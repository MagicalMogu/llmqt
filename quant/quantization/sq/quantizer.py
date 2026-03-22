import torch

from quant.nn_models.modules.linear import get_concrete_linear_module
from quant.quantization.base.quantizer import BaseQuantizer
from quant.quantization.sq.smooth import smooth_lm
from quant.utils.common_utils import (
    clear_memory,
    exclude_layers_to_not_quantize,
    get_best_device,
    get_named_linears,
    set_op_by_name,
)
from quant.utils.sq_clib_utils import get_act_scales, get_static_decoder_layer_scales


# TODO 调研一下除了opt外的其他模型apply sq是如何做精度转换的？ 是否和opt一样？
# 考虑到int8激活及dq是int上下层的精度，所以多个cutlass s8 gemm是需要的

class SqQuantizer(BaseQuantizer):
    def __init__(
        self,
        modelforCausalLM, # Qwen2ModelForCausal类 # only use for awq
        model,
        model_type,
        tokenizer,
        quant_config,
        quant_method,
        w_bit,
        group_size,
        zero_point,
        calib_data,
        duo_scaling,
        modules_to_not_convert=None,
        fake_quant=False, # true时为fake quant, false为real quant
        apply_clip=False,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        super(BaseQuantizer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.quant_config = quant_config
        self.quant_method = quant_method
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.calib_data = calib_data
        self.fake_quant = fake_quant
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )

    def quantize(self):
        # 对每个layer的input act求amax
        act_scales = get_act_scales(
            self.model, self.tokenizer, self.calib_data, self.max_calib_samples, self.max_calib_seq_len
        )

        # smooth
        # 这里的0.5是一个经验值，后续可以调整，或者改成搜索
        smooth_lm(self.model, act_scales, 0.5)

        # 截止到这里，sq的核心ideal已经完成，后面是执行量化。

        # 返回dict，key为linear name，value是linear module
        named_linears = get_named_linears(self.model) # self.modules[i])

        # Filter out the linear layers we don't want to exclude
        named_linears = exclude_layers_to_not_quantize(
            named_linears, self.modules_to_not_convert
        )
        if self.fake_quant:
            for name, linear_layer in named_linears.items():
                print("[info] hit ", name)
                # linear_layer = linear_layer.to(common_device).half() # 这里to common
                linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                    linear_layer.weight.data
                )
            return
        # calib，与上面的get_act_scales的区别在于，这里还要求每个linear的output max
        decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(self.model,
                                                                           self.tokenizer,
                                                                           self.calib_data,
                                                                           self.max_calib_samples,
                                                                           self.max_calib_seq_len)

        # [STEP 4]: scale和clip都apply之后，开始real Quantize weights+替换int8 linear
        if not self.fake_quant:
            self._apply_quant(self.model, decoder_layer_scales, named_linears) # TODO

        clear_memory()

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape

        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0, (
                f"in_features ({org_w_shape[-1]}) must be divisible by "
                f"group_size ({self.group_size})"
            )
            w = w.reshape(-1, self.group_size)

        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2 ** self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -2 ** (self.w_bit - 1)
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(w).sum() == 0
        assert torch.isnan(scales).sum() == 0

       #
        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

    def _apply_quant(self, module, decoder_layer_scales, named_linears):
        dev = get_best_device()
        for name, linear_layer in named_linears.items():
            linear_layer = linear_layer.to(dev)

            # 1. 拿到sq对应的linear module
            q_linear_module = get_concrete_linear_module(self.quant_method)
            proj = name.split(".")[-1]

            try:
                layer_id = int(name.split(".")[3])  # for opt model
            except Exception:
                layer_id = int(name.split(".")[2])  # for non-opt model

            if proj in ["q_proj", "k_proj", "v_proj"]:
                print("[info] hit layer ", layer_id)
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["attn_input_scale"]

            # for llama like
            elif proj in ["o_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["out_input_scale"]
            elif proj in ["gate_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["gate_input_scale"]
            elif proj in ["up_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["up_input_scale"]
            elif proj in ["down_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["down_input_scale"]

            # for opt
            elif proj in ["out_proj"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["out_input_scale"]
            elif proj in ["fc1"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["fc1_input_scale"]
            elif proj in ["fc2"]:
                print("[info] hit ", proj)
                scales = decoder_layer_scales[layer_id]["fc2_input_scale"]
            else:
                print("[warning] this dont hit any proj, pls check. the current is ", proj)
                continue
            
            # 获得量化后的linear module
            q_linear = q_linear_module.from_linear(
                module=linear_layer,
                input_scale=scales,
                dev=dev,
            )

            linear_layer.cpu()
            q_linear.to(dev)
            # 替换原linear，这里的module是指整个模型
            set_op_by_name(module, name, q_linear)
            clear_memory(q_linear)
