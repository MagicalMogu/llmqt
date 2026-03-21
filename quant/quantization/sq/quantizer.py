from quant.quantization.base.quantizer import BaseQuantizer
from quant.utils.common_utils import (
    clear_memory,
    exclude_layers_to_not_quantize,
    get_named_linears,
)


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
        # calib，与上面的get_act_scales的区别在于，这里还要求每个linear的output max，最后
        decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(self.model,
                                                                           self.tokenizer,
                                                                           self.calib_data,
                                                                           self.max_calib_samples,
                                                                           self.max_calib_seq_len)

        # [STEP 4]: scale和clip都apply之后，开始real Quantize weights+替换int8 linear
        if not self.fake_quant:
            self._apply_quant(self.model, decoder_layer_scales, named_linears) # TODO

        clear_memory()
