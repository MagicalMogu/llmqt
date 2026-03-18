from ast import Dict, mod
from collections import defaultdict
from email.policy import default
import functools
import inspect
import logging
from os import name
from tkinter import W
from turtle import end_fill
from xml.dom.minidom import Element
from click import clear
from requests import get
import tqdm
import tokenize
from typing import List

from quant.quantization.awq import scale
import torch
import torch.nn as nn
from quant.quantization.base.quantizer import BaseQuantizer
from quant.utils.awq_clib_utils import get_calib_dataset
from quant.utils.common_utils import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    # set_op_by_name,
    exclude_layers_to_not_quantize,
    # clear_memory,
    get_best_device
)

class AwqQuantizer(BaseQuantizer):
    def __init__(
        self,
        modelforCausalLM, # BaseModelForCausalLM.from_pretrained后的model,项目自定义的, 比如qwen2model
        model, # AutoModelForCausalLM.from_pretrained后的model, transformers库的
        model_type, # 模型类型, 比如"qwen2"
        tokenizer, # AutoTokenizer.from_pretrained后的tokenizer, transformers库的
        quant_config, # QuantConfig.from_pretrained后的quant_config, 项目自定义
        quant_method, 
        w_bits, # 权重量化的bit数
        group_size, # 权重量化的group size
        zero_point, # 权重量化的zero point
        calib_data, # str, 校准数据数据集名字，可以是"c4", "wikitext2"等, huggingface
        duo_scaling, # 是否使用duo-scaling方法进行量化, true or false
        modules_not_convert=None, # 不进行量化的模块列表, list
        fake_quant=False, # 是否使用fake quantization方法进行量化, true or false
        apply_clip=True, # 是否在量化后应用clip方法, true or false
        n_parrallel_calib_sample=None, # 校准时并行处理的样本数量, int
        max_calib_samples=128, # 校准时使用的最大样本数量, int
        max_calib_seq_len=512, # 校准时使用的最大序列长度, int
        max_chunk_memory=1024*1024*1024, # 校准时每个chunk的最大内存, 省显存小技巧
    )-> None:
        super(BaseQuantizer, self).__init__()
        self.awq_model = modelforCausalLM
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.quant_config = quant_config
        self.quant_method = quant_method
        self.w_bits = w_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.calib_data = calib_data
        self.duo_scaling = duo_scaling
        self.fake_quant = fake_quant
        self.apply_clip = apply_clip
        self.n_parrallel_calib_sample = n_parrallel_calib_sample

        if self.model.types == "qwen3_moe":
            self.max_calib_samples = max_calib_samples * 2 # qwen3_moe模型需要更多的校准样本
        else:
            self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_not_convert = (
            modules_not_convert if modules_not_convert is not None else []
        )

        # 名字叫target_modules,不是self.modules, 会和basequantize里的self.modules冲突
        # self.inps 表示捕获到的第一个layer的输入
        # self.target_modules表示需要量化的所有层
        self.target_modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, n_seq_len=self.max_calib_seq_len
        )


    def quantize(self):
        # 遍历decoder层
        for i in tqdm(range(len(self.target_modules)), desc="Quantizing layers"):
            # next是获取该层的第一个参数, 获取该参数所在的设备
            common_device = next(self.target_modules[i].parameters()).device
            if str(common_device) == "cpu" or common_device is None:
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count())
                else:
                    best_device = get_best_device()

                self.target_modules[i]= self.target_modules[i].to(best_device)
                common_device = next(self.target_modules[i].parameters()).device

            # 第0 decoder layer 的输入，传到要量化的层的device里
            self.inps = self.inps.to(common_device)
            # emb table 同理
            self.awq_model.move_embed(self.model, common_device)
            # 返回第 i 个 decoder layer 里需要量化的层的name和层对象（映射字典,name - torch.nn.linear）
            # 举例
            # {'self_attn.q_proj': Linear(in_features=4096, out_features=4096, bias=True), 
            # 'self_attn.k_proj': Linear(in_features=4096, out_features=4096, bias=True), 
            # 'self_attn.v_proj': Linear(in_features=4096, out_features=4096, bias=True), 
            # 'self_attn.o_proj': Linear(in_features=4096, out_features=4096, bias=True), 
            # 'mlp.gate_proj': Linear(in_features=4096, out_features=11008, bias=True), 
            # 'mlp.up_proj': Linear(in_features=4096, out_features=11008, bias=True), 
            # 'mlp.down_proj': Linear(in_features=11008, out_features=4096, bias=True)
            # 
            named_linears = get_named_linears(self.target_modules[i])
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_not_convert
            )
            # calib 返回第i个decoder layer 中每个linear的 input_feature 或者说 activation
            input_feat = self._get_input_feature(self.target_modules[i], named_linears)
            # 返回第i个decoder layer的 config, 包含每个linear的输入特征维度等信息
            # attn qkvo gate up down 7个linear
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.target_modules[i], input_feat, self.module_kwargs
            )
            # 搜寻每个linear的best scale
            # module_config里每个元素是一个dict, 包含该linear的输入特征维度等信息
            # 详细看函数内部
            scale_list = [
                self._search_best_scale(self.target_modules[i], **layer)
                for layer in module_config
            ]
            # 把搜到的scale应用到第i个decoder layer的每个linear上
            self.apply_scale(
                self.target_modules[i], scale_list, input_feat, common_device, self.module_kwargs
            )

            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.target_modules[i], named_linears, input_feat, common_device, self.module_kwargs
                )
                self.apply_clip(self.target_modules[i], clip_list, input_feat, common_device, self.module_kwargs)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.target_modules[i])
                )
            
            # 真正量化的地方，在这之前，上面的weight都是fp16
            # fp16->int4
            # 原地量化, 没有新建一个量化后的模型, 直接把原模型的权重改了
            if not self.fake_quant:
                self._apply_quant(self.target_modules[i], named_linears, common_device)



# 下面是init_quant函数, 主要是获取校准数据, 获取需要量化的层, 获取第一个decoder layer的输入
    def init_quant(self, n_samples=128, n_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data = self.calib_data,
            tokenizer = self.tokenizer,
            n_samples = n_samples,
            max_seq_len = n_seq_len,
            split = "validation",
        )
        samples = torch.cat(samples, dim=0) # [n_samples, seq_len]

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # 捕获decoder layer0的input
        # 实际上是上一个模块的输出，我们做一个假前向传播，触发hook函数，捕获到这个输入
        # 我们把layer0 包装成这个Catcher
        # 因为大部分decoder的forward如下
        # def forward(self, hidden_states, ...):
        # hidden_states就是输入, 也就是我们要捕获的input，则为args[0]或者kwargs的第一个元素
        # 不是百分百通用，但对很多 HuggingFace decoder 模型能工作。
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                # 继承所有属性，防止触发attribute error
                for name, value in module.__dict__.items():
                    setattr(self, name, value)

            def forward(self, *args, **kwargs):
                if (len(args) > 0):
                    hidden_stats = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError("Catch the input of the first layer, stop forward process")

        modules[0] = Catcher(modules[0])
        # forward,捕获到inputs, 还有layer_kwargs
        # 也就是decoder layer的forward里除了hidden_states以外的其他参数, 比如attention mask等
        try:
            self.model(samples.to(best_device))
        except ValueError:
            pass

        print(inps)
        # 还原
        modules[0] = modules[0].module
        # prepare_inputs_for_generation是transformers里生成文本时准备输入的函数
        # 输入samples，根据当前的状态动态地准备好下一步生成文本需要的输入, 比如input_ids, attention_mask等 
        # 把捕获到的inputs和layer_kwargs准备好, 以便后续量化过程中使用
        # ep.prefill阶段，inputs如下
        # inputs = {
        # 'input_ids': initial_input 初始化的输入 [batch, seq_len]
        # 'attention_mask': attention_mask, 初始mask
        # 'use_cache': True 
        # }
        # model_inputs = model.prepare_inputs_for_generation(**inputs)
        # 但是decode阶段，inputs如下
        # inputs = {
        # 'inputs_ids': new_token [batch, 1]
        # 'past_key_values': past 上一轮的kv
        # 'attention_mask': update mask扩展mask
        # }
        # 现在有的layer kwargs是不全的，只是layer0的一个切面，这个函数可以补全回
        # 输入时期的完整状态
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        layer_kwargs.pop("input_ids")

        del samples
        # 去掉一维
        # 省显存
        inps = inps[0]
        modules = [module.to("cpu") for module in modules]
        self.awq_model.move_embed(self.model, "cpu")

        return modules, layer_kwargs, inps

    # 1 decoder layer 2 里面的linear layer，dict name:nn.module
    def _get_input_feature(self, layer, named_linears):

        # 在每个linear module上注册这个钩子，截获输入
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x) # {linear name : input act}
        #增加对一些模型的兼容
        if self.awq_model.model_type == "qwen3_moe":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }
        if self.awq_model.model_type == "llama4":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }

        input_feat = defaultdict(list)
        handles = []
        for name, module in named_linears.items():
            handles.append(
                module.register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )

        sanitized_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)
        try:
            # init_quant 里获得的输入在这里派上用场
            # 当前 layer 前向后，输出会成为下一个 layer 的输入
            self.inps = self._module_forward(self.inps, layer, sanitized_kwargs)
        finally:
            # remove handle，别影响下一个 layer
            for h in handles:
                h.remove()
        
        # 主要用于moe的验证
        # 一个feat_dict[name] 里存的是一个列表 [tensor1, tensor2, tensor3, ...]
        # 一个linear可能会被调用多次的，tensor的形状是[batch,seq_len,in_features]
        # 或者 [token, in_features]
        # 
        def cat_and_assert(k,v):
            x = torch.cat(v, dim=0)
            assert x.shape[0] != 0, (
                f"Failed to collect input features for layer '{k}'. "
                "Please check whether the target module was executed in the forward pass."
            )
            return x

        input_feat = {k: cat_and_assert(k,v) for k,v in input_feat.items()}
        return input_feat

    
    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        过滤掉目标模块的forward方法不支持的参数

        """

        module_signature = inspect.signature(module.forward)
        sanitized_kwargs = {}
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in module_signature.parameters.values()
        )
        for key, value in inputs_kwargs.items():
            if accepts_var_kwargs or key in module_signature.parameters:
                sanitized_kwargs[key] = value
        return sanitized_kwargs

    def _module_forward(
        self,
        x: torch.Tensor,
        module: nn.Module,
        module_kwargs: dict,
    ) -> torch.Tensor:
        """
        对任意子模块做一次前向，并统一返回 tensor 输出。

        这里单独封装一个 helper，主要是为了解决两件事：
        1. 有些模块一次性跑完整个 calibration batch 会占很多显存，所以支持按小批次拆开跑。
        2. Hugging Face 里的模块 forward 有时返回 tuple，这里统一取第一个真正的 hidden states/output。
        """

        if self.n_parrallel_calib_sample is None:
            # 显存足够时，直接把所有校准样本一次性跑完。
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # 显存紧张时，把校准输入按 batch 维拆开。
            # 每次只跑 n_parrallel_calib_sample 条，最后再沿 dim=0 拼回完整输出。
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parrallel_calib_sample, dim=0)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)
                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                # 中间结果先挪到 CPU，避免所有 partial output 一直堆在 GPU 上。
                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad()
    def _search_best_scale(
        self,
        module, # transformer decoder layer
        prev_op,
        layers: List[nn.Linear],
        inp:torch.Tensor,
        module2inspect=None, # 子模块
        kwargs={},
    ):
        # 实际上只需要传入module，其他的是搜索子图里的信息，由get_layers_for_scaling返回的dict

        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]
        
        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # put X to device
        inp = inp.to(next(module2inspect.parameters()).device)
        # [step 1], 计算 w [out channel, in channel] 下，每group的weight均值
        # 把要量化的层的权重拼在一起，方便计算, [o1, in] [o2,in] -> [o1+o2, in]
        weight = torch.cat([layer.weight for layer in layers], dim=0)
        org_shape = weight.shape
        # [(o1+o2)*in//group_size, group_size]
        weight = weight.view(-1, self.group_size)
        # 量化时，哪个通道更敏感、权重分布更大，取决于权重幅值，不取决于它是正还是负。
        # 此时被缩放为[0,1], 越接近1表示越接近该group里最大的权重
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        w_scale = w_scale.view(org_shape) # [o1+o2, in]
        w_mean = w_scale.mean(dim=0)# [in]

        # [step 2]: view成[bs*token, in], 再计算每列activation均值
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.shape[0]
        num_channels = inp_flat.shape[1]
        # multiply by 2 for fp32
        element_size_bytes = inp_flat.element_size() * 2

        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)
        chunk_size = max(chunk_size, 1)

        x_sum = torch.zeros(num_channels, dtype=torch.float32)
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum
        # [in], [bs, in chanels]的X shape下的每列的mean
        x_mean = (x_sum / num_elements).to(inp.dtype)   

        # [STEP3] compute output of module
        # 拿一个基准输出fp16，后续要对比loss function来搜寻最好的scales
        # 对于qkv是self attn, gate up down是mlp, 还有o_proj的情况，计算它们的输出
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            fp16_output = fp16_output.clip(torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)

        # [STEP4] 搜寻scale
        best_scales = self._compute_best_scales(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, layer) for layer in layers]),
            best_scales,
        )
    
    @torch.no_grad()
    def _search_best_clip(
        self, 
        layer: nn.Module, 
        named_linears: dict, 
        input_feat: dict, 
        common_device: torch.device
    ):
        # clip_list = [(op_name, best_clip_value), ...]
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            if any([_ in name for _ in avoid_clipping]):
                continue
            named_linears[name].to(common_device)
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].to("cpu")
        return clip_list


    # 返回值 scales->tensor([in]), 每个输入通道一个scale
    def _compute_best_scales(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: nn.Module,
        layers: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: dict={}
    ):
        """
        Compute loss and select best scales for every inchannels
        L(s) = || Q(W * s) (s^-1 * X) - W*X ||^2
        Q(W*s) 表示对 W*s 进行量化, s是缩放比例
        X是输入特征
        W是权重 fp16
        s per channel, 也就是每个输入通道一个缩放比例

        用网格搜索去找最佳scale
        """

        n_grid = 20
        # 搜索流程的核心是：
        # 1. 保存当前 module2inspect 的原始 fp16 参数快照 org_sd
        # 2. 枚举一系列 ratio，构造候选 scales
        # 3. 把候选 scales 临时应用到 layers 上，重新前向
        # 4. 比较候选输出与 fp16_output 的误差，留下误差最小的一组
        # 5. 每轮结束后都用 org_sd 恢复参数，保证所有候选在同一基线比较
        #
        # 这里常见的参数快照写法：
        # org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}
        # 它的作用不是保存结果，而是为了“回滚”：
        # 网格搜索会反复试不同 scale；如果不恢复原始权重，前一轮的改动会污染下一轮。
        #
      
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        # 这个快照是为了每次试完 candidate scales 后都能恢复原始参数。
        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        # x_mean 和 w_mean 都是 [in]，每个输入通道一个均值。
        for ratio_idx in range(n_grid):
            ratio = ratio_idx / n_grid
            # AWQ 论文原始 proxy 更偏向 activation magnitude；
            # duo_scaling 是对论文里 activation-only proxy 的工程增强版。
            # 论文原始思路更接近只用 activation 的 magnitude x_mean 来刻画通道重要性；
            # 开源实现里常加入 weight 统计 w_mean，形成同时参考 x/w 的 duo_scaling：
            #     scales ~ x_mean^ratio / w_mean^(1-ratio)
            #
            # 这样做的好处：
            # 1. 当某个通道 activation 很小、但权重很大时，只看 x_mean 容易把 scale 压得过小，导致权重W消失
            # 2. 同时利用 w_mean 后，搜索空间更平滑，极端稀疏激活下更稳（ds和llama上有一些极度稀疏但非常重要的权重）
            # 3. ratio 相当于在“更信 activation”与“更信 weight”之间做网格搜索
            #    ratio 越大越偏向 x_mean，越小越偏向 w_mean
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / w_mean.pow(1 - ratio) + 1e-4).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)

            # scales最后有一些triks
            # 1 平衡scale的范围，避免出现最大值和最小值差距过大，到时候权重分布会很不均匀
            # 2 保持数值稳定，在后续的矩阵乘会有数据溢出 消失
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales = scales.view(1, -1).to(device)
            # 极端case的处理，相当于不做awq
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # 求loss function的一部分
            # Q(W*s) * s^-1
            for fc in layers:
                # dot product
                fc.weight.mul_(scales)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales
                )
            # Q(W*s) * s^-1 * X
            # 对子模块跑一个前向,就可以求出实际output
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)

            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            # 回滚
            module2inspect.load_state_dict(org_sd, strict=False)

        if best_ratio == -1:
            logging.debug(history)
            raise ValueError("Failed to find best scales during grid search.")
        
        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    def _compute_best_clip(
            self, 
            weight: torch.Tensor, 
            inp: torch.Tensor,
            n_grid=20,
            max_shrink=0.5,
            n_samples_token=512) -> torch.Tensor:
        """
        返回tensor shape [out_channel, n_group, 1], 每个group一个clip值
        搜寻最佳clip值的流程和搜寻最佳scale类似,也是枚举一系列候选clip值
        应用到权重上 前向计算输出 与fp16输出比较误差 留下误差最小的那个clip值。
        但是这里不是按照输入通道来搜寻 而是分为group来搜寻
        和后面的group-wise量化配合
        """
        assert weight.dim() == 2
        org_w_shape = weight.shape

        # group-wise quantization: [oc, ic] -> [oc, 1, n_group, group_size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        assert org_w_shape[1] % group_size == 0, (
            f"in_features ({org_w_shape[1]}) must be divisible by group_size ({group_size})"
        )

        # input activation: [bs, seq, ic] or [token, ic] -> [1, n_token, n_group, group_size]
        inp = inp.view(-1, inp.shape[-1])
        inp = inp.reshape(1, inp.shape[0], -1, group_size)

        # sample tokens uniformly to reduce search memory/time cost
        step_size = max(1, inp.shape[1] // n_samples_token)
        inp = inp[:, ::step_size]

        w = weight.reshape(org_w_shape[0], 1, -1, group_size)

        # process output channels by chunks to avoid OOM
        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64
        assert org_w_shape[0] % oc_batch_size == 0, (
            f"out_channels ({org_w_shape[0]}) must be divisible by oc_batch_size ({oc_batch_size})"
        )

        w_all = w
        best_max_val_all = []

        for oc_idx in range(org_w_shape[0] // oc_batch_size):
            # 256 out channels per group
            w_chunk = w_all[oc_idx * oc_batch_size : (oc_idx + 1) * oc_batch_size]
            # [oc_batch_size, 1, n_group, group_size]
            # 变成 [oc_batch_size, 1, n_group, 1] 的形式，表示每个group的clip值
            org_max_val = w_chunk.abs().amax(dim=-1, keepdim=True)
            best_max_val = org_max_val.clone()

            min_err = torch.full_like(org_max_val, float("inf"))

            inp_chunk = inp.to(w_chunk.device)
            # inp_chunk [1, n_token, n_group, group_size]
            # w_chunk [oc_batch_size, 1, n_group, group_size] ->
            # -> [oc_batch_size, n_token, n_group]
            # 本质是分块分组之后的矩阵乘法
            # 求这个group的原始输出，作为后续搜索clip值的基准
            org_out = (inp_chunk * w_chunk).sum(dim=-1)

            # try progressively smaller clipping ranges
            for grid_idx in range(int(max_shrink * n_grid)):
                shrink_ratio = 1 - (grid_idx / n_grid)
                max_val = org_max_val * shrink_ratio
                min_val = -max_val

                clipped_w = torch.clamp(w_chunk, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(clipped_w)[0]

                cur_out = (inp_chunk * q_w).sum(dim=-1)
                # average over sampled tokens -> per-(oc, group) error
                cur_err = (cur_out - org_out).pow(2).mean(dim=1).view_as(min_err)
                del clipped_w
                del cur_out
                better = cur_err < min_err
                min_err[better] = cur_err[better]
                best_max_val[better] = max_val[better]

            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)
        clear_memory(inp_chunk)
        clear_memory(org_out)
        # [out_channel, n_group, 1]
        return best_max_val.squeeze(1)


    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape # [5120, 5120]

        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0, (
                f"in_features ({org_w_shape[-1]}) must be divisible by "
                f"group_size ({self.group_size})"
            )
            w = w.view(-1, self.group_size) # [5220*40, 128]

        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        
        if self.zero_point:
            # 1. 非对称量化
             max_val = w.amax(dim=0, keepdim=True)
             min_val = w.amin(dim=0, keepdim=True)
             max_int = 2 ** self.w_bits - 1
             min_int = 0
             scales = (max_val - min_val).clamp(min=1e-5) / max_int
             zeros = (-torch.roud(min_val / scales)).clamp(min=min_int, max=max_int)
             w = (
                 torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
             ) * scales
             zeros = zeros.view(org_w_shape[0], -1)

        else:
            # 2. 对称量化
            # int4的公式
            # scale = absmax / 2^4-1, zp = 0
            # qx = clip(round(x/scale), -8, 7)
            max_val = w.abs().amax(dim=0, keepdim=True).clamp(min=1e-5)
            max_int = 2 ** (self.w_bits - 1) - 1
            min_int = -2 ** (self.w_bits - 1)
            # 实际会把值映射到 -7,7
            scales = max_val / max_int
            zeros = None
            # fake quant 最后需要把scales乘回去，保持数值不变
            # 会包含量化误差，但是数值意义不变
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales
        
        assert torch.isnan(w).sum() == 0
        assert torch.isnan(scales).sum() == 0
        # 
        scales = scales.view(org_w_shape[0], -1)
        # view需要内存连续，reshape会先尝试view，不行了就复制一份数据，保证内存连续
        w = w.reshape(org_w_shape)

        return w, scales, zeros


    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # 分 chunk 算误差，避免一次性把整段输出搬到高精度后占太多显存/内存。
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)
        chunk_size = max(chunk_size, 1)

        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (
                fp16_chunk.to(device) - int_w_chunk.to(device)
            ).float().pow(2).sum().item()
            loss += chunk_loss

        loss /= num_elements
        return loss

    def _apply_quant(
            self,
            layer: nn.Module,
            named_linears: dict,
            common_device: torch.device
    ):
        