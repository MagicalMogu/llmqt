from ast import Dict, mod
from os import name
import tokenize
from typing import List

from quant.quantization.awq import scale
import torch
import torch.nn as nn
from quant.quantization.base.quantizer import BaseQuantizer
from quant.quantization.awq.awq_clib_utils import get_calib_dataset
from quant.quantization.common_utils import (
    # append_str_prefix,
    # get_op_name,
    # get_named_linears,
    # set_op_by_name,
    # exclude_layers_to_not_quantize,
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
            named_linears = self.get_named_linear(self.target_modules[i])
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
            # module_config里每个元素是一个dict, 包含该linear的输入特征维度等信息,
            # 举例
            # {'name': 'self_attn.q_proj', 'input_dim': 4096, 'output_dim': 4096, 'group_size': 128, 'dtype': torch.float16}

            scale_list = [
                self._search_best_scale(self.target_modules[i], **layer)
                for layer in module_config
            ]
            # 把搜到的scale应用到第i个decoder layer的每个linear上
            apply_scale(
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
                self._quantize_weights(self.target_modules[i], named_linears, common_device)



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