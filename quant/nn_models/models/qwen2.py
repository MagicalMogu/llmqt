import stat

import tqdm # 
from typing import List, Self, Tuple
from quant.core.base import BaseModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer as OldQwen2DecoderLayer,
    Qwen2ForCausalLM as OldQwen2ForCausalLM,
)

class Qwen2MOdelForCausalLM(BaseModelForCausalLM):
    layer_type = "Qwen2DecoderLayer"

    @staticmethod
    def get_model_layers(model: OldQwen2ForCausalLM):
        return model.model.layers
        
    @staticmethod
    def move_embed(model: OldQwen2ForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(device)
        

    @staticmethod
    def get_layers_for_scaling(module: OldQwen2DecoderLayer, input_feat, module_kwargs):
        """
        这个函数把decoder layer划分成为几组适合一起搜scale的子图
        dict的意义如下: (有些不需要的可以不传)
        prev op 前面是哪个算子
        哪些层共享输入
        输入是什么
        表示这组属于哪个更大的子模块
        当前 layer 前向需要的附加参数，比如 attention_mask 之类
        """

        layers = []

        # q/k/v 三个投影共享同一个前置 layernorm 输出，适合一起搜索缩放比例。
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # 只有在 v_proj 的输出维度和 o_proj 的输入维度一致时，
        # 才能把 o_proj 作为一组可缩放层处理；GQA/MQA 下通常不满足。
        if module.self_attn.v_proj.weight.shape[0] == module.self_attn.o_proj.weight.shape[1]:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                    # module2inspect=module.self_attn,
                    # kwargs=module_kwargs,
                )
            )

        # gate_proj 和 up_proj 共享 post_attention_layernorm 的输出。
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
                # kwargs=module_kwargs,
            )
        )

        # down_proj 消耗的是 MLP 中间激活，单独作为一组处理。
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
                # module2inspect=module.mlp,
                # kwargs=module_kwargs,
            )
        )

        return layers


# Keep a correctly-cased export for import sites.
Qwen2ModelForCausalLM = Qwen2MOdelForCausalLM
