import torch
from transformers import AutoConfig
from quant.nn_models import *


Quant_CAUSAL_LM_MODEL_MAP = {
    "qwen2": Qwen2ModelForCausalLM,
}

def check_and_get_model_type(model_path, trust_remote_code, **model_init_kwargs):
    # config.json
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code, **model_init_kwargs)
    print(f"Model type: {config.model_type}")
    if config.model_type not in Quant_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"Model type {config.model_type} is not supported for quantization.")
    return config.model_type

class AutoQuantForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "You must instantiate AutoQuantForCausalLM with the `from_pretrained` method."
        )
        
    @classmethod
    def from_pretrained(
        self,
        model_path, 
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ):
        model_type = check_and_get_model_type(
            model_path, trust_remote_code, **model_init_kwargs
        )
        
        return Quant_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path,
            torch_dtype=torch_dtype, # 模型权重类型
            trust_remote_code=trust_remote_code, # 相信远端code或模型
            safetensors=safetensors, # 是否使用safetensors格式加载模型
            device_map=device_map, # 模型加载到哪个设备
            low_cpu_mem_usage=low_cpu_mem_usage, 
            # 是否在加载模型时尽量减少CPU内存使用，如果为True，模型将被分块加载到GPU上，而不是先加载到CPU上再转移到GPU上
            use_cache=use_cache, # 控制是否使用kv cache
            **model_init_kwargs,
        )