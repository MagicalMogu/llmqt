import os
import json
from typing import Dict, Optional
from dataclasses import dataclass, field
from transformers.utils.hub import PushToHubMixin

# 功能：在量化的时候，保存量化的配置
# 在runtime的时候，读取配置，分发到不同的量化算子

# 1. 从json里读取config, 然后初始化本类
# 2. 保存的时候序列化为dict，然后保存为json

@dataclass
class QuantConfig:
    quant_methon: str = field(default="awq")
    
    zero_point: bool = field(default=True) # only use in awq
    q_group_size: int = field(default=0) # only use in awq
    w_bit: int = field(default=8) # only use in awq, default 4 bit

    config_file_name: str = field(default="config.json")
    # 量化时不需要量化的模块名称列表
    modules_to_not_convert: list = field(default_factory=lambda: ["lm_head"])
    fp8_static_quant: bool = field(default=False) # 是否使用静态FP8量化
    per_tensor_quant: bool = field(default=False) # 是否使用逐张量量化
    kv_cache_quant_layers: list = field(default_factory=list) # 需要量化kv cache的层列表，格式为["layer.0", "layer.1", ...]

    @staticmethod
    def _normalize_quant_config_dict(quant_config: Dict) -> Dict:
        normalized = dict(quant_config)
        if "quant_method" in normalized and "quant_methon" not in normalized:
            normalized["quant_methon"] = normalized.pop("quant_method")
        return normalized

    @classmethod
    def from_pretrained(cls, save_dir, **kwargs):
        config_path = os.path.join(save_dir, cls.config_file_name)
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        quant_config = config_dict.get("quantization_config")
        
        if quant_config is None:
            return cls()
        return cls(**cls._normalize_quant_config_dict(quant_config))
    
    @classmethod
    def from_dict(cls, quant_config: Dict={}):
        if not quant_config:
            return cls()
        return cls(**cls._normalize_quant_config_dict(quant_config))

    @property
    def quant_method(self) -> str:
        # 兼容调用侧使用更自然的 quant_method 命名；
        # 先不改底层字段名，避免影响现有序列化格式和已有代码。
        return self.quant_methon

    @quant_method.setter
    def quant_method(self, value: str) -> None:
        self.quant_methon = value
      
    def to_transformer_dict(self):
        return {
            "quant_method": self.quant_methon,
            "zero_point": self.zero_point,
            "q_group_size": self.q_group_size,
            "w_bit": self.w_bit,
            "modules_to_not_convert": self.modules_to_not_convert,
            "fp8_static_quant": self.fp8_static_quant,
            "per_tensor_quant": self.per_tensor_quant,
            "kv_cache_quant_layers": self.kv_cache_quant_layers,
        }
    
    
