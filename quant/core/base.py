import json
import os
import torch
from torch import nn
import transformers
from transformers import PretrainedModel, PretrainedConfig, AutoConfig
from huggingface_hub import snapshot_download, save_torch_state_dict
from .config import QuantConfig

TRANSFORMERS_AUTO_MAPPING_DICT = {
    "qwen2": "AutoModelForCausalLM",
}

class BaseModelForCausalLM(nn.Module):
    def __init__(self,
                 model, # pretrained model
                 model_type, # model type exmp. "qwen2"
                 is_quantized, # 是否已经被quantized了
                 config, # config of model
                 quant_config):
        super().__init__()
        self.model: PretrainedModel = model
        self.model_type: str = model_type
        self.isquantized: bool = is_quantized
        self.config: PretrainedConfig = config
        self.quant_config: QuantConfig = quant_config

    @classmethod
    def _load_config(cls, model_path):
        # 1. download model if path is not a dir
        model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)

        quant_config = QuantConfig.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)

        return model_path, config, quant_config

    @classmethod
    def from_pretrained(cls,
        model_path, 
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ):

        model_weights_path, config, quant_config = cls._load_config(
            cls, model_path, "", safetensors, trust_remote_code=trust_remote_code, **model_init_kwargs
        )
        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)

        if model_init_kwargs.get("low_cpu_mem_usage") is None:
            model_init_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        if model_init_kwargs.get("use_cache") is None:
            model_init_kwargs["use_cache"] = use_cache
           
        model = target_cls.from_pretrained(
            model_weights_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_safetensors=safetensors,
            device_map=device_map,
            **model_init_kwargs
        )

        model.eval()

        return cls(model=model, model_type=config.model_type, is_quantized=False, config=config, quant_config=quant_config)
    

    @torch.no_grad()
    def quantize(
        self,
        tokenizer=None,
        quant_config={},
        calib_data="pileval",
        duo_scaling=True, # 是否使用duo-scaling方法进行量化，默认为True
        fake_quant=False,
        apply_clip=True,
        n_parrallel_sample=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024*1024*1024,
    ):
        
        self.quant_config: QuantConfig = QuantConfig.from_dict(quant_config)
        if hasattr(self, "modules_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert
        

        # dispatch to corresponding quantizer
        quantizer_cls = get_concrete_quantizer_cls(self.quant_config.quant_method)
        self.quantizer = quantizer_cls(
            self,
            self.model,
            self.quant_config,
            tokenizer=tokenizer,
            calib_data=calib_data,
        )
        self.quantizer.quantize()
        self.isquantized = True

    def save_quantized(
        self,
        save_directory,
        safetensors=True,
        shard_size="5GB",
    ):
        os.makedirs(save_directory, exist_ok=True)

        class EmptyModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        # 先保存 config/tokenizer 相关文件，避免立刻把完整权重再写一遍
        if self.quant_config is not None:
            self.model.config.quantization_config = self.quant_config.to_transformer_dict()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.do_sample = True

        self.model.save_pretrained(
            save_directory,
            state_dict=EmptyModule().state_dict(),
            safe_serialization=safetensors,
        )

        # 删除 save_pretrained 生成的空权重文件，稍后用 huggingface_hub 的分片保存真实权重
        default_paths = [
            os.path.join(save_directory, "model.safetensors"),
            os.path.join(save_directory, "pytorch_model.bin"),
        ]
        for path in default_paths:
            if os.path.exists(path):
                os.remove(path)

        save_torch_state_dict(
            state_dict=self.model.state_dict(),
            save_directory=save_directory,
            max_shard_size=shard_size,
            safe_serialization=safetensors,
            force_contiguous=True,
            shared_tensors_to_discard=getattr(self.model, "_tied_weights_keys", None),
        )

        # 额外落地一份量化配置，方便 runtime 直接读取
        if self.quant_config is not None:
            quant_cfg_path = os.path.join(save_directory, self.quant_config.config_file_name)
            with open(quant_cfg_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"quantization_config": self.quant_config.to_transformer_dict()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
