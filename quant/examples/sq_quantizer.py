from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer

# model_path = 'Qwen/Qwen2.5-14B-Instruct'
# quant_path = 'Qwen2.5-14B-Instruct-sq'
# 其他效果一般，在这个模型上效果还可以
# model_path = 'facebook/opt-13b'
# quant_path = 'opt-13b-sq'
# model_path = 'meta-llama/Llama-3.1-8B-Instruct'
# quant_path = 'Llama-3.1-8B-Instruct-sq'
model_path = 'Qwen/Qwen2.5-0.5B-Instruct'
quant_path = 'Qwen2.5-0.5B-Instruct-sq'

# 做8bit量化；zero_point 对 SQ 当前实现理论上不关键，先保留显式配置。
quant_config = {"quant_method": "sq", "zero_point": True}

# Load model
model = AutoQuantForCausalLM.from_pretrained(model_path, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
