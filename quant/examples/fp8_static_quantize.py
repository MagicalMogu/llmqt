from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer

# model_path = "meta-llama/Llama-3.1-8B-Instruct"
# quant_path = "Llama-3.1-8B-Instruct-fp8-static"
# model_path = "Qwen/Qwen3-8B"
# quant_path = "Qwen3-8B-awq-fp8-static"
# model_path = "Qwen/Qwen2.5-14B-Instruct"
# quant_path = "Qwen2.5-14B-Instruct-fp8-static"
# model_path = "Qwen/Qwen3-30B-A3B"
# # quant_path = "Qwen3-30B-A3B-static"
# model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# quant_path = "DeepSeek-R1-Distill-Qwen-32B-static"
# Small test model.
# You can use a Hugging Face repo id directly; the model will then be cached under
# ~/.cache/huggingface/hub on this machine.
# If you already have a local copy, replace model_path with that local directory.
model_path = "Qwen/Qwen2.5-0.5B-Instruct"
quant_path = "Qwen2.5-0.5B-Instruct-fp8-static"

# quant_config = {"quant_method": "fp8_static_quant", "zero_point": True, "q_group_size": ...}
quant_config = {"quant_method": "fp8_static_quant",
                "per_tensor_quant": True}


model = AutoQuantForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer=tokenizer, quant_config=quant_config)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
