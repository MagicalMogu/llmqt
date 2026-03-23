from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer


model_path = "meta-llama/Llama-4-Scout-17B-16E"
quant_path = "Llama-4-Scout-17B-16E-dyn"
quant_config = {"quant_method": "fp8_dynamic_quant"}

model = AutoQuantForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer=tokenizer, quant_config=quant_config)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
