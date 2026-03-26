from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2.5-0.5B-Instruct"
quant_path = "Qwen2.5-0.5B-Instruct-awq"
# 1. Define quantization configuration
quant_config = {
    "quant_method": "awq",
    "w_bit": 4,
    "zero_point": True,
    "q_group_size": 128,
}
# 2. Load the pre-trained model and tokenizer
model = AutoQuantForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 3. Quantize the model with the defined configuration
model.quantize(tokenizer, quant_config=quant_config)
# 4. Save the quantized model and tokenizer
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"Model quantized and saved to {quant_path}")
