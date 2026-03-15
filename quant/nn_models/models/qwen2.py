import stat

import tqdm # 
from typing import List, Tuple
from quant.core.base import BaseModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer as OldQwen2DecoderLayer,
    Qwen2ForCausalLM as OldQwen2ForCausalLM,
)

class Qwen2MOdelForCausalLM(BaseModelForCausalLM):
    layer_type = "Qwen2DecoderLayer"

    @staticmethod
    def get_model_layers(model: OldQwen2ForCausalLM):
    
    @staticmethod
    def move_embed(model: OldQwen2ForCausalLM, device: str):

    @staticmethod
    def get_layers_for scaling(module: OldQwen2DecoderLayer, input)