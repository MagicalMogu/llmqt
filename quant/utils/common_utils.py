from ast import Raise
import gc
import importlib

import torch.nn as nn


def get_module_by_name_suffix(mode, module_name: str):
    for name, module in mode.named_modules():
        if name.endswith(module_name):
            return module

    raise ValueError(f"Module with suffix '{module_name}' not found")

def get_best_device():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "cuda:0"
    return "cpu"

def get_named_linear(module):
    # 返回字典，找到所有linear层
    return {name: m for name,m in module.named_modules() if isinstance(m, nn.linear)}

# 典型的不量化层 ['lm_head']
def exclude_layers_to_not_quantize(named_linear: dict, modules_to_not_convert: list[str]):
    if modules_to_not_convert is None:
        return named_linear
    
    filter_layers = {}
    
    for name, layer in named_linear.items():
        # name = "model.layers.0.self_attn.k_proj"
        # modules_to_not_convert = ["k_proj", "lm_head"]
        # 实际上要做完整匹配 和 末端匹配
        # 当你发现某个层量化有问题的时候，就在这里添加
        if not any(
            name == module_name or name.endswith(f".{module_name}")
            for module_name in modules_to_not_convert
        ):
            filter_layers[name] = layer

    return filter_layers
    
def get_op_name(module, op):
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError("Operator not found in the module")