import gc
import importlib

import torch


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
