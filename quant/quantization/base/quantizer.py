import torch

class BaseQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()