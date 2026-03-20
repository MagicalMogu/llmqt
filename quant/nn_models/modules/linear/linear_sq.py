
.from .linear_base import LinearBase

class SqW8A8BBF16OBF16Linear(LinearBase):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        