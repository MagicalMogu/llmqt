import torch
import torch.nn.functional as F

from .linear_base import LinearBase


def quantize_per_tensor_absmax(w: torch.Tensor):
    abs_max = w.detach().abs().max().clamp(min=1e-5)
    scale = abs_max / 127.0
    int8_weight = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return int8_weight, scale


class SqW8A8BBF16OBF16Linear(LinearBase):
    """
    SmoothQuant INT8 linear.

    `SqW8A8BBF16OBF16` 的含义：
    - `W8`: weight 是 int8
    - `A8`: activation 目标是 int8
    - `BBF16`: bias 保持 bfloat16/这里先用 float16 buffer 近似占位
    - `OBF16`: output 保持 bf16/这里先返回浮点输出
    """

    # For qkv_proj / o_proj / mlp proj
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        weight_scale=1.0,
        input_scale=1.0,
        dev="cuda:0",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "qweight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                device=dev,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (self.out_features,),
                    dtype=torch.float16,
                    requires_grad=False,
                    device=dev,
                ),
            )
        else:
            self.bias = None

        self.register_buffer(
            "weight_scale", torch.tensor(weight_scale, dtype=torch.float16, device=dev)
        )
        self.register_buffer(
            "input_scale", torch.tensor(input_scale, dtype=torch.float16, device=dev)
        )

    @staticmethod
    def from_linear(module: torch.nn.Linear, input_scale, dev="cuda:0"):
        int8_module = SqW8A8BBF16OBF16Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dev=dev,
        )
        # 很糟糕的命名，这里的scale已经是量化时候的映射的乘数了
        # 不是之前calib得到的那个scale了（平滑用）
        # 之前的那个scale已经在smooth py里面（smooth_lm 那个函数)
        # 被乘到权重上了，这里就直接量化到int8就够了
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)

        int8_module.weight_scale.copy_(
            torch.tensor(weight_scale, dtype=torch.float16, device=dev)
        )
        int8_module.input_scale.copy_(
            torch.tensor(input_scale, dtype=torch.float16, device=dev)
        )
        int8_module.qweight.copy_(int8_weight.to(dev))
        if module.bias is not None:
            int8_module.bias.copy_(module.bias.detach().clone().to(dev))
        return int8_module

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape

        # [*, in_features] -> [N, in_features]
        x = x.view(-1, x_shape[-1]).to(self.qweight.device)

        # 当前实现仍是“伪 A8”路径：先按输入 scale 映射，再反量化回 BF16 参与 GEMM。
        # 这样写是为了把量化语义显式展开，便于后续替换成真正的 INT8 kernel。
        x_i8 = x.to(torch.bfloat16) * self.input_scale.item()
        
        x_bf16 = x_i8 / self.input_scale.item()

        weight_bf16 = self.qweight.to(torch.bfloat16) * self.weight_scale.item()
        y = torch.matmul(x_bf16, weight_bf16.t())

        if self.bias is not None:
            y = y + self.bias.to(torch.bfloat16)

        y = y.view(*x_shape[:-1], -1)
        return y
