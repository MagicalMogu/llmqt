import torch

from .linear_base import LinearBase


def per_tensor_quantize(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax.clamp(min=1e-12) / finfo.max
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    # Note: torch.nn.Parameter 不支持 fp8 数据表示，所以这里暂时不 to fp8
    # 喂到kernel 前再 to(fp8)，避免后续计算出问题。 
    # 但这样会占用更多显存，后续可以考虑注册一个 buffer 来存储 fp8 权重
    # qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float()
    return qweight, scale


def static_per_tensor_quantize(w: torch.Tensor, scale):
    scale = torch.as_tensor(scale, device=w.device, dtype=torch.float32)
    fp8_weight = (w / scale).to(torch.float8_e4m3fn)
    return fp8_weight


def per_channel_quantize(w: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    if w.numel() == 0:
        print("[warning] weight is empty! tensor numbers = 0")
        qweight = torch.empty_like(w, dtype=torch.float8_e4m3fn)
        scales = torch.ones((*w.shape[:-1], 1), dtype=torch.float32)
        return qweight, scales
    amax = w.abs().amax(dim=-1, keepdim=True)
    scale = amax.clamp(min=1e-12) / finfo.max
    qweight = (w / scale).clamp(min=finfo.min, max=finfo.max)
    # qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float()
    return qweight, scale


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    if A.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    native_fp8_support = False
    if native_fp8_support:
        need_reshape = A.dim() == 3
        if need_reshape:
            batch_size = A.shape[0]
            A_input = A.reshape(-1, A.shape[-1])
        else:
            batch_size = None
            A_input = A
        # scaled mm是
        output, _ = torch._scaled_mm(
            A_input, # [m, in]
            B.t(), # [in, out]
            out_dtype=out_dtype,
            scale_a=A_scale,
            scale_b=B_scale,
            bias=bias,
        )
        if need_reshape:
            output = output.reshape(
                batch_size, output.shape[0] // batch_size, output.shape[1]
            )
    else:
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale.to(out_dtype),
            B.to(out_dtype) * B_scale.to(out_dtype),
            bias=bias,
        )
    return output


class FP8DynamicLinear(LinearBase):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool,
        dev="cuda:0",
        dtype=torch.bfloat16,
        qdtype=torch.float8_e4m3fn,
        per_tensor=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qdtype = qdtype
        # 和register buff 同作用。 
        # 大部分情况下都用register buff。 除非权重不固定
        self.weight = torch.nn.Parameter(
            torch.randn(
                (self.out_features, self.in_features),
                dtype=dtype,
                device=dev,
                requires_grad=False,
            )
        )

        self.per_tensor = per_tensor
        if self.per_tensor:
            # 整个tensor一个scale
            self.weight_scale = torch.nn.Parameter(
                torch.randn((1,), dtype=torch.float32, device=dev, requires_grad=False)
            )
        else:  # per channel
            # channel 指 in features 维度。 每个输出维度一个scale
            # 所以共 out_features 个scale，shape 是 (out_features, 1)，方便后续广播
            self.weight_scale = torch.nn.Parameter(
                torch.randn(
                    (self.out_features, 1),
                    dtype=torch.float32,
                    device=dev,
                    requires_grad=False,
                )
            )
        # bias 不量化，保持fp32
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(
                    (self.out_features,),
                    dtype=dtype,
                    requires_grad=False,
                    device=dev,
                )
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, module: torch.nn.Linear, per_tensor=True, group_size=0):
        assert (
            group_size == 0
        ), "not support group wise fp8 quant yet! pls set group_size=0"
        dev = module.weight.device
        fp8_dynamic_linear = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dev=dev,
            per_tensor=per_tensor,
        )
        # 这里无论选择 PerTensor 或者 PerChannel，weight scale 的 shape 都得和
        # register buffer 时定义的 weight_scale 对上
        if module.bias is not None:
            fp8_dynamic_linear.bias.data = module.bias.clone()
        if per_tensor:
            fp8_weight, weight_scale = per_tensor_quantize(module.weight)
            weight_scale = weight_scale.to(dev)
            fp8_dynamic_linear.weight_scale.data = torch.tensor(
                weight_scale, device=dev
            )
        else:  # per channel
            fp8_weight, weight_scale = per_channel_quantize(module.weight)
            weight_scale = weight_scale.to(dev)
            fp8_dynamic_linear.weight_scale.data = weight_scale.detach().clone()
        # fp8 weight 在此处还是 bf16 type fp8 val，fwd 时再 to 为 fp8
        # shape 是 (out_features, in_features)
        fp8_dynamic_linear.weight.data = fp8_weight
        return fp8_dynamic_linear

    def forward(self, x):
        # scale is computed in runtime, so naming dyn
        if self.per_tensor:
            qinput, x_scale = per_tensor_quantize(x)
        else:
            #activation的 per token，顺便复用了w的per channel函数
            # 因为刚好都是dim=-1, w.shape [out,in], activation.shape [bs,in]
            # x的shape为 [bs, in]
            qinput, x_scale = per_channel_quantize(x)  # x_scale.shape = [m] keep dim -> [m, 1]
            self.weight_scale = self.weight_scale.t()  # weight scale need to transpose
            # 此时 torch_scaled_mm 视角下:
            # A=[m,in] B.t()=[in,out] A_scale=[m,1] B_scale=[1,out]

        # 进入kernel前，存的fp16转成fp8
        self.weight = self.weight.to(self.qdtype)
        # fp8 rowwise gemm
        output = fp8_gemm(
            A=qinput, # [bs, in]
            A_scale=x_scale, # [m, 1]
            B=self.weight, # [out, in]
            B_scale=self.weight_scale, # [1, out]
            bias=self.bias, # [out, 1]
            out_dtype=x.dtype,
        )
        return output
