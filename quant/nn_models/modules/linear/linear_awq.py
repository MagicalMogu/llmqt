import torch
import torch.nn as nn
from quant.utils.packing_utils import (
    dequantize_gemm,
)

from .linear_base import LinearBase

class AWQLinear_GEMM(LinearBase):

    def __init__(
        self,
        w_bit: int,
        group_size: int,
        in_features,
        out_features,
        bias,
        device
    ):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features

        self.register_buffer(
            "qweight",  # [in_features, out_features // (32 // w_bit)]
            torch.zeros(
                (self.in_features, self.out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=device,
            ),
        )
        # 这里的zeros和scales 是pseudo里面，对称/非对称量化里流出来的参数
        # 本来就是psesudo的一个zero point，int4范围的一个伪整数
        # 注意不要和awq里搜出来的scales弄混
        # 搜clip 和 pseudo量化的时候，都有均值 最大 最小这样的指标，都是按group划分的
        # group在 in features上进行划分
        self.register_buffer(
            "qzeros",  # [in_features // group_size, out_features // (32 // w_bit)]
            torch.zeros(
                (
                    self.in_features // self.group_size,
                    self.out_features // (32 // self.w_bit),
                ),
                dtype=torch.int32,
                device=device,
            ),
        )
        # 解码要用，必须保留fp16，对精度要求很高
        self.register_buffer(
            "scales",  # [in_features // group_size, out_features]
            torch.zeros(
                (self.in_features // self.group_size, self.out_features),
                dtype=torch.float16,
                device=device,
            ),
        )
        # 意义不大不量化
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (self.out_features,),
                    dtype=torch.float16,
                    device=device,
                ),
            )
        else:
            self.bias = None


    @classmethod
    def from_linear(
        cls, 
        linear: nn.Linear, 
        w_bit: int,
        group_size: int,
        init_only: bool = False,
        scales = None,
        zeros = None
    ):
        awq_linear = cls(
            w_bit=w_bit,
            group_size=group_size,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device
        )

        if init_only:
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None

        # 1. 开始量化 到 int4
        # 非对称量化里: qx = round(x / scale + zp) <=> round((x + scale*zp) / scale)
        # 减少重复乘法和广播处理，后面循环里直接 + 
        scale_zeros = zeros * scales

        awq_linear.scales.copy_(scales.clone().half())
        if linear.bias is not None:
            awq_linear.bias.copy_(linear.bias.clone().half())

        pack_num = 32 // awq_linear.w_bit # 32 // 4 = 8 每8个int4打包成一个int32

        # scales*zeros: [in_features // group_size, out_features]
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                # linear.weight.data [out_features, in_features] -> [out_features]
                # 每行是一个输出通道的权重
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // awq_linear.group_size])
                / awq_linear.scales[idx // awq_linear.group_size]
                ).to(torch.int)[:, None]
            )
        # 把list cat成一个tensor, 最后shape [out_features, in_features]
        intweight = torch.cat(intweight, dim=1) 

        # 这里为了方便pack成int32，先转置成 [in_features, out_features]，后续按照行来打包
        intweight = intweight.t().contiguous()  # [in_features, out_features]
        intweight = torch.clamp(intweight, 0, (1 << awq_linear.w_bit) - 1).to(torch.int32)

        # 2. pack 8个int4成1个int32，按order_map重排每组pack_num个元素
        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // pack_num),
            dtype=torch.int32,
            device=intweight.device,
        )
        #
        #  重排的意义是让量化后的权重在内存中更友好，减少访问时的位移和掩码操作，提升计算效率
        if awq_linear.w_bit == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        for col in range(intweight.shape[1] // pack_num):
            # 核心就是个位对齐，按照order_map的顺序把每组pack_num个元素打包成一个int32
            for i in range(pack_num): # 0-7
                # 找到真正的数据列
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                # 放在该放的 col的 第 i 个槽位上
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight.copy_(qweight)

        # zeros: [in_features // group_size, out_features]
        intzeros = torch.clamp(zeros, 0, (1 << awq_linear.w_bit) - 1).to(torch.int32)
        # qzeros: [in_features // group_size, out_features // pack_num]
        # 与 qweight 使用同一套 order_map，保持打包规则一致
        qzeros = torch.zeros(
            (intzeros.shape[0], intzeros.shape[1] // pack_num),
            dtype=torch.int32,
            device=intzeros.device,
        )
        for col in range(intzeros.shape[1] // pack_num):
            for i in range(pack_num):
                qzeros_col = intzeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzeros_col << (i * awq_linear.w_bit)
        awq_linear.qzeros.copy_(qzeros)

        return awq_linear

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        x = x.to(torch.float16)
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        out = dequantize_gemm(
            self.qweight,
            self.qzeros,
            self.scales,
            self.w_bit,
            self.group_size,
        )
        out = torch.matmul(x, out)
        out = out + self.bias if self.bias is not None else out

        # if input_dtype != torch.float16:
        #     out = out.to(dtype=input_dtype)

        return out.reshape(out_shape)
