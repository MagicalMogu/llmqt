import torch

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qweight.device)

    # unpacking columnwise
    # qweight [n, m, 1]
    # shifts -> [1, 1, 8]
    # iweight [n, m, 8] 每个int32被拆成8个int4，最后是 [n, m*8] 的形状
    # 通过位移和掩码操作把每个int32中的8个int4元素提取出来，得到 [n, m, 8] 的形状
    # int8是torch里最小的int
    iweights = torch.bitwise_right_shift(
        qweight[:, :, None], shifts[None, None, :]
    ).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(
            qzeros[:, :, None], shifts[None, None, :]
        ).to(torch.int8)
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros

    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1], dtype=torch.int32, device=iweights.device
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros


def dequantize_gemm(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # 还原回原始的权重顺序，恢复成 [out_channel, in_channel] 的形状
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # 取最后4位, 高位位移有脏数据
    iweight = torch.bitwise_and(iweight, (2 ** bits) - 1)
    izeros = torch.bitwise_and(izeros, (2 ** bits) - 1)

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    fp_weight = (iweight - izeros) * scales

    return fp_weight
