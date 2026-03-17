

import torch
import torch.nn as nn
from quant.utils.common_utils import get_op_by_name, get_op_name


@torch.no_grad()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    assert isinstance(fc1, nn.Linear) 
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)
    # prev op 提前div scales，数学上等价于activation / scales
    # 只改 fc1 的“最后若干个输出通道”
    # 因为 y = W1 X + b1
    # y/s = W1 X/s + b1/s = W1/s X + b1/s
    # scale: [out1] 原始shape
    #x          : [B, in1]
    # fc1.weight : [out1, in1]
    # fc1.bias  : [out1]
    # y = fc1(x) : [B, out1]
    # fc2.weight : [out2, out1]
    # z = fc2(y) : [B, out2]
    #
    # 这个[-scales.size(0):]是个安全冗余，不够干净，疑似cv的代码
    # 只会在project大投影qkv的时候有用
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales)
    # 真正的乘操作
    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0, f"NaN detected in {get_op_name(fc1)}"
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0, f"NaN detected in {get_op_name(fc2)}"

# 和老师写的不完全一样，自写版本
@torch.no_grad()
def scale_fc_fcs(fc: nn.Linear, fcs: list[nn.Linear], scales: torch.Tensor):
    assert isinstance(fc, nn.Linear)
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert all(isinstance(next_fc, nn.Linear) for next_fc in fcs)

    scales = scales.to(fc.weight.device)

    # 一个前置 Linear 的输出被多个后续 Linear 共同消费时，
    # 先把中间 activation 的 / scales 融到前置 fc，
    # 再把补偿项乘回每个后续 fc 的输入列。
    fc.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc.bias is not None:
        fc.bias.div_(scales)

    for next_fc in fcs:
        next_fc.weight.mul_(scales.view(1, -1))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0, f"NaN detected in {get_op_name(fc)}"
    for next_fc in fcs:
        for p in next_fc.parameters():
            assert torch.isnan(p).sum() == 0, f"NaN detected in {get_op_name(next_fc)}"


@torch.no_grad()
def scale_ln_fcs(ln: nn.Module, fcs: list[nn.Linear], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    # GemmaRMSNorm 的输出缩放是 x * (1 + weight)，
    # 所以要先临时还原成真正参与乘法的那部分参数再做 / scales。
    if isinstance(ln, GemmaRMSNorm) or isinstance(ln, Gemma2RMSNorm):
        ln.weight += 1
        ln.weight.div_(scales)
        ln.weight -= 1
    else:
        # layernorm = gamma * (x - mean) / sqrt(var + eps) + beta
        # gamma 是参与乘法的参数，所以要 / scales；beta 是参与加法
        ln.weight.div_(scales)

    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0



def apply_scale(module, scale_list, input_feat_dict=None):

    best_device = next(module.parameters()).device

    for prev_op, layer_names, scales in scale_list:
        # prev_op 和 layer_names 都是字符串，需要转换成实际的模块对象。
        # get op by name 和 get op name两个函数直接互相用了字符串传递模块，避免直接传递模块对象
        prev_op = get_op_by_name(module, prev_op)
        layers = [get_op_by_name(module, layer_name) for layer_name in layer_names]

        prev_op = prev_op.to(best_device)
        scales = scales.to(best_device)

        # 和get_layers_for_scaling搜索的子图一样
        # 需要分别判断需要哪种类型的scales

        if (
            isinstance(prev_op, nn.Linear)
            and type(layers) == list
            and isinstance(layers[0], nn.Linear)

        ):
            scale_fc_fcs(prev_op, layers, scales)

        elif isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            # 实际上prev_op == up_proj, layers[0] == down_proj
            scale_fc_fc(prev_op, layers[0], scales)
        
        elif (
            any(isinstance(prev_op, t) for t in allowed_norms)
            or "rmsnorm" in str(prev_op.__class__).lower()
        ):
            scale_ln_fcs(prev_op, layers, scales)
        
        else:
            raise NotImplementedError(
                f"Unsupported layer type for scaling: {type(prev_op)}"
            )

        if input_feat_dict is not None:
            for layer_name in layer_names:
                # skip the modules that are not quantized
                if layer_name in input_feat_dict:
                    inp = input_feat_dict[layer_name]
                    # inp.div_(scales.view(1, -1)).to(inp.device)  这特么写的啥，无意义
                    inp.div_(scales.view(1, -1))

        prev_op.to("cpu")
        for layer in layers:
            layer.to("cpu")
        scales.to("cpu")
