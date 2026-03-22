import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer

try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
except ImportError:
    LlamaDecoderLayer = ()
    LlamaRMSNorm = ()

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer,
        Qwen2RMSNorm,
    )
except ImportError:
    Qwen2DecoderLayer = ()
    Qwen2RMSNorm = ()


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]

    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        # ln的公式是 y = (x - mean) / sqrt(var + eps) * weight + bias
        # weight的shape是[hidden_dim], bias的shape也是[hidden_dim]
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device = fcs[0].weight.device
    dtype = fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    # w shape [hidden_dim or in_features, out_features]
    # 求每一列的max，得到shape [1, out_features]
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    # sm的公式
    # scale = act_scale^alpha / weight_scale^(1-alpha)
    # 这里的alpha是一个超参数，控制act_scale和weight_scale对最终scale的影响程度
    # alpha越大，act_scale的影响越大，反之weight_scale的影响越大。
    scales = (
        act_scales.pow(alpha) / weight_scales.pow(1 - alpha)
    ).clamp(min=1e-5)

    # w = W * scale
    # a = x / scale （放到prev op上，省去runtime的计算
    #  _都是原地操作，注意
    ln.weight.div_(scales)
    if ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]

    allowed_rms_norms = tuple(
        norm_cls for norm_cls in (LlamaRMSNorm, Qwen2RMSNorm) if norm_cls
    )
    assert allowed_rms_norms
    assert isinstance(ln, allowed_rms_norms)

    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device = fcs[0].weight.device
    dtype = fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        act_scales.pow(alpha) / weight_scales.pow(1 - alpha)
    ).clamp(min=1e-5)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, (LlamaDecoderLayer, Qwen2DecoderLayer)):
            attn_ln = module.input_layernorm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]
            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
