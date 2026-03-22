import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm


def get_act_scales(model, tokenizer, calib_data, num_samples=512, seq_len=512, split="validation"):
    # device = next(model.parameters()).device
    device = "cuda:0"
    model.eval().to(device)
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    # 1.注册钩子函数到每个linear module
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    # 2.对模型做推理，forward
    if isinstance(calib_data, str):
        if calib_data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            dataset = load_dataset(calib_data, split=split)
    dataset = dataset.shuffle(seed=42)
    # dataset = load_dataset("json", data_files=dataset_path, split="train")
    # dataset = dataset.shuffle(seed=42)
    for i in tqdm(range(num_samples), desc="getting input act max to smooth"):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales
@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    tokenizer,
    data,
    num_samples=512,
    seq_len=512,
    split="validation",
):
    model.eval()
    device = next(model.parameters()).device
    print("get static decoder layer scales in ", device)
    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(y, tuple):
            y = y[0]

        x_max = x.detach().abs().max().item()
        y_max = y.detach().abs().max().item()
        
        # act_dict的内容大概为：
        # {
        #   "model.decoder.layers.0.self_attn.q_proj": {
        #       "input": 3.2,
        #       "output": 4.5,
        #   },
        if "input" not in act_dict[name]:
            act_dict[name]["input"] = x_max
        else:
            act_dict[name]["input"] = max(act_dict[name]["input"], x_max)

        if "output" not in act_dict[name]:
            act_dict[name]["output"] = y_max
        else:
            act_dict[name]["output"] = max(act_dict[name]["output"], y_max)

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(
                    functools.partial(stat_io_hook, name=name)
                )
            )

    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            dataset = load_dataset(data, split=split)
    else:
        dataset = data

    dataset = dataset.shuffle(seed=42)

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
        ).input_ids.to(device)
        model(input_ids)

        if act_dict:
            mean_scale = np.mean([v["input"] for v in act_dict.values() if "input" in v])
            pbar.set_description(f"Mean input scale: {mean_scale:.2f}")

    for hook in hooks:
        hook.remove()

    # 由此，得到每个linear的最小最大值
    # 接下来，基于最大最小值，求得scale，保存在scale_dict
    decoder_layer_scales = []
    is_opt = model.config.model_type == "opt"

    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}

        if is_opt:
            scale_dict["attn_input_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["input"] / 127
            )
            scale_dict["q_output_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["output"] / 127
            )
            scale_dict["k_output_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.self_attn.k_proj"]["output"] / 127
            )
            scale_dict["v_output_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.self_attn.v_proj"]["output"] / 127
            )
            scale_dict["out_input_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.self_attn.out_proj"]["input"] / 127
            )
            scale_dict["fc1_input_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.fc1"]["input"] / 127
            )
            scale_dict["fc2_input_scale"] = (
                act_dict[f"model.decoder.layers.{idx}.fc2"]["input"] / 127
            )
        else:
            q_proj_key = f"model.layers.{idx}.self_attn.q_proj"
            if q_proj_key not in act_dict:
                q_proj_key = f"model.model.layers.{idx}.self_attn.q_proj"

            k_proj_key = f"model.layers.{idx}.self_attn.k_proj"
            if k_proj_key not in act_dict:
                k_proj_key = f"model.model.layers.{idx}.self_attn.k_proj"

            v_proj_key = f"model.layers.{idx}.self_attn.v_proj"
            if v_proj_key not in act_dict:
                v_proj_key = f"model.model.layers.{idx}.self_attn.v_proj"

            o_proj_key = f"model.layers.{idx}.self_attn.o_proj"
            if o_proj_key not in act_dict:
                o_proj_key = f"model.model.layers.{idx}.self_attn.o_proj"

            gate_proj_key = f"model.layers.{idx}.mlp.gate_proj"
            if gate_proj_key not in act_dict:
                gate_proj_key = f"model.model.layers.{idx}.mlp.gate_proj"

            up_proj_key = f"model.layers.{idx}.mlp.up_proj"
            if up_proj_key not in act_dict:
                up_proj_key = f"model.model.layers.{idx}.mlp.up_proj"

            down_proj_key = f"model.layers.{idx}.mlp.down_proj"
            if down_proj_key not in act_dict:
                down_proj_key = f"model.model.layers.{idx}.mlp.down_proj"

            scale_dict["attn_input_scale"] = act_dict[q_proj_key]["input"] / 127
            scale_dict["q_output_scale"] = act_dict[q_proj_key]["output"] / 127
            scale_dict["k_output_scale"] = act_dict[k_proj_key]["output"] / 127
            scale_dict["v_output_scale"] = act_dict[v_proj_key]["output"] / 127
            scale_dict["out_input_scale"] = act_dict[o_proj_key]["input"] / 127
            scale_dict["gate_input_scale"] = act_dict[gate_proj_key]["input"] / 127
            scale_dict["gate_output_scale"] = act_dict[gate_proj_key]["output"] / 127
            scale_dict["up_input_scale"] = act_dict[up_proj_key]["input"] / 127
            scale_dict["up_output_scale"] = act_dict[up_proj_key]["output"] / 127
            scale_dict["down_input_scale"] = act_dict[down_proj_key]["input"] / 127
            scale_dict["down_output_scale"] = act_dict[down_proj_key]["output"] / 127
            scale_dict["fc1_input_scale"] = act_dict[gate_proj_key]["input"] / 127
            scale_dict["fc2_input_scale"] = act_dict[down_proj_key]["input"] / 127

        # 把每一层linear的input scale都存在于decoder_layer_scales=list中
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict
