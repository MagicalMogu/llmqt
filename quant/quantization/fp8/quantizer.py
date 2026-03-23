import copy
import logging
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
from tqdm import tqdm

from quant.nn_models.modules.linear import get_concrete_linear_module
from quant.utils.common_utils import (
    append_str_prefix,
    clear_memory,
    exclude_layers_to_not_quantize,
    get_best_device,
    get_named_linears,
    get_op_name,
    set_op_by_name,
)
from quant.quantization.base.quantizer import BaseQuantizer


logger = logging.getLogger(__name__)


class Fp8Quantizer(BaseQuantizer):
    def __init__(
        self,
        modelforCausalLM,
        model,
        model_type,
        tokenizer,
        quant_config,
        quant_method,
        w_bit,
        group_size,
        zero_point,
        calib_data,
        duo_scaling,
        modules_to_not_convert=None,
        fake_quant=False,
        apply_clip=False,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        super(BaseQuantizer, self).__init__()
        self.modelforCausalLM = modelforCausalLM
        self.model = model
        self.model_type = model_type
        self.quant_method = quant_method
        self.tokenizer = tokenizer
        self.quant_config = quant_config
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.calib_data = calib_data
        self.duo_scaling = duo_scaling
        self.fake_quant = fake_quant
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )

        self.device = get_best_device()
        # 
        # 无论动静态量化都使用同一个线性模块，可复用
        # 静态的区别就在于，他会量化activation的量化参数并固化在权重里，而动态的则是在 forward 里根据输入动态计算量化参数。
        # 区别在于前者会在 forward 里根据输入动态计算量化参数，后者则在量化时就计算好并固化在权重里。
        self.dynamic_quant_linear = get_concrete_linear_module("fp8_dynamic_quant")
        # True代表始终多卡量化
        self.parallel = True

    def quantize_layer_on_device(
        self,
        layer,
        device_idx,
        quant_config,
        dynamic_quant_linear=None,
    ):
        """
        在指定设备上量化单层。

        这里先把截图里的接口骨架补齐，后续再继续补实际的
        calibration、replace 以及 pack 流程。
        """
        if dynamic_quant_linear is None:
            dynamic_quant_linear = self.dynamic_quant_linear

        if torch.cuda.is_available():
            target_device = f"cuda:{device_idx}"
        else:
            target_device = self.device

        layer = layer.to(target_device)
        named_linears = get_named_linears(layer)
        named_linears = exclude_layers_to_not_quantize(
            named_linears, self.modules_to_not_convert
        )

        logger.info("Preparing fp8 quantization on %s", target_device)
        return {
            "layer": layer,
            "device": target_device,
            "named_linears": named_linears,
            "quant_config": quant_config,
            "dynamic_quant_linear": dynamic_quant_linear,
        }

    # 修改主循环部分
    def parallel_quantize_layers(self):
        """TODO: implement parallel layer quantization."""
        pass

    # fake multi-gpu quantize
    def quantize(self):
        if self.parallel:
            self.parallel_quantize_layers()
        else:
            layers = self.modelforCausalLM.get_model_layers(self.model)
            calib_tokens = prepare_calib_tokens(
                self.tokenizer,
                self.device,
                self.max_calib_samples,
                self.max_calib_seq_len,
            )
            for i in tqdm(range(len(layers)), desc="FP8 Quantizing weights"):
                # 获取当前layer该被分到哪个device
                common_device = next(layers[i].parameters()).device
                if common_device is None or str(common_device) == "cpu":
                    if torch.cuda.is_available():
                        best_device = "cuda:" + str(i % torch.cuda.device_count())
                    else:
                        best_device = get_best_device()

                    layers[i] = layers[i].to(best_device)
                    common_device = next(layers[i].parameters()).device
                # 1.拿到当前层的所有linear 模块
                named_modules = get_named_linears(layers[i])
                for name, linear in named_modules.items():
                    if (
                        not isinstance(linear, torch.nn.Linear)
                        or name in self.quant_config.modules_to_not_convert
                    ):
                        print("=== skipping ", name)
                        continue
                    print("=== Dynamic Quantizing ", name)
                    # 初始化量化模块
                    # 主要是量化weight

                    q_linear = self.dynamic_quant_linear.from_linear(
                        linear, per_tensor=self.quant_config.per_tensor_quant
                    )
                    # 用q_linear替换原来的名字为name的linear
                    replace_module(layers[i], name, q_linear)
                    del linear.weight
                    del linear.bias
                    del linear
                layers[i].cpu()
                clear_memory()
            
            # 静态量化
            if self.quant_config.per_tensor_quant and (
                self.quant_method == "fp8_static_quant"
            ):
                self._apply_quant_act(self.quant_config, calib_tokens)
            else:
                print(
                    "[info] skip static quant, since per_tensor=False or quant method is not fp8_static_quant"
                )
            clear_memory()

    def _apply_quant_act(self, quant_config, calib_tokens):
        """TODO: implement activation-side quantization helpers."""
        pass
