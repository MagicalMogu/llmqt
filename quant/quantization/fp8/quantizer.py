import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
from tqdm import tqdm

from quant.nn_models.modules.linear import get_concrete_linear_module
from quant.nn_models.modules.linear.linear_fp8 import FP8StaticLinearQuantizer
from quant.utils.common_utils import (
    append_str_prefix,
    clear_memory,
    exclude_layers_to_not_quantize,
    get_best_device,
    get_named_linears,
    get_op_name,
    set_op_by_name,
)
from quant.utils.awq_clib_utils import get_calib_dataset
from quant.quantization.base.quantizer import BaseQuantizer


logger = logging.getLogger(__name__)


def replace_module(module, name, new_module):
    set_op_by_name(module, name, new_module)


def prepare_calib_tokens(tokenizer, device, max_calib_samples, max_calib_seq_len, calib_data="pileval"):
    calib_samples = get_calib_dataset(
        data=calib_data,
        tokenizer=tokenizer,
        n_samples=max_calib_samples,
        max_seq_len=max_calib_seq_len,
        split="validation",
    )
    if len(calib_samples) == 0:
        return torch.empty((0, max_calib_seq_len), dtype=torch.long, device=device)
    return torch.cat(calib_samples, dim=0).to(device)


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

        logger.info("FP8 quantizing one layer on %s", target_device)
        for name, linear in named_linears.items():
            if (
                not isinstance(linear, torch.nn.Linear)
                or name in quant_config.modules_to_not_convert
            ):
                print("=== skipping ", name)
                continue
            print("=== Dynamic Quantizing ", name)
            q_linear = dynamic_quant_linear.from_linear(
                linear, per_tensor=quant_config.per_tensor_quant
            )
            replace_module(layer, name, q_linear)
            del linear

        layer = layer.cpu()
        clear_memory()
        return layer

    def quantize_layer_on_device_thread_safe(
        self,
        layer,
        device_idx,
        quant_config,
        dynamic_quant_linear=None,
    ):
        """线程安全的量化函数"""
        try:
            # 关键：在每个线程中明确设置CUDA设备
            torch.cuda.set_device(device_idx)

            # 将layer数据移动到对应的GPU
            layer = layer.to(f"cuda:{device_idx}")

            named_modules = get_named_linears(layer)

            for name, linear in named_modules.items():
                if (
                    not isinstance(linear, torch.nn.Linear)
                    or name in quant_config.modules_to_not_convert
                ):
                    print(f"=== Device {device_idx}: skipping {name}")
                    continue

                print(f"=== Device {device_idx}: Dynamic Quantizing {name}")

                q_linear = dynamic_quant_linear.from_linear(
                    linear,
                    per_tensor=quant_config.per_tensor_quant
                )

                replace_module(layer, name, q_linear)
                del linear.weight
                del linear.bias
                del linear

            # 量化完成后移回CPU
            layer.cpu()
            clear_memory()
            return layer
        except Exception as e:
            print(f"[error] device {device_idx} quantization failed: {e}")
            raise

    # 修改主循环部分
    def parallel_quantize_layers(self):
        """使用线程池进行多卡并行量化"""
        layers = self.modelforCausalLM.get_model_layers(self.model)
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_layers = len(layers)

        # 准备参数：每个layer分配到不同的设备
        tasks = []
        for i, layer in enumerate(layers):
            device_idx = i % num_devices  # 轮询分配设备
            tasks.append((layer, device_idx))

        results = [None] * len(tasks)  # 预分配结果列表
        completed_count = 0

        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(
                    self.quantize_layer_on_device_thread_safe,
                    layer,
                    device_idx,
                    self.quant_config,
                    self.dynamic_quant_linear,
                ): i
                for i, (layer, device_idx) in enumerate(tasks)
            }

            # 使用tqdm显示进度
            with tqdm(total=len(tasks), desc="FP8 Quantizing weights in parallel") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                        completed_count += 1
                    except Exception as e:
                        print(f"Error quantizing layer {index}: {e}")
                        results[index] = None
                    finally:
                        pbar.update(1)
                        pbar.set_postfix(completed=f"{completed_count}/{len(tasks)}")

        for i, layer in enumerate(results):
            if layer is not None:
                layers[i] = layer

        calib_tokens = prepare_calib_tokens(
            self.tokenizer,
            self.device,
            self.max_calib_samples,
            self.max_calib_seq_len,
            calib_data=self.calib_data,
        )

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
        return results

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
        # 1 准备calibration
        # Replace weight quantizer with a dynamic activation quantizer observer
        for name, dynamic_quant_linear in self.model.named_modules():
            if (
                not isinstance(dynamic_quant_linear, self.dynamic_quant_linear)
                or name in quant_config.modules_to_not_convert
            ):
                continue
            quantizer = FP8StaticLinearQuantizer(
                in_features=dynamic_quant_linear.in_features,
                out_features=dynamic_quant_linear.out_features,
                qdtype=dynamic_quant_linear.qdtype,
                weight=dynamic_quant_linear.weight,
                weight_scale=dynamic_quant_linear.weight_scale,
                bias=dynamic_quant_linear.bias,
                quantize_output=(
                    hasattr(quant_config, "kv_cache_quant_layers")
                    and name in quant_config.kv_cache_quant_layers
                ),
            )
            replace_module(self.model, name, quantizer)

        #2  calibration，也就是用数据跑一边前向
        # Pass through calibration data to measure activation scales
        self.model.to(self.device)
        with torch.inference_mode():
            with tqdm(
                total=calib_tokens.shape[0], desc="Calibrating activation scales"
            ) as pbar:
                for row_idx in range(calib_tokens.shape[0]):
                    self.model(calib_tokens[row_idx].reshape(1, -1))
                    clear_memory()
                    pbar.update(1)

        # 3.用真正的FP8StaticLinear替换掉FP8StaticLinearQuantizer
        # 也就是求到scale后，现在完成静态量化，把权重和激活的scale固化在一起
        static_quant_linear = get_concrete_linear_module("fp8_static_quant")
        # Replace dynamic quantizer observer with StaticLinear for export
        for name, quantizer in self.model.named_modules():
            if (
                not isinstance(quantizer, FP8StaticLinearQuantizer)
                or name in quant_config.modules_to_not_convert
            ):
                print("=== skipping ", name)
                continue
            print("=== static Quantizing ", name)
            static_proj = static_quant_linear.from_linear(
                in_features=quantizer.in_features,
                out_features=quantizer.out_features,
                fp8_weight=quantizer.qweight,
                input_scale=quantizer.input_scale,
                weight_scales=quantizer.weight_scale,
                bias=quantizer.bias,
                output_scale=quantizer.output_scale,
                quantize_output=(
                    hasattr(quant_config, "kv_cache_quant_layers")
                    and name in quant_config.kv_cache_quant_layers
                ),
            )
            replace_module(self.model, name, static_proj)
            del quantizer
        # 4. 量化 kv cache
        # store kv cache quant scale in the parent attention module as `k_scale` and `v_scale`
        if quant_config.kv_cache_quant_layers:
            # Assumes that list is ordered such that [layer0.k_proj, layer0.v_proj, layer1.k_proj, layer1.v_proj, ...]
            # so we make a list of tuples [(layer0.k_proj, layer0.v_proj), (layer1.k_proj, layer1.v_proj), ...]
            kv_proj_pairs = zip(*[iter(quant_config.kv_cache_quant_layers)] * 2)

            for k_proj_name, v_proj_name in kv_proj_pairs:
                parent_module_name = ".".join(k_proj_name.split(".")[:-1])
                assert parent_module_name == ".".join(v_proj_name.split(".")[:-1])
                parent_module = dict(self.model.named_modules())[parent_module_name]

                k_proj = dict(self.model.named_modules())[k_proj_name]
                v_proj = dict(self.model.named_modules())[v_proj_name]
                # !! 核心: 量化v在于把kv cache scale保存到k proj和v proj的parent module的属性中
                parent_module.k_scale = torch.nn.Parameter(
                    k_proj.output_scale, requires_grad=False
                )
                parent_module.v_scale = torch.nn.Parameter(
                    v_proj.output_scale, requires_grad=False
                )

                # Remove output_scale from k_proj and v_proj
                k_proj.output_scale = None
                v_proj.output_scale = None
        clear_memory()
