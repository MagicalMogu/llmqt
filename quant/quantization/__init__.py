from importlib import import_module
from typing import Dict, Type

from .base.quantizer import BaseQuantizer

# 量化方法名到具体实现类的注册表。
# BaseModelForCausalLM 只依赖这个入口，不需要了解 AWQ/SQ/FP8 的实现细节。
_METHOD_TO_QUANTIZER: Dict[str, str] = {
    "awq": "quant.quantization.awq.quantizer.AwqQuantizer",
    "sq": "quant.quantization.sq.quantizer.SqQuantizer",
    "fp8_dynamic_quant": "quant.quantization.fp8.quantizer.Fp8Quantizer",
    "fp8_static_quant": "quant.quantization.fp8.quantizer.Fp8Quantizer",
}


def register_quantizer(method: str, class_path: str) -> None:
    """Register a quantizer implementation for a quantization method."""
    _METHOD_TO_QUANTIZER[method] = class_path


def get_supported_quant_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHOD_TO_QUANTIZER))


def get_concrete_quantizer_cls(quant_method: str) -> Type[BaseQuantizer]:
    """Resolve a quantization method string to its quantizer class."""
    try:
        module_path, class_name = _METHOD_TO_QUANTIZER[quant_method].rsplit(".", 1)
    except KeyError as exc:
        supported = ", ".join(get_supported_quant_methods())
        raise ValueError(
            f"Unsupported quantization method: {quant_method}. "
            f"Supported methods: {supported}"
        ) from exc

    module = import_module(module_path)
    try:
        quantizer_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Quantizer class '{class_name}' was not found in module '{module_path}' "
            f"for quantization method '{quant_method}'."
        ) from exc

    if not issubclass(quantizer_cls, BaseQuantizer):
        raise TypeError(
            f"{quantizer_cls.__name__} must inherit from BaseQuantizer, "
            f"but got {type(quantizer_cls)!r}."
        )

    return quantizer_cls


__all__ = [
    "BaseQuantizer",
    "get_concrete_quantizer_cls",
    "get_supported_quant_methods",
    "register_quantizer",
]
