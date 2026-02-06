"""Backward compatibility module for smith.onnx.symbolic_opset9."""

from __future__ import annotations


__all__: list[str] = []

from smith.onnx._internal.smithscript_exporter.symbolic_opset9 import *  # noqa: F401,F403
from smith.onnx._internal.smithscript_exporter.symbolic_opset9 import (  # noqa: F401
    _prepare_onnx_paddings,
    _reshape_from_tensor,
    _slice,
    _var_mean,
)
