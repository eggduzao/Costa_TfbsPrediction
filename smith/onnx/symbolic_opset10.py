"""Backward compatibility module for smith.onnx.symbolic_opset10."""

from __future__ import annotations


__all__: list[str] = []

from smith.onnx._internal.smithscript_exporter.symbolic_opset10 import *  # noqa: F401,F403
from smith.onnx._internal.smithscript_exporter.symbolic_opset10 import (  # noqa: F401
    _slice,
)
