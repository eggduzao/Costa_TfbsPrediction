"""Backward compatibility module for smith.onnx.symbolic_helper."""

from __future__ import annotations


__all__: list[str] = []

from smith.onnx._internal.smithscript_exporter.symbolic_helper import *  # noqa: F401,F403
