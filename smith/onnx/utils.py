"""Backward compatibility module for smith.onnx.utils."""

from __future__ import annotations


__all__: list[str] = []


from smith.onnx._internal.smithscript_exporter.utils import *  # noqa: F401,F403
