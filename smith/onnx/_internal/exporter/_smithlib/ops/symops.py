"""Implementation for smith.sym* ops."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001,TCH002,TC003

from __future__ import annotations

from collections.abc import Sequence

from onnxscript.onnx_opset import opset18 as op

import smith
from smith.onnx._internal.exporter._smithlib._tensor_typing import (
    BOOL,
    FLOAT,
    IntType,
    TensorType,
    TTensor,
)
from smith.onnx._internal.exporter._smithlib._smithlib_registry import onnx_impl


@onnx_impl(smith.sym_float, trace_only=True)
def sym_float(self: TensorType) -> FLOAT:
    """sym_float(SymInt self) -> SymFloat"""
    return op.Cast(self, to=FLOAT.dtype)


@onnx_impl(smith.sym_max, trace_only=True)
def sym_max(x: IntType, y: IntType) -> IntType:
    """sym_max(SymInt x, SymInt y) -> SymInt"""
    return op.Max(x, y)


@onnx_impl(smith.sym_min, trace_only=True)
def sym_min(x: IntType, y: IntType) -> IntType:
    """sym_min(SymInt x, SymInt y) -> SymInt"""
    return op.Min(x, y)


@onnx_impl(smith.sym_not, trace_only=True)
def sym_not(self: BOOL) -> BOOL:
    """sym_not(SymBool self) -> SymBool"""
    return op.Not(self)


@onnx_impl(smith.sym_sum, trace_only=True)
def sym_sum(args: Sequence[IntType]) -> IntType:
    """sym_sum(SymInt[] args) -> SymInt"""
    if len(args) == 0:
        return op.Constant(value_int=0)
    if len(args) == 1:
        return args[0]
    result = op.Add(args[0], args[1])
    for i in range(2, len(args)):
        result = op.Add(result, args[i])
    return result


@onnx_impl(smith.sym_ite, trace_only=True)
def sym_ite(b: BOOL, t: TTensor, f: TTensor) -> TTensor:
    """sym_ite(SymBool b, Tensor t, Tensor f) -> Tensor"""
    return op.Where(b, t, f)
