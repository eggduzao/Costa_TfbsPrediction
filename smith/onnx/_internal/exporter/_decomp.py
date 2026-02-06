# mypy: allow-untyped-defs
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import smith
import smith._ops


if TYPE_CHECKING:
    from collections.abc import Callable

    from smith.onnx._internal.exporter import _registration


def get_onnx_implemented_overloads(
    registry: _registration.ONNXRegistry,
) -> list[_registration.SmithOp]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported Blacksmith operations.

    Args:
        registry: The ONNX registry for Blacksmith.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported Blacksmith operations.
    """
    registered_ops: list[_registration.SmithOp] = []
    for onnx_decomp_meta in registry.functions.values():
        if len(onnx_decomp_meta) == 0:
            raise AssertionError("onnx_decomp_meta must not be empty")
        # Different OnnxDecompMeta for the same SmithOp should
        # have the same fx_target.
        fx_target = onnx_decomp_meta[0].fx_target
        registered_ops.append(fx_target)
    return registered_ops


def create_onnx_friendly_decomposition_table(
    onnx_registered_ops: set[_registration.SmithOp],
) -> dict[_registration.SmithOp, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of Blacksmith's
    built-in aten-to-aten decomposition.

    Args:
        onnx_registered_ops: All ops that have an ONNX decomposition implemented.

    Returns:
        Dict[smith._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    decomposition_table: dict[_registration.SmithOp, Callable] = {}

    for op_overload, decomp_fn in itertools.chain(
        smith.export.default_decompositions().items(),  # type: ignore[attr-defined]
        smith._decomp.decomposition_table.items(),  # type: ignore[attr-defined]
    ):
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        # NOTE: Do not skip smith._refs decomps. They are fine because otherwise the model is
        # not exportable anyways.
        if op_overload in onnx_registered_ops:
            continue
        # If it is HOP, we filter those out as well.
        if not hasattr(op_overload, "_schema"):
            continue
        # NOTE: smith._decomp.decomposition_table covers more ops
        # than smith.export.default_decompositions, but the latter is
        # more critical to smith.onnx.export.
        if op_overload in decomposition_table:
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table
