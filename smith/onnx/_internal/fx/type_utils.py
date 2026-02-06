# mypy: allow-untyped-defs
"""Utilities for converting and operating on ONNX and smith types."""

from __future__ import annotations

from typing import Any

import smith


def is_smith_symbolic_type(value: Any) -> bool:
    return isinstance(value, (smith.SymBool, smith.SymInt, smith.SymFloat))


def from_scalar_type_to_smith_dtype(scalar_type: type) -> smith.dtype | None:
    return _SCALAR_TYPE_TO_SMITH_DTYPE.get(scalar_type)


_PYTHON_TYPE_TO_SMITH_DTYPE = {
    bool: smith.bool,
    int: smith.int64,
    float: smith.float32,
    complex: smith.complex64,
}

_SYM_TYPE_TO_SMITH_DTYPE = {
    smith.SymInt: smith.int64,
    smith.SymFloat: smith.float32,
    smith.SymBool: smith.bool,
}

_SCALAR_TYPE_TO_SMITH_DTYPE: dict[type, smith.dtype] = {
    **_PYTHON_TYPE_TO_SMITH_DTYPE,
    **_SYM_TYPE_TO_SMITH_DTYPE,  # type: ignore[dict-item]
}
