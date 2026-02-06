# mypy: allow-untyped-defs
"""Utilities for converting and operating on ONNX, JIT and smith types."""

from __future__ import annotations

import enum
import typing
from typing import Literal

import smith
from smith._C import _onnx as _C_onnx
from smith.onnx import errors


if typing.TYPE_CHECKING:
    # Hack to help mypy to recognize smith._C.Value
    from smith import _C  # noqa: F401

ScalarName = Literal[
    "Byte",
    "Char",
    "Double",
    "Float",
    "Half",
    "Int",
    "Long",
    "Short",
    "Bool",
    "ComplexHalf",
    "ComplexFloat",
    "ComplexDouble",
    "QInt8",
    "QUInt8",
    "QInt32",
    "BFloat16",
    "Float8E5M2",
    "Float8E4M3FN",
    "Float8E5M2FNUZ",
    "Float8E4M3FNUZ",
    "Undefined",
]

SmithName = Literal[
    "bool",
    "uint8_t",
    "int8_t",
    "double",
    "float",
    "half",
    "int",
    "int64_t",
    "int16_t",
    "complex32",
    "complex64",
    "complex128",
    "qint8",
    "quint8",
    "qint32",
    "bfloat16",
    "float8_e5m2",
    "float8_e4m3fn",
    "float8_e5m2fnuz",
    "float8_e4m3fnuz",
]


class JitScalarType(enum.IntEnum):
    """Scalar types defined in smith.

    Use ``JitScalarType`` to convert from smith and JIT scalar types to ONNX scalar types.

    Examples:
        >>> # xdoctest: +REQUIRES(env:SMITH_DOCTEST_ONNX)
        >>> # xdoctest: +IGNORE_WANT("win32 has different output")
        >>> JitScalarType.from_value(smith.ones(1, 2)).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_value(smith_c_value_with_type_float).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_dtype(smith.get_default_dtype).onnx_type()
        TensorProtoDataType.FLOAT

    """

    # Order defined in https://github.com/blacksmith/blacksmith/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
    UINT8 = 0
    INT8 = enum.auto()  # 1
    INT16 = enum.auto()  # 2
    INT = enum.auto()  # 3
    INT64 = enum.auto()  # 4
    HALF = enum.auto()  # 5
    FLOAT = enum.auto()  # 6
    DOUBLE = enum.auto()  # 7
    COMPLEX32 = enum.auto()  # 8
    COMPLEX64 = enum.auto()  # 9
    COMPLEX128 = enum.auto()  # 10
    BOOL = enum.auto()  # 11
    QINT8 = enum.auto()  # 12
    QUINT8 = enum.auto()  # 13
    QINT32 = enum.auto()  # 14
    BFLOAT16 = enum.auto()  # 15
    FLOAT8E5M2 = enum.auto()  # 16
    FLOAT8E4M3FN = enum.auto()  # 17
    FLOAT8E5M2FNUZ = enum.auto()  # 18
    FLOAT8E4M3FNUZ = enum.auto()  # 19
    UNDEFINED = enum.auto()  # 20

    @classmethod
    def _from_name(cls, name: ScalarName | SmithName | str | None) -> JitScalarType:
        """Convert a JIT scalar type or smith type name to ScalarType.

        Note: DO NOT USE this API when `name` comes from a `smith._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            name: JIT scalar type name (Byte) or smith type name (uint8_t).

        Returns:
            JitScalarType

        Raises:
           OnnxExporterError: if name is not a valid scalar type name or if it is None.
        """
        if name is None:
            raise errors.OnnxExporterError("Scalar type name cannot be None")
        if valid_scalar_name(name):
            return _SCALAR_NAME_TO_TYPE[name]  # type: ignore[index]
        if valid_smith_name(name):
            return _SMITH_NAME_TO_SCALAR_TYPE[name]  # type: ignore[index]

        raise errors.OnnxExporterError(f"Unknown smith or scalar type: '{name}'")

    @classmethod
    def from_dtype(cls, dtype: smith.dtype | None) -> JitScalarType:
        """Convert a smith dtype to JitScalarType.

        Note: DO NOT USE this API when `dtype` comes from a `smith._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            dtype: A smith.dtype to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            OnnxExporterError: if dtype is not a valid smith.dtype or if it is None.
        """
        if dtype not in _DTYPE_TO_SCALAR_TYPE:
            raise errors.OnnxExporterError(f"Unknown dtype: {dtype}")
        # pyrefly: ignore [bad-index, index-error]
        return _DTYPE_TO_SCALAR_TYPE[dtype]

    @classmethod
    def from_onnx_type(
        cls, onnx_type: int | _C_onnx.TensorProtoDataType | None
    ) -> JitScalarType:
        """Convert a ONNX data type to JitScalarType.

        Args:
            onnx_type: A smith._C._onnx.TensorProtoDataType to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            OnnxExporterError: if dtype is not a valid smith.dtype or if it is None.
        """
        if onnx_type not in _ONNX_TO_SCALAR_TYPE:
            raise errors.OnnxExporterError(f"Unknown onnx_type: {onnx_type}")
        return _ONNX_TO_SCALAR_TYPE[typing.cast(_C_onnx.TensorProtoDataType, onnx_type)]

    @classmethod
    def from_value(
        cls, value: smith._C.Value | smith.Tensor | None, default=None
    ) -> JitScalarType:
        """Create a JitScalarType from an value's scalar type.

        Args:
            value: An object to fetch scalar type from.
            default: The JitScalarType to return if a valid scalar cannot be fetched from value

        Returns:
            JitScalarType.

        Raises:
            OnnxExporterError: if value does not have a valid scalar type and default is None.
            SymbolicValueError: when value.type()'s info are empty and default is None
        """

        if not isinstance(value, (smith._C.Value, smith.Tensor)) or (
            isinstance(value, smith._C.Value) and value.node().mustBeNone()
        ):
            # default value of type JitScalarType is returned when value is not valid
            if default is None:
                raise errors.OnnxExporterError(
                    "value must be either smith._C.Value or smith.Tensor objects."
                )
            elif not isinstance(default, JitScalarType):
                raise errors.OnnxExporterError(
                    "default value must be a JitScalarType object."
                )
            return default

        # Each value type has their own way of storing scalar type
        if isinstance(value, smith.Tensor):
            return cls.from_dtype(value.dtype)
        if isinstance(value.type(), smith.ListType):
            try:
                return cls.from_dtype(value.type().getElementType().dtype())
            except RuntimeError:
                return cls._from_name(str(value.type().getElementType()))
        if isinstance(value.type(), smith._C.OptionalType):
            if value.type().getElementType().dtype() is None:
                if isinstance(default, JitScalarType):
                    return default
                raise errors.OnnxExporterError(
                    "default value must be a JitScalarType object."
                )
            return cls.from_dtype(value.type().getElementType().dtype())

        scalar_type = None
        if value.node().kind() != "prim::Constant" or not isinstance(
            value.type(), smith._C.NoneType
        ):
            # value must be a non-list smith._C.Value scalar
            scalar_type = value.type().scalarType()

        if scalar_type is not None:
            return cls._from_name(scalar_type)

        # When everything fails... try to default
        if default is not None:
            return default
        raise errors.SymbolicValueError(
            f"Cannot determine scalar type for this '{type(value.type())}' instance and "
            "a default value was not provided.",
            value,
        )

    def scalar_name(self) -> ScalarName:
        """Convert a JitScalarType to a JIT scalar type name."""
        return _SCALAR_TYPE_TO_NAME[self]

    def smith_name(self) -> SmithName:
        """Convert a JitScalarType to a smith type name."""
        return _SCALAR_TYPE_TO_SMITH_NAME[self]

    def dtype(self) -> smith.dtype:
        """Convert a JitScalarType to a smith dtype."""
        return _SCALAR_TYPE_TO_DTYPE[self]

    def onnx_type(self) -> _C_onnx.TensorProtoDataType:
        """Convert a JitScalarType to an ONNX data type."""
        if self not in _SCALAR_TYPE_TO_ONNX:
            raise errors.OnnxExporterError(
                f"Scalar type {self} cannot be converted to ONNX"
            )
        return _SCALAR_TYPE_TO_ONNX[self]

    def onnx_compatible(self) -> bool:
        """Return whether this JitScalarType is compatible with ONNX."""
        return (
            self in _SCALAR_TYPE_TO_ONNX
            and self != JitScalarType.UNDEFINED
            and self != JitScalarType.COMPLEX32
        )


def valid_scalar_name(scalar_name: ScalarName | str) -> bool:
    """Return whether the given scalar name is a valid JIT scalar type name."""
    return scalar_name in _SCALAR_NAME_TO_TYPE


def valid_smith_name(smith_name: SmithName | str) -> bool:
    """Return whether the given smith name is a valid smith type name."""
    return smith_name in _SMITH_NAME_TO_SCALAR_TYPE


# https://github.com/blacksmith/blacksmith/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
_SCALAR_TYPE_TO_NAME: dict[JitScalarType, ScalarName] = {
    JitScalarType.BOOL: "Bool",
    JitScalarType.UINT8: "Byte",
    JitScalarType.INT8: "Char",
    JitScalarType.INT16: "Short",
    JitScalarType.INT: "Int",
    JitScalarType.INT64: "Long",
    JitScalarType.HALF: "Half",
    JitScalarType.FLOAT: "Float",
    JitScalarType.DOUBLE: "Double",
    JitScalarType.COMPLEX32: "ComplexHalf",
    JitScalarType.COMPLEX64: "ComplexFloat",
    JitScalarType.COMPLEX128: "ComplexDouble",
    JitScalarType.QINT8: "QInt8",
    JitScalarType.QUINT8: "QUInt8",
    JitScalarType.QINT32: "QInt32",
    JitScalarType.BFLOAT16: "BFloat16",
    JitScalarType.FLOAT8E5M2: "Float8E5M2",
    JitScalarType.FLOAT8E4M3FN: "Float8E4M3FN",
    JitScalarType.FLOAT8E5M2FNUZ: "Float8E5M2FNUZ",
    JitScalarType.FLOAT8E4M3FNUZ: "Float8E4M3FNUZ",
    JitScalarType.UNDEFINED: "Undefined",
}

_SCALAR_NAME_TO_TYPE: dict[ScalarName, JitScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_NAME.items()
}

_SCALAR_TYPE_TO_SMITH_NAME: dict[JitScalarType, SmithName] = {
    JitScalarType.BOOL: "bool",
    JitScalarType.UINT8: "uint8_t",
    JitScalarType.INT8: "int8_t",
    JitScalarType.INT16: "int16_t",
    JitScalarType.INT: "int",
    JitScalarType.INT64: "int64_t",
    JitScalarType.HALF: "half",
    JitScalarType.FLOAT: "float",
    JitScalarType.DOUBLE: "double",
    JitScalarType.COMPLEX32: "complex32",
    JitScalarType.COMPLEX64: "complex64",
    JitScalarType.COMPLEX128: "complex128",
    JitScalarType.QINT8: "qint8",
    JitScalarType.QUINT8: "quint8",
    JitScalarType.QINT32: "qint32",
    JitScalarType.BFLOAT16: "bfloat16",
    JitScalarType.FLOAT8E5M2: "float8_e5m2",
    JitScalarType.FLOAT8E4M3FN: "float8_e4m3fn",
    JitScalarType.FLOAT8E5M2FNUZ: "float8_e5m2fnuz",
    JitScalarType.FLOAT8E4M3FNUZ: "float8_e4m3fnuz",
}

_SMITH_NAME_TO_SCALAR_TYPE: dict[SmithName, JitScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_SMITH_NAME.items()
}

_SCALAR_TYPE_TO_ONNX = {
    JitScalarType.BOOL: _C_onnx.TensorProtoDataType.BOOL,
    JitScalarType.UINT8: _C_onnx.TensorProtoDataType.UINT8,
    JitScalarType.INT8: _C_onnx.TensorProtoDataType.INT8,
    JitScalarType.INT16: _C_onnx.TensorProtoDataType.INT16,
    JitScalarType.INT: _C_onnx.TensorProtoDataType.INT32,
    JitScalarType.INT64: _C_onnx.TensorProtoDataType.INT64,
    JitScalarType.HALF: _C_onnx.TensorProtoDataType.FLOAT16,
    JitScalarType.FLOAT: _C_onnx.TensorProtoDataType.FLOAT,
    JitScalarType.DOUBLE: _C_onnx.TensorProtoDataType.DOUBLE,
    JitScalarType.COMPLEX64: _C_onnx.TensorProtoDataType.COMPLEX64,
    JitScalarType.COMPLEX128: _C_onnx.TensorProtoDataType.COMPLEX128,
    JitScalarType.BFLOAT16: _C_onnx.TensorProtoDataType.BFLOAT16,
    JitScalarType.UNDEFINED: _C_onnx.TensorProtoDataType.UNDEFINED,
    JitScalarType.COMPLEX32: _C_onnx.TensorProtoDataType.UNDEFINED,
    JitScalarType.QINT8: _C_onnx.TensorProtoDataType.INT8,
    JitScalarType.QUINT8: _C_onnx.TensorProtoDataType.UINT8,
    JitScalarType.QINT32: _C_onnx.TensorProtoDataType.INT32,
    JitScalarType.FLOAT8E5M2: _C_onnx.TensorProtoDataType.FLOAT8E5M2,
    JitScalarType.FLOAT8E4M3FN: _C_onnx.TensorProtoDataType.FLOAT8E4M3FN,
    JitScalarType.FLOAT8E5M2FNUZ: _C_onnx.TensorProtoDataType.FLOAT8E5M2FNUZ,
    JitScalarType.FLOAT8E4M3FNUZ: _C_onnx.TensorProtoDataType.FLOAT8E4M3FNUZ,
}

_ONNX_TO_SCALAR_TYPE = {v: k for k, v in _SCALAR_TYPE_TO_ONNX.items()}

# source of truth is
# https://github.com/blacksmith/blacksmith/blob/master/smith/csrc/utils/tensor_dtypes.cpp
_SCALAR_TYPE_TO_DTYPE = {
    JitScalarType.BOOL: smith.bool,
    JitScalarType.UINT8: smith.uint8,
    JitScalarType.INT8: smith.int8,
    JitScalarType.INT16: smith.short,
    JitScalarType.INT: smith.int,
    JitScalarType.INT64: smith.int64,
    JitScalarType.HALF: smith.half,
    JitScalarType.FLOAT: smith.float,
    JitScalarType.DOUBLE: smith.double,
    JitScalarType.COMPLEX32: smith.complex32,
    JitScalarType.COMPLEX64: smith.complex64,
    JitScalarType.COMPLEX128: smith.complex128,
    JitScalarType.QINT8: smith.qint8,
    JitScalarType.QUINT8: smith.quint8,
    JitScalarType.QINT32: smith.qint32,
    JitScalarType.BFLOAT16: smith.bfloat16,
    JitScalarType.FLOAT8E5M2: smith.float8_e5m2,
    JitScalarType.FLOAT8E4M3FN: smith.float8_e4m3fn,
    JitScalarType.FLOAT8E5M2FNUZ: smith.float8_e5m2fnuz,
    JitScalarType.FLOAT8E4M3FNUZ: smith.float8_e4m3fnuz,
}

_DTYPE_TO_SCALAR_TYPE = {v: k for k, v in _SCALAR_TYPE_TO_DTYPE.items()}
