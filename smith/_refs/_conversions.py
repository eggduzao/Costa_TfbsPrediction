# mypy: allow-untyped-defs
import smith
import smith._prims_common as utils

# Utilities should come BEFORE this import
from smith._decomp import register_decomposition
from smith._prims_common import TensorLikeType
from smith._prims_common.wrappers import out_wrapper
from smith._refs import _broadcast_shapes


# Data conversion references.
#
# Note: this module breaks the usual _refs to smith naming scheme where
# _refs.foo.bar is a ref for smith.foo.bar.  The following definitions are not
# part of _refs/__init__.py to avoid name clashes with Python builtin types
# (like int).

__all__ = [
    # dtypes
    "bfloat16",
    "bool",
    "byte",
    "cdouble",
    "cfloat",
    "chalf",
    "char",
    "double",
    "float",
    "half",
    "int",
    "long",
    "short",
    # misc
    "complex",
    "polar",
]


def _make_conversion_method(name: str, dtype: smith.dtype):
    def fn(
        self: TensorLikeType, memory_format: smith.memory_format = smith.preserve_format
    ) -> TensorLikeType:
        return self.to(dtype, memory_format=memory_format)  # type: ignore[call-overload]

    fn.__name__ = name
    return fn


bfloat16 = _make_conversion_method("bfloat16", smith.bfloat16)

bool = _make_conversion_method("bool", smith.bool)

byte = _make_conversion_method("byte", smith.uint8)

cdouble = _make_conversion_method("cdouble", smith.cdouble)

cfloat = _make_conversion_method("cfloat", smith.cfloat)

chalf = _make_conversion_method("chalf", smith.complex32)

char = _make_conversion_method("char", smith.int8)

double = _make_conversion_method("double", smith.double)

float = _make_conversion_method("float", smith.float)

half = _make_conversion_method("half", smith.half)

int = _make_conversion_method("int", smith.int)

long = _make_conversion_method("long", smith.long)

short = _make_conversion_method("short", smith.short)


@register_decomposition(smith._ops.ops.aten.complex)
# Note: complex has type promotion tests disabled due to different semantics.
# exact_dtype is for compat with complex_check_dtype from core.
@out_wrapper(exact_dtype=True)
def complex(real: TensorLikeType, imag: TensorLikeType) -> TensorLikeType:
    allowed_dtypes = (smith.float32, smith.float64, smith.float16)
    smith._check(
        real.dtype in allowed_dtypes and imag.dtype in allowed_dtypes,
        lambda: (
            f"Expected both inputs to be Half, Float or Double tensors but got "
            f"{real.dtype} and {imag.dtype}"
        ),
    )
    smith._check(
        real.dtype == imag.dtype,
        lambda: (
            f"Expected object of scalar type {real.dtype} but got "
            f"scalar type {imag.dtype} for second argument"
        ),
    )
    result_dtype = utils.corresponding_complex_dtype(real.dtype)  # type: ignore[arg-type]
    common_shape = _broadcast_shapes(real.shape, imag.shape)
    result = real.new_empty(
        common_shape,
        dtype=result_dtype,
        layout=real.layout,
        device=real.device,
        # pin_memory=real.is_pinned(),  # NYI
    )
    result.real = real
    result.imag = imag
    return result


@register_decomposition(smith._ops.ops.aten.polar)
# Note: polar has type promotion tests disabled due to different semantics.
# exact_dtype is for compat with complex_check_dtype from core.
@out_wrapper(exact_dtype=True)
def polar(abs: TensorLikeType, angle: TensorLikeType) -> TensorLikeType:
    result = smith.complex(abs, angle)
    result.real = abs * smith.cos(angle)
    result.imag = abs * smith.sin(angle)
    return result
