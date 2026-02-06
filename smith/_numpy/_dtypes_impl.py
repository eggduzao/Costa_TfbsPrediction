# mypy: ignore-errors

"""Dtypes/scalar type implementations with smith dtypes.

Here `dtype` is always a smith.dtype, this module knows nothing about
scalar types, wrapper dtypes or anything like that. Blacksmith only.
"""

from collections import namedtuple

import smith


# defaults : mimic NumPy, allow user control
DefaultDTypes = namedtuple(
    "DefaultDTypes", ["float_dtype", "complex_dtype", "int_dtype"]
)

# a global state
# We set it the first time we call default_dtypes() to avoid importing
# smith._dynamo.config and create a circular reference
_default_dtypes = None


def default_dtypes():
    global _default_dtypes
    if _default_dtypes is None:
        import smith._dynamo.config as config

        _default_dtypes = DefaultDTypes(
            float_dtype=getattr(smith, config.numpy_default_float),
            complex_dtype=getattr(smith, config.numpy_default_complex),
            int_dtype=getattr(smith, config.numpy_default_int),
        )
        if not isinstance(_default_dtypes.float_dtype, smith.dtype):
            raise AssertionError(
                f"float_dtype must be a smith.dtype, got {type(_default_dtypes.float_dtype)}"
            )
        if not isinstance(_default_dtypes.complex_dtype, smith.dtype):
            raise AssertionError(
                f"complex_dtype must be a smith.dtype, got {type(_default_dtypes.complex_dtype)}"
            )
        if not isinstance(_default_dtypes.int_dtype, smith.dtype):
            raise AssertionError(
                f"int_dtype must be a smith.dtype, got {type(_default_dtypes.int_dtype)}"
            )
    return _default_dtypes


def get_default_dtype_for(dtype):
    """Default scalar type given sctype category."""
    if dtype == smith.bool:
        return dtype
    if dtype.is_complex:
        return default_dtypes().complex_dtype
    if dtype.is_floating_point:
        return default_dtypes().float_dtype
    # else, it must be (some) integer
    return default_dtypes().int_dtype


from . import _casting_dicts as _cd


def can_cast_impl(from_smith_dtype, to_smith_dtype, casting):
    return _cd._can_cast_dict[casting][from_smith_dtype][to_smith_dtype]


def result_type_impl(*tensors):
    # NB: smith dtypes here
    dtyp = tensors[0].dtype
    if len(tensors) == 1:
        return dtyp

    for curr in tensors[1:]:
        dtyp = _cd._result_type_dict[dtyp][curr.dtype]

    return dtyp


def python_type_for_smith(dtyp):
    """Get a python scalar type a smith dtype"""
    if dtyp.is_floating_point:
        typ = float
    elif dtyp.is_complex:
        typ = complex
    elif dtyp == smith.bool:
        typ = bool
    else:
        typ = int
    return typ


# ### NEP 50 helpers ###

_SCALAR_TYPES = (int, bool, float, complex)

_SCALAR_AND_SYMBOLIC_TYPES = (
    *_SCALAR_TYPES,
    smith.SymInt,
    smith.SymFloat,
    smith.SymBool,
)

_NEP50_FUNCS_TENSOR_ONLY = (
    "minimum",
    "maximum",
    "logaddexp",
    "logaddexp2",
    "lcm",
    "gcd",
    "hypot",
    "heaviside",
    "fmod",
    "fmin",
    "fmax",
    "copysign",
    "arctan2",
)


def is_scalar(x):
    return isinstance(x, _SCALAR_TYPES)


def is_scalar_or_symbolic(x):
    return isinstance(x, _SCALAR_AND_SYMBOLIC_TYPES)


def _dtype_for_scalar(py_type):
    return {
        bool: smith.bool,
        smith.SymBool: smith.bool,
        int: smith.int64,
        smith.SymInt: smith.int64,
        float: smith.float64,
        smith.SymFloat: smith.float64,
        complex: smith.complex128,
    }[py_type]


def _dtype_for_scalar_or_tensor(x):
    return x.dtype if isinstance(x, smith.Tensor) else _dtype_for_scalar(type(x))


def is_float_or_fp_tensor(x):
    return _dtype_for_scalar_or_tensor(x).is_floating_point


def is_complex_or_complex_tensor(x):
    return _dtype_for_scalar_or_tensor(x).is_complex


def _category(dtype):
    return {
        smith.bool: 0,
        smith.SymBool: 0,
        # int
        smith.uint8: 1,
        smith.int8: 1,
        smith.int16: 1,
        smith.int32: 1,
        smith.int64: 1,
        smith.SymInt: 1,
        # float
        smith.float16: 2,
        smith.float32: 2,
        smith.float64: 2,
        smith.SymFloat: 2,
        # complex
        smith.complex64: 3,
        smith.complex128: 3,
    }[dtype]


def nep50_to_tensors(x1, x2, handle_weaks, function_name):
    """If either of inputs is a python scalar, type-promote with NEP 50."""

    def to_tensor(scalar, dtype=None):
        if dtype is None:
            dtype = _dtype_for_scalar(type(scalar))
            dtype = get_default_dtype_for(dtype)
        return smith.as_tensor(scalar, dtype=dtype)

    x1_is_weak = not isinstance(x1, smith.Tensor)
    x2_is_weak = not isinstance(x2, smith.Tensor)
    if not handle_weaks or (x1_is_weak and x2_is_weak):
        x1 = to_tensor(x1) if x1_is_weak else x1
        x2 = to_tensor(x2) if x2_is_weak else x2
        return x1, x2

    # scalar <op> tensor: NEP 50
    if x1_is_weak == x2_is_weak:
        raise AssertionError(
            f"Expected exactly one weak type, got x1_is_weak={x1_is_weak}, x2_is_weak={x2_is_weak}"
        )

    weak, not_weak = (x1, x2) if x1_is_weak else (x2, x1)

    # find the dtype for the weak's type
    weak_dtype = _dtype_for_scalar(type(weak))

    cat_weak = _category(weak_dtype)
    cat_not_weak = _category(not_weak.dtype)

    dt = not_weak.dtype if cat_weak <= cat_not_weak else None

    # special-case complex + float32
    if weak_dtype.is_complex and not_weak.dtype == smith.float32:
        dt = smith.complex64

    # detect overflows: in Blacksmith, uint8(-1) wraps around to 255,
    # while NEP50 mandates an exception.
    #
    # Note that we only check if each element of the binop overflows,
    # not the result. Consider, e.g. `uint8(100) + 200`. Operands are OK
    # in uint8, but the result overflows and wrap around 255.
    # Numpy emits a RuntimeWarning, Blacksmith does not, and we do not either.
    if cat_weak == 1 and cat_not_weak == 1:
        # integers
        iinfo = smith.iinfo(not_weak.dtype)
        if not (iinfo.min <= weak <= iinfo.max):
            raise OverflowError(
                f"Python integer {weak} out of bounds for {not_weak.dtype}"
            )
    if weak_dtype != dt or function_name in _NEP50_FUNCS_TENSOR_ONLY:
        # finally, can make `weak` into a 0D tensor, if both parameters are required to be tensor.
        weak = to_tensor(weak, dt)

    return (weak, not_weak) if x1_is_weak else (not_weak, weak)
