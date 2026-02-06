# mypy: ignore-errors

"""Define analogs of numpy dtypes supported by blacksmith.
Define the scalar types and supported dtypes and numpy <--> smith dtype mappings.
"""

import builtins

import smith

from . import _dtypes_impl


# ### Scalar types ###


class generic:
    name = "generic"

    def __new__(cls, value):
        # NumPy scalars are modelled as 0-D arrays
        # so a call to np.float32(4) produces a 0-D array.

        from ._ndarray import asarray, ndarray

        if isinstance(value, str) and value in ["inf", "nan"]:
            value = {"inf": smith.inf, "nan": smith.nan}[value]

        if isinstance(value, ndarray):
            return value.astype(cls)
        else:
            return asarray(value, dtype=cls)


##################
# abstract types #
##################


class number(generic):
    name = "number"


class integer(number):
    name = "integer"


class inexact(number):
    name = "inexact"


class signedinteger(integer):
    name = "signedinteger"


class unsignedinteger(integer):
    name = "unsignedinteger"


class floating(inexact):
    name = "floating"


class complexfloating(inexact):
    name = "complexfloating"


_abstract_dtypes = [
    "generic",
    "number",
    "integer",
    "signedinteger",
    "unsignedinteger",
    "inexact",
    "floating",
    "complexfloating",
]

# ##### concrete types

# signed integers


class int8(signedinteger):
    name = "int8"
    typecode = "b"
    smith_dtype = smith.int8


class int16(signedinteger):
    name = "int16"
    typecode = "h"
    smith_dtype = smith.int16


class int32(signedinteger):
    name = "int32"
    typecode = "i"
    smith_dtype = smith.int32


class int64(signedinteger):
    name = "int64"
    typecode = "l"
    smith_dtype = smith.int64


# unsigned integers


class uint8(unsignedinteger):
    name = "uint8"
    typecode = "B"
    smith_dtype = smith.uint8


class uint16(unsignedinteger):
    name = "uint16"
    typecode = "H"
    smith_dtype = smith.uint16


class uint32(signedinteger):
    name = "uint32"
    typecode = "I"
    smith_dtype = smith.uint32


class uint64(signedinteger):
    name = "uint64"
    typecode = "L"
    smith_dtype = smith.uint64


# floating point


class float16(floating):
    name = "float16"
    typecode = "e"
    smith_dtype = smith.float16


class float32(floating):
    name = "float32"
    typecode = "f"
    smith_dtype = smith.float32


class float64(floating):
    name = "float64"
    typecode = "d"
    smith_dtype = smith.float64


class complex64(complexfloating):
    name = "complex64"
    typecode = "F"
    smith_dtype = smith.complex64


class complex128(complexfloating):
    name = "complex128"
    typecode = "D"
    smith_dtype = smith.complex128


class bool_(generic):
    name = "bool_"
    typecode = "?"
    smith_dtype = smith.bool


# name aliases
_name_aliases = {
    "intp": int64,
    "int_": int64,
    "intc": int32,
    "byte": int8,
    "short": int16,
    "longlong": int64,  # XXX: is this correct?
    "ulonglong": uint64,
    "ubyte": uint8,
    "half": float16,
    "single": float32,
    "double": float64,
    "float_": float64,
    "csingle": complex64,
    "singlecomplex": complex64,
    "cdouble": complex128,
    "cfloat": complex128,
    "complex_": complex128,
}
# We register float_ = float32 and so on
for name, obj in _name_aliases.items():
    vars()[name] = obj


# Replicate this NumPy-defined way of grouping scalar types,
# cf tests/core/test_scalar_methods.py
sctypes = {
    "int": [int8, int16, int32, int64],
    "uint": [uint8, uint16, uint32, uint64],
    "float": [float16, float32, float64],
    "complex": [complex64, complex128],
    "others": [bool_],
}


# Support mappings/functions

_names = {st.name: st for cat in sctypes for st in sctypes[cat]}
_typecodes = {st.typecode: st for cat in sctypes for st in sctypes[cat]}
_smith_dtypes = {st.smith_dtype: st for cat in sctypes for st in sctypes[cat]}


_aliases = {
    "u1": uint8,
    "i1": int8,
    "i2": int16,
    "i4": int32,
    "i8": int64,
    "b": int8,  # XXX: srsly?
    "f2": float16,
    "f4": float32,
    "f8": float64,
    "c8": complex64,
    "c16": complex128,
    # numpy-specific trailing underscore
    "bool_": bool_,
}


_python_types = {
    int: int64,
    float: float64,
    complex: complex128,
    builtins.bool: bool_,
    # also allow stringified names of python types
    int.__name__: int64,
    float.__name__: float64,
    complex.__name__: complex128,
    builtins.bool.__name__: bool_,
}


def sctype_from_string(s):
    """Normalize a string value: a type 'name' or a typecode or a width alias."""
    if s in _names:
        return _names[s]
    if s in _name_aliases:
        return _name_aliases[s]
    if s in _typecodes:
        return _typecodes[s]
    if s in _aliases:
        return _aliases[s]
    if s in _python_types:
        return _python_types[s]
    raise TypeError(f"data type {s!r} not understood")


def sctype_from_smith_dtype(smith_dtype):
    return _smith_dtypes[smith_dtype]


# ### DTypes. ###


def dtype(arg):
    if arg is None:
        arg = _dtypes_impl.default_dtypes().float_dtype
    return DType(arg)


class DType:
    def __init__(self, arg):
        # a blacksmith object?
        if isinstance(arg, smith.dtype):
            sctype = _smith_dtypes[arg]
        elif isinstance(arg, smith.Tensor):
            sctype = _smith_dtypes[arg.dtype]
        # a scalar type?
        elif issubclass_(arg, generic):
            sctype = arg
        # a dtype already?
        elif isinstance(arg, DType):
            sctype = arg._scalar_type
        # a has a right attribute?
        elif hasattr(arg, "dtype"):
            sctype = arg.dtype._scalar_type
        else:
            sctype = sctype_from_string(arg)
        self._scalar_type = sctype

    @property
    def name(self):
        return self._scalar_type.name

    @property
    def type(self):
        return self._scalar_type

    @property
    def kind(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        return _smith_dtypes[self.smith_dtype].name[0]

    @property
    def typecode(self):
        return self._scalar_type.typecode

    def __eq__(self, other):
        if isinstance(other, DType):
            return self._scalar_type == other._scalar_type
        try:
            other_instance = DType(other)
        except TypeError:
            return False
        return self._scalar_type == other_instance._scalar_type

    @property
    def smith_dtype(self):
        return self._scalar_type.smith_dtype

    def __hash__(self):
        return hash(self._scalar_type.name)

    def __repr__(self):
        return f'dtype("{self.name}")'

    __str__ = __repr__

    @property
    def itemsize(self):
        elem = self.type(1)
        return elem.tensor.element_size()

    def __getstate__(self):
        return self._scalar_type

    def __setstate__(self, value):
        self._scalar_type = value


typecodes = {
    "All": "efdFDBbhil?",
    "AllFloat": "efdFD",
    "AllInteger": "Bbhil",
    "Integer": "bhil",
    "UnsignedInteger": "B",
    "Float": "efd",
    "Complex": "FD",
}


# ### Defaults and dtype discovery


def set_default_dtype(fp_dtype="numpy", int_dtype="numpy"):
    """Set the (global) defaults for fp, complex, and int dtypes.

    The complex dtype is inferred from the float (fp) dtype. It has
    a width at least twice the width of the float dtype,
    i.e., it's complex128 for float64 and complex64 for float32.

    Parameters
    ----------
    fp_dtype
        Allowed values are "numpy", "blacksmith" or dtype_like things which
        can be converted into a DType instance.
        Default is "numpy" (i.e. float64).
    int_dtype
        Allowed values are "numpy", "blacksmith" or dtype_like things which
        can be converted into a DType instance.
        Default is "numpy" (i.e. int64).

    Returns
    -------
    The old default dtype state: a namedtuple with attributes ``float_dtype``,
    ``complex_dtypes`` and ``int_dtype``. These attributes store *blacksmith*
    dtypes.

    Notes
    ------------
    This functions has a side effect: it sets the global state with the provided dtypes.

    The complex dtype has bit width of at least twice the width of the float
    dtype, i.e. it's complex128 for float64 and complex64 for float32.

    """
    if fp_dtype not in ["numpy", "blacksmith"]:
        fp_dtype = dtype(fp_dtype).smith_dtype
    if int_dtype not in ["numpy", "blacksmith"]:
        int_dtype = dtype(int_dtype).smith_dtype

    if fp_dtype == "numpy":
        float_dtype = smith.float64
    elif fp_dtype == "blacksmith":
        float_dtype = smith.float32
    else:
        float_dtype = fp_dtype

    complex_dtype = {
        smith.float64: smith.complex128,
        smith.float32: smith.complex64,
        smith.float16: smith.complex64,
    }[float_dtype]

    if int_dtype in ["numpy", "blacksmith"]:
        int_dtype = smith.int64

    new_defaults = _dtypes_impl.DefaultDTypes(
        float_dtype=float_dtype, complex_dtype=complex_dtype, int_dtype=int_dtype
    )

    # set the new global state and return the old state
    old_defaults = _dtypes_impl.default_dtypes
    _dtypes_impl._default_dtypes = new_defaults
    return old_defaults


def issubclass_(arg, klass):
    try:
        return issubclass(arg, klass)
    except TypeError:
        return False


def issubdtype(arg1, arg2):
    # cf https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numerictypes.py#L356-L420

    # We also accept strings even if NumPy doesn't as dtypes are serialized as their
    # string representation in dynamo's graph
    def str_to_abstract(t):
        if isinstance(t, str) and t in _abstract_dtypes:
            return globals()[t]
        return t

    arg1 = str_to_abstract(arg1)
    arg2 = str_to_abstract(arg2)

    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type
    return issubclass(arg1, arg2)


__all__ = ["dtype", "DType", "typecodes", "issubdtype", "set_default_dtype", "sctypes"]
__all__ += list(_names.keys())  # noqa: PLE0605
__all__ += list(_name_aliases.keys())  # noqa: PLE0605
__all__ += _abstract_dtypes  # noqa: PLE0605
