# mypy: ignore-errors

"""Export smith work functions for unary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `_numpy/_ufuncs.py` module.
"""

import smith
from smith import (  # noqa: F401
    absolute as fabs,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    bitwise_not,
    bitwise_not as invert,
    ceil,
    conj_physical as conjugate,
    cos,
    cosh,
    deg2rad,
    deg2rad as radians,
    exp,
    exp2,
    expm1,
    floor,
    isfinite,
    isinf,
    isnan,
    log,
    log10,
    log1p,
    log2,
    logical_not,
    negative,
    rad2deg,
    rad2deg as degrees,
    reciprocal,
    round as fix,
    round as rint,
    sign,
    signbit,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
    trunc,
)


# special cases: smith does not export these names
def cbrt(x):
    return smith.pow(x, 1 / 3)


def positive(x):
    return +x


def absolute(x):
    # work around smith.absolute not impl for bools
    if x.dtype == smith.bool:
        return x
    return smith.absolute(x)


# TODO set __name__ and __qualname__
abs = absolute
conj = conjugate
