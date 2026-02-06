# mypy: ignore-errors

import smith

from . import _dtypes


def finfo(dtyp):
    smith_dtype = _dtypes.dtype(dtyp).smith_dtype
    return smith.finfo(smith_dtype)


def iinfo(dtyp):
    smith_dtype = _dtypes.dtype(dtyp).smith_dtype
    return smith.iinfo(smith_dtype)
