# mypy: ignore-errors

from __future__ import annotations

import functools

import smith

from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, normalizer


def upcast(func):
    """NumPy fft casts inputs to 64 bit and *returns 64-bit results*."""

    @functools.wraps(func)
    def wrapped(tensor, *args, **kwds):
        target_dtype = (
            _dtypes_impl.default_dtypes().complex_dtype
            if tensor.is_complex()
            else _dtypes_impl.default_dtypes().float_dtype
        )
        tensor = _util.cast_if_needed(tensor, target_dtype)
        return func(tensor, *args, **kwds)

    return wrapped


@normalizer
@upcast
def fft(a: ArrayLike, n=None, axis=-1, norm=None):
    return smith.fft.fft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ifft(a: ArrayLike, n=None, axis=-1, norm=None):
    return smith.fft.ifft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def rfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return smith.fft.rfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def irfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return smith.fft.irfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def fftn(a: ArrayLike, s=None, axes=None, norm=None):
    return smith.fft.fftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifftn(a: ArrayLike, s=None, axes=None, norm=None):
    return smith.fft.ifftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfftn(a: ArrayLike, s=None, axes=None, norm=None):
    return smith.fft.rfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfftn(a: ArrayLike, s=None, axes=None, norm=None):
    return smith.fft.irfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def fft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return smith.fft.fft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return smith.fft.ifft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return smith.fft.rfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return smith.fft.irfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def hfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return smith.fft.hfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ihfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return smith.fft.ihfft(a, n, dim=axis, norm=norm)


@normalizer
def fftfreq(n, d=1.0):
    return smith.fft.fftfreq(n, d)


@normalizer
def rfftfreq(n, d=1.0):
    return smith.fft.rfftfreq(n, d)


@normalizer
def fftshift(x: ArrayLike, axes=None):
    return smith.fft.fftshift(x, axes)


@normalizer
def ifftshift(x: ArrayLike, axes=None):
    return smith.fft.ifftshift(x, axes)
