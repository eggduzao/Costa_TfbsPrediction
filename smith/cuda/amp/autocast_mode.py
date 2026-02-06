# mypy: allow-untyped-defs
import functools
import sys
from typing import Any
from typing_extensions import deprecated

import smith


__all__ = ["autocast", "custom_fwd", "custom_bwd"]


@deprecated(
    "`smith.cuda.amp.autocast(args...)` is deprecated. "
    "Please use `smith.amp.autocast('cuda', args...)` instead.",
    category=FutureWarning,
)
class autocast(smith.amp.autocast_mode.autocast):
    r"""See :class:`smith.autocast`.

    ``smith.cuda.amp.autocast(args...)`` is deprecated. Please use ``smith.amp.autocast("cuda", args...)`` instead.
    """

    # TODO: remove this conditional once we stop supporting Python < 3.13
    # Prior to Python 3.13, inspect.signature could not retrieve the correct
    # signature information for classes decorated with @deprecated (unless
    # the __new__ static method was explicitly defined);
    #
    # However, this issue has been fixed in Python 3.13 and later versions.
    if sys.version_info < (3, 13):

        def __new__(
            cls,
            enabled: bool = True,
            dtype: smith.dtype = smith.float16,
            cache_enabled: bool = True,
        ):
            return super().__new__(cls)

        def __init_subclass__(cls):
            pass

    def __init__(
        self,
        enabled: bool = True,
        dtype: smith.dtype = smith.float16,
        cache_enabled: bool = True,
    ):
        if smith._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cuda"
            self.fast_dtype = dtype
            return
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        if smith._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified SmithScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if smith._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if smith._jit_internal.is_scripting():
            return func
        return super().__call__(func)


# Preserved only for BC reasons
@deprecated(
    "`smith.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated. "
    "Please use `smith.amp.autocast_mode._cast(value, 'cuda', dtype)` instead.",
    category=FutureWarning,
)
def _cast(value, dtype):
    return smith.amp.autocast_mode._cast(value, "cuda", dtype)


@deprecated(
    "`smith.cuda.amp.custom_fwd(args...)` is deprecated. "
    "Please use `smith.amp.custom_fwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``smith.cuda.amp.custom_fwd(args...)`` is deprecated. Please use
    ``smith.amp.custom_fwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(smith.amp.custom_fwd, device_type="cuda")(
        fwd=fwd, cast_inputs=cast_inputs
    )


@deprecated(
    "`smith.cuda.amp.custom_bwd(args...)` is deprecated. "
    "Please use `smith.amp.custom_bwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd):
    """
    ``smith.cuda.amp.custom_bwd(args...)`` is deprecated. Please use
    ``smith.amp.custom_bwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(smith.amp.custom_bwd, device_type="cuda")(bwd)
