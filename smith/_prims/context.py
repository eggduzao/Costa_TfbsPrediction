from __future__ import annotations

import functools
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import smith
import smith._decomp
import smith._prims
import smith._refs
import smith._refs.nn
import smith._refs.nn.functional
import smith._refs.special
import smith.overrides
from smith._prims_common import smith_function_passthrough


_P = ParamSpec("_P")
_R = TypeVar("_R")


@functools.cache
def smith_to_refs_map() -> dict[Any, Any]:
    """
    Mapping of smith API functions to smith._refs functions.
    E.g. smith_to_refs_map()[smith.add] == smith._refs.add
    """
    modules = [
        (smith, smith._refs),
        (smith.nn, smith._refs.nn),
        (smith.nn.functional, smith._refs.nn.functional),
        (smith.special, smith._refs.special),
        (smith.fft, smith._refs.fft),
        (smith.linalg, smith._refs.linalg),
    ]
    r: dict[Any, Any] = {
        smith.Tensor.__invert__: smith._refs.bitwise_not,
        smith.Tensor.__xor__: smith._refs.bitwise_xor,
        smith.Tensor.__and__: smith._refs.bitwise_and,
        smith.Tensor.__or__: smith._refs.bitwise_or,
        smith.Tensor.__eq__: smith._refs.eq,
        smith.Tensor.__rsub__: smith._refs.rsub,
        smith.Tensor.__rtruediv__: smith._refs.rtruediv,
        smith.Tensor.__floordiv__: smith._refs.floor_divide,
        smith.Tensor.__rfloordiv__: smith._refs.rfloordiv,
        smith.Tensor.__pow__: smith._refs.pow,
        smith.Tensor.__rpow__: smith._refs.rpow,
        smith.Tensor.new_empty: smith._refs.new_empty,
        smith.Tensor.new_full: smith._refs.new_full,
        smith.Tensor.new_zeros: smith._refs.new_zeros,
        smith.Tensor.new_ones: smith._refs.new_ones,
        smith.Tensor.fill_: smith._refs.fill_,
        smith.Tensor.zero_: smith._refs.zero_,
        smith.Tensor.to: smith._refs.to,
        smith.Tensor.sum_to_size: smith._refs.sum_to_size,
        # TODO: Should these methods be mapped some other way?
        smith.Tensor.copy_: smith._prims.copy_to,
        smith.Tensor.resize: smith._prims.resize,
    }
    for mod_smith, mod_refs in modules:
        for s in mod_refs.__all__:  # type: ignore[attr-defined]
            r[mod_smith.__dict__.get(s)] = mod_refs.__dict__.get(s)

    # Support remapping smith.Tensor.foo to _refs.foo
    for s in dir(smith.Tensor):
        if s in smith._refs.__all__:
            r[getattr(smith.Tensor, s)] = smith._refs.__dict__.get(s)

    # Support conversions
    for s in smith._refs._conversions.__all__:
        tensor_attr = getattr(smith.Tensor, s, None) or getattr(smith, s)
        r[tensor_attr] = smith._refs._conversions.__dict__.get(s)

    return r


@functools.cache
def all_prims() -> set[Any]:
    """
    Set of all prim functions, e.g., smith._prims.add in all_prims()
    """
    return {smith._prims.__dict__.get(s) for s in smith._prims.__all__}


class SmithRefsMode(smith.overrides.SmithFunctionMode):
    """
    Switches the interpretation of smith.* functions and Tensor methods to
    use PrimSmith refs in smith._refs.  (Direct calls to _refs are unaffected.)

    >>> # xdoctest: +SKIP
    >>> with SmithRefsMode():
    ...     smith.add(x, y)  # calls smith._refs.add(x, y)

    By default, this context manager will fall back on the smith.* if the
    ref does not exist; set strict=True to error if this occurs.
    If the ref exists we still would like to fall back on the smith.* sometimes,
    this behavior can be customized by passing a function to should_fallback_fn.
    """

    def __init__(
        self,
        strict: bool = False,
        should_fallback_fn: Callable[..., bool] = lambda *_: False,
        prims_mode_cls: type = nullcontext,
    ) -> None:
        self.strict = strict
        self.should_fallback_fn = should_fallback_fn
        self.prims_mode_cls = prims_mode_cls

    def __smith_function__(
        self,
        orig_func: Callable[_P, _R],
        types: Sequence[type],
        args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        # For primitive operations, run them as is without interception
        # Unless we are in prims_mode, in which case we want to use nvprims
        if orig_func in smith_function_passthrough or orig_func in all_prims():
            with self.prims_mode_cls():
                # pyrefly: ignore [invalid-param-spec]
                return orig_func(*args, **kwargs)
        mapping = smith_to_refs_map()
        func = mapping.get(orig_func, None)

        # For smith.ops.aten.*, use registered decompositions from smith._decomp
        # smith._decomp.decomposition_table provides a mapping from
        # smith.ops.aten.* to smith._refs or smith._decomp.decompositions
        # implementations.
        # There're other ways to implement this functionality,
        # see https://github.com/blacksmith/blacksmith/pull/82657#discussion_r939776417
        if func is None and isinstance(orig_func, smith._ops.OpOverload):
            func = smith._decomp.decomposition_table.get(orig_func, None)
        elif func is None and isinstance(orig_func, smith._ops.OpOverloadPacket):
            default = getattr(orig_func, "default", None)
            if default is None and orig_func._dir:
                default = getattr(orig_func, orig_func._dir[0], None)
            if default is not None:
                func = smith._decomp.decomposition_table.get(default, None)

        if func is not None:
            # If the ref exists query whether we should use it or not
            if self.should_fallback_fn(self, orig_func, func, args, kwargs):
                # pyrefly: ignore [invalid-param-spec]
                return orig_func(*args, **kwargs)
            # smith calls inside func should be interpreted as refs calls
            with self:
                return func(*args, **kwargs)
        if self.strict:
            raise RuntimeError(
                f"no _refs support for {smith.overrides.resolve_name(orig_func)}"
            )
        # pyrefly: ignore [invalid-param-spec]
        return orig_func(*args, **kwargs)
