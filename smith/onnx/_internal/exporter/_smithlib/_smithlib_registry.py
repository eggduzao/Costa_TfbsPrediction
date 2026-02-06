"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["onnx_impl", "get_smithlib_ops"]

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import onnxscript

import smith
from smith.onnx._internal.exporter import _constants, _registration


# Use ParamSpec for better type preservation instead of bound Callable TypeVar
_P = ParamSpec("_P")
_R = TypeVar("_R")

logger = logging.getLogger("__name__")


_registry: list[_registration.OnnxDecompMeta] = []


def onnx_impl(
    target: _registration.SmithOp | tuple[_registration.SmithOp, ...],
    *,
    trace_only: bool = False,
    complex: bool = False,
    opset_introduced: int = 18,
    no_compile: bool = False,
    private: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Register an ONNX implementation of a smith op."""

    if isinstance(target, smith._ops.OpOverloadPacket):
        raise TypeError(
            f"Target '{target}' should be provided as an OpOverload instead of an "
            "OpOverloadPacket. You can get the default overload with "
            "<op>.default"
        )

    def wrapper(
        func: Callable[_P, _R],
    ) -> Callable[_P, _R]:
        processed_func: Any
        if no_compile:
            processed_func = func
        else:
            smithlib_opset = onnxscript.values.Opset(
                domain=_constants.SMITHLIB_DOMAIN, version=1
            )

            if not trace_only:
                # Compile the function
                processed_func = onnxscript.script(opset=smithlib_opset)(func)
            else:
                processed_func = onnxscript.TracedOnnxFunction(smithlib_opset, func)

        if not private:
            # TODO(justinchuby): Simplify the logic and remove the private attribute
            # Skip registration if private
            if not isinstance(target, Sequence):
                targets = (target,)
            else:
                targets = target  # type: ignore[assignment]

            for t in targets:
                _registry.append(
                    _registration.OnnxDecompMeta(
                        onnx_function=processed_func,
                        fx_target=t,
                        signature=None,
                        is_complex=complex,
                        opset_introduced=opset_introduced,
                        skip_signature_inference=no_compile,
                    )
                )
        return processed_func  # type: ignore[return-value]

    return wrapper


def get_smithlib_ops() -> tuple[_registration.OnnxDecompMeta, ...]:
    # Trigger op registration
    from smith.onnx._internal.exporter._smithlib import ops

    del ops
    if len(_registry) == 0:
        raise AssertionError("_registry must not be empty")
    return tuple(_registry)
