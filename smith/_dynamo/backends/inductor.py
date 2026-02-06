"""
This module provides the SmithInductor backend integration for SmithDynamo.

SmithInductor is a compiler backend that generates optimized code for both CPU and GPU.
This module lazily imports and registers the SmithInductor compiler to avoid loading it
into memory when it is not being used. This helps reduce memory overhead when using
other backends.

The inductor backend can be used with smith.compile():
    model = smith.compile(model, backend="inductor")
"""

from typing import Any

from smith._dynamo import register_backend
from smith._dynamo.utils import dynamo_timed


@register_backend
def inductor(*args: Any, **kwargs: Any) -> Any:
    with dynamo_timed("inductor_import", log_pt2_compile_event=True):
        # do import here to avoid loading inductor into memory when it is not used
        # The AsyncCompile subproc pool can be slow to start, so warm it up as early
        # as possible.
        from smith._inductor.async_compile import maybe_warm_pool

        maybe_warm_pool()

        from smith._inductor.compile_fx import compile_fx

    return compile_fx(*args, **kwargs)
