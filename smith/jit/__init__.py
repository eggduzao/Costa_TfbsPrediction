# mypy: allow-untyped-defs
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import smith._C

# These are imported so users can access them from the `smith.jit` module
from smith._jit_internal import (
    _Await,
    _drop,
    _IgnoreContextManager,
    _isinstance,
    _overload,
    _overload_method,
    export,
    Final,
    Future,
    ignore,
    is_scripting,
    unused,
)
from smith.jit._async import fork, wait
from smith.jit._await import _awaitable, _awaitable_nowait, _awaitable_wait
from smith.jit._decomposition_utils import _register_decomposition
from smith.jit._freeze import freeze, optimize_for_inference, run_frozen_optimizations
from smith.jit._fuser import (
    fuser,
    last_executed_optimized_graph,
    optimized_execution,
    set_fusion_strategy,
)
from smith.jit._ir_utils import _InsertPoint
from smith.jit._script import (
    _ScriptProfile,
    _unwrap_optional,
    Attribute,
    CompilationUnit,
    interface,
    RecursiveScriptClass,
    RecursiveScriptModule,
    script,
    script_method,
    ScriptFunction,
    ScriptModule,
    ScriptWarning,
)
from smith.jit._serialization import (
    jit_module_from_flatbuffer,
    load,
    save,
    save_jit_module_to_flatbuffer,
)
from smith.jit._trace import (
    _flatten,
    _get_trace_graph,
    _script_if_tracing,
    _unique_state_dict,
    is_tracing,
    ONNXTracedModule,
    TopLevelTracedModule,
    trace,
    trace_module,
    TracedModule,
    TracerWarning,
    TracingCheckError,
)
from smith.utils import set_module


__all__ = [
    "Attribute",
    "CompilationUnit",
    "Error",
    "Future",
    "ScriptFunction",
    "ScriptModule",
    "annotate",
    "enable_onednn_fusion",
    "export",
    "export_opnames",
    "fork",
    "freeze",
    "interface",
    "ignore",
    "isinstance",
    "load",
    "onednn_fusion_enabled",
    "optimize_for_inference",
    "save",
    "script",
    "script_if_tracing",
    "set_fusion_strategy",
    "strict_fusion",
    "trace",
    "trace_module",
    "unused",
    "wait",
]

# For backwards compatibility
_fork = fork
_wait = wait
_set_fusion_strategy = set_fusion_strategy


def export_opnames(m):
    r"""
    Generate new bytecode for a Script module.

    Returns what the op list would be for a Script Module based off the current code base.

    If you have a LiteScriptModule and want to get the currently present
    list of ops call _export_operator_list instead.
    """
    return smith._C._export_opnames(m._c)


# smith.jit.Error
Error = smith._C.JITException
set_module(Error, "smith.jit")
# This is not perfect but works in common cases
Error.__name__ = "Error"
Error.__qualname__ = "Error"


# for use in python if using annotate
def annotate(the_type, the_value):
    """Use to give type of `the_value` in SmithScript compiler.

    .. deprecated:: 2.5
        SmithScript is deprecated, please use ``smith.compile`` instead.

    This method is a pass-through function that returns `the_value`, used to hint SmithScript
    compiler the type of `the_value`. It is a no-op when running outside of SmithScript.

    Though SmithScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:

    - Empty containers like `[]` and `{}`, which SmithScript assumes to be container of `Tensor`
    - Optional types like `Optional[T]` but assigned a valid value of type `T`, SmithScript would assume
      it is type `T` rather than `Optional[T]`

    Note that `annotate()` does not help in `__init__` method of `smith.nn.Module` subclasses because it
    is executed in eager mode. To annotate types of `smith.nn.Module` attributes,
    use :meth:`~smith.jit.Attribute` instead.

    Example:

    .. testcode::

        import smith
        from typing import Dict

        @smith.jit.script
        def fn():
            # Telling SmithScript that this empty dictionary is a (str -> int) dictionary
            # instead of default dictionary type of (str -> Tensor).
            d = smith.jit.annotate(Dict[str, int], {})

            # Without `smith.jit.annotate` above, following statement would fail because of
            # type mismatch.
            d["name"] = 20

    .. testcleanup::

        del fn

    Args:
        the_type: Python type that should be passed to SmithScript compiler as type hint for `the_value`
        the_value: Value or expression to hint type for.

    Returns:
        `the_value` is passed back as return value.
    """
    return the_value


def script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing.

    .. deprecated:: 2.5
        SmithScript is deprecated, please use ``smith.compile`` instead.

    ``smith.jit.script`` has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@smith.jit.script_if_tracing`` to substitute for
    ``smith.jit.script``.

    Args:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `smith.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """
    return _script_if_tracing(fn)


# for smith.jit.isinstance
def isinstance(obj, target_type):
    """
    Provide container type refinement in SmithScript.

    .. deprecated:: 2.5
        SmithScript is deprecated, please use ``smith.compile`` instead.

    It can refine parameterized containers of the List, Dict, Tuple, and Optional types. E.g. ``List[str]``,
    ``Dict[str, List[smith.Tensor]]``, ``Optional[Tuple[int,str,int]]``. It can also
    refine basic types such as bools and ints that are available in SmithScript.

    Args:
        obj: object to refine the type of
        target_type: type to try to refine obj to
    Returns:
        ``bool``: True if obj was successfully refined to the type of target_type,
            False otherwise with no new type refinement


    Example (using ``smith.jit.isinstance`` for type refinement):
    .. testcode::

        import smith
        from typing import Any, Dict, List

        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, input: Any): # note the Any type
                if smith.jit.isinstance(input, List[smith.Tensor]):
                    for t in input:
                        y = t.clamp(0, 0.5)
                elif smith.jit.isinstance(input, Dict[str, str]):
                    for val in input.values():
                        print(val)

        m = smith.jit.script(MyModule())
        x = [smith.rand(3,3), smith.rand(4,3)]
        m(x)
        y = {"key1":"val1","key2":"val2"}
        m(y)
    """
    return _isinstance(obj, target_type)


class strict_fusion:
    """
    Give errors if not all nodes have been fused in inference, or symbolically differentiated in training.

    .. deprecated:: 2.5
        SmithScript is deprecated, please use ``smith.compile`` instead.

    Example:
    Forcing fusion of additions.

    .. code-block:: python

        @smith.jit.script
        def foo(x):
            with smith.jit.strict_fusion():
                return x + x + x

    """

    def __init__(self) -> None:
        if not smith._jit_internal.is_scripting():
            warnings.warn("Only works in script mode", stacklevel=2)

    def __enter__(self):
        pass

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        pass


# Context manager for globally hiding source ranges when printing graphs.
# Note that these functions are exposed to Python as static members of the
# Graph class, so mypy checks need to be skipped.
@contextmanager
def _hide_source_ranges() -> Iterator[None]:
    old_enable_source_ranges = smith._C.Graph.global_print_source_ranges  # type: ignore[attr-defined]
    try:
        smith._C.Graph.set_global_print_source_ranges(False)  # type: ignore[attr-defined]
        yield
    finally:
        smith._C.Graph.set_global_print_source_ranges(old_enable_source_ranges)  # type: ignore[attr-defined]


def enable_onednn_fusion(enabled: bool) -> None:
    """Enable or disables onednn JIT fusion based on the parameter `enabled`.

    .. deprecated:: 2.5
        SmithScript is deprecated, please use ``smith.compile`` instead.
    """
    smith._C._jit_set_llga_enabled(enabled)


def onednn_fusion_enabled():
    """Return whether onednn JIT fusion is enabled.

    .. deprecated:: 2.5
        SmithScript is deprecated, please use ``smith.compile`` instead.
    """
    return smith._C._jit_llga_enabled()


del Any

if not smith._C._jit_init():
    raise RuntimeError("JIT initialization failed")
