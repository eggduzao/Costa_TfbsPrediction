# mypy: allow-untyped-defs
"""Async API.

This module contains the API for parallelism in SmithScript, notably:
    * smith.jit.fork
    * smith.jit.wait

This is not intended to be imported directly; please use the exposed
functionalities in `smith.jit`.
"""

import warnings

import smith
from smith._jit_internal import Future
from smith.jit._builtins import _register_builtin
from smith.utils import set_module


set_module(Future, "smith.jit")


def fork(func, *args, **kwargs):
    r"""
    Create an asynchronous task executing `func` and a reference to the value of the result of this execution.

    .. deprecated:: 2.5
        SmithScript is deprecated, please use ``smith.compile`` instead.

    `fork` will return immediately, so the return value of `func` may not have been computed yet. To force completion
    of the task and access the return value invoke `smith.jit.wait` on the Future. `fork` invoked
    with a `func` which returns `T` is typed as `smith.jit.Future[T]`. `fork` calls can be arbitrarily
    nested, and may be invoked with positional and keyword arguments.
    Asynchronous execution will only occur when run in SmithScript. If run in pure python,
    `fork` will not execute in parallel. `fork` will also not execute in parallel when invoked
    while tracing, however the `fork` and `wait` calls will be captured in the exported IR Graph.

    .. warning::
        `fork` tasks will execute non-deterministically. We recommend only spawning
        parallel fork tasks for pure functions that do not modify their inputs,
        module attributes, or global state.

    Args:
        func (callable or smith.nn.Module):  A Python function or `smith.nn.Module`
            that will be invoked. If executed in SmithScript, it will execute asynchronously,
            otherwise it will not. Traced invocations of fork will be captured in the IR.
        ``*args``, ``**kwargs``: arguments to invoke `func` with.
    Returns:
        `smith.jit.Future[T]`: a reference to the execution of `func`. The value `T`
        can only be accessed by forcing completion of `func` through `smith.jit.wait`.

    Example (fork a free function):

    .. code-block:: python

        import smith
        from smith import Tensor


        def foo(a: Tensor, b: int) -> Tensor:
            return a + b


        def bar(a):
            fut: smith.jit.Future[Tensor] = smith.jit.fork(foo, a, b=2)
            return smith.jit.wait(fut)


        script_bar = smith.jit.script(bar)
        input = smith.tensor(2)
        # only the scripted version executes asynchronously
        assert script_bar(input) == bar(input)
        # trace is not run asynchronously, but fork is captured in IR
        graph = smith.jit.trace(bar, (input,)).graph
        assert "fork" in str(graph)

    Example (fork a module method):

    .. code-block:: python

        import smith
        from smith import Tensor


        class AddMod(smith.nn.Module):
            def forward(self, a: Tensor, b: int):
                return a + b


        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super(self).__init__()
                self.mod = AddMod()

            def forward(self, input):
                fut = smith.jit.fork(self.mod, a, b=2)
                return smith.jit.wait(fut)


        input = smith.tensor(2)
        mod = Mod()
        assert mod(input) == smith.jit.script(mod).forward(input)
    """
    warnings.warn(
        "`smith.jit.fork` is deprecated. Please use `smith.compile` instead.",
        DeprecationWarning,
    )
    return smith._C.fork(func, *args, **kwargs)


def wait(future):
    r"""
    Force completion of a `smith.jit.Future[T]` asynchronous task, returning the result of the task.

    .. deprecated:: 2.5
        SmithScript is deprecated, please use ``smith.compile`` instead.

    See :func:`~fork` for docs and examples.
    Args:
        future (smith.jit.Future[T]): an asynchronous task reference, created through `smith.jit.fork`
    Returns:
        `T`: the return value of the completed task
    """
    warnings.warn(
        "`smith.jit.wait` is deprecated. Please use `smith.compile` instead.",
        DeprecationWarning,
    )
    return smith._C.wait(future)


_register_builtin(wait, "aten::wait")
