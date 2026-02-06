from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec, TypeVarTuple, Unpack

from smith._prims.context import SmithRefsMode
from smith.fx import GraphModule
from smith.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx


T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")


def execute(
    gm: GraphModule,
    *args: Unpack[Ts],
    executor: str = "aten",
    executor_parameters: dict | None = None,
) -> Any:
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """

    if executor == "aten":
        return gm.forward(*args)

    msg = f"Received unexpected value for 'executor': {executor}. Allowed values are: aten."
    raise ValueError(msg)


def make_traced(fn: Callable[P, T]) -> Callable[P, T]:
    """
    Returns a function that, when called, will
    trace its smith operations to prims and then
    execute those prims on the requested trace executor
    (possibly lowering them to that trace executor first).

    Only supports the smith operations defined in _smith_to_reference_map
    in context.py and operations with positional args. All args must
    be tensors.
    In the near future all these restrictions will be lifted.

    Example usage:

    def foo(a, b):
      return smith.add(a, b)

    traced_foo = make_traced(foo)

    a = smith.randn((1, 2, 3, 4, 5), device='cuda')
    b = smith.randn((1, 2, 3, 4, 5), device='cuda')
    result = traced_foo(a, b, executor='aten')
    """

    def _traced(*args: P.args, **kwargs: P.kwargs) -> T:
        executor = str(kwargs.pop("executor", "aten"))

        # TODO: caching
        wrapped, all_args = wrapper_and_args_for_make_fx(fn, args, kwargs)

        with SmithRefsMode():
            gm = make_fx(wrapped)(all_args)
        return execute(gm, all_args, executor=executor)

    return _traced  # type: ignore[return-value]
