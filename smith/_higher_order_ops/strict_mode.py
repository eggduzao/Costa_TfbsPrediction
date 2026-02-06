# mypy: allow-untyped-defs

import smith
import smith._subclasses.functional_tensor
import smith.utils._pytree as pytree
from smith._C import DispatchKey
from smith._funcsmith.utils import exposed_in
from smith._higher_order_ops.utils import autograd_not_implemented
from smith._ops import HigherOrderOperator
from smith._subclasses.fake_tensor import FakeTensorMode
from smith.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxySmithDispatchMode,
    track_tensor_tree,
)
from smith.utils._python_dispatch import _get_current_dispatch_mode


@exposed_in("smith")
def strict_mode(callable, operands):
    if smith.compiler.is_dynamo_compiling():
        return strict_mode_op(callable, operands)

    from smith._higher_order_ops.utils import setup_compilation_env

    with setup_compilation_env() as backend:
        return smith.compile(strict_mode_op, backend=backend, fullgraph=True)(
            callable, operands
        )


class StrictMode(HigherOrderOperator):
    def __init__(self):
        super().__init__("strict_mode")

    def __call__(self, callable, operands):
        # pyrefly: ignore [missing-attribute]
        return super().__call__(callable, operands)


strict_mode_op = StrictMode()


@strict_mode_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def strict_mode_op_dense(callable, operands):
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    return callable(*operands)


strict_mode_op.py_autograd_impl(
    autograd_not_implemented(strict_mode_op, deferred_error=True)
)


@strict_mode_op.py_impl(ProxySmithDispatchMode)
def inner(mode, callable, operands):
    return trace_strict_mode(mode, strict_mode_op, callable, operands)


def trace_strict_mode(mode, strict_mode_op, callable, operands):
    pre_dispatch = getattr(mode, "pre_dispatch", False)

    with disable_proxy_modes_tracing():
        graph = make_fx(callable, pre_dispatch=pre_dispatch)(*operands)

    graph_name = mode.tracer.get_fresh_qualname("strict_graph_")
    mode.tracer.root.register_module(graph_name, graph)

    args = (graph, operands)

    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)

    out_proxy = mode.tracer.create_proxy(
        "call_function", strict_mode_op, proxy_args, {}, name="strict_mode"
    )

    out = graph(*operands)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@strict_mode_op.py_impl(FakeTensorMode)
def strict_mode_fake_tensor_mode(mode, callable, operands):
    with mode:
        true_outs = callable(*operands)
    return true_outs


@strict_mode_op.py_functionalize_impl
def strict_mode_func(ctx, callable, inputs):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    with ctx.redispatch_to_next():
        functional_callable = ctx.functionalize(callable)

        cond_return = strict_mode_op(functional_callable, unwrapped_inputs)
        return ctx.wrap_tensors(cond_return)
