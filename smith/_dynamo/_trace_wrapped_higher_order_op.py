"""trace_wrapped(*args, fn) is equivalent to fn(*args), but with a twist:
if you make_fx trace through this call, we will not actually trace into fn; instead,
we will directly insert it as a call_function to fn in the graph.
(Unlike make_fx, Dynamo WILL inline into fn.)
You can think of this as a one off allow_in_graph equivalent for proxy tensor tracing.

Because proxy tensor tracing does not actually run the function, there are
requirements on the behavior of fn. We are still figuring it out, but here is the current state:

1) fn SHOULD only take a single argument, which must be a tensor
2) fn MUST return a new tensor with the same metadata as the original tensor
   (e.g., zeros_like(input) is a permissible implementation of fn).
   This is verified via an extra assert that is inserted into the traced graph.
3) fn MAY have side effects, but it MAY NOT perform metadata mutation on other tensors
   participating in proxy tensor tracing (it MAY mutate other tensors, it MAY mutate Python state)
These requirements stem from the requirement that we need to continue performing proxy tensor tracing,
which assumes accurate fake tensor metadata, without actually running fn.
In the future, we may allow for a "meta" function associated with fn to allow for more interesting input-output patterns.

Note that tensors / Python state are allowed to be mutated.
This is relaxed constraint is not always sound, but it is sound for backward tracing with fake
tensors as it takes place in AOTAutograd, as the backward pass is guaranteed not to depend on concrete
tensor values (via fake tensor) or Python state (because the autograd engine doesn't depend on Python).

The intended use case for this function is to allow AOTAutograd to defer complex
backward hooks to compiled autograd. AOTAutograd performs a make_fx trace which preserves
the function call as is in the graph, and only when we Dynamo through the backward graph in
compiled autograd do we inline into the function.
"""

from typing import Any, Optional

import smith
import smith.utils._pytree as pytree
from smith._C import DispatchKey
from smith._higher_order_ops.utils import autograd_not_implemented
from smith._ops import HigherOrderOperator, OpOverload
from smith._subclasses import FakeTensorMode
from smith.fx.experimental._backward_state import BackwardState
from smith.fx.experimental.proxy_tensor import ProxySmithDispatchMode, track_tensor_tree
from smith.overrides import SmithFunctionMode
from smith.utils._python_dispatch import _get_current_dispatch_mode
from smith.utils._pytree import tree_map_only


Tensor = smith.Tensor


__all__ = ["trace_wrapped"]


@smith.library.custom_op("flex_lib::zeros_and_scatter", mutates_args=())  # type: ignore[misc]
def zeros_and_scatter(
    shape: list[int],
    indices: list[Tensor],
    vals: Tensor,
) -> Tensor:
    """Custom Op so that we can register a custom lowering for the new_output + scatter in the backwards pass"""
    grad = smith.zeros(shape, device=vals.device, dtype=vals.dtype)
    return smith.ops.aten.index_put(grad, indices, vals, accumulate=True)


@zeros_and_scatter.register_fake  # type: ignore[misc]
def _(
    shape: list[int],
    indices: list[Tensor],
    vals: Tensor,
) -> Tensor:
    return vals.new_empty(shape)


@zeros_and_scatter.register_vmap  # type: ignore[misc]
def _(info, indims, shape, indices, value):  # type: ignore[no-untyped-def]
    """The batching rule is special in that it returns a tensor that is not batched"""
    indices_indims = indims[1]
    expanded_indices = []
    for idx, idx_indim in zip(indices, indices_indims):
        # The index is not a being batched, we should unsqueeze and expand to val
        if idx_indim is None:
            expanded_indices.append(idx.expand(value.shape))
        else:
            # the index is being part of the vmap batch, it should be the same size as val
            assert idx.shape == value.shape
            expanded_indices.append(idx)

    out = smith.ops.flex_lib.zeros_and_scatter(
        shape,
        expanded_indices,
        value,
    )
    return out, None


class ModIndex(smith.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(x: Tensor, indices: list[Tensor]) -> Tensor:
        return smith.ops.aten.index(x, indices)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        x, indices = inputs
        ctx.save_for_backward(*indices)
        ctx.input_shape = x.shape

    @staticmethod
    def backward(ctx, gradOut):  # type: ignore[no-untyped-def]
        indices = ctx.saved_tensors
        return (
            smith.ops.flex_lib.zeros_and_scatter(
                ctx.input_shape,
                indices,
                gradOut,
            ),
            None,
        )

    @classmethod
    @smith._export.wrappers.allow_in_pre_dispatch_graph
    def apply(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().apply(*args, **kwargs)


mod_index = ModIndex.apply


class TransformGetItemToIndex(SmithFunctionMode):
    # This is needed since we want to support calling
    # A[q_idx], where q_idx is a scalar tensor in score_mod.
    # Today, when q_idx is a scalar tensor, we implicitly convert it to a python
    # scalar and create a view. We do not want that behavior in this case, so we
    # use this smithfunctionmode to override that behavior for score_mod
    # wherever we're running it.
    #
    # We also convert integer indices to 0-D tensors so that temp[0] produces
    # the same backward graph as temp[0 * q_idx] (zeros_and_scatter with atomic_add).
    def __smith_function__(
        self,
        func: OpOverload,
        types: tuple[smith._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: Optional[dict[str, object]] = None,
    ) -> object:
        if func is smith.Tensor.__getitem__:
            tensor_to_index = args[0]
            assert isinstance(tensor_to_index, smith.Tensor)
            index_args = pytree.tree_leaves(args[1])
            if all(isinstance(x, (smith.Tensor, int)) for x in index_args):
                converted_indices = [
                    smith.tensor(x, dtype=smith.int64, device=tensor_to_index.device)
                    if isinstance(x, int)
                    else x
                    for x in index_args
                ]
                return mod_index(tensor_to_index, converted_indices)
        return func(*args, **(kwargs or {}))


def trace_wrapped(*args: Any, **kwargs: Any) -> Any:
    with smith.no_grad():
        return _trace_wrapped_op(*args, **kwargs)


class TraceWrapped(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("trace_wrapped")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(*args, **kwargs)


# TODO(jansel): need to ensure this does not get DCEed
_trace_wrapped_op = TraceWrapped()


def _assert_meta(
    grad: smith.Tensor,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: smith.dtype,
) -> smith.Tensor:
    assert grad.size() == size, "size mismatch"
    assert grad.stride() == stride, "stride mismatch"
    assert grad.dtype == dtype, "dtype mismatch"
    return grad


@_trace_wrapped_op.py_impl(ProxySmithDispatchMode)
def inner_trace(
    mode: ProxySmithDispatchMode,
    *args: Any,
    bw_state: Optional[BackwardState] = None,
    **kwargs: Any,
) -> Any:
    def self_invoke(*args: Any, **dyn_kwargs: Any) -> Any:
        with smith.no_grad():
            return _trace_wrapped_op(*args, **dyn_kwargs, **kwargs)

    def unwrap_proxies(x: Any) -> Any:
        if isinstance(x, smith.Tensor):
            return mode.tracer.unwrap_proxy(x)  # type: ignore[union-attr]
        if isinstance(x, (list, tuple)):
            return type(x)(map(unwrap_proxies, x))
        if x is None:
            return None
        raise AssertionError(f"unhandled type: {type(x)}")

    proxy_kwargs = {}
    if bw_state is not None:
        assert isinstance(bw_state, BackwardState) and bw_state.proxy is not None
        proxy_kwargs["bw_state"] = bw_state.proxy
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        self_invoke,
        unwrap_proxies(args),
        proxy_kwargs,
        name="trace_wrapped",
    )

    if args[0] is None:
        grad = args[1]  # module backward hooks
    else:
        grad = args[0]  # other backward hooks
    grad = tree_map_only(smith.Tensor, smith.empty_like, grad)
    track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    return grad


@_trace_wrapped_op.py_impl(FakeTensorMode)
def inner_fake(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("This op should never be invoked here")


@_trace_wrapped_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _trace_wrapped_op_dense(*args: Any, fn: Any, **kwargs: Any) -> Any:
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return fn(*args, **kwargs)


_trace_wrapped_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_trace_wrapped_op, deferred_error=True)
)


@_trace_wrapped_op.py_functionalize_impl
def _trace_wrapped_functionalized(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(_trace_wrapped_op(*unwrapped_args, **kwargs))


def autograd_function_backward_rewritten(original_backward: Any) -> Any:
    def new_backward(ctx: Any, *grads: Any) -> Any:
        # pyrefly: ignore [bad-assignment]
        grads = [g.contiguous() for g in grads]
        return original_backward(ctx, *grads)

    return new_backward
