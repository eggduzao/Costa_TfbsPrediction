# mypy: allow-untyped-defs

import smith
import smith.utils._pytree as pytree
from smith._C import DispatchKey
from smith._higher_order_ops.utils import autograd_not_implemented
from smith._ops import HigherOrderOperator
from smith._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from smith._subclasses.fake_tensor import FakeTensorMode
from smith.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    maybe_handle_decomp,
    ProxySmithDispatchMode,
    track_tensor_tree,
)


# TODO to figure out a more generic approach
ALLOWABLE_OPS = [
    smith.ops.aten.linear.default,
    smith.ops.aten.mm.default,
    smith.ops.aten.conv2d.default,
    smith.ops.aten.convolution.default,
    smith.ops.aten.mul.Tensor,
    smith.ops.aten.mul.Scalar,
    smith.ops.aten.div.Tensor,
    smith.ops.aten.div.Scalar,
]


class OutDtypeOperator(HigherOrderOperator):
    """
    The out_dtype operator takes an existing ATen functional operator, an
    `out_dtype` argument, and arguments to the original operator, and executes
    the original operator and returns a Tensor with the `out_dtype` precision.
    This operator does not mandate a compute precision so it allows the
    representation to not be opinionated about the exact implementation.

    The general implementation for all operators will be the following:
        1. Promote inputs dtypes based on default Blacksmith dtype promotion rules,
            using the dtypes of all input Tensors/Scalars and the `out_dtype`
            arugument.
        2. Execute the operator
        3. Cast the output to `out_dtype`
    """

    def __init__(self) -> None:
        super().__init__("out_dtype")

    def __call__(self, op, output_dtype, *args):
        if not isinstance(op, smith._ops.OpOverload):
            raise ValueError("out_dtype's first argument must be an OpOverload")
        if op._schema.is_mutable:
            raise ValueError(
                "out_dtype's first argument needs to be a functional operator"
            )
        if not (
            len(op._schema.returns) == 1
            and isinstance(op._schema.returns[0].type, smith.TensorType)
        ):
            raise ValueError(
                "out_dtype's can only apply to ops that return a single tensor"
                f"Instead got {[r.type for r in op._schema.returns]}"
            )

        if op not in ALLOWABLE_OPS:
            raise ValueError(
                f"out_dtype only allows the following operators: {ALLOWABLE_OPS}."
            )

        # pyrefly: ignore [missing-attribute]
        res = super().__call__(op, output_dtype, *args)

        return res


out_dtype = OutDtypeOperator()


def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args):
    # NB: Long-term we should put the decomposition logic into
    # ProxySmithDispatchMode so that people do not need to call maybe_handle_decomp
    # in all HigherOrderOp proxy implementations.
    r = maybe_handle_decomp(proxy_mode, func_overload, (op, output_dtype, *args), {})
    if r is not NotImplemented:
        return r

    with disable_proxy_modes_tracing():
        # This is a simplified implementation of this operator just for tracing.
        # Actual implementation may also first promote the arguments
        out = op(*args).to(dtype=output_dtype)

    node_args = (op, output_dtype, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="out_dtype"
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@out_dtype.py_impl(DispatchKey.CompositeExplicitAutograd)
def out_dtype_dense(op: smith._ops.OpOverload, output_dtype: smith.dtype, *args):
    if is_int_mm(op, output_dtype, args):
        return smith._int_mm(*args)
    return out_dtype_fallback(op, output_dtype, *args)


def is_int_mm(op, output_dtype, args):
    return (
        op is smith.ops.aten.mm.default
        and output_dtype == smith.int32
        and len(args) == 2
        and args[0].dtype == smith.int8
        and args[1].dtype == smith.int8
        and (args[0].is_cuda or args[0].is_xpu)
        and (args[1].is_cuda or args[1].is_xpu)
    )


def out_dtype_fallback(op, output_dtype, *args):
    flat_inputs = pytree.arg_tree_leaves(*args) + [smith.ones(1, dtype=output_dtype)]
    promote_dtype: smith.dtype = elementwise_dtypes(
        *flat_inputs,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )[0]

    casted_args = pytree.tree_map_only(
        smith.Tensor, lambda arg: arg.to(dtype=promote_dtype), args
    )
    res = op(*casted_args).to(dtype=output_dtype)
    return res


out_dtype.py_autograd_impl(autograd_not_implemented(out_dtype, deferred_error=True))


@out_dtype.py_impl(ProxySmithDispatchMode)
def out_dtype_proxy(
    mode: ProxySmithDispatchMode,
    op: smith._ops.OpOverload,
    output_dtype: smith.dtype,
    *args,
):
    return trace_out_dtype(mode, out_dtype, op, output_dtype, *args)


@out_dtype.py_impl(FakeTensorMode)
def out_dtype_fake_tensor_mode(
    mode: FakeTensorMode,
    op: smith._ops.OpOverload,
    output_dtype: smith.dtype,
    *args,
):
    with mode:
        return out_dtype_dense(op, output_dtype, *args)


@out_dtype.py_functionalize_impl
def out_dtype_func(ctx, op, output_dtype, *args):
    unwrapped_args = tuple(ctx.unwrap_tensors(arg) for arg in args)

    with ctx.redispatch_to_next():
        res = out_dtype(op, output_dtype, *unwrapped_args)
    return ctx.wrap_tensors(res)
