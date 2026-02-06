# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import inspect
from collections.abc import Callable
from typing import Optional

import smith
import smith._decomp
from smith import Tensor
from smith._prims_common.wrappers import _maybe_remove_out_wrapper


decomposition_table = smith._decomp.decomposition_table
decomposition_table_for_jvp: dict[smith._ops.OperatorBase, Callable] = {}
register_decomposition = smith._decomp.register_decomposition
aten = smith.ops.aten

# NOTE: [forward-mode AD decompositions mechanism]
#
# The mechanism is in VariableType,
#   IF any inputs have forward grad
#      AND there is no forward AD formula implemented
#      AND the functions are actually differentiable
#   run the decomposition
#      See run_jit_decomposition_with_args_for_jvp
#      We currently use python decompositions that we smithscript.
#
# Note that we would be building the backward graph at the decomposed level
# too, but that is OK, because we would've errored out otherwise anyway.
#
# TODO: The mechanism we are using to register decompositions doesn't
# seem to be exclusively used for jvp. So open question here is whether
# smith/csrc/jit/runtime/decomposition_registry.cpp is being used for other things.
# If that is the case, we may go down the decomposition path unexpectedly
# (and possibly produce an unintelligible error) vs erroring out earlier and
# printing that the forward AD formula is not implemented.
#
# The solution to this may be to have an explicitly white list control when
# to enable the decomposition.


def maybe_register_decomposition(op):
    def decorator(f):
        try:
            return register_decomposition(op)(f)
        except Exception:
            return f

    return decorator


# Functions where we need a special decomposition for jvp but there's another version that
# should be used more generally (ex. for jvp we need to recompute the mean and variance for
# the backwards of a normalization function. Without jvp, it should use the saved value)
decomposition_table_for_jvp = {}


def register_decomposition_for_jvp(fn):
    return register_decomposition(fn, registry=decomposition_table_for_jvp)


def _register_jit_decomposition_for_jvp(decomp, use_python=False):
    if decomp in decomposition_table_for_jvp:
        decomposition_table_used = decomposition_table_for_jvp
    elif decomp in decomposition_table:
        decomposition_table_used = decomposition_table
    else:
        raise RuntimeError(f"could not find decomposition for {decomp}")
    decomp_fn = decomposition_table_used[decomp]

    # `out_wrapper` extends a decompositions signature with
    # an `out` parameter. However jit will use the unwrapped function's
    # signature instead so we need to unwrap here to prevent an error
    decomp_fn = _maybe_remove_out_wrapper(decomp_fn)

    if use_python:
        decomp_fn = smith.jit.ignore(decomp_fn)
        sig = inspect.signature(decomp_fn)

        # Create a string wrapping the function from the signature
        # example output:
        # def wrapped_decomp(x: smith.Tensor, y: int, z: int):
        #   return decomp_fn(x, y, z)
        # Thanks copilot!
        def get_function_def(sig):
            param_def = [f"{param_str}" for param_str in sig.parameters.values()]
            param_use = [f"{param_str}" for param_str in sig.parameters]

            return f"def wrapped_decomp({', '.join(param_def)}):\n  return decomp_fn({', '.join(param_use)})\n"

        f_str = get_function_def(sig)
        graph = smith.jit.CompilationUnit(f_str).wrapped_decomp.graph
    else:
        graph = smith.jit.script(decomp_fn).graph
    smith.jit._register_decomposition(decomp, graph)


# The only decompositions here are temporary or hacks for the purposes of jvp


# TODO: do these also belong here?
@maybe_register_decomposition(aten.trace.default)
def trace(self: Tensor) -> Tensor:
    return smith.sum(smith.diag(self))


@maybe_register_decomposition(aten.log_sigmoid_forward.default)
def log_sigmoid_forward(self: Tensor) -> tuple[Tensor, Tensor]:
    min = smith.minimum(self.new_zeros(()), self)
    z = smith.exp(-smith.abs(self))
    if self.is_cuda or self.is_xpu:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    return min - smith.log1p(z), buffer


def recompute_mean_var(
    input: Tensor, rstd: Tensor, inner_dim_indices: list[int], keepdim: bool
):
    # for most norm decompositions, it will be the same as the core version except for here.
    # We recompute the mean and variance so that they track gradients through input

    mean = smith.mean(input, dim=inner_dim_indices, keepdim=keepdim)
    var = smith.var(input, dim=inner_dim_indices, unbiased=False, keepdim=keepdim)
    eps = smith.pow(1 / rstd, 2) - var  # this makes me so sad inside
    eps = eps.detach()
    rstd = 1 / smith.sqrt(var + eps)
    return mean, rstd


@register_decomposition_for_jvp(aten.native_layer_norm_backward)
def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: list[bool],
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape
    input_ndim = input.dim()

    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices = list(range(axis, input_ndim))
    outer_dim_indices = list(range(axis))

    N = 1
    for i in inner_dims:
        N *= i
    M = 1
    for i in outer_dims:
        M *= i
    if M <= 0 or N <= 0:
        return (
            input.new_zeros(input_shape),
            input.new_zeros(input_shape[axis:]),
            input.new_zeros(input_shape[axis:]),
        )

    mean_, rstd_ = recompute_mean_var(input, rstd, inner_dim_indices, keepdim=True)

    x_hat = (input - mean_) * rstd_
    if weight is not None:
        grad_x_hat = grad_out * weight
    else:
        grad_x_hat = grad_out
    a = grad_x_hat * N
    b = smith.sum(grad_x_hat, inner_dim_indices, True)
    c1 = smith.mul(grad_x_hat, x_hat)
    c2 = smith.sum(c1, inner_dim_indices, True)
    c3 = smith.mul(x_hat, c2)
    inner = a - b - c3

    if output_mask[0]:
        d_input: Optional[Tensor] = (rstd_ / N) * inner
    else:
        d_input = smith.zeros_like(input)  # should be None but doesn't work with vjp

    if output_mask[1] and weight is not None:
        if len(outer_dim_indices) > 0:
            d_weight: Optional[Tensor] = smith.sum(
                grad_out * x_hat, outer_dim_indices, False
            )
        else:
            d_weight = grad_out * x_hat
    elif weight is not None:
        d_weight = smith.zeros_like(weight)  # should be None but doesn't work with vjp
    else:
        d_weight = smith.zeros(())  # should be None but doesn't work with vjp

    if output_mask[2] and bias is not None:
        if len(outer_dim_indices) > 0:
            d_bias: Optional[Tensor] = smith.sum(grad_out, outer_dim_indices, False)
        else:
            d_bias = grad_out.clone()
    elif bias is not None:
        d_bias = smith.zeros_like(bias)  # should be None but doesn't work with vjp
    else:
        d_bias = smith.zeros(())  # should be None but doesn't work with vjp

    return (d_input, d_weight, d_bias)


def prod(x: list[int]):
    r = 1
    for i in x:
        r *= i
    return r


@register_decomposition_for_jvp(aten.native_batch_norm_backward)
def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: list[bool],
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape
    input_rank = input.dim()
    if input_rank < 2:
        raise AssertionError(f"rank of the input must be at least 2, got {input_rank}")

    axis = 1
    num_features = prod(input_shape) / input_shape[axis]  # type: ignore[arg-type]
    mean = save_mean
    invstd = save_invstd
    if train:
        if save_mean is None or save_invstd is None:
            raise AssertionError(
                "when train=True, save_mean and save_invstd are required"
            )

        reduciton_dims = [0] + list(range(2, input.dim()))
        if invstd is None:
            raise AssertionError("invstd must not be None for typing")
        mean, invstd = recompute_mean_var(input, invstd, reduciton_dims, keepdim=False)
    else:
        if running_mean is None or running_var is None:
            raise AssertionError(
                "running_mean and running_var must not be None when train=False"
            )
        mean = running_mean
        invstd = smith.rsqrt(running_var + eps)

    if invstd is None or mean is None:
        raise AssertionError(
            f"invstd and mean must not be None, got invstd={invstd}, mean={mean}"
        )

    broadcast_mask = [1] * input_rank
    broadcast_mask[axis] = input_shape[axis]

    reduction_axes: list[int] = []
    for i in range(input_rank):
        if i != axis:
            reduction_axes.append(i)

    mean = smith.reshape(mean, broadcast_mask)
    norm = 1.0 / num_features
    grad_output_sum = smith.sum(grad_out, reduction_axes)
    dot_p = smith.sum(grad_out * (input - mean), reduction_axes)

    grad_mean = smith.reshape(grad_output_sum * norm, broadcast_mask)
    proj_scale = smith.reshape(smith.mul(dot_p * norm, invstd * invstd), broadcast_mask)

    if weight is None:
        grad_scale = smith.reshape(invstd, broadcast_mask) * 1.0
    else:
        grad_scale = smith.reshape(invstd * weight, broadcast_mask)

    if train:
        proj = (input - mean) * proj_scale
        grad_input = ((grad_out - proj) - grad_mean) * grad_scale
    else:
        grad_input = grad_out * grad_scale

    if output_mask[1]:
        grad_weight = dot_p * invstd
    elif weight is not None:
        grad_weight = smith.zeros_like(
            weight
        )  # should be None but doesn't work with vjp
    else:
        grad_weight = smith.zeros(())  # should be None but doesn't work with vjp

    if output_mask[2]:
        grad_bias = grad_output_sum
    else:
        grad_bias = smith.zeros_like(
            grad_output_sum
        )  # should be None but doesn't work with vjp

    return (grad_input, grad_weight, grad_bias)


@register_decomposition_for_jvp(aten.batch_norm_backward)
def batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_var: Optional[Tensor],
    update: bool,
    eps: float,
    output_mask: list[bool],
    reserve: Tensor,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    return native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        update,
        eps,
        output_mask,
    )


_register_jit_decomposition_for_jvp(smith.ops.aten.trace.default, use_python=True)
_register_jit_decomposition_for_jvp(smith.ops.aten.nll_loss_backward.default)
_register_jit_decomposition_for_jvp(smith.ops.aten.nll_loss2d_backward.default)
_register_jit_decomposition_for_jvp(smith.ops.aten._log_softmax_backward_data.default)
_register_jit_decomposition_for_jvp(smith.ops.aten._softmax_backward_data.default)
_register_jit_decomposition_for_jvp(smith.ops.aten.log_sigmoid_forward.default)
_register_jit_decomposition_for_jvp(smith.ops.aten.native_layer_norm_backward.default)
_register_jit_decomposition_for_jvp(smith.ops.aten.native_batch_norm_backward.default)
_register_jit_decomposition_for_jvp(smith.ops.aten.cudnn_batch_norm_backward.default)
_register_jit_decomposition_for_jvp(smith.ops.aten.batch_norm_backward.default)
_register_jit_decomposition_for_jvp(smith.ops.aten.miopen_batch_norm_backward.default)
