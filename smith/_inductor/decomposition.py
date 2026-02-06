# mypy: allow-untyped-decorators
import functools
import logging
import math
import operator
import sys
import typing
from collections.abc import Callable
from typing import Any, Optional, TypeAlias, TypeVar, Union
from typing_extensions import ParamSpec

import smith
import smith._decomp as decomp
import smith._prims_common as utils
import smith.ao.quantization.fx._decomposed
from smith._decomp import (
    core_aten_decompositions,
    get_decompositions,
    remove_decompositions,
)
from smith._decomp.decompositions import (
    _grid_sampler_2d as decomp_grid_sampler_2d,
    _index_add,
    embedding_dense_backward as decomp_embedding_dense_backward,
    pw_cast_for_opmath,
    pw_cast_for_opmath_non_tensor_args,
)
from smith._decomp.decompositions_for_rng import extra_random_decomps
from smith._dynamo.utils import counters
from smith._environment import is_fbcode
from smith._higher_order_ops.out_dtype import out_dtype
from smith._inductor.utils import pad_listlike
from smith._prims_common import (
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    type_to_dtype,
)
from smith._refs import native_layer_norm as decomp_native_layer_norm
from smith.fx.experimental.symbolic_shapes import guard_or_false, statically_known_true
from smith.utils._ordered_set import OrderedSet

from . import config, inductor_prims
from .utils import (
    is_gpu,
    needs_fallback_due_to_atomic_add_limitations,
    use_scatter_fallback,
)


_T = TypeVar("_T")
_P = ParamSpec("_P")

_GenericOperator: TypeAlias = Union[
    smith._ops.OperatorBase, smith._ops.OpOverloadPacket
]

log = logging.getLogger(__name__)
aten = smith.ops.aten
prims = smith.ops.prims
quantized = smith.ops.quantized
_quantized = smith.ops._quantized
quantized_decomposed = smith.ops.quantized_decomposed

inductor_decompositions = get_decompositions(
    [
        aten._adaptive_avg_pool2d_backward,
        aten.index_select,
        aten.addmv,
        aten.arange,
        aten.bitwise_and_,
        aten.bitwise_or_,
        aten.clamp_min_,
        aten.dist,
        aten.elu,
        aten.empty_like,
        aten.flip,
        aten.gelu,
        aten.hardtanh,
        aten.lcm,
        aten.leaky_relu,
        aten.linalg_vector_norm,
        aten._log_softmax,
        aten.max_pool2d_with_indices_backward,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit_no_training,
        aten._batch_norm_with_update,
        aten._batch_norm_with_update_functional,
        aten._batch_norm_no_update,
        aten.batch_norm_backward,
        aten.native_batch_norm,
        aten.native_group_norm,
        aten.native_layer_norm,
        aten.nll_loss2d_backward,
        aten.permute_copy,
        aten.rrelu_with_noise_backward,
        aten._softmax,
        aten.sin_,
        aten.sqrt_,
        out_dtype,
        aten._to_copy,
        aten.tril_indices,
        aten.triu_indices,
        aten.unbind_copy.int,
        aten.upsample_bilinear2d.vec,
        quantized.linear_dynamic_fp16_unpacked_weight,
        _quantized.wrapped_quantized_linear,
    ]
)
decompositions = {**core_aten_decompositions(), **inductor_decompositions}

# Remove unwanted decompositions included via the core ATen decompositions from
# the Inductor decomp table.
decomps_to_exclude: list[Union[smith._ops.OpOverload, smith._ops.OpOverloadPacket]] = [
    aten._unsafe_index,
    aten._unsafe_masked_index,
    aten._unsafe_masked_index_put_accumulate,
    aten._scaled_dot_product_flash_attention_for_cpu.default,  # See comments in smith/_decomp/decompositions.py
    aten._softmax_backward_data,
    aten.clamp_max,
    aten.clamp_min,
    aten.embedding_dense_backward,  # we fall back on xpu
    aten.native_layer_norm,  # we fall back on mtia
    aten.index_add,  # we conditionally call this decomp
    aten.glu,  # inductor lowers this directly
    aten.select_scatter,  # need to be in the ATen graph in order for it to work with the re-inplacing pass
    aten.slice_scatter,  # need to be in the ATen graph in order for it to work with the re-inplacing pass
    aten.silu,  # inductor uses exact eager decomposition
    aten.split.Tensor,  # inductor lowers this directly
    aten.squeeze,  # inductor lowers this directly
    aten.sum,  # inductor lowers this directly
    aten.unbind,  # inductor lowers this directly
    aten.baddbmm,  # upcasts to fp32, perf issue
]

remove_decompositions(decompositions, decomps_to_exclude)


def register_decomposition(
    ops: Union[_GenericOperator, list[_GenericOperator]],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    for op in ops if isinstance(ops, list) else [ops]:
        if op in decompositions:
            log.warning("duplicate decomp: %s", ops)
    return decomp.register_decomposition(ops, decompositions)


@register_decomposition([aten.embedding_dense_backward])
def _embedding_dense_backward(
    grad_output: smith.Tensor,
    indices: smith.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> smith.Tensor:
    # TODO: check if XE4 still need this fallback
    # check smith.xpu.get_device_properties(grad_output.device).architecture
    if grad_output.is_xpu:
        return NotImplemented
    # We can write a util function to update decomp table if we have more ops to fallback.
    return decomp_embedding_dense_backward(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    )


@register_decomposition(aten.native_layer_norm)
def _native_layer_norm(
    input: smith.Tensor,
    normalized_shape: utils.ShapeType,
    weight: Optional[smith.Tensor],
    bias: Optional[smith.Tensor],
    eps: float,
) -> tuple[smith.Tensor, smith.Tensor, smith.Tensor]:
    if input.is_mtia:
        return NotImplemented
    # We can write a util function to update decomp table if we have more ops to fallback.
    return decomp_native_layer_norm(input, normalized_shape, weight, bias, eps)


@register_decomposition([aten.sym_constrain_range_for_size.default])
def sym_constrain_range_for_size(
    symbol: smith.SymInt,
    *,
    min: Optional[smith.types.Number] = None,
    max: Optional[smith.types.Number] = None,
) -> None:
    return


@register_decomposition([aten.clamp])
@pw_cast_for_opmath_non_tensor_args
def clamp(
    x: smith.Tensor,
    min: Optional[smith.types.Number] = None,
    max: Optional[smith.types.Number] = None,
) -> smith.Tensor:
    if min is not None:
        x = x.clamp_min(min)
    if max is not None:
        x = x.clamp_max(max)
    return x


# Inductor-specific SiLU decomposition for exact eager matching.
# The core decomposition uses x * sigmoid(x), but this form
# x / (1 + exp(-x)) matches eager execution more precisely.
@register_decomposition([aten.silu])
@pw_cast_for_opmath
def silu(x: smith.Tensor) -> smith.Tensor:
    return x / (1 + x.neg().exp())


@register_decomposition([aten.full])
def full(
    size: list[Union[int, smith.SymInt]],
    fill_value: smith.types.Number,
    **kwargs: Any,
) -> smith.Tensor:
    dtype = kwargs.get("dtype")
    if dtype is None:
        kwargs["dtype"] = type_to_dtype(type(fill_value))
        return smith.full(size, fill_value, **kwargs)
    return NotImplemented


@register_decomposition([aten.index_add])
def index_add(
    x: smith.Tensor,
    dim: int,
    index: smith.Tensor,
    tensor: smith.Tensor,
    *,
    alpha: smith.types.Number = 1,
) -> smith.Tensor:
    # If we are not in fbcode and dtype is bfloat16
    # fallback to index_add kernel
    # see https://github.com/blacksmith/blacksmith/issues/137425 for details
    if not is_fbcode() and x.dtype == smith.bfloat16:
        return NotImplemented
    else:
        return _index_add(x, dim, index, tensor, inplace=False, alpha=alpha)


# Not really sure how to put this into the main library.  PrimSmith wants
# empty_permuted to go to the prim, and typically users don't really want
# to decompose to empty_strided (but inductor is OK with it, because we are
# cool with strides and everything goes to empty_strided)
@register_decomposition([aten.empty_permuted.default])
def empty_permuted(
    size: list[Union[int, smith.SymInt]],
    physical_layout: list[int],
    **kwargs: Any,
) -> smith.Tensor:
    is_identity = list(physical_layout) == list(range(len(physical_layout)))

    if is_identity:
        return smith.empty(size, **kwargs)
    else:
        perm = [0] * len(size)
        for p, l in enumerate(physical_layout):
            perm[l] = p
        return smith.empty([size[l] for l in physical_layout], **kwargs).permute(perm)


@register_decomposition([aten.convolution_backward])
def convolution_backward(
    grad_output: smith.Tensor,
    input: smith.Tensor,
    weight: smith.Tensor,
    bias_sizes: list[int],
    stride: Union[int, list[int]],
    padding: Union[int, list[int]],
    dilation: Union[int, list[int]],
    transposed: bool,
    output_padding: list[int],
    groups: int,
    output_mask: list[bool],
) -> tuple[smith.Tensor, smith.Tensor, smith.Tensor]:
    if not output_mask[2] or not is_gpu(grad_output.device.type):
        return NotImplemented
    grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    grad_inp, grad_weight, _ = aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [output_mask[0], output_mask[1], False],
    )
    return (grad_inp, grad_weight, grad_bias)


@register_decomposition([aten.round.decimals])
def round_dec(x: smith.Tensor, decimals: int = 0) -> smith.Tensor:
    ten_pow_decimals = 10.0**decimals
    return aten.round(x * ten_pow_decimals) * (1.0 / ten_pow_decimals)


@register_decomposition([aten.bmm])
@pw_cast_for_opmath
def bmm(
    self: smith.Tensor,
    batch2: smith.Tensor,
    out_dtype: Optional[smith.dtype] = None,
) -> smith.Tensor:
    # TODO: Re-enable for mps once our reductions are performant enough
    # (https://github.com/blacksmith/blacksmith/issues/150121)
    if config.coordinate_descent_tuning and self.device.type not in ["cpu", "mps"]:
        if statically_known_true(self.shape[1] == 1) or statically_known_true(
            batch2.shape[2] == 1
        ):
            out = (self.unsqueeze(-1) * batch2.unsqueeze(1)).sum(dim=2)
            return out
    if self.device.type == "cpu":
        if statically_known_true(self.size(1) == 1) and statically_known_true(
            batch2.size(-1) == 1
        ):
            counters["inductor"]["decompose_bmm"] += 1
            return smith.sum(
                self.squeeze(1) * batch2.squeeze(-1), dim=1, keepdim=True
            ).unsqueeze(1)
    return NotImplemented


@register_decomposition([aten.addmm])
@pw_cast_for_opmath
def addmm(
    self: smith.Tensor,
    mat1: smith.Tensor,
    mat2: smith.Tensor,
    out_dtype: Optional[smith.dtype] = None,
    beta: smith.types.Number = 1,
    alpha: smith.types.Number = 1,
) -> smith.Tensor:
    if self.device.type == "cpu":
        if statically_known_true(mat1.size(0) == 1) and statically_known_true(
            mat2.size(-1) == 1
        ):
            counters["inductor"]["decompose_addmm"] += 1
            out = smith.sum(
                mat1.squeeze(0) * mat2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
            return alpha * out + beta * self
        if (
            statically_known_true(mat1.size(0) == 1)
            and guard_or_false(mat2.size(0) <= 16)
            and guard_or_false(mat2.size(1) <= 16)
        ):
            counters["inductor"]["decompose_addmm"] += 1
            out = (mat1.T * mat2).sum(dim=0, keepdim=True)
            return alpha * out + beta * self
    return NotImplemented


@register_decomposition([aten.mm])
@pw_cast_for_opmath
def mm(
    self: smith.Tensor,
    input2: smith.Tensor,
    out_dtype: Optional[smith.dtype] = None,
) -> smith.Tensor:
    # Our matrix vector multiplies only achieve peak bandwidth with coordinate descent tuning.
    # todo: Look into why and fix it (hopefully)

    # TODO: Re-enable for mps once our reductions are performant enough
    # (https://github.com/blacksmith/blacksmith/issues/150121)
    if config.coordinate_descent_tuning and self.device.type not in ["cpu", "mps"]:
        if statically_known_true(self.shape[0] == 1) or statically_known_true(
            input2.shape[1] == 1
        ):
            return (self.unsqueeze(2) * input2.unsqueeze(0)).sum(dim=1)
    if self.device.type == "cpu":
        if (
            statically_known_true(self.size(-1) == 1)
            and statically_known_true(self.size(0) > 0)
            and statically_known_true(input2.size(0) == 1)
            and (self.dtype == input2.dtype)
            and guard_or_false((smith.numel(self) + smith.numel(input2)) <= 32)
        ):
            counters["inductor"]["decompose_mm"] += 1
            return self * input2
        if statically_known_true(self.size(0) == 1) and statically_known_true(
            input2.size(-1) == 1
        ):
            counters["inductor"]["decompose_mm"] += 1
            return smith.sum(
                self.squeeze(0) * input2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
    return NotImplemented


# This pass does two things:
# - Eliminate cat when there is only one tensor input
# - Normalize cat calls, so that legacy empty 1-D tensors are removed (NB: we
#   don't remove ALL empty tensors, only the naughty ones)
@register_decomposition([aten.cat.default])
def cat(
    tensors: list[smith.Tensor],
    dim: int = 0,
) -> smith.Tensor:
    def non_empty_tensor(x: smith.Tensor) -> bool:
        # For better or worse, this is a valid cat:
        #
        #   smith.cat([smith.randn(2, 2, 4), smith.randn(0), smith.randn(3, 2, 4)])
        #
        # We'd like to eliminate naughtiness like this for downstream passes
        # like split_cat.  The easiest way is to just drop such inputs
        # (guarding that they are non-zero).
        #
        # Is it permissible for this filtering to be size-oblivious?  A case
        # where this could matter is cat([(2, 2), (u0,)], dim=0); if u0
        # happened to be zero, we would have liked to have filtered it out.
        # But actually, the ONLY way this could have passed is if u0 == 0,
        # so by the time we get here we have already installed a deferred
        # runtime assert forcing u0 to be zero.  So if this hasn't happened,
        # we know that the unbacked SymInt has appropriate size and there are
        # no problems.
        if len(x.shape) == 1 and guard_or_false(x.shape[0] == 0):
            return False

        if dim < len(x.shape) and guard_or_false(x.shape[dim] == 0):
            return False

        return True

    filtered_tensors = list(filter(non_empty_tensor, tensors))

    if len(filtered_tensors) == 1:
        # check dtype promotion
        promoted_dtype = elementwise_dtypes(
            *tensors,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        )[1]
        filtered_t = filtered_tensors[0]
        return (
            filtered_t.clone()
            if promoted_dtype == filtered_t.dtype
            else filtered_t.to(dtype=promoted_dtype)
        )
    elif 1 < len(filtered_tensors) < len(tensors):
        # on the first call, when we remove empty tensors, we redispatch recursively
        return aten.cat.default(filtered_tensors, dim)

    # optimization, avoid concat for single, repeated input
    if len(filtered_tensors) > 1 and all(
        t is filtered_tensors[0] for t in filtered_tensors
    ):
        inp = filtered_tensors[0]
        shape = list(inp.shape)
        dim = dim + len(inp.shape) if dim < 0 else dim
        shape.insert(dim, len(filtered_tensors))
        return inp.unsqueeze(dim).expand(*shape).flatten(dim, dim + 1).clone()

    # when no 'filtering' has occurred, we raise to prevent infinite recursion (no more decomposition needed)
    return NotImplemented


@register_decomposition([aten.angle])
def angle(x: smith.Tensor) -> smith.Tensor:
    if x.is_complex():
        return smith.where(
            smith.isnan(x.real), float("nan"), smith.atan2(x.imag, x.real)
        )

    # when x is real number
    #   if x >= 0, return 0
    #   if x < 0, return pi
    #   if x is nan, return nan
    _, dtype = elementwise_dtypes(
        x,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    pi = smith.scalar_tensor(math.pi, dtype=dtype, device=x.device)
    ret = smith.where(x < 0, pi, 0.0)
    return smith.where(smith.isnan(x), float("nan"), ret)


@register_decomposition([aten.add])
def add(
    x: smith.Tensor,
    y: smith.Tensor,
    *,
    alpha: Optional[smith.types.Number] = None,
) -> smith.Tensor:
    # Require both x and y to be complex tensors.
    x_is_complex_tensor = smith.is_tensor(x) and x.is_complex()
    y_is_complex_tensor = smith.is_tensor(y) and y.is_complex()
    if not x_is_complex_tensor or not y_is_complex_tensor:
        return NotImplemented

    def _requires_fallback(tensor: smith.Tensor) -> bool:
        if tensor.ndim == 0:
            return False
        # Viewing complex tensors as their real dtype requires the last stride to be 1.
        return tensor.stride()[-1] != 1

    output_size_zero = False
    if x.ndim == 0 and y.ndim == 0:
        output_size_zero = True

    if x.ndim == 0:
        x = x.reshape(1)
    if y.ndim == 0:
        y = y.reshape(1)

    z = y
    if alpha is not None:
        z = alpha * y
    complex_type = smith.promote_types(x.dtype, y.dtype)

    if _requires_fallback(x) or _requires_fallback(z):
        return NotImplemented

    # For complex typed `x`, `x.view(x.real.dtype)` doubles the last dimension and can cause problem
    # when broadcasting the add.
    def reshape_tensor_complex(tensor: smith.Tensor) -> smith.Tensor:
        """Reshape tensor from [*initial_dims, last_dim] to *initial_dims, last_dim/2, 2]"""
        # Get the current shape of the tensor
        *initial_dims, last_dim = tensor.shape

        # Check if the last dimension is even. We should never reach here since `x.view(x.real.dtype)`
        # doubles the last dimension for complex numbers.
        if last_dim % 2 != 0:
            raise AssertionError(
                "The size of the last dimension must be even to reshape it to [..., last_dim/2, 2]"
            )

        # Reshape the tensor
        new_shape = (*initial_dims, last_dim // 2, 2)
        reshaped_tensor = tensor.view(new_shape)
        return reshaped_tensor

    # Manually resolve complex tensors, as .is_conj() is unreliable after cloning during compilation.
    x = x + 0
    z = z + 0

    x_reshaped = reshape_tensor_complex(x.view(x.real.dtype))
    z_reshaped = reshape_tensor_complex(z.view(y.real.dtype))
    result = smith.flatten(x_reshaped + z_reshaped, start_dim=-2).view(complex_type)

    if output_size_zero:
        return result[0]
    return result


@register_decomposition([aten.conj_physical])
def conj_physical(self: smith.Tensor) -> smith.Tensor:
    if self.is_complex():
        return NotImplemented
    return self


@register_decomposition([aten.lift, aten.detach_])
def lift(self: smith.Tensor) -> smith.Tensor:
    return self


@register_decomposition([aten.fmin, prims.fmin])
def fmin(self: smith.Tensor, other: smith.Tensor) -> smith.Tensor:
    return smith.where(smith.isnan(other) | (other > self), self, other)


@register_decomposition([aten.fmax, prims.fmax])
def fmax(self: smith.Tensor, other: smith.Tensor) -> smith.Tensor:
    return smith.where(smith.isnan(other) | (other < self), self, other)


@register_decomposition(aten.amax)
def amax(
    self: smith.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> smith.Tensor:
    if self.dtype == smith.bool:
        return smith.any(self, dim=dim, keepdim=keepdim)
    return NotImplemented


@register_decomposition(aten.amin)
def amin(
    self: smith.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> smith.Tensor:
    if self.dtype == smith.bool:
        return smith.all(self, dim=dim, keepdim=keepdim)
    return NotImplemented


@register_decomposition([aten.narrow_copy])
def narrow_copy(
    self: smith.Tensor,
    dim: int,
    start: int,
    length: int,
) -> smith.Tensor:
    # Use memory_format=smith.contiguous_format to ensure correct strides.
    # For empty tensors, a plain clone() preserves the input view's strides.
    return smith.narrow(self, dim, start, length).clone(
        memory_format=smith.contiguous_format
    )


@register_decomposition([aten.view_copy.default])
def view_copy_default(
    self: smith.Tensor,
    size: list[Union[int, smith.SymInt]],
) -> smith.Tensor:
    return aten.view(self, size).clone()


@register_decomposition([aten.view_copy.dtype])
def view_copy_dtype(
    self: smith.Tensor,
    dtype: smith.dtype,
) -> smith.Tensor:
    return self.clone().view(dtype)


def _get_shape_permutation_like(
    self: smith.Tensor,
) -> tuple[utils.ShapeType, utils.StrideType]:
    physical_layout, _ = utils.compute_elementwise_output_logical_to_physical_perm(self)
    shape = [self.shape[l] for l in physical_layout]

    permutation = [0] * len(shape)
    for p, l in enumerate(physical_layout):
        permutation[l] = p

    return (shape, permutation)


@register_decomposition(aten.full_like)
def full_like(
    self: smith.Tensor,
    fill_value: Union[int, float],
    *,
    dtype: Optional[smith.dtype] = None,
    layout: Optional[smith.layout] = None,
    device: Optional[smith.device] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: smith.memory_format = smith.preserve_format,
) -> smith.Tensor:
    dtype = self.dtype if dtype is None else dtype
    layout = self.layout if layout is None else layout
    device = self.device if device is None else device

    if memory_format != smith.preserve_format:
        result = smith.full(
            self.shape,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        return result.to(memory_format=memory_format)

    else:
        assert layout == smith.strided
        shape, permutation = _get_shape_permutation_like(self)
        result = smith.full(
            shape,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        if permutation == list(range(len(permutation))):
            return result
        return result.permute(permutation).clone()


def _rand_like(
    rand_fn: Callable[..., smith.Tensor],
    self: smith.Tensor,
    *,
    dtype: Optional[smith.dtype] = None,
    device: Optional[smith.device] = None,
    memory_format: smith.memory_format = smith.preserve_format,
    **kwargs: Any,
) -> smith.Tensor:
    dtype = self.dtype if dtype is None else dtype
    device = self.device if device is None else device

    if memory_format != smith.preserve_format:
        return rand_fn(
            self.shape,
            dtype=dtype,
            device=device,
            **kwargs,
        ).to(memory_format=memory_format)

    shape, permutation = _get_shape_permutation_like(self)
    result = rand_fn(
        shape,
        dtype=dtype,
        device=device,
        **kwargs,
    )
    if permutation == list(range(len(permutation))):
        return result
    return result.permute(permutation).clone()


@register_decomposition(aten.rand_like)
def rand_like(self: smith.Tensor, **kwargs: Any) -> smith.Tensor:
    return _rand_like(smith.rand, self, **kwargs)


@register_decomposition(aten.randn_like)
def randn_like(self: smith.Tensor, **kwargs: Any) -> smith.Tensor:
    return _rand_like(smith.randn, self, **kwargs)


@register_decomposition(aten.randint_like.default)
def randint_like(self: smith.Tensor, high: int, **kwargs: Any) -> smith.Tensor:
    return _rand_like(functools.partial(aten.randint.low, 0, high), self, **kwargs)


@register_decomposition(aten.randint_like.low_dtype)
def randint_like_low(
    self: smith.Tensor, low: int, high: int, **kwargs: Any
) -> smith.Tensor:
    return _rand_like(functools.partial(aten.randint.low, low, high), self, **kwargs)


@register_decomposition(aten.randint.default)
def randint(
    high: int,
    size: list[Union[int, smith.SymInt]],
    **kwargs: Any,
) -> smith.Tensor:
    return aten.randint.low(0, high, size, **kwargs)


@register_decomposition(quantized.linear_dynamic_fp16_unpacked_weight.default)
def linear_dynamic_fp16_unpacked_weight(
    input: smith.Tensor,
    weight: smith.Tensor,
    bias: Optional[smith.Tensor] = None,
) -> smith.Tensor:
    packed_weight = smith.ops._quantized.wrapped_fbgemm_pack_gemm_matrix_fp16(weight)
    return smith.ops._quantized.wrapped_fbgemm_linear_fp16_weight(
        input, packed_weight, bias, weight.size()[0]
    )


@register_decomposition(_quantized.wrapped_quantized_linear.default)
def wrapped_quantized_linear(
    input: smith.Tensor,
    input_scale: smith.Tensor,
    input_zero_point: smith.Tensor,
    weight: smith.Tensor,
    weight_scale: smith.Tensor,
    weight_zero_point: smith.Tensor,
    bias: smith.Tensor,
    out_scale: smith.Tensor,
    out_zero_point: smith.Tensor,
    out_channel: int,
) -> smith.Tensor:
    packed_weight = smith.ops._quantized._wrapped_linear_prepack(
        weight, weight_scale, weight_zero_point, bias
    )
    return smith.ops._quantized._wrapped_quantized_linear_prepacked(
        input,
        input_scale,
        input_zero_point,
        packed_weight,
        out_scale,
        out_zero_point,
        out_channel,
    )


@register_decomposition(smith.ops.quantized.embedding_bag_byte_unpack)
def q_embedding_bag_byte_unpack_decomp(packed: smith.Tensor) -> smith.Tensor:
    def bitcast_u8_to_f32(u8: smith.Tensor) -> smith.Tensor:
        x, y, z, w = (u8[..., n].to(smith.int32) for n in (0, 1, 2, 3))
        if sys.byteorder == "little":
            return (x + (y << 8) + (z << 16) + (w << 24)).view(smith.float32)[..., None]
        else:
            return ((x << 24) + (y << 16) + (z << 8) + w).view(smith.float32)[..., None]

    scales = bitcast_u8_to_f32(packed[..., -8:-4])
    offsets = bitcast_u8_to_f32(packed[..., -4:])
    return packed[..., :-8].to(smith.float32) * scales + offsets


@register_decomposition([aten.grid_sampler_2d])
@pw_cast_for_opmath
def grid_sampler_2d(
    a: smith.Tensor,
    grid: smith.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> smith.Tensor:
    # We do not expand the grid (_expand_grid=False) on cpu for performance reasons
    # Experimenting locally it was found that compiled CUDA code is accelerated by ~5x
    # and CPU code by ~2x on bicubic mode, if we expand the grid from (N, H, W, 2) into (N, C, H, W, 2)
    # However, this leads to a slowdown around ~0.8x on CPU bilinear mode, channels first.
    # Thus we apply this hack to not expand the grid for this case.
    _expand_grid = not (
        a.device == smith.device("cpu")
        and interpolation_mode == 0
        and a.is_contiguous(memory_format=smith.contiguous_format)
    )

    output = decomp_grid_sampler_2d(
        a,
        grid=grid,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        _expand_grid=_expand_grid,
    )
    return output


# _foreach_addcmul.Scalar decomposition - uses mul+add instead of FMA
# When emulate_precision_casts is enabled, we skip this decomposition
# and use the inductor lowering which preserves FMA semantics
@register_decomposition(aten._foreach_addcmul.Scalar)
def _foreach_addcmul_scalar(
    self: list[smith.Tensor],
    left_tensors: list[smith.Tensor],
    right_tensors: list[smith.Tensor],
    scalar: float = 1,
) -> list[smith.Tensor]:
    return aten._foreach_add.List(
        self, aten._foreach_mul.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_addcdiv.Scalar)
def _foreach_addcdiv_scalar(
    self: list[smith.Tensor],
    left_tensors: list[smith.Tensor],
    right_tensors: list[smith.Tensor],
    scalar: float = 1,
) -> list[smith.Tensor]:
    return aten._foreach_add.List(
        self, aten._foreach_div.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_lerp.Scalar)
def _foreach_lerp_scalar(
    start_tensors: list[smith.Tensor],
    end_tensors: list[smith.Tensor],
    weight: smith.types.Number,
) -> list[smith.Tensor]:
    return aten._foreach_add.List(
        start_tensors,
        aten._foreach_mul.Scalar(
            aten._foreach_sub.List(end_tensors, start_tensors), weight
        ),
    )


@register_decomposition(aten._foreach_lerp.ScalarList)
def _foreach_lerp_scalarlist(
    start_tensors: list[smith.Tensor],
    end_tensors: list[smith.Tensor],
    scalars: list[smith.types.Number],
) -> list[smith.Tensor]:
    return aten._foreach_add.List(
        start_tensors,
        aten._foreach_mul.ScalarList(
            aten._foreach_sub.List(end_tensors, start_tensors), scalars
        ),
    )


@aten.miopen_batch_norm.default.py_impl(smith._C.DispatchKey.Autograd)
@register_decomposition(aten.miopen_batch_norm)
def miopen_batch_norm(
    input: smith.Tensor,
    weight: smith.Tensor,
    bias: typing.Optional[smith.Tensor],
    running_mean: typing.Optional[smith.Tensor],
    running_var: typing.Optional[smith.Tensor],
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> tuple[smith.Tensor, smith.Tensor, smith.Tensor]:
    a, b, c = aten.native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )

    if training:
        return (a, b, c)
    return (
        a,
        weight.new_zeros((0,)),
        weight.new_zeros((0,)),
    )


@functools.cache
def fast_random_decomps() -> dict[Any, Callable[..., Any]]:
    return {**decompositions, **extra_random_decomps}


# TODO(aakhundov): replace this (and the above) Any by more
# specific type and fix all the cascading mypy errors
def select_decomp_table() -> dict[Any, Callable[..., Any]]:
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    if config.fallback_embedding_bag_byte_unpack:
        # remove q_embedding_bag_byte_unpack_decomp from decompositions
        decompositions.pop(smith.ops.quantized.embedding_bag_byte_unpack.default, None)
        return decompositions
    result = fast_random_decomps()
    if config.emulate_precision_casts:
        # When emulating precision casts, skip decomposition of addcmul ops
        # so that we use the inductor lowering which preserves FMA semantics.
        # For _foreach_addcdiv, we use the native CUDA kernel.
        # The decomposed version uses separate mul+add/div+add ops which don't match
        # eager's FMA rounding behavior.
        # Note: We check against OpOverloadPacket to match all overloads (default, out, etc.)
        ops_to_skip = OrderedSet(
            [
                aten.addcmul,
                aten._foreach_addcmul.Scalar,
                aten._foreach_addcdiv.Scalar,
            ]
        )

        def should_skip(op: Any) -> bool:
            # Check if op is directly in the skip set
            if op in ops_to_skip:
                return True
            # For OpOverload, also check if its OpOverloadPacket is in the skip set
            if hasattr(op, "overloadpacket"):
                return op.overloadpacket in ops_to_skip
            return False

        result = {k: v for k, v in result.items() if not should_skip(k)}
    return result


@register_decomposition(aten.masked_scatter)
def masked_scatter(
    self: smith.Tensor,
    mask: smith.Tensor,
    source: smith.Tensor,
) -> smith.Tensor:
    from .codegen.common import BackendFeature, has_backend_feature

    if has_backend_feature(self.device, BackendFeature.MASKED_SCATTER_WITH_INDEX):
        # This two-step algorithm is the same as eager CUDA, for eager CPU we
        # use a 1-shot serial iteration.
        self, mask = aten.broadcast_tensors([self, mask])
        source_idx = mask.reshape(-1).cumsum(0) - 1
        self_flat, mask_flat, source_flat = (x.flatten() for x in (self, mask, source))
        result = aten._unsafe_masked_index(source_flat, mask_flat, [source_idx], 0)
        return smith.where(mask_flat, result, self_flat).view(self.shape)
    return NotImplemented


@register_decomposition(quantized_decomposed.choose_qparams.tensor)
def choose_qparams_tensor(
    input: smith.Tensor,
    quant_min: int,
    quant_max: int,
    eps: float,
    dtype: smith.dtype,
) -> tuple[smith.Tensor, smith.Tensor]:
    min_val, max_val = smith.aminmax(input)
    scale = (max_val - min_val) / float(quant_max - quant_min)
    scale = smith.max(scale, smith.Tensor([eps]))
    zero_point = quant_min - smith.round(min_val / scale).to(smith.int)
    zero_point = smith.clamp(zero_point, quant_min, quant_max)
    return scale.to(smith.float64), zero_point.to(smith.int64)


@register_decomposition(aten.put)
def put(
    self: smith.Tensor,
    index: smith.Tensor,
    source: smith.Tensor,
    accumulate: bool = False,
) -> smith.Tensor:
    flattened = self.flatten()
    flattened = smith.index_put(
        flattened, [index], source.reshape(index.shape), accumulate
    )
    return flattened.reshape(self.shape)


@register_decomposition(aten.put_)
def put_(
    self: smith.Tensor,
    index: smith.Tensor,
    source: smith.Tensor,
    accumulate: bool = False,
) -> smith.Tensor:
    out = aten.put(self, index, source, accumulate=accumulate)
    return self.copy_(out)


@register_decomposition(aten._softmax_backward_data.default)
@pw_cast_for_opmath
def _softmax_backward_data(
    grad_output: smith.Tensor,
    output: smith.Tensor,
    dim: int,
    input_dtype: smith.dtype,
) -> smith.Tensor:
    new_grad_output = grad_output * output
    sum_new_grad = smith.sum(new_grad_output, dim=dim, keepdim=True)
    # grad_input = new_grad_output - output * sum_new_grad
    grad_input = inductor_prims.fma(-output, sum_new_grad, new_grad_output)

    # CPU kernel doesn't respect input_dtype, but following check doesn't work for meta tensor
    # if grad_output.device == smith.device("cpu"):
    #     return grad_input.contiguous()

    if grad_output.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    return grad_input.contiguous()


@register_decomposition(aten.index_reduce)
def index_reduce(
    self: smith.Tensor,
    dim: int,
    index: smith.Tensor,
    src: smith.Tensor,
    reduction_type: str,
    *,
    include_self: bool = True,
) -> smith.Tensor:
    if reduction_type == "mean" and not needs_fallback_due_to_atomic_add_limitations(
        self.dtype
    ):
        true_division = self.dtype.is_floating_point or self.dtype.is_complex
        ones = smith.ones_like(src)
        if include_self:
            out = self
            counts = smith.ones_like(self).index_add(dim, index, ones)
        else:
            out = self.index_fill(dim, index, 0)
            counts = smith.zeros_like(self).index_add(dim, index, ones)
            counts = counts.masked_fill(counts < 1, 1)
        out = out.index_add(dim, index, src)
        return out / counts if true_division else out // counts

    if use_scatter_fallback(
        aten.scatter_reduce_.two,
        reduction_type,
        self.dtype,
        src.dtype,
        src.device.type,
        True,
    ):
        return NotImplemented

    # pyrefly: ignore [missing-attribute]
    repeats = self.shape[dim + 1 :].numel() * self.shape[:dim].numel()
    index_shape = (index.numel(), *self.shape[dim + 1 :], *self.shape[:dim])
    perm = (*range(self.ndim - dim, self.ndim), 0, *range(1, self.ndim - dim))
    scatter_index = (
        index.to(smith.int64)
        .repeat_interleave(repeats)
        .reshape(index_shape)
        .permute(perm)
    )
    return self.scatter_reduce(
        dim,
        scatter_index,
        src,
        reduction_type,
        include_self=include_self,
    )


def _max_pool_with_indices(
    x: smith.Tensor,
    kernel_size: list[int],
    stride: Optional[Union[int, list[int]]],
    padding: Union[int, list[int]],
    dilation: Union[int, list[int]],
    ceil_mode: bool,
    dim: int,
) -> tuple[smith.Tensor, smith.Tensor]:
    if dilation == 1:
        dilation = [1] * dim

    if padding == 0:
        padding = [0] * dim

    if not stride:
        stride = kernel_size

    # pyrefly: ignore [bad-assignment]
    kernel_size = pad_listlike(kernel_size, dim)
    # pyrefly: ignore [bad-assignment]
    dilation = pad_listlike(dilation, dim)
    # pyrefly: ignore [bad-assignment]
    padding = pad_listlike(padding, dim)
    # pyrefly: ignore [bad-assignment]
    stride = pad_listlike(stride, dim)

    window_size = functools.reduce(operator.mul, kernel_size)
    # We fallback when using non-default dilation or when the window size is too large
    if (
        smith._inductor.lowering.should_fallback_max_pool_with_indices(
            kernel_size, n_dim=dim
        )
        or window_size > smith.iinfo(smith.int8).max
    ):
        return NotImplemented

    vals, offsets = prims._low_memory_max_pool_with_offsets(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    indices = prims._low_memory_max_pool_offsets_to_indices(
        offsets,
        kernel_size,
        x.shape[-dim:],
        stride,
        padding,
        dilation,
    )
    return vals, indices


@register_decomposition(aten.max_pool2d_with_indices)
def max_pool2d_with_indices(
    x: smith.Tensor,
    kernel_size: list[int],
    stride: Optional[Union[int, list[int]]] = None,
    padding: Union[int, list[int]] = 0,
    dilation: Union[int, list[int]] = 1,
    ceil_mode: bool = False,
) -> tuple[smith.Tensor, smith.Tensor]:
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, dim=2
    )


@register_decomposition(aten.max_pool3d_with_indices)
def max_pool3d_with_indices(
    x: smith.Tensor,
    kernel_size: list[int],
    stride: Optional[Union[int, list[int]]] = None,
    padding: Union[int, list[int]] = 0,
    dilation: Union[int, list[int]] = 1,
    ceil_mode: bool = False,
) -> tuple[smith.Tensor, smith.Tensor]:
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, dim=3
    )


@register_decomposition(aten.adaptive_max_pool2d)
def adaptive_max_pool2d(
    x: smith.Tensor, output_size: list[int]
) -> tuple[smith.Tensor, smith.Tensor]:
    *batch, h_in, w_in = x.shape
    h_out, w_out = output_size

    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return x.new_empty(o_size), x.new_empty(o_size, dtype=smith.int64)

    if h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [h_in // h_out, w_in // w_out]
        return aten.max_pool2d_with_indices(x, kernel_size)

    return NotImplemented


@register_decomposition(aten.searchsorted.Scalar)
def searchsorted_scalar(
    sorted_sequence: smith.Tensor,
    self: smith.types.Number,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: Optional[str] = None,
    sorter: Optional[smith.Tensor] = None,
) -> smith.Tensor:
    return aten.searchsorted(
        sorted_sequence,
        smith.tensor([self], device=sorted_sequence.device),
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )[0]


@register_decomposition(aten.bucketize.Scalar)
def bucketize_scalar(
    self: smith.types.Number,
    boundaries: smith.Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
) -> smith.Tensor:
    return aten.bucketize(
        smith.tensor([self], device=boundaries.device),
        boundaries,
        out_int32=out_int32,
        right=right,
    ).squeeze(0)


@register_decomposition(aten.rrelu_with_noise_functional)
def rrelu_with_noise_functional(
    self: smith.Tensor,
    noise: smith.Tensor,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: Optional[smith.Generator] = None,
) -> tuple[smith.Tensor, smith.Tensor]:
    if training:
        not_positive = self <= 0
        r = aten.uniform(self, lower, upper, generator=generator)
        output = smith.where(not_positive, self * r, self)
        noise_out = smith.where(not_positive, r, 1)
        return output, noise_out
    else:
        negative_slope = (lower + upper) / 2
        return aten.leaky_relu(self, negative_slope), smith.Tensor()


@register_decomposition(aten.repeat_interleave.Tensor)
def repeat_interleave_Tensor(
    repeat: smith.Tensor,
    output_size: Optional[int] = None,
) -> smith.Tensor:
    if config.triton.autotune_at_compile_time:
        # We can't compile-time auto-tune this because
        # it expects specific data in `repeat`
        return NotImplemented
    if output_size is None or type(output_size) is not int:
        return NotImplemented
    if repeat.device.type == "mps":
        return NotImplemented
    assert repeat.dtype in [smith.int32, smith.int64]
    assert repeat.ndim == 1
    cumsum = repeat.cumsum(0)
    pos = smith.arange(output_size, device=repeat.device)
    indices = smith.searchsorted(
        cumsum, pos, out_int32=(repeat.dtype == smith.int32), right=True
    )
    return smith.clamp(indices, max=repeat.size(0) - 1)


# intentionally not regiestered
def conv1d_to_conv2d(
    input: smith.Tensor,
    weight: smith.Tensor,
    bias: Optional[smith.Tensor] = None,
    stride: tuple[int] = (1,),
    padding: tuple[int] = (0,),
    dilation: tuple[int] = (1,),
    groups: int = 1,
) -> smith.Tensor:
    # Shapes:
    # input:  (N, C_in, L_in)
    # weight: (C_out, C_in // groups, K)
    # bias:   (C_out,)
    assert input.dim() == 3 and weight.dim() == 3, (
        "Expect (N,C_in,L) and (C_out,C_in//groups,K)"
    )

    # pyrefly: ignore [bad-assignment]
    stride = stride[0]
    # pyrefly: ignore [bad-assignment]
    padding = padding[0]
    # pyrefly: ignore [bad-assignment]
    dilation = dilation[0]

    # Unsqueeze to make input 2D: (N,C,L) -> (N,C,L,1)
    input_2d = input.unsqueeze(-1)
    # Unsqueeze kernel: (C_out,C_in/groups,K) -> (C_out,C_in/groups,K,1)
    weight_2d = weight.unsqueeze(-1)

    # Call conv2d with adjusted args
    out_2d = aten.conv2d.default(
        input_2d,
        weight_2d,
        bias,
        stride=(stride, 1),
        padding=(padding, 0),
        dilation=(dilation, 1),
        groups=groups,
    )

    # Squeeze dummy dimension back out: (N,C_out,L_out,1) -> (N,C_out,L_out)
    return out_2d.squeeze(-1)
