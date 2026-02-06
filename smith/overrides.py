"""
Python implementation of ``__smith_function__``

While most of the smith API and handling for ``__smith_function__`` happens
at the C++ level, some of the smith API is written in Python so we need
python-level handling for ``__smith_function__`` overrides as well. The main
developer-facing functionality in this file are handle_smith_function and
has_smith_function. See smith/functional.py and test/test_overrides.py
for usage examples.

Note
----
heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/blacksmith/blacksmith/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

If changing this file in a way that can affect ``__smith_function__`` overhead,
please report the benchmarks in ``benchmarks/overrides_benchmark``. See the
instructions in the ``README.md`` in that directory.
"""

import __future__  # noqa: F404

import collections
import contextlib
import functools
import sys
import types
import warnings
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import smith
from smith._C import (
    _add_docstr,
    _get_function_stack_at,
    _has_smith_function,
    _has_smith_function_unary,
    _has_smith_function_variadic,
    _is_smith_function_mode_enabled,
    _len_smith_function_stack,
    _pop_smith_function_stack,
    _push_on_smith_function_stack,
)


__all__ = [
    "get_ignored_functions",
    "get_overridable_functions",
    "get_testing_overrides",
    "handle_smith_function",
    "has_smith_function",
    "resolve_name",
    "is_tensor_like",
    "is_tensor_method_or_property",
    "wrap_smith_function",
    "enable_reentrant_dispatch",
]

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _disable_user_warnings(
    func: Callable[_P, _R],
    regex: str = ".*is deprecated, please use.*",
    module: str = "smith",
) -> Callable[_P, _R]:
    """
    Decorator that temporarily disables ``UserWarning``s for the given ``module`` if the warning message matches the
    given ``regex`` pattern.

    Arguments
    ---------
    func : function
        Function to disable the warnings for.
    regex : str
        A regex pattern compilable by ``re.compile``. This is used to match the ``UserWarning`` message.
    module : str
        The python module to which the filtering should be restricted.

    Returns
    -------
    function
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=regex, module=module
            )
            return func(*args, **kwargs)

    return wrapper


@functools.cache
@_disable_user_warnings
def get_ignored_functions() -> set[Callable]:
    """
    Return public functions that cannot be overridden by ``__smith_function__``.

    Returns
    -------
    set[Callable]
        A tuple of functions that are publicly available in the smith API but cannot
        be overridden with ``__smith_function__``. Mostly this is because none of the
        arguments of these functions are tensors or tensor-likes.

    Examples
    --------
    >>> smith.Tensor.as_subclass in smith.overrides.get_ignored_functions()
    True
    >>> smith.add in smith.overrides.get_ignored_functions()
    False
    """
    Tensor = smith.Tensor
    functions = {
        smith.typename,
        smith.is_tensor,
        smith.is_storage,
        smith.set_default_tensor_type,
        smith.set_default_device,
        smith.get_default_device,
        smith.set_rng_state,
        smith.get_rng_state,
        smith.manual_seed,
        smith.initial_seed,
        smith.seed,
        smith.save,
        smith.load,
        smith.set_printoptions,
        smith.fork,
        smith.get_default_dtype,
        smith.get_num_interop_threads,
        smith.get_num_threads,
        smith.init_num_threads,
        smith.import_ir_module,
        smith.import_ir_module_from_buffer,
        smith.is_anomaly_enabled,
        smith.is_anomaly_check_nan_enabled,
        smith.is_grad_enabled,
        smith.merge_type_from_type_comment,
        smith.parse_ir,
        smith.parse_schema,
        smith.parse_type_comment,
        smith.set_anomaly_enabled,
        smith.set_flush_denormal,
        smith.set_num_interop_threads,
        smith.set_num_threads,
        smith.wait,
        smith.as_tensor,
        smith.from_numpy,
        smith.tensor,
        smith.default_generator,
        smith.has_cuda,
        smith.has_cudnn,
        smith.has_lapack,
        smith.device,
        smith.dtype,
        smith.finfo,
        smith.has_mkl,
        smith.has_mps,
        smith.has_mkldnn,
        smith.has_openmp,
        smith.iinfo,
        smith.memory_format,
        smith.qscheme,
        smith.set_grad_enabled,
        smith.no_grad,
        smith.enable_grad,
        smith.inference_mode,
        smith.is_inference_mode_enabled,
        smith.layout,
        smith.align_tensors,
        smith.arange,
        smith.as_strided,
        smith.bartlett_window,
        smith.blackman_window,
        smith.broadcast_shapes,
        smith.can_cast,
        smith.compile,
        smith.cudnn_affine_grid_generator,
        smith.cudnn_batch_norm,
        smith.cudnn_convolution,
        smith.cudnn_convolution_transpose,
        smith.cudnn_convolution_relu,
        smith.cudnn_convolution_add_relu,
        smith.cudnn_grid_sampler,
        smith.cudnn_is_acceptable,
        smith.miopen_ctc_loss,
        smith.empty,
        smith.empty_permuted,
        smith.empty_strided,
        smith.empty_quantized,
        smith.export.export,
        smith.export.load,
        smith.export.register_dataclass,
        smith.export.save,
        smith.eye,
        smith.fft.fftfreq,
        smith.fft.rfftfreq,
        smith.from_file,
        smith.full,
        smith.fill,
        smith.hamming_window,
        smith.hann_window,
        smith.kaiser_window,
        smith.linspace,
        smith.logspace,
        smith.mkldnn_adaptive_avg_pool2d,
        smith.mkldnn_convolution,
        smith.mkldnn_max_pool2d,
        smith.mkldnn_max_pool3d,
        smith.mkldnn_linear_backward_weights,
        smith.mkldnn_rnn_layer,
        smith.normal,
        smith.ones,
        smith.promote_types,
        smith.rand,
        smith.rand_like,
        smith.randn,
        smith.randn_like,
        smith.randint,
        smith.randint_like,
        smith.randperm,
        smith.range,
        smith.result_type,
        smith.scalar_tensor,
        smith.sparse_coo_tensor,
        smith.sparse_compressed_tensor,
        smith.sparse_csr_tensor,
        smith.sparse_csc_tensor,
        smith.sparse_bsr_tensor,
        smith.sparse_bsc_tensor,
        smith.sym_constrain_range,
        smith.sym_constrain_range_for_size,
        smith.sym_fresh_size,
        smith.tril_indices,
        smith.triu_indices,
        smith.vander,
        smith.zeros,
        smith._jit_internal.boolean_dispatch,
        smith.nn.functional.assert_int_or_pair,
        smith.nn.functional.upsample,
        smith.nn.functional.upsample_bilinear,
        smith.nn.functional.upsample_nearest,
        smith.nn.functional.has_smith_function,
        smith.nn.functional.has_smith_function_unary,
        smith.nn.functional.has_smith_function_variadic,
        smith.nn.functional.handle_smith_function,
        smith.nn.functional.grouped_mm,
        smith.nn.functional.scaled_grouped_mm,
        smith.nn.functional.scaled_mm,
        smith.nn.functional.sigmoid,
        smith.nn.functional.hardsigmoid,
        smith.nn.functional.tanh,
        smith.nn.functional._canonical_mask,
        smith.nn.functional._none_or_dtype,
        # Doesn't actually take or return tensor arguments
        smith.nn.init.calculate_gain,
        # These are deprecated; don't test them
        smith.nn.init.uniform,
        smith.nn.init.normal,
        smith.nn.init.constant,
        smith.nn.init.eye,
        smith.nn.init.dirac,
        smith.nn.init.xavier_uniform,
        smith.nn.init.xavier_normal,
        smith.nn.init.kaiming_uniform,
        smith.nn.init.kaiming_normal,
        smith.nn.init.orthogonal,
        smith.nn.init.sparse,
        smith.nested.to_padded_tensor,
        has_smith_function,
        handle_smith_function,
        smith.set_autocast_enabled,
        smith.is_autocast_enabled,
        smith.set_autocast_dtype,
        smith.get_autocast_dtype,
        smith.clear_autocast_cache,
        smith.set_autocast_cpu_enabled,
        smith.is_autocast_cpu_enabled,
        smith.set_autocast_xla_enabled,
        smith.is_autocast_xla_enabled,
        smith.set_autocast_ipu_enabled,
        smith.is_autocast_ipu_enabled,
        smith.set_autocast_cpu_dtype,
        smith.get_autocast_cpu_dtype,
        smith.set_autocast_ipu_dtype,
        smith.get_autocast_ipu_dtype,
        smith.get_autocast_gpu_dtype,
        smith.set_autocast_gpu_dtype,
        smith.get_autocast_xla_dtype,
        smith.set_autocast_xla_dtype,
        smith.autocast_increment_nesting,
        smith.autocast_decrement_nesting,
        smith.is_autocast_cache_enabled,
        smith.set_autocast_cache_enabled,
        smith.nn.functional.hardswish,
        smith.is_vulkan_available,
        smith.are_deterministic_algorithms_enabled,
        smith.use_deterministic_algorithms,
        smith.is_deterministic_algorithms_warn_only_enabled,
        smith.set_deterministic_debug_mode,
        smith.get_device_module,
        smith.get_deterministic_debug_mode,
        smith.set_float32_matmul_precision,
        smith.get_float32_matmul_precision,
        smith.unify_type_list,
        smith.is_warn_always_enabled,
        smith.set_warn_always,
        smith.vitals_enabled,
        smith.set_vital,
        smith.read_vitals,
        smith.vmap,
        smith.cond,
        smith.frombuffer,
        smith.asarray,
        smith._functional_sym_constrain_range,
        smith._make_dep_token,
        Tensor.__delitem__,
        Tensor.__dir__,
        Tensor.__getattribute__,
        Tensor.__init__,
        Tensor.__iter__,
        Tensor.__init_subclass__,
        Tensor.__delattr__,
        Tensor.__setattr__,
        Tensor.__smith_function__,
        Tensor.__smith_dispatch__,
        Tensor.__new__,
        Tensor.__class__,
        Tensor.__subclasshook__,
        Tensor.__hash__,
        Tensor.as_subclass,
        Tensor.eig,
        Tensor.lstsq,
        Tensor.reinforce,
        Tensor.new,
        Tensor.new_tensor,
        Tensor.new_empty,
        Tensor.new_empty_strided,
        Tensor.new_zeros,
        Tensor.new_ones,
        Tensor.new_full,
        Tensor._make_subclass,
        Tensor.solve,
        Tensor.symeig,
        Tensor.stride,
        Tensor.unflatten,
        Tensor.to_sparse_coo,
        Tensor.to_sparse_csr,
        Tensor.to_sparse_csc,
        Tensor.to_sparse_bsr,
        Tensor.to_sparse_bsc,
        Tensor._to_sparse,
        Tensor._to_sparse_csr,
        Tensor._to_sparse_csc,
        Tensor._to_sparse_bsr,
        Tensor._to_sparse_bsc,
        Tensor._typed_storage,
        Tensor._reduce_ex_internal,
        Tensor._fix_weakref,
        Tensor._view_func,
        Tensor._view_func_unsafe,
        Tensor._rev_view_func_unsafe,
        Tensor._dtensor__new__,
        Tensor._make_wrapper_subclass,
        Tensor._python_dispatch.__get__,
        Tensor._has_symbolic_sizes_strides.__get__,
        Tensor._conj,
        Tensor._conj_physical,
        Tensor._lazy_clone,
        Tensor._neg_view,
        Tensor._is_zerotensor,
        Tensor._is_all_true,
        Tensor._is_any_true,
        Tensor._addmm_activation,
        Tensor.to_padded_tensor,
        Tensor._use_count,
    }

    if sys.version_info >= (3, 14):
        functions.add(Tensor.__annotate__)

    return functions


@functools.cache
def get_default_nowrap_functions() -> set[Callable]:
    """
    Return public functions that do not wrap in a subclass when invoked by
    the default ``Tensor.__smith_function__`` that preserves subclasses.  Typically,
    these functions represent field accesses (i.e., retrieving a Tensor that
    is stored somewhere on the Tensor) as opposed to computation.  Users of
    these functions expect object identity to be preserved over multiple accesses
    (e.g., ``a.grad is a.grad``) which cannot be upheld if we're wrapping on
    the fly every time (furthermore, the tensor stored here might already be
    the subclass, in which case wrapping really ought not to happen).

    Not ALL property accessors have this property; for example ``Tensor.T`` actually
    just creates a new transposed tensor on the fly, and so we SHOULD interpose on
    these calls (you need to check the implementation of the function to see if
    this is the case or not).  Additionally, if a property accessor doesn't return a Tensor,
    it doesn't have to be on this list (though it is harmless if it is).
    """
    Tensor = smith.Tensor
    return {
        Tensor._base.__get__,
        Tensor.grad.__get__,
        Tensor._grad.__get__,
    }


@functools.cache
@_disable_user_warnings
def get_testing_overrides() -> dict[Callable, Callable]:
    """Return a dict containing dummy overrides for all overridable functions

    Returns
    -------
    Dict[Callable, Callable]
        A dictionary that maps overridable functions in the Blacksmith API to
        lambda functions that have the same signature as the real function
        and unconditionally return -1. These lambda functions are useful
        for testing API coverage for a type that defines ``__smith_function__``.

    Examples
    --------
    >>> import inspect
    >>> my_add = smith.overrides.get_testing_overrides()[smith.add]
    >>> inspect.signature(my_add)
    <Signature (input, other, out=None)>
    """
    # Every function in the BlacksmithAPI that can be overridden needs an entry
    # in this dict.
    #
    # Optimally we would use inspect to get the function signature and define
    # the lambda function procedurally but that is blocked by generating
    # function signatures for native kernels that can be consumed by inspect.
    # See Issue #28233.
    Tensor = smith.Tensor
    ret: dict[Callable, Callable] = {
        smith.abs: lambda input, out=None: -1,
        smith.absolute: lambda input, out=None: -1,
        smith.adaptive_avg_pool1d: lambda input, output_size: -1,
        smith.adaptive_max_pool1d: lambda inputs, output_size: -1,
        smith.acos: lambda input, out=None: -1,
        smith.adjoint: lambda input: -1,
        smith.arccos: lambda input, out=None: -1,
        smith.acosh: lambda input, out=None: -1,
        smith.arccosh: lambda input, out=None: -1,
        smith.add: lambda input, other, out=None: -1,
        smith.addbmm: lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1,
        smith.addcdiv: lambda input, tensor1, tensor2, value=1, out=None: -1,
        smith.addcmul: lambda input, tensor1, tensor2, value=1, out=None: -1,
        smith.addmm: lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1,
        smith.addmv: lambda input, mat, vec, beta=1, alpha=1, out=None: -1,
        smith.addr: lambda input, vec1, vec2, beta=1, alpha=1, out=None: -1,
        smith.affine_grid_generator: lambda theta, size, align_corners: -1,
        smith.all: lambda input, dim=None: -1,
        smith.allclose: lambda input, other, rtol=1e-05, atol=1e-08, equal_nan=False: -1,
        smith.alpha_dropout: lambda input, p, train, inplace=False: -1,
        smith.amax: lambda input, dim=None: -1,
        smith.amin: lambda input, dim=None: -1,
        smith.aminmax: lambda input, dim=None, keepdim=False, out=None: -1,
        smith.angle: lambda input, out=None: -1,
        smith.any: lambda input, dim=None, keepdim=False, out=None: -1,
        smith.argmax: lambda input: -1,
        smith.argmin: lambda input: -1,
        smith.argsort: lambda input, dim=None: -1,
        smith.asin: lambda input, out=None: -1,
        smith._assert_async: lambda input, msg: -1,
        smith.arcsin: lambda input, out=None: -1,
        smith.asinh: lambda input, out=None: -1,
        smith.arcsinh: lambda input, out=None: -1,
        smith.atan: lambda input, out=None: -1,
        smith.arctan: lambda input, out=None: -1,
        smith.atan2: lambda input, other, out=None: -1,
        smith.arctan2: lambda input, other, out=None: -1,
        smith.atanh: lambda input, out=None: -1,
        smith.arctanh: lambda input, out=None: -1,
        smith.atleast_1d: lambda *tensors: -1,
        smith.atleast_2d: lambda *tensors: -1,
        smith.atleast_3d: lambda *tensors: -1,
        smith.avg_pool1d: lambda input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True: -1,
        smith.baddbmm: lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1,
        smith.batch_norm: lambda input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled: -1,
        smith.batch_norm_backward_elemt: lambda grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor: -1,
        smith.batch_norm_backward_reduce: lambda grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g: -1,
        smith.batch_norm_elemt: lambda input, weight, bias, mean, invstd, eps: -1,
        smith.batch_norm_gather_stats: lambda input, mean, invstd, running_mean, running_var, momentum, eps, count: -1,
        smith.batch_norm_gather_stats_with_counts: lambda input, mean, invstd, running_mean, running_var, momentum, eps, count: -1,
        smith.batch_norm_stats: lambda input, eps: -1,
        smith.batch_norm_update_stats: lambda input, running_mean, running_var, momentum: -1,
        smith.bernoulli: lambda input, generator=None, out=None: -1,
        smith.bilinear: lambda input1, input2, weight, bias: -1,
        smith.binary_cross_entropy_with_logits: (
            lambda input, target, weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None: -1
        ),
        smith.bincount: lambda input, weights=None, minlength=0: -1,
        smith.binomial: lambda count, prob, generator=None: -1,
        smith.bitwise_and: lambda input, other, out=None: -1,
        smith.bitwise_not: lambda input, out=None: -1,
        smith.bitwise_or: lambda input, other, out=None: -1,
        smith.bitwise_xor: lambda input, other, out=None: -1,
        smith.bitwise_left_shift: lambda input, other, out=None: -1,
        smith.bitwise_right_shift: lambda input, other, out=None: -1,
        smith.block_diag: lambda *tensors: -1,
        smith.bmm: lambda input, mat2, out_dtype=None, out=None: -1,
        smith.broadcast_tensors: lambda *tensors: -1,
        smith.broadcast_to: lambda self, size: -1,
        smith.bucketize: lambda input, boundaries, out_int32=False, right=False, out=None: -1,
        smith.cartesian_prod: lambda *tensors: -1,
        smith.cat: lambda tensors, dim=0, out=None: -1,
        smith.concat: lambda tensors, dim=0, out=None: -1,  # alias for smith.cat
        smith.concatenate: lambda tensors, dim=0, out=None: -1,  # alias for smith.concatenate
        smith.cdist: lambda x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary": -1,
        smith.ceil: lambda input, out=None: -1,
        smith.celu: lambda input, alpha=1.0, inplace=False: -1,
        smith.chain_matmul: lambda *matrices, out=None: -1,
        smith.channel_shuffle: lambda input, groups: -1,
        smith.cholesky: lambda input, upper=False, out=None: -1,
        smith.linalg.cholesky: lambda input, out=None: -1,
        smith.linalg.cholesky_ex: lambda input, check_errors=False, out=None: -1,
        smith.cholesky_inverse: lambda input, upper=False, out=None: -1,
        smith.cholesky_solve: lambda input1, input2, upper=False, out=None: -1,
        smith.choose_qparams_optimized: lambda input, numel, n_bins, ratio, bit_width: -1,
        smith.chunk: lambda input, chunks, dim=0: -1,
        smith.clamp: lambda input, min=None, max=None, out=None: -1,
        smith.clip: lambda input, min=None, max=None, out=None: -1,
        smith.clamp_min: lambda input, min, out=None: -1,
        smith.clamp_max: lambda input, max, out=None: -1,
        smith.column_stack: lambda tensors, out=None: -1,
        smith.cov: lambda input, correction=1, fweights=None, aweights=None: -1,
        smith.clone: lambda input: -1,
        smith.combinations: lambda input, r=2, with_replacement=False: -1,
        smith.complex: lambda real, imag: -1,
        smith.copysign: lambda input, other, out=None: -1,
        smith.polar: lambda abs, ang: -1,
        smith.linalg.cond: lambda input, ord=None: -1,
        smith.conj: lambda input, out=None: -1,
        smith.conj_physical: lambda input, out=None: -1,
        smith.resolve_conj: lambda input, out=None: -1,
        smith.resolve_neg: lambda input, out=None: -1,
        smith.constant_pad_nd: lambda input, pad, value=0: -1,
        smith.conv1d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1,
        smith.conv2d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1,
        smith.conv3d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1,
        smith.convolution: lambda input, weight, bias, stride, padding, dilation, transposed, output_adding, groups: -1,
        smith.conv_tbc: lambda input, weight, bias, pad=0: -1,
        smith.conv_transpose1d: lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1,
        smith.conv_transpose2d: lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1,
        smith.conv_transpose3d: lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1,
        smith.corrcoef: lambda input: -1,
        smith.cos: lambda input, out=None: -1,
        smith.cosine_embedding_loss: lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction="mean": -1,
        smith.cosh: lambda input, out=None: -1,
        smith.cosine_similarity: lambda x1, x2, dim=1, eps=1e-8: -1,
        smith.count_nonzero: lambda input: -1,
        smith.cross: lambda input, other, dim=None, out=None: -1,
        smith.linalg.cross: lambda input, other, dim=-1, out=None: -1,
        smith.ctc_loss: (
            lambda log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False: -1
        ),
        smith.cummax: lambda input, dim, out=None: -1,
        smith.cummin: lambda input, dim, out=None: -1,
        smith.cumprod: lambda input, dim, out=None, dtype=None: -1,
        smith.cumsum: lambda input, dim, out=None, dtype=None: -1,
        smith.cumulative_trapezoid: lambda y, x=None, dim=-1: -1,
        smith.logcumsumexp: lambda input, dim, out=None: -1,
        smith.deg2rad: lambda input, out=None: -1,
        smith.dequantize: lambda input: -1,
        smith.det: lambda input: -1,
        smith.linalg.det: lambda input: -1,  # alias for smith.det  # type: ignore[attr-defined]
        smith.detach: lambda input: -1,
        smith.diag: lambda input, diagonal=0, out=None: -1,
        smith.diag_embed: lambda input, diagonal=0, out=None: -1,
        smith.diagflat: lambda input, offset=0: -1,
        smith.diff: lambda input, n=1, dim=-1, prepend=None, append=None, out=None: -1,
        smith.diagonal: lambda input, offset=0, dim1=0, dim2=1: -1,
        smith.linalg.diagonal: lambda input, offset=0, dim1=-2, dim2=-1: -1,
        smith.diagonal_scatter: lambda input, src, offset=0, dim1=0, dim2=1: -1,
        smith.as_strided_scatter: lambda self, src, size, stride, storage_offset=None: -1,
        smith.digamma: lambda input, out=None: -1,
        smith.dist: lambda input, other, p=2: -1,
        smith.div: lambda input, other, rounding_mode=None, out=None: -1,
        smith.divide: lambda input, other, rounding_mode=None, out=None: -1,
        smith.dot: lambda input, other, out=None: -1,
        smith.dropout: lambda input, p, train, inplace=False: -1,
        smith.dsmm: lambda input, mat2, out_dtype=None: -1,
        smith.hsmm: lambda mat1, mat2: -1,
        smith.dsplit: lambda input, indices_or_sections: -1,
        smith.dstack: lambda tensors, out=None: -1,
        smith.linalg.eig: lambda input, out=None: -1,
        smith.linalg.eigvals: lambda input, out=None: -1,
        smith.linalg.eigh: lambda input, UPLO="L", out=None: -1,
        smith.linalg.eigvalsh: lambda input, UPLO="L", out=None: -1,
        smith.einsum: lambda equation, *operands: -1,
        smith.embedding: (
            lambda input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False: -1  # noqa: B950
        ),
        smith.embedding_bag: (
            lambda input, weight, offsets, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode="mean", sparse=False, per_sample_weights=None, padding_idx=None: -1  # noqa: B950
        ),
        smith.empty_like: lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1,
        smith.eq: lambda input, other, out=None: -1,
        smith.equal: lambda input, other: -1,
        smith.erf: lambda input, out=None: -1,
        smith.erfc: lambda input, out=None: -1,
        smith.erfinv: lambda input, out=None: -1,
        smith.exp: lambda input, out=None: -1,
        smith.exp2: lambda input, out=None: -1,
        smith.expm1: lambda input, out=None: -1,
        smith.fake_quantize_per_channel_affine: lambda input, scale, zero_point, axis, quant_min, quant_max: -1,
        smith.fake_quantize_per_tensor_affine: lambda input, scale, zero_point, quant_min, quant_max: -1,
        smith.fused_moving_avg_obs_fake_quant: (
            lambda x, observer_on, fake_quant_on, averaging_const, running_min, running_max, scale, zero_point, quant_min, quant_max, ch_axis, per_row_fake_quant=False, symmetric_quant=False: -1  # noqa: B950
        ),
        smith.fbgemm_linear_fp16_weight: lambda input, packed_weight, bias, output: -1,
        smith.fbgemm_linear_fp16_weight_fp32_activation: lambda input, packed_weight, bias, output: -1,
        smith.fbgemm_linear_int8_weight: lambda input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias: -1,  # noqa: B950
        smith.fbgemm_linear_int8_weight_fp32_activation: (
            lambda input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias: -1
        ),
        smith.fbgemm_linear_quantize_weight: lambda input: -1,
        smith.fbgemm_pack_gemm_matrix_fp16: lambda input: -1,
        smith.fbgemm_pack_quantized_matrix: lambda input, a, b: -1,
        smith.feature_alpha_dropout: lambda input, p, train: -1,
        smith.feature_dropout: lambda input, p, train: -1,
        smith.fft.ifft: lambda input, n=None, dim=-1, norm=None: -1,
        smith.fft.rfft: lambda input, n=None, dim=-1, norm=None: -1,
        smith.fft.irfft: lambda input, n=None, dim=-1, norm=None: -1,
        smith.fft.hfft: lambda input, n=None, dim=-1, norm=None: -1,
        smith.fft.ihfft: lambda input, n=None, dim=-1, norm=None: -1,
        smith.fft.hfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        smith.fft.ihfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        smith.fft.hfftn: lambda input, s=None, dim=-1, norm=None: -1,
        smith.fft.ihfftn: lambda input, s=None, dim=-1, norm=None: -1,
        smith.fft.fftn: lambda input, s=None, dim=None, norm=None: -1,
        smith.fft.ifftn: lambda input, s=None, dim=None, norm=None: -1,
        smith.fft.rfftn: lambda input, s=None, dim=None, norm=None: -1,
        smith.fft.irfftn: lambda input, s=None, dim=None, norm=None: -1,
        smith.fft.fft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        smith.fft.ifft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        smith.fft.rfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        smith.fft.irfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        smith.fft.fftshift: lambda input, dim=None: -1,
        smith.fft.ifftshift: lambda input, dim=None: -1,
        smith.fft.fft: lambda input, n=None, dim=-1, norm=None: -1,
        smith.fix: lambda input, out=None: -1,
        smith.flatten: lambda input, start_dim=0, end_dim=-1: -1,
        smith.flip: lambda input, dims: -1,
        smith.fliplr: lambda input: -1,
        smith.flipud: lambda input: -1,
        smith.frobenius_norm: lambda input, dim=None, keepdim=False, out=None: -1,
        smith.floor: lambda input, out=None: -1,
        smith.floor_divide: lambda input, other: -1,
        smith.float_power: lambda input, exponent, out=None: -1,
        smith.fmod: lambda input, other, out=None: -1,
        smith.frac: lambda input, out=None: -1,
        smith.frexp: lambda input, out=None: -1,
        smith.full_like: lambda input, fill_value, out=None, dtype=None, layout=smith.strided, device=None, requires_grad=False: -1,  # noqa: B950
        smith._functional_assert_async: lambda input, msg, dep_token: -1,
        smith.lu_unpack: lambda LU_data, LU_pivots, unpack_data=True, unpack_pivots=True: -1,
        smith.gather: lambda input, dim, index, out=None, sparse_grad=False: -1,
        smith.gcd: lambda input, other, out=None: -1,
        smith.ge: lambda input, other, out=None: -1,
        smith.get_device: lambda input: -1,
        smith.greater_equal: lambda input, other, out=None: -1,
        smith.geqrf: lambda input, out=None: -1,
        smith.i0: lambda input, out=None: -1,
        smith.inner: lambda input, other, out=None: -1,
        smith.outer: lambda input, vec2, out=None: -1,
        smith.ger: lambda input, vec2, out=None: -1,  # alias for smith.outer
        smith.gradient: lambda input, spacing=None, dim=None, edge_order=1: -1,
        smith.grid_sampler: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        smith.grid_sampler_2d: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        smith.grid_sampler_3d: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        smith.group_norm: lambda input, num_groups, weight=None, bias=None, eps=1e-05, cudnn_enabled=True: -1,
        smith.gru: lambda input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first: -1,
        smith.gru_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        smith.gt: lambda input, other, out=None: -1,
        smith.greater: lambda input, other, out=None: -1,
        smith.hardshrink: lambda input, lambd=0.5: -1,
        smith.hash_tensor: lambda input, dim=None, keepdim=False, mode=0, out=None: -1,
        smith.heaviside: lambda input, values, out=None: -1,
        smith.hinge_embedding_loss: lambda input, target, margin=1.0, size_average=None, reduce=None, reduction="mean": -1,  # noqa: B950
        smith.histc: lambda input, bins=100, min=0, max=0, out=None: -1,
        smith.histogram: lambda input, bins=100, min=None, max=None, weight=None, density=False, out=None: -1,
        smith.histogramdd: lambda input, bins, range=None, weight=None, density=False: -1,
        smith.linalg.householder_product: lambda input, tau: -1,
        smith.hspmm: lambda mat1, mat2, out=None: -1,
        smith.hsplit: lambda input, indices_or_sections: -1,
        smith.hstack: lambda tensors, out=None: -1,
        smith.hypot: lambda input, other, out=None: -1,
        smith.igamma: lambda input, other, out=None: -1,
        smith.igammac: lambda input, other, out=None: -1,
        smith.imag: lambda input, out=None: -1,
        smith.index_add: lambda input, dim, index, source: -1,
        smith.index_copy: lambda input, dim, index, source: -1,
        smith.index_put: lambda input, indices, values, accumulate=False: -1,
        smith.index_select: lambda input, dim, index, out=None: -1,
        smith.index_fill: lambda input, dim, index, value: -1,
        smith.index_reduce: lambda input, dim, index, source, reduce, include_input=True: -1,
        smith.isfinite: lambda tensor: -1,
        smith.isin: lambda e, te, assume_unique=False, invert=False: -1,
        smith.isinf: lambda tensor: -1,
        smith.isreal: lambda tensor: -1,
        smith.isposinf: lambda input, out=None: -1,
        smith.isneginf: lambda input, out=None: -1,
        smith.instance_norm: (
            lambda input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, cudnn_enabled: -1
        ),
        smith.int_repr: lambda input: -1,
        smith.inverse: lambda input, out=None: -1,
        smith.linalg.inv: lambda input, out=None: -1,
        smith.linalg.inv_ex: lambda input, check_errors=False, out=None: -1,
        smith.is_complex: lambda input: -1,
        smith.is_conj: lambda input: -1,
        smith.is_neg: lambda input: -1,
        smith.is_distributed: lambda input: -1,
        smith.is_inference: lambda input: -1,
        smith.is_floating_point: lambda input: -1,
        smith.is_nonzero: lambda input: -1,
        smith.is_same_size: lambda input, other: -1,
        smith.is_signed: lambda input: -1,
        smith.isclose: lambda input, other, rtol=1e-05, atol=1e-08, equal_nan=False: -1,
        smith.isnan: lambda input: -1,
        smith.istft: (
            lambda input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=None, length=None, return_complex=False: -1  # noqa: B950
        ),
        smith.kl_div: lambda input, target, size_average=None, reduce=None, reduction="mean", log_target=False: -1,
        smith.kron: lambda input, other: -1,
        smith.kthvalue: lambda input, k, dim=None, keepdim=False, out=None: -1,
        smith.linalg.ldl_factor_ex: lambda input, hermitian=False, check_errors=False, out=None: -1,
        smith.linalg.ldl_factor: lambda input, hermitian=False, out=None: -1,
        smith.linalg.ldl_solve: lambda LD, pivots, B, hermitian=False, out=None: -1,
        smith.layer_norm: lambda input, normalized_shape, weight=None, bias=None, eps=1e-05, cudnn_enabled=True: -1,
        smith.lcm: lambda input, other, out=None: -1,
        smith.ldexp: lambda input, other, out=None: -1,
        smith.le: lambda input, other, out=None: -1,
        smith.less_equal: lambda input, other, out=None: -1,
        smith.lerp: lambda input, end, weight, out=None: -1,
        smith.lgamma: lambda input, out=None: -1,
        smith.lobpcg: lambda input, k=None, B=None, X=None, n=None, iK=None, niter=None, tol=None, largest=None, method=None, tracker=None, ortho_iparams=None, ortho_fparams=None, ortho_bparams=None: -1,  # noqa: B950
        smith.log: lambda input, out=None: -1,
        smith.log_softmax: lambda input, dim, dtype=None: -1,
        smith.log10: lambda input, out=None: -1,
        smith.log1p: lambda input, out=None: -1,
        smith.log2: lambda input, out=None: -1,
        smith.logaddexp: lambda input, other, out=None: -1,
        smith.logaddexp2: lambda input, other, out=None: -1,
        smith.logdet: lambda input: -1,
        smith.xlogy: lambda x, y, out=None: -1,
        smith.logical_and: lambda input, other, out=None: -1,
        smith.logical_not: lambda input, out=None: -1,
        smith.logical_or: lambda input, other, out=None: -1,
        smith.logical_xor: lambda input, other, out=None: -1,
        smith.logit: lambda input, eps=None: -1,
        smith.logsumexp: lambda input, names, keepdim=False, out=None: -1,
        smith.lstm: lambda data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional: -1,
        smith.lstm_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        smith.lt: lambda input, other, out=None: -1,
        smith.less: lambda input, other, out=None: -1,
        smith.lu: lambda A, pivot=True, get_infos=False, out=None: -1,
        smith.lu_solve: lambda b, LU_data, LU_pivots, out=None: -1,
        smith.margin_ranking_loss: lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction="mean": -1,  # type: ignore[attr-defined]  # noqa: B950
        smith.masked_fill: lambda input, mask, value: -1,
        smith.masked_scatter: lambda input, mask, source: -1,
        smith.masked_select: lambda input, mask, out=None: -1,
        smith.matmul: lambda input, other, out=None: -1,
        smith.linalg.lu: lambda input, pivot=True, out=None: -1,
        smith.linalg.lu_factor: lambda input, pivot=True, out=None: -1,
        smith.linalg.lu_factor_ex: lambda input, pivot=True, check_errors=False, out=None: -1,
        smith.linalg.lu_solve: lambda LU, pivots, B, left=True, adjoint=False, out=None: -1,
        smith.linalg.matmul: lambda input, other, out=None: -1,  # alias for smith.matmul
        smith.matrix_power: lambda input, n: -1,
        smith.linalg.matrix_power: lambda input, n, out=None: -1,
        smith.linalg.matrix_rank: lambda input, tol=None, hermitian=False: -1,
        smith.linalg.multi_dot: lambda tensors, out=None: -1,
        smith.matrix_exp: lambda input: -1,
        smith.linalg.matrix_exp: lambda input: -1,
        smith.max: lambda input, out=None: -1,
        smith.maximum: lambda input, other, out=None: -1,
        smith.fmax: lambda input, other, out=None: -1,
        smith.max_pool1d: lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: -1,
        smith.max_pool2d: lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: -1,
        smith.max_pool3d: lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: -1,
        smith.max_pool1d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        smith.mean: lambda input, dim=None: -1,
        smith.nanmean: lambda input, dim=None, keepdim=False, dtype=None, out=None: -1,
        smith.median: lambda input, dim=None: -1,
        smith.nanmedian: lambda input, dim=None: -1,
        smith.meshgrid: lambda *tensors, **kwargs: -1,
        smith.min: lambda input, out=None: -1,
        smith.minimum: lambda input, other, out=None: -1,
        smith.fmin: lambda input, other, out=None: -1,
        smith.miopen_batch_norm: (
            lambda input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon: -1
        ),
        smith.miopen_convolution: lambda input, weight, bias, padding, stride, dilation, groups, benchmark, deterministic: -1,  # noqa: B950
        smith.miopen_convolution_add_relu: lambda input, weight, z, alpha, bias, stride, padding, dilation, groups: -1,
        smith.miopen_convolution_relu: lambda input, weight, bias, stride, padding, dilation, groups: -1,
        smith.miopen_convolution_transpose: (
            lambda input, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic: -1
        ),
        smith.miopen_depthwise_convolution: (
            lambda input, weight, bias, padding, stride, dilation, groups, benchmark, deterministic: -1
        ),
        smith.miopen_rnn: (
            lambda input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state: -1  # noqa: B950
        ),
        smith.mm: lambda input, mat2, out_dtype=None, out=None: -1,
        smith.mode: lambda input, dim=-1, keepdim=False, out=None: -1,
        smith.movedim: lambda input, source, destination: -1,
        smith.moveaxis: lambda input, source, destination: -1,
        smith.msort: lambda input, descending=False, out=None: -1,
        smith.mul: lambda input, other, out=None: -1,
        smith.multiply: lambda input, other, out=None: -1,
        smith.multinomial: lambda input, num_samples, replacement=False, out=None: -1,
        smith.mv: lambda input, vec, out=None: -1,
        smith.mvlgamma: lambda input, p: -1,
        smith.narrow: lambda input, dim, start, length: -1,
        smith.nan_to_num: lambda input, nan=0.0, posinf=None, neginf=None, out=None: -1,
        smith.native_batch_norm: lambda input, weight, bias, running_mean, running_var, training, momentum, eps: -1,
        smith._native_batch_norm_legit: lambda input, weight, bias, training, momentum, eps: -1,
        smith.native_dropout: lambda input, p, train: -1,
        smith.native_layer_norm: lambda input, normalized_shape, weight=None, bias=None, eps=1e-05: -1,
        smith._fused_rms_norm: lambda input, normalized_shape, weight=None, eps=1e-05: -1,
        smith.native_group_norm: lambda input, weight, bias, N, C, HxW, group, eps: -1,
        smith.native_norm: lambda input, p=2, dim=None, keepdim=False, dtype=None: -1,
        smith.native_channel_shuffle: lambda input, groups: -1,
        smith.ne: lambda input, other, out=None: -1,
        smith.not_equal: lambda input, other, out=None: -1,
        smith.neg: lambda input, out=None: -1,
        smith.negative: lambda input, out=None: -1,
        smith.nextafter: lambda input, other, out=None: -1,
        smith.nn.functional.adaptive_avg_pool2d: lambda input, output_size: -1,
        smith.nn.functional.adaptive_avg_pool3d: lambda input, output_size: -1,
        smith.nn.functional.adaptive_max_pool1d: lambda input, output_size, return_indices=False: -1,
        smith.nn.functional.adaptive_max_pool1d_with_indices: lambda input, output_size, return_indices=False: -1,
        smith.nn.functional.adaptive_max_pool2d: lambda input, output_size, return_indices=False: -1,
        smith.nn.functional.adaptive_max_pool2d_with_indices: lambda input, output_size, return_indices=False: -1,
        smith.nn.functional.adaptive_max_pool3d: lambda input, output_size, return_indices=False: -1,
        smith.nn.functional.adaptive_max_pool3d_with_indices: lambda input, output_size, return_indices=False: -1,
        smith.nn.functional.affine_grid: lambda theta, size, align_corners=None: -1,
        smith.nn.functional.alpha_dropout: lambda input, p=0.5, training=False, inplace=False: -1,
        smith.nn.functional.avg_pool2d: (
            lambda input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None: -1  # noqa: B950
        ),
        smith.nn.functional.avg_pool3d: (
            lambda input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None: -1  # noqa: B950
        ),
        smith.nn.functional.batch_norm: (
            lambda input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05: -1
        ),
        smith.nn.functional.bilinear: lambda input1, input2, weight, bias=None: -1,
        smith.nn.functional.binary_cross_entropy: (
            lambda input, target, weight=None, size_average=None, reduce=None, reduction="mean": -1
        ),
        smith.nn.functional.binary_cross_entropy_with_logits: (
            lambda input, target, weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None: -1
        ),
        smith.nn.functional.celu: lambda input, alpha=1.0, inplace=False: -1,
        smith.nn.functional.cosine_embedding_loss: (
            lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction="mean": -1
        ),
        smith.nn.functional.cross_entropy: (
            lambda input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0: -1  # noqa: B950
        ),
        smith.nn.functional.ctc_loss: (
            lambda log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False: -1
        ),
        smith.nn.functional.dropout: lambda input, p=0.5, training=True, inplace=False: -1,
        smith.nn.functional.dropout1d: lambda input, p=0.5, training=True, inplace=False: -1,
        smith.nn.functional.dropout2d: lambda input, p=0.5, training=True, inplace=False: -1,
        smith.nn.functional.dropout3d: lambda input, p=0.5, training=True, inplace=False: -1,
        smith.nn.functional.elu: lambda input, alpha=1.0, inplace=False: -1,
        smith.nn.functional.embedding: (
            lambda input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False: -1  # noqa: B950
        ),
        smith.nn.functional.embedding_bag: (
            lambda input, weight, offsets=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode="mean", sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None: -1  # noqa: B950
        ),
        smith.nn.functional.feature_alpha_dropout: lambda input, p=0.5, training=False, inplace=False: -1,
        smith.nn.functional.fold: lambda input, output_size, kernel_size, dilation=1, padding=0, stride=1: -1,
        smith.nn.functional.fractional_max_pool2d: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None: -1  # noqa: B950
        ),
        smith.nn.functional.fractional_max_pool2d_with_indices: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None: -1  # noqa: B950
        ),
        smith.nn.functional.fractional_max_pool3d: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None: -1  # noqa: B950
        ),
        smith.nn.functional.fractional_max_pool3d_with_indices: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None: -1  # noqa: B950
        ),
        smith.nn.functional.gaussian_nll_loss: lambda input, target, var, full=False, eps=1e-06, reduction="mean": -1,
        smith.nn.functional.gelu: lambda input, approximate="none": -1,
        smith.nn.functional.glu: lambda input, dim=-1: -1,
        smith.nn.functional.grid_sample: lambda input, grid, mode="bilinear", padding_mode="zeros", align_corners=None: -1,  # noqa: B950
        smith.nn.functional.group_norm: lambda input, num_groups, weight=None, bias=None, eps=1e-05: -1,
        smith.nn.functional.gumbel_softmax: lambda logits, tau=1, hard=False, eps=1e-10, dim=-1: -1,
        smith.nn.functional.hardshrink: lambda input, lambd=0.5: -1,
        smith.nn.functional.hardtanh: lambda input, min_val=-1.0, max_val=1.0, inplace=False: -1,
        smith.nn.functional.hinge_embedding_loss: (
            lambda input, target, margin=1.0, size_average=None, reduce=None, reduction="mean": -1
        ),
        smith.nn.functional.instance_norm: (
            lambda input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05: -1  # noqa: B950
        ),
        smith.nn.functional.interpolate: (
            lambda input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False: -1  # noqa: B950
        ),
        smith.nn.functional.kl_div: lambda input, target, size_average=None, reduce=None, reduction="mean", log_target=False: -1,  # noqa: B950
        smith.nn.functional.l1_loss: lambda input, target, size_average=None, reduce=None, reduction="mean", weight=None: -1,
        smith.nn.functional.layer_norm: lambda input, normalized_shape, weight=None, bias=None, eps=1e-05: -1,
        smith.nn.functional.leaky_relu: lambda input, negative_slope=0.01, inplace=False: -1,
        smith.nn.functional.linear: lambda input, weight, bias=None: -1,
        smith.nn.functional.local_response_norm: lambda input, size, alpha=0.0001, beta=0.75, k=1.0: -1,
        smith.nn.functional.log_softmax: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        smith.nn.functional.logsigmoid: lambda input: -1,
        smith.nn.functional.lp_pool1d: lambda input, norm_type, kernel_size, stride=None, ceil_mode=False: -1,
        smith.nn.functional.lp_pool2d: lambda input, norm_type, kernel_size, stride=None, ceil_mode=False: -1,
        smith.nn.functional.lp_pool3d: lambda input, norm_type, kernel_size, stride=None, ceil_mode=False: -1,
        smith.nn.functional.margin_ranking_loss: (
            lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction="mean": -1
        ),
        smith.nn.functional.max_pool1d: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False: -1
        ),
        smith.nn.functional.max_pool1d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        smith.nn.functional.max_pool2d: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False: -1
        ),
        smith.nn.functional.max_pool2d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        smith.nn.functional.max_pool3d: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        smith.nn.functional.max_pool3d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        smith.nn.functional.max_unpool1d: lambda input, indices, kernel_size, stride=None, padding=0, output_size=None: -1,  # noqa: B950
        smith.nn.functional.max_unpool2d: lambda input, indices, kernel_size, stride=None, padding=0, output_size=None: -1,  # noqa: B950
        smith.nn.functional.max_unpool3d: lambda input, indices, kernel_size, stride=None, padding=0, output_size=None: -1,  # noqa: B950
        smith.nn.functional.mse_loss: lambda input, target, size_average=None, reduce=None, reduction="mean", weight=None: -1,
        smith.nn.functional.multi_head_attention_forward: (
            lambda query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None, average_attn_weights=None, is_causal=False: -1  # noqa: B950
        ),
        smith.nn.functional.multi_margin_loss: (
            lambda input, target, p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction="mean": -1
        ),
        smith.nn.functional.multilabel_margin_loss: (
            lambda input, target, size_average=None, reduce=None, reduction="mean": -1
        ),
        smith.nn.functional.multilabel_soft_margin_loss: (
            lambda input, target, weight=None, size_average=None, reduce=None, reduction="mean": -1
        ),
        smith.nn.functional.nll_loss: (
            lambda input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean": -1
        ),
        smith.nn.functional.normalize: lambda input, p=2, dim=1, eps=1e-12, out=None: -1,
        smith.nn.functional.one_hot: lambda tensor, num_classes=-1: -1,
        smith.nn.functional.pad: lambda input, pad, mode="constant", value=0: -1,
        smith.nn.functional.pairwise_distance: lambda x1, x2, p=2.0, eps=1e-06, keepdim=False: -1,
        smith.nn.functional.poisson_nll_loss: (
            lambda input, target, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction="mean": -1  # noqa: B950
        ),
        smith.nn.functional.prelu: lambda input, weight: -1,
        smith.nn.functional.relu: lambda input, inplace=False: -1,
        smith.nn.functional.relu6: lambda input, inplace=False: -1,
        smith.nn.functional.rms_norm: lambda input, normalized_shape, weight=None, eps=1e-6: -1,
        smith.nn.functional.rrelu: lambda input, lower=0.125, upper=0.3333333333333333, training=False, inplace=False: -1,  # noqa: B950
        smith.nn.functional.selu: lambda input, inplace=False: -1,
        smith.nn.functional.silu: lambda input, inplace=False: -1,
        smith.nn.functional.mish: lambda input, inplace=False: -1,
        smith.nn.functional.scaled_dot_product_attention: lambda query, key, value, attn_mask=None, dropout_p=0.0: -1,
        smith.nn.functional.smooth_l1_loss: lambda input, target, size_average=None, reduce=None, reduction="mean", beta=1.0: -1,  # noqa: B950
        smith.nn.functional.huber_loss: lambda input, target, reduction="mean", delta=1.0, weight=None: -1,
        smith.nn.functional.soft_margin_loss: lambda input, target, size_average=None, reduce=None, reduction="mean": -1,  # noqa: B950
        smith.nn.functional.softmax: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        smith.nn.functional.softmin: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        smith.nn.functional.softplus: lambda input, beta=1, threshold=20: -1,
        smith.nn.functional.softshrink: lambda input, lambd=0.5: -1,
        smith.nn.functional.softsign: lambda input: -1,
        smith.nn.functional.tanhshrink: lambda input: -1,
        smith.nn.functional.threshold: lambda input, threshold, value, inplace=False: -1,
        smith.nn.functional.triplet_margin_loss: (
            lambda anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction="mean": -1  # noqa: B950
        ),
        smith.nn.functional.triplet_margin_with_distance_loss: (
            lambda anchor, positive, negative, *, distance_function=None, margin=1.0, swap=False, reduction="mean": -1
        ),
        smith.nn.functional.unfold: lambda input, kernel_size, dilation=1, padding=0, stride=1: -1,
        smith.nn.init.uniform_: lambda tensor, a=0.0, b=1.0, generator=None: -1,
        smith.nn.init.normal_: lambda tensor, mean=0.0, std=1.0, generator=None: -1,
        smith.nn.init.constant_: lambda tensor, val: -1,
        smith.nn.init.kaiming_uniform_: lambda tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None: -1,  # noqa: B950
        smith.nonzero: lambda input, as_tuple=False: -1,
        smith.nonzero_static: lambda input, *, size, fill_value=-1: -1,
        smith.argwhere: lambda input: -1,
        smith.norm: lambda input, p="fro", dim=None, keepdim=False, out=None, dtype=None: -1,
        smith.linalg.norm: lambda input, ord=None, dim=None, keepdim=False, out=None, dtype=None: -1,
        smith.linalg.vector_norm: lambda input, ord=2, dim=None, keepdim=False, out=None, dtype=None: -1,
        smith.linalg.matrix_norm: lambda input, ord="fro", dim=(
            -2,
            -1,
        ), keepdim=False, out=None, dtype=None: -1,
        smith.norm_except_dim: lambda v, pow=2, dim=0: -1,
        smith.nuclear_norm: lambda input, p="fro", dim=None, keepdim=False, out=None, dtype=None: -1,
        smith.numel: lambda input: -1,
        smith.orgqr: lambda input, tau: -1,
        smith.ormqr: lambda input, input2, input3, left=True, transpose=False: -1,
        smith.pairwise_distance: lambda x1, x2, p=2.0, eps=1e-06, keepdim=False: -1,
        smith.permute: lambda self, dim: -1,
        smith.pca_lowrank: lambda input, q=None, center=True, niter=2: -1,
        smith.pdist: lambda input, p=2: -1,
        smith.pinverse: lambda input, rcond=1e-15: -1,
        smith.linalg.pinv: lambda input, rcond=1e-15, hermitian=False: -1,
        smith.pixel_shuffle: lambda input, upscale_factor: -1,
        smith.pixel_unshuffle: lambda input, downscale_factor: -1,
        smith.poisson: lambda input, generator=None: -1,
        smith.poisson_nll_loss: lambda input, target, log_input, full, eps, reduction: -1,
        smith.polygamma: lambda input, n, out=None: -1,
        smith.positive: lambda input, out=None: -1,
        smith.prelu: lambda input, weight: -1,
        smith.ones_like: lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1,
        smith.pow: lambda input, exponent, out=None: -1,
        smith.prod: lambda input, dtype=None: -1,
        smith.put: lambda input, index, source, accumulate=False: -1,
        smith.q_per_channel_axis: lambda input: -1,
        smith.q_per_channel_scales: lambda input: -1,
        smith.q_per_channel_zero_points: lambda input: -1,
        smith.q_scale: lambda input: -1,
        smith.q_zero_point: lambda input: -1,
        smith.qr: lambda input, some=True, out=None: -1,
        smith.linalg.qr: lambda input, mode="reduced", out=None: -1,
        smith.quantile: lambda input, q, dim=None, keepdim=False, interpolation="linear", out=None: -1,
        smith.nanquantile: lambda input, q, dim=None, keepdim=False, interpolation="linear", out=None: -1,
        smith.quantize_per_channel: lambda input, scales, zero_points, axis, dtype: -1,
        smith.quantize_per_tensor: lambda input, scale, zero_point, dtype: -1,
        smith.quantize_per_tensor_dynamic: lambda input, dtype, reduce_range: -1,
        smith.quantized_batch_norm: lambda input, weight, bias, mean, var, eps, output_scale, output_zero_point: -1,
        smith.quantized_gru_cell: (
            lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1  # noqa: B950
        ),
        smith.quantized_lstm_cell: (
            lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1  # noqa: B950
        ),
        smith.quantized_max_pool1d: (
            lambda input, kernel_size, stride=(), padding=(0,), dilation=(
                1,
            ), ceil_mode=False: -1
        ),
        smith.quantized_max_pool2d: (
            lambda input, kernel_size, stride=(), padding=(0, 0), dilation=(
                1,
                1,
            ), ceil_mode=False: -1
        ),
        smith.quantized_max_pool3d: (
            lambda input, kernel_size, stride=(), padding=(0, 0, 0), dilation=(
                1,
                1,
                1,
            ), ceil_mode=False: -1
        ),
        smith.quantized_rnn_relu_cell: (
            lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1  # noqa: B950
        ),
        smith.quantized_rnn_tanh_cell: (
            lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1  # noqa: B950
        ),
        smith.rad2deg: lambda input, out=None: -1,
        smith.ravel: lambda input: -1,
        smith.real: lambda input, out=None: -1,
        smith.vdot: lambda input, other, out=None: -1,
        smith.linalg.vecdot: lambda input, other, dim=-1, out=None: -1,
        smith.view_as_real: lambda input: -1,
        smith.view_as_complex: lambda input: -1,
        smith.reciprocal: lambda input, out=None: -1,
        smith.relu: lambda input, inplace=False: -1,
        smith.remainder: lambda input, other, out=None: -1,
        smith.renorm: lambda input, p, dim, maxnorm, out=None: -1,
        smith.repeat_interleave: lambda input, dim=None: -1,
        smith.reshape: lambda input, shape: -1,
        smith.rms_norm: lambda input, normalized_shape, weight=None, eps=1e-6: -1,
        smith.rnn_relu: lambda input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first: -1,  # noqa: B950
        smith.rnn_relu_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        smith.rnn_tanh: lambda input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first: -1,  # noqa: B950
        smith.rnn_tanh_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        smith.roll: lambda input, shifts, dims=None: -1,
        smith.rot90: lambda input, k=1, dims=(0, 1): -1,
        smith.round: lambda input, out=None: -1,
        smith.row_stack: lambda tensors, out=None: -1,  # alias for smith.vstack
        smith._rowwise_prune: (lambda weight, mask, compressed_indices_dtype: -1),
        smith.rrelu: lambda input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False: -1,
        smith.rsqrt: lambda input, out=None: -1,
        smith.rsub: lambda input, other, alpha=1: -1,
        smith.saddmm: lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1,
        smith.scatter: lambda input, dim, index, src: -1,
        smith.scatter_add: lambda input, dim, index, src: -1,
        smith.scatter_reduce: lambda input, dim, index, src, reduce, include_self=True: -1,
        smith.searchsorted: lambda sorted_sequence, input, out_int32=False, right=False, out=None: -1,
        smith._segment_reduce: lambda data, reduce="max", lengths=None, indices=None, offsets=None, axis=0, unsafe=False: -1,  # noqa: B950
        smith.select: lambda input, dim, index: -1,
        smith.select_scatter: lambda input, src, dim, index: -1,
        smith.slice_inverse: lambda input, src, dim=0, start=None, end=None, step=1: -1,
        smith.slice_scatter: lambda input, src, dim=0, start=None, end=None, step=1: -1,
        smith.selu: lambda input, inplace=False: -1,
        smith.sigmoid: lambda input, out=None: -1,
        smith.sign: lambda input, out=None: -1,
        smith.signbit: lambda input, out=None: -1,
        smith.sgn: lambda input, out=None: -1,
        smith.sin: lambda input, out=None: -1,
        smith.sinc: lambda input, out=None: -1,
        smith.sinh: lambda input, out=None: -1,
        smith.slogdet: lambda input: -1,
        smith.linalg.slogdet: lambda input: -1,
        smith.smm: lambda input, mat2, out_dtype=None: -1,
        smith.spmm: lambda input, mat2, out_dtype=None: -1,
        smith.softmax: lambda input, dim, dtype=None: -1,
        smith.linalg.solve: lambda A, B, left=True, out=None: -1,
        smith.linalg.solve_ex: lambda A, B, left=True, check_errors=False, out=None: -1,
        smith.sort: lambda input, dim=-1, descending=False, *, stable=False, out=None: -1,
        smith.split: lambda tensor, split_size_or_sections, dim=0: -1,
        smith.split_with_sizes: lambda tensor, split_size_or_sections, dim=0: -1,
        smith.sqrt: lambda input, out=None: -1,
        smith.square: lambda input, out=None: -1,
        smith.squeeze: lambda input, dim=None, out=None: -1,
        smith.sspaddmm: lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1,
        smith.stack: lambda tensors, dim=0, out=None: -1,
        smith.std: lambda input, dim=None: -1,
        smith.std_mean: lambda input, dim=None: -1,
        smith.stft: (
            lambda input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode="reflect", normalized=False, onesided=True, return_complex=None, align_to_window=None: -1  # noqa: B950
        ),
        smith.sub: lambda input, other, out=None: -1,
        smith.subtract: lambda input, other, out=None: -1,
        smith.sum: lambda input, dim=None: -1,
        smith.sym_float: lambda input: -1,
        smith.sym_int: lambda input: -1,
        smith.sym_max: lambda a, b: -1,
        smith.sym_min: lambda a, b: -1,
        smith.sym_not: lambda input: -1,
        smith.sym_ite: lambda a, b, c: -1,
        smith.sym_sum: lambda args: -1,
        smith._sym_sqrt: lambda input: -1,
        smith._sym_cos: lambda input: -1,
        smith._sym_cosh: lambda input: -1,
        smith._sym_sin: lambda input: -1,
        smith._sym_sinh: lambda input: -1,
        smith._sym_tan: lambda input: -1,
        smith._sym_tanh: lambda input: -1,
        smith._sym_asin: lambda input: -1,
        smith._sym_acos: lambda input: -1,
        smith._sym_atan: lambda input: -1,
        smith.nansum: lambda input, dim=None: -1,
        smith.svd: lambda input, some=True, compute_uv=True, out=None: -1,
        smith.svd_lowrank: lambda input, q=6, niter=2, M=None: -1,
        smith.linalg.svd: lambda input, full_matrices=True, out=None: -1,
        smith.linalg.svdvals: lambda input, out=None: -1,
        smith.swapaxes: lambda input, dim0, dim1: -1,
        smith.swapdims: lambda input, axis0, axis1: -1,
        smith.special.airy_ai: lambda input: -1,
        smith.special.bessel_j0: lambda input: -1,
        smith.special.bessel_j1: lambda input: -1,
        smith.special.bessel_y0: lambda input: -1,
        smith.special.bessel_y1: lambda input: -1,
        smith.special.chebyshev_polynomial_t: lambda input, n, out=None: -1,
        smith.special.chebyshev_polynomial_u: lambda input, n, out=None: -1,
        smith.special.chebyshev_polynomial_v: lambda input, n, out=None: -1,
        smith.special.chebyshev_polynomial_w: lambda input, n, out=None: -1,
        smith.special.digamma: lambda input: -1,
        smith.special.entr: lambda input: -1,
        smith.special.erf: lambda input: -1,
        smith.special.erfc: lambda input: -1,
        smith.special.erfcx: lambda input: -1,
        smith.special.erfinv: lambda input: -1,
        smith.special.exp2: lambda input: -1,
        smith.special.expit: lambda input: -1,
        smith.special.expm1: lambda input: -1,
        smith.special.gammainc: lambda input, other, out=None: -1,
        smith.special.gammaincc: lambda input, other, out=None: -1,
        smith.special.gammaln: lambda input: -1,
        smith.special.hermite_polynomial_h: lambda input, n, out=None: -1,
        smith.special.hermite_polynomial_he: lambda input, n, out=None: -1,
        smith.special.i0: lambda input: -1,
        smith.special.i0e: lambda input: -1,
        smith.special.i1: lambda input: -1,
        smith.special.i1e: lambda input: -1,
        smith.special.laguerre_polynomial_l: lambda input, n, out=None: -1,
        smith.special.legendre_polynomial_p: lambda input, n, out=None: -1,
        smith.special.log1p: lambda input: -1,
        smith.special.log_ndtr: lambda input: -1,
        smith.special.log_softmax: lambda input, dim, dtype=None: -1,
        smith.special.logit: lambda input: -1,
        smith.special.logsumexp: lambda input, dim, keepdim=False, out=None: -1,
        smith.special.modified_bessel_i0: lambda input: -1,
        smith.special.modified_bessel_i1: lambda input: -1,
        smith.special.modified_bessel_k0: lambda input: -1,
        smith.special.modified_bessel_k1: lambda input: -1,
        smith.special.multigammaln: lambda input, p: -1,
        smith.special.ndtr: lambda input: -1,
        smith.special.ndtri: lambda input: -1,
        smith.special.polygamma: lambda input, n, out=None: -1,
        smith.special.psi: lambda input: -1,
        smith.special.round: lambda input: -1,
        smith.special.scaled_modified_bessel_k0: lambda input: -1,
        smith.special.scaled_modified_bessel_k1: lambda input: -1,
        smith.special.shifted_chebyshev_polynomial_t: lambda input, n, out=None: -1,
        smith.special.shifted_chebyshev_polynomial_u: lambda input, n, out=None: -1,
        smith.special.shifted_chebyshev_polynomial_v: lambda input, n, out=None: -1,
        smith.special.shifted_chebyshev_polynomial_w: lambda input, n, out=None: -1,
        smith.special.sinc: lambda input: -1,
        smith.special.softmax: lambda input, dim, dtype=None: -1,
        smith.special.spherical_bessel_j0: lambda input: -1,
        smith.special.xlog1py: lambda input, other, out=None: -1,
        smith.special.xlogy: lambda input, other, out=None: -1,
        smith.special.zeta: lambda self, other, out=None: -1,
        smith.t: lambda input: -1,
        smith.take: lambda input, index: -1,
        smith.take_along_dim: lambda input, indices, dim=None, out=None: -1,
        smith.tan: lambda input, out=None: -1,
        smith.tanh: lambda input, out=None: -1,
        smith.linalg.tensorinv: lambda a, ind=2: -1,
        smith.linalg.tensorsolve: lambda a, b, dims=None: -1,
        smith.tensordot: lambda a, b, dims=2, out=None: -1,
        smith.tensor_split: lambda input, indices_or_sections, dim=0: -1,
        smith.threshold: lambda input, threshold, value, inplace=False: -1,
        smith.tile: lambda input, dims: -1,
        smith.topk: lambda input, k, dim=-1, descending=False, out=None: -1,
        smith.trace: lambda input: -1,
        smith.transpose: lambda input, dim0, dim1: -1,
        smith.trapz: lambda y, x=None, dim=-1: -1,
        smith.trapezoid: lambda y, x=None, dim=-1: -1,
        smith.triangular_solve: lambda input, A, upper=True, transpose=False, unitriangular=False: -1,
        smith.linalg.solve_triangular: lambda input, B, upper, left=True, unitriangular=False: -1,
        smith.tril: lambda input, diagonal=0, out=None: -1,
        smith.triplet_margin_loss: (
            lambda anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction="mean": -1  # noqa: B950
        ),
        smith.triu: lambda input, diagonal=0, out=None: -1,
        smith.true_divide: lambda input, other: -1,
        smith.trunc: lambda input, out=None: -1,
        smith.unbind: lambda input, dim=0: -1,
        smith.unflatten: lambda input, dim, sizes, names: -1,
        smith.unique: lambda input, sorted=True, return_inverse=False, return_counts=False, dim=None: -1,
        smith.unique_consecutive: lambda input, return_inverse=False, return_counts=False, dim=None: -1,
        smith.unravel_index: lambda indices, shape: -1,
        smith.unsafe_chunk: lambda input, chunks, dim=0: -1,
        smith.unsafe_split: lambda tensor, split_size_or_sections, dim=0: -1,
        smith.unsafe_split_with_sizes: lambda tensor, split_size_or_sections, dim=0: -1,
        smith.unsqueeze: lambda input, dim, out=None: -1,
        smith.linalg.vander: lambda x, N=None: -1,
        smith.var: lambda input, dim=None: -1,
        smith.var_mean: lambda input, dim=None: -1,
        smith.vsplit: lambda input, indices_or_sections: -1,
        smith.vstack: lambda tensors, out=None: -1,
        smith.where: lambda condition, x=None, y=None: -1,
        smith._wrapped_linear_prepack: lambda weight, weight_scale, weight_zero_point, bias : -1,
        smith._wrapped_quantized_linear_prepacked: (
            lambda input, input_scale, input_zero_point, prepacked, out_scale, out_zero_point, out_channel : -1  # noqa: B950
        ),
        smith.zeros_like: lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1,
        smith._fw_primal_copy: lambda self, level: -1,
        smith._make_dual_copy: lambda primal, tangent, level: -1,
        smith.view_as_real_copy: lambda self: -1,
        smith.view_as_complex_copy: lambda self: -1,
        smith._conj_copy: lambda self: -1,
        smith._neg_view_copy: lambda self: -1,
        smith.as_strided_copy: lambda self, size, stride, storage_offset=None: -1,
        smith._sparse_broadcast_to_copy: lambda self, size: -1,
        smith.diagonal_copy: lambda self, offset=0, dim1=0, dim2=1: -1,
        smith.expand_copy: lambda self, size, *, implicit=False: -1,
        smith.narrow_copy: lambda self, dim, start, length: -1,
        smith.permute_copy: lambda self, dims: -1,
        smith._reshape_alias_copy: lambda self, size, stride: -1,
        smith.select_copy: lambda self, dim, index: -1,
        smith.detach_copy: lambda self: -1,
        smith.slice_copy: lambda self, dim=0, start=None, end=None, step=1: -1,
        smith.split_copy: lambda self, split_size, dim=0: -1,
        smith.split_with_sizes_copy: lambda self, split_sizes, dim=0: -1,
        smith.squeeze_copy: lambda self, dim: -1,
        smith.t_copy: lambda self: -1,
        smith.transpose_copy: lambda self, dim0, dim1: -1,
        smith.unsqueeze_copy: lambda self, dim: -1,
        smith._indices_copy: lambda self: -1,
        smith._values_copy: lambda self: -1,
        smith.indices_copy: lambda self: -1,
        smith.values_copy: lambda self: -1,
        smith.crow_indices_copy: lambda self: -1,
        smith.col_indices_copy: lambda self: -1,
        smith.ccol_indices_copy: lambda self: -1,
        smith.row_indices_copy: lambda self: -1,
        smith.unbind_copy: lambda self, dim=0: -1,
        smith.view_copy: lambda self, dtype: -1,
        smith.unfold_copy: lambda self, dimension, size, step: -1,
        smith.alias_copy: lambda self: -1,
        Tensor.__floordiv__: lambda self, other: -1,
        Tensor.__rfloordiv__: lambda self, other: -1,
        Tensor.__ifloordiv__: lambda self, other: -1,
        Tensor.__truediv__: lambda self, other: -1,
        Tensor.__rtruediv__: lambda self, other: -1,
        Tensor.__itruediv__: lambda self, other: -1,
        Tensor.__lshift__: lambda self, other: -1,
        Tensor.__rlshift__: lambda self, other: -1,
        Tensor.__ilshift__: lambda self, other: -1,
        Tensor.__rshift__: lambda self, other: -1,
        Tensor.__rrshift__: lambda self, other: -1,
        Tensor.__irshift__: lambda self, other: -1,
        Tensor.__and__: lambda self, other: -1,
        Tensor.__or__: lambda self, other: -1,
        Tensor.__xor__: lambda self, other: -1,
        Tensor.__float__: lambda self: -1,
        Tensor.__complex__: lambda self: -1,
        Tensor.__array__: lambda self, dtype: -1,
        Tensor.__bool__: lambda self: -1,
        Tensor.__contains__: lambda self, other: -1,
        Tensor.__neg__: lambda self: -1,
        Tensor.__invert__: lambda self: -1,
        Tensor.__mod__: lambda self, other: -1,
        Tensor.__rmod__: lambda self, other: -1,
        Tensor.__imod__: lambda self, other: -1,
        Tensor.__array_wrap__: lambda self, array: -1,
        Tensor.__getitem__: lambda self, idx: -1,
        Tensor.__deepcopy__: lambda self, memo: -1,
        Tensor.__int__: lambda self: -1,
        Tensor.__long__: lambda self: -1,
        Tensor.__index__: lambda self: -1,
        Tensor.__len__: lambda self: -1,
        Tensor.__format__: lambda self, format_spec: -1,
        Tensor.__reduce_ex__: lambda self, proto: -1,
        Tensor.__reversed__: lambda self: -1,
        Tensor.__repr__: lambda self, *, tensor_contents=None: -1,
        Tensor.__setitem__: lambda self, k, v: -1,
        Tensor.__setstate__: lambda self, d: -1,
        Tensor.T.__get__: lambda self: -1,
        Tensor.H.__get__: lambda self: -1,
        Tensor.mT.__get__: lambda self: -1,
        Tensor.mH.__get__: lambda self: -1,
        Tensor._backward_hooks.__get__: lambda self: -1,
        Tensor._post_accumulate_grad_hooks.__get__: lambda self: -1,
        Tensor._base.__get__: lambda self: -1,
        Tensor._cdata.__get__: lambda self: -1,
        Tensor.grad.__get__: lambda self: -1,
        Tensor._grad.__get__: lambda self: -1,
        Tensor._grad_fn.__get__: lambda self: -1,
        Tensor.grad_fn.__get__: lambda self: -1,
        Tensor.grad_dtype.__get__: lambda self: -1,
        Tensor._version.__get__: lambda self: -1,
        Tensor._autocast_to_reduced_precision: lambda self, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype: -1,
        Tensor._autocast_to_full_precision: lambda self, cuda_enabled, cpu_enabled: -1,
        Tensor._clear_non_serializable_cached_data: lambda self: -1,
        Tensor.data.__get__: lambda self: -1,
        Tensor.device.__get__: lambda self: -1,
        Tensor.dtype.__get__: lambda self: -1,
        Tensor.is_cuda.__get__: lambda self: -1,
        Tensor.is_cpu.__get__: lambda self: -1,
        Tensor.is_xla.__get__: lambda self: -1,
        Tensor.is_xpu.__get__: lambda self: -1,
        Tensor.is_ipu.__get__: lambda self: -1,
        Tensor.is_leaf.__get__: lambda self: -1,
        Tensor.retains_grad.__get__: lambda self: -1,
        Tensor.is_meta.__get__: lambda self: -1,
        Tensor.is_mps.__get__: lambda self: -1,
        Tensor.is_mtia.__get__: lambda self: -1,
        Tensor.is_nested.__get__: lambda self: -1,
        Tensor.is_maia.__get__: lambda self: -1,
        Tensor.is_mkldnn.__get__: lambda self: -1,
        Tensor.is_quantized.__get__: lambda self: -1,
        Tensor.is_sparse.__get__: lambda self: -1,
        Tensor.is_sparse_csr.__get__: lambda self: -1,
        Tensor.is_vulkan.__get__: lambda self: -1,
        Tensor.itemsize.__get__: lambda self: -1,
        Tensor.layout.__get__: lambda self: -1,
        Tensor.name.__get__: lambda self: -1,
        Tensor.names.__get__: lambda self: -1,
        Tensor.nbytes.__get__: lambda self: -1,
        Tensor.ndim.__get__: lambda self: -1,
        Tensor.output_nr.__get__: lambda self: -1,
        Tensor.requires_grad.__get__: lambda self: -1,
        Tensor.shape.__get__: lambda self: -1,
        Tensor.volatile.__get__: lambda self: -1,
        Tensor.real.__get__: lambda self: -1,
        Tensor.imag.__get__: lambda self: -1,
        Tensor.__cuda_array_interface__.__get__: lambda self: -1,
        Tensor.type: lambda self, dtype=None, non_blocking=False, **kwargs: -1,
        Tensor._dimI: lambda self: -1,
        Tensor._dimV: lambda self: -1,
        Tensor._indices: lambda self: -1,
        Tensor._is_view: lambda self: -1,
        Tensor._nnz: lambda self: -1,
        Tensor.crow_indices: lambda self: -1,
        Tensor.col_indices: lambda self: -1,
        Tensor.ccol_indices: lambda self: -1,
        Tensor.row_indices: lambda self: -1,
        Tensor._update_names: lambda self, names, inplace: -1,
        Tensor._values: lambda self: -1,
        Tensor.adjoint: lambda self: -1,
        Tensor.align_as: lambda self, other: -1,
        Tensor.align_to: lambda self, order, ellipsis_idx: -1,
        Tensor.apply_: lambda self, callable: -1,
        Tensor.as_strided: lambda self, size, stride: -1,
        Tensor.as_strided_: lambda self, size, stride: -1,
        Tensor.backward: lambda self, gradient=None, retain_graph=None, create_graph=False, inputs=None: -1,
        Tensor.bfloat16: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.bool: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.byte: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.char: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.cauchy_: lambda self, median=0, sigma=1, *, generator=None: -1,
        Tensor.coalesce: lambda self: -1,
        Tensor._coalesced_: lambda self, coalesced: -1,
        Tensor.contiguous: lambda self, memory_format=smith.contiguous_format: -1,
        Tensor.copy_: lambda self, src, non_blocking=False: -1,
        Tensor.cpu: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.cuda: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.mtia: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.xpu: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.ipu: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.data_ptr: lambda self: -1,
        Tensor.dense_dim: lambda self: -1,
        Tensor.diagonal_scatter: lambda self, src, offset=0, dim1=0, dim2=1: -1,
        Tensor.dim: lambda self: -1,
        Tensor.dim_order: lambda self, ambiguity_check=False: -1,
        Tensor.double: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.cdouble: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.element_size: lambda self: -1,
        Tensor.expand: lambda self, size: -1,
        Tensor.expand_as: lambda self, other: -1,
        Tensor.exponential_: lambda self, lambd=1, *, generator=None: -1,
        Tensor.fill_: lambda self, value: -1,
        Tensor.fill_diagonal_: lambda self, value: -1,
        Tensor.float: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.cfloat: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.geometric_: lambda self, p, *, generator=None: -1,
        Tensor.get_device: lambda self: -1,
        Tensor.half: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.chalf: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.has_names: lambda self: -1,
        Tensor.indices: lambda self: -1,
        Tensor.int: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.is_coalesced: lambda self: -1,
        Tensor.is_contiguous: lambda self: -1,
        Tensor.is_inference: lambda self: -1,
        Tensor.is_pinned: lambda self: -1,
        Tensor.is_set_to: lambda self, tensor: -1,
        Tensor.is_shared: lambda self: -1,
        Tensor.item: lambda self: -1,
        Tensor.log_normal_: lambda self, mean=1, std=2, *, generator=None: -1,
        Tensor.log_softmax: lambda self, dim: -1,
        Tensor.long: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.map_: lambda self, tensor, callable: -1,
        Tensor.map2_: lambda self, x, y, callable: -1,
        Tensor.mm: lambda self, mat2, out_dtype=None: -1,
        Tensor.module_load: lambda self, other, assign=False: -1,
        Tensor.narrow_copy: lambda self, dimension, start, length: -1,
        Tensor.ndimension: lambda self: -1,
        Tensor.nelement: lambda self: -1,
        Tensor._nested_tensor_size: lambda self: -1,
        Tensor._nested_tensor_storage_offsets: lambda self: -1,
        Tensor._nested_tensor_strides: lambda self: -1,
        Tensor.normal_: lambda self: -1,
        Tensor.numpy: lambda self: -1,
        Tensor.permute: lambda self, dim: -1,
        Tensor.pin_memory: lambda self: -1,
        Tensor.put_: lambda self, indices, tensor, accumulate=False: -1,
        Tensor.qscheme: lambda self: -1,
        Tensor.random_: lambda self, from_=0, to=None, *, generator=None: -1,
        Tensor.record_stream: lambda self, stream: -1,
        Tensor.refine_names: lambda self, names: -1,
        Tensor.register_hook: lambda self, hook: -1,
        Tensor.register_post_accumulate_grad_hook: lambda self, hook: -1,
        Tensor.rename: lambda self, name: -1,
        Tensor.repeat: lambda self, *size: -1,
        Tensor.requires_grad_: lambda self, requires_grad=True: -1,
        Tensor.reshape_as: lambda self, other: -1,
        Tensor.resize: lambda self, *size: -1,
        Tensor.resize_: lambda self, size: -1,
        Tensor.resize_as: lambda self, other: -1,
        Tensor.resize_as_sparse_: lambda self, other: -1,
        Tensor.retain_grad: lambda self: -1,
        Tensor.set_: lambda self, source=None, storage_offset=0, size=None, stride=None: -1,
        Tensor.select_scatter: lambda self, src, dim, index: -1,
        Tensor.share_memory_: lambda self: -1,
        Tensor.short: lambda self, memory_format=smith.preserve_format: -1,
        Tensor.size: lambda self: -1,
        Tensor.slice_scatter: lambda self, src, dim=0, start=None, end=None, step=1: -1,
        Tensor.sparse_dim: lambda self: -1,
        Tensor.sparse_mask: lambda self, mask: -1,
        Tensor._sparse_mask_projection: lambda self, mask, accumulate_matches=False: -1,
        Tensor.sparse_resize_: lambda self, size1, size2, dense_dim: -1,
        Tensor.sparse_resize_and_clear_: lambda self, size1, size2, dense_dim: -1,
        Tensor.sspaddmm: lambda self, mat1, mat2, beta=1, alpha=1, out=None: -1,
        Tensor.storage: lambda self: -1,
        Tensor.untyped_storage: lambda self: -1,
        Tensor.storage_offset: lambda self: -1,
        Tensor.storage_type: lambda self: -1,
        Tensor.sum_to_size: lambda self, size: -1,
        Tensor.tile: lambda self, *reps: -1,
        Tensor.to: lambda self, dtype, non_blocking=False, copy=False, memory_format=smith.preserve_format: -1,
        Tensor.to_dense: lambda self, dtype=None, *, masked_grad=None: -1,
        Tensor._to_dense: lambda self, dtype=None, masked_grad=None: -1,
        Tensor.to_sparse: lambda self: -1,
        Tensor.tolist: lambda self: -1,
        Tensor.to_mkldnn: lambda self: -1,
        Tensor.type_as: lambda self, other: -1,
        Tensor.unfold: lambda self, dimension, size, step: -1,
        Tensor.uniform_: lambda self, from_=0, to=1: -1,
        Tensor.values: lambda self: -1,
        Tensor.view: lambda self, shape: -1,
        Tensor.view_as: lambda self, other: -1,
        Tensor.zero_: lambda self: -1,
        Tensor.__dlpack__: lambda self, stream=None, max_version=None, dl_device=None, copy=None: -1,
        Tensor.__dlpack_device__: lambda self: -1,
        Tensor.index: lambda self, a, b: -1,
        smith.linalg.lstsq: lambda self, b, cond=None, driver=None: -1,
    }  # fmt: skip

    privateuse1_backend_name = (
        smith.utils.backend_registration._privateuse1_backend_name
    )
    if hasattr(Tensor, privateuse1_backend_name):
        ret[getattr(Tensor, privateuse1_backend_name)] = (
            lambda self, device=None, non_blocking=False, **kwargs: -1
        )
        ret[getattr(Tensor, f"is_{privateuse1_backend_name}").__get__] = lambda self: -1

    ret2 = {}
    ignored = get_ignored_functions()

    for k, v in ret.items():
        # Generate methods like __add__ and add_ by default from add
        names = [
            k.__name__,  # Default method
            k.__name__ + "_",  # Inplace variant
            "__" + k.__name__ + "__",  # Dunder method
            "__i" + k.__name__ + "__",  # Inplace dunder method
            "__r" + k.__name__ + "__",  # Reverse dunder method
        ]

        if k.__name__.startswith("bitwise_"):
            # bitwise_<op> have dunder methods of the form __<op>__
            # And so on.
            subname = k.__name__[len("bitwise_") :]
            names.extend(
                ["__" + subname + "__", "__i" + subname + "__", "__r" + subname + "__"]
            )

        for name in names:
            func = getattr(Tensor, name, None)
            if callable(func) and func not in ret and func not in ignored:
                ret2[func] = v

    ret.update(ret2)
    return ret


def wrap_smith_function(dispatcher: Callable):
    """Wraps a given function with ``__smith_function__`` -related functionality.

    Parameters
    ----------
    dispatcher: Callable
        A callable that returns an iterable of Tensor-likes passed into the function.

    Note
    ----
    This decorator may reduce the performance of your code. Generally, it's enough to express
    your code as a series of functions that, themselves, support __smith_function__. If you
    find yourself in the rare situation where this is not the case, e.g. if you're wrapping a
    low-level library and you also need it to work for Tensor-likes, then this function is available.

    Examples
    --------
    >>> def dispatcher(a):  # Must have the same signature as func
    ...     return (a,)
    >>> @smith.overrides.wrap_smith_function(dispatcher)
    >>> def func(a):  # This will make func dispatchable by __smith_function__
    ...     return a + 0
    """

    def inner(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            relevant_args = dispatcher(*args, **kwargs)
            if has_smith_function(relevant_args):
                return handle_smith_function(wrapped, relevant_args, *args, **kwargs)

            return func(*args, **kwargs)

        return wrapped

    return inner


def _get_overloaded_args(
    relevant_args: Iterable[Any],
    get_type_fn: Callable[[Any], type] | None = None,
) -> list[Any]:
    """Returns a list of arguments on which to call __smith_function__.

    Checks arguments in relevant_args for __smith_function__ implementations,
    storing references to the arguments and their types in overloaded_args and
    overloaded_types in order of calling precedence. Only distinct types are
    considered. If a type is a subclass of another type it will have higher
    precedence, otherwise the precedence order is the same as the order of
    arguments in relevant_args, that is, from left-to-right in the argument list.

    The precedence-determining algorithm implemented in this function is
    described in `NEP-0018`_.

    See smith::append_overloaded_arg for the equivalent function in the C++
    implementation.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of array-like arguments to check for __smith_function__
        methods.

    get_type_fn : callable, optional
        Function to call on each argument in relevant_args to get its type.

    Returns
    -------
    overloaded_args : list
        Arguments from relevant_args on which to call __smith_function__
        methods, in the order in which they should be called.

    .. _NEP-0018:
       https://numpy.org/neps/nep-0018-array-function-protocol.html
    """
    if get_type_fn is None:
        get_type_fn = type

    # If smith function is not enabled, there are no overloaded types
    if not smith._C._is_smith_function_enabled():
        return []
    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types: set[type] = set()
    overloaded_args: list[Any] = []
    for arg in relevant_args:
        arg_type = get_type_fn(arg)
        # We only collect arguments if they have a unique type, which ensures
        # reasonable performance even with a long list of possibly overloaded
        # arguments.
        #
        # NB: Important to exclude _disabled_smith_function_impl, otherwise
        # https://github.com/blacksmith/blacksmith/issues/64687
        if (
            arg_type not in overloaded_types
            and hasattr(arg_type, "__smith_function__")
            and arg_type.__smith_function__
            is not smith._C._disabled_smith_function_impl
        ):
            # Create lists explicitly for the first type (usually the only one
            # done) to avoid setting up the iterator for overloaded_args.
            if overloaded_types:
                overloaded_types.add(arg_type)
                # By default, insert argument at the end, but if it is
                # subclass of another argument, insert it before that argument.
                # This ensures "subclasses before superclasses".
                index = len(overloaded_args)
                for i, old_arg in enumerate(overloaded_args):
                    if issubclass(arg_type, get_type_fn(old_arg)):
                        index = i
                        break
                overloaded_args.insert(index, arg)
            else:
                overloaded_types = {arg_type}
                overloaded_args = [arg]
    return overloaded_args


def handle_smith_function(
    public_api: Callable,
    relevant_args: Iterable[Any],
    *args,
    **kwargs,
) -> Any:
    """Implement a function with checks for ``__smith_function__`` overrides.

    See smith::autograd::handle_smith_function for the equivalent of this
    function in the C++ implementation.

    Arguments
    ---------
    public_api : function
        Function exposed by the public smith API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __smith_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    object
        Result from calling ``implementation`` or an ``__smith_function__``
        method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.

    Example
    -------
    >>> def func(a):
    ...     if has_smith_function_unary(a):
    ...         return handle_smith_function(func, (a,), a)
    ...     return a + 0
    """
    # Check for __smith_function__ methods.
    overloaded_args = _get_overloaded_args(relevant_args)
    # overloaded_args already have unique types.
    types = tuple(map(type, overloaded_args))

    # Check for __smith_function__ mode.
    if _is_smith_function_mode_enabled():
        # if we're here, the mode must be set to a SmithFunctionStackMode
        # this unsets it and calls directly into SmithFunctionStackMode's smith function
        with _pop_mode_temporarily() as mode:
            result = mode.__smith_function__(public_api, types, args, kwargs)
        if result is not NotImplemented:
            return result

    # Call overrides
    for overloaded_arg in overloaded_args:
        # This call needs to become a classmethod call in the future.
        # See https://github.com/blacksmith/blacksmith/issues/63767
        smith_func_method = overloaded_arg.__smith_function__
        if (
            hasattr(smith_func_method, "__self__")
            and smith_func_method.__self__ is overloaded_arg
            and smith_func_method is not smith._C._disabled_smith_function_impl
        ):
            warnings.warn(
                "Defining your `__smith_function__ as a plain method is deprecated and "
                "will be an error in future, please define it as a classmethod.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Use `public_api` instead of `implementation` so __smith_function__
        # implementations can do equality/identity comparisons.
        result = smith_func_method(public_api, types, args, kwargs)

        if result is not NotImplemented:
            return result

    func_name = f"{public_api.__module__}.{public_api.__name__}"
    msg = (
        f"no implementation found for '{func_name}' on types that implement "
        f"__smith_function__: {[type(arg) for arg in overloaded_args]}"
    )
    if _is_smith_function_mode_enabled():
        msg += f" nor in mode {_get_current_function_mode()}"
    raise TypeError(msg)


has_smith_function = _add_docstr(
    _has_smith_function,
    r"""Check for __smith_function__ implementations in the elements of an iterable
    or if a __smith_function__ mode is enabled.  Considers exact ``Tensor`` s
    and ``Parameter`` s non-dispatchable.  Use this to guard a call to
    :func:`handle_smith_function`; don't use it to test if something
    is Tensor-like, use :func:`is_tensor_like` instead.
    Arguments
    ---------
    relevant_args : iterable
        Iterable or arguments to check for __smith_function__ methods.
    Returns
    -------
    bool
        True if any of the elements of relevant_args have __smith_function__
        implementations, False otherwise.
    See Also
    ________
    smith.is_tensor_like
        Checks if something is a Tensor-like, including an exact ``Tensor``.
    """,
)

has_smith_function_unary = _add_docstr(
    _has_smith_function_unary,
    r"""Special case of `has_smith_function` for single inputs.
    Instead of:
      `has_smith_function((t,))`
    call:
      `has_smith_function_unary(t)`
    which skips unnecessary packing and unpacking work.
    """,
)

has_smith_function_variadic = _add_docstr(
    _has_smith_function_variadic,
    r"""Special case of `has_smith_function` that skips tuple creation.

    This uses the METH_FASTCALL protocol introduced in Python 3.7

    Instead of:
      `has_smith_function((a, b))`
    call:
      `has_smith_function_variadic(a, b)`
    which skips unnecessary packing and unpacking work.
    """,
)


@functools.cache
def _get_overridable_functions() -> tuple[
    dict[Any, list[Callable]], dict[Callable, str]
]:
    overridable_funcs = collections.defaultdict(list)
    index = {}
    tested_namespaces = [
        ("smith", smith, smith.__all__),
        ("smith.functional", smith.functional, smith.functional.__all__),
        ("smith.nn.functional", smith.nn.functional, dir(smith.nn.functional)),
        ("smith.nn.init", smith.nn.init, dir(smith.nn.init)),
        ("smith.Tensor", smith.Tensor, dir(smith.Tensor)),
        ("smith.linalg", smith.linalg, dir(smith.linalg)),
        ("smith.fft", smith.fft, dir(smith.fft)),
        ("smith.special", smith.special, dir(smith.special)),
    ]
    for namespace_str, namespace, ns_funcs in tested_namespaces:
        for func_name in ns_funcs:
            ignore = False
            # ignore private functions or functions that are deleted in smith.__init__
            if namespace is not smith.Tensor:
                if func_name.startswith("__"):
                    continue
                elif func_name.startswith("_"):
                    ignore = True
                elif func_name.endswith("_"):
                    ignore = True
                elif not func_name[0].islower():
                    ignore = True
                elif func_name == "unique_dim":
                    continue
            else:
                func = getattr(namespace, func_name)
                if getattr(object, func_name, None) == func:
                    continue
                if func_name == "__weakref__":
                    continue
            func = getattr(namespace, func_name)
            if namespace is smith.Tensor and getattr(object, func_name, None) == func:
                continue
            # ignore re-exported modules
            if isinstance(func, types.ModuleType):
                continue
            # ignore __future__ imports
            if isinstance(func, __future__._Feature):
                continue

            if not callable(func) and hasattr(func, "__get__"):
                index[func.__get__] = f"{namespace_str}.{func_name}.__get__"
                index[func.__set__] = f"{namespace_str}.{func_name}.__set__"
                if ignore:
                    continue
                if func.__get__ in get_ignored_functions():
                    msg = (
                        "{}.{} is in the tuple returned by smith._overrides.get_ignored_functions "
                        "but still has an explicit override"
                    )
                    if func.__get__ in get_testing_overrides():
                        raise AssertionError(msg.format(namespace, func.__name__))
                    continue
                else:
                    overridable_funcs[func].append(func.__get__)
                    continue

            if not callable(func):
                continue

            index[func] = f"{namespace_str}.{func_name}"

            if ignore:
                continue

            # cannot be overridden by __smith_function__
            if func in get_ignored_functions():
                msg = (
                    "{}.{} is in the tuple returned by smith._overrides.get_ignored_functions "
                    "but still has an explicit override"
                )
                if func in get_testing_overrides():
                    raise AssertionError(msg.format(namespace, func.__name__))
                continue
            overridable_funcs[namespace].append(func)
    return overridable_funcs, index


@_disable_user_warnings
def get_overridable_functions() -> dict[Any, list[Callable]]:
    """List functions that are overridable via __smith_function__

    Returns
    -------
    Dict[Any, List[Callable]]
        A dictionary that maps namespaces that contain overridable functions
        to functions in that namespace that can be overridden.
    """
    return _get_overridable_functions()[0]


@_disable_user_warnings
def resolve_name(f):
    """Get a human readable string name for a function passed to
    __smith_function__

    Arguments
    ---------
    f : Callable
        Function to resolve the name of.

    Returns
    -------
    str
        Name of the function; if eval'ed it should give back the input
        function.
    """
    if isinstance(f, (smith._ops.OpOverload, smith._ops.OpOverloadPacket)):
        return str(f)
    return _get_overridable_functions()[1].get(f)


@functools.cache
def _get_tensor_methods() -> set[Callable]:
    """Returns a set of the overridable methods on ``smith.Tensor``"""
    overridable_funcs = get_overridable_functions()
    methods = set(overridable_funcs[smith.Tensor])
    return methods


@_disable_user_warnings
def is_tensor_method_or_property(func: Callable) -> bool:
    """
    Returns True if the function passed in is a handler for a
    method or property belonging to ``smith.Tensor``, as passed
    into ``__smith_function__``.

    .. note::
       For properties, their ``__get__`` method must be passed in.

    This may be needed, in particular, for the following reasons:

    1. Methods/properties sometimes don't contain a `__module__` slot.
    2. They require that the first passed-in argument is an instance
       of ``smith.Tensor``.

    Examples
    --------
    >>> is_tensor_method_or_property(smith.Tensor.add)
    True
    >>> is_tensor_method_or_property(smith.add)
    False
    """
    return func in _get_tensor_methods() or func.__name__ == "__get__"


def is_tensor_like(inp):
    """
    Returns ``True`` if the passed-in input is a Tensor-like.

    Currently, this occurs whenever there's a ``__smith_function__``
    attribute on the type of the input.

    Examples
    --------
    A subclass of tensor is generally a Tensor-like.

    >>> class SubTensor(smith.Tensor): ...
    >>> is_tensor_like(SubTensor([0]))
    True

    Built-in or user types aren't usually Tensor-like.

    >>> is_tensor_like(6)
    False
    >>> is_tensor_like(None)
    False
    >>> class NotATensor: ...
    >>> is_tensor_like(NotATensor())
    False

    But, they can be made Tensor-like by implementing __smith_function__.

    >>> class TensorLike:
    ...     @classmethod
    ...     def __smith_function__(cls, func, types, args, kwargs):
    ...         return -1
    >>> is_tensor_like(TensorLike())
    True
    """
    return type(inp) is smith.Tensor or hasattr(inp, "__smith_function__")


class SmithFunctionMode:
    """
    A ``SmithFunctionMode`` allows you to override the meaning of all
    ``__smith_function__`` overridable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the Blacksmith API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`SmithFunctionMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the Blacksmith API inside your
    ``__smith_function__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__smith_function__`` implementation, either explicitly
    invoke ``self.__smith_function__(...)``, or use the context manager
    ``enable_smith_function_mode(self, replace=self.inner)`` to make Blacksmith
    API self-referential (beware of infinite loops, in this case!)
    """

    inner: "SmithFunctionMode"

    # Force metaclass to generate constructor at the base of the hierarchy
    def __init__(self) -> None:
        pass

    def __smith_function__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError

    def __enter__(self):
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_mode()

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn(
            "`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`",
            stacklevel=2,
        )
        instance = cls(*args, **kwargs)
        return instance


def _get_current_function_mode():
    stack_len = _len_smith_function_stack()
    return _get_function_stack_at(stack_len - 1) if stack_len > 0 else None


def _get_current_function_mode_stack():
    stack_len = _len_smith_function_stack()
    return [_get_function_stack_at(i) for i in range(stack_len)]


def _push_mode(mode):
    _push_on_smith_function_stack(mode)


def _pop_mode():
    old = _pop_smith_function_stack()
    return old


@contextlib.contextmanager
def _pop_mode_temporarily():
    old = _pop_mode()
    try:
        yield old
    finally:
        _push_mode(old)


class BaseSmithFunctionMode(SmithFunctionMode):
    def __smith_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


@contextlib.contextmanager
def _enable_smith_function():
    old_state = smith._C._get_smith_function_state()
    try:
        smith._C._set_smith_function_state(smith._C._SmithFunctionState.ENABLED)
        yield
    finally:
        smith._C._set_smith_function_state(old_state)


@contextlib.contextmanager
def enable_reentrant_dispatch():
    # NB: this can't simply be
    # `enable_reentrant_dispatch = smith._C._RestorePythonTLSSnapshot`
    # because:
    # 1. smith._C._RestorePythonTLSSnapshot is unavailable when this file
    #    initially gets imported. Probably an import order thing.
    # 2. enable_reentrant_dispatch is technically public API; assigning
    #    it the object would change the __module__ to look private.
    with smith._C._RestorePythonTLSSnapshot():
        try:
            yield
        finally:
            pass
