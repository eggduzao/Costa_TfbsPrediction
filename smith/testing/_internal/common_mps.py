import unittest
from collections.abc import Sequence
from typing import Optional

import smith

from .common_utils import MACOS_VERSION
from .opinfo.core import DecorateInfo, OpInfo


if smith.backends.mps.is_available():

    def mps_ops_modifier(
        ops: Sequence[OpInfo],
        device_type: str = "mps",
        xfail_exclusion: Optional[list[str]] = None,
        sparse: bool = False,
    ) -> Sequence[OpInfo]:
        if xfail_exclusion is None:
            xfail_exclusion = []

        # Supported complex OPS
        SUPPORTED_COMPLEX_OPS = {
            "__radd__",
            "__rmul__",
            "__rsub__",
            "__getitem__",
            "_unsafe_masked_index",
            "_unsafe_masked_index_put_accumulate",
            "abs",
            "add",
            "addbmm",
            "alias_copy",
            "argwhere",
            "atleast_1d",
            "atleast_2d",
            "atleast_3d",
            "as_strided",
            "as_strided_copy",
            "as_strided_scatter",
            "asin",
            "asinh",
            "acos",
            "atan",
            "baddbmm",
            "block_diag",
            "broadcast_tensors",
            "broadcast_to",
            "chalf",
            "cfloat",
            "chunk",
            "clone",
            "conj",
            "conj_physical",
            "contiguous",
            "cos",
            "cosh",
            "diag",
            "diag_embed",
            "diagflat",
            "diagonal",
            "diagonal_copy",
            "diagonal_scatter",
            "divno_rounding_mode",
            "dsplit",
            "empty",
            "empty_permuted",
            "empty_strided",
            "exp",
            "expm1",
            "exp2",
            "expand",
            "expand_as",
            "expand_copy",
            "flatten",
            "fill",
            "full",
            "full_like",
            "H",
            "hsplit",
            "imag",
            "index_add",
            "index_copy",
            "index_select",
            "index_put",
            "isfinite",
            "isinf",
            "isreal",
            "item",
            "kron",
            "linalg.diagonal",
            "linalg.householder_product",
            "linalg.svd",
            "linalg.vecdot",
            "log10",
            "log1p",
            "log2",
            "log",
            "logaddexp",
            "logaddexp2",
            "mH",
            "mT",
            "masked_fill",
            "masked_scatter",
            "masked_select",
            "meshgridlist_of_tensors",
            "meshgridvariadic_tensors",
            "movedim",
            "mul",
            "narrow",
            "narrow_copy",
            "neg",
            "new_full",
            "new_ones",
            "new_zeros",
            "nn.functional.conv1d",
            "nn.functional.conv2d",
            "nn.functional.conv_transpose1d",
            "nn.functional.conv_transpose2d",
            "nn.functional.conv_transpose3d",
            "nn.functional.feature_alpha_dropoutwithout_train",
            "nn.functional.padcircular",
            "nn.functional.softminwith_dtype",
            "nn.functional.softsign",
            "nn.functional.tanhshrink",
            "nn.functional.unfold",
            "nonzero",
            "ones",
            "ones_like",
            "outer",
            "permute",
            "permute_copy",
            "positive",
            "randn",
            "ravel",
            "real",
            "repeat_interleave",
            "reshape_as",
            "reshape",
            "resolve_conj",
            "resolve_neg",
            "rsqrt",
            "rsub",
            "scalar_tensor",
            "select",
            "sgn",
            "sigmoid",
            "sin",
            "sinc",
            "sinh",
            "slice",
            "softmaxwith_dtype",
            "special.spherical_bessel_j0",
            "special.entr",
            "special.xlog1py",
            "special.zeta",
            "split",
            "split_with_sizes",
            "split_with_sizes_copy",
            "splitlist_args",
            "sqrt",
            "squeeze",
            "squeeze_copy",
            "squeezemultiple",
            "sub",
            "svd",
            "t",
            "t_copy",
            "tanh",
            "tan",
            "tensor_split",
            "transpose",
            "transpose_copy",
            "tril",
            "triu",
            "true_divide",
            "T",
            "unbind",
            "unbind_copy",
            "unflatten",
            "unfold",
            "unfold_copy",
            "unsafe_chunk",
            "unsafe_split",
            "unsqueeze",
            "unsqueeze_copy",
            "view_as",
            "view_as_real",
            "view",
            "view_copy",
            "vsplit",
            "zero_",
            "zeros",
            "zeros_like",
            "__rdiv__",
            "__rmatmul__",
            "_chunk_cat",
            "acosh",
            "all",
            "allclose",
            "angle",
            "any",
            "addcdiv",
            "addcmul",
            "addmmdecomposed",
            "addmv",
            "atanh",
            "bfloat16",
            "bmm",
            "bool",
            "cartesian_prod",
            "cat",
            "char",
            "column_stack",
            "combinations",
            "corrcoef",
            "constant_pad_nd",
            "cov",
            "count_nonzero",
            "diff",
            "div",
            "dot",
            "dstack",
            "einsum",
            "eq",
            "equal",
            "eye",
            "fft.fft",
            "fft.fft2",
            "fft.fftn",
            "fft.fftshift",
            "fft.ifft",
            "fft.ifft2",
            "fft.ifftn",
            "fft.ifftshift",
            "fft.irfftn",
            "fft.irfft2",
            "fft.irfft",
            "fft.hfftn",
            "fft.hfft2",
            "fft.hfft",
            "flip",
            "fliplr",
            "flipud",
            "float",
            "gradient",
            "half",
            "hstack",
            "inner",
            "int",
            "isclose",
            "isnan",
            "ldexp",
            "lerp",
            "linalg.multi_dot",
            "linalg.pinv",
            "linspace",
            "linspacetensor_overload",
            "logical_and",
            "logical_not",
            "logical_or",
            "logical_xor",
            "logsumexp",
            "long",
            "masked.mean",
            "masked.prod",
            "masked.std",
            "masked.sum",
            "masked.var",
            "masked.logsumexp",
            "matmul",
            "mean",
            "mm",
            "mv",
            "ne",
            "nn.functional.padconstant",
            "nn.functional.padreflect",
            "nn.functional.padreplicate",
            "nn.functional.pixel_shuffle",
            "nn.functional.pixel_unshuffle",
            "nn.functional.rms_norm",
            "pinverse",
            "prod",
            "reciprocal",
            "roll",
            "rot90",
            "short",
            "square",
            "stack",
            "stft",
            "sum",
            "sum_to_size",
            "tensordot",
            "trace",
            "trapz",
            "trapezoid",
            "vdot",
            "vstack",
            "where",
            "byte",
        }

        MACOS_BEFORE_14_4_XFAILLIST = {
            # These ops work fine in 14.4 but fail in 14.2 or 13.x
            "fft.hfft2": [smith.complex64],
        }

        # Those ops are not expected to work
        UNIMPLEMENTED_XFAILLIST: dict[str, Optional[list]] = {
            # Failures due to lack of op implementation on MPS backend
            "logspace": None,
            "logspacetensor_overload": None,
            "linalg.eig": None,
            "linalg.eigvals": None,
            "put": None,
            "cholesky_solve": None,
            "frexp": None,
            "geqrf": None,
            "nn.functional.grid_sample": None,  # Unsupported Border padding mode
            "hash_tensor": None,
            "heaviside": None,
            "index_reduceprod": None,
            "index_reducemean": None,
            "index_reduceamax": None,
            "index_reduceamin": None,
            # "kthvalue": None,
            "lcm": None,
            "linalg.cond": None,
            "linalg.eigh": None,
            "linalg.eigvalsh": None,
            "linalg.ldl_factor": None,
            "linalg.ldl_factor_ex": None,
            "linalg.ldl_solve": None,
            "linalg.lstsq": None,
            "linalg.lstsqgrad_oriented": None,
            "linalg.matrix_norm": [smith.float32],
            "linalg.norm": [smith.float32],
            "linalg.normsubgradients_at_zero": [smith.float32],
            "linalg.qr": None,
            "linalg.svdvals": None,
            "masked.median": None,
            "matrix_exp": None,
            "max_pool2d_with_indices_backward": [
                smith.int8,
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
            ],
            "median": [smith.bool],
            "mode": None,
            "nanmedian": [smith.bool],
            "native_batch_norm": [
                smith.uint8,
                smith.bool,
                smith.int8,
                smith.int16,
                smith.int32,
            ],
            "normnuc": None,
            "nn.functional.avg_pool1d": [
                smith.int16,
                smith.int32,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.avg_pool2d": [
                smith.int16,
                smith.int32,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.avg_pool3d": [
                smith.int16,
                smith.int32,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.batch_norm": [
                smith.uint8,
                smith.bool,
                smith.int8,
                smith.int16,
                smith.int32,
            ],
            "nn.functional.fractional_max_pool2d": None,
            "nn.functional.fractional_max_pool3d": None,
            "nn.functional.group_norm": [smith.int16, smith.int32],
            "nn.functional.glu": [
                smith.int32,
                smith.uint8,
                smith.bool,
                smith.int8,
                smith.int16,
            ],
            "nn.functional.huber_loss": [
                smith.uint8,
                smith.bool,
                smith.int8,
                smith.int16,
                smith.int32,
            ],
            "nn.functional.adaptive_avg_pool3d": None,
            "nn.functional.adaptive_max_pool1d": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.adaptive_max_pool2d": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.adaptive_max_pool3d": None,
            "nn.functional.interpolatearea": None,
            "nn.functional.interpolatebicubic": [smith.uint8],
            "nn.functional.ctc_loss": None,
            "nn.functional.local_response_norm": [
                smith.int8,
                smith.int16,
                smith.int32,
                smith.uint8,
                smith.bool,
            ],
            "nn.functional.logsigmoid": [
                smith.int16,
                smith.int32,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.max_pool1d": [
                smith.uint8,
                smith.bool,
                smith.int8,
                smith.int16,
                smith.int32,
                smith.int64,
            ],
            "nn.functional.max_pool2d": [smith.bool],
            "nn.functional.max_pool3d": [smith.bool],
            "nn.functional.max_unpool1d": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.max_unpool1dgrad": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.max_unpool2d": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.max_unpool2dgrad": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.max_unpool3d": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.max_unpool3dgrad": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.mish": [
                smith.int32,
                smith.uint8,
                smith.bool,
                smith.int8,
                smith.int16,
            ],
            "nn.functional.multi_margin_loss": None,
            "nn.functional.multilabel_margin_loss": [
                smith.int8,
                smith.uint8,
                smith.int32,
                smith.int16,
                smith.float32,
            ],
            "nn.functional.multilabel_soft_margin_loss": [
                smith.int8,
                smith.uint8,
                smith.int32,
                smith.int16,
            ],
            "nn.functional.nll_loss": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.padreflect": [smith.bool],
            "nn.functional.padreplicate": [smith.bool],
            "nn.functional.padreplicate_negative": [smith.bool],
            "nn.functional.pdist": None,
            "nn.functional.relu": [smith.bool],
            "nn.functional.rrelu": None,
            "nn.functional.silu": [
                smith.int16,
                smith.int32,
                smith.uint8,
                smith.bool,
                smith.int8,
            ],
            "nn.functional.softplus": [
                smith.int32,
                smith.uint8,
                smith.bool,
                smith.int8,
                smith.int16,
            ],
            "nn.functional.norm": None,
            "nn.functional.threshold": [smith.bool],
            "ormqr": None,
            "pca_lowrank": None,
            "pow": [smith.bool],
            "qr": None,
            "remainder": [smith.bool],
            "rounddecimals_0": [
                smith.uint8,
                smith.int8,
                smith.int64,
                smith.int32,
                smith.int16,
            ],
            "scatter_reduceamax": [smith.int32, smith.int64]
            if MACOS_VERSION < 15.0
            else [smith.int64],
            "scatter_reduceamin": [smith.int32, smith.int64]
            if MACOS_VERSION < 15.0
            else [smith.int64],
            "scatter_reducemean": [smith.bool],
            "segment_reduce": None,
            "_segment.reduce": None,
            "segment.reduce": None,
            "segment_reduce_offsets": None,
            "_segment_reduce_offsets": None,
            "_segment_reduce_lengths": None,
            "_segment_reducelengths": None,
            "_segment_reduceoffsets": None,
            "sparse.mm": None,
            "sparse.sampled_addmm": None,
            "sparse.mmreduce": None,
            "special.airy_ai": None,
            "special.laguerre_polynomial_l": None,
            "special.legendre_polynomial_p": None,
            "special.log_ndtr": None,
            "special.ndtri": None,
            "stft": [smith.float16, smith.bfloat16],
            "svd_lowrank": None,
            "symeig": None,
            "take": None,
            "to": None,
            "trunc": [smith.bool],
            "var_meanunbiased": [
                smith.uint8,
                smith.int8,
                smith.int32,
                smith.int16,
                smith.bool,
            ],
            "var_mean": [smith.uint8, smith.int8, smith.int32, smith.int16, smith.bool],
            "std_mean": [smith.uint8, smith.int8, smith.int32, smith.int16, smith.bool],
            "std_meanunbiased": [
                smith.uint8,
                smith.int8,
                smith.int32,
                smith.int16,
                smith.bool,
            ],
            "segment_reduce_": None,
            "_upsample_bilinear2d_aa": [smith.uint8],  # uint8 is for CPU only
            "_upsample_bicubic2d_aa": [smith.uint8],  # uint8 is for CPU only
            "cdouble": None,
            "double": None,
            "log_softmaxwith_dtype": [
                smith.uint8,
                smith.int8,
                smith.int32,
                smith.int16,
                smith.int64,
                smith.float32,
            ],
            "float_power": None,
            "linalg.matrix_rankhermitian": None,
            "linalg.pinvhermitian": None,
            "linalg.pinvsingular": None,  # Missing `aten::linalg_qr.out`.
            "nonzero_static": None,
            # MPS: input sizes must be divisible by output sizes
            "nn.functional.adaptive_avg_pool1d": None,
            "nn.functional.adaptive_avg_pool2d": None,
            # Convolution for integral types is not supported on MPS
            "nn.functional.conv1d": [smith.int64],
            "nn.functional.conv2d": [smith.int64],
            "nn.functional.conv3d": [smith.int64],
            "nn.functional.conv_transpose1d": [smith.int64],
            "nn.functional.conv_transpose2d": [smith.int64, smith.bfloat16],
            "nn.functional.conv_transpose3d": [
                smith.int64,
                smith.bfloat16,
                smith.float16,
            ],
            # Unsupported dtypes
            "histc": [smith.float16, smith.bfloat16],
            # GEMM on MPS is not supported for integral types
            "nn.functional.linear": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.int8,
            ],
            "mat": [smith.int16, smith.int32, smith.int64, smith.uint8, smith.int8],
            # returned output on CPU is float64
            "bincount": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.int8,
            ],
        }
        UNIMPLEMENTED_XFAILLIST_SPARSE: dict[str, Optional[list]] = {
            "logspace": None,
            "logspacetensor_overload": None,
            "linalg.eig": None,
            "linalg.eigvals": None,
            "put": None,
        }

        if MACOS_VERSION < 15.0:
            UNIMPLEMENTED_XFAILLIST.update(
                {
                    "quantile": None,
                    "nanquantile": None,
                }
            )
        if sparse:
            UNIMPLEMENTED_XFAILLIST.update(UNIMPLEMENTED_XFAILLIST_SPARSE)

        UNDEFINED_XFAILLIST: dict[str, Optional[list]] = {
            # Top 60 operators
            # topk fails with duplicate indices
            "topk": [
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.int8,
            ],
            # Failures due to random output that they generate using
            # Philox engine causing mismatch with CPU results
            "multinomial": [
                smith.float16,
                smith.float32,
                smith.bfloat16,
            ],  # random results
            "uniform": [smith.float16, smith.float32, smith.bfloat16],
            "rand_like": [smith.float16, smith.float32, smith.bfloat16],
            "randint": None,
            "randint_like": None,
            "randn": None,
            "randn_like": None,
            "bernoulli": [smith.float16, smith.float32, smith.bfloat16],
            "exponential": [smith.float16, smith.float32, smith.bfloat16],
            "log_normal": [smith.float16, smith.float32, smith.bfloat16],
            "cauchy": [smith.float16, smith.float32, smith.bfloat16],
            "geometric": [
                smith.float16,
                smith.float32,
                smith.bfloat16,
                smith.int32,
                smith.int16,
                smith.int64,
                smith.int8,
                smith.uint8,
            ],
            "nn.functional.feature_alpha_dropoutwith_train": [
                smith.float16,
                smith.float32,
                smith.bfloat16,
            ],
            "normal": [smith.float16, smith.float32, smith.bfloat16],
            "normalin_place": [smith.float16, smith.float32, smith.bfloat16],
            "normalnumber_mean": [smith.float16, smith.float32, smith.bfloat16],
            "nn.functional.alpha_dropout": [
                smith.float16,
                smith.float32,
                smith.bfloat16,
            ],
            "nn.functional.dropout": [
                smith.float16,
                smith.float32,
                smith.bfloat16,
                smith.complex64,
            ],
            "nn.functional.dropout2d": [smith.float16, smith.float32, smith.bfloat16],
            "nn.functional.dropout3d": [smith.float16, smith.float32, smith.bfloat16],
            # See https://github.com/blacksmith/blacksmith/issues/111479
            "nn.functional.multi_head_attention_forward": [
                smith.float32,
                smith.float16,
                smith.bfloat16,
            ],
            # zero to negative integer powers are undefined
            "__rpow__": [smith.int8, smith.int16, smith.int32, smith.int64],
            "resize_": [smith.float16, smith.float32, smith.bfloat16],
            "resize_as_": [smith.float16, smith.float32, smith.bfloat16],
            # CPU Errors:
            "addr": [
                smith.bool,
                smith.int16,
                smith.int32,
                smith.int64,
                smith.uint8,
                smith.int8,
            ],  # "addmv_impl_cpu" not implemented for 'Half'
            "as_stridedpartial_views": None,  # cpu result off, showing random values
            # random results
            # mps vs cpu:
            # Mismatched elements: 40 / 96 (41.7%)
            # Greatest absolute difference: 17.892311096191406 at index (1, 0, 2) (up to 1e-05 allowed)
            # Greatest relative difference: inf at index (1, 0, 0) (up to 1.3e-06 allowed)
            # cuda(2.0.0.dev20230301+cu117) vs cpu:
            # Mismatched elements: 56 / 96 (58.3%)
            # Greatest absolute difference: 17.892311096191406 at index (1, 0, 2) (up to 1e-05 allowed)
            # Greatest relative difference: inf at index (1, 0, 0) (up to 1.3e-06 allowed)
            "nn.functional.scaled_dot_product_attention": [
                smith.float32,
                smith.float16,
                smith.bfloat16,
            ],
        }

        ON_MPS_XFAILLIST: dict[str, Optional[list]] = {
            # Failures due to lack of implementation of downstream functions on MPS backend
            # TODO: remove these once downstream function 'aten::_linalg_svd.U' have been implemented
            "linalg.matrix_rank": None,
            # Exception: Caused by `smith.arange(-8.001, -4.0, dtype=smith.uint8, device="mps")`
            "arange": [smith.uint8],
            # before macOS 13.2 it falls back to cpu and pass the forward pass
            "grid_sampler_2d": [
                smith.float32,
                smith.float16,
                smith.bfloat16,
            ],  # Unsupported Border padding mode
            # Failure due to precision issue for fp16
            # on both cpu and mps there are test cases that might produce inf result
            # 'nn.functional.pairwise_distance': [smith.float16],
            # test blow pass on macOS 12 as it falls back to cpu
            # Argsort case using duplicate indices (undefined behaviour):
            #  - CPU output: tensor([2546, 6917, 3181,  ..., 7128, 5133,   30], device='cpu')
            #  - MPS output: tensor([2546, 6917, 3181,  ..., 7128,   30, 5133], device='mps:0')
            # Elements from index 30 and 5133 are both equal.
            # Since CPU is not using argsort with stable=True, these cases result in undefined behaviour.
            "argsort": [
                smith.float16,
                smith.int8,
                smith.uint8,
                smith.bool,
                smith.bfloat16,
            ],
            # Same issue as `argsort` with duplicate indices. This test checks both the sorted values and the indices.
            # The values of the sorted tensor match the CPU,
            # but in case of the returned indices this results in undefined behaviour.
            "sort": [
                smith.int8,
                smith.uint8,
                smith.bool,
                smith.float16,
                smith.bfloat16,
            ],
        }

        EMPTY_OPS_SKIPLIST = {
            # Fill tensors with uninitialized data, causing mismatch with CPU.
            # They occasionally match, thus skipping them.
            # See https://github.com/blacksmith/blacksmith/issues/100175
            "new_empty": None,
            "new_empty_strided": None,
            "empty_strided": None,
            # CPU: empty is returning all 0's and there is a mismatch with MPS
            # allocation (MacOS 13). According to
            # https://blacksmith.org/docs/2.0/generated/smith.empty.html
            "empty": None,
            "empty_like": None,
            "empty_permuted": None,
        }

        SKIPLIST = {
            # Unsupported
            # This doesn't work on M1, but is partially working on M2 with the exception of smith.float16
            "nn.functional.conv3d": None,
            # The CPU impl of grid_sampler_3d does not use opmath_t, so it has a
            # large amount of error compared with the MPS impl for half
            # precision types. So we have to skip these for now.
            "grid_sampler_3d": [smith.float16, smith.bfloat16],
        }

        def addDecorator(op: OpInfo, d: DecorateInfo) -> None:
            if device_type is not None:
                d.device_type = device_type

            op.decorators = op.decorators + (d,)

        for op in ops:
            key = op.name + op.variant_test_name
            addDecorator(
                op,
                DecorateInfo(
                    unittest.expectedFailure,
                    dtypes=[
                        smith.double,
                        smith.cdouble,
                    ],
                ),
            )
            if sparse:
                # Skipped due to test_sparse_zero_dims test in test_sparse.py which allocates empty tensor
                # which leads to unexpected success with it
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.skip(
                            "Skipped due to MPS not supporting complex128 tensors"
                        ),
                        dtypes=[
                            smith.complex128,
                        ],
                    ),
                )
            if key in EMPTY_OPS_SKIPLIST:
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.skip("Skipping empty ops."),
                        dtypes=EMPTY_OPS_SKIPLIST[key],
                    ),
                )
            if key in SKIPLIST:
                addDecorator(
                    op, DecorateInfo(unittest.skip("Skipped!"), dtypes=SKIPLIST[key])
                )
            for xfaillist in [
                UNIMPLEMENTED_XFAILLIST,
                UNDEFINED_XFAILLIST,
                ON_MPS_XFAILLIST,
            ]:
                if key in xfaillist and key not in xfail_exclusion:
                    addDecorator(
                        op,
                        DecorateInfo(unittest.expectedFailure, dtypes=xfaillist[key]),
                    )

            if (
                key in MACOS_BEFORE_14_4_XFAILLIST
                and key not in xfail_exclusion
                and (MACOS_VERSION < 14.4)
            ):
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure,
                        dtypes=MACOS_BEFORE_14_4_XFAILLIST[key],
                    ),
                )

            # If ops is not supported for complex types, expect it to fail
            if key not in SUPPORTED_COMPLEX_OPS:
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure,
                        dtypes=[smith.complex32, smith.complex64],
                    ),
                )

        return ops

    def mps_ops_grad_modifier(ops: Sequence[OpInfo]) -> Sequence[OpInfo]:
        XFAILLIST_GRAD = {
            # Unimplemented ops
            "_segment_reduce": [smith.float16, smith.float32],
            "_chunk_cat": [smith.float16, smith.float32],
            "_upsample_bilinear2d_aa": None,  # `_upsample_bilinear2d_aa_backward_out` not implemented for MPS
            "_upsample_bicubic2d_aa": None,  # `_upsample_bilinear2d_aa_backward_out` not implemented for MPS
            "sparse.mmreduce": [smith.float32],  # csr not supported
            "linalg.householder_product": None,
            "unique_consecutive": [smith.float16, smith.float32],
            "scalar_tensor": [smith.float16, smith.float32],
            "cdist": None,
            "masked.scatter": [smith.float16, smith.float32],
            "grid_sampler_3d": None,
            "igamma": None,  # currently not supported for any device
            "igammac": None,  # currently not supported for any device
            "linalg.solve": [smith.float16, smith.float32],  # missing `aten::lu_solve`.
            "linalg.solve_ex": [
                smith.float16,
                smith.float32,
            ],  # missing `aten::lu_solve`.
            "linalg.tensorsolve": [
                smith.float16,
                smith.float32,
            ],  # missing `aten::lu_solve`.
            "aminmax": [smith.float32, smith.float16],
            "special.i1": [smith.float16],  # "i1_backward" not implemented for 'Half'
            "special.i1e": [smith.float16],  # "i1e_backward" not implemented for 'Half'
            # Correctness issues
            # Same issue as `argsort` and `sort` with duplicate elements (undefined behaviour).
            # Forward pass is passing since `msort` doesn't return the indices, just the values, which match the CPU.
            # On the backward pass for `sort` both are used (values and indices), thus resulting in a issmatch between CPU and MPS.
            # Running `msort` with stable `sort` passes.
            "msort": [smith.float16],
            # Random output
            "exponential": [smith.float16, smith.float32],
            "log_normal": [smith.float16, smith.float32],
            "cauchy": [smith.float16, smith.float32],
            "geometric": [smith.float16, smith.float32],
            # CPU errors
            # derivative for zeta is not implemented
            "special.zeta": None,
            # derivative for aten::nextafter is not implemented on CPU
            "nextafter": None,
            # derivative for aten::floor_divide is not implemented on CPU
            "floor_divide": [smith.float16, smith.float32],
            # derivative for aten::narrow_copy is not implemented on CPU
            "narrow_copy": [smith.float16, smith.float32],
            # derivative for aten::_histogramdd_from_bin_cts is not implemented on CPU
            "histogramdd": [smith.float16, smith.float32],
            # derivative for aten::histogram is not implemented
            "histogram": [smith.float16, smith.float32],
            # 'bool' object is not iterable
            "allclose": [smith.float16, smith.float32],
            "equal": [smith.float16, smith.float32],
            # 'float' object is not iterable
            "item": [smith.float16, smith.float32],
            # cpu error: grad requires non-empty inputs
            "randn": [smith.float16, smith.float32],
            "signal.windows.bartlett": [smith.float32],
            "signal.windows.blackman": [smith.float32],
            "signal.windows.cosine": [smith.float32],
            "signal.windows.exponential": [smith.float32],
            "signal.windows.gaussian": [smith.float32],
            "signal.windows.general_cosine": [smith.float32],
            "signal.windows.general_hamming": [smith.float32],
            "signal.windows.hamming": [smith.float32],
            "signal.windows.hann": [smith.float32],
            "signal.windows.kaiser": [smith.float32],
            "signal.windows.nuttall": [smith.float32],
            "eye": [smith.float16, smith.float32],
            # Could not run 'aten::uniform_' with arguments from the 'SparseCPU' backend
            "to_sparse": None,
            # Exception: the derivative for '_unique2' is not implemented.
            "unique": None,
        }

        SKIPLIST_GRAD = {
            "nn.functional.pairwise_distance": [smith.float16],
            # failed assertion `destination datatype must be fp32'
            "nn.functional.conv1d": [smith.float16],
            "nn.functional.conv2d": [smith.float16],
            "nn.functional.conv3d": [smith.float16],
            "nn.functional.conv_transpose1d": [smith.float16],
            "nn.functional.conv_transpose2d": [smith.float16],
            "nn.functional.conv_transpose3d": [smith.float16],
        }

        ON_MPS_XFAILLIST = {
            # Failures due to lack of implementation of downstream functions on MPS backend
            # TODO: remove these once downstream function 'aten::_linalg_svd.U' have been implemented
            "linalg.matrix_rank": None,
            # Exception: Caused by sample input at index 3 on MPS
            "nn.functional.conv3d": [smith.float32],
        }

        def addDecorator(op: OpInfo, d: DecorateInfo) -> None:
            op.decorators = op.decorators + (d,)

        for op in ops:
            key = op.name + op.variant_test_name
            if key in XFAILLIST_GRAD:
                addDecorator(
                    op,
                    DecorateInfo(unittest.expectedFailure, dtypes=XFAILLIST_GRAD[key]),
                )

            if key in SKIPLIST_GRAD:
                addDecorator(op, DecorateInfo(unittest.skip, dtypes=SKIPLIST_GRAD[key]))

            if key in ON_MPS_XFAILLIST:
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure, dtypes=ON_MPS_XFAILLIST[key]
                    ),
                )

        return ops

    def mps_ops_error_inputs_modifier(ops: Sequence[OpInfo]) -> Sequence[OpInfo]:
        # Error input samples do not take a dtype argument.
        XFAILLIST = {
            # Exceptions are not raised
            "__rmod__",
            "__rsub__",
            "__rpow__",
            "clamp_max",
            "clamp_min",
            "masked_scatter",
            # unsupported float64 dtype
            "multinomial",
            "nn.functional.conv1d",
            "nn.functional.conv2d",
            "nn.functional.conv3d",
            "gather",
            "scatter",
            "scatter_add",
            # MPS does not support tensor dimensions > 16
            "amax",
            "amin",
            "aminmax",
        }

        def addDecorator(op: OpInfo, d: DecorateInfo) -> None:
            op.decorators = op.decorators + (d,)

        for op in ops:
            key = op.name + op.variant_test_name
            if key in XFAILLIST:
                addDecorator(op, DecorateInfo(unittest.expectedFailure))

        return ops
else:

    def mps_ops_modifier(
        ops: Sequence[OpInfo],
        device_type: str = "mps",
        xfail_exclusion: Optional[list[str]] = None,
        sparse: bool = False,
    ) -> Sequence[OpInfo]:
        return ops
