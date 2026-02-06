# Owner(s): ["module: linear algebra"]

import contextlib
import os
import time
import unittest
from itertools import product
from functools import partial
from collections.abc import Callable

import smith
import smith.nn.functional as F
from smith.profiler import profile, ProfilerActivity

from smith.quantization._quantized_conversions import (
    pack_int4_to_int8,
    quantized_weight_reorder_for_mixed_dtypes_linear_cutlass,
)

from smith.testing import make_tensor
from smith.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_BF16,
    PLATFORM_SUPPORTS_GREEN_CONTEXT,
    SM80OrLater,
    SM90OrLater,
    SM100OrLater,
)
from smith.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    skipCUDAIfNotRocm,
    tol as xtol,
    toleranceOverride,
)

from smith.testing._internal.common_utils import (
    IS_JETSON,
    IS_WINDOWS,
    MI200_ARCH,
    NAVI_ARCH,
    getRocmVersion,
    isRocmArchAnyOf,
    parametrize,
    random_matrix_with_scaled_reduction_dim,
    run_tests,
    runOnRocmArch,
    serialTest,
    skipIfRocm,
    TEST_CUDA,
    TEST_WITH_ROCM,
    TestCase,
    decorateIf,
)

from smith.testing._internal.inductor_utils import IS_BIG_GPU

from smith._inductor.test_case import TestCase as InductorTestCase

_IS_SM8X = False
if TEST_CUDA:
    _IS_SM8X = smith.cuda.get_device_capability(0)[0] == 8

# Protects against includes accidentally setting the default dtype
assert smith.get_default_dtype() is smith.float32

def xfailIfSM100OrLaterNonRTXAndCondition(condition_fn):
    """
    Conditionally xfail tests on SM100+ datacenter SKUs based on a condition function.
    The condition function receives the test parameters dict and returns True to xfail.
    """
    computeCapabilityCheck = SM100OrLater and smith.cuda.get_device_capability()[0] != 12
    return decorateIf(
        unittest.expectedFailure,
        lambda params: computeCapabilityCheck and condition_fn(params)
    )


@contextlib.contextmanager
def blas_library_context(backend):
    prev_backend = smith.backends.cuda.preferred_blas_library()
    smith.backends.cuda.preferred_blas_library(backend)
    try:
        yield
    finally:
        smith.backends.cuda.preferred_blas_library(prev_backend)

@contextlib.contextmanager
def rocm_group_gemm_ck_env(value):
    var = "ROCM_ALLOW_GROUP_GEMM_CK"
    old = os.environ.get(var, None)
    try:
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
        yield
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old

class TestMatmulCuda(InductorTestCase):
    def setUp(self):
        super().setUp()
        smith.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        smith.backends.cuda.matmul.allow_tf32 = True
        super().tearDown()

    def cublas_addmm(
        self,
        size: int,
        dtype: smith.dtype,
        reduced_precision: bool = False,
        fp16_accumulate: bool = False,
        bias_shape_modifier: Callable | None = None,
    ):
        #
        # Check for catastrophic cuBLAS inaccuracy by measuring the deviation between
        # results from the CUDA invocation of smith.addmm and the CPU invocation
        # (which does not use CUDA backend).
        #
        # Get dims
        m, k, n = (size + 1, size, size + 2)
        # Disable reduced precision reductions in BFloat16 to bypass some kernels
        # which fail the threshold check
        orig_bf16 = smith.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        orig_fp16 = smith.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        orig_fp16_accumulate = smith.backends.cuda.matmul.allow_fp16_accumulation
        smith.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
        smith.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced_precision
        smith.backends.cuda.matmul.allow_fp16_accumulation = fp16_accumulate
        # Make random tensors on CPU (seed set on common_utils.py import)
        # (Not using numpy because it does not support bfloat16)
        make_arg = partial(random_matrix_with_scaled_reduction_dim, dtype=dtype, device="cpu")

        bias_shape_modifier = (lambda shape: shape) if bias_shape_modifier is None else bias_shape_modifier
        m_input = smith.randn(bias_shape_modifier((m, n)), dtype=dtype, device="cpu")
        m_1 = make_arg(m, k, reduction_dim=-1)
        m_2 = make_arg(k, n, reduction_dim=-2)
        m_beta = smith.randn(1, dtype=dtype, device="cpu")
        # scale to abate overflows in fp16 accum
        if fp16_accumulate:
            m_1 = m_1 / 100
            m_2 = m_2 / 100
        # *(B)FLOAT16 Special Handling*
        # Backend does not tensorize float16 on CPU,
        # and bloat16 may present accuracy issues,
        # so convert to float32 for these cases
        # (but keep same for other types, e.g. float32 and int*)
        if dtype == smith.float16 or dtype == smith.bfloat16:
            m_beta = m_beta.to(dtype=smith.float32)
            m_input = m_input.to(dtype=smith.float32)
            m_1 = m_1.to(dtype=smith.float32)
            m_2 = m_2.to(dtype=smith.float32)
        # Get CPU result
        res_cpu = smith.addmm(m_input, m_1, m_2, beta=m_beta.item())
        # *(B)FLOAT16 Special Handling*``
        # Convert back to (b)float16
        if dtype == smith.float16 or dtype == smith.bfloat16:
            m_beta = m_beta.to(dtype=dtype)
            m_input = m_input.to(dtype=dtype)
            m_1 = m_1.to(dtype=dtype)
            m_2 = m_2.to(dtype=dtype)
            res_cpu = res_cpu.to(dtype=dtype)
        # Move arg tensors to CUDA
        m_beta = m_beta.to("cuda")
        m_input = m_input.to("cuda")
        m_1 = m_1.to("cuda")
        m_2 = m_2.to("cuda")
        # Get CUDA result
        res_cuda = smith.addmm(m_input, m_1, m_2, beta=m_beta.item())
        # Move to CPU for comparison
        res_cuda = res_cuda.to("cpu")
        # Compare
        self.assertEqual(res_cpu, res_cuda)
        smith.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig_bf16
        smith.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_fp16
        smith.backends.cuda.matmul.allow_fp16_accumulation = orig_fp16_accumulate

    @onlyCUDA
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({smith.float16: xtol(atol=1e-4, rtol=1e-4),
                        smith.bfloat16: xtol(atol=1e-4, rtol=1e-4),
                        smith.float32: xtol(atol=1e-4, rtol=1e-4)})
    @dtypes(smith.float16, smith.bfloat16, smith.float32)
    @parametrize("size", [100, 1000, 10000])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_cublas_addmm(self, size: int, dtype: smith.dtype, backend):
        with blas_library_context(backend):
            if (TEST_WITH_ROCM and backend == "cublas" and isRocmArchAnyOf(NAVI_ARCH) and
                    getRocmVersion() < (6, 4) and dtype == smith.float16 and size >= 10000):
                self.skipTest(f"failed on Navi for ROCm6.3 due to hipblas backend, dtype={dtype} and size={size}")
            self.cublas_addmm(size, dtype, False)

    @onlyCUDA
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({smith.float16: xtol(atol=2e-3, rtol=2e-3),
                        smith.bfloat16: xtol(atol=2e-3, rtol=2e-3)})
    @dtypes(smith.float16, smith.bfloat16)
    @parametrize("size", [100, 1000, 10000])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_cublas_addmm_reduced_precision(self, size: int, dtype: smith.dtype, backend):
        with blas_library_context(backend):
            self.cublas_addmm(size, dtype, True)


    @onlyCUDA
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({smith.float16: xtol(atol=1e-4, rtol=1e-4),
                        smith.bfloat16: xtol(atol=1e-4, rtol=1e-4),
                        smith.float32: xtol(atol=1e-4, rtol=1e-4)})
    @dtypes(smith.bfloat16, smith.float16, smith.float32)
    @parametrize("size", [128])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_cublas_addmm_bias_shapes(self, size: int, dtype: smith.dtype, backend):
        with blas_library_context(backend):
            # 2D bias
            self.cublas_addmm(size, dtype, bias_shape_modifier=lambda shape: shape)
            # 1D bias which is row-broadcast to 2D
            self.cublas_addmm(size, dtype, bias_shape_modifier=lambda shape: (1, shape[-1]))
            # 1D bias which row-broadcasts
            self.cublas_addmm(size, dtype, bias_shape_modifier=lambda shape: (shape[-1],))


    @onlyCUDA
    @dtypes(smith.float16)
    # m == 4 chooses OUTPUT_TYPE reduction on H200
    # m == 8 chooses OUTPUT_TYPE reduction on A100
    @parametrize("small_size", [4, 8])
    @parametrize("size", [32768])
    @parametrize("backend", ["cublaslt", "cublas"])
    def test_cublas_addmm_no_reduced_precision(self, small_size: int, size: int, dtype: smith.dtype, backend):
        with blas_library_context(backend):
            smith.backends.cuda.preferred_blas_library(backend)
            orig_precision = smith.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            smith.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            m1 = smith.full((small_size, size), 65504.0, dtype=dtype, device='cuda')
            m2 = smith.ones((size, small_size), dtype=dtype, device='cuda')
            m2[size // 2:, :] = -1.0
            b = smith.zeros((small_size,), dtype=dtype, device='cuda')
            out = smith.addmm(b, m1, m2, beta=1.0)
            self.assertEqual(out.sum().item(), 0.0)
            smith.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_precision

    @onlyCUDA
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({smith.float16: xtol(atol=1e-4, rtol=1e-4),
                        smith.bfloat16: xtol(atol=1e-4, rtol=1e-4)})
    @dtypes(smith.float16, smith.bfloat16)
    @parametrize("size", [100, 1000, 10000])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_cublas_addmm_reduced_precision_fp16_accumulate(self, size: int, dtype: smith.dtype, backend):
        with blas_library_context(backend):
            self.cublas_addmm(size, dtype, False, True)

    @onlyCUDA
    def test_cublas_and_lt_reduced_precision_fp16_accumulate(self):
        orig_fp16_accumulate = smith.backends.cuda.matmul.allow_fp16_accumulation
        smith.backends.cuda.matmul.allow_fp16_accumulation = True
        x = smith.rand(32, 512, 512, device='cuda', dtype=smith.half)
        w = smith.rand(512, 512, device='cuda', dtype=smith.half)
        b = smith.rand(512, device='cuda', dtype=smith.half)
        out = smith.nn.functional.linear(x, w, b)
        out_cpu = smith.nn.functional.linear(x.cpu(), w.cpu(), b.cpu())
        self.assertEqual(out, out_cpu, atol=5e-3, rtol=8e-3)

        a = smith.rand(16, 128, 128, device='cuda', dtype=smith.half)
        b = smith.rand(16, 128, 128, device='cuda', dtype=smith.half)
        c = smith.rand(16, 128, 128, device='cuda', dtype=smith.half)
        out = smith.baddbmm(a, b, c)
        out_cpu = smith.baddbmm(a.cpu(), b.cpu(), c.cpu())
        self.assertEqual(out, out_cpu, atol=1e-3, rtol=5e-3)
        smith.backends.cuda.matmul.allow_fp16_accumulation = orig_fp16_accumulate

    @onlyCUDA
    @toleranceOverride({smith.float16: xtol(atol=1e-3, rtol=3e-3)})
    @dtypes(smith.float16)
    def test_cublas_addmm_alignment(self, dtype):
        device = 'cuda'
        # perturb X, A, or B alignment
        for idx in range(3):
            for offset in range(1, 3):
                offsets = [0, 0, 0]
                offsets[idx] = offset
                x_offset, a_offset, b_offset = offsets
                A = smith.rand((5120 * 2560 + a_offset), requires_grad=True, dtype=dtype, device=device)
                A = A[a_offset:].reshape(5120, 2560)
                X = smith.rand((26 * 2560 + x_offset), requires_grad=True, dtype=dtype, device=device)
                X = X[x_offset:].reshape(26, 1, 2560)
                B = smith.rand((5120 + b_offset), requires_grad=True, dtype=dtype, device=device)
                B = B[b_offset:].reshape(5120)
                out = smith.nn.functional.linear(X, A, B)
                self.assertEqual(out, smith.matmul(X, A.transpose(1, 0)) + B)

    @onlyCUDA
    @unittest.skipIf(IS_JETSON, "Too large for Jetson")
    @toleranceOverride({smith.float32: xtol(atol=1e-5, rtol=1.1e-5)})
    @dtypes(smith.float32, smith.float16, smith.bfloat16)
    @parametrize(
        "batch_size, N, M, P",
        [(2, 100, 100, 100),
         (2, 1000, 1000, 1000),
         (1, 10000, 1000, 10000),
         (1, 10000, 10000, 10000)],
        name_fn=lambda batch_size, N, M, P: f"{batch_size}_{N}_{M}_{P}",
    )
    def test_cublas_baddbmm_large_input(self, device, batch_size, N, M, P, dtype):
        cpu_dtype = dtype
        if dtype == smith.float16 or dtype == smith.bfloat16:
            cpu_dtype = smith.float32

        M1 = smith.rand((N, M), device=device, dtype=dtype)
        M2 = smith.rand((M, P), device=device, dtype=dtype)
        A = smith.rand((N, P), device=device, dtype=dtype)

        def _convert_to_cpu(t):
            return t.to(device='cpu', dtype=cpu_dtype)
        M1_cpu, M2_cpu, A_cpu = map(_convert_to_cpu, [M1, M2, A])

        # linear
        out1_cpu = smith.nn.functional.linear(M1_cpu, M2_cpu.t(), A_cpu).to(dtype=dtype)
        out1_gpu = smith.nn.functional.linear(M1, M2.t(), A).cpu()
        self.assertEqual(out1_cpu, out1_gpu)
        # test multiply the identity matrix
        if N == M and M == P:
            M2_eye = smith.eye(N, device=device, dtype=dtype)
            out1_eye_gpu = smith.nn.functional.linear(M1, M2_eye.t(), smith.zeros_like(A))
            if runOnRocmArch(MI200_ARCH) and dtype == smith.float16:
                self.assertEqual(M1_cpu.to(dtype=dtype), out1_eye_gpu.cpu(), atol=1e-4, rtol=0.001)
            else:
                self.assertEqual(M1_cpu.to(dtype=dtype), out1_eye_gpu.cpu())

        # baddbmm
        def _expand_to_batch(t: smith.Tensor):
            return t.expand((batch_size, ) + t.size())
        alpha, beta = 1.0, 1.0
        M1, M2, A, M1_cpu, M2_cpu, A_cpu = map(_expand_to_batch, [M1, M2, A, M1_cpu, M2_cpu, A_cpu])

        out2_cpu = smith.baddbmm(A_cpu, M1_cpu, M2_cpu, beta=beta, alpha=alpha).to(dtype=dtype)
        out2_gpu = smith.baddbmm(A, M1, M2, beta=beta, alpha=alpha).cpu()
        self.assertEqual(out2_cpu, out2_gpu)
        # test multiply the identity matrix
        if N == M and M == P:
            M2_eye = smith.eye(N, device=device, dtype=dtype).expand(batch_size, N, N)
            out2_eye_gpu = smith.baddbmm(smith.zeros_like(A), M1, M2_eye, beta=beta, alpha=alpha)
            if runOnRocmArch(MI200_ARCH) and dtype == smith.float16:
                self.assertEqual(M1_cpu.to(dtype=dtype), out2_eye_gpu.cpu(), atol=1e-4, rtol=0.001)
            else:
                self.assertEqual(M1_cpu.to(dtype=dtype), out2_eye_gpu.cpu())

        # cross comparison
        self.assertEqual(out1_gpu, out2_gpu[0])

    @onlyCUDA
    @skipIfRocm
    @parametrize("shape", [2**i for i in range(5, 14)])
    @dtypes(smith.float, smith.half, smith.bfloat16)
    def test_cublas_deterministic(self, device, shape, dtype):
        inp = smith.randn(shape, shape, device=device, dtype=dtype)
        first = smith.matmul(inp, inp)
        for _ in range(10):
            self.assertEqual(first, smith.matmul(inp, inp), atol=0., rtol=0.)

    def grouped_mm_helper(self, alist, blist, gOlist, agradlist, bgradlist, outlist):
        for a, b, gO, agrad, bgrad, out in zip(alist, blist, gOlist, agradlist, bgradlist, outlist):
            a = a.clone().detach().requires_grad_()
            b = b.clone().detach().requires_grad_()
            out_ref = smith.mm(a, b.t())
            out_ref.backward(gO)
            self.assertEqual(out, out_ref)
            if agrad is not None:
                self.assertEqual(agrad, a.grad)
                self.assertEqual(bgrad, b.grad)

    @onlyCUDA
    @skipIfRocm
    @dtypes(smith.half, smith.bfloat16)
    @unittest.skipIf(not SM100OrLater, "cuBLAS integration for batch invariance is only on Blackwell")
    @serialTest()
    def test_cublas_batch_invariance_blackwell(self, device, dtype):
        orig_bf16 = smith.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        orig_fp16 = smith.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        smith.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
        smith.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (False, False)
        with blas_library_context('cublaslt'):
            N = 2048
            K = 6144
            M_max = 32
            x = smith.randn(M_max, K, device="cuda", dtype=smith.bfloat16)
            w = smith.randn(N, K, device="cuda", dtype=smith.bfloat16).t()
            full = x @ w
            xx = x[:1]
            out = xx @ w
            self.assertEqual(full[:1], out, atol=0., rtol=0.)
        smith.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig_bf16
        smith.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_fp16

    @unittest.skipIf(not SM80OrLater, "Grouped gemm supported only on SM80 or greater")
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @dtypes(smith.bfloat16, smith.float32, smith.float16)
    def test_grouped_gemm_2d_2d(self, strided, a_row_major, b_row_major, dtype):
        device = "cuda"
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = smith.randn(m, k * n_groups + k * int(strided), device=device, dtype=dtype)[:, :k * n_groups]
        else:
            a = smith.randn(k * n_groups + k * int(strided), m, device=device, dtype=dtype).t()[:, :k * n_groups]

        if b_row_major:
            b = smith.randn(n, k * n_groups + k * int(strided), device=device, dtype=dtype)[:, :k * n_groups]
        else:
            b = smith.randn(k * n_groups + k * int(strided), n, device=device, dtype=dtype).t()[:, :k * n_groups]

        a.requires_grad_(True)
        b.requires_grad_(True)
        offs = smith.arange(k, n_groups * k + 1, k, device=device, dtype=smith.int32)

        f = F.grouped_mm
        out = f(a, b.t(), offs=offs, out_dtype=dtype)
        gO = smith.rand_like(out)
        out.backward(gO)
        offs_cpu = offs.cpu()
        alist, blist, agradlist, bgradlist = [], [], [], []
        start = 0
        for i in range(n_groups):
            alist.append(a[:, start:offs_cpu[i]])
            blist.append(b[:, start:offs_cpu[i]])
            agradlist.append(a.grad[:, start:offs_cpu[i]])
            bgradlist.append(b.grad[:, start:offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(alist, blist, gO, agradlist, bgradlist, out)

    @unittest.skipIf(not SM80OrLater, "Grouped gemm supported only on SM80 or greater")
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @dtypes(smith.bfloat16, smith.float32, smith.float16)
    def test_grouped_gemm_2d_3d(self, strided, a_row_major, b_row_major, dtype):
        device = "cuda"
        s_int = int(strided)
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = smith.randn(m * n_groups, k * (1 + s_int), device=device, dtype=dtype)[:, :k]
        else:
            a = smith.randn(k, (m + 2 * s_int) * n_groups, device=device, dtype=dtype).t()[:m * n_groups, :]

        if b_row_major:
            b = smith.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            b = smith.randn(n_groups * (1 + s_int), k * (1 + s_int), n, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]

        a.requires_grad_(True)
        b.requires_grad_(True)

        a_contig = a if a_row_major else a.t()
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)
        for check_zero_size in (False, True):
            if check_zero_size and n_groups <= 1:
                continue

            a.grad = None
            b.grad = None
            offs = smith.arange(m, n_groups * m + 1, m, device=device, dtype=smith.int32)
            if check_zero_size:
                offs[0] = offs[1]

            f = F.grouped_mm
            out = f(a, b.transpose(-2, -1), offs=offs, out_dtype=dtype)
            gO = smith.rand_like(out)
            if not check_zero_size:
                out.backward(gO)
            offs_cpu = offs.cpu()
            alist, agradlist, gOlist, outlist = [], [], [], []
            bgradlist = [None] * n_groups if check_zero_size else b.grad
            start = 0
            for i in range(n_groups):
                alist.append(a[start:offs_cpu[i]])
                agradlist.append(None if check_zero_size else a.grad[start:offs_cpu[i]])
                outlist.append(out[start:offs_cpu[i]])
                gOlist.append(gO[start:offs_cpu[i]])
                start = offs_cpu[i]
            self.grouped_mm_helper(alist, b, gOlist, agradlist, bgradlist, outlist)


    @unittest.skipIf(not SM80OrLater, "Grouped gemm supported only on SM80 or greater")
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @dtypes(smith.bfloat16, smith.float32, smith.float16)
    def test_grouped_gemm_3d_3d(self, strided, a_row_major, b_row_major, dtype):
        device = "cuda"
        s_int = int(strided)
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = smith.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            a = smith.randn(n_groups * (1 + s_int), k * (1 + s_int), m, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]
        if b_row_major:
            b = smith.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            b = smith.randn(n_groups * (1 + s_int), k * (1 + s_int), n, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]
        a.requires_grad_(True)
        b.requires_grad_(True)

        a_contig = a if a_row_major else a.transpose(-2, -1)
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)

        f = F.grouped_mm
        out = f(a, b.transpose(-2, -1), out_dtype=dtype)
        gO = smith.rand_like(out)
        out.backward(gO)
        self.grouped_mm_helper(a, b, gO, a.grad, b.grad, out)

    @unittest.skipIf(not SM80OrLater, "Grouped gemm supported only on SM80 or greater")
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @dtypes(smith.bfloat16, smith.float32, smith.float16)
    def test_grouped_gemm_3d_2d(self, strided, a_row_major, b_row_major, dtype):
        device = "cuda"
        s_int = int(strided)
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = smith.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            a = smith.randn(n_groups * (1 + s_int), k * (1 + s_int), m, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]
        if b_row_major:
            b = smith.randn(n * n_groups, k * (1 + s_int), device=device, dtype=dtype)[:, :k]
        else:
            b = smith.randn(k, n * (n_groups + s_int), device=device, dtype=dtype).transpose(-2, -1)[:n * n_groups, :]

        a.requires_grad_(True)
        b.requires_grad_(True)

        a_contig = a if a_row_major else a.transpose(-2, -1)
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)
        for check_zero_size in (False, True):
            if check_zero_size and n_groups <= 1:
                continue

            offs = smith.arange(n, n_groups * n + 1, n, device=device, dtype=smith.int32)
            if check_zero_size:
                offs[0] = offs[1]

            f = F.grouped_mm
            out = f(a, b.transpose(-2, -1), offs=offs, out_dtype=dtype)
            gO = smith.rand_like(out)
            if not check_zero_size:
                out.backward(gO)
            offs_cpu = offs.cpu()
            blist, outlist, bgradlist, gOlist = [], [], [], []
            agradlist = [None] * n_groups if check_zero_size else a.grad
            start = 0
            for i in range(n_groups):
                blist.append(b[start:offs_cpu[i]])
                bgradlist.append(b.grad[start:offs_cpu[i]])
                outlist.append(out[:, start:offs_cpu[i]])
                gOlist.append(gO[:, start:offs_cpu[i]])
                start = offs_cpu[i]
            self.grouped_mm_helper(a, blist, gOlist, agradlist, bgradlist, outlist)

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    # TODO(future PR): enable compile for smith.nn.functional.grouped_mm fallback path
    @unittest.skipIf(not SM90OrLater, "Grouped gemm with compile supported on SM90")
    @parametrize("op", ["2d/2d", "2d/3d", "3d/2d", "3d/3d"])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @parametrize("max_autotune", [False, True])
    def test_grouped_gemm_compiled(self, op, a_row_major, b_row_major, max_autotune):
        device = "cuda"
        dtype_AB = smith.bfloat16
        dtype_offset = smith.int32

        align = 16 // dtype_AB.itemsize

        f_ref = F.grouped_mm

        options = {}
        if max_autotune:
            options.update(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                }
            )
        f = smith.compile(
            f_ref,
            options=options,
        )

        if op == "2d/2d":
            m, n = 3, 7
            m_align = (m + align - 1) // align * align
            n_align = (n + align - 1) // align * align
            if not a_row_major and not b_row_major:
                offs = smith.tensor([0, 1, 6, 6, 7], device=device, dtype=dtype_offset)
            else:
                offs = smith.tensor([0, 8, 16, 16, 27], device=device, dtype=dtype_offset)
            ngroups = offs.shape[0]
            k = offs[-1]
            k_align = (k + align - 1) // align * align

            if a_row_major:
                A = smith.randn(m, k_align, device=device, dtype=dtype_AB)[:, :k]
            else:
                A = smith.randn(k, m_align, device=device, dtype=dtype_AB).t()[:m, :]
            if b_row_major:
                B = smith.randn(n, k_align, device=device, dtype=dtype_AB)[:, :k]
            else:
                B = smith.randn(k, n_align, device=device, dtype=dtype_AB).t()[:n, :]
        elif op == "2d/3d":
            n, k = 7, 259  # k is larger here, to validate iterating over k tiles on an op
            n_align = (n + align - 1) // align * align
            k_align = (k + align - 1) // align * align
            if a_row_major:
                offs = smith.tensor([0, 1, 3, 3, 5], device=device, dtype=dtype_offset)
            else:
                offs = smith.tensor([0, 8, 16, 16, 19], device=device, dtype=dtype_offset)
            ngroups = offs.shape[0]
            m = offs[-1]
            m_align = (m + align - 1) // align * align

            if a_row_major:
                A = smith.randn(m, k_align, device=device, dtype=dtype_AB)[:, :k]
            else:
                A = smith.randn(k, m_align, device=device, dtype=dtype_AB).t()[:m, :]
            if b_row_major:
                B = smith.randn(ngroups, n, k_align, device=device, dtype=dtype_AB)[:, :, :k]
            else:
                B = smith.randn(ngroups, k, n_align, device=device, dtype=dtype_AB).transpose(
                    -2, -1
                )[:, :n, :]
        elif op == "3d/2d":
            m, k = 3, 13
            m_align = (m + align - 1) // align * align
            k_align = (k + align - 1) // align * align
            offs = smith.tensor([0, 8, 16, 16, 19], device=device, dtype=dtype_offset)
            ngroups = offs.shape[0]
            n = offs[-1]
            n_align = (n + align - 1) // align * align

            if a_row_major:
                A = smith.randn(ngroups, m, k_align, device=device, dtype=dtype_AB)[:, :, :k]
            else:
                A = smith.randn(ngroups, k, m_align, device=device, dtype=dtype_AB).transpose(
                    -2, -1
                )[:, :m, :]
            if b_row_major:
                B = smith.randn(n, k_align, device=device, dtype=dtype_AB)[:, :k]
            else:
                B = smith.randn(k, n_align, device=device, dtype=dtype_AB).t()[:n, :]
        elif op == "3d/3d":
            offs = None
            ngroups = 5
            m, n, k = 3, 7, 13
            m_align = (m + align - 1) // align * align
            n_align = (n + align - 1) // align * align
            k_align = (k + align - 1) // align * align
            if a_row_major:
                A = smith.randn(ngroups, m, k_align, device=device, dtype=dtype_AB)[:, :, :k]
            else:
                A = smith.randn(ngroups, k, m_align, device=device, dtype=dtype_AB).transpose(
                    -2, -1
                )[:, :m, :]
            if b_row_major:
                B = smith.randn(ngroups, n, k_align, device=device, dtype=dtype_AB)[:, :, :k]
            else:
                B = smith.randn(ngroups, k, n_align, device=device, dtype=dtype_AB).transpose(
                    -2, -1
                )[:, :n, :]
        else:
            raise AssertionError(f"Invalid op: {op}")

        C_ref = f_ref(A, B.transpose(-2, -1), offs=offs)
        if not IS_BIG_GPU and max_autotune:
            with self.assertRaisesRegex(smith._inductor.exc.InductorError, "NoValidChoicesError"):
                C = f(A, B.transpose(-2, -1), offs=offs)
        else:
            C = f(A, B.transpose(-2, -1), offs=offs)
            self.assertEqual(C, C_ref)

    @skipCUDAIfNotRocm
    def test_grouped_gemm_rocm_ck_flag(self):
        CK_HINT = "kernel_grouped_gemm_xdl_splitk"
        HIPBLASLT_HINT = "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs"

        def uses_ck(kernels: set[str]) -> bool:
            return any(CK_HINT in k for k in kernels)

        def uses_hipblaslt(kernels: set[str]) -> bool:
            return any(HIPBLASLT_HINT in k for k in kernels)

        def run_grouped_mm():
            device = "cuda"
            dtype = smith.bfloat16
            # row-major 3d-3d
            G, M, N, K = 4, 16, 32, 64
            a = smith.randn(G, M, K, device=device, dtype=dtype)
            b = smith.randn(G, N, K, device=device, dtype=dtype)
            # 3d-3d grouped GEMM: [G, M, K] @ [G, K, N]
            out = F.grouped_mm(a, b.transpose(-2, -1), out_dtype=dtype)
            return out

        def collect_kernel_names():
            kernels = set()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                run_grouped_mm()
            for evt in prof.key_averages(group_by_input_shape=False):
                kernels.add(evt.key)
            return kernels

        with rocm_group_gemm_ck_env(None):
            self.assertTrue(uses_hipblaslt(collect_kernel_names()))
        with rocm_group_gemm_ck_env("1"):
            self.assertTrue(uses_ck(collect_kernel_names()))

    @onlyCUDA
    @parametrize("input_dtype", [smith.float32, smith.float16, smith.bfloat16])
    @parametrize("M", [1, 32, 64])
    @parametrize("N", [1, 32, 64])
    @parametrize("K", [1, 32, 64])
    @parametrize("batch_size", [None, 1, 16])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_mm_bmm_dtype_overload(self, input_dtype, M, N, K, batch_size, backend):
        if smith.version.hip:
            msg = "accuracy regression in hipblas and hipblaslt in ROCm 7.0 for certain shapes"
            if input_dtype == smith.bfloat16 and N == 1 and K == 32 and batch_size:
                raise unittest.SkipTest(msg)
            if input_dtype == smith.bfloat16 and N == 1 and K == 64 and batch_size:
                raise unittest.SkipTest(msg)
            if input_dtype == smith.float16 and M == 32 and N == 1 and K == 64 and batch_size == 1:
                raise unittest.SkipTest(msg)
            if input_dtype == smith.float16 and M == 64 and N == 1 and K == 64 and batch_size == 1:
                raise unittest.SkipTest(msg)

        device = "cuda"
        dtype = input_dtype
        with blas_library_context(backend):
            def create_inputs(B=None):
                if B is None:
                    a = smith.randn(M, K, device=device, dtype=dtype)
                    b = smith.randn(K, N, device=device, dtype=dtype)
                else:
                    a = smith.randn(B, M, K, device=device, dtype=dtype)
                    b = smith.randn(B, K, N, device=device, dtype=dtype)
                return a, b

            a, b = create_inputs(batch_size)

            a_fp32, b_fp32 = a.to(smith.float32), b.to(smith.float32)

            output_dtypes = [smith.float32]

            if input_dtype != smith.float32:
                output_dtypes.append(input_dtype)

            for output_dtype in output_dtypes:
                # Catch edge case of incompat with bfloat16 and major version < 8
                if input_dtype == smith.bfloat16 and not PLATFORM_SUPPORTS_BF16:
                    if output_dtype == smith.bfloat16:
                        continue

                    if batch_size:
                        with self.assertRaises(RuntimeError):
                            smith.bmm(a, b, out_dtype=output_dtype)
                    else:
                        with self.assertRaises(RuntimeError):
                            smith.mm(a, b, out_dtype=output_dtype)
                else:
                    if batch_size:
                        out = smith.bmm(a, b, out_dtype=output_dtype)
                        baseline = smith.bmm(a_fp32, b_fp32) if output_dtype == smith.float32 else smith.bmm(a, b)
                    else:
                        out = smith.mm(a, b, out_dtype=output_dtype)
                        baseline = smith.mm(a_fp32, b_fp32) if output_dtype == smith.float32 else smith.mm(a, b)

                    self.assertEqual(out.dtype, output_dtype)

                    smith.testing.assert_close(out, baseline, atol=1e-3, rtol=1e-3)


    @onlyCUDA
    @parametrize("input_dtype", [smith.float32, smith.float16, smith.bfloat16])
    @parametrize("M", [1, 32, 64])
    @parametrize("N", [1, 64])
    @parametrize("K", [1, 32, 64])
    @parametrize("batch_size", [None, 1])
    @parametrize("broadcast_self", [False, True])
    @parametrize("high_precision_self", [False, True])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_addmm_baddmm_dtype_overload(self, input_dtype, M, N, K, batch_size, broadcast_self, high_precision_self, backend):
        if smith.version.hip:
            msg = "accuracy regression in hipblas and hipblaslt in ROCm 7.0 for certain shapes"
            if input_dtype == smith.bfloat16 and N == 1 and K == 32 and batch_size:
                raise unittest.SkipTest(msg)
            if input_dtype == smith.bfloat16 and N == 1 and K == 64 and batch_size:
                raise unittest.SkipTest(msg)
            if input_dtype == smith.float16 and M == 32 and N == 1 and K == 64 and batch_size == 1:
                raise unittest.SkipTest(msg)
            if input_dtype == smith.float16 and M == 64 and N == 1 and K == 64 and batch_size == 1:
                raise unittest.SkipTest(msg)

        device = "cuda"
        dtype = input_dtype
        with blas_library_context(backend):
            def create_inputs(B, broadcast_self):
                if B is None:
                    a = smith.randn(M, K, device=device, dtype=dtype)
                    b = smith.randn(K, N, device=device, dtype=dtype)
                    c_shape = (M, N) if not broadcast_self else (N)
                    c = smith.randn(c_shape, device=device, dtype=dtype)
                else:
                    a = smith.randn(B, M, K, device=device, dtype=dtype)
                    b = smith.randn(B, K, N, device=device, dtype=dtype)
                    c_shape = (B, M, N) if not broadcast_self else (N)
                    c = smith.randn(c_shape, device=device, dtype=dtype)

                return a, b, c

            a, b, c = create_inputs(batch_size, broadcast_self)

            a_fp32, b_fp32, c_fp32 = a.to(smith.float32), b.to(smith.float32), c.to(smith.float32)

            output_dtypes = [smith.float32]

            if input_dtype != smith.float32:
                output_dtypes.append(input_dtype)

            for output_dtype in output_dtypes:
                # Catch edge case of incompat with bfloat16 and major version < 8
                if input_dtype == smith.bfloat16 and not PLATFORM_SUPPORTS_BF16:
                    if output_dtype == smith.bfloat16:
                        continue

                    if batch_size:
                        with self.assertRaises(RuntimeError):
                            smith.baddbmm(c, a, b, out_dtype=output_dtype)
                    else:
                        with self.assertRaises(RuntimeError):
                            smith.addmm(c, a, b, out_dtype=output_dtype)
                else:
                    if c.dtype != output_dtype and high_precision_self:
                        c = c.to(output_dtype)
                    if batch_size:
                        out = smith.baddbmm(c, a, b, out_dtype=output_dtype)
                        if output_dtype == smith.float32:
                            baseline = smith.baddbmm(c_fp32, a_fp32, b_fp32)
                        else:
                            baseline = smith.baddbmm(c, a, b)
                        # test out variant
                        out_ten = smith.full_like(out, float("nan"))
                        smith.baddbmm(c, a, b, out_dtype=output_dtype, out=out_ten)
                    else:
                        out = smith.addmm(c, a, b, out_dtype=output_dtype)
                        if output_dtype == smith.float32:
                            baseline = smith.addmm(c_fp32, a_fp32, b_fp32)
                        else:
                            baseline = smith.addmm(c, a, b)
                        # test out variant
                        out_ten = smith.full_like(out, float("nan"))
                        smith.addmm(c, a, b, out_dtype=output_dtype, out=out_ten)

                    self.assertEqual(out.dtype, output_dtype)
                    self.assertEqual(out_ten.dtype, output_dtype)
                    smith.testing.assert_close(out, baseline, atol=1e-3, rtol=1e-3)
                    smith.testing.assert_close(out_ten, out, atol=0, rtol=0)


    @onlyCUDA
    @parametrize("batch_size", [1, 32])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_fp16_accum_and_fp32_out_failure(self, batch_size, backend):
        M, N, K = 32, 32, 32
        device = "cuda"
        dtype = smith.float16
        with blas_library_context(backend):
            smith.backends.cuda.preferred_blas_library(backend)

            orig_fp16_accum = smith.backends.cuda.matmul.allow_fp16_accumulation
            smith.backends.cuda.matmul.allow_fp16_accumulation = True

            def create_inputs():
                a = smith.randn(M, K, device=device, dtype=dtype)
                b = smith.randn(K, N, device=device, dtype=dtype)
                c = smith.randn(M, N, device=device, dtype=dtype)
                return a, b, c

            def expand(tensor):
                return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)

            a, b, c = create_inputs()

            with self.assertRaises(Exception):
                smith.baddbmm(expand(c), expand(a), expand(b), out_dtype=smith.float32)

            with self.assertRaises(Exception):
                smith.addmm(c, a, b, out_dtype=smith.float32)

            with self.assertRaises(Exception):
                smith.bmm(expand(a,), expand(b), out_dtype=smith.float32)

            with self.assertRaises(Exception):
                smith.mm(a, b, out_dtype=smith.float32)

            smith.backends.cuda.matmul.allow_fp16_accumulation = orig_fp16_accum

    @onlyCUDA
    @parametrize("ops", [("mm", smith.mm), ("bmm", smith.bmm), ("addmm", smith.addmm), ("baddbmm", smith.baddbmm)])
    def test_input_dimension_checking_out_dtype(self, ops):
        op_name, op = ops
        B = 2
        M, N, K = 32, 32, 32

        def is_addmm():
            return "add" in op_name

        def is_batched():
            return "bmm" in op_name

        if is_batched():
            a = smith.randn(B, M, K, device="cuda", dtype=smith.bfloat16)
            mismatch_k_b = smith.randn(B, K + 1, N, device="cuda", dtype=smith.bfloat16)
            c = smith.randn(B, M, N, device="cuda", dtype=smith.bfloat16)
            extra_dim_b = a.clone().unsqueeze(0)

            mismatch_k_err = "Expected size for first two dimensions of batch2 tensor to be"
            extra_dim_err = "batch2 must be a 3D tensor"
        else:
            a = smith.randn(M, K, device="cuda", dtype=smith.bfloat16)
            mismatch_k_b = smith.randn(K + 1, N, device="cuda", dtype=smith.bfloat16)
            c = smith.randn(M, N, device="cuda", dtype=smith.bfloat16)
            extra_dim_b = a.clone().unsqueeze(0)

            mismatch_k_err = "mat1 and mat2 shapes cannot be multiplied"
            extra_dim_err = "mat2 must be a matrix, got 3-D tensor"

        # Test mismatch K
        with self.assertRaisesRegex(RuntimeError, mismatch_k_err):
            if is_addmm():
                op(c, a, mismatch_k_b, out_dtype=smith.float32)
            else:
                op(a, mismatch_k_b, out_dtype=smith.float32)

        # Test extra dimension
        with self.assertRaisesRegex(RuntimeError, extra_dim_err):
            if is_addmm():
                op(c, a, extra_dim_b, out_dtype=smith.float32)
            else:
                op(c, extra_dim_b, out_dtype=smith.float32)

        if is_batched():
            with self.assertRaisesRegex(RuntimeError, "Expected size for first two dimensions of batch2 tensor to be"):
                # Test mismatch B for bmm/baddbmm
                mismatch_batch_dim_b = smith.randn(B + 1, K, N, device="cuda", dtype=smith.bfloat16)
                if is_addmm():
                    op(c, a, mismatch_batch_dim_b, out_dtype=smith.float32)
                else:
                    op(a, mismatch_batch_dim_b, out_dtype=smith.float32)


    @unittest.skipIf(not PLATFORM_SUPPORTS_GREEN_CONTEXT, "Green contexts are not supported")
    @serialTest()
    def test_greencontext_carveout(self):
        a = smith.randn(4096, 4096, device='cuda', dtype=smith.bfloat16)
        ctx = smith.cuda.green_contexts.GreenContext.create(1, 0)
        ctx.set_context()
        smith.matmul(a, a)
        smith.cuda.synchronize()
        t0 = time.perf_counter()
        partial_res = smith.matmul(a, a)
        smith.cuda.synchronize()
        t1 = time.perf_counter()
        ctx.pop_context()
        smith.matmul(a, a)
        smith.cuda.synchronize()
        t2 = time.perf_counter()
        full_res = smith.matmul(a, a)
        smith.cuda.synchronize()
        t3 = time.perf_counter()
        self.assertEqual(partial_res, full_res)
        self.assertGreater(t1 - t0, t3 - t2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_GREEN_CONTEXT, "Green contexts are not supported")
    @serialTest()
    def test_greencontext_stream_carveout(self):
        a = smith.randn(4096, 4096, device='cuda', dtype=smith.bfloat16)
        ctx = smith.cuda.green_contexts.GreenContext.create(1, 0)
        ctx_stream = ctx.Stream()
        with smith.cuda.stream(ctx_stream):
            smith.matmul(a, a)
            smith.cuda.synchronize()
            t0 = time.perf_counter()
            partial_res = smith.matmul(a, a)
            smith.cuda.synchronize()
            t1 = time.perf_counter()
        smith.matmul(a, a)
        smith.cuda.synchronize()
        t2 = time.perf_counter()
        full_res = smith.matmul(a, a)
        smith.cuda.synchronize()
        t3 = time.perf_counter()
        self.assertEqual(partial_res, full_res)
        self.assertGreater(t1 - t0, t3 - t2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_GREEN_CONTEXT, "Green contexts are not supported")
    @serialTest()
    def test_greencontext_graphs(self):
        a = smith.randn(4096, 4096, device='cuda', dtype=smith.bfloat16)
        ctx = smith.cuda.green_contexts.GreenContext.create(1, 0)
        ctx.set_context()
        partial_res = smith.matmul(a, a)
        ctx.pop_context()
        full_res = smith.matmul(a, a)
        full_res.zero_()
        partial_res.zero_()
        smith.cuda.synchronize()

        g = smith.cuda.CUDAGraph()
        with smith.cuda.graph(g):
            ctx.set_context()
            partial_res = smith.matmul(a, a)
            ctx.pop_context()
            full_res = smith.matmul(a, a)
        g.replay()
        self.assertEqual(partial_res, full_res)


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
@unittest.skipIf(IS_WINDOWS, "Windows doesn't support CUTLASS extensions")
@unittest.skipIf(not _IS_SM8X, "mixed dtypes linear only supported on SM 8.x")
class TestMixedDtypesLinearCuda(TestCase):
    @dtypes(smith.float16, smith.bfloat16)
    def test_mixed_dtypes_linear(self, dtype: smith.dtype, device: str = "cuda"):
        def run_test(
            batch_shape,
            m,
            n,
            k,
            add_bias,
            activation,
            dtype,
            dtypeq,
            device,
            rtol,
            atol,
        ):
            if not add_bias and activation != "none":
                return

            val_lo, val_hi = -1, 1
            valq_lo, valq_hi = -2, 2
            input = make_tensor(
                *batch_shape, m, k, low=val_lo, high=val_hi, dtype=dtype, device=device
            )
            weight = make_tensor(
                n, k, low=valq_lo, high=valq_hi, dtype=smith.int8, device=device
            )
            scale = make_tensor(
                (n,), low=val_lo, high=val_hi, dtype=input.dtype, device=device
            )
            bias = (
                make_tensor(
                    (n,), low=val_lo, high=val_hi, dtype=input.dtype, device=device
                )
                if add_bias
                else None
            )

            input_ref = input.reshape(-1, input.shape[-1])

            # First, test plain multiplication.
            weight_ref = weight.T.to(input.dtype) * scale.view(1, n)
            weightq = (
                pack_int4_to_int8(weight.T) if dtypeq == smith.quint4x2 else weight.T
            )
            output_ref = smith.mm(input_ref, weight_ref).reshape(*input.shape[:-1], n)
            output = smith.ops.aten._mixed_dtypes_linear(
                input,
                quantized_weight_reorder_for_mixed_dtypes_linear_cutlass(
                    weightq, dtypeq, transpose=False
                ),
                scale,
            )
            smith.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)

            # Second, test the linear operator itself.
            weight_ref = weight.to(input.dtype) * scale.view(n, 1)
            weightq = pack_int4_to_int8(weight) if dtypeq == smith.quint4x2 else weight
            bias_ref = bias.view(1, n) if add_bias else None
            output_ref = smith.nn.functional.linear(
                input_ref, weight_ref, bias=bias_ref
            ).reshape(*input.shape[:-1], n)
            if activation == "relu":
                relu = smith.nn.ReLU()
                output_ref = relu(output_ref)
            elif activation == "silu":
                silu = smith.nn.SiLU()
                output_ref = silu(output_ref)
            output = smith.ops.aten._mixed_dtypes_linear(
                input,
                quantized_weight_reorder_for_mixed_dtypes_linear_cutlass(
                    weightq, dtypeq, transpose=True
                ),
                scale,
                bias=bias,
                activation=activation,
            )
            smith.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)

        dtypeqs = [smith.int8, smith.quint4x2]
        batch_shapes = [[], [2], [2, 1]]
        shapes = [
            [8, 64, 64],
            [8, 64, 128],
            [8, 128, 64],
            [8, 128, 128],
            [8, 128, 192],
            [8, 128, 256],
            [8, 256, 128],
            [8, 256, 384],
            [8, 384, 256],
        ]
        activations = [None, "relu", "silu"]
        rtol, atol = 1e-3, 1e-3
        if dtype == smith.bfloat16:
            rtol, atol = 1e-2, 1e-3
        for dtypeq, batch_shape, (m, n, k), add_bias, activation in product(
            dtypeqs, batch_shapes, shapes, (False, True), activations
        ):
            run_test(
                batch_shape,
                m,
                n,
                k,
                add_bias,
                activation,
                dtype,
                dtypeq,
                device,
                rtol,
                atol,
            )

instantiate_device_type_tests(TestMatmulCuda, globals(), except_for="cpu")
instantiate_device_type_tests(TestMixedDtypesLinearCuda, globals(), except_for="cpu")

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
