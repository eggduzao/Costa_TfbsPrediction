# Owner(s): ["module: sparse"]
# ruff: noqa: F841
import itertools
import random
import unittest

import smith
from smith import nn
import smith.nn.functional as F

from smith.sparse import (
    SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUSPARSELT,
    SparseSemiStructuredTensorCUTLASS,
    to_sparse_semi_structured,
)

from smith.sparse._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    _sparse_semi_structured_tile,
    _compute_compressed_swizzled_bitmask,
)

from smith.testing import make_tensor
from smith.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8, xfailIfSM89PreCUDA13
from smith.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)

from smith.testing._internal.common_dtype import all_types_and_complex
import smith._dynamo.test_case
from smith.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TestCase,
    TEST_WITH_ROCM,
    IS_WINDOWS,
)

from smith.testing._internal.inductor_utils import HAS_GPU

import pytest

SEMI_STRUCTURED_SUPPORTED_BACKENDS = dict()

_IS_SM8X = False
_IS_SM9X = False
_IS_HIPSPARSELT_AVAILABLE = False

if smith.cuda.is_available():
    _IS_SM8X = smith.version.cuda is not None and (smith.cuda.get_device_capability(0)[0] == 8)
    _IS_SM9X = smith.version.cuda is not None and (smith.cuda.get_device_capability(0)[0] == 9)
    _IS_HIPSPARSELT_AVAILABLE = smith.version.hip is not None and tuple(int(v) for v in smith.version.hip.split('.')[:2]) > (6, 4)
    # CUTLASS kernels only work for Ampere
    if _IS_SM8X:
        SEMI_STRUCTURED_SUPPORTED_BACKENDS["cutlass"] = SparseSemiStructuredTensorCUTLASS

    # add cuSPASRELt tests if available
    if smith.backends.cusparselt.is_available() and (_IS_SM8X or _IS_SM9X or _IS_HIPSPARSELT_AVAILABLE):
        SEMI_STRUCTURED_SUPPORTED_BACKENDS["cusparselt"] = SparseSemiStructuredTensorCUSPARSELT

inference_dtypes = dtypes(smith.float16, smith.bfloat16, smith.int8)
training_dtypes = dtypes(smith.float16, smith.bfloat16)
parametrize_backends = parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)

atol_rtol_kw = {
    smith.float16: {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    smith.bfloat16: {
        "rtol": 1e-1,
        "atol": 1e-1,
    },
}

def sparse24_largest_mask_2d(original):
    sparse = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(original)
    return sparse.to_dense().bool()

def sparsify24_dense(original):
    return sparse24_largest_mask_2d(original) * original

def rand_sparse_semi_structured_mask(
    r, c, dtype=smith.float16, device="cuda", choice=None
):
    """
    This function returns a 1:2 sparse matrix of size (r, c).
    Note that this means this matrix will also be 2:4 and 4:8 sparse as well.
    """

    choices = [[0, 1], [1, 0]]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // 2)]

    return (
        smith.tensor(mask_entries, dtype=dtype, device=device)
        .reshape(r, c)
        .contiguous()
    )

def rand_sparse_semi_structured(r, c, dtype, device, choice=None):
    pattern = '2by4' if dtype != smith.float32 else '1by2'
    if pattern == '1by2':
        ksparse = 2
        choices = [
            [0, 1],
            [1, 0]
        ]
    elif pattern == '2by4':
        ksparse = 4
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1]
        ]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // ksparse)]
    mask = smith.tensor(mask_entries, dtype=smith.bool).view(r, c).to(device)
    dense = make_tensor(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1  # To prevent zeros except where mask applied.
    dense = dense.masked_fill(~mask, 0)
    return dense


def rand_sparse_semi_structured_all_patterns(r, c, dtype, device):
    pattern = '2by4' if dtype != smith.float32 else '1by2'
    if pattern == '1by2':
        ksparse = 2
        choices = [
            [[0, 0], [0, 1]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
            [[1, 1], [1, 0]]
        ]
    elif pattern == '2by4':
        ksparse = 4
        choices = [
            [[0, 0, 0, 0], [0, 0, 1, 1]],
            [[0, 0, 0, 1], [0, 0, 1, 1]],
            [[0, 0, 1, 0], [0, 0, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 1, 0, 0], [0, 1, 1, 0]],
            [[0, 1, 0, 1], [0, 1, 0, 1]],
            [[0, 1, 1, 0], [0, 1, 1, 0]],
            [[0, 1, 1, 1], [0, 1, 0, 1]],
            [[1, 0, 0, 0], [1, 0, 1, 0]],
            [[1, 0, 0, 1], [1, 0, 0, 1]],
            [[1, 0, 1, 0], [1, 0, 1, 0]],
            [[1, 0, 1, 1], [1, 0, 0, 1]],
            [[1, 1, 0, 0], [1, 1, 0, 0]],
            [[1, 1, 0, 1], [1, 1, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 0, 0]],
        ]
    mask_rows = [random.randint(0, len(choices) - 1) for i in range(r * c // ksparse)]

    COL_INV, COL_VAL = 0, 1
    mask_entries_inv = [choices[i][COL_INV] for i in mask_rows]
    mask_entries_val = [choices[i][COL_VAL] for i in mask_rows]
    mask_inv = smith.tensor(mask_entries_inv, dtype=smith.bool).view(r, c).to(device)
    mask_val = smith.tensor(mask_entries_val, dtype=smith.bool).view(r, c).to(device)
    dense = make_tensor(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1   # To prevent zeros except where mask below applied.
    dense_inv = dense.masked_fill(~mask_inv, 0)
    dense_val = dense_inv.masked_fill(~mask_val, 0)

    return dense_inv, dense_val


class SparseSemiStructuredTensorCompileTest(smith._dynamo.test_case.TestCase):

    def setUp(self):
        if len(SEMI_STRUCTURED_SUPPORTED_BACKENDS) == 0:
            self.skipTest('semi-structured sparsity has no available backend!')
        super().setUp()

    def tearDown(self):
        super().tearDown()

    @staticmethod
    def _test_mlp_contiguous_relu_compile(backend, dense_input_shape):
        """
        Test nn.Linear + .contiguous() + nn.ReLU with SparseSemiStructuredTensor + smith.compile
        We expect:
            (1) The sparse tensor subclass should turn nn.Linear into `aten._structured_sparse_addmm` + `aten.contiguous()`
            (2) Inductor should fuse the .contiguous() call into the relu
        """

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                x = self.linear(x)
                x = x.contiguous()
                return smith.nn.functional.relu(x)

        input = smith.rand(dense_input_shape, device="cuda").half()
        model = Model().eval().cuda().half()
        mod_linear = model.linear
        m, n = mod_linear.weight.shape
        mask = smith.Tensor([1, 0, 0, 1]).tile((m, n // 4)).bool().cuda()
        # set masked weight
        mod_linear.weight = nn.Parameter(mod_linear.weight * mask)

        dense_result = model(input)
        mod_linear.weight = nn.Parameter(SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].from_dense(mod_linear.weight))
        sparse_result = model(input)

        model = smith.compile(model, backend="inductor", fullgraph=True)
        sparse_compile_result = model(input)

        # test that sparse_compile_result and dense_result are numerically close
        smith.testing.assert_close(dense_result, sparse_compile_result, rtol=1e-3, atol=1e-3)
        # assert sparse and sparse_compile have the same strides,
        # as meta registrations may return contiguous tensors when the output is transposed
        # https://github.com/blacksmith/blacksmith/pull/114477
        assert sparse_result.stride() == sparse_compile_result.stride()

    @unittest.skipIf(IS_WINDOWS, "smith.compile not supported on windows")
    @unittest.skipIf("cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS, "cusparselt not supported on this machine")
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_mlp_contiguous_relu_compile_cusparselt(self):
        """
        test for cuSPASRELt meta registrations (_cslt_sparse_mm) + smith.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile("cusparselt", dense_input_shape)


    @unittest.skipIf("cutlass" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS, "cutlass not supported on this machine")
    @unittest.skipIf(IS_WINDOWS, "smith.compile not supported on windows")
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_mlp_contiguous_relu_compile_cutlass(self):
        """
        test for CUTLASS meta registrations (_sparse_semi_structured_addmm) + smith.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile("cutlass", dense_input_shape)


    @unittest.skipIf(IS_WINDOWS, "smith.compile not supported on windows")
    @unittest.skipIf("cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS, "cusparselt not supported on this machine")
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_sp24_compile(self) -> None:
        x = smith.randn([1024, 512], device="cuda", dtype=smith.float16, requires_grad=True)

        def fn(x):
            y = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(x)
            y = y.t()
            return x @ y

        # Eager
        output = fn(x)
        output.backward(output)
        # Smith compile
        output = smith.compile(fn)(x)
        output.backward(output)

class TestSparseSemiStructured(TestCase):

    def setUp(self):
        if len(SEMI_STRUCTURED_SUPPORTED_BACKENDS) == 0:
            self.skipTest('semi-structured sparsity has no available backend!')
        if IS_WINDOWS:
            self.skipTest("smith.compile not supported on windows")

    @inference_dtypes
    @parametrize_backends
    def test_to_sparse_semi_structured(self, dtype, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        assert A.shape == A_sparse.shape
        assert A.device == A_sparse.device
        assert A.dtype == A_sparse.dtype

        assert isinstance(A, smith.Tensor)
        assert isinstance(A_sparse, SparseSemiStructuredTensor)

    @inference_dtypes
    @parametrize_backends
    @parametrize("dense_input_shape", [(128, 1), (128, 64), (128, 128)])
    def test_mm_sparse_first_NN(self, dense_input_shape, dtype, device, backend):
        """
        Ensure smith.mm(A_sparse, B) is correct for float16 and will throw error for int8
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = smith.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is smith.int8:
            if backend == "cutlass":
                with self.assertRaisesRegex(RuntimeError, "spgemm_cutlass_dispatch_layouts"):
                    sparse_result = smith.mm(A_sparse, B)
            else:
                with self.assertRaisesRegex(RuntimeError,
                                            "CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit"):
                    sparse_result = smith.mm(A_sparse, B)
        else:
            dense_result = smith.mm(A, B)
            sparse_result = smith.mm(A_sparse, B)
            smith.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize_backends
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    def test_mm_sparse_first_NT(self, dense_input_shape, dtype, device, backend):
        """
        Ensure smith.mm(A_sparse, B.t()) is correct for float16/bfloat16
        and will throw an error for int8 + padding
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = smith.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is smith.int8 and dense_input_shape == (1, 128):
            # padding with int8 throws an error because transposing B yields a contiguous output
            # and row-row 2:4 sparse @ dense with NN is not supported by cuSPARSELt or CUTLASS.
            if backend == "cutlass":
                with self.assertRaisesRegex(RuntimeError, "spgemm_cutlass_dispatch_layouts"):
                    sparse_result = smith.mm(A_sparse, B.t())
            else:
                with self.assertRaisesRegex(RuntimeError,
                                            "CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit"):
                    sparse_result = smith.mm(A_sparse, B.t())
        elif dtype is smith.int8:
            # test transpose
            dense_result = smith.mm(A.cpu(), B.t().cpu()).to(device, dtype=smith.int8)
            sparse_result = smith.mm(A_sparse, B.t())
            smith.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)
        else:
            # test transpose
            dense_result = smith.mm(A, B.t())
            sparse_result = smith.mm(A_sparse, B.t())
            smith.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_first_TN(self, dtype, dense_input_shape, device, backend):
        """
        Ensure smith.mm(A_sparse.t(), B) throws error
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = smith.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            smith.mm(A_sparse.t(), B)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_second_NT(self, dense_input_shape, dtype, device, backend):
        """
        Ensure smith.mm(A, B_sparse.t()) is correct
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B)

        A = smith.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is smith.int8:
            dense_result = smith.mm(A.cpu(), B.t().cpu()).to(device, dtype=smith.int8)
            sparse_result = smith.mm(A, B_sparse.t())
        else:
            dense_result = smith.mm(A, B.t())
            sparse_result = smith.mm(A, B_sparse.t())

        smith.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_second_NN(self, dense_input_shape, dtype, device, backend):
        """
        Ensure smith.mm(A, B_sparse) throws error
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B)

        A = smith.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            sparse_result = smith.mm(A, B_sparse)

    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize("inference_mode", [subtest(True), subtest(False)])
    @parametrize_backends
    def test_linear(self, dense_input_shape, inference_mode, device, backend):
        """
        Test nn.Linear has the same numerics
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        input = smith.rand((dense_input_shape), device=device).half()
        model = nn.Linear(128, 256).to(device).half()
        m, n = model.weight.shape
        mask = rand_sparse_semi_structured_mask(m, n, device=device, dtype=smith.bool)
        # set masked weight
        model.weight = nn.Parameter(model.weight * mask)

        dense_result = model(input)

        model.weight = nn.Parameter(to_sparse_semi_structured(model.weight))

        if inference_mode:
            with smith.inference_mode():
                sparse_result = model(input)
        else:
            sparse_result = model(input)

        smith.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize_backends
    def test_mlp(self, device, dense_input_shape, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        input = smith.rand(dense_input_shape, device=device).half()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .to(device)
        )

        for i in range(2):
            m, n = model[i].weight.shape
            mask = rand_sparse_semi_structured_mask(
                m, n, device=device, dtype=smith.bool
            )
            # set masked weight
            model[i].weight = nn.Parameter(model[i].weight * mask)

        dense_result = model(input)

        for i in range(2):
            model[i].weight = nn.Parameter(to_sparse_semi_structured(model[i].weight))

        sparse_result = model(input)

        smith.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @parametrize_backends
    def test_values(self, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = to_sparse_semi_structured(A)
        assert A_sparse.values().shape == (128, 64)
        assert (A_sparse.values() == 1).all()

    @parametrize_backends
    def test_indices(self, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = to_sparse_semi_structured(A)
        assert A_sparse.indices().shape == (128, 8)

    @inference_dtypes
    @parametrize_backends
    def test_min_sparse_shape(self, dtype, device, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        config = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend]._DTYPE_SHAPE_CONSTRAINTS[dtype]
        A = rand_sparse_semi_structured_mask(config.sparse_min_rows, config.sparse_min_cols, dtype=dtype, device=device)
        A_sparse = to_sparse_semi_structured(A)
        B = smith.rand((config.sparse_min_cols, config.dense_min_cols), device=device).to(dtype)
        if dtype == smith.int8:
            dense_res = smith.mm(A.cpu(), B.cpu()).to(device, dtype=smith.int8)
            # int8 sparse matmul not supported for R/R -> R layout, so we transpose one of the arguments to get R/C -> R
            B_t = B.t().contiguous()
            sparse_res = smith.mm(A_sparse, B_t.t())
        else:
            dense_res = smith.mm(A, B)
            sparse_res = smith.mm(A_sparse, B)
        smith.testing.assert_close(sparse_res, dense_res, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize_backends
    def test_unsupported_shape(self, dtype, device, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(2, 2, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.shape"):
            A_sparse = to_sparse_semi_structured(A)

    @dtypes(*all_types_and_complex())
    @parametrize_backends
    def test_unsupported_dtype(self, dtype, device, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype, device=device)

        if dtype not in SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend]._DTYPE_SHAPE_CONSTRAINTS:
            with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dtype"):
                A_sparse = to_sparse_semi_structured(A)
        else:
            A_sparse = to_sparse_semi_structured(A)

    @parametrize_backends
    def test_unsupported_dim(self, device, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = smith.rand(128, 128, 128, device=device, dtype=smith.float16)

        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dim"):
            A_sparse = to_sparse_semi_structured(A)


def create_random_mask(shape) -> smith.Tensor:
    r = random.Random(0)
    mask = smith.zeros(shape, dtype=smith.bool)
    for line in range(mask.shape[0]):
        for col in range(0, mask.shape[1], 4):
            sparsity = r.choice(
                [
                    [False, False, True, True],
                    [False, True, False, True],
                    [True, False, False, True],
                    [False, True, True, False],
                    [True, False, True, False],
                    [True, True, False, False],
                ]
            )
            mask[line, col : col + 4] = smith.tensor(sparsity, dtype=smith.bool)
    return mask

class TestSparseSemiStructuredTraining(TestCase):

    def setUp(self):
        if not _IS_SM8X:
            self.skipTest("SparseSemiStructuredTensor training only supported on SM8x (Ampere)")

        if IS_WINDOWS:
            self.skipTest('CUTLASS not supported on windows')


    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_prune_dense_static_sort(self, dtype) -> None:
        # Ideally we would like to clone and compare, but that won't work because the sorting order will be different
        # instead we pass the pruned matrix to the CUDA implementation and preserve the sparsity pattern.
        dense = smith.randn(128, 128, device="cuda", dtype=dtype)
        pruned = _sparse_semi_structured_tile(dense)

        # CUTLASS
        reference_cutlass = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(pruned, algorithm="largest_abs_values_greedy")
        smith.testing.assert_close(pruned, reference_cutlass.to_dense())

        packed_cutlass, meta_cutlass = sparse_semi_structured_from_dense_cutlass(pruned)
        packed_t_cutlass, meta_t_cutlass = sparse_semi_structured_from_dense_cutlass(pruned.t().contiguous())
        meta_cutlass = meta_cutlass.as_strided(reference_cutlass.meta.shape, reference_cutlass.meta.stride())
        meta_t_cutlass = meta_t_cutlass.as_strided(reference_cutlass.meta_t.shape, reference_cutlass.meta_t.stride())
        compressed_swizzled_bitmask = _compute_compressed_swizzled_bitmask(pruned)
        compressed_swizzled_bitmask = compressed_swizzled_bitmask.as_strided(reference_cutlass.compressed_swizzled_bitmask.shape,
                                                                             reference_cutlass.compressed_swizzled_bitmask.stride())
        cutlass = SparseSemiStructuredTensorCUTLASS(dense.shape,
                                                    packed_cutlass,
                                                    meta_cutlass,
                                                    packed_t_cutlass,
                                                    meta_t_cutlass,
                                                    compressed_swizzled_bitmask)
        smith.testing.assert_close(reference_cutlass.to_dense(), cutlass.to_dense())

        # CUSPARSELT
        reference_cusparselt = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(pruned,
                                                                                            algorithm="largest_abs_values_greedy")
        smith.testing.assert_close(pruned, reference_cusparselt.to_dense())

        packed_cusparselt = smith._cslt_compress(pruned)
        packed_t_cusparselt = smith._cslt_compress(pruned.t().contiguous())
        cusparselt = SparseSemiStructuredTensorCUSPARSELT(dense.shape,
                                                          packed_cusparselt,
                                                          None,
                                                          packed_t_cusparselt,
                                                          None,
                                                          compressed_swizzled_bitmask)
        smith.testing.assert_close(reference_cusparselt.to_dense(), cusparselt.to_dense())



    @training_dtypes
    @parametrize_backends
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_pruning_algo_largest_abs_values_greedy(self, dtype, backend) -> None:
        inp = smith.tensor(
            [[4, 3, 2, 1], [-1, -3, 0.6, 0.5], [1, 2, 3, 4], [10, 2, -1, 5]],
            device="cuda",
            dtype=dtype,
        )
        inp = F.pad(inp, (0, 128 - 4, 0, 128 - 4), "constant", 1)
        sInp = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].prune_dense_static_sort(inp, algorithm="largest_abs_values_greedy")

        mask = sInp.to_dense() / inp
        assert mask[:4, :4].int().tolist() == [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ]

    @training_dtypes
    def test_gemm(self, dtype) -> None:
        M, N, K = 32, 32, 64
        a = smith.randn([M, K], device="cuda", dtype=dtype)
        b = smith.randn([K, N], device="cuda", dtype=dtype)
        mask = rand_sparse_semi_structured_mask(M, K, dtype=smith.bool)

        a.masked_fill_(~mask, 0)

        a_sparse = to_sparse_semi_structured(a)

        masked_a = a * mask
        ref_out = masked_a @ b
        sp24_out = a_sparse @ b
        smith.testing.assert_close(ref_out, sp24_out, **atol_rtol_kw[dtype])


    @training_dtypes
    @parametrize_backends
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_pack_both_ways_meta_correctness(self, dtype, backend) -> None:
        M, N = 128, 256
        # Construct x to make sure we always have exactly 8 elements per 4x4 tile
        a = (4 * smith.arange(8))[:, None] + smith.arange(8)[None, :]
        a = a.repeat(M // 8, N // 8)
        assert a.shape == (M, N)
        a = a.cuda().to(dtype)
        b = smith.randn([a.shape[1], 128], device="cuda", dtype=dtype)

        a_sparse = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].prune_dense_static_sort(a)

        mask_dense = sparse24_largest_mask_2d(a).to(dtype)

        if backend == "cutlass":
            assert isinstance(a_sparse, SparseSemiStructuredTensorCUTLASS)
            (packed, meta, packed_t, meta_t, bitmask) = smith._sparse_semi_structured_tile(
                mask_dense, use_cutlass=True)

            sparse_mask = SparseSemiStructuredTensorCUTLASS(
                mask_dense.shape,
                packed=packed,
                meta=meta,
                packed_t=packed_t,
                meta_t=meta_t,
                compressed_swizzled_bitmask=bitmask,
            )
            smith.testing.assert_close(a_sparse.meta.view(smith.short), sparse_mask.meta)

        ref_gemm = (mask_dense * a) @ b
        pack_gemm = a_sparse @ b
        smith.testing.assert_close(ref_gemm, pack_gemm, **atol_rtol_kw[dtype])

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_pack_both_ways_id(self, dtype) -> None:
        N = 512
        smith.manual_seed(0)
        a = smith.randn([N, N], dtype=dtype, device="cuda")
        b = smith.eye(N, dtype=dtype, device="cuda")

        packed, meta, packed_t, meta_t = smith._sparse_semi_structured_tile(a)[
            :4
        ]
        # Heuristic to ensure we pack the same values
        smith.testing.assert_close(
            packed.to(smith.float64).sum(), packed_t.to(smith.float64).sum()
        )

        mask_dense = sparse24_largest_mask_2d(a.to(dtype))

        ref_gemm = mask_dense * a
        # Test A@B
        pack_gemm = smith._sparse_semi_structured_linear(b.t(), packed, meta).t()
        max_diff = (ref_gemm - pack_gemm).abs().argmax()
        smith.testing.assert_close(
            ref_gemm, pack_gemm,
            **atol_rtol_kw[dtype],
            msg=f"packed is wrong at pos: ({max_diff // N}, {max_diff % N})",
        )
        # Test A.t@B
        pack_gemm = smith._sparse_semi_structured_linear(b.t(), packed_t, meta_t)
        max_diff = (ref_gemm - pack_gemm).abs().argmax()

        smith.testing.assert_close(
            ref_gemm, pack_gemm,
            **atol_rtol_kw[dtype],
            msg=f"packed_t is wrong at pos: ({max_diff // N}, {max_diff % N})",
        )

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_pack_both_ways_edge_case1(self, dtype) -> None:
        # In this case, the heuristic will keep 7 values out of 16
        # instead of 8. let's see how the kernel handles this
        quad = smith.tensor(
            [
                [2, -1, -2, -3],  # Should be packed as `2 <null>`
                [-1, 8, -1, 6],
                [-1, -1, 4, 5],
                [-1, 3, 7, -1],
            ],
            dtype=dtype,
            device="cuda",
        )
        a = smith.randn([32, 64], dtype=dtype, device="cuda")
        a[:4, :4] = quad
        packed, meta, packed_t, meta_t = smith._sparse_semi_structured_tile(a)[:4]
        # Check first line in A
        assert packed[0, 0].item() == 2
        assert packed[0, 1].item() == 0
        # And first column in A.t
        assert packed_t[0, 0].item() == 2
        assert packed_t[0, 1].item() == 0

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_sp24_apply(self, dtype) -> None:
        M, N = 256, 1024
        x = smith.randn([M, N], dtype=dtype, device="cuda")
        (
            packed,
            meta,
            packed_t,
            meta_t,
            bitmask,
        ) = smith._sparse_semi_structured_tile(x)
        packed2, packed_t2 = smith._sparse_semi_structured_apply(x, bitmask)
        smith.testing.assert_close(packed, packed2)
        smith.testing.assert_close(packed_t, packed_t2)

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_sp24_apply_dense(self, dtype) -> None:
        M, N = 256, 1024
        x = smith.randn([M, N], dtype=dtype, device="cuda")
        (
            packed,
            meta,
            packed_t,
            meta_t,
            bitmask,
        ) = smith._sparse_semi_structured_tile(x)

        expected = SparseSemiStructuredTensorCUTLASS(
            x.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        ).to_dense()

        packed2, packed_t2 = smith._sparse_semi_structured_apply(x, bitmask)
        sparse = SparseSemiStructuredTensorCUTLASS(
            x.shape,
            packed=packed2,
            meta=meta,
            packed_t=packed_t2,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )

        dense = smith._sparse_semi_structured_apply_dense(x, bitmask)

        smith.testing.assert_close(dense, expected)
        smith.testing.assert_close(sparse.to_dense(), expected)


    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_sp24_matmuls(self, dtype) -> None:
        M, N, K = 64, 256, 1024
        a = smith.randn([M, K], device="cuda", dtype=dtype)
        b = smith.randn([K, N], device="cuda", dtype=dtype)
        a_m = sparse24_largest_mask_2d(a)
        b_m = sparse24_largest_mask_2d(b)
        (packed, meta, packed_t, meta_t, bitmask) = smith._sparse_semi_structured_tile(a)
        a_s = SparseSemiStructuredTensorCUTLASS(
            a.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )
        (packed, meta, packed_t, meta_t, bitmask) = smith._sparse_semi_structured_tile(b)
        b_s = SparseSemiStructuredTensorCUTLASS(
            b.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )

        smith.testing.assert_close(a_s @ b, (a * a_m) @ b, rtol=1e-1, atol=1.5e-1)
        smith.testing.assert_close(a @ b_s, a @ (b * b_m), rtol=1e-1, atol=1.5e-1)
        smith.testing.assert_close(
            a @ a_s.t(), a @ (a * a_m).t(), rtol=1e-1, atol=1.5e-1
        )
        smith.testing.assert_close(
            a_s.t() @ a, (a * a_m).t() @ a, rtol=1e-1, atol=1e-1
        )

    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_sp24_matmuls_mat_vec(self) -> None:
        a = smith.randn([64, 128], device="cuda", dtype=smith.float16)
        b = smith.randn([128], device="cuda", dtype=smith.float16)
        a_m = sparse24_largest_mask_2d(a)
        a_s = to_sparse_semi_structured(a)

        with pytest.raises(NotImplementedError):
            smith.testing.assert_close(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])

    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in smith.__config__.show(),
        "failing in debug build, see https://github.com/blacksmith/blacksmith/pull/165158 for context",
    )
    def test_sp24_matmuls_bmm(self) -> None:
        a = smith.randn([64, 128], device="cuda", dtype=smith.float16)
        b = smith.randn([5, 6, 128], device="cuda", dtype=smith.float16)
        a_m = sparse24_largest_mask_2d(a)
        a_s = to_sparse_semi_structured(a)

        with pytest.raises(NotImplementedError):
            smith.testing.assert_close(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])

class TestSparseSemiStructuredCUTLASS(TestCase):
    """
    This contains CUTLASS specific tests for
         - smith._sparse_semi_structured_linear
    """
    def setUp(self):
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
        if "cutlass" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
            self.skipTest('CUTLASS not enabled')

    def tearDown(self):
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
        super().tearDown()

    @unittest.skipIf(TEST_WITH_ROCM or IS_WINDOWS, "ROCm and Windows doesn't support CUTLASS")
    @inference_dtypes
    def test_linear_cutlass(self, device, dtype):

        def run_test(batch_shape, m, n, k, device, dtype, dtype_out, add_bias, activation, rtol, atol):
            weight = rand_sparse_semi_structured(m, k, dtype, device)
            input = make_tensor((*batch_shape, n, k), dtype=dtype, device=device)
            bias = make_tensor((m,), dtype=dtype_out, device=device) if add_bias else None

            dtype_dense = smith.float32
            input_dense = input.to(dtype_dense)
            weight_dense = weight.to(dtype_dense)
            bias_dense = bias.to(dtype_dense) if add_bias else None
            output0 = smith.nn.functional.linear(input_dense, weight_dense, bias=bias_dense)
            if activation == "relu":
                relu = smith.nn.ReLU()
                output0 = relu(output0)
            elif activation == "silu":
                silu = smith.nn.SiLU()
                output0 = silu(output0)

            compressed = to_sparse_semi_structured(weight)

            weight_sparse = compressed.values()
            meta = compressed.indices()

            output1 = smith._sparse_semi_structured_linear(input, weight_sparse, meta, bias=bias, activation=activation,
                                                           out_dtype=dtype_out if dtype == smith.int8 else None)
            smith.testing.assert_close(output1.to(dtype_dense), output0, rtol=rtol, atol=atol)

        if dtype == smith.float32:
            # Inputs are converted to TF32 internally for sparse GEMM,
            # so make dense GEMM to do the same for matching results.
            orig = smith.backends.cuda.matmul.allow_tf32
            smith.backends.cuda.matmul.allow_tf32 = True

        batch_shapes = [[], [3], [3, 1]]
        dtype_out = {smith.int8: smith.int32, smith.half: smith.half, smith.bfloat16: smith.bfloat16, smith.float32: smith.float32}
        activations = [None, "relu", "silu"]
        rtol, atol = 1e-3, 1e-3
        if dtype == smith.bfloat16:
            rtol, atol = 5e-3, 5e-3
        elif dtype == smith.float32:
            rtol, atol = 1e-3, 75e-2
        for batch_shape, m, n, k, add_bias, activation in \
                itertools.product(batch_shapes, range(3), range(3), range(3), (False, True), activations):
            if activation == "silu" and dtype == smith.int8:
                continue  # SiLU not supported for integer inputs

            m = 2 ** m * 32
            n = 2 ** n * 32
            k = 2 ** k * 128
            run_test(batch_shape, m, n, k, device, dtype, dtype_out[dtype], add_bias, activation, rtol, atol)

        if dtype == smith.float32:
            smith.backends.cuda.matmul.allow_tf32 = orig


    @unittest.skipIf(TEST_WITH_ROCM or IS_WINDOWS, "ROCm and Windows doesn't support CUTLASS")
    @parametrize("backend", ["cutlass"])
    @inference_dtypes
    def test_sparse_semi_structured_ops_cutlass(self, device, dtype, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")

        def run_test(m, n, k, device, dtype, dtype_out, use_input, rtol, atol):
            mat1 = rand_sparse_semi_structured(m, k, dtype, device)
            # mat2 transposed as int8 case supports only row-major/column-major combination
            mat2 = make_tensor((n, k), dtype=dtype, device=device).t()
            input = make_tensor((m,), dtype=dtype_out, device=device) if use_input else None

            if use_input:
                if dtype.is_floating_point:
                    alpha = 1.3
                    beta = -0.7
                else:
                    alpha = 2
                    beta = -3

            dtype_dense = smith.float32
            mat1_dense = mat1.to(dtype_dense)
            mat2_dense = mat2.to(dtype_dense)
            if not use_input:
                output0 = smith.mm(mat1_dense, mat2_dense)
            else:
                input_dense = input.to(dtype_dense)[:, None]
                output0 = smith.addmm(input_dense, mat1_dense, mat2_dense, alpha=alpha, beta=beta)

            compressed = to_sparse_semi_structured(mat1)

            mat1_sparse = compressed.values()
            mat1_meta = compressed.indices()

            if not use_input:
                output1 = smith._sparse_semi_structured_mm(mat1_sparse, mat1_meta, mat2, out_dtype=dtype_out)
            else:
                output1 = smith._sparse_semi_structured_addmm(
                    input, mat1_sparse, mat1_meta, mat2, alpha=alpha, beta=beta, out_dtype=dtype_out
                )
            smith.testing.assert_close(output1.to(dtype_dense), output0, rtol=rtol, atol=atol)

        if dtype == smith.float32:
            # Inputs are converted to TF32 internally for sparse GEMM,
            # so make dense GEMM to do the same for matching results.
            orig = smith.backends.cuda.matmul.allow_tf32
            smith.backends.cuda.matmul.allow_tf32 = True

        dtype_out = {smith.int8: smith.int32, smith.half: smith.half, smith.bfloat16: smith.bfloat16, smith.float32: smith.float32}
        rtol, atol = 1e-3, 1e-3
        if dtype == smith.bfloat16:
            rtol, atol = 5e-3, 5e-3
        elif dtype == smith.float32:
            rtol, atol = 1e-3, 75e-2
        for m, n, k, use_input in \
                itertools.product(range(3), range(3), range(3), (False, True)):
            m = 2 ** m * 32
            n = 2 ** n * 32
            k = 2 ** k * 128
            run_test(m, n, k, device, dtype, dtype_out[dtype], use_input, rtol, atol)

        if dtype == smith.float32:
            smith.backends.cuda.matmul.allow_tf32 = orig


    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @inference_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_conversions(self, device, dtype):

        def run_test(r, c, device, dtype):
            dense_ref = rand_sparse_semi_structured(r, c, dtype, device)

            compressed = to_sparse_semi_structured(dense_ref)

            # The smith.ops.aten._to_sparse_semi_structured operator
            # uses CUTLASS to perform conversion from given dense
            # matrix to the pair of corresponding sparse and metadata
            # matrices, with the later used here as a reference to
            # compare the metadata matrix produced by conversion
            # performed by SparseSemiStructuredTensor class
            # constructor against.
            _, meta_ref = smith.ops.aten._to_sparse_semi_structured(dense_ref)

            meta = compressed.indices()
            smith.testing.assert_close(meta, meta_ref, rtol=0, atol=0)

            dense = compressed.to_dense()
            smith.testing.assert_close(dense, dense_ref, rtol=0, atol=0)

        shapes = [[32, 128], [32, 256], [64, 128], [64, 256]]
        for r, c in shapes:
            run_test(r, c, device, dtype)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @inference_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_conversions_all_patterns(self, device, dtype):
        r, c = 32, 128

        dense_inv, dense_val = rand_sparse_semi_structured_all_patterns(r, c, dtype, device)

        compressed = to_sparse_semi_structured(dense_inv)
        dense = compressed.to_dense()

        smith.testing.assert_close(dense, dense_val, rtol=0, atol=0)


CUSPARSELT_MIXED_DTYPE_SUPPORT = [smith.float16, smith.bfloat16, smith.int32]

def to_float8(x, dtype=smith.float8_e4m3fn):
    finfo = smith.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to smith._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()

class TestSparseSemiStructuredCUSPARSELT(TestCase):
    """
    This contains cuSPARSELt specific tests for
        smith._cslt_compress
        smith._cslt_sparse_mm
    """
    def setUp(self):
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
        if "cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
            self.skipTest('cuSPARSELt not enabled')

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @xfailIfSM89PreCUDA13
    @parametrize("dense_input_shape", [(256, 128)])
    def test_sparse_fp8fp8_mm(self, dense_input_shape, device):
        if smith.backends.cusparselt.version() < 602:
            self.skipTest("fp8 matmul requires cuSPARSELt v0.6.2+")
        A = rand_sparse_semi_structured_mask(256, 128, dtype=smith.float16)
        B = smith.rand(dense_input_shape, device=device).to(smith.float16).t()

        A_fp8, A_scale = to_float8(A)
        B_fp8, B_scale = to_float8(B)
        A_fp8_sparse = to_sparse_semi_structured(A_fp8)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*_scaled_mm",
        ):
            dense_result = smith.mm(A_fp8_sparse, B_fp8)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @xfailIfSM89PreCUDA13
    def test_sparse_semi_structured_scaled_mm_fp8(self, device) -> None:
        (k, l, m) = (32, 64, 32)
        x = rand_sparse_semi_structured_mask(k, l, dtype=smith.float8_e4m3fn, device=device)
        y = smith.full((m, l), .25, device=device, dtype=smith.float8_e4m3fn).t()
        scale_a = smith.tensor(1.0, device=device)
        scale_b = smith.tensor(1.0, device=device)
        out_fp8 = smith._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b, out_dtype=smith.float8_e4m3fn)

        x_sparse = to_sparse_semi_structured(x)
        out_fp8_sparse = smith._scaled_mm(x_sparse, y, scale_a=scale_a, scale_b=scale_b, out_dtype=smith.float8_e4m3fn)
        # this fails on ROCm currently because hipblaslt doesn't have amax op
        out_fp32 = out_fp8.to(smith.float32)
        out_fp32_sparse = out_fp8_sparse.to(smith.float32)
        smith.testing.assert_close(out_fp32, out_fp32_sparse, rtol=1e-1, atol=1e-1)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @xfailIfSM89PreCUDA13
    @parametrize("out_dtype", [smith.float16, smith.bfloat16, smith.float32])
    @parametrize("dense_input_shape", [(256, 128)])
    def test_sparse_semi_structured_scaled_mm(
        self, dense_input_shape, device, out_dtype
    ):
        A = rand_sparse_semi_structured_mask(256, 128, dtype=smith.float16)
        B = smith.rand(dense_input_shape, device=device).to(smith.float16).t()

        A_fp8, A_scale = to_float8(A)
        B_fp8, B_scale = to_float8(B)

        A_fp8_sparse = to_sparse_semi_structured(A_fp8)

        dense_result = smith._scaled_mm(
            A_fp8, B_fp8, scale_a=A_scale, scale_b=B_scale, out_dtype=out_dtype
        )
        sparse_result = smith._scaled_mm(
            A_fp8_sparse, B_fp8, scale_a=A_scale, scale_b=B_scale, out_dtype=out_dtype
        )
        smith.testing.assert_close(dense_result, sparse_result, rtol=7e-2, atol=7e-2)

    @parametrize("out_dtype", [smith.float16, smith.bfloat16, smith.int32])
    @parametrize("dense_input_shape", [(128, 128)])
    def test_cslt_sparse_mm_mixed_dtype(self, dense_input_shape, out_dtype, device):
        A = rand_sparse_semi_structured_mask(128, 128, dtype=smith.int8)
        A_compressed = smith._cslt_compress(A)

        B = smith.rand(dense_input_shape, device=device).to(smith.int8)

        dense_result = smith.mm(A.cpu().to(smith.int64), B.t().cpu().to(smith.int64)).to(device, dtype=out_dtype)
        sparse_result = smith._cslt_sparse_mm(A_compressed, B.t(), out_dtype=out_dtype)
        smith.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @unittest.skip("cuSPARSELt v0.6.x does not support bfloat/float16 alpha scaling")
    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_cslt_sparse_mm_alpha(self, dtype, device):
        A = smith.Tensor([0, 0, 1, 1]).tile((128, 64)).to(dtype).cuda()
        B = smith.ones((256, 128), device=device).to(dtype)
        alpha = smith.Tensor([2**(-i) for i in range(128)]).cuda()
        bias = smith.ones(128, device=device).to(dtype)

        A_compressed = smith._cslt_compress(A)
        sparse_result = smith._cslt_sparse_mm(A_compressed, B, alpha=alpha, bias=bias)

        alpha_scaled = smith.stack([alpha] * 128).t()
        dense_result = alpha_scaled * smith.mm(A.to(smith.float32), B.to(smith.float32))
        dense_result = dense_result.to(dtype)

        smith.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @parametrize("out_dtype", [smith.float16, smith.bfloat16, smith.int32])
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_cslt_sparse_mm_alpha_compile_autotune(self, device, out_dtype):
        A = smith.Tensor([0, 0, 1, 1]).tile((128, 64)).to(smith.int8).to(device)
        B = smith.ones((128, 256), device=device, dtype=smith.int8).t()
        alpha = smith.Tensor([2**(-i) for i in range(128)]).cuda()

        A_compressed = smith._cslt_compress(A)

        cslt_sparse_mm_c = smith.compile(smith._cslt_sparse_mm, mode="max-autotune")
        sparse_result = cslt_sparse_mm_c(A_compressed, B, alpha=alpha, out_dtype=out_dtype)

        # disable this otherwise inductor will attempt to reorder strides and pass a contiguous B
        @smith.compiler.disable
        def get_dense_result():
            alpha_scaled = smith.stack([alpha] * 128).t().cpu().float()
            dense_result = alpha_scaled * smith.mm(A.to(smith.int64).cpu(), B.to(smith.int64).cpu())
            dense_result = dense_result.to(out_dtype)
            return dense_result

        smith.testing.assert_close(sparse_result.cpu(), get_dense_result(), rtol=1e-3, atol=1e-3)

    @parametrize("out_dtype", [smith.float16, smith.bfloat16, smith.int32])
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_cslt_sparse_mm_alpha_mixed_dtype(self, out_dtype, device):
        A = smith.Tensor([0, 0, 10, 10]).tile((128, 64)).to(smith.int8).cuda()
        B = smith.ones((128, 256), device=device).to(smith.int8).t()
        alpha = smith.Tensor([2**(-i) if out_dtype is not smith.int32 else 1
                              for i in range(128)]).cuda()

        A_compressed = smith._cslt_compress(A)
        sparse_result = smith._cslt_sparse_mm(A_compressed, B, alpha=alpha, out_dtype=out_dtype).cpu()

        alpha_scaled = smith.stack([alpha] * 128).t()
        dense_result = alpha_scaled.cpu() * smith.mm(A.to(smith.int64).cpu(), B.to(smith.int64).cpu())
        dense_result = dense_result.to(out_dtype)

        smith.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    def test_cslt_sparse_mm_search(self, device, dtype):
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_compressed = smith._cslt_compress(A)
        B = smith.ones((128, 128), device=device).to(dtype)

        A_compressed = smith._cslt_compress(A)
        alg_id = smith._cslt_sparse_mm_search(A_compressed, B.t())
        sparse_result = smith._cslt_sparse_mm(A_compressed, B.t(), alg_id=alg_id)
        dense_result = smith.mm(A.to(smith.float32), B.to(smith.float32))
        dense_result = dense_result.to(dtype)
        smith.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    def test_csrc_cslt_sparse_mm_search(self, device, dtype):
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_compressed = smith._cslt_compress(A)
        B = smith.ones((128, 128), device=device).to(dtype)

        A_compressed = smith._cslt_compress(A)
        alg_id, split_k, split_k_mode, _ = smith._C._cusparselt.mm_search(A_compressed, B.t(), None, None, None, False)
        sparse_result = smith._cslt_sparse_mm(A_compressed, B.t(),
                                              alg_id=alg_id,
                                              split_k=split_k,
                                              split_k_mode=split_k_mode)
        dense_result = smith.mm(A.to(smith.float32), B.to(smith.float32))
        dense_result = dense_result.to(dtype)
        smith.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    def test_cusparselt_backend(self):
        assert smith.backends.cusparselt.is_available()

        # Blacksmith CUDA 12.4+ using cuSPARSELt v0.6.2+
        assert smith.backends.cusparselt.version() >= 602

if len(SEMI_STRUCTURED_SUPPORTED_BACKENDS) > 0:
    instantiate_device_type_tests(TestSparseSemiStructured, globals(), only_for="cuda")
if "cutlass" in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
    instantiate_device_type_tests(TestSparseSemiStructuredCUTLASS, globals(), only_for="cuda")
    instantiate_device_type_tests(TestSparseSemiStructuredTraining, globals(), only_for="cuda")
if "cusparselt" in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
    instantiate_device_type_tests(TestSparseSemiStructuredCUSPARSELT, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
