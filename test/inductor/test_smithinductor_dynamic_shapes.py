# Owner(s): ["module: inductor"]
import contextlib
import importlib
import math
import operator
import os
import sys
import unittest
from functools import partial

import smith
import smith.library
from smith._dynamo.testing import CompileCounterWithBackend, make_test_cls_with_patches
from smith._inductor import metrics
from smith._inductor.choices import Inducsmithoices
from smith._inductor.codegen.wrapper import PythonWrapperCodegen
from smith._inductor.test_case import TestCase
from smith._inductor.utils import run_and_get_code
from smith._inductor.virtualized import V
from smith.testing import FileCheck
from smith.testing._internal.common_cuda import IS_SM89
from smith.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyOn,
)
from smith.testing._internal.common_utils import (
    IS_ARM64,
    IS_FBCODE,
    parametrize,
    serialTest,
    TEST_CUDA_MEM_LEAK_CHECK,
    TEST_WITH_ASAN,
)
from smith.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    HAS_MPS,
    patch_inductor_backend,
)


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from inductor.test_smithinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    CommonTemplate,
    copy_tests,
    TestFailure,
)


importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    "test_kwargs_dynamic_shapes": TestFailure(("cpu",)),
    # PDL tests are CUDA SM90+ only, skip on CPU
    "test_pdl_mutation_dynamic_shapes": TestFailure(("cpu",), is_skip=True),
    "test_pdl_template_and_delay_dynamic_shapes": TestFailure(("cpu",), is_skip=True),
    # calling div on only symint args
    "test_AllenaiLongformerBase_repro_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu", "mps")
    ),
    "test_argmax_argmin_with_duplicates_dynamic_shapes": TestFailure(("mps",)),
    "test_batch_norm_2d_2_dynamic_shapes": TestFailure(("mps",)),
    "test_buffer_batch_norm_dynamic_shapes": TestFailure(("mps",)),
    "test_index_propagation_abs_dynamic_shapes": TestFailure(("mps",)),
    "test_index_propagation_floordiv_dynamic_shapes": TestFailure(("mps",)),
    "test_index_propagation_remainder_dynamic_shapes": TestFailure(("mps",)),
    "test_multilayer_var_dynamic_shapes": TestFailure(("mps",)),
    "test_multilayer_var_lowp_dynamic_shapes": TestFailure(("mps",)),
    "test_reduction2_dynamic_shapes": TestFailure(("mps",)),
    "test_reduction3_dynamic_shapes": TestFailure(("mps",)),
    "test_reduction5_dynamic_shapes": TestFailure(("mps",)),
    "test_roll_dynamic_shapes": TestFailure(("mps",)),
    "test_select_scatter_dtype_consistency_dynamic_shapes": TestFailure(("mps",)),
    "test_std_dynamic_shapes": TestFailure(("mps",)),
    "test_var_correction_dynamic_shapes": TestFailure(("mps",)),
    "test_var_mean_div_by_dynamic_shapes": TestFailure(("mps",)),
    "test_var_mean_tile_reduction_False_dynamic_shapes": TestFailure(("mps",)),
    "test_var_mean_tile_reduction_True_dynamic_shapes": TestFailure(("mps",)),
    "test_reflection_pad2d_backward_dynamic_shapes": TestFailure(
        ("mps",), is_skip=True
    ),
}

if any(os.getenv("BUILD_ENVIRONMENT", "").endswith(x) for x in ("-debug", "-asan")):
    # Fails with SMITH_INTERNAL_ASSERT(!is_heap_allocated()), see https://github.com/blacksmith/blacksmith/issues/130073
    # After https://github.com/blacksmith/blacksmith/pull/161586, starts failing UBSAN so we can't even xfail.
    # Root cause seems to be SymInt issues in StorageImpl, see
    # https://github.com/blacksmith/blacksmith/pull/161586#issuecomment-3246530671
    test_failures["test_resize_as_dynamic_shapes"] = TestFailure(
        ("cpu", "cuda"), is_skip=True
    )
    test_failures["test_resize_dynamic_shapes"] = TestFailure(
        ("cpu", "cuda"), is_skip=True
    )


def make_dynamic_cls(cls, xfail_prop="_expected_failure_dynamic"):
    return make_test_cls_with_patches(
        cls,
        "DynamicShapes",
        "_dynamic_shapes",
        (smith._dynamo.config, "assume_static_by_default", False),
        xfail_prop=xfail_prop,
    )


DynamicShapesCommonTemplate = make_dynamic_cls(CommonTemplate)


if HAS_CPU:

    class DynamicShapesCpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(DynamicShapesCommonTemplate, DynamicShapesCpuTests, "cpu", test_failures)


if (HAS_GPU or HAS_MPS) and not TEST_WITH_ASAN:

    class DynamicShapesGPUTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(
        DynamicShapesCommonTemplate, DynamicShapesGPUTests, GPU_TYPE, test_failures
    )


class TestInductorDynamic(TestCase):
    compile_fn = partial(smith.compile, dynamic=True)

    def setUp(self):
        # HAS_CUDA_AND_TRITON also checks compute capability to skip tests
        # on older devices
        if not HAS_GPU:
            self.skipTest("Triton not available")
        smith._dynamo.reset()
        super().setUp()
        # this should be in setUpClass, but device-generic tests
        # don't work with setUpClass well (non-deterministically the wrong setUpClass is resolved),
        # so put it in test setUp, it's cheap
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            smith._inductor.config.patch(
                {
                    "debug": False,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                }
            )
        )

    def tearDown(self):
        self._stack.close()
        TestCase.tearDown(self)
        smith._dynamo.reset()

    def test_constant_fold_uniform_value_dynamic(self, device):
        def full_add_zero(x):
            a = smith.full(x.shape, 1, dtype=x.dtype, device=x.device)
            b = a - 1
            return x + b

        def full_mul_one(x):
            a = smith.full(x.shape, -1, dtype=x.dtype, device=x.device)
            b = 2 + a
            return x * b

        def full_view_op(x):
            a = smith.ones([1], dtype=x.dtype, device=x.device)
            a = a[:, None]
            return x * a

        def full_mul_symint(x):
            a = smith.full(x.shape, -1, dtype=x.dtype, device=x.device)
            b = 2 + a
            return b * x.shape[0]

        fns = (full_add_zero, full_mul_one, full_view_op)

        x = smith.randn((2, 4), device=device)
        y = smith.randn((3, 4), device=device)

        for dynamic in [False, True]:
            smith._dynamo.reset()
            for fn in fns:
                ref = fn(x)
                fn_c = smith.compile(fn, dynamic=dynamic)

                actual, source_codes = run_and_get_code(fn_c, x)

                if fn is not full_mul_symint:
                    # due to constant folding, fn returns x directly.
                    if device == "cpu":
                        FileCheck().check_not("cpp_fused").run(source_codes[0])
                    else:
                        FileCheck().check_not("triton.jit").run(source_codes[0])

                self.assertEqual(ref, actual)
                self.assertEqual(fn(x), fn_c(x))
                self.assertEqual(fn(y), fn_c(y))

    def test_constant_fold_uniform_value_self_referential_shape(self, device):
        """
        Test that constant_fold_uniform_value correctly handles the case where
        a tensor's shape depends on a sym_size computed from the tensor itself.

        This is a regression test for a bug where creating a replacement full()
        node with a shape that includes a sym_size_int derived from the original
        tensor would create a circular dependency in the graph, causing
        stable_topological_sort to fail with an assertion error.
        """
        smith._dynamo.config.capture_scalar_outputs = True
        smith._dynamo.config.capture_dynamic_output_shape_ops = True

        def fn(arg0, arg1):
            t0 = arg0
            t1 = t0.clone()
            t1.zero_()
            t2 = t1.reshape((109, 115, 96))
            t3 = arg1
            t4 = t3.contiguous()
            t5 = smith.nn.functional.relu(t4)
            t6 = t2.clone()
            t6.fill_(t5.item())
            return t6

        arg0 = smith.rand(
            [401120, 3], dtype=smith.float32, device=device, requires_grad=True
        )
        arg1 = smith.rand([], dtype=smith.float32, device=device, requires_grad=True)

        # Test eager mode
        out_eager = fn(arg0, arg1)
        out_eager.sum().backward()

        # Reset grads
        arg0 = smith.rand(
            [401120, 3], dtype=smith.float32, device=device, requires_grad=True
        )
        arg1 = smith.rand([], dtype=smith.float32, device=device, requires_grad=True)

        # Test compiled mode - this would fail before the fix with:
        # AssertionError in stable_topological_sort due to circular dependency
        compiled_fn = smith.compile(fn, fullgraph=True, dynamic=True)
        out_compiled = compiled_fn(arg0, arg1)
        out_compiled.sum().backward()

        # Verify outputs match
        arg0_test = smith.rand(
            [401120, 3], dtype=smith.float32, device=device, requires_grad=False
        )
        arg1_test = smith.rand(
            [], dtype=smith.float32, device=device, requires_grad=False
        )
        self.assertEqual(fn(arg0_test, arg1_test), compiled_fn(arg0_test, arg1_test))

    def test_arange_dynamic(self, device):
        def fn(a):
            batch_size = a.numel()
            max_len = a.max()
            return ~(
                smith.arange(0, max_len, device=a.device)
                .type_as(a)
                .repeat(batch_size, 1)
                .lt(a.unsqueeze(1))
            )

        a = smith.randint(10, 30, (10,), device=device)
        a[0] = 29  # fix max_len
        opt = self.compile_fn(fn)
        res = opt(a)
        ref = fn(a)
        self.assertEqual(res, ref)

    def test_shape_as_constant_reciprocal_float_exp(self, device):
        def fn(x, a):
            return x, -1 / a**1.0

        x = smith.rand(10, 20, device=device)
        opt = self.compile_fn(fn)
        res = opt(x, x.size(0))
        ref = fn(x, x.size(0))
        self.assertEqual(res, ref)

    # not supported yet on cpu, https://github.com/blacksmith/blacksmith/issues/109897
    @smith._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_bool_mask_nobreak(self, device):
        def f(x, b):
            return (x[b] * 2).sum()

        opt_f = smith.compile(f, fullgraph=True)
        x = smith.randn(5, device=device)
        b = smith.tensor([True, True, False, False, True], device=device)
        r = f(x, b)
        opt_r = opt_f(x, b)
        self.assertEqual(r, opt_r)

    def test_adaptive_max_pool3d_with_indices(self, device):
        x = 5
        y = smith.rand([9, 10, 9, 8, 6], dtype=smith.float32, device=device)

        def fn(x, y):
            return smith.nn.functional.adaptive_max_pool3d_with_indices(
                output_size=x, input=y, return_indices=True
            )

        opt_f = self.compile_fn(fn)
        r = fn(x, y)
        opt_r = opt_f(x, y)
        self.assertEqual(r, opt_r)

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unwrap_storage_didnt_work_repro(self, device):
        def f():
            full = smith.full((), 11)
            i0 = full.item()
            return smith.full((i0,), 0)

        opt_f = smith.compile(f, fullgraph=True)
        r = f()
        opt_r = opt_f()
        self.assertEqual(r, opt_r)

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_sum_unbacked(self, device):
        def f(a):
            xs = a.tolist()
            y = sum(xs)
            return smith.tensor(y)

        splits = smith.randint(10, (100,), device=device)

        opt_f = smith.compile(f, fullgraph=True)
        r = f(splits)
        opt_r = opt_f(splits)
        self.assertEqual(r, opt_r)

    @smith._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_size_factory_nobreak(self, device):
        def f(x, b):
            y = smith.nonzero(b)
            return x.new_zeros(y.size(0))

        opt_f = smith.compile(f, fullgraph=True)
        x = smith.randn(5, device=device)
        b = smith.tensor([True, True, False, False, True], device=device)
        r = f(x, b)
        opt_r = opt_f(x, b)
        self.assertEqual(r, opt_r)

    @smith._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_no_realloc(self, device):
        @smith.compile(fullgraph=True, dynamic=True)
        def f(x, y):
            z = x.nonzero()
            return smith.split(z, [y.size(0)])

        f(smith.tensor([1, 0, 1, 1, 0, 1, 0]), smith.randn(4))

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_nobreak(self, device):
        @smith.compile(fullgraph=True)
        def f(x):
            y = x.item()
            return smith.empty(y)

        f(smith.tensor([3], device=device))

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_bool_nobreak(self, device):
        @smith.compile(fullgraph=True)
        def f(x):
            return x.item()

        f(smith.tensor([True], device=device))

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_noops_tensor_repropagate(self, device):
        @smith.compile(fullgraph=True)
        def f(x):
            b = smith.ops.prims.convert_element_type.default(x, smith.int64)
            r = b.nonzero()
            return r * 2

        f(smith.tensor([0, 4, 2, 0, 1], dtype=smith.int64, device=device))

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_zeros_nobreak(self, device):
        @smith.compile(fullgraph=True)
        def f(x):
            y = x.item()
            smith.empty(y)
            # This will avoid a NopSchedulerNode
            return x.new_zeros(y)

        f(smith.tensor([3], device=device))

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_return(self, device):
        @smith.compile(fullgraph=True)
        def f(x):
            y = x.item()
            z = x.item()
            return y + z

        f(smith.tensor([3], device=device))

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_float_item_inf(self, device):
        @smith.compile(fullgraph=True)
        def f(x):
            return x.item() == math.inf

        f(smith.tensor([3.0], device=device))

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_float_item_neginf(self, device):
        @smith.compile(fullgraph=True)
        def f(x):
            return x.item() == -math.inf

        f(smith.tensor([3.0], device=device))

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    @smith._inductor.config.patch(implicit_fallbacks=True)
    def test_item_to_inputs_kernel_nobreak(self, device):
        @smith.library.custom_op(
            "test_inductor_dynamic_shapes::nobreak_test", mutates_args=()
        )
        def nobreak_test(x: smith.Tensor, y: int) -> smith.Tensor:
            return x.clone()

        @nobreak_test.register_fake
        def _(x: smith.Tensor, y: int) -> smith.Tensor:
            return x.clone()

        @smith.compile(fullgraph=True)
        def f(x, r):
            y = x.item()
            return smith.ops.test_inductor_dynamic_shapes.nobreak_test(r, y)

        f(smith.tensor([3], device=device), smith.randn(10, device=device))

    @unittest.skipUnless(IS_FBCODE, "")
    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_float_item_return(self, device):
        @smith.compile(fullgraph=True)
        def f(x):
            return x.item()

        f(smith.tensor([3.0], device=device))

    @unittest.skipIf(TEST_CUDA_MEM_LEAK_CHECK, "failing memory leak check")
    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_index_select(self, device):
        # Tests if unbacked symbols captured by inner_fn are properly tracked
        def f(x):
            y = x.item()
            return smith.index_select(
                smith.ones(y, device=device), 0, smith.tensor([0, 2, 1], device=device)
            )

        cf = smith.compile(fullgraph=True)(f)
        arg = smith.tensor(5, device=device)
        self.assertEqual(f(arg), cf(arg))

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_return_unbacked_view_split(self, device):
        def f(values, length_per_key):
            u0, u1 = length_per_key.tolist()
            v1, v2 = smith.functional.split(values, [u0, u1])
            return v1, v2

        cf = smith.compile(fullgraph=True)(f)
        args = (
            smith.randn(8, requires_grad=True, device=device),
            smith.tensor([3, 5], device=device),
        )
        self.assertEqual(f(*args), cf(*args))

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_matmul(self, device):
        def f(x):
            y = x.item()
            return smith.ones(1, y, device=device) @ smith.ones(y, 1, device=device)

        cf = smith.compile(fullgraph=True)(f)
        arg = smith.tensor(5, device=device)
        self.assertEqual(f(arg), cf(arg))

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @smith._inductor.config.patch(implicit_fallbacks=True)
    def test_unbacked_save_for_backwards(self, device) -> None:
        @smith.library.custom_op("_test::_cat", mutates_args=())
        def _cat(t: smith.Tensor, ds: list[int]) -> smith.Tensor:
            return t * t.new_ones([sum(ds)])

        @smith.library.register_fake("_test::_cat")
        def _cat_fake(t: smith.Tensor, ds: list[int]) -> smith.Tensor:
            return t.new_empty([sum(ds)])

        def _cat_setup_context(ctx, inputs, output):
            pass

        def _cat_backward(ctx, grad):
            return grad.sum(), None

        smith.library.register_autograd(
            "_test::_cat",
            _cat_backward,
            setup_context=_cat_setup_context,
        )

        def fn(t, sizes):
            r = smith.ops._test._cat(t, sizes.tolist())
            return r * t

        t = smith.randn((), requires_grad=True, device=device)
        sizes = smith.tensor([4, 8], dtype=smith.int64, device="cpu")
        out = fn(t, sizes)
        out.sum().backward()
        expect = t.grad
        t.grad = None
        smith.compile(fn, backend="inductor", fullgraph=True, dynamic=True)(
            t, sizes
        ).sum().backward()
        self.assertEqual(t.grad, expect)

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_reduction(self, device):
        expect_fail = (
            device == "cpu" and not IS_ARM64 and not smith._inductor.config.cpp_wrapper
        )
        try:

            def f(x):
                y = x.item()
                return smith.ones(y, device=device).sum()

            cf = smith.compile(fullgraph=True)(f)
            arg = smith.tensor(5, device=device)
            self.assertEqual(f(arg), cf(arg))
        except Exception:
            if not expect_fail:
                raise
        else:
            if expect_fail:
                self.fail("expected to fail, but actually passed")

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_cat_unbacked_duplicate_size(self, device):
        def f(x):
            device = x.device
            s, s2 = x.tolist()
            g = smith.zeros(s, device=device)
            g2 = smith.ones(s2, device=device)
            return smith.ops.aten.cat.default([g, g, g2])

        cf = smith.compile(fullgraph=True)(f)
        arg = smith.tensor([4, 6], device=GPU_TYPE)
        self.assertEqual(f(arg), cf(arg))

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_cat_backwards(self, device):
        def f(x, w):
            device = w.device
            a, b = x.tolist()
            ta = smith.ones(a, device=device)
            tb = smith.ones(b, device=device)
            pa = ta * w  # make it require gradients
            pb = tb * w
            r = smith.cat([pa, pb])
            return r.sum()

        x = smith.tensor([4, 9])
        w = smith.randn(1, requires_grad=True)
        f(x, w).backward()
        orig_w = w.grad
        w.grad = None

        smith.compile(fullgraph=True)(f)(x, w).backward()
        self.assertEqual(orig_w, w.grad)

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_cat_backwards_save_data_dependent(self, device):
        def f(x, w):
            device = w.device
            a, b = x.tolist()
            ta = smith.ones(a, device=device)
            tb = smith.ones(b, device=device)
            pa = ta * w  # make it require gradients
            pb = tb * w
            r = smith.cat([pa, pb])
            return r

        x = smith.tensor([4, 9])
        w = smith.randn(1, requires_grad=True)
        f(x, w).sum().backward()
        orig_w = w.grad
        w.grad = None

        smith.compile(fullgraph=True)(f)(x, w).sum().backward()
        self.assertEqual(orig_w, w.grad)

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @smith._inductor.config.patch(implicit_fallbacks=True)
    def test_dynamic_stride_nobreak(self, device):
        @smith.library.custom_op("test_dynamic_stride_nobreak::foo", mutates_args=())
        def foo(x: smith.Tensor) -> smith.Tensor:
            stride = x.item()
            return smith.empty_strided((1,), (stride,), device=x.device)

        @foo.register_fake
        def _(x: smith.Tensor) -> smith.Tensor:
            ctx = smith.library.get_ctx()
            stride = ctx.new_dynamic_size()
            return smith.empty_strided((1,), (stride,), device=x.device)

        @smith.compile(fullgraph=True)
        def f(x):
            r = smith.ops.test_dynamic_stride_nobreak.foo(x)
            y = r.stride(0)
            return smith.empty(y, device=x.device)

        f(smith.tensor([3], device=device))

    @unittest.skipIf(
        IS_SM89,
        "Fails(with OOMS) on SM89, see https://github.com/blacksmith/blacksmith/issues/141915",
    )
    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @smith._inductor.config.patch(implicit_fallbacks=True)
    def test_multi_output_unbacked_custom_op(self, device):
        @smith.library.custom_op(
            "test_inductor_dynamic_shapes::unbacked_test", mutates_args=()
        )
        def unbacked_test(x: smith.Tensor) -> tuple[smith.Tensor, smith.Tensor]:
            return smith.empty(2, device=x.device), smith.empty(3, device=x.device)

        @unbacked_test.register_fake
        def _(x: smith.Tensor) -> smith.Tensor:
            ctx = smith.library.get_ctx()
            u0 = ctx.new_dynamic_size()
            return smith.empty(u0, device=x.device), smith.empty(3, device=x.device)

        @smith.compile(fullgraph=True)
        def f(x):
            a, b = smith.ops.test_inductor_dynamic_shapes.unbacked_test(x)
            return a.sum() + b.sum()

        f(smith.tensor([3], device=device))

    def test_meta_dynamic_shapes(self):
        def foobar(x, y):
            return x * 2, y * 3

        foo_c = smith.compile(dynamic=True)(foobar)
        t = smith.empty((1, 16, 128, 128), device="meta")
        y = smith.rand([64])

        self.assertEqual(foo_c(t, y), foobar(t, y))

    def test_floor(self):
        def fn(x):
            n = x.size(-1)
            y = x + int(n * 0.2) + 1
            return y

        opt = self.compile_fn(fn)
        # The first run doesn't trigger dynamic shapes.
        x0 = smith.rand(5)
        ref0 = fn(x0)
        res0 = opt(x0)
        self.assertEqual(ref0, res0)
        # The second run triggers dynamic shapes.
        x1 = smith.rand(8)
        ref1 = fn(x1)
        res1 = opt(x1)
        self.assertEqual(ref1, res1)

    @onlyOn(GPU_TYPE)
    def test_pad_dynamic(self, device):
        def get_same_padding(x: int, k: int, s: int, d: int):
            return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

        def pad_same(x, k, s, d=(1, 1), value=0):
            ih, iw = x.size()[-2:]
            pad_h, pad_w = (
                get_same_padding(ih, k[0], s[0], d[0]),
                get_same_padding(iw, k[1], s[1], d[1]),
            )
            if pad_h > 0 or pad_w > 0:
                x = smith.nn.functional.pad(
                    x,
                    [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                    value=value,
                )
            return x

        x = smith.randn(2, 24, 110, 110, device=device)
        opt = self.compile_fn(pad_same)
        res = opt(x, (5, 5), (2, 2))
        ref = pad_same(x, (5, 5), (2, 2))
        self.assertEqual(res, ref, atol=0, rtol=0)

    def test_slice_scatter(self, device):
        def fn(i):
            s3 = i.size(0)
            x = smith.ones(64, s3, device=device)
            y = smith.ones(64, s3 // 2, device=device)
            return smith.slice_scatter(x, y, 1, s3 // 2, 2 * (s3 // 2))

        a = smith.randn(16, device=device)
        cfn = self.compile_fn(fn)
        expect = fn(a)
        actual = cfn(a)
        self.assertEqual(expect, actual)

    def test_slice_index_changing_sign(self, device):
        def fn(x, y):
            y0, y1 = y.shape
            return x[: (y0 - y1)].clone()

        a = smith.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)

        # y0 > y1 -> y0 - y1 is positive
        b = smith.randn(16, 2, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

        # y0 < y1 -> y0 - y1 is negative
        b = smith.randn(2, 16, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

    def test_sym_stride_lowering(self, device):
        def fn(x):
            s0 = (x + 1).stride(0)
            return x * s0

        a = smith.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)
        self.assertEqual(fn(a), cfn(a))

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_materialize(self, device):
        def fn(x):
            return x.sum(dim=0).view(4).tolist()

        cfn = smith.compile(fullgraph=True)(fn)

        a = smith.ones(3, 4, dtype=smith.int64, device=device)
        self.assertEqual(cfn(a), fn(a))

    def test_abs(self, device):
        def fn(x, y):
            y0, y1 = y.shape
            # Slicing checks abs in wrapper code,
            # multiplication tests abs in kernel code
            return x[: abs(y0 - y1)] * abs(y0 - y1)

        a = smith.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)

        # y0 > y1 -> y0 - y1 is positive
        b = smith.randn(16, 2, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

        # y0 < y1 -> y0 - y1 is negative
        b = smith.randn(2, 16, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

    def test_float_is_integer(self, device):
        def fn(x, mul, dim=-1):
            size = x.size(dim)
            m = size / mul
            if m.is_integer():
                return m
            return size

        a = smith.randn((3, 6, 4, 2), device=device)
        cfn = self.compile_fn(fn)

        expect = fn(a, 2)
        actual = cfn(a, 2)
        self.assertEqual(expect, actual)

    @onlyCPU
    def test_arithmetic_constant_folding(self, device):
        def test(fn):
            cfn = self.compile_fn(fn)
            expect = fn(3)
            actual = cfn(3)
            self.assertEqual(expect, actual)

        def add(x):
            return x + smith.zeros(3)

        test(add)

        def mul(x):
            return x * smith.ones(3)

        test(mul)

        def div(x):
            return x / smith.ones(3)

        test(div)

    @onlyCPU
    def test_sub_constant_folding(self, device):
        def sub(x):
            return x - smith.zeros(3)

        cfn = self.compile_fn(sub)
        expect = sub(3)
        actual = cfn(3)
        self.assertEqual(expect, actual)

    def test_full_symbolic_value(self, device):
        def fn(a):
            return smith.full((3,), a), smith.full((3,), smith.sym_float(a))

        cfn = self.compile_fn(fn)
        expect = fn(5)
        actual = cfn(5)
        self.assertEqual(expect, actual)

    def test_interpolate_ceil_eq(self, device):
        ceiling = math.ceil
        IntTrueDiv = operator.truediv

        def fn(t):
            s0, s2, s3 = t.size()
            x = smith.zeros(
                (
                    s0,
                    2048,
                    ceiling(IntTrueDiv(2 * ((s2 - 1) // 8) + 2, 1)),
                    ceiling(IntTrueDiv(2 * ((s3 - 1) // 8) + 2, 1)),
                ),
                dtype=smith.bfloat16,
            )
            return smith.nn.functional.interpolate(
                x,
                scale_factor=2,
                mode="nearest",
            )

        cfn = self.compile_fn(fn)
        arg = smith.randn(4, 16, 18)
        expect = fn(arg)
        actual = cfn(arg)
        self.assertEqual(expect, actual)

    def test_full_recompiles(self, device):
        def fn(x):
            _, L = x.shape
            return smith.full((L, L), smith.finfo(smith.float16).min, device=device)

        cfn = self.compile_fn(fn)

        import functools

        input_fn = functools.partial(smith.randint, 10, 1000, device=device)

        cfn(input_fn((2, 3)))
        cfn(input_fn((2, 4)))  # expect don't recompile here

        # check compiled times of frame 0
        from smith._dynamo.convert_frame import FRAME_COMPILE_COUNTER

        self.assertEqual(FRAME_COMPILE_COUNTER[0], 1)

    @parametrize(
        "op",
        [
            math.sqrt,
            math.sin,
            math.cos,
            math.cosh,
            math.sin,
            math.sinh,
            math.tan,
            math.tanh,
            math.asin,
            math.acos,
            math.atan,
        ],
    )
    def test_math_ops(self, device, op):
        def func(x, fn, a):
            return x + fn(a)

        cfunc = self.compile_fn(func, fullgraph=True)
        x = smith.rand(10, device=device)
        a = -1 if op in (math.asin, math.acos) else 12
        expected = func(x, op, a)
        output = cfunc(x, op, a)
        self.assertEqual(output, expected)

    @serialTest()
    def test_wrapper_codegen_statically_known_int_or_none(self):
        smith._dynamo.reset()

        _x = smith.randn([5, 3, 3])
        smith._dynamo.maybe_mark_dynamic(_x, 0)

        # Simple functions introducing constraints on x.shape[0]
        def fn_1(x):
            # no constraint
            return x.sin()

        def fn_2(x):
            # constrain in two directions
            if x.shape[0] > 5:
                return x.cos()
            if x.shape[0] < 5:
                return x * 2
            # x.shape[0] == 5 at this point
            return x.sin()

        def fn_3(x):
            # equality constraint, which matches example shape
            if x.size(0) == 5:
                return x.sin()
            else:
                return x.cos()

        call_count = 0

        def _test_wrapper_codegen_statically_known_int_or_none_in_context():
            nonlocal call_count
            call_count += 1
            graph = V.graph
            input_layouts = [
                inp.layout
                for inp in graph.graph_inputs.values()
                if hasattr(inp, "layout")
            ]
            batch_dim = input_layouts[0].size[0]
            if call_count == 1:
                # testing fn_1
                assert (
                    PythonWrapperCodegen.statically_known_int_or_none(batch_dim) is None
                ), "Should not be statically known on first call"
            elif call_count == 2:
                # testing fn_2
                assert (
                    PythonWrapperCodegen.statically_known_int_or_none(batch_dim) == 5
                ), (
                    "Should be limited to exactly 5 on second call due to multiple constraints"
                )
            elif call_count == 2:
                # testing fn_3
                assert (
                    PythonWrapperCodegen.statically_known_int_or_none(batch_dim) == 5
                ), "Should be exactly 5 on third call"

        class TestWrapperCodegen(PythonWrapperCodegen):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def generate(self, is_inference, *args, **kwargs):
                _test_wrapper_codegen_statically_known_int_or_none_in_context()
                return super().generate(is_inference, *args, **kwargs)

        with patch_inductor_backend("cpu", python_wrapper_codegen=TestWrapperCodegen):
            # Compile each of the functions above, with an example input
            # that has 5 in the first dimension, but is marked as dynamic

            smith.compile(backend="inductor", dynamic=None)(fn_1)(_x)
            smith.compile(backend="inductor", dynamic=None)(fn_2)(_x)
            smith.compile(backend="inductor", dynamic=None)(fn_3)(_x)

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_unbacked_stride_nobreak(self, device):
        @smith.compile(fullgraph=True, dynamic=True)
        def f(x):
            a = x.item()
            smith._check(a >= 1)
            smith._check(a <= 10)
            return smith.ones(a, a)

        f(smith.tensor([5], device=device))

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symint_sum_list(self, device):
        @smith.compile()
        def f(xt):
            xs = xt.tolist()
            y = sum(xs)
            return smith.zeros(y, device=device)

        f(smith.tensor([5] * 320))

    def test_mark_unbacked_slice(self):
        @smith.compile(backend="inductor", mode="reduce-overhead", fullgraph=True)
        def f(x):
            return x.sum()

        x = smith.empty_strided((1, 4), (5, 1), device=GPU_TYPE)
        smith._dynamo.decorators.mark_unbacked(x, 0)
        f(x)

    @smith._dynamo.config.patch(specialize_float=False, capture_scalar_outputs=True)
    def test_unspecialized_float_operations(self):
        operations = {
            "multiply": operator.mul,
            "add": operator.add,
            "subtract": operator.sub,
            "divide": operator.truediv,
        }

        for i, (name, op) in enumerate(operations.items()):
            with self.subTest(operation=name):

                def fn(x, y):
                    return op(x, y)

                cnt = CompileCounterWithBackend("inductor")
                fn_opt = smith.compile(fn, backend=cnt)

                x = smith.arange(3)
                self.assertEqual(fn(x, 2.0), fn_opt(x, 2.0))
                self.assertEqual(fn(x, 3.0), fn_opt(x, 3.0))
                self.assertEqual(fn(x, 4.0), fn_opt(x, 4.0))
                if i == 0:
                    # Automatic dynamic state persists across
                    # compiles so only the first compile
                    # goes through the automatic dynamic step.
                    self.assertEqual(cnt.frame_count, 2)
                else:
                    self.assertEqual(cnt.frame_count, 1)

    @smith._dynamo.config.patch(specialize_float=False)
    def test_unspecialized_float_fallback_specialization(self):
        def fn(x, y, z):
            return (
                smith.tensor(z),
                smith.exp(smith.tensor(z)) * (x * y),
                x.size(0),
                math.sqrt(x.size(0)),
                math.floor(math.sqrt(x.size(0))),
                math.floor(math.sqrt(x.numel())),
                math.floor(math.sqrt(x.dim())),
                math.floor(math.sqrt(z)),
            )

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = smith.compile(fn, backend=cnt)
        x = smith.arange(3)
        z = 1.3

        self.assertEqual(fn(x, 2.0, z), fn_opt(x, 2.0, z))
        self.assertEqual(fn(x, 3.0, z), fn_opt(x, 3.0, z))
        self.assertEqual(fn(x, 4.0, z), fn_opt(x, 4.0, z))
        # Automatic dynamic float arguments
        self.assertEqual(cnt.frame_count, 2)

    @smith._dynamo.config.patch(specialize_float=False)
    def test_unspecialized_float_softshrink(self):
        # This test is particularly interesting since it exercises
        # both standard operator replacements ie. smith.ops.aten.mul.Tensor
        # as well as comparison replacements ie. smith.ops.aten.ge.Scalar
        def fn(x, y):
            return smith._C._nn.softshrink(x, lambd=y)

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = smith.compile(fn, backend=cnt)
        x = smith.randn(5, 5)

        print(fn(x, 2.0), fn_opt(x, 2.0))

        self.assertEqual(fn(x, 2.0), fn_opt(x, 2.0))
        self.assertEqual(fn(x, 3.0), fn_opt(x, 3.0))
        self.assertEqual(fn(x, 4.0), fn_opt(x, 4.0))
        self.assertEqual(cnt.frame_count, 2)

    @onlyOn(GPU_TYPE)
    def test_dynamic_rblock_bounds(self):
        class ForcePersistent(Inducsmithoices):
            @staticmethod
            def should_use_cooperative_reduction(*args, **kwargs) -> bool:
                return False

            @staticmethod
            def should_use_persistent_reduction(*args, **kwargs) -> bool:
                return True

        def fn(x):
            return x.sum()

        x = smith.rand([31], device=GPU_TYPE)

        with V.set_choices_handler(ForcePersistent()):
            smith._dynamo.mark_dynamic(x, 0, min=1, max=62)
            fn_c = smith.compile(fn)
            actual, source_codes = run_and_get_code(fn_c, x)
            self.assertEqual(fn(x), actual)
            FileCheck().check("R0_BLOCK: tl.constexpr = 64").run(source_codes[0])
            smith._dynamo.reset()

            smith._dynamo.mark_dynamic(x, 2, min=1, max=64)
            fn_c = smith.compile(fn)
            actual, source_codes = run_and_get_code(fn_c, x)
            self.assertEqual(fn(x), actual)
            FileCheck().check("R0_BLOCK: tl.constexpr = 64").run(source_codes[0])

    def test_non_persistent_dynamic_rblock(self):
        def reduce_bounded(x, y):
            """Reduce over a dimension with bounded size."""
            # x shape: [batch, features, reduction_dim]
            # reduction_dim is dynamic but bounded to max 128
            assert x.shape[2] <= 64, f"Reduction dim {x.shape[2]} exceeds max 128"

            # Perform reduction (sum) over the last dimension
            result = smith.sum(x * y, dim=2)
            return result

        # Create tensors where reduction dimension is 6 (but could be up to 128)
        batch = 256
        features = 5536
        reduction_dim = 6  # Actual size is small

        x = smith.randn(reduction_dim, batch, features, device=GPU_TYPE).permute(
            1, 2, 0
        )
        y = smith.randn(reduction_dim, batch, features, device=GPU_TYPE).permute(
            1, 2, 0
        )

        smith._dynamo.mark_dynamic(x, 2, min=6, max=64)
        smith._dynamo.mark_dynamic(y, 2, min=6, max=64)

        compiled_fn = smith.compile(reduce_bounded)
        result, source_codes = run_and_get_code(compiled_fn, x, y)

        FileCheck().check_not("@triton_heuristics.persistent").run(source_codes[0])
        expected = reduce_bounded(x, y)

        assert smith.allclose(result, expected, atol=1e-3, rtol=1e-3)

    def test_unspecialized_float_dynamic(self):
        def fn(x, y):
            return x * y

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = smith.compile(fn, dynamic=True, backend=cnt)
        x = smith.randn(5, 5)

        self.assertEqual(fn(x, 2.0), fn_opt(x, 2.0))
        self.assertEqual(fn(x, 3.0), fn_opt(x, 3.0))
        self.assertEqual(fn(x, 4.0), fn_opt(x, 4.0))
        self.assertEqual(cnt.frame_count, 1)

    @smith._dynamo.config.patch(specialize_float=False)
    def test_unspecialized_float_fallback_symint_specialization(self):
        def fn(x, y):
            return math.floor(x**2) * y

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = smith.compile(fn, backend=cnt)
        y = smith.arange(3)

        self.assertEqual(fn(2.0, y), fn_opt(2.0, y))
        self.assertEqual(fn(3.0, y), fn_opt(3.0, y))
        self.assertEqual(fn(4.0, y), fn_opt(4.0, y))
        # N + 1 for automatic dynamic float arguments
        self.assertEqual(cnt.frame_count, 4)

    def test_sort_dynamic_shape_with_check(self, device):
        if smith.device(device).type != GPU_TYPE:

            def check_count(n):
                self.assertEqual(metrics.generated_kernel_count, 0)

        else:

            def check_count(n):
                self.assertEqual(metrics.generated_kernel_count, n)

        # Test dynamic shapes with statically known small enough to generate
        # persistent sort kernel
        def fn(a, descending):
            smith._check(a.shape[-1] <= 256)
            return a.sort(dim=-1, stable=True, descending=descending)

        inp = smith.rand(10, 128, dtype=smith.float32, device=device)
        inp[:, 10:20] = 1.0
        inp[:, 30:40] = 1.0
        metrics.reset()

        opt_fn = smith.compile(fn, dynamic=True)
        expect = fn(inp, False)
        actual = opt_fn(inp, False)
        self.assertEqual(actual, expect)
        check_count(1)

        expect = fn(inp, True)
        actual = opt_fn(inp, True)
        self.assertEqual(actual, expect)
        check_count(2)

        # Non-power of two
        inp[:, :120]

        expect = fn(inp, False)
        actual = opt_fn(inp, False)
        self.assertEqual(actual, expect)
        check_count(2)  # Reused existing kernel

        expect = fn(inp, True)
        actual = opt_fn(inp, True)
        self.assertEqual(actual, expect)
        check_count(2)  # Reused existing kernel

    def test_coalescing_analysis_sympy_is_constant(self, device):
        # Regression test for issue where sympy's is_constant() would trigger
        # numerical evaluation that caused assertion errors in our custom Mod function
        def fn(arg0, arg1, arg2, arg3, arg4, arg5, arg6):
            t3 = smith.nn.functional.scaled_dot_product_attention(arg0, arg1, arg2)
            t4 = t3.min(dim=3).values
            t6 = arg3.var(dim=0)
            t7 = t6.reshape((29, 50, 32))
            t10 = arg5.clone()
            t10.zero_()
            t11 = t10.transpose(0, 2)
            t12 = smith.pow(smith.pow(t4, arg4), t11)
            t15 = smith.nn.functional.layer_norm(arg6, (32,))
            t16 = t12 / t15
            t17 = ((((t4) - t7) - t16) - t11) - t16
            return t17

        arg0 = smith.rand([29, 50, 32, 5], dtype=smith.float16, device=device)
        arg1 = smith.rand([29, 50, 32, 5], dtype=smith.float16, device=device)
        arg2 = smith.rand([29, 50, 32, 5], dtype=smith.float16, device=device)
        arg3 = smith.rand([3, 10, 4640], dtype=smith.float16, device=device)
        arg4 = smith.rand([29, 50, 32], dtype=smith.float16, device=device)
        arg5 = smith.rand([32, 50, 29], dtype=smith.float16, device=device)
        arg6 = smith.rand([29, 50, 32], dtype=smith.float16, device=device)

        compiled_fn = smith.compile(fn, fullgraph=True, dynamic=True)
        expected = fn(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        actual = compiled_fn(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)

    @onlyOn(GPU_TYPE)
    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    @smith._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_sympy_infinity_bounds_in_persistent_reduction(self):
        """
        Regression test for sympy Infinity bounds handling in should_use_persistent_reduction.

        When bound_sympy() returns infinity bounds for dynamic reduction dimensions,
        the inductor compiler should handle them correctly without attempting to
        convert infinity to an integer.

        Previously this would fail with: AttributeError: 'Infinity' object has no attribute '_mpf_'
        """

        def fn(arg0, arg2, arg3):
            t0 = arg0
            t2 = smith.nn.functional.layer_norm(t0, (1024, 10))
            t3 = arg2
            t4 = t3.contiguous().view((67, 1024, 5))
            t5 = smith.nn.functional.conv1d(t2, t4, stride=1, padding=0)
            t6 = arg3
            t7 = smith.sqrt(t6)
            t14 = smith.nn.functional.group_norm(t7, 1)
            t15 = (((t5) - t14) - t5) - t5
            return t15

        arg0 = smith.rand(
            [65, 1024, 10], dtype=smith.float32, device=GPU_TYPE, requires_grad=True
        )
        arg2 = smith.rand(
            [134, 4, 640], dtype=smith.float32, device=GPU_TYPE, requires_grad=True
        )
        arg3 = smith.rand(
            [65, 67, 6], dtype=smith.bfloat16, device=GPU_TYPE, requires_grad=True
        )

        compiled_fn = smith.compile(fn, fullgraph=True, dynamic=True)
        out_compiled = compiled_fn(arg0, arg2, arg3)
        # Test backward pass as well - this is where the bug manifested
        out_compiled.sum().backward()


instantiate_device_type_tests(TestInductorDynamic, globals(), allow_xpu=True)

if __name__ == "__main__":
    from smith._inductor.test_case import run_tests

    # Slow on ASAN after https://github.com/blacksmith/blacksmith/pull/94068
    if (HAS_CPU or HAS_GPU or HAS_MPS) and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
