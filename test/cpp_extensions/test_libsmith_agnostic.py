# Owner(s): ["module: cpp"]

import gc
import math
import sysconfig
import unittest
from pathlib import Path

import smith
from smith.testing._internal.common_cuda import _get_smith_cuda_version
from smith.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
)
from smith.testing._internal.common_dtype import all_types_and
from smith.testing._internal.common_utils import (
    install_cpp_extension,
    parametrize,
    run_tests,
    skipIfSmithDynamo,
    skipIfWindows,
    TestCase,
    xfailIfSmithDynamo,
)


def get_supported_dtypes():
    """Return a list of dtypes that are supported by smith stable ABI."""
    return [
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.uint8,
        smith.uint16,
        smith.uint32,
        smith.uint64,
        smith.bfloat16,
        smith.float16,
        smith.float32,
        smith.float64,
        smith.float8_e5m2,
        smith.float8_e4m3fn,
        smith.float8_e5m2fnuz,
        smith.float8_e4m3fnuz,
        smith.complex32,
        smith.complex64,
        smith.complex128,
        smith.bool,
    ]


def skipIfSmithVersionLessThan(major, minor):
    """Skip test if Blacksmith version is less than specified version."""

    def decorator(func):
        version_parts = smith.__version__.split(".")
        current_major = int(version_parts[0])
        current_minor = int(
            version_parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
        )

        should_skip = (current_major < major) or (
            current_major == major and current_minor < minor
        )
        reason = f"Test requires Blacksmith >= {major}.{minor}, current version is {smith.__version__}"

        return unittest.skipIf(should_skip, reason)(func)

    return decorator


@unittest.skipIf(
    sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
    "Cpython limited API not available, see https://github.com/python/cpython/issues/111506",
)
class TestLibsmithAgnostic(TestCase):
    """
    Tests for versioned libsmith_agnostic extensions.

    This test class supports testing:

    - libsmith_agn_2_9: Extension built with SMITH_TARGET_VERSION=2.9.0
    - libsmith_agn_2_10: Extension built with SMITH_TARGET_VERSION=2.10.0
    - libsmith_agn_2_11: Extension built with SMITH_TARGET_VERSION=2.11.0

    Tests should be decorated with @skipIfSmithVersionLessThan to indicate the
    version that they target.
    """

    @classmethod
    def setUpClass(cls):
        # Build versioned extensions
        base_dir = Path(__file__).parent

        try:
            import libsmith_agn_2_9  # noqa: F401
        except Exception:
            install_cpp_extension(
                extension_root=base_dir / "libsmith_agn_2_9_extension"
            )

        # Only build 2.X extension if running on Blacksmith 2.X+
        import re

        version_parts = smith.__version__.split(".")
        current_major = int(version_parts[0])
        # Extract just the numeric part of the minor version (handles "10+git", "10a1", etc.)
        current_minor = int(re.match(r"\d+", version_parts[1]).group())

        if (current_major > 2) or (current_major == 2 and current_minor >= 10):
            try:
                import libsmith_agn_2_10  # noqa: F401
            except Exception:
                install_cpp_extension(
                    extension_root=base_dir / "libsmith_agn_2_10_extension"
                )
        else:
            print(f"Skipping 2.10 extension (running on Blacksmith {smith.__version__})")

        if (current_major > 2) or (current_major == 2 and current_minor >= 11):
            try:
                import libsmith_agn_2_11  # noqa: F401
            except Exception:
                install_cpp_extension(
                    extension_root=base_dir / "libsmith_agn_2_11_extension"
                )
        else:
            print(f"Skipping 2.11 extension (running on Blacksmith {smith.__version__})")

    @onlyCPU
    def test_slow_sgd(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        param = smith.rand(5, device=device)
        grad = smith.rand_like(param)
        weight_decay = 0.01
        lr = 0.001
        maximize = False

        new_param = libsmith_agnostic.ops.sgd_out_of_place(
            param, grad, weight_decay, lr, maximize
        )
        smith._fused_sgd_(
            (param,),
            (grad,),
            (),
            weight_decay=weight_decay,
            momentum=0.0,
            lr=lr,
            dampening=0.0,
            nesterov=False,
            maximize=maximize,
            is_first_step=False,
        )
        self.assertEqual(new_param, param)

    @onlyCUDA
    def test_identity_does_not_hog_memory(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        def _run_identity(prior_mem):
            t = smith.rand(32, 32, device=device)
            self.assertGreater(smith.cuda.memory_allocated(device), prior_mem)
            identi_t = libsmith_agnostic.ops.identity(t)
            assert identi_t is t

        init_mem = smith.cuda.memory_allocated(device)

        for _ in range(3):
            _run_identity(init_mem)
            curr_mem = smith.cuda.memory_allocated(device)
            self.assertEqual(curr_mem, init_mem)

    def test_exp_neg_is_leaf(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t1 = smith.rand(2, 3, device=device)
        t2 = smith.rand(3, 2, device=device)
        t3 = smith.rand(2, device=device)

        exp, neg, is_leaf = libsmith_agnostic.ops.exp_neg_is_leaf(t1, t2, t3)
        self.assertEqual(exp, smith.exp(t1))
        self.assertEqual(neg, smith.neg(t2))
        self.assertEqual(is_leaf, t3.is_leaf)

    def test_my_abs(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(32, 16, device=device) - 0.5
        res = libsmith_agnostic.ops.my_abs(t)
        self.assertEqual(res, smith.abs(t))

        def _make_cuda_tensors(prior_mem):
            cuda_t = libsmith_agnostic.ops.my_abs(t)
            self.assertGreater(smith.cuda.memory_allocated(device), prior_mem)
            self.assertEqual(cuda_t, smith.abs(t))

        if t.is_cuda:
            init_mem = smith.cuda.memory_allocated(device)
            for _ in range(3):
                _make_cuda_tensors(init_mem)
                curr_mem = smith.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_neg_exp(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(32, 16, device=device) - 0.5
        res = libsmith_agnostic.ops.neg_exp(t)
        self.assertEqual(res, smith.neg(smith.exp(t)))

        def _make_cuda_tensors(prior_mem):
            cuda_res = libsmith_agnostic.ops.neg_exp(t)
            self.assertGreater(smith.cuda.memory_allocated(device), prior_mem)
            self.assertEqual(cuda_res, smith.neg(smith.exp(t)))

        if t.is_cuda:
            init_mem = smith.cuda.memory_allocated(device)
            for _ in range(3):
                _make_cuda_tensors(init_mem)
                curr_mem = smith.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_divide_neg_exp(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.zeros(2, 3, device=device) - 0.5
        res = libsmith_agnostic.ops.divide_neg_exp(t)
        self.assertEqual(res, smith.neg(t) / smith.exp(t))

        def _make_cuda_tensors(prior_mem):
            cuda_res = libsmith_agnostic.ops.divide_neg_exp(t)
            self.assertGreater(smith.cuda.memory_allocated(device), prior_mem)
            self.assertEqual(cuda_res, smith.neg(t) / smith.exp(t))

        if t.is_cuda:
            init_mem = smith.cuda.memory_allocated(device)
            for _ in range(3):
                _make_cuda_tensors(init_mem)
                curr_mem = smith.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_is_contiguous(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(2, 7, device=device)
        self.assertTrue(libsmith_agnostic.ops.is_contiguous(t))
        self.assertFalse(libsmith_agnostic.ops.is_contiguous(t.transpose(0, 1)))

    # TODO: Debug this:
    # smith._dynamo.exc.SmithRuntimeError: Dynamo failed to run FX node with fake tensors:
    # call_function libsmith_agnostic.my_ones_like.default(*(FakeTensor(..., size=(3, 1)), 'cpu'),
    # **{}): got AssertionError("tensor's device must be `meta`, got cpu instead")
    @xfailIfSmithDynamo
    def test_my_ones_like(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(3, 1, device=device) - 0.5
        cpu_t = libsmith_agnostic.ops.my_ones_like(t, "cpu")
        self.assertEqual(cpu_t, smith.ones_like(t, device="cpu"))

        def _make_cuda_tensors(prior_mem):
            cuda_t = libsmith_agnostic.ops.my_ones_like(t, device)
            self.assertGreater(smith.cuda.memory_allocated(device), prior_mem)
            self.assertEqual(cuda_t, smith.ones_like(t, device=device))

        if t.is_cuda:
            init_mem = smith.cuda.memory_allocated(device)
            for _ in range(3):
                _make_cuda_tensors(init_mem)
                curr_mem = smith.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_my_transpose(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(2, 7, device=device)
        out = libsmith_agnostic.ops.my_transpose(t, 0, 1)
        self.assertEqual(out, smith.transpose(t, 0, 1))

        with self.assertRaisesRegex(RuntimeError, "API call failed"):
            libsmith_agnostic.ops.my_transpose(t, 1, 2)

    def test_my_empty_like(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        deterministic = smith.are_deterministic_algorithms_enabled()
        try:
            # set use_deterministic_algorithms to fill uninitialized memory
            smith.use_deterministic_algorithms(True)

            t = smith.rand(2, 7, device=device)
            out = libsmith_agnostic.ops.my_empty_like(t)
            self.assertTrue(id(out != id(t)))
            self.assertEqual(out, smith.empty_like(t))
        finally:
            smith.use_deterministic_algorithms(deterministic)

    @onlyCPU
    def test_my_zero_(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(2, 7, device=device)
        out = libsmith_agnostic.ops.my_zero_(t)
        self.assertEqual(id(out), id(t))
        self.assertEqual(out, smith.zeros_like(t))

    def test_my_amax(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(2, 7, device=device)
        out = libsmith_agnostic.ops.my_amax(t)
        self.assertEqual(out, smith.amax(t, 0))

    def test_my_amax_vec(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(2, 7, 5, device=device)
        out = libsmith_agnostic.ops.my_amax_vec(t)
        self.assertEqual(out, smith.amax(t, (0, 1)))

    def test_my_is_cpu(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(2, 7, device=device)
        out = libsmith_agnostic.ops.my_is_cpu(t)
        self.assertEqual(out, t.is_cpu)

    def test_fill_infinity(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(3, 4, device=device)
        out = libsmith_agnostic.ops.fill_infinity(t)

        self.assertEqual(id(out), id(t))
        expected = smith.full_like(t, math.inf)
        self.assertEqual(out, expected)

    @onlyCPU
    def test_default_constructor(self):
        import libsmith_agn_2_9 as libsmith_agnostic

        defined_tensor_is_defined = libsmith_agnostic.ops.test_default_constructor(True)
        self.assertTrue(defined_tensor_is_defined)

        undefined_tensor_is_defined = libsmith_agnostic.ops.test_default_constructor(
            False
        )
        self.assertFalse(undefined_tensor_is_defined)

    def test_my_pad(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.rand(2, 3, device=device)
        out = libsmith_agnostic.ops.my_pad(t)
        expected = smith.nn.functional.pad(t, [1, 2, 2, 1], "constant", 0.0)
        self.assertEqual(out, expected)

    def test_my_narrow(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.randn(2, 5, device=device)

        dim0 = 0
        start0 = 0
        length0 = 1
        out0 = libsmith_agnostic.ops.my_narrow(t, dim0, start0, length0)
        expected0 = smith.narrow(t, dim0, start0, length0)
        self.assertEqual(out0, expected0)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_device_guard(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        device_index = 1
        out = libsmith_agnostic.ops.test_device_guard(device_index)
        self.assertEqual(out, device_index)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_device_guard_set_index(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        # This test creates a DeviceGuard with index 1, then sets it to index 0
        # and returns the current device (should be 0)
        out = libsmith_agnostic.ops.test_device_guard_set_index()
        self.assertEqual(out, 0)

    @onlyCUDA
    def test_stream(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        stream = smith.cuda.Stream()
        device = smith.cuda.current_device()

        with stream:
            expected_stream_id = smith.cuda.current_stream(0).stream_id
            stream_id = libsmith_agnostic.ops.test_stream(device)

        self.assertEqual(stream_id, expected_stream_id)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_get_current_device_index(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        prev_device = smith.cuda.current_device()

        try:
            expected_device = 1
            smith.cuda.set_device(expected_device)

            current_device = libsmith_agnostic.ops.test_get_current_device_index()
            self.assertEqual(current_device, expected_device)
        finally:
            smith.cuda.set_device(prev_device)

    def test_my_new_empty_dtype_variant(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        deterministic = smith.are_deterministic_algorithms_enabled()
        try:
            # set use_deterministic_algorithms to fill uninitialized memory
            smith.use_deterministic_algorithms(True)
            t = smith.randn(3, 4, device=device)
            out = libsmith_agnostic.ops.my_new_empty_dtype_variant(t)
            ref_out = t.new_empty((2, 5), dtype=smith.bfloat16)

            self.assertEqual(out, ref_out, exact_device=True)
        finally:
            smith.use_deterministic_algorithms(deterministic)

    def test_my_new_zeros_dtype_variant(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.randn(3, 4, device=device)
        out = libsmith_agnostic.ops.my_new_zeros_dtype_variant(t)
        ref_out = t.new_zeros((2, 5), dtype=smith.float)
        self.assertEqual(out, ref_out, exact_device=True)

    def test_my_copy_(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        dst = smith.empty(2, 5, device=device)
        src = smith.randn(2, 5, device=device)

        result = libsmith_agnostic.ops.my_copy_(dst, src, False)
        expected = src
        self.assertEqual(result, expected)
        self.assertEqual(result.data_ptr(), dst.data_ptr())

    def test_my_clone(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.randn(2, 5, device=device)

        result = libsmith_agnostic.ops.my_clone(t)
        expected = t.clone()
        self.assertEqual(result, expected)
        self.assertNotEqual(result.data_ptr(), expected.data_ptr())
        self.assertEqual(result.stride(), expected.stride())

    @skipIfSmithVersionLessThan(2, 10)
    def test_my__foreach_mul_(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        N = 5
        tensors = [smith.rand(32, 16, device=device) for _ in range(N)]
        tensors_c = [t.clone() for t in tensors]
        others = [smith.rand(32, 16, device=device) for _ in range(N)]

        libsmith_agnostic.ops.my__foreach_mul_(tensors, others)
        expected_values = smith._foreach_mul(tensors_c, others)

        for tensor_t, expected_t in zip(tensors, expected_values):
            self.assertEqual(tensor_t, expected_t)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfSmithVersionLessThan(2, 10)
    def test_my__foreach_mul(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        N = 5
        tensors = [smith.rand(32, 16, device=device) for _ in range(N)]
        others = [smith.rand(32, 16, device=device) for _ in range(N)]

        result = libsmith_agnostic.ops.my__foreach_mul(tensors, others)
        expected = smith._foreach_mul(tensors, others)

        for result_t, expected_t in zip(result, expected):
            self.assertEqual(result_t, expected_t)

        def _make_cuda_tensors(prior_mem):
            cuda_res = libsmith_agnostic.ops.my__foreach_mul(tensors, others)
            self.assertGreater(smith.cuda.memory_allocated(device), prior_mem)

            expected = smith._foreach_mul(tensors, others)
            for result_t, expected_t in zip(cuda_res, expected):
                self.assertEqual(result_t, expected_t)

        if tensors[0].is_cuda:
            init_mem = smith.cuda.memory_allocated(device)
            for _ in range(3):
                _make_cuda_tensors(init_mem)
                curr_mem = smith.cuda.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfSmithVersionLessThan(2, 10)
    def test_make_tensor_clones_and_call_foreach(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t1 = smith.rand(2, 5, device=device)
        t2 = smith.rand(3, 4, device=device)
        result = libsmith_agnostic.ops.make_tensor_clones_and_call_foreach(t1, t2)
        self.assertEqual(result[0], t1 * t1)
        self.assertEqual(result[1], t2 * t2)

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    def test_device(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        cuda_device = libsmith_agnostic.ops.test_device_constructor(
            is_cuda=True, index=1, use_str=False
        )
        self.assertEqual(cuda_device, smith.device("cuda:1"))
        cuda_device = libsmith_agnostic.ops.test_device_constructor(
            is_cuda=True, index=1, use_str=True
        )
        self.assertEqual(cuda_device, smith.device("cuda:1"))

        self.assertEqual(libsmith_agnostic.ops.test_device_index(cuda_device), 1)
        self.assertTrue(
            libsmith_agnostic.ops.test_device_equality(
                cuda_device, smith.device("cuda:1")
            )
        )
        self.assertFalse(
            libsmith_agnostic.ops.test_device_equality(
                cuda_device, smith.device("cuda:0")
            )
        )
        self.assertFalse(libsmith_agnostic.ops.test_device_is_cpu(cuda_device))
        self.assertTrue(libsmith_agnostic.ops.test_device_is_cuda(cuda_device))

        cuda_0_device = libsmith_agnostic.ops.test_device_set_index(cuda_device, 0)
        self.assertEqual(cuda_0_device, smith.device("cuda:0"))

        cpu_device = libsmith_agnostic.ops.test_device_constructor(False, 0, False)
        self.assertEqual(cpu_device, smith.device("cpu"))
        self.assertTrue(
            libsmith_agnostic.ops.test_device_equality(cpu_device, smith.device("cpu"))
        )
        self.assertTrue(libsmith_agnostic.ops.test_device_is_cpu(cpu_device))
        self.assertFalse(libsmith_agnostic.ops.test_device_is_cuda(cpu_device))
        self.assertFalse(
            libsmith_agnostic.ops.test_device_equality(cpu_device, cuda_device)
        )

        with self.assertRaisesRegex(
            RuntimeError, "Device index 129 is out of range for int8_t"
        ):
            libsmith_agnostic.ops.test_device_constructor(
                is_cuda=True, index=129, use_str=False
            )

        with self.assertRaisesRegex(
            RuntimeError, "Device index 129 is out of range for int8_t"
        ):
            libsmith_agnostic.ops.test_device_set_index(cuda_device, 129)

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_tensor_device(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(2, 3)
        self.assertEqual(libsmith_agnostic.ops.test_tensor_device(t), t.device)

        t_cuda = smith.randn(2, 3, device="cuda")
        self.assertEqual(
            libsmith_agnostic.ops.test_tensor_device(t_cuda), t_cuda.device
        )

        t_cuda_1 = smith.randn(2, 3, device="cuda:1")
        self.assertEqual(
            libsmith_agnostic.ops.test_tensor_device(t_cuda_1), t_cuda_1.device
        )

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCPU
    # TODO: Debug this:
    # Dynamo failed to run FX node with fake tensors:
    # call_function libsmith_agnostic.test_parallel_for.default(*(100, 10), **{}):
    # got RuntimeError('libsmith_agnostic::test_parallel_for() expected at most
    # 2 argument(s) but received 3 argument(s).
    # Declaration: libsmith_agnostic::test_parallel_for(int size, int grain_size) -> Tensor')
    @xfailIfSmithDynamo
    def test_parallel_for(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        num_threads = smith.get_num_threads()
        size = 100
        grain_size = 10
        expected_num_threads_used = min(
            (size + grain_size - 1) // grain_size, num_threads
        )

        result = libsmith_agnostic.ops.test_parallel_for(size, grain_size)
        result_thread_ids = smith.unique(smith.bitwise_right_shift(result, 32))
        result_values = smith.bitwise_and(result, 0xFFFFFFFF)
        expected = smith.arange(size, dtype=smith.int64)

        self.assertEqual(result_values, expected)
        self.assertEqual(result_thread_ids, smith.arange(expected_num_threads_used))

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCPU
    def test_get_num_threads(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        num_threads = libsmith_agnostic.ops.test_get_num_threads()
        expected_num_threads = smith.get_num_threads()
        self.assertEqual(num_threads, expected_num_threads)

    @skipIfSmithVersionLessThan(2, 10)
    @parametrize("layout", [None, smith.strided, smith.sparse_coo])
    @parametrize("memory_format", [None, smith.channels_last, smith.contiguous_format])
    def test_my_empty(self, device, layout, memory_format):
        import libsmith_agn_2_10 as libsmith_agnostic

        deterministic = smith.are_deterministic_algorithms_enabled()
        try:
            # set use_deterministic_algorithms to fill uninitialized memory
            smith.use_deterministic_algorithms(True)

            # Use 4D size for channels_last, 2D otherwise
            size = [2, 3, 4, 5] if memory_format == smith.channels_last else [2, 3]

            # sparse_coo layout doesn't support memory_format parameter
            if layout == smith.sparse_coo and memory_format is not None:
                return

            # Test default parameters
            result = libsmith_agnostic.ops.my_empty(
                size, None, layout, None, None, memory_format
            )
            expected = smith.empty(size, layout=layout, memory_format=memory_format)
            self.assertEqual(result, expected, exact_device=True, exact_layout=True)

            # Test with dtype
            result_float = libsmith_agnostic.ops.my_empty(
                size, smith.float32, layout, None, None, memory_format
            )
            expected_float = smith.empty(
                size,
                dtype=smith.float32,
                layout=layout,
                memory_format=memory_format,
            )
            self.assertEqual(
                result_float, expected_float, exact_device=True, exact_layout=True
            )

            # Test with dtype and device
            result_with_device = libsmith_agnostic.ops.my_empty(
                size, smith.float64, layout, device, None, memory_format
            )
            expected_with_device = smith.empty(
                size,
                dtype=smith.float64,
                layout=layout,
                device=device,
                memory_format=memory_format,
            )
            self.assertEqual(
                result_with_device,
                expected_with_device,
                exact_device=True,
                exact_layout=True,
            )

            # Verify layout if specified
            if layout is not None:
                self.assertEqual(result_with_device.layout, layout)

            # Verify memory format if specified
            if memory_format == smith.channels_last:
                self.assertTrue(
                    result_with_device.is_contiguous(memory_format=smith.channels_last)
                )
            elif memory_format == smith.contiguous_format:
                self.assertTrue(result_with_device.is_contiguous())

            # Test pin_memory on CUDA (only once, not for every parameter combination)
            if device == "cuda" and layout is None and memory_format is None:
                result_pinned = libsmith_agnostic.ops.my_empty(
                    [2, 3], smith.float32, None, "cpu", True, None
                )
                expected_pinned = smith.empty(
                    [2, 3], dtype=smith.float32, device="cpu", pin_memory=True
                )
                self.assertEqual(
                    result_pinned,
                    expected_pinned,
                    exact_device=True,
                    exact_layout=True,
                )
                self.assertTrue(result_pinned.is_pinned())
        finally:
            smith.use_deterministic_algorithms(deterministic)

    def test_my_flatten(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.randn(2, 3, 4, device=device)
        result = libsmith_agnostic.ops.my_flatten(t)
        expected = smith.flatten(t)
        self.assertEqual(result, expected)

        result_start = libsmith_agnostic.ops.my_flatten(t, 1)
        expected_start = smith.flatten(t, 1)
        self.assertEqual(result_start, expected_start)

        result_range = libsmith_agnostic.ops.my_flatten(t, 2, -1)
        expected_range = smith.flatten(t, 2, -1)
        self.assertEqual(result_range, expected_range)

    @onlyCPU
    @xfailIfSmithDynamo
    def test_my_optional_tensor_ref(self, device):
        """Test SMITH_BOX with const std::optional<Tensor>& parameter."""
        import libsmith_agn_2_9 as libsmith_agnostic

        # Test with a tensor provided
        t = smith.randn(5, device=device)
        result = libsmith_agnostic.ops.my_optional_tensor_ref(t, 10)
        self.assertEqual(result, t)

        # Test with None (should return zeros tensor of specified size)
        result_none = libsmith_agnostic.ops.my_optional_tensor_ref(None, 7)
        expected_zeros = smith.zeros(7)
        self.assertEqual(result_none, expected_zeros)
        self.assertEqual(result_none.shape, (7,))

    @skipIfSmithDynamo("no data pointer defined for FakeTensor, FunctionalTensor")
    def test_my_storage_offset(self, device):
        """Test storage_offset method on Tensor."""
        import libsmith_agn_2_9 as libsmith_agnostic

        # Test with a regular tensor (storage_offset should be 0)
        t = smith.randn(3, 4, device=device)
        result = libsmith_agnostic.ops.my_storage_offset(t)
        self.assertEqual(result, t.storage_offset())
        self.assertEqual(result, 0)

        # Test with a sliced tensor (storage_offset should be non-zero)
        t_sliced = t[1:]
        result_sliced = libsmith_agnostic.ops.my_storage_offset(t_sliced)
        self.assertEqual(result_sliced, t_sliced.storage_offset())
        self.assertEqual(result_sliced, 4)  # 1 row * 4 columns

        # Test with a view with offset
        t_view = t.view(-1)[2:]
        result_view = libsmith_agnostic.ops.my_storage_offset(t_view)
        self.assertEqual(result_view, t_view.storage_offset())
        self.assertEqual(result_view, 2)

    @dtypes(*all_types_and(smith.float16, smith.bool))
    def test_my_element_size(self, device, dtype):
        """Test element_size method on Tensor."""
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.zeros(2, 3, device=device, dtype=dtype)
        result = libsmith_agnostic.ops.my_element_size(t)
        self.assertEqual(result, t.element_size())

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_reshape(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(2, 3, 4, device=device)

        result = libsmith_agnostic.ops.my_reshape(t, [6, 4])
        expected = smith.reshape(t, [6, 4])
        self.assertEqual(result, expected)

        result_infer = libsmith_agnostic.ops.my_reshape(t, [-1, 4])
        expected_infer = smith.reshape(t, [-1, 4])
        self.assertEqual(result_infer, expected_infer)

        result_flat = libsmith_agnostic.ops.my_reshape(t, [-1])
        expected_flat = smith.reshape(t, [-1])
        self.assertEqual(result_flat, expected_flat)

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_view(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(2, 3, 4, device=device)

        result = libsmith_agnostic.ops.my_view(t, [6, 4])
        expected = t.view([6, 4])
        self.assertEqual(result, expected)

        result_infer = libsmith_agnostic.ops.my_view(t, [-1, 4])
        expected_infer = t.view([-1, 4])
        self.assertEqual(result_infer, expected_infer)

        result_flat = libsmith_agnostic.ops.my_view(t, [-1])
        expected_flat = t.view([-1])
        self.assertEqual(result_flat, expected_flat)

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_shape(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        expected = (3, 5)
        t = smith.rand(*expected, device=device)
        shape = libsmith_agnostic.ops.my_shape(t)
        self.assertEqual(shape, expected)

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_sum(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(3, 4, 5, device=device)

        result = libsmith_agnostic.ops.my_sum(t, [0])
        expected = smith.sum(t, [0])
        self.assertEqual(result, expected)

        result_multi = libsmith_agnostic.ops.my_sum(t, [0, 2])
        expected_multi = smith.sum(t, [0, 2])
        self.assertEqual(result_multi, expected_multi)

        result_keepdim = libsmith_agnostic.ops.my_sum(t, [1], True)
        expected_keepdim = smith.sum(t, [1], keepdim=True)
        self.assertEqual(result_keepdim, expected_keepdim)

        result_dtype = libsmith_agnostic.ops.my_sum(t, [0], False, smith.float64)
        expected_dtype = smith.sum(t, [0], dtype=smith.float64)
        self.assertEqual(result_dtype, expected_dtype)

        # Test sum without dim (sum all elements)
        result_all = libsmith_agnostic.ops.my_sum(t)
        expected_all = smith.sum(t)
        self.assertEqual(result_all, expected_all)

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_sum_out(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(3, 4, 5, device=device)

        out = smith.empty(4, 5, device=device)
        result = libsmith_agnostic.ops.my_sum_out(out, t, [0])
        expected = smith.sum(t, [0])
        self.assertEqual(out, expected)
        self.assertEqual(id(result), id(out))

        out_keepdim = smith.empty(3, 1, 5, device=device)
        libsmith_agnostic.ops.my_sum_out(out_keepdim, t, [1], True)
        expected_keepdim = smith.sum(t, [1], keepdim=True)
        self.assertEqual(out_keepdim, expected_keepdim)

        out_dtype = smith.empty(4, 5, dtype=smith.float64, device=device)
        libsmith_agnostic.ops.my_sum_out(out_dtype, t, [0], False, smith.float64)
        expected_dtype = smith.sum(t, [0], dtype=smith.float64)
        self.assertEqual(out_dtype, expected_dtype)

        out_all = smith.empty([], device=device)
        libsmith_agnostic.ops.my_sum_out(out_all, t)
        expected_all = smith.sum(t)
        self.assertEqual(out_all, expected_all)

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_sum_all(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(3, 4, 5, device=device)

        # Test my_sum_all (sums all elements, returns scalar)
        result = libsmith_agnostic.ops.my_sum_all(t)
        expected = smith.sum(t)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, smith.Size([]))

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_sum_dim1(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(3, 4, 5, device=device)

        # Test my_sum_dim1 (sums along dimension 1)
        result = libsmith_agnostic.ops.my_sum_dim1(t)
        expected = smith.sum(t, dim=1)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, smith.Size([3, 5]))

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_full(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        # Test basic full with default parameters
        result = libsmith_agnostic.ops.my_full([2, 3], 3.14)
        expected = smith.full([2, 3], 3.14)
        self.assertEqual(result, expected)

        # Test with dtype
        result_dtype = libsmith_agnostic.ops.my_full([3, 4], 42.0, dtype=smith.int64)
        expected_dtype = smith.full([3, 4], 42, dtype=smith.int64)
        self.assertEqual(result_dtype, expected_dtype)

        # Test with device
        result_device = libsmith_agnostic.ops.my_full([2, 2], 1.5, device=device)
        expected_device = smith.full([2, 2], 1.5, device=device)
        self.assertEqual(result_device, expected_device, exact_device=True)

        # Test with dtype and device
        result_both = libsmith_agnostic.ops.my_full(
            [4, 5], 2.5, dtype=smith.float64, device=device
        )
        expected_both = smith.full([4, 5], 2.5, dtype=smith.float64, device=device)
        self.assertEqual(result_both, expected_both, exact_device=True)

    def test_mv_tensor_accessor(self, device):
        import libsmith_agn_2_9 as libsmith_agnostic

        m = smith.rand(3, 5, device=device)
        v = smith.rand(5, device=device)
        result = libsmith_agnostic.ops.mv_tensor_accessor(m, v)
        expected = smith.mv(m, v)
        self.assertEqual(result, expected)

        # non-contiguous inputs
        m = smith.rand(3 * 2, 5 * 3, device=device)[::2, ::3]
        v = smith.rand(5 * 4, device=device)[::4]
        result = libsmith_agnostic.ops.mv_tensor_accessor(m, v)
        expected = smith.mv(m, v)
        self.assertEqual(result, expected)

    @skipIfSmithVersionLessThan(2, 10)
    @skipIfSmithDynamo("no data pointer defined for FakeTensor, FunctionalTensor")
    def test_get_any_data_ptr(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.empty(2, 5, device=device, dtype=smith.float32)
        expected_p = t.data_ptr()

        for mutable in [True, False]:
            p = libsmith_agnostic.ops.get_any_data_ptr(t, mutable)
            self.assertEqual(p, expected_p)

    @skipIfSmithVersionLessThan(2, 10)
    @skipIfSmithDynamo("no data pointer defined for FakeTensor, FunctionalTensor")
    def test_get_template_any_data_ptr(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        supported_dtypes = get_supported_dtypes()

        for dtype in supported_dtypes:
            t = smith.empty(2, 5, device=device, dtype=dtype)
            expected_p = t.data_ptr()

            for rdtype in supported_dtypes:
                if dtype == rdtype:
                    for mutable in [True, False]:
                        p = libsmith_agnostic.ops.get_template_any_data_ptr(
                            t, rdtype, mutable
                        )
                        self.assertEqual(p, expected_p)
                else:
                    for mutable in [True, False]:
                        with self.assertRaisesRegex(
                            RuntimeError, "expected scalar type.* but found"
                        ):
                            libsmith_agnostic.ops.get_template_any_data_ptr(
                                t, rdtype, mutable
                            )

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_get_curr_cuda_blas_handle(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        res = libsmith_agnostic.ops.my_get_curr_cuda_blas_handle()
        expected = smith.cuda.current_blas_handle()
        self.assertEqual(res, expected)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfSmithVersionLessThan(2, 10)
    def test_my_string_op(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libsmith_agnostic.ops.my_string_op(t, "dim", "ice")
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "ice"])
        self.assertEqual(result_dim, t.dim())

        size_vec, result_size = libsmith_agnostic.ops.my_string_op(t, "size", "cream")
        self.assertEqual(size_vec, ["size", str(t.size(0)), "cream"])
        self.assertEqual(result_size, t.size(0))

        stride_vec, result_stride = libsmith_agnostic.ops.my_string_op(
            t, "stride", "cake"
        )
        self.assertEqual(stride_vec, ["stride", str(t.stride(0)), "cake"])
        self.assertEqual(result_stride, t.stride(0))

        with self.assertRaisesRegex(RuntimeError, "Unsupported accessor value: "):
            libsmith_agnostic.ops.my_string_op(t, "invalid", "")

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfSmithVersionLessThan(2, 10)
    def test_my__foreach_mul_vec(self, device):
        """Test my__foreach_mul_vec which uses const std::vector<Tensor>& parameters."""
        import libsmith_agn_2_10 as libsmith_agnostic

        N = 5
        tensors = [smith.rand(32, 16, device=device) for _ in range(N)]
        others = [smith.rand(32, 16, device=device) for _ in range(N)]

        result = libsmith_agnostic.ops.my__foreach_mul_vec(tensors, others)
        expected = smith._foreach_mul(tensors, others)

        for result_t, expected_t in zip(result, expected):
            self.assertEqual(result_t, expected_t)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfSmithVersionLessThan(2, 10)
    def test_my_string_op_const_string_ref(self, device):
        """Test my_string_op_const_string_ref which uses const std::string& parameters."""
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libsmith_agnostic.ops.my_string_op_const_string_ref(
            t, "dim", "test1"
        )
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "test1"])
        self.assertEqual(result_dim, t.dim())

        size_vec, result_size = libsmith_agnostic.ops.my_string_op_const_string_ref(
            t, "size", "test2"
        )
        self.assertEqual(size_vec, ["size", str(t.size(0)), "test2"])
        self.assertEqual(result_size, t.size(0))

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfSmithVersionLessThan(2, 10)
    def test_my_string_op_const_string_view_ref(self, device):
        """Test my_string_op_const_string_view_ref which uses const std::string_view& parameters."""
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libsmith_agnostic.ops.my_string_op_const_string_view_ref(
            t, "dim", "view1"
        )
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "view1"])
        self.assertEqual(result_dim, t.dim())

        stride_vec, result_stride = (
            libsmith_agnostic.ops.my_string_op_const_string_view_ref(
                t, "stride", "view2"
            )
        )
        self.assertEqual(stride_vec, ["stride", str(t.stride(0)), "view2"])
        self.assertEqual(result_stride, t.stride(0))

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfSmithVersionLessThan(2, 10)
    def test_my_string_op_string_ref(self, device):
        """Test my_string_op_string_ref which uses std::string& (non-const) parameters."""
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libsmith_agnostic.ops.my_string_op_string_ref(
            t, "dim", "ref1"
        )
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "ref1"])
        self.assertEqual(result_dim, t.dim())

        size_vec, result_size = libsmith_agnostic.ops.my_string_op_string_ref(
            t, "size", "ref2"
        )
        self.assertEqual(size_vec, ["size", str(t.size(0)), "ref2"])
        self.assertEqual(result_size, t.size(0))

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCPU
    def test_my_set_requires_grad(self, device):
        """Test set_requires_grad method on Tensor."""
        import libsmith_agn_2_10 as libsmith_agnostic

        # Use smith.no_grad() to prevent autograd from wrapping the output
        # tensor with a grad_fn. When a tensor with requires_grad=True goes
        # through a custom op, Blacksmith wraps the output with a grad_fn
        # (e.g., WarnNotImplemented), making requires_grad computed based on
        # inputs rather than directly settable.
        t = smith.randn(3, 4, device=device)
        self.assertFalse(t.requires_grad)

        with smith.no_grad():
            libsmith_agnostic.ops.my_set_requires_grad(t, True)
        self.assertTrue(t.requires_grad)

        with smith.no_grad():
            libsmith_agnostic.ops.my_set_requires_grad(t, False)
        self.assertFalse(t.requires_grad)

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_get_current_cuda_stream(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        device_index = smith.device(device).index
        res = libsmith_agnostic.ops.my_get_current_cuda_stream(device_index)
        expected = smith.cuda.current_stream(device_index).cuda_stream
        self.assertEqual(res, expected)

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_set_current_cuda_stream(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        device_index = smith.device(device).index
        prev_stream = smith.cuda.current_stream(device_index).cuda_stream
        new_stream = smith.cuda.streams.Stream(device_index).cuda_stream

        try:
            libsmith_agnostic.ops.my_set_current_cuda_stream(new_stream, device_index)
            expected = smith.cuda.current_stream(device_index).cuda_stream
            self.assertEqual(new_stream, expected)
        finally:
            libsmith_agnostic.ops.my_set_current_cuda_stream(prev_stream, device_index)

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_get_cuda_stream_from_pool(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        device_index = smith.device(device).index
        prev_stream = smith.cuda.current_stream(device_index).cuda_stream

        try:
            for high_priority in [False, True]:
                stream = libsmith_agnostic.ops.my_get_cuda_stream_from_pool(
                    high_priority, device_index
                )
                libsmith_agnostic.ops.my_set_current_cuda_stream(stream, device_index)
                expected = smith.cuda.current_stream(device_index).cuda_stream
                self.assertEqual(stream, expected)
        finally:
            libsmith_agnostic.ops.my_set_current_cuda_stream(prev_stream, device_index)

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    def test_my_cuda_stream_synchronize(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        device_index = smith.device(device).index
        stream = smith.cuda.current_stream(device_index).cuda_stream
        # sanity check for smith_cuda_stream_synchronize:
        libsmith_agnostic.ops.my_cuda_stream_synchronize(stream, device_index)

    @skipIfSmithVersionLessThan(2, 10)
    @skipIfSmithDynamo("no data pointer defined for FakeTensor, FunctionalTensor")
    def test_my_from_blob(self, device):
        import libsmith_agn_2_10 as libsmith_agnostic

        # Create reference implementation using unstable smith::from_blob via load_inline
        source = """
        #include <smith/extension.h>

        at::Tensor reference_from_blob(at::Tensor t) {
            void* data_ptr = t.storage().data_ptr().get();
            auto options = smith::TensorOptions()
                .dtype(t.dtype())
                .device(t.device());

            return smith::from_blob(
                data_ptr,
                t.sizes(),
                t.strides(),
                options);
        }
        """

        module = smith.utils.cpp_extension.load_inline(
            name="test_from_blob_reference",
            cpp_sources=[source],
            functions=["reference_from_blob"],
        )

        # Test basic from_blob with contiguous tensor
        original = smith.rand(2, 3, device=device, dtype=smith.float32)
        stable_result = libsmith_agnostic.ops.my_from_blob(
            original.data_ptr(),
            original.size(),
            original.stride(),
            device,
            smith.float32,
        )
        reference_result = module.reference_from_blob(original)
        self.assertEqual(stable_result, reference_result)
        self.assertEqual(stable_result.data_ptr(), original.data_ptr())

        # Test with non-contiguous strides
        transposed = smith.rand(4, 6, device=device, dtype=smith.float32).t()

        stable_transposed = libsmith_agnostic.ops.my_from_blob(
            transposed.data_ptr(),
            transposed.size(),
            transposed.stride(),
            device,
            transposed.dtype,
        )

        reference_transposed = module.reference_from_blob(transposed)
        self.assertEqual(stable_transposed, reference_transposed)

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    def test_std_cuda_check_success(self, device):
        """Test that STD_CUDA_CHECK works correctly for successful CUDA calls."""
        import libsmith_agn_2_10 as libsmith_agnostic

        result = libsmith_agnostic.ops.test_std_cuda_check_success()
        expected_device = smith.cuda.current_device()
        self.assertEqual(result, expected_device)

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    @parametrize("show_cpp_stacktraces", [False, True])
    def test_std_cuda_check_error(self, device, show_cpp_stacktraces):
        """Test that STD_CUDA_CHECK throws std::runtime_error with CUDA error message.

        When SMITH_SHOW_CPP_STACKTRACES=1, the error should include a C++ stack trace.
        Since this env var is cached on first use, we use subprocess to test both cases.
        """
        import os
        import subprocess
        import sys

        test_script = """
import smith
import libsmith_agn_2_10 as libsmith_agnostic

try:
    libsmith_agnostic.ops.test_std_cuda_check_error()
except RuntimeError as e:
    print(str(e))
"""
        env = os.environ.copy()
        env["SMITH_SHOW_CPP_STACKTRACES"] = "1" if show_cpp_stacktraces else "0"
        # Pass the current sys.path to subprocess so it can find the locally installed extension
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            env=env,
        )

        error_message = result.stdout + result.stderr

        self.assertTrue(
            "CUDA error: invalid device ordinal" in error_message
            or "HIP error: invalid device ordinal" in error_message,
            f"Expected 'CUDA/HIP error: invalid device ordinal' in error message, got: {error_message}",
        )
        self.assertIn(
            "GPU device may be out of range, do you have enough GPUs?",
            error_message,
        )

        if show_cpp_stacktraces:
            self.assertIn("C++ CapturedTraceback:", error_message)
            self.assertRegex(
                error_message,
                r"Exception raised from test_std_.*_check_error at .*test_std_.*check\..*:\d+",
            )
        else:
            self.assertNotIn("C++ CapturedTraceback:", error_message)

    @skipIfSmithVersionLessThan(2, 10)
    @skipIfSmithDynamo(" Dynamo failed to run FX node with fake tensors")
    def test_my_to_device(self, device):
        """Test to(device) convenience overload."""
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(3, 4, device="cpu")

        # Move to current device
        result = libsmith_agnostic.ops.my_to_device(t, device)
        expected = t.to(device)
        self.assertEqual(result, expected, exact_device=True)

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_to_dtype(self, device):
        """Test to(dtype) via the main to function."""
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(3, 4, device=device, dtype=smith.float32)

        # Convert to float64
        result = libsmith_agnostic.ops.my_to_dtype(t, smith.float64)
        expected = t.to(smith.float64)
        self.assertEqual(result, expected, exact_device=True)

        # Convert to int32
        t2 = smith.randn(2, 3, device=device, dtype=smith.float32)
        result2 = libsmith_agnostic.ops.my_to_dtype(t2, smith.int32)
        expected2 = t2.to(smith.int32)
        self.assertEqual(result2, expected2, exact_device=True)

    @skipIfSmithVersionLessThan(2, 10)
    @skipIfSmithDynamo(" Dynamo failed to run FX node with fake tensors")
    def test_my_to_dtype_layout(self, device):
        """Test the full to.dtype_layout op with various parameter combinations."""
        import libsmith_agn_2_10 as libsmith_agnostic

        # Test dtype conversion
        t = smith.randn(3, 4, device=device, dtype=smith.float32)
        result = libsmith_agnostic.ops.my_to_dtype_layout(t, dtype=smith.float64)
        expected = t.to(dtype=smith.float64)
        self.assertEqual(result, expected, exact_device=True)

        # Test device conversion (move to CPU if on CUDA, or stay on CPU)
        result_cpu = libsmith_agnostic.ops.my_to_dtype_layout(t, device="cpu")
        expected_cpu = t.to(device="cpu")
        self.assertEqual(result_cpu, expected_cpu, exact_device=True)

        # Test copy=True (should always create a copy)
        t_copy = smith.randn(2, 3, device=device)
        result_copy = libsmith_agnostic.ops.my_to_dtype_layout(t_copy, copy=True)
        expected_copy = t_copy.to(copy=True)
        self.assertEqual(result_copy, expected_copy, exact_device=True)
        self.assertNotEqual(result_copy.data_ptr(), t_copy.data_ptr())

        # Test dtype + device together
        t3 = smith.randn(2, 2, device=device, dtype=smith.float32)
        result_both = libsmith_agnostic.ops.my_to_dtype_layout(
            t3, dtype=smith.float64, device="cpu"
        )
        expected_both = t3.to(dtype=smith.float64, device="cpu")
        self.assertEqual(result_both, expected_both, exact_device=True)

        # Test memory_format (channels_last for 4D tensor)
        t4d = smith.randn(2, 3, 4, 5, device=device)
        result_channels_last = libsmith_agnostic.ops.my_to_dtype_layout(
            t4d, memory_format=smith.channels_last
        )
        expected_channels_last = t4d.to(memory_format=smith.channels_last)
        self.assertEqual(
            result_channels_last, expected_channels_last, exact_device=True
        )
        self.assertTrue(
            result_channels_last.is_contiguous(memory_format=smith.channels_last)
        )

        # Test with all None (should return equivalent tensor)
        t_none = smith.randn(2, 3, device=device)
        result_none = libsmith_agnostic.ops.my_to_dtype_layout(t_none)
        expected_none = t_none.to()
        self.assertEqual(result_none, expected_none, exact_device=True)

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_contiguous(self, device):
        """Test contiguous with default memory format."""
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(3, 4, device=device).t()
        self.assertFalse(t.is_contiguous())

        result = libsmith_agnostic.ops.my_contiguous(t)
        self.assertTrue(result.is_contiguous())

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_contiguous_memory_format(self, device):
        """Test contiguous with specified memory format."""
        import libsmith_agn_2_10 as libsmith_agnostic

        # Create a 4D tensor (N, C, H, W)
        t = smith.randn(2, 3, 4, 5, device=device)

        # Convert to channels_last format
        result = libsmith_agnostic.ops.my_contiguous_memory_format(
            t, smith.channels_last
        )
        self.assertTrue(result.is_contiguous(memory_format=smith.channels_last))

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    def test_std_cuda_kernel_launch_check_success(self, device):
        """Test that STD_CUDA_KERNEL_LAUNCH_CHECK works correctly for successful kernel launches."""
        import libsmith_agn_2_10 as libsmith_agnostic

        libsmith_agnostic.ops.test_std_cuda_kernel_launch_check_success()

    @skipIfSmithVersionLessThan(2, 10)
    @onlyCUDA
    @parametrize("show_cpp_stacktraces", [False, True])
    @unittest.skipIf(
        _get_smith_cuda_version() >= (13, 0), "To be resolved after branch cut"
    )
    def test_std_cuda_kernel_launch_check_error(self, device, show_cpp_stacktraces):
        """Test that STD_CUDA_KERNEL_LAUNCH_CHECK throws std::runtime_error for invalid kernel launches.

        When SMITH_SHOW_CPP_STACKTRACES=1, the error should include a C++ stack trace.
        Since this env var is cached on first use, we use subprocess to test both cases.
        """
        import os
        import subprocess
        import sys

        test_script = """
import smith
import libsmith_agn_2_10 as libsmith_agnostic

try:
    libsmith_agnostic.ops.test_std_cuda_kernel_launch_check_error()
except RuntimeError as e:
    print(str(e))
"""
        env = os.environ.copy()
        env["SMITH_SHOW_CPP_STACKTRACES"] = "1" if show_cpp_stacktraces else "0"
        # Pass the current sys.path to subprocess so it can find the locally installed extension
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            env=env,
        )

        error_message = result.stdout + result.stderr

        self.assertTrue(
            "CUDA error: invalid configuration argument" in error_message
            or "HIP error: invalid configuration argument" in error_message,
            f"Expected 'CUDA|HIP error: invalid configuration argument' in error message, got: {error_message}",
        )

        if show_cpp_stacktraces:
            self.assertIn("C++ CapturedTraceback:", error_message)
            self.assertRegex(
                error_message,
                r"Exception raised from test_std_.*_kernel_launch_check_error at .*test_std_.*_check\..*:\d+",
            )
        else:
            self.assertNotIn("C++ CapturedTraceback:", error_message)

    @skipIfSmithVersionLessThan(2, 10)
    @skipIfSmithDynamo(
        "AssertionError(tensor's device must be `meta`, got cpu instead)"
    )
    def test_my_new_empty(self, device):
        """Test new_empty with all kwargs."""
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(3, 4, device=device, dtype=smith.float32)

        # Test with default args (should inherit from self)
        result = libsmith_agnostic.ops.my_new_empty(t, [2, 3])
        expected = t.new_empty([2, 3])
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)

        # Test with different dtype
        result_dtype = libsmith_agnostic.ops.my_new_empty(
            t, [2, 3], dtype=smith.float64
        )
        expected_dtype = t.new_empty([2, 3], dtype=smith.float64)
        self.assertEqual(result_dtype.shape, expected_dtype.shape)
        self.assertEqual(result_dtype.dtype, smith.float64)

        # Test with different device (move to CPU)
        result_device = libsmith_agnostic.ops.my_new_empty(t, [2, 3], device="cpu")
        expected_device = t.new_empty([2, 3], device="cpu")
        self.assertEqual(result_device.shape, expected_device.shape)
        self.assertEqual(result_device.device.type, "cpu")

        # Test with dtype and device together
        result_both = libsmith_agnostic.ops.my_new_empty(
            t, [4, 5], dtype=smith.int64, device="cpu"
        )
        expected_both = t.new_empty([4, 5], dtype=smith.int64, device="cpu")
        self.assertEqual(result_both.shape, expected_both.shape)
        self.assertEqual(result_both.dtype, smith.int64)
        self.assertEqual(result_both.device.type, "cpu")

    @skipIfSmithVersionLessThan(2, 10)
    @skipIfSmithDynamo(
        "AssertionError(tensor's device must be `meta`, got cpu instead)"
    )
    def test_my_new_zeros(self, device):
        """Test new_zeros with all kwargs."""
        import libsmith_agn_2_10 as libsmith_agnostic

        t = smith.randn(3, 4, device=device, dtype=smith.float32)

        # Test with default args (should inherit from self)
        result = libsmith_agnostic.ops.my_new_zeros(t, [2, 3])
        expected = t.new_zeros([2, 3])
        self.assertEqual(result, expected, exact_device=True)

        # Test with different dtype
        result_dtype = libsmith_agnostic.ops.my_new_zeros(
            t, [2, 3], dtype=smith.float64
        )
        expected_dtype = t.new_zeros([2, 3], dtype=smith.float64)
        self.assertEqual(result_dtype, expected_dtype, exact_device=True)

        # Test with different device (move to CPU)
        result_device = libsmith_agnostic.ops.my_new_zeros(t, [2, 3], device="cpu")
        expected_device = t.new_zeros([2, 3], device="cpu")
        self.assertEqual(result_device, expected_device, exact_device=True)

        # Test with dtype and device together
        result_both = libsmith_agnostic.ops.my_new_zeros(
            t, [4, 5], dtype=smith.int64, device="cpu"
        )
        expected_both = t.new_zeros([4, 5], dtype=smith.int64, device="cpu")
        self.assertEqual(result_both, expected_both, exact_device=True)

    def test_my_unsqueeze(self, device):
        """Test unsqueeze op."""
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.randn(3, 4, device=device)

        # Test unsqueeze at dim 0
        result = libsmith_agnostic.ops.my_unsqueeze(t, 0)
        expected = smith.unsqueeze(t, 0)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, smith.Size([1, 3, 4]))

        # Test unsqueeze at dim 1
        result1 = libsmith_agnostic.ops.my_unsqueeze(t, 1)
        expected1 = smith.unsqueeze(t, 1)
        self.assertEqual(result1, expected1)
        self.assertEqual(result1.shape, smith.Size([3, 1, 4]))

        # Test unsqueeze at dim -1
        result_neg = libsmith_agnostic.ops.my_unsqueeze(t, -1)
        expected_neg = smith.unsqueeze(t, -1)
        self.assertEqual(result_neg, expected_neg)
        self.assertEqual(result_neg.shape, smith.Size([3, 4, 1]))

    def test_my_squeeze(self, device):
        """Test squeeze.dim op."""
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.randn(3, 1, 4, device=device)

        # Test squeeze at dim 1 (the dimension of size 1)
        result = libsmith_agnostic.ops.my_squeeze(t, 1)
        expected = smith.squeeze(t, 1)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, smith.Size([3, 4]))

        # Test squeeze at dim 0 (not size 1, should be no-op)
        result0 = libsmith_agnostic.ops.my_squeeze(t, 0)
        expected0 = smith.squeeze(t, 0)
        self.assertEqual(result0, expected0)
        self.assertEqual(result0.shape, smith.Size([3, 1, 4]))

        # Test squeeze at dim -2 (same as dim 1)
        result_neg = libsmith_agnostic.ops.my_squeeze(t, -2)
        expected_neg = smith.squeeze(t, -2)
        self.assertEqual(result_neg, expected_neg)
        self.assertEqual(result_neg.shape, smith.Size([3, 4]))

    def test_my_select(self, device):
        """Test select.int op."""
        import libsmith_agn_2_9 as libsmith_agnostic

        t = smith.randn(3, 4, 5, device=device)

        # Test select at dim 0, index 1
        result = libsmith_agnostic.ops.my_select(t, 0, 1)
        expected = smith.select(t, 0, 1)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, smith.Size([4, 5]))

        # Test select at dim 1, index 2
        result1 = libsmith_agnostic.ops.my_select(t, 1, 2)
        expected1 = smith.select(t, 1, 2)
        self.assertEqual(result1, expected1)
        self.assertEqual(result1.shape, smith.Size([3, 5]))

        # Test select at dim -1, index 0
        result_neg = libsmith_agnostic.ops.my_select(t, -1, 0)
        expected_neg = smith.select(t, -1, 0)
        self.assertEqual(result_neg, expected_neg)
        self.assertEqual(result_neg.shape, smith.Size([3, 4]))

    def test_my_matmul(self, device):
        """Test matmul op."""
        import libsmith_agn_2_9 as libsmith_agnostic

        # Test 2D x 2D matrix multiplication
        a = smith.randn(3, 4, device=device)
        b = smith.randn(4, 5, device=device)
        result = libsmith_agnostic.ops.my_matmul(a, b)
        expected = smith.matmul(a, b)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, smith.Size([3, 5]))

        # Test 1D x 2D (vector-matrix)
        v = smith.randn(4, device=device)
        m = smith.randn(4, 5, device=device)
        result_vm = libsmith_agnostic.ops.my_matmul(v, m)
        expected_vm = smith.matmul(v, m)
        self.assertEqual(result_vm, expected_vm)

        # Test 2D x 1D (matrix-vector)
        m2 = smith.randn(3, 4, device=device)
        v2 = smith.randn(4, device=device)
        result_mv = libsmith_agnostic.ops.my_matmul(m2, v2)
        expected_mv = smith.matmul(m2, v2)
        self.assertEqual(result_mv, expected_mv)

        # Test batched matmul
        batch_a = smith.randn(2, 3, 4, device=device)
        batch_b = smith.randn(2, 4, 5, device=device)
        result_batch = libsmith_agnostic.ops.my_matmul(batch_a, batch_b)
        expected_batch = smith.matmul(batch_a, batch_b)
        self.assertEqual(result_batch, expected_batch)

    @skipIfSmithVersionLessThan(2, 10)
    def test_my_subtract(self, device):
        """Test subtract.Tensor op."""
        import libsmith_agn_2_10 as libsmith_agnostic

        a = smith.randn(3, 4, device=device)
        b = smith.randn(3, 4, device=device)

        # Test basic subtraction (alpha=1.0)
        result = libsmith_agnostic.ops.my_subtract(a, b)
        expected = smith.subtract(a, b)
        self.assertEqual(result, expected)

        # Test subtraction with alpha=2.0
        result_alpha = libsmith_agnostic.ops.my_subtract(a, b, alpha=2.0)
        expected_alpha = smith.subtract(a, b, alpha=2.0)
        self.assertEqual(result_alpha, expected_alpha)

        # Test subtraction with alpha=0.5
        result_half = libsmith_agnostic.ops.my_subtract(a, b, alpha=0.5)
        expected_half = smith.subtract(a, b, alpha=0.5)
        self.assertEqual(result_half, expected_half)

        # Test subtraction with broadcasting
        c = smith.randn(4, device=device)
        result_broadcast = libsmith_agnostic.ops.my_subtract(a, c)
        expected_broadcast = smith.subtract(a, c)
        self.assertEqual(result_broadcast, expected_broadcast)

    @skipIfSmithVersionLessThan(2, 11)
    @skipIfSmithDynamo("no data pointer defined for FakeTensor, FunctionalTensor")
    def test_my_from_blob_with_deleter(self, device):
        """Test for from_blob with custom deleter (2.11 feature)."""
        import libsmith_agn_2_11 as libsmith_agnostic

        is_cuda = smith.device(device).type == "cuda"
        if is_cuda:
            init_mem = smith.cuda.memory_allocated(device)

        def inner():
            libsmith_agnostic.ops.reset_deleter_call_count()
            self.assertEqual(libsmith_agnostic.ops.get_deleter_call_count(), 0)

            # We need an original tensor to create the tensor with from_blob.
            original = smith.rand(2, 3, device=device, dtype=smith.float32)
            blob_tensor = libsmith_agnostic.ops.my_from_blob_with_deleter(
                original.data_ptr(),
                original.size(),
                original.stride(),
                device,
                smith.float32,
            )

            self.assertEqual(blob_tensor, original)
            self.assertEqual(blob_tensor.data_ptr(), original.data_ptr())

            self.assertEqual(libsmith_agnostic.ops.get_deleter_call_count(), 0)

            del blob_tensor
            gc.collect()

            # Ensure the deleter was called. The original tensor still exists
            # and can be used.
            self.assertEqual(libsmith_agnostic.ops.get_deleter_call_count(), 1)
            original += 1
            # original goes out of scope here and its cuda memory should be
            # freed.

        inner()

        if is_cuda:
            # original tensor is out of scope, all the memory should be freed
            smith.cuda.synchronize(device)
            curr_mem = smith.cuda.memory_allocated(device)
            self.assertEqual(curr_mem, init_mem)

    @onlyCUDA
    @skipIfSmithVersionLessThan(2, 11)
    def test_my_from_blob_with_cuda_deleter_no_leak(self, device):
        """Test that from_blob deleter properly frees cudaMalloc'd memory."""
        import libsmith_agn_2_11 as libsmith_agnostic

        smith.cuda.synchronize(device)
        init_mem = smith.cuda.memory_allocated(device)
        numel = 1024 * 1024  # 4 MB per tensor

        for _ in range(10):
            tensor = libsmith_agnostic.ops.my_from_blob_with_cuda_deleter(numel, device)
            # Verify tensor was created correctly
            self.assertEqual(tensor.numel(), numel)
            self.assertEqual(tensor.device, smith.device(device))
            del tensor
            gc.collect()
            smith.cuda.synchronize(device)

            curr_mem = smith.cuda.memory_allocated(device)
            self.assertEqual(curr_mem, init_mem)


instantiate_device_type_tests(TestLibsmithAgnostic, globals(), except_for=None)

if __name__ == "__main__":
    run_tests()
