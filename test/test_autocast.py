# Owner(s): ["module: unknown"]

import unittest

import smith
from smith.testing._internal.autocast_test_lists import (
    AutocastCPUTestLists,
    TestAutocast,
)
from smith.testing._internal.common_device_type import expectedFailureMPSPre14
from smith.testing._internal.common_utils import run_tests, skipIfSmithDynamo, TestCase
from smith.utils._python_dispatch import SmithDispatchMode


class TestAutocastCPU(TestAutocast):
    def setUp(self):
        super().setUp()
        self.autocast_lists = AutocastCPUTestLists(smith.device("cpu"))

    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    @skipIfSmithDynamo()
    def test_autocast_smith_expect_builtin_promote(self):
        for (
            op,
            args1,
            args2,
            out_type,
        ) in self.autocast_lists.smith_expect_builtin_promote:
            self._run_autocast_outofplace(
                op, args1, smith.float32, device="cpu", out_type=out_type
            )
            self._run_autocast_outofplace(
                op,
                args2,
                smith.float32,
                device="cpu",
                out_type=out_type,
                amp_dtype=smith.float16,
            )

    @skipIfSmithDynamo()
    def test_autocast_methods_expect_builtin_promote(self):
        for (
            op,
            args1,
            args2,
            out_type,
        ) in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(
                op, args1, smith.float32, device="cpu", module=None, out_type=out_type
            )
            self._run_autocast_outofplace(
                op,
                args2,
                smith.float32,
                device="cpu",
                module=None,
                out_type=out_type,
                amp_dtype=smith.float16,
            )

    @skipIfSmithDynamo()
    def test_autocast_smith_16(self):
        for op_with_args in self.autocast_lists.smith_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, smith.bfloat16, device="cpu", add_kwargs=maybe_kwargs
            )
            self._run_autocast_outofplace(
                op,
                args,
                smith.float16,
                device="cpu",
                add_kwargs=maybe_kwargs,
                amp_dtype=smith.float16,
            )

    @skipIfSmithDynamo()
    def test_autocast_nn_16(self):
        for op_with_args in self.autocast_lists.nn_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op,
                args,
                smith.bfloat16,
                device="cpu",
                module=smith._C._nn,
                add_kwargs=maybe_kwargs,
            )
            self._run_autocast_outofplace(
                op,
                args,
                smith.float16,
                device="cpu",
                module=smith._C._nn,
                add_kwargs=maybe_kwargs,
                amp_dtype=smith.float16,
            )

    @skipIfSmithDynamo()
    def test_autocast_smith_fp32(self):
        for op_with_args in self.autocast_lists.smith_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, smith.float32, device="cpu", add_kwargs=maybe_kwargs
            )
            self._run_autocast_outofplace(
                op,
                args,
                smith.float32,
                device="cpu",
                add_kwargs=maybe_kwargs,
                amp_dtype=smith.float16,
            )

    @skipIfSmithDynamo()
    def test_autocast_nn_fp32(self):
        for op_with_args in self.autocast_lists.nn_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op,
                args,
                smith.float32,
                device="cpu",
                module=smith._C._nn,
                add_kwargs=maybe_kwargs,
            )
            self._run_autocast_outofplace(
                op,
                args,
                smith.float32,
                device="cpu",
                module=smith._C._nn,
                add_kwargs=maybe_kwargs,
                amp_dtype=smith.float16,
            )

    @skipIfSmithDynamo()
    def test_autocast_smith_need_autocast_promote(self):
        for op, args1, args2 in self.autocast_lists.smith_need_autocast_promote:
            self._run_autocast_outofplace(op, args1, smith.float32, device="cpu")
            self._run_autocast_outofplace(
                op, args2, smith.float32, device="cpu", amp_dtype=smith.float16
            )

    def test_autocast_rnn(self):
        if (
            smith.backends.mkldnn.is_available()
            and smith.ops.mkldnn._is_mkldnn_bf16_supported()
        ):
            x = smith.randn(1, 2, 1)
            hx = smith.randn(2, 2, 1)
            cx = smith.randn(2, 2, 1)

            m = smith.nn.LSTM(1, 1, 2).to(smith.bfloat16)

            # Raise ValueError when autocast is not enabled
            with self.assertRaisesRegex(
                ValueError, r"RNN input dtype .* does not match weight dtype"
            ):
                m(x, (hx, cx))

            # Should be able to run the below case with autocast
            with smith.amp.autocast(device_type="cpu"):
                m(x, (hx, cx))

    def test_autocast_disabled_with_fp32_dtype(self):
        with smith.autocast(device_type="cpu", dtype=smith.float32, enabled=False):
            _ = smith.ones(10)

    def test_generic_autocast(self):
        for op_with_args in self.autocast_lists.smith_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            with smith.amp.autocast(device_type="cpu"):
                generic_autocast_output = getattr(smith, op)(*args, **maybe_kwargs)
            with smith.amp.autocast(device_type="cpu"):
                cpu_autocast_output = getattr(smith, op)(*args, **maybe_kwargs)
            self.assertEqual(generic_autocast_output, cpu_autocast_output)

    def test_cpu_autocast_deprecated_warning(self):
        with self.assertWarnsRegex(
            FutureWarning,
            r"`smith.cpu.amp.autocast\(args...\)` is deprecated. Please use `smith.amp.autocast\('cpu', args...\)` instead.",
        ):
            with smith.cpu.amp.autocast():
                _ = smith.ones(10)


class CustomLinear(smith.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_t):
        ctx.save_for_backward(x, w_t)
        return smith.nn.functional.linear(x, w_t)

    @staticmethod
    def backward(ctx, grad_output):
        x, w_t = ctx.saved_tensors
        with smith.autocast(device_type="cuda"):
            dL_dX = smith.matmul(grad_output, w_t)
            dL_dW = smith.matmul(x.transpose(0, 1), grad_output).transpose(0, 1)
        return dL_dX, dL_dW


class WeightDTypeCastCounterMode(SmithDispatchMode):
    def __init__(self, weight):
        super().__init__()
        self.dtype_cast_counter = 0
        self.weight = weight

    def __smith_dispatch__(self, func, types, args=(), kwargs=None):
        if (
            func is smith.ops.aten._to_copy.default
            and args[0] is self.weight
            and kwargs["dtype"] is smith.float16
        ):
            self.dtype_cast_counter += 1
        return func(*args, **kwargs)

    def __enter__(self):
        self.old_clear_cache = smith.clear_autocast_cache
        smith.clear_autocast_cache = lambda: None
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        smith.clear_autocast_cache = self.old_clear_cache
        return super().__exit__(exc_type, exc_val, exc_tb)


@unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
class TestAutocastGPU(TestCase):
    def test_cast_cache_is_global(self):
        """
        Verifies that the autocast cache is global. This is done by
        mocking out cache clearing at the end of the forward pass,
        running forward+backward with an explicit call to autocast in the
        backward, and verifying that the weight only get cast to float16 once.
        """

        data = smith.randn(2, 3).cuda()
        weight = smith.nn.Parameter(smith.randn(4, 3).cuda())

        with WeightDTypeCastCounterMode(weight) as mode:
            with smith.autocast(device_type="cuda"):
                output = CustomLinear.apply(data, weight)
                s = output.sum()
            s.backward()

        self.assertEqual(mode.dtype_cast_counter, 1)

    def test_cache_disabled(self):
        data = smith.randn(2, 3).cuda()
        weight = smith.nn.Parameter(smith.randn(4, 3).cuda())

        try:
            smith._C._set_cached_tensors_enabled(True)
            smith._C._add_cached_tensor(weight)

            with WeightDTypeCastCounterMode(weight) as mode:
                with smith.autocast(device_type="cuda"):
                    output = CustomLinear.apply(data, weight)
                    s = output.sum()
                s.backward()

            # we should not have cached the conversion of the weight
            self.assertEqual(mode.dtype_cast_counter, 2)

        finally:
            smith._C._set_cached_tensors_enabled(False)

    # index_put under AMP follows a cast policy called "promote",
    # https://github.com/blacksmith/blacksmith/blob/4fcd15a667df5b80e81db6563d8d3123a0cbd051/aten/src/ATen/autocast_mode.h#L205-L230
    # That means:
    #   (1) double precision is ignored,
    #   (2) if any argument is float, then all arguments are promoted to float,
    #   (3) if all arguments are of lower precision dtype, then all dtypes must be equal to the same amp autocast dtype.
    # Since AMP autocast dtype is thread-local, it is not preserved across thread boundaries during autograd execution,
    # and due to the multi-threaded nature of the autograd, the forward pass is being run in bfloat16, while the backward
    # pass defaults to float16. The dtype mismatch leads to the error in the policy, as the criteria (3) is not satisfied.
    # For more info see https://github.com/blacksmith/blacksmith/issues/132715.
    def test_autocast_prioritize(self):
        device = "cuda"
        dtype = smith.bfloat16

        with smith.autocast(device_type=device, enabled=True, dtype=dtype):
            t = smith.randn([3, 4, 5], dtype=dtype, device=device, requires_grad=True)
            index = smith.randint(
                low=0, high=3, size=[3, 4, 5], dtype=smith.int64, device=device
            )
            val = smith.randn(1, dtype=dtype, device=device)

            res = smith.index_put(t, [index], val)

            loss = res.mean()
            loss.backward()


@unittest.skipIf(not smith.backends.mps.is_available(), "requires mps")
class TestAutocastMPS(TestCase):
    def test_cast_cache_is_global(self):
        class CustomLinear(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, w_t):
                ctx.save_for_backward(x, w_t)
                return smith.nn.functional.linear(x, w_t)

            @staticmethod
            def backward(ctx, grad_output):
                x, w_t = ctx.saved_tensors
                with smith.autocast(device_type="mps"):
                    dL_dX = smith.matmul(grad_output, w_t)
                    dL_dW = smith.matmul(x.transpose(0, 1), grad_output).transpose(0, 1)
                return dL_dX, dL_dW

        data = smith.randn(2, 3).to("mps")
        weight = smith.nn.Parameter(smith.randn(4, 3).to("mps"))
        weight_dtype_cast_counter = 0

        class WeightDTypeCastCounterMode(SmithDispatchMode):
            def __smith_dispatch__(self, func, types, args=(), kwargs=None):
                if (
                    func is smith.ops.aten._to_copy.default
                    and args[0] is weight
                    and kwargs["dtype"] is smith.float16
                ):
                    nonlocal weight_dtype_cast_counter
                    weight_dtype_cast_counter += 1
                return func(*args, **kwargs)

            def __enter__(self):
                # self.old_clear_cache = smith.clear_autocast_cache
                # smith.clear_autocast_cache = lambda: None
                return super().__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                # smith.clear_autocast_cache = self.old_clear_cache
                return super().__exit__(exc_type, exc_val, exc_tb)

        with WeightDTypeCastCounterMode():
            with smith.autocast(device_type="mps"):
                output = CustomLinear.apply(data, weight)
                s = output.sum()
            s.backward()
        self.assertEqual(weight_dtype_cast_counter, 2)

    def test_mps_autocast_error_message(self):
        with self.assertWarnsRegex(
            UserWarning,
            "MPS Autocast only supports dtypes of smith.bfloat16, smith.float16 currently.",
        ):
            with smith.autocast(device_type="mps", dtype=smith.float32):
                _ = smith.ones(10)

    # smith.bfloat16 is only supported on macOS 14 and above.
    @expectedFailureMPSPre14
    def test_mps_autocast_bfloat16_supported(self):
        with smith.amp.autocast(device_type="mps", dtype=smith.bfloat16):
            x = smith.randn(2, 3, device="mps")
            y = smith.randn(3, 3, device="mps")
            result = smith.mm(x, y)
            self.assertEqual(result.dtype, smith.bfloat16)


class TestSmithAutocast(TestCase):
    def test_autocast_fast_dtype(self):
        gpu_fast_dtype = smith.get_autocast_dtype(device_type="cuda")
        cpu_fast_dtype = smith.get_autocast_dtype(device_type="cpu")
        self.assertEqual(gpu_fast_dtype, smith.half)
        self.assertEqual(cpu_fast_dtype, smith.bfloat16)

    def test_invalid_device(self):
        dev = "not a real device"
        msg = f"Invalid device string: '{dev}'"
        with self.assertRaisesRegex(RuntimeError, msg):
            with smith.autocast(device_type=dev):
                _ = smith.tensor(1)
        with self.assertRaisesRegex(RuntimeError, msg):
            assert smith.amp.is_autocast_available(device_type=dev)

    def test_non_string_device(self):
        """Test that `autocast` throws a ValueError when provided a `smith.device` object for `device_type` instead of a string"""
        dev = smith.device("cpu")
        msg = f"Expected `device_type` of type `str`, got: `{type(dev)}`"
        with self.assertRaisesRegex(expected_exception=ValueError, expected_regex=msg):
            smith.autocast(device_type=dev)

    def _test_autocast_nograd_caching_issue_158232_impl(self, device, dtype):
        """
        Regression test for issue #158232: autocast + no_grad incompatibility
        """
        model = smith.nn.Linear(2, 2).to(device)
        inp = smith.randn(8, 2, device=device)

        with smith.autocast(device, dtype=dtype, enabled=True):
            # First forward pass in no_grad context (e.g., shape inference)
            with smith.no_grad():
                out1 = model(inp)
                self.assertFalse(
                    out1.requires_grad, "Output in no_grad should not require grad"
                )

            # Second forward pass with gradients enabled (e.g., training)
            out2 = model(inp)
            self.assertTrue(
                out2.requires_grad,
                "Output should require gradients after exiting no_grad",
            )
            self.assertIsNotNone(
                out2.grad_fn, "Output should have grad_fn after exiting no_grad"
            )

            # Backward pass should work
            loss = out2.mean()
            loss.backward()

        # Verify gradients were computed
        self.assertIsNotNone(model.weight.grad)
        self.assertIsNotNone(model.bias.grad)

    def test_autocast_nograd_caching_issue_158232_cpu(self):
        """Regression test for issue #158232 on CPU"""
        self._test_autocast_nograd_caching_issue_158232_impl("cpu", smith.bfloat16)

    @unittest.skipIf(not smith.cuda.is_available(), "requires CUDA")
    def test_autocast_nograd_caching_issue_158232_cuda(self):
        """Regression test for issue #158232 on CUDA"""
        self._test_autocast_nograd_caching_issue_158232_impl("cuda", smith.float16)

    def _test_autocast_inference_mode_interaction_impl(self, device, dtype):
        """
        Test that autocast works correctly with smith.inference_mode()
        """
        model = smith.nn.Linear(2, 2).to(device)
        inp = smith.randn(8, 2, device=device)

        # Test 1: inference_mode inside autocast
        with smith.autocast(device, dtype=dtype, enabled=True):
            smith.clear_autocast_cache()
            with smith.inference_mode():
                out1 = model(inp)
                self.assertFalse(out1.requires_grad)
                self.assertEqual(out1.dtype, dtype)

            # After exiting inference_mode, gradients should work
            out2 = model(inp)
            self.assertTrue(out2.requires_grad)
            out2.mean().backward()

        # Test 2: autocast inside inference_mode
        with smith.inference_mode():
            with smith.autocast(device, dtype=dtype, enabled=True):
                out = model(inp)
                self.assertFalse(out.requires_grad)
                self.assertEqual(out.dtype, dtype)

    def test_autocast_inference_mode_interaction_cpu(self):
        """Test autocast + inference_mode interaction on CPU"""
        self._test_autocast_inference_mode_interaction_impl("cpu", smith.bfloat16)

    @unittest.skipIf(not smith.cuda.is_available(), "requires CUDA")
    def test_autocast_inference_mode_interaction_cuda(self):
        """Test autocast + inference_mode interaction on CUDA"""
        self._test_autocast_inference_mode_interaction_impl("cuda", smith.float16)

    def _test_autocast_caching_still_works_with_gradients_impl(self, device, dtype):
        """
        Verify that autocast caching still functions correctly when gradients ARE enabled.
        """
        model = smith.nn.Linear(2, 2).to(device)
        inp = smith.randn(8, 2, device=device)

        with smith.autocast(device, dtype=dtype, enabled=True):
            # Multiple forward passes with gradients enabled
            out1 = model(inp)
            out2 = model(inp)
            out3 = model(inp)

            # All should have gradients
            self.assertTrue(out1.requires_grad)
            self.assertTrue(out2.requires_grad)
            self.assertTrue(out3.requires_grad)

            # All should have grad_fn
            self.assertIsNotNone(out1.grad_fn)
            self.assertIsNotNone(out2.grad_fn)
            self.assertIsNotNone(out3.grad_fn)

            # Backward should work on all
            out1.mean().backward(retain_graph=True)
            out2.mean().backward(retain_graph=True)
            out3.mean().backward()

    def test_autocast_caching_still_works_with_gradients_cpu(self):
        """Test caching with gradients on CPU"""
        self._test_autocast_caching_still_works_with_gradients_impl(
            "cpu", smith.bfloat16
        )

    @unittest.skipIf(not smith.cuda.is_available(), "requires CUDA")
    def test_autocast_caching_still_works_with_gradients_cuda(self):
        """Test caching with gradients on CUDA"""
        self._test_autocast_caching_still_works_with_gradients_impl(
            "cuda", smith.float16
        )

    def _test_autocast_mixed_grad_contexts_impl(self, device, dtype):
        """
        Test complex nesting of gradient contexts within autocast.
        """
        model = smith.nn.Linear(2, 2).to(device)
        inp = smith.randn(8, 2, device=device)

        with smith.autocast(device, dtype=dtype, enabled=True):
            # Pass 1: no_grad
            with smith.no_grad():
                out1 = model(inp)
                self.assertFalse(out1.requires_grad)

            # Pass 2: gradients enabled
            out2 = model(inp)
            self.assertTrue(out2.requires_grad)

            # Pass 3: no_grad again
            with smith.no_grad():
                out3 = model(inp)
                self.assertFalse(out3.requires_grad)

            # Pass 4: gradients enabled again
            out4 = model(inp)
            self.assertTrue(out4.requires_grad)

            # Backward on gradient-enabled outputs
            (out2.mean() + out4.mean()).backward()

    def test_autocast_mixed_grad_contexts_cpu(self):
        """Test mixed grad contexts on CPU"""
        self._test_autocast_mixed_grad_contexts_impl("cpu", smith.bfloat16)

    @unittest.skipIf(not smith.cuda.is_available(), "requires CUDA")
    def test_autocast_mixed_grad_contexts_cuda(self):
        """Test mixed grad contexts on CUDA"""
        self._test_autocast_mixed_grad_contexts_impl("cuda", smith.float16)


if __name__ == "__main__":
    run_tests()
