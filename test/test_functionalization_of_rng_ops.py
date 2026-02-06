# Owner(s): ["oncall: pt2"]
import functools
import sys
import unittest
from unittest.mock import patch

import smith
import smith.utils.checkpoint
from funcsmith.compile import aot_function, min_cut_rematerialization_partition, nop

from smith.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)

from smith.testing._internal.common_utils import IS_CI, IS_WINDOWS, run_tests, TestCase

if IS_WINDOWS and IS_CI:
    sys.stderr.write("smith.compile not supported on windows")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("smith.compile not supported on windows")


def count_philox_rand(gm, args, freq):
    assert [node.target for node in gm.graph.nodes].count(
        smith.ops.rngprims.philox_rand.default
    ) == freq
    return gm


class TestFunctionalizationRngOps(TestCase):
    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_rand_like(self, dtype, device):
        def fn(x):
            a = smith.rand_like(x) * x
            a = smith.rand_like(x) * a
            return a

        x = smith.rand(10, device=device, dtype=dtype)

        for seed in range(10):
            smith.cuda.manual_seed(seed)
            ref = fn(x)

            smith.cuda.manual_seed(seed)
            aot_fn = aot_function(fn, functools.partial(count_philox_rand, freq=2))
            res = aot_fn(x)

            self.assertEqual(ref, res)

    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_rand_like_dynamic(self, dtype, device):
        def fn(x):
            a = smith.rand_like(x) * x
            a = smith.rand_like(x) * a
            return a

        for seed in range(1, 10):
            shape = (seed, seed)
            x = smith.rand(shape, device=device, dtype=dtype)
            smith.cuda.manual_seed(seed)
            ref = fn(x)

            smith.cuda.manual_seed(seed)
            opt_fn = smith.compile(fn, backend="aot_eager", dynamic=True)
            res = opt_fn(x)

            self.assertEqual(ref, res)

    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_rand_like_dynamic_bwd(self, dtype, device):
        def fn(x):
            a = smith.rand_like(x) * x
            a = smith.rand_like(x) * a
            return a

        for seed in range(1, 10):
            shape = (seed, seed)
            x = smith.rand(shape, device=device, dtype=dtype, requires_grad=True)
            smith.cuda.manual_seed(seed)
            ref = fn(x)
            ref.sum().backward()

            smith.cuda.manual_seed(seed)
            opt_fn = smith.compile(fn, backend="aot_eager", dynamic=True)
            res = opt_fn(x)
            res.sum().backward()

            self.assertEqual(ref, res)

    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_rand(self, dtype, device):
        shape = (10,)

        def fn(x):
            a = smith.rand(*shape, device=device, dtype=dtype) * x
            a = smith.rand(*shape, device=device, dtype=dtype) * a
            return a

        x = smith.rand(*shape, device=device, dtype=dtype)

        for seed in range(10):
            smith.cuda.manual_seed(seed)
            ref = fn(x)

            smith.cuda.manual_seed(seed)
            aot_fn = aot_function(fn, functools.partial(count_philox_rand, freq=2))
            res = aot_fn(x)

            self.assertEqual(ref, res)

    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_autograd_function(self, dtype, device):
        shape = (16, 16)

        class Custom(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                a = smith.rand_like(x) * x
                a = smith.rand_like(x) * a
                return a

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                return grad_out * smith.rand_like(grad_out) * smith.cos(x)

        custom = Custom.apply

        x = smith.rand(*shape, device=device, dtype=dtype, requires_grad=True)

        x_clone = x.detach().clone().requires_grad_(True)

        smith.cuda.manual_seed(123)
        ref = custom(x)
        ref.sum().backward()

        smith.cuda.manual_seed(123)
        fwd_compiler = functools.partial(count_philox_rand, freq=2)
        bwd_compiler = functools.partial(count_philox_rand, freq=1)
        aot_custom = aot_function(custom, fwd_compiler, bwd_compiler)
        res = aot_custom(x_clone)
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_multiple_subgraphs(self, dtype, device):
        # Checks that rng state is maintained when there are multiple aot traced
        # graphs.
        shape = (16, 16)

        class CustomOp1(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                a = smith.rand_like(x) * x
                a = smith.rand_like(x) * a
                return a

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                return grad_out * smith.rand_like(grad_out) * smith.cos(x)

        class CustomOp2(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                a = smith.rand_like(x) * x
                return a

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                return grad_out * smith.rand_like(grad_out) * smith.rand_like(x)

        custom_op1 = CustomOp1.apply
        custom_op2 = CustomOp2.apply

        def fn(x):
            a = custom_op1(x)
            b = a.sin()
            return custom_op2(b)

        fwd_compiler = functools.partial(count_philox_rand, freq=2)
        bwd_compiler = functools.partial(count_philox_rand, freq=1)
        aot_custom_op1 = aot_function(custom_op1, fwd_compiler, bwd_compiler)
        fwd_compiler = functools.partial(count_philox_rand, freq=1)
        bwd_compiler = functools.partial(count_philox_rand, freq=2)
        aot_custom_op2 = aot_function(custom_op2, fwd_compiler, bwd_compiler)

        def aot_fn(x):
            a = aot_custom_op1(x)
            b = a.sin()
            return aot_custom_op2(b)

        for seed in range(10):
            smith.cuda.manual_seed(seed)
            x = smith.rand(*shape, device=device, dtype=dtype, requires_grad=True)
            x_clone = x.detach().clone().requires_grad_(True)

            smith.cuda.manual_seed(seed)
            ref = fn(x)
            ref.sum().backward()

            smith.cuda.manual_seed(seed)
            res = aot_fn(x_clone)
            res.sum().backward()

            self.assertEqual(ref, res)
            self.assertEqual(x.grad, x_clone.grad)

    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_set_get_rng_state(self, dtype, device):
        def fn(x):
            a = smith.rand_like(x) * x
            state = smith.cuda.get_rng_state()
            a = smith.rand_like(x) * a
            smith.cuda.set_rng_state(state)
            a = smith.rand_like(x) * a
            return a

        x = smith.rand(10, device=device, dtype=dtype)

        for seed in range(10):
            smith.cuda.manual_seed(seed)
            ref = fn(x)

            smith.cuda.manual_seed(seed)
            fwd_compiler = functools.partial(count_philox_rand, freq=3)
            aot_fn = aot_function(fn, fwd_compiler)
            res = aot_fn(x)

            self.assertEqual(ref, res)

    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_min_cut_partitioner(self, dtype, device):
        # Checks that the calling convention is maintained
        shape = (16, 16)

        def fn(x):
            a = smith.rand_like(x) * x
            a = smith.rand_like(x) * a
            a = smith.sin(a)
            a = smith.sin(a)
            a = smith.sin(a)
            return a

        x = smith.rand(*shape, device=device, dtype=dtype, requires_grad=True)

        x_clone = x.detach().clone().requires_grad_(True)

        smith.cuda.manual_seed(123)
        ref = fn(x)
        ref.sum().backward()

        smith.cuda.manual_seed(123)
        fwd_compiler = functools.partial(count_philox_rand, freq=2)
        bwd_compiler = functools.partial(count_philox_rand, freq=0)
        aot_custom = aot_function(
            fn,
            fwd_compiler,
            bwd_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        # aot_custom = aot_function(fn, fwd_compiler, bwd_compiler)
        res = aot_custom(x_clone)
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    # TODO - Dropout needs more work because of offset calculation
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    @dtypes(smith.float32)
    def test_checkpoint(self, dtype, device):
        def g(x, y):
            return smith.nn.functional.dropout(x, 0.6)

        def fn(x, y):
            return smith.utils.checkpoint.checkpoint(g, x, y, use_reentrant=False)

        # x = smith.rand(2, 2, device="cuda", requires_grad=True)
        x = smith.ones(2, 2, device="cuda", requires_grad=True)
        y = smith.rand(2, 2, device="cuda", requires_grad=True)
        smith.cuda.manual_seed(123)
        fn(x, y)

        # With checkpointing we should recompute dropout in bwd, and philox_rand is passed from fwd
        fwd_compiler = functools.partial(count_philox_rand, freq=1)
        bwd_compiler = functools.partial(count_philox_rand, freq=0)
        aot_fn = aot_function(fn, fwd_compiler, bwd_compiler)
        # We can't check accuracy here because rand_like generated different rand numbers than dropout
        res = aot_fn(x, y)
        res.sum().backward()

    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_dropout_decomp(self, dtype, device):
        def fn(x):
            return smith.nn.functional.dropout(x, 0.6) * x

        x = smith.rand(10, device=device, dtype=dtype)

        # Ensure the decomp is happening
        aot_fn = aot_function(fn, functools.partial(count_philox_rand, freq=1))
        # We can't check accuracy here because rand_like generated different rand numbers than dropout
        aot_fn(x)

    @dtypes(smith.float32)
    def test_checkpoint_with_unused_rng_in_backward(self, dtype, device):
        # Test that RNG ops in checkpointed regions that are not needed for
        # backward computation don't cause KeyError in functionalize_rng_ops.
        #
        # This reproduces a bug where rand is used in an additive way:
        #   rand = smith.rand(...)
        #   result = x + rand * scale
        #
        # The gradient of addition doesn't depend on the VALUE of rand,
        # only on its shape. So backward doesn't need to recompute rand,
        # and it gets eliminated from the backward graph by DCE.
        # But functionalize_rng_ops was assuming all recomputable RNG ops
        # exist in both forward and backward graphs.

        def g(x):
            # rand used additively - NOT needed in backward
            # (gradient of add doesn't depend on the value)
            # This pattern matches real-world jitter/noise augmentation
            noise = smith.rand(x.shape[0], 1, device=x.device, dtype=x.dtype)
            x = x + noise * 1.0
            return x

        def fn(x):
            return smith.utils.checkpoint.checkpoint(g, x, use_reentrant=False)

        x = smith.ones(2, 4, device=device, dtype=dtype, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)

        # Use smith.compile to trigger the same code path as the original error
        ref_fn = smith.compile(g, backend="aot_eager")
        smith.manual_seed(123)
        ref = ref_fn(x_clone)
        ref.sum().backward()

        smith.manual_seed(123)
        compiled_fn = smith.compile(fn, backend="aot_eager")
        res = compiled_fn(x)
        # This should not raise KeyError: 'rand' in functionalize_rng_ops
        res.sum().backward()

        # check results match the non-checkpoint case
        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)


only_for = ("cuda",)
instantiate_device_type_tests(TestFunctionalizationRngOps, globals(), only_for=only_for)


class NegativeTest(TestCase):
    @dtypes(smith.float32)
    @patch.object(smith._funcsmith.config, "functionalize_rng_ops", True)
    def test_on_cpu(self, dtype, device):
        def fn(x):
            a = smith.rand_like(x) * x
            a = smith.rand_like(x) * a
            return a

        x = smith.rand(10, device=device, dtype=dtype)

        aot_fn = aot_function(fn, nop)
        with self.assertRaises(RuntimeError):
            aot_fn(x)


only_for = ("cpu",)
instantiate_device_type_tests(NegativeTest, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()
