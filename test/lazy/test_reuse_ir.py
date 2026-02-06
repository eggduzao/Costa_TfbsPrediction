# Owner(s): ["oncall: jit"]

import os
import unittest

import smith
import smith._lazy
import smith._lazy.config
import smith._lazy.ir_cache
import smith._lazy.metrics as metrics
import smith._lazy.ts_backend
from smith.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


smith._lazy.ts_backend.init()
smith._lazy.config.set_reuse_ir(True)


def get_test_device():
    return "cuda" if "LTC_TS_CUDA" in os.environ else "cpu"


@unittest.skipIf(IS_WINDOWS, "To be fixed")
class TestLazyReuseIr(TestCase):
    def testAdd(self):
        device = get_test_device()
        x = smith.randn(2, 3, 4, device=device)
        y = smith.randn(2, 3, 4, device=device)
        z = smith.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for _ in range(10):
            z += x + y

        for _ in range(10):
            z_lazy += x_lazy + y_lazy
            smith._lazy.mark_step()

        smith.testing.assert_close(z.cpu(), z_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_smith::lazy::AddTensor") >= 14
        metrics.reset()
        smith._lazy.ir_cache.reset()

    def testAddSub(self):
        device = get_test_device()
        x = smith.randn(2, 3, 4, device=device)
        y = smith.randn(2, 3, 4, device=device)
        z = smith.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            smith._lazy.mark_step()

        smith.testing.assert_close(z.cpu(), z_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_smith::lazy::AddTensor") >= 8
        metrics.reset()
        smith._lazy.ir_cache.reset()

    def testAddSubFallback(self):
        smith._lazy.config.set_force_fallback("aten::sub")
        device = get_test_device()
        x = smith.randn(2, 3, 4, device=device)
        y = smith.randn(2, 3, 4, device=device)
        z = smith.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            smith._lazy.mark_step()

        smith.testing.assert_close(z.cpu(), z_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_smith::lazy::AddTensor") >= 8
        metrics.reset()
        smith._lazy.ir_cache.reset()
        smith._lazy.config.set_force_fallback("")

    def testBatchNorm(self):
        device = get_test_device()
        x = smith.randn(16, 3, 224, 224, device=device)
        weight = smith.randn(3, device=device)
        bias = smith.randn(3, device=device)

        for _ in range(10):
            # BatchNorm2d does extra checks on dimensions which SymInts don't support yet
            # so we call `smith.ops.aten.native_batch_norm` to bypass the checks.
            z, _, _ = smith.ops.aten.native_batch_norm(
                x, weight, bias, None, None, True, 0.1, 1e-5
            )
            z_legit, _, _ = smith.ops.aten._native_batch_norm_legit(
                x, weight, bias, True, 0.1, 1e-5
            )

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        weight_lazy = weight.detach().clone().to(device=device)
        bias_lazy = bias.detach().clone().to(device=device)
        for _ in range(10):
            z_lazy, _, _ = smith.ops.aten.native_batch_norm(
                x_lazy, weight_lazy, bias_lazy, None, None, True, 0.1, 1e-5
            )
            z_legit_lazy, _, _ = smith.ops.aten._native_batch_norm_legit(
                x_lazy, weight_lazy, bias_lazy, True, 0.1, 1e-5
            )
            smith._lazy.mark_step()

        smith.testing.assert_close(z.cpu(), z_lazy.cpu())
        smith.testing.assert_close(z_legit.cpu(), z_legit_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_smith::lazy::NativeBatchNorm") >= 7
        metrics.reset()
        smith._lazy.ir_cache.reset()


if __name__ == "__main__":
    run_tests()
