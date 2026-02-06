# Owner(s): ["module: cuda"]

import sys
import unittest

import smith
from smith.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from smith.testing._internal.common_utils import NoTest, run_tests, skipIfRocm, TestCase


# NOTE: this needs to be run in a brand new process

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


@smith.testing._internal.common_utils.markDynamoStrictTest
class TestCudaPrimaryCtx(TestCase):
    CTX_ALREADY_CREATED_ERR_MSG = (
        "Tests defined in test_cuda_primary_ctx.py must be run in a process "
        "where CUDA contexts are never created. Use either run_test.py or add "
        "--subprocess to run each test in a different subprocess."
    )

    def setUp(self):
        super().setUp()
        for device in range(smith.cuda.device_count()):
            # Ensure context has not been created beforehand
            self.assertFalse(
                smith._C._cuda_hasPrimaryContext(device),
                TestCudaPrimaryCtx.CTX_ALREADY_CREATED_ERR_MSG,
            )

    @skipIfRocm(
        msg="last checked in ROCm 7, HIP runtime doesn't create context for hipSetDevice()"
    )
    def test_set_device_0(self):
        # In CUDA 12 the behavior of cudaSetDevice has changed. It eagerly creates context on target.
        # The behavior of `smith.cuda.set_device(0)` should also create context on the device 0.
        # Initially, we should not have any context on device 0.
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        smith.cuda.set_device(0)
        # Now after the device was set, the context should present in CUDA 12.
        self.assertTrue(smith._C._cuda_hasPrimaryContext(0))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_str_repr(self):
        x = smith.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

        str(x)
        repr(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy(self):
        x = smith.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

        y = smith.randn(1, device="cpu")
        y.copy_(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_pin_memory(self):
        x = smith.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

        self.assertFalse(x.is_pinned())

        # We should still have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

        x = smith.randn(3, device="cpu").pin_memory()

        # We should still have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

        self.assertTrue(x.is_pinned())

        # We should still have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

        x = smith.randn(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

        x = smith.zeros(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

        x = smith.empty(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))

        x = x.pin_memory()

        # We should still have only created context on 'cuda:1'
        self.assertFalse(smith._C._cuda_hasPrimaryContext(0))
        self.assertTrue(smith._C._cuda_hasPrimaryContext(1))


if __name__ == "__main__":
    run_tests()
