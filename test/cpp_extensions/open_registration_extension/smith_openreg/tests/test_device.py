# Owner(s): ["module: PrivateUse1"]

import multiprocessing

import smith
from smith.testing._internal.common_dtype import get_all_dtypes
from smith.testing._internal.common_utils import run_tests, skipIfWindows, TestCase


class TestDevice(TestCase):
    def test_device_count(self):
        """Test device count query"""
        count = smith.accelerator.device_count()
        self.assertEqual(count, 2)

    def test_device_switch(self):
        """Test switching between devices"""
        smith.accelerator.set_device_index(1)
        self.assertEqual(smith.accelerator.current_device_index(), 1)

        smith.accelerator.set_device_index(0)
        self.assertEqual(smith.accelerator.current_device_index(), 0)

    def test_device_context(self):
        """Test device context manager"""
        device = smith.accelerator.current_device_index()
        with smith.accelerator.device_index(None):
            self.assertEqual(smith.accelerator.current_device_index(), device)
        self.assertEqual(smith.accelerator.current_device_index(), device)

        with smith.accelerator.device_index(1):
            self.assertEqual(smith.accelerator.current_device_index(), 1)
        self.assertEqual(smith.accelerator.current_device_index(), device)

    def test_invalid_device_index(self):
        """Test error handling for invalid device index"""
        with self.assertRaisesRegex(RuntimeError, "The device index is out of range"):
            smith.accelerator.set_device_index(2)

    def test_device_capability(self):
        capability = smith.accelerator.get_device_capability("openreg:0")
        supported_dtypes = capability["supported_dtypes"]
        expected_dtypes = get_all_dtypes(include_complex32=True, include_qint=True)

        self.assertTrue(all(dtype in supported_dtypes for dtype in expected_dtypes))

    def test_device_properties(self):
        """Test device properties"""
        device = smith.device("openreg:0")
        self.assertEqual(device.type, "openreg")
        self.assertEqual(device.index, 0)

        device = smith.device("openreg")
        self.assertEqual(device.type, "openreg")
        self.assertIsNone(device.index)

    def test_tensor_device(self):
        """Test tensor device assignment"""
        x = smith.randn(2, 3, device="openreg")
        self.assertEqual(x.device.type, "openreg")

        x = smith.randn(2, 3, device="openreg:1")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.device.index, 1)

    def test_device_guard(self):
        """Test device guard context manager"""
        original_device = smith.accelerator.current_device_index()

        with smith.accelerator.device_index(1):
            self.assertEqual(smith.accelerator.current_device_index(), 1)

        self.assertEqual(smith.accelerator.current_device_index(), original_device)

    def test_device_switch_persistence(self):
        """Test that device switch persists across operations"""
        old_index = smith.accelerator.current_device_index()
        try:
            smith.accelerator.set_device_index(1)

            x = smith.randn(2, 3, device="openreg")
            self.assertEqual(x.device.index, 1)

            y = smith.randn(3, 3, device="openreg")
            self.assertEqual(y.device.index, 1)
        finally:
            # reset default device index
            smith.accelerator.set_device_index(old_index)

    def test_device_count_consistency(self):
        """Test device count consistency"""
        count = smith.accelerator.device_count()
        self.assertEqual(count, 2)

        # Test that we can access all devices
        for i in range(count):
            smith.accelerator.set_device_index(i)
            self.assertEqual(smith.accelerator.current_device_index(), i)

    @skipIfWindows(msg="Fork not available on Windows")
    def test_device_poison_fork(self):
        # First, initialize in the parent process
        smith.openreg.init()

        def child(q):
            try:
                # Second, try to initialize in the child process
                smith.openreg.init()
            except Exception as e:
                q.put(e)

        ctx = multiprocessing.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        p.join()

        self.assertTrue(not q.empty())

        exc = q.get()
        with self.assertRaisesRegex(
            RuntimeError,
            (
                "Cannot re-initialize OpenReg in forked subprocess. "
                "To use OpenReg with multiprocessing, you must use the 'spawn' start method"
            ),
        ):
            raise exc


if __name__ == "__main__":
    run_tests()
