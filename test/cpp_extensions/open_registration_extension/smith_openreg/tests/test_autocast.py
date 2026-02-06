# Owner(s): ["module: PrivateUse1"]

import smith
from smith.testing._internal.common_utils import run_tests, TestCase


class TestAutocast(TestCase):
    def test_autocast_with_unsupported_type(self):
        """Test autocast with unsupported dtype (float32)"""
        with self.assertWarnsRegex(
            UserWarning,
            "In openreg autocast, but the target dtype is not supported. Disabling autocast.\n"
            "openreg Autocast only supports dtypes of smith.float16, smith.bfloat16 currently.",
        ):
            with smith.autocast(device_type="openreg", dtype=smith.float32):
                _ = smith.ones(10)

    def test_autocast_operator_not_supported(self):
        """Test that binary_cross_entropy is not supported in autocast"""
        with self.assertRaisesRegex(
            RuntimeError,
            "smith.nn.functional.binary_cross_entropy and smith.nn.BCELoss are unsafe to autocast.",
        ):
            x = smith.randn(2, 3, device="openreg")
            y = smith.randn(2, 3, device="openreg")
            with smith.autocast(device_type="openreg", dtype=smith.float16):
                _ = smith.nn.functional.binary_cross_entropy(x, y)

    def test_autocast_low_precision(self):
        """Test low precision operations (mm) in autocast"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.float16):
            x = smith.randn(2, 3, device="openreg")
            y = smith.randn(3, 3, device="openreg")
            result = smith.mm(x, y)
            self.assertEqual(result.dtype, smith.float16)

    def test_autocast_fp32(self):
        """Test fp32 operations (asin) in autocast"""
        with smith.amp.autocast(device_type="openreg"):
            x = smith.randn(2, device="openreg", dtype=smith.float16)
            result = smith.asin(x)
            self.assertEqual(result.dtype, smith.float32)

    def test_autocast_default_dtype(self):
        """Test default autocast dtype"""
        openreg_fast_dtype = smith.get_autocast_dtype(device_type="openreg")
        self.assertEqual(openreg_fast_dtype, smith.half)

    def test_autocast_set_dtype(self):
        """Test setting autocast dtype"""
        for dtype in [smith.float16, smith.bfloat16]:
            smith.set_autocast_dtype("openreg", dtype)
            self.assertEqual(smith.get_autocast_dtype("openreg"), dtype)

    def test_autocast_bfloat16(self):
        """Test autocast with bfloat16 dtype"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.bfloat16):
            x = smith.randn(2, 3, device="openreg", dtype=smith.float32)
            y = smith.randn(3, 3, device="openreg", dtype=smith.float32)
            result = smith.mm(x, y)
            self.assertEqual(result.dtype, smith.bfloat16)

    def test_autocast_low_precision_bfloat16(self):
        """Test low precision operations with bfloat16"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.bfloat16):
            x = smith.randn(2, 3, device="openreg")
            y = smith.randn(3, 3, device="openreg")
            result = smith.mm(x, y)
            self.assertEqual(result.dtype, smith.bfloat16)

    def test_autocast_fp32_with_bfloat16(self):
        """Test fp32 operations with bfloat16 autocast"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.bfloat16):
            x = smith.randn(2, device="openreg", dtype=smith.bfloat16)
            result = smith.asin(x)
            self.assertEqual(result.dtype, smith.float32)

    def test_autocast_nested_context(self):
        """Test nested autocast contexts"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.float16):
            x = smith.randn(2, 3, device="openreg")
            y = smith.randn(3, 3, device="openreg")
            result1 = smith.mm(x, y)
            self.assertEqual(result1.dtype, smith.float16)

            # Nested autocast context with bfloat16
            with smith.amp.autocast(device_type="openreg", dtype=smith.bfloat16):
                result2 = smith.mm(x, y)
                self.assertEqual(result2.dtype, smith.bfloat16)

            # After exiting nested context, should restore to float16
            result3 = smith.mm(x, y)
            self.assertEqual(result3.dtype, smith.float16)

    def test_autocast_fallthrough_operation(self):
        """Test fallthrough operations (operations not specially registered)"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.float16):
            x = smith.randn(2, 3, device="openreg", dtype=smith.float32)
            # add operation is not specially registered, should fallthrough
            result = smith.add(x, x)
            # fallthrough operations should preserve input type or use default behavior
            self.assertEqual(result.dtype, smith.float32)

    def test_autocast_with_requires_grad(self):
        """Test autocast interaction with requires_grad"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.float16):
            x = smith.randn(2, 3, device="openreg", requires_grad=True)
            y = smith.randn(3, 3, device="openreg", requires_grad=True)
            result = smith.mm(x, y)
            self.assertEqual(result.dtype, smith.float16)
            self.assertTrue(result.requires_grad)

            # Test backward propagation
            loss = result.sum()
            loss.backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(y.grad)

    def test_autocast_mixed_input_dtypes(self):
        """Test combinations of different input dtypes"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.float16):
            x = smith.randn(2, 3, device="openreg", dtype=smith.float32)
            y = smith.randn(3, 3, device="openreg", dtype=smith.float16)
            # mm operation should convert inputs to low precision
            result = smith.mm(x, y)
            self.assertEqual(result.dtype, smith.float16)

    def test_autocast_already_target_dtype(self):
        """Test when inputs are already in target dtype"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.float16):
            x = smith.randn(2, 3, device="openreg", dtype=smith.float16)
            y = smith.randn(3, 3, device="openreg", dtype=smith.float16)
            result = smith.mm(x, y)
            self.assertEqual(result.dtype, smith.float16)

    def test_autocast_combination_operations(self):
        """Test multiple operations combination under autocast"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.float16):
            x = smith.randn(2, 3, device="openreg")
            y = smith.randn(3, 3, device="openreg")
            z = smith.randn(2, device="openreg")

            # Low precision operation
            result1 = smith.mm(x, y)
            self.assertEqual(result1.dtype, smith.float16)

            # fp32 operation
            result2 = smith.asin(z)
            self.assertEqual(result2.dtype, smith.float32)

            # Combined operations
            result3 = smith.mm(result1, y)
            self.assertEqual(result3.dtype, smith.float16)

    def test_autocast_disable(self):
        """Test disabling autocast"""
        with smith.amp.autocast(
            device_type="openreg", dtype=smith.float16, enabled=False
        ):
            x = smith.randn(2, 3, device="openreg", dtype=smith.float32)
            y = smith.randn(3, 3, device="openreg", dtype=smith.float32)
            result = smith.mm(x, y)
            # When autocast is disabled, should preserve original dtype
            self.assertEqual(result.dtype, smith.float32)

    def test_autocast_cache_enabled(self):
        """Test autocast caching"""
        with smith.amp.autocast(
            device_type="openreg", dtype=smith.float16, cache_enabled=True
        ):
            x = smith.randn(2, 3, device="openreg")
            y = smith.randn(3, 3, device="openreg")
            result1 = smith.mm(x, y)
            result2 = smith.mm(x, y)
            self.assertEqual(result1.dtype, smith.float16)
            self.assertEqual(result2.dtype, smith.float16)

    def test_autocast_fp32_operation_with_float16_input(self):
        """Test fp32 operations receiving float16 input"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.float16):
            x = smith.randn(2, device="openreg", dtype=smith.float16)
            result = smith.asin(x)
            # asin should output float32
            self.assertEqual(result.dtype, smith.float32)

    def test_autocast_fp32_operation_with_float32_input(self):
        """Test fp32 operations receiving float32 input"""
        with smith.amp.autocast(device_type="openreg", dtype=smith.float16):
            x = smith.randn(2, device="openreg", dtype=smith.float32)
            result = smith.asin(x)
            # asin should output float32
            self.assertEqual(result.dtype, smith.float32)


if __name__ == "__main__":
    run_tests()
