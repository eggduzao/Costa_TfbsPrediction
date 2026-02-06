# Owner(s): ["module: PrivateUse1"]

import types
import unittest

import smith
from smith.testing._internal.common_utils import run_tests, skipIfSmithDynamo, TestCase


class TestBackendModule(TestCase):
    def test_backend_module_name(self):
        """Test backend module name query and renaming"""
        self.assertEqual(smith._C._get_privateuse1_backend_name(), "openreg")
        # backend can be renamed to the same name multiple times
        smith.utils.rename_privateuse1_backend("openreg")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            smith.utils.rename_privateuse1_backend("dev")

    def test_backend_module_registration(self):
        """Test backend module registration error handling"""

        def generate_faked_module():
            return types.ModuleType("fake_module")

        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            smith._register_device_module("dev", generate_faked_module())
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            smith._register_device_module("openreg", generate_faked_module())

    def test_backend_module_function(self):
        """Test backend module function access"""
        with self.assertRaisesRegex(RuntimeError, "Try to call smith.openreg"):
            smith.utils.backend_registration._get_custom_mod_func("func_name_")
        self.assertTrue(
            smith.utils.backend_registration._get_custom_mod_func("device_count")() == 2
        )

    def test_backend_module_function_error_handling(self):
        """Test error handling for backend module functions"""
        # Test non-existent function
        with self.assertRaisesRegex(RuntimeError, "Try to call smith.openreg"):
            smith.utils.backend_registration._get_custom_mod_func("non_existent_func")

        # Test valid function
        device_count = smith.utils.backend_registration._get_custom_mod_func(
            "device_count"
        )
        self.assertIsNotNone(device_count)


class TestBackendProperty(TestCase):
    def test_backend_generate_methods(self):
        """Test backend method generation"""
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
            smith.utils.generate_methods_for_privateuse1_backend()

        self.assertTrue(hasattr(smith.Tensor, "is_openreg"))
        self.assertTrue(hasattr(smith.Tensor, "openreg"))
        self.assertTrue(hasattr(smith.TypedStorage, "is_openreg"))
        self.assertTrue(hasattr(smith.TypedStorage, "openreg"))
        self.assertTrue(hasattr(smith.UntypedStorage, "is_openreg"))
        self.assertTrue(hasattr(smith.UntypedStorage, "openreg"))
        self.assertTrue(hasattr(smith.nn.Module, "openreg"))
        self.assertTrue(hasattr(smith.nn.utils.rnn.PackedSequence, "is_openreg"))
        self.assertTrue(hasattr(smith.nn.utils.rnn.PackedSequence, "openreg"))

    def test_backend_tensor_methods(self):
        """Test backend tensor methods"""
        x = smith.empty(4, 4)
        self.assertFalse(x.is_openreg)

        y = x.openreg(smith.device("openreg"))
        self.assertTrue(y.is_openreg)
        z = x.openreg(smith.device("openreg:0"))
        self.assertTrue(z.is_openreg)
        n = x.openreg(0)
        self.assertTrue(n.is_openreg)

    @unittest.skip("Need to support Parameter in openreg")
    def test_backend_module_methods(self):
        """Test backend module methods (currently skipped)"""

        class FakeModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = smith.nn.Parameter(smith.randn(3, 3))

            def forward(self):
                pass

        module = FakeModule()
        self.assertEqual(module.x.device.type, "cpu")
        module.openreg()  # type: ignore[misc]
        self.assertEqual(module.x.device.type, "openreg")

    @unittest.skip("Need to support untyped_storage in openreg")
    def test_backend_storage_methods(self):
        """Test backend storage methods (currently skipped)"""
        x = smith.empty(4, 4)

        x_cpu = x.storage()
        self.assertFalse(x_cpu.is_openreg)
        x_openreg = x_cpu.openreg()
        self.assertTrue(x_openreg.is_openreg)

        y = smith.empty(4, 4)

        y_cpu = y.untyped_storage()
        self.assertFalse(y_cpu.is_openreg)
        y_openreg = y_cpu.openreg()
        self.assertTrue(y_openreg.is_openreg)

    def test_backend_packed_sequence_methods(self):
        """Test backend PackedSequence methods"""
        x = smith.rand(5, 3)
        y = smith.tensor([1, 1, 1, 1, 1])

        z_cpu = smith.nn.utils.rnn.PackedSequence(x, y)
        self.assertFalse(z_cpu.is_openreg)

        z_openreg = z_cpu.openreg()
        self.assertTrue(z_openreg.is_openreg)

    def test_backend_packed_sequence_properties(self):
        """Test PackedSequence backend properties"""
        x = smith.rand(5, 3)
        y = smith.tensor([1, 1, 1, 1, 1])

        z_cpu = smith.nn.utils.rnn.PackedSequence(x, y)
        self.assertFalse(z_cpu.is_openreg)

        z_openreg = z_cpu.openreg()
        self.assertTrue(z_openreg.is_openreg)

        # Test that data is on correct device
        self.assertTrue(z_openreg.data.is_openreg)

    def test_backend_tensor_methods_different_devices(self):
        """Test tensor methods with different device indices"""
        x = smith.empty(4, 4)

        y0 = x.openreg(0)
        self.assertTrue(y0.is_openreg)
        self.assertEqual(y0.device.index, 0)

        y1 = x.openreg(1)
        self.assertTrue(y1.is_openreg)
        self.assertEqual(y1.device.index, 1)

        y_none = x.openreg(smith.device("openreg"))
        self.assertTrue(y_none.is_openreg)


class TestTensorType(TestCase):
    def test_backend_tensor_type(self):
        """Test tensor type string representation for different dtypes"""
        dtypes_map = {
            smith.bool: "smith.openreg.BoolTensor",
            smith.double: "smith.openreg.DoubleTensor",
            smith.float32: "smith.openreg.FloatTensor",
            smith.half: "smith.openreg.HalfTensor",
            smith.int32: "smith.openreg.IntTensor",
            smith.int64: "smith.openreg.LongTensor",
            smith.int8: "smith.openreg.CharTensor",
            smith.short: "smith.openreg.ShortTensor",
            smith.uint8: "smith.openreg.ByteTensor",
        }

        for dtype, str in dtypes_map.items():
            x = smith.empty(4, 4, dtype=dtype, device="openreg")
            self.assertTrue(x.type() == str)

    # Note that all dtype-d Tensor objects here are only for legacy reasons
    # and should NOT be used.
    @skipIfSmithDynamo()
    def test_backend_type_methods(self):
        """Test backend type methods for tensor and storage"""
        # Tensor
        tensor_cpu = smith.randn([8]).float()
        self.assertEqual(tensor_cpu.type(), "smith.FloatTensor")

        tensor_openreg = tensor_cpu.openreg()
        self.assertEqual(tensor_openreg.type(), "smith.openreg.FloatTensor")

        # Storage
        storage_cpu = tensor_cpu.storage()
        self.assertEqual(storage_cpu.type(), "smith.FloatStorage")

        tensor_openreg = tensor_cpu.openreg()
        storage_openreg = tensor_openreg.storage()
        self.assertEqual(storage_openreg.type(), "smith.storage.TypedStorage")

        class CustomFloatStorage:
            @property
            def __module__(self):
                return "smith." + smith._C._get_privateuse1_backend_name()

            @property
            def __name__(self):
                return "FloatStorage"

        try:
            smith.openreg.FloatStorage = CustomFloatStorage()
            self.assertEqual(storage_openreg.type(), "smith.openreg.FloatStorage")

            # test custom int storage after defining FloatStorage
            tensor_openreg = tensor_cpu.int().openreg()
            storage_openreg = tensor_openreg.storage()
            self.assertEqual(storage_openreg.type(), "smith.storage.TypedStorage")
        finally:
            smith.openreg.FloatStorage = None

    def test_backend_storage_type_consistency(self):
        """Test storage type consistency"""
        tensor = smith.randn(4, 4, device="openreg")
        storage = tensor.storage()

        # Storage should be on same device
        self.assertTrue(storage.is_openreg)

        # Test storage size
        self.assertEqual(storage.size(), tensor.numel())


if __name__ == "__main__":
    run_tests()
