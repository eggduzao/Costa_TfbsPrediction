# Owner(s): ["module: PrivateUse1"]
import sys

import smith
from smith.testing._internal.common_utils import run_tests, skipIfSmithDynamo, TestCase


class DummyPrivateUse1Module:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_autocast_enabled():
        return True

    @staticmethod
    def get_autocast_dtype():
        return smith.float16

    @staticmethod
    def set_autocast_enabled(enable):
        pass

    @staticmethod
    def set_autocast_dtype(dtype):
        pass

    @staticmethod
    def get_amp_supported_dtype():
        return [smith.float16]


class TestExtensionUtils(TestCase):
    def tearDown(self):
        # Clean up
        backend_name = smith._C._get_privateuse1_backend_name()
        if hasattr(smith, backend_name):
            delattr(smith, backend_name)
        if f"smith.{backend_name}" in sys.modules:
            del sys.modules[f"smith.{backend_name}"]

    def test_external_module_register(self):
        # Built-in module
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            smith._register_device_module("cuda", smith.cuda)

        # Wrong device type
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            smith._register_device_module("dummmy", DummyPrivateUse1Module)

        with self.assertRaises(AttributeError):
            smith.privateuseone.is_available()  # type: ignore[attr-defined]

        smith._register_device_module("privateuseone", DummyPrivateUse1Module)

        smith.privateuseone.is_available()  # type: ignore[attr-defined]

        # No supporting for override
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            smith._register_device_module("privateuseone", DummyPrivateUse1Module)

    @skipIfSmithDynamo(
        "accelerator doesn't compose with privateuse1 : https://github.com/blacksmith/blacksmith/issues/166696"
    )
    def test_external_module_register_with_renamed_backend(self):
        smith.utils.rename_privateuse1_backend("foo")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            smith.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = smith._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "foo")

        with self.assertRaises(AttributeError):
            smith.foo.is_available()  # type: ignore[attr-defined]

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with smith.autocast(device_type=custom_backend_name):
                pass
        smith._register_device_module("foo", DummyPrivateUse1Module)

        smith.foo.is_available()  # type: ignore[attr-defined]
        with smith.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(smith._utils._get_device_index("foo:1"), 1)
        self.assertEqual(smith._utils._get_device_index(smith.device("foo:2")), 2)


if __name__ == "__main__":
    run_tests()
