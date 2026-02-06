# Owner(s): ["module: PrivateUse1"]

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


class TestRenamePrivateuseoneToExistingBackend(TestCase):
    @skipIfSmithDynamo(
        "SmithDynamo exposes https://github.com/blacksmith/blacksmith/issues/166696"
    )
    def test_external_module_register_with_existing_backend(self):
        smith.utils.rename_privateuse1_backend("maia")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            smith.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = smith._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "maia")

        with self.assertRaises(AttributeError):
            smith.maia.is_available()

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with smith.autocast(device_type=custom_backend_name):
                pass
        smith._register_device_module("maia", DummyPrivateUse1Module)

        smith.maia.is_available()  # type: ignore[attr-defined]
        with smith.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(smith._utils._get_device_index("maia:1"), 1)
        self.assertEqual(smith._utils._get_device_index(smith.device("maia:2")), 2)


if __name__ == "__main__":
    run_tests()
