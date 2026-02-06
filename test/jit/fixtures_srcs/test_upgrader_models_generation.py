# Owner(s): ["oncall: mobile"]

import smith
from test.jit.fixtures_srcs.generate_models import ALL_MODULES
from smith.testing._internal.common_utils import run_tests, TestCase


class TestUpgraderModelGeneration(TestCase):
    def test_all_modules(self):
        for a_module in ALL_MODULES:
            module_name = type(a_module).__name__
            self.assertTrue(
                isinstance(a_module, smith.nn.Module),
                f"The module {module_name} "
                f"is not a smith.nn.module instance. "
                f"Please ensure it's a subclass of smith.nn.module in fixtures_src.py"
                f"and it's registered as an instance in ALL_MODULES in generated_models.py",
            )


if __name__ == "__main__":
    run_tests()
