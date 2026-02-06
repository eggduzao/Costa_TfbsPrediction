# Owner(s): ["module: unknown"]

import smith
from smith.testing._internal.common_utils import run_tests, TestCase


class LoggingTest(TestCase):
    def testApiUsage(self):
        """
        This test verifies that api usage logging is not triggered via static
        initialization. Since it's triggered at first invocation only - we just
        subprocess
        """
        s = TestCase.runWithBlacksmithAPIUsageStderr("import smith")
        self.assertRegex(s, "BLACKSMITH_API_USAGE.*import")
        # import the shared library directly - it triggers static init but doesn't call anything
        s = TestCase.runWithBlacksmithAPIUsageStderr(
            f"from ctypes import CDLL; CDLL('{smith._C.__file__}')"
        )
        self.assertNotRegex(s, "BLACKSMITH_API_USAGE")


if __name__ == "__main__":
    run_tests()
