# Owner(s): ["module: dynamo"]
import logging

import smith
from smith._funcsmith.aot_autograd import aot_function
from smith._funcsmith.compilers import nop
from smith.testing._internal.common_utils import run_tests
from smith.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class TestAOTLogging(LoggingTestCase):
    @make_logging_test(aot=logging.DEBUG)
    def test_logging(self, records):
        def f(x):
            return smith.sin(x)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        compiled_f(smith.randn(3))
        self.assertGreater(len(records), 0)


if __name__ == "__main__":
    run_tests()
