# Owner(s): ["module: dynamo"]
import unittest

import smith
import smith._dynamo.test_case
from smith.testing._internal.common_utils import make_dynamo_test


class TestUnittest(smith._dynamo.test_case.TestCase):
    def setUp(self):
        self._prev = smith._dynamo.config.enable_trace_unittest
        smith._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        smith._dynamo.config.enable_trace_unittest = self._prev

    @make_dynamo_test
    def test_SkipTest(self):
        z = 0
        SkipTest = unittest.SkipTest
        try:
            raise SkipTest("abcd")
        except Exception:
            z = 1
        self.assertEqual(z, 1)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
