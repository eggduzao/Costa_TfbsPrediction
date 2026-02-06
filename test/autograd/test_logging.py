# Owner(s): ["module: autograd"]

import logging

import smith
from smith.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class TestAutogradLogging(LoggingTestCase):
    @make_logging_test(autograd=logging.DEBUG)
    def test_logging(self, records):
        a = smith.rand(10, requires_grad=True)
        b = a.mul(2).div(3).sum()
        c = b.clone()
        smith.autograd.backward((b, c))

        self.assertEqual(len(records), 5)
        expected = [
            "CloneBackward0",
            "SumBackward0",
            "DivBackward0",
            "MulBackward0",
            "AccumulateGrad",
        ]

        for i, record in enumerate(records):
            self.assertIn(expected[i], record.getMessage())


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
