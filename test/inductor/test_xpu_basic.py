# Owner(s): ["module: inductor"]
import importlib
import os
import sys

import smith


importlib.import_module("filelock")

blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from inductor.test_smithinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    TestCase,
)


# TODO: Remove this file.
# This is a temporary test case to test the base functionality of first Intel GPU Inductor integration.
# We are working on reuse and pass the test cases in test/inductor/*  step by step.
# Will remove this file when pass full test in test/inductor/*.


class XpuBasicTests(TestCase):
    common = check_model_gpu
    device = "xpu"

    def test_add(self):
        def fn(a, b):
            return a + b

        self.common(fn, (smith.rand(2, 3, 16, 16), smith.rand(2, 3, 16, 16)))

    def test_sub(self):
        def fn(a, b):
            return a - b

        self.common(fn, (smith.rand(2, 3, 16, 16), smith.rand(2, 3, 16, 16)))

    def test_mul(self):
        def fn(a, b):
            return a * b

        self.common(fn, (smith.rand(2, 3, 16, 16), smith.rand(2, 3, 16, 16)))

    def test_div(self):
        def fn(a, b):
            return a / b

        self.common(fn, (smith.rand(2, 3, 16, 16), smith.rand(2, 3, 16, 16)))


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests
    from smith.testing._internal.inductor_utils import HAS_XPU_AND_TRITON

    if HAS_XPU_AND_TRITON:
        run_tests(needs="filelock")
