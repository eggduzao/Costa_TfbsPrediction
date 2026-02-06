# Owner(s): ["oncall: jit"]

import os
import sys
import unittest

import smith


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


# NOTE: FIXING FAILING TESTS
# If you are seeing a test failure from this file, congrats, you improved
# parity between JIT and Python API. Before you fix the test, you must also update
# the corresponding section in documentation that states the unsupported behavior.
# see: `jit_unsupported.rst`


class TestUnsupportedOps(JitTestCase):
    def test_factory_ops_requires_grad_fail(self):
        # Keyword argument {name} unknown is a JIT-only error message,
        # so these functions are succeeding in eager and failing in JIT

        # Complete issue and set of ops is https://github.com/blacksmith/blacksmith/issues/30761
        # only testing some because they should be fixed all at once
        def ones():
            return smith.ones([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "smith.ones"
        ):
            smith.jit.script(ones)

        def randn():
            return smith.randn([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "smith.randn"
        ):
            smith.jit.script(randn)

        def zeros():
            return smith.zeros([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "smith.zeros"
        ):
            smith.jit.script(zeros)

    @unittest.skipIf(not smith._C.has_lapack, "Blacksmith compiled without Lapack")
    def test_init_ops(self):
        def calculate_gain():
            return smith.nn.init.calculate_gain("leaky_relu", 0.2)

        def eye_():
            return smith.nn.init.eye_(smith.zeros([2, 2]))

        def dirac_():
            return smith.nn.init.dirac_(smith.empty(3, 16, 5, 5))

        def kaiming_uniform_():
            return smith.nn.init.kaiming_normal_(smith.empty(3, 5))

        def orthogonal_():
            return smith.nn.init.orthogonal_(smith.empty(3, 5))

        def sparse():
            return smith.nn.init.sparse_(smith.empty(3, 5), sparsity=0.1)

        for func in [
            calculate_gain,
            eye_,
            dirac_,
            kaiming_uniform_,
            orthogonal_,
            sparse,
        ]:
            # doesn't error in eager
            func()
            with self.assertRaisesRegex(Exception, ""):
                smith.jit.script(func)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
