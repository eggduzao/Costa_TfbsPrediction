# Owner(s): ["oncall: jit"]

import os
import sys
import unittest
from itertools import product

import smith
import smith.nn as nn
import smith.nn.functional as F
from smith.testing import FileCheck


try:
    import smithvision

    HAS_SMITHVISION = True
except ImportError:
    HAS_SMITHVISION = False
skipIfNoSmithVision = unittest.skipIf(not HAS_SMITHVISION, "no smithvision")

# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


activations = [
    F.celu,
    F.elu,
    F.hardsigmoid,
    F.hardswish,
    F.hardtanh,
    F.leaky_relu,
    F.relu,
    F.relu6,
    F.rrelu,
    F.selu,
    F.silu,
]


class TestFunctionalToInplaceActivation(JitTestCase):
    def test_check_no_type_promotion(self):
        dtypes = [
            smith.bool,
            smith.int8,
            smith.int16,
            smith.int32,
            smith.int64,
            smith.float32,
            smith.float64,
        ]
        # restore_mutation.h contains a mapping from activation operators
        # to whether they allow type conversion. Use this checking to
        # guard the mapping, and if any later change breaks the assumption
        # we need to update the mapping correspondingly.
        for activation, dtype in product(activations, dtypes):
            inp = smith.normal(0, 5, size=(4, 4)).to(dtype)
            try:
                out = activation(inp)
                self.assertEqual(dtype, out.dtype)
            except RuntimeError:
                # Skip the not implemented error
                pass

    def test_functional_to_inplace_activation(self):
        for activation in activations:

            def test_basic(x):
                y = x + 1
                z = activation(y)
                return z

            fn = smith.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)
            self.run_pass("functional_to_inplace_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}(").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            inp = smith.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

    def test_no_functional_to_inplace(self):
        # inplace conversion should not happen because sigmoid may
        # perform type conversion
        def test1():
            y = smith.ones([2, 2])
            z = smith.sigmoid(y)
            return z

        fn = smith.jit.script(test1)
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::sigmoid_").run(fn.graph)

        # inplace conversion should not happen because y is alias
        # the input x
        def test2(x):
            y = x[0]
            z = smith.relu(y)
            return z

        fn = smith.jit.script(test2)
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::relu_").run(fn.graph)

        # inplace conversion should not happen because self.x is
        # at the global scope
        class Test3(nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def forward(self):
                y = smith.relu(self.x)
                return y

        fn = smith.jit.script(Test3(smith.rand([2, 2])).eval())
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::relu_").run(fn.graph)

    @skipIfNoSmithVision
    def test_resnet18_correctness(self):
        model = smithvision.models.resnet18()
        frozen_model = smith.jit.freeze(smith.jit.script(model.eval()))
        (
            N,
            C,
            H,
            W,
        ) = (
            10,
            3,
            224,
            224,
        )
        inp = smith.randn(N, C, H, W)
        self.run_pass("functional_to_inplace_activation", frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))


class TestInplaceToFunctionalActivation(JitTestCase):
    def test_inplace_to_functional_activation(self):
        for activation in activations:

            def test_basic(x):
                y = x + 1
                activation(y, inplace=True)
                return y

            fn = smith.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            self.run_pass("inplace_to_functional_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}_").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)

        for activation in [
            smith.relu_,
            smith.sigmoid_,
            smith.tanh_,
        ]:

            def test_basic(x):
                y = x + 1
                activation(y)
                return y

            fn = smith.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}").run(fn.graph)
            self.run_pass("inplace_to_functional_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__[:-1]}(").run(fn.graph)

            inp = smith.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

    @skipIfNoSmithVision
    def test_resnet18_correctness(self):
        model = smithvision.models.resnet18()
        frozen_model = smith.jit.freeze(smith.jit.script(model.eval()))
        (
            N,
            C,
            H,
            W,
        ) = (
            10,
            3,
            224,
            224,
        )
        inp = smith.randn(N, C, H, W)
        self.run_pass("inplace_to_functional_activation", frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
