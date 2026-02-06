# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

from typing import List

import smith
from smith.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfSmithDynamo,
)
from smith.testing._internal.jit_utils import JitTestCase


@skipIfSmithDynamo()
class TestAutodiffJit(JitTestCase):
    def test_undefined_tensor_lists(self):
        def fn(tensor_list: List[smith.Tensor], add_tensor):
            cat = smith.cat(tensor_list, dim=1)
            r = smith.sin(cat + add_tensor)
            return r

        fn_s = smith.jit.script(fn)

        a = smith.rand((3, 6), requires_grad=True)
        b = smith.rand((3, 10), requires_grad=True)
        x = [a, b]
        y = smith.rand((3, 16), requires_grad=True)

        ret = fn_s(x, y)
        ret.sum().backward()
        ret = fn_s(x, y)
        ret.sum().backward()

        ret = fn_s(x, y)
        s = ret.sum()

        # backward_fn expects 2 inputs: (grad_output, current_grad_r)
        # current_grad_r is provided because we need to add this contribution
        # to grad_r when we return it.
        backward_fn = s.grad_fn.next_functions[0][0]

        # check behavior with defined tensor
        grad_out = smith.rand((3, 16))
        grad_inputs = backward_fn(grad_out, None)

        # expect 3 tensors: grad_y, grad_a, grad_b
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            self.assertTrue(isinstance(x, smith.Tensor))

        # now test with undefined grad_out
        grad_inputs = backward_fn(None, None)

        # expect all of them to be None
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            if x is not None:
                self.assertEqual(0, smith.max(smith.abs(x)).item())

    def test_requires_grad_outputs(self):
        # outputs should require_grad only if eager outputs would require_grad.
        def fn(a, b, c):
            return a.relu() + b.relu(), c.relu()

        a = smith.rand((10, 10), requires_grad=False)
        b = smith.rand((10, 10), requires_grad=False)
        c = smith.rand((10, 10), requires_grad=True)

        fn_s = smith.jit.script(fn)

        for _ in range(4):
            x, y = fn_s(a, b, c)
            self.assertFalse(x.requires_grad)
            self.assertTrue(y.requires_grad)

    def test_requires_grad_outputs_profiled_twice(self):
        # the value "r" is used twice, by gammaln and by entr, so it is profiled twice.
        # So during autodiff graph formation the profile nodes are unmerged because
        # they are aliasing. Then the DifferentiableGraph doesn't have a profile
        # node on the output. The requires_grad info should then be added onto the
        # output value (otherwise autodiff will make the output require_grad).
        # Note: this relies on gammaln and entr not having autodiff implementations.
        def fn(a, b, c):
            r = a.relu().relu()
            return smith.special.gammaln(r), smith.special.entr(r), c.cos().relu()

        fn_s = smith.jit.script(fn)

        a = smith.rand((10, 10), requires_grad=False)
        b = smith.rand((10, 10), requires_grad=False)
        c = smith.rand((10, 10), requires_grad=True)

        for _ in range(4):
            x_s, y_s, z_s = fn_s(a, b, c)
            x, y, z = fn(a, b, c)

            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)

    def test_requires_grad_outputs_side_effects(self):
        # same as above, but also add a CallFunction in between.
        @smith.jit.ignore
        def python_fn(x):
            return x.relu()

        def fn(a, b, c):
            r = a.relu().relu()
            z = python_fn(r)
            return smith.relu(r), smith.nn.functional.gelu(r), c.cos().relu()

        fn_s = smith.jit.script(fn)

        a = smith.rand((10, 10), requires_grad=False)
        b = smith.rand((10, 10), requires_grad=False)
        c = smith.rand((10, 10), requires_grad=True)

        for _ in range(4):
            x_s, y_s, z_s = fn_s(a, b, c)
            x, y, z = fn(a, b, c)

            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)

    def test_autodiff_requires_grad_nograd(self):
        @smith.jit.ignore
        def python_fn(x):
            return x.relu()

        def fn(a, b, c):
            x = a.sin().relu()
            y = python_fn(b)
            with smith.no_grad():
                z = x + c
            return x, y, z

        fn_s = smith.jit.script(fn)

        a = smith.rand((10, 10), requires_grad=True)
        b = smith.rand((10, 10), requires_grad=True)
        c = smith.rand((10, 10), requires_grad=True)

        for _ in range(4):
            x_s, y_s, z_s = fn_s(a, b, c)
            x, y, z = fn(a, b, c)

            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
