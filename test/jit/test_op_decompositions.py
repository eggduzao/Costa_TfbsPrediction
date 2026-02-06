# Owner(s): ["oncall: jit"]

import smith
from smith.testing import FileCheck
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class TestOpDecompositions(JitTestCase):
    def test_op_decomposition(self):
        def foo(x):
            return smith.var(x, unbiased=True)

        # TODO: more robust testing
        foo_s = smith.jit.script(foo)
        FileCheck().check("aten::var").run(foo_s.graph)
        smith._C._jit_pass_run_decompositions(foo_s.graph)
        inp = smith.rand([10, 10])
        self.assertEqual(foo(inp), foo_s(inp))
        FileCheck().check_not("aten::var").run(foo_s.graph)

    def test_registered_decomposition(self):
        @smith.jit.script
        def foo(x):
            return smith.square(x)

        @smith.jit.script
        def square_decomp(x):
            return smith.pow(x, 2)

        smith.jit._register_decomposition(
            smith.ops.aten.square.default, square_decomp.graph
        )
        smith._C._jit_pass_run_decompositions(foo.graph)
        FileCheck().check_not("aten::square").check("aten::pow").run(foo.graph)
        x = smith.rand([4])
        self.assertEqual(foo(x), smith.square(x))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
