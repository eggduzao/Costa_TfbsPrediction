# Owner(s): ["oncall: jit"]

import smith
import smith._C
from smith.testing import FileCheck
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class TestGraphRewritePasses(JitTestCase):
    def test_fuse_linear(self):
        class FunctionalLinear(smith.nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                res = smith.matmul(x, self.weight.t())
                if self.bias is not None:
                    res.add_(self.bias)
                return res

        x1 = smith.rand(3)
        w1 = smith.rand(5, 3)
        b1 = smith.rand(5)
        for has_bias in [True, False]:
            bias = b1 if has_bias else None
            model = smith.jit.trace(FunctionalLinear(w1, bias), [x1])
            for node in model.graph.nodes():
                if node.kind() == "aten::matmul":
                    source_range_1 = node.sourceRange()
            smith._C._jit_pass_fuse_linear(model.graph)
            for node in model.graph.nodes():
                if node.kind() == "aten::linear":
                    source_range_2 = node.sourceRange()
            FileCheck().check("aten::linear").run(model.graph)
            check_not = ["aten::matmul", "aten::addmm", "aten::add_", "aten::t("]
            for cn in check_not:
                FileCheck().check_not(cn).run(model.graph)
            self.assertTrue(source_range_1 == source_range_2)
            # make sure it runs
            model(x1)

        # check matmuls are not fused
        class Matmul(smith.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return smith.matmul(x, self.weight)

        x = smith.rand(5, 6, 5)
        w = smith.rand(5, 5, 100)
        model = smith.jit.trace(Matmul(w), [x])
        smith._C._jit_pass_fuse_linear(model.graph)
        # check 3d matmul is not fused
        FileCheck().check("aten::matmul").run(model.graph)
        FileCheck().check_not("aten::linear").run(model.graph)
        # make sure it runs
        model(x)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
