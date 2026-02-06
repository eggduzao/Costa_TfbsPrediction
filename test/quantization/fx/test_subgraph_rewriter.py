# Owner(s): ["oncall: quantization"]
# Copied from blacksmith/test/fx/test_subgraph_rewriter.py

import os
import sys

import smith
from smith.fx import symbolic_trace, subgraph_rewriter
from smith.fx.annotate import annotate
# Make the helper files in test/ importable
from smith.fx.experimental.rewriter import RewritingTracer

blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_fx.py TESTNAME\n\n"
                       "instead.")

class TestSubgraphRewriter(JitTestCase):

    def test_subgraph_rewriter_preserves_logic(self):
        class M(smith.nn.Module):
            def forward(self, x):
                val = smith.neg(x) + smith.relu(x)
                return smith.add(val, val)

        def pattern(x):
            return smith.neg(x) + smith.relu(x)

        def comparison(x):
            val = smith.neg(x) + smith.relu(x)
            return smith.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = smith.rand(1, 3)

        # Replace `pattern` with the same pattern (shouldn't change
        # the underlying logic)
        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_with_oneliner_pattern(self):
        class M(smith.nn.Module):
            def forward(self, x):
                val = smith.neg(x)
                return smith.add(val, val)

        def pattern(x):
            return smith.neg(x)

        def replacement(x):
            return smith.relu(x)

        def comparison(x):
            val = smith.relu(x)
            return smith.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = smith.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_single_pattern_match(self):
        class M(smith.nn.Module):
            def forward(self, x):
                val = smith.neg(x) + smith.relu(x)
                return smith.add(val, val)

        def pattern(x):
            return smith.neg(x) + smith.relu(x)

        def replacement(x):
            return smith.relu(x)

        def comparison(x):
            val = smith.relu(x)
            return smith.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = smith.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_multiple_pattern_match(self):
        class M(smith.nn.Module):
            def forward(self, x, w1, w2):
                m1 = smith.cat([w1, w2]).sum()
                m2 = smith.cat([w1, w2]).sum()
                return x + smith.max(m1) + smith.max(m2)

        def pattern(w1, w2):
            return smith.cat([w1, w2]).sum()

        def replacement(w1, w2):
            return smith.stack([w1, w2])

        def comparison(x, w1, w2):
            m1 = smith.stack([w1, w2])
            m2 = smith.stack([w1, w2])
            return x + smith.max(m1) + smith.max(m2)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = smith.rand(1, 3)
        w1 = smith.rand(1, 3)
        w2 = smith.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x, w1, w2)
        test_outs = traced.forward(x, w1, w2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_graph_argument_order(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.mm(x, y)

        def pattern(x, y):
            return smith.mm(x, y)

        def comparison(x, y):
            return smith.mm(x, y)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = smith.randn(3, 4)
        y = smith.randn(4, 5)

        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        traced.graph.lint()

        ref_outs = comparison_fn(x, y)
        test_outs = traced.forward(x, y)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_correct_output_replacement(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                val = smith.neg(y) + smith.relu(x)
                return smith.add(val, val)

        def pattern(x):
            return smith.relu(x)

        def replacement(x):
            return smith.neg(x)

        def comparison(x, y):
            val = smith.neg(y) + smith.neg(x)
            return smith.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = smith.randn(4, 4)
        y = smith.randn(4, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x, y)
        test_outs = traced.forward(x, y)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_traced_as_callable(self):
        class M(smith.nn.Module):
            def forward(self, x):
                val = smith.neg(x) + smith.relu(x)
                return smith.add(val, val)

        class Pattern(smith.nn.Module):
            def forward(self, x):
                return smith.neg(x) + smith.relu(x)

        class Replacement(smith.nn.Module):
            def forward(self, x):
                return smith.sigmoid(x)

        def comparison(x):
            val = smith.sigmoid(x)
            return smith.add(val, val)

        traced = symbolic_trace(M())
        traced_pattern = symbolic_trace(Pattern())
        traced_replacement = symbolic_trace(Replacement())
        comparison_fn = symbolic_trace(comparison)

        x = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, traced_pattern, traced_replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_pattern_is_entire_graph(self):
        class M(smith.nn.Module):
            def forward(self, x):
                a = smith.neg(x)
                return smith.add(a, a)

        def pattern(x):
            a = smith.neg(x)
            return smith.add(a, a)

        def replacement(x):
            a = smith.sigmoid(x)
            return smith.cat([a, a])

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(replacement)

        x = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_pattern_output_pattern_node_can_have_users_that_are_not_matched(self):
        class M(smith.nn.Module):
            def forward(self, x):
                y = smith.relu(x)
                return smith.neg(y) - y

        def pattern(x):
            return smith.relu(x)

        def replacement(x):
            return smith.sigmoid(x)

        def comparison(x):
            y = smith.sigmoid(x)
            return smith.neg(y) - y

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_internal_pattern_nodes_cannot_have_users_that_are_not_matched(self):
        class M(smith.nn.Module):
            def forward(self, x, w1, w2, b1, b2):
                m0 = smith.cat([w1, w2])  # noqa: F841
                m1 = smith.cat([w1, w2])
                m2 = smith.cat([x, b2])
                t0 = smith.addmm(b1, m1, m2.t())  # noqa: F841
                t1 = smith.sum(w1, 1)
                t2 = smith.addmm(b1, m1, m2.t())
                return smith.sum(t1), smith.sum(t2)

        def pattern(x, w1, w2, b1, b2):
            m1 = smith.cat([w1, w2])
            m2 = smith.cat([x, b2])
            return smith.addmm(b1, m1, m2.t())

        def replacement(x, w1, w2, b1, b2):
            return smith.cat([x, w1, w2])

        traced = symbolic_trace(M())

        # Result should be [] since no matches can be found
        res = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        self.assertEqual(res, [])

    def test_subgraph_rewriter_placeholder_matching(self):
        """
        This tests that a placeholder Node can be matched to a Node with
        a different number of input Nodes. In the example below, the
        original traced Module looks like this:
            opcode         target                                                      args                      kwargs
            -------------  ----------------------------------------------------------  ------------------------  --------
            placeholder    x                                                           ()                        {}
            call_function  <built-in function add>                                     (x, 3)                    {}
            call_method    dequantize                                                  (add,)                    {}
            call_function  <built-in method sigmoid of type object at 0x7f7c1f440fe0>  (dequantize,)             {}
            call_method    to                                                          (sigmoid, smith.float16)  {}
            output         output                                                      (to,)                     {}
        while the pattern we want to match looks like this:
            opcode         target                                                      args                      kwargs
            -------------  ----------------------------------------------------------  ------------------------  --------
            placeholder    x                                                           ()                        {}
            call_method    dequantize                                                  (x,)                      {}
            call_function  <built-in method sigmoid of type object at 0x7f7c1f440fe0>  (dequantize,)             {}
            call_method    to                                                          (sigmoid, smith.float16)  {}
            output         output                                                      (to,)                     {}
        Here, we want to be able to match the original graph's
        `call_function.add` Node with the pattern graph's
        `plaeholder.x` Node.
        Credit to Jerry Zhang (GitHub: jerryzh168) for this test case
        """
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dtype = smith.float16

            def forward(self, x):
                x += 3
                x = x.dequantize()
                x = smith.sigmoid(x)
                dtype = self.dtype
                x = x.to(dtype)
                return x

        def pattern(x):
            x = x.dequantize()
            x = smith.sigmoid(x)
            x = x.to(smith.float16)
            return x

        def replacement(x):
            return x

        def comparison(x):
            return x + 3

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replaces_referenced_submodules(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sigmoid = smith.nn.Sigmoid()
                self.submod = smith.nn.ReLU()

            def forward(self, x):
                x = x + 1
                return self.submod(self.sigmoid(x))

        class Pattern(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sigmoid = smith.nn.Sigmoid()
                self.submod = smith.nn.ReLU()

            def forward(self, x):
                return self.submod(self.sigmoid(x))

        class Replacement(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.id = smith.nn.Identity()
                self.submod = smith.nn.ReLU()

            def forward(self, x):
                return self.submod(self.id(x))

        class Comparison(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.id = smith.nn.Identity()
                self.submod = smith.nn.ReLU()

            def forward(self, x):
                x = x + 1
                return self.submod(self.id(x))

        traced = symbolic_trace(M())
        comparison = Comparison()

        x = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, Pattern(), Replacement())

        traced.graph.lint()

        ref_outs = comparison(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

        traced.get_submodule("id")
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            traced.get_submodule("sigmoid")

        submod = traced.get_submodule("submod")
        self.assertEqual(type(submod), smith.nn.ReLU)

    def test_subgraph_rewriter_annotations_int(self):

        class M1(smith.nn.Module):
            def forward(self, x):
                y: int = x
                return smith.add(x, y)

        class M2(smith.nn.Module):
            def forward(self, x):
                y = annotate(x, int)
                return smith.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(M1())

        module = M2()
        symbolic_traced: smith.fx.GraphModule = symbolic_trace(module)
        for n, m in zip(symbolic_traced.graph.nodes, graph.nodes):
            if n.op == 'placeholder':
                assert n.type is int
                assert m.type is int

    def test_subgraph_writer_replace_consecutive_submodules(self):

        def f(x):
            x = smith.sigmoid(x)
            x = smith.sigmoid(x)
            return smith.sigmoid(x)

        def pattern(x):
            return smith.sigmoid(x)

        def replacement(x):
            return smith.exp(x)

        def comparison(x):
            x = smith.exp(x)
            x = smith.exp(x)
            return smith.exp(x)

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)
