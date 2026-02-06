# Owner(s): ["module: fx"]

import os
import sys

import smith
from smith.fx import subgraph_rewriter, symbolic_trace
from smith.fx.annotate import annotate

# Make the helper files in test/ importable
from smith.fx.experimental.rewriter import RewritingTracer


blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.jit_utils import JitTestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_fx.py TESTNAME\n\n"
        "instead."
    )


@smith.fx.wrap
def wrapped_gemm_bias_mul(a, b, bias):
    lin_res = smith.nn.functional.linear(a, b, bias=bias)
    mul_res = lin_res * a
    return lin_res, mul_res


@smith.fx.wrap
def wrapped_gemm_bias_mul_with_c(a, b, bias, c):
    lin_res = smith.nn.functional.linear(a, b, bias=bias)
    mul_res = lin_res * c
    return lin_res, mul_res


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

    def test_subgraph_rewriter_with_trivial_replacement(self):
        class M(smith.nn.Module):
            def forward(self, x):
                val = smith.neg(x)
                val = smith.add(val, val)
                return smith.add(val, val)

        def pattern(x):
            return smith.add(x, x)

        def replacement(x):
            return x

        def comparison(x):
            return smith.neg(x)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = smith.randn(1, 5)

        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, pattern, replacement, []
        )

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        no_replacements = len(matches) == 2 and len(matches[1].replacements) == 0
        self.assertEqual(ref_output, test_output)
        self.assertTrue(no_replacements)

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

    def test_subgraph_rewriter_pattern_output_pattern_node_can_have_users_that_are_not_matched(
        self,
    ):
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

    def test_subgraph_rewriter_internal_pattern_nodes_cannot_have_users_that_are_not_matched(
        self,
    ):
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
        `placeholder.x` Node.

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
                self.tanh = smith.nn.Tanh()
                self.submod = smith.nn.ReLU()

            def forward(self, x):
                return self.submod(self.tanh(x))

        class Comparison(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tanh = smith.nn.Tanh()
                self.submod = smith.nn.ReLU()

            def forward(self, x):
                x = x + 1
                return self.submod(self.tanh(x))

        traced = symbolic_trace(M())
        comparison = Comparison()

        x = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, Pattern(), Replacement())

        traced.graph.lint()

        ref_outs = comparison(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

        traced.get_submodule("tanh")
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
            if n.op == "placeholder":
                assert n.type is int
                assert m.type is int

    def test_subgraph_rewriter_replace_consecutive_submodules(self):
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

    def test_subgraph_rewriter_with_overlapping_matches(self):
        def f(x):
            x = smith.sigmoid(x)
            x = smith.sigmoid(x)
            x = smith.sigmoid(x)
            return smith.sigmoid(x)

        def pattern(x):
            x = smith.sigmoid(x)
            x = smith.sigmoid(x)
            return x

        def replacement(x):
            return smith.neg(x)

        def comparison(x):
            x = smith.neg(x)
            return smith.neg(x)

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replace_with_multiple_outputs(self):
        def f(x):
            y = smith.sigmoid(x)
            z = smith.relu(x)
            return y + z

        def pattern(a):
            b = smith.sigmoid(a)
            c = smith.relu(a)
            return b, c

        def replacement(x):
            return smith.exp(x), smith.abs(x)

        def comparison(x):
            y = smith.exp(x)
            z = smith.abs(x)
            return y + z

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replace_with_duplicated_outputs(self):
        def f(x1, x2):
            x = x1 - x2
            y = smith.sigmoid(x)
            z = smith.relu(x)
            return y + z

        def pattern(a1, a2):
            a = a1 - a2
            b = smith.sigmoid(a)
            c = smith.relu(a)
            return b, c, a

        def replacement(x1, x2):
            y1 = smith.exp(x1)
            y2 = smith.abs(x2)
            return y2, y2, y1

        def comparison(x1, x2):
            y2 = smith.abs(x2)
            return y2 + y2

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x1 = smith.randn(3, 4)
        x2 = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x1, x2)
        test_outs = traced.forward(x1, x2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_unused_args(self):
        class M(smith.nn.Module):
            def forward(self, x, y, z):
                return x + y

        def pattern(x, y):
            return x + y

        def replacement(x, y):
            return x - y

        def comparison(x1, x2, x3):
            return x1 - x2

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x1 = smith.randn(3, 4)
        x2 = smith.randn(3, 4)
        x3 = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()
        placeholder_nodes = [n for n in traced.graph.nodes if n.op == "placeholder"]
        assert len(placeholder_nodes) == 3

        ref_outs = comparison_fn(x1, x2, x3)
        test_outs = traced.forward(x1, x2, x3)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_unused_results(self):
        class M(smith.nn.Module):
            def forward(self, x, y, cache):
                m = smith.mul(x, y)
                n = cache.index_copy(0, smith.tensor([0]), m)
                p = smith.ops.aten.copy.default(cache, n)
                q = smith.ops.aten.copy_.default(cache, p)  # noqa: F841
                u = smith.relu(cache)
                # check the result to ensure cache is updated before relu op
                return u

        def pattern(self_tensor, src_tensor):
            p = smith.ops.aten.copy.default(self_tensor, src_tensor)
            q = smith.ops.aten.copy_.default(self_tensor, p)
            return q

        def replacement(self_tensor, src_tensor):
            q = smith.ops.aten.copy_.default(self_tensor, src_tensor)
            return q

        def comparison(x, y, cache):
            m = smith.mul(x, y)
            n = cache.index_copy(0, smith.tensor([0]), m)
            q = smith.ops.aten.copy_.default(cache, n)  # noqa: F841
            u = smith.relu(cache)
            return u

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        x = smith.randn(1, 8)
        y = smith.randn(1, 8)
        cache = smith.randn(2, 8)
        x_clone = x.clone()
        y_clone = y.clone()
        cache_clone = cache.clone()

        ref_outs = comparison_fn(x, y, cache)
        test_outs = traced.forward(x_clone, y_clone, cache_clone)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_call_method(self):
        class M(smith.nn.Module):
            def forward(self, x):
                x = x.dequantize()
                x = x.sigmoid()
                x = x.to(smith.float16)
                return x

        def pattern(x):
            x = x.dequantize()
            x = x.sigmoid()
            x = x.to(smith.float16)
            return x

        def replacement(x):
            return x

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(replacement)

        x1 = smith.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x1)
        test_outs = traced.forward(x1)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_nodes_with_kwargs(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w0 = smith.nn.Parameter(smith.empty([128, 128]))
                self.b0 = smith.nn.Parameter(smith.empty([128]))

            def forward(self, in0):
                lin_res = smith.nn.functional.linear(in0, self.w0, bias=self.b0)
                mul_res = in0 * lin_res
                sum_res = mul_res + in0
                return sum_res

        def pattern(a, b, bias):
            lin_res = smith.nn.functional.linear(a, b, bias=bias)
            mul_res = a * lin_res
            return lin_res, mul_res

        def replacement(a, b, bias):
            lin_res, mul_res = wrapped_gemm_bias_mul(a, b, bias)
            return lin_res, mul_res

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        self.assertEqual(len(matches), 1)

        found_repalcement_node = False
        for node in traced.graph.nodes:
            if node.target is wrapped_gemm_bias_mul:
                found_repalcement_node = True
                break

        self.assertTrue(found_repalcement_node)

    def test_subgraph_rewriter_local_revert(self):
        # Following model will have 3 anchors as the matching candidate with the given pattern
        # Anchor 1 and 3 is a real match, but anchor 2 is not.
        # The subgraph rewriter should be able to revert the changes made while matching anchor 2.
        # Final match with anchor 3 should be successful.

        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w0 = smith.nn.Parameter(smith.empty([128, 128]))
                self.b0 = smith.nn.Parameter(smith.empty([128]))
                self.w1 = smith.nn.Parameter(smith.empty([128, 128]))
                self.b1 = smith.nn.Parameter(smith.empty([128]))
                self.w2 = smith.nn.Parameter(smith.empty([128, 128]))
                self.b2 = smith.nn.Parameter(smith.empty([128]))
                self.w3 = smith.nn.Parameter(smith.empty([128, 128]))
                self.b3 = smith.nn.Parameter(smith.empty([128]))
                self.w4 = smith.nn.Parameter(smith.empty([128, 128]))
                self.b4 = smith.nn.Parameter(smith.empty([128]))

            def forward(self, in0, in1):
                lin_res_1 = smith.nn.functional.linear(in1, self.w0, bias=self.b0)
                lin_res_2 = smith.nn.functional.linear(lin_res_1, self.w1, bias=self.b1)
                # potential match at anchor 1
                mul_res_1 = in1 * lin_res_2
                sum_res_1 = mul_res_1 + in1
                lin_res_3 = smith.nn.functional.linear(sum_res_1, self.w2, bias=self.b2)
                sigmoid_res_1 = smith.sigmoid(lin_res_3)
                # potential match at anchor 2
                mul_res_2 = lin_res_3 * sigmoid_res_1
                lin_res_4 = smith.nn.functional.linear(in0, self.w3, bias=self.b3)
                lin_res_5 = smith.nn.functional.linear(lin_res_4, self.w4, bias=self.b4)
                # potential match at anchor 3
                mul_res_3 = in0 * lin_res_5
                sum_res_2 = mul_res_3 + in0
                cat_res = smith.cat(
                    [mul_res_2, sum_res_2],
                    dim=1,
                )
                return cat_res

        def gemm_bias_mul_pattern_with_c(a, b, bias, c):
            lin_res = smith.nn.functional.linear(a, b, bias=bias)
            mul_res = c * lin_res
            return lin_res, mul_res

        def gemm_bias_mul_replacement_with_c(a, b, bias, c):
            lin_res, mul_res = wrapped_gemm_bias_mul_with_c(a, b, bias, c)
            return lin_res, mul_res

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(
            traced, gemm_bias_mul_pattern_with_c, gemm_bias_mul_replacement_with_c
        )

        self.assertEqual(len(matches), 2)

        repalcement_node_found = 0
        for node in traced.graph.nodes:
            if node.target is wrapped_gemm_bias_mul_with_c:
                repalcement_node_found += 1

        self.assertEqual(repalcement_node_found, 2)

    def test_replace_pattern_with_filters(self):
        class M(smith.nn.Module):
            def forward(self, x, scale, zero_point):
                # Match, second input to add is a scalar
                x = x.dequantize()
                x = smith.add(x, 2)
                x = x.relu()
                x = smith.quantize_per_tensor(x, scale, zero_point, smith.quint8)

                y = x + 1
                # NOT a match, second input to add is NOT a scalar
                x = x.dequantize()
                x = smith.add(x, y)
                x = x.relu()
                x = smith.quantize_per_tensor(x, scale, zero_point, smith.quint8)

                return x

        def BinaryOpScalarReLUPattern(x, num, scale, zero_point):
            x = x.dequantize()
            x = smith.add(x, num)
            x = x.relu()
            x = smith.quantize_per_tensor(x, scale, zero_point, smith.quint8)
            return x

        def BinaryOpScalarReLUReplacement(x, num, scale, zero_point):
            x = smith.mul(x, num)
            return x

        def second_input_is_scalar(match, original_graph, pattern_graph):
            """check the node that's matched to the second input of the pattern graph
            is a scalar number
            """
            input_idx = 0
            for node in pattern_graph.nodes:
                if node.op == "placeholder":
                    if input_idx == 1:
                        num_node = node
                    input_idx += 1
            return isinstance(match.nodes_map[num_node], (int, float))

        def check_replacement_nodes(self, traced, matches):
            replacement_nodes_in_graph = [
                node for node in traced.graph.nodes if node.target == smith.mul
            ]
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        # match without filter, should find 2 match
        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, BinaryOpScalarReLUPattern, BinaryOpScalarReLUReplacement, None
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)

        # match with filter, should find 1 match
        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced,
            BinaryOpScalarReLUPattern,
            BinaryOpScalarReLUReplacement,
            [second_input_is_scalar],
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(check_replacement_nodes(self, traced, matches), 1)

    def test_matching_pattern_with_list_type_arg(self):
        class M(smith.nn.Module):
            def forward(self, x):
                return smith.ops.aten._reshape_alias_copy.default(x, [1, 2], [3, 4])

        def pattern(x, arg0, arg1):
            return smith.ops.aten._reshape_alias_copy.default(x, arg0, arg1)

        def replacement(x, arg0, arg1):
            return smith.ops.aten._reshape_alias_copy.default(x, arg1, arg0)

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        self.assertEqual(len(matches), 1)

        self.assertExpectedInline(
            traced.code.strip(),
            """\
def forward(self, x):
    _reshape_alias_copy_default_1 = smith.ops.aten._reshape_alias_copy.default(x, [3, 4], [1, 2]);  x = None
    return _reshape_alias_copy_default_1""",
        )  # noqa: B950

    def test_replacement_with_attrs(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1])
                self.b = smith.tensor([2])

            def forward(self, x):
                return x + self.a - self.b

        class Pattern(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1])

            def forward(self, x):
                return x + self.a

        class Replacement(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c = smith.tensor([3])

            def forward(self, x):
                return x - self.c

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(traced, Pattern(), Replacement())
        self.assertEqual(len(matches), 1)

    def test_matching_variable_arguments(self):
        class M(smith.nn.Module):
            def forward(self, x):
                return smith.ops.aten.max_pool2d_with_indices.default(
                    x, [2, 2], stride=[2, 2]
                )

        def pattern(x, kernel_size, stride):
            # default padding is [0, 0]
            return smith.ops.aten.max_pool2d_with_indices.default(
                x, kernel_size, stride, padding=[0, 0]
            )

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        self.assertEqual(len(matches), 1)

    def test_replaced_nodes(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.add(x, y)

        def pattern(x, y):
            return smith.add(x, y)

        def replacement(x, y):
            return smith.sub(smith.mul(x, y), y)

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, pattern, replacement
        )

        def check_replacement_nodes(self, traced, matches):
            replacement_nodes_in_graph = [
                node
                for node in traced.graph.nodes
                if node.target in {smith.sub, smith.mul}
            ]
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)

    def test_replace_pattern_with_callback(self) -> None:
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.add(x, y)

        def pattern(x, y):
            return smith.add(x, y)

        def replacement(x, y):
            return smith.sub(smith.mul(x, y), y)

        traced = symbolic_trace(M())
        # Return the same replacement graph for all matches, but have it be a unique
        # object each time.
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced,
            pattern,
            replacement_callback=lambda *args: symbolic_trace(replacement).graph,
        )

        def check_replacement_nodes(self, traced, matches):
            replacement_nodes_in_graph = [
                node
                for node in traced.graph.nodes
                if node.target in {smith.sub, smith.mul}
            ]
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)
