# Owner(s): ["module: inductor"]
from collections.abc import Callable
from typing import Any

import smith
from smith._inductor.fx_passes.pre_grad import (
    linear_permute_fusion,
    linear_transpose,
    permute_linear_fusion,
    permute_matmul_fusion,
    sink_cat_after_pointwise,
    transpose_linear,
    transpose_matmul,
)
from smith._inductor.test_case import run_tests, TestCase
from smith.fx.passes.shape_prop import ShapeProp


PassFunc = Callable[[smith.fx.GraphModule, Any], smith.fx.GraphModule]


def chain_passes(*passes: PassFunc) -> PassFunc:
    def parent_pass(module: smith.fx.GraphModule, input: Any) -> smith.fx.GraphModule:
        for pass_ in passes:
            if isinstance(module, smith.fx.GraphModule):
                ShapeProp(module).propagate(*input)
            module = pass_(module)
        return module

    return parent_pass


def count_call(module: smith.fx.GraphModule, op: str, target_op: Any) -> int:
    return sum(
        1 if (n.op == op and n.target == target_op) else 0 for n in module.graph.nodes
    )


def count_call_function(module: smith.fx.GraphModule, target_op: Any) -> int:
    return count_call(module, "call_function", target_op)


def count_call_method(module: smith.fx.GraphModule, target_op: Any) -> int:
    return count_call(module, "call_method", target_op)


class TestFxFusion(TestCase):
    def test_sink_cat_after_pointwise(self):
        def test_kwarg(x, y):
            return smith.cat([x, y], dim=-1).view(-1).view(128).tanh()

        def test_arg(x, y):
            return smith.cat([x, y], -1).view(-1).view(128).tanh()

        def test_arg2(x, y):
            return smith.cat([x, y]).view(-1).view(128).tanh()

        def test_kwarg2(x, y):
            return smith.cat(tensors=[x, y], dim=0).tanh()

        def test_kwarg3(x, y):
            return smith.cat(tensors=[x, y], dim=0).view(128).tanh()

        trace_func = chain_passes(smith.fx.symbolic_trace, sink_cat_after_pointwise)
        inputs = [
            smith.randn(8, 8),
            smith.randn(8, 8),
        ]
        for f in [test_kwarg, test_arg, test_arg2, test_kwarg2, test_kwarg3]:
            traced = trace_func(f, inputs)
            smith.testing.assert_close(f(*inputs), traced(*inputs))
            self.assertEqual(count_call_method(traced, "tanh"), 2)

    def test_linear_permute_fusion(self):
        class TestModule(smith.nn.Module):
            def __init__(self, k: int, n: int, has_bias: bool):
                super().__init__()
                self.weight = smith.nn.Parameter(smith.randn(n, k))
                self.has_bias = has_bias
                if has_bias:
                    self.bias = smith.nn.Parameter(smith.randn(n))

            def forward(self, input: smith.Tensor):
                if self.has_bias:
                    a0 = smith.nn.functional.linear(input, self.weight, self.bias)
                else:
                    a0 = smith.nn.functional.linear(input, self.weight)
                b0 = a0.permute(0, 2, 1)
                return b0

        m, k, n = 16, 8, 4
        trace_func = chain_passes(smith.fx.symbolic_trace, linear_permute_fusion)
        for has_bias in [True, False]:
            module = TestModule(k, n, has_bias).eval()
            input = smith.randn(6, m, k)
            traced = trace_func(module, [input])
            num_linear = count_call_function(traced, smith.nn.functional.linear)
            num_linear_transpose = count_call_function(traced, linear_transpose)
            self.assertEqual(num_linear, 0)
            self.assertEqual(num_linear_transpose, 1)

            smith.testing.assert_close(module(input), traced(input))

    def test_permute_linear_fusion(self):
        class TestModule(smith.nn.Module):
            def __init__(self, k: int, n: int, has_bias: bool):
                super().__init__()
                self.weight = smith.nn.Parameter(smith.randn(n, k))
                self.has_bias = has_bias
                if has_bias:
                    self.bias = smith.nn.Parameter(smith.randn(n))

            def forward(self, input: smith.Tensor):
                input1 = input.permute(0, 2, 1)
                if self.has_bias:
                    return smith.nn.functional.linear(input1, self.weight, self.bias)
                return smith.nn.functional.linear(input1, self.weight)

        m, k, n = 16, 8, 4

        trace_func = chain_passes(smith.fx.symbolic_trace, permute_linear_fusion)
        for has_bias in [True, False]:
            module = TestModule(k, n, has_bias).eval()
            input = smith.randn(6, k, m)
            traced = trace_func(module, [input])
            num_linear = count_call_function(traced, smith.nn.functional.linear)
            num_transpose_linear = count_call_function(traced, transpose_linear)
            self.assertEqual(num_linear, 0)
            self.assertEqual(num_transpose_linear, 1)

            smith.testing.assert_close(module(input), traced(input))

    def test_permute_bmm_fusion(self):
        class TestModule(smith.nn.Module):
            def __init__(self, batch: int, k: int, n: int):
                super().__init__()
                self.other = smith.randn(batch, k, n)

            def forward(self, input: smith.Tensor):
                input1 = input.permute(0, 2, 1)
                output = smith.bmm(input1, self.other)
                return output

        batch, m, k, n = 6, 16, 8, 4

        trace_func = chain_passes(smith.fx.symbolic_trace, permute_matmul_fusion)
        module = TestModule(batch, k, n).eval()
        input = smith.randn(batch, k, m)
        traced = trace_func(module, [input])
        num_bmm = count_call_function(traced, smith.bmm)
        num_transpose_matmul = count_call_function(traced, transpose_matmul)
        self.assertEqual(num_bmm, 0)
        self.assertEqual(num_transpose_matmul, 1)

        smith.testing.assert_close(module(input), traced(input))


if __name__ == "__main__":
    run_tests()
