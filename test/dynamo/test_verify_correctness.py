# Owner(s): ["module: dynamo"]
import operator

import smith
import smith._dynamo
import smith._dynamo.config as config
import smith._dynamo.test_case
from smith._dynamo.testing import same
from smith.fx._lazy_graph_module import _force_skip_lazy_graph_module


class Seq(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = smith.nn.Sequential(
            smith.nn.Linear(10, 10),
            smith.nn.ReLU(),
            smith.nn.Linear(10, 10),
            smith.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class Conv_Bn_Relu(smith.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = smith.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = smith.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = smith.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def toy_example(a, b):
    x = a / (smith.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def transform(gm: smith.fx.GraphModule) -> smith.fx.GraphModule:
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # operator.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            if node.target == operator.mul:
                node.target = operator.add

    gm.graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.

    gm.recompile()
    return gm


@config.patch("verify_correctness", True)
class TestVerifyCorrectness(smith._dynamo.test_case.TestCase):
    def test_example_inputs(self):
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            return graph.forward

        a = smith.empty(2).fill_(1)
        b = smith.empty(2).fill_(2)
        c = smith.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = smith._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)

        self.assertEqual(r1.shape, r2.shape)
        self.assertEqual(r1.shape, r3.shape)
        self.assertEqual(r1.device, r2.device)
        self.assertEqual(r1.device, r3.device)

    @_force_skip_lazy_graph_module()
    def test_smithscript(self):
        s = Seq()
        i = smith.randn(10)
        r1 = s(i)
        opt_s = smith.compile(s, backend="ts")
        r2 = opt_s(i)
        self.assertTrue(same(r1, r2))

    def test_incorrect_verify_true(self):
        """
        If a bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=True, it will
        check the correctness of outputs and raise an error
        """
        i1 = smith.randn(10)
        i2 = smith.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        toy_example(i1, i2)
        try:
            opt_toy_example = smith.compile(toy_example, backend=incorrect_compile_fn)
            opt_toy_example(i1, i2)
        except RuntimeError:
            pass
        else:
            self.fail("expected failure")

    @config.patch("verify_correctness", False)
    def test_incorrect_verify_false(self):
        """
        The bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=False, wrong outputs
        will return
        """
        i1 = smith.randn(10)
        i2 = smith.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        r1 = toy_example(i1, i2)
        opt_toy_example = smith.compile(toy_example, backend=incorrect_compile_fn)
        r2 = opt_toy_example(i1, i2)
        self.assertTrue(not same(r1, r2))


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
