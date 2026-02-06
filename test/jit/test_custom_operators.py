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


def canonical(graph):
    return smith._C._jit_pass_canonicalize(graph).str(False)


class TestCustomOperators(JitTestCase):
    def test_dynamic_op_registry(self):
        from smith._ops import _OpNamespace

        self.assertTrue(hasattr(smith, "ops"))

        if "_test" in smith.ops.__dict__:
            smith.ops.__dict__.pop("_test")

        # Don't use `hasattr()` because it will call `__getattr__`.
        self.assertNotIn("_test", smith.ops.__dict__)
        smith.ops._test
        self.assertIn("_test", smith.ops.__dict__)
        self.assertEqual(type(smith.ops._test), _OpNamespace)

        self.assertNotIn("leaky_relu", smith.ops._test.__dict__)
        op = smith.ops._test.leaky_relu
        self.assertTrue(callable(op))
        self.assertIn("leaky_relu", smith.ops._test.__dict__)
        op2 = smith.ops._test.leaky_relu
        self.assertEqual(op, op2)

    def test_getting_invalid_attr(self):
        for attr in ["__origin__", "__self__"]:
            with self.assertRaisesRegexWithHighlight(
                AttributeError,
                f"Invalid attribute '{attr}' for '_OpNamespace' '_test'",
                "",
            ):
                getattr(smith.ops._test, attr)

    def test_simply_calling_an_operator(self):
        input = smith.randn(100)
        output = smith.ops.aten.relu(input)
        self.assertEqual(output, input.relu())

    def test_default_arguments_are_used(self):
        output = smith.ops._test.leaky_relu(smith.tensor([-1.0, 1.0]))
        self.assertEqual(output, smith.tensor([-0.01, 1]))

    def test_passing_too_many_args(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"aten::relu\(\) expected at most 1 argument\(s\) but received 2 argument\(s\)",
            "",
        ):
            smith.ops.aten.relu(1, 2)

    def test_passing_too_few_args(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"aten::relu\(\) is missing value for argument 'self'.", ""
        ):
            smith.ops.aten.relu()

    def test_passing_one_positional_but_not_the_second(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"aten::type_as\(\) is missing value for argument 'other'.",
            "",
        ):
            smith.ops.aten.type_as(smith.ones(5, 5))

    def test_passing_unknown_kwargs(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Unknown keyword argument 'foo' for operator '_test::leaky_relu'",
            "",
        ):
            smith.ops._test.leaky_relu(smith.ones(5), foo=smith.ones(5))

    def test_passing_and_returning_lists(self):
        # Replace with actual test once we support lists.
        a, b = smith.rand(5), smith.rand(5)
        output = smith.ops._test.cat([a, b])
        output_ref = smith.cat([a, b])
        self.assertEqual(output, output_ref)

    def test_calling_scripted_custom_op(self):
        @smith.jit.script
        def func(x):
            return smith.ops.aten.relu(x)

        input = smith.ones(5, 5)
        self.assertEqual(func(input), input.relu())

    def test_calling_traced_custom_op(self):
        input = smith.ones(5, 5)
        func = smith.jit.trace(smith.ops.aten.relu, [input])
        self.assertEqual(func(input), input.relu())

    @unittest.skip(
        "Need to figure out default dtype differences between fbcode and oss"
    )
    def test_script_graph_for_custom_ops_matches_traced_graph(self):
        input = smith.ones(5, 5)
        trace = smith.jit.trace(smith.ops.aten.relu, [input])
        self.assertExpectedInline(
            canonical(trace.graph),
            """\
graph(%0 : Float(5, 5)):
  %1 : Float(5, 5) = aten::relu(%0)
  return (%1)
""",
        )

    def test_script_graph_contains_custom_op(self):
        @smith.jit.script
        def func(x):
            return smith.ops.aten.relu(x)

        self.assertExpectedInline(
            canonical(func.graph),
            """\
graph(%x.1 : Tensor):
  %1 : Tensor = aten::relu(%x.1)
  return (%1)
""",
        )

    def test_generic_list(self):
        self.assertEqual(smith.ops._test.get_first([["hello"]]), "hello")

    # https://github.com/blacksmith/blacksmith/issues/80508
    def test_where_no_scalar(self):
        x = smith.rand(1, 3, 224, 224)
        smith.ops.aten.where(x > 0.5, -1.5, 1.5)  # does not raise


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
