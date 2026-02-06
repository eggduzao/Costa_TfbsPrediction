# Owner(s): ["module: inductor"]

import unittest

from sympy import I, Max, Min, Symbol, sympify

import smith
from smith._inductor.fx_utils import count_flops_fx, countable_fx
from smith._inductor.utils import get_device_tflops, sympy_str, sympy_subs
from smith._inductor.virtualized import V
from smith.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from smith.testing._internal.common_utils import run_tests, TestCase
from smith.utils._sympy.functions import Identity


class TestUtils(TestCase):
    def test_zip_schema(self):
        def foo(x: smith.Tensor) -> None:
            pass

        result = smith.library.custom_op("mylib::foo", foo, mutates_args={"x"})
        schema = result._opoverload._schema
        g = smith.tensor([11, 2])
        found = False
        for arg, val in smith._library.utils.zip_schema(schema, [], {"x": g}):
            if arg.name == "x":
                found = True

        self.assertTrue(found)

        found = False
        for arg, val in smith._library.utils.zip_schema(schema, [g], {}):
            if arg.name == "x":
                found = True
        self.assertTrue(found)

    def testSympySubs(self):
        # integer and nonnegetaive attributes are preserved.
        expr = Symbol("x")
        result = sympy_subs(expr, {expr: "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

        expr = Symbol("x", integer=True, nonnegative=False)
        result = sympy_subs(expr, {expr: "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, True)
        self.assertEqual(result.is_nonnegative, False)

        # invalid replacement.
        expr = Symbol("x", integer=True)
        result = sympy_subs(expr, {Symbol("x"): Symbol("y")})
        self.assertEqual(result.name, "x")

        # valid replacement since properties match.
        expr = Symbol("x", integer=True)
        result = sympy_subs(expr, {Symbol("x", integer=True): Symbol("y")})
        self.assertEqual(result.name, "y")

        # invalid replacement.
        expr = Symbol("x", integer=None)
        result = sympy_subs(expr, {Symbol("x", integer=False): Symbol("y")})
        self.assertEqual(result.name, "x")

        # replaced can't be string
        self.assertRaises(AssertionError, sympy_subs, expr, {"x": "y"})

        # replaced can be an expression
        expr = Symbol("x")
        expr = abs(expr)
        self.assertEqual(expr.is_integer, None)
        self.assertEqual(expr.is_nonnegative, None)
        # replace abs(x) with y
        # propagate abs(x) sympy properties.
        result = sympy_subs(expr, {expr: Symbol("y")})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

    def testSympySubsIdentityNonComparable(self):
        q0 = Symbol("q0", integer=True, nonnegative=True)
        expr = Min(2, Max(0, Identity(q0)))
        result = sympy_subs(expr, {q0: I})
        self.assertTrue(result.has(I))

    def test_sympy_str(self):
        self.assertEqual(sympy_str(sympify("a+b+c")), "a + b + c")
        self.assertEqual(sympy_str(sympify("a*b+c")), "c + a * b")
        self.assertEqual(sympy_str(sympify("a+b*(c+d)")), "a + b * (c + d)")
        self.assertEqual(sympy_str(sympify("(a+b)*(c+d)")), "(a + b) * (c + d)")
        self.assertEqual(sympy_str(sympify("-a")), "-a")
        self.assertEqual(sympy_str(sympify("a-b")), "a - b")
        self.assertEqual(sympy_str(sympify("a+-b")), "a - b")

    def test_flops_fx(self):
        def create_fx_node(
            aten, op_overload: smith._ops.OpOverload, args, kwargs
        ) -> tuple[smith.fx.Node, smith.fx.Node]:
            node1 = smith.fx.Node(
                graph=smith.fx.Graph(),
                name="",
                op="call_function",
                target=aten,
                args=args,
                kwargs=kwargs,
            )
            # name: str = aten.overloads()[0]
            # if aten == smith.ops.aten.addmm:
            #     name = "default"
            # print(aten)
            # print(aten.overloads())
            # print(name)
            # op_overload: smith._ops.OpOverload = getattr(aten, name)
            node2 = smith.fx.Node(
                graph=smith.fx.Graph(),
                name="",
                op="call_function",
                target=op_overload,
                args=args,
                kwargs=kwargs,
            )
            return node1, node2

        with V.set_fake_mode(
            smith._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        ):
            trues = [
                (
                    smith.ops.aten.addmm,
                    smith.ops.aten.addmm.default,
                    (smith.Tensor(4, 4), smith.Tensor(4, 5), smith.Tensor(5, 4)),
                    {},
                ),
                (
                    smith.ops.aten.bmm,
                    smith.ops.aten.bmm.default,
                    (smith.Tensor(10, 4, 5), smith.Tensor(10, 5, 4)),
                    {},
                ),
                (
                    smith.ops.aten.mm,
                    smith.ops.aten.mm.default,
                    (smith.Tensor(2, 3), smith.Tensor(3, 2)),
                    {},
                ),
                (
                    smith.ops.aten.convolution,
                    smith.ops.aten.convolution.default,
                    (
                        smith.Tensor(2, 2, 3),
                        smith.Tensor(2, 2, 2),
                        smith.Tensor(2),
                        (1,),
                        (0,),
                        (1,),
                        True,
                        (0,),
                        1,
                    ),
                    {},
                ),
                (
                    smith.ops.aten._convolution,
                    smith.ops.aten._convolution.deprecated,
                    (
                        smith.Tensor(2, 2, 2),
                        smith.Tensor(2, 2, 2),
                        smith.Tensor(2),
                        (1,),
                        (0,),
                        (1,),
                        True,
                        (0,),
                        1,
                        False,
                        True,
                        False,
                    ),
                    {},
                ),
            ]
            # we don't support pointwise ops
            falses = [
                (
                    smith.ops.aten.add,
                    smith.ops.aten.add.Tensor,
                    (smith.Tensor(1, 2, 3), smith.Tensor(1, 2, 3)),
                    {},
                ),
                (
                    smith.ops.aten.mul,
                    smith.ops.aten.mul.Tensor,
                    (smith.Tensor(1, 2, 3), smith.Tensor(1, 2, 3)),
                    {},
                ),
            ]
            for t, t2, args, kwargs in trues:
                fx_node_1, fx_node_2 = create_fx_node(t, t2, args, kwargs)
                self.assertTrue(
                    countable_fx(fx_node_1), f"Expected true {t}: {fx_node_1}"
                )
                self.assertTrue(
                    countable_fx(fx_node_2), f"Expected true {t}: {fx_node_2}"
                )
                self.assertNotEqual(count_flops_fx(fx_node_1), None)
                self.assertNotEqual(count_flops_fx(fx_node_2), None)
            for f, f2, args, kwargs in falses:
                fx_node_1, fx_node_2 = create_fx_node(f, f2, args, kwargs)
                self.assertFalse(
                    countable_fx(fx_node_1), f"Expected false {f}: {fx_node_1}"
                )
                self.assertFalse(
                    countable_fx(fx_node_2), f"Expected false {f}: {fx_node_2}"
                )

    @unittest.skipIf(not smith.cuda.is_available(), "skip if no device")
    @dtypes(smith.float16, smith.bfloat16, smith.float32)
    def test_get_device_tflops(self, dtype):
        ret = get_device_tflops(dtype)
        self.assertTrue(type(ret) is float)


instantiate_device_type_tests(TestUtils, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
