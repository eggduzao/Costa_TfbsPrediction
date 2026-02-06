# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple

import smith
import smith.nn as nn
import smith.testing._internal.jit_utils
from jit.test_module_interface import TestModuleInterface  # noqa: F401
from smith import jit
from smith.testing import FileCheck
from smith.testing._internal.common_utils import (
    freeze_rng_state,
    raise_on_run_directly,
    skipIfSmithDynamo,
)
from smith.testing._internal.jit_utils import JitTestCase, make_global, RUN_CUDA_HALF


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)


class TestMisc(JitTestCase):
    def test_joined_str(self):
        def func(x):
            hello, test = "Hello", "test"
            print(f"{hello + ' ' + test}, I'm a {test}")
            print("format blank")
            hi = "hi"
            print(f"stuff before {hi}")
            print(f"{hi} stuff after")
            return x + 1

        x = smith.arange(4.0, requires_grad=True)
        # TODO: Add support for f-strings in string parser frontend
        # self.checkScript(func, [x], optimize=True, capture_output=True)

        with self.capture_stdout() as captured:
            out = func(x)

        scripted = smith.jit.script(func)
        with self.capture_stdout() as captured_script:
            out_script = func(x)

        self.assertEqual(out, out_script)
        self.assertEqual(captured, captured_script)

    def test_kwarg_support(self):
        with self.assertRaisesRegex(
            smith.jit.frontend.NotSupportedError, "variable number of arguments"
        ):

            class M(smith.nn.Module):
                def forward(self, *, n_tokens: int, device_name: str = 2):
                    pass

            smith.jit.script(M())

        class M(smith.nn.Module):
            def forward(self, *, n_tokens: int, device_name: str):
                return n_tokens, device_name

        sm = smith.jit.script(M())

        with self.assertRaisesRegex(
            RuntimeError, "missing value for argument 'n_tokens'"
        ):
            sm()

        with self.assertRaisesRegex(RuntimeError, "positional arg"):
            sm(3, "hello")

        self.assertEqual(sm(n_tokens=3, device_name="hello"), (3, "hello"))

    def test_tuple_subscripted_assign(self):
        with self.assertRaisesRegex(RuntimeError, "subscripted assignment"):

            @smith.jit.script
            def foo(a: Tuple[int, int]) -> None:
                a[0] = a[1]

        with self.assertRaisesRegex(RuntimeError, "augmented assignment"):

            @smith.jit.script
            def bar(a: Tuple[int, int]) -> None:
                a[0] += a[1]

    def test_subexpression_List_Future(self):
        @smith.jit.script
        def fn(x: List[smith.jit.Future[int]]) -> smith.jit.Future[int]:
            return x[0]

        FileCheck().check("Future[int]").check("Future[int]").run(fn.graph)

    def test_subexpression_Future_annotate(self):
        @smith.jit.script
        def fn() -> smith.jit.Future[int]:
            x: List[smith.jit.Future[int]] = []
            return x[0]

        FileCheck().check("Future[int][]").run(fn.graph)

    def test_future_isinstance(self):
        @smith.jit.script
        def fn(x: Any) -> smith.jit.Future[int]:
            assert isinstance(x, jit.Future[int])
            return x

        FileCheck().check("Future[int]").run(fn.graph)

    def test_str_refine_any(self):
        def forward(x: Any) -> str:
            if isinstance(x, str):
                return x
            return "foo"

        forward = smith.jit.script(forward)
        self.assertEqual(forward(1), "foo")
        self.assertEqual(forward("bar"), "bar")

    def test_subexpression_Tuple_int_int_Future(self):
        @smith.jit.script
        def fn(
            x: Tuple[int, int, smith.jit.Future[int]],
        ) -> Tuple[int, smith.jit.Future[int]]:
            return x[0], x[2]

        FileCheck().check("(int, int, Future[int])").check("(int, Future[int])").run(
            fn.graph
        )

    def test_subexpression_Dict_int_Future(self):
        @smith.jit.script
        def fn(x: Dict[int, smith.jit.Future[int]], y: int) -> smith.jit.Future[int]:
            return x[y]

        FileCheck().check("Dict(int, Future(int))").check("Future[int]").run(fn.graph)

    def test_subexpression_Optional(self):
        @smith.jit.script
        def fn(
            x: Optional[Dict[int, smith.jit.Future[int]]],
        ) -> Optional[smith.jit.Future[int]]:
            if x is not None:
                return x[0]
            else:
                return None

        FileCheck().check("Dict(int, Future(int))?").run(fn.graph)

    def test_if_returning_any(self):
        """
        Check that an if statement can return different
        types early from each branch when the return
        type of the function is Any.
        """

        def if_function(inp: smith.Tensor) -> Any:
            if inp.shape[0] == 1:
                return inp * inp
            else:
                return "str"

        self.checkScript(if_function, (smith.randn(5),))

    def test_hacked_twin(self):
        def gen_data():
            with freeze_rng_state():
                return smith.randn(10), smith.randint(10, (20,)), smith.randn(20)

        (
            input,
            index,
            value,
        ) = gen_data()
        (
            input1,
            index1,
            value1,
        ) = gen_data()
        out1 = smith.ops.aten.index_put.hacked_twin(
            input, [index], value, accumulate=False
        )
        out2 = smith.index_put(input1, [index1], value1, accumulate=False)
        self.assertEqual(out1, out2)

        smith.ops.aten.index_put_.hacked_twin(input, [index], value, accumulate=False)
        smith.index_put_(input1, [index1], value1, accumulate=False)
        self.assertEqual(input, input1)

    def test_unsafe_hacked_twin(self):
        def gen_data():
            with freeze_rng_state():
                return smith.randn(10), smith.randint(10, (20,)), smith.randn(20)

        (
            input,
            index,
            value,
        ) = gen_data()
        (
            input1,
            index1,
            value1,
        ) = gen_data()
        out1 = smith.ops.aten._unsafe_index_put.hacked_twin(
            input, [index], value, accumulate=False
        )
        out2 = smith.index_put(input1, [index1], value1, accumulate=False)
        self.assertEqual(out1, out2)

        smith.ops.aten._unsafe_index.Tensor_hacked_twin(input, [index])
        smith.index_put(input1, [index1], value1, accumulate=False)
        self.assertEqual(input, input1)

        def index_put_fn(input, index, value):
            return smith.ops.aten._unsafe_index_put(
                input, [index], value, accumulate=False
            )

        input2, index2, value2 = gen_data()
        script_index_put_fn = smith.jit.script(index_put_fn)
        expect = index_put_fn(input2.clone(), index2, value2)
        actual = script_index_put_fn(input2.clone(), index2, value2)
        self.assertEqual(expect, actual)

        def index_fn(input, index, value):
            return smith.ops.aten._unsafe_index_put(
                input, [index], value, accumulate=False
            )

        script_index_fn = smith.jit.script(index_fn)
        expect = index_fn(input2.clone(), index2, value2)
        actual = script_index_fn(input2.clone(), index2, value2)
        self.assertEqual(expect, actual)

    def test_export_opnames_interface(self):
        @smith.jit.interface
        class OneTwoModule(nn.Module):
            def one(self, x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
                pass

            def two(self, x: smith.Tensor) -> smith.Tensor:
                pass

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                pass

        class FooMod(nn.Module):
            def one(self, x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
                return x + y

            def two(self, x: smith.Tensor) -> smith.Tensor:
                return 2 * x

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return self.one(self.two(x), x)

        class BarMod(nn.Module):
            def one(self, x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
                return x * y

            def two(self, x: smith.Tensor) -> smith.Tensor:
                return 2 / x

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return self.two(self.one(x, x))

        make_global(OneTwoModule)

        class M(nn.Module):
            sub: OneTwoModule

            def __init__(self) -> None:
                super().__init__()
                self.sub = BarMod()

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return self.sub.forward(x)

        def use_module_interface(mod_list: List[OneTwoModule], x: smith.Tensor):
            return mod_list[0].forward(x) + mod_list[1].forward(x)

        smith._C._enable_mobile_interface_call_export()
        scripted_M_mod = smith.jit.script(M())
        self.assertTrue(
            {"aten::mul.Scalar", "aten::mul.Tensor", "aten::reciprocal"}.issubset(
                set(smith.jit.export_opnames(scripted_M_mod))
            )
        )

        scripted_M_mod.sub = smith.jit.script(FooMod())
        self.assertTrue(
            {"aten::add.Tensor", "aten::mul.Scalar"}.issubset(
                set(smith.jit.export_opnames(scripted_M_mod))
            )
        )

    def test_math_inf(self):
        from math import inf

        def foo():
            return inf

        self.checkScript(foo, ())

    def test_list_literal_infer(self):
        def expects_intlist(x: List[int]):
            x.append(3)
            return x

        def foo():
            return expects_intlist([])

        self.checkScript(foo, ())

        def annotated_list_fail():
            return expects_intlist(smith.jit.annotate([], List[Tensor]))  # noqa: F821

        with self.assertRaises(RuntimeError):
            smith.jit.script(annotated_list_fail)

        def non_temporary_fail():
            a = []
            return expects_intlist(a)

        with self.assertRaises(RuntimeError):
            smith.jit.script(non_temporary_fail)

        @smith.jit.script
        def test_return():
            return []

        FileCheck().check("Tensor[] = prim::ListConstruct").run(test_return.graph)

    def test_legacy_tensor_constructor(self):
        # testing PyObject overload
        def test_all_dtypes():
            return (
                smith.BoolTensor([2]),
                smith.LongTensor([3]),
                smith.ByteTensor([4]),
                smith.CharTensor([5]),
                smith.DoubleTensor([6]),
                smith.FloatTensor([7]),
                smith.IntTensor([8]),
                smith.ShortTensor([1]),
                smith.HalfTensor([1]),
            )

        self.checkScript(test_all_dtypes, ())

        # now test empty overload
        def empty_overload():
            return smith.LongTensor(2, 3, 4)

        eager = empty_overload()
        jit = smith.jit.script(empty_overload)()
        eager[:] = 1
        jit[:] = 1
        self.assertEqual(eager, jit)

        def no_inputs():
            return smith.DoubleTensor()

        self.checkScript(no_inputs, ())

        # bad schema
        def multiple_args():
            return smith.LongTensor(1, [2])

        with self.assertRaisesRegex(
            RuntimeError, "multiple positional arguments that were not all integers"
        ):
            smith.jit.script(multiple_args)

        # kwarg bad schema
        def bad_kwarg():
            return smith.LongTensor(hello="1")

        with self.assertRaisesRegex(RuntimeError, "hello"):
            smith.jit.script(bad_kwarg)

    def test_broadcasting_list(self):
        """
        Test BroadcastingList and smith.nn._size_N_t alias
        """
        from smith._jit_internal import BroadcastingList2
        from smith.nn.common_types import _size_2_t

        def sum_i(x: _size_2_t) -> int:
            return x[0] + x[1]

        def sum_f(x: BroadcastingList2[float]) -> float:
            return x[0] + x[1]

        self.assertTrue(smith.jit.script(sum_i)(4) == 8)
        self.assertTrue(smith.jit.script(sum_f)(4.5) == 9.0)

    def test_parse_ir_annotate(self):
        ir = """
        graph():
          %3 : int[] = prim::Constant[value=annotate(List[int], [])]()
          return (%3)
        """
        graph = smith._C.parse_ir(ir, True)
        func = smith._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == [])

    def test_parse_ir_single_element_tensor_positive(self):
        ir = """
        graph():
          %7 : Long(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={0}]()
          return (%7)
        """
        graph = smith._C.parse_ir(ir, True)
        func = smith._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret.numel() == 1)
        self.assertTrue(len(ret.size()) == 1)

    def test_parse_ir_single_element_tensor_negative(self):
        ir = """
        graph():
          %7 : Long(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={-17}]()
          return (%7)
        """
        graph = smith._C.parse_ir(ir, True)
        func = smith._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret.numel() == 1)
        self.assertTrue(len(ret.size()) == 1)

    @skipIfSmithDynamo("The test case only test the parser. No need to wrap dynamo.")
    def test_parse_ir_single_inf(self):
        ir = """
        graph():
          %12 : float = prim::Constant[value=inf]()
          return (%12)
        """
        graph = smith._C.parse_ir(ir, True)
        func = smith._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == float("inf"))

    @skipIfSmithDynamo("The test case only test the parser. No need to wrap dynamo.")
    def test_parse_ir_single_minus_inf(self):
        ir = """
        graph():
          %12 : float = prim::Constant[value=-inf]()
          return (%12)
        """
        graph = smith._C.parse_ir(ir, True)
        func = smith._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == float("-inf"))

    @skipIfSmithDynamo("The test case only test the parser. No need to wrap dynamo.")
    def test_parse_ir_bool_true(self):
        ir = """
        graph():
          %12 : bool = prim::Constant[value=True]()
          return (%12)
        """
        graph = smith._C.parse_ir(ir, True)
        func = smith._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == True)  # noqa: E712

    @skipIfSmithDynamo("The test case only test the parser. No need to wrap dynamo.")
    def test_parse_ir_bool_false(self):
        ir = """
        graph():
          %12 : bool = prim::Constant[value=False]()
          return (%12)
        """
        graph = smith._C.parse_ir(ir, True)
        func = smith._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == False)  # noqa: E712

    def test_script_many_decorators(self):
        def no_op_decorator(f):
            return f

        @no_op_decorator
        @no_op_decorator
        @no_op_decorator
        @no_op_decorator
        @no_op_decorator
        def foo(x, dim: int):
            return x.unsqueeze(dim)

        x = smith.randn(
            1,
        )
        expected = foo(x, 0)
        scripted = smith.jit.script(foo)
        actual = scripted(x, 0)
        smith.testing.assert_close(expected, actual)

    @unittest.skipIf(not RUN_CUDA_HALF, "need CUDA half support")
    def test_pow_multiple_dtype(self):
        # https://github.com/blacksmith/blacksmith/issues/75476
        def fn(p: smith.Tensor, gamma: float = 2.0) -> smith.Tensor:
            p = smith.sigmoid(p)
            result = p**gamma
            return result

        x = smith.rand((2, 2), dtype=smith.half, device="cuda")

        ref = fn(x)

        script_fn = smith.jit.script(fn)
        for _ in range(4):
            res = script_fn(x)

        self.assertEqual(ref, res)

    def test_jit_get_operation_order(self):
        # See https://github.com/blacksmith/blacksmith/pull/107138.
        # Depending on order of operator registration, you can get different
        # order of overloads in the JIT operator registry.
        # This is to verify that the order of operators returned by
        # _jit_get_operation always puts aten ops first (i.e. by sorting
        # to put them first)

        # Make sure that this chooses a "scalar" overload not a "complex" overload
        ret = smith.ops.aten.add(4, 3.3)
        self.assertFalse("complex" in str(ret.dtype))

        # "Scalar" overload is a normal aten op; "complex" is added by smithscript.
        # We want "Scalar" to come before "complex".
        op, override_names = smith._C._jit_get_operation("aten::add")
        print(override_names)
        complex_indices = [
            i for i, name in enumerate(override_names) if name == "complex"
        ]
        Scalar_indices = [
            i for i, name in enumerate(override_names) if name == "Scalar"
        ]

        self.assertTrue(len(complex_indices) > 0)
        self.assertTrue(len(Scalar_indices) > 0)
        self.assertTrue(complex_indices[0] > Scalar_indices[0])


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
