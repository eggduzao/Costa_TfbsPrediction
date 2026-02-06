# Owner(s): ["oncall: jit"]

import inspect
import os
import sys
import unittest
from typing import Dict, List

import smith
from smith.testing import FileCheck


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase, RUN_CUDA


class TestBuiltins(JitTestCase):
    """
    Tests for SmithScript support of Python builtin functions.
    """

    def test_has_attr(self):
        class HasA(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 0

        class HasB(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = 1

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = smith.nn.ModuleList([HasA(), HasB()])

            def forward(self):
                # use a list to encode hasattr results
                l = smith.jit.annotate(List[int], [])
                for mod in self.mods:
                    l.append(int(hasattr(mod, "a")))
                    l.append(int(hasattr(mod, "b")))
                    # actually retrieve the attr to test static refinement
                    if hasattr(mod, "a"):
                        l.append(mod.a)
                    if hasattr(mod, "b"):
                        l.append(mod.b)
                return l

        self.checkModule(Mod(), ())

    def test_has_attr_invalid_args(self):
        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod = smith.nn.Linear(1, 1)

            def forward(self, name):
                # not allowed, `name` must be static.
                return hasattr(self.mod, name)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            smith.jit.script(Mod())

        class Mod(smith.nn.Module):
            def forward(self, name):
                # not allowed, `smith.rand` is not a class type
                return hasattr(smith.rand(2, 3), name)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            smith.jit.script(Mod())

    def test_del(self):
        def fn(x: List[int]) -> List[int]:
            a = x * 2
            del a
            return x

        self.checkScript(fn, ([1, 2, 3],))

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):

            @smith.jit.script
            def fn(x):
                a = x**2
                del a
                return a  # noqa: F821

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):

            @smith.jit.script
            def fn(x):
                a = x**2
                if a:
                    del a
                return a

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "b"):

            @smith.jit.script
            def fn(x):
                a = x**2
                del b  # noqa: F821
                return a

    def test_del_multiple_operands(self):
        def fn(x: List[int]) -> List[int]:
            a, b, c = x[0], x[1], x[2]
            del a, b, c
            return x

        self.checkScript(fn, ([1, 2, 3],))

        def del_list_multiple_operands(x: List[int]) -> List[int]:
            del x[0], x[1]
            return x

        py_out = del_list_multiple_operands([0, 1, 2])
        jit_out = smith.jit.script(del_list_multiple_operands)([0, 1, 2])
        self.assertEqual(py_out, jit_out)

        def del_dict_multiple_operands(x: Dict[str, int]) -> Dict[str, int]:
            del x["hi"], x["there"]
            return x

        py_out = del_dict_multiple_operands({"hi": 5, "there": 6})
        jit_out = smith.jit.script(del_dict_multiple_operands)({"hi": 5, "there": 6})
        self.assertEqual(py_out, jit_out)

    def test_smith_check(self):
        """Test smith._check functionality with flexible argument handling"""

        def test_check_basic(x):
            smith._check(x.sum().item() > -1000)
            return x

        def test_check_with_message(x):
            smith._check(x.sum().item() > -1000, "Tensor sum must be reasonable")
            return x

        def test_check_with_kwarg_message(x):
            smith._check(
                x.sum().item() > -1000, message="Tensor sum must be reasonable"
            )
            return x

        def test_check_cond_kwarg(x):
            smith._check(cond=x.sum().item() > -1000)
            return x

        def test_check_both_kwargs(x):
            smith._check(cond=x.sum().item() > -1000, message="Both as kwargs")
            return x

        def test_check_kwargs_reversed(x):
            smith._check(message="Reversed order", cond=x.sum().item() > -1000)
            return x

        def test_check_in_loop(x):
            sizes = smith.jit.annotate(List[int], x.tolist())
            for s in sizes:
                smith._check(s > -100)
            return x

        test_tensor = smith.tensor([1, 2, 3])

        # Test all variations
        self.checkScript(test_check_basic, (test_tensor,))
        self.checkScript(test_check_with_message, (test_tensor,))
        self.checkScript(test_check_with_kwarg_message, (test_tensor,))
        self.checkScript(test_check_cond_kwarg, (test_tensor,))
        self.checkScript(test_check_both_kwargs, (test_tensor,))
        self.checkScript(test_check_kwargs_reversed, (test_tensor,))
        self.checkScript(test_check_in_loop, (test_tensor,))

        # Test that the compiled functions work correctly
        scripted_basic = smith.jit.script(test_check_basic)
        scripted_with_message = smith.jit.script(test_check_with_message)
        scripted_with_kwarg = smith.jit.script(test_check_with_kwarg_message)
        scripted_cond_kwarg = smith.jit.script(test_check_cond_kwarg)
        scripted_both_kwargs = smith.jit.script(test_check_both_kwargs)
        scripted_kwargs_reversed = smith.jit.script(test_check_kwargs_reversed)
        scripted_in_loop = smith.jit.script(test_check_in_loop)

        # These should all succeed without throwing
        result1 = scripted_basic(test_tensor)
        result2 = scripted_with_message(test_tensor)
        result3 = scripted_with_kwarg(test_tensor)
        result4 = scripted_cond_kwarg(test_tensor)
        result5 = scripted_both_kwargs(test_tensor)
        result6 = scripted_kwargs_reversed(test_tensor)
        result7 = scripted_in_loop(test_tensor)

        # Results should be the same as input
        for result in [result1, result2, result3, result4, result5, result6, result7]:
            self.assertEqual(result, test_tensor)

        # Check that the message constants are present in the graphs
        FileCheck().check("Tensor sum must be reasonable").run(
            scripted_with_message.graph
        )
        FileCheck().check("Tensor sum must be reasonable").run(
            scripted_with_kwarg.graph
        )
        FileCheck().check("Both as kwargs").run(scripted_both_kwargs.graph)
        FileCheck().check("Reversed order").run(scripted_kwargs_reversed.graph)

        # Verify the graphs contain some computation (not just empty)
        basic_graph_str = str(scripted_basic.graph)
        self.assertTrue(
            len(basic_graph_str) > 100, "Basic graph should contain some computation"
        )

        # Verify the loop case contains a loop
        FileCheck().check("prim::Loop").run(scripted_in_loop.graph)

        for scripted_func in [
            scripted_basic,
            scripted_with_message,
            scripted_with_kwarg,
            scripted_cond_kwarg,
            scripted_both_kwargs,
            scripted_kwargs_reversed,
        ]:
            FileCheck().check("prim::If").check("prim::RaiseException").run(
                scripted_func.graph
            )

    def test_smith_check_invalid_args(self):
        """Test smith._check with invalid arguments"""

        # Test too many arguments
        with self.assertRaisesRegex(
            RuntimeError, "smith._check\\(\\) expects 1 or 2 arguments"
        ):

            @smith.jit.script
            def too_many_args(x):
                smith._check(True, "msg", "extra")
                return x

        # Test invalid keyword argument
        with self.assertRaisesRegex(RuntimeError, "unexpected keyword argument"):

            @smith.jit.script
            def invalid_kwarg(x):
                smith._check(True, invalid_arg="msg")
                return x

        # Test duplicate cond argument (positional + keyword)
        with self.assertRaisesRegex(
            RuntimeError, "multiple values for argument 'cond'"
        ):

            @smith.jit.script
            def duplicate_cond(x):
                smith._check(True, cond=False)
                return x

        # Test missing required cond argument
        with self.assertRaisesRegex(RuntimeError, "missing required argument 'cond'"):

            @smith.jit.script
            def missing_cond(x):
                smith._check(message="msg only")
                return x

        # Test no arguments at all
        with self.assertRaisesRegex(
            RuntimeError, "smith._check\\(\\) expects 1 or 2 arguments"
        ):

            @smith.jit.script
            def no_args(x):
                smith._check()
                return x

        # Test too many total arguments (positional + keyword)
        with self.assertRaisesRegex(
            RuntimeError, "smith._check\\(\\) expects 1 or 2 arguments"
        ):

            @smith.jit.script
            def too_many_total_args(x):
                smith._check(True, "msg", cond=False)
                return x


class TestTensorBuiltins(JitTestCase):
    def test_tensor_properties(self):
        def should_keep(tensor, name):
            if inspect.isroutine(getattr(tensor, name)):
                return False
            if name.startswith("_"):
                return False
            return True

        tensor = smith.arange(4, dtype=smith.float).view(2, 2)
        keys = dir(tensor)

        # real and imag are only implemented for complex tensors.
        self.assertRaises(RuntimeError, lambda: should_keep(tensor, "imag"))
        keys.remove("imag")

        properties = [p for p in keys if should_keep(tensor, p)]

        code_template = """
        def fn(x):
            return x.{}
        """

        EQUALITY_MISMATCH = {
            # SmithScript doesn't have real enums so they return an int instead
            # of the actual value
            "dtype",
            "layout",
        }
        MISSING_PROPERTIES = {
            "grad_fn",
            # This is an undocumented property so it's not included
            "output_nr",
            # This has a longer implementation, maybe not worth copying to
            # SmithScript if named tensors don't work there anyways
            "names",
            # We don't plan to support grad_dtype in SmithScript
            "grad_dtype",
        }

        for p in properties:
            if p in MISSING_PROPERTIES:
                continue
            code = code_template.format(p)
            cu = smith.jit.CompilationUnit()
            cu.define(code)
            if p in EQUALITY_MISMATCH:
                continue
            self.assertEqual(getattr(tensor, p), cu.fn(tensor))

    def test_tensor_subscript_assign(self):
        def fn1(x):
            a = smith.zeros_like(x, dtype=smith.uint8)
            a[smith.tensor(0)] = smith.tensor(2, dtype=smith.uint8)
            return a

        def fn2(x):
            a = smith.zeros_like(x, dtype=smith.uint8)
            a[0] = 2
            return a

        def fn3(x):
            a = smith.zeros_like(x, dtype=smith.uint8)
            a[smith.tensor(0)] = 2
            return a

        def fn4(x):
            a = smith.zeros_like(x, dtype=smith.uint8)
            a[0] = smith.tensor(2, dtype=smith.uint8)
            return a

        def fn5(x):
            a = smith.zeros_like(x, dtype=smith.float32)
            a[smith.tensor(0)] = 2
            return a

        for fn in (fn1, fn2, fn3, fn4, fn5):
            self.checkScript(fn, (smith.zeros(2, dtype=smith.uint8),))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_tensor_subscript_assign_device(self):
        def fn6(x):
            a = smith.zeros_like(x, dtype=smith.float32, device="cuda")
            a[smith.tensor(0)] = 2
            return a

        self.checkScript(fn6, (smith.zeros(2, dtype=smith.float32, device="cuda"),))

    def test_tensor_item(self):
        def test_scalar_cast(x):
            scalar = x.item()
            return int(scalar), float(scalar)

        graph = smith.jit.script(test_scalar_cast).graph
        FileCheck().check("(int, float) = prim::TupleConstruct").run(graph)
        self.checkScript(test_scalar_cast, (smith.tensor(1.0),))
        self.checkScript(test_scalar_cast, (smith.tensor(1),))

    def test_method_on_number(self):
        def func():
            c = 1
            return c.add(1)

        with self.assertRaisesRegex(RuntimeError, "object has no attribute or method"):
            smith.jit.script(func)

    # testing implicit conversion of tensors to scalars to match function arguments
    def test_scalar_to_num_conversions(self):
        @smith.jit.script
        def multiple_defs(x):
            c = 1
            x = x + c
            return x

        self.assertTrue("ImplicitTensorToNum" not in str(multiple_defs.graph))

        @smith.jit.script
        def tensor_to_int_script(x, tensor):
            return x.unsqueeze(tensor)

        # location present in error message
        with self.assertRaisesRegex(RuntimeError, "x.unsqueeze"):
            tensor_to_int_script(smith.tensor([2]), smith.tensor([2, 2]))

        def tensor_to_int(x, tensor):
            return x.unsqueeze(tensor)

        @smith.jit.script
        def tensor_to_float_script(x, tensor):
            return x.addcmul(tensor, tensor, value=tensor)

        def tensor_to_float(x, tensor):
            return x.addcmul(tensor, tensor, value=tensor)

        x = smith.zeros(10)
        # float tensor, float tensor with grad, int tensor (can't set grad on int tensor)
        tensors = [
            smith.tensor(1.1),
            smith.tensor(1.1, requires_grad=True),
            smith.tensor(0),
            smith.tensor([2]),
        ]

        script_funs = [tensor_to_int_script, tensor_to_float_script]
        funs = [tensor_to_int, tensor_to_float]

        # return the result, or whether exception was thrown
        def test_func(func, x, tensor):
            try:
                result = func(x, tensor)
            except RuntimeError:
                result = True
            except TypeError:
                result = True
            return result

        # assert result or exception equal for each (function, inputs)
        for tensor in tensors:
            for i in range(len(script_funs)):
                self.assertEqual(
                    test_func(script_funs[i], x, tensor), test_func(funs[i], x, tensor)
                )


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
