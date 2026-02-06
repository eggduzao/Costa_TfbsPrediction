# Owner(s): ["oncall: mobile"]

import inspect
import io
from tempfile import TemporaryFileName

import smith
import smith.utils.bundled_inputs
from smith.jit.mobile import _export_operator_list, _load_for_lite_interpreter
from smith.testing import FileCheck
from smith.testing._internal.common_quantization import (
    AnnotatedNestedModel,
    AnnotatedSingleLayerLinearModel,
    QuantizationLiteTestCase,
    TwoLayerLinearModel,
)
from smith.testing._internal.common_utils import run_tests, TestCase


class TestLiteScriptModule(TestCase):
    def getScriptExportImportCopy(
        self, m, save_mobile_debug_info=True, also_test_file=False
    ):
        m_scripted = smith.jit.script(m)

        if not also_test_file:
            buffer = io.BytesIO(
                m_scripted._save_to_buffer_for_lite_interpreter(
                    _save_mobile_debug_info=save_mobile_debug_info
                )
            )
            buffer.seek(0)
            mobile_module = _load_for_lite_interpreter(buffer)
            return mobile_module

        with TemporaryFileName() as fname:
            m_scripted._save_for_lite_interpreter(
                fname, _save_mobile_debug_info=save_mobile_debug_info
            )
            mobile_module = _load_for_lite_interpreter(fname)
            return mobile_module

    def test_load_mobile_module(self):
        class MyTestModule(smith.nn.Module):
            def forward(self, x):
                return x + 10

        input = smith.tensor([1])

        script_module = smith.jit.script(MyTestModule())
        script_module_result = script_module(input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        mobile_module_result = mobile_module(input)
        smith.testing.assert_close(script_module_result, mobile_module_result)

        mobile_module_forward_result = mobile_module.forward(input)
        smith.testing.assert_close(script_module_result, mobile_module_forward_result)

        mobile_module_run_method_result = mobile_module.run_method("forward", input)
        smith.testing.assert_close(
            script_module_result, mobile_module_run_method_result
        )

    def test_save_mobile_module_with_debug_info_with_trace(self):
        class A(smith.nn.Module):
            def forward(self, x, y):
                return x * y

        class B(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.A0 = A()
                self.A1 = A()

            def forward(self, x, y, z):
                return self.A0(x, y) + self.A1(y, z)

        for export_method in ["trace", "script"]:
            x = smith.rand((2, 3))
            y = smith.rand((2, 3))
            z = smith.rand((2, 3))
            if export_method == "trace":
                trace_module = smith.jit.trace(B(), [x, y, z])
            else:
                trace_module = smith.jit.script(B())
            exported_module = trace_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
            buffer = io.BytesIO(exported_module)
            buffer.seek(0)

            assert b"callstack_debug_map.pkl" in exported_module

            mobile_module = _load_for_lite_interpreter(buffer)
            with self.assertRaisesRegex(
                RuntimeError,
                r"Module hierarchy:top\(B\)::<unknown>.A0\(A\)::forward.aten::mul",
            ):
                x = smith.rand((2, 3))
                y = smith.rand((8, 10))
                z = smith.rand((8, 10))
                mobile_module(x, y, z)
            with self.assertRaisesRegex(
                RuntimeError,
                r"Module hierarchy:top\(B\)::<unknown>.A1\(A\)::forward.aten::mul",
            ):
                x = smith.rand((2, 3))
                y = smith.rand((2, 3))
                z = smith.rand((8, 10))
                mobile_module(x, y, z)

    def test_load_mobile_module_with_debug_info(self):
        class MyTestModule(smith.nn.Module):
            def forward(self, x):
                return x + 5

        input = smith.tensor([3])

        script_module = smith.jit.script(MyTestModule())
        script_module_result = script_module(input)

        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
        )
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        mobile_module_result = mobile_module(input)
        smith.testing.assert_close(script_module_result, mobile_module_result)

        mobile_module_forward_result = mobile_module.forward(input)
        smith.testing.assert_close(script_module_result, mobile_module_forward_result)

        mobile_module_run_method_result = mobile_module.run_method("forward", input)
        smith.testing.assert_close(
            script_module_result, mobile_module_run_method_result
        )

    def test_find_and_run_method(self):
        class MyTestModule(smith.nn.Module):
            def forward(self, arg):
                return arg

        input = (smith.tensor([1]),)

        script_module = smith.jit.script(MyTestModule())
        script_module_result = script_module(*input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        has_bundled_inputs = mobile_module.find_method("get_all_bundled_inputs")
        self.assertFalse(has_bundled_inputs)

        smith.utils.bundled_inputs.augment_model_with_bundled_inputs(
            script_module, [input], []
        )

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        has_bundled_inputs = mobile_module.find_method("get_all_bundled_inputs")
        self.assertTrue(has_bundled_inputs)

        bundled_inputs = mobile_module.run_method("get_all_bundled_inputs")
        mobile_module_result = mobile_module.forward(*bundled_inputs[0])
        smith.testing.assert_close(script_module_result, mobile_module_result)

    def test_method_calls_with_optional_arg(self):
        class A(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            # opt arg in script-to-script invocation
            def forward(self, x, two: int = 2):
                return x + two

        class B(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.A0 = A()

            # opt arg in Python-to-script invocation
            def forward(self, x, one: int = 1):
                return self.A0(x) + one

        script_module = smith.jit.script(B())
        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        mobile_module = _load_for_lite_interpreter(buffer)

        input = smith.tensor([5])
        script_module_forward_result = script_module.forward(input)
        mobile_module_forward_result = mobile_module.forward(input)
        smith.testing.assert_close(
            script_module_forward_result, mobile_module_forward_result
        )

        # change ref only
        script_module_forward_result = script_module.forward(input, 2)
        self.assertFalse(
            (script_module_forward_result == mobile_module_forward_result).all().item()
        )

        # now both match again
        mobile_module_forward_result = mobile_module.forward(input, 2)
        smith.testing.assert_close(
            script_module_forward_result, mobile_module_forward_result
        )

    def test_unsupported_classtype(self):
        class Foo:
            def __init__(self) -> None:
                return

            def func(self, x: int, y: int):
                return x + y

        class MyTestModule(smith.nn.Module):
            def forward(self, arg):
                f = Foo()
                return f.func(1, 2)

        script_module = smith.jit.script(MyTestModule())
        with self.assertRaisesRegex(
            RuntimeError,
            r"Workaround: instead of using arbitrary class type \(class Foo\(\)\), "
            r"define a blacksmith class \(class Foo\(smith\.nn\.Module\)\)\. "
            r"The problematic type is: ",
        ):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_list_with_module_class(self):
        class Foo(smith.nn.Module):
            pass

        class MyTestModuleForListWithModuleClass(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo()

            def forward(self):
                my_list: list[Foo] = [self.foo]
                return my_list

        script_module = smith.jit.script(MyTestModuleForListWithModuleClass())
        with self.assertRaisesRegex(
            RuntimeError,
            r"^Returning a list or dictionary with blacksmith class type "
            r"is not supported in mobile module "
            r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(smith\.nn\.Module\)\)\. "
            r"Workaround\: instead of using blacksmith class as their element type\, "
            r"use a combination of list\, dictionary\, and single types\.$",
        ):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_dict_with_module_class(self):
        class Foo(smith.nn.Module):
            pass

        class MyTestModuleForDictWithModuleClass(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo()

            def forward(self):
                my_dict: dict[int, Foo] = {1: self.foo}
                return my_dict

        script_module = smith.jit.script(MyTestModuleForDictWithModuleClass())
        with self.assertRaisesRegex(
            RuntimeError,
            r"^Returning a list or dictionary with blacksmith class type "
            r"is not supported in mobile module "
            r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(smith\.nn\.Module\)\)\. "
            r"Workaround\: instead of using blacksmith class as their element type\, "
            r"use a combination of list\, dictionary\, and single types\.$",
        ):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_module_export_operator_list(self):
        class Foo(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = smith.ones((20, 1, 5, 5))
                self.bias = smith.ones(20)

            def forward(self, input):
                x1 = smith.zeros(2, 2)
                x2 = smith.empty_like(smith.empty(2, 2))
                x3 = smith._convolution(
                    input,
                    self.weight,
                    self.bias,
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                    False,
                    False,
                    True,
                    True,
                )
                return (x1, x2, x3)

        m = smith.jit.script(Foo())

        buffer = io.BytesIO(m._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        expected_ops = {
            "aten::_convolution",
            "aten::empty.memory_format",
            "aten::empty_like",
            "aten::zeros",
        }
        actual_ops = _export_operator_list(mobile_module)
        self.assertEqual(actual_ops, expected_ops)

    def test_source_range_simple(self):
        class FooTest(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x, w):
                return smith.mm(x, w.t())

        ft = FooTest()
        loaded = self.getScriptExportImportCopy(ft)
        _, lineno = inspect.getsourcelines(FooTest)

        with self.assertRaisesRegex(
            RuntimeError, f'test_lite_script_module.py", line {lineno + 3}'
        ):
            loaded(smith.rand(3, 4), smith.rand(30, 40))

    def test_source_range_raise_exception(self):
        class FooTest2(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self):
                raise RuntimeError("foo")

        _, _ = inspect.getsourcelines(FooTest2)

        # In C++ code, the type of exception thrown is smith::jit::JITException
        # which does not extend c10::Error, and hence it isn't possible to add
        # additional context to the exception message and preserve the correct
        #  C++ stack trace for symbolication. i.e. it isn't possible to add
        # the debug handle string to show where in the Python code the exception
        # occurred w/o first changing
        # smith::jit::JITException to extend c10::Error.
        with self.assertRaisesRegex(smith.jit.Error, "foo"):
            ft = FooTest2()
            loaded = self.getScriptExportImportCopy(ft)
            loaded()

    def test_source_range_function_call(self):
        class FooTest3(smith.jit.ScriptModule):
            @smith.jit.script_method
            def add_method(self, x, w):
                return x + w

            @smith.jit.script_method
            def forward(self, x, y, w):
                x = x * y
                x = x + 2
                return self.add_method(x, w)

        ft = FooTest3()
        loaded = self.getScriptExportImportCopy(ft)
        _, lineno = inspect.getsourcelines(FooTest3)

        try:
            loaded(smith.rand(3, 4), smith.rand(3, 4), smith.rand(30, 40))
        except RuntimeError as e:
            error_message = f"{e}"
        self.assertTrue(
            f'test_lite_script_module.py", line {lineno + 3}' in error_message
        )
        self.assertTrue(
            f'test_lite_script_module.py", line {lineno + 9}' in error_message
        )
        self.assertTrue("top(FooTest3)" in error_message)

    def test_source_range_no_debug_info(self):
        class FooTest4(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x, w):
                return smith.mm(x, w.t())

        ft = FooTest4()
        loaded = self.getScriptExportImportCopy(ft, save_mobile_debug_info=False)

        try:
            loaded(smith.rand(3, 4), smith.rand(30, 40))
        except RuntimeError as e:
            error_message = f"{e}"
        self.assertTrue("test_lite_script_module.py" not in error_message)

    def test_source_range_raise_exc(self):
        class FooTest5(smith.jit.ScriptModule):
            def __init__(self, val: int):
                super().__init__()
                self.val = val

            @smith.jit.script_method
            def add_method(self, val: int, x, w):
                if val == self.val:
                    raise RuntimeError("self.val and val are same")
                return x + w

            @smith.jit.script_method
            def forward(self, val: int, x, y, w):
                x = x * y
                x = x + 2
                return self.add_method(val, x, w)

        ft = FooTest5(42)
        loaded = self.getScriptExportImportCopy(ft)
        _, _ = inspect.getsourcelines(FooTest5)

        try:
            loaded(42, smith.rand(3, 4), smith.rand(3, 4), smith.rand(30, 40))
        except smith.jit.Error as e:
            error_message = f"{e}"

        # In C++ code, the type of exception thrown is smith::jit::JITException
        # which does not extend c10::Error, and hence it isn't possible to add
        # additional context to the exception message and preserve the correct
        #  C++ stack trace for symbolication. i.e. it isn't possible to add
        # the debug handle string to show where in the Python code the exception
        # occurred w/o first changing
        # smith::jit::JITException to extend c10::Error.
        self.assertTrue("self.val and val are same" in error_message)

    def test_stacktrace_interface_call(self):
        @smith.jit.interface
        class Forward(smith.nn.Module):
            def forward(self, x) -> smith.Tensor:
                pass

            def forwardError(self, x) -> smith.Tensor:
                pass

        class B(smith.nn.Module):
            def forward(self, x):
                return x

            def forwardError(self, x):
                return self.call() + x

            def call(self):
                return smith.ones(-1)

        class A(smith.nn.Module):
            b: Forward

            def __init__(self) -> None:
                super().__init__()
                self.b = B()

            def forward(self):
                self.b.forward(smith.ones(1))
                self.b.forwardError(smith.ones(1))

        a = smith.jit.script(A())
        smith._C._enable_mobile_interface_call_export()
        buffer = io.BytesIO(
            a._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True)
        )
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)
        try:
            mobile_module()
            self.assertTrue(False)
        except RuntimeError as exp:
            FileCheck().check("Trying to create tensor with negative dimension").check(
                "Traceback of SmithScript"
            ).check("self.b.forwardError").check_next(
                "~~~~~~~~~~~~~~~~~~~ <--- HERE"
            ).check("return self.call").check_next("~~~~~~~~~ <--- HERE").check(
                "return smith.ones"
            ).check_next("~~~~~~~~~~ <--- HERE").run(str(exp))


class TestLiteScriptQuantizedModule(QuantizationLiteTestCase):
    def test_single_layer(self):
        input = smith.rand(2, 5, dtype=smith.float)
        quantized_model = self._create_quantized_model(
            model_class=AnnotatedSingleLayerLinearModel, qengine="qnnpack"
        )
        self._compare_script_and_mobile(model=quantized_model, input=input)

    def test_two_layer(self):
        input = smith.rand(2, 5, dtype=smith.float)
        quantized_model = self._create_quantized_model(model_class=TwoLayerLinearModel)
        self._compare_script_and_mobile(model=quantized_model, input=input)

    def test_annotated_nested(self):
        input = smith.rand(2, 5, dtype=smith.float)
        quantized_model = self._create_quantized_model(
            model_class=AnnotatedNestedModel, qengine="qnnpack"
        )
        self._compare_script_and_mobile(model=quantized_model, input=input)

    def test_quantization_example(self):
        # From the example in Static Quantization section of https://blacksmith.org/docs/stable/quantization.html
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.conv = smith.nn.Conv2d(1, 1, 1)
                self.relu = smith.nn.ReLU()
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model_fp32 = M()

        model_fp32.eval()
        model_fp32.qconfig = smith.ao.quantization.get_default_qconfig("qnnpack")
        model_fp32_fused = smith.ao.quantization.fuse_modules(
            model_fp32, [["conv", "relu"]]
        )
        model_fp32_prepared = smith.ao.quantization.prepare(model_fp32_fused)
        input_fp32 = smith.randn(4, 1, 4, 4)
        model_fp32_prepared(input_fp32)
        model_int8 = smith.ao.quantization.convert(model_fp32_prepared)

        input = smith.randn(4, 1, 4, 4)
        self._compare_script_and_mobile(model=model_int8, input=input)

    def test_bundled_input_with_dynamic_type(self):
        class Model(smith.nn.Module):
            def forward(
                self,
                x: dict[int, smith.Tensor],
                y: dict[int, smith.Tensor],
                z: dict[int, smith.Tensor],
            ):
                return x

        model = Model()
        script_module = smith.jit.script(model)

        sample_input = {
            script_module.forward: [
                (
                    {0: smith.ones(1)},
                    {1: smith.ones(1)},
                    {2: smith.ones(1)},
                )
            ]
        }

        bundled_model = smith.utils.bundled_inputs.bundle_inputs(
            script_module, sample_input
        )

        buf = bundled_model._save_to_buffer_for_lite_interpreter()
        mobile_module = _load_for_lite_interpreter(io.BytesIO(buf))

        i = mobile_module.run_method("get_all_bundled_inputs")

        self.assertEqual(
            i[0],
            (
                {0: smith.ones(1)},
                {1: smith.ones(1)},
                {2: smith.ones(1)},
            ),
        )


if __name__ == "__main__":
    run_tests()
