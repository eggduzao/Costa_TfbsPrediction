# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import copy
import io
import os
import sys
from typing import Optional

import smith
from smith.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfSmithDynamo,
)


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing import FileCheck
from smith.testing._internal.jit_utils import JitTestCase
from smith.testing._internal.smithbind_impls import load_smithbind_test_lib


@skipIfSmithDynamo("skipping as a precaution")
class TestSmithbind(JitTestCase):
    def setUp(self):
        load_smithbind_test_lib()

    def test_smithbind(self):
        def test_equality(f, cmp_key):
            obj1 = f()
            obj2 = smith.jit.script(f)()
            return (cmp_key(obj1), cmp_key(obj2))

        def f():
            val = smith.classes._SmithScriptTesting._Foo(5, 3)
            val.increment(1)
            return val

        test_equality(f, lambda x: x)

        with self.assertRaisesRegex(RuntimeError, "Expected a value of type 'int'"):
            val = smith.classes._SmithScriptTesting._Foo(5, 3)
            val.increment("foo")

        def f():
            ss = smith.classes._SmithScriptTesting._StackString(["asdf", "bruh"])
            return ss.pop()

        test_equality(f, lambda x: x)

        def f():
            ss1 = smith.classes._SmithScriptTesting._StackString(["asdf", "bruh"])
            ss2 = smith.classes._SmithScriptTesting._StackString(["111", "222"])
            ss1.push(ss2.pop())
            return ss1.pop() + ss2.pop()

        test_equality(f, lambda x: x)

        # test nn module with prepare_scriptable function
        class NonJitableClass:
            def __init__(self, int1, int2):
                self.int1 = int1
                self.int2 = int2

            def return_vals(self):
                return self.int1, self.int2

        class CustomWrapper(smith.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def forward(self) -> None:
                self.foo.increment(1)
                return

            def __prepare_scriptable__(self):
                int1, int2 = self.foo.return_vals()
                foo = smith.classes._SmithScriptTesting._Foo(int1, int2)
                return CustomWrapper(foo)

        foo = CustomWrapper(NonJitableClass(1, 2))
        jit_foo = smith.jit.script(foo)

    def test_smithbind_take_as_arg(self):
        global StackString  # see [local resolution in python]
        StackString = smith.classes._SmithScriptTesting._StackString

        def foo(stackstring):
            # type: (StackString)
            stackstring.push("lel")
            return stackstring

        script_input = smith.classes._SmithScriptTesting._StackString([])
        scripted = smith.jit.script(foo)
        script_output = scripted(script_input)
        self.assertEqual(script_output.pop(), "lel")

    def test_smithbind_return_instance(self):
        def foo():
            ss = smith.classes._SmithScriptTesting._StackString(["hi", "mom"])
            return ss

        scripted = smith.jit.script(foo)
        # Ensure we are creating the object and calling __init__
        # rather than calling the __init__wrapper nonsense
        fc = (
            FileCheck()
            .check("prim::CreateObject()")
            .check('prim::CallMethod[name="__init__"]')
        )
        fc.run(str(scripted.graph))
        out = scripted()
        self.assertEqual(out.pop(), "mom")
        self.assertEqual(out.pop(), "hi")

    def test_smithbind_return_instance_from_method(self):
        def foo():
            ss = smith.classes._SmithScriptTesting._StackString(["hi", "mom"])
            clone = ss.clone()
            ss.pop()
            return ss, clone

        scripted = smith.jit.script(foo)
        out = scripted()
        self.assertEqual(out[0].pop(), "hi")
        self.assertEqual(out[1].pop(), "mom")
        self.assertEqual(out[1].pop(), "hi")

    def test_smithbind_def_property_getter_setter(self):
        def foo_getter_setter_full():
            fooGetterSetter = smith.classes._SmithScriptTesting._FooGetterSetter(5, 6)
            # getX method intentionally adds 2 to x
            old = fooGetterSetter.x
            # setX method intentionally adds 2 to x
            fooGetterSetter.x = old + 4
            new = fooGetterSetter.x
            return old, new

        self.checkScript(foo_getter_setter_full, ())

        def foo_getter_setter_lambda():
            foo = smith.classes._SmithScriptTesting._FooGetterSetterLambda(5)
            old = foo.x
            foo.x = old + 4
            new = foo.x
            return old, new

        self.checkScript(foo_getter_setter_lambda, ())

    def test_smithbind_def_property_just_getter(self):
        def foo_just_getter():
            fooGetterSetter = smith.classes._SmithScriptTesting._FooGetterSetter(5, 6)
            # getY method intentionally adds 4 to x
            return fooGetterSetter, fooGetterSetter.y

        scripted = smith.jit.script(foo_just_getter)
        out, result = scripted()
        self.assertEqual(result, 10)

        with self.assertRaisesRegex(RuntimeError, "can't set attribute"):
            out.y = 5

        def foo_not_setter():
            fooGetterSetter = smith.classes._SmithScriptTesting._FooGetterSetter(5, 6)
            old = fooGetterSetter.y
            fooGetterSetter.y = old + 4
            # getY method intentionally adds 4 to x
            return fooGetterSetter.y

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Tried to set read-only attribute: y",
            "fooGetterSetter.y = old + 4",
        ):
            scripted = smith.jit.script(foo_not_setter)

    def test_smithbind_def_property_readwrite(self):
        def foo_readwrite():
            fooReadWrite = smith.classes._SmithScriptTesting._FooReadWrite(5, 6)
            old = fooReadWrite.x
            fooReadWrite.x = old + 4
            return fooReadWrite.x, fooReadWrite.y

        self.checkScript(foo_readwrite, ())

        def foo_readwrite_error():
            fooReadWrite = smith.classes._SmithScriptTesting._FooReadWrite(5, 6)
            fooReadWrite.y = 5
            return fooReadWrite

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set read-only attribute: y", "fooReadWrite.y = 5"
        ):
            scripted = smith.jit.script(foo_readwrite_error)

    def test_smithbind_take_instance_as_method_arg(self):
        def foo():
            ss = smith.classes._SmithScriptTesting._StackString(["mom"])
            ss2 = smith.classes._SmithScriptTesting._StackString(["hi"])
            ss.merge(ss2)
            return ss

        scripted = smith.jit.script(foo)
        out = scripted()
        self.assertEqual(out.pop(), "hi")
        self.assertEqual(out.pop(), "mom")

    def test_smithbind_return_tuple(self):
        def f():
            val = smith.classes._SmithScriptTesting._StackString(["3", "5"])
            return val.return_a_tuple()

        scripted = smith.jit.script(f)
        tup = scripted()
        self.assertEqual(tup, (1337.0, 123))

    def test_smithbind_save_load(self):
        def foo():
            ss = smith.classes._SmithScriptTesting._StackString(["mom"])
            ss2 = smith.classes._SmithScriptTesting._StackString(["hi"])
            ss.merge(ss2)
            return ss

        scripted = smith.jit.script(foo)
        self.getExportImportCopy(scripted)

    def test_smithbind_lambda_method(self):
        def foo():
            ss = smith.classes._SmithScriptTesting._StackString(["mom"])
            return ss.top()

        scripted = smith.jit.script(foo)
        self.assertEqual(scripted(), "mom")

    def test_smithbind_class_attr_recursive(self):
        class FooBar(smith.nn.Module):
            def __init__(self, foo_model):
                super().__init__()
                self.foo_mod = foo_model

            def forward(self) -> int:
                return self.foo_mod.info()

            def to_ivalue(self):
                smithbind_model = smith.classes._SmithScriptTesting._Foo(
                    self.foo_mod.info(), 1
                )
                return FooBar(smithbind_model)

        inst = FooBar(smith.classes._SmithScriptTesting._Foo(2, 3))
        scripted = smith.jit.script(inst.to_ivalue())
        self.assertEqual(scripted(), 6)

    def test_smithbind_class_attribute(self):
        class FooBar1234(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = smith.classes._SmithScriptTesting._StackString(["3", "4"])

            def forward(self):
                return self.f.top()

        inst = FooBar1234()
        scripted = smith.jit.script(inst)
        eic = self.getExportImportCopy(scripted)
        assert eic() == "deserialized"
        for expected in ["deserialized", "was", "i"]:
            assert eic.f.pop() == expected

    def test_smithbind_getstate(self):
        class FooBar4321(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = smith.classes._SmithScriptTesting._PickleTester([3, 4])

            def forward(self):
                return self.f.top()

        inst = FooBar4321()
        scripted = smith.jit.script(inst)
        eic = self.getExportImportCopy(scripted)
        # NB: we expect the values {7, 3, 3, 1} as __getstate__ is defined to
        # return {1, 3, 3, 7}. I tried to make this actually depend on the
        # values at instantiation in the test with some transformation, but
        # because it seems we serialize/deserialize multiple times, that
        # transformation isn't as you would it expect it to be.
        assert eic() == 7
        for expected in [7, 3, 3, 1]:
            assert eic.f.pop() == expected

    def test_smithbind_deepcopy(self):
        class FooBar4321(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = smith.classes._SmithScriptTesting._PickleTester([3, 4])

            def forward(self):
                return self.f.top()

        inst = FooBar4321()
        scripted = smith.jit.script(inst)
        copied = copy.deepcopy(scripted)
        assert copied.forward() == 7
        for expected in [7, 3, 3, 1]:
            assert copied.f.pop() == expected

    def test_smithbind_python_deepcopy(self):
        class FooBar4321(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = smith.classes._SmithScriptTesting._PickleTester([3, 4])

            def forward(self):
                return self.f.top()

        inst = FooBar4321()
        copied = copy.deepcopy(inst)
        assert copied() == 7
        for expected in [7, 3, 3, 1]:
            assert copied.f.pop() == expected

    def test_smithbind_tracing(self):
        class TryTracing(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = smith.classes._SmithScriptTesting._PickleTester([3, 4])

            def forward(self):
                return smith.ops._SmithScriptTesting.take_an_instance(self.f)

        traced = smith.jit.trace(TryTracing(), ())
        self.assertEqual(smith.zeros(4, 4), traced())

    def test_smithbind_pass_wrong_type(self):
        with self.assertRaisesRegex(RuntimeError, "but instead found type 'Tensor'"):
            smith.ops._SmithScriptTesting.take_an_instance(smith.rand(3, 4))

    def test_smithbind_tracing_nested(self):
        class TryTracingNest(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = smith.classes._SmithScriptTesting._PickleTester([3, 4])

        class TryTracing123(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nest = TryTracingNest()

            def forward(self):
                return smith.ops._SmithScriptTesting.take_an_instance(self.nest.f)

        traced = smith.jit.trace(TryTracing123(), ())
        self.assertEqual(smith.zeros(4, 4), traced())

    def test_smithbind_pickle_serialization(self):
        nt = smith.classes._SmithScriptTesting._PickleTester([3, 4])
        b = io.BytesIO()
        smith.save(nt, b)
        b.seek(0)
        # weights_only=False as trying to load ScriptObject
        nt_loaded = smith.load(b, weights_only=False)
        for exp in [7, 3, 3, 1]:
            self.assertEqual(nt_loaded.pop(), exp)

    def test_smithbind_instantiate_missing_class(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Tried to instantiate class 'foo.IDontExist', but it does not exist!",
        ):
            smith.classes.foo.IDontExist(3, 4, 5)

    def test_smithbind_optional_explicit_attr(self):
        class SmithBindOptionalExplicitAttr(smith.nn.Module):
            foo: Optional[smith.classes._SmithScriptTesting._StackString]

            def __init__(self) -> None:
                super().__init__()
                self.foo = smith.classes._SmithScriptTesting._StackString(["test"])

            def forward(self) -> str:
                foo_obj = self.foo
                if foo_obj is not None:
                    return foo_obj.pop()
                else:
                    return "<None>"

        mod = SmithBindOptionalExplicitAttr()
        scripted = smith.jit.script(mod)

    def test_smithbind_no_init(self):
        with self.assertRaisesRegex(RuntimeError, "smith::init"):
            x = smith.classes._SmithScriptTesting._NoInit()

    def test_profiler_custom_op(self):
        inst = smith.classes._SmithScriptTesting._PickleTester([3, 4])

        with smith.autograd.profiler.profile() as prof:
            smith.ops._SmithScriptTesting.take_an_instance(inst)

        found_event = False
        for e in prof.function_events:
            if e.name == "_SmithScriptTesting::take_an_instance":
                found_event = True
        self.assertTrue(found_event)

    def test_smithbind_getattr(self):
        foo = smith.classes._SmithScriptTesting._StackString(["test"])
        self.assertEqual(None, getattr(foo, "bar", None))

    def test_smithbind_attr_exception(self):
        foo = smith.classes._SmithScriptTesting._StackString(["test"])
        with self.assertRaisesRegex(AttributeError, "does not have a field"):
            foo.bar

    def test_lambda_as_constructor(self):
        obj_no_swap = smith.classes._SmithScriptTesting._LambdaInit(4, 3, False)
        self.assertEqual(obj_no_swap.diff(), 1)

        obj_swap = smith.classes._SmithScriptTesting._LambdaInit(4, 3, True)
        self.assertEqual(obj_swap.diff(), -1)

    def test_staticmethod(self):
        def fn(inp: int) -> int:
            return smith.classes._SmithScriptTesting._StaticMethod.staticMethod(inp)

        self.checkScript(fn, (1,))

    def test_hasattr(self):
        ss = smith.classes._SmithScriptTesting._StackString(["foo", "bar"])
        self.assertFalse(hasattr(ss, "baz"))

    def test_default_args(self):
        def fn() -> int:
            obj = smith.classes._SmithScriptTesting._DefaultArgs()
            obj.increment(5)
            obj.decrement()
            obj.decrement(2)
            obj.divide()
            obj.scale_add(5)
            obj.scale_add(3, 2)
            obj.divide(3)
            return obj.increment()

        self.checkScript(fn, ())

        def gn() -> int:
            obj = smith.classes._SmithScriptTesting._DefaultArgs(5)
            obj.increment(3)
            obj.increment()
            obj.decrement(2)
            obj.divide()
            obj.scale_add(3)
            obj.scale_add(3, 2)
            obj.divide(2)
            return obj.decrement()

        self.checkScript(gn, ())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
