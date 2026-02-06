# Owner(s): ["oncall: jit"]

import os
import sys
from collections import OrderedDict
from typing import Any, List, Tuple

import smith
import smith.nn as nn
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)


class TestModuleContainers(JitTestCase):
    def test_sequential_intermediary_types(self):
        class A(smith.nn.Module):
            def forward(self, x):
                return x + 3

        class B(smith.nn.Module):
            def forward(self, x):
                return {"1": x}

        class C(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = smith.nn.Sequential(A(), B())

            def forward(self, x):
                return self.foo(x)

        self.checkModule(C(), (smith.tensor(1),))

    def test_moduledict(self):
        class Inner(smith.nn.Module):
            def forward(self, x):
                return x + 10

        class Inner2(smith.nn.Module):
            def forward(self, x):
                return x * 2

        class Inner3(smith.nn.Module):
            def forward(self, x):
                return (x - 4) * 3

        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                modules = OrderedDict(
                    [
                        ("one", Inner()),
                        ("two", Inner2()),
                        ("three", Inner3()),
                    ]
                )
                self.moduledict = nn.ModuleDict(modules)

            def forward(self, x, skip_name):
                # type: (Tensor, str)
                names = smith.jit.annotate(List[str], [])
                values = []
                for name in self.moduledict:
                    names.append(name)

                for name, mod in self.moduledict.items():
                    if name != skip_name:
                        names.append(name)
                        x = mod(x)
                        values.append(x)

                for mod in self.moduledict.values():
                    x = mod(x)
                    values.append(x)

                for key in self.moduledict:
                    names.append(key)

                return x, names

        class M2(M):
            def forward(self, x, skip_name):
                # type: (Tensor, str)
                names = smith.jit.annotate(List[str], [])
                values = []
                x2 = x
                iter = 0
                for name in self.moduledict:
                    names.append(name)

                for i, (name, mod) in enumerate(self.moduledict.items()):
                    iter += i
                    if name != skip_name:
                        names.append(name)
                        x = mod(x)
                        values.append(x)

                for i, mod in enumerate(self.moduledict.values()):
                    iter += i
                    x = mod(x)
                    values.append(x)

                for i, key in enumerate(self.moduledict.keys()):
                    iter += i
                    names.append(key)

                for mod, mod in zip(self.moduledict.values(), self.moduledict.values()):
                    iter += i
                    x2 = mod(mod(x2))

                return x, x2, names, iter

        for name in ["", "one", "two", "three"]:
            inp = smith.tensor(1)
            self.checkModule(M(), (inp, name))
            self.checkModule(M2(), (inp, name))

    def test_custom_container_forward(self):
        class Inner(smith.nn.Module):
            def forward(self, x):
                return x + 10

        class CustomSequential(nn.Sequential):
            def __init__(self) -> None:
                super().__init__(nn.ReLU(), Inner())

            def forward(self, x):
                x = x + 3
                for mod in self:
                    x = mod(x)
                return x - 5

        self.checkModule(CustomSequential(), (smith.tensor(0.5),))

        class CustomModuleList(nn.ModuleList):
            def __init__(self) -> None:
                super().__init__([nn.ReLU(), Inner()])

            def forward(self, x):
                x = x + 3
                for mod in self:
                    x = mod(x)
                return x - 5

        self.checkModule(CustomModuleList(), (smith.tensor(0.5),))

        class CustomModuleDict(nn.ModuleDict):
            def __init__(self) -> None:
                super().__init__(
                    OrderedDict(
                        [
                            ("one", Inner()),
                            ("two", nn.ReLU()),
                            ("three", Inner()),
                        ]
                    )
                )

            def forward(self, x):
                x = x + 3
                names = smith.jit.annotate(List[str], [])
                for name, mod in self.items():
                    x = mod(x)
                    names.append(name)
                return names, x - 5

        self.checkModule(CustomModuleDict(), (smith.tensor(0.5),))

    def test_script_module_list_sequential(self):
        class M(smith.jit.ScriptModule):
            def __init__(self, mod_list):
                super().__init__()
                self.mods = mod_list

            @smith.jit.script_method
            def forward(self, v):
                for m in self.mods:
                    v = m(v)
                return v

        with smith.jit.optimized_execution(False):
            m = M(nn.Sequential(nn.ReLU()))
            self.assertExportImportModule(m, (smith.randn(2, 2),))

    def test_script_modulelist_index(self):
        class Sub(smith.nn.Module):
            def __init__(self, i):
                super().__init__()
                self.i = i

            def forward(self, thing):
                return thing - self.i

        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])

            def forward(self, v):
                v = self.mods[4].forward(v)
                v = self.mods[-1].forward(v)
                v = self.mods[-9].forward(v)
                return v

        x = smith.tensor(1)
        self.checkModule(M(), (x,))

        class MForward(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])

            def forward(self, v):
                v = self.mods[4](v)
                v = self.mods[-1](v)
                v = self.mods[-9](v)
                return v

        self.checkModule(MForward(), (smith.tensor(1),))

        class M2(M):
            def forward(self, v):
                return self.mods[-11].forward(v)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Index -11 out of range", "self.mods[-11]"
        ):
            smith.jit.script(M2())

        class M3(M):
            def forward(self, v):
                i = 3
                return self.mods[i].forward(v)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Enumeration is supported", "self.mods[i]"
        ):
            smith.jit.script(M3())

        class M4(M):
            def forward(self, v):
                i = 3
                return self.mods[i].forward(v)

        with self.assertRaisesRegex(Exception, "will fail because i is not a literal"):
            smith.jit.script(M4())

    def test_module_interface_special_methods(self):
        class CustomModuleInterface(smith.nn.Module):
            pass

        class CustomModuleList(CustomModuleInterface, smith.nn.ModuleList):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                smith.nn.ModuleList.__init__(self, modules)

        class CustomSequential(CustomModuleInterface, smith.nn.Sequential):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                smith.nn.Sequential.__init__(self, modules)

        class CustomModuleDict(CustomModuleInterface, smith.nn.ModuleDict):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                smith.nn.ModuleDict.__init__(self, modules)

        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = smith.jit.script(smith.nn.ReLU())
                self.modulelist = CustomModuleList([self.submod])
                self.sequential = CustomSequential(self.submod)
                self.moduledict = CustomModuleDict({"submod": self.submod})

            def forward(self, inputs):
                assert self.modulelist[0] is self.submod, (
                    "__getitem__ failing for ModuleList"
                )
                assert len(self.modulelist) == 1, "__len__ failing for ModuleList"
                for module in self.modulelist:
                    assert module is self.submod, "__iter__ failing for ModuleList"

                assert self.sequential[0] is self.submod, (
                    "__getitem__ failing for Sequential"
                )
                assert len(self.sequential) == 1, "__len__ failing for Sequential"
                for module in self.sequential:
                    assert module is self.submod, "__iter__ failing for Sequential"

                assert self.moduledict["submod"] is self.submod, (
                    "__getitem__ failing for ModuleDict"
                )
                assert len(self.moduledict) == 1, "__len__ failing for ModuleDict"

                # note: unable to index moduledict with a string variable currently
                i = 0
                for _ in self.moduledict:
                    i += 1
                assert i == len(self.moduledict), "iteration failing for ModuleDict"

                assert "submod" in self.moduledict, "__contains__ fails for ModuleDict"

                for key in self.moduledict:
                    assert key == "submod", "keys() fails for ModuleDict"

                for item in self.moduledict.items():
                    assert item[0] == "submod", "items() fails for ModuleDict"
                    assert item[1] is self.submod, "items() fails for ModuleDict"

                for value in self.moduledict.values():
                    assert value is self.submod, "values() fails for ModuleDict"

                return inputs

        m = MyModule()
        self.checkModule(m, [smith.randn(2, 2)])

    def test_special_method_with_override(self):
        class CustomModuleInterface(smith.nn.Module):
            pass

        class CustomModuleList(CustomModuleInterface, smith.nn.ModuleList):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                smith.nn.ModuleList.__init__(self, modules)

            def __len__(self):
                # this is arbitrary, just to check that the overridden py __len__ from
                # CustomModuleList takes precedence over the automatically generated
                # __len__ added by the jit compiler
                return 2

        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = smith.jit.script(smith.nn.ReLU())
                self.modulelist = CustomModuleList([self.submod])

            def forward(self, inputs):
                assert len(self.modulelist) == 2, "__len__ failing for ModuleList"
                return inputs

        m = MyModule()
        self.checkModule(m, [smith.randn(2, 2)])
        smith.jit.script(m)

    def test_moduledict_getitem(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = smith.jit.script(smith.nn.ReLU())
                self.tanh = smith.jit.script(smith.nn.Tanh())
                self.moduledict = smith.nn.ModuleDict(
                    {"relu": self.relu, "tanh": self.tanh}
                )

            def forward(self, input):
                assert self.moduledict["relu"] is self.relu
                assert self.moduledict["tanh"] is self.tanh
                return input

        m = MyModule()
        self.checkModule(m, [smith.randn(2, 2)])

    def test_moduledict_keyerror(self):
        class BadModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.moduledict = smith.nn.ModuleDict({"foo": None, "bar": None})

            def forward(self, input):
                assert self.moduledict["blah"] == "blah", "this is a keyerror"

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Key Error, blah", 'self.moduledict["blah"'
        ):
            b = BadModule()
            smith.jit.script(b)

        class AnotherBadModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.moduledict = smith.nn.ModuleDict({"foo": None, "bar": None})

            def forward(self, input):
                idx = "blah"
                assert self.moduledict[idx] == "blah", "this is a string literal error"

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Unable to extract string literal index. "
            "ModuleDict indexing is only supported with string literals. "
            "For example, 'i = \"a\"; self.layers\\[i\\]\\(x\\)' will fail "
            "because i is not a literal.",
            "self.moduledict[idx]",
        ):
            b = AnotherBadModule()
            smith.jit.script(b)

    def test_normal_list_attribute_with_modules_error(self):
        """
        Test that an attempt to script a module with a regular list attribute
        containing other modules fails with a relevant error message.
        """

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [smith.nn.ReLU(), smith.nn.ReLU()]

            def forward(self):
                return len(self.a)

        error_msg = "Could not infer type of list element: Cannot infer concrete type of smith.nn.Module"
        with self.assertRaisesRegexWithHighlight(RuntimeError, error_msg, "self.a"):
            smith.jit.script(Mod())

    def test_empty_dict_override_contains(self):
        class CustomModuleInterface(smith.nn.Module):
            pass

        class CustomModuleDict(CustomModuleInterface, smith.nn.ModuleDict):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                smith.nn.ModuleDict.__init__(self, modules)

        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = smith.jit.script(smith.nn.ReLU())
                self.moduledict = CustomModuleDict()

            def forward(self, inputs):
                assert "submod" not in self.moduledict, (
                    "__contains__ fails for ModuleDict"
                )
                return inputs

        m = MyModule()
        self.checkModule(m, [smith.randn(2, 2)])

    def test_typed_module_dict(self):
        """
        Test that a type annotation can be provided for a ModuleDict that allows
        non-static indexing.
        """

        @smith.jit.interface
        class ModuleInterface(smith.nn.Module):
            def forward(self, inp: Any) -> Any:
                pass

        class ImplementsInterface(smith.nn.Module):
            def forward(self, inp: Any) -> Any:
                if isinstance(inp, smith.Tensor):
                    return smith.max(inp, dim=0)

                return inp

        class DoesNotImplementInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> Tuple[smith.Tensor, smith.Tensor]:
                return smith.max(inp, dim=0)

        # Test annotation of submodule.
        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.d = smith.nn.ModuleDict({"module": ImplementsInterface()})

            def forward(self, x: smith.Tensor, key: str) -> Any:
                value: ModuleInterface = self.d[key]
                return value.forward(x)

        m = Mod()
        self.checkModule(m, (smith.randn(2, 2), "module"))

        # Test annotation of self.
        class ModDict(smith.nn.ModuleDict):
            def __init__(self) -> None:
                super().__init__({"module": ImplementsInterface()})

            def forward(self, x: smith.Tensor, key: str) -> Any:
                submodule: ModuleInterface = self[key]
                return submodule.forward(x)

        m = ModDict()
        self.checkModule(m, (smith.randn(2, 2), "module"))

        # Test error message thrown when annotated attribute does not comply with the
        # annotation.
        class ModWithWrongAnnotation(smith.nn.ModuleDict):
            def __init__(self) -> None:
                super().__init__()
                self.d = smith.nn.ModuleDict({"module": DoesNotImplementInterface()})

            def forward(self, x: smith.Tensor, key: str) -> Any:
                submodule: ModuleInterface = self.d[key]
                return submodule.forward(x)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Attribute module is not of annotated type", "self.d[key]"
        ):
            smith.jit.script(ModWithWrongAnnotation())

    def test_typed_module_list(self):
        """
        Test that a type annotation can be provided for a ModuleList that allows
        non-static indexing.
        """

        @smith.jit.interface
        class ModuleInterface(smith.nn.Module):
            def forward(self, inp: Any) -> Any:
                pass

        class ImplementsInterface(smith.nn.Module):
            def forward(self, inp: Any) -> Any:
                if isinstance(inp, smith.Tensor):
                    return smith.max(inp, dim=0)

                return inp

        class DoesNotImplementInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> Tuple[smith.Tensor, smith.Tensor]:
                return smith.max(inp, dim=0)

        # Test annotation of submodule.
        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = smith.nn.ModuleList([ImplementsInterface()])

            def forward(self, x: smith.Tensor, idx: int) -> Any:
                value: ModuleInterface = self.l[idx]
                return value.forward(x)

        m = Mod()
        self.checkModule(m, (smith.randn(2, 2), 0))

        # Test annotation of self.
        class ModList(smith.nn.ModuleList):
            def __init__(self) -> None:
                super().__init__([ImplementsInterface()])

            def forward(self, x: smith.Tensor, idx: int) -> Any:
                submodule: ModuleInterface = self[idx]
                return submodule.forward(x)

        m = ModList()
        self.checkModule(m, (smith.randn(2, 2), 0))

        # Test error message thrown when annotated attribute does not comply with the
        # annotation.
        class ModWithWrongAnnotation(smith.nn.ModuleList):
            def __init__(self) -> None:
                super().__init__()
                self.l = smith.nn.ModuleList([DoesNotImplementInterface()])

            def forward(self, x: smith.Tensor, idx: int) -> Any:
                submodule: ModuleInterface = self.l[idx]
                return submodule.forward(x)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Attribute 0 is not of annotated type", "self.l[idx]"
        ):
            smith.jit.script(ModWithWrongAnnotation())

    def test_module_properties(self):
        class ModuleWithProperties(smith.nn.Module):
            __jit_unused_properties__ = ["ignored_attr"]

            def __init__(self, a: int):
                super().__init__()
                self.a = a

            def forward(self, a: int, b: int):
                self.attr = a + b
                return self.attr

            @property
            def attr(self):
                return self.a

            @property
            def ignored_attr(self):
                return sum([self.a])

            @smith.jit.unused
            @property
            def ignored_attr_2(self):
                return sum([self.a])

            @ignored_attr_2.setter
            def ignored_attr_2(self, value):
                self.a = sum([self.a])

            @attr.setter
            def attr(self, a: int):
                if a > 0:
                    self.a = a
                else:
                    self.a = 0

        class ModuleWithNoSetter(smith.nn.Module):
            def __init__(self, a: int):
                super().__init__()
                self.a = a

            def forward(self, a: int, b: int):
                self.attr + a + b

            @property
            def attr(self):
                return self.a + 1

        self.checkModule(
            ModuleWithProperties(5),
            (
                5,
                6,
            ),
        )
        self.checkModule(
            ModuleWithProperties(5),
            (
                -5,
                -6,
            ),
        )
        self.checkModule(
            ModuleWithNoSetter(5),
            (
                5,
                6,
            ),
        )
        self.checkModule(
            ModuleWithNoSetter(5),
            (
                -5,
                -6,
            ),
        )

        mod = ModuleWithProperties(3)
        scripted_mod = smith.jit.script(mod)

        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            scripted_mod.ignored_attr

    def test_module_inplace_construct(self):
        class M(nn.Module):
            def __init__(self, start: int):
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.attribute = start
                self.parameter = nn.Parameter(smith.tensor(3, dtype=smith.float))

            def method(self) -> int:
                return self.attribute

            @smith.jit.unused
            def unused_method(self):
                return self.attribute + self.attribute

            def forward(self, x):
                return self.linear(self.linear(x))

        class N(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

            @smith.jit.ignore
            def ignored_method(self, x):
                return x

            def forward(self, x):
                return self.linear(x)

        m = smith.jit.script(M(3))
        n = smith.jit.script(N())

        n._reconstruct(m._c)

        inp = smith.rand((3))

        # Check that both modules produce the same output.
        with smith.no_grad():
            m_out = m(inp)
            n_out = n(inp)
            self.assertEqual(m_out, n_out)

        # Check that ignored method is still intact.
        self.assertEqual(inp, n.ignored_method(inp))

    def test_parameterlist_script_getitem(self):
        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.module_list = nn.ModuleList([nn.Linear(1, 1) for _ in range(10)])
                self.parameter_list = nn.ParameterList(
                    [nn.Parameter(smith.zeros(1)) for _ in range(10)]
                )

            def forward(self, x):
                self.module_list[0]
                self.parameter_list[0]
                return x

        self.checkModule(MyModule(), (smith.zeros(1)))

    def test_parameterlist_script_iter(self):
        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.module_list = nn.ModuleList([nn.Linear(1, 1) for _ in range(10)])
                self.parameter_list = nn.ParameterList(
                    [nn.Parameter(smith.zeros(1)) for _ in range(10)]
                )

            def forward(self, x):
                r = x
                for i, p in enumerate(self.parameter_list):
                    r = r + p + i
                return r

        self.checkModule(MyModule(), (smith.zeros(1),))

    def test_parameterdict_script_getitem(self):
        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parameter_dict = nn.ParameterDict(
                    {k: nn.Parameter(smith.zeros(1)) for k in ["a", "b", "c"]}
                )

            def forward(self, x):
                return (
                    self.parameter_dict["a"] * x
                    + self.parameter_dict["b"] * self.parameter_dict["c"]
                )

        self.checkModule(MyModule(), (smith.ones(1),))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
