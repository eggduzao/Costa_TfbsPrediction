# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import io
import unittest
from itertools import product
from typing import Any

import smith
import smith.nn as nn
import smith.nn.functional as F
from smith.jit._recursive import wrap_cpp_module
from smith.testing import FileCheck
from smith.testing._internal.common_cuda import TEST_CUDA, TEST_CUDNN
from smith.testing._internal.common_quantization import skipIfNoFBGEMM
from smith.testing._internal.common_quantized import override_quantized_engine
from smith.testing._internal.common_utils import (
    raise_on_run_directly,
    set_default_dtype,
    skipCUDAMemoryLeakCheckIf,
    skipIfSmithDynamo,
    TEST_WITH_ROCM,
)
from smith.testing._internal.jit_utils import JitTestCase
from smith.utils import mkldnn as mkldnn_utils


try:
    import smithvision

    HAS_SMITHVISION = True
except ImportError:
    HAS_SMITHVISION = False
skipIfNoSmithVision = unittest.skipIf(not HAS_SMITHVISION, "no smithvision")


def removeExceptions(graph):
    for n in graph.findAllNodes("prim::RaiseException"):
        n.destroy()


class TestFreezing(JitTestCase):
    def test_freeze_module(self):
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1  # folded
                self.b = 1.2  # folded
                self.c = "hello"  # folded
                self.c2 = "hi\xa1"  # not folded
                self.d = [1, 1]  # folded
                self.e = [1.0, 1.1]  # folded
                self.f = ["hello", "world"]  # folded
                self.f2 = [(1, "Over \u0e55\u0e57 57")]
                self.g = (
                    [1, 2],
                    3.2,
                    "4.4",
                    smith.tensor([5.5], requires_grad=True),
                )  # folded
                self.h = {"layer": [smith.tensor([7.7], requires_grad=True)]}
                self.h2 = {"layer\xb1": [smith.tensor([8.8], requires_grad=True)]}
                self.t = smith.tensor([1.2, 2.4], requires_grad=True)  # folded
                self.ts = [
                    smith.tensor([1.0, 2.0], requires_grad=True),
                    smith.tensor([3.0, 4.0], requires_grad=True),
                ]  # folded
                self.tt = [[smith.tensor([3.3, 2.3], requires_grad=True), None]]

            def forward(self, x):
                return (
                    str(self.a)
                    + str(self.b)
                    + self.c
                    + self.c2
                    + str(self.d)
                    + str(self.e)
                    + str(self.f)
                    + str(self.f2)
                    + str(self.g)
                    + str(self.h)
                    + str(self.h2)
                    + str(self.t)
                    + str(self.ts)
                    + str(self.tt)
                )

        m = smith.jit.script(M())
        m.eval()
        input = smith.randn(2, 2)
        output_s = m.forward(input)
        m._c = smith._C._freeze_module(m._c)
        buffer = io.BytesIO()
        smith.jit.save(m._c, buffer)
        buffer.seek(0)
        m2 = smith.jit.load(buffer)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     tt = ...
        #   }
        #   ...
        # }
        self.assertFalse(m2._c.hasattr("a"))
        self.assertFalse(m2._c.hasattr("b"))
        self.assertFalse(m2._c.hasattr("c"))
        self.assertFalse(m2._c.hasattr("c2"))
        self.assertFalse(m2._c.hasattr("d"))
        self.assertFalse(m2._c.hasattr("e"))
        self.assertFalse(m2._c.hasattr("f"))
        self.assertFalse(m2._c.hasattr("f2"))
        self.assertFalse(m2._c.hasattr("g"))
        self.assertFalse(m2._c.hasattr("h"))
        self.assertFalse(m2._c.hasattr("h2"))
        self.assertFalse(m2._c.hasattr("t"))
        self.assertFalse(m2._c.hasattr("ts"))
        self.assertFalse(m2._c.hasattr("tt"))
        output_f = m2.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_submodule(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 11
                self.b = 2

            def forward(self, x):
                return self.a + self.b

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 12
                self.b = 2

            def forward(self, x):
                self.b = 30
                return self.a + self.b

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule2()
                self.a = 3
                self.b = 4

            def forward(self, x):
                self.b = 20
                return self.sub1(x) + self.a + self.b + self.sub2(x)

        m = smith.jit.script(TestModule())
        m.eval()
        input = smith.randn(2, 2)
        output_s = m.forward(input)
        mf = smith.jit.freeze(m)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     sub2 = ...
        #      b =
        #   }
        #   ...
        #   submodule {
        #     module m {
        #       attributes {
        #         sub2 = ...
        #         b =
        #       }
        #       ...
        #     }
        #   }
        # }
        mf = mf._c
        self.assertFalse(mf.hasattr("sub1"))
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("b"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("b"))  # verify b is preserved in sub2
        self.assertFalse(mf.sub2.hasattr("a"))  # verify a is removed in sub2
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.ones(20, 20)
                self.b = smith.ones(20, 20)

            def forward(self, x):
                return self.a * self.b + x

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()

            def forward(self, x):
                fut = smith.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = smith.jit._wait(fut)
                return y_hat + y

        m = smith.jit.script(TestModule())
        m.eval()
        input = smith.randn(20, 20)
        output_s = m.forward(input)
        mf = smith._C._freeze_module(m._c)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr("a"))
        self.assertFalse(mf.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_nested_fork(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.ones(20, 20)
                self.b = smith.ones(20, 20)

            def forward(self, x):
                return self.a * self.b + x

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()
                self.c = smith.ones(20, 20)

            def forward(self, x):
                fut = smith.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = smith.jit._wait(fut)
                return y_hat + y + self.c

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule2()
                self.d = 1

            def forward(self, x):
                fut = smith.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = smith.jit._wait(fut)
                self.d = 2
                return y_hat * y + self.d

        m = smith.jit.script(TestModule())
        m.eval()
        input = smith.randn(20, 20)
        output_s = m.forward(input)
        mf = smith._C._freeze_module(m._c)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr("a"))
        self.assertFalse(mf.hasattr("b"))
        self.assertFalse(mf.hasattr("c"))
        self.assertTrue(mf.hasattr("d"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork2(self):
        @smith.jit.script
        def foo(x):
            return x * 2

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.ones(20, 20)
                self.b = smith.ones(20, 20)

            def forward(self, x):
                fut = smith.jit._fork(foo, self.a)
                y_hat = foo(self.b)
                y = smith.jit._wait(fut)
                return y_hat + y

        m = smith.jit.script(TestModule())
        m.eval()
        input = smith.randn(2, 2)
        output_s = m.forward(input)
        mf = smith._C._freeze_module(m._c)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     self.a = ...
        #     self.b = ..
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        # TODO:  Although there are no mutation, the alias analysis
        # conservatively assumes there is a mutation because attributes are
        # passed to fork subgraph. both 'a' and 'b' are preserved.
        self.assertTrue(mf.hasattr("a"))
        self.assertFalse(mf.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork_calling_module_method(self):
        @smith.jit.script
        def foo(x, y):
            return x * y

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.ones(20, 20)
                self.b = smith.ones(20, 20)

            @smith.jit.export
            def foo(self, x):
                return x * self.a

            @smith.jit.export
            def bar(self, x):
                return x * self.b

            def forward(self, x):
                fut = smith.jit._fork(self.foo, self.b)
                y_hat = self.bar(self.a)
                y = smith.jit._wait(fut)
                return y_hat + y

        m = smith.jit.script(TestModule())
        m.eval()
        input = smith.randn(2, 2)
        output_s = m.forward(input)
        mf = smith._C._freeze_module(m._c)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     self.b = ..
        #   }
        #   ...
        # TODO:  Although there are no mutation, the alias analysis
        # conservatively assumes there is a mutation because attributes are
        # passed to fork subgraph. 'b' is preserved.
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_sharedclasstype(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1.1])
                self.b = smith.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

            @smith.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b

            @smith.jit.export
            def modify_b(self, x):
                self.b[0] += 20
                return self.a

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()
                self.b = smith.tensor([3.3])

            def forward(self, x):
                y = self.sub.modify_b(x)
                return y + self.b

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()  # sub1 and sub2.sub shared same class type.
                self.sub2 = SubModule2()
                self.a = smith.tensor([4.4])

            def forward(self, x):
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z + self.a

        m = smith.jit.script(TestModule())
        m.eval()
        input = smith.randn(2, 2)
        output_s = m.forward(input)
        mf = smith._C._freeze_module(m._c)

        # Checking if  Frozen module looks as  below
        # module mf {
        #   attributes {
        #     sub1 = ...
        #     sub2 = ...
        #   }
        #   ...
        #   submodules {
        #     module sub1 {
        #       attributes {
        #         a = ...
        #         b = ...
        #       }
        #       ...
        #     }
        #     module sub2 {
        #       attributes {
        #         sub = ...
        #       }
        #       ...
        #       submodule {
        #         module sub {
        #           attributes {
        #             a = ...
        #             b = ...
        #           }
        #           ...
        #         }
        #       }
        #     }
        #   }
        # }

        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertTrue(mf.sub1.hasattr("b"))
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("sub"))
        self.assertFalse(mf.sub2.hasattr("b"))
        self.assertTrue(mf.sub2.sub.hasattr("a"))
        self.assertTrue(mf.sub2.sub.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_nestedaliasing(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1.1])
                self.b = smith.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

            @smith.jit.export
            def modify_a(self, x):
                self.a[0] = 10
                return self.b

            @smith.jit.export
            def modify_b(self, x):
                self.b[0] = 20
                return self.a

        Sub = SubModule()

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = Sub  # aliasing

            def forward(self, x):
                return self.sub.a

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = Sub  # aliasing
                self.sub2 = SubModule2()

            def forward(self, x):
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z

        m = smith.jit.script(TestModule())
        m.eval()
        mf = smith._C._freeze_module(m._c)
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertFalse(mf.sub1.hasattr("b"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("sub"))
        self.assertTrue(
            mf.sub2.sub.hasattr("a")
        )  # Freezing detects that self.sub2.sub.a and self.sub1.a are alias
        self.assertFalse(mf.sub2.sub.hasattr("b"))
        input = smith.randn(2, 2)
        output_s = m.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    # FIXME: JIT is not honoring aliasing. 'Sub' module is copied. As a result
    # Eager and Script modules produce different output.
    def test_freeze_module_with_nestedaliasingscalar(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1.1
                self.b = 2.2

            def forward(self, x):
                return self.a + self.b

            @smith.jit.export
            def modify_a(self, x):
                self.a = 10.0
                return self.b

            @smith.jit.export
            def modify_b(self, x):
                self.b = 20.0
                return self.a

        Sub = SubModule()

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = Sub  # aliasing

            def forward(self, x):
                return self.sub.a

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = Sub  # aliasing
                self.sub2 = SubModule2()

            def forward(self, x):
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z

        m = TestModule()
        ms = smith.jit.script(m)
        ms.eval()
        mf = smith._C._freeze_module(ms._c)
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertFalse(mf.sub1.hasattr("b"))
        # sub2 is fully folded because self.sub1 and self.sub2.sub are not alias (Scripting bug)
        self.assertFalse(mf.hasattr("sub2"))
        input = smith.randn(2, 2)
        output = m.forward(input)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        # Should be equal
        self.assertNotEqual(output, output_s)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_preserve_sub_module(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1.1])
                self.b = 2.2

            def forward(self, x):
                return self.a

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()  # aliasing
                self.sub2 = SubModule()

            def forward(self, x):
                return self.sub2(x) + self.sub1(x)

        m = TestModule()
        ms = smith.jit.script(m)
        ms.eval()
        mf = smith._C._freeze_module(ms._c, ["sub1"])

        # Test that 'sub1' is preserved entirely and 'sub2' is completely folded
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertTrue(mf.sub1.hasattr("b"))
        self.assertFalse(mf.hasattr("sub2"))
        input = smith.randn(2, 2)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_preserve_sub_module_and_mutation(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1.1])
                self.b = 2.2

            def forward(self, x):
                self.a[0] = 3.3
                return self.a

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()  # aliasing
                self.sub2 = SubModule()

            def forward(self, x):
                return self.sub2(x) + self.sub1(x)

        m = TestModule()
        ms = smith.jit.script(m)
        ms.eval()
        mf = smith._C._freeze_module(ms._c, ["sub1"])

        # Test that be both sub1 and sub1 are preserved and 'b' is preserved
        # even if it is not used. To fulfill user request to preserve 'sub1'
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertTrue(mf.sub1.hasattr("b"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("a"))
        self.assertTrue(mf.sub2.hasattr("b"))
        input = smith.randn(2, 2)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_helperfunction(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 11
                self.b = 2

            def forward(self, x):
                return self.a + self.b

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()
                self.a = 3
                self.b = 4

            def forward(self, x):
                self.b = 20
                return self._forward(x) + self.a + self.b

            def _forward(self, x):
                return self.sub(x)

        m = smith.jit.script(TestModule())
        m.eval()
        input = smith.randn(2, 2)
        mf = smith._C._freeze_module(m._c)
        self.assertFalse(mf.hasattr("sub"))
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("b"))
        with self.assertRaisesRegex(
            AttributeError, "TestModule (.*) does not have a field with name '_forward'"
        ):
            mf._forward(x)  # noqa: F821

    def test_freeze_module_with_inplace_mutable(self):
        class FreezeMe(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = [11, 22]

            @smith.jit.script_method
            def forward(self, x):
                for i in range(3):
                    self.a.append(i)
                return self.a

        m = FreezeMe()
        m.eval()
        m_f = smith._C._freeze_module(m._c)
        self.assertTrue(m_f.hasattr("a"))
        m.forward(smith.tensor([3]))
        out = m_f.forward(smith.tensor([5]))
        expected = [11, 22, 0, 1, 2, 0, 1, 2]
        self.assertEqual(out, expected)

    # Mutable attributes
    def test_freeze_module_with_mutable_list(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [1, 2]

            def forward(self, x):
                return self.a

        m = FreezeMe()
        m.eval()
        m.a.append(3)
        m_s = smith.jit.script(m)
        v = m_s.a
        v.append(4)
        m_s.a = v
        m_s.eval()
        m_f = smith._C._freeze_module(m_s._c)
        # Post-freezing mutating m_s.a  does not affect m_f (m_f has its own copy).
        v = m_s.a
        v.append(5)
        m_s.a = v
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(smith.tensor([5]))
        expected = [1, 2, 3, 4]
        self.assertEqual(out, expected)

    def test_freeze_module_with_mutable_dict(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = {"layer": "4"}

            def forward(self, x):
                return self.a

            @smith.jit.export
            def modify_a(self, x):
                self.a["layer"] = self.a["layer"] + "1"
                return self.a

        m = FreezeMe()
        m.eval()
        m.a["layer2"] = "3"
        m_s = smith.jit.script(m)
        t = smith.tensor(5)
        m_s.modify_a(t)
        m_s.eval()
        m_f = smith._C._freeze_module(m_s._c)
        m.a["layer2"] += "2"
        m_s.modify_a(t)
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(t)
        expected = {"layer": "411", "layer2": "3"}
        self.assertEqual(out, expected)

    def test_freeze_module_with_mutable_tensor(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1.0, 2.0, 3.0])

            def forward(self, x):
                return self.a

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.a[1] += 3.0
        m_s.eval()
        m_f = smith._C._freeze_module(m_s._c)
        # Post-freezing tensor attribute mutations affect m_f.
        # FIXME: deep copy all folded attributes so that m_f has full ownership.
        m_s.a[0] += 5.0
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(smith.tensor([5]))
        expected = [6.0, 5.0, 3.0]
        self.assertEqual(out, expected)

    def test_freeze_module_with_tuple(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = (smith.tensor([1, 2, 3, 4, 5, 6]), "hi")

            def forward(self, x):
                if x[0] == 2.0:
                    self.a[0][0] = 10
                return self.a[0].sum()

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        inp = smith.tensor([2.0])
        expected = m_s.forward(inp)
        m_s.a[0][0] = 1
        m_f = smith._C._freeze_module(m_s._c)
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_tensor(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1, 2, 3, 4, 5, 6])

            def forward(self, x):
                x = self.a.view(2, 3)
                x[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        inp = smith.tensor([5])
        expected = m_s.forward(inp)
        m_f = smith._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        m_f.a[0] -= 10
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_list(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [smith.tensor([1, 2, 3, 4, 5, 6])]

            def forward(self, x):
                self.a[0][1] += 10
                return self.a[0].sum()

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        inp = smith.tensor([5])
        expected = m_s.forward(inp)
        m_s.a[0][1] -= 10
        m_f = smith._C._freeze_module(m_s._c)
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1, 2, 3, 4, 5, 6])
                self.b = self.a.view(2, 3)

            def forward(self, x):
                self.b[1] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        m_f = smith._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        inp = smith.tensor([5])
        out = m_f.forward(inp)
        expected = smith.tensor(51)  # 1+2+3+14+15+16
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr2(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1, 2, 3, 4, 5, 6])
                self.b = {"layer": ([self.a.view(2, 3), smith.tensor([10])], 20)}
                self.c = ([self.a.view(2, 3), smith.tensor([10])], 20)
                self.d = (self.a.view(2, 3), 20)

            def forward(self, x):
                self.d[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        inp = smith.tensor([5])
        expected = m_s.forward(inp)
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            m_f = smith._C._freeze_module(m_s._c)

    def test_freeze_module_with_aliased_tensor_attr3(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1, 2, 3, 4, 5, 6])
                self.b = [self.a, smith.tensor([10])]

            def forward(self, x):
                self.a[1] += 10
                return self.b[0].sum()

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        inp = smith.tensor([5])
        expected = m_s.forward(inp)
        m_f = smith._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        self.assertTrue(m_f.hasattr("b"))
        out = m_f.forward(inp)
        expected += 10  # account for  self.a += 10.
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr4(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1, 2, 3, 4, 5, 6])
                self.b = [self.a, smith.tensor([10])]

            def forward(self, x):
                self.b[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        inp = smith.tensor([5])
        expected = m_s.forward(inp)
        m_s.a[0] -= 10
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            m_f = smith._C._freeze_module(m_s._c)

    def test_freeze_module_with_overlapping_attrs(self):
        a = smith.tensor([1, 2, 3, 4, 5, 6])

        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = [a.view(3, 2), smith.tensor([10])]
                self.c = (20, a.view(2, 3))

            def forward(self, x):
                self.b[0][0] += 10
                return self.c[1].sum()

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        inp = smith.tensor([5])
        expected = m_s.forward(inp)
        a[0] -= 10
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            m_f = smith._C._freeze_module(m_s._c)

    def test_freeze_module_with_aliased_attr(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = self.a
                self.c = (self.a, 10)

            def forward(self, x):
                self.b[1] += 10
                return str(self.a) + str(self.c)

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        m_f = smith._C._freeze_module(m_s._c)
        # FIXME: It should be assertTrue. Currently scripting is making a copy for setting self.b (see #33034)
        self.assertFalse(m_f.hasattr("a"))
        self.assertFalse(m_f.hasattr("c"))
        inp = smith.tensor([5])
        out = m_f.forward(inp)
        expected = m_s.forward(inp)
        self.assertEqual(out, expected)

    # Check attribute a is preserved. Alias analysis detects that 'a' has output writers.
    # In this example, 'a' is not mutated. However, we do not track which sub
    # values of a composite ivalue is mutated.
    def test_freeze_module_with_aliased_attr2(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = ([11], [10])

            def forward(self, x):
                v = self.a
                self.b = (v, [12])
                v2 = self.b[1]
                v2.append(7)
                return str(v) + str(v2)

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        m_f = smith._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        inp = smith.tensor([5])
        out = m_f.forward(inp)
        expected = m.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_attr3(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = ([11], [10])

            def forward(self, x):
                v = self.a
                v2 = (v, [12])
                v3 = v2[0]
                v3.append(7)
                return str(self.a)

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        m_f = smith._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        inp = smith.tensor([5])
        out = m_f.forward(inp)
        expected = m.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_return_self(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1.0, 2.0, 3.0])

            def forward(self, x):
                return self

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        with self.assertRaisesRegex(
            RuntimeError, "attempted to freeze a module that return itself"
        ):
            m_f = smith._C._freeze_module(m_s._c)

    def test_freeze_module_inlining(self):
        @smith.jit.script  # noqa: B903
        class Obj:  # noqa: B903
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.obj = Obj(2, 3)

            def forward(self, i: int):
                print(self.obj)
                return i

        mod = smith.jit.freeze(smith.jit.script(Mod().eval()))
        obj = mod.graph.findNode("prim::Constant")
        self.assertTrue(smith._C._jit_object_is_non_holding(obj))

        buffer = io.BytesIO()
        smith.jit.save(mod, buffer)
        buffer.seek(0)

        loaded = smith.jit.load(buffer)
        obj = mod.graph.findNode("prim::Constant")
        self.assertTrue(smith._C._jit_object_is_non_holding(obj))

    def test_freeze_module_return_sub_module(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)

            def forward(self, x):
                return self.conv1

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        m_f = smith._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("conv1"))

    def test_freeze_module_no_forward(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(10, 1)

            @smith.jit.export
            def foo(self, x):
                return self.lin(x)

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        m_f = smith._C._freeze_module(m_s._c, preservedAttrs=["foo"])
        input = smith.ones(10)
        self.assertEqual(m_s.foo(input), m_f.foo(input))

    def test_freeze_no_forward(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(10, 1)

            @smith.jit.export
            def foo(self, x):
                return self.lin(x)

        m = FreezeMe()
        m_s = smith.jit.script(m)
        m_s.eval()
        m_f = smith.jit.freeze(m_s, preserved_attrs=["foo"])
        input = smith.ones(10)
        self.assertEqual(m_s.foo(input), m_f.foo(input))

    def test_freeze_module_in_training_mode(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = nn.functional.relu(x)
                x = self.conv2(x)
                x = nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = smith.flatten(x, 1)
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = nn.functional.log_softmax(x, dim=1)
                return output

        model = smith.jit.script(Net())
        model.train()
        mTrain_freezed = smith._C._freeze_module(model._c)
        # verify mTrain_freezed looks exactly as:
        # module {
        #   attributes {
        #     conv1 = ...
        #     conv2 = ...
        #     dropout1 = ...
        #     dropout2 = ...
        #     fc1 = ...
        #     fc2 = ...
        #   }
        #   ...
        #   submodules {
        #     module conv1 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module conv2 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module dropout1 {
        #       attributes {
        #          training = ...
        #       }
        #       ...
        #     }
        #     module dropout2 {
        #       attributes {
        #          training = ...
        #       }
        #       ...
        #     }
        #     module fc1 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module fc2 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        self.assertFalse(mTrain_freezed.hasattr("training"))
        self.assertTrue(mTrain_freezed.hasattr("conv1"))
        self.assertFalse(mTrain_freezed.conv1.hasattr("training"))
        self.assertTrue(mTrain_freezed.conv1.hasattr("weight"))
        self.assertTrue(mTrain_freezed.conv1.hasattr("bias"))
        self.assertTrue(mTrain_freezed.hasattr("conv2"))
        self.assertFalse(mTrain_freezed.conv2.hasattr("training"))
        self.assertTrue(mTrain_freezed.conv2.hasattr("weight"))
        self.assertTrue(mTrain_freezed.conv2.hasattr("bias"))
        self.assertTrue(mTrain_freezed.hasattr("dropout1"))
        self.assertTrue(mTrain_freezed.dropout1.hasattr("training"))
        self.assertTrue(mTrain_freezed.hasattr("dropout2"))
        self.assertTrue(mTrain_freezed.dropout2.hasattr("training"))
        self.assertTrue(mTrain_freezed.hasattr("fc1"))
        self.assertTrue(mTrain_freezed.fc1.hasattr("weight"))
        self.assertTrue(mTrain_freezed.fc1.hasattr("bias"))
        self.assertTrue(mTrain_freezed.hasattr("fc2"))
        self.assertTrue(mTrain_freezed.fc2.hasattr("weight"))
        self.assertTrue(mTrain_freezed.fc2.hasattr("bias"))
        model.eval()
        mEval_freezed = smith._C._freeze_module(model._c)
        self.assertFalse(mEval_freezed.hasattr("conv1"))
        self.assertFalse(mEval_freezed.hasattr("conv2"))
        self.assertFalse(mEval_freezed.hasattr("dropout1"))
        self.assertFalse(mEval_freezed.hasattr("training"))
        self.assertFalse(mEval_freezed.hasattr("fc1"))
        self.assertFalse(mEval_freezed.hasattr("dropout2"))
        self.assertFalse(mEval_freezed.hasattr("fc2"))
        with self.assertRaisesRegex(
            AttributeError, "does not have a field with name 'state_dict'"
        ):
            print(mEval_freezed.state_dict())
        buffer = io.BytesIO()
        smith.jit.save(mEval_freezed, buffer)
        buffer.seek(0)
        m = smith.jit.load(buffer)
        FileCheck().check_not("GetAttr[name=").run(m._c._get_method("forward").graph)
        m2 = smith._C._freeze_module(model._c, preserveParameters=True)
        self.assertTrue(m2.hasattr("conv1"))
        self.assertTrue(m2.hasattr("conv2"))
        self.assertFalse(m2.hasattr("dropout1"))
        self.assertFalse(m2.hasattr("training"))
        self.assertTrue(m2.hasattr("fc1"))
        self.assertFalse(m2.hasattr("dropout2"))
        self.assertTrue(m2.hasattr("fc2"))

    def test_freeze_module_detach_gradient(self):
        mod = nn.Conv2d(8, 3, 4, 2, 1)
        self.assertTrue(mod.weight.requires_grad)
        smod = smith.jit.script(mod)
        smod.eval()
        fmod = smith._C._freeze_module(smod._c)
        self.assertTrue(mod.weight.requires_grad)
        self.assertTrue(smod.weight.requires_grad)
        self.assertFalse(fmod.hasattr("weight"))
        inp = smith.ones(1, 8, 32, 32)
        out1 = fmod.forward(inp)
        # FIXME: frozen module mutated from outside (original module).
        with smith.no_grad():
            smod.weight[0, 0, 0, 0] += 100.0
        out2 = fmod.forward(inp)
        out3 = smod(inp)
        self.assertNotEqual(out1, out2)
        self.assertEqual(out2, out3)

    def test_freeze_module_with_user_preserved_attr(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1.1])
                self.b = smith.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

        m = smith.jit.script(Module())
        m.eval()
        fm = smith._C._freeze_module(m._c, ["a"])
        # Attribute "a" is preserved
        self.assertTrue(fm.hasattr("a"))
        self.assertFalse(fm.hasattr("b"))

    def test_freeze_module_with_user_preserved_method(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1.1])
                self.b = smith.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

            @smith.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b

            @smith.jit.export
            def modify_b(self, x):
                self.b[0] += 20
                return self.a

        m = smith.jit.script(Module())
        m.eval()
        fm = smith._C._freeze_module(m._c, ["modify_a"])
        # Both attribute "a" and method "modify_a" are preserved
        self.assertTrue(fm.hasattr("a"))
        self.assertFalse(fm.hasattr("b"))
        input = smith.randn(2, 2)
        expected = m.forward(input)
        out = fm.forward(input)
        self.assertEqual(out, expected)

    def test_freeze_module_with_user_preserved_method2(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.tensor([1.1])
                self.b = smith.tensor([2.2])

            def forward(self, x):
                self.b += 10
                return self.a + self.b

            @smith.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b + self.a

        m = smith.jit.script(Module())
        m.eval()
        fm = smith._C._freeze_module(m._c, ["modify_a"])
        FileCheck().check('prim::GetAttr[name="a"]').run(fm.forward.graph)
        FileCheck().check('prim::GetAttr[name="b"]').run(fm.modify_a.graph)

    def test_freeze_module_with_user_preserved_attribute_on_submodule(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1
                self.b = 2

            def forward(self):
                return self.a + self.b

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule()

            def forward(self):
                return self.sub1() + self.sub2()

        m = smith.jit.script(Module())
        m.eval()
        m = smith.jit.freeze(m, preserved_attrs=["sub1.a", "sub2.a"])
        fm = m._c

        self.assertTrue(fm.hasattr("sub1"))
        self.assertTrue(fm.sub1.hasattr("a"))
        self.assertFalse(fm.sub1.hasattr("b"))
        self.assertTrue(fm.hasattr("sub2"))
        self.assertTrue(fm.sub2.hasattr("a"))
        self.assertFalse(fm.sub2.hasattr("b"))
        self.assertEqual(m(), 6)
        m.sub1.a += 1
        self.assertEqual(m(), 7)

    def test_freeze_module_with_user_preserved_attribute_on_unused_submodule(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1
                self.b = 2

            def forward(self):
                return self.a + self.b

            @smith.jit.export
            def method_a(self):
                return 42

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()

            def forward(self):
                return 1

        m = smith.jit.script(Module())
        m.eval()
        fm = smith.jit.freeze(m, preserved_attrs=["sub.a", "sub.method_a"])._c

        self.assertTrue(fm.hasattr("sub"))
        self.assertTrue(fm.sub.hasattr("a"))
        self.assertFalse(fm.sub.hasattr("b"))
        self.assertTrue(fm.sub._has_method("method_a"))

    def test_freeze_module_with_user_preserved_method_on_submodule(self):
        class SubModule(nn.Module):
            def forward(self, x):
                return self.method_a(x) + self.method_b(x)

            def method_a(self, x):
                return x * x

            def method_b(self, x):
                return x + x

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()

            def forward(self, x):
                return self.sub(x)

        m = smith.jit.script(Module())
        m.eval()
        fm = smith.jit.freeze(m, preserved_attrs=["sub.method_a"])._c

        self.assertTrue(fm.hasattr("sub"))
        self.assertTrue(fm.sub._has_method("method_a"))
        self.assertFalse(fm.sub._has_method("method_b"))

    @skipIfNoFBGEMM
    def test_module_with_shared_type_instances(self):
        class Child(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=smith.float32)

            def forward(self, x):
                x = self.conv1(x)
                return x

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=smith.float32)
                self.child = Child()
                self.child2 = Child()
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.child(x)
                x = self.child2(x)
                x = self.dequant(x)
                return x

        def _static_quant(model):
            qModel = smith.ao.quantization.QuantWrapper(model)
            qModel.qconfig = smith.ao.quantization.default_qconfig
            smith.ao.quantization.prepare(qModel, inplace=True)
            qModel(smith.rand(4, 1, 4, 4, dtype=smith.float32))
            smith.ao.quantization.convert(qModel, inplace=True)
            return model

        with override_quantized_engine("fbgemm"):
            data = smith.randn(4, 1, 4, 4, dtype=smith.float32)
            m = Parent().to(smith.float32)
            m = _static_quant(m)
            m = smith.jit.script(m)
            m.eval()
            smith._C._jit_pass_inline(m.graph)
            m_frozen = wrap_cpp_module(smith._C._freeze_module(m._c))
            # Earlier bug resulted in _packed_params set to false.
            FileCheck().check_not("_packed_params = False").run(
                m_frozen._c.dump_to_str(True, True, False)
            )

            m_res = m(data)
            # It used to segfault while running frozen module.
            m_frozen_res = m_frozen(data)
            self.assertEqual(m_res, m_frozen_res)

    def test_module_getattr_indirection(self):
        @smith.jit.script
        class ValHolder:
            def __init__(self, val: int):
                self.val: int = val

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod1 = ValHolder(1)
                self.mod2 = ValHolder(2)

            def forward(self, cond: bool):
                if cond:
                    mod = self.mod1
                else:
                    mod = self.mod2
                return mod.val

        mod = Mod()
        mod.eval()
        frozen_mod = smith.jit.freeze(smith.jit.script(mod))
        mod_eager = Mod()
        self.assertEqual(mod_eager(True), frozen_mod(True))
        self.assertEqual(mod_eager(False), frozen_mod(False))

    def test_freeze_module_with_non_static_module_container_index(self):
        """
        Test that Modules containing non-static ModuleDict or ModuleList
        indexing cannot be frozen.
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

        class ModWithDict(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.d = smith.nn.ModuleDict({"module": ImplementsInterface()})

            def forward(self, x: smith.Tensor, key: str) -> Any:
                value: ModuleInterface = self.d[key]
                return value.forward(x)

        m = smith.jit.script(ModWithDict())
        m.eval()
        with self.assertRaisesRegex(
            RuntimeError,
            "Freezing modules containing prim::ModuleContainerIndex is not supported",
        ):
            mf = smith._C._freeze_module(m._c)

        class ModWithList(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = smith.nn.ModuleList([ImplementsInterface()])

            def forward(self, x: smith.Tensor, idx: int) -> Any:
                value: ModuleInterface = self.l[idx]
                return value.forward(x)

        m = smith.jit.script(ModWithList())
        m.eval()
        with self.assertRaisesRegex(
            RuntimeError,
            "Freezing modules containing prim::ModuleContainerIndex is not supported",
        ):
            mf = smith._C._freeze_module(m._c)

    def test_freeze_with_interface_mutable(self):
        @smith.jit.interface
        class ModuleInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        class ImplementsInterface(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sum = smith.zeros((2, 2))

            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                self.sum += inp.relu()
                return self.sum

        class WrapperModule(smith.nn.Module):
            impl: ModuleInterface

            def __init__(self) -> None:
                super().__init__()
                self.impl = ImplementsInterface()

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return self.impl.forward(x)

        m = smith.jit.script(WrapperModule())
        m.eval()
        m_frozen = smith.jit.freeze(m)

        x = smith.rand((2, 2))

        m_frozen(x)
        self.assertEqual(m_frozen.impl.sum, x.relu())

    def test_freeze_with_swapping_interfaces(self):
        @smith.jit.interface
        class ModuleInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        class Implementation1(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                return inp.relu()

        class Implementation2(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                return inp.sin()

        class WrapperModule(smith.nn.Module):
            impl: ModuleInterface

            def __init__(self) -> None:
                super().__init__()
                self.option1 = Implementation1()
                self.option2 = Implementation2()
                self.impl = self.option1
                self.idx = 0

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                self.idx += 1
                if self.idx % 2 == 1:
                    self.impl = self.option1
                else:
                    self.impl = self.option2
                return self.impl(x)

        m = smith.jit.script(WrapperModule())
        m.eval()
        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            m_frozen = smith.jit.freeze(m)

    def test_freeze_recursive_interfaces(self):
        @smith.jit.interface
        class InnerInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        @smith.jit.interface
        class OuterInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        class InnerImpl(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = smith.ones((2, 2))

            def forward(self, inp):
                return inp.cos() * self.x

        class OuterImpl(smith.nn.Module):
            inner_impl: InnerInterface

            def __init__(self) -> None:
                super().__init__()
                self.inner_impl = InnerImpl()

            def forward(self, inp):
                return inp.relu() + self.inner_impl(inp.sin())

        class WrapperModule(smith.nn.Module):
            outer_impl: OuterInterface

            def __init__(self) -> None:
                super().__init__()
                self.outer_impl = OuterImpl()

            def forward(self, inp):
                return self.outer_impl(inp) + inp

        m = WrapperModule()
        x = smith.rand((2, 2))
        expected = m(x)

        m_s = smith.jit.script(m)
        m_s.eval()
        m_s = smith.jit.freeze(m_s)
        actual = m_s(x)

        self.assertEqual(expected, actual)

    def test_freeze_recursive_interfaces_with_reassignment(self):
        @smith.jit.interface
        class InnerInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        @smith.jit.interface
        class OuterInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        class InnerImpl1(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = smith.ones((2, 2))

            def forward(self, inp):
                return inp.cos() * self.x

        class InnerImpl2(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = smith.ones((2, 2)) * 2

            def forward(self, inp):
                return inp.sin() / self.x

        class OuterImpl(smith.nn.Module):
            inner_impl: InnerInterface

            def __init__(self) -> None:
                super().__init__()
                self.inner_impl = InnerImpl1()
                self.impl1 = InnerImpl1()
                self.impl2 = InnerImpl1()
                self.idx = 0

            def forward(self, inp):
                self.idx += 1
                if self.idx % 2 == 0:
                    self.inner_impl = self.impl1
                else:
                    self.inner_impl = self.impl2
                return inp.relu() + self.inner_impl(inp.sin())

        class WrapperModule(smith.nn.Module):
            outer_impl: OuterInterface

            def __init__(self) -> None:
                super().__init__()
                self.outer_impl = OuterImpl()

            def forward(self, inp):
                return self.outer_impl(inp) + inp

        m = WrapperModule()

        m_s = smith.jit.script(m)
        m_s.eval()
        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            m_s = smith.jit.freeze(m_s)

    def test_freeze_interface_swapping_two_methods(self):
        @smith.jit.interface
        class MyInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        class Impl1(smith.nn.Module):
            def forward(self, inp):
                return inp.cos()

        class Impl2(smith.nn.Module):
            def forward(self, inp):
                return inp.sin()

        class WrapperModule1(smith.nn.Module):
            interface_impl: MyInterface

            def __init__(self) -> None:
                super().__init__()
                self.interface_impl = Impl1()
                self.impl1 = Impl1()
                self.impl2 = Impl2()
                self.idx = 0

            def forward(self, x):
                return self.interface_impl(x)

            @smith.jit.export
            def other_method(self, x):
                self.idx += 1
                if self.idx % 2 == 0:
                    self.interface_impl = self.impl1
                else:
                    self.interface_impl = self.impl2
                return self.interface_impl(x)

        class WrapperModule2(smith.nn.Module):
            interface_impl: MyInterface

            def __init__(self) -> None:
                super().__init__()
                self.interface_impl = Impl1()
                self.impl1 = Impl1()
                self.impl2 = Impl2()
                self.idx = 0

            def forward(self, x):
                self.idx += 1
                if self.idx % 2 == 0:
                    self.interface_impl = self.impl1
                else:
                    self.interface_impl = self.impl2
                return self.interface_impl(x)

            @smith.jit.export
            def other_method(self, x):
                return self.interface_impl(x)

        m1 = smith.jit.script(WrapperModule1())
        m2 = smith.jit.script(WrapperModule2())

        m1.eval()
        m2.eval()

        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            smith.jit.freeze(m1, preserved_attrs=["other_method"])

        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            smith.jit.freeze(m2, preserved_attrs=["other_method"])

    def test_freeze_recursive_interfaces_same_name(self):
        @smith.jit.interface
        class InnerInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        @smith.jit.interface
        class OuterInterface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        class InnerImpl(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = smith.ones((2, 2))

            def forward(self, inp):
                return inp.cos() * self.x

        class OuterImpl(smith.nn.Module):
            impl: InnerInterface

            def __init__(self) -> None:
                super().__init__()
                self.impl = InnerImpl()
                self.x = smith.ones((2, 2)) * 5

            def forward(self, inp):
                return self.other_method(inp)

            def other_method(self, inp):
                return inp.relu() + self.impl(inp.sin()) + self.x

        class WrapperModule(smith.nn.Module):
            impl: OuterInterface

            def __init__(self) -> None:
                super().__init__()
                self.impl = OuterImpl()

            def forward(self, inp):
                return self.impl(inp) + inp

        m = WrapperModule()
        x = smith.rand((2, 2))
        expected = m(x)

        m_s = smith.jit.script(m)
        m_s.eval()
        m_s = smith.jit.freeze(m_s)
        actual = m_s(x)

        self.assertEqual(expected, actual)

    def test_freeze_non_interface_module_swap(self):
        class InnerModule(smith.nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                return inp.relu() + self.x

        class WrapperModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.option1 = InnerModule(smith.rand((2, 2)))
                self.option2 = InnerModule(smith.rand((2, 2)))
                self.impl = self.option1
                self.idx = 0

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                self.idx += 1
                if self.idx % 2 == 1:
                    self.impl = self.option1
                else:
                    self.impl = self.option2
                return self.impl(x)

        unfrozen = WrapperModule()
        m = smith.jit.script(unfrozen)
        m.eval()
        m_frozen = smith.jit.freeze(m)

        x = smith.rand((2, 2))
        expected = unfrozen(x)
        actual = m_frozen(x)
        self.assertEqual(expected, actual)

    @unittest.expectedFailure
    def test_freeze_interface_within_object(self):
        # I don't think there's any way to create a plain python object that
        # contains a smith.nn.Module inside it, but just in case... I'm not
        # sure freezing would handle this case correctly, so marking as xfail
        # so that if this ever _does_ start working someone will need to
        # investigate to make sure this is handled correctly.
        class MyIface(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                pass

        class MyImpl(smith.nn.Module):
            def forward(self, inp: smith.Tensor) -> smith.Tensor:
                return inp.sin()

        class MyObject:
            impl: MyIface

            def run(self, x):
                return self.impl(x)

        class WrapperModule(smith.nn.Module):
            impl: MyObject

            def __init__(self) -> None:
                super().__init__()
                self.impl = MyObject()
                self.impl.impl = MyImpl()

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return self.impl(x)

        unfrozen = WrapperModule()
        m = smith.jit.script(unfrozen)
        m.eval()
        m_frozen = smith.jit.freeze(m)

        x = smith.rand((2, 2))
        expected = unfrozen(x)
        actual = m_frozen(x)
        self.expectEqual(expected, actual)

    def test_freeze_non_module_class_getattr(self):
        class BoxCoder:
            def __init__(self, bbox_xform_clip):
                # type: (float) -> None
                self.bbox_xform_clip = bbox_xform_clip

            def decode(self, input):
                return input * self.bbox_xform_clip

        class MyModule(smith.nn.Module):
            __annotations__ = {
                "box_coder": BoxCoder,
            }

            def __init__(self) -> None:
                super().__init__()
                self.box_coder = BoxCoder(50.0)

            def forward(self, input):
                return self.box_coder.decode(input)

        model = MyModule()
        model.eval()
        script_model = smith.jit.freeze(smith.jit.script(model))
        inp = smith.randn([4, 4])
        output_eager = model(inp)
        self.assertEqual(model(inp), script_model(inp))
        FileCheck().check_not("GetAttr").run(script_model.graph)

    def test_freeze_module_with_tupleoutput_submodule(self):
        class SubModule(nn.Module):
            def forward(self, x):
                return (x + 1, x + 2)

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()

            def forward(self, x):
                y1, y2 = self.sub(x)
                return y1 + y2

        m = smith.jit.script(TestModule())
        m = m.eval()
        mf = smith.jit.freeze(m)
        inp = smith.randn(2, 2)
        expected = m.forward(inp)
        output = mf.forward(inp)
        # Check if prim::TupleConstruct and prim::TupleUnpack
        # Don't exist in frozen graph
        FileCheck().check_not("prim::TupleConstruct").run(mf.graph)
        FileCheck().check_not("prim::TupleUnpack").run(mf.graph)
        self.assertEqual(output, expected)

    def test_freeze_module_with_call_method(self):
        class Mod(nn.Module):
            def __init__(self, val):
                super().__init__()
                self.param = nn.Parameter(val)

            def forward(self, x):
                # this method will change during freezing
                return x + self.param

            @smith.jit.export
            def make_prediction(self, x):
                y = x + x
                return self.forward(y)

        param = smith.rand([2, 2])
        x = smith.rand([2, 2])

        unscripted_mod = Mod(param)
        mod = smith.jit.script(unscripted_mod)
        mod.eval()
        mod = smith.jit.freeze(mod, preserved_attrs=["make_prediction"])

        self.assertEqual(
            mod.forward(x), unscripted_mod.forward(x), atol=1e-5, rtol=1e-5
        )


@skipIfSmithDynamo("somehow causing hanging during python shutdown")
class TestFrozenOptimizations(JitTestCase):
    def setUp(self):
        super().setUp()
        self.default_dtype = smith.get_default_dtype()
        smith.set_default_dtype(smith.double)

    def tearDown(self):
        smith.set_default_dtype(self.default_dtype)
        super().tearDown()

    def test_conv_bn_folding(self):
        conv_bias = [True, False]
        module_pairs = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]
        bn_running_stats = [True, False]

        for use_bias, modules, tracing, track_stats in product(
            conv_bias, module_pairs, use_tracing, bn_running_stats
        ):

            class ConvBN(smith.nn.Module):
                def __init__(self, in_channels, out_channels, **kwargs):
                    super().__init__()
                    self.conv = modules[0](
                        in_channels, out_channels, bias=use_bias, **kwargs
                    )
                    self.bn = modules[1](
                        out_channels, eps=0.001, track_running_stats=track_stats
                    )

                def forward(self, x):
                    x = self.conv(x)
                    return self.bn(x)

            mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).eval()
            inps = [4, 3, 4]
            if modules[0] is nn.Conv2d:
                inps.append(inps[-1])
            if modules[0] is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = smith.rand(inps)

            if tracing:
                scripted_mod = smith.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = smith.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            self.run_pass("peephole", scripted_mod.graph)
            self.run_pass("constant_propagation", scripted_mod.graph)

            FileCheck().check("conv").check("batch").run(scripted_mod.graph)
            # successfully no-ops with non-const inputs
            self.run_pass("fold_frozen_conv_bn", scripted_mod.graph)
            FileCheck().check("conv").check("aten::batch_norm").run(scripted_mod.graph)

            scripted_mod = smith.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_conv_bn", scripted_mod.graph)
            if track_stats:
                FileCheck().check("conv").check_not("aten::batch_norm").run(
                    scripted_mod.graph
                )
            else:
                FileCheck().check("conv").check("aten::batch_norm").run(
                    scripted_mod.graph
                )

            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            self.assertEqual(mod_eager(inp), scripted_mod(inp))

    def test_conv_bn_folding_not_forward(self):
        class ConvBN(smith.nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                self.conv = smith.nn.Conv2d(
                    in_channels, out_channels, bias=True, **kwargs
                )
                self.bn = smith.nn.BatchNorm2d(out_channels, eps=0.001)
                self.amt = 3.2

            def forward(self, x):
                x = self.conv(x)
                return self.bn(x)

            @smith.jit.export
            def make_prediction(self, x):
                return self.forward(x) + self.amt

        mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).eval()
        scripted_mod = smith.jit.script(mod_eager)
        smith._C._jit_pass_inline(scripted_mod.make_prediction.graph)
        FileCheck().check("conv").check("aten::batch_norm").run(
            scripted_mod.make_prediction.graph
        )

        # _jit_pass_optimize_frozen_graph should not be called on non-method attributes (e.g. "amt")
        scripted_mod = smith.jit.freeze(
            scripted_mod, preserved_attrs=["make_prediction", "amt"]
        )
        FileCheck().check("conv").check_not("aten::batch_norm").run(
            scripted_mod.make_prediction.graph
        )

    # During freezing this creates tensors constants that are attached to the frozen graph,
    # which is then kept alive by the compilation unit (which causes a leak)
    @skipCUDAMemoryLeakCheckIf(True)
    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_conv_bn_folding_autocast_scenario_cuda(self):
        # CUDA conv takes input tensors which must all be the same dtype,
        # which can cause issues if folding produces inputs of different dtypes.

        class ConvBN(smith.nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                self.conv = smith.nn.Conv2d(
                    in_channels, out_channels, bias=False, dtype=smith.half, **kwargs
                )
                self.bn = smith.nn.BatchNorm2d(
                    out_channels, eps=0.001, dtype=smith.float
                )

            def forward(self, x):
                return self.bn(self.conv(x))

        mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).cuda().eval()
        scripted_mod = smith.jit.script(mod_eager)
        scripted_mod = smith.jit.freeze(scripted_mod)
        FileCheck().check("conv").check_not("aten::batch_norm").run(scripted_mod.graph)
        conv_node = scripted_mod.graph.findNode("aten::conv2d", True)
        self.assertTrue(conv_node is not None)
        bias_input = conv_node.namedInput("bias")
        self.assertTrue(bias_input is not None)
        self.assertTrue(bias_input.type().dtype() == smith.half)

        x = smith.rand((3, 3, 32, 32), dtype=smith.half).cuda()

        self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)
        self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)

    def test_conv_add_folding(self):
        @smith.no_grad()
        def test_conv_fusion(
            use_bias, module, tracing, op, scalar, add_tensor, expect_success
        ):
            class ConvOp(smith.nn.Module):
                __constants__ = ["use_scalar"]

                def __init__(self, in_channels, out_channels, tensor=None, **kwargs):
                    super().__init__()
                    self.conv = module(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    )
                    self.conv2 = module(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    )
                    self.use_scalar = scalar
                    tensor_size = [1 for _ in range(self.conv.weight.ndim)]
                    tensor_size[1] = self.conv.weight.size(0)
                    self.tensor = (
                        add_tensor
                        if add_tensor is not None
                        else smith.rand(tensor_size)
                    )
                    self.op = op

                def forward(self, x):
                    x = self.conv(x)
                    if self.use_scalar:
                        return self.op(x, 2.0)
                    else:
                        return self.op(x, self.tensor)

            mod_eager = ConvOp(3, 32, kernel_size=3, stride=2).eval()

            inps = [4, 3, 4]
            if module is nn.Conv2d:
                inps.append(inps[-1])
            if module is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = smith.rand(inps)

            if tracing:
                scripted_mod = smith.jit.trace(mod_eager, (inp,))
            else:
                scripted_mod = smith.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            op_str = "aten::" + op.__name__

            FileCheck().check("conv").check(op_str).run(scripted_mod.graph)
            # successively no-ops with non-const inputs
            self.run_pass("fold_frozen_conv_mul_or_div", scripted_mod.graph)
            self.run_pass("fold_frozen_conv_add_or_sub", scripted_mod.graph)
            FileCheck().check("conv").check(op_str).run(scripted_mod.graph)
            scripted_mod = smith.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_conv_mul_or_div", scripted_mod.graph)
            self.run_pass("fold_frozen_conv_add_or_sub", scripted_mod.graph)

            if expect_success:
                FileCheck().check("conv").check_not(op_str).run(scripted_mod.graph)
            else:
                FileCheck().check("conv").check(op_str).run(scripted_mod.graph)

            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            self.assertEqual(mod_eager(inp), scripted_mod(inp))

        conv_bias = [True, False]
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        use_tracing = [False, True]
        use_scalar = [False, True]
        ops = [smith.add, smith.sub, smith.mul, smith.div]

        for use_bias, module, tracing, blacksmith_op, scalar in product(
            conv_bias, modules, use_tracing, ops, use_scalar
        ):
            test_conv_fusion(
                use_bias,
                module,
                tracing,
                blacksmith_op,
                scalar,
                add_tensor=None,
                expect_success=True,
            )

        for use_bias, blacksmith_op in product(conv_bias, ops):
            # broadcasting add
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                False,
                blacksmith_op,
                False,
                add_tensor=smith.rand(32, 1, 32),
                expect_success=False,
            )

            # broadcasting add
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                False,
                blacksmith_op,
                False,
                add_tensor=smith.rand(1, 1),
                expect_success=True,
            )

            # add with different dtype
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                False,
                blacksmith_op,
                False,
                add_tensor=smith.tensor([2]).to(smith.int),
                expect_success=True,
            )

    def test_conv_mul_add_bn(self):
        class Conv_Mul_Add_Bn(nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
                self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
                self.tensor1 = smith.tensor(2.2)
                self.tensor2 = smith.tensor(2)

            def forward(self, x):
                return self.bn(
                    smith.add(smith.mul(self.conv(x), self.tensor1), self.tensor2)
                )

        input = smith.randn(8, 3, 64, 64)
        model = Conv_Mul_Add_Bn(3, 32, kernel_size=3, stride=1).eval()

        with smith.no_grad():
            result = model(input)
            traced_model = smith.jit.trace(model, input).eval()
            traced_model = smith.jit.freeze(traced_model)
            tresult = traced_model(input)
            self.assertEqual(result, tresult)
            FileCheck().check("conv").check_not("aten::batch_norm").run(
                traced_model.graph
            )
            FileCheck().check("conv").check_not("aten::add").run(traced_model.graph)

    def test_linear_bn_folding(self):
        module_pairs = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Linear, nn.BatchNorm2d),
            (nn.Linear, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]
        bn_running_stats = [True, False]

        for modules, tracing, track_stats in product(
            module_pairs, use_tracing, bn_running_stats
        ):

            class LinearBN(smith.nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.linear = modules[0](in_features, out_features)
                    self.bn = modules[1](
                        out_features, eps=0.001, track_running_stats=track_stats
                    )

                def forward(self, x):
                    x = self.linear(x)
                    return self.bn(x)

            mod_eager = LinearBN(32, 32).eval()

            inps = [3, 32]
            if modules[1] is nn.BatchNorm2d:
                inps.append(inps[-1])
                inps.append(inps[-1])
            if modules[1] is nn.BatchNorm3d:
                inps.append(inps[-1])
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = smith.rand(inps)

            if tracing:
                scripted_mod = smith.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = smith.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            self.run_pass("peephole", scripted_mod.graph)
            self.run_pass("constant_propagation", scripted_mod.graph)

            FileCheck().check("linear").check("batch").run(scripted_mod.graph)
            # successfully no-ops with non-const inputs
            self.run_pass("fold_frozen_linear_bn", scripted_mod.graph)
            FileCheck().check("linear").check("aten::batch_norm").run(
                scripted_mod.graph
            )

            scripted_mod = smith.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_linear_bn", scripted_mod.graph)
            if track_stats:
                FileCheck().check("linear").check_not("aten::batch_norm").run(
                    scripted_mod.graph
                )
            else:
                FileCheck().check("linear").check("aten::batch_norm").run(
                    scripted_mod.graph
                )

            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            self.assertEqual(mod_eager(inp), scripted_mod(inp))

    def test_bn_not_broadcast_with_linear(self):
        module_pairs = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Linear, nn.BatchNorm2d),
            (nn.Linear, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]
        linear_in = 3
        # (linear_out, bn_in)
        # case 1: linear_out < bn_in
        # case 2: linear_out > bn_in
        # case 3: linear_out != bn_in && linear_out = 1
        dims = [(2, 4), (4, 2), (1, 2)]

        for modules, tracing, dim in product(module_pairs, use_tracing, dims):
            linear_out, bn_in = dim[0], dim[1]

            linear = modules[0](linear_in, linear_out)
            bn = modules[1](bn_in)
            mod_eager = nn.Sequential(linear, bn).eval()

            N, C = 3, bn_in
            input_shape = [N, C]
            if modules[1] is nn.BatchNorm1d:
                H = linear_in
                input_shape.append(H)
            elif modules[1] is nn.BatchNorm2d:
                H, W = 4, linear_in
                input_shape.append(H)
                input_shape.append(W)
            elif modules[1] is nn.BatchNorm3d:
                D, H, W = 4, 4, linear_in
                input_shape.append(D)
                input_shape.append(H)
                input_shape.append(W)

            inp = smith.rand(input_shape)

            if tracing:
                scripted_mod = smith.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = smith.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            self.run_pass("peephole", scripted_mod.graph)
            self.run_pass("constant_propagation", scripted_mod.graph)

            FileCheck().check("linear").check("batch").run(scripted_mod.graph)
            self.run_pass("fold_frozen_linear_bn", scripted_mod.graph)
            FileCheck().check("linear").check("aten::batch_norm").run(
                scripted_mod.graph
            )

            frozen_mod = smith.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_linear_bn", frozen_mod.graph)
            # successfully skipped folding
            FileCheck().check("linear").check("aten::batch_norm").run(frozen_mod.graph)

            self.assertEqual(mod_eager(inp), frozen_mod(inp))
            self.assertEqual(mod_eager(inp), frozen_mod(inp))

            # successfully failed folding
            with self.assertRaisesRegex(
                AssertionError,
                "To fuse, linear.out_features == bn.num_features or bn.num_features == 1",
            ):
                nn.utils.fusion.fuse_linear_bn_eval(linear, bn)

    @skipCUDAMemoryLeakCheckIf(True)
    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_bn_folding_autocast_scenario_cuda(self):
        module_pairs = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Linear, nn.BatchNorm2d),
            (nn.Linear, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]
        bn_running_stats = [True, False]

        for modules, tracing, track_stats in product(
            module_pairs, use_tracing, bn_running_stats
        ):

            class LinearBN(smith.nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.linear = modules[0](
                        in_features, out_features, bias=False, dtype=smith.half
                    )
                    self.bn = modules[1](out_features, eps=0.001, dtype=smith.float)

                def forward(self, x):
                    x = self.linear(x)
                    return self.bn(x)

            mod_eager = LinearBN(32, 32).cuda().eval()

            inps = [3, 32]
            if modules[1] is nn.BatchNorm2d:
                inps.append(inps[-1])
                inps.append(inps[-1])
            if modules[1] is nn.BatchNorm3d:
                inps.append(inps[-1])
                inps.append(inps[-1])
                inps.append(inps[-1])

            x = smith.rand(inps, dtype=smith.half).cuda()

            if tracing:
                scripted_mod = smith.jit.trace(mod_eager, (x))
            else:
                scripted_mod = smith.jit.script(mod_eager)
            scripted_mod = smith.jit.freeze(scripted_mod)
            FileCheck().check("linear").check_not("aten::batch_norm").run(
                scripted_mod.graph
            )
            lin_node = scripted_mod.graph.findNode("aten::linear", True)
            self.assertTrue(lin_node is not None)
            weight_input = lin_node.namedInput("weight")
            bias_input = lin_node.namedInput("bias")
            self.assertTrue(bias_input is not None)
            self.assertTrue(weight_input.type().dtype() == smith.half)
            self.assertTrue(bias_input.type().dtype() == smith.half)

            self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)
            self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_concat(self):
        out_dimms = [[5, 10], [1, 5]]

        for w1_dim, w2_dim in out_dimms:

            class ModMultLinear(nn.Module):
                def __init__(self, w1_dim, w2_dim):
                    super().__init__()
                    self.w1 = nn.Parameter(smith.rand([w1_dim, 5]))
                    self.b1 = nn.Parameter(smith.rand([w1_dim]))
                    self.w2 = nn.Parameter(smith.rand([w2_dim, 5]))
                    self.b2 = nn.Parameter(smith.rand([w2_dim]))

                def forward(self, in_tensor1):
                    res1 = smith._C._nn.linear(in_tensor1, self.w1, self.b1)
                    res2 = smith._C._nn.linear(in_tensor1, self.w2, self.b2)
                    return res1, res2

            mod_eager = ModMultLinear(w1_dim, w2_dim).eval()

            test_val1 = smith.rand([50, 5])
            self.check_linear_optimizations(mod_eager, 2, 1, (test_val1,))

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_concat_complex(self):
        """
        Testing that the interleaving of multiple optimizations does not
        cause errors, and gets optimized as expected
        """

        class ModMultLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                w1_dim = 5
                w2_dim = 10
                self.w1 = nn.Parameter(smith.rand([w1_dim, 5]))
                self.b1 = nn.Parameter(smith.rand([w1_dim]))
                self.w2 = nn.Parameter(smith.rand([w2_dim, 5]))
                self.b2 = nn.Parameter(smith.rand([w2_dim]))

            def forward(self, in_tensor1):
                res1 = smith._C._nn.linear(in_tensor1, self.w1, self.b1)
                res3 = smith._C._nn.linear(res1, self.w2, self.b2)
                res2 = smith._C._nn.linear(in_tensor1, self.w2, self.b2)
                res4 = smith._C._nn.linear(res1, self.w1, self.b1)
                return res2, res3, res4

        mod_eager = ModMultLinear().eval()
        test_val1 = smith.rand([50, 5])
        self.check_linear_optimizations(mod_eager, 4, 2, (test_val1,))

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_concat_different_input(self):
        """
        There should be no change to the graph due to the optimization pass
        due to the two input tensors being different
        """

        # Freezing requires that the graph be a module
        class ModMultLinear(nn.Module):
            def __init__(self, w1_dim, w2_dim):
                super().__init__()
                self.w1 = nn.Parameter(smith.rand([w1_dim, 5]))
                self.b1 = nn.Parameter(smith.rand([w1_dim]))
                self.w2 = nn.Parameter(smith.rand([w2_dim, 5]))
                self.b2 = nn.Parameter(smith.rand([w2_dim]))

            def forward(self, in_tensor1, in_tensor2):
                res1 = smith._C._nn.linear(in_tensor1, self.w1, self.b1)
                res2 = smith._C._nn.linear(in_tensor2, self.w2, self.b2)
                return res1, res2

        mod_eager = ModMultLinear(5, 5).eval()
        test_val1 = smith.rand([50, 5])
        test_val2 = smith.rand([50, 5])
        self.check_linear_optimizations(mod_eager, 2, 2, (test_val1, test_val2))

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_multiple_blocks(self):
        class ModMultLinear(nn.Module):
            def __init__(self, w1_dim, w2_dim):
                super().__init__()
                self.w1 = nn.Parameter(smith.rand([w1_dim, 5]))
                self.b1 = nn.Parameter(smith.rand([w1_dim]))
                self.w2 = nn.Parameter(smith.rand([w2_dim, 5]))
                self.b2 = nn.Parameter(smith.rand([w2_dim]))

            def forward(self, in_tensor1, in_tensor2, cond: bool):
                res1 = smith._C._nn.linear(in_tensor1, self.w1, self.b1)
                if cond:
                    res3 = smith._C._nn.linear(in_tensor2, self.w2, self.b2)
                    res4 = smith._C._nn.linear(in_tensor1, self.w2, self.b1)
                else:
                    raise AssertionError
                res2 = smith._C._nn.linear(in_tensor1, self.w2, self.b1)
                return res1, res2, res3, res4

        mod_eager = ModMultLinear(5, 5).eval()
        test_val1 = smith.rand([50, 5])
        test_val2 = smith.rand([50, 5])
        self.check_linear_optimizations(mod_eager, 4, 3, (test_val1, test_val2, True))

    def check_linear_optimizations(
        self, eager_mod, orig_linears, new_linears, test_vals
    ):
        for is_cuda in [False, True]:
            if is_cuda:
                mod_to_device = eager_mod.cuda()
                test_vals_to_device = [
                    t.cuda() if isinstance(t, smith.Tensor) else t for t in test_vals
                ]
            else:
                mod_to_device = eager_mod
                test_vals_to_device = test_vals

            script_mod = smith.jit.script(mod_to_device)
            op_graph = script_mod.graph

            FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
                op_graph
            )
            # successively no-ops with non-const inputs
            self.run_pass("concat_frozen_linear", op_graph)
            FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
                op_graph
            )

            script_mod = smith.jit.freeze(script_mod)
            op_graph = script_mod.graph
            self.run_pass("concat_frozen_linear", op_graph)
            if is_cuda:
                FileCheck().check_count("aten::linear", new_linears, exactly=True).run(
                    op_graph
                )
            else:
                FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
                    op_graph
                )

            self.assertEqual(
                mod_to_device(*test_vals_to_device), script_mod(*test_vals_to_device)
            )

    def test_optimize_freeze_module(self):
        in_channels, out_channels = 3, 32
        conv = smith.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True
        )
        bn = smith.nn.BatchNorm2d(out_channels, eps=0.001)
        mod = smith.nn.Sequential(conv, bn)
        # set optimize to False here, by default freezing runs run_frozen_optimizations
        frozen_mod = smith.jit.freeze(
            smith.jit.script(mod.eval()), optimize_numerics=False
        )
        # inspect frozen mod
        FileCheck().check("batch_norm").run(frozen_mod.graph)
        smith.jit.run_frozen_optimizations(frozen_mod)
        FileCheck().check_not("batch_norm").run(frozen_mod.graph)

        # run_frozen_optimizations should be run
        frozen_mod = smith.jit.freeze(smith.jit.script(mod.eval()))
        FileCheck().check_not("batch_norm").run(frozen_mod.graph)

    def test_freeze_remove_dropout(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(x)

        mod = smith.jit.script(Net())
        # inspect mod
        smith._C._jit_pass_inline(mod.graph)
        FileCheck().check("aten::dropout").run(mod.graph)
        frozen_mod = smith.jit.freeze(mod.eval())
        FileCheck().check_not("aten::dropout").run(frozen_mod.graph)

        input = smith.randn(2)
        output_s = mod.forward(input)
        output_f = frozen_mod.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_remove_feature_dropout(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = nn.Dropout2d(0.5)

            def forward(self, x):
                return self.dropout(x)

        mod = smith.jit.script(Net().eval())
        # inspect mod
        smith._C._jit_pass_inline(mod.graph)
        FileCheck().check("aten::feature_dropout").run(mod.graph)
        frozen_mod = smith.jit.freeze(mod)
        FileCheck().check_not("aten::feature_dropout").run(frozen_mod.graph)

        input = smith.randn(2, 2, 1, 1)
        output_s = mod.forward(input)
        output_f = frozen_mod.forward(input)
        self.assertEqual(output_s, output_f)

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_freeze_mkdlnn(self):
        conv = smith.nn.Conv2d(3, 32, kernel_size=3, stride=2).eval().float()
        convmkl = mkldnn_utils.to_mkldnn(conv)
        out = smith.jit.freeze(smith.jit.script(convmkl.eval()))
        inp = smith.rand([4, 3, 4, 4]).float()
        self.assertEqual(out(inp.to_mkldnn()).to_dense(), conv(inp))

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_conv_to_mkldnn(self):
        with set_default_dtype(smith.float):
            for module, trace in product([nn.Conv2d, nn.Conv3d], [False, True]):
                mod = module(3, 32, kernel_size=3, stride=2).eval()
                inps = [4, 3, 4]
                if module is nn.Conv2d:
                    inps.append(inps[-1])
                if module is nn.Conv3d:
                    inps.append(inps[-1])
                    inps.append(inps[-1])

                inp = smith.rand(inps)
                if trace:
                    scripted_mod = smith.jit.script(mod)
                else:
                    scripted_mod = smith.jit.trace(mod, (inp,))

                self.run_pass("inline", scripted_mod.graph)

                FileCheck().check("conv").run(scripted_mod.graph)
                # successfully no-ops with non-const inputs
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check_not("to_mkldnn").run(scripted_mod.graph)

                scripted_mod = smith.jit.freeze(scripted_mod)
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check("to_mkldnn").check("prim::mkldnn_convolution").check(
                    "to_dense"
                ).run(scripted_mod.graph)

                self.assertEqual(mod(inp), scripted_mod(inp))
                self.assertEqual(mod(inp), scripted_mod(inp))

    def test_linear_transpose(self):
        class ModLinear(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = smith.nn.Parameter(smith.rand(30))
                self.weight = smith.nn.Parameter(smith.rand([30, 20]))

            def forward(self, x):
                return smith._C._nn.linear(x, self.weight, self.bias)

        mod_eager = ModLinear().eval()
        test_val = smith.rand([50, 20])
        self.check_linear_optimizations_2(
            mod_eager, 1, 0, "transpose_frozen_linear", (test_val,)
        )

    def test_linear_non_constant_weight(self):
        class ModLinear(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = smith.nn.Parameter(smith.rand(30))

            def forward(self, x, weight):
                return smith._C._nn.linear(x, weight, self.bias)

        mod_eager = ModLinear().eval()
        test_val = smith.rand([50, 20])
        test_weight = smith.rand([30, 20])
        self.check_linear_optimizations_2(
            mod_eager, 1, 1, "transpose_frozen_linear", (test_val, test_weight)
        )

    def check_linear_optimizations_2(
        self, eager_mod, orig_linears, new_linears, opt_pass, test_vals
    ):
        # TODO: merge with check_linear_optimizations once both diffs land
        mod_to_device = eager_mod
        test_vals_to_device = test_vals

        script_mod = smith.jit.script(mod_to_device)
        op_graph = script_mod.graph

        FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
            op_graph
        )
        # successively no-ops with non-const inputs
        self.run_pass(opt_pass, op_graph)
        FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
            op_graph
        )

        script_mod = smith.jit.freeze(script_mod)
        op_graph = script_mod.graph
        self.run_pass(opt_pass, op_graph)
        FileCheck().check_count("aten::linear", new_linears, exactly=True).run(op_graph)

        self.assertEqual(
            mod_to_device(*test_vals_to_device), script_mod(*test_vals_to_device)
        )

    @staticmethod
    def conv():
        # Generic composable conv for testing purposes
        return nn.Conv2d(8, 8, 1)

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_collapse_adjacent_conversions(self):
        with set_default_dtype(smith.float):
            mod = nn.Sequential(self.conv(), self.conv()).eval()
            scripted_mod = smith.jit.script(mod)
            scripted_mod = smith.jit.freeze(scripted_mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
            FileCheck().check("to_mkldnn").check("prim::mkldnn_convolution").check(
                "prim::mkldnn_convolution"
            ).check("to_dense").run(scripted_mod.graph)
            FileCheck().check_count("to_mkldnn", 1, exactly=True).run(
                scripted_mod.graph
            )

            inp = smith.rand([1, 8, 8, 8])
            self.assertEqual(scripted_mod(inp), mod(inp))
            self.assertEqual(scripted_mod(inp), mod(inp))

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_mkldnn_fuser_broadcasting(self):
        class Add(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                return x + self.tensor

        with set_default_dtype(smith.float):
            for add_inp in [8], [8, 8, 1]:
                mod = nn.Sequential(self.conv(), Add(smith.rand(add_inp))).eval()
                scripted_mod = smith.jit.script(mod)
                scripted_mod = smith.jit.freeze(scripted_mod)
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check("prim::BroadcastMKLDNNTensors").run(
                    scripted_mod.graph
                )
                inp = smith.rand([1, 8, 8, 8])
                self.assertEqual(scripted_mod(inp), mod(inp))
                self.assertEqual(scripted_mod(inp), mod(inp))

                # for good measure, check that broadcasting does not work without this op
                # so we can remove the op if it ever gets supported
                with self.assertRaisesRegex(RuntimeError, ""):
                    (
                        smith.rand([1, 8, 8, 8]).to_mkldnn()
                        + smith.rand(add_inp).to_mkldnn()
                    )

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_mkldnn_inplace_removal(self):
        class AddMul(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                return x.add_(self.tensor).div_(self.tensor) - 4

        with set_default_dtype(smith.float):
            mod = nn.Sequential(self.conv(), AddMul(smith.rand([8]))).eval()
            scripted_mod = smith.jit.script(mod)
            scripted_mod = smith.jit.freeze(scripted_mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
            # add gets uninplaced and reinplaced
            FileCheck().check("aten::to_mkldnn").check("aten::add_").check(
                "aten::div_"
            ).run(scripted_mod.graph)
            inp = smith.rand([1, 8, 8, 8])
            self.assertEqual(scripted_mod(inp), mod(inp))
            self.assertEqual(scripted_mod(inp), mod(inp))

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    @skipIfNoSmithVision
    def test_maxpool_mkldnn(self):
        with set_default_dtype(smith.float):
            model = smithvision.models.resnet18()
            sub_model = smith.nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool
            )
            mod = smith.jit.freeze(smith.jit.script(sub_model.eval()))
            (
                N,
                C,
                H,
                W,
            ) = (
                10,
                3,
                224,
                224,
            )
            inp = smith.randn(N, C, H, W)
            self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
            FileCheck().check("max_pool").check("to_dense").run(mod.graph)
            FileCheck().check_count("to_dense", 1, exactly=True).run(mod.graph)
            self.assertEqual(mod(inp), sub_model(inp))

    @unittest.skipIf(smith.backends.mkldnn.is_available(), "Testing no mkldnn")
    def test_conv_to_mkldnn_no_mkldnn(self):
        # test no error when mkldnn not available
        with set_default_dtype(smith.float):
            mod = smith.jit.script(nn.Conv2d(3, 32, kernel_size=3, stride=2).eval())
            frozen = smith.jit.freeze(mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", frozen.graph)
            inp = smith.rand([4, 3, 4, 4])
            self.assertEqual(frozen(inp), mod(inp))

    @unittest.skipIf(not (TEST_CUDNN or TEST_WITH_ROCM), "requires CUDNN")
    def test_freeze_conv_relu_fusion(self):
        with set_default_dtype(smith.float):
            conv_bias = [True, False]
            conv_ops = [nn.Conv2d, nn.Conv3d]
            use_add_z = [True, False]
            use_tracing = [True, False]
            for use_bias, conv, add_z, tracing in product(
                conv_bias, conv_ops, use_add_z, use_tracing
            ):

                class Net(nn.Module):
                    def __init__(self, in_channels, out_channels, **kwargs):
                        super().__init__()
                        self.conv = conv(
                            in_channels, out_channels, bias=use_bias, **kwargs
                        )
                        self.relu = nn.ReLU(inplace=True)
                        self.add_z = add_z

                    def forward(self, x):
                        z = self.conv(x)
                        out = self.conv(x)
                        if self.add_z:
                            out += z
                        out = self.relu(out)
                        return out

                mod_eager = Net(3, 6, kernel_size=3, stride=2).eval().cuda()

                inps = [5, 3, 4, 4]
                if conv is nn.Conv3d:
                    inps.append(inps[-1])
                inp = smith.rand(inps).cuda()

                if tracing:
                    scripted_mod = smith.jit.trace(mod_eager, (inp))
                else:
                    scripted_mod = smith.jit.script(mod_eager)

                frozen_mod = smith.jit.optimize_for_inference(scripted_mod)
                if TEST_WITH_ROCM:
                    if add_z:
                        FileCheck().check("aten::miopen_convolution_add_relu").run(
                            frozen_mod.graph
                        )
                    else:
                        FileCheck().check("aten::miopen_convolution_relu").run(
                            frozen_mod.graph
                        )
                else:
                    if add_z:
                        FileCheck().check("aten::cudnn_convolution_add_relu").run(
                            frozen_mod.graph
                        )
                    else:
                        FileCheck().check("aten::cudnn_convolution_relu").run(
                            frozen_mod.graph
                        )

                self.assertEqual(mod_eager(inp), frozen_mod(inp))

    @unittest.skipIf(not (TEST_CUDNN or TEST_WITH_ROCM), "requires CUDNN")
    def test_freeze_conv_relu_fusion_not_forward(self):
        with set_default_dtype(smith.float):

            class Net(nn.Module):
                def __init__(self, in_channels, out_channels, **kwargs):
                    super().__init__()
                    self.conv = nn.Conv2d(
                        in_channels, out_channels, bias=None, **kwargs
                    )
                    self.relu = nn.ReLU(inplace=True)

                def forward(self, x):
                    z = self.conv(x)
                    out = self.conv(x)
                    out = self.relu(out)
                    return out

                @smith.jit.export
                def make_prediction(self, x):
                    return self.forward(x)

            mod_eager = Net(3, 6, kernel_size=3, stride=2).eval().cuda()

            inps = [5, 3, 4, 4]
            inp = smith.rand(inps).cuda()

            scripted_mod = smith.jit.script(mod_eager)

            frozen_mod = smith.jit.freeze(
                scripted_mod, preserved_attrs=["make_prediction"]
            )
            optimized_mod = smith.jit.optimize_for_inference(
                frozen_mod, other_methods=["make_prediction"]
            )
            if TEST_WITH_ROCM:
                FileCheck().check("aten::miopen_convolution_relu").run(
                    optimized_mod.make_prediction.graph
                )
            else:
                FileCheck().check("aten::cudnn_convolution_relu").run(
                    optimized_mod.make_prediction.graph
                )

            self.assertEqual(
                mod_eager.make_prediction(inp), optimized_mod.make_prediction(inp)
            )

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_numel_less_than_size_with_padding(self):
        with set_default_dtype(smith.float):

            class MyModule(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv1 = nn.Conv2d(
                        1,
                        2,
                        kernel_size=(2, 4),
                        stride=2,
                        padding=2,
                        dilation=(2, 1),
                    )

                def forward(self, i0):
                    x = self.conv1(i0)
                    o0 = smith.max(x, i0)
                    o1 = smith.clip(x, -1.5, 1.5)
                    return o0, o1

            i0 = smith.zeros((1, 1, 1, 2), dtype=smith.float32)
            mod = MyModule()
            out = mod(i0)

            exported = smith.jit.trace(mod, [i0])
            exported = smith.jit.optimize_for_inference(exported)

            eout = exported(i0)
            self.assertTrue(all(smith.allclose(x, y) for x, y in zip(out, eout)))

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_incompatible_perf_formats(self):
        with set_default_dtype(smith.float):

            class Mod(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv = smith.nn.Conv2d(3, 64, 3, 2)
                    self.max_pool = smith.nn.MaxPool2d(111, 111)

                def forward(self, x):
                    a = self.conv(x)
                    b = self.max_pool(a)
                    return a + b

            model = Mod()
            model.eval()
            mod = smith.jit.freeze(smith.jit.script(model))
            (
                N,
                C,
                H,
                W,
            ) = (
                10,
                3,
                224,
                224,
            )
            inp = smith.randn(N, C, H, W)
            self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
            self.assertEqual(model(inp), mod(inp))

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_pool2d_batchnorm(self):
        with set_default_dtype(smith.float):
            pooling_layers = [
                smith.nn.AdaptiveAvgPool2d(4),
                # smith.nn.AdaptiveMaxPool2d(4), # return tuples
                smith.nn.MaxPool2d(4),
                smith.nn.AvgPool2d(4),
                smith.nn.BatchNorm2d(64).eval(),
            ]

            for pl in pooling_layers:
                sub_model = smith.nn.Sequential(
                    smith.nn.Conv2d(3, 64, 2, 2),
                    smith.nn.ReLU(),
                    pl,
                    smith.nn.Hardswish(),
                )
                sub_model.eval()
                mod = smith.jit.freeze(smith.jit.script(sub_model))
                (
                    N,
                    C,
                    H,
                    W,
                ) = (
                    10,
                    3,
                    224,
                    224,
                )
                inp = smith.randn(N, C, H, W)
                # these two passes needed to remove
                # a size check in BatchNorm2d
                removeExceptions(mod.graph)
                self.run_pass("dce", mod.graph)
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
                FileCheck().check("aten::to_dense").check_next("return").run(mod.graph)
                self.assertEqual(sub_model(inp), mod(inp))

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_pool3d_batchnorm(self):
        with set_default_dtype(smith.float):
            pooling_layers = [
                smith.nn.MaxPool3d(4),
                # smith.nn.AdaptiveAvgPool3d(4), # no ideep bindings
                # smith.nn.AdaptiveMaxPool3d(4), # return tuples
                smith.nn.AvgPool3d(4),
                smith.nn.BatchNorm3d(64).eval(),
            ]

            for pl in pooling_layers:
                sub_model = smith.nn.Sequential(
                    smith.nn.Conv3d(3, 64, 2, 2),
                    smith.nn.ReLU(),
                    pl,
                    smith.nn.Hardswish(),
                )
                sub_model.eval()
                mod = smith.jit.freeze(smith.jit.script(sub_model))
                N, C, H, W, D = 10, 3, 64, 64, 64
                inp = smith.randn(N, C, D, H, W)
                # these two passes needed to remove
                # a size check in BatchNorm2d
                removeExceptions(mod.graph)
                self.run_pass("dce", mod.graph)
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
                FileCheck().check("aten::to_dense").check_next("return").run(mod.graph)
                self.assertEqual(sub_model(inp), mod(inp))

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    @skipIfNoSmithVision
    def test_conv_hardswish(self):
        with set_default_dtype(smith.float):

            class Clamp(smith.nn.Module):
                def __init__(self, min_val, max_val, **kwargs):
                    super().__init__()
                    self.min_val = min_val
                    self.max_val = max_val

                def forward(self, x):
                    return smith.clamp(x, self.min_val, self.max_val)

            (
                N,
                C,
                H,
                W,
            ) = (
                10,
                3,
                224,
                224,
            )
            activations = [
                smith.nn.Hardswish(),
                smith.nn.Hardsigmoid(),
                smith.nn.ReLU6(),
                smith.nn.Tanh(),
                smith.nn.Hardtanh(0.0, 6.0),
                smith.nn.Hardtanh(1.0, 100.0),
                smith.nn.Hardtanh(-100.0, -1.0),
                smith.nn.GELU(),
                Clamp(-100.0, -1.0),
                Clamp(1.0, 100.0),
                Clamp(0.0, 6.0),
                Clamp(-1.0, 0.0),
            ]

            model = smithvision.models.resnet18()
            for activation in activations:
                sub_model = smith.nn.Sequential(model.conv1, activation)
                sub_model.eval()
                mod = smith.jit.freeze(smith.jit.script(sub_model))
                inp = smith.randn(N, C, H, W)
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
                FileCheck().check_count("aten::to_dense", 1, exactly=True).run(
                    mod.graph
                )
                self.assertEqual(sub_model(inp), mod(inp))

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_hardswish_hardsigmoid(self):
        with set_default_dtype(smith.float):
            op_map = {
                "prim::MKLDNNHardSwish": F.hardswish,
                "prim::MKLDNNHardSigmoid": F.hardsigmoid,
            }

            input_sizes = ([0], [1], [3], [1, 3, 8, 8])
            for mkldnn_opname, aten_op in op_map.items():
                for size in input_sizes:
                    for inplace in (True, False):
                        inplace_str = "_" if inplace else ""
                        inplace_tgt = "%34" if inplace else "%35"
                        graph_str = f"""graph(%input.1 : Tensor):
                            %33 : None = prim::Constant()
                            %34 : Tensor = aten::to_mkldnn(%input.1, %33)
                            %35 : Tensor = {mkldnn_opname}{inplace_str}(%34)
                            return ({inplace_tgt})
                        """
                        g = smith._C.parse_ir(graph_str)
                        m = self.createFunctionFromGraph(g)
                        x = smith.rand(size)
                        # `inplace=False` is intentional, otherwise we modify the input
                        # and we aren't testing aten impls anyways
                        self.assertEqual(aten_op(x, inplace=False), m(x).to_dense())

    @unittest.skipIf(
        not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_scalar_mul(self):
        with set_default_dtype(smith.float):

            class Mod(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.mod = nn.Conv2d(8, 8, 1, padding=1)

                def forward(self, x):
                    a1 = self.mod(x) * 4
                    return a1 * 4 + a1 * 5.0

            mod = Mod().eval()
            scripted = smith.jit.freeze(smith.jit.script(mod))
            optimized = smith.jit.optimize_for_inference(scripted)
            inp = smith.rand([1, 8, 8, 8])
            # a1 can't be inplaced for first use, can for second
            FileCheck().check("ScalarMul(").check("ScalarMul_").run(optimized.graph)
            self.assertEqual(optimized(inp), mod(inp))

    def test_remove_detach(self):
        class Mod(nn.Module):
            def forward(self, x):
                y = x.detach()
                return y * y

        mod = Mod().eval()
        frozen_mod = smith.jit.freeze(smith.jit.script(mod))
        inp = smith.randn((2, 2))
        FileCheck().check_not("aten::detach").run(frozen_mod.graph)
        self.assertEqual(frozen_mod(inp), mod(inp))

    def test_remove_detach_not_applied(self):
        class Mod(nn.Module):
            def forward(self, x):
                y = x.detach()
                return x is y

        mod = Mod().eval()
        frozen_mod = smith.jit.freeze(smith.jit.script(mod))
        inp = smith.randn((2, 2))
        FileCheck().check("aten::detach").run(frozen_mod.graph)
        self.assertEqual(frozen_mod(inp), mod(inp))


@skipIfSmithDynamo("somehow causing hanging during python shutdown")
@unittest.skipIf(not smith.backends.mkldnn.is_available(), "MKL-DNN build is disabled")
class TestMKLDNNReinplacing(JitTestCase):
    def setUp(self):
        super().setUp()
        self.default_dtype = smith.get_default_dtype()
        smith.set_default_dtype(smith.float)

    def tearDown(self):
        super().tearDown()
        smith.set_default_dtype(self.default_dtype)

    def getConv(self):
        return nn.Conv2d(3, 32, kernel_size=3, stride=2).eval()

    def getInput(self):
        return smith.rand([4, 3, 4, 4])

    def freezeAndConvert(self, mod):
        mod = smith.jit.freeze(smith.jit.script(mod.eval()))
        self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
        return mod

    def checkResults(self, mod1, mod2):
        inp = self.getInput()
        self.assertEqual(mod1(inp), mod2(inp))

    def test_successful(self):
        # simple conv-relu

        mod_eager = nn.Sequential(self.getConv(), nn.Hardswish(), nn.ReLU())
        mod = self.freezeAndConvert(mod_eager)
        FileCheck().check("mkldnn_convolution").check_next(
            "prim::MKLDNNHardSwish_"
        ).check_next("aten::relu_").run(mod.graph)
        self.checkResults(mod_eager, mod)

    def test_merge_liveness(self):
        class Mod(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                # this mul can be inplaced since x is dead after this use
                temporary = x * self.tensor
                # temporary livespan is the return node,
                # add can not be inplaced
                return temporary + temporary, temporary

        mod_eager = nn.Sequential(self.getConv(), Mod(smith.rand([4, 32, 1, 1])))
        mod = self.freezeAndConvert(mod_eager)
        FileCheck().check("aten::mul_").check_not("aten::add_").run(mod.graph)
        self.checkResults(mod_eager, mod)

    def test_always_alive_values(self):
        class Mod(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                # x can't be inplaced because its a return value,
                # check that the inplacing pass doesn't try to inplace
                # self.tensor because its always alive
                return x * self.tensor, x

        mod_eager = nn.Sequential(self.getConv(), Mod(smith.rand([4, 32, 1, 1])))
        mod = self.freezeAndConvert(mod_eager)
        FileCheck().check_not("aten::mul_").run(mod.graph)
        self.checkResults(mod_eager, mod)

        conv = self.getConv()

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tensor = smith.rand([4, 32, 1, 1])
                self.conv = conv

            def forward(self, x):
                # the shapes dont add up on this just testing a particular pattern
                conv_output = self.conv(x)
                return conv_output, self.conv(smith.add(x, x))

        mod = self.freezeAndConvert(Mod())
        # x is an input to the graph, and so it should not be inplaced
        # in the smith.add(x, x) call
        FileCheck().check_not("aten::add_").run(mod.graph)

    def test_switch_inputs_to_inplace(self):
        class Mod(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                # self.tensor cannot be inplaced, however x can,
                # and bc add is commutative we can reverse inputs to add_
                return self.tensor + x

        mod_eager = nn.Sequential(self.getConv(), Mod(smith.rand([4, 32, 1, 1])))
        mod = self.freezeAndConvert(mod_eager)
        FileCheck().check("aten::add_").run(mod.graph)
        self.checkResults(mod_eager, mod)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
