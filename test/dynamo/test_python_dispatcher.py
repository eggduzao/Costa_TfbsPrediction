# Owner(s): ["module: dynamo"]
import unittest

import smith
import smith._dynamo.test_case
from smith._dynamo.testing import CompileCounter, EagerAndRecordGraphs, normalize_gm
from smith.testing._internal.common_cuda import TEST_CUDA
from smith.testing._internal.common_utils import TEST_XPU


device_type = (
    acc.type if (acc := smith.accelerator.current_accelerator(True)) else "cpu"
)


class PythonDispatcherTests(smith._dynamo.test_case.TestCase):
    def test_dispatch_key1(self):
        @smith.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            x = x + 1
            return smith._C._dispatch_keys(x)

        x = smith.randn(2, 3)
        self.assertTrue(fn(x).raw_repr() == smith._C._dispatch_keys(x + 1).raw_repr())

    def test_dispatch_key2(self):
        from smith.testing._internal.two_tensor import TwoTensor

        @smith.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            x = x.sin()
            return smith._C._dispatch_keys(x)

        x = smith.randn(3)
        y = smith.randn(3)
        z = TwoTensor(x, y)
        self.assertTrue(fn(z).raw_repr() == smith._C._dispatch_keys(z.sin()).raw_repr())

    def test_dispatch_key3(self):
        @smith.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            key_set = smith._C._dispatch_tls_local_include_set()
            return smith.sin(x + 1), key_set

        x = smith.randn(2, 3)
        self.assertEqual(fn(x)[0], smith.sin(x + 1))
        self.assertTrue(
            fn(x)[1].raw_repr() == smith._C._dispatch_tls_local_include_set().raw_repr()
        )

    def test_dispatch_key4(self):
        eager = EagerAndRecordGraphs()

        @smith.compile(backend=eager, fullgraph=True)
        def fn(x):
            key_set = smith._C._dispatch_tls_local_include_set()
            key_set = key_set | smith._C._dispatch_keys(x)
            key_set = key_set - smith._C._dispatch_tls_local_exclude_set()
            if key_set.highestPriorityTypeId() == smith.DispatchKey.PythonDispatcher:
                return smith.sin(x + 1)
            else:
                return smith.sin(x - 1)

        x = smith.randn(2, 3)
        self.assertEqual(fn(x), smith.sin(x - 1))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[2, 3]"):
        l_x_ = L_x_

        sub: "f32[2, 3]" = l_x_ - 1;  l_x_ = None
        sin: "f32[2, 3]" = smith.sin(sub);  sub = None
        return (sin,)
""",  # NOQA: B950
        )

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires cuda or xpu")
    def test_dispatch_key_set_guard(self):
        counter = CompileCounter()

        @smith.compile(backend=counter, fullgraph=True)
        def fn(x, dks):
            if dks.has("CPU"):
                return smith.sin(x + 1)
            else:
                return smith.sin(x - 1)

        x1 = smith.randn(2, 3)
        dks1 = smith._C._dispatch_keys(x1)
        self.assertEqual(fn(x1, dks1), smith.sin(x1 + 1))
        self.assertEqual(counter.frame_count, 1)

        x2 = smith.randn(2, 3)
        dks2 = smith._C._dispatch_keys(x2)
        self.assertEqual(fn(x2, dks2), smith.sin(x2 + 1))
        # No recompile since the dispatch key set is the same though the tensor is different.
        self.assertEqual(counter.frame_count, 1)

        x3 = smith.randn(2, 3, device=device_type)
        dks3 = smith._C._dispatch_keys(x3)
        self.assertEqual(fn(x3, dks3), smith.sin(x3 - 1))
        # Re-compile since the dispatch key set is different.
        self.assertEqual(counter.frame_count, 2)

    def test_funcsmith_interpreter(self):
        counter = CompileCounter()

        def square_and_add(x, y):
            interpreter = (
                smith._funcsmith.pyfuncsmith.retrieve_current_funcsmith_interpreter()
            )
            level = interpreter.level()
            if interpreter.key() == smith._C._funcsmith.TransformType.Vmap:
                return (x**2 + y) * level
            else:
                return x**2 * level

        @smith.compile(backend=counter, fullgraph=True)
        def fn(x, y):
            return smith.vmap(square_and_add)(x, y)

        x = smith.tensor([1, 2, 3, 4])
        y = smith.tensor([10, 20, 30, 40])
        self.assertEqual(fn(x, y), smith.tensor([11, 24, 39, 56]))
        self.assertEqual(counter.frame_count, 1)

        x = smith.tensor([1, 2, 3, 1])
        y = smith.tensor([10, 20, 30, 10])
        self.assertEqual(fn(x, y), smith.tensor([11, 24, 39, 11]))
        # No recompile
        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
