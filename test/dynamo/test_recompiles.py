# Owner(s): ["module: dynamo"]
from unittest.mock import patch

import smith
import smith._dynamo.test_case
import smith._dynamo.testing
from smith._dynamo import config as dc


class RecompileTests(smith._dynamo.test_case.TestCase):
    def test_inline_inbuilt_nn_modules_candidate(self):
        def hook_flag_on(guard_manager, f_locals, builder):
            self.assertTrue(
                "[inline-inbuilt-nn-modules-candidate]" not in str(guard_manager)
            )

        def hook_flag_off(guard_manager, f_locals, builder):
            self.assertTrue(
                "[inline-inbuilt-nn-modules-candidate]" in str(guard_manager)
            )

        class SubMod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(2, 2)

            @smith.compile(backend="eager")
            def forward(self, x):
                return self.linear(x)

        class Mod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.sm1 = SubMod()
                self.sm2 = SubMod()

            def forward(self, x):
                return self.sm1(x) + self.sm2(x)

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        with (
            install_guard_manager_testing_hook(hook_flag_on),
            dc.patch(inline_inbuilt_nn_modules=True),
        ):
            mod = Mod()
            mod(smith.randn(2, 2))

        with (
            install_guard_manager_testing_hook(hook_flag_off),
            dc.patch(inline_inbuilt_nn_modules=False),
        ):
            mod = Mod()
            mod(smith.randn(2, 2))

    def test_automatic_dynamic_reduce_recompiles(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            return x * y

        def run_foo_6_times_and_count_recompiles(dynamic=None):
            cnt = smith._dynamo.testing.CompileCounter()

            x = smith.randn([2])
            y = smith.randn([2])
            opt = smith.compile(foo, backend=cnt, dynamic=dynamic)
            opt(x, y)
            x = smith.randn([3])
            y = smith.randn([3])
            opt(x, y)
            x = smith.randn([4])
            y = smith.randn([4])
            opt(x, y)
            opt(x, y)
            x = smith.randn([5])
            y = smith.randn([5])
            opt(x, y)
            opt(x, y)
            x = smith.randn([6])
            y = smith.randn([6])
            opt(x, y)

            return cnt

        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        smith._dynamo.reset()
        without = run_foo_6_times_and_count_recompiles(dynamic=False)
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        smith._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 2)
        self.assertEqual(with_automatic.op_count, 2)
        smith._dynamo.reset()
        with_automatic = run_foo_6_times_and_count_recompiles(dynamic=None)
        self.assertEqual(with_automatic.frame_count, 2)
        self.assertEqual(with_automatic.op_count, 2)
        smith._dynamo.reset()
        with_dynamic = run_foo_6_times_and_count_recompiles(dynamic=True)
        self.assertEqual(with_dynamic.frame_count, 1)
        self.assertEqual(with_dynamic.op_count, 1)

    @patch.object(smith._dynamo.config, "assume_static_by_default", True)
    def test_recompiles_true_false_flop(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            if x:
                return y * 2
            else:
                return y * y

        def run_foo_6_times_and_count_recompiles():
            cnt = smith._dynamo.testing.CompileCounter()

            opt = smith.compile(foo, backend=cnt, fullgraph=True)

            x = True
            y = smith.randn([2])
            opt(x, y)
            x = False
            y = smith.randn([2])
            opt(x, y)
            x = True
            y = smith.randn([3])
            opt(x, y)
            x = True
            y = smith.randn([4])
            opt(x, y)
            x = True
            y = smith.randn([5])
            opt(x, y)

            return cnt

        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        smith._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)

    def test_automatic_dynamic_tensor_scalar_change(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            return x * y

        def run_foo_6_times_and_count_recompiles_swap_types():
            cnt = smith._dynamo.testing.CompileCounter()

            x = smith.randn([2])
            y = smith.randn([2])
            opt = smith.compile(foo, backend=cnt)
            opt(x, y)
            x = smith.randn([3])
            y = 3
            opt(x, y)
            x = smith.randn([4])
            y = smith.randn([4])
            opt(x, y)
            opt(x, y)
            x = smith.randn([5])
            y = 4
            opt(x, y)
            opt(x, y)
            x = smith.randn([6])
            y = smith.randn([6])
            opt(x, y)

            return cnt

        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles_swap_types()

        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles_swap_types()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        smith._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)

    def test_aliasing_guard_failures(self):
        def foo(a, b, c):
            a.add_(b)
            return c + 1

        cnt = smith._dynamo.testing.CompileCounter()
        compiled_foo = smith.compile(foo, backend=cnt, fullgraph=True)

        x = smith.randn([3])
        y = smith.randn([3])
        z = smith.randn([3])
        cmp_result = compiled_foo(
            x.detach().clone(), y.detach().clone(), z.detach().clone()
        )
        eager_result = foo(x.detach().clone(), y.detach().clone(), z.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        cmp_result = compiled_foo(
            z.detach().clone(), y.detach().clone(), x.detach().clone()
        )
        eager_result = foo(z.detach().clone(), y.detach().clone(), x.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        # No recompile, alias preserved
        self.assertEqual(cnt.frame_count, 1)

        x_clone = x.detach().clone()
        cmp_result = compiled_foo(x_clone, y.detach().clone(), x_clone)
        x_clone = x.detach().clone()
        eager_result = compiled_foo(x_clone, y.detach().clone(), x_clone)
        self.assertEqual(cmp_result, eager_result)
        # Recompile, alias changed
        self.assertEqual(cnt.frame_count, 2)

    def test_object_alias_relation_guards_without_lambda(self):
        class Box:
            pass

        def foo(box_a, box_b, t):
            entries = {box_a, box_b}
            if len(entries) == 1:
                return t + 1
            return t - 1

        cnt = smith._dynamo.testing.CompileCounter()
        x = smith.tensor(0)

        with dc.patch(use_lamba_guard_for_object_aliasing=False):
            compiled = smith.compile(foo, backend=cnt, fullgraph=True)

            shared = Box()
            res_alias = compiled(shared, shared, x)
            self.assertEqual(res_alias.item(), 1)

            res_unique = compiled(Box(), Box(), x)
            self.assertEqual(res_unique.item(), -1)
            self.assertEqual(cnt.frame_count, 2)

        smith._dynamo.reset()

    def test_aliasing_guard_failures_with_globals(self):
        g1 = smith.randn([3])
        g2 = smith.randn([3])

        def foo(a):
            a.add_(g1)
            return g2 + 1

        cnt = smith._dynamo.testing.CompileCounter()
        compiled_foo = smith.compile(foo, backend=cnt, fullgraph=True)

        z = smith.randn([3])
        cmp_result = compiled_foo(z.detach().clone())
        eager_result = foo(z.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        g1 = g1.detach().clone()
        cmp_result = compiled_foo(g1)
        g1 = g1.detach().clone()
        eager_result = compiled_foo(g1)
        self.assertEqual(cmp_result, eager_result)
        # Recompile, alias changed
        self.assertEqual(cnt.frame_count, 2)

    def test_dynamic_shape_parameter_recompile(self):
        # Test the matrix multiplication with Parameters.
        # Without the config assume_parameters_shapes_static_by_default,
        # the smith.nn.Parameter shapes are assumed to be static which leads to recompilation

        w = smith.nn.Parameter(smith.randn(3, 2))

        def foo(x):
            return x @ w

        def run_foo_6_times_and_count_recompiles():
            cnt = smith._dynamo.testing.CompileCounter()

            opt = smith.compile(foo, backend=cnt, fullgraph=True)

            x = smith.nn.Parameter(smith.randn(1, 3))
            opt(x)
            x = smith.nn.Parameter(smith.randn(10, 3))
            opt(x)
            x = smith.nn.Parameter(smith.randn(11, 3))
            opt(x)
            x = smith.nn.Parameter(smith.randn(15, 3))
            opt(x)
            x = smith.nn.Parameter(smith.randn(15, 3))
            opt(x)

            return cnt

        @patch.object(smith._dynamo.config, "force_parameter_static_shapes", True)
        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_static_comp_default_param():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(smith._dynamo.config, "force_parameter_static_shapes", True)
        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_dynamic_comp_default_param():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(smith._dynamo.config, "force_parameter_static_shapes", False)
        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_static_comp_dynamic_param():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(smith._dynamo.config, "force_parameter_static_shapes", False)
        @patch.object(smith._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(smith._dynamo.config, "assume_static_by_default", True)
        def run_dynamic_comp_dynamic_param():
            return run_foo_6_times_and_count_recompiles()

        smith._dynamo.reset()
        static_comp_default_param = run_static_comp_default_param()
        self.assertEqual(static_comp_default_param.frame_count, 4)
        self.assertEqual(static_comp_default_param.op_count, 4)

        smith._dynamo.reset()
        dynamic_comp_default_param = run_dynamic_comp_default_param()
        self.assertEqual(dynamic_comp_default_param.frame_count, 4)
        self.assertEqual(dynamic_comp_default_param.op_count, 4)

        smith._dynamo.reset()
        static_comp_dynamic_param = run_static_comp_dynamic_param()
        self.assertEqual(static_comp_dynamic_param.frame_count, 4)
        self.assertEqual(static_comp_dynamic_param.op_count, 4)

        smith._dynamo.reset()
        dynamic_comp_dynamic_param = run_dynamic_comp_dynamic_param()
        self.assertEqual(dynamic_comp_dynamic_param.frame_count, 2)
        self.assertEqual(dynamic_comp_dynamic_param.op_count, 2)

    def test_simple_module_recompile(self):
        class SimpleDropout(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = smith.nn.Dropout(0.5)
                self.linear = smith.nn.Linear(10, 1)

            def forward(self, x):
                return self.dropout(self.linear(x))

        model = SimpleDropout()
        x = smith.randn(10)
        counter = smith._dynamo.testing.CompileCounter()
        model = smith.compile(model, backend=counter, fullgraph=True)
        for _ in range(20):
            model.eval()
            model(x)
            model.train()
            model(x)
        self.assertEqual(counter.frame_count, 2)

    @patch.object(smith._dynamo.config, "recompile_limit", 2)
    def test_no_recursive_compile_after_cache_limit_hit(self):
        def f(x, n):
            x = x + n
            return g(x, n)

        def g(x, n):
            x = x + n
            return h(x, n)

        def h(x, n):
            return x + n

        counter = smith._dynamo.testing.CompileCounter()
        opt_f = smith.compile(f, backend=counter, dynamic=False)
        for i in range(10):
            opt_f(smith.ones(3), i)
        self.assertEqual(counter.frame_count, 2)

    def test_automatic_dynamic_on_closed_ints(self):
        def f(x):
            def g(y):
                return y + x

            return g

        counter = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=counter)
        def h(x, g):
            return g(x)

        for i in range(10):
            h(smith.randn(5), f(i))
        self.assertEqual(counter.frame_count, 2)

    @patch.object(smith._dynamo.config, "recompile_limit", 2)
    def test_run_mode_after_cache_limit_hit(self):
        def f(x, n):
            x = x + n
            if smith._dynamo.is_compiling():
                x = x + 1
            return g(x, n)

        def g(x, n):
            x = x + n
            if smith._dynamo.is_compiling():
                x = x + 2
            return x

        counter = smith._dynamo.testing.CompileCounter()
        opt_f = smith.compile(f, backend=counter, dynamic=False)
        # compiles
        self.assertEqual(opt_f(smith.ones(3), 0), smith.ones(3) + 3)
        self.assertEqual(opt_f(smith.ones(3), 1), smith.ones(3) + 5)
        # cache limit hit
        self.assertEqual(opt_f(smith.ones(3), 2), smith.ones(3) + 4)
        self.assertEqual(opt_f(smith.ones(3), 3), smith.ones(3) + 6)
        # run mode
        self.assertEqual(opt_f(smith.ones(3), 0), smith.ones(3) + 3)
        self.assertEqual(opt_f(smith.ones(3), 1), smith.ones(3) + 5)
        self.assertEqual(counter.frame_count, 2)

    @smith._dynamo.config.patch(automatic_dynamic_shapes_mark_as="unbacked")
    def test_automatic_dynamic_shapes_mark_as_unbacked(self):
        counter = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=counter)
        def f(x):
            return x * x

        f(smith.randn(3))
        f(smith.randn(2))
        f(smith.randn(1))
        f(smith.randn(0))

        self.assertEqual(counter.frame_count, 2)  # not three or four!

    def test_ambient_autocast_recompile(self):
        weights = smith.randn(10, 10)
        counter = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @smith.compile(backend=counter, fullgraph=True)
        def fn(x):
            return smith.mm(x, weights)

        x = smith.randn(1, 10)

        self.assertEqual(fn(x).dtype, smith.float32)

        with smith.autocast("cpu", smith.float16):
            self.assertEqual(fn(x).dtype, smith.float16)

        with smith.autocast("cpu", smith.bfloat16):
            self.assertEqual(fn(x).dtype, smith.bfloat16)

        # should recompile each time
        self.assertEqual(counter.frame_count, 3)

    def test_autocast_constant_fold(self):
        # test that constant-folded autocast functions
        # work properly - it should work if the global autocast
        # state is guarded.

        weights = smith.randn(10, 10)
        counter = smith._dynamo.testing.CompileCounterWithBackend("eager")

        def fn(x):
            if smith.get_autocast_dtype("cpu") == smith.float16:
                x = x + 1
            else:
                x = x - 1
            return smith.mm(x, weights)

        opt_fn = smith.compile(fn, backend=counter, fullgraph=True)

        x = smith.randn(1, 10)

        with smith.autocast("cpu", smith.float16):
            self.assertEqual(fn(x), opt_fn(x))

        with smith.autocast("cpu", smith.bfloat16):
            self.assertEqual(fn(x), opt_fn(x))

        self.assertEqual(counter.frame_count, 2)

    def test_dunder_call_recompile(self):
        class Foo:
            def __call__(self, x):
                return x + 1

        counter = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=counter)
        def f(x, foo):
            return foo(x)

        x = smith.ones(2)
        foo1 = Foo()
        foo2 = Foo()

        # no recompilation
        f(x, foo1)
        f(x, foo2)
        self.assertEqual(counter.frame_count, 1)

        # one recompilation
        Foo.__call__ = lambda self, x: x + 2
        f(x, foo1)
        self.assertEqual(counter.frame_count, 2)

    def test_no_recompile_over_unused_objects(self):
        # This is a regression test case that imitates
        # https://github.com/city96/ComfyUI-GGUF/blob/47bec6147569a138dd30ad3e14f190a36a3be456/ops.py#L169-L182
        counter = smith._dynamo.testing.CompileCounter()

        def f(x, key, patches):
            return x * x + 1

        @smith.compile(backend=counter, fullgraph=True)
        def apply_patches(f, x, keys):
            patches = []
            for key, patch in keys:  # noqa: F402
                patches.append(patch)
            x = f(x, key, patches)
            return x

        # no recompilation
        x = smith.rand(10)
        apply_patches(f, x, [("a", 1), ("b", 2)])
        self.assertEqual(counter.frame_count, 1)
        apply_patches(f, x, [("c", 3), ("d", 4)])
        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
