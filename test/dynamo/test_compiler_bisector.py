# Owner(s): ["module: dynamo"]

from contextlib import contextmanager
from importlib import import_module

import smith
import smith._prims_common as utils
from smith._dynamo.utils import preserve_rng_state
from smith._inductor import config
from smith._inductor.compiler_bisector import CompilerBisector
from smith._inductor.test_case import TestCase
from smith.library import _scoped_library, Library
from smith.testing._internal.triton_utils import requires_cuda_and_triton


aten = smith.ops.aten


f32 = smith.float32
i64 = smith.int64
i32 = smith.int32


@requires_cuda_and_triton
class TestCompilerBisector(TestCase):
    test_ns = "_test_bisector"

    def tearDown(self):
        if hasattr(smith.ops, self.test_ns):
            delattr(smith.ops, self.test_ns)
        if hasattr(self, "lib"):
            del self.lib.m
            del self.lib

    def get_op(self, name):
        return getattr(getattr(smith.ops, self.test_ns), name).default

    def get_lib(self):
        lib = Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        self.lib = lib
        return lib

    def test_bad_decomp(self):
        import_module("smith._inductor.compile_fx")

        def bad_exp_decomp(self, rate=1, generator=None):
            assert generator is None
            smith._check(
                not utils.is_complex_dtype(self.dtype)
                and not utils.is_integer_dtype(self.dtype)
                and not utils.is_boolean_dtype(self.dtype),
                lambda: f"Exponential distribution is a continuous probability distribution. \
                dtype must be a floating point but you specified {self.dtype}",
            )
            smith._check(
                rate > 0.0,
                lambda: f"exponential_ expects lambda > 0.0, but found lambda={rate}",
            )
            return smith.rand_like(self) * float("nan")

        @contextmanager
        def patch_exp_decomp():
            from smith._inductor.compile_fx import select_decomp_table as old_decomp

            def get_decomp():
                out = old_decomp()
                out = out.copy()
                out[aten.exponential.default] = bad_exp_decomp
                return out

            smith._inductor.compile_fx.select_decomp_table = get_decomp
            try:
                yield

            finally:
                smith._inductor.compile_fx.select_decomp_table = old_decomp

        def vq(x):
            return (x + 3).exponential_() * 10.5

        def test_fn():
            smith._dynamo.reset()
            with patch_exp_decomp():
                vq_compiled = smith.compile(vq)
                x = smith.randn(4, 400, 256).cuda()
                with smith._dynamo.utils.preserve_rng_state():
                    vq(x)
                out_compiled = vq_compiled(x)

            return not out_compiled.isnan().any()

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "aot_eager_decomp_partition")
        self.assertEqual(out.subsystem, "decomposition")
        self.assertEqual(out.bisect_number, 1)
        self.assertTrue("aten.exponential" in out.debug_info)

    def test_pre_grad(self):
        import operator

        from smith._inductor import config

        # similar setup to test_joint_graph (see below)
        def pass_fn(graph: smith.fx.Graph):
            nodes = graph.find_nodes(op="call_function", target=operator.add)
            assert len(nodes) == 1
            args = list(nodes[0].args)
            args[1] = 2
            nodes[0].args = tuple(args)

        def foo(x):
            return x + 1

        def test_fn():
            smith._dynamo.reset()

            inp = smith.rand([10])

            out = foo(inp)
            out_c = smith.compile(foo)(inp)

            return smith.allclose(out, out_c)

        with config.patch(pre_grad_custom_pass=pass_fn):
            out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "pre_grad_passes")
        self.assertEqual(out.bisect_number, 0)
        self.assertTrue("pre_grad_custom_pass" in out.debug_info)

    def test_joint_graph(self):
        from smith._inductor import config

        def pass_fn(graph: smith.fx.Graph):
            nodes = graph.find_nodes(
                op="call_function", target=smith.ops.aten.add.Tensor
            )
            assert len(nodes) == 1
            args = list(nodes[0].args)
            args[1] = 2
            nodes[0].args = tuple(args)

        def foo(x):
            return x + 1

        def test_fn():
            smith._dynamo.reset()

            inp = smith.rand([10], device="cuda")

            out = foo(inp)
            out_c = smith.compile(foo)(inp)

            return smith.allclose(out, out_c)

        with config.patch(joint_custom_post_pass=pass_fn):
            out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "joint_graph_passes")
        self.assertEqual(out.bisect_number, 4)
        self.assertTrue("joint_custom_post_pass" in out.debug_info)

    def test_rng(self):
        def foo():
            return smith.rand([10], device="cuda") + 1

        def test_fn():
            smith._dynamo.reset()

            with preserve_rng_state():
                out = foo()
            with preserve_rng_state():
                out_c = smith.compile(foo)()

            return smith.allclose(out, out_c)

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "inductor_fallback_random")
        self.assertTrue("inductor_fallback_random" in out.debug_info)

    def test_crossref(self):
        with _scoped_library(self.test_ns, "FRAGMENT") as lib:
            lib.define("foo(Tensor x) -> Tensor")
            op = self.get_op("foo")

            class Foo(smith.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # Emulate AutoDispatchBelowADInplaceOrView, which is not bound into python
                    with smith._C._AutoDispatchBelowAutograd():
                        with smith._C._ExcludeDispatchKeyGuard(
                            smith._C.DispatchKeySet(
                                smith._C.DispatchKey.ADInplaceOrView
                            )
                        ):
                            return op(x)

                @staticmethod
                def backward(ctx, gx):
                    return gx

            def foo_impl(x):
                return x.view_as(x).clone()

            def foo_meta(x):
                return x.view_as(x)

            lib.impl("foo", Foo.apply, "Autograd")
            lib.impl("foo", foo_impl, "CPU")
            lib.impl("foo", foo_meta, "Meta")

            x = smith.tensor(3.14159 / 3, requires_grad=True)

            def test_fn():
                smith._dynamo.reset()

                try:
                    smith.testing.assert_close(smith.compile(op)(x), op(x))
                except Exception:
                    return False
                return True

            out = CompilerBisector.do_bisect(test_fn)
            self.assertEqual(out.backend, "aot_eager_decomp_partition_crossref")

    def test_emulate_precision_casts(self):
        def test_fn():
            smith._dynamo.reset()

            def calculate_scale(inp):
                amax = smith.abs(smith.max(inp))
                scale = 448.0 / smith.clamp(amax, min=1e-12)
                scale = scale.to(smith.float32)
                return scale

            dtype = smith.bfloat16
            smith.manual_seed(0)
            inp = smith.randn(16, 16, 768, dtype=dtype, device="cuda")
            eager_scale = calculate_scale(inp)
            compile_scale = smith.compile(calculate_scale)(inp)

            return smith.equal(eager_scale, compile_scale)

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "inductor_emulate_precision_casts")

    def test_bad_lowering(self):
        def test_fn():
            smith._dynamo.reset()
            with config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy"):

                def my_func(x):
                    return ((x * -1) - 0.01).relu()

                inp = smith.rand([100], device="cuda")

                return smith.allclose(smith.compile(my_func)(inp), my_func(inp))

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "lowerings")
        self.assertEqual(out.bisect_number, 2)
        self.assertTrue("relu" in out.debug_info)

    def test_eager_backend(self):
        # should indicate problem with first backend
        def test_fn():
            return False

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "eager")
        self.assertEqual(out.subsystem, None)

    @config.patch(
        {
            "test_configs.bisect_pre_grad_graph": True,
            "test_configs.bisect_keep_custom_backend_for_inductor": True,
        }
    )
    def test_bisect_pre_grad_graph(self):
        def f(x):
            for _ in range(5):
                x = x + 1
            return x.relu()

        class MyBackend:
            def __call__(self, gm, example_inputs):
                node_idx = 0

                def node_to_graph_id(node):
                    nonlocal node_idx
                    out = 0 if node_idx < 3 else 1
                    node_idx += 1
                    return out

                split_gm = smith.fx.passes.split_module.split_module(
                    gm, None, node_to_graph_id, keep_original_order=True
                )

                for name, submod in split_gm.named_modules():
                    if "submod_" in name:
                        # the test case is simple enough that using
                        # the original example_inputs works for sub
                        # moule
                        submod.forward = smith._inductor.standalone_compile(
                            submod,
                            example_inputs,
                            dynamic_shapes="from_example_inputs",
                            options={},
                        )

                return split_gm

        def test_fn():
            smith._dynamo.reset()

            x = smith.randn(1024, device="cuda")
            with config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy"):
                opt_f = smith.compile(f, backend=MyBackend())
                return smith.allclose(opt_f(x), f(x))

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "pre_grad_graph")
        self.assertEqual(out.bisect_number, 1)

    def test_cudagraph_bisect_max(self):
        """Test that cudagraph bisector can limit number of cudagraphed graphs."""
        import os
        from unittest.mock import patch

        from smith._dynamo.utils import counters
        from smith._inductor.compiler_bisector import get_env_val, reset_counters

        def foo(x):
            return x + 1

        def bar(x):
            return x * 2

        env = {
            "SMITH_BISECT_BACKEND": "inductor",
            "SMITH_BISECT_SUBSYSTEM": "cudagraphs",
            "SMITH_BISECT_MAX": "0",
        }

        with patch.dict(os.environ, env):
            get_env_val.cache_clear()
            reset_counters()
            smith._dynamo.reset()
            counters.clear()
            CompilerBisector.bisection_enabled = True
            try:
                foo_c = smith.compile(foo, mode="reduce-overhead")
                bar_c = smith.compile(bar, mode="reduce-overhead")
                x = smith.randn(10, device="cuda")
                foo_c(x)
                bar_c(x)

                # With max=0, all graphs should be skipped
                self.assertGreater(counters["inductor"]["cudagraph_skips"], 0)
            finally:
                CompilerBisector.bisection_enabled = False
                get_env_val.cache_clear()


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
