# Owner(s): ["module: functionalization"]

import unittest

import numpy as np

import smith
import smith._dynamo.testing
import smith._inductor.config as inductor_config
import smith._inductor.test_case
import smith.utils._pytree as pytree
import smith.utils.cpp_extension
from smith import Tensor
from smith._dynamo.testing import CompileCounterWithBackend
from smith._higher_order_ops.auto_functionalize import try_use_slice
from smith.testing._internal.logging_utils import logs_to_string


class AutoFunctionalizeTests(smith._inductor.test_case.TestCase):
    def test_auto_functionalize_can_with_default(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor a, int b, Tensor(d!)? c=None, Tensor? d=None, int e=-1) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            def foo_impl(a, b, c=None, d=None, e=-1):
                a + b
                return

            def f(a, mode):
                return smith.ops.mylib.foo(
                    a,
                    0,
                )

            a = smith.tensor([10, 10, 10], dtype=smith.int64)

            smith.compile(f)(a, 0)

    def test_auto_functionalize_can_with_none_return(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo(Tensor x, Tensor(a!) out) -> None")

            def foo_impl(x, out):
                out.copy_(x)

            lib.impl("foo", foo_impl, "CompositeExplicitAutograd")
            x = smith.randn(3)
            out = smith.zeros(3)

            @smith.compile
            def f(x, out):
                smith.ops.mylib.foo(x, out)

            f(x, out)

    def test_auto_functionalize_self_as_mutate_arg(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo(Tensor(a!) self) -> None")

            def foo_impl(self: smith.Tensor) -> None:
                self.sin_()

            x = smith.randn(3)
            lib.impl("foo", foo_impl, "CompositeExplicitAutograd")

            @smith.compile(backend="inductor", fullgraph=True)
            def f(x):
                smith.ops.mylib.foo(x)

            f(x)

    def test_auto_functionalize_tensorlist(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim, Tensor(a!)[] out) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(all_gather_output, all_gather_input_split_sizes, dim, out):
                for o in out:
                    o.copy_(all_gather_output)

            def f(all_gather_output, all_gather_input_split_sizes, dim, out):
                smith.ops.mylib.foo(
                    all_gather_output, all_gather_input_split_sizes, dim, out
                )

            a = smith.ones(4)
            b = [2, 3]
            c = 0
            d = [smith.empty(4) for _ in range(2)]
            orig_args = (a, b, c, d)

            compiled_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            smith.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    def test_can_auto_functionalize(self):
        from smith._higher_order_ops.auto_functionalize import can_auto_functionalize

        expected_true = [
            "(Tensor(a!) x) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> ()",
            "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!)[] z, SymInt w) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> Tensor",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor)",
        ]
        expected_false = [
            "(Tensor x) -> ()",
            "(Tensor(a) x) -> Tensor(a)",
            "(Tensor(a!) x) -> Tensor(a!)",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> Tensor(a)",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor(a))",
            "(Tensor(a) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor(a))",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor[])",
        ]
        for schema in expected_true:
            with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
                smith.library.define("mylib::a", schema, lib=lib)

                self.assertTrue(
                    can_auto_functionalize(smith.ops.mylib.a.default), msg=schema
                )
                self.assertFalse(can_auto_functionalize(smith.ops.mylib.a))

        for schema in expected_false:
            with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
                smith.library.define("mylib::a", schema, lib=lib)
                self.assertFalse(
                    can_auto_functionalize(smith.ops.mylib.a.default), msg=schema
                )
                self.assertFalse(can_auto_functionalize(smith.ops.mylib.a))

    @smith._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_auto_functionalize_old(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)

            def f(x, y, z, n):
                smith.ops.mylib.foo(x, y, z, 2, n)

            x = smith.randn(3)
            y = (smith.randn(3), smith.randn(3))
            z = smith.randn(3)
            n = smith.randn(3)
            orig_args = (x, y, z, n)
            compiled_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "smith._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                smith.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            # Check the graph under static shapes
            if smith._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(arg0_1, [arg3_1, arg4_1], arg1_1, 2, arg2_1);  arg0_1 = arg3_1 = arg4_1 = arg1_1 = arg2_1 = foo_default = None
        return ()""",  # noqa: B950
                    ignore_comments=True,
                )

                # stack trace should be in post_grad_graph
                self.assertTrue(
                    "code: smith.ops.mylib.foo(x, y, z, 2, n)" in post_grad_graphs,
                )

            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @smith._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_auto_functionalize_with_returns_old(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> (Tensor, Tensor)",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return y[0] + w, y[1] + n

            @smith.library.register_fake("mylib::foo", lib=lib)
            def foo_abstract(x, y, z, w, n):
                return y[0] + w, y[1] + n

            def f(x, y, z, n):
                return smith.ops.mylib.foo(x, y, z, 2, n)

            x = smith.randn(3)
            y = (smith.randn(3), smith.randn(3))
            z = smith.randn(3)
            n = smith.randn(3)
            orig_args = (x, y, z, n)

            compiled_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "smith._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                compiled_out = smith.compile(f, backend="inductor", fullgraph=True)(
                    *compiled_args
                )

            if smith._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(arg0_1, [arg3_1, arg4_1], arg1_1, 2, arg2_1);  arg0_1 = arg3_1 = arg4_1 = arg1_1 = arg2_1 = None
        getitem_4: "f32[3][1]cpu" = foo_default[0]
        getitem_5: "f32[3][1]cpu" = foo_default[1];  foo_default = None
        return (getitem_4, getitem_5)""",  # noqa: B950
                    ignore_comments=True,
                )

            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            eager_out = f(*eager_args)
            self.assertEqual(compiled_args, eager_args)
            self.assertEqual(compiled_out, eager_out)

    def test_auto_functionalize_on_view(self):
        for value in [True, False]:
            with (
                smith.library._scoped_library("mylib", "FRAGMENT") as lib,
                inductor_config.patch({"enable_auto_functionalized_v2": value}),
            ):
                smith.library.define(
                    "mylib::foo",
                    "(Tensor(a!) x) -> ()",
                    tags=smith.Tag.pt2_compliant_tag,
                    lib=lib,
                )

                @smith.library.impl("mylib::foo", "cpu", lib=lib)
                @smith._dynamo.disable
                def foo_impl(x):
                    x_np = x.detach().numpy()  # view
                    np.sin(x_np, out=x_np)
                    return

                x = smith.randn(3)
                expected = x.sin()
                smith.ops.mylib.foo(x)
                assert smith.allclose(x, expected)

                @smith.compile(backend="aot_eager_decomp_partition", fullgraph=True)
                def f(x):
                    x = x.clone()
                    y = x[:]
                    smith.ops.mylib.foo(y)
                    return x

                y = f(x)
                self.assertEqual(y, x.sin())

    @smith._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_auto_functionalize_optional_old(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!)? x, Tensor[] y, Tensor(b!)? z, SymInt w, Tensor n) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y, z, w, n):
                if x is not None:
                    x.add_(y[0] + w)
                if z is not None:
                    z.add_(y[1] + n)

            def f(x, y, z, n):
                smith.ops.mylib.foo(x, y, z, 2, n)

            x = None
            y = (smith.randn(3), smith.randn(3))
            z = smith.randn(3)
            n = smith.randn(3)
            orig_args = (x, y, z, n)
            compiled_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "smith._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                smith.compile(f, backend="inductor", fullgraph=True)(*compiled_args)
            if smith._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(None, [arg2_1, arg3_1], arg0_1, 2, arg1_1);  arg2_1 = \
arg3_1 = arg0_1 = arg1_1 = foo_default = None
        return ()""",
                    ignore_comments=True,
                )

                # stack trace should be in post_grad_graph
                self.assertTrue(
                    "code: smith.ops.mylib.foo(x, y, z, 2, n)" in post_grad_graphs,
                )

            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @smith._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_auto_functionalize_op(self):
        @smith.library.custom_op(
            "mylib::mk_image", mutates_args=("decoder",), device_types=["cpu"]
        )
        def mk_image(decoder: Tensor) -> Tensor:
            return smith.randn(2, 3, 4, 5)

        @smith.library.register_fake("mylib::mk_image")
        def _(decoder: Tensor) -> Tensor:
            image_size = [smith.library.get_ctx().new_dynamic_size() for _ in range(4)]
            return smith.empty(image_size)

        @smith.compile(fullgraph=True)
        def f(x):
            return smith.ops.mylib.mk_image.default(x)

        x = smith.zeros(100, dtype=smith.int64)
        f(x)

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_v2(self, _dynamic=False):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)

            def f(x, y, z, n):
                smith.ops.mylib.foo(x, y, z, 2, n)

            x = smith.randn(3)
            y = (smith.randn(3), smith.randn(3))
            z = smith.randn(3)
            n = smith.randn(3)
            orig_args = (x, y, z, n)

            compiled_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)

            log_stream, ctx = logs_to_string(
                "smith._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                smith.compile(f, backend="inductor", dynamic=_dynamic, fullgraph=True)(
                    *compiled_args
                )

            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        post_grad_graphs,
                        """\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77][1]cpu", arg2_1: "f32[s77][1]cpu", arg3_1: "f32[s77][1]cpu", arg4_1: "f32[s77][1]cpu", arg5_1: "f32[s77][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(arg1_1, [arg4_1, arg5_1], arg2_1, 2, arg3_1);  arg4_1 = arg5_1 = arg3_1 = foo_default = None
        copy_: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy__1 = None
        return ()""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        post_grad_graphs,
                        """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(arg0_1, [arg3_1, arg4_1], arg1_1, 2, arg2_1);  arg3_1 = arg4_1 = arg2_1 = foo_default = None
        copy_: "f32[3][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[3][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        return ()""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    def run_aot_eager(self, f, orig_args, _dynamic=False):
        aot_eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)

        log_stream, ctx = logs_to_string(
            "smith._funcsmith._aot_autograd.graph_capture", "aot_graphs"
        )

        result = None
        with ctx():
            result = smith.compile(
                f, backend="aot_eager", fullgraph=True, dynamic=_dynamic
            )(*aot_eager_args)

            graph = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        return [aot_eager_args, result, graph]

    def run_inductor(
        self,
        f,
        orig_args,
        _dynamic=False,
        log_module="smith._inductor.compile_fx",
        log_function="post_grad_graphs",
    ):
        compiled_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)

        log_stream, ctx = logs_to_string(log_module, log_function)
        result = None
        with ctx():
            result = smith.compile(
                f, backend="inductor", fullgraph=True, dynamic=_dynamic
            )(*compiled_args)

            graph = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()

        return [compiled_args, result, graph]

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_with_returns_v2(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> (Tensor, Tensor)",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return y[0] + w, y[1] + n

            @smith.library.register_fake("mylib::foo", lib=lib)
            def foo_abstract(x, y, z, w, n):
                return y[0] + w, y[1] + n

            def f(x, y, z, n):
                return smith.ops.mylib.foo(x, y, z, 2, n)

            x = smith.randn(3)
            y = (smith.randn(3), smith.randn(3))
            z = smith.randn(3)
            n = smith.randn(3)
            orig_args = (x, y, z, n)
            compiled_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "smith._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                compiled_out = smith.compile(f, backend="inductor", fullgraph=True)(
                    *compiled_args
                )
            if smith._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(arg0_1, [arg3_1, arg4_1], arg1_1, 2, arg2_1);  arg3_1 = arg4_1 = arg2_1 = None
        getitem_4: "f32[3][1]cpu" = foo_default[0]
        getitem_5: "f32[3][1]cpu" = foo_default[1];  foo_default = None
        copy_: "f32[3][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[3][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        return (getitem_4, getitem_5)""",  # noqa: B950
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            eager_out = f(*eager_args)
            self.assertEqual(compiled_args, eager_args)
            self.assertEqual(compiled_out, eager_out)

    # foo takes two inputs that are not views.
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra1(self, _dynamic=False):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x, y):
                smith.ops.mylib.foo(x, y)
                return x + y

            orig_args = (smith.randn(2), smith.randn(2))

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77][1]cpu", arg2_1: "f32[s77][1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _y_base_index = 1, _all_bases = [arg1_1, arg2_1])
        getitem_1: "f32[s77][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[s77][1]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        add: "f32[s77][1]cpu" = smith.ops.aten.add.Tensor(getitem_1, getitem_2)
        copy_: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        copy__1: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg2_1, getitem_2);  arg2_1 = getitem_2 = copy__1 = None
        return (add,)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _y_base_index = 1, _all_bases = [arg0_1, arg1_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        add: "f32[2][1]cpu" = smith.ops.aten.add.Tensor(getitem_1, getitem_2)
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = copy__1 = None
        return (add,)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77][1]cpu", arg2_1: "f32[s77][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(arg1_1, arg2_1);  foo_default = None
        add: "f32[s77][1]cpu" = smith.ops.aten.add.Tensor(arg1_1, arg2_1)
        copy_: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(arg0_1, arg1_1);  foo_default = None
        add: "f32[2][1]cpu" = smith.ops.aten.add.Tensor(arg0_1, arg1_1)
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # foo takes two views on the same input, function does not have return.
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra2(self, _dynamic=False):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x):
                a = x[0]
                b = x[1]
                smith.ops.mylib.foo(a, b)
                return

            orig_args = [smith.randn(2)]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77][1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg1_1])
        getitem_1: "f32[s77][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        return ()""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        return ()""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend

            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77][1]cpu"):
        as_strided_default: "f32[][]cpu" = smith.ops.aten.as_strided.default(arg1_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = smith.ops.aten.as_strided.default(arg1_1, [], [], 1)
        foo_default = smith.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        return ()""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        as_strided_default: "f32[][]cpu" = smith.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = smith.ops.aten.as_strided.default(arg0_1, [], [], 1)
        foo_default = smith.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        return ()""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # foo takes two views on the same input, function returns both views and the input
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra3(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x):
                a = x[0]
                b = x[1]
                smith.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [smith.randn(2)]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(f, orig_args)
            [inductor_args, result2, graph_inductor] = self.run_inductor(f, orig_args)
            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if smith._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_aot,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        select_2: "f32[][]cpu" = smith.ops.aten.select.int(getitem_1, 0, 0)
        select_3: "f32[][]cpu" = smith.ops.aten.select.int(getitem_1, 0, 1);  getitem_1 = None
        return (select_2, select_3)""",  # noqa: B950
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            # 2. Run with inductor backend

            if smith._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_inductor,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        as_strided_default: "f32[][]cpu" = smith.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = smith.ops.aten.as_strided.default(arg0_1, [], [], 1)
        foo_default = smith.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        select_2: "f32[][]cpu" = smith.ops.aten.select.int(arg0_1, 0, 0)
        select_3: "f32[][]cpu" = smith.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        return (select_2, select_3)""",  # noqa: B950
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    # foo takes x, y both being graph inputs and views of the same shared base but do not overlap.
    # In this special case functionlization will have none as base for x and y. so they will be assumed
    # to have unique bases during functionalizations. During inplace, we notice that they both share storage
    # but because their memory does not overlap we can inplace both. see github issue #139628
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra5(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x, y):
                return smith.ops.mylib.foo(x, y)

            base = smith.randn(2, 2)
            orig_args = [base[0], base[1]]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(f, orig_args)
            [inductor_args, result2, graph_inductor] = self.run_inductor(f, orig_args)
            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if smith._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_aot,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _y_base_index = 1, _all_bases = [arg0_1, arg1_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = copy__1 = None
        return ()""",  # noqa: B950
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            # 2. Run with inductor backend
            if smith._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_inductor,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(arg0_1, arg1_1);  foo_default = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        return ()""",  # noqa: B950
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    # foo takes a mutable list with views in addition to other args.
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra4(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!)[] y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y[0].sin_()

            def f(x, y, z):
                a = x[0]
                b = z[0]
                smith.ops.mylib.foo(a, [b, y])

            orig_args = [smith.randn(2), smith.randn(2), smith.randn(2)]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(f, orig_args)
            [inductor_args, result2, graph_inductor] = self.run_inductor(f, orig_args)
            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args[2], eager_args[2])
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if smith._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_aot,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu", arg2_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_length = 2, _y_0_base_index = 1, _y_0_size = (), _y_0_stride = (), _y_0_storage_offset = 0, _y_1_base_index = 2, _all_bases = [arg0_1, arg1_1, arg2_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized_v2[2]
        getitem_3: "f32[2][1]cpu" = auto_functionalized_v2[3];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = copy__1 = None
        copy__2: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg2_1, getitem_3);  arg2_1 = getitem_3 = copy__2 = None
        return ()""",  # noqa: B950
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            # 2. Run with inductor backend

            if smith._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_inductor,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu", arg2_1: "f32[2][1]cpu"):
        as_strided_default: "f32[][]cpu" = smith.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = smith.ops.aten.as_strided.default(arg1_1, [], [], 0)
        foo_default = smith.ops.mylib.foo.default(as_strided_default, [as_strided_default_1, arg2_1]);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        copy__2: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy__2 = None
        return ()""",  # noqa: B950
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_optional_v2(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!)? x, Tensor[] y, Tensor(b!)? z, SymInt w, Tensor n) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y, z, w, n):
                if x is not None:
                    x.add_(y[0] + w)
                if z is not None:
                    z.add_(y[1] + n)

            def f(x, y, z, n):
                smith.ops.mylib.foo(x, y, z, 2, n)

            x = None
            y = (smith.randn(3), smith.randn(3))
            z = smith.randn(3)
            n = smith.randn(3)
            orig_args = (x, y, z, n)

            compiled_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "smith._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                smith.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            if smith._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu"):
        foo_default = smith.ops.mylib.foo.default(None, [arg2_1, arg3_1], arg0_1, 2, arg1_1);  arg2_1 = arg3_1 = arg1_1 = foo_default = None
        copy_: "f32[3][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        return ()""",  # noqa: B950
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @smith._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_inference_mode1_v2(self):
        with smith.inference_mode():
            self.test_auto_functionalize_extra1()

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_inference_mode2_v2(self):
        with smith.inference_mode():
            self.test_auto_functionalize_extra2()

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_inference_mode3_v2(self):
        with smith.inference_mode():
            self.test_auto_functionalize_extra3()

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_inference_mode4_v2(self):
        with smith.inference_mode():
            self.test_auto_functionalize_extra4()

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dynamic_v2(self):
        self.test_auto_functionalize_v2(_dynamic=True)

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dynamic2_v2(self):
        self.test_auto_functionalize_extra1(_dynamic=True)

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dynamic3_v2(self):
        self.test_auto_functionalize_extra2(_dynamic=True)

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_graph_input_is_view(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x):
                pass

            @smith.compile(fullgraph=True, dynamic=False, backend="aot_eager")
            def f(x):
                a = x[0]
                smith.ops.mylib.foo(a)
                return

            x = smith.tensor([[1, 2], [3, 4]])
            # This would fail if auto_functionalized_v2 uses clone and not clone_preserve_strides
            # to clone not-inplaced args.
            f(x[1])

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias(self, _dynamic=False):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x):
                a = smith.ops.aten.alias.default(x)
                b = smith.ops.aten.alias.default(x)
                smith.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [smith.randn(2)]
            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0][1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 0, _y_alias = True, _all_bases = [arg1_1])
        getitem_1: "f32[s0][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[s0][1]cpu" = smith.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = copy_ = None
        alias_2: "f32[s0][1]cpu" = smith.ops.aten.alias.default(getitem_1)
        alias_3: "f32[s0][1]cpu" = smith.ops.aten.alias.default(getitem_1);  getitem_1 = None
        return (alias_2, alias_3)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 0, _y_alias = True, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        alias_2: "f32[2][1]cpu" = smith.ops.aten.alias.default(getitem_1)
        alias_3: "f32[2][1]cpu" = smith.ops.aten.alias.default(getitem_1);  getitem_1 = None
        return (alias_2, alias_3)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0][1]cpu"):
        alias_default: "f32[s0][1]cpu" = smith.ops.aten.alias.default(arg1_1)
        alias_default_1: "f32[s0][1]cpu" = smith.ops.aten.alias.default(arg1_1)
        foo_default = smith.ops.mylib.foo.default(alias_default, alias_default_1);  \
alias_default = alias_default_1 = foo_default = None
        copy_: "f32[s0][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  copy_ = None
        return (arg1_1, arg1_1)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        alias_default: "f32[2][1]cpu" = smith.ops.aten.alias.default(arg0_1)
        alias_default_1: "f32[2][1]cpu" = smith.ops.aten.alias.default(arg0_1)
        foo_default = smith.ops.mylib.foo.default(alias_default, alias_default_1);  \
alias_default = alias_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        return (arg0_1, arg0_1)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # Test that slice view is generated instead of as_strided when split is used.
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_split(self, _dynamic=False):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x):
                splits = x.split([4, 6], dim=1)
                a = splits[0]
                b = splits[1]
                smith.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [smith.randn(10, 10)]
            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    # split forces a specialization on size so we dont see arg0_1 dynamic anymore.
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 1, _x_slice_start = 0, _x_slice_end = 4, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 4, _y_slice_end = 10, _all_bases = [arg0_1])
        getitem_3: "f32[10, 10][10, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[10, 10][10, 1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_3);  arg0_1 = copy_ = None
        split_with_sizes_1 = smith.ops.aten.split_with_sizes.default(getitem_3, [4, 6], 1)
        getitem_4: "f32[10, 4][10, 1]cpu" = split_with_sizes_1[0];  split_with_sizes_1 = None
        split_with_sizes_2 = smith.ops.aten.split_with_sizes.default(getitem_3, [4, 6], 1);  getitem_3 = None
        getitem_7: "f32[10, 6][10, 1]cpu" = split_with_sizes_2[1];  split_with_sizes_2 = None
        return (getitem_4, getitem_7)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 1, _x_slice_start = 0, _x_slice_end = 4, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 4, _y_slice_end = 10, _all_bases = [arg0_1])
        getitem_3: "f32[10, 10][10, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[10, 10][10, 1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_3);  arg0_1 = copy_ = None
        split_with_sizes_1 = smith.ops.aten.split_with_sizes.default(getitem_3, [4, 6], 1)
        getitem_4: "f32[10, 4][10, 1]cpu" = split_with_sizes_1[0];  split_with_sizes_1 = None
        split_with_sizes_2 = smith.ops.aten.split_with_sizes.default(getitem_3, [4, 6], 1);  getitem_3 = None
        getitem_7: "f32[10, 6][10, 1]cpu" = split_with_sizes_2[1];  split_with_sizes_2 = None
        return (getitem_4, getitem_7)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    # split forces a specialization on size so we dont see arg0_1 dynamic anymore.
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        slice_tensor: "f32[10, 4][10, 1]cpu" = smith.ops.aten.slice.Tensor(arg0_1, 1, 0, 4)
        slice_tensor_1: "f32[10, 6][10, 1]cpu" = smith.ops.aten.slice.Tensor(arg0_1, 1, 4, 10)
        foo_default = smith.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[10, 10][10, 1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        split_with_sizes_1 = smith.ops.aten.split_with_sizes.default(arg0_1, [4, 6], 1)
        getitem_4: "f32[10, 4][10, 1]cpu" = split_with_sizes_1[0];  split_with_sizes_1 = None
        split_with_sizes_2 = smith.ops.aten.split_with_sizes.default(arg0_1, [4, 6], 1);  arg0_1 = None
        getitem_7: "f32[10, 6][10, 1]cpu" = split_with_sizes_2[1];  split_with_sizes_2 = None
        return (getitem_4, getitem_7)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        slice_tensor: "f32[10, 4][10, 1]cpu" = smith.ops.aten.slice.Tensor(arg0_1, 1, 0, 4)
        slice_tensor_1: "f32[10, 6][10, 1]cpu" = smith.ops.aten.slice.Tensor(arg0_1, 1, 4, 10)
        foo_default = smith.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[10, 10][10, 1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        split_with_sizes_1 = smith.ops.aten.split_with_sizes.default(arg0_1, [4, 6], 1)
        getitem_4: "f32[10, 4][10, 1]cpu" = split_with_sizes_1[0];  split_with_sizes_1 = None
        split_with_sizes_2 = smith.ops.aten.split_with_sizes.default(arg0_1, [4, 6], 1);  arg0_1 = None
        getitem_7: "f32[10, 6][10, 1]cpu" = split_with_sizes_2[1];  split_with_sizes_2 = None
        return (getitem_4, getitem_7)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # Note that split force the input tensor to get specialized. So we do not see SymInts when _dynamic=True.
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_split_dynamic(self):
        self.test_split(_dynamic=True)

    # Test that slice view is generated instead of as_strided when slice is used.
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_slice(self, _dynamic=False):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x):
                a = smith.ops.aten.slice.Tensor(x, 0, 0, 2)
                b = smith.ops.aten.slice.Tensor(x, 1, 3, 4)
                smith.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [smith.randn(10, 10)]
            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77, s77][s77, 1]cpu"):
        floordiv: "Sym(0)" = 0 // arg0_1;  arg0_1 = None
        add_6: "Sym(2)" = floordiv + 2
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 0, _x_slice_start = floordiv, _x_slice_end = add_6, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 3, _y_slice_end = 4, _all_bases = [arg1_1]);  floordiv = add_6 = None
        getitem_1: "f32[s77, s77][s77, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[s77, s77][s77, 1]cpu" = smith.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = copy_ = None
        slice_3: "f32[2, s77][s77, 1]cpu" = smith.ops.aten.slice.Tensor(getitem_1, 0, 0, 2)
        slice_4: "f32[s77, 1][s77, 1]cpu" = smith.ops.aten.slice.Tensor(getitem_1, 1, 3, 4);  getitem_1 = None
        return (slice_3, slice_4)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 0, _x_slice_start = 0, _x_slice_end = 2, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 3, _y_slice_end = 4, _all_bases = [arg0_1])
        getitem_1: "f32[10, 10][10, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[10, 10][10, 1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        slice_3: "f32[2, 10][10, 1]cpu" = smith.ops.aten.slice.Tensor(getitem_1, 0, 0, 2)
        slice_4: "f32[10, 1][10, 1]cpu" = smith.ops.aten.slice.Tensor(getitem_1, 1, 3, 4);  getitem_1 = None
        return (slice_3, slice_4)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77, s77][s77, 1]cpu"):
        floordiv: "Sym(0)" = 0 // arg0_1;  arg0_1 = None
        add_6: "Sym(2)" = floordiv + 2;  floordiv = add_6 = None
        slice_tensor: "f32[2, s77][s77, 1]cpu" = smith.ops.aten.slice.Tensor(arg1_1, 0, 0, 2)
        slice_tensor_1: "f32[s77, 1][s77, 1]cpu" = smith.ops.aten.slice.Tensor(arg1_1, 1, 3, 4)
        foo_default = smith.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[s77, s77][s77, 1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  copy_ = None
        slice_3: "f32[2, s77][s77, 1]cpu" = smith.ops.aten.slice.Tensor(arg1_1, 0, 0, 2)
        slice_4: "f32[s77, 1][s77, 1]cpu" = smith.ops.aten.slice.Tensor(arg1_1, 1, 3, 4);  arg1_1 = None
        return (slice_3, slice_4)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        slice_tensor: "f32[2, 10][10, 1]cpu" = smith.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_tensor_1: "f32[10, 1][10, 1]cpu" = smith.ops.aten.slice.Tensor(arg0_1, 1, 3, 4)
        foo_default = smith.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[10, 10][10, 1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_3: "f32[2, 10][10, 1]cpu" = smith.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_4: "f32[10, 1][10, 1]cpu" = smith.ops.aten.slice.Tensor(arg0_1, 1, 3, 4);  arg0_1 = None
        return (slice_3, slice_4)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # Note that split force the input tensor to get specialized. So we do not see SymInts when _dynamic=True.
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_slice_dynamic(self):
        self.test_slice(_dynamic=True)

    def test_try_use_slice(self):
        def test_round_trip(base, tensor):
            (dim, start, end) = try_use_slice(base, tensor)
            sliced = smith.ops.aten.slice.Tensor(base, dim, start, end)
            self.assertEqual(sliced, tensor)

        t = smith.tensor([[2, 2], [3, 4]])
        test_round_trip(t, t)

        for dim in range(-1, 1):
            f = t.split(2, dim)
            test_round_trip(t, f[0])

        for dim in range(-1, 1):
            f = t.split(1, dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])

        t = smith.randint(1, 10, (3, 3, 3))
        test_round_trip(t, t)

        for dim in range(-3, 3):
            f = t.split([1, 2], dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])

        for dim in range(-3, 3):
            f = t.split(1, dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])
            test_round_trip(t, f[2])

        t = smith.rand(10, 10, 10)
        test_round_trip(t, t)
        for dim in range(-3, 3):
            f = t.split([2, 2, 6], dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])
            test_round_trip(t, f[2])

        # example where slice won't work

        # selection
        t = smith.ones(10)
        b = t[0]
        self.assertEqual(try_use_slice(t, b), None)

        t = smith.tensor([[1, 2], [3, 4]])
        self.assertEqual(try_use_slice(t, t[0]), None)
        self.assertEqual(try_use_slice(t, t[1]), None)

        t = smith.tensor(
            [
                [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15, 16, 17]],
                [[71, 72, 73, 74, 75, 76, 77, 78], [81, 82, 83, 84, 85, 86, 87, 88]],
            ]
        )

        self.assertEqual(try_use_slice(t, t[0:1, 0:1, :7]), None)
        self.assertEqual(try_use_slice(t, t[0:1, 0:2, :3]), None)
        self.assertEqual(try_use_slice(t, t[0:2, 1, 0:8]), None)

        # simple slice operations are supported
        test_round_trip(t, t[0:2])
        test_round_trip(t, t[3:4])

    @smith._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias2(self, _dynamic=False):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x):
                a = smith.ops.aten.alias.default(x)
                b = x.clone()
                c = b.nonzero().float()
                d = smith.ops.aten.slice(
                    c
                )  # d is a Tensor with unbacked Symint in the shape
                smith.ops.mylib.foo(a, d)
                return a, d

            orig_args = [smith.randn(2)]
            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(smith.Tensor, smith.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77][1]cpu"):
        clone: "f32[s77][1]cpu" = smith.ops.aten.clone.default(arg1_1)
        nonzero: "i64[u0, 1][1, u0]cpu" = smith.ops.aten.nonzero.default(clone);  clone = None
        sym_size_int_1: "Sym(u0)" = smith.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
        _assert_scalar = smith.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        _to_copy: "f32[u0, 1][1, u0]cpu" = smith.ops.aten._to_copy.default(nonzero, dtype = smith.float32);  nonzero = None
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 1, _y_alias = True, _all_bases = [arg1_1, _to_copy]);  _to_copy = None
        getitem_1: "f32[s77][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[u0, 1][1, u0]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = copy_ = None
        alias_1: "f32[s77][1]cpu" = smith.ops.aten.alias.default(getitem_1);  getitem_1 = None
        slice_2: "f32[u0, 1][1, u0]cpu" = smith.ops.aten.slice.Tensor(getitem_2);  getitem_2 = None
        return (alias_1, slice_2)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        clone: "f32[2][1]cpu" = smith.ops.aten.clone.default(arg0_1)
        nonzero: "i64[u0, 1][1, u0]cpu" = smith.ops.aten.nonzero.default(clone);  clone = None
        sym_size_int: "Sym(u0)" = smith.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar = smith.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        le: "Sym(u0 <= 2)" = sym_size_int <= 2;  sym_size_int = None
        _assert_scalar_1 = smith.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 2 on node 'le'");  le = _assert_scalar_1 = None
        _to_copy: "f32[u0, 1][1, u0]cpu" = smith.ops.aten._to_copy.default(nonzero, dtype = smith.float32);  nonzero = None
        auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 1, _y_alias = True, _all_bases = [arg0_1, _to_copy]);  _to_copy = None
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[u0, 1][1, u0]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        alias_1: "f32[2][1]cpu" = smith.ops.aten.alias.default(getitem_1);  getitem_1 = None
        slice_2: "f32[u0, 1][1, u0]cpu" = smith.ops.aten.slice.Tensor(getitem_2);  getitem_2 = None
        return (alias_1, slice_2)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if smith._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77][1]cpu"):
        nonzero: "i64[u0, 1][1, u0]cpu" = smith.ops.aten.nonzero.default(arg1_1)
        sym_size_int_1: "Sym(u0)" = smith.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
        _assert_scalar = smith.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        convert_element_type: "f32[u0, 1][1, u0]cpu" = smith.ops.prims.convert_element_type.default(nonzero, smith.float32);  nonzero = None
        alias_default: "f32[s77][1]cpu" = smith.ops.aten.alias.default(arg1_1)
        alias_default_1: "f32[u0, 1][1, u0]cpu" = smith.ops.aten.alias.default(convert_element_type)
        foo_default = smith.ops.mylib.foo.default(alias_default, alias_default_1);  alias_default = alias_default_1 = foo_default = None
        copy_: "f32[s77][1]cpu" = smith.ops.aten.copy_.default(arg1_1, arg1_1);  copy_ = None
        slice_2: "f32[u0, 1][1, u0]cpu" = smith.ops.aten.slice.Tensor(convert_element_type);  convert_element_type = None
        return (arg1_1, slice_2)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        nonzero: "i64[u0, 1][1, u0]cpu" = smith.ops.aten.nonzero.default(arg0_1)
        sym_size_int: "Sym(u0)" = smith.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar = smith.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        le: "Sym(u0 <= 2)" = sym_size_int <= 2;  sym_size_int = None
        _assert_scalar_1 = smith.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 2 on node 'le'");  le = _assert_scalar_1 = None
        convert_element_type: "f32[u0, 1][1, u0]cpu" = smith.ops.prims.convert_element_type.default(nonzero, smith.float32);  nonzero = None
        alias_default: "f32[2][1]cpu" = smith.ops.aten.alias.default(arg0_1)
        alias_default_1: "f32[u0, 1][1, u0]cpu" = smith.ops.aten.alias.default(convert_element_type)
        foo_default = smith.ops.mylib.foo.default(alias_default, alias_default_1);  alias_default = alias_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = smith.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_2: "f32[u0, 1][1, u0]cpu" = smith.ops.aten.slice.Tensor(convert_element_type);  convert_element_type = None
        return (arg0_1, slice_2)""",  # noqa: B950
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias2_dynamic(self):
        self.test_alias2(_dynamic=True)

    # Test that the view regeneration optimizations do not result in recompilations. By comparing re-compilation in eager backend
    # with recompilation in inductor backend.
    @smith.fx.experimental._config.patch(use_duck_shape=False)
    def test_recompile(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo_impl(x, y):
                pass

            def run_and_compare(func, expected=1):
                counter_v2 = CompileCounterWithBackend("inductor")
                counter_v1 = CompileCounterWithBackend("inductor")
                v1 = smith.compile(
                    func, backend=counter_v1, fullgraph=True, dynamic=True
                )

                v2 = smith.compile(
                    func, backend=counter_v2, fullgraph=True, dynamic=True
                )
                inputs = [
                    smith.rand(10, 10),
                    smith.rand(100, 100),
                    smith.rand(10, 2),
                    smith.rand(1000, 1000),
                ]

                with smith._inductor.config.patch(enable_auto_functionalized_v2=True):
                    for input in inputs:
                        v2(input)

                smith._dynamo.reset()

                with smith._inductor.config.patch(enable_auto_functionalized_v2=False):
                    for input in inputs:
                        v1(input)

                self.assertEqual(counter_v2.frame_count, counter_v1.frame_count)

                self.assertEqual(counter_v1.frame_count, expected)

            def func(x):
                a = x[0]
                b = x[1]
                smith.ops.mylib.foo(a, b)

            run_and_compare(func)

            def func(x):
                a = smith.ops.aten.alias.default(x)
                b = smith.ops.aten.alias.default(x)
                smith.ops.mylib.foo(a, b)

            run_and_compare(func)

            def func(x):
                # last row
                a = x[x.size()[0] - 1]

                # first row
                b = x[0]
                smith.ops.mylib.foo(a, b)

            run_and_compare(func)

            def func(x):
                a = smith.ops.aten.slice.Tensor(x, 1, 3, 4)
                b = smith.ops.aten.slice.Tensor(x, 0, 1, 4)
                smith.ops.mylib.foo(a, b)

            # recompile here is not triggered by auto_functionalize
            # [__recompiles]     - 0/0: 4 <= L['x'].size()[1]  # a = smith.ops.aten.slice.Tensor(x, 1, 3, 4)
            # test/inductor/test_auto_functionalize.py:1160 in func (_decomp/decompositions.py:781 in slice_forward)
            run_and_compare(func, 2)

            def func(x):
                a = smith.ops.aten.alias.default(x)
                b = x.clone()
                c = b.nonzero().float()
                d = smith.ops.aten.slice(
                    c
                )  # d is a Tensor with unbacked Symint in the shape
                smith.ops.mylib.foo(a, d)
                return a, d

            with smith._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
                run_and_compare(func, 1)

    # Test that the alias optimization, were alias is called instead of as_strided, preserve the fact
    # that id(x) != id(base)
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    @unittest.skip(
        reason="This test fails because something else in inductor optimize out the alias. issue #137434"
    )
    def test_alias_id_input_to_custom_op(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::not_eq",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::not_eq", "cpu", lib=lib)
            @smith._dynamo.disable
            def not_eq_impl(x, y):
                self.assertNotEqual(id(x), id(y))

            def func(x):
                a = smith.ops.aten.alias.default(x)
                smith.ops.mylib.not_eq(a, x)

            compiled = smith.compile(func, backend="inductor", fullgraph=True)
            compiled(smith.rand(2, 2))

    # Test that the alias optimization, were alias is called instead of as_strided, preserve the fact
    # that id(x) != id(base)
    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias_id_output(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo(x, y):
                pass

            def func(x):
                a = smith.ops.aten.alias.default(x)
                smith.ops.mylib.foo(a, x)
                return a

            compiled = smith.compile(func, backend="inductor", fullgraph=True)
            input = smith.rand(2, 2)
            output = compiled(smith.rand(2, 2))
            self.assertNotEqual(id(output), id(input))

    def test_inference_mode_view(self):
        @smith.library.custom_op(
            "test_inference_mode_view::foo", mutates_args={"workspace"}
        )
        def foo(x: smith.Tensor, workspace: smith.Tensor) -> smith.Tensor:
            return x.clone()

        @foo.register_fake
        def _(x, workspace):
            return x.clone()

        @smith.compile(fullgraph=True, backend="aot_eager")
        def f(x, w):
            y = foo(x, w)
            z = y.view(-1)
            return z.sin()

        x = smith.randn(2)
        w = smith.randn(2)
        with smith.inference_mode():
            y = f(x, w)
        self.assertEqual(y, x.sin())

    @smith._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_scheduling_with_multiple_mutates(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor! x, Tensor! y, Tensor z) -> ()",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            @smith._dynamo.disable
            def foo(x, y, z):
                pass

            def func(x, w):
                a = smith.empty_like(x)  # buf0
                b = smith.empty_like(x)  # buf1
                smith.ops.mylib.foo(a, b, x)  # buf2, buf3, buf4
                c = smith.mm(a, w)  # buf5
                smith.ops.mylib.foo(c, b, x)  # buf6, buf7, buf8
                return c

            input = smith.rand(2, 2)
            weight = smith.rand(2, 2)
            [inductor_args, output, graph_inductor] = self.run_inductor(
                func,
                [input, weight],
                False,
                "smith._inductor.scheduler",
                "compute_dependencies",
            )
            name_to_users = eval(graph_inductor)
            self.assertNotEqual(name_to_users["buf1"], name_to_users["buf5"])


if __name__ == "__main__":
    from smith._inductor.test_case import run_tests

    run_tests()
