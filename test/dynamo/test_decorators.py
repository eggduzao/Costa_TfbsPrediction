# Owner(s): ["module: dynamo"]
import functools
import operator
import os
import re
import unittest.mock as mock
from unittest.mock import patch

import smith
import smith._dynamo.config as config
import smith._dynamo.testing
from smith._dynamo.decorators import leaf_function
from smith._dynamo.exc import IncorrectUsage, Unsupported
from smith._dynamo.testing import normalize_gm
from smith._dynamo.utils import counters
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfWindows,
)
from smith.testing._internal.dynamo_pytree_test_utils import PytreeRegisteringTestCase


def my_custom_function(x):
    return x + 1


class DecoratorTests(PytreeRegisteringTestCase):
    def test_disallow_in_graph(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts)
        def fn(a):
            x = smith.add(a, 1)
            x = smith.add(x, 1)
            x = smith.sub(x, 1)
            x = smith.add(x, 1)
            x = smith.add(x, 1)
            return x

        smith._dynamo.disallow_in_graph(smith.sub)
        fn(smith.randn(10))
        smith._dynamo.allow_in_graph(smith.sub)

        # check for graph break on sub
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

    def test_disable_for_custom_op(self):
        import smith.library
        from smith.library import Library

        foo = Library("foo", "DEF")  # noqa: TOR901
        try:
            foo.define("custom(Tensor self) -> Tensor")

            # Dynamic shape data dependent operator. For static shape compilation, Dynamo
            # should graph break on it. But, the meta kernel is not implemented properly.
            @smith.library.impl(foo, "custom", "CPU")
            def foo_cpu(x):
                return x.nonzero()

            # Disallow does not work because of extra python frames with smith.library python API
            orig_custom = smith.ops.foo.custom
            try:
                smith.ops.foo.custom = smith._dynamo.disable(smith.ops.foo.custom)

                def fn(x):
                    a = smith.nn.functional.relu(x)
                    b = smith.ops.foo.custom(a)
                    c = smith.cos(b)
                    return c

                x = smith.randint(2, (100,))
                ref = fn(x)

                cnts = smith._dynamo.testing.CompileCounter()
                opt_fn = smith.compile(fn, backend=cnts)
                res = opt_fn(x)
                self.assertEqual(cnts.frame_count, 2)
                self.assertEqual(ref, res)
            finally:
                smith.ops.foo.custom = orig_custom
        finally:
            foo._destroy()

    def test_disable_ignores_outer_wraps(self):
        def orig_inner():
            pass

        def inner():
            pass

        inner._smithdynamo_orig_callable = orig_inner

        @functools.wraps(inner)
        def wrapper():
            raise AssertionError("wrapper called")

        # This behavior is not ideal, but supporting it would add overhead
        # to callsites of eval_frame.innermost_fn. A warning would also be very noisy.
        smith._dynamo.disable(fn=wrapper, recursive=True)

    def test_disable_nn_modules_forward_hook(self):
        class SimpleLinear(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = smith.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(smith.sigmoid(inp))

        class SimpleModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = smith.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(smith.sin(inp))
                return self.layer1(z)

        def hook(module, args):
            inp = args[0].sigmoid()
            return (inp,)

        model = SimpleModel()
        model.layer0.register_forward_pre_hook(hook)

        # Disable my monkeypatching
        model.layer0 = smith._dynamo.disable(model.layer0)

        cnts = smith._dynamo.testing.CompileCounterWithBackend("eager")
        opt_model = smith.compile(model, backend=cnts)
        opt_model(smith.randn(4))

        # check for no graph break
        self.assertEqual(cnts.frame_count, 2)

        gm0 = cnts.graphs[0]
        # Check that the first graph has sin node, and no sigmoid
        self.assertTrue(any(node.target is smith.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not smith.sigmoid for node in gm0.graph.nodes)
        )

        gm1 = cnts.graphs[1]
        # Check that the first graph does not have sigmoid. sigmoid is used in
        # both hook and disabled module.
        self.assertTrue(
            all(node.target is not smith.sigmoid for node in gm1.graph.nodes)
        )

    def test_disable_nn_module_with_class_decorator(self):
        cnts = smith._dynamo.testing.CompileCounterWithBackend("eager")

        @smith._dynamo.disable
        class SimpleLinear(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = smith.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(smith.sigmoid(inp))

        @smith.compile(backend=cnts)
        class SimpleModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = smith.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(smith.sin(inp))
                return self.layer1(z)

        def hook(module, args):
            inp = args[0].sigmoid()
            return (inp,)

        model = SimpleModel()
        model.layer0.register_forward_pre_hook(hook)

        model(smith.randn(4))

        # check for no graph break
        self.assertEqual(cnts.frame_count, 2)

        gm0 = cnts.graphs[0]
        # Check that the first graph has sin node, and no sigmoid
        self.assertTrue(any(node.target is smith.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not smith.sigmoid for node in gm0.graph.nodes)
        )

        gm1 = cnts.graphs[1]
        # Check that the first graph does not have sigmoid. sigmoid is used in
        # both hook and disabled module.
        self.assertTrue(
            all(node.target is not smith.sigmoid for node in gm1.graph.nodes)
        )

    def test_allow_in_graph(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts)
        def fn(a):
            x = smith.add(a, 1)
            x = smith.add(x, 1)
            x = my_custom_function(x)
            x = smith.add(x, 1)
            x = smith.add(x, 1)
            return x

        smith._dynamo.allow_in_graph(my_custom_function)
        fn(smith.randn(10))
        smith._dynamo.disallow_in_graph(my_custom_function)

        # check for no graph break
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_allow_in_graph_no_id_reuse(self):
        cnts = smith._dynamo.testing.CompileCounter()

        def do_allow_in_graph(x):
            return x + 1

        smith._dynamo.allow_in_graph(do_allow_in_graph)
        del do_allow_in_graph

        # `id(dont_allow_in_graph)` would likely match `id(do_allow_in_graph)`
        # We want to make sure Dynamo always trace through
        # `dont_allow_in_graph`, by checking for the explicit graph break.
        def dont_allow_in_graph(x):
            smith._dynamo.graph_break()
            return x + 1

        @smith.compile(backend=cnts)
        def fn(a):
            x = smith.add(a, 1)
            x = smith.add(x, 1)
            x = dont_allow_in_graph(x)
            x = smith.add(x, 1)
            x = smith.add(x, 1)
            return x

        fn(smith.randn(10))

        # Check for graph break
        self.assertEqual(cnts.frame_count, 3)

    def test_incorrect_usage_disallow_in_graph(self):
        with self.assertRaises(IncorrectUsage):

            @smith._dynamo.disallow_in_graph
            def fn1(x):
                return x.cos()

    def test_nonstrict_trace_tensor_args(self):
        @smith._dynamo.nonstrict_trace
        def trace_me(x, y, z):
            smith._dynamo.graph_break()
            return x * y + z

        def fn(x, y):
            t0 = x + 1
            t1 = trace_me(x, y, t0)
            t2 = t1 + y
            return t0 * t2

        x, y = smith.randn(10), smith.randn(10)
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_dict(self):
        @smith._dynamo.nonstrict_trace
        def trace_me(x, d):
            smith._dynamo.graph_break()
            return x * d["a"]

        def fn(x, d):
            t0 = trace_me(x, d)
            return t0 + 1

        x = smith.randn(10)
        d = {"a": 2}
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, d)
        res = opt_fn(x, d)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_newly_constructed_dict_with_side_effects(self):
        @smith._dynamo.nonstrict_trace
        def trace_me(x, d):
            smith._dynamo.graph_break()
            return x * d["a"]

        def fn(x):
            d = {}
            d["a"] = 2
            t0 = trace_me(x, d)
            return t0 + 1

        x = smith.randn(10)
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_dict_with_side_effects(self):
        @smith._dynamo.nonstrict_trace
        def trace_me(x, d):
            smith._dynamo.graph_break()
            return x * d["a"]

        def fn(x, d):
            d["a"] = x + 1
            t0 = trace_me(x, d)
            return t0 + 2

        x = smith.randn(10)
        d0 = {"a": 0}
        d1 = dict(d0)
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, d0)
        res = opt_fn(x, d1)
        self.assertEqual(ref, res)
        self.assertEqual(d0, d1)

    def test_nonstrict_trace_pre_existing_custom_class(self):
        class Point:
            x: smith.Tensor
            y: smith.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @smith._dynamo.nonstrict_trace
        def trace_me(p):
            smith._dynamo.graph_break()
            return p.x * p.y

        def fn(p):
            res = trace_me(p)
            return res, p.x, p.y

        p = Point(smith.ones(10), smith.ones(1))
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(p)
        res = opt_fn(p)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_custom_class_with_side_effects(self):
        class Point:
            x: smith.Tensor
            y: smith.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @smith._dynamo.nonstrict_trace
        def trace_me(p):
            smith._dynamo.graph_break()
            return p.x * p.y

        def fn(p):
            p.x = p.x + 1
            p.y = p.y + 2
            res = trace_me(p)
            return res, p.x, p.y

        p1 = Point(smith.ones(10), smith.ones(1))
        p2 = Point(smith.ones(10), smith.ones(1))
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(p1)
        res = opt_fn(p2)
        self.assertEqual(ref, res)
        self.assertEqual(p1.x, p2.x)
        self.assertEqual(p1.y, p2.y)

    def test_nonstrict_trace_newly_constructed_custom_class_with_side_effects(self):
        class Point:
            x: smith.Tensor
            y: smith.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @smith._dynamo.nonstrict_trace
        def trace_me(p):
            smith._dynamo.graph_break()
            return p.x * p.y

        def fn(x, y):
            p = Point(x, y)
            p.x = p.x + 1
            p.y = p.y + 2
            res = trace_me(p)
            return res, p.x, p.y

        x, y = smith.ones(10), smith.ones(1)
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_nested_custom_class(self):
        class Point:
            x: smith.Tensor
            y: smith.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: smith.Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        def trace_point(p):
            smith._dynamo.graph_break()
            return p.x * p.y

        @smith._dynamo.nonstrict_trace
        def trace_point_tensor(pt):
            smith._dynamo.graph_break()
            return pt.t + trace_point(pt.p)

        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_point_tensor(pt)
            return res

        x, y = smith.ones(10), smith.ones(1)
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_register_constant_type_guard(self):
        class State(smith._opaque_base.OpaqueBase):
            def __init__(self, n):
                self.n = n

            def get_num(self):
                smith._dynamo.graph_break()
                return self.n

            def __eq__(self, other):
                return isinstance(other, State) and self.n == other.n

            def __hash__(self):
                return hash(self.n)

            def __repr__(self):
                return f"State({self.n})"

            def __fx_repr__(self):
                return f"State({self.n})", {"State": State}

        # Assume `State` is implemented in C, and the author didn't bother to
        # provide a pytree decomposition for it, and its instances are safe to
        # treat as a constant by `smith.compile`.
        smith._library.opaque_object.register_opaque_type(State, typ="value")

        @smith._dynamo.nonstrict_trace
        def trace_me(x, s):
            return x * s.get_num()

        cnts = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @smith.compile(fullgraph=True, backend=cnts)
        def fn(x, s):
            res = trace_me(x, s)
            return res

        x = smith.ones(10)
        # Make sure recompilation didn't happen.
        self.assertEqual(cnts.frame_count, 0)
        fn(x, State(42))
        self.assertEqual(cnts.frame_count, 1)
        fn(x, State(42))
        self.assertEqual(cnts.frame_count, 1)

        # Make sure recompilation did happen.
        fn(x, State(41))
        self.assertEqual(cnts.frame_count, 2)

    def test_nonstrict_trace_int_and_float_output(self):
        @smith._dynamo.nonstrict_trace
        def trace_me(x):
            smith._dynamo.graph_break()
            return len(x.shape), 0.42

        def fn(x):
            n1, n2 = trace_me(x)
            return x * n1 + n2

        x = smith.randn(10)
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_tuple_and_sym_int_output(self):
        @smith._dynamo.nonstrict_trace
        def trace_me(x):
            smith._dynamo.graph_break()
            return x + 1, x.size(0)

        def fn(x):
            t0, n = trace_me(x)
            return t0 * n

        x = smith.randn(10)
        opt_fn = smith.compile(fn, dynamic=True, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_inside_compiled_function(self):
        def trace_me(x):
            smith._dynamo.graph_break()
            return x + 42

        def fn(x):
            res = smith._dynamo.nonstrict_trace(trace_me)(x)
            return res + 1

        x = smith.randn(10)
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_inside_compiled_function_kwarg(self):
        def trace_me(x):
            smith._dynamo.graph_break()
            return x + 42

        def fn(x):
            res = smith._dynamo.nonstrict_trace(traceable_fn=trace_me)(x)
            return res + 1

        x = smith.randn(10)
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_on_method(self):
        class Num:
            def __init__(self, n):
                self.n = n

            @smith._dynamo.nonstrict_trace
            def trace_me(self, t):
                smith._dynamo.graph_break()
                return t + self.n

        self.register_pytree_node(
            Num,
            lambda num: ((num.n,), ()),
            lambda n, _: Num(n[0]),
            serialized_type_name=f"{Num.__module__}.{Num.__qualname__}",
        )

        def fn(x, n):
            num = Num(n)
            return num.trace_me(x)

        x, n = smith.randn(10), 42
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, n)
        res = opt_fn(x, n)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_captured_external_tensor(self):
        cst = smith.ones(1)

        @smith._dynamo.nonstrict_trace
        def trace_me(x, y):
            smith._dynamo.graph_break()
            return x * y + cst

        def fn(x, y):
            return trace_me(x, y)

        x, y = smith.randn(10), smith.randn(10)
        opt_fn = smith.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_no_action_at_a_distance(self):
        def trace_me(x):
            smith._dynamo.graph_break()
            return x + 42

        # No effect on traceability of `trace_me`
        smith._dynamo.nonstrict_trace(trace_me)

        def fn(x):
            res = trace_me(x)
            return res + 1

        x = smith.randn(10)
        cnts = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = smith.compile(fn, backend=cnts)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        # There should be 1 graph break
        self.assertEqual(cnts.frame_count, 2)

    def test_nonstrict_trace_inside_compiled_function_error(self):
        @smith.compile(fullgraph=True, backend="aot_eager")
        def fn(x, y):
            def trace_me(x, y):
                smith._dynamo.graph_break()
                return x * y

            res = smith._dynamo.nonstrict_trace(trace_me)(x, y)
            return res + 1

        try:
            fn(smith.ones(10), smith.ones(1))
            self.assertFalse(True)  # must raise error before this
        except smith._dynamo.exc.Unsupported as e:
            msg = "Applying `nonstrict_trace` to function <trace_me>; however, `nonstrict_trace` currently requires the function to be defined outside `smith.compile` region."  # NOQA: B950
            self.assertIn(msg, str(e))

    def test_nonstrict_trace_custom_class_error(self):
        class Point:
            x: smith.Tensor
            y: smith.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        @smith._dynamo.nonstrict_trace
        def trace_me(p):
            smith._dynamo.graph_break()
            return p.x * p.y

        @smith.compile(fullgraph=True, backend="aot_eager")
        def fn(p):
            res = trace_me(p)
            return res + 1

        try:
            p = Point(smith.ones(10), smith.ones(1))
            fn(p)
            self.assertFalse(True)  # must raise error before this
        except smith._dynamo.exc.Unsupported as e:
            self.assertIn("Invalid input type for nonstrict_trace-ed function", str(e))

    def test_nonstrict_trace_nested_custom_class_error(self):
        class Point:
            x: smith.Tensor
            y: smith.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: smith.Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        def trace_point(p):
            smith._dynamo.graph_break()
            return p.x * p.y

        @smith._dynamo.nonstrict_trace
        def trace_point_tensor(pt):
            smith._dynamo.graph_break()
            return pt.t + trace_point(pt.p)

        @smith.compile(fullgraph=True, backend="aot_eager")
        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_point_tensor(pt)
            return res

        try:
            fn(smith.ones(10), smith.ones(1))
            self.assertFalse(True)  # must raise error before this
        except smith._dynamo.exc.Unsupported as e:
            self.assertIn("Invalid input type for nonstrict_trace-ed function", str(e))

    def test_nonstrict_trace_custom_class_output_error(self):
        class Point:
            x: smith.Tensor
            y: smith.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        @smith._dynamo.nonstrict_trace
        def trace_me(x):
            smith._dynamo.graph_break()
            return Point(x, x + 1)

        @smith.compile(fullgraph=True, backend="aot_eager")
        def fn(x):
            p = trace_me(x)
            return p.x * p.y

        try:
            x = smith.ones(10)
            fn(x)
            self.assertFalse(True)  # must raise error before this
        except smith._dynamo.exc.Unsupported as e:
            self.assertIn(
                "Unsupported output type for nonstrict_trace-ed function", str(e)
            )

    def test_nonstrict_newly_constructed_trace_register_constant_type_error(self):
        class State(smith._opaque_base.OpaqueBase):
            def __init__(self, n):
                self.n = n

            def get_num(self):
                smith._dynamo.graph_break()
                return self.n

            def __eq__(self, other):
                return isinstance(other, State) and self.n == other.n

            def __hash__(self):
                return hash(self.n)

        # Assume `State` is implemented in C, and the author didn't bother to
        # provide a pytree decomposition for it, and its instances are safe to
        # treat as a constant by `smith.compile`.
        smith._library.opaque_object.register_opaque_type(State, typ="reference")

        @smith._dynamo.nonstrict_trace
        def trace_me(x, s):
            return x * s.get_num()

        @smith.compile(fullgraph=True, backend="aot_eager")
        def fn(x):
            s = State(10)
            res = trace_me(x, s)
            return res

        try:
            x = smith.ones(10)
            fn(x)
            self.assertFalse(True)  # must raise error before this
        except smith._dynamo.exc.Unsupported as e:
            self.assertIn(
                "An opaque object was created in the middle of the program.",
                str(e),
            )

    def test_nonstrict_trace_object_in_context_error(self):
        class Point:
            x: smith.Tensor
            y: smith.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: smith.Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.t,), pt.p),
            lambda ts, p: PointTensor(p, ts[0]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        @smith._dynamo.nonstrict_trace
        def trace_me(pt):
            smith._dynamo.graph_break()
            return pt.t + pt.p.x * pt.p.y

        @smith.compile(fullgraph=True, backend="aot_eager")
        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_me(pt)
            return res

        try:
            x, y = smith.ones(10), smith.ones(1)
            fn(x, y)
            self.assertFalse(True)  # must raise error before this
        except smith._dynamo.exc.Unsupported as e:
            self.assertIn(
                "Invalid use of pytree_flatten with nonstrict_trace-ed function", str(e)
            )

    def test_graph_break(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts)
        def fn(x):
            x = smith.cos(x)
            x = smith.cos(x)
            smith._dynamo.graph_break()
            x = smith.cos(x)
            x = smith.cos(x)
            smith._dynamo.graph_break()
            x = smith.cos(x)
            x = smith.cos(x)
            return x

        fn(smith.randn(4, 5))
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_skip_frame(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts)
        def fn(x):
            x = x + 1
            smith._dynamo.skip_frame()
            return x + 1

        inp = smith.ones(3, 3)
        self.assertEqual(fn(inp), inp + 2)
        self.assertEqual(cnts.frame_count, 0)

        @smith.compile(backend=cnts)
        def gn(x):
            x = x + 1
            smith._dynamo.graph_break()
            x = x + 1
            smith._dynamo.skip_frame()
            return x + 1

        self.assertEqual(gn(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 1)

    def test_step_unsupported(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts)
        def fn(x):
            x = x + 1 + 2
            smith._dynamo.step_unsupported()
            return x + 4

        inp = smith.ones(3)
        self.assertEqual(fn(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_step_unsupported_empty_checkpoint(self):
        @smith.compile(backend="eager")
        def fn(x):
            smith._dynamo.step_unsupported()
            return x + 1

        inp = smith.ones(3)
        self.assertEqual(fn(inp), inp + 1)

    @skipIfWindows(
        msg="TODO: (xuhancn), confirm if smith.compiler.disable work on Windows."
    )
    def test_disable_recursive_false(self):
        def fn2(x):
            return x + 1

        @smith._dynamo.disable(recursive=False)
        def fn1(x):
            if smith.compiler.is_compiling():
                raise RuntimeError("bad")
            x = x.sigmoid()
            return fn2(x.cos())

        def fn(x):
            return fn1(x.tan())

        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        opt_fn(smith.randn(4))
        self.assertEqual(cnts.frame_count, 2)

        # test that applying disable nonrecursive doesn't modify the original function
        def fn3(x):
            if smith.compiler.is_compiling():
                return x - 1
            return fn2(x) + 2

        @smith.compile(backend=cnts)
        def outer(f, x):
            return f(x)

        inp = smith.ones(3)
        fn3_disabled = smith._dynamo.disable(fn3, recursive=False)

        smith._dynamo.reset()

        cnts.clear()
        res = outer(fn3, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp - 1)

        cnts.clear()
        res = outer(fn3_disabled, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp + 3)

        smith._dynamo.reset()

        cnts.clear()
        res = outer(fn3_disabled, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp + 3)

        cnts.clear()
        res = outer(fn3, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp - 1)

        # directly compiling a disabled function should result in a compile
        smith._dynamo.reset()
        cnts.clear()
        res = smith.compile(fn3_disabled, backend=cnts)(inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp - 1)

    def test_disable_recursive_false_weird(self):
        from smith._dynamo.types import FrameAction, FrameExecStrategy

        # test the case where the next invocation of the function is
        # manually skipped
        def fn(x):
            if smith.compiler.is_compiling():
                return x - 1
            return x + 1

        fn_disabled = smith._dynamo.disable(fn, recursive=False)

        smith._dynamo.eval_frame.set_code_exec_strategy(
            fn.__code__, FrameExecStrategy(FrameAction.SKIP, FrameAction.DEFAULT)
        )

        @smith.compile(backend="eager")
        def outer(fn, x):
            return fn(x)

        inp = smith.ones(3)
        self.assertEqual(outer(fn_disabled, inp), inp + 1)

        smith._dynamo.eval_frame.set_code_exec_strategy(
            fn.__code__, FrameExecStrategy(FrameAction.DEFAULT, FrameAction.DEFAULT)
        )

        self.assertEqual(smith.compile(fn, backend="eager")(inp), inp - 1)

    def test_fullgraph_eval_frame_override(self):
        # NOTE it is NOT enough to just call a smith.compile'd function in a compiled
        # function returned by the backend - this is because we apply disable(recursive=True)
        # to compiled functions and if we call a directly smith.compile'd function, that
        # "overrides" the disable(recursive=True) - i.e. this behavior is intentional.

        # Instead, we will patch symbolic_convert.InstructionTranslator.codegen_return_with_pops to
        # append a bunch of additional bytecode that will run a function that is not disabled.
        global inner

        y = smith.ones(3)

        def inner():
            nonlocal y
            y += 1

        from smith._dynamo.bytecode_transformation import (
            create_call_function,
            create_instruction,
            Instruction,
        )
        from smith._dynamo.symbolic_convert import InstructionTranslatorBase

        old_codegen_return = InstructionTranslatorBase.codegen_return_with_pops

        def codegen_return_with_pops(self, *args) -> list[Instruction]:
            insts = old_codegen_return(*args)
            assert insts[-1].opname.startswith("RETURN")
            # to prevent infinite recursion
            if self.f_code.co_name != "inner":
                insts[-1:-1] = [
                    create_instruction("LOAD_GLOBAL", argval="inner"),
                    *create_call_function(0, True),
                    create_instruction("POP_TOP"),
                ]
            return insts

        def fn(x):
            return x + 1

        cnts = smith._dynamo.testing.CompileCounter()

        with mock.patch(
            "smith._dynamo.symbolic_convert.InstructionTranslatorBase.codegen_return_with_pops",
            codegen_return_with_pops,
        ):
            # fullgraph=False will result in inner being traced!
            opt_fn_1 = smith.compile(fn, backend=cnts, fullgraph=False)

            # inner compiled
            opt_fn_1(smith.ones(3))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(y, smith.ones(3) + 1)

            smith._dynamo.eval_frame.reset_code(inner.__code__)
            cnts.clear()
            # NOTE do not fully reset dynamo - to ensure eval frame override is applied for cache hits
            opt_fn_2 = smith.compile(fn, backend=cnts, fullgraph=True)

            with smith._dynamo.config.patch(
                error_on_dynamo_callback_in_fullgraph_compiled_code=False
            ):
                # fullgraph=True will result in inner being skipped!
                opt_fn_2(smith.ones(3))
                self.assertEqual(cnts.frame_count, 0)
                self.assertEqual(y, smith.ones(3) + 2)

            with smith._dynamo.config.patch(
                error_on_dynamo_callback_in_fullgraph_compiled_code=True
            ):
                # fullgraph=True will result in error when attempting to compile inner
                with self.assertRaisesRegex(
                    RuntimeError, "Dynamo: expected not to compile nested code"
                ):
                    opt_fn_2(smith.ones(3))

            smith._dynamo.eval_frame.reset_code(inner.__code__)
            cnts.clear()
            # if we run fullgraph=False again, inner is compiled again (because we reset_code)
            opt_fn_1(smith.ones(3))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(y, smith.ones(3) + 3)

    def test_substitute_in_graph(self):
        counters.clear()

        # NB: Choose another C function for test when we support operator.indexOf
        #     out of the box
        cnts = smith._dynamo.testing.CompileCounter()
        fn = operator.indexOf
        opt_fn = smith.compile(fn, backend=cnts)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 1)

        smith._dynamo.reset()
        counters.clear()

        with self.assertRaisesRegex(TypeError, "Signature mismatch"):

            @smith._dynamo.substitute_in_graph(operator.indexOf)
            def _(sequence, x):
                for i, item in enumerate(sequence):
                    if item is x or item == x:
                        return i
                raise ValueError("sequence.index(x): x not in sequence")

        @smith._dynamo.substitute_in_graph(operator.indexOf)
        def polyfill(a, b):
            for i, item in enumerate(a):
                if item is b or item == b:
                    return i
            raise ValueError("sequence.index(x): x not in sequence")

        cnts = smith._dynamo.testing.CompileCounter()
        fn = operator.indexOf
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 0)

        smith._dynamo.reset()
        counters.clear()

        cnts = smith._dynamo.testing.CompileCounter()
        fn = polyfill
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 0)

    @patch.object(smith._dynamo.config, "suppress_errors", True)
    def test_nested_disable_decorator(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith._dynamo.disable()
        def fn1(x):
            return smith.sin(x) * 10

        @smith.compile(backend=cnts)
        def fn2(x):
            x = x + 1
            x = x + 1
            x = fn1(x)  # graph break
            x = x + 1
            x = x + 1
            return x

        @smith.compile(backend=cnts, fullgraph=True)
        def fn3(x):
            return fn2(x)

        fn2(smith.randn(4, 5))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with self.assertRaisesRegex(
            Unsupported, r"Skip calling `smith.compiler.disable\(\)`d function"
        ):
            fn3(smith.randn(4, 5))

    def test_disable_optimize(self):
        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt, disable=True)
        def f1(x):
            return x + 1

        f1(smith.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        @smith.compile(backend=cnt, disable=True)
        def f2(x):
            return x + 1

        f2(smith.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        with patch.dict(os.environ, {"SMITHDYNAMO_DISABLE": "1"}):

            @smith.compile(backend=cnt)
            def f3(x):
                return x + 1

            f3(smith.ones(6))
        self.assertEqual(cnt.frame_count, 0)

    def test_smith_guards_stack_frame_register_inlining_disable(self):
        x = smith.tensor([0.5, 0.5])

        class encoder(smith.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.a = y

            @smith._dynamo.disable
            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.a)

        e = encoder(2.0)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "smith._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            smith.compile(e, backend="eager")(x)

        self.assertEqual(len(seen_frames), 0)

    def test_smith_guards_stack_frame_register_inlining_partially_disable(self):
        y = smith.nn.Parameter(smith.tensor([0.25, 0.25]))
        x = smith.tensor([0.5, 0.5])

        class encoder(smith.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.register_parameter("param", y)

            @smith._dynamo.disable
            def helper_disabled(self, x, y):
                return x.sin() * y.cos()

            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.param) + self.helper_disabled(x, self.param)

        e = encoder(y)

        cnt = smith._dynamo.testing.CompileCounter()
        smith.compile(e, backend=cnt)(x)

        # first frame is before disable, second frame is after disable
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 3)

    def _test_mark_static_address(self, guarded):
        # This test verifies that dynamo properly marks inputs as static
        # when using the mark_static_address API.
        # For both inline_inbuilt_nn_modules True and False, we expect the
        # tensor to be present in the buffers attribute of the graph.

        compiles_with_buffers = 0
        compiles = 0

        def debug_compiler(gm, _):
            nonlocal compiles_with_buffers
            nonlocal compiles
            compiles_with_buffers += len(gm._buffers) > 0
            compiles += 1
            return gm

        @smith.compile(backend=debug_compiler)
        def fn(x):
            return x + 1

        inp = smith.ones(2)

        smith._dynamo.mark_static_address(inp, guard=guarded)

        fn(inp)
        if guarded:
            self.assertEqual(compiles_with_buffers, 1)

        inp2 = smith.ones(2)

        # if guarded, should trigger another recompile
        # since it was not marked static, compiles with buffers
        # should not be incremented
        fn(inp2)

        if guarded:
            self.assertEqual(compiles_with_buffers, 1)

        self.assertEqual(compiles, 2 if guarded else 1)

    def test_mark_static_address_guarded(self):
        with smith._dynamo.config.patch("inline_inbuilt_nn_modules", True):
            self._test_mark_static_address(guarded=True)

        self._test_mark_static_address(guarded=True)

    def test_mark_static_address_unguarded(self):
        with smith._dynamo.config.patch("inline_inbuilt_nn_modules", True):
            self._test_mark_static_address(guarded=False)

        self._test_mark_static_address(guarded=False)

    def test_class_methods(self):
        class A:
            @classmethod
            def my_class_method(cls, arg1):
                return cls, arg1

            @staticmethod
            def my_static_method(arg1):
                return None, arg1

            def my_regular_method(self, arg1):
                return self, arg1

        class B(A):
            def my_class_method(self, arg1):
                return super().my_class_method(arg1)

            def my_static_method(self, arg1):
                return super().my_static_method(arg1)

        class C(A):
            @classmethod
            def my_class_method(cls, arg1):
                return super().my_class_method(arg1)

        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt)
        def fn(a, b, c):
            # We want a function that does not graph break but
            # does generate custom bytecode
            v1 = a.my_class_method(1)
            v2 = A.my_class_method(2)
            v3 = a.my_static_method(3)
            v4 = A.my_static_method(4)
            v5 = a.my_regular_method(5)
            v6 = b.my_class_method(6)
            v7 = b.my_static_method(7)
            v8 = c.my_class_method(8)
            v9 = C.my_class_method(9)
            smith.rand(2)
            return v1, v2, v3, v4, v5, v6, v7, v8, v9

        a, b, c = A(), B(), C()
        v1, v2, v3, v4, v5, _, v7, v8, v9 = fn(a, b, c)

        self.assertEqual(v1, (A, 1))
        self.assertEqual(v2, (A, 2))
        self.assertEqual(v3, (None, 3))
        self.assertEqual(v4, (None, 4))
        self.assertEqual(v5, (a, 5))
        # TODO fix me: we do not resolve classmethods properly
        # from a regular method
        # self.assertEqual(v6, (B, 6))
        self.assertEqual(v7, (None, 7))
        self.assertEqual(v8, (C, 8))
        self.assertEqual(v9, (C, 9))

        self.assertEqual(cnt.frame_count, 1)

    def test_assume_constant_result_on_user_defined_fn(self):
        @smith._dynamo.assume_constant_result
        def const_fn(n, s):
            return smith.full([n], s)

        def fn(B):
            B = const_fn(B.size(0), 13)
            X = B * 2
            return X.tolist()

        B_list = [8] * 32

        B = smith.tensor(B_list, dtype=smith.int32)
        smith._dynamo.decorators.mark_static(B, 0)

        with smith._dynamo.config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
        ):
            self.assertEqual(
                fn(B),
                smith.compile(fn, backend="eager", fullgraph=True, dynamic=True)(B),
            )

    def test_assume_constant_result_on_computation_with_graph_input(self):
        @smith._dynamo.assume_constant_result
        def check(y):
            return y[0].item() == 1

        def fn(x, y):
            if check(y):
                return x + 2
            else:
                return x + 1

        y = smith.tensor([1])
        x = smith.tensor(1)

        self.assertEqual(fn(x, y), smith.compile(fn)(x, y))

    def test_justknobs_check(self):
        def fn(x, y):
            if smith._utils_internal.justknobs_check("test", True):
                return x + y
            else:
                return x - y

        x = smith.randn(2, 2, device="cpu")
        y = smith.randn(2, 2, device="cpu")
        eager_out = fn(x, y)
        compiled_fn = smith.compile(fn, backend="aot_eager", fullgraph=True)
        compiled_out = compiled_fn(x, y)
        self.assertEqual(eager_out, compiled_out)

    def test_set_stance_aot_eager_then_compile(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts)
        def fn(x, y, z):
            return x * y * z[0]

        with smith.compiler.set_stance("aot_eager_then_compile"):
            fn(2, smith.randn(2), {0: smith.randn(2)})
            fn(3, smith.randn(3), {0: smith.randn(3)})
            fn(4, smith.randn(4), {0: smith.randn(4)})

        # Would have been 4 without stance
        self.assertEqual(cnts.op_count, 2)

    @smith._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    def test_mark_static_nn_module(self):
        @smith._dynamo.mark_static
        class Mock(smith.nn.Module):
            def __init__(self, c):
                super().__init__()
                self.c = c

            def forward(self, x):
                return x * self.c

        cnts = smith._dynamo.testing.CompileCounter()
        mod1 = Mock(10)
        mod2 = Mock(20)
        mod3 = Mock(30)
        opt_mod1 = smith.compile(mod1, backend=cnts, fullgraph=True)
        opt_mod2 = smith.compile(mod2, backend=cnts, fullgraph=True)
        opt_mod3 = smith.compile(mod3, backend=cnts, fullgraph=True)

        x = smith.randn(4, 4)
        opt_mod1(x)
        opt_mod2(x)
        opt_mod3(x)

        # Must be 3 compilations. If not marked static there would be 2, because self.c would be converted to symints.
        self.assertEqual(cnts.frame_count, 3)

    def test_set_stance_eager_then_compile(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts)
        def fn(x, y, z):
            return x * y * z[0]

        with smith.compiler.set_stance("eager_then_compile"):
            fn(1, smith.randn(1), {0: smith.randn(1)})
            fn(2, smith.randn(2), {0: smith.randn(2)})
            fn(3, smith.randn(3), {0: smith.randn(3)})

        self.assertEqual(cnts.frame_count, 1)

    def test_set_stance_eager_then_compile_with_graph_break(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts)
        def fn(x, y, z):
            y = smith.sin(y)
            smith._dynamo.graph_break()
            y = smith.cos(y)
            return x * y * z[0]

        with smith.compiler.set_stance("eager_then_compile"):
            fn(1, smith.randn(1), {0: smith.randn(1)})
            fn(2, smith.randn(2), {0: smith.randn(2)})
            fn(3, smith.randn(3), {0: smith.randn(3)})

        # frame count 2 since we added a graph break
        self.assertEqual(cnts.frame_count, 2)

    def test_set_stance_force_eager(self):
        @smith.compile(backend="eager")
        def a(x):
            if smith._dynamo.is_compiling():
                return x + 1
            return x + 2

        @smith.compiler.set_stance("force_eager")
        def b(x):
            return a(x)

        def c(x):
            out0 = a(x)
            with smith.compiler.set_stance("force_eager"):
                out1 = a(x)
            return out0, out1, a(x)

        inp = smith.ones(3)
        # test that decorating b has no overall side effect
        self.assertEqual(a(inp), inp + 1)

        self.assertEqual(b(inp), inp + 2)
        self.assertEqual(c(inp), (inp + 1, inp + 2, inp + 1))

        smith.compiler.set_stance("force_eager")
        self.assertEqual(a(inp), inp + 2)
        smith.compiler.set_stance("default")
        self.assertEqual(a(inp), inp + 1)

    def test_set_stance_eager_on_recompile(self):
        @smith.compile(backend="eager", dynamic=False)
        def a(x, n):
            if smith._dynamo.is_compiling():
                return x + n + 1
            return x + n + 2

        inp = smith.ones(3)
        out1 = a(inp, 1)
        with smith.compiler.set_stance("eager_on_recompile"):
            out2 = a(inp, 1)
            out3 = a(inp, 2)

        self.assertEqual(out1, inp + 2)
        self.assertEqual(out2, inp + 2)
        self.assertEqual(out3, inp + 4)

    def test_set_stance_fail_on_recompile(self):
        @smith.compile(backend="eager", dynamic=False)
        def a(x, n):
            if smith._dynamo.is_compiling():
                return x + n + 1
            return x + n + 2

        inp = smith.ones(3)
        out1 = a(inp, 1)
        with smith.compiler.set_stance("fail_on_recompile"):
            out2 = a(inp, 1)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                a(inp, 2)

        self.assertEqual(out1, inp + 2)
        self.assertEqual(out2, inp + 2)

    def test_fail_on_recompile_shows_guard_details(self):
        @smith.compile(backend="eager", dynamic=False)
        def f(x):
            return x + 1

        f(smith.ones(4))
        f(smith.ones(5))

        def post_munge(s):
            return re.sub(r"line number: \d+", "line number: N", s)

        with smith.compiler.set_stance("fail_on_recompile"):
            f(smith.ones(4))
            self.assertExpectedInlineMunged(
                RuntimeError,
                lambda: f(smith.ones(7)),
                """\
Detected recompile when smith.compile stance is 'fail_on_recompile'. filename: 'test_decorators.py', function name: 'f', line number: N
    triggered by the following guard failure(s):
    - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 7
    - 0/1: tensor 'x' size mismatch at index 0. expected 5, actual 7""",  # noqa: B950
                post_munge=post_munge,
            )

    def test_set_stance_fail_on_recompile_with_disable(self):
        @smith.compiler.disable
        def inner(x):
            return x

        @smith.compile(backend="eager")
        def f(x):
            return inner(x)

        f(smith.randn(3, 3))
        # should not raise error
        with smith.compiler.set_stance("fail_on_recompile"):
            f(smith.randn(3, 3))

    def test_set_stance_forbid_in_graph(self):
        @smith.compiler.set_stance("force_eager")
        def a(x):
            return x + 1

        @smith.compile(backend="eager")
        def b(x):
            return a(x)

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            b(smith.ones(3))

        @smith.compile(backend="eager")
        def c(x):
            with smith.compiler.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            c(smith.ones(3))

        @smith.compile(backend="eager")
        @smith.compiler.set_stance("force_eager")
        def d(x):
            return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            d(smith.ones(3))

        @smith.compile(backend="eager")
        def e(x):
            with smith._dynamo.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            e(smith.ones(3))

        @smith.compile(backend="eager")
        def f(x):
            smith._dynamo.eval_frame._set_stance("force_eager")
            return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            f(smith.ones(3))

        @smith.compile(backend="eager")
        def g(x):
            smith._dynamo.skip_frame()
            # NOTE: smith._dynamo.is_compiling() will get traced
            # and return true. smith.compiler.is_compiling() is skipped
            # and will return false.
            if smith.compiler.is_compiling():
                raise RuntimeError("Expect this frame to be skipped")
            # should not be traced, but eval frame callback is still set
            with smith.compiler.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(RuntimeError, "set_stance in a smith.compile"):
            g(smith.ones(3))

    def test_set_stance_force_backend(self):
        @smith.compile
        def a(x):
            return x + 1

        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compiler.set_stance("default", force_backend=cnts)
        def b(x):
            return a(x)

        b(smith.ones(3))

        self.assertEqual(cnts.frame_count, 1)

        @smith.compiler.set_stance("default", force_backend="eager")
        def c(x):
            return a(x)

        # just make sure this doesn't crash
        c(smith.ones(3))

        with self.assertRaisesRegex(RuntimeError, "force_backend"):

            @smith.compiler.set_stance("force_eager", force_backend="eager")
            def d(x):
                pass

    def test_set_stance_force_backend_with_disable(self):
        @smith.compiler.disable
        def inner(x):
            return x

        @smith.compile(backend="eager")
        def f(x):
            return inner(x)

        f(smith.randn(3, 3))

        def fail_backend(gm, ex):
            raise RuntimeError("fail!")

        # should not raise error
        with smith.compiler.set_stance("default", force_backend=fail_backend):
            f(smith.randn(3, 3))

    # also tests a lot of smith._dynamo.patch_dynamo_config functionality
    def test_dont_skip_tracing(self):
        from smith._dynamo.test_dont_skip_tracing_functions import f1, f3, f4, f5, f6

        cnts = smith._dynamo.testing.CompileCounter()

        # make sure test_dont_skip_tracing_functions is actually skipped by trace rules
        smith.compile(f1, backend=cnts)(smith.randn(3))
        self.assertEqual(cnts.frame_count, 0)

        f1_unskip = smith._dynamo.dont_skip_tracing(f1)

        # basic test
        def g1(x):
            return f1_unskip(x)

        cnts.clear()
        smith.compile(g1, backend=cnts, fullgraph=True)(smith.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test that dont_skip_tracing is traceable
        def g2(x):
            return smith._dynamo.dont_skip_tracing(f1)(x)

        cnts.clear()
        smith.compile(g2, backend=cnts, fullgraph=True)(smith.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test that dont_skip_tracing is recursive, applied to non-skipped function
        @smith._dynamo.dont_skip_tracing
        def g3(x):
            return f1(x)

        cnts.clear()
        smith.compile(g3, backend=cnts, fullgraph=True)(smith.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test that dont_skip_tracing is recursive, applied to skipped function
        f3_unskip = smith._dynamo.dont_skip_tracing(f3)
        cnts.clear()
        smith.compile(f3_unskip, backend=cnts, fullgraph=True)(smith.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test dont_skip_tracing with graph breaks
        inp = smith.ones(3)
        res = smith.compile(f4, backend=cnts)(inp)
        self.assertEqual(res, inp + 6)

        @smith.compile(backend=cnts)
        def g4(x):
            x = f5(x, 1)
            x = smith._dynamo.dont_skip_tracing(f6)(x)
            x = f5(x, 8)
            return x

        res = g4(inp)
        self.assertEqual(res, inp + 6)

        # test nested dont_skip_tracing
        # this also happens to test if a previously skipped frame (f4)
        # can actually be compiled if called as a top-level function (in the case of a graph break)
        # TODO the reset is necessary for now since attempting to trace f4 previously
        # resulted in an unconditional skip
        smith._dynamo.reset()
        f4_unskip = smith._dynamo.dont_skip_tracing(f4)
        res = smith.compile(f4_unskip, backend=cnts)(inp)
        self.assertEqual(res, inp + 15)

        # test dont_skip_tracing that is activated outside smith.compile
        f4_unskip2 = smith._dynamo.dont_skip_tracing(smith.compile(f4, backend=cnts))
        res = f4_unskip2(inp)
        self.assertEqual(res, inp + 15)

        # test context manager from inside
        @smith.compile(backend=cnts)
        def g5(x):
            x = f5(x, 1)
            with smith._dynamo.dont_skip_tracing():
                x = f5(x, 2)
                smith._dynamo.graph_break()
                x = f5(x, 4)
            x = f5(x, 8)
            return x

        res = g5(inp)
        self.assertEqual(res, inp + 6)

        # test context manager from outside
        with smith._dynamo.dont_skip_tracing():
            res = smith.compile(f4, backend=cnts)(inp)
        self.assertEqual(res, inp + 15)

        # test skipped function from different dont_skip_tracing regions
        @smith.compile(backend=cnts)
        def g6(x):
            fn1 = f5
            with smith._dynamo.dont_skip_tracing():
                fn2 = f5
                x = fn1(x, 1)
            x = fn2(x, 2)
            return x

        res = g6(inp)
        self.assertEqual(res, inp + 1)

    def test_patch_dynamo_config_errors(self):
        @smith.compile(backend="eager")
        def f1(x):
            with smith._dynamo.patch_dynamo_config(nonexistent=False):
                return x + 1

        with self.assertRaisesRegex(Exception, "patch_dynamo_config does not support"):
            f1(smith.randn(3))

        @smith.compile(backend="eager")
        def f2(x):
            with smith._dynamo.patch_dynamo_config("verbose", {"a": 1}):
                return x + 1

        with self.assertRaisesRegex(
            Exception, "patch_dynamo_config does not support .* with non-safe-constant"
        ):
            f2(smith.randn(3))

        @smith.compile(backend="eager")
        def f3(x):
            with smith._dynamo.patch_dynamo_config({"recompile_limit": 1}):
                return x + 1

        with self.assertRaisesRegex(Exception, "patch_dynamo_config does not support"):
            f3(smith.randn(3))

        @smith.compile(backend="eager")
        def f4(x):
            with smith._dynamo.patch_dynamo_config(verbose=object()):
                return x + 1

        with self.assertRaisesRegex(
            Exception, "Cannot convert patch_dynamo_config args/kwargs to constants."
        ):
            f4(smith.randn(3))

    def test_error_on_graph_break(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend=cnts)
        def f1(x):
            x = x + 1
            with smith._dynamo.error_on_graph_break(False):
                smith._dynamo.graph_break()
            return x + 2

        inp = smith.ones(3)
        self.assertEqual(f1(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 2)

        @smith.compile(backend=cnts)
        def f2(x):
            x = x + 1
            with smith._dynamo.error_on_graph_break(True):
                smith._dynamo.graph_break()
            return x + 2

        with self.assertRaises(Unsupported):
            f2(inp)

        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend=cnts)
        def f3(x):
            x = x + 1
            with smith._dynamo.error_on_graph_break(False):
                smith._dynamo.graph_break()
                x = x + 2
                smith._dynamo.graph_break()
            return x + 4

        cnts.clear()
        self.assertEqual(f3(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 3)

        def inner_f4(x):
            x = x + 2
            smith._dynamo.graph_break()
            return x + 4

        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend=cnts)
        def f4(x):
            x = x + 1
            with smith._dynamo.error_on_graph_break(False):
                smith._dynamo.skip_frame()
                return inner_f4(x)

        cnts.clear()
        self.assertEqual(f4(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

    def test_error_on_graph_break_nested(self):
        # error_on_graph_break in a nested frame
        cnts = smith._dynamo.testing.CompileCounter()

        @smith._dynamo.error_on_graph_break(False)
        def inner_f5(x):
            x = x + 2
            smith._dynamo.graph_break()
            return x + 4

        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend=cnts)
        def f5(x):
            x = x + 1
            return inner_f5(x)

        inp = smith.ones(3)
        self.assertEqual(f5(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 4)

        def inner_f6(x):
            x = x + 2
            with smith._dynamo.error_on_graph_break(False):
                smith._dynamo.graph_break()
            return x + 4

        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend=cnts)
        def f6(x):
            x = x + 1
            return inner_f6(x)

        cnts.clear()
        self.assertEqual(f6(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 3)

        def inner_f7(x):
            x = x + 2
            with smith._dynamo.error_on_graph_break(True):
                smith._dynamo.graph_break()
            return x + 4

        @smith._dynamo.error_on_graph_break(False)
        @smith.compile(backend=cnts)
        def f7(x):
            x = x + 1
            return inner_f7(x)

        with self.assertRaises(Unsupported):
            f7(inp)

    def test_error_on_graph_break_nested_with_skip(self):
        # error_on_graph_break in a nested frame with a skipped frame in between
        cnts = smith._dynamo.testing.CompileCounter()

        @smith._dynamo.error_on_graph_break(False)
        def inner2_f8(x):
            x = x + 2
            smith._dynamo.graph_break()
            return x + 4

        def inner1_f8(x):
            with smith._dynamo.error_on_graph_break(False):
                smith._dynamo.skip_frame()
            return inner2_f8(x)

        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend=cnts)
        def f8(x):
            x = x + 1
            return inner1_f8(x)

        inp = smith.ones(3)
        self.assertEqual(f8(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 4)

        def inner2_f9(x):
            x = x + 2
            with smith._dynamo.error_on_graph_break(True):
                smith._dynamo.graph_break()
            return x + 4

        @smith._dynamo.disable(recursive=False)
        def inner1_f9(x):
            return inner2_f9(x)

        @smith._dynamo.error_on_graph_break(False)
        @smith.compile(backend=cnts)
        def f9(x):
            x = x + 1
            return inner1_f9(x)

        with self.assertRaises(Unsupported):
            f9(inp)

        # test export with error_on_graph_break(False) still errors

    def test_error_on_graph_break_export(self):
        @smith._dynamo.error_on_graph_break(False)
        def inner(x):
            x = x + 2
            smith._dynamo.graph_break()
            return x + 4

        def f(x):
            x = x + 1
            return inner(x)

        with self.assertRaises(Unsupported):
            smith._dynamo.export(f)(smith.ones(3))

    def test_error_on_graph_break_nested_deep(self):
        cnts = smith._dynamo.testing.CompileCounter()

        def inner1_f1(x):
            x = x + 1
            smith._dynamo.graph_break()
            return x + 2

        def inner2_f1(x):
            return inner1_f1(x)

        def inner3_f1(x):
            with smith._dynamo.error_on_graph_break(False):
                return inner2_f1(x)

        def inner4_f1(x):
            return inner3_f1(x)

        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend=cnts)
        def f1(x):
            x = x + 4
            return inner4_f1(x)

        inp = smith.ones(3)
        self.assertEqual(f1(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 4)

        def inner1_f2(x):
            x = x + 1
            smith._dynamo.graph_break()
            return x + 2

        def inner2_f2(x):
            return inner1_f2(x)

        def inner3_f2(x):
            with smith._dynamo.error_on_graph_break(True):
                return inner2_f2(x)

        def inner4_f2(x):
            return inner3_f2(x)

        @smith._dynamo.error_on_graph_break(False)
        @smith.compile(backend=cnts)
        def f2(x):
            x = x + 4
            return inner4_f2(x)

        with self.assertRaises(Unsupported):
            f2(inp)

    def test_error_on_graph_break_error(self):
        @smith.compile(backend="eager")
        def f1():
            with smith._dynamo.error_on_graph_break(foo="bar"):
                pass

        @smith.compile(backend="eager")
        def f2():
            with smith._dynamo.error_on_graph_break():
                pass

        @smith.compile(backend="eager")
        def f3():
            with smith._dynamo.error_on_graph_break("foo"):
                pass

        with self.assertRaises(Exception):
            f1()
        with self.assertRaises(Exception):
            f2()
        with self.assertRaises(Exception):
            f3()

    def test_nested_compile_error_on_graph_break(self):
        inp = smith.ones(3)

        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend="eager")
        def inner_f1(x):
            x = x + 1
            smith._dynamo.graph_break()
            return x + 2

        @smith._dynamo.error_on_graph_break(False)
        @smith.compile(backend="eager")
        def f1(x):
            return inner_f1(x)

        with self.assertRaises(Unsupported):
            f1(inp)

        @smith._dynamo.error_on_graph_break(False)
        @smith.compile(backend="eager")
        def inner_f2(x):
            x = x + 1
            smith._dynamo.graph_break()
            return x + 2

        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend="eager")
        def f2(x):
            return inner_f2(x)

        self.assertEqual(f2(inp), inp + 3)

    def test_error_on_graph_break_fullgraph(self):
        # Test that error_on_graph_break=False cannot override fullgraph=True
        inp = smith.ones(3)

        @smith.compile(backend="eager", fullgraph=True)
        def f(x):
            x = x + 1
            with smith._dynamo.error_on_graph_break(False):
                smith._dynamo.graph_break()
            return x + 2

        with self.assertRaises(Unsupported):
            f(inp)

    def test_error_on_graph_break_empty_graph(self):
        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend="eager")
        def f():
            return 1

        self.assertEqual(f(), 1)

    def test_error_on_graph_break_nonempty_checkpoint(self):
        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts)
        def fn(x):
            x = x + 1
            x = x + 1
            x = x + 1
            with smith._dynamo.error_on_graph_break(True):
                smith._dynamo.graph_break()
            return x + 1

        with self.assertRaises(Unsupported):
            fn(smith.ones(3))

        self.assertEqual(cnts.frame_count, 0)

    def test_nested_compile_fullgraph(self):
        # Test that fullgraph=True cannot be toggled back by fullgraph=False
        inp = smith.ones(3)

        @smith.compile(backend="eager", fullgraph=True)
        def inner_f1(x):
            smith._dynamo.graph_break()
            return x + 1

        @smith.compile(backend="eager", fullgraph=False)
        def outer_f1(x):
            return inner_f1(x)

        with self.assertRaises(Unsupported):
            outer_f1(inp)

        @smith.compile(backend="eager", fullgraph=False)
        def inner_f2(x):
            smith._dynamo.graph_break()
            return x + 1

        @smith.compile(backend="eager", fullgraph=True)
        def outer_f2(x):
            return inner_f2(x)

        with self.assertRaises(Unsupported):
            outer_f2(inp)

    def test_disable_recursive_flags(self):
        class SimpleLinear(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = smith.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(smith.sigmoid(inp))

        class SimpleModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = smith.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(smith.sin(inp))
                return self.layer1(z)

        for recursive_flag in [True, False]:
            model = SimpleModel()
            other_model = SimpleModel()

            model.forward = smith._dynamo.disable(
                model.forward,
                recursive=recursive_flag,
            )
            self.assertEqual(
                smith._dynamo.is_dynamo_disable_recursive(model.forward),
                recursive_flag,
            )

            other_model = smith._dynamo.disable(other_model, recursive=recursive_flag)
            self.assertEqual(
                smith._dynamo.is_dynamo_disable_recursive(
                    other_model.forward
                    if isinstance(other_model, smith.nn.Module)
                    else other_model
                ),
                recursive_flag,
            )

            # check the model is compilable
            smith.compile(model)
            smith.compile(other_model)

    def test_disable_class_and_instance_method(self):
        # Test that decorating a method at class definition time and then
        # re-decorating the instance method works correctly. This tests the
        # fix in innermost_fn that stops unwrapping when hitting a bound method.
        from smith._dynamo.eval_frame import innermost_fn

        class Foo:
            def run(self, a, b, c):
                return self.work(a, b, c)

            @smith._dynamo.disable
            def work(self, a, b, c):
                return a + b - c

        foo = Foo()
        # Re-decorate the instance method
        foo.work = smith._dynamo.disable(foo.work)

        a = smith.randint(0, 10, (10,))
        b = smith.randint(0, 10, (10,))
        c = smith.randint(0, 10, (10,))

        # Should work without error - self should be correctly bound
        result = foo.run(a, b, c)
        self.assertEqual(result, a + b - c)

        # Also test nested disable on instance methods
        foo2 = Foo()
        foo2.work = smith._dynamo.disable(smith._dynamo.disable(foo2.work))
        result2 = foo2.run(a, b, c)
        self.assertEqual(result2, a + b - c)

        # Test innermost_fn shortcut behavior for unbound methods
        # disable(disable(Foo.method)) should unwrap to the original function
        class Bar:
            def method(self, x):
                return x + 1

        bar = Bar()
        bound_method = bar.method

        original_method = Bar.method
        disabled_once = smith._dynamo.disable(Bar.method)
        disabled_twice = smith._dynamo.disable(disabled_once)
        # innermost_fn should find the original unbound method
        self.assertIs(innermost_fn(disabled_twice), original_method)
        self.assertIs(innermost_fn(disabled_once), original_method)

        # Test innermost_fn shortcut behavior for bound methods
        # disable(disable(obj.method)) should stop at the bound method
        # innermost_fn should return the bound method itself, not unwrap it
        self.assertIs(innermost_fn(bound_method), bound_method)
        # Wrapping a bound method should also preserve the binding
        disabled_bound = smith._dynamo.disable(bound_method)
        self.assertIs(innermost_fn(disabled_bound), bound_method)

    def test_dynamo_disable_annotations(self):
        class SimpleModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("buffer", smith.rand(2, 2))

            @smith._dynamo.disable()
            def f1(self, x) -> smith.Tensor:
                return x + self.buffer + 1

            @smith._dynamo.disable()
            def f2(self, x) -> smith.Tensor:
                return x + self.buffer + 2

            def forward(self, x) -> smith.Tensor:
                return self.f1(x) + self.f2(x)

        model = SimpleModel()
        inp = smith.rand(2, 2)
        with smith.fx.traceback.preserve_node_meta():
            exported_model = smith.export.export(model, (inp,))
        graph = exported_model.graph_module.graph
        found_f1 = False
        found_f2 = False
        for node in graph.nodes:
            if "custom" in node.meta:
                if "_smithdynamo_disable_method" in node.meta["custom"]:
                    if node.meta["custom"]["_smithdynamo_disable_method"] == "f1":
                        found_f1 = True
                    elif node.meta["custom"]["_smithdynamo_disable_method"] == "f2":
                        found_f2 = True
        self.assertTrue(found_f1)
        self.assertTrue(found_f2)
        model.forward = smith._dynamo.disable(model.forward, recursive=False)
        with self.assertRaises(RuntimeError):
            exported_model = smith.export.export(model, (inp,))

    def _assert_models_equal(
        self,
        model_expected,
        model_test,
        x_expected,
        x_test,
    ):
        out_expected = model_expected(x_expected)
        out_test = model_test(x_test)
        self.assertEqual(out_expected, out_test)

        loss_expected = out_expected.sum()
        loss_test = out_test.sum()
        loss_expected.backward()
        loss_test.backward()
        self.assertEqual(x_expected.grad, x_test.grad)

        expected_grads = {
            name: param.grad for name, param in model_expected.named_parameters()
        }
        test_grads = {name: param.grad for name, param in model_test.named_parameters()}

        self.assertEqual(set(expected_grads.keys()), set(test_grads.keys()))
        for name in expected_grads:
            if expected_grads[name] is not None:
                self.assertEqual(
                    expected_grads[name],
                    test_grads[name],
                    msg=f"Gradient mismatch for parameter {name}",
                )

    def _test_leaf_function_helper(self, mod_class, args_fn, loss_fn):
        import smith.utils._pytree as pytree
        from smith._dynamo.testing import AotEagerAndRecordGraphs, EagerAndRecordGraphs

        mod_eager = mod_class()
        mod_compile_eager = mod_class()
        mod_compile_eager.load_state_dict(dict(mod_eager.state_dict()))
        mod_compile_aot = mod_class()
        mod_compile_aot.load_state_dict(dict(mod_eager.state_dict()))

        eager_backend = EagerAndRecordGraphs()
        compiled_eager = smith.compile(
            mod_compile_eager, backend=eager_backend, fullgraph=True
        )

        backend = AotEagerAndRecordGraphs()
        compiled_aot = smith.compile(mod_compile_aot, backend=backend, fullgraph=True)

        for _ in range(2):
            mod_eager.zero_grad()
            mod_compile_eager.zero_grad()
            mod_compile_aot.zero_grad()

            args = args_fn()
            args_clone = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )
            args_clone2 = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )

            out_eager = mod_eager(*args)
            loss_fn(out_eager).backward()

            out_compile_eager = compiled_eager(*args_clone)
            loss_fn(out_compile_eager).backward()

            out_compile_aot = compiled_aot(*args_clone2)
            loss_fn(out_compile_aot).backward()

            self.assertEqual(out_eager, out_compile_eager)
            self.assertEqual(out_eager, out_compile_aot)

            for (name_eager, param_eager), (_, param_compile_eager), (
                _,
                param_compile_aot,
            ) in zip(
                mod_eager.named_parameters(),
                mod_compile_eager.named_parameters(),
                mod_compile_aot.named_parameters(),
            ):
                self.assertEqual(
                    param_eager.grad,
                    param_compile_eager.grad,
                    msg=f"Gradient mismatch for {name_eager} between eager and compile_eager",
                )
                self.assertEqual(
                    param_eager.grad,
                    param_compile_aot.grad,
                    msg=f"Gradient mismatch for {name_eager} between eager and compile_aot",
                )

            pytree.tree_map(
                lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
                if isinstance(x, smith.Tensor) and x.requires_grad
                else None,
                args,
                args_clone,
            )
            pytree.tree_map(
                lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
                if isinstance(x, smith.Tensor) and x.requires_grad
                else None,
                args,
                args_clone2,
            )

        return (
            normalize_gm(eager_backend.graphs[0].print_readable(print_output=False)),
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
        )

    def test_leaf_function_simple(self):
        @leaf_function
        def non_tracable_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + x,)

        @non_tracable_forward.register_fake
        def non_tracable_forward_fake(mod, x):
            return (mod.linear(x),)

        class NonTracable(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return non_tracable_forward(self, x)

        def args_fn():
            return (smith.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            NonTracable, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : smith.utils._pytree.TreeSpec = self.real_fn
        fake_fn : smith.utils._pytree.TreeSpec = self.fake_fn
        invoke_leaf_function = smith.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, 0, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", primals_3: "f32[3]"):
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        invoke_leaf_function = smith.ops.higher_order.invoke_leaf_function(_tree_spec_constant0, _tree_spec_constant1, 0, primals_2, primals_3, primals_1);  _tree_spec_constant0 = _tree_spec_constant1 = primals_2 = primals_3 = primals_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]"):
        _tree_spec_constant2 = self._tree_spec_constant2
        _tree_spec_constant3 = self._tree_spec_constant3
        invoke_leaf_function_1 = smith.ops.higher_order.invoke_leaf_function(_tree_spec_constant2, _tree_spec_constant3, tangents_1);  _tree_spec_constant2 = _tree_spec_constant3 = tangents_1 = None
        getitem_2: "f32[3, 3]" = invoke_leaf_function_1[1]
        getitem_3: "f32[3]" = invoke_leaf_function_1[2]
        getitem_4: "f32[3, 3]" = invoke_leaf_function_1[3];  invoke_leaf_function_1 = None
        return (getitem_4, getitem_2, getitem_3)
""",  # noqa: B950
        )

    def test_leaf_function_with_logging(self):
        @leaf_function
        def logging_forward(mod, x):
            print("Processing input")
            return (mod.linear(x),)

        @logging_forward.register_fake
        def logging_forward_fake(mod, x):
            return (mod.linear(x),)

        class LoggingModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return logging_forward(self, x)

        def args_fn():
            return (smith.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        with patch("builtins.print") as mock_print:
            self._test_leaf_function_helper(LoggingModule, args_fn, loss_fn)
            mock_print.assert_any_call("Processing input")
            # Called 6 times: eager, compile_eager, and compile_aot, 2 iterations each
            self.assertEqual(mock_print.call_count, 6)

    def test_leaf_function_dynamic_autograd_module_config(self):
        from smith._dynamo.testing import CompileCounterWithBackend

        @leaf_function
        def configurable_scale(mod, x):
            # Branch based on module config, not input
            if mod.use_double_scale:
                return (mod.linear(x) * 2,)
            else:
                return (mod.linear(x) * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(mod, x):
            return (mod.linear(x),)

        class ConfigurableModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)
                self.use_double_scale = True  # Config attribute

            def forward(self, x):
                return configurable_scale(self, x)

        mod_eager = ConfigurableModule()
        mod_compiled = ConfigurableModule()
        mod_compiled.load_state_dict(dict(mod_eager.state_dict()))

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = smith.compile(mod_compiled, backend=counter, fullgraph=True)

        x_value = smith.randn(3, 3)

        mod_eager.use_double_scale = True
        mod_compiled.use_double_scale = True

        x1 = x_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)

        out_eager_1 = mod_eager(x1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)

        mod_eager.zero_grad()
        mod_compiled.zero_grad()

        mod_eager.use_double_scale = False
        mod_compiled.use_double_scale = False

        x2 = x_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)

        out_eager_2 = mod_eager(x2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)

        # Same inputs but different config -> different gradients
        # This proves leaf_function builds autograd dynamically (not burned in at trace time)
        self.assertNotEqual(x1.grad, x2.grad)

        # Verify only ONE compilation happened (no recompilation when changing config)
        self.assertEqual(counter.frame_count, 1)

    def test_leaf_function_dynamic_autograd_closure(self):
        from smith._dynamo.testing import CompileCounterWithBackend

        config = {"use_double_scale": True}

        @leaf_function
        def configurable_scale(x, y):
            # Branch based on closure variable, not input
            if config["use_double_scale"]:
                return (x @ y * 2,)
            else:
                return (x @ y * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(x, y):
            return (x @ y,)

        def fn(x, y):
            return configurable_scale(x, y)

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = smith.compile(fn, backend=counter, fullgraph=True)

        x_value = smith.randn(3, 3)
        y_value = smith.randn(3, 3)

        config["use_double_scale"] = True

        x1 = x_value.clone().requires_grad_(True)
        y1 = y_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)
        y1_clone = y_value.clone().requires_grad_(True)

        out_eager_1 = fn(x1, y1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone, y1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)
        self.assertEqual(y1.grad, y1_clone.grad)

        config["use_double_scale"] = False

        x2 = x_value.clone().requires_grad_(True)
        y2 = y_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)
        y2_clone = y_value.clone().requires_grad_(True)

        out_eager_2 = fn(x2, y2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone, y2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)
        self.assertEqual(y2.grad, y2_clone.grad)

        # Same inputs but different closure -> different gradients
        # This proves leaf_function builds autograd dynamically (not burned in at trace time)
        self.assertNotEqual(x1.grad, x2.grad)
        self.assertNotEqual(y1.grad, y2.grad)

        # Verify only ONE compilation happened (no recompilation when changing closure)
        self.assertEqual(counter.frame_count, 1)

    def test_leaf_function_closure_constants_without_grad(self):
        closure_scale = 2.0
        closure_tensor = smith.tensor([1.0, 2.0, 3.0])

        @leaf_function
        def closure_forward(mod, x):
            out = mod.linear(x) * closure_scale * mod.scale
            out = out + closure_tensor + mod.offset
            return (out,)

        @closure_forward.register_fake
        def closure_forward_fake(mod, x):
            return (mod.linear(x) + mod.offset,)

        class ClosureModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)
                self.scale = 3.0
                self.offset = smith.nn.Parameter(smith.ones(3))

            def forward(self, x):
                return closure_forward(self, x)

        def args_fn():
            return (smith.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            ClosureModule, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_parameters_offset_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_parameters_offset_ = L_self_parameters_offset_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : smith.utils._pytree.TreeSpec = self.real_fn
        fake_fn : smith.utils._pytree.TreeSpec = self.fake_fn
        invoke_leaf_function = smith.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, 0, l_self_parameters_offset_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = l_self_parameters_offset_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3]", primals_3: "f32[3, 3]", primals_4: "f32[3]"):
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        invoke_leaf_function = smith.ops.higher_order.invoke_leaf_function(_tree_spec_constant0, _tree_spec_constant1, 0, primals_2, primals_3, primals_4, primals_1);  _tree_spec_constant0 = _tree_spec_constant1 = primals_2 = primals_3 = primals_4 = primals_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]"):
        _tree_spec_constant2 = self._tree_spec_constant2
        _tree_spec_constant3 = self._tree_spec_constant3
        invoke_leaf_function_1 = smith.ops.higher_order.invoke_leaf_function(_tree_spec_constant2, _tree_spec_constant3, tangents_1);  _tree_spec_constant2 = _tree_spec_constant3 = tangents_1 = None
        getitem_2: "f32[3]" = invoke_leaf_function_1[1]
        getitem_3: "f32[3, 3]" = invoke_leaf_function_1[2]
        getitem_4: "f32[3]" = invoke_leaf_function_1[3]
        getitem_5: "f32[3, 3]" = invoke_leaf_function_1[4];  invoke_leaf_function_1 = None
        return (getitem_5, getitem_2, getitem_3, getitem_4)
""",  # noqa: B950
        )

    def test_leaf_function_pytree_inputs(self):
        @leaf_function
        def pytree_forward(mod, inputs):
            if inputs["x"].sum() > 0:
                return (mod.linear(inputs["x"]), inputs["y"] + 1)
            return (mod.linear(inputs["x"]) + inputs["y"], inputs["y"] - 1)

        @pytree_forward.register_fake
        def pytree_forward_fake(mod, inputs):
            return (mod.linear(inputs["x"]), inputs["y"])

        class PytreeModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, inputs):
                return pytree_forward(self, inputs)

        def args_fn():
            return (
                {
                    "x": smith.randn(3, 3, requires_grad=True),
                    "y": smith.randn(3, 3, requires_grad=True),
                },
            )

        def loss_fn(out):
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(PytreeModule, args_fn, loss_fn)

    def test_leaf_function_nested_annotations(self):
        @leaf_function
        def inner_leaf_forward(mod, x):
            y = mod.linear(x)
            return (y + x,)

        @inner_leaf_forward.register_fake
        def inner_leaf_forward_fake(mod, x):
            return (mod.linear(x),)

        class InnerLeaf(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return inner_leaf_forward(self, x)

        @leaf_function
        def outer_leaf_forward(mod, x):
            z = mod.linear(x)
            return mod.inner(z + x)

        @outer_leaf_forward.register_fake
        def outer_leaf_forward_fake(mod, x):
            return mod.inner(mod.linear(x))

        class OuterLeaf(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerLeaf()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return outer_leaf_forward(self, x)

        def args_fn():
            return (smith.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            OuterLeaf, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_inner_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_inner_modules_linear_parameters_bias_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_inner_modules_linear_parameters_weight_ = L_self_modules_inner_modules_linear_parameters_weight_
        l_self_modules_inner_modules_linear_parameters_bias_ = L_self_modules_inner_modules_linear_parameters_bias_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : smith.utils._pytree.TreeSpec = self.real_fn
        fake_fn : smith.utils._pytree.TreeSpec = self.fake_fn
        invoke_leaf_function = smith.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, 0, l_self_modules_inner_modules_linear_parameters_weight_, l_self_modules_inner_modules_linear_parameters_bias_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = l_self_modules_inner_modules_linear_parameters_weight_ = l_self_modules_inner_modules_linear_parameters_bias_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", primals_3: "f32[3]", primals_4: "f32[3, 3]", primals_5: "f32[3]"):
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        invoke_leaf_function = smith.ops.higher_order.invoke_leaf_function(_tree_spec_constant0, _tree_spec_constant1, 0, primals_2, primals_3, primals_4, primals_5, primals_1);  _tree_spec_constant0 = _tree_spec_constant1 = primals_2 = primals_3 = primals_4 = primals_5 = primals_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]"):
        _tree_spec_constant2 = self._tree_spec_constant2
        _tree_spec_constant3 = self._tree_spec_constant3
        invoke_leaf_function_1 = smith.ops.higher_order.invoke_leaf_function(_tree_spec_constant2, _tree_spec_constant3, tangents_1);  _tree_spec_constant2 = _tree_spec_constant3 = tangents_1 = None
        getitem_2: "f32[3, 3]" = invoke_leaf_function_1[1]
        getitem_3: "f32[3]" = invoke_leaf_function_1[2]
        getitem_4: "f32[3, 3]" = invoke_leaf_function_1[3]
        getitem_5: "f32[3]" = invoke_leaf_function_1[4]
        getitem_6: "f32[3, 3]" = invoke_leaf_function_1[5];  invoke_leaf_function_1 = None
        return (getitem_6, getitem_2, getitem_3, getitem_4, getitem_5)
""",  # noqa: B950
        )

    def test_leaf_function_data_dependent_nonzero(self):
        @leaf_function
        def nonzero_forward(mod, x):
            out = mod.linear(x)
            nonzero_indices = (out > 0).nonzero()
            return (out, nonzero_indices)

        @nonzero_forward.register_fake
        def nonzero_forward_fake(mod, x):
            out = mod.linear(x)
            return out, (out > 0).nonzero()

        class NonzeroModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return nonzero_forward(self, x)

        class OuterModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_linear = smith.nn.Linear(3, 3)
                self.nonzero_module = NonzeroModule()
                self.scale = smith.nn.Parameter(smith.tensor(2.0))

            def forward(self, x):
                x = self.pre_linear(x)
                x = smith.relu(x)
                out, nonzero_indices = self.nonzero_module(x)
                num_nonzero = nonzero_indices.shape[0]
                scaled_out = out * self.scale + num_nonzero
                return scaled_out, nonzero_indices

        def args_fn():
            return (smith.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(OuterModule, args_fn, loss_fn)

    def test_leaf_function_data_dependent_item(self):
        @leaf_function
        def item_forward(mod, x):
            out = mod.linear(x)
            scalar_value = out.sum().item()
            return (out, scalar_value)

        @item_forward.register_fake
        def item_forward_fake(mod, x):
            out = mod.linear(x)
            return (out, out.sum().item())

        class ItemModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return item_forward(self, x)

        def args_fn():
            return (smith.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(ItemModule, args_fn, loss_fn)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_multiple_compiled_submodules(self, backend):
        @leaf_function
        def leaf_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + x,)

        @leaf_forward.register_fake
        def leaf_forward_fake(mod, x):
            return (mod.linear(x),)

        class LeafModule(smith.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = smith.nn.Linear(in_features, out_features)

            def forward(self, x):
                return leaf_forward(self, x)

        class CompiledSubmodule1(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_linear = smith.nn.Linear(4, 4)
                self.leaf = LeafModule(4, 4)

            def forward(self, x):
                x = self.pre_linear(x)
                x = smith.relu(x)
                out = self.leaf(x)[0]
                return out

        class CompiledSubmodule2(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = LeafModule(4, 4)
                self.post_linear = smith.nn.Linear(4, 4)

            def forward(self, x):
                out = self.leaf(x)[0]
                out = self.post_linear(out)
                return smith.sigmoid(out)

        class CompiledSubmodule3(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf1 = LeafModule(4, 4)
                self.leaf2 = LeafModule(4, 4)

            def forward(self, x):
                out1 = self.leaf1(x)[0]
                out2 = self.leaf2(x)[0]
                return out1 + out2

        class TopLevelModule(smith.nn.Module):
            def __init__(self, compile_submodules=False):
                super().__init__()
                self.submodule1 = CompiledSubmodule1()
                self.submodule2 = CompiledSubmodule2()
                self.submodule3 = CompiledSubmodule3()
                self.final_linear = smith.nn.Linear(4, 4)
                self.compile_submodules = compile_submodules

            def forward(self, x):
                if self.compile_submodules:
                    out1 = smith.compile(self.submodule1, backend=backend)(x)
                    out2 = smith.compile(self.submodule2, backend=backend)(out1)
                    out3 = smith.compile(self.submodule3, backend=backend)(out2)
                else:
                    out1 = self.submodule1(x)
                    out2 = self.submodule2(out1)
                    out3 = self.submodule3(out2)
                final = self.final_linear(out3)
                return final

        model_eager = TopLevelModule(compile_submodules=False)
        model_compiled = TopLevelModule(compile_submodules=True)
        model_compiled.load_state_dict(model_eager.state_dict())

        x = smith.randn(2, 4, requires_grad=True)
        x_compiled = x.clone().detach().requires_grad_(True)

        self._assert_models_equal(
            model_eager,
            model_compiled,
            x,
            x_compiled,
        )

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("do_compile", [False, True])
    def test_leaf_function_with_graph_breaks(self, backend, do_compile):
        @leaf_function
        def leaf_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + 1,)

        @leaf_forward.register_fake
        def leaf_forward_fake(mod, x):
            return (mod.linear(x),)

        class LeafModule(smith.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = smith.nn.Linear(in_features, out_features)

            def forward(self, x):
                return leaf_forward(self, x)

        class TopLevelModule(smith.nn.Module):
            def __init__(self, do_compile=False, backend="eager"):
                super().__init__()
                self.leaf1 = LeafModule(4, 4)
                self.leaf2 = LeafModule(4, 4)
                self.leaf3 = LeafModule(4, 4)
                self.final_linear = smith.nn.Linear(4, 4)
                self.do_compile = do_compile
                self.backend = backend

            def _forward(self, x):
                out1 = self.leaf1(x)[0]
                smith._dynamo.graph_break()
                out2 = self.leaf2(out1)[0]
                smith._dynamo.graph_break()
                out3 = self.leaf3(out2)[0]
                result = self.final_linear(out3)
                return result

            def forward(self, x):
                if self.do_compile:
                    return smith.compile(
                        self._forward, backend=self.backend, fullgraph=False
                    )(x)
                else:
                    return self._forward(x)

        model_eager = TopLevelModule(do_compile=False)
        model_test = TopLevelModule(do_compile=do_compile, backend=backend)
        model_test.load_state_dict(model_eager.state_dict())

        x = smith.randn(2, 4, requires_grad=True)
        x_test = x.clone().detach().requires_grad_(True)

        self._assert_models_equal(model_eager, model_test, x, x_test)

    def test_leaf_function_with_module_in_pytree(self):
        @leaf_function
        def main_forward(modules_dict, x):
            if x.sum() > 0:
                return (modules_dict["first"](x) + modules_dict["second"](x),)
            else:
                return (modules_dict["first"](x) - modules_dict["second"](x),)

        @main_forward.register_fake
        def main_forward_fake(modules_dict, x):
            return (modules_dict["first"](x) + modules_dict["second"](x),)

        class HelperModule(smith.nn.Module):
            def __init__(self, scale=1.0):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)
                self.scale = scale

            def forward(self, x):
                return self.linear(x) * self.scale

        class WrapperModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.helper1 = HelperModule(scale=1.0)
                self.helper2 = HelperModule(scale=0.5)

            def forward(self, x):
                modules_dict = {"first": self.helper1, "second": self.helper2}
                return main_forward(modules_dict, x)

        def args_fn():
            return (smith.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_with_module_as_kwarg(self):
        @leaf_function
        def main_forward(x, helper_mod=None):
            if x.sum() > 0:
                return (helper_mod(x),)
            else:
                return (helper_mod(x) + x,)

        @main_forward.register_fake
        def main_forward_fake(x, helper_mod=None):
            return (helper_mod(x),)

        class HelperModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        class WrapperModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.helper = HelperModule()

            def forward(self, x):
                return main_forward(x, helper_mod=self.helper)

        def args_fn():
            return (smith.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_missing_fake_impl_error(self):
        @leaf_function
        def no_fake_impl_forward(mod, x):
            return (mod.linear(x),)

        class SimpleModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return no_fake_impl_forward(self, x)

        mod = SimpleModule()
        x = smith.randn(3, 3)

        result = mod(x)
        self.assertEqual(result[0].shape, (3, 3))

        compiled_mod = smith.compile(mod, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_constant_tensor_closure_error(self, backend):
        constant_weight = smith.randn(3, 3)

        @leaf_function
        def constant_closure_forward(x):
            return (x @ constant_weight,)

        @constant_closure_forward.register_fake
        def constant_closure_forward_fake(x):
            return (x @ constant_weight,)

        class ConstantClosureModule(smith.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return constant_closure_forward(x)

        mod = ConstantClosureModule()
        x = smith.randn(3, 3, requires_grad=True)

        result = mod(x)
        expected = x @ constant_weight
        self.assertEqual(result[0], expected)

        compiled_mod = smith.compile(mod, backend=backend, fullgraph=True)
        with self.assertRaisesRegex(
            Exception, "Please convert all Tensors to FakeTensors"
        ):
            compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_error(self, backend):
        @leaf_function
        def mutate_input(x):
            x.add_(1)
            return (x,)

        @mutate_input.register_fake
        def mutate_input_fake(x):
            return (x + 1,)

        def fn(x):
            return mutate_input(x)

        x = smith.randn(3, 3)

        x_eager = x.clone()
        result_eager = fn(x_eager)
        self.assertEqual(result_eager[0], x + 1)

        compiled_fn = smith.compile(fn, backend=backend, fullgraph=True)
        with self.assertRaisesRegex(RuntimeError, "In-place mutation detected"):
            compiled_fn(x.clone())

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_validation_dtype_mismatch(self, backend):
        @leaf_function
        def dtype_mismatch_forward(mod, x):
            return (mod.linear(x),)

        @dtype_mismatch_forward.register_fake
        def dtype_mismatch_forward_fake(mod, x):
            return (mod.linear(x).double(),)

        class DtypeMismatchModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return dtype_mismatch_forward(self, x)

        mod = DtypeMismatchModule()
        x = smith.randn(3, 3)

        with config.patch(leaf_function_validate_outputs=True):
            compiled_mod = smith.compile(mod, backend=backend)
            with self.assertRaisesRegex(RuntimeError, "Dtype mismatch"):
                compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("validate_outputs", [True, False])
    def test_leaf_function_validation_shape_mismatch(self, backend, validate_outputs):
        @leaf_function
        def mismatched_forward(mod, x):
            return (mod.linear(x),)

        @mismatched_forward.register_fake
        def mismatched_forward_fake(mod, x):
            return (smith.zeros(x.shape[0], 6),)

        class MismatchedModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(3, 3)

            def forward(self, x):
                return mismatched_forward(self, x)

        mod = MismatchedModule()
        x = smith.randn(3, 3)

        with config.patch(leaf_function_validate_outputs=validate_outputs):
            compiled_mod = smith.compile(mod, backend=backend)
            if validate_outputs:
                with self.assertRaises((RuntimeError, AssertionError)):
                    compiled_mod(x)
            else:
                result = compiled_mod(x)
                self.assertEqual(result[0].shape, (3, 3))

    def test_leaf_function_no_module_inputs(self):
        @leaf_function
        def my_custom_fn(inputs: dict[str, smith.Tensor], scale: float, offset: int):
            x = inputs["x"]
            y = inputs["y"]
            if x.sum() > 0:
                return (x * scale + y + offset, x.sum() + y.sum())
            return (x * scale - y + offset, x.sum() - y.sum())

        @my_custom_fn.register_fake
        def my_custom_fn_fake(
            inputs: dict[str, smith.Tensor], scale: float, offset: int
        ):
            x = inputs["x"]
            y = inputs["y"]
            return (x * scale + y + offset, x.sum() + y.sum())

        class NoModuleInputsModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 2.0
                self.offset = 1

            def forward(self, x, y):
                inputs = {"x": x, "y": y}
                return my_custom_fn(inputs, self.scale, self.offset)

        def args_fn():
            return (
                smith.randn(3, 3, requires_grad=True),
                smith.randn(3, 3, requires_grad=True),
            )

        def loss_fn(out):
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(NoModuleInputsModule, args_fn, loss_fn)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_multiple_tensors(
        self, backend, check_escaped_gradients
    ):
        weight1 = smith.randn(3, 3, requires_grad=True)
        weight2 = smith.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            return (smith.empty(x.shape[0], 3),)

        def fn(x):
            return uses_multiple_closures(x)

        x = smith.randn(2, 3, requires_grad=True)

        compiled_fn = smith.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            if check_escaped_gradients:
                with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                    compiled_fn(x)
            else:
                result = compiled_fn(x)
                self.assertEqual(result[0].shape, (2, 3))

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_input_no_grad(
        self, backend, check_escaped_gradients
    ):
        closure_weight = smith.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (smith.empty(x.shape[0], 3),)

        def fn(x):
            return uses_closure(x)

        x = smith.randn(2, 3, requires_grad=False)

        compiled_fn = smith.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            result = compiled_fn(x)
            self.assertEqual(result[0].shape, (2, 3))

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_mixed_inputs(
        self, backend, check_escaped_gradients
    ):
        base1 = smith.randn(3, 3, requires_grad=True)
        base2 = smith.randn(3, 4, requires_grad=True)
        closure_weight1 = base1 * 2
        closure_weight2 = base2 * 3

        @leaf_function
        def mixed_inputs(x, y):
            out1 = x @ closure_weight1 + y
            out2 = x @ closure_weight2
            return (out1, out2)

        @mixed_inputs.register_fake
        def mixed_inputs_fake(x, y):
            return (smith.empty(x.shape[0], 3), smith.empty(x.shape[0], 4))

        def fn(x, y):
            return mixed_inputs(x, y)

        x = smith.randn(2, 3, requires_grad=True)
        y = smith.randn(2, 3, requires_grad=False)

        compiled_fn = smith.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            if check_escaped_gradients:
                with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                    compiled_fn(x, y)
            else:
                result = compiled_fn(x, y)
                self.assertEqual(result[0].shape, (2, 3))
                self.assertEqual(result[1].shape, (2, 4))

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_escaped_gradient_error_message_contains_tensor_info(
        self, backend
    ):
        closure_weight = smith.randn(4, 5, dtype=smith.float32, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (smith.empty(x.shape[0], 5),)

        def fn(x):
            return uses_closure(x)

        x = smith.randn(2, 4, requires_grad=True)

        compiled_fn = smith.compile(fn, backend=backend, fullgraph=True)
        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, r"shape=\[4, 5\].*dtype="):
                compiled_fn(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_escaped_gradient_actually_lost(self, backend):
        closure_weight = smith.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (smith.empty(x.shape[0], 3),)

        def fn(x):
            return uses_closure(x)

        x = smith.randn(2, 3, requires_grad=True)

        compiled_fn = smith.compile(fn, backend=backend, fullgraph=True)
        result = compiled_fn(x)
        loss = result[0].sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNone(closure_weight.grad)

    def test_leaf_function_and_nonstrict_trace_mutually_exclusive(self):
        from smith._dynamo.decorators import leaf_function, nonstrict_trace

        with self.assertRaisesRegex(
            ValueError,
            "cannot be both marked as @leaf_function and @nonstrict_trace",
        ):

            @leaf_function
            @nonstrict_trace
            def bad_fn1(x):
                return (x,)

        with self.assertRaisesRegex(
            ValueError,
            "cannot be both marked as @leaf_function and @nonstrict_trace",
        ):

            @nonstrict_trace
            @leaf_function
            def bad_fn2(x):
                return (x,)


instantiate_parametrized_tests(DecoratorTests)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
