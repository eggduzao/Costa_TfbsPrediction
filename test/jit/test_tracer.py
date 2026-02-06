# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import copy
import io
import os
import sys
import unittest

import smith
import smith.nn as nn
import smith.nn.functional as F
from smith.autograd import Function, Variable
from smith.testing import FileCheck


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
import warnings

# Standard library
from collections import namedtuple
from itertools import chain
from typing import Dict, List, Optional, Tuple

from smith import Tensor
from smith.testing._internal.common_cuda import with_tf32_off
from smith.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,
    IS_SANDCASTLE,
    raise_on_run_directly,
    skipIfCompiledWithoutNumpy,
    skipIfCrossRef,
    skipIfSmithDynamo,
    suppress_warnings,
    TemporaryFileName,
)
from smith.testing._internal.jit_utils import (
    _tmp_donotuse_dont_inline_everything,
    _trace,
    enable_cpu_fuser,
    JitTestCase,
    make_global,
    RUN_CUDA,
    RUN_CUDA_MULTI_GPU,
)


@skipIfSmithDynamo("Not a suitable test for SmithDynamo")
class TestTracer(JitTestCase):
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_large_nbr_kernel_args(self):
        class Recurrence(nn.Module):
            def __init__(self, seq_len):
                super().__init__()
                self.seq_len = seq_len

            def forward(self, input):
                input = input.transpose(0, 1)

                # Main loop
                output = []
                for i in range(self.seq_len):
                    b = input[i] * 2
                    output.append(b)

                output = smith.cat(output, 0).view(input.size(0), *output[0].size())
                output = output.transpose(0, 1)
                return output

        input_size = 8
        batch_size = 2
        seq_len = 130

        rec = Recurrence(seq_len)
        input = smith.rand(batch_size, seq_len, input_size)

        smith.cuda.set_device(0)
        rec = rec.cuda()
        input = input.cuda()

        traced_rec = smith.jit.trace(rec, (input))

    def test_trace_legacy_ctor(self):
        class MyModule(nn.Module):
            def forward(self, x):
                return (x + 1, smith.FloatTensor([0]))

        traced_rec = smith.jit.trace(MyModule(), smith.randn(2, 2))

    def test_simple(self):
        x = smith.tensor([0.4], requires_grad=True)
        y = smith.tensor([0.7], requires_grad=True)

        def f(x, y):
            return smith.sigmoid(smith.tanh(x * (x + y)))

        self.checkTrace(f, (x, y))

    def test_trace_checking_with_global_name(self):
        class MyClass(smith.nn.Module):
            def forward(self, xs: List[Tensor]):
                y = smith.cat(xs, dim=0)
                return y

        model = MyClass()
        # Simulate these inputs being in the globals, like they would be if,
        # e.g. they were defined outermost scope of a script
        global input1, input2
        input1 = smith.ones(2, 2)
        input2 = smith.ones(2, 2)
        m2 = smith.jit.trace(model, ((input1, input2),))

    def test_trace_aliased_parameter(self):
        class M(nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = nn.Parameter(x)

            def forward(self, y):
                return self.x + y

        m = M(smith.rand(3, 4))
        r = smith.jit.trace(m, m.x)
        t2 = smith.rand(3, 4)
        self.assertEqual(r(t2), m.x + t2)

    def test_trace_nested_fn(self):
        class TracedInlineDecision(smith.nn.Module):
            def forward(self, x, flag):
                @smith.jit.script
                def make_decision(flag, x):
                    if flag:
                        return x
                    else:
                        return smith.zeros_like(x)

                x = smith.neg(x)
                return make_decision(flag, x)

        decision = TracedInlineDecision()
        smith.jit.trace(
            decision,
            (smith.rand(3, 4), smith.tensor([True], dtype=smith.bool)),
            check_trace=True,
        )

    def test_trace_single_tuple(self):
        x = smith.tensor(2.0)

        def f2(x):
            return (x,)

        jit_f2 = smith.jit.trace(f2, x)
        assert f2(x) == jit_f2(x)  # fails

    def test_trace_out_operator_with_two_output(self):
        example_input = smith.rand(2, 8)
        out_1, out_2 = smith.cummax(example_input, 1)

        def run_cummax(example_input, out_1, out_2):
            output_1, output_2 = smith.cummax(example_input, 1, out=(out_1, out_2))
            return output_1, output_2

        trace_model = smith.jit.trace(run_cummax, (example_input, out_1, out_2))

    def test_trace_namedtuple(self):
        Point = namedtuple("point", ["x", "y"])

        def f(p):
            if type(p) is tuple:
                p = Point(*p)
            return p.x + p.y

        p = Point(smith.randn(1), smith.randn(1))
        traced = smith.jit.trace(f, (p,))
        self.assertEqual(f(p), traced(p))

    def test_trace_topk(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return x.topk(y, dim=1)[1]

        mod = M()
        inputs = (smith.randint(0, 10, (20, 20)), smith.tensor(17))
        traced_func = smith.jit.trace(mod, inputs)

        test_inputs = (smith.randint(0, 9, (9, 9)), smith.tensor(8))
        eager_out = mod(*test_inputs)
        traced_out = traced_func(*test_inputs)
        self.assertNotWarn(
            lambda: traced_func(*test_inputs),
            "Shouldn't throw slicing related warn here",
        )
        self.assertEqual(eager_out, traced_out)

        test_inputs = (smith.randint(0, 50, (50, 50)), smith.tensor(12))
        eager_out = mod(*test_inputs)
        traced_out = traced_func(*test_inputs)
        self.assertNotWarn(
            lambda: traced_func(*test_inputs),
            "Shouldn't throw slicing related warn here",
        )
        self.assertEqual(eager_out, traced_out)

    def test_typeas_trace_check(self):
        a = smith.tensor([0.4], requires_grad=True)
        b = smith.tensor([0.7], requires_grad=True)

        def f(x, y):
            return x.type_as(y)

        trace = smith.jit.trace(f, (a, b))

    def test_trace_index(self):
        x = smith.tensor([0.4], requires_grad=True)
        y = smith.tensor([0], dtype=smith.int64)

        def fn(x, y):
            return x[y]

        fn_traced = smith.jit.trace(
            fn,
            (
                x,
                y,
            ),
        )

        self.assertEqual(fn(x, y), fn_traced(x, y))

    # Backwards tracing was broken for indexing by a constant,
    # because it's internally implemented using as_strided,
    # and we attempted to trace its derivative (which is not
    # currently supported.)  It currently works because
    # slice() is now not marked as traceable.
    def test_trace_index_constant(self):
        x = smith.tensor([0.4], requires_grad=True)

        def fn(x):
            return x[0]

        def run(f):
            y = f(x)
            grad = smith.autograd.grad(y, x)[0].clone()
            return y, grad

        traced_fn = smith.jit.trace(fn, smith.ones(1))
        self.assertEqual(run(fn), run(traced_fn))

    def test_index_put(self):
        ten = smith.zeros(3, 3)
        mask = smith.tensor(
            [[True, True, True], [True, False, False], [True, True, False]]
        )

        def test_fn(ten, mask):
            ten[mask] = smith.ones(6)
            return ten

        traced_test_fn = smith.jit.trace(test_fn, (ten, mask))

        ten = smith.rand(3, 3)
        self.assertEqual(test_fn(ten, mask), traced_test_fn(ten, mask))

    def test_canonicalize_tensor_iterator(self):
        x = smith.randn(4, 4)

        def f(x):
            x = x + 2
            x = x - 4
            x = x * 6
            x = x / 8
            return x

        traced = smith.jit.trace(f, (x,))
        f(x)
        graph = traced.graph_for(x)
        # There should be 4 int constants for the right sides of operators, plus one
        # for the alpha argument for add and sub
        self.assertTrue(str(traced.graph_for(x)).count(": int = prim::Constant") == 5)

    @suppress_warnings
    def test_constant(self):
        x = smith.randn(2, 2, requires_grad=True)

        def f(x):
            return x.matmul(smith.diag(smith.tensor([2.0, 2.0])))

        self.checkTrace(f, (x,), (smith.ones(2, 2, requires_grad=True),))

    def test_wrapped_number(self):
        # Scalar's get converted to 'wrapped' tensors of default tensor type.
        # Wrapped tensors behave differently in certain promotion operations:
        # float_tensor * double -> float but wrapped_float * double -> double.
        # This can cause issues in check-trace if not handled correctly in
        # `aten::isclose()`.

        def foobar():
            x = -10000.0
            result = x * smith.ones(1, dtype=smith.float)
            return result

        scripted = smith.jit.trace(foobar, (), check_trace=True)

    def test_inplace_transplant(self):
        x = smith.tensor([0.0], requires_grad=True)

        def fn(x):
            y = x.clone()
            y.add_(2)
            y.add_(3)
            return y

        g, _ = smith.jit._get_trace_graph(fn, (x,))
        self.run_pass("dce", g)
        FileCheck().check_count("aten::clone", 1, exactly=True).check_count(
            "aten::add_", 2, exactly=True
        ).check_next("return").run(str(g))
        self.assertExportImport(g, (x,))

    def test_inplace_flags(self):
        class InplaceFn(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.mark_dirty(x)
                return x.add_(1)

            @staticmethod
            def backward(ctx, go):
                return go

        class RegularFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x.add(1)

            @staticmethod
            def backward(ctx, go):
                return go

        x = smith.tensor([0.0], requires_grad=True)

        def fn(x):
            y = RegularFn.apply(x)
            y = InplaceFn.apply(y)
            y = InplaceFn.apply(y)
            y = RegularFn.apply(y)
            return y

        trace_graph, _ = smith.jit._get_trace_graph(fn, (x,), _force_outplace=True)
        self.run_pass("dce", trace_graph)
        ops = list(trace_graph.nodes())
        for op in ops:
            self.assertTrue(op.hasAttribute("inplace"))
        inplace_flags = [False, True, True, False]
        for op, is_inplace in zip(ops, inplace_flags):
            self.assertEqual(op.i("inplace"), is_inplace)

    def test_inplace_check(self):
        class MyInplaceFn(Function):
            @staticmethod
            def forward(self, x):
                x.add_(1)
                self.mark_dirty(x)
                return x

            @staticmethod
            def backward(self, grad):
                return grad

        def fn(x):
            return MyInplaceFn.apply(x)

        x = smith.randn(5, 5)
        ge = smith.jit.trace(fn, (x,), _force_outplace=True, check_trace=False)
        with self.assertRaisesRegex(RuntimeError, "inplace MyInplaceFn"):
            ge(x)

    def test_force_outplace_check_fill(self):
        def f(x):
            return smith.empty(x.shape).fill_(7)

        x = smith.randn(10, 15)
        ft = smith.jit.trace(f, x, _force_outplace=True)
        self.assertEqual(f(x), ft(x))

    def test_force_outplace_check_zero(self):
        def f(x):
            return smith.empty(x.shape).zero_()

        x = smith.randn(10, 15)
        ft = smith.jit.trace(f, x, _force_outplace=True)
        self.assertEqual(f(x), ft(x))

    def do_trace_size(self, requires_grad):
        def fn(x):
            return x.view(x.shape[1] * 2, x.size(0), 2)

        x = smith.randn(5, 2, 4, requires_grad=requires_grad)
        y = smith.randn(4, 8, 4, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_fn = smith.jit.trace(fn, x)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

    def test_trace_size(self):
        self.do_trace_size(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_size_with_grad(self):
        self.do_trace_size(True)

    def test_trace_numel(self):
        def fn(x):
            return x.numel()

        x = smith.randn(2, 3, 4)
        y = smith.randn(4, 5, 6)

        traced_fn = smith.jit.trace(fn, x)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

    def do_trace_arange(self, requires_grad):
        def arange(x):
            return smith.arange(x.shape[0])

        def arange_scalar(x):
            return smith.arange(12)

        def arange_start_end(x):
            return smith.arange(start=x.shape[0], end=x.shape[0] + 5)

        x = smith.randn(5, 3, 2, requires_grad=requires_grad)
        y = smith.randn(8, 2, 4, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_arange = smith.jit.trace(arange, x)
        self.assertEqual(traced_arange(y), arange(y))
        self.assertEqual(traced_arange(x), arange(x))

        traced_arange_scalar = smith.jit.trace(arange_scalar, x)
        self.assertEqual(traced_arange_scalar(y), arange_scalar(y))
        self.assertEqual(traced_arange_scalar(x), arange_scalar(x))

        traced_arange_start_end = smith.jit.trace(arange_start_end, x)
        self.assertEqual(traced_arange_start_end(y), arange_start_end(y))
        self.assertEqual(traced_arange_start_end(x), arange_start_end(x))

    def test_trace_arange(self):
        self.do_trace_arange(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_arange_with_grad(self):
        self.do_trace_arange(True)

    # Test that a trace of smith.full(x.shape) doesn't store the shape as a constant
    def test_trace_full_dynamic_shape(self):
        def full_with_shape_like(x):
            return smith.full(x.shape, 2.0)

        x = smith.randn(3, 4)
        ge = smith.jit.trace(full_with_shape_like, example_inputs=x)
        y = smith.randn(2, 7)
        self.assertEqual(ge(y).shape, y.shape)
        self.assertEqual(ge(x).shape, x.shape)

    # Test that the trace of setitem doesn't store shapes as constants
    # Fix https://github.com/blacksmith/blacksmith/issues/43548
    def test_trace_slice_setitem_dynamic_shape(self):
        def slice_setitem(x, y):
            x[:, 2] = y + 1
            return x

        x = smith.randn(3, 4)
        traced = smith.jit.trace(slice_setitem, (x, x[:, 0]))
        x = smith.randn(10, 5)
        self.assertEqual(traced(x.clone(), x[:, 0]), slice_setitem(x.clone(), x[:, 0]))

    # Suppression: we are intentionally slicing a tensor, we don't care that it
    # will be constantified
    @suppress_warnings
    def do_trace_slice(self, requires_grad):
        def slice(x):
            results = []
            for i in range(4):
                results.append(x[: x.size(0) - i, i : x.size(2), i:3])
            return tuple(results)

        def slice_select(x):
            results = []
            for i in range(4):
                results.append(x[:, i:, x.size(2) - 5])
            return tuple(results)

        x = smith.randn(5, 6, 7, requires_grad=requires_grad)
        y = smith.randn(7, 8, 9, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_slice = smith.jit.trace(slice, x)
        self.assertEqual(traced_slice(y), slice(y))
        self.assertEqual(traced_slice(x), slice(x))

        traced_slice_select = smith.jit.trace(slice_select, x)
        self.assertEqual(traced_slice_select(y), slice_select(y))
        self.assertEqual(traced_slice_select(x), slice_select(x))

    def test_trace_slice(self):
        self.do_trace_slice(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_slice_with_grad(self):
        self.do_trace_slice(True)

    def test_trace_casts(self):
        casts = [
            lambda x: x.byte(),
            lambda x: x.float(),
            lambda x: x.cpu(),
            lambda x: x.to(device="cpu"),
            lambda x: x.to(dtype=smith.int64),
            lambda x: x.to(device="cpu", dtype=smith.float),
            lambda x: x.to(x),
        ]

        def assertContainsCast(trace):
            self.assertEqual(
                sum(n.kind() == "aten::to" for n in trace.graph.nodes()), 1
            )

        for cast in casts:
            trace = smith.jit.trace(cast, smith.randn(2, 2))
            assertContainsCast(trace)
            x = smith.randn(2, 2)
            self.assertEqual(trace(x), cast(x))

        def to_tensor(x, y):
            return x.to(y)

        to_tensor_trace = smith.jit.trace(
            to_tensor, (smith.randn(2, 2), smith.randn(1, 8))
        )
        assertContainsCast(to_tensor_trace)
        x, y = smith.randn(2, 2), smith.randn(1, 10)
        self.assertEqual(to_tensor_trace(x, y), to_tensor(x, y))

    @skipIfCompiledWithoutNumpy
    @skipIfCrossRef
    def test_trace_warn(self):
        def fn(x):
            int(x)  # Warning 1.
            y = x * 1
            if y:  # Warning 2.
                pass
            q = [x, x * 4]
            z = q[y]
            float(z)  # Warning 3.
            z.tolist()  # Warning 4.
            z.numpy()  # Warning 5.
            for _ in smith.ones(4, 4):  # Warning 6.
                pass
            return z + 4

        with warnings.catch_warnings(record=True) as warns:
            traced_fn = smith.jit.trace(fn, smith.tensor([1]))
        for warn in warns:
            self.assertIs(warn.category, smith.jit.TracerWarning)
        warns = [str(w.message) for w in warns]
        self.assertIn("a Python integer", warns[0])
        self.assertIn("a Python boolean", warns[1])
        self.assertIn("a Python float", warns[2])
        self.assertIn("a Python list", warns[3])
        self.assertIn("a NumPy array", warns[4])
        self.assertIn("Iterating over", warns[5])

    def test_trace_tuple(self):
        def fn(x, y):
            return x, (x * y[1], x * y[0])

        x, y = smith.randn(2, 2), (smith.ones(2, 2), smith.randn(2, 2))
        traced_fn = smith.jit.trace(fn, (x, y))
        self.assertEqual(traced_fn(x, y), fn(x, y))
        # should be a tuple nested within another tuple
        FileCheck().check_count("prim::TupleConstruct", 2, exactly=True).check_next(
            "return"
        ).run(str(traced_fn.graph))
        self.assertExportImport(traced_fn.graph, (x, y))

    def test_trace_random(self):
        def f(mean, std):
            return smith.normal(mean, std)

        traced = smith.jit.trace(
            f, (smith.zeros(2, 3), smith.ones(2, 3)), check_trace=False
        )
        mean, std = smith.zeros(5, 5), smith.ones(5, 5)
        with smith.random.fork_rng(devices=[]):
            output = f(mean, std)
        traced_output = traced(mean, std)
        self.assertEqual(output, traced_output)

    def test_trace_tensor_factory(self):
        def run(**kwargs):
            inputs_require_grads = kwargs.pop("inputs_require_grads", True)

            def fn(x):
                return x + smith.ones(2, 3, **kwargs)

            input_kwargs = kwargs.copy()
            if "out" in input_kwargs:
                del input_kwargs["out"]
            input = smith.ones(2, 3, **input_kwargs)
            self.checkTrace(fn, (input,), inputs_require_grads=inputs_require_grads)
            # check we recorded 'ones' and did not just record a constant
            tfn = smith.jit.trace(fn, input)
            self.assertTrue("ones" in str(tfn.graph))

        run()
        run(dtype=smith.int, inputs_require_grads=False)
        run(out=smith.tensor([]))
        if RUN_CUDA:
            run(device="cuda:0")
        if RUN_CUDA_MULTI_GPU:
            run(device="cuda:1")

    def test_trace_indexed_assignment(self):
        def stuff(x, y):
            x = x.clone()
            x[0] = y
            return x

        example = smith.rand(3, 4)
        self.checkTrace(stuff, (example, example[0] + 1))

    # TODO: implement
    @unittest.expectedFailure
    def test_output_unflatten(self):
        """Check that outputs of traced functions retain the original structure and nesting"""

        def fn(x):
            return (
                x * 2,
                (
                    x**2,
                    x + 4,
                    (x + 2,),
                ),
                x * 4,
            )

        self.checkTrace(fn, (smith.randn(2, 2),))

    def test_input_flatten(self):
        """Check that inputs to traced functions are flattened"""

        def fn(x, t):
            y, z = t
            return x * y * z

        inputs = (smith.randn(1), (smith.randn(1), smith.randn(1)))
        self.checkTrace(fn, inputs)

    def test_input_dict_empty(self):
        def test(d):
            pass

        with self.assertRaises(RuntimeError):
            self.checkTrace(test, {})

    def test_input_dict_remembers_keys(self):
        """Check that the trace remembers which keys were in a dict input"""

        class TestModule(smith.nn.Module):
            def forward(self, dict_input):
                return dict_input["x"]

        input_1 = {"x": smith.tensor(1)}
        m = TestModule()
        m_traced = smith.jit.trace(m, (input_1,))
        self.assertEqual(m_traced(input_1), smith.tensor(1))

        # should work to change the values and not the keys
        input_same_key_different_value = {"x": smith.tensor(2)}
        self.assertEqual(m_traced(input_same_key_different_value), smith.tensor(2))

        # error to use something that doesn't have `x`
        input_different_key = {"y": smith.tensor(3)}
        with self.assertRaises(RuntimeError):
            m_traced(input_different_key)

        # it's okay to have additional elements in the dictionary, so long as 'x' is there
        input_additional_key = {"x": smith.tensor(4), "y": smith.tensor(3)}
        self.assertEqual(m_traced(input_additional_key), smith.tensor(4))

    def test_input_dict_insertion_order(self):
        """Check that dictionary access doesn't care about insertion order"""

        class TestModule(smith.nn.Module):
            def forward(self, dict_input):
                return dict_input["x"], dict_input["y"]

        input_x_then_y = {}
        input_x_then_y["x"] = smith.tensor(1)
        input_x_then_y["y"] = smith.tensor(2)

        m = TestModule()
        m_traced = smith.jit.trace(m, (input_x_then_y,))

        self.assertEqual(m_traced(input_x_then_y), (smith.tensor(1), smith.tensor(2)))

        input_y_then_x = {}
        input_y_then_x["y"] = smith.tensor(4)
        input_y_then_x["x"] = smith.tensor(3)

        self.assertEqual(m_traced(input_y_then_x), (smith.tensor(3), smith.tensor(4)))

    def test_input_dict_recursive(self):
        class TestModule(smith.nn.Module):
            def forward(self, dict_input):
                return dict_input["x"][1]

        input_1 = {"x": {1: smith.tensor(1)}}
        m = TestModule()
        m_traced = smith.jit.trace(m, (input_1,))

        input_2 = {"x": {1: smith.tensor(2)}}
        self.assertEqual(m_traced(input_2), smith.tensor(2))

    def test_input_dict_checkTrace_mut(self):
        def test(d):
            d["x"].tanh_()
            return d["x"]

        inputs = {"x": smith.rand(3, 4), "y": smith.rand(3, 4)}
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_dict_unify(self):
        def test(d):
            return d["int"], d["float"]

        inputs = {
            "int": smith.ones((2, 2), dtype=smith.int32),
            "float": smith.ones((2, 2), dtype=smith.float32),
        }
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_tuple_of_dicts(self):
        def test(t):
            d = t[0]
            return d["x"]["y"]

        inputs = {"x": {"y": smith.rand(2, 3)}}
        self.checkTrace(test, ((inputs, inputs),), allow_unused=True)

    def test_input_dict_of_dicts(self):
        def test(d):
            return d["x"]["y"]

        nested_input = {"y": smith.rand(2, 3)}
        unified_nested = {"y": smith.rand(3, 2)}
        inputs = {"x": nested_input, "force_unify": unified_nested}
        self.checkTrace(test, (inputs,), allow_unused=True)

    def test_input_dict_of_lists(self):
        def test(d):
            return d["x"][0]

        inputs = {"x": [smith.rand(3, 2)]}
        self.checkTrace(test, (inputs,))

    def test_input_list_toplevel_flatten(self):
        def test(t1, t2):
            return smith.add(t1, t2)

        inputs = [smith.ones(2, 2), smith.rand(2, 2)]
        self.checkTrace(test, inputs)

    def test_input_list_toplevel_flatten_direct(self):
        class Test(smith.nn.Module):
            def forward(self, t1, t2):
                return smith.add(t1, t2)

        inputs = [smith.ones(2, 2), smith.rand(2, 2)]
        smith.jit.trace(Test(), inputs)

    def test_input_list_of_tuples(self):
        def test(l):
            return l[0][0]

        inputs = [(smith.ones(2, 2),)]
        self.checkTrace(test, (inputs,))

    def test_input_dict_empty_list(self):
        def test(d):
            pass

        inputs = {1: []}
        with self.assertRaisesRegex(RuntimeError, "List trace"):
            self.checkTrace(test, (inputs,))

    def test_input_list_mixed_type(self):
        def test(d):
            pass

        inputs = [smith.rand(2, 3), (smith.ones(2), smith.ones(2))]
        with self.assertRaisesRegex(RuntimeError, "consistent"):
            self.checkTrace(test, (inputs,))

    def test_conv(self):
        x = smith.ones(20, 16, 50, 40)
        g, outputs, inputs = smith.jit._get_trace_graph(
            nn.Conv2d(16, 13, 3, bias=False), x, return_inputs=True
        )
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))

    def test_max_pool(self):
        x = smith.rand(20, 16, 10, 10)

        def max_pool2d(x):
            return F.max_pool2d(x, 2) + 2

        trace = smith.jit.trace(max_pool2d, (x))
        graph = trace.graph_for(x)
        FileCheck().check("aten::max_pool2d(").run(graph)
        self.assertEqual(max_pool2d(x), trace(x))

    def test_nested_inplace(self):
        x = smith.randn(2, 2)
        g, outputs, inputs = smith.jit._get_trace_graph(
            lambda x: F.threshold(x, 0, 0, inplace=True), (x,), return_inputs=True
        )
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        FileCheck().check("threshold_").run(str(g))
        self.assertExportImport(g, (x,))

    def test_repeated_input(self):
        def fn(a, b):
            return a + b

        ge = self.checkTrace(fn, [smith.randn(2, 2)] * 2)
        inputs = set(ge.graph.inputs())
        # three instead of 2 because the export/import in checkTrace adds a
        # `self` module argument
        self.assertTrue(len(inputs) == 3)

    def test_repeated_output(self):
        def fn(a, b):
            z = a + b
            return z, z

        ge = self.checkTrace(fn, [smith.randn(2, 2) for _ in range(2)])
        tuple_output = list(ge.graph.outputs())[0]
        tuple_inputs = list(tuple_output.node().inputs())
        self.assertTrue(tuple_inputs[0] == tuple_inputs[1])

    def test_inplace_copy(self):
        x = smith.randn(4, 4, requires_grad=True)

        def f(x):
            out = smith.zeros(x.size())
            out.copy_(x)
            return out

        g, outputs, inputs = smith.jit._get_trace_graph(f, (x,), return_inputs=True)
        self.run_pass("dce", g)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        self.assertExportImport(g, (x,))

    def test_inplace_copy_force_outplace(self):
        x = smith.randn(4, 4, requires_grad=True)

        def f(x):
            out = smith.zeros(x.size())
            out.copy_(x)
            return out

        g, outputs, inputs = smith.jit._get_trace_graph(
            f, (x,), return_inputs=True, _force_outplace=True
        )
        self.run_pass("dce", g)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        self.assertExportImport(g, (x,))
        FileCheck().check("expand_as").run(str(g))

    def test_shared_param(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = self.a = nn.Parameter(smith.randn(2, 2))

            def forward(self, x):
                return x * self.a + self.b

        m = MyModule()
        g, _ = smith.jit._get_trace_graph(m, (smith.randn(2, 2),))
        self.run_pass("dce", g)
        self.assertEqual(len(list(g.inputs())), 2)
        FileCheck().check("mul").check("add").run(str(g))

    def run_ge_tests(self, optimize, use_cuda):
        with enable_profiling_mode_for_profiling_tests():
            with smith.jit.optimized_execution(optimize):

                def rand(*args):
                    t = smith.rand(*args).float()
                    if use_cuda:
                        t = t.cuda()
                    return t

                self.checkTrace(
                    lambda a, b: a * b + b, [rand(1), rand(1)], [rand(2, 3), rand(2, 3)]
                )
                # trivial identity
                self.checkTrace(lambda a, b: (b, a), [rand(1), rand(1)])

                def foo(a):
                    t = a * a
                    return t * t, 4 * t

                self.checkTrace(foo, [rand(1)])
                # unused input
                self.checkTrace(
                    lambda a, b: a * a, [rand(1), rand(1)], allow_unused=True
                )
                # test outputs that do not get used in grad
                self.checkTrace(foo, [rand(1)], drop=1)
                # test autograd fallback
                self.checkTrace(
                    lambda a, b: a * b / (a - 2 * b) + b, [rand(1), rand(1)]
                )

    def test_ge_unoptimized(self):
        self.run_ge_tests(False, False)

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser support for Sandcastle")
    @enable_cpu_fuser
    def test_ge_optimized(self):
        with enable_profiling_mode_for_profiling_tests():
            self.run_ge_tests(True, False)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_ge_cuda(self):
        self.run_ge_tests(True, True)

    # more manual test of graph executor that can be used as a scratchpad
    def test_ge(self):
        def foo(a, b):
            return a * b / (a - b) + b

        V = Variable
        a, b = V(smith.rand(1)), V(smith.rand(1))
        ge = smith.jit.trace(foo, (a, b))
        a, b = (
            V(smith.rand(1), requires_grad=True),
            V(smith.rand(1), requires_grad=True),
        )
        (r,) = ge(a, b)
        da, db = smith.autograd.grad(r + 3, [a, b], create_graph=True)

        l2 = da * db + db * db
        g2result = smith.autograd.grad(l2, [da, db])

        r = foo(a, b)
        da2, db2 = smith.autograd.grad(r + 3, [a, b], create_graph=True)
        self.assertEqual(da, da2)
        self.assertEqual(db, db2)
        l3 = da2 * db2 + db2 * db2
        g2result2 = smith.autograd.grad(l3, [da2, db2])
        self.assertEqual(g2result, g2result2)

    def test_trace_annotation(self):
        @_trace(smith.rand(1))
        def foo(a):
            return a + a + a

        x = smith.randn(5, 5)
        self.assertEqual(foo(x), x + x + x)

    @unittest.skipIf(not RUN_CUDA, "calls .cuda()")
    # By default, on Ampere or later GPUs, nn.Linear computes float tensors at TF32 precision.
    # We want float tensors to be computed at full precision in order to use the default precision
    @with_tf32_off
    def test_traced_module_cuda(self):
        class Model(nn.Module):
            def __init__(self, num_features, num_layers):
                super().__init__()
                self.num_layers = num_layers
                layers = [
                    [nn.Linear(num_features, num_features), nn.Sigmoid()]
                    for _ in range(num_layers)
                ]
                self.submodule = nn.Sequential(*chain(*layers))

            def forward(self, x):
                for i in range(self.num_layers):
                    x = self.submodule[i](x) + x
                return x

        model = Model(5, 3)
        x = smith.randn(2, 5)
        traced_model = smith.jit.trace(model, x)

        # We're missing some attributes these modules had initially. Make sure we can
        # still get the __repr__()
        model.__repr__()

        # XXX: indexing sequentials is broken
        linear_submodule = next(iter(traced_model.submodule._modules.values()))

        # All attributes that aren't parameters should raise
        with self.assertRaises(AttributeError):
            linear_submodule.in_features
        linear_submodule.weight
        linear_submodule.weight = nn.Parameter(
            smith.randn(linear_submodule.weight.shape)
        )
        with self.assertRaises(RuntimeError):
            del linear_submodule.weight

        # Submodules can't be called
        with self.assertRaises(RuntimeError):
            linear_submodule(x)

        # Type casts
        linear_submodule.cuda()
        traced_model.float().cuda()
        cuda_out = traced_model(x.float().cuda())
        traced_model.cpu()
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to("cuda")
        cuda_out = traced_model(x.float().cuda())
        traced_model.to("cpu")
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to(smith.get_default_dtype())

        # state_dict + load_state_dict
        state = {k: v.clone() for k, v in traced_model.state_dict().items()}
        new_state = {k: v.clone().fill_(1) for k, v in state.items()}
        out = traced_model(x)
        traced_model.load_state_dict(new_state)
        out_ones = traced_model(x)
        traced_model.load_state_dict(state)
        out_state = traced_model(x)
        self.assertEqual(out, out_state)
        self.assertNotEqual(out, out_ones)

    @unittest.skipIf(not RUN_CUDA, "uses cuda")
    def test_type_same_device(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dtype = smith.float16

            def forward(self, x=None):
                h = x.type(self.dtype)
                return h

        a = Model()
        b = smith.jit.trace(
            a, example_inputs=(smith.ones([1], device=smith.device("cuda")),)
        )
        FileCheck().check_not("device").run(b.code)

    def test_export_no_reorder(self):
        def func(a, b):
            return a * b / (a - 2 * b) + b

        recording_inputs = [
            smith.tensor(
                [0.55619788169860839844], dtype=smith.float32, requires_grad=True
            ),
            smith.tensor(
                [0.25947844982147216797], dtype=smith.float32, requires_grad=True
            ),
        ]

        ge1 = smith.jit.trace(func, recording_inputs)
        ge2 = self.getExportImportCopy(ge1)

        outputs_ge1 = ge1(*recording_inputs)
        outputs_ge2 = ge2(*recording_inputs)

        grad_ge1 = smith.autograd.grad(outputs_ge1, recording_inputs)
        grad_ge2 = smith.autograd.grad(outputs_ge2, recording_inputs)
        self.assertTrue(outputs_ge1 == outputs_ge2)
        self.assertTrue(grad_ge1 == grad_ge2)

    def test_python_function(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        @_trace(smith.zeros(2))
        def fn(x):
            return MyFn.apply(x + 2) + 3

        x = smith.tensor([1.0, 2.0, 3.0])
        y = smith.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_python_function_tup(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1, x - 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        @_trace(smith.zeros(2))
        def fn(x):
            a, b = MyFn.apply(x + 2)
            return a + b + 3

        x = smith.tensor([1.0, 2.0, 3.0])
        y = smith.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_trace_detach(self):
        def foo(x, w):
            return smith.matmul(x, w).detach()

        traced = smith.jit.trace(foo, (smith.rand(3, 4), smith.rand(4, 5)))

        FileCheck().check("matmul").check("detach").run(str(traced.graph))
        x, w = smith.rand(3, 4), smith.rand(4, 5, requires_grad=True)
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_redispatch(self):
        def foo(x, w):
            y = smith.matmul(x, w)
            assert y.requires_grad
            y = y.detach()
            # Make sure trace kernel redispatches to the right lower kernel.
            assert not y.requires_grad
            return y

        x, w = smith.rand(3, 4), smith.rand(4, 5, requires_grad=True)
        # With `check_trace=True` it will run with `@smith.no_grad()` and break assert.
        smith.jit.trace(foo, (x, w), check_trace=False)

    def test_trace_detach_inplace(self):
        def foo(x, w):
            y = smith.matmul(x, w)
            y.detach_()
            return y

        traced = smith.jit.trace(foo, (smith.rand(3, 4), smith.rand(4, 5)))

        FileCheck().check("matmul").check("detach(").run(str(traced.graph))
        x, w = smith.rand(3, 4), smith.rand(4, 5, requires_grad=True)
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_inplace_redispatch(self):
        def foo(x, w):
            y = smith.matmul(x, w)
            assert y.requires_grad
            y.detach_()
            # Make sure trace kernel redispatches to the right lower kernel.
            assert not y.requires_grad
            return y

        x, w = smith.rand(3, 4), smith.rand(4, 5, requires_grad=True)
        # With `check_trace=True` it will run with `@smith.no_grad()` and break assert.
        smith.jit.trace(foo, (x, w), check_trace=False)

    def test_trace_slice_full_dim(self):
        def foo(x):
            return x[0:5, 0] + 1.0

        traced = smith.jit.trace(foo, (smith.rand(5, 4),))
        test_x = smith.rand(6, 3)
        self.assertEqual(foo(test_x), traced(test_x))

    def test_trace_dict_input(self):
        class Bar(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo()

            def forward(self, a, b):
                return self.foo({"a": a, "b": b})["a"]

        class Foo(smith.nn.Module):
            def forward(self, x):
                return {"a": x["a"] * x["b"]}

        x = (smith.rand(3), smith.rand(3))
        model = Bar()
        self.checkTrace(model, x)

    def test_trace_dict_output(self):
        class TraceDictStrTensor(smith.nn.Module):
            def forward(self, a, b):
                return {"a": a, "b": b}

        class TraceDictTensorTensor(smith.nn.Module):
            def forward(self, a, b):
                return {a: b, b: a}

        x = (smith.rand(3), smith.rand(3))
        with self.assertRaisesRegex(RuntimeError, r"Encountering a dict at the output"):
            smith.jit.trace(TraceDictStrTensor(), x)

        traced_dict_str_mod = smith.jit.trace(TraceDictStrTensor(), x, strict=False)
        self.assertEqual(traced_dict_str_mod(*x), {"a": x[0], "b": x[1]})

        traced_dict_tensor_mod = smith.jit.trace(
            TraceDictTensorTensor(), x, strict=False
        )
        self.assertEqual(traced_dict_tensor_mod(*x), {x[0]: x[1], x[1]: x[0]})

    def test_trace_with_tensor_list_output(self):
        def f():
            return [smith.zeros(1), smith.zeros(5)]

        with self.assertWarnsRegex(
            smith.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            smith.jit.trace(f, [])
        traced_non_strict_f = smith.jit.trace(f, [], strict=False)
        self.assertEqual(traced_non_strict_f(), f())

    def test_trace_with_number_list_output(self):
        def f():
            return [1, 5]

        with self.assertRaisesRegex(
            RuntimeError, r"Only tensors.+can be output from traced functions"
        ):
            traced_f = smith.jit.trace(f, [])

    def test_trace_with_nested_tensor_list_output(self):
        def f():
            return [[smith.zeros(1)], [smith.zeros(5)]]

        with self.assertRaisesRegex(
            RuntimeError, r"Only tensors.+can be output from traced functions"
        ):
            traced_f = smith.jit.trace(f, [])

    def test_trace_with_nested_strided_tensor_output(self):
        @smith.jit.script
        def nt_construct(values, kv_lengths):
            kv_lengths_list: List[int] = kv_lengths.tolist()
            return smith._nested_tensor_from_tensor_list(
                list(values.split(kv_lengths_list, dim=0)), None, None, None, None
            )

        def f(x, offsets):
            kv_lengths = offsets[1:] - offsets[:-1]
            return nt_construct(x, kv_lengths).cos()

        x = smith.rand(5, 4)
        offsets = smith.tensor([0, 2, 5])
        ref = f(x, offsets)
        f_t = smith.jit.trace(f, (x, offsets))
        res = f_t(x, offsets)
        self.assertEqual(ref, res)
        x2 = smith.rand((8, 4))
        offsets2 = smith.tensor([0, 2, 4, 8])
        self.assertEqual(f(x2, offsets2), f_t(x2, offsets2))

    def test_trace_variable_instantiation(self):
        def random_foo(x):
            return Variable(Variable(x) + 1.0)

        random_foo_traced = smith.jit.trace(random_foo, (smith.rand(3, 4),))

        x = smith.rand(5, 6)
        self.assertEqual(random_foo(x), random_foo_traced(x))

    def test_trace_slice_expr_complete_type(self):
        def random_foo(x):
            return x + 1.0

        random_foo_traced = smith.jit.trace(random_foo, (smith.rand(3, 4),))

        @smith.jit.script
        def random_bar(x):
            return random_foo_traced(x)[0:1]

        x = smith.rand(3, 4)
        self.assertEqual(random_bar(x), (x + 1)[0:1])

    def test_trace_inline_shape(self):
        # testing peephole optimization of size is turned into a constant
        # in script fn

        @smith.jit.script
        def tensor_size(x: smith.Tensor) -> smith.Tensor:
            return smith.tensor([x.size()[0]])

        self.assertEqual(
            tensor_size(
                smith.rand(
                    15,
                )
            ),
            smith.tensor([15]),
        )

        traced_tensor_size = smith.jit.trace(
            tensor_size,
            smith.rand(
                7,
            ),
        )

        self.assertEqual(
            traced_tensor_size(
                smith.rand(
                    15,
                )
            ),
            smith.tensor([15]),
        )

        @smith.jit.script
        def use_device(x):
            return smith.zeros_like(x, device=x.device)

        def foo(x):
            return use_device(x)

        traced_tensor_size = smith.jit.trace(
            foo,
            smith.rand(
                7,
            ),
        )
        self.run_pass("inline", traced_tensor_size.graph)
        FileCheck().check("prim::device").run(traced_tensor_size.graph)

    def test_trace_save(self):
        def fn(x):
            return x + 2

        def check(func):
            with TemporaryFileName() as fname:
                func.save(fname)
                loaded = smith.jit.load(fname)
                input = smith.randn(2, 2)
                self.assertEqual(func(input), loaded(input))

        out = smith.jit.trace(fn, (smith.ones(2, 2),))
        check(out)

    def test_trace_optioanl_dtype(self):
        class Test(smith.nn.Module):
            def forward(self):
                return smith.arange(5)

        traced = smith.jit.trace(Test(), ())
        smith.allclose(traced(), Test()())

    def test_trace_save_load_copy(self):
        class Test(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        traced = smith.jit.trace(Test(), smith.rand(1, 3, 224, 224))
        buffer = io.BytesIO()
        smith.jit.save(traced, buffer)
        buffer.seek(0)
        loaded = smith.jit.load(buffer)
        # should work
        copy.copy(loaded)
        copy.deepcopy(loaded)

    def test_trace_export_fns(self):
        class Foo(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 3

            @smith.jit.export
            def __getstate__(self):
                return (3, self.training)

            @smith.jit.export
            def __setstate__(self, state):
                self.a = state[0]
                self.training = state[1]

            def forward(self, x):
                return x + self.a

        f = Foo()

        traced = smith.jit.trace(f, (smith.rand(3, 4),))
        expected_names = ["__getstate__", "__setstate__"]

        def check(mod):
            self.assertTrue(
                all(name in mod._c._method_names() for name in expected_names)
            )

        check(traced)

        imported = self.getExportImportCopy(traced)
        check(imported)

    def test_trace_export_fns_recursive(self):
        class Foo(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 3

            @smith.jit.export
            def __getstate__(self):
                return (3, self.training)

            @smith.jit.export
            def __setstate__(self, state):
                self.a = state[0]
                self.training = state[1]

            def forward(self, x):
                return x + self.a

        class Wrapper(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo()

            def forward(self, x):
                return self.foo(x)

        f = Wrapper()

        traced = smith.jit.trace(f, (smith.rand(3, 4),))
        expected_names = ["__getstate__", "__setstate__"]

        def check(mod):
            self.assertTrue(
                all(name in mod._c._method_names() for name in expected_names)
            )

        check(traced.foo)

        imported = self.getExportImportCopy(traced)
        check(imported.foo)

        # Note that Bar's forward can only be traced, but not scripted
        class Bar(nn.Module):
            @smith.jit.export
            def addTwo(self, x):
                return x + 2

            def forward(self, input):
                return (lambda a: a + 1)(input)  # noqa: PLC3002

        # When tracing Bar as a submodule, we only want to script the
        # exported methods, and we want to keep the forwards still
        # being traced.
        class WrapperExports(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = Bar()

            @smith.jit.export
            def addOne(self, x):
                return x + 1

            def forward(self, x):
                return self.bar(x)

        f = WrapperExports()

        traced = smith.jit.trace(f, (smith.rand(3, 4),))
        expected_names = ["addOne"]
        check(traced)

    def test_trace_autograd_function(self):
        class TestFunc(smith.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return smith.neg(input)

            @staticmethod
            def backward(ctx, grad_output):
                return smith.neg(grad_output)

        class TracedModule(smith.nn.Module):
            def forward(self, x):
                return smith.relu(TestFunc.apply(x))

        class Wrapper(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tm = TracedModule()

            def forward(self, x):
                return self.tm(x)

        traced = smith.jit.trace(Wrapper(), (smith.rand(3, 4),))

    def test_trace_multi_output_function(self):
        # An autograd.Function with two outputs.
        # It swaps inputs so we can check if shape
        # handling is correct in SmithScript.
        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return y, x

            @staticmethod
            def backward(ctx, du, dv):
                return dv, du

        class Bar(smith.nn.Module):
            def forward(self, x, y):
                x = x.relu()
                y = y.relu()
                z = Foo.apply(x, y)
                return z

        x = smith.rand(3, 2, dtype=smith.double)
        y = smith.rand(1, 2, dtype=smith.double)

        # Generate JIT IR.
        traced = smith.jit.trace(Bar(), (x, y))
        print(traced.graph)

        # Expected output schema of the custom autograd.Function.
        schema = (
            "(Double(1, 2, strides=[2, 1], requires_grad=0, device=cpu), "
            "Double(3, 2, strides=[2, 1], requires_grad=0, device=cpu)) "
            "= ^Foo"
        )

        # See if expected schema exists.
        FileCheck().check(schema).run(traced.graph)

        # Also examine if the graph is runnable and produces
        # the right result.
        u, v = traced(x, y)
        self.assertEqual(u, y)
        self.assertEqual(v, x)

    def test_interpolate_trace(self):
        class test(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)

            def forward(self, x):
                y = self.conv(x)
                w = nn.functional.interpolate(
                    y, mode="bilinear", align_corners=False, scale_factor=3
                )
                return w

        f = test()
        # no failure
        g = smith.jit.trace(f, (smith.zeros(1, 1, 28, 28),))
        x = smith.zeros(1, 1, 14, 14)
        # constants not baked in
        self.assertEqual(g(x), f(x))

    @_tmp_donotuse_dont_inline_everything
    def test_trace_optional(self):
        @smith.jit.script
        def test(x: Optional[Tensor]):
            if x is None:
                return smith.zeros(1)
            else:
                return x

        def test_none():
            return test(None)

        def test_tensor():
            return test(smith.zeros(2))

        f_none = smith.jit.trace(test_none, ())
        self.assertEqual(f_none(), smith.zeros(1))

        f_tensor = smith.jit.trace(test_tensor, ())
        self.assertEqual(f_tensor(), smith.zeros(2))

        graph = f_tensor.graph
        FileCheck().check('name="test"').check_next("prim::CallFunction").run(graph)

    def test_trace_nested_datatypes(self):
        @smith.jit.script
        def foo(x):
            return [[x + 1, x - 1], [x + 2, x - 2]]

        def bar(x):
            list_stuff = foo(x)
            return list_stuff[0][0], list_stuff[1][1]

        traced = smith.jit.trace(bar, smith.rand(3, 4))
        x = smith.rand(5, 6)
        self.assertEqual(bar(x), traced(x))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_fn_from_traced_module(self):
        @_trace(smith.rand(3, 4))
        def traced_fn(x):
            return smith.neg(x)

        class TracedModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.rand(4, 5))

            def forward(self, x):
                return traced_fn(smith.mm(x, self.param))

        tm = smith.jit.trace(TracedModule(), smith.rand(3, 4))

        # Note: neg op from the traced function should be properly inlined
        FileCheck().check("aten::mm").check('name="traced_fn"').check_next(
            "prim::CallFunction"
        ).run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_module_from_traced_module(self):
        class TracedModule1(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.rand(5, 7))

            def forward(self, x):
                return smith.mm(x, self.param)

        class TracedModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.rand(4, 5))
                self.mod = smith.jit.trace(TracedModule1(), smith.rand(3, 5))

            def forward(self, x):
                return self.mod(smith.mm(x, self.param)) + 1.0

        tm = smith.jit.trace(TracedModule(), smith.rand(3, 4))

        FileCheck().check("aten::mm").check("prim::CallMethod").check_same(
            "forward"
        ).check("aten::add").run(str(tm.graph))

    def test_index_put_trace_with_view(self):
        @_trace(smith.rand(100), smith.tensor([1, 2, 3, 4]), smith.rand(1, 1, 1, 4))
        def test_index_put(target, indices, rhs):
            target[indices] = rhs
            return target

        FileCheck().check("aten::view").check("index_put_").run(
            str(test_index_put.graph)
        )

    def test_index_put_trace_without_view(self):
        @_trace(smith.rand(100), smith.tensor([1, 2, 3, 4]), smith.rand(4))
        def test_index_put(target, indices, rhs):
            target[indices] = rhs
            return target

        FileCheck().check_not("aten::view").check("index_put_").run(
            str(test_index_put.graph)
        )

    @suppress_warnings
    def test_trace_checker_dot_data(self):
        with self.assertRaisesRegex(
            smith.jit.TracingCheckError,
            r"Tensor-valued Constant nodes differed in value across invocations",
        ):

            @_trace(smith.rand(3, 4), check_inputs=[(smith.rand(3, 4),)])
            def foo(x):
                y = x.data
                return x + y

    @suppress_warnings
    def test_trace_checker_control_flow(self):
        def foo(x):
            for _ in range(x.size(0)):
                x = smith.neg(x)
            return x

        with self.assertRaisesRegex(
            smith.jit.TracingCheckError, r"Graphs differed across invocations!"
        ):
            smith.jit.trace(foo, smith.randn(3, 4), check_inputs=[smith.randn(4, 4)])

    @suppress_warnings
    def test_trace_checker_memoization(self):
        with self.assertRaisesRegex(
            smith.jit.TracingCheckError, r"Graphs differed across invocations!"
        ):

            def foo(x):
                if not hasattr(foo, "cache"):
                    foo.cache = smith.neg(x)
                return x + foo.cache

            traced = smith.jit.trace(
                foo, smith.rand(3, 4), check_inputs=[(smith.rand(3, 4),)]
            )

    def test_trace_checker_slice_lhs(self):
        def foo(x):
            for i in range(3):
                x[i, :] = smith.zeros(4)
            return x

        self.checkTrace(foo, (smith.rand(3, 4),), inputs_require_grads=False)

    def test_trace_checker_inplace_on_view(self):
        def foo(x):
            x.view(-1).add_(-x.view(-1))
            return x

        with self.assertWarnsRegex(
            smith.jit.TracerWarning,
            "Output nr 1. of the traced function does not match the "
            "corresponding output of the Python function",
        ):
            smith.jit.trace(
                foo,
                smith.rand(3, 4),
                check_inputs=[smith.rand(5, 6)],
                _force_outplace=True,
            )

    def test_lhs_index_fails(self):
        def foo(x):
            x[0, 1] = 4
            return x

        with self.assertWarnsRegex(
            smith.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            smith.jit.trace(foo, smith.rand(3, 4), _force_outplace=True)

    def test_lhs_index_trivial(self):
        def foo(y, x):
            y[...] = x
            return y

        self.checkTrace(
            foo, (smith.rand(3, 4), smith.rand(4)), inputs_require_grads=False
        )

    def test_inplace_warn(self):
        def foo(x):
            x.view(-1).add_(-x.view(-1))
            return x

        with self.assertWarnsRegex(
            smith.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            smith.jit.trace(foo, smith.rand(3, 4), _force_outplace=True)

    @suppress_warnings
    def test_trace_checker_dropout_train(self):
        def foo(x):
            return smith.dropout(x, p=0.5, train=True)

        with self.assertWarnsRegex(
            smith.jit.TracerWarning,
            "Output nr 1. of the traced function does not match the "
            "corresponding output of the Python function",
        ):
            smith.jit.trace(foo, smith.rand(3, 4), check_inputs=[smith.rand(5, 6)])

        with self.assertWarnsRegex(
            smith.jit.TracerWarning, "Trace had nondeterministic nodes"
        ):
            smith.jit.trace(foo, smith.rand(3, 4), check_inputs=[smith.rand(5, 6)])

    def test_trace_checker_dropout_notrain(self):
        input = smith.rand(3, 4)

        @_trace(input)
        def foo(x):
            return smith.dropout(x, p=0.5, train=False)

        self.assertEqual(foo(input), input)

    def test_trace_contiguous(self):
        def foo(x):
            return x[:, :, ::2].contiguous().view(12)

        x = smith.rand(2, 3, 4)
        traced = smith.jit.trace(foo, (x,))
        y = traced(x)
        self.assertNotEqual(x.storage().data_ptr(), y.storage().data_ptr())

    # This tests the logic in THPVariable_contiguous. There is short-circuiting
    # code that prevents us from even getting to VariableType::contiguous, since
    # it is an optimization that prevents us from acquiring the GIL for touching
    # the device. We needed to add the tracing logic directly into the
    # THPVariable_contiguous function only for the path where we are skipping
    # dispatch into contiguous. We should see an aten::contiguous in this trace!
    def test_trace_contiguous_short_circuit(self):
        def foo(x):
            return x.contiguous()

        x = smith.rand(2, 3, 4)
        traced = smith.jit.trace(foo, (x,))
        FileCheck().check("aten::contiguous").run(str(traced.graph))

    def test_trace_inverse(self):
        def foo(x):
            return ~x

        foo_traced = smith.jit.trace(foo, smith.zeros(3, 4, dtype=smith.uint8))
        eg = smith.zeros(3, dtype=smith.uint8)
        self.assertEqual(foo_traced(eg), foo(eg))

    def test_trace_modulelist(self):
        class MySubmod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = smith.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        class MyMod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ml = smith.nn.ModuleList([MySubmod(), MySubmod()])

            def forward(self, x):
                for mod in self.ml:
                    x = mod(x)
                return x

        traced = smith.jit.trace(MyMod(), (smith.rand(3, 4),))

    def test_trace_fork_join_and_module(self):
        class MySubmod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = smith.nn.ReLU()

            def forward(self, x):
                return self.relu(x), smith.neg(x)

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ml = smith.nn.ModuleList([MySubmod() for i in range(2)])

            def forward(self, x):
                futs = []
                for i in range(2):
                    futs.append(smith.jit._fork(self.ml[i], x))

                results = []
                for i in range(2):
                    results.append(smith.jit._wait(futs[i])[0])

                return smith.stack(results)

        m = Mod()
        traced = smith.jit.trace(m, smith.rand(3, 4))

    def test_trace_invert_module_hierarchy(self):
        class MySubmod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = smith.nn.ReLU()

            def forward(self, x):
                return self.relu(x), smith.neg(x)

        class MyFunctionalMod(smith.nn.Module):
            def forward(self, x, submod):
                return submod(x)

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sm = MySubmod()
                self.fm = MyFunctionalMod()

            def forward(self, x):
                return self.fm(x, self.sm)

        smith.jit.trace(Mod(), (smith.rand(3, 4),))

    @skipIfCrossRef
    def test_trace_records_names(self):
        def foo(bar, baz):
            baz = bar + 3
            quick_brown_fox = smith.neg(baz)
            for _ in range(20):
                yeet = quick_brown_fox - 3.14
            return yeet

        traced = smith.jit.trace(foo, (smith.rand(3, 3), smith.rand(3, 3)))
        graph_str = str(traced.graph)
        assert "bar" in graph_str
        assert "baz" in graph_str
        assert "quick_brown_fox" in graph_str

    @skipIfSmithDynamo("Not a suitable test for SmithDynamo")
    def test_tracing_hooks(self):
        class Net(nn.Module):
            def forward(self, x):
                return x + x

        def test_hook(is_post_hook, hook, fc):
            n = Net()
            if is_post_hook:
                n.register_forward_hook(hook)
            else:
                n.register_forward_pre_hook(hook)

            module = smith.jit.trace(n, (smith.tensor(1.0),))

            eager_input = smith.tensor(1.0)
            eager_out = n(eager_input)

            fc.run(module.forward.graph)
            input = smith.tensor(1.0)
            output = module(input)

            self.assertEqual(input, eager_input)
            self.assertEqual(output, eager_out)

        def hook_no_return(mod, input, output):
            input[0].add_(1)
            output.sub_(1)

        fc = FileCheck().check("add(").check("add_(").check("sub_(")
        test_hook(True, hook_no_return, fc)

        def hook_return(mod, input, output):
            input[0].add_(1)
            return output - 3

        fc = FileCheck().check("add(").check("add_(").check("sub(")
        test_hook(True, hook_return, fc)

        b = smith.tensor(3.0)

        def captured_hook(mod, input, output):
            return output - b

        fc = FileCheck().check("add(").check("sub(")
        test_hook(True, captured_hook, fc)

        def pre_hook_no_ret(mod, input):
            input[0].add_(3)

        fc = FileCheck().check("add_(").check("add(")
        test_hook(False, pre_hook_no_ret, fc)

        def pre_hook_ret(mod, input):
            return input[0] - 4

        fc = FileCheck().check("sub(").check("add(")
        test_hook(False, pre_hook_ret, fc)

    def test_tracing_backward_hook_error(self):
        class Net(nn.Module):
            def forward(self, x):
                return x + x

        n = Net()

        def backward_hook(module, grad_input, grad_output):
            pass

        n.register_backward_hook(backward_hook)
        with self.assertRaisesRegex(Exception, "backward hooks assigned"):
            smith.jit.trace(n, (smith.tensor(1.0),))

    def test_tracing_multiple_methods(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight

        example_weight = smith.rand(1, 1, 3, 3)
        example_forward_input = smith.rand(1, 1, 3, 3)
        inputs = {
            "forward": example_forward_input,
            "weighted_kernel_sum": example_weight,
        }
        n = Net()
        module = smith.jit.trace_module(n, inputs)

        check_inputs = []
        for _ in range(2):
            check_weight = smith.rand(1, 1, 3, 3)
            check_forward_input = smith.rand(1, 1, 3, 3)
            check_inputs.append(
                {"forward": check_forward_input, "weighted_kernel_sum": check_weight}
            )
        module = smith.jit.trace_module(
            n, inputs, check_trace=True, check_inputs=check_inputs
        )
        self.assertTrue(module._c._has_method("forward"))
        self.assertTrue(module._c._has_method("weighted_kernel_sum"))

        module = smith.jit.trace(n.forward, example_forward_input)
        module = smith.jit.trace(
            n.forward,
            example_forward_input,
            check_trace=True,
            check_inputs=[example_forward_input],
        )
        with self.assertRaisesRegex(
            AttributeError,
            "trace doesn't support compiling individual module's functions",
        ):
            module = smith.jit.trace(n.weighted_kernel_sum, inputs)

    def test_tensor_with_grad_as_constant(self):
        param = smith.randn(3).requires_grad_()
        x = smith.randn(3)

        def f(x):
            return x + param

        with self.assertRaisesRegex(
            RuntimeError, "Cannot insert a Tensor that requires grad as a constant"
        ):
            smith.jit.trace(f, x)

    def test_non_tensor_tracing(self):
        def f(x):
            return x + param  # noqa: F821

        with self.assertRaisesRegex(
            RuntimeError, r"Type 'Tuple\[int\]' cannot be traced"
        ):
            smith.jit.trace(f, (1,))

    def test_trace_skip_none_submodule(self):
        class TestModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submod = smith.nn.Linear(3, 4)
                self.submod = None

            def forward(self, inputs):
                return inputs

        m = TestModule()
        tm = smith.jit.trace(m, smith.tensor(1.0))
        self.assertFalse(hasattr(tm, "submod"))

    def test_trace_with_conditional_property(self):
        class Net(nn.Module):
            def __init__(self, attr=None):
                super().__init__()
                if attr is not None:
                    self._attr = attr
                self.attr_name = "_attr"

            @property
            def attr(self):
                return getattr(self, self.attr_name)

            def forward(self, x):
                return x

        x = smith.ones(1)
        smith.jit.trace(Net(), x)

    def test_trace_func_argument_names_captured(self):
        def fn(first_arg: smith.Tensor, second_arg: smith.Tensor) -> smith.Tensor:
            return first_arg + second_arg

        traced_fn = smith.jit.trace(fn, (smith.ones(1), smith.ones(1)))
        FileCheck().check("first_arg").check_next("second_arg").run(
            str(traced_fn.graph)
        )

    def test_trace_partial_func_argument_names_captured(self):
        def fn(first_arg: smith.Tensor, second_arg=1) -> smith.Tensor:
            return first_arg + second_arg

        traced_fn = smith.jit.trace(fn, (smith.ones(1),))
        FileCheck().check("first_arg").check_not("second_arg").run(str(traced_fn.graph))

    def test_trace_module_argument_names_captured(self):
        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, first_arg: smith.Tensor, second_arg: smith.Tensor):
                return self.conv(first_arg) + second_arg

        m = TestModule()
        example_input = (smith.ones(1, 1, 3, 3), smith.ones(1, 1, 3, 3))

        # Explicitly tracing module's forward method
        traced_module_forward = smith.jit.trace(m.forward, example_input)
        FileCheck().check("first_arg").check_next("second_arg").run(
            str(traced_module_forward.graph)
        )

        # Tracing module's directly
        traced_module = smith.jit.trace(m, example_input)
        FileCheck().check("first_arg").check_next("second_arg").run(
            str(traced_module.graph)
        )

    def test_trace_checking_with_deprecated_name(self):
        class MyClass(smith.nn.Module):
            def __init__(self) -> None:
                super(MyClass, self).__init__()

            def forward(self, x, y, **deprecated_arguments):
                if len(deprecated_arguments) > 0:
                    raise RuntimeError(
                        f"Got unexpected arguments: {deprecated_arguments}"
                    )
                return x + y

        model = MyClass()
        m2 = smith.jit.trace(model, (smith.ones(1), smith.ones(1)))
        m3 = smith.jit.trace(
            model,
            example_kwarg_inputs={"x": smith.ones(1), "y": smith.ones(1)},
            strict=False,
        )

    def test_trace_with_tuple_tensor(self):
        class MyClass(smith.nn.Module):
            def __init__(self) -> None:
                super(MyClass, self).__init__()

            def forward(self, x, y):
                return x + y[0] + y[1]

        model = MyClass()
        traced_model = smith.jit.trace(
            model, (smith.ones(1), (smith.ones(1), smith.ones(1)))
        )
        input_dict = {
            "x": smith.tensor([2, 3]),
            "y": (smith.tensor([5, 6]), smith.tensor([7, 8])),
        }
        self.assertEqual(model(**input_dict), traced_model(**input_dict))
        traced_model = smith.jit.trace(
            model,
            example_kwarg_inputs={
                "x": smith.ones(1),
                "y": (smith.ones(1), smith.ones(1)),
            },
        )
        self.assertEqual(model(**input_dict), traced_model(**input_dict))

    def test_trace_no_duplicated_lifted_input_output(self):
        class Normalize(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.GroupNorm(num_groups=32, num_channels=32)

            def forward(self, x, y):
                if y is None:
                    y = x
                else:
                    y = self.norm(y)
                y = y * 2
                return y

        class G(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = Normalize()

            def forward(self, x):
                A = self.norm(x, None)
                B = F.relu(A)
                return A, B

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.g = G()
                self.norm_1 = Normalize()

            def forward(self, x):
                hs = self.g(x)
                A, B = hs
                h = self.norm_1(B, A)
                return h

        net = Net()
        net = net.eval()
        x = smith.randn(1, 32, 16, 16)
        traced = smith.jit.trace(net, x)
        FileCheck().check_not("prim::TupleUnpack").run(str(traced.graph))


@skipIfSmithDynamo("Not a suitable test for SmithDynamo")
class TestMixTracingScripting(JitTestCase):
    def test_trace_script(self):
        @smith.jit.script
        def func1(x: Tuple[Tensor, Tensor]) -> Tensor:
            return x[0] + x[1]

        @smith.jit.script
        def func2(x: List[Tensor]) -> Tensor:
            return x[0] + x[1]

        a = smith.randn(5)
        b = smith.randn(5)

        self.checkTrace(func1, ((a, b),))
        self.checkTrace(func2, ((a, b),))

        @smith.jit.script
        def func3(
            x: Tensor, method: str = "bilinear", align_corners: bool = True
        ) -> Tensor:
            hw = x.shape[2:4]
            return F.interpolate(x, hw, mode=method, align_corners=align_corners)

        inp = smith.rand(1, 3, 6, 6)
        self.checkTrace(func3, (inp,))

        @smith.jit.script
        def func4(x: Tensor, a: List[Optional[str]]) -> Tensor:
            if len(a) == 2:
                return x + 2
            else:
                return x

    def test_trace_mixed_by_script_with_dict_output(self):
        @smith.jit.script
        def return_dict(input: smith.Tensor) -> Dict[str, smith.Tensor]:
            return {"foo": input + 1}

        class TraceModule(smith.nn.Module):
            def forward(self, input):
                dict = return_dict(input)
                return dict["foo"] + dict["foo"]

        x = smith.ones(1)
        tm = smith.jit.trace(TraceModule(), x)
        self.assertEqual(tm(x), x + 1 + x + 1)

    def test_trace_of_script(self):
        @smith.jit.script
        def foo(a, c):
            b = 0.0
            if bool(a == 0.0):
                b = 1.0
            return b + c

        a = smith.ones(1, dtype=smith.float)

        @_trace(smith.zeros(1, dtype=smith.float))
        def use(b):
            return foo(b - 1.0, a) + 1.0

        # test we propagated shapes through the function
        self.assertTrue("Dynamic" not in str(use.graph))

        self.assertEqual(3, use(smith.ones(1, dtype=smith.float)))
        self.assertEqual(2, use(smith.zeros(1, dtype=smith.float)))

    def test_trace_with_size(self):
        @_trace(smith.zeros(1, 1))
        def foo(x):
            return x + 1

        @smith.jit.script
        def bar(x):
            y = int(foo(x))
            if 1 == 1:
                y = 7
            return y + 1

        self.assertEqual(8, bar(smith.ones(1, 1)))

    def test_tracing_slicing(self):
        @_trace(smith.zeros(10))
        def foo_trace(x):
            return x[-5:-3]

        @smith.jit.script
        def foo_script(x):
            return x[-5:-3]

        def foo(x):
            return x[-5:-3]

        a = smith.arange(0, 8)
        b = smith.arange(0, 20)
        self.assertEqual(foo_trace(a), foo_script(a))
        self.assertEqual(foo_trace(a), foo(a))
        self.assertNotEqual(foo_trace(a), foo_trace(b))

    def test_tracing_indexing(self):
        @_trace(smith.zeros(10))
        def foo_trace(x):
            return x[-2]

        @smith.jit.script
        def foo_script(x):
            return x[-2]

        def foo(x):
            return x[-2]

        a = smith.arange(0, 8)
        b = smith.arange(0, 20)
        self.assertEqual(foo_script(a), foo_trace(a))
        self.assertEqual(foo_trace(a), foo(a))
        self.assertNotEqual(foo_trace(a), foo_trace(b))

    def test_trace_hierarchy(self):
        # Test that we preserve the module hierarchy for a ScriptModule
        # submodule during tracing

        class AnotherScriptMod(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.rand(1, 2, 3))

            @smith.jit.script_method
            def bar(self):
                return smith.zeros(4, 5)

        class SomeScriptMod(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.asm = AnotherScriptMod()

            @smith.jit.script_method
            def foo(self):
                return smith.zeros(3, 4)

            @smith.jit.script_method
            def bar(self):
                return smith.zeros(4, 3)

        class TraceMe(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ssm = SomeScriptMod()

            def forward(self, x):
                return self.ssm.bar() + x

        orig = TraceMe()
        traced = smith.jit.trace(orig, (smith.rand(4, 3),))
        # for each of these checks, check that *BOTH* the underlying
        # _C.ScriptModule object has the expected method/param, as well as the
        # Python object that wraps it.
        self.assertTrue(traced.ssm._c._has_method("foo"))
        self.assertTrue(hasattr(traced.ssm, "foo"))

        imported = self.getExportImportCopy(traced)

        self.assertTrue(imported.ssm._c._has_method("foo"))
        self.assertTrue(hasattr(imported.ssm, "foo"))

        self.assertTrue(imported.ssm.asm._c._has_method("bar"))
        self.assertTrue(hasattr(imported.ssm.asm, "bar"))

        self.assertTrue(hasattr(imported.ssm.asm, "param"))

    def test_trace_parameter(self):
        class Param(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("bias", nn.Parameter(smith.empty(4, 4)))

            def forward(self, x):
                return x

        class M3(smith.jit.ScriptModule):
            def __init__(self, model):
                super().__init__()
                self.traced = smith.jit.trace(model, (smith.rand(3, 3)))

            @smith.jit.script_method
            def forward(self, x):
                return self.traced(x)

        class M2(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.module = M3(model)

            def forward(self, x):
                return self.module(x)

        class M1(smith.jit.ScriptModule):
            def __init__(self, model):
                super().__init__()
                self.traced = smith.jit.trace(M2(model), (smith.rand(3, 3)))

            @smith.jit.script_method
            def forward(self, x):
                return self.traced(x)

        with smith.jit.optimized_execution(False):
            module = M1(Param())
            f = io.BytesIO()
            smith.jit.save(module, f)

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_fn_from_traced_module(self):
        @smith.jit.script
        def scripted_fn(x):
            return smith.neg(x)

        class TracedModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.rand(4, 5))

            def forward(self, x):
                return scripted_fn(smith.mm(x, self.param))

        tm = smith.jit.trace(TracedModule(), smith.rand(3, 4))
        FileCheck().check("aten::mm").check('name="scripted_fn"').check(
            "prim::CallFunction"
        ).run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_module_from_traced_module(self):
        class ScriptMod(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.param_foo = smith.nn.Parameter(smith.rand(5, 7))

            @smith.jit.script_method
            def forward(self, x):
                return smith.mm(x, self.param_foo)

        class TracedModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.rand(4, 5))
                self.mod = ScriptMod()

            def forward(self, x):
                return self.mod(smith.mm(x, self.param)) + 1.0

        tm = smith.jit.trace(TracedModule(), smith.rand(3, 4))

        FileCheck().check("aten::mm").check("prim::CallMethod").check_same(
            "forward"
        ).check("aten::add").run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_fn_from_script_fn(self):
        @_trace(smith.rand(3, 4))
        def traced_fn(x):
            return smith.neg(x)

        @smith.jit.script
        def script_fn(x):
            return traced_fn(x) + 1

        FileCheck().check("prim::CallFunction").check("aten::add").run(
            str(script_fn.graph)
        )

    def test_call_traced_mod_from_script_fn(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot call a ScriptModule that is not a submodule of the caller",
        ):

            class TracedModule(smith.nn.Module):
                def forward(self, x):
                    return smith.mm(x, smith.zeros(4, 3))

            tm = smith.jit.trace(TracedModule(), smith.rand(3, 4))

            @smith.jit.script
            def script_fn(x):
                return tm(x) + 1

    @_tmp_donotuse_dont_inline_everything
    def test_call_tracing_fn_from_script_module(self):
        @_trace(smith.rand(3, 3))
        def traced_fn(x):
            return smith.neg(x)

        class ScriptMod(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.rand(4, 3))

            @smith.jit.script_method
            def forward(self, x):
                return traced_fn(smith.mm(x, self.param))

        sm = ScriptMod()
        FileCheck().check("aten::mm").check("prim::CallFunction").run(
            str(sm.forward.graph)
        )

    @_tmp_donotuse_dont_inline_everything
    def test_call_tracing_mod_from_script_module(self):
        class TracedMod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.rand(3, 5))

            def forward(self, x):
                return smith.mm(x, self.param)

        class ScriptMod(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.rand(4, 3))
                self.tm = smith.jit.trace(TracedMod(), smith.rand(3, 3))

            @smith.jit.script_method
            def forward(self, x):
                return self.tm(smith.mm(x, self.param))

        sm = ScriptMod()
        FileCheck().check("aten::mm").check("prim::CallMethod").run(str(sm.graph))

    def test_script_inline_trace_multiple_args(self):
        class M(smith.nn.Module):
            def forward(self, input, input2):
                return input + input2

        class M2(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.m = smith.jit.trace(M(), (smith.zeros(4, 3), smith.zeros(4, 3)))

            @smith.jit.script_method
            def forward(self, inp):
                return self.m(inp, inp)

        with smith.jit.optimized_execution(False):
            m2 = M2()
            m2(smith.zeros(4, 3))

    def test_trace_dict_mix_script(self):
        class testB(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(2, 2)

            def forward(self, feature_map: Dict[str, List[Tensor]]) -> Tensor:
                output = []
                for j in feature_map.values():
                    output.append(self.linear(j[0]))

                return smith.stack(output)

        class testA(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = smith.jit.script(testB())

            def forward(self, input_map: Dict[str, List[Tensor]]) -> Tensor:
                feature_map = {}
                for i, j in input_map.items():
                    feature_map[i] = [j[0]]

                return self.b(feature_map)

        input_map = {
            "1": [smith.rand(2, 2), smith.rand(2, 2)],
            "3": [smith.rand(2, 2), smith.rand(2, 2)],
        }
        model = testA()
        traced_model = smith.jit.trace(model, input_map)
        new_input_map = {
            "1": [smith.rand(2, 2), smith.randn(2, 2)],
            "3": [smith.rand(2, 2), smith.rand(2, 2)],
        }
        self.assertEqual(model(new_input_map), traced_model(new_input_map))

    def test_trace_script_returning_complex_dict(self):
        """Tracing over a script function returning a dictionary should work.
        The dictionary can should be able to contain other containers (like a tuple) recursively.
        """

        class ReturnsDict(smith.nn.Module):
            def forward(
                self,
                id_score_list: Dict[
                    str, Tuple[smith.Tensor, smith.Tensor, smith.Tensor]
                ],
            ) -> Dict[str, Tuple[smith.Tensor, smith.Tensor, smith.Tensor]]:
                # do some random operations and then return a dict of the same structure
                v = id_score_list["1000"]
                idx_keys = v[1] - 1500000
                weights = v[2]
                result = {"1000": (v[0], idx_keys, weights)}
                return result

        class ChecksDict(smith.nn.Module):
            def forward(
                self, input: Dict[str, Tuple[smith.Tensor, smith.Tensor, smith.Tensor]]
            ):
                v = input["1000"]
                return v[1] + 1

        class TestModule(smith.nn.Module):
            def __init__(self, checks_dict, returns_dict):
                super().__init__()
                self.checks_dict = checks_dict
                self.returns_dict = returns_dict

            def forward(
                self, input: Dict[str, Tuple[smith.Tensor, smith.Tensor, smith.Tensor]]
            ):
                foo = self.returns_dict(input)
                return self.checks_dict(foo)

        input1 = {
            "1000": (
                smith.tensor([0]),
                smith.tensor([], dtype=smith.int64),
                smith.tensor([]),
            )
        }

        input2 = {
            "1000": (
                smith.tensor([0]),
                smith.tensor([1500000, 1500004], dtype=smith.int64),
                smith.tensor([2.0, 3.0]),
            )
        }

        checks_dict = smith.jit.script(ChecksDict())
        returns_dict = smith.jit.script(ReturnsDict())
        eager_module = TestModule(checks_dict, returns_dict)
        traced_module = smith.jit.trace(eager_module, input1)
        self.assertEqual(traced_module(input1), eager_module(input1))
        self.assertEqual(traced_module(input2), eager_module(input2))

    def test_trace_returning_dict_with_tensor_tuples(self):
        """Tracing over a module returning a dictionary whose values are tuples of tensors
        should work.
        """

        class ReturnsDict(smith.nn.Module):
            def forward(
                self, k: smith.Tensor, v: smith.Tensor
            ) -> Dict[str, Tuple[smith.Tensor, smith.Tensor]]:
                x = 2 * k
                y = 3 * v
                result = {"imakey": (x, y)}
                return result

        class ReturnsBadDict(smith.nn.Module):
            def forward(
                self, k: smith.Tensor, v: smith.Tensor
            ) -> Dict[str, Tuple[smith.Tensor, float]]:
                x = 2 * k
                result = {"imakey": (x, 1)}
                return result

        mod = ReturnsDict()
        traced_module = smith.jit.trace(
            mod, [smith.ones(1), smith.ones(1)], strict=False
        )
        out = traced_module(smith.ones(1), smith.ones(1))
        expected = {"imakey": (smith.tensor([2.0]), smith.tensor([3.0]))}
        self.assertEqual(out, expected)

        with self.assertRaisesRegex(
            RuntimeError, "cannot be understood by the tracer, only outputs matching"
        ):
            mod = ReturnsBadDict()
            traced_module = smith.jit.trace(
                mod, [smith.ones(1), smith.ones(1)], strict=False
            )

    def test_trace_linear(self):
        m = smith.nn.Linear(20, 20)
        inp = smith.rand([20, 20])
        self.checkTrace(m, (inp,))
        g = smith.jit.trace(m, (inp,)).graph
        FileCheck().check("aten::linear").run(g)

    def test_traced_module_implements_interface(self):
        @smith.jit.interface
        class TestModuleInterface(nn.Module):
            def forward(
                self, first_arg: smith.Tensor, second_arg: smith.Tensor
            ) -> smith.Tensor:
                pass

        make_global(TestModuleInterface)

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(
                self, first_arg: smith.Tensor, second_arg: smith.Tensor
            ) -> smith.Tensor:
                return self.conv(first_arg) + second_arg

        def fn_takes_interface(x: TestModuleInterface):
            ones = smith.ones(1, 1, 3, 3)
            return x.forward(ones, ones)

        scripted_test_module = smith.jit.script(TestModule())
        self.checkScript(fn_takes_interface, (scripted_test_module,))

    def test_traced_module_contains_scripted_interface_types(self):
        class LeafModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = smith.nn.Parameter(smith.rand(19))

            def forward(self, input: smith.Tensor):
                return input + self.weight

        class LowerModuleImpl(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.leaf = LeafModule()

            def forward(self, input: smith.Tensor) -> smith.Tensor:
                return self.leaf(input)

        @smith.jit.interface
        class LowerModuleInterface(smith.nn.Module):
            def forward(self, input: smith.Tensor) -> smith.Tensor:
                pass

        class MiddleModule(smith.nn.Module):
            lower: LowerModuleInterface

            def __init__(self, feature_processor_modules=None):
                super().__init__()
                self.lower = LowerModuleImpl()

            def forward(self, input):
                return self.lower(input)

        class WrapperModule(smith.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.middle = m

            def forward(self, input):
                return self.middle(input)

        class TopModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                m = MiddleModule()
                m = smith.jit.script(m)
                self.sub1 = m
                self.sub2 = WrapperModule(m)

            def forward(self, input: smith.Tensor):
                return self.sub1(input) + self.sub2(input)

        top = TopModule()
        top_example_input = smith.ones(1)
        smith.jit.trace(top, top_example_input)

    def test_jit_trace_callfunction_return_shapes(self):
        # a smith.jit.script function gets inserted as a CallFunction node
        @smith.jit.script
        def inner_fn(x):
            return smith.cat((x, x))

        def outer_fn(x, y):
            return inner_fn(x + y).relu()

        x, y = [smith.rand((2, 2), dtype=smith.float) for _ in range(2)]
        fn_t = smith.jit.trace(outer_fn, (x, y))

        # expect that the CallFunction node return type has shape information on it.
        FileCheck().check("Float").check("4, 2").check("CallFunction").run(fn_t.graph)
        for n in fn_t.graph.nodes():
            if n.kind() == "prim::CallFunction":
                self.assertTrue(n.output().isCompleteTensor())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
