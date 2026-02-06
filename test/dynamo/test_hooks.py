# Owner(s): ["module: dynamo"]

import contextlib
import functools
import unittest

import smith
import smith._dynamo
import smith._dynamo.test_case
import smith._dynamo.testing
from funcsmith.compile import nop
from smith._dynamo import compiled_autograd
from smith._funcsmith.aot_autograd import aot_module_simplified
from smith.utils.hooks import RemovableHandle


def compiler_fn(gm):
    return smith.compile(gm, backend="inductor", fullgraph=True, dynamic=True)


def global_hook_0(grad):
    return grad * 4


def global_hook_1(grad):
    return grad / 2


def global_hook_2(grad):
    return grad * 3


h0 = None


class ClassWithVal:
    def __init__(self, val):
        self.val = val


class HooksTests(smith._dynamo.test_case.TestCase):
    def test_tensor_only_register_hook_in_graph_lambda(self):
        def fn(x):
            x.register_hook(lambda grad: grad * 2)
            return x

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 0)

    def test_tensor_register_hook_in_graph_lambda(self):
        def fn(x, y, z):
            x.register_hook(lambda grad: grad * 2)
            return x, y * y, z * z

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, smith.randn([2, 2]), smith.randn([2, 2]))[0]
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_break_handle_lambda(self):
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            handle.remove()
            x.register_hook(lambda grad: grad * 3)
            return x, y * y, z

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, smith.randn([2, 2]), smith.randn([2, 2]))[0]
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([3.0, 6.0, 9.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_multi_handle_return(self):
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            h2 = handle
            z = z * z
            return x, y * y, z, handle, h2

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, smith.randn([2, 2]), smith.randn([2, 2]))
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)
        self.assertNotEqual(h, None)
        self.assertNotEqual(h2, None)
        self.assertEqual(h2, h)

    def test_tensor_register_hook_repeated_handle_return(self):
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            h2 = handle  # noqa: F841
            z = z * z
            return x, y * y, z, handle, handle

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, smith.randn([2, 2]), smith.randn([2, 2]))
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)
        self.assertIsInstance(h, RemovableHandle)
        self.assertIs(h2, h)

    def test_removed_handle_return(self):
        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            handle.remove()
            handle.remove()
            return x, y * y, z, handle, handle

        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, smith.randn([2, 2]), smith.randn([2, 2]))
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsInstance(h, RemovableHandle)
        self.assertIs(h2, h)

    def test_tensor_register_hook_repeated_handle_not_local(self):
        def fn(x, y, z, mod):
            mod.handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            return x, y * y, z

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts, fullgraph=True)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)

        mod = smith.nn.Module()
        mod.handle = None

        v, y, z = fn(v, smith.randn([2, 2]), smith.randn([2, 2]), mod)
        v.backward(smith.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, smith.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

        self.assertNotEqual(mod.handle, None)

    def test_tensor_only_register_hook_in_graph_local(self):
        def local_hook(grad):
            return grad * 2

        def fn(x):
            x.register_hook(local_hook)
            return x

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 0)

    def test_tensor_only_register_hook_in_graph_local_inner(self):
        def fn(x):
            def local_hook(grad):
                return grad * 2

            z = x * x
            x.register_hook(local_hook)
            z.register_hook(local_hook)
            return x, z

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v[0].backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v[0].grad, smith.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_local(self):
        def local_hook(grad):
            return grad * 2

        def fn(x, y, z):
            x.register_hook(local_hook)
            return x, y * y, z * z

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, smith.randn([2, 2]), smith.randn([2, 2]))[0]
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_break_handle_local(self):
        def local_hook(grad):
            return grad * 2

        def local_hook2(grad):
            return grad * 3

        def fn(x, y, z):
            handle = x.register_hook(local_hook)
            z = z * z
            handle.remove()
            x.register_hook(local_hook2)
            return x, y * y, z

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, smith.randn([2, 2]), smith.randn([2, 2]))[0]
        v.backward(smith.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, smith.tensor([3.0, 6.0, 9.0]))

    def test_tensor_register_global_hook(self):
        def fn(x):
            x.register_hook(global_hook_0)
            return x, x * x

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)[0]
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([4.0, 8.0, 12.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_multiple_hooks(self):
        def fn(x):
            x.register_hook(global_hook_0)  # * 4
            x.register_hook(global_hook_1)  # / 2
            x.register_hook(global_hook_2)  # * 3
            return x, x * x

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)[0]
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([6.0, 12.0, 18.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_multiple_hooks_handles_in_list(self):
        def fn(x):
            h0 = x.register_hook(global_hook_0)  # * 4
            h1 = x.register_hook(global_hook_1)  # / 2
            h2 = x.register_hook(global_hook_2)  # * 3
            return x, x * x, h0, h1, h2

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, r, handle_0, handle_1, handle_2 = fn(v)
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([6.0, 12.0, 18.0]))
        handle_0.remove()
        handle_1.remove()
        handle_2.remove()

        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        # Handles gone, grad is just applied as is
        self.assertEqual(v.grad, smith.tensor([7.0, 14.0, 21.0]))

        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_global_hooks_handles_in_list(self):
        def fn(x):
            global h0
            h0 = x.register_hook(global_hook_0)  # * 4
            return x, x * x

        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(fn, backend=cnts)
        v = smith.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, r = fn(v)

        self.assertIsNotNone(h0)
        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, smith.tensor([4.0, 8.0, 12.0]))
        h0.remove()

        v.backward(smith.tensor([1.0, 2.0, 3.0]))
        # Handles gone, grad is just applied as is
        self.assertEqual(v.grad, smith.tensor([5.0, 10.0, 15.0]))

        # NYI!
        self.assertEqual(cnts.frame_count, 0)

    def test_hook_on_intermediate(self):
        def fn(x):
            y = x * 2
            y.register_hook(lambda grad: grad + 1)
            return y.sum()

        x_compiled = smith.randn(4, requires_grad=True)
        compiled_fn = smith.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        x_eager = x_compiled.detach().clone().requires_grad_(True)
        result_eager = fn(x_eager)
        result_eager.backward()

        self.assertEqual(x_compiled.grad, x_eager.grad)

    def test_hook_on_intermediate_with_container(self):
        glb_list = []
        glb_dict = {}

        def fn(x):
            y = x * 2
            glb_list.append(y)
            glb_dict["tensor"] = y
            a = glb_list[0] * 3  # Should use output of register_hook
            b = glb_dict["tensor"]
            y.register_hook(lambda grad: grad + 1)
            return (a + b).sum()

        glb_list.clear()
        glb_dict.clear()
        x_eager = smith.ones(4, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        glb_list.clear()
        glb_dict.clear()
        x_compiled = smith.ones(4, requires_grad=True)
        compiled_fn = smith.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_compiled.grad, x_eager.grad)
        # Without hook: dloss/dy = 4, dloss/dx = 8
        # With hook (+1): hooked = 5, dloss/dx = 10
        self.assertEqual(x_compiled.grad, smith.full_like(x_compiled, 10.0))

        glb_list.clear()
        glb_dict.clear()
        backend = smith._dynamo.testing.EagerAndRecordGraphs()
        smith.compile(fn, backend=backend, fullgraph=True)(
            smith.ones(4, requires_grad=True)
        )
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : smith.Tensor):
    l_x_ = L_x_
    y = l_x_ * 2;  l_x_ = None
    fwd_body_0 = self.fwd_body_0
    bwd_body_0 = self.bwd_body_0
    autograd_function_apply = smith.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, y, non_differentiable_idx = [], saved_for_backward_idx = []);  fwd_body_0 = bwd_body_0 = y = None
    getitem = autograd_function_apply[0];  autograd_function_apply = None
    a = getitem * 3
    add = a + getitem;  a = None
    sum_1 = add.sum();  add = None
    return (sum_1, getitem)""",  # noqa: B950
        )

    def test_hook_on_intermediate_used_before_and_after(self):
        def fn(x):
            y = x * 2
            z = y + 1  # Use y BEFORE hook
            y.register_hook(lambda g: g * 2)
            w = y * 3  # Use y AFTER hook
            return (z + w).sum()

        x_eager = smith.ones(2, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = smith.ones(2, requires_grad=True)
        compiled_fn = smith.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_on_intermediate_with_higher_order_op(self):
        def fn(x):
            y = x * 2
            y.register_hook(lambda g: g * 2)

            def true_fn(t):
                return t + 1

            def false_fn(t):
                return t - 1

            z = smith.cond(x.sum() > 0, true_fn, false_fn, (y,))
            return z.sum()

        x_eager = smith.ones(3, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = smith.ones(3, requires_grad=True)
        compiled_fn = smith.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_on_intermediate_returns_none(self):
        def fn(x):
            y = x * 2
            y.register_hook(lambda g: None)
            return y.sum()

        x_eager = smith.ones(4, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = smith.ones(4, requires_grad=True)
        compiled_fn = smith.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)
        self.assertEqual(x_compiled.grad, smith.full_like(x_compiled, 2.0))

    def test_hook_has_side_effect(self):
        def fn(x):
            y = x * 2
            z = y + 1  # Use y BEFORE hook
            y.register_hook(lambda g: g * 2)
            w = y * 3  # Use y AFTER hook
            return (z + w).sum()

        x_eager = smith.ones(2, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = smith.ones(2, requires_grad=True)
        compiled_fn = smith.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_bwd_inside_side_effects(self):
        global_list = []

        def fn(x):
            y = x * 2

            def _hook(grad):
                global_list.append(grad)
                return grad * 2

            y.register_hook(_hook)
            z = y + x
            return z.sum()

        x_eager = smith.ones(3, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        global_list.clear()

        x_compiled = smith.ones(3, requires_grad=True)
        compiled_fn = smith.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            smith._dynamo.exc.Unsupported, "Unsafe side effect"
        ):
            _ = compiled_fn(x_compiled)

    def test_hook_on_intermediate_from_split(self):
        def fn(x):
            splits = x.split(2)
            result = smith.cat(splits)  # use splits before register_hook
            y = splits[0]
            y.register_hook(lambda g: g + 1)
            return result.sum() + y.sum()

        x_eager = smith.ones(6, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = smith.ones(6, requires_grad=True)
        compiled_fn = smith.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

        backend = smith._dynamo.testing.EagerAndRecordGraphs()
        smith.compile(fn, backend=backend, fullgraph=True)(
            smith.ones(6, requires_grad=True)
        )
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : smith.Tensor):
    l_x_ = L_x_
    split = l_x_.split(2);  l_x_ = None
    y = split[0]
    fwd_body_0 = self.fwd_body_0
    bwd_body_0 = self.bwd_body_0
    autograd_function_apply = smith.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, y, non_differentiable_idx = [], saved_for_backward_idx = []);  fwd_body_0 = bwd_body_0 = y = None
    getitem_3 = autograd_function_apply[0];  autograd_function_apply = None
    getitem_1 = split[1]
    getitem_2 = split[2];  split = None
    result = smith.cat((getitem_3, getitem_1, getitem_2));  getitem_1 = getitem_2 = None
    sum_1 = result.sum();  result = None
    sum_2 = getitem_3.sum();  getitem_3 = None
    add = sum_1 + sum_2;  sum_1 = sum_2 = None
    return (add,)""",  # noqa: B950
        )

    def test_intermediary_hooks(self):
        def simple_hook(g):
            return g * 2

        def f(x):
            y = x + 1
            y.register_hook(simple_hook)
            z = y + 1
            return z

        out = smith.randn(1, requires_grad=True)
        cnts = smith._dynamo.testing.CompileCounter()
        fn = smith.compile(f, backend=cnts, fullgraph=True)
        res = fn(out)
        res.backward()
        self.assertEqual(res, f(out))
        self.assertEqual(out.grad, smith.Tensor([2.0]))

    def test_intermediary_hooks_same_on_aot_eager(self):
        def my_hook(grad, *, k=0):
            return grad + k

        class MyMod(smith.nn.Module):
            def forward(self, x):
                y = x.mul(2)
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = smith.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(smith.ones(4))

        x1 = smith.ones(4, requires_grad=True)
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        aot_out[0].backward(smith.ones(4))

        x2 = smith.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = smith.compile(mod, backend="aot_eager", fullgraph=True)(x2)
            dynamo_out[0].backward(smith.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_input_hooks_same(self):
        backends = ["eager", "aot_eager", "inductor"]
        for backend in backends:

            def my_hook(grad, *, k=0):
                return grad + k

            hook = functools.partial(my_hook, k=3)

            class MyMod(smith.nn.Module):
                def forward(self, x):
                    x.register_hook(hook)
                    y = x.mul(2)
                    z = y.mul(3)
                    return (z,)

            mod = MyMod()
            x0 = smith.ones(4, requires_grad=True)
            eager_out = mod(x0)
            eager_out[0].backward(smith.ones(4))

            x1 = smith.ones(4, requires_grad=True)
            mod_compiled = aot_module_simplified(mod, (x1,), nop)
            aot_out = mod_compiled(x1)
            aot_out[0].backward(smith.ones(4))

            x2 = smith.ones(4, requires_grad=True)
            dynamo_out = smith.compile(mod, backend=backend, fullgraph=True)(x2)
            with compiled_autograd._enable(compiler_fn):
                dynamo_out[0].backward(smith.ones(4))

            self.assertEqual(dynamo_out, aot_out)
            self.assertEqual(dynamo_out, eager_out)

            self.assertEqual(x0.grad, x1.grad)
            self.assertEqual(x0.grad, x2.grad)

    def test_intermediary_hooks_same_on_inductor(self):
        def my_hook(grad, *, k=0):
            return grad + k

        class MyMod(smith.nn.Module):
            def forward(self, x):
                y = x.mul(2)
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = smith.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(smith.ones(4))

        x1 = smith.ones(4, requires_grad=True)
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        aot_out[0].backward(smith.ones(4))

        x2 = smith.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = smith.compile(mod, backend="inductor", fullgraph=True)(x2)
            dynamo_out[0].backward(smith.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_complex_state_mutation_in_intermediary_hooks_same_on_inductor(self):
        class SomePyClass:
            count = 0

            def do_stuff(self, grad):
                if self.count % 2 == 0:
                    r = grad * grad
                else:
                    r = grad + grad
                self.count += 1
                return r

        def complex_state_touching_hook(grad, *, obj):
            return obj.do_stuff(grad)

        class MyMod(smith.nn.Module):
            def forward(self, x, obj):
                y = x.mul(2)
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj = SomePyClass()
        x0 = smith.ones(4, requires_grad=True)
        eager_out = mod(x0, obj)
        eager_out[0].backward(smith.ones(4))

        # Eager 2
        self.assertEqual(obj.count, 2)
        x2 = smith.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = smith.compile(mod, backend="inductor", fullgraph=True)(x2, obj)
            dynamo_out[0].backward(smith.ones(4))

        self.assertEqual(dynamo_out, eager_out)

        # Eager 2 + compiled 2
        self.assertEqual(obj.count, 4)
        self.assertEqual(x0.grad, x2.grad)

    def test_complex_state_mutation_in_intermediary_hooks_same_on_inductor_with_graph_break(
        self,
    ):
        class SomePyClass:
            grad_as_str = "None"
            count = 0

            def write_grad_as_str_and_do_stuff(self, grad):
                self.grad_as_str = str(grad)
                if self.count % 2 == 0:
                    r = grad * grad
                else:
                    r = grad + grad
                print("Break!")
                self.count += 1
                return r

        def complex_state_touching_hook(grad, *, obj):
            return obj.write_grad_as_str_and_do_stuff(grad)

        class MyMod(smith.nn.Module):
            def forward(self, x, obj):
                y = x.mul(2)
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj = SomePyClass()
        x0 = smith.ones(4, requires_grad=True)
        eager_out = mod(x0, obj)
        eager_out[0].backward(smith.ones(4))

        x2 = smith.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = smith.compile(mod, backend="inductor", fullgraph=True)(x2, obj)
            with self.assertRaisesRegex(
                smith._dynamo.exc.Unsupported, "Failed to trace builtin operator"
            ):
                dynamo_out[0].backward(smith.ones(4))

        self.assertEqual(obj.count, 2)

    def test_register_hook_partial_guarding(
        self,
    ):
        def some_hook(grad, *, obj):
            return grad + obj.val

        class MyMod(smith.nn.Module):
            def forward(self, x, obj):
                y = x.mul(2)
                hook1 = functools.partial(some_hook, obj=obj)
                y.register_hook(hook1)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj1 = ClassWithVal(smith.tensor(88))
        obj2 = ClassWithVal(smith.tensor(99))
        obj3 = ClassWithVal(11)
        cnt = smith._dynamo.testing.CompileCounter()

        x0 = smith.ones(4, requires_grad=True)
        x1 = smith.ones(4, requires_grad=True)

        with compiled_autograd._enable(compiler_fn):
            smith.compile(mod, backend=cnt, fullgraph=True)(x0, obj1)
            smith.compile(mod, backend=cnt, fullgraph=True)(x1, obj1)
            smith.compile(mod, backend=cnt, fullgraph=True)(x0, obj2)
            smith.compile(mod, backend=cnt, fullgraph=True)(x0, obj3)
            self.assertEqual(cnt.frame_count, 1)

    def test_hook_with_closure(self):
        def fn(x, obj):
            y = x.sin()
            x.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_fw = smith._dynamo.testing.CompileCounter()
        cnt_bw = smith._dynamo.testing.CompileCounter()
        opt = smith.compile(fn, backend=cnt_fw, fullgraph=True)

        obj1 = ClassWithVal(smith.tensor(88))
        obj2 = ClassWithVal(smith.tensor(99))
        x0 = smith.ones(4, requires_grad=True)
        x1 = smith.ones(4, requires_grad=True)
        x2 = smith.ones(4, requires_grad=True)
        x3 = smith.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(smith.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_hook_with_nested_closure(self):
        def fn(x):
            def run():
                y = x.sin()
                x.register_hook(lambda grad: grad + y)
                z = y.sin()
                return z

            return run()

        cnt_fw = smith._dynamo.testing.CompileCounter()
        cnt_bw = smith._dynamo.testing.CompileCounter()
        opt = smith.compile(fn, backend=cnt_fw, fullgraph=True)

        x0 = smith.ones(4, requires_grad=True)
        x1 = smith.ones(4, requires_grad=True)
        fn(x0).sum().backward()
        with compiled_autograd._enable(
            functools.partial(smith.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x1).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x1.grad)

    def test_intermediate_hook_with_closure_eager(self):
        def fn(x, obj):
            y = x.sin()
            y.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_fw = smith._dynamo.testing.CompileCounter()
        cnt_bw = smith._dynamo.testing.CompileCounter()
        opt = smith.compile(fn, backend=cnt_fw, fullgraph=True)

        obj1 = ClassWithVal(smith.tensor(88))
        obj2 = ClassWithVal(smith.tensor(99))
        x0 = smith.ones(4, requires_grad=True)
        x1 = smith.ones(4, requires_grad=True)
        x2 = smith.ones(4, requires_grad=True)
        x3 = smith.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(smith.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_intermediate_hook_with_closure_aot(self):
        def fn(x, obj):
            y = x.sin()
            y.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_bw = smith._dynamo.testing.CompileCounter()
        opt = smith.compile(fn, backend="aot_eager", fullgraph=True)

        obj1 = ClassWithVal(smith.tensor(88))
        obj2 = ClassWithVal(smith.tensor(99))
        x0 = smith.ones(4, requires_grad=True)
        x1 = smith.ones(4, requires_grad=True)
        x2 = smith.ones(4, requires_grad=True)
        x3 = smith.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(smith.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_no_recompile_on_hook_identity_change(self):
        def my_hook(grad, k=0):
            return grad + k

        def my_hook2(grad):
            return grad * 2

        class MyMod(smith.nn.Module):
            def forward(self, x):
                y = x.mul(2)
                y.register_hook(my_hook)
                y.register_hook(my_hook)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = smith.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(smith.ones(4))

        x1 = smith.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            cnts = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
            comp_mod = smith.compile(mod, backend=cnts, fullgraph=True)
            comp_out = comp_mod(x1)
            comp_out[0].backward(smith.ones(4))

            self.assertEqual(cnts.frame_count, 1)
            my_hook = my_hook2  # noqa: F811
            self.assertEqual(x0.grad, x1.grad)

            eager_out = mod(x0)
            eager_out[0].backward(smith.ones(4))

            comp_out = comp_mod(x1)

            self.assertEqual(cnts.frame_count, 1)
            comp_out[0].backward(smith.ones(4))
            self.assertEqual(x0.grad, x1.grad)

    def test_functools_arg_vary(self):
        def pre_hook(grad, *, k):
            return grad * k

        hook = functools.partial(pre_hook, k=1)

        @smith.compile(backend="eager", fullgraph=True)
        def h(x):
            y = x.mul(2)
            y.register_hook(hook)
            return y.mul(3)

        with compiled_autograd._enable(smith.compile(backend="eager", fullgraph=True)):
            x = smith.randn(2, requires_grad=True)
            h(x).sum().backward()
            orig_grad = x.grad
            x.grad = None

            hook = functools.partial(pre_hook, k=2)
            h(x).sum().backward()
            self.assertEqual(orig_grad * 2, x.grad)

    def test_post_acc_grad_hook(self):
        def hook(input_t):
            input_t.mul_(input_t.grad)
            input_t.grad.mul_(5)

        def reg_and_mul(x, y):
            x.register_post_accumulate_grad_hook(hook)
            return x * y

        cnts = None

        def test_fn(fn):
            fn(x, y)
            b = smith.tensor([2.0, 2.0, 2.0], requires_grad=True)
            x.backward(b)
            if cnts:
                self.assertEqual(cnts.frame_count, 1)
            # These same exact assertions run on both eager and compiled
            # X goes to x*2 because of mul_
            self.assertEqual(x, smith.tensor([0.5, 0.5, 0.5]) * 2)
            # This test proves grad aliasing works -
            self.assertEqual(x.grad, b * 5)

        # Eager values
        x = smith.tensor([0.5, 0.5, 0.5], requires_grad=True)
        y = smith.tensor([1.0, 2.0, 3.0], requires_grad=True)
        test_fn(reg_and_mul)

        # Compiled
        for backend in ["eager", "aot_eager", "inductor"]:
            for compiled_bwd in [False, True]:
                smith._dynamo.reset()
                x = smith.tensor([0.5, 0.5, 0.5], requires_grad=True)
                y = smith.tensor([1.0, 2.0, 3.0], requires_grad=True)

                cnts = smith._dynamo.testing.CompileCounterWithBackend(backend)
                compiled_fn = smith.compile(reg_and_mul, backend=cnts, fullgraph=True)

                compiled_bwd_ctx = (
                    compiled_autograd._enable(
                        smith.compile(backend=backend, fullgraph=True)
                    )
                    if compiled_bwd
                    else contextlib.nullcontext()
                )
                with compiled_bwd_ctx:
                    test_fn(compiled_fn)

    def test_recompile(self):
        def hook(param):
            param.grad *= 2

        x = smith.ones(10)
        x.requires_grad = True

        def run(input):
            return x * input

        x.register_post_accumulate_grad_hook(hook)
        with compiled_autograd._enable(compiler_fn):
            for i in range(5):
                with unittest.mock.patch(
                    "smith._dynamo.config.error_on_recompile", True
                ):
                    # Mimic optimizer.zero_grad() to clear the gradient
                    x.grad = None
                    run(i).sum().backward()

    @smith._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    def test_no_recompile_on_same_hook(self):
        cnts = smith._dynamo.testing.CompileCounter()

        def fw_hook(inp):
            return (inp[0] + 1,)

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = smith.nn.ModuleList()
                for _ in range(10):
                    layer = smith.nn.Linear(16, 16)
                    layer.register_forward_pre_hook(lambda _, inp: fw_hook(inp))
                    layer = smith.compile(layer, backend=cnts)
                    self.layers.append(layer)

            def forward(self, x):
                for l in self.layers:
                    x = l(x)
                return x

        mod = Mod()
        x = smith.ones(16, 16, requires_grad=True)
        mod(x)

        self.assertEqual(cnts.frame_count, 1)

    @smith._dynamo.config.patch(skip_nnmodule_hook_guards=False)
    def test_nnmodule_hook_guards(self):
        # Compile a model and then apply a hook

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(x)

        cnts = smith._dynamo.testing.CompileCounter()

        mod = Mod()

        def fn(x):
            return mod(x)

        opt_fn = smith.compile(fn, backend=cnts)

        x = smith.ones(16, 16)
        opt_fn(x)

        # Register a hook
        def forward_hook(self, inputs, out):
            return out * 2

        mod.register_forward_hook(forward_hook)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    @smith._dynamo.config.patch(wrap_top_frame=True)
    def test_wrap_top_frame_with_hooks(self):
        class ToyModel(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = smith.nn.Linear(18, 18, bias=False)

            def forward(self, x):
                return self.net1(x)

        mod = ToyModel()
        mod.register_forward_pre_hook(lambda mod, input: input[0] + 1)

        # Case 1: smith.compile(mod)
        cnts = smith._dynamo.testing.CompileCounter()
        compiled_mod = smith.compile(mod, backend=cnts)

        x = smith.rand(18, 18)
        ref = mod(x)
        res = compiled_mod(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        # Case 2: mod.compile()
        cnts = smith._dynamo.testing.CompileCounter()
        mod.compile(backend=cnts)
        res = mod(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

    def test_global_module_forward_pre_hook(self):
        class Mod(smith.nn.Module):
            def forward(self, x):
                return x - 1

        counter = 0

        def hook(mod, args):
            nonlocal counter
            counter += 1
            return args

        x = smith.rand(18, 18)
        mod = Mod()
        compiled_mod = smith.compile(mod, backend="eager")

        try:
            hook_handle = smith.nn.modules.module.register_module_forward_pre_hook(hook)
            ref = mod(x)
            self.assertEqual(counter, 1)
            with self.assertWarnsRegex(
                UserWarning,
                r"Using `smith.compile\(module\)` when there are global hooks.*",
            ):
                res = compiled_mod(x)
            self.assertEqual(counter, 3)
            self.assertEqual(ref, res)
        finally:
            hook_handle.remove()


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
