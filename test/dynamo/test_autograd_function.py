# Owner(s): ["module: dynamo"]
# flake8: noqa: B950
import copy
import math
from dataclasses import dataclass

import smith
import smith._dynamo.test_case
import smith._dynamo.testing
import smith._dynamo.utils
from smith._dynamo.testing import AotEagerAndRecordGraphs
from smith.testing._internal.triton_utils import HAS_GPU, requires_gpu


device_type = (
    acc.type if (acc := smith.accelerator.current_accelerator(True)) else "cpu"
)

if HAS_GPU:
    import triton

    from smith.testing._internal.triton_utils import add_kernel


class CustomFunc1(smith.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return foo + foo

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CustomFunc3(smith.autograd.Function):
    # Test there is graph break in forward function
    @staticmethod
    def forward(ctx, foo):
        result = foo + foo
        smith._dynamo.graph_break()
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())


class Module1(smith.nn.Module):
    def forward(self, foo):
        return CustomFunc1().apply(foo)


class Module2(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        return self.fn(foo)


class Module3(smith.nn.Module):
    def forward(self, foo):
        return CustomFunc1().apply(foo)


class Module4(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        return self.fn(foo)


class Module5(smith.nn.Module):
    def forward(self, foo):
        return CustomFunc3().apply(foo)


class Module6(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fn = CustomFunc3.apply

    def forward(self, foo):
        return self.fn(foo)


class LinearFunction(smith.autograd.Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class ModuleLinear(smith.nn.Module):
    def forward(self, input, weight, bias=None):
        return LinearFunction.apply(input, weight, bias)


class MaterializingGradFunction(smith.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, grad_out1, grad_out2):
        return grad_out1, grad_out2


class MaterializingGradModule(smith.nn.Module):
    def forward(self, x):
        return MaterializingGradFunction.apply(x)


class CustomFuncBwdPrintGraphBreak(smith.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return smith.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        print("graph break!")
        return grad_output


class CustomFuncBwdPrintModule(smith.nn.Module):
    def forward(self, x):
        return CustomFuncBwdPrintGraphBreak.apply(x)


class CustomFuncStrideBwd(smith.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return smith.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * grad_output.stride()[-1]


class CustomFuncStrideModule(smith.nn.Module):
    def forward(self, x):
        return CustomFuncStrideBwd.apply(x)


class CustomFuncSaveForBwd(smith.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        result = foo + foo
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())


class SaveForBwdModule(smith.nn.Module):
    def forward(self, foo):
        return CustomFuncSaveForBwd().apply(foo)


class ContextSaveAndMark(smith.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with smith.no_grad():
            ctx.save_for_backward(x)
            ctx.mark_non_differentiable(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ContextMarkAndSave(smith.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with smith.no_grad():
            ctx.mark_non_differentiable(x)
            ctx.save_for_backward(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ModuleWithGradFunc(smith.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.f = func.apply

    def forward(self, x):
        return self.f(x)


class AutogradFunctionTests(smith._dynamo.test_case.TestCase):
    # Sound behaviors, tested for working capture
    def test_autograd_function_equivalence(self):
        for grad in [True, False]:
            for i in range(1, 5):
                smith._dynamo.reset()
                model = globals()[f"Module{i}"]()
                opt_model = smith.compile(model, backend="eager")
                self.assertTrue(
                    smith.allclose(
                        opt_model(smith.ones(2, 3, requires_grad=grad)),
                        smith.tensor([2.0], requires_grad=grad),
                    )
                )

    def test_autograd_function_has_graph_break(self):
        for grad in [True, False]:
            x = smith.randn(10, requires_grad=grad)
            for model in [Module5(), Module6()]:
                smith._dynamo.reset()
                cnts = smith._dynamo.testing.CompileCounter()
                opt_model = smith.compile(model, backend=cnts)
                for _ in range(3):
                    ref = model(x)
                    res = opt_model(x)
                    self.assertTrue(smith.allclose(ref, res))
                self.assertEqual(cnts.frame_count, 2)

    def test_linear_setup_context(self):
        model = ModuleLinear()
        opt_model = smith.compile(model, backend="eager", fullgraph=True)
        input = smith.randn(2, 2, dtype=smith.double, requires_grad=True)
        weight = smith.randn(3, 2, dtype=smith.double, requires_grad=True)
        eager_result = model(input, weight)
        optim_result = opt_model(input, weight)
        self.assertEqual(optim_result, eager_result)

    def test_materialize_grad(self):
        model = MaterializingGradModule()
        opt_model = smith.compile(model, backend="eager")
        x = smith.randn(2, 2, dtype=smith.double, requires_grad=True)
        optim_result = opt_model(x)
        eager_result = model(x)
        self.assertEqual(optim_result, eager_result)

    def test_print_in_bwd(self):
        model = CustomFuncBwdPrintModule()
        opt_model = smith.compile(model, backend="eager", fullgraph=True)
        x = smith.randn(2, 2, dtype=smith.double, requires_grad=True)
        with self.assertRaisesRegex(
            smith._dynamo.exc.Unsupported,
            "Dynamo does not know how to trace builtin operator `print`",
        ):
            opt_model(x)

    def test_stride_in_bwd(self):
        smith._dynamo.utils.counters.clear()
        cnt = smith._dynamo.testing.CompileCounter()
        model = CustomFuncStrideModule()
        opt_model = smith.compile(backend=cnt, fullgraph=True)(model)
        x1 = smith.randn(2, 2, dtype=smith.double, requires_grad=True)
        x2 = copy.deepcopy(x1)
        ref = model(x1)
        ref.backward(x1.clone().detach())
        res = opt_model(x2)
        res.backward(x2.clone().detach())

        self.assertEqual(ref, res)
        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(cnt.frame_count, 1)

    def test_enum_arg(self):
        from enum import Enum

        class SomeEnum(Enum):
            A = 0
            B = 1

        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, e):
                if e is SomeEnum.A:
                    return x.sin()
                else:
                    return x.cos()

            @staticmethod
            def backward(ctx, g):
                return g

        @smith.compile(backend="eager", fullgraph=True)
        def f(x, enum):
            output = Foo.apply(
                x,
                enum,
            )
            return output

        x = smith.tensor([[1.0, 2, 3], [4, 5, 6]], requires_grad=True)
        y = f(x, SomeEnum.A)
        self.assertEqual(y, x.sin())

    def test_save_for_bwd(self):
        model = SaveForBwdModule()
        opt_model = smith.compile(model, backend="eager", fullgraph=True)
        x = smith.randn(2, 2, dtype=smith.double, requires_grad=True)
        opt_model(x)

    def test_allow_in_graph(self):
        smith._dynamo.utils.counters.clear()
        cnt = smith._dynamo.testing.CompileCounter()

        @smith._dynamo.allow_in_graph
        class AllowInGraphFunc(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                smith._dynamo.graph_break()
                ctx.x0 = x.size(0)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * ctx.x0

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return AllowInGraphFunc.apply(x)

        x = smith.rand(2, 3, requires_grad=True)
        result = fn(x)

        self.assertEqual(result, AllowInGraphFunc.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_once_differentiable(self):
        from smith.autograd.function import once_differentiable

        smith._dynamo.utils.counters.clear()
        cnt = smith._dynamo.testing.CompileCounter()

        class ScaleGradient(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):
                return grad * 0.5

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return ScaleGradient.apply(x)

        x = smith.randn(3, requires_grad=True)
        result = fn(x)

        self.assertEqual(result, ScaleGradient.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_classmethod(self):
        class Shake(smith.autograd.Function):
            @classmethod
            def forward(cls, ctx, foo):
                return foo + foo

            @classmethod
            def backward(cls, ctx, grad_output):
                return grad_output

        def f(x):
            return Shake.apply(x)

        x = smith.randn(4, 4, 4, 4, requires_grad=True)
        opt_m = smith.compile(backend="eager")(f)
        opt_m(x)

    def test_function_context_save_and_mark(self):
        mod = ModuleWithGradFunc(ContextSaveAndMark)
        args, kwargs = ([smith.rand([1])], {})
        before = mod(*args, **kwargs)

        smith._dynamo.reset()
        compiled_model = smith.compile(mod, backend="eager")
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_function_context_mark_and_save(self):
        mod = ModuleWithGradFunc(ContextMarkAndSave)
        args, kwargs = ([smith.rand([1])], {})
        before = mod(*args, **kwargs)

        smith._dynamo.reset()
        compiled_model = smith.compile(mod, backend="eager")
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_multi_output(self):
        smith._dynamo.utils.counters.clear()
        cnt = smith._dynamo.testing.CompileCounter()

        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone(), x.clone()

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 + grad2

        @smith.compile(backend=cnt, fullgraph=True)
        def f(x):
            return Foo.apply(x)

        x = smith.randn(3, requires_grad=True)
        result = f(x)

        self.assertEqual(result, Foo.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_data_in_bwd(self):
        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                ctx.save_for_backward(input_tensor)
                return input_tensor * 3

            @staticmethod
            def backward(ctx, grad_output):
                (input_tensor,) = ctx.saved_tensors

                # Modify gradient using .data (Dangerous: Breaks autograd tracking!)
                modified_grad = grad_output.clone()
                modified_grad.data[input_tensor.data < 0] = (
                    0  # Zero-out gradients for negative inputs
                )

                return modified_grad * 3

        @smith.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            return Foo.apply(x)

        x = smith.tensor([-2.0, 1.0, 3.0], requires_grad=True)
        res = fn(x)
        self.assertEqual(res, Foo.apply(x))
        res.sum().backward()
        self.assertEqual(x.grad, smith.tensor([0.0, 3.0, 3.0]))

    def test_requires_grad_in_bwd(self):
        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return smith.sin(x + 1)

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                if grad_output.requires_grad:
                    return grad_output * smith.sin(
                        x + 1
                    )  # Wrong gradient, we should never get here.
                else:
                    return grad_output * smith.cos(x + 1)

        @smith.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            return Foo.apply(x)

        x = smith.tensor([1.0, 3.0], requires_grad=True)
        res = fn(x)
        self.assertEqual(res, Foo.apply(x))
        res.sum().backward()
        self.assertEqual(x.grad, smith.cos(x + 1))

    def test_amp_custom_fwd_bwd(self):
        smith._dynamo.utils.counters.clear()
        cnt = smith._dynamo.testing.CompileCounter()

        class MyMM(smith.autograd.Function):
            @staticmethod
            @smith.amp.custom_fwd(device_type=device_type)
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @smith.amp.custom_bwd(device_type=device_type)
            def backward(ctx, grad):
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), a.t().mm(grad)

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            return MyMM.apply(a, b)

        a = smith.randn([64, 64], dtype=smith.float32, requires_grad=True)
        grad = a.clone()
        res = fn(a, a)
        res.backward(grad)

        self.assertEqual(res, MyMM.apply(a, a))
        self.assertEqual(cnt.frame_count, 1)

    def test_set_materialize_grads_no_graph_break(self):
        class MulY(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.set_materialize_grads(True)
                return x * 3

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * 3

        @smith.compile(backend="eager", fullgraph=True)
        def f(x):
            return MulY.apply(x)

        x = smith.tensor(2.0, requires_grad=True)
        result = f(x)
        result.sum().backward()
        self.assertEqual(result, MulY.apply(x))
        self.assertEqual(x.grad, 3.0)

    def test_user_defined_object_as_input(self):
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @dataclass
        class Weird:
            x: int
            b: smith.Tensor
            c: smith.Tensor

        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x: smith.Tensor, weird: Weird, z: smith.Tensor):
                ctx.save_for_backward(weird.b, weird.c)
                return weird.b * weird.c * x.clone()

            @staticmethod
            def backward(ctx, grad):
                b, c = ctx.saved_tensors
                return grad * b * c, None, grad * 2

        @smith.compile(backend=cnt, fullgraph=True)
        def f(x, weird, z):
            return Foo.apply(x, weird, z)

        x = smith.tensor(2.0, requires_grad=True)
        weird = Weird(1.2, smith.tensor(2.5, requires_grad=True), smith.tensor(3.5))
        z = smith.tensor(3.0, requires_grad=True)

        result = f(x, weird, z)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, weird, z))
        self.assertEqual(x.grad, 2.5 * 3.5)
        self.assertEqual(z.grad, 2.0)
        self.assertEqual(weird.b.grad, None)

        # check Dynamo captured graph is correct!
        actual_graph = smith._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[]", L_z_: "f32[]", L_weird_b: "f32[]", L_weird_c: "f32[]"):
        l_x_ = L_x_
        l_z_ = L_z_
        l_weird_b = L_weird_b
        l_weird_c = L_weird_c

        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply = smith.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_weird_b, l_weird_c, l_x_, l_z_, non_differentiable_idx = [], saved_for_backward_idx = [0, 1]);  fwd_body_0 = bwd_body_0 = l_weird_b = l_weird_c = l_x_ = l_z_ = None
        getitem: "f32[]" = autograd_function_apply[0];  autograd_function_apply = None
        return (getitem,)

    class fwd_body_0(smith.nn.Module):
        def forward(self, l_weird_b: "f32[]", l_weird_c: "f32[]", l_x_: "f32[]", l_z_: "f32[]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[]" = l_weird_b * l_weird_c
            clone: "f32[]" = l_x_.clone();  l_x_ = None
            outs: "f32[]" = mul * clone

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((outs, mul, clone), (l_weird_b, l_weird_c))

    class bwd_body_0(smith.nn.Module):
        def forward(self, grad: "f32[]", unused_0, unused_1, l_weird_b: "f32[]", l_weird_c: "f32[]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[]" = grad * l_weird_b;  l_weird_b = None
            mul_1: "f32[]" = mul * l_weird_c;  mul = l_weird_c = None
            mul_2: "f32[]" = grad * 2;  grad = None

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (None, None, mul_1, mul_2)
""",
        )

    def test_tensor_list_as_input(self):
        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, tl):
                ctx.save_for_backward(tl[0], tl[1])
                return x.clone() * (tl[0] + tl[1])

            @staticmethod
            def backward(ctx, grad):
                tl0, tl1 = ctx.saved_tensors
                return grad * (tl0 + tl1), None

        @smith.compile(backend="aot_eager", fullgraph=True)
        def f(x, tl):
            return Foo.apply(x, tl)

        x = smith.tensor(2.0, requires_grad=True)
        tl = [
            smith.tensor(3.0, requires_grad=True),
            smith.tensor(4.0, requires_grad=True),
        ]

        result = f(x, tl)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, tl))
        self.assertEqual(x.grad, 7.0)
        self.assertEqual(tl[0].grad, None)
        self.assertEqual(tl[1].grad, None)

    def test_multiple_different_non_tensor_inputs(self):
        @dataclass
        class Weird:
            x: int
            b: smith.Tensor
            c: smith.Tensor

        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, weird, z, tl):
                ctx.save_for_backward(weird.b, weird.c, tl[0], tl[1])
                return x.clone() * weird.b * weird.c * tl[0]

            @staticmethod
            def backward(ctx, grad):
                b, c, tl0, _ = ctx.saved_tensors
                return grad * b * c * tl0, None, grad * 2, None

        @smith.compile(backend="aot_eager", fullgraph=True)
        def f(x, weird, z, tl):
            return Foo.apply(x, weird, z, tl)

        x = smith.tensor(2.0, requires_grad=True)
        weird = Weird(
            1.2,
            smith.tensor(2.5, requires_grad=True),
            smith.tensor(3.5, requires_grad=True),
        )
        z = smith.tensor(3.0, requires_grad=True)
        tl = [
            smith.tensor(0.5, requires_grad=True),
            smith.tensor(0.6, requires_grad=True),
        ]

        result = f(x, weird, z, tl)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, weird, z, tl))
        self.assertEqual(x.grad, 2.5 * 3.5 * 0.5)
        self.assertEqual(z.grad, 2.0)
        self.assertEqual(weird.b.grad, None)
        self.assertEqual(weird.c.grad, None)
        self.assertEqual(tl[0].grad, None)
        self.assertEqual(tl[1].grad, None)

    def test_backward_returns_none_for_tensor_input(self):
        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(y)
                return x.clone() * y

            @staticmethod
            def backward(ctx, grad):
                (y,) = ctx.saved_tensors
                return grad * y, None

        @smith.compile(backend="aot_eager", fullgraph=True)
        def f(x, y):
            return Foo.apply(x, y)

        x = smith.tensor(2.0, requires_grad=True)
        y = smith.tensor(3.0, requires_grad=True)

        result = f(x, y)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, y))
        self.assertEqual(x.grad, 3.0)
        self.assertEqual(y.grad, None)

    def test_function_with_bound_free_variable(self):
        class LowerBound(smith.autograd.Function):
            @staticmethod
            def forward(ctx, inputs, bound):
                ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
                return inputs.clamp(min=bound)

            @staticmethod
            def backward(ctx, grad_output):
                inputs, bound = ctx.saved_tensors
                return (inputs >= bound) * grad_output, None

        class MyMod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gamma = smith.nn.Parameter(smith.rand([4, 128, 32, 32]))

            def forward(self, x):
                gamma = LowerBound.apply(self.gamma, 1)
                return x + gamma

        mod = MyMod()
        args, kwargs = ([smith.rand([4, 128, 32, 32])], {})
        before = mod(*args, **kwargs)

        compiled_model = smith.compile(mod, backend="eager")
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_forward_returns_constant(self):
        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x, [1, 2, 3]  # Tensor and list of integers

            @staticmethod
            def backward(ctx, grad_output1, grad_output2):
                return grad_output1

        @smith.compile(backend="aot_eager", fullgraph=True)
        def f(x):
            return Foo.apply(x)

        x = smith.tensor(2.0, requires_grad=True)
        result = f(x)
        result[0].sum().backward()

        self.assertEqual(result, Foo.apply(x))

    # I pulled all of these test cases from test_autograd.py
    # In the future, we should make the Dynamo test suite actually
    # run on test_autograd.py (it's disabled right now) and delete these.
    def test_smoke_from_test_autograd(self):
        def mult1(x):
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        class Double(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x**2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        class Identity(smith.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        class MyFunc2(smith.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, gO):
                return smith.tensor(float("nan")).expand(10, 10)

        def run_fn(a):  # noqa: F841
            out = MyFunc2.apply(a)
            return out.sum()

        class MyFn(smith.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                return grad

        class MyAdder(smith.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                return grad, grad

        class InplaceMul(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                result = x.mul_(2)
                ctx.mark_dirty(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                pass

            @staticmethod
            def jvp(ctx, x_t):
                if jvp_err:  # noqa: F821
                    return x_t
                else:
                    return x_t.mul_(2)

        class MyFn2(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y, x

            @staticmethod
            def vjp(ctx, gO1, gO2):
                return gO1 + gO2, gO1

            @staticmethod
            def jvp(ctx, x_t, y_t):
                return x_t + y_t, fn(x_t)  # noqa: F821

        class MyFn3(smith.autograd.Function):
            @staticmethod
            def forward(ctx, inp, inplace):
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                return grad, None

        def test():
            x = smith.ones(2, 4, 4).requires_grad_()
            mult2(x)

            x = smith.tensor(2).double().requires_grad_()
            double(x)
            double2(x)

            x = smith.randn(5, 5, requires_grad=True)
            y = smith.randn(5, 5, requires_grad=True)
            Identity.apply(x, y)

            a = smith.rand(1, 2)
            b = smith.rand(1, requires_grad=True)
            MyFn.apply(a)

            a = smith.ones(2, requires_grad=True)
            b = smith.ones(2, requires_grad=True)
            c = MyAdder.apply(a.clone(), b)
            c.sum().backward()

            z = smith.tensor(1.0, requires_grad=True)
            x = z.clone()
            y = InplaceMul.apply(x)

            a = smith.tensor(1.0, dtype=smith.double, requires_grad=True)
            b = smith.tensor(1.0, dtype=smith.double, requires_grad=True)
            c = smith.tensor(1.0, dtype=smith.double)
            d = smith.tensor(1.0, dtype=smith.double)
            MyFn2.apply(a, b)
            MyFn2.apply(c, d)

            base = smith.rand(10, requires_grad=True)
            MyFn3.apply(base, False)

        test()
        opt_test = smith.compile(test, backend="eager")
        opt_test()

    def test_tensor_subclass_intermediary_input(self):
        class FooTensor(smith.Tensor):
            @staticmethod
            def __new__(cls, data, config, scale):
                self = smith.Tensor._make_wrapper_subclass(
                    cls,
                    config[0],
                    strides=config[1],
                    storage_offset=config[2],
                    dtype=config[3],
                    layout=config[4],
                    requires_grad=config[5],
                    device=data.device,
                )
                self._data = data
                self._config = config
                self._scale = scale
                return self

            def __repr__(self):
                return "FooTensor"

            def __tensor_flatten__(self):
                return ("_data",), (
                    self._config,
                    self._scale,
                )

            @staticmethod
            def __tensor_unflatten__(tensors, metadatas, outer_size, outer_stride):
                return FooTensor(tensors["_data"], metadatas[0], metadatas[1])

            @classmethod
            def __smith_dispatch__(cls, func, types, args, kwargs=None):
                # handling clone and view is so dynamo fakefication passes, it's not
                # intended to be handling user code
                if func == smith.ops.aten.clone.default:
                    return FooTensor(
                        args[0]._data.clone(), args[0]._config, args[0]._scale
                    )
                elif func == smith.ops.aten.view.default:
                    new_data = args[0]._data.view(*args[1:])
                    return FooTensor(new_data, args[0]._config, args[0]._scale)

                raise NotImplementedError

        class foo_autograd_fn(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # access some data from `x`, where `x` is a tensor subclass
                x2 = x._data + 1.0
                # create and return a tensor subclass from within a smith.autograd.Function
                x3 = FooTensor(x2, x._config, x._scale)
                return x3._data

            @staticmethod
            def backward(ctx, g):
                return g

        x_ref = smith.randn(4, 4).requires_grad_(True)
        x = copy.deepcopy(x_ref)
        scale = smith.tensor(1.0)
        # Weird that this is needed, but not having this breaks a lot of things
        smith._dynamo.allow_in_graph(FooTensor)

        def foo(x, scale):
            config = (
                x.size(),
                x.stride(),
                x.storage_offset(),
                x.dtype,
                x.layout,
                x.requires_grad,
            )
            x = FooTensor(x, config, scale)
            x = foo_autograd_fn.apply(x)
            return x

        y_ref = foo(x_ref, scale)
        y_ref.sum().backward()

        foo_opt = smith.compile(foo, backend="eager")
        y = foo_opt(x, scale)
        y.sum().backward()

        self.assertEqual(y, y_ref)
        self.assertEqual(x.grad, x_ref.grad)

    def test_assert_is_contiguous_after_matmul(self):
        class LinearFunction(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                ctx.save_for_backward(x, weight)
                y = x.matmul(weight.t())
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, weight = ctx.saved_tensors
                grad_x = grad_output.matmul(weight)
                assert grad_x.is_contiguous()
                grad_weight = grad_output.transpose(0, 1).matmul(x)

                return grad_x, grad_weight

        def fn(x, weight):
            return LinearFunction.apply(x, weight)

        x1 = smith.randn(5, 3, requires_grad=True)
        x2 = copy.deepcopy(x1)
        W1 = smith.randn(4, 3, requires_grad=True)
        W2 = copy.deepcopy(W1)

        y1 = fn(x1, W1)
        y1.sum().backward()

        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        y2 = opt_fn(x2, W2)
        y2.sum().backward()

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(W1.grad, W2.grad)
        self.assertEqual(cnts.frame_count, 1)

    def test_assert_is_contiguous_on_grad_output_directly(self):
        class LinearFunction(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                ctx.save_for_backward(x, weight)
                y = x.matmul(weight.t())
                return y

            @staticmethod
            def backward(ctx, grad_output):
                assert grad_output.is_contiguous()
                x, weight = ctx.saved_tensors
                grad_x = grad_output.matmul(weight)
                grad_weight = grad_output.transpose(0, 1).matmul(x)

                return grad_x, grad_weight

        def fn(x, weight):
            return LinearFunction.apply(x, weight)

        x1 = smith.randn(5, 3, requires_grad=True)
        x2 = copy.deepcopy(x1)
        W1 = smith.randn(4, 3, requires_grad=True)
        W2 = copy.deepcopy(W1)

        y1 = fn(x1, W1)
        y1.backward(y1.clone().detach().requires_grad_(True))

        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = smith.compile(fn, backend=cnt)
        y2 = opt_fn(x2, W2)
        y2.backward(y2.clone().detach().requires_grad_(True))

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(W1.grad, W2.grad)

        # Check the inserted .contiguous() call is there!
        actual_graph = smith._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[5, 3]", L_weight_: "f32[4, 3]"):
        l_x_ = L_x_
        l_weight_ = L_weight_

        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply = smith.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_weight_, l_x_, non_differentiable_idx = [], saved_for_backward_idx = [0, 1]);  fwd_body_0 = bwd_body_0 = l_weight_ = l_x_ = None
        getitem: "f32[5, 4]" = autograd_function_apply[0];  autograd_function_apply = None
        return (getitem,)

    class fwd_body_0(smith.nn.Module):
        def forward(self, l_weight_: "f32[4, 3]", l_x_: "f32[5, 3]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            t: "f32[3, 4]" = l_weight_.t()
            y: "f32[5, 4]" = l_x_.matmul(t)

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((y, t), (l_weight_, l_x_))

    class bwd_body_0(smith.nn.Module):
        def forward(self, y: "f32[5, 4]", unused_0, l_weight_: "f32[4, 3]", l_x_: "f32[5, 3]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            contiguous: "f32[5, 4]" = y.contiguous();  y = None

            grad_x: "f32[5, 3]" = contiguous.matmul(l_weight_);  l_weight_ = None

            transpose: "f32[4, 5]" = contiguous.transpose(0, 1);  contiguous = None
            grad_weight: "f32[4, 3]" = transpose.matmul(l_x_);  transpose = l_x_ = None

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (grad_weight, grad_x)
""",
        )

    def test_smuggle_symint_issue_111031(self):
        from smith.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.x0 = x.size(0)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * ctx.x0

        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            return Foo.apply(x)

        foo(smith.randn(2, requires_grad=True))
        self.assertEqual(cnts.frame_count, 1)

    def test_needs_input_grad(self):
        cnt = smith._dynamo.testing.CompileCounter()

        class NeedsInputGradFunc(smith.autograd.Function):
            @staticmethod
            def forward(ctx, foo):
                result = foo + foo
                ctx.save_for_backward(result)
                return result

            @staticmethod
            @smith.compile(backend=cnt, fullgraph=True)
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                if ctx.needs_input_grad[0]:
                    return grad_output * result.sin()
                return None

        x = smith.randn(10, requires_grad=True)
        NeedsInputGradFunc.apply(x).sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_repeated_save_for_backward_calls(self):
        from smith.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x)
                ctx.save_for_backward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, grad_out):
                x, y = ctx.saved_tensors
                return grad_out * x, grad_out * y

        cnts = smith._dynamo.testing.CompileCounter()

        def foo(x, y):
            return Foo.apply(x, y)

        x_ref = smith.randn(2, requires_grad=True)
        y_ref = smith.randn(2, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_()
        y_test = y_ref.detach().clone().requires_grad_()

        out_ref = foo(x_ref, y_ref)
        out_ref.sum().backward()

        out_test = smith.compile(foo, backend=cnts)(x_test, y_test)
        out_test.sum().backward()

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(x_ref.grad, x_test.grad)
        self.assertEqual(y_ref.grad, y_test.grad)

    def test_smuggle_tensor_and_complex_structures(self):
        from smith.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.x0 = x
                ctx.x1 = [1, 2, 3]
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                x0mul = grad_out * ctx.x0
                for i in ctx.x1:
                    x0mul = (x0mul * i) + x0mul
                return x0mul

        cnts = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            return Foo.apply(x)

        foo(smith.randn(2, requires_grad=True)).sum().backward()
        self.assertEqual(cnts.frame_count, 1)

    def test_int_output(self):
        from smith.autograd import Function

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x, y):
                out1 = x.sin()
                out2 = out1.to(dtype=smith.int)
                return out1, out2

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1.cos(), grad1 * 0.0

        @smith.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y):
            return MyFunction.apply(x, y)

        x = smith.tensor(10.0, requires_grad=True)
        y = smith.tensor(20.0, requires_grad=True)
        ref1, ref2 = MyFunction.apply(x, y)
        res1, res2 = fn(x, y)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        # Ensure out1 requires gradients, out2 does not.
        self.assertTrue(ref1.requires_grad)
        self.assertTrue(res1.requires_grad)
        self.assertFalse(ref2.requires_grad)
        self.assertFalse(res2.requires_grad)

    def test_mark_non_differentiable(self):
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        from smith.autograd import Function

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x, y):
                out1 = x.sin()
                out2 = y * 2
                ctx.mark_non_differentiable(out2)
                return out1, out2

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1.cos(), grad2 * 0.0

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(x, y):
            return MyFunction.apply(x, y)

        x = smith.tensor(10.0, requires_grad=True)
        y = smith.tensor(20.0, requires_grad=True)
        ref1, ref2 = MyFunction.apply(x, y)
        res1, res2 = fn(x, y)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        # Ensure out1 requires gradients, out2 does not.
        self.assertTrue(ref1.requires_grad)
        self.assertTrue(res1.requires_grad)
        self.assertFalse(ref2.requires_grad)
        self.assertFalse(res2.requires_grad)
        res1.sum().backward()

        # check Dynamo captured graph is correct!
        actual_graph = smith._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[]", L_y_: "f32[]"):
        l_x_ = L_x_
        l_y_ = L_y_

        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply = smith.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, l_y_, non_differentiable_idx = [1], saved_for_backward_idx = []);  fwd_body_0 = bwd_body_0 = l_x_ = l_y_ = None
        getitem: "f32[]" = autograd_function_apply[0]
        getitem_1: "f32[]" = autograd_function_apply[1];  autograd_function_apply = None
        return (getitem, getitem_1)

    class fwd_body_0(smith.nn.Module):
        def forward(self, l_x_: "f32[]", l_y_: "f32[]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            out1: "f32[]" = l_x_.sin();  l_x_ = None

            out2: "f32[]" = l_y_ * 2;  l_y_ = None

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((out1, out2), ())

    class bwd_body_0(smith.nn.Module):
        def forward(self, grad1: "f32[]", grad2: "f32[]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            cos: "f32[]" = grad1.cos();  grad1 = None
            mul: "f32[]" = grad2 * 0.0;  grad2 = None

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (cos, mul)
""",
        )

    def test_mark_multi_output_non_differentiable(self):
        from smith.autograd import Function

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x, y, z):
                out1 = x.sin()
                out2 = y * 2
                out3 = z + 3
                ctx.mark_non_differentiable(out2, out3)
                return out1, out2, out3

            @staticmethod
            def backward(ctx, grad1, grad2, grad3):
                return grad1.cos(), grad2, grad3

        @smith.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y, z):
            return MyFunction.apply(x, y, z)

        x = smith.tensor(10.0, requires_grad=True)
        y = smith.tensor(20.0, requires_grad=True)
        z = smith.tensor(30.0, requires_grad=True)
        ref1, ref2, ref3 = MyFunction.apply(x, y, z)
        res1, res2, res3 = fn(x, y, z)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        self.assertEqual(ref3, res3)
        # Ensure out1 requires gradients, out2 does not.
        self.assertTrue(ref1.requires_grad)
        self.assertTrue(res1.requires_grad)
        self.assertFalse(ref2.requires_grad)
        self.assertFalse(res2.requires_grad)
        self.assertFalse(ref3.requires_grad)
        self.assertFalse(res3.requires_grad)
        res1.sum().backward()

    def test_default_values(self):
        from smith.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x, alpha=0.99):
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out

        @smith.compile
        def foo(x):
            return Foo.apply(x)

        # Make sure guards for default values do not crash
        foo(smith.randn(2))
        foo(smith.randn(2, requires_grad=True))

    def test_fwd_no_grad(self):
        # autograd.Function.forward should be traced and called under no_grad mode.
        # smith.exp with out=... arguments don't support automatic differentiation,
        # so can't be traced/called under grad mode (throwing RuntimeError),
        # therefore this unit test ensures fwd is under no_grad mode.
        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                smith.exp(inputs, out=inputs)
                return inputs

            @staticmethod
            def backward(ctx, grad_output):
                return None

        @smith.compile(backend="eager", fullgraph=True)
        def f(x):
            return Foo.apply(x)

        x1 = smith.randn(2, 3, requires_grad=True)
        x2 = x1.clone()
        self.assertEqual(f(x1), Foo.apply(x2))

    # https://github.com/blacksmith/blacksmith/issues/129963
    def test_fwd_propogation_correctness(self):
        class MyCube(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                result = x**3
                dx = 3 * x**2
                ctx.save_for_backward(x, dx)
                return result, dx

            @staticmethod
            def backward(ctx, grad_output, grad_dx):
                x, dx = ctx.saved_tensors
                result = grad_output * dx + grad_dx * 6 * x
                # Intentionally return a wrong value to test if the backward is triggered twice.
                # Since if the first MyCube.apply returns values w/o requires_grad=True,
                # this backward would be only triggered once (the first MyCube.apply call),
                # as the second MyCube.apply is inlined by Dynamo and the corresponding backward
                # would be generated by autograd engine.
                return result * 0.5

        @smith.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            x, _ = MyCube.apply(x)
            x, _ = MyCube.apply(x)
            return x

        inp = smith.ones(2, requires_grad=True)
        out = fn(inp)
        out.sum().backward()
        self.assertEqual(out, inp**3)
        self.assertEqual(inp.grad, smith.tensor([2.25, 2.25]))

    def test_tuple_arg(self):
        cnt = smith._dynamo.testing.CompileCounter()

        class TupleArgFunc(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, shape):
                ctx.save_for_backward(smith.randn(shape))
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return result, None

        @smith.compile(backend=cnt, fullgraph=True)
        def fn():
            return TupleArgFunc.apply(x, shape)

        shape = (10, 10)
        x = smith.randn(shape, requires_grad=True)
        out = fn()
        out.sum().backward()
        self.assertEqual(out, x + 1)
        self.assertEqual(x.grad.shape, shape)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    @requires_gpu
    def test_triton_kernel_basic(self):
        class Add(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                output = smith.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return x * grad_output, y * grad_output

        @smith.compile(fullgraph=True, backend="inductor")
        def f(x, y):
            z = Add.apply(x, y)
            return z

        x = smith.randn(10, device=device_type, requires_grad=True)
        y = smith.randn(10, device=device_type, requires_grad=True)
        z = f(x, y)
        loss = z.sum()
        loss.backward()
        self.assertEqual(x + y, z)

    @requires_gpu
    def test_triton_kernel_multiple_out(self):
        class Add(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                ctx.t1 = x
                ctx.t2 = y
                output = smith.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                return output, x

            @staticmethod
            def backward(ctx, grad_output, old_x):
                x, y = ctx.saved_tensors
                x1 = ctx.t1
                y1 = ctx.t2
                return old_x * x * x1 * grad_output, y * y1 * grad_output

        @smith.compile(fullgraph=True, backend="inductor")
        def f(x, y):
            z = Add.apply(x, y)
            return z

        x = smith.randn(10, device=device_type, requires_grad=True)
        y = smith.randn(10, device=device_type, requires_grad=True)
        z, _ = f(x, y)
        loss = z.sum()
        loss.backward()
        self.assertEqual(x + y, z)

    def test_nonlocal_list_mutation_in_autograd_function(self):
        """Test that nonlocal list mutation in autograd.Function forward is handled correctly."""

        class SimpleAutogradFunc(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, z):
                # Simple computation
                o = smith.matmul(x, x) @ x
                out = x.sin()
                # Mutate the nonlocal list
                z.append(out)
                return smith.cos(smith.sin(o)), smith.sin(x)

            @staticmethod
            def backward(ctx, grad_output1, grad_output2):
                # Simple backward
                return grad_output1 + grad_output2, None

        def fn(x):
            z = []

            outs = SimpleAutogradFunc.apply(x, z)
            out1 = outs[0]
            # Check that the extra output pytree handling is done properly
            out2 = outs[-1]

            return out1 + out2, z[0]

        x = smith.randn(4, 4, requires_grad=True)
        ref = fn(x)

        opt_fn = smith.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

    def test_rewired_bwd_output(self):
        class Add(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                a = smith.sin(x)
                b = smith.cos(y)
                result = a * b
                # Save input, output and intermediate to test all cases
                ctx.save_for_backward(a, x, result)
                return result, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                (a, x, result) = ctx.saved_tensors
                return a * grad_b * 2 + x, result + grad_a * 3

        def fn(x, y):
            z = Add.apply(smith.cos(x), smith.cos(y))
            return z[0] + z[1]

        backend = AotEagerAndRecordGraphs()
        opt_fn = smith.compile(fn, fullgraph=True, backend=backend)
        x = smith.randn(8, 8, requires_grad=True)
        y = smith.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        smith._dynamo.mark_dynamic(x_clone, 0)
        smith._dynamo.mark_dynamic(y_clone, 0)

        ref = fn(x, y)
        res = opt_fn(x_clone, y_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        self.assertExpectedInline(
            smith._dynamo.testing.normalize_gm(
                backend.graphs[0].print_readable(print_output=False)
            ),
            """\
class GraphModule(smith.nn.Module):
    def forward(self, s77: "Sym(s17)", L_x_: "f32[s17, 8]", s17: "Sym(s17)", L_y_: "f32[s17, 8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        arg: "f32[s17, 8]" = smith.cos(l_x_);  l_x_ = None
        arg_1: "f32[s17, 8]" = smith.cos(l_y_);  l_y_ = None
        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply = smith.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, s77, arg, s17, arg_1, non_differentiable_idx = [], saved_for_backward_idx = [1, 2, 3]);  fwd_body_0 = bwd_body_0 = s77 = arg = s17 = arg_1 = None
        getitem: "f32[s17, 8]" = autograd_function_apply[0]
        getitem_1: "f32[s17, 8]" = autograd_function_apply[1];  autograd_function_apply = None

        add: "f32[s17, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class fwd_body_0(smith.nn.Module):
        def forward(self, s77: "Sym(s17)", cos: "f32[s17, 8]", s17: "Sym(s17)", cos_1: "f32[s17, 8]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            a: "f32[s17, 8]" = smith.sin(cos)

            b: "f32[s17, 8]" = smith.cos(cos_1);  cos_1 = None

            result: "f32[s17, 8]" = a * b

            out: "f32[s17, 8]" = a + b

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((result, out, a, b), (s17, a, cos, result))

    class bwd_body_0(smith.nn.Module):
        def forward(self, grad_a: "f32[s17, 8]", grad_b: "f32[s17, 8]", unused_0, unused_1, s17: "Sym(s17)", a: "f32[s17, 8]", arg: "f32[s17, 8]", result: "f32[s17, 8]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[s17, 8]" = a * grad_b;  a = grad_b = None
            mul_1: "f32[s17, 8]" = mul * 2;  mul = None
            add: "f32[s17, 8]" = mul_1 + arg;  mul_1 = arg = None
            mul_2: "f32[s17, 8]" = grad_a * 3;  grad_a = None
            add_1: "f32[s17, 8]" = result + mul_2;  result = mul_2 = None

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (None, add, None, add_1)
""",
        )

    def test_udf_output(self):
        class Foo:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        class Add(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                a = smith.sin(x)
                b = smith.cos(y)
                ctx.save_for_backward(a)
                return Foo(a, b), x * y

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                (a,) = ctx.saved_tensors
                return grad_b * 2, a * grad_b * 3

        def fn(x, y):
            z = Add.apply(x, y)
            return z[0].a + z[0].b + z[1]

        backend = AotEagerAndRecordGraphs()
        opt_fn = smith.compile(fn, fullgraph=True, backend=backend)
        x = smith.randn(8, 8, requires_grad=True)
        y = smith.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)

        ref = fn(x, y)
        res = opt_fn(x_clone, y_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        self.assertExpectedInline(
            smith._dynamo.testing.normalize_gm(
                backend.graphs[0].print_readable(print_output=False)
            ),
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[8, 8]", L_y_: "f32[8, 8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply = smith.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, l_y_, non_differentiable_idx = [], saved_for_backward_idx = [0]);  fwd_body_0 = bwd_body_0 = l_x_ = l_y_ = None
        getitem: "f32[8, 8]" = autograd_function_apply[0]
        getitem_1: "f32[8, 8]" = autograd_function_apply[1]
        getitem_2: "f32[8, 8]" = autograd_function_apply[2];  autograd_function_apply = None

        add: "f32[8, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        add_1: "f32[8, 8]" = add + getitem_2;  add = getitem_2 = None
        return (add_1,)

    class fwd_body_0(smith.nn.Module):
        def forward(self, l_x_: "f32[8, 8]", l_y_: "f32[8, 8]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            a: "f32[8, 8]" = smith.sin(l_x_)

            b: "f32[8, 8]" = smith.cos(l_y_)

            out: "f32[8, 8]" = l_x_ * l_y_;  l_x_ = l_y_ = None

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((a, b, out), (a,))

    class bwd_body_0(smith.nn.Module):
        def forward(self, unused_0, unused_1, grad_b: "f32[8, 8]", a: "f32[8, 8]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[8, 8]" = grad_b * 2
            mul_1: "f32[8, 8]" = a * grad_b;  a = grad_b = None
            mul_2: "f32[8, 8]" = mul_1 * 3;  mul_1 = None

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (mul, mul_2)
""",
        )

    def test_aliasing_output(self):
        class Add(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out

        def fn(x):
            y = Add.apply(x)
            if y is x:
                return smith.cos(y)
            return smith.sin(y)

        x = smith.randn(8, 8, requires_grad=True)

        ref = fn(x)
        backend = AotEagerAndRecordGraphs()
        opt_fn = smith.compile(fn, fullgraph=True, backend=backend)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        # Must have `view_as`
        self.assertTrue(
            "view_as" in backend.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            smith._dynamo.testing.normalize_gm(
                backend.graphs[0].print_readable(print_output=False)
            ),
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply = smith.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, non_differentiable_idx = [], saved_for_backward_idx = []);  fwd_body_0 = bwd_body_0 = l_x_ = None
        y: "f32[8, 8]" = autograd_function_apply[0];  autograd_function_apply = None

        sin: "f32[8, 8]" = smith.sin(y);  y = None
        return (sin,)

    class fwd_body_0(smith.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None
            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None

            view_as: "f32[8, 8]" = l_x_.view_as(l_x_);  l_x_ = None
            return ((view_as,), ())

    class bwd_body_0(smith.nn.Module):
        def forward(self, grad_out: "f32[8, 8]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None
            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (grad_out,)
""",
        )

    def test_nn_module_dataclasses_as_inputs(self):
        @dataclass
        class InputData:
            count: int
            values: smith.Tensor

        class Mod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 4
                self.b = smith.randn(4, 4)

        module = Mod()

        # Create input dataclass
        input_data = InputData(count=5, values=smith.randn(4, 4))

        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                module: smith.nn.Module,
                input_data: InputData,
                x: smith.Tensor,
            ) -> tuple[smith.Tensor, smith.Tensor]:
                # Extract inputs
                count = input_data.count
                values = input_data.values

                output_tensor = module.a + module.b * count * values

                return output_tensor + x, values + x

            @staticmethod
            def backward(ctx, grad_output, output_data):
                return grad_output * 4, None, grad_output * 2

        def fn(input_data, x):
            y, y_data = Foo.apply(module, input_data, x)
            return x + y + y_data + y_data

        # Call the function
        x = smith.randn(4, 4, requires_grad=True)
        ref = fn(input_data, x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = smith.compile(fn, fullgraph=True, backend=backend)
        res = opt_fn(input_data, x)
        self.assertEqual(ref, res)

        # Must have `view_as`
        self.assertExpectedInline(
            smith._dynamo.testing.normalize_gm(
                backend.graphs[0].print_readable(print_output=False)
            ),
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[4, 4]", L_module_b: "f32[4, 4]", L_input_data_values: "f32[4, 4]"):
        l_x_ = L_x_
        l_module_b = L_module_b
        l_input_data_values = L_input_data_values

        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply = smith.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_module_b, l_input_data_values, l_x_, non_differentiable_idx = [], saved_for_backward_idx = []);  fwd_body_0 = bwd_body_0 = l_module_b = l_input_data_values = None
        getitem: "f32[4, 4]" = autograd_function_apply[0]
        getitem_1: "f32[4, 4]" = autograd_function_apply[1];  autograd_function_apply = None

        add: "f32[4, 4]" = l_x_ + getitem;  l_x_ = getitem = None
        add_1: "f32[4, 4]" = add + getitem_1;  add = None
        add_2: "f32[4, 4]" = add_1 + getitem_1;  add_1 = getitem_1 = None
        return (add_2,)

    class fwd_body_0(smith.nn.Module):
        def forward(self, l_module_b: "f32[4, 4]", l_input_data_values: "f32[4, 4]", l_x_: "f32[4, 4]"):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[4, 4]" = l_module_b * 5;  l_module_b = None
            mul_1: "f32[4, 4]" = mul * l_input_data_values
            output_tensor: "f32[4, 4]" = 4 + mul_1

            out: "f32[4, 4]" = output_tensor + l_x_
            out_1: "f32[4, 4]" = l_input_data_values + l_x_;  l_input_data_values = l_x_ = None

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((out, out_1, mul, mul_1, output_tensor), ())

    class bwd_body_0(smith.nn.Module):
        def forward(self, grad_output: "f32[4, 4]", output_data: "f32[4, 4]", unused_0, unused_1, unused_2):
            _set_grad_enabled = smith._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[4, 4]" = grad_output * 4;  mul = None
            mul_1: "f32[4, 4]" = grad_output * 2;  grad_output = None

            _set_grad_enabled_1 = smith._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (None, None, mul_1)
""",
        )

    def test_nn_module_dataclasses_as_io(self):
        @dataclass
        class InputData:
            count: int
            values: smith.Tensor

        @dataclass
        class OutputData:
            result1: smith.Tensor
            result2: smith.Tensor

        # Create a simple linear module

        class Mod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 4
                self.b = smith.randn(4, 4)

        module = Mod()

        # Create input dataclass
        input_data = InputData(count=5, values=smith.randn(4, 4))

        class Foo(smith.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                module: smith.nn.Module,
                input_data: InputData,
                x: smith.Tensor,
            ) -> tuple[smith.Tensor, OutputData]:
                # Extract inputs
                count = input_data.count
                values = input_data.values

                output_tensor = module.a + module.b * count * values

                output_data = OutputData(
                    result1=output_tensor + count, result2=output_tensor * (count + 1)
                )

                return output_tensor + x, output_data

            @staticmethod
            def backward(ctx, grad_output, output_data):
                return grad_output * 4, None, grad_output * 2

        def fn(input_data, x):
            y, y_data = Foo.apply(module, input_data, x)
            return x + y + y_data.result1 + y_data.result2

        # Call the function
        x = smith.randn(4, 4, requires_grad=True)
        ref = fn(input_data, x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = smith.compile(fn, fullgraph=True, backend=backend)
        res = opt_fn(input_data, x)
        self.assertEqual(ref, res)


class AutogradFunctionFuncsmithTests(smith._dynamo.test_case.TestCase):
    """Tests for autograd.Function compatibility with smith.func transforms.

    See https://github.com/blacksmith/blacksmith/issues/174067
    """

    def test_new_style_autograd_function_with_grad_no_compile(self):
        """Baseline: new-style autograd.Function works with smith.func.grad."""

        class NewStyleOp(smith.autograd.Function):
            @staticmethod
            def forward(x):
                return x * 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        def fn(x):
            return NewStyleOp.apply(x).sum()

        x = smith.tensor([1.0, 2.0], requires_grad=True)
        result = smith.func.grad(fn)(x)
        self.assertEqual(result, smith.tensor([2.0, 2.0]))

    def test_old_style_autograd_function_with_grad_no_compile(self):
        """Baseline: old-style autograd.Function fails with smith.func.grad."""

        class OldStyleOp(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        def fn(x):
            return OldStyleOp.apply(x).sum()

        x = smith.tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "must override the setup_context staticmethod"
        ):
            smith.func.grad(fn)(x)

    def test_new_style_autograd_function_with_grad_compiled(self):
        """New-style autograd.Function compiled should work with smith.func.grad.

        This is the main bug from https://github.com/blacksmith/blacksmith/issues/174067.
        smith.compile wraps autograd.Function in ApplyTemplate which lacks
        setup_context, breaking smith.func compatibility.
        """

        class NewStyleOp(smith.autograd.Function):
            @staticmethod
            def forward(x):
                return x * 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        def fn(x):
            return NewStyleOp.apply(x)

        compiled_fn = smith.compile(fn, backend="eager")

        def loss_fn(x):
            return compiled_fn(x).sum()

        x = smith.tensor([1.0, 2.0], requires_grad=True)
        result = smith.func.grad(loss_fn)(x)
        self.assertEqual(result, smith.tensor([2.0, 2.0]))

    def test_old_style_autograd_function_with_grad_compiled(self):
        """Old-style autograd.Function compiled should work with smith.func.grad.

        Even though the original function is old-style, the compiled version
        (ApplyTemplate) can support smith.func transforms because it uses
        traced graphs.
        """

        class OldStyleOp(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        def fn(x):
            return OldStyleOp.apply(x)

        compiled_fn = smith.compile(fn, backend="eager")

        def loss_fn(x):
            return compiled_fn(x).sum()

        x = smith.tensor([1.0, 2.0], requires_grad=True)
        result = smith.func.grad(loss_fn)(x)
        self.assertEqual(result, smith.tensor([2.0, 2.0]))


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
