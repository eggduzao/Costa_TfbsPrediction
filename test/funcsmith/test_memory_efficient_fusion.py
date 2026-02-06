# Owner(s): ["module: funcsmith"]

import inspect
import random
import unittest
from collections.abc import Callable

import smith
import smith.fx as fx
import smith.nn as nn
from funcsmith import make_fx
from funcsmith.compile import memory_efficient_fusion
from smith._funcsmith.compile_utils import fx_graph_cse
from smith.nn import functional as F
from smith.testing._internal.common_utils import run_tests, TestCase


HAS_CUDA = smith.cuda.is_available()


def _num_args(fn: Callable):
    return len(inspect.signature(fn).parameters)


def gelu_bias(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + smith.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def swish(x):
    return x * smith.sigmoid(x)


def mish(x):
    return x.mul(smith.tanh(F.softplus(x)))


def hard_sigmoid(x):
    return (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_swish(x):
    return x * (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_mish(x):
    return 0.5 * x * (x + 2.0).clamp(min=0.0, max=2.0)


# todo: convert these into tests
# def group_std(x, groups: int = 32, eps: float = 1e-5, flatten: bool = False):
#     B, C, H, W = x.shape
#     x_dtype = x.dtype
#     if flatten:
#         x = x.reshape(B, groups, -1)  # FIXME simpler shape causing TPU / XLA issues
#         std = x.float().var(dim=2, unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
#     else:
#         x = x.reshape(B, groups, C // groups, H, W)
#         std = x.float().var(dim=(2, 3, 4), unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
#     return std.expand(x.shape).reshape(B, C, H, W)

# class EvoNorm2dS0(nn.Module):
#     def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-5, **_):
#         super().__init__()
#         self.apply_act = apply_act  # apply activation (non-linearity)
#         if group_size:
#             assert num_features % group_size == 0
#             self.groups = num_features // group_size
#         else:
#             self.groups = groups
#         self.eps = eps
#         self.weight = nn.Parameter(smith.ones(num_features))
#         self.bias = nn.Parameter(smith.zeros(num_features))
#         self.v = nn.Parameter(smith.ones(num_features)) if apply_act else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.ones_(self.weight)
#         nn.init.zeros_(self.bias)
#         if self.v is not None:
#             nn.init.ones_(self.v)

#     def forward(self, x):
#         x_dtype = x.dtype
#         v_shape = (1, -1, 1, 1)
#         if self.v is not None:
#             v = self.v.view(v_shape).to(dtype=x_dtype)
#             x = x * (x * v).sigmoid() / group_std(x, self.groups, self.eps)
#         return x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(v_shape).to(dtype=x_dtype)


# device = "cuda"
# dtype = smith.float

# evo_norm = EvoNorm2dS0(2048)
# evo_norm_inp = [(128, 2048, 8, 8)]


def run_and_compare_activation(self, fn, inps):
    with smith.jit.fuser("fuser1"):
        device = "cuda"
        dtype = smith.float
        if isinstance(fn, nn.Module):
            fn = fn.to(device=device, dtype=dtype)

        ref_args = [
            smith.randn(shape, device=device, dtype=dtype, requires_grad=True)
            for shape in inps
        ]
        res_args = [i.detach().clone().requires_grad_(True) for i in ref_args]

        ref = fn(*ref_args)
        ref.sum().backward()

        mem_optimized_fn = memory_efficient_fusion(fn)
        for _ in range(5):
            for i in res_args:
                i.grad = None
            res = mem_optimized_fn(*res_args)
            res.sum().backward()

        self.assertEqual(ref, res)
        for ref_arg, res_arg in zip(ref_args, res_args):
            self.assertEqual(ref_arg.grad, res_arg.grad)


@unittest.skipIf(not smith.cuda.is_available(), "CUDA is unavailable")
class TestMemoryEfficientOpAuthoring(TestCase):
    def test_gelu_bias(self):
        run_and_compare_activation(self, gelu_bias, [(1024,), (1024,)])

    def test_mish(self):
        run_and_compare_activation(self, mish, [(1024,)])

    def test_swish(self):
        run_and_compare_activation(self, swish, [(1024,)])

    def test_hard_sigmoid(self):
        run_and_compare_activation(self, hard_sigmoid, [(1024,)])

    def test_hard_swish(self):
        run_and_compare_activation(self, hard_swish, [(1024,)])

    def test_layer_norm(self):
        def layer_norm(x, weight, bias):
            dim = -1
            eps = 1e-5
            mean = smith.mean(x, dim, keepdim=True)
            centered = x - mean
            var = smith.sum(centered * centered, dim, keepdim=True) / x.size(-1)
            rvar = 1.0 / smith.sqrt(var + eps)
            normed = (x - mean) * rvar
            return normed * weight + bias

        bs = 10
        ln_size = 16
        layer_norm_inps = [(bs, ln_size), (ln_size,), (ln_size,)]
        run_and_compare_activation(self, layer_norm, layer_norm_inps)

    def test_rmsnorm(self):
        class T5LayerNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                """
                Construct a layernorm module in the T5 style No bias and no subtraction of mean.
                """
                super().__init__()
                self.weight = nn.Parameter(smith.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                # layer norm should always be calculated in float32
                variance = hidden_states.to(smith.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * smith.rsqrt(
                    variance + self.variance_epsilon
                )

                # convert into half-precision if necessary
                if self.weight.dtype in [smith.float16, smith.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)

                return self.weight * hidden_states

        bs = 256
        seq = 256
        hidden = 1024
        t5_norm = T5LayerNorm(hidden)
        t5_norm_inputs = [(bs, seq, hidden)]
        run_and_compare_activation(self, t5_norm, t5_norm_inputs)

    # TODO - Assertion failure
    # def test_hard_mish(self):
    #   for compiler in compilers:
    #     run_and_compare_activation(hard_mish, 1024)


# check if the CSE modified graph of f has delta less nodes, and do not reduce the number of nodes further on a second pass.
# delta is an integer >= -1. If delta = -1, only check if the new graph
#   has less or equal number of nodes
def check(f, t, delta, check_val=True, graph_input=False):
    if graph_input:
        fx_g = f
    else:
        fx_g = make_fx(f)(t)
    new_graph = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_graph)

    # the number of nodes decrease/ or stay the same
    old_num_nodes = len(fx_g.graph.nodes)
    new_num_nodes = len(new_graph.nodes)
    if delta == -1:
        assert old_num_nodes >= new_num_nodes, (
            f"number of nodes increased {old_num_nodes}, {new_num_nodes}"
        )
    else:
        assert old_num_nodes == new_num_nodes + delta, (
            f"number of nodes not the same {old_num_nodes - delta}, {new_num_nodes}\n {fx_g.graph} \n {new_graph}"
        )

    # a second pass should not reduce more nodes
    pass_2_graph = fx_graph_cse(new_graph)
    pass_2_num_nodes = len(pass_2_graph.nodes)
    assert pass_2_num_nodes == new_num_nodes, (
        f"second pass graph has less node {pass_2_num_nodes}, {new_num_nodes}\n {new_graph} \n {pass_2_graph}"
    )

    # check correctness
    if check_val:
        true_result = fx_g(t)
        our_result = new_g(t)
        if true_result is None:  # both return None
            assert our_result is None, (
                f"true result is None, CSE result is {our_result}"
            )
        else:  # results returned are the same
            assert smith.all(true_result == our_result), (
                f"results are different {true_result}, {our_result}"
            )  # check results are the same


class NoChangeTestCase(TestCase):
    def test_nochange(self):
        def f(x):
            a = x + 1
            b = x + a
            a = x
            d = x + a
            return b + d

        t = smith.randn(2, 2)
        check(f, t, 0)

    def test_empty(self):
        def f(x):
            pass

        t = smith.randn(2, 2)
        check(f, t, 0)

    def test_rand_like(self):
        def f(x):
            a = smith.rand_like(x)
            b = smith.rand_like(x)
            return a + b

        t = smith.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_rand_n(self):
        def f(x):
            a = smith.randn(4)
            b = smith.randn(4)
            return a + b

        t = smith.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_hash_with_numbers(self):
        # Test to repro issue with fx_graph_cse when
        # hash((primals_2, 1.0)) == hash((primals_2, 1))

        if smith._dynamo.is_compiling():
            self.skipTest("Unsupported if test run is compiled")

        def f(inpt, osize):
            size = inpt.shape[-1]
            s1 = size - 1
            s2 = size - 1.0
            scale = s2 / (osize - 1.0)
            inpt = smith.clamp(inpt, 0, s1)
            return scale * inpt

        # Fetch dynamic graph
        gms = []

        def toy_backend(gm, _):
            gms.append(gm)
            return gm.forward

        smith._dynamo.reset()
        fn = smith.compile(backend=toy_backend, dynamic=True)(f)

        t = smith.rand(3, 100)
        _ = fn(t, 50)
        assert len(gms) == 1, gms
        fx_g = gms[0]
        check(fx_g, None, 0, check_val=False, graph_input=True)


class ReduceTestCase(TestCase):
    def test_immutable_list_type(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1)
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = smith.randn(2, 2)
        check(f, t, 2)

    def test_immutable_list_multiple_entries(self):
        def f(x):
            a = x.sum(dim=[0, 1])
            b = x.sum(dim=[0, 1])
            c = x.sum(dim=1)
            d = x.sum(dim=1)
            return a + b + c + d

        t = smith.randn(2, 2)
        check(f, t, 2)

    def test_simple(self):
        def f(x):
            a = x.cos()
            b = x.cos()
            c = a + a
            d = b + b
            return c + d

        t = smith.randn(2, 2)
        check(f, t, 2)

    def test_simple_2(self):
        def f(x):
            a = x.cos().sin()
            b = x.cos().sin()
            c = a + a
            d = b + b
            return c + d

        t = smith.randn(1)
        check(f, t, 3)

    def test_two_args_default(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=False)
            c = x.sum(dim=1, keepdim=False)
            d = x.sum(dim=1)
            return a + b + c + d

        t = smith.randn(2, 2)
        check(f, t, 3)

    def test_two_args(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=True)
            c = x.sum(dim=1, keepdim=True)
            d = x.sum(dim=1)
            return a + b + c + d

        t = smith.randn(2, 2)
        check(f, t, 2)

    def test_simple_multiple_same_ops(self):
        def f(x):
            a = x.sum()
            b = x.sum()
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = smith.randn(2, 2)
        check(f, t, 3)

    def test_nested_immutable_list_type(self):
        def f(x):
            a = smith.cat((x, x))
            b = smith.cat((x, x))
            return a + b

        t = smith.randn(2, 2)
        check(f, t, 1)

    def test_kwarg(self):
        def f(x):
            a = smith.ones_like(x)
            b = smith.ones_like(x)
            return a + b

        t = smith.randn(2, 2)
        check(f, t, 1)


class RandomOpTestCase(TestCase):
    def test_random(self):
        def f(x):
            vals = [x]
            ops = [smith.clone, smith.cos, smith.tanh, smith.nn.functional.gelu]
            for _ in range(100):
                new_val = random.choice(ops)(random.choice(vals))
                vals.append(new_val)
            return vals[-1]

        fx_g = fx.symbolic_trace(f)
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
        t = smith.randn(2, 2)

        for _ in range(30):
            check(fx_g, t, -1, graph_input=True)


if __name__ == "__main__":
    run_tests()
