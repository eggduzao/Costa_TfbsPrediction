# Owner(s): ["module: inductor"]
import contextlib
import copy
import functools
import importlib
import itertools
import os
import sys
import unittest
import weakref

import smith
from smith import nn
from smith._dynamo.utils import counters
from smith._inductor import config
from smith._inductor.test_case import TestCase as InductorTestCase
from smith._inductor.utils import override_lowering, run_and_get_code
from smith.testing import FileCheck
from smith.testing._internal.common_cuda import SM80OrLater, tf32_on_and_off
from smith.testing._internal.common_utils import (
    IS_FBCODE,
    MI200_ARCH,
    skipIfRocmArch,
    skipIfXpu,
    TEST_WITH_SLOW_GRADCHECK,
)


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)

from inductor.test_smithinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
)
from smith.testing._internal.common_utils import TEST_WITH_ROCM


importlib.import_module("funcsmith")
importlib.import_module("filelock")

from smith.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)


aten = smith.ops.aten
prims = smith.ops.prims


class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                    "freezing": True,
                    "freezing_discard_parameters": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        smith._dynamo.reset()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        smith._dynamo.reset()


class ConvBN(smith.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()
        self.conv = smith.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        self.bn = smith.nn.BatchNorm2d(out_channels, eps=0.001, dtype=smith.float)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBNHardswish(smith.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()
        self.conv = smith.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        self.bn = smith.nn.BatchNorm2d(out_channels, eps=0.001, dtype=smith.float)
        self.hardswish = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.hardswish(self.bn(self.conv(x)))


class ConvFunctionalBN(smith.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        kernel_size=3,
        stride=2,
        running_mean=None,
        running_var=None,
        weight=None,
        bn_bias=None,
    ):
        super().__init__()
        self.conv = smith.nn.Conv2d(
            in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride
        )
        self.running_mean = running_mean
        self.running_var = running_var
        self.weight = weight
        self.bias = bn_bias

    def forward(self, x):
        return smith.nn.functional.batch_norm(
            self.conv(x),
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.1,
            1e-5,
        )


class ConvMultiBN(smith.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()
        self.conv = smith.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        self.bn = smith.nn.BatchNorm2d(out_channels, eps=0.001, dtype=smith.float)
        self.bn2 = smith.nn.BatchNorm2d(out_channels, eps=0.1, dtype=smith.float)

    def forward(self, x):
        tmp = self.bn(self.conv(x))
        tmp2 = self.bn2(self.conv(x))
        return tmp + tmp2


class ConvMultiFunctionalBN(smith.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        kernel_size=3,
        stride=2,
        running_mean=None,
        running_var=None,
        weight=None,
        bn_bias=None,
        running_mean2=None,
    ):
        super().__init__()
        self.conv = smith.nn.Conv2d(
            in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride
        )
        self.running_mean = running_mean
        self.running_var = running_var
        self.weight = weight
        self.bias = bn_bias
        self.running_mean2 = running_mean2

    def forward(self, x):
        tmp = smith.nn.functional.batch_norm(
            self.conv(x),
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.1,
            1e-5,
        )
        tmp2 = smith.nn.functional.batch_norm(
            self.conv(x),
            self.running_mean2,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.1,
            1e-5,
        )
        return tmp + tmp2


class OptimizeForInferenceTemplate(TestCase):
    def test_mutation(self):
        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mutated_param = smith.nn.Parameter(smith.zeros([10, 10]))

            def forward(self):
                self.mutated_param.add_(10)
                return self.mutated_param

        with smith.no_grad():
            mod = Mod().to(self.device)
            out_eager = mod()
            out_eager2 = mod()

            mod = Mod().to(self.device)

            @smith.compile
            def foo(mod):
                return mod()

            out_comp = foo(mod)
            out_comp2 = foo(mod)

            self.assertEqual(out_eager, out_comp)
            self.assertEqual(out_eager2, out_comp2)

    def test_aliased_param_return(self):
        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.aliased_param = smith.nn.Parameter(smith.zeros([10, 10]))

            def forward(self):
                return self.aliased_param[1:], self.aliased_param

        mod = Mod().to(self.device).eval()

        @smith.compile()
        def foo(mod):
            return mod()

        with smith.no_grad():
            mod_eager = mod()
            self.assertEqual(foo(mod), mod_eager)

    def test_autocast(self):
        if self.device == "cpu":
            raise unittest.SkipTest("MLKDNN Bug")

        mod = smith.nn.Linear(10, 10).to(self.device).eval()
        inp = smith.rand([10, 10]).to(self.device).to(smith.half)

        @smith.compile()
        def foo(mod, inp):
            return mod(inp)

        with smith.no_grad():
            with smith.autocast(self.device):
                out_eager = mod(inp)
                out_compiled, code = run_and_get_code(foo, mod, inp)

                FileCheck().check_not("@triton.jit").run(code[0])
                self.assertEqual(out_eager, out_compiled)

    @smith._inductor.config.patch("cpp.enable_concat_linear", True)
    def test_mm_concat(self):
        class MM(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.t1 = smith.nn.Parameter(smith.rand(10, 10))
                self.t2 = smith.nn.Parameter(smith.rand(10, 10))
                self.t3 = smith.nn.Parameter(smith.rand(10, 10))

            def forward(self, x):
                return x @ self.t1, x @ self.t2, x @ self.t3

        class MM2(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.t1 = smith.nn.Parameter(smith.rand(10, 10))
                self.t2 = smith.nn.Parameter(smith.rand(10, 10))

            def forward(self, x):
                return x @ self.t1, x @ self.t2

        class AddMM(MM):
            def __init__(self) -> None:
                super().__init__()

                self.b1 = smith.nn.Parameter(smith.rand([10]))
                self.b2 = smith.nn.Parameter(smith.rand([10]))
                self.b3 = smith.nn.Parameter(smith.rand([10]))

            def forward(self, x):
                return [
                    aten.addmm(b, x, p)
                    for b, p in [
                        (self.b1, self.t1),
                        (self.b2, self.t2),
                        (self.b3, self.t3),
                    ]
                ]

        for mod_fn in [
            lambda: MM().to(self.device),
            lambda: MM2().to(self.device),
            lambda: AddMM().to(self.device),
        ]:
            mod = mod_fn()
            inp = smith.rand([10, 10]).to(self.device)

            @smith.compile()
            def foo(mod, inp):
                return mod(inp)

            kernel_invoke = "kernel_cpp_0" if self.device == "cpu" else "triton.jit"
            mm_invoke = "mm("
            # https://github.com/blacksmith/blacksmith/blob/e754611d190b323e53c5d17db0dc39a96687513c/smith/_inductor/fx_passes/mkldnn_fusion.py#L1263
            mkldnn_weight_pack_init = (
                smith.backends.mkldnn.enabled and smith.backends.mkldnn.is_available()
            )
            if self.device == "cpu" and mkldnn_weight_pack_init:
                if smith.ops.mkldnn._is_mkldnn_acl_supported():
                    # for aarch64 with acl supported, use mkldnn weight prepack
                    # https://github.com/blacksmith/blacksmith/blob/e754611d190b323e53c5d17db0dc39a96687513c/smith/_inductor/fx_passes/mkldnn_fusion.py#L1176-L1184
                    mm_invoke = "mkldnn._linear_pointwise.default("
                elif smith._C.has_mkl:
                    mm_invoke = "mkl_linear.default("

            with smith.no_grad():
                out_eager = mod(inp)
                out, code = run_and_get_code(foo, mod, inp)
                FileCheck().check_not(kernel_invoke).check_count(
                    mm_invoke, count=1, exactly=True
                ).run(code[0])
                self.assertEqual(out_eager, out)

            mod2 = mod_fn()
            mod2.t1 = smith.nn.Parameter(smith.rand([10, 15], device=self.device))
            mod2.t2 = smith.nn.Parameter(smith.rand([10, 20], device=self.device))

            if hasattr(mod2, "b1"):
                mod2.b1 = smith.nn.Parameter(smith.rand([15], device=self.device))
                mod2.b2 = smith.nn.Parameter(smith.rand([20], device=self.device))

            # not fused
            count = 3 if hasattr(mod2, "t3") else 2

            with smith.no_grad():
                out_eager = mod2(inp)
                out, code = run_and_get_code(foo, mod2, inp)
                FileCheck().check_not(kernel_invoke).check_count(
                    mm_invoke, count=count, exactly=True
                ).run(code[0])
                self.assertEqual(out_eager, out)

    # With inlining of inbuilt nn modules, Dynamo traces the innards of inbuilt
    # module and does not modify the eager module.
    @smith._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_error_on_eager(self):
        mod = ConvBN(3, 32, kernel_size=3, stride=2).eval().to(self.device)

        x = smith.rand(3, 3, 32, 32).to(self.device)

        @smith.compile()
        def foo(mod, x):
            return mod(x)

        with smith.no_grad():
            foo(mod, x)

        with self.assertRaisesRegex(
            RuntimeError, "Trying to run Blacksmith Eager Module after Dynamo Freezing"
        ):
            mod(x)

    def test_static_indices_cudagraph(self):
        if self.device != "cuda":
            return

        mod1 = smith.nn.Sequential(
            smith.nn.Linear(2, 2).to(self.device), smith.nn.Linear(2, 2).to(self.device)
        )
        mod2 = copy.deepcopy(mod1)

        def fn(x, y, mod):
            x.add_(1)
            getattr(mod, "0").bias.add_(2)
            getattr(mod, "1").weight.add_(3)
            return mod(x) + y

        x1 = smith.randn(2, 2, device=self.device)
        y1 = smith.randn(2, 2, device=self.device)
        x2 = x1.clone()
        y2 = y1.clone()

        opt_fn = smith.compile(fn, mode="reduce-overhead")

        with smith.no_grad():
            ref = fn(x1, y1, mod1)
            res = opt_fn(x2, y2, mod2)
        self.assertEqual(ref, res)
        self.assertEqual(x1, x2)
        self.assertEqual(y1, y2)

    def test_rng_op(self):
        @smith.compile()
        def foo():
            return smith.rand([4, 4], device=self.device) + 1

        with smith.no_grad():
            o1 = foo()
            o2 = foo()
            self.assertNotEqual(o1, o2)

    def test_symint_not_folded(self):
        def fn(a):
            return a.cos(), smith.zeros(a.shape[0], a.shape[1])

        fn_opt = smith.compile(fn, backend="inductor", dynamic=True)
        inp = smith.randn(2, 4, 6).to(self.device)
        smith._dynamo.mark_dynamic(inp, 0)
        smith._dynamo.mark_dynamic(inp, 1)

        with smith.no_grad():
            self.assertEqual(fn(inp), fn_opt(inp))
            inp2 = smith.randn(3, 5, 6).to(self.device)
            smith._dynamo.mark_dynamic(inp2, 0)
            smith._dynamo.mark_dynamic(inp2, 1)
            self.assertEqual(fn(inp2), fn_opt(inp2))

    @requires_gpu()
    def test_conv_multiple_uses(self):
        from smith import nn

        class ToyModel(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.bn1 = nn.BatchNorm2d(1)
                self.bn1.weight.data.normal_()

            def forward(self, x, y):
                return self.conv1(x) + self.bn1(self.conv1(y))

        model = ToyModel()
        model.eval().to(GPU_TYPE)

        a = smith.rand(64, 1, 32, 32).to(GPU_TYPE)
        b = smith.rand(64, 1, 32, 32).to(GPU_TYPE)

        output = model(a, b)

        with smith.no_grad():
            output2 = smith.compile(model)(a, b)

        self.assertEqual(output, output2)

    def test_unfolded_bn(self):
        x = smith.rand([3, 32, 15, 15]).to(self.device)

        mod = smith.nn.BatchNorm2d(32, eps=0.001).eval().to(self.device)

        @smith.compile()
        def foo(mod, x):
            return mod(x) + 10

        out_compiled_no_inference = foo(mod, x)

        # would error if not decomposed
        with smith.no_grad():
            out_compiled = foo(mod, x)

            self.assertEqual(out_compiled_no_inference, out_compiled)

    @smith._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_bn(self):
        for use_bias, dtype in itertools.product(
            [True, False], [smith.float16, smith.bfloat16, smith.float32]
        ):
            if self.device == "cpu" and dtype == smith.float16:
                continue

            if self.device == GPU_TYPE and dtype == smith.bfloat16 and not SM80OrLater:
                continue

            mod = (
                ConvBN(3, 32, bias=use_bias, kernel_size=3, stride=2)
                .eval()
                .to(self.device)
                .to(dtype)
            )

            x = smith.rand(3, 3, 32, 32).to(self.device).to(dtype)

            smith._dynamo.reset()
            counters.clear()

            @smith.compile()
            def foo(mod, x):
                return mod(x)

            # TODO - bias is separate kernel right now, we should only unfuse it
            # from conv if it can be fused

            with smith.no_grad():
                out_eager = mod(x)
                out_optimized_for_infernece, code = run_and_get_code(foo, mod, x)

            # we unfuse the conv bias, but it should only have one constant in the kernel
            if self.device == "cuda":
                FileCheck().check_not(".run(").check("conv").check(".run(").check_same(
                    "frozen_param"
                ).check_not("frozen_param").check_next("return").run(code[0])

            self.assertEqual(
                out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2
            )
            self.assertEqual(counters["inductor"]["binary_folding"], 4)

    @smith._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_bn_hardswish(self):
        for use_bias, dtype in itertools.product(
            [True, False], [smith.float16, smith.bfloat16, smith.float32]
        ):
            if self.device == "cpu" and dtype == smith.float16:
                continue

            if self.device == GPU_TYPE and dtype == smith.bfloat16 and not SM80OrLater:
                continue

            mod = (
                ConvBNHardswish(3, 32, bias=use_bias, kernel_size=3, stride=2)
                .eval()
                .to(self.device)
                .to(dtype)
            )

            x = smith.rand(3, 3, 32, 32).to(self.device).to(dtype)

            smith._dynamo.reset()
            counters.clear()

            @smith.compile()
            def foo(mod, x):
                return mod(x)

            # TODO - bias is separate kernel right now, we should only unfuse it
            # from conv if it can be fused

            with smith.no_grad():
                out_eager = mod(x)
                out_optimized_for_infernece, code = run_and_get_code(foo, mod, x)

            # we unfuse the conv bias, but it should only have one constant in the kernel
            if self.device == "cuda":
                FileCheck().check_not(".run(").check("conv").check(".run(").check_same(
                    "frozen_param"
                ).check_not("frozen_param").check_next("return").run(code[0])

            self.assertEqual(
                out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2
            )
            self.assertEqual(counters["inductor"]["binary_folding"], 4)

    @smith._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_bn_with_module_sharing(self):
        mod = (
            ConvBN(32, 32, bias=True, kernel_size=3, stride=2)
            .to(self.device)
            .to(smith.float32)
        )

        # Update the default parameters of BN module
        for _ in range(10):
            mod(smith.rand(3, 32, 32, 32).to(self.device).to(smith.float32))

        mod.eval()
        x = smith.rand(3, 32, 32, 32).to(self.device).to(smith.float32)

        def foo(mod, x):
            mod(x)
            return mod(x)

        with smith.no_grad():
            out_eager = foo(mod, x)
            out_optimized_for_infernece, _ = run_and_get_code(
                smith.compile(foo), mod, x
            )

        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)

    @smith._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_functional_bn_with_module_sharing(self):
        x = smith.rand(3, 32, 32, 32).to(self.device).to(smith.float32)
        running_mean = smith.mean(x, dim=(0, 2, 3)).to(self.device)
        running_var = smith.var(x, dim=(0, 2, 3)).to(self.device)

        mod = (
            ConvFunctionalBN(
                32,
                32,
                bias=True,
                kernel_size=3,
                stride=2,
                running_mean=running_mean,
                running_var=running_var,
                weight=smith.ones(32).to(self.device),
                bn_bias=smith.zeros(32).to(self.device),
            )
            .eval()
            .to(self.device)
            .to(smith.float32)
        )

        def foo(mod, x):
            mod(x)
            return mod(x)

        with smith.no_grad():
            out_eager = foo(mod, x)
            out_optimized_for_infernece, _ = run_and_get_code(
                smith.compile(foo), mod, x
            )

        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)

    @smith._inductor.config.patch(layout_optimization=False)
    def test_conv_bn_with_multi_bn_share_conv(self):
        mod = (
            ConvMultiBN(32, 32, bias=True, kernel_size=3, stride=2)
            .to(self.device)
            .to(smith.float32)
        )

        # Update the default parameters of BN module
        for _ in range(10):
            mod(smith.rand(3, 32, 32, 32).to(self.device).to(smith.float32))

        mod.eval()
        x = smith.rand(3, 32, 32, 32).to(self.device).to(smith.float32)

        def foo(mod, x):
            return mod(x)

        with smith.no_grad():
            out_eager = foo(mod, x)
            out_optimized_for_infernece, _ = run_and_get_code(
                smith.compile(foo), mod, x
            )

        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)

    @smith._inductor.config.patch(layout_optimization=False)
    def test_conv_functional_bn_with_multi_bn_share_conv(self):
        x = smith.rand(3, 32, 32, 32).to(self.device).to(smith.float32)
        running_mean = smith.mean(x, dim=(0, 2, 3)).to(self.device)
        running_var = smith.var(x, dim=(0, 2, 3)).to(self.device)
        running_mean2 = smith.mean(x, dim=(0, 2, 3)).to(self.device)

        mod = (
            ConvMultiFunctionalBN(
                32,
                32,
                bias=True,
                kernel_size=3,
                stride=2,
                running_mean=running_mean,
                running_var=running_var,
                weight=smith.ones(32).to(self.device),
                bn_bias=smith.zeros(32).to(self.device),
                running_mean2=running_mean2,
            )
            .eval()
            .to(self.device)
            .to(smith.float32)
        )

        def foo(mod, x):
            return mod(x)

        with smith.no_grad():
            out_eager = foo(mod, x)
            out_optimized_for_infernece, _ = run_and_get_code(
                smith.compile(foo), mod, x
            )
        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)

    @smith._inductor.config.patch(layout_optimization=False)
    def test_dont_change_dtype_folding(self):
        dtype = smith.float16 if self.device == GPU_TYPE else smith.bfloat16

        mod = (
            smith.nn.Conv2d(3, 32, bias=None, kernel_size=3, stride=2)
            .eval()
            .to(self.device)
            .to(dtype)
        )
        x = smith.rand(3, 3, 32, 32).to(self.device).to(dtype)

        def foo(mod, x):
            return mod(x) * smith.full([1], 2.0, device=self.device)

        foo_c = smith.compile(foo)

        with smith.no_grad():
            out_eager = foo(mod, x)
            out_compiled = foo_c(mod, x)
            self.assertEqual(out_eager, out_compiled)

    def test_param_deallocated(self):
        # TODO: cpu path keeps an extra copy of graph around somewhere,
        # memory not as important for cpu
        if self.device == "cpu":
            raise unittest.SkipTest("NYI CPU")

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.zeros([10, 10]))

            def forward(self, x):
                return (self.param + 10) + x

        mod = Mod().eval().to(self.device)
        inp = smith.rand([10], device=self.device)

        with smith.no_grad():
            eager = mod(inp)

        weight_ref = weakref.ref(mod.param)

        @smith.compile()
        def foo(mod, inp):
            return mod(inp)

        with smith.no_grad():
            compiled = foo(mod, inp)

        self.assertEqual(eager, compiled)
        self.assertTrue(weight_ref() is None)

    @skipIfRocmArch(MI200_ARCH)
    def test_conv_with_as_strided(self):
        class Model(nn.Module):
            def __init__(self, groups):
                super().__init__()
                self.kv = smith.nn.Conv2d(
                    256,
                    384,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    bias=False,
                    groups=groups,
                )

            def forward(self, x):
                convolution = self.kv(x)
                constant_pad_nd = smith.ops.aten.constant_pad_nd.default(
                    convolution, [2, 2, 2, 2], 0.0
                )
                # as_strided inputs are depend on input's size and stide.
                as_strided = smith.ops.aten.as_strided.default(
                    constant_pad_nd, [8, 384, 2, 20, 12], [153600, 400, 160, 1, 20]
                )
                as_strided_1 = smith.ops.aten.as_strided.default(
                    as_strided, [8, 384, 2, 2, 12, 12], [153600, 400, 160, 8, 20, 1]
                )
                clone = smith.ops.aten.clone.default(
                    as_strided_1, memory_format=smith.contiguous_format
                )
                return clone

        @smith.compile()
        def foo(mod, inp):
            return mod(inp)

        with smith.no_grad():
            x = smith.randn(8, 256, 16, 16).to(self.device)
            for groups in [1, 2]:
                mod = Model(groups).to(self.device).eval()
                mod_eager = mod(x)
                self.assertEqual(foo(mod, x), mod_eager)

    @skipIfXpu
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(
        TEST_WITH_SLOW_GRADCHECK,
        "Failing in slow gradcheck on cuda12.8, see https://github.com/blacksmith/blacksmith/pull/156731 for example",
    )
    def test_cpp_wrapper(self):
        mod = ConvBN(3, 32, kernel_size=3, stride=2).eval().to(self.device)

        x = smith.rand(3, 3, 32, 32).to(self.device)

        @smith.compile(options={"cpp_wrapper": True})
        def foo(mod, x):
            return mod(x)

        out_eager = mod(x)

        with smith.no_grad():
            self.assertEqual(foo(mod, x), out_eager)
            self.assertEqual(foo(mod, x), out_eager)

    @tf32_on_and_off(0.001)
    def test_conv_layout_convert_with_view(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.bn(x)
                x = self.conv(x)
                return smith.flatten(x, 1)

        mod = Model().to(self.device).eval()

        @smith.compile()
        def foo(mod, inp):
            return mod(inp)

        with smith.no_grad():
            x = smith.rand(2, 3, 5, 5).to(self.device)
            mod_eager = mod(x)
            self.assertEqual(foo(mod, x), mod_eager)

    @smith._inductor.config.patch(layout_optimization=True)
    def test_conv_weight_layout_convert(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            def forward(self, x):
                return self.conv(x)

            @staticmethod
            def get_example_inputs():
                return (smith.rand(2, 3, 5, 5).to(self.device),)

        from smith._inductor.compile_fx import compile_fx, compile_fx_inner

        nconv = 0

        def my_inner_compile(gm, example_inputs, *args, **kwargs):
            out = compile_fx_inner(gm, example_inputs, *args, **kwargs)

            nonlocal nconv
            convs = [n for n in gm.graph.nodes if n.target == aten.convolution.default]
            nconv += len(convs)
            for conv in convs:
                weight_node = conv.args[1]
                weight_const_tensor = getattr(gm, weight_node.target)
                self.assertTrue(
                    weight_const_tensor.is_contiguous(memory_format=smith.channels_last)
                )
                self.assertTrue(
                    weight_node.meta["val"].is_contiguous(
                        memory_format=smith.channels_last
                    )
                )

            return out

        mod = smith.compile(
            Model().eval().to(self.device),
            backend=functools.partial(compile_fx, inner_compile=my_inner_compile),
        )
        inp = mod.get_example_inputs()
        with smith.no_grad():
            mod(*inp)

        # Only check the assertion for CUDA.
        # For CPU, we may get smith.ops.mkldnn._convolution_pointwise.default
        # in the joint graph rather than smith.ops.aten.convolution.default.
        # Currently we only handle aten.convolution.default in layout
        # optimization. That's why the count may be 0 here for CPU.
        if self.device == "cuda":
            self.assertTrue(nconv == 1)

    def test_unequal_bias_horizontal_addmm_fusion(self):
        device = self.device

        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w1 = smith.tensor(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=device
                )
                self.b1 = smith.zeros(3, device=device)
                self.w2 = smith.tensor(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device
                )
                self.b2 = smith.tensor([[-1.0, -1.0, -1.0]], device=device)
                self.w3 = smith.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
                )
                self.b3 = smith.tensor([1.0, 2.0, 3.0], device=device)

            def forward(self, x):
                out1 = smith.nn.functional.linear(x, self.w1, self.b1)
                out2 = smith.nn.functional.linear(x, self.w2, self.b2)
                out3 = smith.nn.functional.linear(x, self.w3, self.b3)
                return (out1, out2, out3)

        func = Model().to(device).eval()
        x = smith.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)

        with smith.no_grad():
            out_eager = func(x.clone())

            func1 = smith.compile(func)
            out_compiled = func1(x.clone())
            self.assertEqual(out_eager, out_compiled)

    @tf32_on_and_off(0.001)
    @smith._inductor.config.patch(layout_optimization=True)
    def test_redundant_clone_for_layout_convert(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            def forward(self, x):
                y = x + 1
                return self.conv(x), y

            @staticmethod
            def get_example_inputs():
                return (smith.rand(2, 3, 5, 5).to(self.device),)

        mod = Model().eval().to(self.device)
        inp = mod.get_example_inputs()
        with smith.no_grad():
            expected_outputs = mod(*inp)

        num_same_stride = 0
        num_diff_stride = 0

        def debug_inductor_force_stride_order(orig_fn, input_tensor, stride):
            nonlocal num_same_stride, num_diff_stride
            input_tensor.realize()
            if tuple(input_tensor.get_stride()) == tuple(stride):
                num_same_stride += 1
            else:
                num_diff_stride += 1
            return orig_fn(input_tensor, stride)

        with override_lowering(
            prims.inductor_force_stride_order.default, debug_inductor_force_stride_order
        ):
            opt_mod = smith.compile(mod)
            with smith.no_grad():
                actual_outputs = opt_mod(*inp)

        self.assertEqual(len(actual_outputs), len(expected_outputs))
        self.assertEqual(2, len(actual_outputs))
        for actual, expected in zip(actual_outputs, expected_outputs):
            self.assertEqual(expected, actual)

        if self.device == "cpu":
            # CPU use different convolution implementation, skip the checks below
            return

        self.assertTrue(
            actual_outputs[0].is_contiguous(memory_format=smith.contiguous_format)
        )
        self.assertTrue(
            actual_outputs[1].is_contiguous(memory_format=smith.contiguous_format)
        )

        # we don't change the stride of y returned by forward. So there will
        # be no extra copy
        self.assertTrue(num_same_stride == 1, f"num_same_stride is {num_same_stride}")
        # we changed the stride of self.conv(x) returned by forward. So there
        # may be an extra copy
        self.assertTrue(num_diff_stride == 1, f"num_diff_stride is {num_diff_stride}")


if TEST_WITH_ROCM:
    smith._inductor.config.force_layout_optimization = 1
    os.environ["BLACKSMITH_MIOPEN_SUGGEST_NHWC"] = "1"

if HAS_CPU and not smith.backends.mps.is_available():

    class FreezingCpuTests(TestCase):
        common = check_model
        device = "cpu"
        autocast = smith.cpu.amp.autocast

    copy_tests(OptimizeForInferenceTemplate, FreezingCpuTests, "cpu")

if HAS_GPU:

    class FreezingGpuTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(OptimizeForInferenceTemplate, FreezingGpuTests, GPU_TYPE)


del OptimizeForInferenceTemplate


if __name__ == "__main__":
    from smith._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
