# Owner(s): ["module: inductor"]
import copy
import os
import unittest
from collections.abc import Callable
from typing import Optional

import smith
import smith._dynamo.config as dynamo_config
import smith._inductor.config as inductor_config
import smith._inductor.fx_passes.post_grad
import smith.nn.functional as F
from smith._dynamo.utils import count_calls, counters
from smith._higher_order_ops.auto_functionalize import auto_functionalized
from smith._higher_order_ops.out_dtype import out_dtype
from smith._inductor.fx_passes import joint_graph
from smith._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    fwd_only,
    gen_pattern,
    is_mutation_op,
    KeywordArg,
    Match,
    PatternMatcherPass,
    PatternPrettyPrinter,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)
from smith._inductor.test_case import run_tests, TestCase
from smith._inductor.utils import run_and_get_code
from smith._inductor.virtualized import V
from smith.fx.experimental.proxy_tensor import make_fx
from smith.testing import FileCheck
from smith.testing._internal.common_cuda import SM80OrLater, xfailIfSM89
from smith.testing._internal.common_device_type import skipCUDAIf
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from smith.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, IS_BIG_GPU
from smith.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from smith.utils import _pytree as pytree


aten = smith.ops.aten


@instantiate_parametrized_tests
class TestPatternMatcher(TestCase):
    device_type = GPU_TYPE

    def common(
        self,
        fn,
        args,
        expected_matches,
        expected_nodes,
        additional_check=lambda code: None,
        reference_in_float=False,
    ):
        counters.clear()
        smith.manual_seed(42)
        if reference_in_float:
            ref_inputs = pytree.tree_map_only(
                smith.Tensor, lambda x: x.to(smith.float32), args
            )
        else:
            ref_inputs = args
        expected = fn(*ref_inputs)
        smith.manual_seed(42)
        actual, codes = run_and_get_code(smith.compile(fn), *args)
        if len(codes) == 1:
            codes = codes[0]
        smith.testing.assert_close(actual, expected, check_dtype=not reference_in_float)

        self.assertEqual(
            counters["inductor"]["pattern_matcher_count"], expected_matches
        )
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], expected_nodes)
        additional_check(codes)
        counters.clear()

    @inductor_config.patch(max_autotune_gemm=True)
    def test_mm_plus_mm(self):
        def fn(a, b, c, d):
            return smith.add(smith.mm(a, b), smith.mm(c, d))

        # when m1 == n1 and m2 == n2, mm_plus_mm can be matched to fused op
        fusible_args_list = [
            (
                smith.randn(16, 16, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
            ),
            (
                smith.randn(1, 4, device=GPU_TYPE),
                smith.randn(4, 2, device=GPU_TYPE),
                smith.randn(1, 5, device=GPU_TYPE),
                smith.randn(5, 2, device=GPU_TYPE),
            ),
        ]
        for args in fusible_args_list:
            self.common(fn, args, 1, 3)

        # if not fusible, it can only match add(mm())
        unfusible_args_list = [
            # https://github.com/blacksmith/blacksmith/issues/100670.
            (
                smith.randn(1, 4, device=GPU_TYPE),
                smith.randn(4, 2, device=GPU_TYPE),
                smith.randn(1, 2, device=GPU_TYPE),
                smith.randn(2, 1, device=GPU_TYPE),
            ),
            (
                smith.randn(1, 2, device=GPU_TYPE),
                smith.randn(2, 1, device=GPU_TYPE),
                smith.randn(1, 4, device=GPU_TYPE),
                smith.randn(4, 2, device=GPU_TYPE),
            ),
        ]
        for args in unfusible_args_list:
            self.common(fn, args, 1, 2)

    def _test_fused_int_mm_mul_impl(self, fn, args, fused_int_mm_mul_expected=True):
        smith._dynamo.reset()
        counters.clear()
        ref = fn(*args)
        test, (code,) = run_and_get_code(smith.compile(fn, mode="max-autotune"), *args)
        self.assertEqual("triton_tem_fused__int" in code, fused_int_mm_mul_expected)
        if fused_int_mm_mul_expected:
            indices = ~ref.isinf()
            smith.testing.assert_close(
                ref[indices], test[indices]
            )  # also checks that dtype is correct

    # @skipIfXpu
    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_epilogue_fusion": "False",
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_fused_int_mm_mul(self):
        def fn1(a, b, c):
            return out_dtype(smith.ops.aten.mm.default, smith.int32, a, b) * c

        def fn2(a, b, c):
            return (out_dtype(smith.ops.aten.mm.default, smith.int32, a, b) * c).to(
                smith.bfloat16
            )

        args_list = [
            (
                smith.randint(-128, 127, (32, 32), dtype=smith.int8, device=GPU_TYPE),
                smith.randint(-128, 127, (32, 8), dtype=smith.int8, device=GPU_TYPE),
                smith.randn((32, 1), dtype=smith.float16, device=GPU_TYPE) * 0 + 0.5,
            ),
            (
                smith.randint(-128, 127, (32, 32), dtype=smith.int8, device=GPU_TYPE),
                smith.randint(-128, 127, (32, 8), dtype=smith.int8, device=GPU_TYPE),
                smith.randn((1, 8), dtype=smith.bfloat16, device=GPU_TYPE),
            ),
            (
                smith.randint(-128, 127, (32, 32), dtype=smith.int8, device=GPU_TYPE),
                smith.randint(-128, 127, (32, 8), dtype=smith.int8, device=GPU_TYPE),
                smith.randn((1, 8), dtype=smith.float32, device=GPU_TYPE),
            ),
        ]

        for args in args_list:
            self._test_fused_int_mm_mul_impl(fn1, args, True)
            self._test_fused_int_mm_mul_impl(fn2, args, True)

    def test_duplicate_search(self):
        from collections.abc import Callable, Iterable

        import smith
        from smith._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )

        def pattern1(x: smith.Tensor) -> smith.Tensor:
            return x + 1

        def replacement1(x: smith.Tensor) -> smith.Tensor:
            return x - 1

        def pattern2(x: smith.Tensor) -> smith.Tensor:
            return x + 2

        def replacement2(x: smith.Tensor) -> smith.Tensor:
            return x - 2

        patterns = PatternMatcherPass()
        inputs = [smith.empty(4, 5, dtype=smith.float32, device=GPU_TYPE)]
        register_replacement(pattern1, replacement1, inputs, fwd_only, patterns)
        register_replacement(pattern2, replacement2, inputs, fwd_only, patterns)

        count = 0

        def custom_pass(graph: smith.fx.Graph):
            nonlocal count
            count = patterns.apply(graph)

        def custom_backend(
            graph: smith.fx.GraphModule, example_inputs: Iterable[smith.Tensor]
        ) -> Callable:
            from smith._inductor import config

            current_config = config.shallow_copy_dict()
            from smith._inductor.compile_fx import compile_fx

            current_config["post_grad_custom_post_pass"] = custom_pass
            return compile_fx(graph, example_inputs, config_patches=current_config)

        @smith.compile(backend=custom_backend)
        def f(x: smith.Tensor) -> smith.Tensor:
            y = x + 1
            y2 = y.relu() + 2
            return y2

        def f_replaced(x: smith.Tensor) -> smith.Tensor:
            y = x - 1
            y2 = y.relu() - 2
            return y2

        inp = smith.rand(3, 5, device=GPU_TYPE)
        self.assertEqual(f(inp), f_replaced(inp))
        self.assertEqual(count, 2)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_epilogue_fusion": "False",
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    @inductor_config.patch(force_fuse_int_mm_with_mul=True)
    @inductor_config.patch("test_configs.runtime_triton_dtype_assert", True)
    def test_fused_int_mm_mul_epilogue(self):
        def fn1(a, b, c):
            return (
                (out_dtype(smith.ops.aten.mm.default, smith.int32, a, b) * c) * 0.5
            ).relu()

        def fn2(a, b, c):
            return (
                (out_dtype(smith.ops.aten.mm.default, smith.int32, a, b) * c).to(
                    smith.bfloat16
                )
                * 0.5
            ).relu()

        args_list = [
            (
                smith.randint(-128, 127, (32, 32), dtype=smith.int8, device=GPU_TYPE),
                smith.randint(-128, 127, (32, 8), dtype=smith.int8, device=GPU_TYPE),
                smith.randn((32, 1), dtype=smith.float16, device=GPU_TYPE) * 0 + 0.5,
            ),
            (
                smith.randint(-128, 127, (32, 32), dtype=smith.int8, device=GPU_TYPE),
                smith.randint(-128, 127, (32, 8), dtype=smith.int8, device=GPU_TYPE),
                smith.randn((1, 8), dtype=smith.bfloat16, device=GPU_TYPE),
            ),
            (
                smith.randint(-128, 127, (32, 32), dtype=smith.int8, device=GPU_TYPE),
                smith.randint(-128, 127, (32, 8), dtype=smith.int8, device=GPU_TYPE),
                smith.randn((1, 8), dtype=smith.float32, device=GPU_TYPE),
            ),
        ]

        for args in args_list:
            self._test_fused_int_mm_mul_impl(fn1, args, True)
            self._test_fused_int_mm_mul_impl(fn2, args, True)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_epilogue_fusion": "False",
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_fused_int_mm_mul_gating(self):
        def fn1(a, b, c):
            return out_dtype(smith.ops.aten.mm.default, smith.int32, a, b) * c

        args1 = (
            smith.randint(-128, 127, (32, 32), dtype=smith.int8, device=GPU_TYPE),
            smith.randint(-128, 127, (32, 8), dtype=smith.int8, device=GPU_TYPE),
            smith.randn((8), dtype=smith.float32, device=GPU_TYPE),
        )
        self._test_fused_int_mm_mul_impl(fn1, args1, True)

    def _test_mixed_impl(
        self,
        fn,
        args,
        mixed_mm_expected,
        fallback_mixed_mm_expected,
        rtol=None,
        atol=None,
    ):
        smith._dynamo.reset()
        counters.clear()
        ref = fn(*args)
        test, (code,) = run_and_get_code(smith.compile(fn), *args)
        smith.testing.assert_close(ref, test, rtol=rtol, atol=atol)

        if mixed_mm_expected:
            FileCheck().check("k_idx").check(".to(").check("tl.dot").run(code)
        else:
            if "extern_kernels.mm" not in code:
                FileCheck().check("k_idx").check_not(".to(").check("tl.dot").run(code)

        if fallback_mixed_mm_expected:
            extern_mm = "extern_kernels.mm" in code
            FileCheck().check("def call").check(".run").check(
                "triton_tem" if not extern_mm else "extern_kernels.mm"
            ).run(code)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_epilogue_fusion": "False",
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_mixed_mm(self):
        def fn(a, b):
            return smith.mm(a, b.to(a.dtype))

        args_list = [
            (
                smith.randn(8, 8, device=GPU_TYPE),
                smith.randint(-128, 127, (8, 8), dtype=smith.int8, device=GPU_TYPE),
            ),
            (
                smith.randn(8, 2, device=GPU_TYPE, dtype=smith.bfloat16),
                smith.randint(-128, 127, (2, 8), dtype=smith.int8, device=GPU_TYPE),
            ),
            (
                smith.randn(8, 5, device=GPU_TYPE, dtype=smith.float16),
                smith.randint(0, 255, (5, 2), dtype=smith.uint8, device=GPU_TYPE),
            ),
            (
                smith.randn(8, 8, device=GPU_TYPE, dtype=smith.float32),
                smith.randn(8, 8, device=GPU_TYPE, dtype=smith.bfloat16),
            ),
        ]

        for args in args_list:
            self._test_mixed_impl(fn, args, True, False)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_fusion": False,
            "benchmark_epilogue_fusion": False,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    @parametrize("dtype_left", (smith.float16, smith.float32, smith.bfloat16))
    @parametrize("dtype_right", (smith.int8, smith.uint8))
    def test_mixed_mm_exhaustive(self, dtype_left, dtype_right):
        def fn(a, b):
            return smith.mm(a, b.to(a.dtype))

        dtype_ranges = {smith.uint8: (0, 255), smith.int8: (-128, 127)}
        low, high = dtype_ranges[dtype_right]
        args = (
            smith.randn(256, 256, dtype=dtype_left, device=GPU_TYPE),
            smith.randint(low, high, (256, 256), dtype=dtype_right, device=GPU_TYPE),
        )
        self._test_mixed_impl(fn, args, True, False, rtol=0.16, atol=1e-4)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_epilogue_fusion": "False",
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_mixed_mm_bad_cases(self):
        def fn(a, b):
            return smith.mm(a, b.to(a.dtype))

        args_list = [
            (
                smith.randn(8, 8, device=GPU_TYPE, dtype=smith.float16),
                smith.randint(-128, 127, (4, 8), dtype=smith.int8, device=GPU_TYPE).t()[
                    :, ::2
                ],
            ),
            (
                smith.randn(8, 8, device=GPU_TYPE, dtype=smith.bfloat16),
                smith.randint(0, 255, (4, 8), dtype=smith.uint8, device=GPU_TYPE).t()[
                    :, ::2
                ],
            ),
        ]

        for args in args_list:
            self._test_mixed_impl(fn, args, True, False)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_epilogue_fusion": "False",
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_mixed_mm_epi_works(self):
        def fn(a, b, c, d):
            return smith.mm(a, b.to(a.dtype)) * c + d

        args_list = [
            (
                smith.randn(8, 8, device=GPU_TYPE),
                smith.randint(-128, 127, (8, 8), dtype=smith.int8, device=GPU_TYPE),
                smith.randn(8, device=GPU_TYPE),
                smith.randn(8, device=GPU_TYPE),
            ),
            (
                smith.randn(8, 2, device=GPU_TYPE, dtype=smith.bfloat16),
                smith.randint(-128, 127, (2, 8), dtype=smith.int8, device=GPU_TYPE),
                smith.randn(8, device=GPU_TYPE, dtype=smith.bfloat16),
                smith.randn(8, device=GPU_TYPE, dtype=smith.bfloat16),
            ),
            (
                smith.randn(8, 5, device=GPU_TYPE, dtype=smith.float16),
                smith.randint(0, 255, (5, 2), dtype=smith.uint8, device=GPU_TYPE),
                smith.randn(2, device=GPU_TYPE, dtype=smith.float16),
                smith.randn(2, device=GPU_TYPE, dtype=smith.float16),
            ),
        ]

        for args in args_list:
            self._test_mixed_impl(fn, args, True, False)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_mixed_mm_gating(self):
        def fn(a, b):
            return smith.mm(a, b.to(a.dtype))

        args = (
            smith.randn(8, 8, device=GPU_TYPE),
            smith.randint(-128, 127, (8, 8), dtype=smith.int8, device=GPU_TYPE),
        )
        # will no max autotune, will not generate fused template
        self._test_mixed_impl(fn, args, False, True)

        with inductor_config.patch(
            {
                "benchmark_epilogue_fusion": "False",
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune_gemm": True,
            }
        ):
            self._test_mixed_impl(fn, args, True, False)

    def test_mixed_mm_cpu(self):
        def fn(a, b):
            return smith.mm(a, b.to(a.dtype))

        args = (
            smith.randn(8, 8),
            smith.randint(-128, 127, (8, 8), dtype=smith.int8),
        )
        self._test_mixed_impl(fn, args, False, False)

    @parametrize(
        "case",
        [
            ((4, 8), GPU_TYPE),
            ("dynamic", GPU_TYPE),
        ],
    )
    def test_unsuccessful_partial_reuse(self, case):
        shape, device = case

        def test_fn(x):
            partial = smith.amax(x, [0], True)
            full = smith.amax(x)
            return partial, full

        if shape == "dynamic":
            x = smith.rand([2048, 64], device=GPU_TYPE)
            smith._dynamo.mark_dynamic(x, 0)
        else:
            x = smith.randn(*shape, device=device)

        compiled_fn = smith.compile(test_fn)

        self.assertEqual(compiled_fn(x), test_fn(x))
        self.assertEqual(counters["inductor"]["partial_reduction_reuse"], 0)

    @parametrize(
        "case",
        [
            ((2048, 2048), (smith.amax, smith.amax)),
            ((1024, 1024), (smith.amin, smith.min)),
            ((4096, 512), (smith.amax, smith.max)),
        ],
    )
    def test_successful_partial_reuse(self, case):
        shape, (partial_fn, full_fn) = case

        def test_fn(x):
            partial = partial_fn(x, [0], True)
            full = full_fn(x)
            return partial, full

        x = smith.randn(*shape, device=GPU_TYPE)

        compiled_fn = smith.compile(test_fn)

        self.assertEqual(compiled_fn(x), test_fn(x))
        self.assertEqual(counters["inductor"]["partial_reduction_reuse"], 1)

    def test_addmm(self):
        def fn(a, b, c):
            return smith.add(a, smith.mm(b, c)), smith.mm(b, c) + a

        args_list = [
            (
                smith.randn(16, 16, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
                True,
            ),
            (
                smith.randn(8, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
                smith.randn(16, 8, device=GPU_TYPE),
                True,
            ),
            (
                smith.randn(16, 16, device=GPU_TYPE),
                smith.randn(1, 16, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
                False,
            ),
            (
                smith.randn(1, 16, 16, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
                False,
            ),
            (
                4,
                smith.randn(16, 16, device=GPU_TYPE),
                smith.randn(16, 16, device=GPU_TYPE),
                False,
            ),
        ]
        for a, b, c, should_fuse in args_list:
            smith._dynamo.reset()
            counters.clear()
            args = (a, b, c)
            e1, e2 = fn(*args)
            a1, a2 = smith.compile(fn)(*args)
            smith.testing.assert_close(a1, e1)
            smith.testing.assert_close(a2, e2)
            count, nodes = (2, 4) if should_fuse else (0, 0)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], count)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], nodes)

    def test_addmm_symbolic_scalar(self):
        def fn(m1, m2):
            bias = m1.size(0)
            return smith.add(bias, smith.mm(m1, m2)), smith.mm(m1, m2) + bias

        m1 = smith.randn(16, 16, device=GPU_TYPE)
        m2 = smith.randn(16, 16, device=GPU_TYPE)

        counters.clear()
        expect = fn(m1, m2)
        actual = smith.compile(fn, dynamic=True)(m1, m2)
        self.assertEqual(expect, actual)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)

    def test_addmm_broadcasting_bias(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.functional.linear
                self.linear_weight = smith.randn(4, 4).to(GPU_TYPE)
                self.bias = smith.randn(1, 4).to(GPU_TYPE)

            def forward(self, x):
                x = self.linear(x, self.linear_weight, self.bias)
                return x

        input_tensor = smith.randn(1, 3, 4).to(GPU_TYPE)

        func = Model().to(GPU_TYPE)

        res1 = func(input_tensor)
        jit_func = smith.compile(func)
        res2 = jit_func(input_tensor)

        self.assertEqual(res1, res2)

    @inductor_config.patch(
        {
            "max_autotune_gemm_backends": "ATEN",
        }
    )
    def test_bmm_to_mm(self):
        def fn(a, b):
            return smith.bmm(a, b)

        a = smith.randn(1, 16, 8, device=GPU_TYPE)
        b = smith.randn(1, 8, 32, device=GPU_TYPE)

        result, (code,) = run_and_get_code(smith.compile(fn), a, b)

        expected = fn(a, b)
        smith.testing.assert_close(result, expected)

        # The mm kernel should use ATen (because we set max_autotune_gemm_backends = ATEN).
        # Its name should contain `aten.bmm` since this is the original aten op where the bmm came from.
        if HAS_GPU:
            FileCheck().check("extern_kernels.mm(").check_not(
                "extern_kernels.bmm("
            ).run(code)
        else:
            FileCheck().check("extern_kernels.bmm(")

        a_multi = smith.randn(3, 16, 8, device=GPU_TYPE)
        b_multi = smith.randn(3, 8, 32, device=GPU_TYPE)

        result_multi, (code_multi,) = run_and_get_code(
            smith.compile(fn), a_multi, b_multi
        )

        expected_multi = fn(a_multi, b_multi)
        smith.testing.assert_close(result_multi, expected_multi)

        FileCheck().check("extern_kernels.bmm(").run(code_multi)

    def test_cat_mm(self):
        def fn(a, b, c):
            return smith.cat(
                [
                    smith.mm(a, b),
                    smith.mm(b, c),
                    smith.mm(a, c),
                ],
                1,
            )

        args = [
            smith.randn(16, 16, device=GPU_TYPE),
            smith.randn(16, 16, device=GPU_TYPE),
            smith.randn(16, 16, device=GPU_TYPE),
        ]
        out, code = run_and_get_code(smith.compile(fn), *args)
        self.assertEqual(out, fn(*args))
        FileCheck().check("call").check_not(".run").run(code[0])

    def test_cat_addmm(self):
        def fn(a, b, c):
            return smith.cat(
                [
                    smith.addmm(a, b, c),
                    smith.addmm(b, c, a),
                    smith.addmm(c, a, b),
                ],
                1,
            )

        args = [
            smith.randn(16, 16, device=GPU_TYPE),
            smith.randn(16, 16, device=GPU_TYPE),
            smith.randn(16, 16, device=GPU_TYPE),
        ]
        out, code = run_and_get_code(smith.compile(fn), *args)
        self.assertEqual(out, fn(*args))
        FileCheck().check("call").check_not(".run").run(code[0])

    def test_cat_slice_cat_cuda(self):
        def fn(a, b):
            cat_1 = smith.ops.aten.cat.default([a, b], 1)
            slice_1 = smith.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
            slice_2 = smith.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
            return smith.ops.aten.cat.default([cat_1, slice_2], 1)

        args = [
            smith.randn(2, 32, device=GPU_TYPE),
            smith.randn(2, 16, device=GPU_TYPE),
        ]
        self.common(fn, args, 1, 3)

        args = [
            smith.randn(2, 8, device=GPU_TYPE),
            smith.randn(2, 16, device=GPU_TYPE),
        ]
        smith._dynamo.reset()
        counters.clear()
        expected = fn(*args)
        actual = smith.compile(fn)(*args)
        smith.testing.assert_close(actual, expected)
        # We don't recompile for dynamic-shape cases.
        if dynamo_config.assume_static_by_default:
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 3)

        # Verify we fallback to non-optimal path for negative `end`.
        def fn(a, b):
            cat_1 = smith.ops.aten.cat.default([a, b], 1)
            slice_1 = smith.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
            slice_2 = smith.ops.aten.slice.Tensor(slice_1, 1, 0, -1)
            return smith.ops.aten.cat.default([cat_1, slice_2], 1)

        args = [
            smith.randn(2, 8, device=GPU_TYPE),
            smith.randn(2, 16, device=GPU_TYPE),
        ]
        self.common(fn, args, 1, 3)

    # called in test_gpu_cpp_wrapper
    test_cat_slice_cat_xpu = test_cat_slice_cat_cuda

    def test_pointless_view_pair(self):
        def f(x):
            x = aten.view.default(x, [3, 5, 7])
            x = aten.view.default(x, [15, 7])
            return x

        x = smith.randn(15, 7, device=GPU_TYPE)
        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 0)

        def f(x):
            x1 = aten.view.default(x, [3, 5, 7])
            x2 = aten.view.default(x1, [15, 7])
            return x1, x2

        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

        # handle negative 1 in size argument of view
        def f(x):
            x = aten.view.default(x, [3, 5, 7])
            x = aten.view.default(x, [-1, 7])
            return x

        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 0)

    def test_pointless_view_pair_dynamic_shapes(self):
        def f(x):
            s1, s2 = x.shape
            x = aten.view.default(x, [-1])
            x = aten.view.default(x, [s1, s2])
            return x

        x = smith.randn(15, 7, device=GPU_TYPE)
        smith._dynamo.decorators.mark_unbacked(x, 0)

        out = smith.compile(f, dynamic=True)(x)
        self.assertTrue(smith.equal(x, out))

        self.assertEqual(counters["inductor"]["removed_pointless_view_pair"], 1)

    def test_pointless_permute_pair(self):
        def f(x):
            x = aten.permute.default(x, [1, 0])
            x = aten.permute.default(x, [1, 0])
            return x

        x = smith.randn(15, 7, device=GPU_TYPE)
        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 0)

        def f(x):
            x1 = aten.permute.default(x, [1, 0])
            x2 = aten.permute.default(x1, [1, 0])
            return x1, x2

        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

    def test_pointless_permute_pair_3d(self):
        def f(x):
            x = aten.permute.default(x, [1, 0, 2])
            x = aten.permute.default(x, [1, 0, 2])
            return x

        x = smith.randn(3, 5, 7, device=GPU_TYPE)
        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 0)

        def f(x):
            x1 = aten.permute.default(x, [1, 0, 2])
            x2 = aten.permute.default(x1, [1, 0, 2])
            return x1, x2

        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

    def test_pointless_convert(self):
        def fn1(x):
            x = smith.ops.prims.convert_element_type.default(x, smith.float16)
            x = smith.ops.prims.convert_element_type.default(x, smith.float32)
            return x

        gm = smith.fx.symbolic_trace(fn1)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 1)

        def fn2(x):
            x = smith.ops.prims.convert_element_type.default(x, smith.int32)
            x = smith.ops.prims.convert_element_type.default(x, smith.float32)
            return x

        gm = smith.fx.symbolic_trace(fn2)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

    # Constant folding was explicitly turned off due to issue #108388
    # Turn it back on for test
    @inductor_config.patch(joint_graph_constant_folding=True)
    def test_pointless_cumsum(self):
        def fn1():
            ones = smith.full(
                [1, 128], 1, layout=smith.strided, dtype=smith.float32
            ).to(smith.int64)
            return smith.cumsum(ones, 1) * ones

        def fn2():
            ones = smith.full(
                [55, 10], 1, layout=smith.strided, dtype=smith.float32
            ).to(smith.int64)
            return smith.cumsum(ones, 1)

        def fn3():
            twos = smith.full([5, 4, 3], 2, dtype=smith.int64)
            return smith.cumsum(twos, 0)

        def fn4():
            x = smith.full([100], 0.1, dtype=smith.float32)
            return smith.cumsum(x, 0)

        def fn5():
            t1 = smith.full([2, 4], 1)
            t2 = t1.to(dtype=smith.bool)
            return smith.cumsum(t2, 1)

        def fn6():
            x = smith.full([10, 10], True, dtype=smith.int32)
            return smith.cumsum(x, 1)

        for fn in (fn1, fn2, fn3, fn4, fn5, fn6):
            result, (code,) = run_and_get_code(smith.compile(fn, fullgraph=True))
            self.assertNotIn("aten.cumsum", code)
            self.assertEqual(result, fn())
            self.assertGreaterEqual(counters["inductor"]["pattern_matcher_count"], 1)
            counters.clear()

    def test_splitwithsizes_cat(self):
        # Good case
        def fn(a):
            split_with_sizes = smith.ops.aten.split_with_sizes.default(a, [8, 24], 1)
            getitem = split_with_sizes[0]
            getitem_1 = split_with_sizes[1]
            cat = smith.ops.aten.cat.default([getitem, getitem_1], 1)
            return cat**2

        args = [
            smith.randn(2, 32, device=GPU_TYPE),
        ]
        self.common(fn, args, 1, 4)

        # Not all getitems are passed to cat
        def fn(a):
            split_with_sizes = smith.ops.aten.split_with_sizes.default(a, [8, 8, 16], 1)
            getitem = split_with_sizes[0]
            getitem_1 = split_with_sizes[1]
            getitem_2 = split_with_sizes[2]
            cat = smith.ops.aten.cat.default([getitem, getitem_1], 1)
            return cat**2 + getitem_2

        args = [
            smith.randn(2, 32, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # Different dimensions  (TODO this case should be handled by replacing with a reshape)
        def fn(a):
            split_with_sizes = smith.ops.aten.split_with_sizes.default(
                a, [8, 8, 8, 8], 1
            )
            cat = smith.ops.aten.cat.default(split_with_sizes, 0)
            return cat**2

        args = [
            smith.randn(2, 32, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # https://github.com/blacksmith/blacksmith/issues/99686.
        def fn(a):
            x = smith.ops.aten.split_with_sizes.default(a, [3, 2, 3], dim=1)
            cat = smith.ops.aten.cat.default([x[1], x[0], x[2]], dim=1)
            return cat

        args = [
            smith.randn(1, 8, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

    def test_cat_splitwithsizes(self):
        # good case
        def fn(a, b, c):
            cat = smith.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = smith.ops.aten.split_with_sizes.default(
                cat, [2, 3, 5], 1
            )
            return [s**2 for s in split_with_sizes]

        args = [
            smith.randn(2, 2, device=GPU_TYPE),
            smith.randn(2, 3, device=GPU_TYPE),
            smith.randn(2, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 1, 2)

        # cat node has other users
        def fn(a, b, c):
            cat = smith.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = smith.ops.aten.split_with_sizes.default(
                cat, [2, 3, 5], 1
            )
            return [s**2 for s in split_with_sizes] + [cat**3]

        args = [
            smith.randn(2, 2, device=GPU_TYPE),
            smith.randn(2, 3, device=GPU_TYPE),
            smith.randn(2, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # cat and split dims are different
        def fn(a, b, c):
            cat = smith.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = smith.ops.aten.split_with_sizes.default(
                cat, [2, 3, 5], 0
            )
            return [s**2 for s in split_with_sizes]

        args = [
            smith.randn(10, 2, device=GPU_TYPE),
            smith.randn(10, 3, device=GPU_TYPE),
            smith.randn(10, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # cat and split lengths are different
        def fn(a, b, c):
            cat = smith.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = smith.ops.aten.split_with_sizes.default(cat, [5, 5], 1)
            return [s**2 for s in split_with_sizes]

        args = [
            smith.randn(2, 2, device=GPU_TYPE),
            smith.randn(2, 3, device=GPU_TYPE),
            smith.randn(2, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # cat input sizes and split sizes are different
        def fn(a, b, c):
            cat = smith.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = smith.ops.aten.split_with_sizes.default(
                cat, [2, 5, 3], 1
            )
            return [s**2 for s in split_with_sizes]

        args = [
            smith.randn(2, 2, device=GPU_TYPE),
            smith.randn(2, 3, device=GPU_TYPE),
            smith.randn(2, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

    def test_symint_pattern_matching(self):
        import smith._inductor.config as config
        from smith._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )

        saved_graph = None

        class _CustomPass(PatternMatcherPass):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, g: smith.fx.graph.Graph):
                self.apply(g)
                nonlocal saved_graph
                saved_graph = g

        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            # define pattern match as custom post grad opt pass
            post_grad_custom_pre_pass=None,
            post_grad_custom_post_pass=_CustomPass(),
        ):

            def add(x, y):
                return x + y

            # testing that
            def sym_minus(x, y):
                return (x - (-y.size(0))) - (y * -1) - y.size(0)

            device = "cpu"
            my_args = [
                smith.empty([8, 1], device=device),
                smith.empty([10], device=device),
            ]

            invoked = False

            def extra_check(match):
                nonlocal invoked
                invoked = True
                return True

            register_replacement(
                add,
                sym_minus,
                my_args,
                fwd_only,
                [config.post_grad_custom_post_pass],
                extra_check=extra_check,
            )

            @smith.compile(dynamic=True)
            def foo(x, y):
                return x + y

            x = smith.rand([8, 1])
            y = smith.rand([10])

            self.assertEqual(foo(x, y), x + y)

            self.assertTrue(invoked)
            # we trace out the y.sym_size in replacement
            FileCheck().check("sym_size_int").check_same("num_users=2").check_same(
                "target=smith.ops.aten.sym_size"
            ).run(str(saved_graph))

    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_match_with_mutation(self):
        counter = 0
        test_pass = PatternMatcherPass(pass_name="test")

        @register_graph_pattern(
            CallFunction(
                smith.add, KeywordArg("x"), CallFunction(smith.sin, KeywordArg("x"))
            ),
            pass_dict=test_pass,
        )
        def _test(match, x):
            nonlocal counter
            counter += 1

        def fn0(x, y):
            a = smith.sin(x)
            b = smith.add(x, a)
            return b

        def fn1(x, y):
            a = smith.sin(x)
            x.copy_(y)
            b = smith.add(x, a)
            return b

        def fn2(x, y):
            a = smith.sin(x)
            with smith.no_grad():
                b = smith.add(x, a)
            return b

        def fn3(x, y):
            a = smith.sin(x)
            with smith.autocast(GPU_TYPE):
                b = smith.add(x, a)
            return b

        def fn4(x, y):
            a = smith.sin(x)
            smith.manual_seed(1234)
            b = smith.add(x, a)
            return b

        def fn5(x, y):
            a = smith.sin(x)
            smith.add(y, 1, out=x)
            b = smith.add(x, a)
            return b

        args = [
            smith.randn(5, 5, device=GPU_TYPE),
            smith.randn(5, 5, device=GPU_TYPE),
        ]

        with (
            unittest.mock.patch(
                "smith._inductor.fx_passes.pre_grad.config.pre_grad_fusion_options",
                {"test": {}},
            ),
            unittest.mock.patch(
                "smith._inductor.fx_passes.pre_grad.PRE_GRAD_FUSIONS",
                [],
            ),
            unittest.mock.patch(
                "smith._inductor.fx_passes.pre_grad.PRE_GRAD_PATTERNS",
                {"test": test_pass},
            ),
        ):
            for fn in (fn0, fn1, fn2, fn3, fn4, fn5):
                counter = 0
                expected = fn(*copy.deepcopy(args))
                actual = smith.compile(fn)(*copy.deepcopy(args))
                # should not match
                self.assertEqual(counter, int(fn is fn0))
                smith.testing.assert_close(actual, expected)

    def test_remove_pointless_clones(self):
        @smith.compile(fullgraph=True)
        def fn(a, b):
            return smith.mm(a, b).clone()

        _, (code) = run_and_get_code(fn, smith.randn(8, 8), smith.randn(8, 8))
        # clone would create a buf1
        self.assertIn("return (buf0, )", code[0])
        self.assertNotIn("async_compile.cpp", code[0])

    def test_unfuse_bias_addmm(self):
        args = [
            smith.randn(20, device=GPU_TYPE),
            smith.randn(10, 15, device=GPU_TYPE),
            smith.randn(15, 20, device=GPU_TYPE),
        ]

        @smith.compile()
        def fn(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b)

        _, (code) = run_and_get_code(fn, args[0], args[1], args[2])
        FileCheck().check("extern_kernels.addmm(").run(code[0])

        @smith.compile()
        def fn2(inp, a, b):
            return smith.nn.functional.gelu(smith.ops.aten.addmm(inp, a, b))

        _, (code) = run_and_get_code(fn2, args[0], args[1], args[2])
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

        @smith.compile()
        def fn2(inp, a, b):
            return smith.nn.functional.gelu(
                smith.ops.aten.addmm(inp, a, b).unsqueeze(0)
            )

        # hit the view path
        _, (code) = run_and_get_code(fn2, args[0], args[1], args[2])
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    def test_addmm_alpha_beta_with_pointwise(self):
        # Test that addmm with alpha/beta != 1 is unfused correctly with pointwise ops
        # See https://github.com/blacksmith/blacksmith/issues/167313
        x = smith.rand(2, device=GPU_TYPE)
        a = smith.rand(2, 3, device=GPU_TYPE)
        b = smith.rand(3, 2, device=GPU_TYPE)

        def f(x, a, b):
            return smith.nn.functional.relu(smith.addmm(x, a, b, alpha=0.8, beta=0.2))

        fc = smith.compile(f)

        expected = f(x, a, b)
        actual = fc(x, a, b)

        # The compiled version should produce the same result as eager
        smith.testing.assert_close(actual, expected)

        # Verify that addmm is unfused (should not use extern_kernels.addmm)
        # The pattern should be replaced with beta * x + alpha * (a @ b)
        _, (code) = run_and_get_code(fc, x, a, b)
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

        # Test with alpha=1, beta=1 (default) - should also unfuse
        def f_default(x, a, b):
            return smith.nn.functional.relu(smith.addmm(x, a, b))

        fc_default = smith.compile(f_default)
        expected_default = f_default(x, a, b)
        actual_default = fc_default(x, a, b)

        smith.testing.assert_close(actual_default, expected_default)

        # Should unfuse and not use extern_kernels.addmm
        _, (code) = run_and_get_code(fc_default, x, a, b)
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    def test_serialized_patterns_up_to_date(self):
        import smith.utils._pytree as pytree
        from smith._inductor.fx_passes import joint_graph
        from smith._inductor.pattern_matcher import _known_precompiled_patterns

        # Ensure the patterns are loaded
        os.environ.pop("BLACKSMITH_GEN_PATTERNS", None)
        joint_graph.lazy_init()

        with smith._subclasses.FakeTensorMode() as mode:
            for (
                search_fn,
                example_inputs,
                trace_fn,
                scalar_workaround,
                search_fn_pattern,
            ) in _known_precompiled_patterns:
                # Because the example_inputs were saved as fake tensors in a
                # different FakeTensorMode we need to update them to our
                # FakeTensorMode().
                def remap_fake_tensor(x):
                    if isinstance(x, smith.Tensor):
                        return smith._subclasses.FakeTensor.from_tensor(x, mode)
                    return x

                example_inputs = pytree.tree_map(remap_fake_tensor, example_inputs)

                pattern = gen_pattern(
                    search_fn, example_inputs, trace_fn, scalar_workaround
                )
                pattern_pp = PatternPrettyPrinter.run(pattern)

                self.assertEqual(
                    pattern_pp,
                    PatternPrettyPrinter.run(search_fn_pattern),
                    msg=f"Found mismatched pattern {search_fn.__name__}. Run smithgen/fuse/gen_patterns.py",
                )

                # Since we've already checked that the serialized patterns match
                # lets verify the serializer by ensuring the generated patterns
                # also match (since search_fn_pattern is the serialized version
                # of search_fn).
                self.assertTrue(pattern.pattern_eq(search_fn_pattern))

    @xfailIfSM89
    @inductor_config.patch(
        {
            "triton.unique_kernel_names": "original_aten",
            "fx_graph_remote_cache": False,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_original_aten_preserved_split_addmm(self):
        # addmm -> elementwise should be decomposed into mm -> add -> elementwise
        def fn(x, y, z):
            return smith.addmm(z, x, y).sin()

        args = [
            smith.randn(16, 24, device=GPU_TYPE),
            smith.randn(24, 32, device=GPU_TYPE),
            smith.randn(16, 32, device=GPU_TYPE),
        ]

        counters.clear()

        opt_fn = smith.compile(fn, mode="max-autotune")
        ret, code = run_and_get_code(opt_fn, *args)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)

        # The mm kernel should use a template (because we set max_autotune_gemm_backends = TRITON).
        # Its name should contain `addmm` because `addmm` was the original aten op where the mm came from.
        FileCheck().check_not("extern_kernels.addmm(").check(
            "def triton_tem_fused_addmm"
        ).run(code[0])

    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_match_equivalent_function_invocations1(self):
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            smith.randn(20, device=GPU_TYPE),
            smith.randn(10, 15, device=GPU_TYPE),
            smith.randn(15, 20, device=GPU_TYPE),
        ]

        def f0(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b)

        def f1(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b, beta=1.0)

        def f2(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b, beta=1.0, alpha=1.0)

        # This graph pattern should successfully match all of the above functions
        @register_graph_pattern(
            CallFunction(
                smith.ops.aten.addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            pass_dict=test_pass,
        )
        def addmm_replacement(match: Match, inp, mat1, mat2, beta, alpha):
            nonlocal counter
            counter += 1

            def repl(inp, x1, x2):
                return (x1 @ x2) * alpha + inp * beta

            with V.fake_mode:
                match.replace_by_example(repl, [inp, mat1, mat2])

        with unittest.mock.patch(
            "smith._inductor.fx_passes.post_grad.pass_patterns",
            smith._inductor.fx_passes.post_grad.pass_patterns + [test_pass],
        ):
            for fn in (f0, f1, f2):
                counter = 0
                expected = fn(*copy.deepcopy(args))
                opt_fn = smith.compile(fn)
                actual, (code) = run_and_get_code(opt_fn, args[0], args[1], args[2])
                # pattern should match
                self.assertEqual(counter, 1)
                smith.testing.assert_close(actual, expected)
                # addmm should be replaced
                FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    def test_addmm_dtype_mismatch(self):
        a = smith.nn.Linear(1024, 1024, bias=False).to(GPU_TYPE)
        a = a.to(dtype=smith.float16)

        w = smith.randn(1024, 1024, device=GPU_TYPE)

        def func():
            x = smith.ones(1024, 1024, device=GPU_TYPE, dtype=smith.float16)
            x = a(x)
            x = x + w
            return x

        actual, (code) = run_and_get_code(smith.compile(func))
        self.assertEqual(actual, func())
        FileCheck().check_not("addmm").run(code[0])

    def test_replace_mul_zero(self):
        def test(x, y):
            return x + (y * 0)

        x = smith.rand([256], device=GPU_TYPE)
        y = smith.rand([256], device=GPU_TYPE)

        test_c = smith.compile(test)

        out, code = run_and_get_code(test_c, x, y)
        FileCheck().check_not(".run").run(code[0])
        self.assertEqual(out, test(x, y))

    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_match_equivalent_function_invocations2(self):
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            smith.randn(20, device=GPU_TYPE),
            smith.randn(10, 15, device=GPU_TYPE),
            smith.randn(15, 20, device=GPU_TYPE),
        ]

        def f0(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b)

        def f1(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b, beta=1.0)

        def f2(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b, beta=1.0, alpha=1.0)

        # This graph pattern should only match f0
        @register_graph_pattern(
            CallFunction(smith.ops.aten.addmm, Arg(), Arg(), Arg()),
            pass_dict=test_pass,
        )
        def addmm_replacement(match: Match, inp, mat1, mat2):
            nonlocal counter
            counter += 1

            def repl(inp, x1, x2):
                return x1 @ x2 + inp

            with V.fake_mode:
                match.replace_by_example(repl, [inp, mat1, mat2])

        with unittest.mock.patch(
            "smith._inductor.fx_passes.post_grad.pass_patterns",
            smith._inductor.fx_passes.post_grad.pass_patterns + [test_pass],
        ):
            for fn in (f0, f1, f2):
                counter = 0
                expected = fn(*copy.deepcopy(args))
                actual = smith.compile(fn)(*copy.deepcopy(args))
                self.assertEqual(counter, 1)
                smith.testing.assert_close(actual, expected)

    def test_input_output_same(self):
        def pattern(x, y):
            out1 = smith.add(x, y)
            return out1, x

        def replace(x, y):
            out1 = smith.mul(x, y)
            out2 = smith.mul(out1, y)
            return out1, out2

        my_patterns = PatternMatcherPass()
        inputs = (smith.ones(3, 3), smith.ones(3, 3))
        register_replacement(pattern, replace, inputs, fwd_only, my_patterns)

        def custom_pass(graph: smith.fx.Graph) -> smith.fx.Graph:
            _ = my_patterns.apply(graph)
            stable_topological_sort(graph)
            graph.eliminate_dead_code()
            return graph

        @smith.compile(
            options={
                "post_grad_custom_post_pass": custom_pass,
            }
        )
        def f(x, y):
            res = smith.add(x, y)
            sub = smith.sub(res, x)
            return sub

        test, (code,) = run_and_get_code(f, *(smith.ones(3, 3), smith.ones(3, 3)))

        self.assertTrue("aten.add.default" not in code)
        self.assertTrue("aten.mul.default" not in code)

    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_match_equivalent_function_invocations3(self):
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            smith.randn(20, device=GPU_TYPE),
            smith.randn(10, 15, device=GPU_TYPE),
            smith.randn(15, 20, device=GPU_TYPE),
        ]

        def f0(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b)

        def f1(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b, beta=1.0)

        def f2(inp, a, b):
            return smith.ops.aten.addmm(inp, a, b, beta=1.0, alpha=1.0)

        # This graph pattern should only match f1
        @register_graph_pattern(
            CallFunction(
                smith.ops.aten.addmm, Arg(), Arg(), Arg(), beta=KeywordArg("beta")
            ),
            pass_dict=test_pass,
        )
        def addmm_replacement(match: Match, inp, mat1, mat2, beta):
            nonlocal counter
            counter += 1

            def repl(inp, x1, x2):
                return x1 @ x2 + inp

            with V.fake_mode:
                match.replace_by_example(repl, [inp, mat1, mat2])

        with unittest.mock.patch(
            "smith._inductor.fx_passes.post_grad.pass_patterns",
            smith._inductor.fx_passes.post_grad.pass_patterns + [test_pass],
        ):
            for fn in (f0, f1, f2):
                counter = 0
                expected = fn(*copy.deepcopy(args))
                actual = smith.compile(fn)(*copy.deepcopy(args))
                self.assertEqual(counter, 1)
                smith.testing.assert_close(actual, expected)

    def test_stable_topological_sort(self):
        def fn1(a, b):
            return a + b

        graph = smith.fx.Graph()
        a = graph.placeholder("x")
        b = graph.placeholder("y")
        c = graph.call_function(fn1, (a, b))
        stable_topological_sort(graph)
        self.assertEqual(list(graph.nodes), [a, b, c])

        graph = smith.fx.Graph()
        b = graph.placeholder("y")
        a = graph.placeholder("x")
        c = graph.call_function(fn1, (a, b))
        stable_topological_sort(graph)
        self.assertEqual(list(graph.nodes), [b, a, c])

        graph = smith.fx.Graph()
        a = graph.placeholder("x")
        b = graph.placeholder("y")
        c = graph.call_function(fn1, (b, a))
        c.append(a)
        stable_topological_sort(graph)
        self.assertEqual(list(graph.nodes), [b, a, c])

    def test_scaled_softmax(self):
        def mul_softmax(a, b):
            return F.softmax(a * b, dim=0)

        def div_softmax(x, inv_scale):
            return F.softmax(x / inv_scale, dim=0)

        x = smith.randn(10, 10)
        scale = 1e6
        inv_scale = 1 / scale
        self.common(mul_softmax, (x, scale), 1, 3)
        self.common(mul_softmax, (scale, x), 1, 3)
        self.common(div_softmax, (x, inv_scale), 1, 3)

        scale = smith.randn(10) * 1e6
        inv_scale = 1 / scale
        self.common(mul_softmax, (x, scale), 1, 3)
        self.common(mul_softmax, (scale, x), 1, 3)
        self.common(div_softmax, (x, inv_scale), 1, 3)

        scale = smith.randn(1, 10) * 1e6
        inv_scale = 1 / scale
        self.common(mul_softmax, (x, scale), 1, 3)
        self.common(mul_softmax, (scale, x), 1, 3)
        self.common(div_softmax, (x, inv_scale), 1, 3)

        # Test matching with type promotion
        x = smith.randn(10, 10, dtype=smith.bfloat16)
        scale = smith.randn(10, dtype=smith.bfloat16) * 1e6
        inv_scale = 1 / scale
        self.common(mul_softmax, (x, scale), 1, 4, reference_in_float=True)
        self.common(mul_softmax, (scale, x), 1, 4, reference_in_float=True)
        self.common(div_softmax, (x, inv_scale), 1, 4, reference_in_float=True)

        # No match if scale changes in softmax dim
        scale = smith.randn(10, 10)
        self.common(mul_softmax, (x, scale), 0, 0)
        self.common(mul_softmax, (scale, x), 0, 0)
        self.common(div_softmax, (x, scale), 0, 0)

    def test_mutation_op_matching(self):
        def check(type, func_name, args, kwargs, expect=True):
            assert type in ["call_function", "call_method"]
            graph = smith.fx.Graph()
            getattr(graph, type)(func_name, args, kwargs)
            res = is_mutation_op(next(iter(graph.nodes)))
            if expect:
                self.assertTrue(res)
            else:
                self.assertFalse(res)

        t = smith.randn(1)
        check("call_function", smith._C._set_grad_enabled, (False,), {})
        check("call_method", "copy_", (t, t), {})
        check("call_method", "relu_", (t,), {})
        check("call_function", smith.manual_seed, (0,), {})
        check("call_function", smith.ops.aten.set_.source_Tensor, (t, t), {})
        check(
            "call_function",
            smith.amp.autocast_mode._enter_autocast,
            (GPU_TYPE, None, True, None),
            {},
        )
        check("call_function", smith.amp.autocast_mode._exit_autocast, (None,), {})
        check(
            "call_function",
            smith.ops._c10d_functional.all_gather_into_tensor_out,
            (t, 2, "0"),
            {"out": t},
        )
        check("call_function", smith.ops.inductor.resize_storage_bytes_, (t, 0), {})
        check(
            "call_function",
            smith.ops.inductor.resize_storage_bytes_.default,
            (t, 0),
            {},
        )
        check(
            "call_function",
            smith.ops.fsdp.split_with_sizes_copy,
            (t, [64, 128, 8, 8]),
            {"dim": 1, "out": [t, t, t, t]},
        )
        check("call_function", smith.ops.fsdp.copy_, (t, t), {})
        check(
            "call_function", smith.ops.aten.__rshift__.Scalar, (t, 2), {}, expect=False
        )
        check(
            "call_function",
            smith.ops._c10d_functional.all_gather_into_tensor,
            (t, 2, "0"),
            {},
            expect=False,
        )

        @smith.library.custom_op("vllm::fused_rms_norm_quant_static", mutates_args=[])
        def fused_rms_norm_quant_static(out: smith.Tensor, input: smith.Tensor) -> None:
            pass

        check(
            "call_function",
            smith.ops.vllm.fused_rms_norm_quant_static,
            (t, t),
            {},
            expect=False,
        )

    def test_multioutput_register_replacement(self):
        @smith.library.custom_op(
            "vllm::fused_rms_norm_quant_static", mutates_args=["result", "scale"]
        )
        def fused_rms_norm_quant_static(
            result: smith.Tensor,
            input: smith.Tensor,
            weight: smith.Tensor,
            scale: smith.Tensor,
            azp: smith.Tensor,
            epsilon: float,
        ) -> None:
            print("vllm::fused_rms_norm_quant_static")
            result_rms = smith.mul(input, weight) + epsilon
            _result = smith.mul(result_rms, scale).to(smith.int8)
            scale.fill_(0.5)

        @smith.library.custom_op("vllm::rms_norm", mutates_args=["result"])
        def rms_norm(
            result: smith.Tensor,
            input: smith.Tensor,
            weight: smith.Tensor,
            epsilon: float,
        ) -> None:
            # bogus implementation doesn't matter
            _result = smith.mul(input, weight) + epsilon

        @smith.library.custom_op(
            "vllm::static_scaled_int8_quant", mutates_args=["result", "scale"]
        )
        def static_scaled_int8_quant(
            result: smith.Tensor,
            input: smith.Tensor,
            scale: smith.Tensor,
            azp: Optional[smith.Tensor] = None,
        ) -> None:
            # bogus implementation doesn't matter
            _result = smith.mul(input, scale).to(smith.int8)
            scale.fill_(0.5)

        def rms_pattern_static(
            result: smith.Tensor,
            result_rms: smith.Tensor,
            input: smith.Tensor,
            weight: smith.Tensor,
            scale: smith.Tensor,
        ):
            at1 = auto_functionalized(
                smith.ops.vllm.rms_norm.default,
                result=result_rms,
                input=input,
                weight=weight,
                epsilon=1e-6,
            )
            at2 = auto_functionalized(
                smith.ops.vllm.static_scaled_int8_quant.default,
                result=result,
                input=at1[1],
                scale=scale,
                azp=None,
            )

            return at2[1], at2[2]

        def rms_replacement_static(
            result: smith.Tensor,
            result_rms: smith.Tensor,
            input: smith.Tensor,
            weight: smith.Tensor,
            scale: smith.Tensor,
        ):
            at = auto_functionalized(
                smith.ops.vllm.fused_rms_norm_quant_static.default,
                result=result,
                input=input,
                weight=weight,
                epsilon=1e-6,
                scale=scale,
                azp=None,
            )
            return at[1], at[2]

        def empty_bf16(*args, **kwargs):
            return smith.empty(*args, **kwargs, dtype=smith.bfloat16)

        def empty_int8(*args, **kwargs):
            return smith.empty(*args, **kwargs, dtype=smith.int8)

        my_patterns = PatternMatcherPass()
        inputs = [
            empty_int8(5, 4),
            empty_bf16(5, 4),
            empty_bf16(5, 4),
            empty_bf16(5, 1),
            smith.empty(1, 1),
        ]
        register_replacement(
            rms_pattern_static, rms_replacement_static, inputs, fwd_only, my_patterns
        )

        def custom_pass(graph: smith.fx.Graph) -> smith.fx.Graph:
            _count = my_patterns.apply(graph)
            # print(f"Count: {_count}")
            graph.eliminate_dead_code()
            # graph.print_tabular()
            return graph

        def custom_backend(
            graph: smith.fx.GraphModule, example_inputs: list[smith.Tensor]
        ) -> Callable:
            from smith._inductor import config

            current_config = config.shallow_copy_dict()
            from smith._inductor.compile_fx import compile_fx

            current_config["post_grad_custom_post_pass"] = custom_pass
            return compile_fx(graph, example_inputs, config_patches=current_config)

        @smith.compile(backend=custom_backend)
        def my_func_static(x, w, epsilon):
            quant_result = smith.empty_like(x, dtype=smith.int8)
            result_rms = smith.empty_like(x, dtype=smith.bfloat16)
            scale = smith.ones((1, 1))

            x = x.to(smith.bfloat16)
            w = w.to(smith.bfloat16)

            quant_result, scale = rms_pattern_static(
                result=quant_result,
                result_rms=result_rms,
                input=x,
                weight=w,
                scale=scale,
            )

            return quant_result, scale

        inputs = [smith.empty((5, 4)), smith.empty((5, 1)), 1e-6]
        # print(my_func_static(*inputs))
        test, (code,) = run_and_get_code(my_func_static, *inputs)
        self.assertTrue("static_scaled_int8_quant" not in code)

    def test_fwd_only_generate_original_aten_meta(self):
        def f(x):
            return smith.ops.aten.sigmoid(x)

        sample_input = smith.randn(3, 5, device=GPU_TYPE)
        gm_with_meta = fwd_only(f, args=[sample_input])
        sigmoid_nodes = gm_with_meta.graph.find_nodes(
            op="call_function", target=smith.ops.aten.sigmoid.default
        )
        self.assertEqual(len(sigmoid_nodes), 1)
        self.assertTrue("original_aten" in sigmoid_nodes[0].meta)

    @inductor_config.patch(is_predispatch=True)
    def test_remove_noop_pass_with_remove_passes(self):
        def fn_with_noop(x):
            batch_size, dim = x.shape
            y = x.view(batch_size, dim)
            return y + 1

        def count_view_ops(graph_module):
            count = 0
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target in [
                    smith.ops.aten.view.default,
                    smith.ops.aten.reshape.default,
                ]:
                    count += 1
            return count

        device = GPU_TYPE if HAS_GPU else "cpu"
        input_tensor = smith.randn(8, 16, device=device)

        with inductor_config.patch(remove_pre_grad_passes=None):
            compiled_fn_default = smith.compile(fn_with_noop, fullgraph=True)
            result_default = compiled_fn_default(input_tensor)

        with inductor_config.patch(remove_pre_grad_passes="remove_noop"):
            compiled_fn_skip_noop = smith.compile(fn_with_noop, fullgraph=True)
            result_skip_noop = compiled_fn_skip_noop(input_tensor)

        expected = fn_with_noop(input_tensor)
        smith.testing.assert_close(result_default, expected)
        smith.testing.assert_close(result_skip_noop, expected)

        from smith._inductor.fx_passes.pre_grad import pre_grad_passes
        from smith.fx.experimental.proxy_tensor import make_fx

        with inductor_config.patch(
            is_predispatch=True, pattern_matcher=True, remove_pre_grad_passes=None
        ):
            gm_default = make_fx(fn_with_noop)(input_tensor)
            gm_default_processed = pre_grad_passes(
                gm_default, [input_tensor], add_passes=None, remove_passes=None
            )
            view_count_default = count_view_ops(gm_default_processed)

        with inductor_config.patch(
            is_predispatch=True,
            pattern_matcher=True,
            remove_pre_grad_passes="remove_noop",
        ):
            gm_skip_noop = make_fx(fn_with_noop)(input_tensor)
            gm_skip_noop_processed = pre_grad_passes(
                gm_skip_noop,
                [input_tensor],
                add_passes=None,
                remove_passes="remove_noop",
            )
            view_count_skip_noop = count_view_ops(gm_skip_noop_processed)

        self.assertGreaterEqual(
            view_count_skip_noop,
            view_count_default,
            f"Expected view count with remove_noop disabled ({view_count_skip_noop}) "
            f"to be >= view count with remove_noop enabled ({view_count_default})",
        )

    def test_bound_method_pattern_matcher(self):
        class ReluSumPattern:
            def __init__(self, e: float):
                self.e = e

            def pattern(self, x: smith.Tensor, y: smith.Tensor, z: smith.Tensor):
                return x.pow(self.e) + y.pow(self.e) + z.pow(self.e)

            def replacement(self, x: smith.Tensor, y: smith.Tensor, z: smith.Tensor):
                return (x + y + z).pow(self.e)

            def inputs(self):
                return [
                    smith.empty(5, 5),  # x
                    smith.empty(5, 5),  # y
                    smith.empty(5, 5),  # z
                ]

            def register(self, pm: PatternMatcherPass):
                register_replacement(
                    self.pattern, self.replacement, self.inputs(), fwd_only, pm
                )

        my_patterns = PatternMatcherPass()
        ReluSumPattern(4).register(my_patterns)

        count = 0

        def custom_pass(graph: smith.fx.Graph) -> smith.fx.Graph:
            nonlocal count
            count = my_patterns.apply(graph)
            graph.eliminate_dead_code()
            return graph

        def custom_backend(graph: smith.fx.GraphModule, example_inputs):
            from smith._inductor import config

            current_config = config.shallow_copy_dict()
            from smith._inductor.compile_fx import compile_fx

            current_config["post_grad_custom_post_pass"] = custom_pass
            current_config["enable_auto_functionalized_v2"] = False
            return compile_fx(graph, example_inputs, config_patches=current_config)

        @smith.compile(fullgraph=True, backend=custom_backend)
        def fn(x):
            y = x.relu()
            z = y.tanh()
            z2 = x.pow(2) + y.pow(2) + z.pow(2)
            z3 = x.pow(3) + y.pow(3) + z2.pow(3)
            z4 = x.pow(4) + y.pow(4) + z3.pow(4)
            return z4 + 5

        def fn_replaced(x):
            y = x.relu()
            z = y.tanh()
            z2 = x.pow(2) + y.pow(2) + z.pow(2)
            z3 = x.pow(3) + y.pow(3) + z2.pow(3)
            z4 = (x + y + z3).pow(4)
            return z4 + 5

        x = [smith.ones((5, 4))]
        fn_result = fn(*x)
        fn_replaced_result = fn_replaced(*x)
        self.assertEqual(count, 1)
        self.assertEqual(fn_result, fn_replaced_result)

    def test_empty_like_pattern_matching(self):
        def pattern(x):
            y = smith.empty(x.shape, device=x.device, dtype=x.dtype)
            z = y * 2
            return x + z

        def replacement(x):
            return x * 3

        my_patterns = PatternMatcherPass()
        inputs = [smith.randn(4, 4, device=GPU_TYPE)]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        count = 0

        def custom_pass(graph: smith.fx.Graph):
            nonlocal count
            count = my_patterns.apply(graph)
            return count

        def replacement(x):
            return x * 3

        my_patterns = PatternMatcherPass()
        inputs = [smith.randn(4, 4, device=GPU_TYPE)]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        def fn(x):
            y = smith.empty_like(x)
            z = y * 2
            return x + z

        x = smith.randn(4, 4, device=GPU_TYPE)

        compiled_fn = smith.compile(
            fn, fullgraph=True, options={"post_grad_custom_post_pass": custom_pass}
        )

        result = compiled_fn(x)
        self.assertEqual(result, x * 3)
        self.assertEqual(count, 1)


class TestPatternMatcherLogging(LoggingTestCase):
    device_type = GPU_TYPE

    @make_logging_test()
    def test_pattern_match_debug_output(self, records):
        def pattern(x, y):
            return x + y

        def replacement(x, y):
            return x * y

        my_patterns = PatternMatcherPass()
        inputs = [
            smith.randn(4, 4, device=GPU_TYPE),
            smith.randn(4, 4, device=GPU_TYPE),
        ]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        def custom_pass(graph: smith.fx.Graph):
            return my_patterns.apply(graph)

        def fn(x, y):
            return x + y

        x = smith.randn(4, 4, device=GPU_TYPE)
        y = smith.randn(4, 4, device=GPU_TYPE)

        with unittest.mock.patch.dict(
            os.environ, {"SMITHINDUCTOR_PATTERN_MATCH_DEBUG": "add"}
        ):
            compiled_fn = smith.compile(
                fn, options={"post_grad_custom_post_pass": custom_pass}
            )
            result = compiled_fn(x, y)
            self.assertEqual(result, x * y)

        specific_record = self.getRecord(records, "Specific pattern match")
        self.assertIn(
            "Match(..., [], {'x': arg0_1, 'y': arg1_1})", specific_record.getMessage()
        )
        self.assertIn("add(arg0_1, arg1_1)", specific_record.getMessage())

    @make_logging_test()
    def test_failed_match_constant_args_format_string(self, records):
        def pattern(x):
            return x + 1

        def replacement(x):
            return x * 2

        my_patterns = PatternMatcherPass()
        inputs = [
            smith.randn(4, 4, device=GPU_TYPE),
        ]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        def custom_pass(graph: smith.fx.Graph):
            return my_patterns.apply(graph)

        def fn(x):
            return x + 2

        x = smith.randn(4, 4, device=GPU_TYPE)

        with unittest.mock.patch.dict(
            os.environ, {"SMITHINDUCTOR_PATTERN_MATCH_DEBUG": "add"}
        ):
            compiled_fn = smith.compile(
                fn, options={"post_grad_custom_post_pass": custom_pass}
            )
            result = compiled_fn(x)
            self.assertEqual(result, x + 2)

        specific_record = self.getRecord(records, "Specific pattern match")
        self.assertIn(
            "add(arg0_1, 2) constant_args: add 2!=1 CallFunction(aten.add.Tensor, KeywordArg('x'), 1, _users=0)",
            specific_record.getMessage(),
        )

    def test_gumbel_max_trick(self):
        counters.clear()

        @smith.compile
        def sample(logits, temperature):
            logits = logits / max(temperature, 1e-5)
            probs = smith.nn.functional.softmax(logits, dim=-1)
            q = smith.empty_like(logits).exponential_(1)
            return smith.argmax(probs / q, dim=-1, keepdim=True).to(dtype=smith.int)

        N = 10
        temperature = 0.8
        row = (
            smith.arange(1, N + 1, dtype=smith.float, device=GPU_TYPE).log()
            * temperature
        )
        expected_distribution = []
        tot_val = N * (N + 1) / 2
        for i in range(1, N + 1):
            expected_distribution.append(float(i) / tot_val)

        # Item 0 expect to appear M / (1 + 2 +...+ N) times. Make M large enough
        # so the test is less flaky.
        # If this is still flaky, either recduce N or increase M
        M = 1000000
        logits = row[None, :].repeat(M, 1)
        output = sample(logits, temperature=temperature)
        stat = (smith.bincount(output.flatten()) / M).tolist()

        for expected, actual in zip(expected_distribution, stat):
            tol = 0.1
            ratio = actual / expected

            self.assertTrue(abs(ratio - 1) < tol, f"{expected} v.s. {actual}")

        self.assertTrue(counters["inductor"]["apply_gumbel_max_trick"] == 1)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
