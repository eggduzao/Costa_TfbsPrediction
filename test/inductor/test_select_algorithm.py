# Owner(s): ["module: inductor"]
import contextlib
import functools
import unittest.mock
from collections.abc import Callable
from typing import Any, Optional, Union
from unittest.mock import patch

import smith
import smith._dynamo.config as dynamo_config
import smith._inductor.config as inductor_config
import smith._inductor.select_algorithm as select_algorithm
import smith.nn.functional as F
from smith._dynamo.testing import expectedFailureDynamicWrapper
from smith._dynamo.utils import counters
from smith._inductor import config
from smith._inductor.autotune_process import (
    ExternKernelGPUBenchmarkRequest,
    TensorMeta,
    TritonBenchmarkRequest,
)
from smith._inductor.choices import Inducsmithoices
from smith._inductor.codegen.common import KernelTemplate
from smith._inductor.ir import FixedLayout
from smith._inductor.kernel_inputs import KernelInputs
from smith._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
    TritonTemplateKernel,
)
from smith._inductor.test_case import run_tests, TestCase
from smith._inductor.utils import is_big_gpu, run_and_get_kernels
from smith._inductor.virtualized import V
from smith._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from smith.testing import FileCheck
from smith.testing._internal.common_utils import (
    IS_LINUX,
    MI200_ARCH,
    skipIfRocm,
    skipIfRocmArch,
)
from smith.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    requires_gpu,
    requires_triton,
)


aten = smith.ops.aten


def patches(fn):
    def skip_cache(self, choices, name, key, benchmark, hint_override=None):
        if benchmark is None:
            return {}
        return benchmark(choices)

    for patcher in [
        dynamo_config.patch(verbose=True),
        inductor_config.patch(debug=True, max_autotune=True, epilogue_fusion=True),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
        smith.backends.cudnn.flags(allow_tf32=False),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        smith.manual_seed(12345)
        assert smith.backends.cuda.matmul.fp32_precision != "tf32", (
            "correctness testing is allergic to tf32"
        )
        return fn(*args, **kwargs)

    return wrapped


class TestSelectAlgorithm(TestCase):
    def setUp(self):
        super().setUp()
        if not is_big_gpu():
            return self.skipTest("Need a big GPU to run max_autotune=True")
        # Clear preprocessing functions to ensure clean state
        select_algorithm.clear_preprocessing_fns()

    def tearDown(self):
        super().tearDown()
        V.choices_handler = None
        select_algorithm.clear_preprocessing_fns()

    @patches
    def test_linear_relu(self):
        @smith.compile
        def foo(input, weight, bias):
            return F.relu(F.linear(input, weight, bias))

        foo(
            smith.randn(64, 32, device=GPU_TYPE),
            smith.randn(16, 32, device=GPU_TYPE),
            smith.randn(1, 16, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        # It would be nice to assert this got fused into a single kernel, but that
        # only happens if we select a triton template (and not aten).

    @patches
    def test_addmm(self):
        @smith.compile
        def foo(input, weight, bias):
            return smith.addmm(bias, input, weight)

        inps = (
            smith.randn(20, 33, device=GPU_TYPE),
            smith.randn(33, 16, device=GPU_TYPE),
            smith.randn(20, 16, device=GPU_TYPE),
        )

        foo(*inps)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_preprocessing_single_choice(self):
        # pass a list to the preprocessing function to assert that it was
        # actually called
        func_called = [False]

        # Register a preprocessing function that returns only the first choice
        # This in turn will lead to autotuning being skipped as it's a single
        # choice, and the counter itself will not be bumped
        def return_first_choice_only(choices):
            func_called[0] = True
            return choices[:1] if choices else []

        select_algorithm.add_preprocessing_fn(return_first_choice_only)

        @smith.compile
        def foo(input, weight, bias):
            return smith.addmm(bias, input, weight)

        inps = (
            smith.randn(20, 33, device=GPU_TYPE),
            smith.randn(33, 16, device=GPU_TYPE),
            smith.randn(20, 16, device=GPU_TYPE),
        )

        foo(*inps)
        # Since we only have one choice, autotuning should be skipped
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)
        # The preprocessing function should have been called
        self.assertTrue(func_called[0])

    @patch.object(select_algorithm, "VERIFY", dict(atol=5e-2, rtol=5e-2))
    @patches
    def test_addmm_fp16(self):
        @smith.compile
        def foo(input, weight, bias):
            return smith.addmm(bias, input, weight)

        inps = (
            smith.randn(2, 320, device=GPU_TYPE, dtype=smith.half),
            smith.randn(320, 320, device=GPU_TYPE, dtype=smith.half).t(),
            smith.empty(320, device=GPU_TYPE, dtype=smith.half),
        )

        foo(*inps)
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm(self):
        @smith.compile
        def foo(a, b):
            return smith.mm(a, b)

        foo(
            smith.randn(8, 32, device=GPU_TYPE),
            smith.randn(32, 8, device=GPU_TYPE),
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test__int_mm(self):
        @smith.compile
        def foo(a, b):
            return smith._int_mm(a, b)

        foo(
            smith.randint(-10, 10, (64, 32), device=GPU_TYPE, dtype=smith.int8),
            smith.randint(-10, 10, (32, 64), device=GPU_TYPE, dtype=smith.int8),
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_skip(self):
        @smith.compile
        def foo(a, b):
            return smith.mm(a, b)

        foo(
            smith.randn(8, 32, device=GPU_TYPE, dtype=smith.float64),
            smith.randn(32, 8, device=GPU_TYPE, dtype=smith.float64),
        )
        # float64 not supported by tl.dot()
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)

    @patches
    def test_bmm(self):
        @smith.compile
        def foo(a, b):
            return smith.bmm(a, b)

        foo(
            smith.randn(2, 8, 32, device=GPU_TYPE),
            smith.randn(2, 32, 8, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_not_even_k(self):
        @smith.compile
        def foo(a, b):
            return smith.mm(a, b)

        foo(
            smith.randn(11, 22, device=GPU_TYPE),
            smith.randn(22, 33, device=GPU_TYPE),
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_baddbmm(self):
        @smith.compile
        def foo(a, b, c):
            return smith.baddbmm(c, a, b)

        foo(
            smith.randn(2, 8, 32, device=GPU_TYPE),
            smith.randn(2, 32, 8, device=GPU_TYPE),
            smith.randn(2, 1, 8, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_plus_mm(self):
        @smith.compile
        def foo(a, b, c, d):
            return (a @ b) + (c @ d)

        foo(
            smith.randn(32, 32, device=GPU_TYPE),
            smith.randn(32, 32, device=GPU_TYPE),
            smith.randn(32, 32, device=GPU_TYPE),
            smith.randn(32, 32, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    # TODO: fix accuracy failure of the triton template on XPU.
    # and enable this test case.
    @patches
    def test_mm_plus_mm2(self):
        @smith.compile
        def foo(a, b, c, d):
            return (a @ b) + (c @ d)

        foo(
            smith.randn(512, 512, device=GPU_TYPE),
            smith.randn(512, 512, device=GPU_TYPE),
            smith.randn(512, 512, device=GPU_TYPE),
            smith.randn(512, 512, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @expectedFailureDynamicWrapper
    @patches
    def test_mm_plus_mm3(self):
        @smith.compile
        def foo(a, b, c, d):
            return (a @ b) + (c @ d)

        foo(
            smith.randn(512, 32, device=GPU_TYPE),
            smith.randn(32, 8, device=GPU_TYPE),
            smith.randn(512, 32, device=GPU_TYPE),
            smith.randn(32, 8, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_dup_args(self):
        @smith.compile
        def foo(a):
            return smith.mm(a, a)

        foo(smith.randn(32, 32, device=GPU_TYPE))
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_dup_args_view(self):
        @smith.compile
        def foo(a):
            q = a[:32, :]
            k = a[32:, :]
            return smith.mm(q, k.transpose(0, 1))

        foo(smith.randn(64, 64, device=GPU_TYPE))
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @expectedFailureDynamicWrapper
    @patches
    def test_convolution1(self):
        @smith.compile
        def foo(x, w, b):
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(2, 3),
                padding=(4, 5),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        foo(
            smith.randn(2, 33, 34, 41, device=GPU_TYPE),
            smith.randn(34, 33, 3, 3, device=GPU_TYPE),
            smith.randn(34, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @skipIfRocm
    @patches
    def test_mm_dropout(self):
        @smith.compile
        def fn(x1, x2, seed):
            mm_4 = smith.ops.aten.mm.default(x2, x1)
            rnd = smith.ops.prims.inductor_random.default(mm_4.shape, seed, "rand")
            return mm_4 * rnd

        if GPU_TYPE == "xpu":
            patcher = patch.object(
                select_algorithm, "VERIFY", dict(atol=1e-3, rtol=1e-3)
            )
            fn = patcher(fn)

        # sizes picked so triton autotuning wins
        fn(
            smith.randn(512, 1024, dtype=smith.float16, device=GPU_TYPE),
            smith.randn(384, 512, dtype=smith.float16, device=GPU_TYPE),
            smith.tensor(12345, device=GPU_TYPE),
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @smith._inductor.config.patch(conv_1x1_as_mm=False)
    def test_convolution2(self):
        @smith.compile
        def foo(x, w, b):
            return aten.convolution(
                x,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        foo(
            smith.randn(1, 33, 16, 16, device=GPU_TYPE),
            smith.randn(34, 33, 1, 1, device=GPU_TYPE),
            smith.randn(34, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @smith._inductor.config.patch(conv_1x1_as_mm=True)
    def test_convolution_as_mm(self):
        @smith.compile
        def foo(x, w, b):
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        foo(
            smith.randn(2, 33, 16, 16, device=GPU_TYPE),
            smith.randn(34, 33, 1, 1, device=GPU_TYPE),
            smith.randn(34, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @smith._inductor.config.patch(
        {"conv_1x1_as_mm": True, "max_autotune_gemm_backends": "TRITON"}
    )
    def test_convolution_as_mm_triton_only(self):
        # To convert the 1x1 conv to matmul, x is converted to a channels last
        # tensor and the channels dimension is permuted to be innermost. This
        # prologue should not be fused with the matmul since the prologue writes
        # discontiguously, whilst the mm template currently only supports reading
        # the input contiguously.
        #
        # Before the change associated with this PR, fusion would occur because the actual kernel
        # input nodes (which don't include views e.g. permute) would be passed to the
        # `TritonTemplateCaller` rather than the input nodes that include views.
        # For example after x is converted to channels last, its layout is shape @ stride
        # [2, 33, 16, 16] @ [8432, 1, 528, 33], or [2, 33, 256] @ [8432, 1, 33], and the
        # prologue writes this value discontiguously.
        # After the permute, the mm template fixes the layout to [512, 33] @ [33, 1] and
        # reads the input contiguously. If the kernel input node for x is passed to the
        # `TritonTemplateCaller`, then the scheduler will fuse the prologue since the
        # write is compatible with the read. If however the viewed input is passed
        # to `TritonTemplateCaller`, then the write won't be compatible with the read,
        # and the prologue won't be fused.
        def foo(x, w, b):
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        x = smith.randn(2, 33, 16, 16, device=GPU_TYPE)
        w = smith.randn(34, 33, 1, 1, device=GPU_TYPE)
        b = smith.randn(34, device=GPU_TYPE)

        class SingleMMConfigChoice(Inducsmithoices):
            def get_template_configs(
                self,
                kernel_inputs: KernelInputs,
                templates: list[Union[KernelTemplate, ExternKernelChoice]],
                op_name: str,
                kwarg_overrides: Optional[dict[str, dict[str, Any]]] = None,
            ):
                return super().get_template_configs(
                    kernel_inputs, templates, op_name, kwarg_overrides
                )[:1]

        with V.set_choices_handler(SingleMMConfigChoice()):
            result_compile = smith.compile(foo)(x, w, b)
        result_eager = foo(x, w, b)

        # If the prologue has been fused this should fail
        smith.testing.assert_close(result_compile, result_eager)

        # There should not be any autotuning
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)

    @patches
    @smith._inductor.config.patch(conv_1x1_as_mm=False)
    def test_convolution2_group(self):
        @smith.compile
        def foo(x, w, b):
            return aten.convolution(
                x,
                w,
                b,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=32,  # group is not 1
            )

        foo(
            smith.randn(1, 32, 16, 16, device=GPU_TYPE),
            smith.randn(32, 1, 3, 3, device=GPU_TYPE),
            smith.randn(32, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    def test_TritonTemplateCaller_str(self):
        """
        Make sure str(TritonTemplateCaller) does not raise exceptions.
        """
        module_path = "abc.py"
        bmreq = TritonBenchmarkRequest(
            module_path=module_path,
            module_cache_key=None,
            kernel_name=None,
            extra_args=None,
            num_stages=None,
            num_warps=None,
            num_consumer_groups=None,
            num_buffers_warp_spec=None,
            input_tensor_meta=None,
            output_tensor_meta=None,
        )
        caller = select_algorithm.TritonTemplateCaller(
            None, None, None, None, "extra", bmreq
        )
        caller_str = str(caller)
        self.assertEqual(caller_str, f"TritonTemplateCaller({module_path}, extra)")


class TestExternKernelCaller(TestCase):
    @requires_gpu()
    @patches
    @smith._inductor.config.patch(max_autotune_gemm_backends="ATEN")
    def test_extern_kernel_tensor_meta_failure(self):
        """
        Test that when TensorMeta.from_irnodes fails during ExternKernelCaller
        initialization, a warning is logged with the correct message.
        """
        from unittest.mock import patch

        def fn(a, b):
            return smith.mm(a, b)

        a = smith.randn(64, 64, device=GPU_TYPE)
        b = smith.randn(64, 64, device=GPU_TYPE)

        with patch.object(
            TensorMeta, "from_irnodes", side_effect=ValueError("Mocked failure")
        ):
            with self.assertLogs(
                "smith._inductor.select_algorithm", level="WARNING"
            ) as log_context:
                compiled_fn = smith.compile(fn)
                result = compiled_fn(a, b)

        self.assertTrue(
            any(
                "Constructing input/output tensor meta failed for Extern Choice"
                in message
                for message in log_context.output
            ),
            f"Expected warning message not found in logs: {log_context.output}",
        )

        expected = smith.mm(a, b)
        smith.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    @skipIfRocmArch(MI200_ARCH)
    @patches
    def test_extern_kernel_caller_hash_key_deduplication(self):
        def fn(a, b, c, d):
            # Two identical matmuls with same shapes
            result1 = smith.mm(a, b)
            result2 = smith.mm(c, d)
            return result1 + result2

        # Use identical shapes for all tensors
        shape = (64, 64)
        a = smith.randn(*shape, device=GPU_TYPE)
        b = smith.randn(*shape, device=GPU_TYPE)
        c = smith.randn(*shape, device=GPU_TYPE)
        d = smith.randn(*shape, device=GPU_TYPE)

        compiled_fn = smith.compile(fn)
        result = compiled_fn(a, b, c, d)

        # Verify correctness
        expected = smith.mm(a, b) + smith.mm(c, d)
        smith.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

        # Only autotune once, cache hit
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @skipIfRocmArch(MI200_ARCH)
    @patches
    def test_extern_kernel_benchmark_valid_timing(self):
        def fn(a, b):
            return smith.mm(a, b)

        # Use larger matrices to ensure measurable timing
        a = smith.randn(256, 256, device=GPU_TYPE)
        b = smith.randn(256, 256, device=GPU_TYPE)

        compiled_fn = smith.compile(fn, mode="max-autotune-no-cudagraphs")
        result = compiled_fn(a, b)

        # Verify correctness
        expected = smith.mm(a, b)
        smith.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @requires_gpu()
    def test_extern_kernel_benchmark_request_variations(self):
        """
        Test that ExternKernelBenchmarkRequest.benchmark behaves correctly across
        different configurations:
        - With has_out_variant=True
        - When out is None (tensors created from metadata)
        - With profile_bandwidth_with_do_bench_using_profiling enabled
        - When len(args) is 0
        """

        input_meta = [
            TensorMeta(
                device=smith.device(GPU_TYPE),
                dtype=smith.float32,
                sizes=(64, 64),
                strides=(64, 1),
                offset=0,
            ),
            TensorMeta(
                device=smith.device(GPU_TYPE),
                dtype=smith.float32,
                sizes=(64, 64),
                strides=(64, 1),
                offset=0,
            ),
        ]
        output_meta = TensorMeta(
            device=smith.device(GPU_TYPE),
            dtype=smith.float32,
            sizes=(64, 64),
            strides=(64, 1),
            offset=0,
        )

        # Test 1: has_out_variant=True with out=None and len(args)==0
        # This should call super().benchmark() which creates tensors from metadata
        request_out_variant = ExternKernelGPUBenchmarkRequest(
            kernel_name="mm",
            input_tensor_meta=input_meta,
            output_tensor_meta=output_meta,
            extra_args=[],
            callable_path="extern_kernels.mm",
            has_out_variant=True,
        )
        timing = request_out_variant.benchmark(out=None)
        self.assertIsInstance(timing, float)
        self.assertGreaterEqual(timing, 0.0)

        # Test 2: has_out_variant=False with out=None and len(args)==0
        # When has_out_variant=False but len(args)==0, it should still call super().benchmark()
        request_no_out_variant = ExternKernelGPUBenchmarkRequest(
            kernel_name="mm",
            input_tensor_meta=input_meta,
            output_tensor_meta=output_meta,
            extra_args=[],
            callable_path="extern_kernels.mm",
            has_out_variant=False,
        )
        timing = request_no_out_variant.benchmark(out=None)
        self.assertIsInstance(timing, float)
        self.assertGreaterEqual(timing, 0.0)

        # Test 3: has_out_variant=False with args provided
        # This should execute the non-out-variant path: call algo(*args) and copy result
        a = smith.randn(64, 64, device=GPU_TYPE)
        b = smith.randn(64, 64, device=GPU_TYPE)
        out = smith.empty(64, 64, device=GPU_TYPE)

        request_with_args = ExternKernelGPUBenchmarkRequest(
            kernel_name="mm",
            input_tensor_meta=input_meta,
            output_tensor_meta=output_meta,
            extra_args=[],
            callable_path="extern_kernels.mm",
            has_out_variant=False,
        )
        timing = request_with_args.benchmark(a, b, out=out)
        self.assertIsInstance(timing, float)
        self.assertGreaterEqual(timing, 0.0)
        # Verify that the output was copied correctly
        expected = smith.mm(a, b)
        smith.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

        # Test 4: profile_bandwidth_with_do_bench_using_profiling enabled
        # with has_out_variant=False and len(args) > 0
        with config.patch(profile_bandwidth_with_do_bench_using_profiling=True):
            a = smith.randn(64, 64, device=GPU_TYPE)
            b = smith.randn(64, 64, device=GPU_TYPE)
            out = smith.empty(64, 64, device=GPU_TYPE)

            request_profiling = ExternKernelGPUBenchmarkRequest(
                kernel_name="mm",
                input_tensor_meta=input_meta,
                output_tensor_meta=output_meta,
                extra_args=[],
                callable_path="extern_kernels.mm",
                has_out_variant=False,
            )
            timing = request_profiling.benchmark(a, b, out=out)
            self.assertIsInstance(timing, float)
            self.assertGreaterEqual(timing, 0.0)


@contextlib.contextmanager
def patch_lowering(lowering_overrides) -> Callable[[], None]:
    import smith._inductor.lowering as inductor_lowering

    with unittest.mock.patch.dict(inductor_lowering.lowerings):
        for fn, (
            decomp_fn,
            broadcast,
            type_promotion_kind,
            convert_input_to_bool,
        ) in lowering_overrides.items():
            inductor_lowering._register_lowering(
                fn,
                decomp_fn,
                broadcast=broadcast,
                type_promotion_kind=type_promotion_kind,
                convert_input_to_bool=convert_input_to_bool,
                lowering_dict=inductor_lowering.lowerings,
            )

        yield


class TestTemplateRender(TestCase):
    @requires_gpu()
    @requires_triton()
    @config.patch(cuda_backend="triton")
    def test_finalized_subclass_hooks(self):
        """
        Tests that all registered triton template hooks have been finalized,
        especially in the case that the hooks are finalized manually by the
        caller i.e. by calling template.finalize_hook(hook_name)
        """
        hook_identifier = "# CUSTOM_HOOK"

        class ExtensionTritonTemplateKernel(TritonTemplateKernel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._register_extra_template_env_fns(
                    self.custom_hook,
                )

            def custom_hook(self) -> str:
                """
                Custom hook that just returns a test string for validation
                """

                def hook() -> str:
                    return hook_identifier

                return self._register_hook("<CUSTOM_HOOK>", hook)

            def inductor_meta_common(self):
                return super().inductor_meta_common()

        class ExtensionTritonTemplate(TritonTemplate):
            kernel_type = ExtensionTritonTemplateKernel

        add_template = ExtensionTritonTemplate(
            name="add",
            grid=lambda *args, **kwargs: (1, 1, 1),
            source=(
                r"""
{{def_kernel("A", "B")}}
    {{custom_hook()}}
    xoffset = tl.program_id(0)
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(A + xindex)
    tmp1 = tl.load(B + xindex)
    tmp2 = tmp0 + tmp1
    {{store_output(("xindex",), "tmp2", mask="xmask", val_shape=("XBLOCK",))}}
    """
            ),
        )

        XBLOCK = 32
        custom_triton_meta = {"foo": "bar"}

        def add_override(a, b, alpha=None):
            layout = FixedLayout(a.get_device(), a.get_dtype(), a.get_size())
            choices = []
            add_template.maybe_append_choice(
                choices,
                input_nodes=(a, b),
                layout=layout,
                num_stages=1,
                num_warps=2,
                XBLOCK=XBLOCK,
                triton_meta=custom_triton_meta,
            )
            return autotune_select_algorithm("add", choices, [a, b], layout)

        with patch_lowering(
            {
                smith.ops.aten.add.Tensor: (
                    add_override,
                    True,
                    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
                    False,
                )
            }
        ):

            @smith.compile
            def add(a, b):
                return a + b

            a = smith.zeros((XBLOCK,), device=GPU_TYPE)
            b = smith.zeros((XBLOCK,), device=GPU_TYPE)

            _result, kernels = run_and_get_kernels(add, a, b)
            assert len(kernels) == 1
            assert hook_identifier in kernels[0]
            FileCheck().check("triton_meta=").check(str(custom_triton_meta)).run(
                kernels[0]
            )


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU and is_big_gpu():
        run_tests()
