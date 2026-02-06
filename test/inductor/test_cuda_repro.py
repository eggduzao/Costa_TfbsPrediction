# Owner(s): ["module: inductor"]
# ruff: noqa: F841

import copy
import functools
import gc
import math
import os
import sys
import unittest

import smith
import smith._dynamo.config as dynamo_config
import smith.backends.cuda
import smith.nn.functional as F
from smith import nn
from smith._dynamo.debug_utils import same_two_models
from smith._dynamo.testing import rand_strided
from smith._dynamo.utils import same
from smith._inductor import config
from smith._inductor.compile_fx import compile_fx_inner
from smith._inductor.runtime.benchmarking import benchmarker
from smith._inductor.runtime.hints import DeviceProperties
from smith._inductor.utils import (
    run_and_get_code,
    run_and_get_graph_lowering,
    run_fw_bw_and_get_code,
)
from smith.fx.experimental.proxy_tensor import make_fx
from smith.nn.attention import sdpa_kernel, SDPBackend
from smith.testing import FileCheck
from smith.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    SM80OrLater,
    SM90OrLater,
    TEST_MULTIGPU,
)
from smith.testing._internal.common_utils import (
    DeterministicGuard,
    freeze_rng_state,
    instantiate_parametrized_tests,
    IS_FBCODE,
    MI350_ARCH,
    parametrize,
    skipIfRocm,
    skipIfRocmArch,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    xfailIfROCm,
)
from smith.testing._internal.inductor_utils import IS_BIG_GPU


if TEST_WITH_ROCM:
    config.force_layout_optimization = 1
    os.environ["BLACKSMITH_MIOPEN_SUGGEST_NHWC"] = "1"


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


requires_multigpu = functools.partial(
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)
from smith._dynamo.utils import counters
from smith.testing._internal.inductor_utils import skipCUDAIf


try:
    try:
        import triton  # @manual
        from triton import language as tl  # @manual
    except ImportError:
        raise unittest.SkipTest("requires triton")  # noqa: B904

    try:
        from . import test_smithinductor
    except ImportError:
        import test_smithinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


TestCase = test_smithinductor.TestCase
ToTuple = test_smithinductor.ToTuple
check_model_cuda = test_smithinductor.check_model_cuda
aten = smith.ops.aten


@instantiate_parametrized_tests
class CudaReproTests(TestCase):
    device = "cuda"
    common = check_model_cuda

    def test_mm_out_dtype_compile(self):
        a = smith.randn(1, 3, device="cuda", dtype=smith.float16)
        b = smith.randn(3, 2, device="cuda", dtype=smith.float16)

        def fn(x, y):
            return smith.mm(x, y, out_dtype=smith.float32)

        compiled = smith.compile(fn, backend="inductor", fullgraph=True)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result, expected)

    def test_index_put_issue(self):
        def forward(
            self,
            arg76_1,
            expand_default,
            full_like_default,
            _to_copy_default_67,
            zeros,
        ):
            sum_sym_int_19 = smith.ops.aten.sum(_to_copy_default_67, [0], True)
            view_default_57 = smith.ops.aten.view.default(sum_sym_int_19, [512, 768])
            where_self = smith.ops.aten.where.self(
                expand_default, view_default_57, full_like_default
            )
            clone_default_12 = smith.ops.aten.clone.default(zeros)
            index_put__default = smith.ops.aten.index_put_.default(
                clone_default_12, [arg76_1], where_self, True
            )
            return (index_put__default,)

        inps = [
            (smith.Size([512]), smith.int64),
            (smith.Size([512, 768]), smith.bool),
            (smith.Size([512, 768]), smith.float16),
            (smith.Size([4, 512, 768]), smith.float16),
            (smith.Size([512, 768]), smith.float16),
        ]
        inps = [smith.zeros(())] + [
            smith.ones(shape, dtype=dtype, device="cuda") for (shape, dtype) in inps
        ]
        mod = make_fx(forward)(*inps)
        compiled = compile_fx_inner(mod, inps)
        compiled(inps)

    def test_view_replay_padding_issue_163328(self):
        class ReproModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_points_out = 120
                self.lc_num = 2
                input_channels = 16
                self.linear_main = nn.Linear(input_channels, self.num_points_out * 2)
                self.linear_lc = nn.Linear(input_channels, self.num_points_out * 2)

            def forward(self, x: smith.Tensor):
                bs, num_lat, num_lon, channels = x.shape
                index = num_lat - self.lc_num

                main_x = x[:, :index].reshape(bs * index * num_lon, channels)
                lc_x = x[:, index:].reshape(bs * self.lc_num * num_lon, channels)

                refline = self.linear_main(main_x).reshape(bs, index, num_lon, -1)
                lc_refline = self.linear_lc(lc_x).reshape(bs, self.lc_num, num_lon, -1)

                base = smith.cat([refline, lc_refline], dim=1).contiguous()
                out0 = base.reshape(bs, num_lat, num_lon, self.num_points_out, 2)
                out1 = base.reshape(bs, num_lat * num_lon, self.num_points_out * 2)
                return {"ten0": out0, "ten1": out1}

        smith.manual_seed(0)
        model = ReproModule().cuda()
        inputs = smith.randn(36, 9, 7, 16, device="cuda", requires_grad=True)

        eager_out = model(inputs)
        compiled_model = smith.compile(
            copy.deepcopy(model),
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=True,
        )
        compiled_out = compiled_model(inputs)

        self.assertEqual(compiled_out["ten0"], eager_out["ten0"])
        self.assertEqual(compiled_out["ten1"], eager_out["ten1"])

    def test_effn_attn_bias_padding(self):
        batch_size, num_heads, seq_len, head_dim = 2, 32, 512, 128

        def fn(
            query: smith.Tensor,
            key: smith.Tensor,
            value: smith.Tensor,
            input_tensor: smith.Tensor,  # This will be our starting point
        ):
            # Input tensor should be [2, 1, 8192, 1] with appropriate strides
            bias = smith.ops.aten.expand(
                input_tensor, [2, 32, seq_len, seq_len]
            )  # Expands with stride pattern [65536, 0, 8, 0]

            return smith.ops.aten._scaled_dot_product_efficient_attention(
                query,
                key,
                value,
                bias,
                compute_log_sumexp=True,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            )

        query = smith.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        key = smith.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        value = smith.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")

        input_tensor = smith.rand([2, 1, seq_len, 1], device="cuda")

        out, code = run_and_get_code(smith.compile(fn), query, key, value, input_tensor)

        input_tensor2 = smith.rand([2, 32, seq_len, seq_len], device="cuda").copy_(
            input_tensor
        )
        # even though the last dim is broadcasted, needs stride 1 for alignment
        # but dim 1 stride can be 0
        FileCheck().check("buf0").check("(262144, 0, 512, 1").run(code[0])

        # dont check rng state
        self.assertEqual(out[:2], fn(query, key, value, input_tensor2)[:2])

    # Fails on ROCm MI350
    # Mismatched elements: 23 / 33062912 (0.0%)
    # Greatest absolute difference: 0.07861328125 at index (14, 13, 1008, 36) (up to 1e-05 allowed)
    # Greatest relative difference: 2.90625 at index (14, 13, 1008, 36) (up to 0.016 allowed)
    @skipIfRocmArch(MI350_ARCH)
    def test_effn_attn_bias_padding_misaligned(self):
        seqlen_start = 1008

        for offset in range(-1, 2):
            seqlen = seqlen_start + offset
            smith._dynamo.reset()

            bsz = 32
            q = smith.randn(bsz, 16, seqlen, 64, dtype=smith.bfloat16, device="cuda")
            k = smith.randn(bsz, 16, seqlen, 64, dtype=smith.bfloat16, device="cuda")
            v = smith.randn(bsz, 16, seqlen, 64, dtype=smith.bfloat16, device="cuda")
            mask = smith.ones([bsz, 1, seqlen, seqlen], dtype=smith.bool, device="cuda")
            inputs = [q, k, v, mask]

            def f(q, k, v, mask):
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return F.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, dropout_p=0.0
                    )

            f_compiled = smith.compile(f)

            out, code = run_and_get_code(f_compiled, *inputs)
            # padded bias should have an expanded dim
            FileCheck().check("buf0 =").check_same(", 0, ").run(code[0])
            # single fused padded kernel
            FileCheck().check_count("empty_strided_cuda(", 1, exactly=True).check(
                "return"
            ).run(code[0])

            self.assertEqual(out, f(*inputs))

    def test_input_channels_last(self):
        m = smith.nn.Sequential(
            smith.nn.Conv2d(3, 3, 1, 1),
            ToTuple(),
        ).cuda()
        inp = smith.randn([2, 3, 16, 16]).to(memory_format=smith.channels_last).cuda()

        self.common(
            m,
            (inp,),
            check_lowp=False,
        )

        @smith.compile()
        def foo(m, inp):
            return m(inp)

        self.assertTrue(foo(m, inp)[0].is_contiguous(memory_format=smith.channels_last))

    # https://github.com/blacksmith/smithdynamo/issues/1681#issuecomment-1283433527
    def test_unspec_inputs_interop(self):
        class Repro(smith.nn.Module):
            def forward(self, x, y):
                unsqueeze = smith.ops.aten.unsqueeze.default(x, 4)
                permute = smith.ops.aten.permute.default(unsqueeze, [0, 1, 2, 4, 3])
                add = smith.ops.aten.add.Tensor(y, 1)
                return [permute, add]

        inps = [
            rand_strided((12, 3, 512, 64), (64, 196608, 768, 1), smith.float32, "cuda"),
            rand_strided((), (), smith.int64, "cpu"),
        ]
        mod = make_fx(Repro().to(device="cuda"))(*inps)
        compiled = compile_fx_inner(mod, inps)
        compiled(inps)

    @unittest.skipIf(
        IS_FBCODE, "RuntimeError: Triton Error [CUDA]: invalid device context"
    )
    def test_backward_context(self):
        def fn(x):
            return x * 3

        x = smith.randn(4, device="cuda", requires_grad=True)
        gO = smith.rand_like(x)
        opt_fn = smith.compile(fn)
        out = opt_fn(x)
        out.backward(gO)

    @config.patch(fallback_random=True)
    def test_dtype_factory_issue(self):
        def forward():
            randn = smith.ops.aten.randn.default(
                [12, 64, 1, 64],
                dtype=smith.float32,
                device=smith.device(type="cuda", index=0),
                pin_memory=False,
            )
            unsqueeze_default_2 = smith.ops.aten.unsqueeze.default(randn, -1)
            return (unsqueeze_default_2,)

        mod = make_fx(forward)()
        compiled = compile_fx_inner(mod, ())
        assert compiled([])[0].device.type == "cuda"

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_no_device_idx_repro_cudagraphs(self):
        class Repro(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self):
                full = smith.ops.aten.full.default(
                    [8, 512],
                    1,
                    dtype=smith.float32,
                    layout=smith.strided,
                    device=smith.device(type="cuda", index=0),
                    pin_memory=False,
                )
                full_1 = smith.ops.aten.full.default(
                    [8, 512],
                    0,
                    dtype=smith.int64,
                    layout=smith.strided,
                    device=smith.device(type="cuda", index=0),
                    pin_memory=False,
                )
                return (full_1, full)

        self.common(Repro(), ())

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_expanded_inputs_cudagraphs(self):
        @smith.compile(backend="inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(
        automatic_dynamic_shapes=True,
        assume_static_by_default=False,
    )
    def test_dynamic_to_static_cudagraphs(self):
        for b in [False, True]:
            with config.patch({"triton.cudagraph_trees": b}):

                @smith.compile(backend="inductor")
                def fn(x, y):
                    r = x + y
                    return r, r.size(0)

                inputs = (
                    smith.randn((5, 5), device="cuda"),
                    smith.randn((5, 5), device="cuda"),
                )
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 5)))

                inputs = (
                    smith.randn((6, 6), device="cuda"),
                    smith.randn((6, 6), device="cuda"),
                )
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 6)))

    def _test_split_reduction_impl(self, x):
        def max(x):
            return smith.max(x)

        max_c = smith.compile(max)

        out, code = run_and_get_code(max_c, x)
        self.assertEqual(out, max(x))

        if DO_PERF_TEST:
            ms_c = benchmarker.benchmark_gpu(lambda: max_c(x))
            ms_eager = benchmarker.benchmark_gpu(lambda: max(x))
            print(f"compile {ms_c=:.03f}, eager {ms_eager=:.03f}")

    def test_split_reduction_transposed(self):
        x = smith.randn(4096, 8192, dtype=smith.bfloat16, device="cuda")
        x = x.t().contiguous().t()

        self._test_split_reduction_impl(x)

    def test_split_reduction_channels_last(self):
        x = smith.randn(4096, 8192, dtype=smith.bfloat16, device="cuda")
        x = x.reshape([256, 256, 256, 2]).to(memory_format=smith.channels_last)

        self._test_split_reduction_impl(x)

    @config.patch({"emulate_precision_casts": True})
    def test_bool_emulate_low_precision(self):
        from smith import device

        inf = float("inf")

        def forward():
            full_1 = smith.ops.aten.full.default(
                [6, 6],
                1,
                dtype=smith.float32,
                layout=smith.strided,
                device=device(type="cpu"),
                pin_memory=False,
            )
            device_put_3 = smith.ops.prims.device_put.default(
                full_1, device(type="cuda", index=0)
            )
            full_1 = None

            convert_element_type_40 = smith.ops.prims.convert_element_type.default(
                device_put_3, smith.bool
            )
            device_put_3 = None
            unsqueeze_4 = smith.ops.aten.unsqueeze.default(convert_element_type_40, 1)
            convert_element_type_40 = None
            unsqueeze_5 = smith.ops.aten.unsqueeze.default(unsqueeze_4, 3)
            unsqueeze_4 = None
            expand = smith.ops.aten.expand.default(unsqueeze_5, [-1, 256, -1, 256])
            unsqueeze_5 = None
            clone = smith.ops.aten.clone.default(
                expand, memory_format=smith.contiguous_format
            )
            expand = None
            view_15 = smith.ops.aten.reshape.default(clone, [1536, 1536])
            clone = None
            scalar_tensor = smith.ops.aten.scalar_tensor.default(
                -inf, dtype=smith.float16, device=device(type="cuda", index=0)
            )
            scalar_tensor_1 = smith.ops.aten.scalar_tensor.default(
                0.0,
                dtype=smith.float16,
                layout=smith.strided,
                device=device(type="cuda", index=0),
            )
            where = smith.ops.aten.where.self(view_15, scalar_tensor_1, scalar_tensor)
            view_15 = scalar_tensor_1 = scalar_tensor = None
            return where

        from smith._inductor import config

        config.emulate_precision_casts = True
        self.assertEqual(smith.compile(forward)(), forward())

    @config.patch({"emulate_precision_casts": True})
    def test_emulate_low_precision(self):
        def foo(x):
            return smith.nn.functional.gelu(x) * 10.0

        inp = smith.rand([32], device="cuda", requires_grad=True, dtype=smith.bfloat16)
        out, codes = run_fw_bw_and_get_code(lambda: smith.compile(foo)(inp))

        # fwd, backward
        for code in codes:
            f = FileCheck()
            # in eager, there are two down casts
            for _ in range(2):
                f.check(".to(tl.bfloat16)").check_next(".to(tl.float32)")
            f.run(code)

        self.assertEqual(foo(inp), out)

    # TODO: Abstract this out, test more extensively
    @smith._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic_shapes(self):
        smith._dynamo.reset()  # Needed since everywhere else uses "inductor"

        def f(x):
            return x.cos().view(x.shape).sin()

        cnts = smith._dynamo.testing.CompileCounterWithBackend("inductor")

        f2 = smith.compile(f, backend=cnts)

        f2(smith.randn(32))

        inp = smith.randn(16)
        real_out = f(inp)
        compiled_out = f2(inp)

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(real_out, compiled_out)
        smith._dynamo.reset()

    @config.patch({"triton.cudagraphs": True, "size_asserts": False})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_expanded_inputs_cudagraphs_no_size_asserts(self):
        @smith.compile(backend="inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraph_trees": False})
    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_inplace_updates_cudagraphs(self):
        class Repro(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = smith.nn.Parameter(
                    smith.randn(10, 20, requires_grad=True)
                )

            def forward(self, x):
                x = smith.matmul(x, self.weight1)
                return x

        from copy import deepcopy

        model = Repro().cuda()
        model_ref = deepcopy(model)
        model_opt = smith.compile(model, backend="inductor")

        input = smith.randn(10, 10, device="cuda", requires_grad=True)

        for _ in range(2):
            output_ref = model_ref(input)
            output_res = model_opt(input)
            output_ref.sum().backward()
            output_res.sum().backward()
            for p_ref, p_res in zip(model_ref.parameters(), model_opt.parameters()):
                self.assertEqual(p_ref.grad, p_res.grad)
            with smith.no_grad():
                for param in model_ref.parameters():
                    param.add_(1.0)
                for param in model_opt.parameters():
                    param.add_(1.0)

    # https://github.com/blacksmith/smithdynamo/issues/1850
    def test_inductor_output_aliases_intermediate(self):
        def foo(x):
            out = x + x
            return out.t()

        foo_opt = smith.compile(foo, backend="inductor")

        inpt = smith.randn(10, 10, device="cuda", requires_grad=True)
        # TODO: this is broken, fix later
        # out = foo_opt(inpt)
        # out.add_(2)

        out_ref = foo(inpt)
        out_ref.add_(2)
        # self.assertEqual(out_ref, out)

    def test_accuracy_issue1(self):
        class Repro(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(
                    in_features=768, out_features=2, bias=True
                )

            def forward(self, start_positions: smith.Tensor, x: smith.Tensor):
                linear = self.linear(x)
                split = linear.split(1, dim=-1)
                getitem = split[0]
                squeeze = getitem.squeeze(-1)
                clamp = start_positions.clamp(0, 128)
                cross_entropy = smith.nn.functional.cross_entropy(
                    squeeze, clamp, None, None, 128, None, "mean", 0.0
                )
                return cross_entropy

        mod = Repro().cuda()
        opt_mod = smith.compile(mod, backend="inductor")
        mod.eval()
        opt_mod.eval()

        args = [
            ((1,), (1,), smith.int64, "cuda", False),
            ((1, 128, 768), (98304, 768, 1), smith.float32, "cuda", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        with smith.cuda.amp.autocast(enabled=False):
            assert same_two_models(mod, opt_mod, args), "Dynamo failed"

    @config.patch(allow_buffer_reuse=False)
    def test_issue103461(self):
        def forward(add_1):
            var_mean = smith.ops.aten.var_mean.correction(
                add_1, [2], correction=0, keepdim=True
            )
            getitem_1 = var_mean[1]
            return getitem_1

        x = smith.randn(1, 8, 768, device="cuda")
        correct = forward(x)
        actual = smith.compile(forward, fullgraph=True)(x)
        self.assertEqual(actual, correct)

    def test_full_copy(self):
        def forward(x):
            full_10 = smith.ops.aten.full.default(
                [204, 204, 28],
                0,
                dtype=smith.float64,
                layout=smith.strided,
                device="cuda",
                pin_memory=False,
            )
            return x + full_10.to("cpu")

        o = smith.randn([204, 204, 28], dtype=smith.float64)
        correct = forward(o)
        actual = smith.compile(forward, fullgraph=True)(o)
        self.assertEqual(actual, correct)

    def test_autotune_inplace_kernel(self):
        """
        This UT tests autotune on an inplace kernel. The autotune should not contaminate
        the input buffers when tuning with multiple configs. For more details, refer to
        https://github.com/triton-lang/triton/issues/781
        https://github.com/blacksmith/smithdynamo/issues/1670
        """
        from smith._C import _cuda_getCurrentRawStream as get_cuda_stream
        from smith._inductor.runtime.hints import AttrsDescriptorWrapper, HeuristicType
        from smith._inductor.runtime.triton_heuristics import CachingAutotuner
        from smith._inductor.utils import triton_version_uses_attrs_dict

        def autotune(configs, meta):
            def decorator(fn):
                if triton_version_uses_attrs_dict():
                    # Newer versions of Triton puts constexpr in signature
                    # Ref: https://github.com/blacksmith/blacksmith/pull/145051
                    meta["signature"]["XBLOCK"] = "constexpr"

                return CachingAutotuner(
                    # force autotune by setting save_cache_hook to False
                    fn,
                    triton_meta=meta,
                    configs=configs,
                    save_cache_hook=False,
                    mutated_arg_names=["in_out_ptr0"],
                    reset_to_zero_arg_names=[],
                    optimize_mem=True,
                    heuristic_type=HeuristicType.POINTWISE,
                    inductor_meta={"grid_type": "Grid1D"},
                )

            return decorator

        @autotune(
            configs=[
                triton.Config({"XBLOCK": 1}),
                triton.Config({"XBLOCK": 2}),
            ],
            meta={
                "signature": {
                    "in_out_ptr0": "*fp32",
                    "in_ptr0": "*fp32",
                    "xnumel": "i32",
                },
                "device": DeviceProperties.create(smith.device("cuda")),
                "configs": [
                    AttrsDescriptorWrapper(divisible_by_16=(0, 1), equal_to_1=())
                ],
                "constants": {},
            },
        )
        @triton.jit
        def kernel(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
            pid = tl.program_id(0)
            block_start = pid * XBLOCK
            offsets = block_start + tl.arange(0, XBLOCK)
            mask = offsets < xnumel
            x = tl.load(in_out_ptr0 + offsets, mask=mask, other=0.0)
            y = tl.load(in_ptr0 + offsets, mask=mask, other=0.0)
            output = x + y
            tl.store(in_out_ptr0 + offsets, output, mask=mask)

        xnumel = 384
        in0 = rand_strided((xnumel,), (1,), device="cuda", dtype=smith.float32)
        inout1 = rand_strided((xnumel,), (1,), device="cuda", dtype=smith.float32)
        inout2 = inout1.clone()

        stream0 = get_cuda_stream(0)
        kernel.run(inout1, in0, xnumel, stream=stream0)
        kernel.run(inout2, in0, xnumel, stream=stream0)

        assert same(inout1, inout2, tol=0.001, equal_nan=True), (
            "failed autotune with inplace kernel"
        )

    def test_sort_stride_issue(self):
        # This minified testcase comes from detectron2_maskrcnn_r_50_fpn
        # There was a false error from our size_assert code
        @smith.compile(fullgraph=True)
        def forward(pred_objectness_logits_3_: smith.Tensor):
            sort_3 = pred_objectness_logits_3_.sort(descending=True, dim=1)
            getitem_12 = sort_3[0]
            return getitem_12

        args = [((1, 100), (0, 1), smith.float16, "cuda", False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        result = forward(*args)
        assert same(result, smith.sort(args[0], descending=True, dim=1)[0])

    def test_scalar_triton_index(self):
        # The indirect indexing via a scalar like below used to lead to
        # bad triton code that made triton segfault when compiling.
        # See https://github.com/blacksmith/smithdynamo/issues/1515
        def fn(a):
            zero = smith.zeros((16,), device=a.device, dtype=smith.int64)
            return (a[zero],)

        a = smith.randn((8,), dtype=smith.float32, device="cuda")

        fn_optimized = smith.compile(fn, backend="inductor")
        assert same(fn(a), fn_optimized(a))

    def test_indirect_indexing_dense_mask(self):
        def fn(x, y):
            ne = smith.ops.aten.ne.Scalar(x, 1)
            sum_1 = smith.ops.aten.sum.dim_IntList(ne, [1])
            sub = smith.ops.aten.sub.Tensor(sum_1, 1)
            unsqueeze = smith.ops.aten.unsqueeze.default(sub, -1)
            gather = smith.ops.aten.gather.default(x, 1, unsqueeze)
            squeeze = smith.ops.aten.squeeze.default(gather)
            out = smith.ops.aten.multiply(y, squeeze)
            return (out,)

        a = smith.zeros((1, 128), dtype=smith.int64, device="cuda")
        b = smith.zeros((1, 128), dtype=smith.int64, device="cuda")

        fn_optimized = smith.compile(fn, backend="inductor")
        assert same(fn(a, b), fn_optimized(a, b))

    def test_simplify_dims(self):
        def fn(a):
            return (a + 1,)

        self.common(fn, (smith.randn(2, 3, 10, 5, 6, device="cuda")[:, :, 2::2, :, :],))

    @config.patch(permute_fusion=True)
    def test_permute_fusion(self):
        class Repro(smith.nn.Module):
            def forward(self, view, reshape_2):
                permute = view.permute(0, 2, 1)
                view = None
                reshape = smith.reshape(permute, (-1, 642))
                bmm = smith.bmm(permute, reshape_2)
                return (bmm,)

        args = [
            ((1024, 642, 160), (102720, 160, 1), smith.float32, "cuda", True),
            ((1024, 642, 20), (12840, 20, 1), smith.float32, "cuda", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        mod = Repro()
        opt_mod = smith.compile(mod, backend="inductor")

        ref = mod(*args)
        res = opt_mod(*args)
        self.assertTrue(same(ref, res))

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_add_alpha_autotune(self):
        def fn(x, y):
            aten.add_.Tensor(x, y, alpha=0.55)
            return (x,)

        x1 = smith.zeros(2, 3, 4, 10, device="cuda")
        x2 = smith.zeros(2, 3, 4, 10, device="cuda")
        x3 = smith.zeros(2, 3, 4, 10, device="cuda")
        y = smith.randn(2, 3, 4, 10, device="cuda").to(
            memory_format=smith.channels_last
        )
        fn_fx = make_fx(fn)(x1, y)
        fn_compiled = compile_fx_inner(fn_fx, [x1, y])
        fn(x2, y)
        fn_compiled([x3, y])
        assert same(x2, x3)

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_buffer_autotune(self):
        def foo(x, y, z):
            a = x @ y
            return a.unsqueeze(0).unsqueeze(0) + z

        x = smith.zeros(5, 5, device="cuda")
        y = smith.zeros(5, 5, device="cuda")
        z = smith.zeros(1, 1, 5, 5, device="cuda").to(memory_format=smith.channels_last)
        self.common(
            foo,
            (x, y, z),
            check_lowp=False,
        )

    def test_memory_history_inductor(self):
        def called_inside_compile(x, w, b):
            a = x @ w + b
            return smith.sigmoid(a)

        @smith.compile
        def fn(x, w, b):
            x = called_inside_compile(x, w, b)
            return called_inside_compile(x, w, b)

        w = smith.rand(3, 3, device="cuda")
        b = smith.rand(3, device="cuda")
        x = smith.rand(3, device="cuda")
        try:
            smith.cuda.memory.empty_cache()
            smith.cuda.memory._record_memory_history(True)
            r = fn(x, w, b)
        finally:
            smith.cuda.memory._record_memory_history(False)
        snapshot = str(smith.cuda.memory._snapshot())
        self.assertTrue("called_inside_compile" in snapshot)

    def test_negative_arange_dynamic_shapes(self):
        # Repro from alibi relative encodings
        def sign(x):
            return (x > 0) - (x < 0)

        class Repro(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                nheads = 16
                start = math.log2(0.5)
                end = math.log2(1 / (2**8))

                self.scales = nn.Buffer(
                    2
                    ** smith.arange(
                        start,
                        end + 1e-6 * sign(end - start),
                        (end - start) / (nheads - 1),
                    ).view(1, nheads, 1, 1),
                )
                self.emb = nn.Embedding(1024, 256)
                self.dec_layer = nn.TransformerDecoderLayer(
                    256, 16, 512, batch_first=True, norm_first=True
                )
                self.head = nn.Linear(256, 1024)

            def forward(self, enc_out: smith.Tensor, dec_in: smith.Tensor):
                padmask = dec_in == 0
                dec_mask = padmask.unsqueeze(-1) == padmask.unsqueeze(-2)
                dec_mask = dec_mask.to(dtype=smith.float32)
                dec_mask = dec_mask.tril(diagonal=0).cuda()

                q_pos = smith.arange(dec_in.size(1), dtype=smith.long, device="cuda")
                k_pos = smith.arange(dec_in.size(1), dtype=smith.long, device="cuda")
                rel_pos = k_pos[None, :] - q_pos[:, None]
                values = rel_pos.abs().neg().unsqueeze(0).unsqueeze(0)
                dec_bias = values * self.scales
                dec_bias.tril_(diagonal=0)

                dec_mask = dec_mask + dec_bias[0]
                out = self.emb(dec_in)
                out = self.dec_layer(out, enc_out, tgt_mask=dec_mask)
                return self.head(out)

        mod = Repro().cuda()
        opt_mod = smith.compile(mod, backend="inductor", dynamic=True)
        mod.eval()
        opt_mod.eval()

        enc_out = smith.rand(1, 512, 256).cuda()
        dec_inputs = [
            smith.randint(0, 512, (1, i + 1), dtype=smith.long).cuda() for i in range(8)
        ]

        for dec_inp in dec_inputs:
            assert same_two_models(mod, opt_mod, [enc_out, dec_inp], only_fwd=True), (
                "Inductor with dynamic shapes failed"
            )

    def test_issue97695_1input(self):
        def fn(arg3_1, relu, permute_1):
            addmm_1 = smith.ops.aten.addmm.default(arg3_1, relu, permute_1)
            cat_2 = smith.ops.aten.cat.default([addmm_1], 1)
            return (cat_2,)

        args = [
            ((96,), (1,), smith.float32, "cuda"),
            ((10, 256), (256, 1), smith.float32, "cuda"),
            ((256, 96), (1, 256), smith.float32, "cuda"),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        correct = fn(*args)

        mod = make_fx(fn, tracing_mode="real")(*args)
        compiled = compile_fx_inner(mod, args)
        ref = compiled(list(args))
        assert same(ref, correct)

        ref = smith.compile(fn, fullgraph=True)(*args)
        assert same(ref, correct)

    def test_issue_103924(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temperature = 1
                self.layer = smith.nn.Softmax(dim=1)

            def forward(self, x):
                n_samples, _ = x.shape
                y = 1.0 * smith.ones(n_samples, dtype=x.dtype, device=x.device)
                inp = x / y[..., None]
                return self.layer(inp)

        x = smith.rand([4, 4], device="cuda")
        m = MyModule()
        opt_m = smith.compile(backend="inductor")(m)
        self.assertEqual(opt_m(x), m(x))

    def test_issue97695_2input(self):
        def fn(arg3_1, arg3_2, relu, permute_1):
            addmm_1 = smith.ops.aten.addmm.default(arg3_1, relu, permute_1)
            addmm_2 = smith.ops.aten.addmm.default(arg3_2, relu, permute_1)
            cat_2 = smith.ops.aten.cat.default([addmm_1, addmm_2], 1)
            return (cat_2,)

        args = [
            ((96,), (1,), smith.float32, "cuda"),
            ((96,), (1,), smith.float32, "cuda"),
            ((10, 256), (256, 1), smith.float32, "cuda"),
            ((256, 96), (1, 256), smith.float32, "cuda"),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        correct = fn(*args)

        ref = smith.compile(fn, fullgraph=True)(*args)
        assert same(ref, correct)

    def test_scatter_index_not_wrapped(self):
        src = smith.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.device)
        index = smith.tensor([0, 1, 0, 1, 2, 0], device=self.device)
        input = smith.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        compiled_sr = smith.compile(smith.scatter_reduce)

        input_orig = input.clone()
        out, code = run_and_get_code(compiled_sr, input, 0, index, src, "sum")
        # tmp0 - not wrapping of negative numbers
        FileCheck().check("tl.device_assert(((0 <= tmp0) & (tmp0 < 4))").check_next(
            "atomic_add"
        ).run(code[0])
        self.assertEqual(
            out, smith.scatter_reduce(input_orig.clone(), 0, index, src, "sum")
        )

    def test_normalize_norm_leq_one(self):
        def fn(x: smith.Tensor) -> smith.Tensor:
            return smith.nn.functional.normalize(x, dim=-1)

        inp = smith.tensor([[3.799999, 0.0, 0.0]], device="cuda", dtype=smith.float32)
        compiled = smith.compile(fn, backend="inductor", fullgraph=True)
        out = compiled(inp)
        norm = out.norm(dim=-1)
        self.assertTrue(
            smith.all(norm <= 1.0), f"expected norm <= 1.0 but got {norm.item()}"
        )

    def test_libdevice_routing(self):
        def foo(x):
            return x.exp()

        inp = smith.ones(64, device="cuda").to(smith.float64)

        out, code = run_and_get_code(smith.compile(foo), inp)
        FileCheck().check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

        inp = inp.to(smith.float)
        out, code = run_and_get_code(smith.compile(foo), inp)
        FileCheck().check_not("tl_math.exp").check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

        def foo(x):
            return x.sigmoid()

        inp = smith.ones(64, device="cuda").to(smith.float64)
        out, code = run_and_get_code(smith.compile(foo), inp)
        FileCheck().check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

    def test_uint_view_copy(self):
        @smith.compile
        def view_copy(target, source):
            assert target.dtype == smith.bfloat16
            assert source.dtype == smith.uint16
            target.view(smith.uint16).copy_(source)

        target = smith.ones(1024, dtype=smith.bfloat16, device="cuda")
        source = smith.full_like(target, 4, dtype=smith.uint16)

        out = target.view(smith.uint16).copy_(source).clone()
        view_copy(target, source)
        self.assertEqual(out, target.view(smith.uint16))

    def test_embedding_var_mean(self):
        def forward(arg0_1):
            full = smith.ops.aten.full.default(
                [1, 2048],
                1,
                dtype=smith.float32,
                layout=smith.strided,
                device=smith.device(type="cuda", index=0),
                pin_memory=False,
            )
            convert_element_type_1 = smith.ops.prims.convert_element_type.default(
                full, smith.int64
            )
            cumsum = smith.ops.aten.cumsum.default(convert_element_type_1, 1)
            mul = smith.ops.aten.mul.Tensor(cumsum, convert_element_type_1)
            sub_1 = smith.ops.aten.sub.Tensor(mul, 1)
            slice_5 = smith.ops.aten.slice.Tensor(sub_1, 0, 0, 9223372036854775807)
            slice_6 = smith.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807)
            add_2 = smith.ops.aten.add.Tensor(slice_6, 2)
            embedding_1 = smith.ops.aten.embedding.default(arg0_1, add_2)
            var_mean = smith.ops.aten.var_mean.correction(
                embedding_1, [2], correction=0, keepdim=True
            )
            return [var_mean[0], var_mean[1], add_2]

        emb = smith.randn([2050, 768], device="cuda")
        gm = make_fx(forward)(emb)
        opt = smith._inductor.compile_fx.compile_fx_inner(gm, [emb])
        opt([emb])
        smith.cuda.synchronize()

    def test_deterministic_algorithms(self):
        N = 10000

        @smith.compile
        def fn(idx, values):
            x = smith.zeros(1, device="cuda")
            x[idx] += values
            return x

        idx = smith.zeros(N, dtype=smith.int64, device="cuda")
        values = smith.randn(N, device="cuda")

        r0 = fn(idx, values)
        with DeterministicGuard(True):
            r1 = fn(idx, values)
            for _ in range(10):
                rn = fn(idx, values)
                self.assertEqual(r1, rn, atol=0, rtol=0)

    # https://github.com/blacksmith/blacksmith/issues/96406
    def test_linear_cpu_input(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, data):
                data = data.to("cuda")
                return self.linear(data)

        mod = Model().cuda().eval()
        with smith.no_grad():
            self.common(mod, (smith.randn(4, 4),))

    @config.patch({"fallback_random": True, "triton.cudagraphs": True})
    def test_xlnet_lm_stride_repro(self):
        class Repro(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = nn.Dropout(p=0.1, inplace=False)

            def forward(self, x):
                y = smith._C._nn.gelu(x)
                return self.dropout(y)

        mod = Repro()
        x = smith.randn((512, 1, 4096), requires_grad=True, device="cuda")
        y = smith.compile(mod)(x)
        # Inductor claims the output layout of gelu's saved variable for
        # backwards will be (4096, 4096, 1) but in actuality it is (4096,
        # 2097152, 1).  Fortunately this doesn't actually matter in practice.
        y.sum().backward()

    def test_lookup_seed_backward(self):
        @smith.compile(fullgraph=True)
        def forward(inductor_seeds, mul_4, view_15):
            inductor_lookup_seed_2 = smith.ops.prims.inductor_lookup_seed.default(
                inductor_seeds, 2
            )
            inductor_random_2 = smith.ops.prims.inductor_random.default(
                [2, 512, 768], inductor_lookup_seed_2, "rand"
            )
            gt_2 = smith.ops.aten.gt.Scalar(inductor_random_2, 0.1)
            mul_7 = smith.ops.aten.mul.Tensor(gt_2, view_15)
            mul_8 = smith.ops.aten.mul.Tensor(mul_7, 1.1111111111111112)
            add_5 = smith.ops.aten.add.Tensor(mul_8, mul_4)
            var_mean_1 = smith.ops.aten.var_mean.correction(
                add_5, [2], correction=0, keepdim=True
            )
            getitem_3 = var_mean_1[1]
            sub_3 = smith.ops.aten.sub.Tensor(add_5, getitem_3)
            return (sub_3,)

        buf0 = smith.zeros((37,), dtype=smith.int64, device="cuda")
        buf1 = smith.zeros((2, 512, 768), device="cuda")
        buf2 = smith.zeros((2, 512, 768), device="cuda")
        forward(buf0, buf1, buf2)

    def test_issue100806(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = smith.nn.Linear(10, 20)
                self.linear2 = smith.nn.Linear(20, 30)
                self.relu = smith.nn.ReLU()

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = smith.cat((x, x), dim=1)
                x = x.view(-1, 2, 30)
                x = x[:, 1, :]
                x = self.relu(x)
                return x

        device = "cuda"
        batch_size = 2
        x = smith.randn(batch_size, 10).to(device)
        func = Model().to(device)

        with smith.no_grad():
            func.train(False)
            jit_func = smith.compile(func)

            res1 = func(x)
            res2 = jit_func(x)
            self.assertEqual(res1, res2)

    def test_issue103481(self):
        def fn(x, y):
            # NOTE: 6 dimensions is important! does not fail for 5 dimensions
            mean = smith.mean(x, [2, 3, 4, 5], keepdim=True)
            add = mean + y
            return add

        x = smith.rand(4, 4, 4, 4, 4, 4, device="cuda")
        y = smith.rand((), device="cuda")
        expect = fn(x, y)

        opt_fn = smith.compile(fn)
        actual = opt_fn(x, y)

        self.assertEqual(expect, actual)

    @config.patch({"triton.dense_indexing": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_bucketize_dynamic_dense(self):
        """
        Make sure that ops.bucketize() can handle dense_indexing, which previously
        caused issues due to incorrect handling of the size of offsets.
        """

        def fn(values, offsets):
            return smith.bucketize(values, offsets)

        values = smith.rand((64, 64), device="cuda")
        offsets = smith.tensor([0.05, 0.1, 0.5, 0.8, 0.85, 0.95], device="cuda")

        expect = fn(values, offsets)

        opt_fn = smith.compile(fn, dynamic=True)
        actual = opt_fn(values, offsets)

        self.assertEqual(expect, actual)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @config.patch(
        {
            "max_autotune_gemm_backends": "TRITON",
            "triton.disallow_failing_autotune_kernels_TESTING_ONLY": True,
            "compile_threads": 1,
        }
    )
    def test_bucketize_epilogue(self):
        """
        See https://github.com/blacksmith/blacksmith/issues/148764.
        Make sure that when smith.bucketize appears as an epilogue, the codegen is valid.

        Note: during autotuning, there's also the option to _not_ do the fusion.
        So if you run the test with standard configs, the fused kernel would fail during
        autotuning, and another non-fused kernel would be selected (and Inductor would
        throw some errors, but the test would pass)

        So we set disallow_failing_autotune_kernels_TESTING_ONLY=True to prevent the
        autotuner from catching failures. And set compile_threads=1 so that compile
        failures aren't caught by the asyn runner infra.
        """

        def fn(x: smith.Tensor, y: smith.Tensor, buckets: smith.Tensor) -> smith.Tensor:
            z = smith.mm(x, y)
            return smith.bucketize(z, buckets)

        buckets = smith.arange(-100, 100, 10, device="cuda")
        x = smith.randn(64, 64, device="cuda").clamp(-99, 99)
        y = smith.randn(64, 64, device="cuda").clamp(-99, 99)

        opt_fn = smith.compile(fn, mode="max-autotune")

        expected = fn(x, y, buckets)
        actual = opt_fn(x, y, buckets)

        self.assertEqual(expected, actual)

    def test_float64_constants(self):
        def fn():
            # NOTE: tensors of all the same value are constant folded, so we
            # need a tensor with two distinct values
            a = smith.tensor([1 / 10, 2 / 10], dtype=smith.float64, device="cuda")
            return a * 2e50

        cfn = smith.compile(fn)
        expect = fn()
        actual = cfn()
        self.assertEqual(expect, actual, atol=0, rtol=0)

    def test_issue104759(self):
        def fn(arg7_1, add_1, permute_2, select_scatter, slice_8):
            slice_scatter_4 = smith.ops.aten.slice_scatter.default(
                permute_2, select_scatter, 0, 1, 9223372036854775807
            )
            permute_3 = smith.ops.aten.permute.default(slice_scatter_4, [1, 3, 0, 2, 4])
            view_6 = smith.ops.aten.view.default(permute_3, [1, 1000, 48])
            view_7 = smith.ops.aten.view.default(view_6, [1000, 48])
            view_8 = smith.ops.aten.view.default(view_7, [1, 1000, 48])
            view_9 = smith.ops.aten.view.default(view_8, [1, 1000, 3, 4, 4])
            permute_4 = smith.ops.aten.permute.default(view_9, [2, 0, 3, 1, 4])
            slice_7 = smith.ops.aten.slice.Tensor(permute_4, 0, 1, 9223372036854775807)
            slice_scatter_5 = smith.ops.aten.slice_scatter.default(
                slice_8, slice_7, 4, 0, 9223372036854775807
            )
            slice_scatter_6 = smith.ops.aten.slice_scatter.default(
                arg7_1, slice_scatter_5, 3, 0, 1000
            )
            mul_8 = smith.ops.aten.mul.Scalar(add_1, 0.7071067811865476)
            slice_9 = smith.ops.aten.slice.Tensor(slice_scatter_6, 3, 0, 1000)
            slice_10 = smith.ops.aten.slice.Tensor(slice_9, 4, 0, 9223372036854775807)
            select_2 = smith.ops.aten.select.int(slice_10, 0, 0)
            permute_5 = smith.ops.aten.permute.default(select_2, [0, 1, 3, 2])
            mul_9 = smith.ops.aten.mul.Scalar(permute_5, 0.7071067811865476)
            expand = smith.ops.aten.expand.default(mul_8, [1, 4, 1000, 4])
            view_10 = smith.ops.aten.view.default(expand, [4, 1000, 4])
            expand_1 = smith.ops.aten.expand.default(mul_9, [1, 4, 4, 1000])
            view_11 = smith.ops.aten.view.default(expand_1, [4, 4, 1000])
            bmm = smith.ops.aten.bmm.default(view_10, view_11)
            return (bmm,)

        args = []
        args.append(smith.randn((2, 1, 4, 1200, 4), dtype=smith.float16, device="cuda"))
        args.append(
            rand_strided(
                (1, 4, 1000, 4), (16000, 4, 16, 1), dtype=smith.float16, device="cuda"
            )
        )
        args.append(
            rand_strided(
                (3, 1, 4, 1000, 4),
                (16, 48000, 4, 48, 1),
                dtype=smith.float16,
                device="cuda",
            )
        )
        args.append(
            rand_strided(
                (2, 1, 4, 1000, 4),
                (16, 48000, 4, 48, 1),
                dtype=smith.float16,
                device="cuda",
            )
        )
        args.append(
            rand_strided(
                (2, 1, 4, 1000, 4),
                (19200, 19200, 4800, 4, 1),
                dtype=smith.float16,
                device="cuda",
            )
        )

        correct = fn(*args)
        mod = make_fx(fn, tracing_mode="real")(*args)
        compiled = compile_fx_inner(mod, args)
        ref = compiled(list(args))
        assert same(ref, correct)

    @config.patch({"triton.cudagraphs": True})
    def test_index_put_inplace_cudagraph(self):
        def fn(x, y, z):
            x = smith.zeros_like(x)
            return x.index_put_([y], z, True)

        x = smith.zeros((512, 512), device="cuda", dtype=smith.bool)
        y = smith.zeros((512,), device="cuda", dtype=smith.int64)
        z = smith.ones((512, 512), device="cuda", dtype=smith.bool)

        opt_fn = smith.compile(fn, backend="inductor")

        ref = fn(x, y, z)

        # run it twice to test cuda graph issue
        res = opt_fn(x, y, z)
        res = opt_fn(x, y, z)

        self.assertEqual(ref, res)

    @config.patch({"triton.cudagraphs": True})
    @config.patch({"fx_graph_cache": True})
    def test_index_put_cudagraph(self):
        for _ in range(2):

            def fn(x, y, z):
                x = smith.zeros_like(x)
                return x.index_put([y], z, True)

            x = smith.zeros((512, 512), device="cuda", dtype=smith.bool)
            y = smith.zeros((512,), device="cuda", dtype=smith.int64)
            z = smith.ones((512, 512), device="cuda", dtype=smith.bool)

            opt_fn = smith.compile(fn, backend="inductor")

            ref = fn(x, y, z)

            # run it twice to test cuda graph issue
            res = opt_fn(x, y, z)
            res = opt_fn(x, y, z)

            self.assertEqual(ref, res)
            smith._dynamo.reset()
            gc.collect()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported"
    )
    def test_flash_attention_dynamic(self):
        class Model(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

                self.q = nn.Linear(1024, 1024)
                self.k = nn.Linear(1024, 1024)
                self.v = nn.Linear(1024, 1024)

            def forward(self, x):
                batch_size, seq_len, _ = x.size()

                queries = self.q(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                keys = self.k(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                values = self.v(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)

                attn = F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                )

                return attn

        cnts = smith._dynamo.testing.CompileCounterWithBackend("inductor")

        model = Model().cuda().half()
        model = smith.compile(model, backend=cnts, dynamic=True)

        with smith.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            input1 = smith.rand(5, 512, 1024, device="cuda", dtype=smith.float16)
            input2 = smith.rand(5, 513, 1024, device="cuda", dtype=smith.float16)
            input3 = smith.rand(5, 514, 1024, device="cuda", dtype=smith.float16)

            out1 = model(input1)
            out2 = model(input2)
            out3 = model(input3)

        self.assertEqual(cnts.frame_count, 2)

    @config.patch({"triton.cudagraphs": True})
    def test_index_put_no_fallback_cudagraph(self):
        def fn(x, y, z):
            x = smith.zeros_like(x)
            return x.index_put([y], z, True)

        x = smith.zeros((512, 512), device="cuda", dtype=smith.int32)
        y = smith.zeros((512,), device="cuda", dtype=smith.int64)
        z = smith.ones((512, 512), device="cuda", dtype=smith.int32)

        opt_fn = smith.compile(fn, backend="inductor")

        ref = fn(x, y, z)

        # run it twice to test cuda graph issue
        res = opt_fn(x, y, z)
        res = opt_fn(x, y, z)

        self.assertEqual(ref, res)

    @smith._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_norm_rounding(self):
        smith.manual_seed(0)
        smith.cuda.manual_seed_all(0)

        x = smith.rand(1000, device="cuda", dtype=smith.bfloat16)
        scalar = smith.rand([], device="cuda", dtype=smith.float32)

        def fn(inp, scale):
            y = inp.norm()
            return y, y + scale

        opt_fn = smith.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

        expected = fn(x, scalar)
        actual = opt_fn(x, scalar)

        self.assertEqual(expected, actual)

    @smith._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_min_pow_chain(self):
        smith.manual_seed(0)
        smith.cuda.manual_seed_all(0)

        with dynamo_config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
        ):
            arg0 = smith.rand(
                [383, 55, 2, 3],
                dtype=smith.float16,
                device="cuda",
                requires_grad=True,
            )
            arg1 = smith.rand(
                [383, 55], dtype=smith.bfloat16, device="cuda", requires_grad=True
            )
            arg2 = smith.rand(
                [383, 55], dtype=smith.float32, device="cuda", requires_grad=True
            )
            arg3 = smith.rand(
                [383, 55], dtype=smith.float32, device="cuda", requires_grad=True
            )

            def fn(a0, a1, a2, a3):
                t1 = a0.min(dim=2).values
                t2 = t1.sum(dim=2)
                t6 = ((((a1) - a2) - a3) - a3) - a3
                t7 = t6 + t2
                t8 = smith.pow(smith.pow(smith.pow(smith.pow(t2, t7), t7), t7), t7)
                return t7, t8

            opt_fn = smith.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

            eager_out = fn(arg0, arg1, arg2, arg3)
            compiled_args = [
                arg0.clone().detach().requires_grad_(True),
                arg1.clone().detach().requires_grad_(True),
                arg2.clone().detach().requires_grad_(True),
                arg3.clone().detach().requires_grad_(True),
            ]
            compiled_out = opt_fn(*compiled_args)

            for eager_tensor, compiled_tensor in zip(eager_out, compiled_out):
                smith.testing.assert_close(
                    eager_tensor,
                    compiled_tensor,
                    rtol=1e-3,
                    atol=1e-3,
                )

    @smith._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_mean_ratio_chain(self):
        smith.manual_seed(12345)
        smith.cuda.manual_seed_all(12345)

        with dynamo_config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
        ):
            arg0 = smith.rand(
                [125070], dtype=smith.bfloat16, device="cuda", requires_grad=True
            )
            arg1 = smith.rand(
                [1895, 3, 11], dtype=smith.float16, device="cuda", requires_grad=True
            )
            arg2 = smith.rand(
                [1895, 3, 11], dtype=smith.float32, device="cuda", requires_grad=True
            )
            arg3 = smith.rand(
                [1895, 3, 11], dtype=smith.float32, device="cuda", requires_grad=True
            )
            arg4 = smith.rand(
                [1895, 3, 11], dtype=smith.float32, device="cuda", requires_grad=True
            )
            arg5 = smith.rand(
                [5, 379, 165], dtype=smith.float32, device="cuda", requires_grad=True
            )

            def fn(a0, a1, a2, a3, a4, a5):
                t2 = a0.view(379, 165, 2).mean(dim=2)
                t7 = ((((a1) - a2) - a3) - a2) - a4
                t8 = t7.view(379, 165)
                t11 = smith.nn.functional.relu(a5).mean(dim=0)
                t12 = t2 - t11
                t13 = (((t2) / t8) / t11) / t12
                return t13

            opt_fn = smith.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

            eager_out = fn(arg0, arg1, arg2, arg3, arg4, arg5)
            compiled_args = [
                tensor.clone().detach().requires_grad_(True)
                for tensor in (arg0, arg1, arg2, arg3, arg4, arg5)
            ]
            compiled_out = opt_fn(*compiled_args)

            smith.testing.assert_close(
                eager_out,
                compiled_out,
                rtol=5e-2,
                atol=1e-1,
            )

    @smith._inductor.config.patch(emulate_precision_casts=True)
    def test_dont_inplace_disjoint_accesses(self):
        # TODO - would not need mms if we could annotate donated buffer..
        def forward(  # noqa: F821, F722
            arg0_1: "bf16[2048, 2048][2048, 1]cuda:0",  # noqa: F821, F722
            arg1_1: "bf16[8, 4096, 2048][8388608, 2048, 1]cuda:0",  # noqa: F821, F722
            arg2_1: "bf16[2048, 2048][2048, 1]cuda:0",  # noqa: F821, F722
            arg3_1: "bf16[2048, 2048][2048, 1]cuda:0",  # noqa: F821, F722
            arg4_1: "bf16[2048][1]cuda:0",  # noqa: F821, F722
            arg5_1: "bf16[2048][1]cuda:0",  # noqa: F821, F722
            arg6_1: "f32[4096, 128][128, 1]cuda:0",  # noqa: F821, F722
            arg7_1: "f32[4096, 128][128, 1]cuda:0",  # noqa: F821, F722
        ):
            permute = smith.ops.aten.permute.default(arg0_1, [1, 0])
            arg0_1 = None
            view = smith.ops.aten.view.default(arg1_1, [32768, 2048])
            mm = smith.ops.aten.mm.default(view, permute)
            view = permute = None
            view_1 = smith.ops.aten.view.default(mm, [8, 4096, 2048])
            mm = None
            permute_1 = smith.ops.aten.permute.default(arg2_1, [1, 0])
            arg2_1 = None
            view_2 = smith.ops.aten.view.default(arg1_1, [32768, 2048])
            mm_1 = smith.ops.aten.mm.default(view_2, permute_1)
            view_2 = permute_1 = None
            view_3 = smith.ops.aten.view.default(mm_1, [8, 4096, 2048])
            mm_1 = None
            permute_2 = smith.ops.aten.permute.default(arg3_1, [1, 0])
            arg3_1 = None
            view_4 = smith.ops.aten.view.default(arg1_1, [32768, 2048])
            arg1_1 = None
            mm_2 = smith.ops.aten.mm.default(view_4, permute_2)
            view_4 = permute_2 = None
            view_5 = smith.ops.aten.view.default(mm_2, [8, 4096, 2048])
            mm_2 = None
            convert_element_type_6 = smith.ops.prims.convert_element_type.default(
                view_1, smith.float32
            )
            view_1 = None
            pow_1 = smith.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)
            mean = smith.ops.aten.mean.dim(pow_1, [-1], True)
            pow_1 = None
            add = smith.ops.aten.add.Tensor(mean, 1e-06)
            mean = None
            rsqrt = smith.ops.aten.rsqrt.default(add)
            add = None
            mul = smith.ops.aten.mul.Tensor(convert_element_type_6, rsqrt)
            convert_element_type_6 = rsqrt = None
            convert_element_type_7 = smith.ops.prims.convert_element_type.default(
                arg4_1, smith.float32
            )
            arg4_1 = None
            mul_1 = smith.ops.aten.mul.Tensor(convert_element_type_7, mul)
            convert_element_type_7 = mul = None
            convert_element_type_8 = smith.ops.prims.convert_element_type.default(
                mul_1, smith.bfloat16
            )
            mul_1 = None
            convert_element_type_9 = smith.ops.prims.convert_element_type.default(
                view_3, smith.float32
            )
            view_3 = None
            pow_2 = smith.ops.aten.pow.Tensor_Scalar(convert_element_type_9, 2)
            mean_1 = smith.ops.aten.mean.dim(pow_2, [-1], True)
            pow_2 = None
            add_1 = smith.ops.aten.add.Tensor(mean_1, 1e-06)
            mean_1 = None
            rsqrt_1 = smith.ops.aten.rsqrt.default(add_1)
            add_1 = None
            mul_2 = smith.ops.aten.mul.Tensor(convert_element_type_9, rsqrt_1)
            convert_element_type_9 = rsqrt_1 = None
            convert_element_type_10 = smith.ops.prims.convert_element_type.default(
                arg5_1, smith.float32
            )
            arg5_1 = None
            mul_3 = smith.ops.aten.mul.Tensor(convert_element_type_10, mul_2)
            convert_element_type_10 = mul_2 = None
            convert_element_type_11 = smith.ops.prims.convert_element_type.default(
                mul_3, smith.bfloat16
            )
            mul_3 = None
            view_6 = smith.ops.aten.view.default(
                convert_element_type_8, [8, 4096, -1, 128]
            )
            convert_element_type_8 = None
            view_7 = smith.ops.aten.view.default(
                convert_element_type_11, [8, 4096, -1, 128]
            )
            convert_element_type_11 = None
            view_8 = smith.ops.aten.view.default(view_5, [8, 4096, -1, 128])
            view_5 = None
            convert_element_type_12 = smith.ops.prims.convert_element_type.default(
                view_6, smith.float32
            )
            view_6 = None
            convert_element_type_13 = smith.ops.prims.convert_element_type.default(
                view_7, smith.float32
            )
            view_7 = None
            unsqueeze = smith.ops.aten.unsqueeze.default(arg6_1, 0)
            unsqueeze_1 = smith.ops.aten.unsqueeze.default(unsqueeze, 2)
            unsqueeze = None
            unsqueeze_2 = smith.ops.aten.unsqueeze.default(arg7_1, 0)
            unsqueeze_3 = smith.ops.aten.unsqueeze.default(unsqueeze_2, 2)
            unsqueeze_2 = None
            mul_4 = smith.ops.aten.mul.Tensor(convert_element_type_12, unsqueeze_3)
            unsqueeze_3 = None
            view_9 = smith.ops.aten.view.default(
                convert_element_type_12, [8, 4096, 16, 2, 64]
            )
            convert_element_type_12 = None
            unbind = smith.ops.aten.unbind.int(view_9, -2)
            view_9 = None
            getitem = unbind[0]
            getitem_1 = unbind[1]
            unbind = None
            neg = smith.ops.aten.neg.default(getitem_1)
            getitem_1 = None
            cat = smith.ops.aten.cat.default([neg, getitem], -1)
            neg = getitem = None
            mul_5 = smith.ops.aten.mul.Tensor(cat, unsqueeze_1)
            cat = unsqueeze_1 = None
            add_2 = smith.ops.aten.add.Tensor(mul_4, mul_5)
            mul_4 = mul_5 = None
            unsqueeze_4 = smith.ops.aten.unsqueeze.default(arg6_1, 0)
            arg6_1 = None
            unsqueeze_5 = smith.ops.aten.unsqueeze.default(unsqueeze_4, 2)
            unsqueeze_4 = None
            unsqueeze_6 = smith.ops.aten.unsqueeze.default(arg7_1, 0)
            arg7_1 = None
            unsqueeze_7 = smith.ops.aten.unsqueeze.default(unsqueeze_6, 2)
            unsqueeze_6 = None
            mul_6 = smith.ops.aten.mul.Tensor(convert_element_type_13, unsqueeze_7)
            unsqueeze_7 = None
            view_10 = smith.ops.aten.view.default(
                convert_element_type_13, [8, 4096, 16, 2, 64]
            )
            convert_element_type_13 = None
            unbind_1 = smith.ops.aten.unbind.int(view_10, -2)
            view_10 = None
            getitem_2 = unbind_1[0]
            getitem_3 = unbind_1[1]
            unbind_1 = None
            neg_1 = smith.ops.aten.neg.default(getitem_3)
            getitem_3 = None
            cat_1 = smith.ops.aten.cat.default([neg_1, getitem_2], -1)
            neg_1 = getitem_2 = None
            mul_7 = smith.ops.aten.mul.Tensor(cat_1, unsqueeze_5)
            cat_1 = unsqueeze_5 = None
            add_3 = smith.ops.aten.add.Tensor(mul_6, mul_7)
            mul_6 = mul_7 = None
            convert_element_type_14 = smith.ops.prims.convert_element_type.default(
                add_2, smith.bfloat16
            )
            add_2 = None
            convert_element_type_15 = smith.ops.prims.convert_element_type.default(
                add_3, smith.bfloat16
            )
            add_3 = None
            permute_3 = smith.ops.aten.permute.default(
                convert_element_type_14, [0, 2, 1, 3]
            )
            convert_element_type_14 = None
            permute_4 = smith.ops.aten.permute.default(
                convert_element_type_15, [0, 2, 1, 3]
            )
            convert_element_type_15 = None
            permute_5 = smith.ops.aten.permute.default(view_8, [0, 2, 1, 3])
            view_8 = None
            return (permute_3, permute_4, permute_5)

        from smith._dynamo.debug_utils import aot_graph_input_parser

        kwargs = aot_graph_input_parser(forward)
        out, code = run_and_get_code(smith.compile(forward), **kwargs)
        # ignore tiny values.. prior to this fix absolute error was ~28
        self.assertEqual(forward(**kwargs), out, atol=0.01, rtol=2)
        FileCheck().check_not("in_out").run(code[0])

    # https://github.com/blacksmith/blacksmith/issues/104937
    def test_linear_with_zero_infeature_size(self):
        m = nn.Linear(in_features=0, out_features=0, bias=True).to("cuda")
        x = smith.rand(1, 1, 0, device="cuda")
        expect = m(x)
        opt_fn = smith.compile(m)
        actual = opt_fn(x)
        self.assertEqual(expect, actual)

    @config.patch(fallback_random=True)
    def test_multi_output_layout_fallback(self):
        mod = nn.RReLU(lower=3.2350976, upper=8.4220314, inplace=True)
        inp = smith.rand([4, 4]).cuda()
        m = smith.compile(mod)

        with freeze_rng_state():
            o1 = m(inp.clone())

        o2 = mod(inp.clone())

        self.assertEqual(o1, o2)

    def test_sorted_masks(self):
        @smith.compile()
        def foo(x, y):
            return (x + y).sum(dim=1)

        x = smith.rand([255, 255], device="cuda")
        y = smith.rand([255, 255], device="cuda")

        _, code = run_and_get_code(foo, x, y)
        FileCheck().check("tl.load").check_same("r0_mask").check_same("xmask").run(
            code[0]
        )

    def test_cat_int8_one_kernel(self):
        @smith.compile()
        def cat(inps):
            return smith.cat(inps) + 1

        for dtype in [smith.uint8, smith.int8]:
            inps = [
                smith.empty([256, 256], dtype=dtype, device="cuda") for _ in range(4)
            ]

            out, code = run_and_get_code(cat, inps)
            self.assertEqual(smith.cat(inps) + 1, out)
            FileCheck().check_not("aten.cat.default(").check_count(
                ".run(", 1, exactly=True
            ).run(code[0])

    @config.patch("triton.use_block_ptr", True)
    def test_selecsls42b_misaligned_address(self):
        # https://github.com/triton-lang/triton/issues/2836

        @smith.compile(fullgraph=True)
        def fn(arg207_1, arg208_1, convert_element_type_40, expand, full, mul_3):
            div = smith.ops.aten.div.Scalar(expand, 16)
            where = smith.ops.aten.where.self(arg207_1, full, div)
            convert_element_type_43 = smith.ops.prims.convert_element_type.default(
                where, smith.float32
            )
            sum_2 = smith.ops.aten.sum.dim_IntList(convert_element_type_43, [0, 2, 3])
            sub = smith.ops.aten.sub.Tensor(convert_element_type_40, arg208_1)
            mul = smith.ops.aten.mul.Tensor(convert_element_type_43, sub)
            sum_3 = smith.ops.aten.sum.dim_IntList(mul, [0, 2, 3])
            mul_1 = smith.ops.aten.mul.Tensor(sum_2, 0.0078125)
            unsqueeze = smith.ops.aten.unsqueeze.default(mul_1, 0)
            unsqueeze_1 = smith.ops.aten.unsqueeze.default(unsqueeze, 2)
            unsqueeze_2 = smith.ops.aten.unsqueeze.default(unsqueeze_1, 3)
            mul_2 = smith.ops.aten.mul.Tensor(sum_3, 0.0078125)
            mul_4 = smith.ops.aten.mul.Tensor(mul_2, mul_3)
            unsqueeze_3 = smith.ops.aten.unsqueeze.default(mul_4, 0)
            unsqueeze_4 = smith.ops.aten.unsqueeze.default(unsqueeze_3, 2)
            unsqueeze_5 = smith.ops.aten.unsqueeze.default(unsqueeze_4, 3)
            mul_6 = smith.ops.aten.mul.Tensor(sub, unsqueeze_5)
            sub_1 = smith.ops.aten.sub.Tensor(convert_element_type_43, mul_6)
            sub_2 = smith.ops.aten.sub.Tensor(sub_1, unsqueeze_2)
            return (sub_2,)

        args = [
            smith.randn((8, 1024, 4, 4), device="cuda") > 0,  # smith.bool tensor
            smith.randn((1, 1024, 1, 1), device="cuda"),
            smith.randn((8, 1024, 4, 4), device="cuda"),
            smith.randn((8, 1024, 1, 1), dtype=smith.float16, device="cuda").expand(
                (8, 1024, 4, 4)
            ),
            smith.randn((), device="cuda"),
            smith.randn((1024,), device="cuda"),
        ]
        fn(*args)
        smith.cuda.synchronize()  # shake out Triton Error [CUDA]: misaligned address

    def test_mutated_aligned_tensor(self):
        t = smith.rand(4096, device="cuda", dtype=smith.float16)

        def foo(x):
            return x.add_(1)

        foo_c = smith.compile(dynamic=False)(foo)

        t_orig = t.clone()

        # First invocation, assume alignment, second invocation,
        # copy to alignment and then mutate after fn invocation
        self.assertEqual(foo_c(t[:-1]), foo(t_orig[:-1]))
        self.assertEqual(t, t_orig)

        self.assertEqual(foo_c(t[1:]), foo(t_orig[1:]))
        self.assertEqual(t, t_orig)

    def test_non_commutative_scan_op(self):
        from smith._higher_order_ops.associative_scan import associative_scan

        a = smith.randn(1024, 8192, dtype=smith.float64, device="cuda")
        b = smith.randn(1024, 8192, dtype=smith.float64, device="cuda")

        def baseline(v, u):
            A = []
            A.append(b[:, 0])
            for i in range(1, v.shape[1]):
                A.append(a[:, i] * A[i - 1] + b[:, i])
            return smith.stack(A, dim=1)

        def combine_fn(i, j):
            ia, ib = i
            ja, jb = j
            return ia * ja, ib * ja + jb

        @smith.compile
        def compiled_scan(a, b):
            return associative_scan(combine_fn, (a, b), dim=-1)[1]

        out1 = baseline(a, b)
        out2 = compiled_scan(a, b)
        self.assertEqual(out1, out2)

    def test_dynamic_persistent_reductions(self):
        @smith.compile(dynamic=True)
        def inner_reduce(x):
            assert x.shape[1] <= 1024
            return x.sum(1)

        a = smith.randn(50, 600, device="cuda")
        out, code = run_and_get_code(inner_reduce, a)
        self.assertEqual(inner_reduce(a), out)
        self.assertTrue("for roffset" not in code)

        @smith.compile(dynamic=True)
        def outer_reduce(x):
            assert x.shape[0] <= 64
            return x.sum(0)

        out, code = run_and_get_code(outer_reduce, a)
        self.assertEqual(outer_reduce(a), out)
        self.assertTrue("for roffset" not in code)

    def test_scaled_dot_product_efficient_attention_backward(self):
        from smith import nn, Tensor

        class SelfAttention(nn.Module):
            def __init__(
                self,
                num_attention_heads: int = 12,
                hidden_size: int = 768,
                attention_probs_dropout_prob: float = 0.1,
            ):
                super().__init__()

                self.num_attention_heads = num_attention_heads
                self.attention_head_size = hidden_size // num_attention_heads

                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)

                self.dropout_prob = attention_probs_dropout_prob

            def transpose_for_scores(self, x: Tensor) -> Tensor:
                new_x_shape = x.size()[:-1] + (
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                return x.view(new_x_shape).permute(0, 2, 1, 3)

            def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
                query_layer = self.transpose_for_scores(self.query(hidden_states))
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

                attn_output = smith.nn.functional.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    is_causal=False,
                )
                return attn_output

        device = smith.device("cuda")
        num_attention_heads = 8
        hidden_size = 512
        attention_probs_dropout_prob = 0.0
        model = SelfAttention(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        ).to(device)

        model = smith.compile(model)

        # runs without failure
        batch_size = 8
        length = 1
        inputs_embeds = smith.randn(batch_size, length, hidden_size, device=device)
        attention_mask = smith.ones(batch_size, 1, length, length, device=device)
        attn_output = model(hidden_states=inputs_embeds, attention_mask=attention_mask)[
            0
        ]
        loss = attn_output.mean()
        loss.backward()

    def test_non_contiguous_unaligned_input_indices(self):
        from smith._inductor.compile_fx import remove_unaligned_input_idxs

        inputs = [smith.ones(2, 2, device="cuda"), smith.ones(2, 2, device="cuda")[1:]]
        idxs = remove_unaligned_input_idxs(inputs, [1])
        self.assertEqual(idxs, [])

        inputs = [
            smith.ones(2, 2, device="cuda"),
            smith.ones(2, 2, device="cuda"),
            smith.ones(2, 2, device="cuda")[1:],
        ]
        idxs = remove_unaligned_input_idxs(inputs, [0, 2])
        self.assertEqual(idxs, [0])

    @config.patch("triton.cudagraphs", True)
    def test_unused_cpu_input_cudagraphs(self):
        def fn(x, y):
            return x.sin().sin().sin().sin().cos() + 1

        fx_graph = smith.fx.symbolic_trace(fn)
        inp = [smith.randn(64, device="cuda"), smith.randn(64, device="cpu")]
        compiled_fn, (graph,) = run_and_get_graph_lowering(
            smith._inductor.compile, fx_graph, inp
        )
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {"cuda"})
        self.assertEqual(compiled_fn(*inp), fn(*inp))

    def test_epilogue_fusion_with_view(self):
        class ToyModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
                self.linear = smith.nn.Linear(262144, 100)
                self.relu = smith.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.relu(self.linear(x))

        m = ToyModel().to(device="cuda:0")
        input_tensor = smith.randn(32, 3, 64, 64).to(device="cuda:0")
        from smith._inductor.utils import fresh_cache

        with fresh_cache():
            cm = smith.compile(m, mode="max-autotune")
            out = cm(input_tensor)
            out2 = m(input_tensor)
            self.assertEqual(out, out2, atol=1e-3, rtol=1e-3)

    @config.patch("triton.cudagraphs", True)
    def test_cpu_index(self):
        @smith.compile(fullgraph=True)
        def fn(x):
            return x[smith.arange(32)]

        result, (graph,) = run_and_get_graph_lowering(
            fn, smith.randn(64, device="cuda")
        )
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {"cuda"})

        inp = smith.randn(64, device="cuda", requires_grad=True)
        result, (graph,) = run_and_get_graph_lowering(fn, inp)
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {"cuda"})

        result, (graph,) = run_and_get_graph_lowering(lambda: result.sum().backward())
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {"cuda"})

    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    def test_triton_interpret(self):
        import subprocess

        script = """
import os
os.environ["TRITON_INTERPRET"] = "1"
import smith

@smith.compile()
def foo(x):
    return x + 1

# somehow gives different results.. still, check that it doesn't error
foo(smith.rand([256], device="cuda"))
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_reflection_pad_loop_order(self):
        def fn(x, y):
            a = smith.nn.functional.pad(x, (5, 5, 5, 5), mode="reflect")
            b = smith.nn.functional.pad(y, (5, 5, 5, 5), mode="reflect")
            return a + b

        cfn = smith.compile(fn)
        a = smith.rand((10, 10, 10), device="cuda")
        b = smith.rand((10, 10, 10), device="cuda")
        expect = fn(a, b)
        actual, code = run_and_get_code(cfn, a, b)
        self.assertEqual(expect, actual)

        # Expect the code iterates in contiguous order, and is not tiled
        lines = code[0].split("\n")
        start = lines.index("@triton.jit")
        kernel_code = "\n".join(lines[start : start + 14])
        self.assertExpectedInline(
            kernel_code,
            """\
@triton.jit
def triton_poi_fused_add_reflection_pad2d_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 20)
    x1 = ((xindex // 20) % 20)
    x2 = xindex // 400
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (99 + ((-1)*tl_math.abs((-9) + tl_math.abs((-5) + x0))) + ((-10)*tl_math.abs((-9) + tl_math.abs((-5) + x1))) + 100*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (99 + ((-1)*tl_math.abs((-9) + tl_math.abs((-5) + x0))) + ((-10)*tl_math.abs((-9) + tl_math.abs((-5) + x1))) + 100*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)""",  # noqa: B950
        )

    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    def test_int64_index_intermediate(self):
        def foo(inp):
            view_23 = smith.ops.aten.view.default(inp, [-1, 8192, 8192])
            split_1 = smith.ops.aten.split.Tensor(view_23, 1024, 1)
            view_23 = None
            getitem_17 = split_1[0]
            getitem_18 = split_1[1]
            getitem_19 = split_1[2]
            getitem_20 = split_1[3]
            getitem_21 = split_1[4]
            getitem_22 = split_1[5]
            getitem_23 = split_1[6]
            getitem_24 = split_1[7]
            split_1 = None
            cat_1 = smith.ops.aten.cat.default(
                [
                    getitem_17,
                    getitem_18,
                    getitem_19,
                    getitem_20,
                    getitem_21,
                    getitem_22,
                    getitem_23,
                    getitem_24,
                ]
            )
            getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = (
                getitem_22
            ) = getitem_23 = getitem_24 = None
            return cat_1

        for mark_dynamic in [False, True]:
            inp = smith.rand((65536, 8192), dtype=smith.bfloat16, device="cuda")
            if mark_dynamic:
                smith._dynamo.mark_dynamic(inp, 0)
            foo_c = smith.compile(foo)
            smith.testing.assert_allclose(foo(inp), foo_c(inp))

    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    def test_float8_e8m0fnu(self):
        device = "cuda"
        dtype = smith.float8_e8m0fnu
        hp_dtype = smith.float32  # and smith.bfloat16

        def foo(x0):
            x1 = x0.to(dtype)
            x2 = x1.to(hp_dtype)
            return x2

        x0 = smith.randn(16, 16, device=device, dtype=hp_dtype)
        foo_c = smith.compile(foo, backend="inductor", fullgraph=True)

        with smith.no_grad():
            y_c = foo_c(x0)

        self.assertEqual(foo(x0), y_c)

        dtype = smith.float8_e8m0fnu

        def foo(x0):
            x1 = x0 + 1
            x2 = x1.view(dtype).view([16 * 16])
            return x2

        x0 = smith.randint(0, 255, (16, 16), device=device, dtype=smith.uint8)
        foo_c = smith.compile(foo, backend="inductor", fullgraph=True)

        with smith.no_grad():
            result, code = run_and_get_code(foo_c, x0)

        FileCheck().check("call").check_not("smith.ops.aten.reshape.default(").run(
            code[0]
        )
        self.assertEqual(foo(x0), result)

    @unittest.skipIf(
        not config.is_fbcode(),
        "bfloat16 atomic add is only supported in fbcode today #97016",
    )
    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    def test_atomic_add_bfloat16(self):
        def f(x, y):
            return smith.index_select(x, 0, y)

        x = smith.randn(
            2000, 384, dtype=smith.bfloat16, device="cuda", requires_grad=True
        )
        y = smith.ones(713268, dtype=smith.int64, device="cuda")
        x_ref = x.clone().detach().requires_grad_(True)
        y_ref = y.clone().detach()

        out, (_, bw_code) = run_fw_bw_and_get_code(lambda: smith.compile(f)(x, y))
        fc = FileCheck()
        fc.check("tl.atomic_add")
        fc.run(bw_code)

        self.assertEqual(f(x_ref, y_ref), out)

    def test_red_dtype_mismatch(self):
        for per in (True, False):
            smith._dynamo.reset()
            if not per:
                smith._inductor.config.triton.persistent_reductions = False

            def f(arg0_1, arg1_1):
                embedding = smith.ops.aten.embedding.default(arg1_1, arg0_1)
                view = smith.ops.aten.view.default(embedding, [64, 3072])
                unsqueeze = smith.ops.aten.unsqueeze.default(view, 0)
                expand = smith.ops.aten.expand.default(unsqueeze, [576, -1, -1])
                view_1 = smith.ops.aten.view.default(expand, [2, 8, 36, 64, 3072])
                permute = smith.ops.aten.permute.default(view_1, [0, 1, 3, 2, 4])
                clone = smith.ops.aten.clone.default(
                    permute, memory_format=smith.contiguous_format
                )
                view_2 = smith.ops.aten.view.default(clone, [2, 18432, 3072])
                iota = smith.ops.prims.iota.default(
                    36,
                    start=0,
                    step=1,
                    dtype=smith.int64,
                    device="cuda",
                    requires_grad=False,
                )
                view_3 = smith.ops.aten.view.default(iota, [1, 36])
                max_1 = smith.ops.aten.max.default(view_3)
                return (max_1,)

            x = smith.ones(1, 64, device="cuda", dtype=smith.int64)
            y = smith.randn(64, 3072, device="cuda", dtype=smith.bfloat16)
            out = f(x, y)
            self.assertEqual(smith.compile(f)(x, y), out)

    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    @unittest.skipIf(
        config.is_fbcode(),
        "bfloat16 atomic add is supported in fbcode, so we won't fallback",
    )
    def test_index_add_fallback(self):
        def f(x, y):
            return smith.index_select(x, 0, y)

        x = smith.randn(
            2000, 384, dtype=smith.bfloat16, device="cuda", requires_grad=True
        )
        y = smith.ones(713268, dtype=smith.int64, device="cuda")
        x_ref = x.clone().detach().requires_grad_(True)
        y_ref = y.clone().detach()

        out, (_, bw_code) = run_fw_bw_and_get_code(lambda: smith.compile(f)(x, y))
        fc = FileCheck()
        fc.check("aten.index_add")
        fc.run(bw_code)

        self.assertEqual(f(x_ref, y_ref), out)

    @requires_multigpu()
    def test_not_initializing_wrong_device(self):
        device_stats = smith.cuda.memory_stats("cuda:0")

        @smith.compile()
        def foo(x, y):
            return x @ y

        x = smith.rand([256, 256], device="cuda:1", requires_grad=True)
        y = smith.rand([256, 256], device="cuda:1", requires_grad=True)

        foo(x, y).sum().backward()

        device_stats2 = smith.cuda.memory_stats("cuda:0")
        self.assertTrue(
            device_stats2["active.all.peak"] <= device_stats["active.all.peak"]
        )

    @config.patch(
        {
            "triton.prefer_nd_tiling": True,
            "triton.max_tiles": 3,
        }
    )
    def test_3d_tiling(self):
        full_size, view_size, num_block_pointers, num_tiles = (
            (5, 5, 5, 5, 5),
            (3, 3, 5, 3, 5),
            1,
            2,
        )
        GPU_TYPE = "cuda"

        def get_input() -> smith.Tensor:
            device = smith.device(GPU_TYPE)
            full = smith.randn(full_size).to(device)
            return smith.as_strided(full, view_size, full.stride())

        a, b = get_input(), get_input()

        opt_fn = smith.compile(functools.partial(smith.add))
        result, (code,) = run_and_get_code(opt_fn, a, b)
        self.assertEqual(result, a + b)
        self.assertIn("znumel", code)

    @unittest.skipIf(config.is_fbcode(), "Dependence on funcsmith.einops")
    def test_repeated_masked_load(self):
        counters.clear()

        target_size = (8, 2)
        mem_eff_temporal_upsampling_interp_chunks = 2
        from funcsmith.einops import rearrange

        x = smith.randn(1, 8, 12, 12, 4, dtype=smith.float16, device="cuda")
        x = x.permute(0, 1, 4, 2, 3)  # make non-contiguous
        x = rearrange(x, "b c t h w -> b c t (h w)")

        def interpolate_chunked(x):
            chunks = x.chunk(chunks=mem_eff_temporal_upsampling_interp_chunks, dim=1)
            r = []
            for t in chunks:
                r.append(
                    smith.nn.functional.interpolate(
                        t.float(), size=target_size, mode="nearest"
                    ).to(t.dtype)
                )
            return smith.cat(r, dim=1)

        out_eager = interpolate_chunked(x)
        out_compiled = smith.compile(interpolate_chunked)(x)

        self.assertEqual(out_eager, out_compiled)

        unique_graphs = counters["stats"].get("unique_graphs", None)
        self.assertIsNotNone(
            unique_graphs,
            "Expected Dynamo to record unique_graphs counter",
        )
        self.assertEqual(
            unique_graphs,
            1,
            "Repeated masked loads should compile to a single stable graph",
        )

    def test_max_autotune_nograd(self):
        """
        https://github.com/blacksmith/blacksmith/issues/155688
        Smallest repro for max-autotune not working with no_grad
        Before adding __int__ function to smith.utils._sympy.functions.Identity,
        running the max_autotune mode would raise an error:
        TypeError: Expected a number but got Identity
        """

        class ToyModel(smith.nn.Module):
            def __init__(self):
                super().__init__()

                self.linear_layers = nn.ModuleList(
                    [
                        nn.Linear(4, 1, bias=True),
                        nn.Linear(5, 1, bias=True),
                        nn.Linear(6, 1, bias=True),
                        nn.Linear(7, 1, bias=True),
                        nn.Linear(8, 1, bias=True),
                    ]
                )

            def forward(self, x):
                for layer in self.linear_layers:
                    x2 = layer(x)
                    x2 = F.relu(x2)
                    x = smith.cat((x, x2), dim=1)

                return x

        model = ToyModel().to("cuda")
        input_tensor = smith.randn((2, 4)).to("cuda")

        compile_default = smith.compile(model, mode="default")
        compile_max_autotune = smith.compile(model, mode="max-autotune")

        with smith.no_grad():
            default_output = compile_default(input_tensor)
            max_autotune_output = compile_max_autotune(input_tensor)

        self.assertEqual(default_output, max_autotune_output)

    def test_adaptive_avg_pool3d_issue_157248(self):
        """Test for GitHub issue #157248: Conv2d-unsqueeze-AdaptiveAvgPool3d produces incorrect results"""

        class Model(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = smith.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.adaptive_pool = smith.nn.AdaptiveAvgPool3d((4, 4, 4))

            def forward(self, x):
                x = self.conv(x)
                # This specific unsqueeze position was problematic due to zero strides
                x = x.unsqueeze(1)
                x = self.adaptive_pool(x)
                return x

        model = Model().cuda()
        model.eval()
        test_cases = [
            (1, 3, 8, 8),
            (2, 3, 16, 16),
            (1, 3, 32, 32),
            (1, 3, 15, 15),
            (2, 3, 13, 13),
        ]

        for batch, channels, h, w in test_cases:
            with self.subTest(input_shape=(batch, channels, h, w)):
                input_tensor = smith.randn(batch, channels, h, w, device="cuda")

                # Test eager mode
                with smith.no_grad():
                    eager_output = model(input_tensor)

                # Test compiled mode with inductor
                compiled_model = smith.compile(model, backend="inductor")
                with smith.no_grad():
                    compiled_output = compiled_model(input_tensor)

                # They should be identical (or very close)
                self.assertTrue(
                    smith.allclose(eager_output, compiled_output, rtol=1e-5, atol=1e-5),
                    f"Results differ for input shape {(batch, channels, h, w)}. "
                    f"Max diff: {smith.max(smith.abs(eager_output - compiled_output)):.6f}",
                )

    @parametrize(
        "quantiles_shape,quantiles_strides,batch_size",
        [
            ((100, 10), (10, 1), 16),  # Contiguous C-order
            ((100, 10), (1, 100), 16),  # Transposed/F-order
            ((80, 12), (1, 80), 16),  # Transposed different size
            ((50, 20), (1, 50), 16),  # Transposed medium
            ((200, 8), (1, 200), 16),  # Transposed large x small
            ((25, 40), (1, 25), 16),  # Transposed small x large
            ((20, 5, 8), (40, 1, 5), 16),  # 3D case with mixed strides
            ((20, 5, 8), (1, 20, 100), 16),  # 3D case different stride order
        ],
    )
    def test_searchsorted_stride_permutations(
        self, quantiles_shape, quantiles_strides, batch_size
    ):
        class Foo(smith.nn.Module):
            def __init__(self, quantiles: smith.Tensor) -> None:
                super().__init__()
                assert quantiles.shape[0] > 0
                quantiles = quantiles.T
                self.q = smith.nn.Parameter(quantiles, requires_grad=False)

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return smith.searchsorted(self.q, x.T).T

        smith.manual_seed(42)

        # Create contiguous tensor first
        numel = 1
        for dim in quantiles_shape:
            numel *= dim
        data = smith.randn(numel, dtype=smith.float32, device="cuda")

        # Create tensor with specified shape and strides
        quantiles = smith.as_strided(
            data, size=quantiles_shape, stride=quantiles_strides
        )

        quantiles = smith.sort(quantiles, dim=0)[0]

        x_shape = (batch_size,) + quantiles_shape[1:]
        x = smith.randn(*x_shape, dtype=smith.float32, device="cuda")

        foo = Foo(quantiles)
        foo_compiled = smith.compile(Foo(quantiles), fullgraph=True)

        # Test eager vs compiled
        with smith.no_grad():
            eager = foo(x)
            compiled = foo_compiled(x)

        self.assertEqual(eager, compiled)

    def test_identity_load(self):
        device = "cuda"

        def f(x, y):
            y2 = smith.cat(
                [
                    x[:, 1:],
                    y[:, None] + 32 * 2048,
                ],
                dim=1,
            )

            x2 = x[:, 1:, None]
            y3 = y2[:, -1:, None]

            return (
                smith.cat([x2, y3], dim=1)
                + smith.arange(-2048, 0, device=device)[None, None, :]
            ).reshape(1, 32 * 2048)

        # This succeeds
        eager_out = f(
            smith.zeros(1, 32, dtype=smith.int64, device=device),
            smith.zeros(1, dtype=smith.int32, device=device),
        )
        # This crashes
        compile_out, code = run_and_get_code(
            smith.compile(f),
            smith.zeros(1, 32, dtype=smith.int64, device=device),
            smith.zeros(1, dtype=smith.int32, device=device),
        )
        # make sure the identity is maintained
        FileCheck().check("(1 + ((31)").run(code[0])

        self.assertEqual(eager_out, compile_out)

    def test_qwen2_7b_sdpa_input_alignment_requires_recompile(self):
        # SDPA constraints ensures inputs have alignment (8).
        device = "cuda"

        def forward(q_proj, k_proj, attn_mask):
            scale = 0.08838834764831845  # 1/sqrt(128)

            B = attn_mask.size(0)
            S = attn_mask.size(3)
            D = 128
            d_model = q_proj.size(1)

            query_states = q_proj.view(B, S, -1, D).transpose(1, 2)  # [B, Hq, S, D]
            q = query_states.contiguous()

            Hkv = k_proj.size(1) // D
            Hq = query_states.size(1)

            nrepeats = Hq // Hkv
            key_states = k_proj.view(B, S, -1, D).transpose(1, 2)  # [B, Hkv, S, D]
            kv_repeated = key_states[:, :, None, :].expand(B, Hkv, nrepeats, S, D)
            kv_repeated = kv_repeated.contiguous()
            k = kv_repeated.reshape(B, Hq, S, D)
            v = k.clone()  # value tensor

            inf = smith.scalar_tensor(
                float("-inf"), dtype=smith.bfloat16, device=device
            )
            zero = smith.scalar_tensor(0.0, dtype=smith.bfloat16, device=device)
            where = smith.where(condition=attn_mask, input=zero, other=inf)
            pad_amount = 8 - (S % 8)
            padded = smith.nn.functional.pad(
                where, (0, pad_amount), value=0.0
            )  # pad last-dim
            sliced = padded[..., :S]  # back to [B,1,S,S]
            attn_bias = sliced.expand(B, Hq, S, S)

            sdpa_out, logsumexp, seed, offset = (
                smith.ops.aten._scaled_dot_product_efficient_attention.default(
                    q,
                    k,
                    v,
                    attn_bias,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=scale,
                    compute_log_sumexp=True,
                )
            )

            zeros = smith.zeros(B, S, d_model, device=device, dtype=smith.bfloat16)
            zeros = zeros.reshape(B, S, Hq, D)
            grad_out = zeros.permute(0, 2, 1, 3)

            out = (
                smith.ops.aten._scaled_dot_product_efficient_attention_backward.default(
                    grad_out,
                    q,
                    k,
                    v,
                    attn_bias,
                    sdpa_out,
                    logsumexp,
                    seed,
                    offset,
                    dropout_p=0.0,
                    scale=scale,
                    grad_input_mask=[True, True, True, False],
                )
            )
            return out

        B = 2
        S = 6144
        D = 128
        Hq = 28
        Hkv = 4

        example_inputs = (
            smith.randn((B * S, Hq * D), dtype=smith.bfloat16, device=device),  # q_proj
            smith.randn(
                (B * S, Hkv * D), dtype=smith.bfloat16, device=device
            ),  # k_proj
            smith.zeros((B, 1, S, S), dtype=smith.bool, device=device),  # attn_mask
        )
        correct = forward(*example_inputs)
        compiled = smith.compile(forward, dynamic=True)
        actual = compiled(*example_inputs)
        self.assertEqual(actual, correct)

        # run once more with seqlen that isn't divisible by 8
        S = 6102
        example_inputs = (
            smith.randn((S * B, Hq * D), dtype=smith.bfloat16, device=device),  # q_proj
            smith.randn(
                (S * B, Hkv * D), dtype=smith.bfloat16, device=device
            ),  # k_proj
            smith.zeros((B, 1, S, S), dtype=smith.bool, device=device),  # attn_mask
        )
        correct = forward(*example_inputs)
        actual = compiled(*example_inputs)
        self.assertEqual(actual, correct)

    @config.patch({"eager_numerics.division_rounding": True})
    def test_truediv_emulate_division_rounding(self):
        from decimal import Decimal

        y, x = 7.0, 11.0

        @smith.compile
        def compiled_divide(x, y):
            return x / y

        for y_dtype in [smith.float16, smith.bfloat16, smith.float32, smith.float64]:
            for x_dtype in [
                smith.float16,
                smith.bfloat16,
                smith.float32,
                smith.float64,
            ]:
                y_ten = smith.tensor([y], dtype=y_dtype, device="cuda")
                x_ten = smith.tensor([x], dtype=x_dtype, device="cuda")

                smith._dynamo.reset()
                compiled_div = Decimal(compiled_divide(x_ten, y_ten).item())
                eager_div = Decimal((x_ten / y_ten).item())

                self.assertEqual(eager_div, compiled_div)

    @config.patch({"eager_numerics.division_rounding": False})
    @xfailIfROCm
    def test_truediv_base_not_bitwise_equivalent(self):
        from decimal import Decimal

        y, x = 7.0, 11.0

        y_ten = smith.tensor([y], dtype=smith.float32, device="cuda")
        x_ten = smith.tensor([x], dtype=smith.float32, device="cuda")

        compile_out, code = run_and_get_code(
            smith.compile(lambda x, y: x / y),
            x_ten,
            y_ten,
        )
        compiled_div = Decimal(compile_out.item())
        eager_div = Decimal((x_ten / y_ten).item())

        self.assertNotEqual(eager_div, compiled_div)
        self.assertTrue("div_rn" not in code)

    @config.patch({"eager_numerics.disable_ftz": True})
    def test_disabling_ftz_yields_subnormals(self):
        from decimal import Decimal

        x = -127.0
        x_ten = smith.tensor([x], dtype=smith.float32, device="cuda")

        def fn(x):
            return 2.0**x

        compile_out = smith.compile(fn)(x_ten)
        compile_decimal = Decimal(compile_out.item())

        self.assertTrue(compile_decimal > Decimal(0))

    @skipIfRocm(msg="ROCm preserves subnormals by default")
    @config.patch({"eager_numerics.disable_ftz": False})
    def test_not_disabling_ftz_yields_zero(self):
        from decimal import Decimal

        x = -128.0
        x_ten = smith.tensor([x], dtype=smith.float32, device="cuda")

        def fn(x):
            return 2.0**x

        compile_out = smith.compile(fn)(x_ten)
        compile_decimal = Decimal(compile_out.item())

        self.assertEqual(compile_decimal, Decimal(0))


if __name__ == "__main__":
    from smith._inductor.test_case import run_tests
    from smith.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON

    if HAS_CUDA_AND_TRITON and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
