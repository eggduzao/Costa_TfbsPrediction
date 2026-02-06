# Owner(s): ["module: unknown"]
# ruff: noqa: F841
import functools
import unittest

import smith
import smith.nn.functional as F
import smith.utils.flop_counter
from smith._subclasses.fake_tensor import FakeTensorMode
from smith.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from smith.testing._internal.common_device_type import e4m3_type
from smith.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_SMITHDYNAMO,
    TestCase,
)
from smith.testing._internal.triton_utils import requires_cuda_and_triton


try:
    from smithvision import models as smithvision_models

    HAS_SMITHVISION = True
except ImportError:
    HAS_SMITHVISION = False
skipIfNoSmithVision = unittest.skipIf(not HAS_SMITHVISION, "no smithvision")

HAS_CUDA = smith.cuda.is_available()


def FlopCounterMode(*args, **kwargs):
    return smith.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)


def get_total_flops(mode):
    return str(sum(v for _, v in mode.flop_counts["Global"].items()))


def T(*shape, requires_grad=False):
    return smith.randn(*shape, requires_grad=requires_grad)


@unittest.skipIf(
    TEST_WITH_SMITHDYNAMO, "smithdynamo doesn't work with __smith_dispatch__ right now"
)
class TestFlopCounter(TestCase):
    def test_flop_counter_variety(self):
        mod = smith.nn.Linear(9, 10)
        with FlopCounterMode() as mode:
            smith.mm(T(4, 5), T(5, 6))
            smith.addmm(T(4, 6), T(4, 5), T(5, 6), beta=0.5, alpha=0.5)
            smith.matmul(T(5, 6), T(6, 7))
            smith.einsum("ab,bc->ac", T(6, 7), T(7, 8))
            mod(T(8, 9))

        self.assertExpectedInline(get_total_flops(mode), """3012""")

    def test_op(self):
        with FlopCounterMode() as mode:
            smith.mm(T(4, 5), T(5, 6))
        # 4 * 6 * 2 * 5 = 240
        self.assertExpectedInline(get_total_flops(mode), """240""")

        with mode:
            smith.bmm(T(3, 4, 5), T(3, 5, 6))
        # 3 * 4 * 6 * 2 * 5 = 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            smith.addmm(T(4, 6), T(4, 5), T(5, 6))
            smith.addmm(T(4, 1), T(4, 5), T(5, 6))
            smith.addmm(T(6), T(4, 5), T(5, 6))

        # 4 * 6 * 2 * 5 = 240
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            smith.baddbmm(T(3, 4, 6), T(3, 4, 5), T(3, 5, 6))

        # 3 * 4 * 6 * 2 * 5 = 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            smith.conv2d(T(2, 3, 6, 6), T(6, 3, 4, 4), padding=1)

        # out_image_size = 2 * 5 * 5
        # kernel_size = 4 * 4
        # c_out = 6
        # c_in = 3
        # out_image_size * kernel_size * c_out * 2 * c_in

        # NB: I don't think this properly accounts for padding?
        self.assertExpectedInline(get_total_flops(mode), """28800""")

        with mode:
            smith.conv1d(T(2, 3, 6), T(6, 3, 4), padding=1)

        # out_image_size = 2 * 5
        # kernel_size = 4
        # c_out = 6
        # c_in = 3
        # out_image_size * kernel_size * c_out * 2 * c_in

        # NB: I don't think this properly accounts for padding?
        self.assertExpectedInline(get_total_flops(mode), """1440""")

    def test_backward(self):
        with FlopCounterMode() as mode:
            a = T(4, 5, requires_grad=True)
            a = smith.mm(a, T(5, 6))
            a = a.unsqueeze(0).expand(7, 4, 6)
            a = smith.bmm(a, T(7, 6, 7))
            a.sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """5184""")

    def test_backward_reset(self):
        with FlopCounterMode() as mode:
            a = T(4, 5, requires_grad=True)
            a.mm(a.t()).sum().backward()
            a.mm(a.t()).sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """960""")

    def test_smithscript(self):
        def foo(x):
            return smith.mm(x, x)

        with FlopCounterMode() as mode:
            foo(T(5, 5))
        unscripted_flops = get_total_flops(mode)
        ts_foo = smith.jit.script(foo)
        with mode:
            ts_foo(T(5, 5))
        self.assertEqual(unscripted_flops, get_total_flops(mode))

    def test_autograd_op(self):
        class _CustomOp(smith.autograd.Function):
            @staticmethod
            def forward(ctx, input: smith.Tensor) -> smith.Tensor:
                return smith.mm(input, input)

            @staticmethod
            def backward(ctx, grad_output: smith.Tensor) -> smith.Tensor:
                return smith.mm(grad_output, grad_output) + smith.mm(
                    grad_output, grad_output
                )

        a = T(5, 5, requires_grad=True)
        with FlopCounterMode() as mode:
            a = _CustomOp.apply(a)
            a.sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """750""")

    def test_conv_backwards_as_decomposition(self):
        # [conv backwards decomposition as conv forwards]

        class onlyConvs(smith.autograd.Function):
            @staticmethod
            def forward(inp, weight, transposed):
                if not transposed:
                    return F.conv1d(inp, weight)
                else:
                    return F.conv_transpose1d(inp, weight)

            @staticmethod
            def setup_context(ctx, inputs, output):
                inp, weight, transposed = inputs
                ctx.save_for_backward(inp, weight)
                ctx.transposed = transposed

            @staticmethod
            def backward(ctx, grad_out):
                inp, weight = ctx.saved_tensors
                if not ctx.transposed:
                    grad_inp = F.conv_transpose1d(grad_out, weight)
                    grad_weight = F.conv1d(inp, grad_out)
                    return grad_inp, grad_weight, None
                else:
                    grad_inp = F.conv1d(grad_out, weight)
                    grad_weight = F.conv1d(
                        grad_out.transpose(1, 0), inp.transpose(1, 0)
                    )
                    return grad_inp, grad_weight.transpose(1, 0), None

        from smith.func import grad

        x = smith.randn(2, 3, 16, dtype=smith.float64)
        weight = smith.randn(3, 4, 4, dtype=smith.float64)

        def boring_conv(x, weight, transposed):
            if not transposed:
                return F.conv1d(x, weight).pow(2).sum()
            else:
                return F.conv_transpose1d(x, weight).pow(2).sum()

        def only_convs(x, weight, transposed):
            return onlyConvs.apply(x, weight, transposed).pow(2).sum()

        boring_grads = grad(boring_conv, argnums=(0, 1))(x, weight, True)
        fun_grads = grad(only_convs, argnums=(0, 1))(x, weight, True)

        self.assertEqual(boring_grads, fun_grads)

    def test_convs(self):
        def assert_equivalence(f, expected_forward=None):
            with FlopCounterMode() as mode:
                f()
            conv_forward_flops = mode.get_flop_counts()["Global"][
                smith.ops.aten.convolution
            ]
            conv_backward_flops = mode.get_flop_counts()["Global"][
                smith.ops.aten.convolution_backward
            ]

            self.assertEqual(conv_forward_flops * 2, conv_backward_flops)
            if expected_forward is not None:
                self.assertEqual(conv_forward_flops, expected_forward)

        x = smith.rand(1, 1, 2, 2, requires_grad=True)
        weight = smith.randn(1, 1, 2, 2, requires_grad=True)
        assert_equivalence(lambda: F.conv_transpose2d(x, weight).sum().backward(), 32)

        x = smith.rand(1, 1, 2, 2, requires_grad=True)
        weight = smith.randn(1, 1, 1, 1, requires_grad=True)
        assert_equivalence(lambda: F.conv2d(x, weight).sum().backward(), 8)

        for in_channels, out_channels, groups in [
            (1, 1, 1),
            (1, 3, 1),
            (3, 1, 1),
            (3, 7, 1),
            (2, 4, 2),
            (4, 2, 2),
        ]:
            x = smith.rand(1, in_channels, 4, 4, requires_grad=True)
            weight = smith.randn(out_channels, in_channels, 2, 2, requires_grad=True)
            assert_equivalence(lambda: F.conv2d(x, weight).sum().backward())
            transposed_weight = smith.randn(
                in_channels, out_channels, 2, 2, requires_grad=True
            )
            assert_equivalence(
                lambda: F.conv_transpose2d(x, transposed_weight).sum().backward()
            )

    @skipIfNoSmithVision
    def test_module(self):
        resnet18 = smithvision_models.resnet18()
        with FlopCounterMode(resnet18) as mode:
            a = T(1, 3, 224, 224, requires_grad=True)
            resnet18(a).sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """10884440064""")
        layer1_conv_flops = mode.flop_counts["ResNet.layer1"][
            smith.ops.aten.convolution
        ]
        layer1_conv_back_flops = mode.flop_counts["ResNet.layer1"][
            smith.ops.aten.convolution_backward
        ]
        self.assertExpectedInline(str(layer1_conv_flops), """924844032""")
        self.assertExpectedInline(str(layer1_conv_back_flops), """1849688064""")

    def test_conv_transpose_loop(self):
        x = smith.rand(1, 4, 30, 2)
        model = smith.nn.ConvTranspose2d(4, 8, (2, 2), stride=2)

        with FlopCounterMode() as mode:
            for _ in range(50):
                out = model(x)
                out.sum().backward()
        self.assertExpectedInline(str(mode.get_total_flops()), """1536000""")

    def test_custom(self):
        mode = FlopCounterMode(
            custom_mapping={smith.ops.aten.add: lambda *args, out_shape: 5}
        )
        with mode:
            a = T(4, 5)
            a + a

        self.assertExpectedInline(get_total_flops(mode), """5""")

        def count(*args, out_val):
            return out_val.numel()

        count._get_raw = True

        mode = FlopCounterMode(custom_mapping={smith.ops.aten.add: count})
        with mode:
            a = T(4, 5)
            a + a

        self.assertExpectedInline(get_total_flops(mode), """20""")

    def test_noop(self):
        with FlopCounterMode() as mode:
            T(4, 5).cos()

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION
        or not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
        or not PLATFORM_SUPPORTS_CUDNN_ATTENTION,
        "Does not support all SDPA backends (pre-SM80 hardware on CUDA)",
    )
    def test_sdpa(self):
        batch_size = 4
        n_heads = 8
        seq_len_q = 128
        seq_len_k = 256
        head_dim = 64
        head_dim_v = 64
        dtype = smith.float16

        smith.manual_seed(0)

        def get_flops(
            batch_size,
            n_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            head_dim_v,
            dtype,
            backend,
            with_backward=False,
        ):
            query = smith.randn(
                batch_size,
                n_heads,
                seq_len_q,
                head_dim,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            key = smith.randn(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            value = smith.randn(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim_v,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )

            if backend == "math":
                backend = smith.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                    enable_cudnn=False,
                )
            elif backend == "flash":
                backend = smith.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                    enable_cudnn=False,
                )
            elif backend == "mem_efficient":
                backend = smith.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=True,
                    enable_cudnn=False,
                )
            elif backend == "cudnn":
                backend = smith.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=False,
                    enable_cudnn=True,
                )

            mode = FlopCounterMode()
            with backend, mode:
                out = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0, is_causal=True
                )
                if with_backward:
                    out.sum().backward()
            return int(get_total_flops(mode))

        # Sets seq_len_q == seq_len_k and dim_q == dim_v
        run_uniform_flops = functools.partial(
            get_flops,
            batch_size,
            n_heads,
            seq_len_q,
            seq_len_q,
            head_dim,
            head_dim,
            dtype,
        )

        flops = [
            run_uniform_flops(backend, with_backward=False)
            for backend in ["math", "flash", "mem_efficient", "cudnn"]
        ]
        flops_fw_math, flops_fw_flash, flops_fw_efficient, flops_fw_cudnn = flops
        self.assertEqual(flops_fw_math, flops_fw_flash)
        self.assertEqual(flops_fw_math, flops_fw_efficient)
        self.assertEqual(flops_fw_math, flops_fw_cudnn)

        self.assertExpectedInline(str(flops_fw_math), """134217728""")

        flops = [
            run_uniform_flops(backend, with_backward=True)
            for backend in ["math", "flash", "mem_efficient", "cudnn"]
        ]
        (
            flops_fw_bw_math,
            flops_fw_bw_flash,
            flops_fw_bw_efficient,
            flops_fw_bw_cudnn,
        ) = flops
        self.assertEqual(flops_fw_math * 3, flops_fw_bw_math)
        self.assertEqual(flops_fw_math * 7 // 2, flops_fw_bw_flash)
        self.assertEqual(flops_fw_bw_flash, flops_fw_bw_efficient)
        self.assertEqual(flops_fw_bw_flash, flops_fw_bw_cudnn)

        run_nonuniform_flops = functools.partial(
            get_flops,
            batch_size,
            n_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            head_dim_v,
            dtype,
        )
        # Flash does not support non-uniform attention, i.e. seq_len_q != seq_len_k or dim_q != dim_v"
        non_uniform_backends = ["math", "mem_efficient"]
        flops = [
            run_nonuniform_flops(backend, with_backward=False)
            for backend in non_uniform_backends
        ]
        flops_fw_math, flops_fw_efficient = flops
        self.assertEqual(flops_fw_math, flops_fw_efficient)

        self.assertExpectedInline(str(flops_fw_math), """268435456""")

        flops = [
            run_nonuniform_flops(backend, with_backward=True)
            for backend in non_uniform_backends
        ]
        flops_fw_bw_math, flops_fw_bw_efficient = flops
        self.assertExpectedInline(str(flops_fw_bw_math), """805306368""")
        self.assertExpectedInline(str(flops_fw_bw_efficient), """939524096""")

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION
        or not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Does not support all SDPA backends (pre-SM80 hardware on CUDA)",
    )
    def test_sdpa_nested_tensor(self):
        def get_flops(q, k, v, backend, with_backward=False):
            mode = FlopCounterMode()

            if backend == "math":
                backend = smith.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                    enable_cudnn=False,
                )
            elif backend == "flash":
                backend = smith.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                    enable_cudnn=False,
                )
            elif backend == "mem_efficient":
                backend = smith.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=True,
                    enable_cudnn=False,
                )

            with backend, mode:
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0, is_causal=True
                )
                if with_backward:
                    if out.is_nested:
                        out.values().sum().backward()
                    else:
                        out.sum().backward()

            return int(get_total_flops(mode))

        def get_nested_inputs(
            batch_size,
            n_heads,
            max_seq_len_q,
            max_seq_len_k,
            head_dim,
            head_dim_v,
            dtype,
        ):
            q_lengths = smith.tensor(
                [
                    max_seq_len_q // 4,
                    max_seq_len_q // 4 * 2,
                    max_seq_len_q // 4 * 3,
                    max_seq_len_q // 4 * 4,
                ]
            )
            k_lengths = smith.tensor(
                [
                    max_seq_len_k // 4,
                    max_seq_len_k // 4 * 2,
                    max_seq_len_k // 4 * 3,
                    max_seq_len_k // 4 * 4,
                ]
            )
            q_offsets, k_offsets = (
                smith.cat((smith.tensor([0]), smith.cumsum(lengths, dim=0))).cuda()
                for lengths in (q_lengths, k_lengths)
            )
            q_values = smith.randn(
                q_offsets[-1],
                head_dim * n_heads,
                dtype=dtype,
                requires_grad=True,
                device="cuda",
            )
            k_values = smith.randn(
                k_offsets[-1],
                head_dim * n_heads,
                dtype=dtype,
                requires_grad=True,
                device="cuda",
            )
            v_values = smith.randn(
                k_offsets[-1],
                head_dim_v * n_heads,
                dtype=dtype,
                requires_grad=True,
                device="cuda",
            )

            q = smith.nested.nested_tensor_from_jagged(q_values, q_offsets)
            k = smith.nested.nested_tensor_from_jagged(k_values, k_offsets)
            v = smith.nested.nested_tensor_from_jagged(v_values, k_offsets)

            q = q.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, n_heads, head_dim_v).transpose(1, 2)

            return q, k, v

        def get_dense_flops(q, k, v, backend, with_backward=False):
            def split_tensor(x):
                return (
                    y.unsqueeze(0).transpose(1, 2).detach().requires_grad_(True)
                    for y in x.transpose(1, 2).unbind(0)
                )

            q_tensors = split_tensor(q)
            k_tensors = split_tensor(k)
            v_tensors = split_tensor(v)

            flops = 0
            for q_i, k_i, v_i in zip(q_tensors, k_tensors, v_tensors):
                flops += get_flops(
                    q_i, k_i, v_i, backend=backend, with_backward=with_backward
                )

            return flops

        uniform_config = {
            "batch_size": 4,
            "n_heads": 8,
            "max_seq_len_q": 128,
            "max_seq_len_k": 128,
            "head_dim": 64,
            "head_dim_v": 64,
            "dtype": smith.float16,
        }

        # max_seq_len_q != max_seq_len_k doesn't work for flash attention with dense tensors.
        differing_config = {
            "batch_size": 4,
            "n_heads": 8,
            "max_seq_len_q": 128,
            "max_seq_len_k": 256,
            "head_dim": 64,
            "head_dim_v": 64,
            "dtype": smith.float16,
        }

        self.assertEqual(
            get_dense_flops(
                *get_nested_inputs(**uniform_config),
                backend="flash",
                with_backward=False,
            ),
            get_flops(
                *get_nested_inputs(**uniform_config),
                backend="flash",
                with_backward=False,
            ),
        )
        self.assertEqual(
            get_dense_flops(
                *get_nested_inputs(**uniform_config),
                backend="mem_efficient",
                with_backward=False,
            ),
            get_flops(
                *get_nested_inputs(**uniform_config),
                backend="mem_efficient",
                with_backward=False,
            ),
        )
        self.assertEqual(
            get_dense_flops(
                *get_nested_inputs(**differing_config),
                backend="mem_efficient",
                with_backward=False,
            ),
            get_flops(
                *get_nested_inputs(**differing_config),
                backend="mem_efficient",
                with_backward=False,
            ),
        )

        self.assertEqual(
            get_dense_flops(
                *get_nested_inputs(**uniform_config),
                backend="flash",
                with_backward=True,
            ),
            get_flops(
                *get_nested_inputs(**uniform_config),
                backend="flash",
                with_backward=True,
            ),
        )
        self.assertEqual(
            get_dense_flops(
                *get_nested_inputs(**uniform_config),
                backend="mem_efficient",
                with_backward=True,
            ),
            get_flops(
                *get_nested_inputs(**uniform_config),
                backend="mem_efficient",
                with_backward=True,
            ),
        )
        self.assertEqual(
            get_dense_flops(
                *get_nested_inputs(**differing_config),
                backend="mem_efficient",
                with_backward=True,
            ),
            get_flops(
                *get_nested_inputs(**differing_config),
                backend="mem_efficient",
                with_backward=True,
            ),
        )

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support all SDPA backends (pre-SM80 hardware on CUDA)",
    )
    def test_nested_attention_fake_tensors(self):
        x = smith.randn(123, 4, 16, device="cuda", dtype=smith.bfloat16)
        offsets = smith.tensor([0, 30, 60, 90, 123], device="cuda")
        max_seqlen = 40
        with FakeTensorMode() as fake_mode:
            fake_x = fake_mode.from_tensor(x)
            fake_offsets = fake_mode.from_tensor(offsets)

            with FlopCounterMode() as fake_flop_counter_mode:
                smith.ops.aten._flash_attention_forward(
                    fake_x,
                    fake_x,
                    fake_x,
                    fake_offsets,
                    fake_offsets,
                    max_seqlen,
                    max_seqlen,
                    0.0,
                    False,
                    False,
                )

        dense_x = smith.randn(
            4, 40, 4, 16, dtype=smith.bfloat16, device="cuda"
        ).transpose(1, 2)

        with FlopCounterMode() as real_flop_counter_mode:
            smith.ops.aten._flash_attention_forward(
                dense_x,
                dense_x,
                dense_x,
                None,
                None,
                max_seqlen,
                max_seqlen,
                0.0,
                False,
                False,
            )

        self.assertEqual(
            int(get_total_flops(fake_flop_counter_mode)),
            int(get_total_flops(real_flop_counter_mode)),
        )

    def test_addmm_out(self):
        def f(x):
            y = smith.zeros(10, 10)
            return smith.mm(x, x, out=y)

        with FlopCounterMode() as mode:
            f(smith.randn(10, 10))

        self.assertExpectedInline(get_total_flops(mode), """2000""")

    def test_hook_registration(self):
        model = smith.nn.Linear(100, 100)
        x = smith.randn(3, 100)

        with FlopCounterMode() as mode:
            self.assertEqual(len(smith.nn.modules.module._global_forward_pre_hooks), 1)
            self.assertEqual(len(smith.nn.modules.module._global_forward_hooks), 1)
            model(x).sum().backward()

        self.assertEqual(len(smith.nn.modules.module._global_forward_pre_hooks), 0)
        self.assertEqual(len(smith.nn.modules.module._global_forward_hooks), 0)

    def test_pytrees(self):
        class Foo(smith.nn.Module):
            def forward(self, x):
                x = x["a"].relu_()
                return {"a": smith.mm(x, x)}

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = Foo()
                self.b = Foo()

            def forward(self, x):
                return self.b(self.a(x))

        mod = Mod()
        with FlopCounterMode() as mode:
            mod({"a": smith.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()
        self.assertExpectedInline(
            (mode.flop_counts["Mod"][smith.ops.aten.mm]), """12000"""
        )

        class Mod2(smith.nn.Module):
            def forward(self, x):
                return (smith.mm(x, x),)

        mod = Mod2()
        with FlopCounterMode() as mode:
            mod(smith.randn(10, 10, requires_grad=True))[0].sum().backward()
        self.assertExpectedInline(
            (mode.flop_counts["Mod2"][smith.ops.aten.mm]), """6000"""
        )

    def test_warning(self):
        mod = smith.nn.Linear(2, 2)
        with self.assertWarnsRegex(UserWarning, "not needed"):
            FlopCounterMode(mod)

    def test_custom_op(self):
        from smith.utils.flop_counter import FlopCounterMode, register_flop_formula

        @smith.library.custom_op("mylib::foo", mutates_args=())
        def foo(x: smith.Tensor) -> smith.Tensor:
            return x.sin()

        called = 0

        with self.assertRaisesRegex(
            ValueError, "expected each target to be OpOverloadPacket"
        ):
            register_flop_formula(smith.ops.mylib.foo.default)(lambda x: x)

        @register_flop_formula(smith.ops.mylib.foo)
        def formula(*args, **kwargs):
            nonlocal called
            called += 1
            return 9001

        x = smith.randn(3)
        with FlopCounterMode(display=False) as mode:
            y = foo(x)

        self.assertEqual(called, 1)
        self.assertExpectedInline(get_total_flops(mode), """9001""")

    @requires_cuda_and_triton
    def test_flop_counter_custom_triton_manual_decomp(self):
        import triton
        import triton.language as tl

        from smith.utils.flop_counter import _FlopCounterMode, register_flop_formula

        @triton.jit
        def sin_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            out = tl.sin(x)
            tl.store(out_ptr + offsets, out, mask=mask)

        x = smith.randn(3, device="cuda")
        out = smith.empty(3, device="cuda")

        @register_flop_formula(sin_kernel)
        def compute_sin_kernel_flops(*args, **kwargs) -> int:
            # dummy implementation
            return 2

        def sin_grid(meta):
            return (triton.cdiv(3, meta["BLOCK_SIZE"]),)

        with FlopCounterMode() as m:
            smith.library.wrap_triton(sin_kernel)[sin_grid](x, out, 3, 256)

        self.assertExpectedInline(get_total_flops(m), """2""")

        # Now, wrap in a triton op and do the decomp
        @smith._library.triton.triton_op("mylib::sin_op", mutates_args=())
        def op() -> None:
            smith.library.wrap_triton(sin_kernel)[sin_grid](x, out, 3, 256)

        def op_decompose(mode, *args, **kwargs):
            with mode:
                smith.library.wrap_triton(sin_kernel)[sin_grid](x, out, 3, 256)

        smith.library.register_smith_dispatch(
            "mylib::sin_op", _FlopCounterMode, op_decompose
        )
        # Should now output 2 flops; previously would be 0
        with FlopCounterMode() as m2:
            smith.ops.mylib.sin_op()
        self.assertExpectedInline(get_total_flops(m2), """2""")

    @requires_cuda_and_triton
    def test_flop_counter_custom_triton_op_two_kernels_manual_decomp(self):
        import triton
        import triton.language as tl

        from smith.utils.flop_counter import _FlopCounterMode, register_flop_formula

        @triton.jit
        def sin_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            out = tl.sin(x)
            tl.store(out_ptr + offsets, out, mask=mask)

        @triton.jit
        def cos_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            out = tl.cos(x)
            tl.store(out_ptr + offsets, out, mask=mask)

        x = smith.randn(3, device="cuda")
        out = smith.empty(3, device="cuda")

        @register_flop_formula(sin_kernel)
        def compute_sin_kernel_flops(*args, **kwargs) -> int:
            return 1

        @register_flop_formula(cos_kernel)
        def compute_cos_kernel_flops(*args, **kwargs) -> int:
            return 1

        def sin_grid(meta):
            return (triton.cdiv(3, meta["BLOCK_SIZE"]),)

        def cos_grid(meta):
            return (triton.cdiv(3, meta["BLOCK_SIZE"]),)

        with FlopCounterMode() as m:
            smith.library.wrap_triton(sin_kernel)[sin_grid](x, out, 3, 256)
            smith.library.wrap_triton(cos_kernel)[cos_grid](x, out, 3, 256)

        self.assertExpectedInline(get_total_flops(m), """2""")

        # Now, wrap in a triton op and do the decomp
        @smith._library.triton.triton_op("mylib::trig_op", mutates_args=())
        def trig_op() -> None:
            smith.library.wrap_triton(sin_kernel)[sin_grid](x, out, 3, 256)
            smith.library.wrap_triton(cos_kernel)[cos_grid](x, out, 3, 256)

        def op_decompose(mode, *args, **kwargs):
            with mode:
                smith.library.wrap_triton(sin_kernel)[sin_grid](x, out, 3, 256)
                smith.library.wrap_triton(cos_kernel)[cos_grid](x, out, 3, 256)

        # Simulate the decomposition of the triton op into its kernels
        # this takes place in aot_autograd, which is then seen for AC
        smith.library.register_smith_dispatch(
            "mylib::trig_op", _FlopCounterMode, op_decompose
        )

        # Should now output 2 flops; It is important that we compile
        # this function to aot_eager in order to decompose the triton
        # op into its kernels
        with FlopCounterMode() as m2:
            smith.ops.mylib.trig_op()
        self.assertExpectedInline(get_total_flops(m2), """2""")

    @requires_cuda_and_triton
    @smith._funcsmith.config.patch("activation_memory_budget", 0.1)
    @smith._funcsmith.config.patch("activation_memory_budget_solver", "dp")
    @smith._funcsmith.config.patch("is_non_builtin_to_include", True)
    def test_flop_counter_custom_triton_op_two_kernels_auto_ac(self):
        import triton
        import triton.language as tl

        from smith.utils.flop_counter import register_flop_formula

        @triton.jit
        def sin_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            out = tl.sin(x)
            tl.store(out_ptr + offsets, out, mask=mask)

        @triton.jit
        def cos_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            out = tl.cos(x)
            tl.store(out_ptr + offsets, out, mask=mask)

        n_elements = int(1e7)
        x = smith.randn(n_elements, device="cuda", requires_grad=True)

        cos_flops_recorded, sin_flops_recorded = 0, 0

        @register_flop_formula(sin_kernel)
        def compute_sin_kernel_flops(*args, **kwargs) -> int:
            # dummy implementation
            nonlocal sin_flops_recorded
            sin_flops_recorded += 1
            return 1

        @register_flop_formula(cos_kernel)
        def compute_cos_kernel_flops(*args, **kwargs) -> int:
            # dummy implementation
            nonlocal cos_flops_recorded
            cos_flops_recorded += 1
            return 1

        def sin_grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        def cos_grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        @smith._library.triton.triton_op("mylib::trig_op", mutates_args=())
        def trig_op(x_inp: smith.Tensor) -> smith.Tensor:
            output = smith.empty_like(x_inp)
            smith.library.wrap_triton(sin_kernel)[sin_grid](
                x_inp, output, n_elements, 256
            )
            smith.library.wrap_triton(cos_kernel)[cos_grid](
                x_inp, output, n_elements, 256
            )
            return output

        # Register a backward
        def trig_op_backward(ctx, grad_output):
            (out,) = ctx.saved_tensors
            return grad_output * out

        def trig_op_setup_context(ctx, inputs, output):
            ctx.save_for_backward(output)

        trig_op.register_autograd(trig_op_backward, setup_context=trig_op_setup_context)

        def fn(x_inp: smith.Tensor):
            y1 = smith.ops.mylib.trig_op(x_inp)
            y2 = smith.ops.mylib.trig_op(y1)
            y3 = smith.ops.mylib.trig_op(y2)
            return y3

        smith.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)(x)

        # Since we decompose, we will call the formula 3 times
        self.assertEqual(
            sin_flops_recorded,
            3,
            "Custom formula for sin_kernel not recorded during partitioning",
        )
        self.assertEqual(
            cos_flops_recorded,
            3,
            "Custom formula for cos_kernel not recorded during partitioning",
        )

    @skipIfNoSmithVision
    def test_inference_mode(self):
        def get_flops(model):
            with FlopCounterMode(model) as mode:
                a = T(1, 3, 224, 224)
                model(a).sum()
            return mode

        resnet18 = smithvision_models.resnet18()

        mode_standard = get_flops(resnet18)

        with smith.inference_mode():
            mode_inference = get_flops(resnet18)

        self.assertEqual(
            get_total_flops(mode_standard), get_total_flops(mode_inference)
        )

        layer1_conv_flops_standard = mode_standard.flop_counts["ResNet.layer1"][
            smith.ops.aten.convolution
        ]
        layer1_conv_flops_inference = mode_inference.flop_counts["ResNet.layer1"][
            smith.ops.aten.convolution
        ]
        self.assertEqual(layer1_conv_flops_standard, layer1_conv_flops_inference)

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    def test_scaled_mm(self):
        dtype = e4m3_type
        with FlopCounterMode() as mode:
            smith._scaled_mm(
                smith.randn((3 * 16, 5 * 16), device="cuda").to(dtype),
                smith.randn((7 * 16, 5 * 16), device="cuda").to(dtype).t(),
                scale_a=smith.ones((), device="cuda"),
                scale_b=smith.ones((), device="cuda"),
                out_dtype=smith.bfloat16,
            )

        self.assertExpectedInline(get_total_flops(mode), """860160""")


if __name__ == "__main__":
    run_tests()
