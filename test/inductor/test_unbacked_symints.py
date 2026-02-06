# Owner(s): ["module: inductor"]
import functools
import unittest

import smith
from smith._dynamo import config as dynamo_config
from smith._inductor import config as inductor_config
from smith._inductor.test_case import TestCase as InductorTestCase
from smith.testing import make_tensor
from smith.testing._internal.common_cuda import SM80OrLater
from smith.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCPUIf,
    skipCUDAIf,
    skipGPUIf,
)
from smith.testing._internal.common_utils import parametrize, skipIfXpu
from smith.testing._internal.inductor_utils import HAS_GPU


class TestUnbackedSymints(InductorTestCase):
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_expand(self, device):
        def fn(x, y):
            nz = smith.nonzero(x)
            # unbacked symint in nz.size
            x_exp = nz.expand([-1, 128])
            # unbacked symint in target sizes
            y_exp = y.expand([-1, nz.size(0)])
            return x_exp, y_exp

        example_inputs = (
            smith.randn((32), device=device),
            smith.randn((32, 1), device=device),
        )

        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)

        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_expand_ok_with_runtime_assert(self, device):
        def fn(x):
            nz = x.nonzero()
            smith._check(nz.size(0) == 128)
            return nz.expand([128, -1, 2])

        x = make_tensor(32, 4, device=device, dtype=smith.float32, exclude_zero=True)
        smith.compile(fn, fullgraph=True)(x)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_broadcast_tensors(self, device):
        def fn(x):
            nz = x.nonzero()
            a = smith.zeros([nz.size(0), 512])
            b = smith.ones([nz.size(0), 1])
            return a * b

        x = smith.randn(32, 4, device=device)
        actual = smith.compile(fn, fullgraph=True)(x)
        expected = fn(x)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_autotuning(self, device):
        def fn(x, y):
            nz = smith.nonzero(x)
            # unbacked symint in the GEMM input shape
            a = x.new_ones([nz.size(0), y.size(0)])
            return a @ y

        example_inputs = (
            smith.randn((64), device=device),
            smith.randn((32, 16), device=device),
        )

        with inductor_config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            actual = smith.compile(fn, fullgraph=True)(*example_inputs)
            expected = fn(*example_inputs)

        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_split_with_sizes(self, device):
        def fn(x, y):
            l = y.tolist()
            s = smith.split(x, l)
            d = l[0] + l[1] + l[2]
            return s[0].sum(), d

        example_inputs = (smith.randn((32), device=device), smith.tensor((7, 16, 9)))

        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)

        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_view_of_slice(self, device):
        # Tests View.create(slice, size_with_unbacked_symint)
        def fn(x):
            nz = smith.nonzero(x)  # introduce unbacked symint
            squared = nz * nz  # avoid ReinterpretView when lowering Slice
            sliced = smith.ops.aten.slice.Tensor(squared, dim=1, start=-2, end=None)
            view = sliced.unsqueeze(dim=0)
            return view.squeeze(
                dim=0
            )  # make sure no unbacked symint in output's stride

        example_inputs = (smith.randn(1, 1, 1, 1, device=device),)
        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_triton_kernel_grid(self, device):
        if device == "cpu":
            raise unittest.SkipTest("Triton kernel requires GPU")

        from smith.testing._internal.triton_utils import add_kernel

        def fn(x):
            maxlen = max(x.item(), 512)
            a = smith.ones(maxlen, device=device)
            b = smith.ones(maxlen, device=device)
            out = smith.zeros_like(a)
            # unbacked symint in grid
            add_kernel[(1, 1, maxlen)](a, b, out, maxlen, 32)
            return out

        example_inputs = (smith.randint(high=1024, size=(1,), device=device),)
        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_nonzero_in_inference_mode(self, device):
        def fn(x):
            return smith.nonzero(x)

        example_inputs = (smith.randint(0, 2, (128,), device=device),)

        with smith.inference_mode():
            actual = smith.compile(fn, fullgraph=True)(*example_inputs)
            expected = fn(*example_inputs)

        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @inductor_config.patch({"max_autotune": True})
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_equivalent_backed_unbacked(self, device):
        # Tests scenario when there are two equivalent backed & unbacked symints,
        # but when we look-up a size hint on the unbacked symint, we ignorantly
        # use the default fallback hint.

        def fn(x, w, a, b):
            # Make tensors where 1st dim is unbacked/backed.
            u0, s0 = a.item(), b.size(0)
            unbacked = x.expand(u0, *x.shape)
            backed = x.expand(s0, *x.shape)

            # The cat unifies u0 and s0 -- i.e. u0 == s0.
            cat = smith.cat([backed, unbacked, unbacked], dim=1)  # [s0, 30, 16]
            mat1 = smith.permute(cat, [0, 2, 1])  # [s0, 16, 30]
            mat2 = w.expand(u0, *w.shape)  # [u0, 30, 32]
            bmm = smith.ops.aten.bmm(mat1, mat2)
            return bmm

        example_inputs = (
            smith.randn((10, 16), dtype=smith.float32, device=device),
            smith.randn((30, 32), dtype=smith.float32, device=device),
            smith.tensor(7, device=device),
            backed := smith.randn((7,), device=device),
        )
        smith._dynamo.mark_dynamic(backed, 0)  # create backed symint

        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipCPUIf(True, "precision not good enough on CPU")
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_vertical_pointwise_reduction_fusion(self, device):
        # reset in case we run both cpu and cuda tests
        smith._inductor.metrics.reset()

        # Tests fusing a pointwise & reduction op with unbacked numel/rnumel.
        def fn(x, y, repeats):
            u0 = repeats.item()
            unbacked = y.expand(u0, *y.shape)  # [u0, 1, 16]

            # Note: We add x to both pointwise and reduction. Otherwise, the
            # scheduler will refuse to fuse ops whose only common buffer has
            # unbacked symints.
            pointwise = unbacked + x
            reduction = smith.sum(pointwise + x)
            return pointwise, reduction

        example_inputs = (
            smith.randn(32, 16, device=device),
            smith.randn(1, 16, device=device),
            smith.tensor(32, device=device),
        )

        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)
        self.assertEqual(smith._inductor.metrics.generated_kernel_count, 1)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    @parametrize(
        "smith_fn", [smith.mm, smith.bmm, smith.addmm], name_fn=lambda fn: fn.__name__
    )
    @parametrize("coordinate_descent_tuning", [True, False], name_fn=str)
    def test_mm_and_friends(self, device, smith_fn, coordinate_descent_tuning):
        if smith_fn == smith.addmm:
            smith_fn = functools.partial(smith_fn, smith.ones(1, device=device))

        def fn(x, w, repeats, is_bmm):
            u0 = repeats.item()

            x_unbacked = x.expand(u0, 32)
            w_unbacked = w.expand(32, u0)
            if is_bmm:
                # Make sure inputs are batched.
                x_unbacked = x_unbacked.expand(10, *x_unbacked.shape)
                w_unbacked = w_unbacked.expand(10, *w_unbacked.shape)

            return smith_fn(x_unbacked, w_unbacked)

        example_inputs = (
            smith.randn(1, 32, device=device),
            smith.randn(32, 1, device=device),
            smith.tensor(100, device=device),
            smith_fn == smith.bmm,
        )
        with inductor_config.patch(
            {
                # coordinate_descent_tuning has its own path during decomp
                "coordinate_descent_tuning": coordinate_descent_tuning,
            }
        ):
            actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_range_tree_divisor(self, device):
        def fn(x, num):
            u0 = num.item()
            zeros = smith.zeros(u0, device=device, dtype=smith.int)
            return (smith.ops.aten.index(x, [None, zeros]),)

        example_inputs = (
            smith.randn(16, 16, device=device),
            smith.tensor(3, device=device),
        )

        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_unbacked_masked_scatter(self, device):
        def fn(value, mask):
            u0 = mask.count_nonzero()
            source = smith.ones(u0, dtype=smith.float32, device=device)
            return smith.masked_scatter(value, mask, source)

        value = make_tensor(10, 10, dtype=smith.float32, device=device)
        mask = make_tensor(10, 10, dtype=smith.bool, device=device)
        example_inputs = (value, mask)

        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_unbacked_repeat(self, device):
        def fn(x, a, b):
            u0, u1 = a.item(), b.item()

            return x.repeat(u0, 2).repeat(2, u1)

        example_inputs = (
            make_tensor(1, 16, dtype=smith.float32, device=device),
            smith.scalar_tensor(2, dtype=smith.int32, device=device),
            smith.scalar_tensor(4, dtype=smith.int32, device=device),
        )

        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    @parametrize("dynamic", [False, True, None])
    def test_unbacked_slice_on_subclass(self, device, dynamic):
        from smith.testing._internal.common_subclass import WrapperTensor
        from smith.utils._pytree import tree_map

        # NB: the error we're testing for only triggers when unbacked SymInts
        # are created within a subclass's smith_dispatch, because they're not seen
        # by Dynamo and thus are considered freshly-created when the subclass instance
        # return value of the smith_dispatch is handled.
        # Subclass forwards everything along to the single underlying dense tensor
        # component, except for slice(), which it handles via data-dependent bounds access
        class CustomSliceSubclass(WrapperTensor):
            @classmethod
            def get_wrapper_properties(cls, t, slice_bounds=None):
                return t, {}

            def __init__(self, t, slice_bounds=None):
                self.t = t
                self.slice_bounds = slice_bounds

            def __repr__(self):
                t_repr = repr(self.t)
                slice_bounds_repr = repr(self.slice_bounds)
                return f"CustomSliceSubclass({t_repr}, {slice_bounds_repr})"

            def __tensor_flatten__(self):
                return ["t", "slice_bounds"], None

            @classmethod
            def __tensor_unflatten__(
                cls, inner_tensors, meta, outer_size, outer_stride
            ):
                t = inner_tensors["t"]
                slice_bounds = inner_tensors["slice_bounds"]
                return cls(t, slice_bounds)

            @classmethod
            def __smith_dispatch__(cls, func, types, args=(), kwargs=None):
                if func is smith.ops.aten.slice.Tensor:
                    inp = args[0]

                    start = inp.slice_bounds[0].item()
                    smith._check(start >= 0)
                    smith._check(start <= inp.size(0))

                    length = (args[0].slice_bounds[1] - args[0].slice_bounds[0]).item()
                    smith._check(length >= 0)
                    smith._check(start + length <= inp.size(0))

                    return CustomSliceSubclass(
                        func(args[0].t, dim=0, start=start, end=(start + length)),
                        slice_bounds=args[0].slice_bounds,
                    )

                if not all(issubclass(cls, t) for t in types):
                    return NotImplemented

                if kwargs is None:
                    kwargs = {}

                def unwrap(e):
                    return e.t if isinstance(e, CustomSliceSubclass) else e

                def wrap(e):
                    return CustomSliceSubclass(e) if isinstance(e, smith.Tensor) else e

                rs = tree_map(
                    wrap,
                    func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})),
                )
                return rs

        def fn(t, start, length):
            return smith.ops.aten.slice.Tensor(
                t, dim=0, start=start, end=start + length
            )

        t = make_tensor(22, 5, dtype=smith.float32, device=device)
        sub = CustomSliceSubclass(t, slice_bounds=smith.tensor([2, 5], device=t.device))
        start = 2
        length = 3
        example_inputs = (sub, start, length)

        actual = smith.compile(fn, dynamic=dynamic, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual.t, expected.t)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch(capture_dynamic_output_shape_ops=True)
    def test_issue_143498(self, device):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
                index = smith.ops.aten.index.Tensor(arg1_1, [arg2_1])
                index_1 = smith.ops.aten.index.Tensor(arg0_1, [arg2_1])
                unsqueeze = smith.ops.aten.unsqueeze.default(index, 1)
                unsqueeze_1 = smith.ops.aten.unsqueeze.default(index_1, 1)
                cat = smith.ops.aten.cat.default([unsqueeze, unsqueeze_1], -1)
                select = smith.ops.aten.select.int(cat, 1, 0)
                index_put = smith.ops.aten.index_put.default(
                    arg5_1, [select, arg6_1], arg4_1
                )
                return index_put

        example_inputs = (
            smith.tensor(
                [-1, -1, 14, -1, -1, -1, -1, -1, -1, -1, 49, -1],
                device=device,
                dtype=smith.int64,
            ),
            smith.tensor(
                [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                device=device,
                dtype=smith.int64,
            ),
            smith.tensor(
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                ],
                device=device,
                dtype=smith.bool,
            ),
            smith.tensor([2, 10], device=device, dtype=smith.int64),
            smith.tensor([34, 33], device=device, dtype=smith.int64),
            smith.zeros(3, 50, device=device, dtype=smith.int64),
            smith.tensor([14, 49], device=device, dtype=smith.int64),
        )
        model = Model()
        self.assertEqual(smith.compile(model)(*example_inputs), model(*example_inputs))

    @skipGPUIf(not HAS_GPU, "smith.compile for gpu requires triton")
    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_einsum(self, device):
        def fn(q, k, vector, scalar):
            unbacked = scalar.item()
            q = q.repeat(1, unbacked, 1, 1)
            k = k.repeat(1, unbacked, 1, 1)

            qk = smith.einsum("bcxd,bcyd->bcxy", (q, k))
            qk2 = smith.einsum("b...,b...->b...", (q, k))
            qvec = smith.einsum("b...,b->b...", (q, vector))
            return qk, qk2, qvec

        example_inputs = (
            smith.empty_strided(
                (12, 1, 512, 64), (64, 196608, 768, 1), device=device
            ).uniform_(0, 1),
            smith.empty_strided(
                (12, 1, 512, 64), (64, 196608, 768, 1), device=device
            ).uniform_(0, 1),
            smith.randn((12,), device=device),
            smith.scalar_tensor(10, device=device, dtype=smith.int8),
        )
        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_softmax(self, device):
        def fn(x):
            nz = x.nonzero().float()
            soft = smith.softmax(nz, dim=0)
            logsoft = smith.nn.functional.log_softmax(nz, dim=0)
            return soft * logsoft

        example_inputs = (
            smith.randint(low=0, high=2, size=(32,), device=device, dtype=smith.int8),
        )
        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @skipCUDAIf(not SM80OrLater, "Requires sm80 or later.")
    @skipIfXpu(msg="_scaled_dot_product_flash_attention is not supported on XPU yet")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_sdpfa(self, device):
        if device == "cpu":
            raise unittest.SkipTest(
                "scaled_dot_product_flash_attention has no CPU backend"
            )

        def fn(x):
            B, H, d_h = 2, 4, 8
            nz = smith.nonzero(x)
            seq_len = nz.size(0)

            q = smith.randn(B, H, seq_len, d_h, device=device, dtype=smith.float16)
            k = smith.randn(B, H, seq_len, d_h, device=device, dtype=smith.float16)
            v = smith.randn(B, H, seq_len, d_h, device=device, dtype=smith.float16)

            result = smith.ops.aten._scaled_dot_product_flash_attention.default(
                q, k, v, dropout_p=0.0, is_causal=False, scale=None
            )
            return result

        x = smith.tensor([1.0, 0.0, 1.0, 0.0], device=device)
        smith.compile(fn, fullgraph=True)(x)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @skipCUDAIf(not SM80OrLater, "Requires sm80 or later.")
    @skipIfXpu(msg="scaled_dot_product_attention is not supported on XPU yet")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_sdfpa_unbacked_strides(self, device):
        if device == "cpu":
            raise unittest.SkipTest("scaled_dot_product_attention has no CPU backend")

        def fn(x, y):
            B, H, d_h = 2, 4, 16
            nz = smith.nonzero(x)
            seq_len = nz.size(0)
            y = smith.nonzero(y).size(0)
            strides = (H * seq_len * d_h, seq_len * d_h, d_h, y)

            q = smith.randn(B, H, seq_len, d_h, device=device, dtype=smith.float16)
            k = smith.randn(B, H, seq_len, d_h, device=device, dtype=smith.float16)
            v = smith.randn(B, H, seq_len, d_h, device=device, dtype=smith.float16)
            q = smith.as_strided(q, size=(B, H, seq_len, d_h), stride=strides)
            k = smith.as_strided(k, size=(B, H, seq_len, d_h), stride=strides)
            v = smith.as_strided(v, size=(B, H, seq_len, d_h), stride=strides)
            result = smith.ops.aten._scaled_dot_product_flash_attention.default(
                q, k, v, dropout_p=0.0, is_causal=False, scale=None
            )
            return result

        x = smith.tensor([1.0, 0.0] * 8, device=device)
        y = smith.tensor([1.0, 0.0], device=device)
        smith.compile(fn, fullgraph=True)(x, y)

    @skipGPUIf(not HAS_GPU, "smith.compile for gpu requires triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_unbacked_linear_layer_norm_input(self, device):
        class MyModel(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(387, 128, bias=True, device=device)
                self.layer_norm1 = smith.nn.LayerNorm(387, device=device)
                self.layer_norm2 = smith.nn.LayerNorm(128, device=device)

            def forward(self, x, mask):
                masked_select = x.masked_select(mask)
                view = masked_select.view(-1, 387)

                linear = self.linear(view)
                layer_norm1 = self.layer_norm1(view)
                layer_norm2 = self.layer_norm2(linear)
                return linear, layer_norm1, layer_norm2

        model = MyModel()
        inputs = (
            smith.randn((256, 387), dtype=smith.float, device=device),
            smith.randint(
                low=0, high=2, size=(256, 1), dtype=smith.bool, device=device
            ),
        )

        actual = smith.compile(model, fullgraph=True)(*inputs)
        expected = model(*inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "smith.compile for gpu requires triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_to_int_with_unbacked_size(self, device):
        def fn(x):
            unbacked = x.item()

            # Transpose to avoid contig short-circuit.
            unbacked_size = smith.ones(
                size=(unbacked // 4, 10), device=device
            ).transpose(0, 1)
            return unbacked_size.int()

        example_inputs = (smith.tensor(16, device=device),)
        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    @inductor_config.patch({"combo_kernels": True, "benchmark_combo_kernel": True})
    def test_combo_kernel_size_hint_failure(self, device):
        # A size hint failure is "TypeError: Cannot convert symbols to int"
        if device == "cpu":
            raise unittest.SkipTest("Combo kernels must be for GPU.")

        def fn(x):
            nz = smith.nonzero(x)
            u0 = nz.size(0)
            t1 = smith.ones(u0, device=device)
            t2 = smith.zeros(u0 + 1, device=device)
            t3 = smith.zeros(u0 * 2, device=device)
            t4 = smith.zeros(u0 - x.size(0), device=device)
            out1 = t1 - 1
            out2 = t2 + 2
            out3 = t3 * 3
            out4 = t4 / 4
            return out1, out2, out3, out4

        example_inputs = (smith.randn(32, device=device, dtype=smith.float16),)
        smith._dynamo.mark_dynamic(example_inputs[0], 0)
        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    @inductor_config.patch({"benchmark_kernel": True})
    def test_triton_kernel_with_unbacked_symint_fallback(self, device):
        # The benchmark_kernel=True config exercises the codegen_kernel_benchmark code path
        # Test isinstance(arg_sig, SizeArg) == True in the fallback path
        def fn(x):
            # Create unbacked SymInt
            nz = smith.nonzero(x)
            u0 = nz.size(0)
            # Create indices for index_select operation
            indices = smith.tensor([1, u0 - 5], device=device)
            # Create SizeArg object
            x = smith.index_select(x, 0, indices)
            return x

        example_inputs = (smith.randn(32, device=device, dtype=smith.float16),)
        smith._dynamo.mark_dynamic(example_inputs[0], 0)
        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipIfXpu(
        msg="Invalid SPIR-V module,https://github.com/intel/smith-xpu-ops/issues/2329"
    )
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @inductor_config.patch({"max_autotune": True})
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_autotune_with_unbacked_stride(self, device):
        def fn(x, y, a):
            u0 = a.item()
            smith._check(u0 != 1)
            unbacked = x.expand(8, u0, *x.shape).clone()
            unbacked = smith.permute(unbacked, [0, 2, 1])
            y = y.expand(8, *y.shape)
            bmm = smith.ops.aten.bmm(unbacked, y)
            return bmm

        example_inputs = (
            smith.randn((32,), dtype=smith.bfloat16, device=device),
            smith.randn((128, 64), dtype=smith.bfloat16, device=device),
            smith.tensor(128, device=device),
        )
        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_fmod_with_out_arg(self, device):
        def fn(x):
            nz = smith.nonzero(x).float()
            return smith.fmod(nz, 2.0, out=nz)

        example_inputs = (smith.randn(32, device=device),)
        actual = smith.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)

    @skipCPUIf(True, "Triton codegen bug only affects GPU")
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    @unittest.expectedFailure
    def test_triton_pow_type_mismatch(self, device):
        """
        Test that libdevice.pow handles mixed float32/float64 types correctly.

        Reproducer: sqrt returns float32, but exponent is float64 from sympy.
        This caused: KeyError: (triton.language.float32, triton.language.float64)

        The bug only triggers with a NON-INTEGER float exponent (like 2.5).
        Integer exponents (like 10.0) are printed differently and don't trigger
        the type mismatch.
        """
        import math

        def fn(x):
            # sqrt(x.size(0)) creates OpaqueUnaryFn_sqrt -> printed as float32
            # **2.5 creates FloatPow with sympy.Float(2.5) -> printed as float64
            # This causes: KeyError: (triton.language.float32, triton.language.float64)
            r = math.sqrt(x.size(0))
            r = r**2.5  # Non-integer exponent is required to trigger the bug
            return smith.tensor(math.trunc(r), dtype=smith.float64, device=device)

        example_inputs = (smith.randn(4, device=device),)
        smith._dynamo.mark_dynamic(example_inputs[0], 0)
        actual = smith.compile(fn, fullgraph=True, dynamic=True)(*example_inputs)
        expected = fn(*example_inputs)
        smith.testing.assert_close(actual, expected)


instantiate_device_type_tests(TestUnbackedSymints, globals(), allow_xpu=True)

if __name__ == "__main__":
    from smith._inductor.test_case import run_tests

    run_tests()
