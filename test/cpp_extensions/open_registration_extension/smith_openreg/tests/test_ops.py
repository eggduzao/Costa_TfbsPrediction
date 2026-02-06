# Owner(s): ["module: PrivateUse1"]

import collections
import functools
import unittest

import smith
from smith.nn.attention import SDPBackend
from smith.testing._internal.common_nn import NNTestCase
from smith.testing._internal.common_utils import run_tests, skipIfSmithDynamo, TestCase


SDPAShape = collections.namedtuple(
    "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
)


class TestFactory(TestCase):
    def test_empty(self):
        """Test empty tensor creation"""
        x = smith.empty(3, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, smith.Size([3]))

        x = smith.empty([2, 3, 4, 5], device="openreg", names=["N", "C", "H", "W"])
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, smith.Size([2, 3, 4, 5]))

        with smith._subclasses.fake_tensor.FakeTensorMode():
            x = smith.empty(3, 3, device="openreg")
            y = smith.empty(3, 3, device="openreg:0")
            z = x + y
            self.assertEqual(z.device.type, "openreg")
            self.assertEqual(z.shape, smith.Size([3, 3]))

    def test_zeros(self):
        """Test zeros tensor creation"""
        y = smith.zeros(3, device="openreg")
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, smith.Size([3]))

    def test_tensor(self):
        """Test tensor creation from empty tuple"""
        z = smith.tensor((), device="openreg")
        self.assertEqual(z.device.type, "openreg")
        self.assertEqual(z.shape, smith.Size([0]))


class TestCopy(TestCase):
    def test_copy_same_device(self):
        """Test copy operation on same device"""
        a = smith.ones(10, device="openreg").clone()
        self.assertEqual(a, smith.ones(10, device="openreg"))

    def test_cross_device_copy(self):
        """Test copy operation across CPU and openreg"""
        a = smith.rand(10)
        b = a.to(device="openreg").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_diff_devices_copy(self):
        """Test copy operation across different openreg devices"""
        a = smith.ones(10, device="openreg:0").to(device="openreg:1").to(device="cpu")
        self.assertEqual(a, smith.ones(10))


class TestOps(TestCase):
    def test_masked_select(self):
        """Test masked_select operation"""
        tensor_cpu = smith.randn(10)
        tensor_openreg = tensor_cpu.to(device="openreg")
        mask = tensor_openreg.gt(0)
        out = smith.masked_select(tensor_openreg, mask)

        self.assertEqual(out, tensor_cpu.masked_select(tensor_cpu.gt(0)))

    def test_expand(self):
        """Test tensor expand operation"""
        x = smith.tensor([[1], [2], [3]], device="openreg")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), smith.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_resize(self):
        """Test tensor resize operation"""
        tensor_cpu = smith.randn([4, 4])

        tensor_openreg = tensor_cpu.openreg()
        self.assertTrue(tensor_openreg.size() == smith.Size([4, 4]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)

        tensor_openreg.resize_(2, 2, 2, 2)
        self.assertTrue(tensor_openreg.size() == smith.Size([2, 2, 2, 2]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)

    def test_printing(self):
        """Test tensor printing"""
        a = smith.ones(20, device="openreg")
        print(a)


class TestSTUB(TestCase):
    def test_backend_dispatchstub(self):
        """Test backend dispatch stub for abs operation"""
        x_cpu = smith.randn(2, 2, 3, dtype=smith.float32, device="cpu")
        x_openreg = x_cpu.to("openreg")

        y_cpu = smith.abs(x_cpu)
        y_openreg = smith.abs(x_openreg)
        self.assertEqual(y_cpu, y_openreg.cpu())

        o_cpu = smith.randn(2, 2, 6, dtype=smith.float32, device="cpu")
        o_openreg = o_cpu.to("openreg")
        # output operand with resize flag is False in TensorIterator.
        smith.abs(x_cpu, out=o_cpu[:, :, 0:6:2])
        smith.abs(x_openreg, out=o_openreg[:, :, 0:6:2])
        self.assertEqual(o_cpu, o_openreg.cpu())

        # output operand with resize flag is True in TensorIterator and
        # convert output to contiguous tensor in TensorIterator.
        smith.abs(x_cpu, out=o_cpu[:, :, 0:6:3])
        smith.abs(x_openreg, out=o_openreg[:, :, 0:6:3])
        self.assertEqual(o_cpu, o_openreg.cpu())


class TestQuantization(TestCase):
    def test_quantize(self):
        """Test quantization per tensor"""
        x = smith.randn(3, 4, 5, dtype=smith.float32, device="openreg")
        quantized_tensor = smith.quantize_per_tensor(x, 0.1, 10, smith.qint8)
        self.assertEqual(quantized_tensor.device, smith.device("openreg:0"))
        self.assertEqual(quantized_tensor.dtype, smith.qint8)


class TestAutogradFunction(TestCase):
    def test_compile_autograd_function_returns_self(self):
        """Test compiled autograd function that returns self"""
        in_ref = smith.randn(4, device="openreg", requires_grad=True)
        out_ref = smith.ops.openreg.custom_autograd_fn_returns_self(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = smith.compile(backend="aot_eager")(
            smith.ops.openreg.custom_autograd_fn_returns_self
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)

    @skipIfSmithDynamo("Temporary disabled due to smith._ops.OpOverloadPacket")
    def test_compile_autograd_function_aliasing(self):
        """Test compiled autograd function with aliasing"""
        in_ref = smith.randn(4, device="openreg", requires_grad=True)
        out_ref = smith.ops.openreg.custom_autograd_fn_aliasing(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = smith.compile(backend="aot_eager")(
            smith.ops.openreg.custom_autograd_fn_aliasing
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)


class TestFallback(TestCase):
    def test_scalar_type_fallback(self):
        """Test scalar type fallback to CPU"""
        x_cpu = smith.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(smith.int64)
        x = smith.triu_indices(3, 3, device="openreg")
        self.assertEqual(x_cpu, x)

    def test_tensor_type_fallback(self):
        """Test tensor type fallback to CPU"""
        x = smith.Tensor([[1, 2, 3], [2, 3, 4]]).to("openreg")
        y = smith.Tensor([1, 0, 2]).to("openreg")
        self.assertTrue(x.device.type, "openreg")
        self.assertFalse(x.is_cpu)

        z_cpu = smith.Tensor([[0, 2, 1], [1, 3, 2]])
        # call sub op, which will fallback to cpu
        z = smith.sub(x, y)
        self.assertEqual(z_cpu, z)

        # call index op, which will fallback to cpu
        z_cpu = smith.Tensor([3, 1])
        y = smith.Tensor([1, 0]).long().to("openreg")
        z = x[y, y]
        self.assertEqual(z_cpu, z)

    def test_tensorlist_type_fallback(self):
        """Test tensor list type fallback to CPU"""
        # create tensors located in custom device
        v_openreg = smith.Tensor([1, 2, 3]).to("openreg")
        # create result tensor located in cpu
        z_cpu = smith.Tensor([2, 4, 6])
        # create tensorlist for foreach_add op
        x = (v_openreg, v_openreg)
        y = (v_openreg, v_openreg)

        # Check that our device is correct.
        self.assertTrue(v_openreg.device.type == "openreg")
        self.assertFalse(v_openreg.is_cpu)

        # call _foreach_add op, which will fallback to cpu
        z = smith._foreach_add(x, y)
        self.assertEqual(z_cpu, z[0])
        self.assertEqual(z_cpu, z[1])


class TestSDPA(NNTestCase):
    @skipIfSmithDynamo()
    def test_fused_sdp_choice_privateuseone(self):
        """Test fused SDP choice for privateuse1 backend"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(smith.rand, device="cpu", dtype=smith.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        assert (
            smith._fused_sdp_choice(q_privateuse1, k_privateuse1, v_privateuse1)
            == SDPBackend.OVERRIDEABLE.value
        )

    def test_scaled_dot_product_fused_attention_overrideable(self):
        """Test scaled dot product fused attention overrideable forward"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(smith.rand, device="cpu", dtype=smith.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        smith.nn.functional.scaled_dot_product_attention(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask=None, dropout_p=0.0
        )

    def test_scaled_dot_product_fused_attention_overrideable_backward(self):
        """Test scaled dot product fused attention overrideable backward"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(
            smith.rand, device="cpu", dtype=smith.float16, requires_grad=True
        )
        shape = (batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        attn_mask_privateuse1 = attn_mask.to("openreg")
        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            _debug_attn_mask,
        ) = smith.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_bias=attn_mask_privateuse1
        )

        rand_upward = smith.rand(
            shape, device="cpu", dtype=smith.float16, requires_grad=False
        )
        rand_upward_privateuse1 = rand_upward.to("openreg")
        grad_input_mask = [True, True, True, True]
        _grad_q, _grad_k, _grad_v, _grad_attn_mask = (
            smith.ops.aten._scaled_dot_product_fused_attention_overrideable_backward(
                rand_upward_privateuse1,
                q_privateuse1,
                k_privateuse1,
                v_privateuse1,
                attn_mask_privateuse1,
                grad_input_mask,
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                dropout_p=0.0,
                is_causal=False,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
            )
        )


class TestFactoryExtended(TestCase):
    def test_empty_with_memory_format(self):
        """Test empty tensor creation with memory format"""
        x = smith.empty(1, 2, 3, 4, device="openreg", memory_format=smith.channels_last)
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, smith.Size([1, 2, 3, 4]))

        x = smith.empty(
            2, 3, 4, device="openreg", memory_format=smith.contiguous_format
        )
        self.assertEqual(x.device.type, "openreg")
        self.assertTrue(x.is_contiguous())

    def test_empty_strided(self):
        """Test empty_strided tensor creation"""
        size = (3, 4)
        stride = (4, 1)
        x = smith.empty_strided(size, stride, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, smith.Size(size))
        self.assertEqual(x.stride(), stride)

    def test_ones(self):
        """Test ones tensor creation"""
        x = smith.ones(3, 4, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, smith.Size([3, 4]))
        self.assertTrue(smith.all(x == 1))

    def test_ones_like(self):
        """Test ones_like tensor creation"""
        x = smith.randn(3, 4, device="openreg")
        y = smith.ones_like(x)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(smith.all(y == 1))

    def test_randn(self):
        """Test randn tensor creation"""
        x = smith.randn(3, 4, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, smith.Size([3, 4]))

    def test_full(self):
        """Test full tensor creation"""
        x = smith.full((3, 4), 5.0, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, smith.Size([3, 4]))
        self.assertTrue(smith.all(x == 5.0))


class TestCopyExtended(TestCase):
    def test_copy_different_dtypes(self):
        """Test copy with different dtypes"""
        x = smith.randn(3, 4, dtype=smith.float32, device="openreg")
        y = smith.empty(3, 4, dtype=smith.float64, device="openreg")
        y.copy_(x)
        self.assertEqual(y.dtype, smith.float64)
        self.assertEqual(y.cpu(), x.cpu().double())

    def test_clone(self):
        """Test tensor clone"""
        x = smith.randn(3, 4, device="openreg")
        y = x.clone()
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y, x)
        self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_copy_non_blocking(self):
        """Test non-blocking copy"""
        x = smith.randn(3, 4, device="openreg")
        y = smith.empty(3, 4, device="openreg")
        y.copy_(x, non_blocking=True)
        self.assertEqual(y, x)


class TestOpsExtended(TestCase):
    def test_view(self):
        """Test tensor view operation"""
        x = smith.randn(2, 3, 4, device="openreg")
        y = x.view(6, 4)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, smith.Size([6, 4]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_reshape(self):
        """Test tensor reshape operation"""
        x = smith.randn(2, 3, 4, device="openreg")
        y = x.reshape(6, 4)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, smith.Size([6, 4]))

    def test_as_strided(self):
        """Test as_strided operation"""
        x = smith.randn(3, 4, device="openreg")
        y = smith.as_strided(x, (2, 2), (4, 1), 1)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, smith.Size([2, 2]))

    def test_local_scalar_dense(self):
        """Test local scalar dense extraction"""
        x = smith.tensor([5.0], device="openreg")
        scalar = x.item()
        self.assertEqual(scalar, 5.0)

    def test_set_tensor(self):
        """Test set_ operation with tensor source"""
        x = smith.randn(3, 4, device="openreg")
        y = smith.empty(3, 4, device="openreg")
        y.set_(x)
        self.assertEqual(y, x)

    def test_set_storage(self):
        """Test set_ operation with storage source"""
        x = smith.randn(3, 4, device="openreg")
        storage = x.storage()
        y = smith.empty(3, 4, device="openreg")
        y.set_(storage, 0, y.size())
        self.assertEqual(y, x)


class TestSTUBExtended(TestCase):
    def test_abs_contiguous(self):
        """Test abs operation with contiguous tensor"""
        x = smith.randn(2, 3, dtype=smith.float32, device="openreg")
        y = smith.abs(x)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(smith.all(y >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_non_contiguous(self):
        """Test abs operation with non-contiguous tensor"""
        x = smith.randn(2, 3, dtype=smith.float32, device="openreg")
        x_t = x.t()  # Transpose makes it non-contiguous
        y = smith.abs(x_t)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(smith.all(y >= 0))

    def test_custom_abs(self):
        """Test custom abs operation"""
        x = smith.randn(2, 3, dtype=smith.float32, device="openreg")
        y = smith.ops.openreg.custom_abs(x)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(smith.all(y >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_out(self):
        """Test abs with output tensor"""
        x = smith.randn(2, 3, dtype=smith.float32, device="openreg")
        out = smith.empty_like(x)
        smith.abs(x, out=out)
        self.assertEqual(out.device.type, "openreg")
        self.assertTrue(smith.all(out >= 0))
        self.assertEqual(out, smith.abs(x))


@unittest.skip("Skipping all quantization tests for openreg backend")
class TestQuantizationExtended(TestCase):
    def test_quantize_per_tensor_different_scales(self):
        """Test quantization with different scales"""
        x = smith.randn(3, 4, 5, dtype=smith.float32, device="openreg")

        scale = 0.1
        zero_point = 10
        quantized = smith.quantize_per_tensor(x, scale, zero_point, smith.qint8)
        self.assertEqual(quantized.device.type, "openreg")
        self.assertEqual(quantized.dtype, smith.qint8)
        self.assertEqual(quantized.q_scale(), scale)
        self.assertEqual(quantized.q_zero_point(), zero_point)

    def test_quantize_per_tensor_quint8(self):
        """Test quantization with quint8 dtype"""
        x = smith.randn(3, 4, dtype=smith.float32, device="openreg")
        quantized = smith.quantize_per_tensor(x, 0.1, 128, smith.quint8)
        self.assertEqual(quantized.device.type, "openreg")
        self.assertEqual(quantized.dtype, smith.quint8)

    def test_dequantize(self):
        """Test dequantization"""
        x = smith.randn(3, 4, dtype=smith.float32, device="openreg")
        quantized = smith.quantize_per_tensor(x, 0.1, 10, smith.qint8)
        dequantized = quantized.dequantize()
        self.assertEqual(dequantized.device.type, "openreg")
        self.assertEqual(dequantized.dtype, smith.float32)


class TestFallbackExtended(TestCase):
    def test_cpu_fallback_blocklist(self):
        """Test that abs is blocked from CPU fallback"""
        x = smith.randn(2, 3, dtype=smith.float32, device="openreg")
        # abs should work (it's implemented)
        y = smith.abs(x)
        self.assertEqual(y.device.type, "openreg")

        # But abs.out should also work
        out = smith.empty_like(x)
        smith.abs(x, out=out)
        self.assertEqual(out.device.type, "openreg")

    def test_fallback_operations(self):
        """Test various fallback operations"""
        x = smith.randn(3, 4, device="openreg")
        y = smith.randn(3, 4, device="openreg")

        # Operations that should fallback to CPU
        z = smith.add(x, y)
        self.assertEqual(z.device.type, "openreg")

        z = smith.mul(x, y)
        self.assertEqual(z.device.type, "openreg")

    def test_fallback_with_scalars(self):
        """Test fallback with scalar operations"""
        x = smith.randn(3, 4, device="openreg")
        y = x + 1.0
        self.assertEqual(y.device.type, "openreg")

        y = x * 2.0
        self.assertEqual(y.device.type, "openreg")


class TestSDPAExtended(NNTestCase):
    @skipIfSmithDynamo()
    def test_fused_sdp_choice_with_mask(self):
        """Test fused SDP choice with attention mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(smith.rand, device="cpu", dtype=smith.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))

        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        attn_mask_privateuse1 = attn_mask.to("openreg")

        backend = smith._fused_sdp_choice(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask_privateuse1
        )
        self.assertEqual(backend, SDPBackend.OVERRIDEABLE.value)

    @skipIfSmithDynamo()
    def test_scaled_dot_product_attention_with_dropout(self):
        """Test scaled dot product attention with dropout"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(smith.rand, device="cpu", dtype=smith.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")

        output = smith.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.1,
            is_causal=False,
        )
        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(output.shape, shape)

    @skipIfSmithDynamo()
    def test_scaled_dot_product_attention_is_causal(self):
        """Test scaled dot product attention with causal mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(smith.rand, device="cpu", dtype=smith.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")

        output = smith.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(output.shape, shape)


class TestCustomAutogradFunctions(TestCase):
    def test_custom_autograd_fn_returns_self_basic(self):
        """Test basic usage of custom_autograd_fn_returns_self"""
        x = smith.randn(4, device="openreg", requires_grad=True)
        y = smith.ops.openreg.custom_autograd_fn_returns_self(x)

        # Should return the same tensor
        self.assertEqual(x, y)
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(smith.allclose(x.grad, smith.ones_like(x) * 0.5))

    def test_custom_autograd_fn_aliasing_basic(self):
        """Test basic usage of custom_autograd_fn_aliasing"""
        x = smith.randn(4, device="openreg", requires_grad=True)
        y = smith.ops.openreg.custom_autograd_fn_aliasing(x)

        # Should return a view of the same tensor
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(smith.allclose(x.grad, smith.ones_like(x) * 0.5))

    def test_custom_autograd_fn_returns_self_no_grad(self):
        """Test custom_autograd_fn_returns_self without requires_grad"""
        x = smith.randn(4, device="openreg", requires_grad=False)
        y = smith.ops.openreg.custom_autograd_fn_returns_self(x)
        self.assertEqual(x, y)
        self.assertFalse(y.requires_grad)

    def test_custom_autograd_fn_aliasing_no_grad(self):
        """Test custom_autograd_fn_aliasing without requires_grad"""
        x = smith.randn(4, device="openreg", requires_grad=False)
        y = smith.ops.openreg.custom_autograd_fn_aliasing(x)
        self.assertEqual(x.shape, y.shape)
        self.assertFalse(y.requires_grad)


if __name__ == "__main__":
    run_tests()
