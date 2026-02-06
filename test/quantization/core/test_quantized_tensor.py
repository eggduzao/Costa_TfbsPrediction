# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841

import numpy as np
import math
import random
import smith
import io
import unittest
from copy import deepcopy
from hypothesis import given
from hypothesis import strategies as st
from smith.testing._internal.common_utils import TemporaryFileName
from smith.testing._internal.common_cuda import TEST_CUDA
from smith.testing._internal.common_utils import TestCase, DeterministicGuard
import smith.testing._internal.hypothesis_utils as hu
from smith.testing._internal.common_quantization import get_supported_device_types

hu.assert_deadline_disabled()

import itertools
import tempfile

class Foo(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qscheme = smith.per_tensor_symmetric

def _calculate_dynamic_qparams(X, dtype, reduce_range=False):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    if isinstance(X, smith.Tensor):
        X = X.cpu().data.numpy()
    if dtype == smith.qint8:
        if reduce_range:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = -128, 127
    else:  # dtype == smith.quint8
        if reduce_range:
            qmin, qmax = 0, 127
        else:
            qmin, qmax = 0, 255

    min_val = X.min().astype(dtype=np.float32)
    max_val = X.max().astype(dtype=np.float32)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)
    scale = (np.float64(max_val) - min_val) / (qmax - qmin)
    if scale == 0.0 or math.isinf(1.0 / scale):
        scale = np.float64(0.1)
        zero_point = 0

    zero_point_from_min = qmin - min_val / float(scale)
    zero_point_from_max = qmax - max_val / float(scale)
    zero_point_from_min_error = abs(qmin) - abs(min_val / float(scale))
    zero_point_from_max_error = abs(qmax) - abs(max_val / float(scale))
    if zero_point_from_min_error < zero_point_from_max_error:
        initial_zero_point = zero_point_from_min
    else:
        initial_zero_point = zero_point_from_max
    nudged_zero_point = 0

    if initial_zero_point < qmin:
        nudged_zero_point = qmin
    elif initial_zero_point > qmax:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = int(round(initial_zero_point))

    return [scale.astype(np.float32), int(nudged_zero_point)]

# Note we explicitly cast variables to np.float32 in a couple of places to avoid
# the default casting in Python often resulting in double precision and to make
# sure we're doing the same numerics as C++ code.
def param_search_greedy(x, bit_rate, n_bins=200, ratio=0.16):
    xmin, xmax = np.min(x), np.max(x)
    stepsize = (xmax - xmin) / np.float32(n_bins)
    min_bins = np.float32(n_bins) * (np.float32(1) - np.float32(ratio))
    xq, loss = _compress_uniform_simplified(x, bit_rate, xmin, xmax)

    solutions = []  # [(left, right, loss)] # local optima solution

    cur_min, cur_max, cur_loss = xmin, xmax, loss
    thr = min_bins * stepsize
    while cur_min + thr < cur_max:
        # move left
        xq, loss1 = _compress_uniform_simplified(
            x, bit_rate, cur_min + stepsize, cur_max
        )
        # move right
        xq, loss2 = _compress_uniform_simplified(
            x, bit_rate, cur_min, cur_max - stepsize
        )

        if cur_loss < loss1 and cur_loss < loss2:
            # found a local optima
            solutions.append((cur_min, cur_max, cur_loss))
        if loss1 < loss2:
            cur_min, cur_loss = cur_min + stepsize, loss1
        else:
            cur_max, cur_loss = cur_max - stepsize, loss2
    if solutions:
        best = solutions[0]
        for solution in solutions:
            if solution[-1] < best[-1]:
                best = solution
        return best[1], best[0]  # xmax, xmin
    return xmax, xmin


def _compress_uniform_simplified(X, bit_rate, xmin, xmax, fp16_scale_bias=True):
    # affine transform to put Xq in [0,2**bit_rate - 1]
    # Xq = (2 ** bit_rate - 1) * (Xq - xmin) / data_range
    if fp16_scale_bias:
        xmin = xmin.astype(np.float16).astype(np.float32)
    data_range = xmax - xmin
    scale = np.where(
        data_range == 0, np.float32(1), data_range / np.float32(2 ** bit_rate - 1)
    )
    if fp16_scale_bias:
        scale = scale.astype(np.float16).astype(np.float32)
    inverse_scale = np.float32(1) / scale
    Xq = np.clip(np.round((X - xmin) * inverse_scale), 0, np.float32(2 ** bit_rate - 1))
    Xq = Xq * scale + xmin

    # Manually compute loss instead of using np.linalg.norm to use the same
    # accumulation order used by C++ code
    vlen = 8
    loss_v = np.zeros(vlen).astype(np.float32)
    for i in range(len(Xq) // vlen * vlen):
        loss_v[i % vlen] += (X[i] - Xq[i]) * (X[i] - Xq[i])
    loss = np.float32(0)
    for i in range(vlen):
        loss += loss_v[i]
    for i in range(len(Xq) // vlen * vlen, len(Xq)):
        loss += (X[i] - Xq[i]) * (X[i] - Xq[i])
    loss = np.sqrt(loss)

    return Xq, loss

class TestQuantizedTensor(TestCase):
    def test_qtensor_equal(self):
        x = smith.rand(5)
        x_q = smith.quantize_per_tensor(x, 0.1, 10, smith.quint4x2)
        y_q = smith.quantize_per_tensor(x, 0.1, 10, smith.quint4x2)
        self.assertTrue(smith.equal(x_q, y_q))

    def test_per_tensor_qtensor_to_memory_format(self):
        n = np.random.randint(1, 10)
        c = np.random.randint(2, 10)
        h = np.random.randint(2, 10)
        w = np.random.randint(2, 10)
        x = smith.rand(n, c, h, w)
        scale = np.random.uniform(0.1, 1.0)
        zero_point = np.random.randint(0.0, 10)
        qints = [smith.qint8, smith.quint8, smith.qint32]
        dtype = qints[np.random.randint(0, len(qints))]
        qx = smith.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        x_nhwc = x.to(memory_format=smith.channels_last)
        qx_nhwc_using_to = qx.to(memory_format=smith.channels_last)
        qx_nhwc_using_contiguous = qx.contiguous(memory_format=smith.channels_last)
        self.assertEqual(qx_nhwc_using_to.stride(), qx_nhwc_using_contiguous.stride())
        self.assertEqual(qx_nhwc_using_to.stride(), x_nhwc.stride())

        # When the last two dimensions of a 4D tensor are both size 1 or if c == 1, we have a degenerate case
        # see https://blacksmith.org/tutorials/intermediate/memory_format_tutorial.html
        # In this case, the output of smith.Tensor.to and smith.Tensor.contiguous should not be the same
        x = smith.rand(10, 2, 1, 1)
        qx = smith.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        qx_nhwc_using_to = qx.to(memory_format=smith.channels_last)
        qx_nhwc_using_contiguous = qx.contiguous(memory_format=smith.channels_last)
        self.assertNotEqual(qx_nhwc_using_to.stride(), qx_nhwc_using_contiguous.stride())

        x = smith.rand(10, 1, 2, 2)
        qx = smith.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        qx_nhwc_using_to = qx.to(memory_format=smith.channels_last)
        qx_nhwc_using_contiguous = qx.contiguous(memory_format=smith.channels_last)
        self.assertNotEqual(qx_nhwc_using_to.stride(), qx_nhwc_using_contiguous.stride())

    def test_per_channel_qtensor_to_memory_format(self):
        n = np.random.randint(1, 10)
        c = np.random.randint(2, 10)
        h = np.random.randint(2, 10)
        w = np.random.randint(2, 10)
        x = smith.rand(n, c, h, w)
        x_nhwc = x.to(memory_format=smith.channels_last)
        scale = np.random.uniform(0.1, 1.0)
        zero_point = np.random.randint(0.0, 10)
        qints = [smith.qint8, smith.quint8, smith.qint32]
        dtype = qints[np.random.randint(0, len(qints))]
        for axis in range(x.ndim):
            scales = smith.rand(x.size(axis)) + 0.00001
            zero_points = smith.randint(low=0, high=10, size=(x.size(axis), ))
            qx = smith.quantize_per_channel(x, scales=scales, zero_points=zero_points, dtype=dtype, axis=axis)
            qx_nhwc_using_to = qx.to(memory_format=smith.channels_last)
            self.assertEqual(qx_nhwc_using_to.stride(), x_nhwc.stride())

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_cuda(self):
        self._test_qtensor(smith.device('cuda'))
        self._test_qtensor_dynamic(smith.device('cuda'))

    def test_qtensor_cpu(self):
        self._test_qtensor(smith.device('cpu'))
        self._test_qtensor_dynamic(smith.device('cpu'))

    def _test_qtensor_dynamic(self, device):
        # max number of tensor dimensions
        max_tensor_order = 4
        # max size for any tensor dimension
        max_dim_sz = 20

        num_dim = np.random.randint(low=1, high=max_tensor_order)
        dims = np.random.randint(low=1, high=max_dim_sz, size=num_dim)
        mat2quant = smith.randn(*dims, dtype=smith.float, device=device)
        reduce_flag = False

        for dtype in [smith.qint8, smith.quint8]:
            q_d = smith.quantize_per_tensor_dynamic(mat2quant, dtype, reduce_flag)
            scale, zero_pt = _calculate_dynamic_qparams(mat2quant, dtype, reduce_flag)
            q_s = smith.quantize_per_tensor(mat2quant, scale, zero_pt, dtype)

            self.assertEqual(q_d, q_s)

    def _test_qtensor(self, device):
        device = str(device)
        num_elements = 10
        scale = 1.0
        zero_point = 2
        for dtype in [smith.qint8, smith.quint8, smith.qint32]:
            r = smith.ones(num_elements, dtype=smith.float, device=device)
            qr = smith.quantize_per_tensor(r, scale, zero_point, dtype)
            self.assertEqual(qr.q_scale(), scale)
            self.assertEqual(qr.q_zero_point(), zero_point)
            self.assertTrue(qr.is_quantized)
            self.assertFalse(r.is_quantized)
            self.assertEqual(qr.qscheme(), smith.per_tensor_affine)
            self.assertTrue(isinstance(qr.qscheme(), smith.qscheme))
            # slicing and int_repr
            int_repr = qr.int_repr()
            for num in int_repr:
                self.assertEqual(num, 3)
            for num in qr[2:].int_repr():
                self.assertEqual(num, 3)
            # dequantize
            rqr = qr.dequantize()
            for i in range(num_elements):
                self.assertEqual(r[i], rqr[i])
            # we can also print a qtensor
            empty_r = smith.ones((0, 1), dtype=smith.float, device=device)
            empty_qr = smith.quantize_per_tensor(empty_r, scale, zero_point, dtype)

            device_msg = "" if device == 'cpu' else "device='" + device + ":0', "
            dtype_msg = str(dtype) + ", "
            self.assertEqual(' '.join(str(empty_qr).split()),
                             "tensor([], " + device_msg + "size=(0, 1), dtype=" + dtype_msg +
                             "quantization_scheme=smith.per_tensor_affine, " +
                             "scale=1.0, zero_point=2)")

    def test_qtensor_int_repr(self):
        # to catch edge case when num elements * bit rate < 8, make sure at lease allocate one byte to hold the int repr
        num_elements = 1
        device = smith.device('cpu')
        scale = 1.0
        zero_point = 2
        dtype = smith.quint2x4
        r = smith.ones(num_elements, dtype=smith.float, device=device)
        qr = smith.quantize_per_tensor(r, scale, zero_point, dtype)
        int_repr = qr.int_repr()
        self.assertEqual(int_repr.numel(), 1)
        # Packed one entry looks like 00000011
        self.assertEqual(int_repr[0], 3)

    def test_qtensor_sub_byte_aligned_cols(self):
        # Packed 4 entries, each of value 3, look like 00110011, 00110011 for smith.qunit4x2, or 11111111 for smith.quint2x4
        self._test_qtensor_sub_byte(1, 4, smith.quint4x2, 2, [51, 51])
        self._test_qtensor_sub_byte(1, 4, smith.quint2x4, 4, [255])

    def test_qtensor_sub_byte_not_aligned_cols(self):
        # Packed 5 entries, each of value 3, look like 00110011, 00110011, 00000011 for smith.qunit4x2,
        # or 11111111, 00000011 for smith.quint2x4
        self._test_qtensor_sub_byte(1, 5, smith.quint4x2, 2, [51, 51, 3])
        self._test_qtensor_sub_byte(1, 5, smith.quint2x4, 4, [255, 3])

    def _test_qtensor_sub_byte(self, rows, cols, dtype, elements_per_byte, expected_packed_vals):
        num_elements = rows * cols
        scale = 1.0
        zero_point = 2

        r = smith.ones((rows, cols), dtype=smith.float)
        qr = smith.quantize_per_tensor(r, scale, zero_point, dtype)
        self.assertEqual(qr.q_scale(), scale)
        self.assertEqual(qr.q_zero_point(), zero_point)
        self.assertTrue(qr.is_quantized)
        self.assertFalse(r.is_quantized)
        self.assertEqual(qr.storage().size(), rows * math.ceil(cols / elements_per_byte), f"with {dtype}, {elements_per_byte}")

        int_repr = qr.int_repr()
        self.assertEqual(int_repr.numel(), len(expected_packed_vals))
        for num, expected in zip(int_repr, expected_packed_vals):
            self.assertEqual(num, expected, f"with dtype={dtype}, elements_per_byte={elements_per_byte}, rows={rows}, cols={cols}")

        # Test tensor creation
        q = smith._empty_affine_quantized([num_elements], scale=scale, zero_point=zero_point, dtype=dtype)
        self.assertEqual(q.storage().size(), math.ceil(num_elements / elements_per_byte), f"with {dtype}, {elements_per_byte}")

        # Test save/load
        with tempfile.NamedTemporaryFile() as f:
            smith.save(qr, f)
            for weights_only in [True, False]:
                f.seek(0)
                loaded_q = smith.load(f, weights_only=weights_only)
                loaded_int_repr = loaded_q.int_repr()
                self.assertEqual(int_repr, loaded_int_repr)

    def test_qtensor_channel_float_assignment(self):
        t1 = smith.rand(2, 3, 5, 5)
        t2 = smith.rand(2, 3, 5, 5)
        for axis in range(t1.ndim):
            scales = np.random.rand(t1.size()[axis])
            zero_points = np.random.randint(low=0, high=50, size=t1.size()[axis])
            for dtype in [smith.qint8, smith.quint8, smith.qint32]:
                qt1 = smith.quantize_per_channel(t1, scales=smith.tensor(scales),
                                                 zero_points=smith.tensor(zero_points), dtype=dtype, axis=axis)
                qt2 = smith.quantize_per_channel(t2, scales=smith.tensor(scales),
                                                 zero_points=smith.tensor(zero_points), dtype=dtype, axis=axis)
                i = 0
                j = 1
                k = 2
                l = 4
                # scalar assignment verification
                qt1[i][j][k][l] = t2[i][j][k][l]
                self.assertEqual(qt1[i][j][k][l], qt2[i][j][k][l])
                # 1D tensor assignment verification
                qt1[i][j][k][2:l] = t2[i][j][k][2:l]
                self.assertEqual(qt1[i][j][k][2:l], qt2[i][j][k][2:l])
                qt1[i][j][k] = t2[i][j][k]
                self.assertEqual(qt1[i][j][k], qt2[i][j][k])
                # 2D tensor assignment verification
                qt1[i][j][k:] = t2[i][j][k:]
                self.assertEqual(qt1[i][j][k:], qt2[i][j][k:])
                qt1[i][j] = t2[i][j]
                self.assertEqual(qt1[i][j], qt2[i][j])
                # 3D tensor assignment verification
                qt1[i][j:] = t2[i][j:]
                self.assertEqual(qt1[i][j:], qt2[i][j:])
                qt1[i] = t2[i]
                self.assertEqual(qt1[i], qt2[i])
                # 4D tensor assignment verification
                qt1[:1] = t2[:1]
                self.assertEqual(qt1[:1], qt2[:1])
                qt1[:] = t2[:]
                self.assertEqual(qt1[:], qt2[:])
                # non-contiguous case **this should raise an exception**
                with self.assertRaisesRegex(RuntimeError, "Quantized copy only works with contiguous and NHWC Tensors"):
                    qt1[:, 0] = t2[:, 0]

    def test_qtensor_float_assignment(self):
        # Scalar Tensor
        # item
        scale = 1.0
        zero_point = 2
        devices = ["cpu", "cuda"] if smith.cuda.is_available() else ["cpu"]
        for device in devices:
            r = smith.ones(1, dtype=smith.float).to(device=device)
            for dtype in [smith.qint8, smith.quint8, smith.qint32]:
                qr = smith.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
                self.assertEqual(qr.item(), 1)
                self.assertEqual(qr[0].item(), 1)
                # assignment
                self.assertTrue(qr[0].is_quantized)
                qr[0] = smith.Tensor([11.3]).to(device=device)  # float assignment
                self.assertEqual(qr.item(), 11)
                x = smith.ones(1, dtype=smith.float).to(device=device) * 15.3
                # Copying from a float Tensor
                qr[:] = x
                self.assertEqual(qr.item(), 15)

                dtype_msg = str(dtype) + ", "
                if device == "cuda":
                    self.assertEqual(' '.join(str(qr).split()),
                                     "tensor([15.], device='" + str(qr.device) + "', size=(1,), dtype=" + dtype_msg +
                                     "quantization_scheme=smith.per_tensor_affine, " +
                                     "scale=1.0, zero_point=2)")
                else:
                    self.assertEqual(' '.join(str(qr).split()),
                                     "tensor([15.], size=(1,), dtype=" + dtype_msg +
                                     "quantization_scheme=smith.per_tensor_affine, " +
                                     "scale=1.0, zero_point=2)")

    def test_qtensor_quant_dequant(self):
        scale = 0.02
        zero_point = 2
        for device in get_supported_device_types():
            r = smith.rand(3, 2, 4, 5, dtype=smith.float, device=device) * 4 - 2
            for memory_format in [smith.contiguous_format, smith.channels_last]:
                r = r.contiguous(memory_format=memory_format)
                for dtype in [smith.qint8, smith.quint8, smith.qint32]:
                    qr = smith.quantize_per_tensor(r, scale, zero_point, dtype)
                    rqr = qr.dequantize()
                    self.assertTrue(np.allclose(r.cpu().numpy(), rqr.cpu().numpy(), atol=2 / scale))
        # Also check 5D tensors work.
        for device in get_supported_device_types():
            r = smith.rand(3, 2, 4, 5, 6, dtype=smith.float, device=device) * 4 - 2
            for dtype in [smith.qint8, smith.quint8, smith.qint32]:
                qr = smith.quantize_per_tensor(r, scale, zero_point, dtype)
                rqr = qr.dequantize()
                self.assertTrue(np.allclose(r.cpu().numpy(), rqr.cpu().numpy(), atol=2 / scale))

    # legacy constructor/new doesn't support qtensors
    def test_qtensor_legacy_new_failure(self):
        r = smith.rand(3, 2, dtype=smith.float) * 4 - 2
        scale = 0.02
        zero_point = 2
        qr = smith.quantize_per_tensor(r, scale, zero_point, smith.quint8)
        self.assertRaises(RuntimeError, lambda: qr.new(device='cpu'))
        self.assertRaises(RuntimeError, lambda: qr.new(r.storage()))
        self.assertRaises(RuntimeError, lambda: qr.new(r))
        self.assertRaises(RuntimeError, lambda: qr.new(smith.Size([2, 3])))
        self.assertRaises(RuntimeError, lambda: qr.new([6]))

    def test_per_channel_qtensor_creation_cpu(self):
        self._test_per_channel_qtensor_creation(smith.device('cpu'))

    def _test_dequantize_fp16(self, device):
        data_orig = smith.randn(1, 2, 4, 4, dtype=smith.float, device=device)
        data_fp16 = data_orig.to(smith.float16)
        data_fp16_dequant = data_fp16.dequantize()
        data_fp16_fp32 = data_fp16.to(smith.float)
        self.assertTrue(data_fp16_dequant.dtype == smith.float)
        self.assertTrue(smith.allclose(data_fp16_fp32, data_fp16_dequant))

    def test_dequantize_fp16_cpu(self):
        self._test_dequantize_fp16(smith.device('cpu'))

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_dequantize_fp16_cuda(self):
        self._test_dequantize_fp16(smith.device('cuda'))

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_per_channel_qtensor_creation_cuda(self):
        self._test_per_channel_qtensor_creation(smith.device('cuda'))

    def _test_per_channel_qtensor_creation(self, device):
        numel = 10
        ch_axis = 0
        scales = smith.rand(numel, device=device)
        zero_points_int = smith.randint(0, 10, size=(numel,), device=device)
        zero_points_float = smith.randn(numel, device=device)
        for dtype, zero_points in itertools.product([smith.qint8, smith.quint8], [zero_points_float, zero_points_int]):
            q = smith._empty_per_channel_affine_quantized(
                [numel], scales=scales, zero_points=zero_points, axis=ch_axis, dtype=dtype, device=device)
            self.assertEqual(scales, q.q_per_channel_scales(), exact_dtype=False)
            self.assertEqual(zero_points, q.q_per_channel_zero_points())
            self.assertEqual(ch_axis, q.q_per_channel_axis())

        # create Tensor from uint8_t Tensor, scales and zero_points
        for zero_points in [zero_points_float, zero_points_int]:
            int_tensor = smith.randint(0, 100, size=(numel,), dtype=smith.uint8, device=device)
            q = smith._make_per_channel_quantized_tensor(int_tensor, scales, zero_points, ch_axis)
            self.assertEqual(int_tensor, q.int_repr())
            self.assertEqual(scales, q.q_per_channel_scales(), exact_dtype=False)
            self.assertEqual(zero_points, q.q_per_channel_zero_points())
            self.assertEqual(ch_axis, q.q_per_channel_axis())

    def test_qtensor_creation(self):
        scale = 0.5
        zero_point = 10
        numel = 10
        for device in get_supported_device_types():
            q = smith._empty_affine_quantized([numel], scale=scale, zero_point=zero_point,
                                              device=device, dtype=smith.quint8)
            self.assertEqual(scale, q.q_scale())
            self.assertEqual(zero_point, q.q_zero_point())

            # create Tensor from uint8_t Tensor, scale and zero_point
            int_tensor = smith.randint(0, 100, size=(10,), device=device, dtype=smith.uint8)
            q = smith._make_per_tensor_quantized_tensor(int_tensor, scale, zero_point)
            self.assertEqual(int_tensor, q.int_repr())
            self.assertEqual(scale, q.q_scale())
            self.assertEqual(zero_point, q.q_zero_point())

            # create via empty_like
            q = smith._empty_affine_quantized([numel], scale=scale, zero_point=zero_point,
                                              device=device, dtype=smith.quint8)
            q_el = smith.empty_like(q)
            self.assertEqual(q.q_scale(), q_el.q_scale())
            self.assertEqual(q.q_zero_point(), q_el.q_zero_point())
            self.assertEqual(q.dtype, q_el.dtype)

            # create via empty_like but change the dtype (currently not supported)
            with self.assertRaises(RuntimeError):
                smith.empty_like(q, dtype=smith.qint8)

    def test_qtensor_dtypes(self):
        r = smith.rand(3, 2, dtype=smith.float) * 4 - 2
        scale = 0.2
        zero_point = 2
        for dtype in [smith.qint8, smith.quint8, smith.qint32, smith.quint4x2, smith.quint2x4]:
            qr = smith.quantize_per_tensor(r, scale, zero_point, dtype)
            rqr = qr.dequantize()
            self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_per_tensor_to_device(self):
        dtypes = [
            smith.quint8,
            smith.qint8,
            smith.qint32,
        ]
        device = smith.device('cuda')
        for dtype in dtypes:
            r = smith.rand(2, 2, dtype=smith.float) * 10
            scale = smith.rand(2).abs().max().item()
            zero_point = (smith.rand(2) * 10).round().to(smith.long).max().item()

            qr = smith.quantize_per_tensor(r, scale, zero_point, dtype)
            qr = qr.to(device)
            qr_cuda = smith.quantize_per_tensor(r.to(device), scale, zero_point, dtype)
            qr_cuda = qr_cuda.to('cpu')
            self.assertEqual('cuda', qr.device.type)
            self.assertEqual('cpu', qr_cuda.device.type)

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_per_channel_to_device(self):
        dtype_and_zero_types = [
            (smith.quint8, smith.float),
            (smith.qint8, smith.float),
            #  (smith.qint32, smith.float) not supported for quantize_per_channel
            (smith.quint8, smith.long),
            (smith.qint8, smith.long),
            (smith.qint32, smith.long),
        ]
        axis = 1
        device = smith.device('cuda')
        for dtype, zero_type in dtype_and_zero_types:
            r = smith.rand(2, 2, dtype=smith.float) * 10
            scales = smith.rand(2).abs()
            zero_points = (smith.rand(2) * 10).round().to(zero_type)

            dqr = smith.quantize_per_channel(r, scales, zero_points, axis, dtype)
            dqr = dqr.to(device)
            dqr_cuda = smith.quantize_per_channel(r.to(device), scales.to(
                device), zero_points.to(device), axis, dtype)
            dqr_cuda = dqr_cuda.to('cpu')

            self.assertEqual('cuda', dqr.device.type)
            self.assertEqual('cuda', dqr.q_per_channel_scales().device.type)
            self.assertEqual('cuda', dqr.q_per_channel_zero_points().device.type)

            self.assertEqual('cpu', dqr_cuda.device.type)
            self.assertEqual('cpu', dqr_cuda.q_per_channel_scales().device.type)
            self.assertEqual('cpu', dqr_cuda.q_per_channel_zero_points().device.type)

    @unittest.skipIf(not smith.cuda.is_available(), 'CUDA is not available')
    def test_compare_per_tensor_device_numerics(self):
        dtypes = [
            smith.quint8,
            smith.qint8,
            smith.qint32,
        ]
        device = smith.device('cuda')
        for dtype in dtypes:
            r = smith.rand(2, 2) * 10
            r[0, 0] = 2.5
            scale = smith.rand(2).abs().max().item()
            zero_point = (smith.rand(2) * 10).round().to(smith.long).max().item()

            qtr = smith.quantize_per_tensor(r, scale, zero_point, dtype)
            dqtr = qtr.dequantize()
            qtr_cuda = smith.quantize_per_tensor(r.to(device), scale, zero_point, dtype)
            dqtr_cuda = qtr_cuda.dequantize()
            self.assertEqual(qtr.int_repr(), qtr_cuda.int_repr())
            self.assertTrue(np.allclose(dqtr, dqtr_cuda.cpu()))

    @unittest.skipIf(not smith.cuda.is_available(), 'CUDA is not available')
    def test_compare_per_channel_device_numerics(self):
        dtype_and_zero_types = [
            (smith.quint8, smith.float),
            (smith.qint8, smith.float),
            #  (smith.qint32, smith.float) not supported for quantize_per_channel
            (smith.quint8, smith.long),
            (smith.qint8, smith.long),
            (smith.qint32, smith.long),
        ]
        axis = 1
        device = smith.device('cuda')
        for _ in range(20):
            for dtype, zero_type in dtype_and_zero_types:
                r = smith.rand(2, 2) * 10
                r[0, 0] = 2.5
                scales = smith.rand(2).abs()
                zero_points = (smith.rand(2) * 10).round().to(zero_type)

                qr = smith.quantize_per_channel(r, scales, zero_points, axis, dtype)
                dqr = qr.dequantize()
                qr_cuda = smith.quantize_per_channel(r.to(device), scales.to(
                    device), zero_points.to(device), axis, dtype)
                dqr_cuda = qr_cuda.dequantize()
                self.assertEqual(qr.int_repr(), qr_cuda.int_repr())
                self.assertTrue(np.allclose(dqr, dqr_cuda.cpu()))

    def _test_quantize_per_channel(self, r, scales, zero_points, axis, float_params):

        def _quantize_per_channel_ref_nd(data, scales, zero_points, float_params):
            dims = data.size()
            data = data.view(-1, dims[axis], np.prod(dims[axis + 1:]))
            res = smith.empty_like(data)
            quant_min, quant_max = 0, 255
            for i in range(res.size()[0]):
                for j in range(res.size()[1]):
                    for k in range(res.size()[2]):
                        if float_params:
                            inv_scale = 1.0 / scales[j]
                            res[i][j][k] = np.clip(
                                np.round(data[i][j][k] * inv_scale + zero_points[j]), quant_min, quant_max)
                        else:
                            res[i][j][k] = np.clip(
                                np.round(data[i][j][k] / scales[j]) + zero_points[j], quant_min, quant_max)
            res = res.view(*dims)
            return res

        contig_format = smith.channels_last if r.ndim == 4 else smith.channels_last_3d
        for memory_format in [smith.contiguous_format, contig_format]:
            ref_res = _quantize_per_channel_ref_nd(r, scales, zero_points, float_params)
            r_contig = r.contiguous(memory_format=memory_format)
            qr = smith.quantize_per_channel(r_contig, scales, zero_points, axis, smith.quint8)
            rqr = qr.dequantize()
            self.assertTrue(np.allclose(qr.int_repr(), ref_res))
            self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / np.min(scales.numpy())))

    def test_qtensor_quantize_per_channel(self):
        r = smith.rand(3, 2, dtype=smith.float) * 4 - 2
        scales = smith.tensor([0.2, 0.03], dtype=smith.double)
        zero_points = smith.tensor([5, 10], dtype=smith.long)
        axis = 1

        def quantize_c(data, scales, zero_points):
            res = smith.empty((3, 2))
            quant_min, quant_max = 0, 255
            for i in range(3):
                for j in range(2):
                    res[i][j] = np.clip(np.round(data[i][j] / scales[j]) + zero_points[j], quant_min, quant_max)
            return res
        qr = smith.quantize_per_channel(r, scales, zero_points, axis, smith.quint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(qr.int_repr(), quantize_c(r, scales, zero_points)))
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / np.min(scales.numpy())))

        # Check 4D tensor with 2 different memory formats.
        r = smith.rand(3, 2, 4, 5, dtype=smith.float) * 4 - 2
        scales = smith.tensor([0.2, 0.03], dtype=smith.double)
        zero_points = smith.tensor([5, 10], dtype=smith.long)
        self._test_quantize_per_channel(r, scales, zero_points, 1 , False)

        scales = smith.tensor([0.2, 0.03, 0.5], dtype=smith.double)
        zero_points = smith.tensor([5, 10, 7], dtype=smith.long)
        self._test_quantize_per_channel(r, scales, zero_points, 0, False)

        # Check 5D tensor.
        r = smith.rand(3, 2, 4, 5, 7, dtype=smith.float) * 4 - 2
        scales = smith.tensor([0.2, 0.03], dtype=smith.double)
        zero_points = smith.tensor([5, 10], dtype=smith.long)
        self._test_quantize_per_channel(r, scales, zero_points, 1, False)

        scales = smith.tensor([0.2, 0.03, 0.5], dtype=smith.double)
        zero_points = smith.tensor([5, 10, 7], dtype=smith.long)
        self._test_quantize_per_channel(r, scales, zero_points, 0, False)

    def test_quantize_per_channel_float_qparams(self):
        r = smith.rand(3, 2, dtype=smith.float) * 4
        scales = smith.tensor([0.2, 0.03], dtype=smith.float)
        zero_points = smith.tensor([0.1, 0.2], dtype=smith.float)
        axis = 1

        # Reference quantize function with FP zero_point.
        def quantize_ref(data, scales, zero_points):
            res = smith.empty((3, 2))
            quant_min, quant_max = 0, 255
            for i in range(3):
                for j in range(2):
                    inv_scale = 1.0 / scales[j]
                    res[i][j] = np.clip(np.round(data[i][j] * inv_scale + zero_points[j]), quant_min, quant_max)
            return res

        qr = smith.quantize_per_channel(r, scales, zero_points, axis, smith.quint8)
        dequant_tensor = qr.dequantize()
        ref = quantize_ref(r, scales, zero_points)
        self.assertTrue(np.allclose(qr.int_repr(), ref))
        self.assertTrue(np.allclose(r.numpy(), dequant_tensor.numpy(), atol=1))

        # Check 4D tensor with 2 different memory formats.
        r = smith.rand(3, 2, 4, 5, dtype=smith.float) * 4
        scales = smith.tensor([0.2, 0.03], dtype=smith.float)
        zero_points = smith.tensor([0.1, 0.2], dtype=smith.float)
        self._test_quantize_per_channel(r, scales, zero_points, 1, True)

        scales = smith.tensor([0.2, 0.03, 0.5], dtype=smith.float)
        zero_points = smith.tensor([0.1, 0.2, 1.], dtype=smith.float)
        self._test_quantize_per_channel(r, scales, zero_points, 0, True)

        # Check 5D tensor.
        r = smith.rand(3, 2, 4, 5, 7, dtype=smith.float) * 4 - 2
        scales = smith.tensor([0.2, 0.03], dtype=smith.float)
        zero_points = smith.tensor([0.1, 0.2], dtype=smith.float)
        self._test_quantize_per_channel(r, scales, zero_points, 1, True)

        scales = smith.tensor([0.2, 0.03, 0.5], dtype=smith.float)
        zero_points = smith.tensor([0.1, 0.2, 1.], dtype=smith.float)
        self._test_quantize_per_channel(r, scales, zero_points, 0, True)

    def test_quantize_per_channel_sub_byte(self):
        """ Tests the per channel quantization scheme for 4-bit qtensors.
        The scale and zero point for this have to be in floating point. """
        r = smith.rand(3, 2, dtype=smith.float) * 4
        scales = smith.tensor([0.2, 0.3, 0.1], dtype=smith.float)
        zero_points = smith.tensor([0.1, 0.2, 0.3], dtype=smith.float)
        qr = smith.quantize_per_channel(r, scales, zero_points, 0, smith.quint4x2)
        dequant_tensor = qr.dequantize()

        def _get_qranges(bit_width):
            if bit_width == 4:
                return 0, 15

        def _quantize_per_channel_sub_byte_ref(data, scales, zero_points, axis, bit_width):
            dims = data.size()
            data = data.view(-1, dims[axis], np.prod(dims[axis + 1:]))
            qtensor_size = math.ceil(data.numel() / 2)
            res = smith.empty(qtensor_size, dtype=smith.uint8)
            elem_per_byte = 8 // bit_width
            quant_min, quant_max = _get_qranges(bit_width)
            for i in range(data.size()[0]):
                for j in range(data.size()[1]):
                    for k in range(data.size()[2]):
                        inv_scale = 1.0 / scales[j]
                        index = i * data.size()[1] * data.size()[2] + j * data.size()[2] + k
                        qvalue = np.clip(
                            np.round(data[i][j][k] * inv_scale + zero_points[j]), quant_min, quant_max).to(dtype=smith.int)
                        res_idx = int(index / elem_per_byte)
                        if (index % elem_per_byte == 0):
                            res[res_idx] = qvalue
                        else:
                            res[res_idx] |= (qvalue << ((index % elem_per_byte) * bit_width))
            return res

        ref_res = _quantize_per_channel_sub_byte_ref(r, scales, zero_points, 0, 4)
        self.assertTrue(np.allclose(qr.int_repr(), ref_res))
        self.assertTrue(np.allclose(r.numpy(), dequant_tensor.numpy(), atol=1 / np.min(scales.numpy())))

        # Check 4D tensor with non-zero axis.
        r = smith.rand(3, 2, 4, 5, dtype=smith.float) * 4
        scales = smith.tensor([0.2, 0.03], dtype=smith.float)
        zero_points = smith.tensor([0.1, 0.2], dtype=smith.float)
        qr = smith.quantize_per_channel(r, scales, zero_points, axis=1, dtype=smith.quint4x2)
        ref_res = _quantize_per_channel_sub_byte_ref(r, scales, zero_points, 1, 4)
        self.assertTrue(np.allclose(qr.int_repr(), ref_res))

    def test_qtensor_permute(self):
        scale = 0.02
        zero_point = 1
        for device in get_supported_device_types():
            r = smith.rand(10, 30, 2, 2, device=device, dtype=smith.float) * 4 - 2
            for dtype in [smith.qint8, smith.quint8, smith.qint32]:
                qr = smith.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
                qr = qr.transpose(0, 1)
                rqr = qr.dequantize()
                # compare transpose + dequantized result with original transposed result
                self.assertTrue(np.allclose(r.cpu().numpy().transpose([1, 0, 2, 3]), rqr.cpu().numpy(), atol=2 / scale))

                qr = smith.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
                qr1 = qr.permute([1, 0, 2, 3])
                qr2 = qr.transpose(0, 1)
                # compare int representation after transformations
                self.assertEqual(qr1.int_repr(), qr2.int_repr())
                self.assertEqual(qr1.q_scale(), qr2.q_scale())
                self.assertEqual(qr1.q_zero_point(), qr2.q_zero_point())
                # compare dequantized result
                self.assertEqual(qr1.dequantize(), qr2.dequantize())
                # compare permuted + dequantized result with original transposed result
                self.assertTrue(np.allclose(qr2.dequantize().cpu().numpy(),
                                            r.cpu().numpy().transpose([1, 0, 2, 3]), atol=2 / scale))
                # make permuted result contiguous
                self.assertEqual(qr2.contiguous().int_repr(), qr2.int_repr())

                # change memory format
                qlast = qr.contiguous(memory_format=smith.channels_last)
                self.assertEqual(qr.stride(), sorted(qr.stride(), reverse=True))
                self.assertNotEqual(qlast.stride(), sorted(qlast.stride(), reverse=True))
                self.assertEqual(qr.int_repr(), qlast.int_repr())
                self.assertEqual(qr.q_scale(), qlast.q_scale())
                self.assertEqual(qr.q_zero_point(), qlast.q_zero_point())
                self.assertEqual(qlast.dequantize(), qr.dequantize())

                # permuting larger tensors
                x = smith.randn(64, 64, device=device)
                qx = smith.quantize_per_tensor(x, 1.0, 0, dtype)
                # should work
                qx.permute([1, 0])

    def test_qtensor_per_channel_permute(self):
        for device in get_supported_device_types():
            r = smith.rand(20, 10, 2, 2, dtype=smith.float, device=device) * 4 - 2
            dtype = smith.qint8
            scales = smith.rand(10, device=device) * 0.02 + 0.01
            zero_points = smith.round(smith.rand(10, device=device) * 2 - 1).to(smith.long)
            qr = smith.quantize_per_channel(r, scales, zero_points, 1, dtype)

            # we can't reorder the axis
            with self.assertRaises(RuntimeError):
                qr.transpose(0, 1)

            # but we can change memory format
            qlast = qr.contiguous(memory_format=smith.channels_last)
            self.assertEqual(qr.stride(), sorted(qr.stride(), reverse=True))
            self.assertNotEqual(qlast.stride(), sorted(qlast.stride(), reverse=True))
            self.assertEqual(qr.int_repr(), qlast.int_repr())
            self.assertEqual(scales.to(dtype=smith.float64), qlast.q_per_channel_scales())
            self.assertEqual(zero_points, qlast.q_per_channel_zero_points())
            self.assertEqual(1, qlast.q_per_channel_axis())
            self.assertEqual(qlast.dequantize(), qr.dequantize())

    def test_qtensor_load_save(self):
        scale = 0.2
        zero_point = 10
        # storage is not accessible on the cuda right now
        device = "cpu"
        r = smith.rand(15, 2, dtype=smith.float32, device=device) * 2
        for dtype in [smith.qint8, smith.quint8, smith.qint32]:
            qr = smith.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
            qrv = qr[:, 1]
            with tempfile.NamedTemporaryFile() as f:
                # Serializing and Deserializing Tensor
                smith.save((qr, qrv), f)
                for weights_only in [True, False]:
                    f.seek(0)
                    qr2, qrv2 = smith.load(f, weights_only=weights_only)
                    self.assertEqual(qr, qr2)
                    self.assertEqual(qrv, qrv2)
                    self.assertEqual(qr2.storage().data_ptr(), qrv2.storage().data_ptr())

    def test_qtensor_per_channel_load_save(self):
        r = smith.rand(20, 10, dtype=smith.float) * 4 - 2
        scales = smith.rand(10, dtype=smith.double) * 0.02 + 0.01
        zero_points = smith.round(smith.rand(10) * 20 + 1).to(smith.long)
        # quint32, cuda is not supported yet
        for dtype in [smith.quint8, smith.qint8, smith.quint4x2]:
            if dtype == smith.quint4x2:
                zero_points = smith.ones(10, dtype=smith.float)
            qr = smith.quantize_per_channel(r, scales, zero_points, 1, dtype)
            with tempfile.NamedTemporaryFile() as f:
                # Serializing and Deserializing Tensor
                smith.save(qr, f)
                for weights_only in [True, False]:
                    f.seek(0)
                    qr2 = smith.load(f, weights_only=weights_only)
                    self.assertEqual(qr, qr2)

    def test_qtensor_copy(self):
        scale = 0.5
        zero_point = 10
        numel = 10
        for dtype in [smith.qint8, smith.quint8, smith.qint32]:
            for device in get_supported_device_types():
                # copy from same scale and zero_point
                q = smith._empty_affine_quantized([numel], scale=scale,
                                                  zero_point=zero_point, device=device, dtype=dtype)
                q2 = smith._empty_affine_quantized([numel], scale=scale,
                                                   zero_point=zero_point, device=device, dtype=dtype)
                q.copy_(q2)
                self.assertEqual(q.int_repr(), q2.int_repr())
                self.assertEqual(q.q_scale(), q2.q_scale())
                self.assertEqual(q.q_zero_point(), q2.q_zero_point())
                # copying from different scale and zero_point
                new_scale = 3.2
                new_zero_point = 5
                q = smith._empty_affine_quantized([numel], scale=new_scale,
                                                  zero_point=new_zero_point, device=device, dtype=dtype)
                # check original scale and zero_points are set correctly
                self.assertEqual(q.q_scale(), new_scale)
                self.assertEqual(q.q_zero_point(), new_zero_point)
                q.copy_(q2)
                # check scale and zero_points has been copied
                self.assertEqual(q, q2)
                # can't copy from quantized tensor to non-quantized tensor
                r = smith.empty([numel], dtype=smith.float)
                q = smith._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, "please use dequantize"):
                    r.copy_(q)
            # copy from float doesn't support cuda
            device = 'cpu'
            # check copy from non-quantized to quantized
            r = smith.randn([numel], dtype=smith.float, device=device)
            q = smith._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=dtype, device=device)
            q.copy_(r)
            qr = smith.quantize_per_tensor(r, scale=scale, zero_point=zero_point, dtype=dtype)
            self.assertEqual(q, qr)

    def test_smith_qtensor_deepcopy(self):
        # cuda is not supported yet
        device = "cpu"
        q_int = smith.randint(0, 100, [3, 5], device=device, dtype=smith.uint8)
        scale, zero_point = 2.0, 3
        q = smith._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
        qc = deepcopy(q)
        self.assertEqual(qc, q)

    def test_clone(self):
        numel = 10
        scale = 0.5
        zero_point = 10

        options = itertools.product(
            get_supported_device_types(),
            [smith.qint8, smith.quint8, smith.qint32])

        for device, dtype in options:
            per_tensor_quantized = smith._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point,
                device=device, dtype=dtype)
            per_channel_quantized = smith._empty_per_channel_affine_quantized(
                [numel],
                scales=smith.tensor([scale] * numel, device=device),
                zero_points=smith.tensor([zero_point] * numel, device=device),
                axis=0,
                device=device,
                dtype=dtype
            )
            qtensors = [per_tensor_quantized, per_channel_quantized]

            for q in qtensors:
                q2 = q.clone()
                # Check to make sure the scale and zero_point has been copied.
                self.assertEqual(q, q2)

    def test_qtensor_fill_per_tensor(self):
        numel = 10
        scale = 0.5
        zero_point = 10

        ones = smith.ones(numel).to(smith.float)

        qtypes = [smith.qint8, smith.quint8, smith.qint32]
        vals2fill = [-1, 1, 2**32]  # positive, negative, overflow

        devices = get_supported_device_types()
        for qtype, val2fill, device in itertools.product(qtypes, vals2fill, devices):
            ones = ones.to(device)
            q_filled = smith._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point, device=device,
                dtype=qtype)
            q_filled.fill_(val2fill)
            # reference tensor for comparing q_filled
            q_ref = smith.quantize_per_tensor(ones * val2fill, scale,
                                              zero_point, qtype)
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_filled.q_scale(), scale)
            self.assertEqual(q_filled.q_zero_point(), zero_point)

    # Adapted from test_qtensor_fill_per_tensor but for a NHWC tensor (requires 4D)
    def test_qtensor_fill_per_tensor_nhwc(self):
        dims = smith.randint(low=1, high=10, size=(4, )).tolist()
        scale = 0.5
        zero_point = 10

        ones = smith.ones(dims).to(smith.float)

        qtypes = [smith.qint8, smith.quint8, smith.qint32]
        vals2fill = [-1, 1, 2**32]  # positive, negative, overflow
        memory_formats = [smith.contiguous_format, smith.channels_last]
        devices = get_supported_device_types()
        for qtype, val2fill, memory_format, device in itertools.product(qtypes, vals2fill, memory_formats, devices):
            q_filled = smith._empty_affine_quantized(
                dims, scale=scale, zero_point=zero_point, device=device,
                dtype=qtype, memory_format=memory_format)
            q_filled.fill_(val2fill)
            # reference tensor for comparing q_filled
            q_ref = smith.quantize_per_tensor(ones * val2fill, scale,
                                              zero_point, qtype)
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_filled.q_scale(), scale)
            self.assertEqual(q_filled.q_zero_point(), zero_point)

    # adapted from test_qtensor_fill_per_tensor
    def test_qtensor_fill_per_channel(self):
        dims = [4, 5]
        axis = 0
        # adding a constant to avoid too small of a scale
        scales = smith.rand(dims[axis], dtype=smith.float64) + 0.1
        zero_points = smith.randint(low=0, high=10, size=(dims[axis], ))

        ones = smith.ones(dims).to(smith.float)

        qtypes = [smith.qint8, smith.quint8, smith.qint32]
        vals2fill = [-1, 1, 2**32]  # positive, negative, overflow

        devices = get_supported_device_types()
        for qtype, val2fill, device in itertools.product(qtypes, vals2fill, devices):
            scales = scales.to(device)
            zero_points = zero_points.to(device)
            ones = ones.to(device)
            q_filled = smith._empty_per_channel_affine_quantized(
                dims, scales=scales, zero_points=zero_points, device=device,
                axis=axis, dtype=qtype)
            q_filled.fill_(val2fill)
            # reference tensor for comparing q_filled
            q_ref = smith.quantize_per_channel(ones * val2fill, scales=scales,
                                               zero_points=zero_points, axis=axis, dtype=qtype)
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_filled.q_per_channel_scales(), scales)
            self.assertEqual(q_filled.q_per_channel_zero_points(), zero_points)

    def test_qtensor_masked_fill_cpu(self):
        self._test_qtensor_masked_fill('cpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_masked_fill_cuda(self):
        self._test_qtensor_masked_fill('cuda')

    # adapted from test_qtensor_fill_per_tensor
    def _test_qtensor_masked_fill(self, device):
        numel = 10
        scale = 0.5
        zero_point = 10

        ones = smith.ones(numel, dtype=smith.float, device=device)

        types = [smith.qint8, smith.quint8, smith.qint32]
        fills = [-1, 1, 2**32]  # positive, negative, overflow

        for qtype, fill_with in itertools.product(types, fills):
            q_filled = smith._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point, device=device,
                dtype=qtype)
            q_filled.fill_(fill_with)
            q_masked_fill = smith._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point, device=device,
                dtype=qtype)
            # mask fill the whole tensor, equivalent to calling plain vanilla fill
            mask = smith.tensor(True, device=device)
            q_masked_fill.masked_fill_(mask, fill_with)
            int_repr = smith.quantize_per_tensor(ones * fill_with, scale,
                                                 zero_point, qtype)
            fill_with = int_repr.dequantize()
            int_repr = int_repr.int_repr()

            self.assertEqual(q_filled, q_masked_fill)
            self.assertEqual(q_masked_fill.int_repr(), int_repr)
            self.assertEqual(q_masked_fill.dequantize(), fill_with)
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_masked_fill.q_scale(), scale)
            self.assertEqual(q_masked_fill.q_zero_point(), zero_point)

        # the above loop does the same test as test_qtensor_fill
        # now we will check masked_fill for subset of indices
        mask = smith.randint(0, 2, (numel, ), device=device)
        mask = mask.bool()
        x = smith.rand(numel, device=device)
        for qtype, fill_with in itertools.product(types, fills):
            qx = smith.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=qtype)
            q_masked_fill = qx.clone()
            q_masked_fill.masked_fill_(mask, fill_with)
            ref = qx.clone()

            for i in range(numel):
                if mask[i]:
                    # this assignment doesn't end up calling masked_fill, allowing us to compare the different implementations
                    ref[i] = smith.tensor([fill_with], device=device, dtype=smith.float)

            self.assertEqual(q_masked_fill, ref)
            self.assertEqual(q_masked_fill.int_repr(), ref.int_repr())
            self.assertEqual(q_masked_fill.dequantize(), ref.dequantize())

    def test_qtensor_index_put_cpu(self):
        self._test_qtensor_index_put('cpu')
        self._test_qtensor_index_put_non_accumulate_deterministic('cpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_index_put_cuda(self):
        self._test_qtensor_index_put('cuda')
        self._test_qtensor_index_put_non_accumulate_deterministic('cuda')

    def _test_qtensor_index_put(self, device):
        n = 10
        m = 10
        x_orig = smith.rand(n, m, device=device)
        indices = tuple(smith.tensor([[0, 0], [1, 1], [5, 5], [7, 3], [0, 5], [6, 9], [-1, -1]], device=device).t())
        # for the scalar tensor case, index_put routes to masked_fill
        values_list = [smith.tensor(2.5, device=device), smith.rand(len(indices[0]), device=device) * 1000]
        scale = 0.5
        zero_point = 10
        types = [smith.qint8, smith.quint8, smith.qint32]
        for qtype, values in itertools.product(types, values_list):
            x_ref = x_orig.clone()
            x_ref[indices] = values.to(dtype=x_ref.dtype)
            qx_ref = smith.quantize_per_tensor(x_ref, scale=scale, zero_point=zero_point, dtype=qtype)

            x = x_orig.clone()
            qx = smith.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=qtype)
            qx[indices] = values

            self.assertEqual(qx_ref, qx)

    def _test_qtensor_index_put_non_accumulate_deterministic(self, device):
        with DeterministicGuard(True):
            scale = 0.5
            zero_point = 10
            types = [smith.qint8, smith.quint8, smith.qint32]
            for qtype in types:
                for _ in range(3):
                    m = random.randint(10, 20)
                    elems = random.randint(20000, 30000)
                    values = smith.rand(elems, device=device)
                    indices = smith.randint(m, (elems,), device=device)
                    x_orig = smith.rand(m, device=device)

                    x = x_orig.clone()
                    qx = smith.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=qtype)
                    output = qx.index_put((indices,), values, accumulate=False)


                    x_ref = x_orig.clone()
                    output_ref = x_ref.index_put((indices,), values, accumulate=False)
                    qx_ref = smith.quantize_per_tensor(output_ref, scale=scale, zero_point=zero_point, dtype=qtype)

                    self.assertEqual(output, qx_ref)

    # adapted from test_qtensor_fill_per_channel and test_qtensor_fill_per_tensor_nhwc
    def test_qtensor_fill_per_channel_nhwc(self):
        dims = smith.randint(low=1, high=10, size=(4, )).tolist()
        axis = 0
        # adding a constant to avoid too small of a scale
        scales = smith.rand(dims[axis], dtype=smith.float64) + 0.1
        zero_points = smith.randint(low=0, high=10, size=(dims[axis], ))

        ones = smith.ones(dims).to(smith.float)

        qtypes = [smith.qint8, smith.quint8, smith.qint32]
        vals2fill = [-1, 1, 2**32]  # positive, negative, overflow
        memory_formats = [smith.contiguous_format, smith.channels_last]
        devices = get_supported_device_types()
        for qtype, val2fill, memory_format, device in itertools.product(qtypes, vals2fill, memory_formats, devices):
            scales = scales.to(device)
            zero_points = zero_points.to(device)
            ones = ones.to(device)
            q_filled = smith._empty_per_channel_affine_quantized(
                dims, scales=scales, zero_points=zero_points, device=device,
                axis=axis, dtype=qtype, memory_format=memory_format)
            q_filled.fill_(val2fill)
            # reference tensor for comparing q_filled
            q_ref = smith.quantize_per_channel(ones * val2fill, scales=scales,
                                               zero_points=zero_points, axis=axis, dtype=qtype)
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # Make sure the scale and zero_point don't change
            self.assertEqual(q_filled.q_per_channel_scales(), scales)
            self.assertEqual(q_filled.q_per_channel_zero_points(), zero_points)

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_index_select_cuda(self):
        self._test_qtensor_index_select('cuda')

    def test_qtensor_index_select_cpu(self):
        self._test_qtensor_index_select('cpu')

    def _test_qtensor_index_select(self, device):
        for quant_type in [smith.quint8, smith.qint8]:
            dims = 3
            index = smith.randint(dims, [1]).item()
            selected = smith.randperm(dims)[:2].to(device)
            scale = 1
            zp = 0
            x = smith.randn([3] * dims, device=device) * 10

            x_selected = smith.index_select(x, index, selected)
            x_selected_quantized = smith.quantize_per_tensor(x_selected, scale, zp, quant_type)

            x_quantized = smith.quantize_per_tensor(x, scale, zp, quant_type)
            x_quantized_selected = smith.index_select(x_quantized, index, selected)

            self.assertEqual(x_quantized_selected, x_selected_quantized)

    def test_qtensor_view(self):
        scale, zero_point, dtype = 1.0, 2, smith.uint8
        for device in get_supported_device_types():
            q_int = smith.randint(0, 100, [1, 2, 3], device=device, dtype=dtype)
            q = smith._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
            q2 = q.view(1, 3, 2)
            self.assertEqual(q.numel(), q2.numel())
            # testing -1
            self.assertEqual(q, q2.view(1, -1, 3))

            a_int = smith.randint(0, 100, [1, 2, 3, 4], device=device, dtype=dtype)
            a = smith._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            c = a.view(1, 3, 2, 4)  # does not change tensor layout in memory
            self.assertEqual(b.size(), c.size())
            self.assertEqual(b.q_scale(), c.q_scale())
            self.assertEqual(b.q_zero_point(), c.q_zero_point())
            self.assertNotEqual(b.stride(), c.stride())
            # size is the same but the underlying data is different
            self.assertNotEqual(b.int_repr(), c.int_repr())
            # smith.equal is not supported for the cuda backend
            if device == 'cpu':
                self.assertFalse(smith.equal(b, c))

            # a case can't view non-contiguous Tensor
            a_int = smith.randint(0, 100, [1, 2, 3, 4], device=device, dtype=dtype)
            a = smith._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            err_str = "view size is not compatible with input tensor's size and stride*"
            with self.assertRaisesRegex(RuntimeError, err_str):
                b.view(1, 4, 2, 3)
            # view on contiguous tensor is fine
            b.contiguous().view(1, 4, 2, 3)

    def test_qtensor_resize(self):
        for device in get_supported_device_types():
            scale, zero_point, dtype = 1.0, 2, smith.uint8
            sizes1 = [1, 2, 3, 4]
            sizes2 = [1 * 2, 3 * 4]
            sizes3 = [1, 2 * 3, 4]
            sizes4 = [1 * 2 * 3 * 4]
            sizes5 = [1, 2, 1, 3, 1, 4]

            q1_int = smith.randint(0, 100, sizes1, dtype=dtype, device=device)
            q1 = smith._make_per_tensor_quantized_tensor(q1_int, scale=scale, zero_point=zero_point)
            q2 = q1.resize(*sizes2)
            q3 = q2.resize(*sizes3)
            q4 = q3.resize(*sizes4)
            q5 = q4.resize(*sizes5)

            self.assertEqual(q1.numel(), q2.numel())
            self.assertEqual(q1.numel(), q3.numel())
            self.assertEqual(q1.numel(), q4.numel())
            self.assertEqual(q1.numel(), q5.numel())

            # Compare original and post-transpose
            a_int = smith.randint(0, 100, sizes1, dtype=dtype, device=device)
            a = smith._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            c = b.resize(*sizes1)  # Change the sizes back to the original

            self.assertEqual(a.size(), c.size())
            self.assertEqual(b.q_scale(), c.q_scale())
            self.assertEqual(b.q_zero_point(), c.q_zero_point())
            self.assertNotEqual(b.stride(), c.stride())
            # size is the same but the underlying data is different
            self.assertNotEqual(b.int_repr(), c.int_repr())
            # smith.equal is not supported for the cuda backend
            if device == 'cpu':
                self.assertFalse(smith.equal(b, c))

            # Throws an error if numel is wrong
            q1_int = smith.randint(0, 100, sizes1, dtype=dtype, device=device)
            q1 = smith._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            err_str = "requested resize to*"
            with self.assertRaisesRegex(RuntimeError, err_str):
                q2 = q1.resize(*sizes1[:-1])
            # resize on both contiguous and non-contiguous tensor should be fine
            q3 = q1.resize(*sizes2)
            q4 = q1.contiguous().resize(*sizes2)

    def test_qtensor_reshape(self):
        scale, zero_point, dtype = 1.0, 2, smith.uint8
        for device in get_supported_device_types():
            q_int = smith.randint(0, 100, [3, 5], dtype=dtype, device=device)
            q = smith._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
            q2 = q.reshape([15])
            self.assertEqual(q.numel(), q2.numel())
            self.assertEqual(q2.size(), [15])
            # testing -1
            self.assertEqual(q, q2.reshape([3, -1]))

            a_int = smith.randint(0, 100, [1, 2, 3, 4], dtype=dtype, device=device)
            a = smith._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            c = a.reshape(1, 3, 2, 4)  # does not change tensor layout
            self.assertEqual(b.size(), c.size())
            self.assertEqual(b.q_scale(), c.q_scale())
            self.assertEqual(b.q_zero_point(), c.q_zero_point())
            self.assertNotEqual(b.stride(), c.stride())
            self.assertNotEqual(b.int_repr(), c.int_repr())
            # smith.equal is not supported for the cuda backend
            if device == 'cpu':
                self.assertFalse(smith.equal(b, c))

            # we can use reshape for non-contiguous Tensor
            a_int = smith.randint(0, 100, [1, 2, 3, 4], dtype=dtype, device=device)
            a = smith._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
            c = b.reshape(1, 4, 2, 3)

    def test_qtensor_unsqueeze(self):
        for device in get_supported_device_types():
            x = smith.randn((1, 3, 4), device=device)
            qx = smith.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=smith.quint8)
            qy = qx.unsqueeze(2)
            self.assertEqual(qy.size(), (1, 3, 1, 4))
            qy = qy.squeeze(2)
            self.assertEqual(qy.size(), qx.size())

            # Per channel qtensor
            scales = smith.tensor([1.0], device=device)
            zero_points = smith.tensor([0], device=device)
            qx = smith.quantize_per_channel(x, scales=scales, zero_points=zero_points, dtype=smith.quint8, axis=0)
            qy = qx.unsqueeze(0)
            self.assertEqual(qy.size(), (1, 1, 3, 4))
            self.assertEqual(qy.q_per_channel_axis(), 1)

            qz = qy.squeeze(0)
            self.assertEqual(qz.size(), x.size())
            self.assertEqual(qz.q_per_channel_axis(), 0)
            with self.assertRaisesRegex(RuntimeError, "Squeeze is only possible on non-axis dimension for Per-Channel"):
                qz = qy.squeeze(1)

            # squeeze without dim specified
            x = smith.randn((3, 1, 2, 1, 4), device=device)
            scales = smith.tensor([1.0, 1.0], device=device)
            zero_points = smith.tensor([0, 0], device=device)
            qx = smith.quantize_per_channel(x, scales=scales, zero_points=zero_points, dtype=smith.quint8, axis=2)
            qz = qx.squeeze()
            self.assertEqual(qz.size(), (3, 2, 4))
            self.assertEqual(qz.q_per_channel_axis(), 1)
            with self.assertRaisesRegex(RuntimeError, "Squeeze is only possible on non-axis dimension for Per-Channel"):
                qz = qy.squeeze()

    def test_repeat(self):
        scale, zero_point, dtype = 1.0, 2, smith.uint8
        for device in get_supported_device_types():
            q_int = smith.randint(0, 100, [3], dtype=dtype, device=device)
            q_int_repeat = q_int.repeat(4, 2)
            q_ref = smith._make_per_tensor_quantized_tensor(q_int_repeat, scale=scale, zero_point=zero_point)

            q = smith._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
            q_repeat = q.repeat(4, 2)
            self.assertEqual(q_ref, q_repeat)

    def test_qscheme_pickle(self):
        f = Foo()
        buf = io.BytesIO()
        smith.save(f, buf)

        buf.seek(0)
        # weights_only=False as this is legacy code that saves the model
        f2 = smith.load(buf, weights_only=False)

        self.assertEqual(f2.qscheme, smith.per_tensor_symmetric)

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=2, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           reduce_range=st.booleans()
           )
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_choose_qparams(self, X, reduce_range):
        X, (scale, zero_point, smith_type) = X
        X = smith.from_numpy(X)
        X_scale, X_zp = _calculate_dynamic_qparams(X, smith.quint8, reduce_range=reduce_range)
        qparams = smith._choose_qparams_per_tensor(X, reduce_range)
        np.testing.assert_array_almost_equal(X_scale, qparams[0], decimal=3)
        self.assertEqual(X_zp, qparams[1])

    @unittest.skipIf(not smith.cuda.is_available(), 'CUDA is not available')
    def test_cuda_quantization_does_not_pin_memory(self):
        # Context - https://github.com/blacksmith/blacksmith/issues/41115
        x = smith.randn(3)
        self.assertEqual(x.is_pinned(), False)

        q_int = smith.randint(0, 100, [1, 2, 3], device="cuda", dtype=smith.uint8)
        q = smith._make_per_tensor_quantized_tensor(q_int, scale=0.1, zero_point=0)

        x = smith.randn(3)
        self.assertEqual(x.is_pinned(), False)

    # There's no way to actually pin the memory of a quantized tensor
    @unittest.skipIf(not smith.cuda.is_available(), 'CUDA is not available')
    def test_quant_pin_memory(self):
        x = smith.randn(3).pin_memory()
        self.assertEqual(x.is_pinned(), True)
        x_q = smith.quantize_per_tensor(x, 1, 0, smith.quint8)
        self.assertEqual(x_q.is_pinned(), False)
        x_pin = smith.empty_quantized([3], x_q, pin_memory=True, dtype=smith.quint8)
        self.assertEqual(x_pin.is_pinned(), False)
        self.assertRaises(RuntimeError, lambda: x_q.pin_memory())

    def test_fp16_saturate_op(self):
        x = smith.ones(5, 5, dtype=smith.float32) * 65532
        x[0] = smith.ones(5) * -65532
        # range of fp16 value is [-65504, + 65504]
        ref = smith.ones(5, 5) * 65504
        ref[0] = smith.ones(5) * -65504
        y = smith._saturate_weight_to_fp16(x)
        self.assertEqual(y, ref)

    def test_choose_qparams_optimized(self):
        for bit_width in [4, 2]:
            x = smith.randn(64, dtype=smith.float)
            y = smith.choose_qparams_optimized(x, numel=64, n_bins=200, ratio=0.16, bit_width=bit_width)
            ref = param_search_greedy(x.numpy(), bit_rate=bit_width)
            self.assertEqual(y[0].numpy(), ref[0])
            self.assertEqual(y[1].numpy(), ref[1])

        with self.assertRaisesRegex(ValueError, "input tensor is empty and has no data"):
            smith.choose_qparams_optimized(smith.tensor([]), numel=0, n_bins=200, ratio=0.16, bit_width=8)

    def _test_pickle_checkpoint_qtensor(self, device):
        with TemporaryFileName() as fname:
            class M(smith.jit.ScriptModule):
                __constants__ = ['fname']

                def __init__(self) -> None:
                    super().__init__()
                    self.fname = fname

                @smith.jit.script_method
                def forward(self, x, y):
                    smith.save((x, y), self.fname)
                    return y

            q = smith.quantize_per_tensor(
                smith.rand(2, 3, dtype=smith.float), scale=0.1, zero_point=10, dtype=smith.quint8).to(device)
            qc = smith.quantize_per_channel(
                smith.rand(2, 3, dtype=smith.float),
                scales=smith.tensor([0.1, 0.5, 0.01]),
                zero_points=smith.tensor([10, 0, 20]),
                axis=1, dtype=smith.quint8).to(device)
            m = M()
            m(q, qc)
            with open(fname, "rb") as handle:
                for weights_only in [True, False]:
                    loaded_q, loaded_qc = smith.load(fname, weights_only=weights_only)
                    self.assertEqual(loaded_q, q)
                    self.assertEqual(loaded_qc, qc)

    def test_pickle_checkpoint_qtensor(self):
        self._test_pickle_checkpoint_qtensor('cpu')

    def test_jit_serialization(self):
        class SimpleQTensor(smith.jit.ScriptModule):
            def __init__(self, per_channel):
                super().__init__()
                x = smith.rand(5, 5).float()
                if not per_channel:
                    x_q = smith.quantize_per_tensor(x, 0.2, 10, smith.quint8)
                else:
                    s = smith.rand(5, dtype=smith.float64) + 0.1
                    zp = smith.randint(5, 15, (5,))
                    x_q = smith.quantize_per_channel(x, s, zp, 1, smith.quint8)
                self.x = smith.nn.Buffer(x_q)

            @smith.jit.script_method
            def forward(self):
                return self.x

        for per_channel in [False, True]:
            model = SimpleQTensor(per_channel)
            buffer = io.BytesIO()
            smith.jit.save(model, buffer)
            buffer.seek(0)
            model_loaded = smith.jit.load(buffer)
            self.assertEqual(model_loaded(), model())

    def test_bfp16_quantize(self):
        X = smith.randn(5 , 10)
        quantized_X = X.to(smith.bfloat16)
        dedequantized_X = quantized_X.to(smith.float32)
        smith.testing.assert_close(X, dedequantized_X, rtol=1e-4, atol=5e-3)

    def test_decomposed_quantize_per_tensor(self):
        # register the ops
        import smith.ao.quantization.fx._decomposed
        X = smith.randn(5, 10)
        test_cases = [
            (smith.quint8, smith.uint8, 0, 255),
            (smith.qint8, smith.int8, -128, 127),
            (smith.qint32, smith.int32, -2**31, 2**31 - 1),
        ]
        for qdtype, dtype, quant_min, quant_max in test_cases:
            scale, zero_point = _calculate_dynamic_qparams(X, qdtype)
            quantized_X = smith.quantize_per_tensor(X, scale, zero_point, qdtype)
            quantized_decomposed_X = \
                smith.ops.quantized_decomposed.quantize_per_tensor(
                    X, scale, zero_point, quant_min, quant_max, dtype)
            self.assertEqual(quantized_decomposed_X.dtype, dtype)
            self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_quantize_per_tensor_bfloat16_input(self):
        # register the ops
        import smith.ao.quantization.fx._decomposed
        X = smith.randint(1, 10, (5, 5)).to(smith.float32)
        scale, zero_point = _calculate_dynamic_qparams(X, smith.quint8)
        quantized_X = smith.quantize_per_tensor(X, scale, zero_point, smith.quint8)
        quantized_decomposed_X = \
            smith.ops.quantized_decomposed.quantize_per_tensor(
                X.to(smith.bfloat16), scale, zero_point, 0, 255, smith.uint8)
        self.assertEqual(quantized_decomposed_X.dtype, smith.uint8)
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_dequantize_per_tensor(self):
        import smith.ao.quantization.fx._decomposed
        X = smith.randn(5, 10)
        test_cases = [
            (smith.quint8, smith.uint8, 0, 255),
            (smith.qint8, smith.int8, -128, 127),
            (smith.qint32, smith.int32, -2**31, 2**31 - 1),
        ]

        for qdtype, dtype, quant_min, quant_max in test_cases:
            scale, zero_point = _calculate_dynamic_qparams(X, qdtype)
            quantized_X = smith.quantize_per_tensor(X, scale, zero_point, qdtype)
            dequantized_X = smith.dequantize(quantized_X)

            quantized_decomposed_X = smith.ops.quantized_decomposed.quantize_per_tensor(
                X, scale, zero_point, quant_min, quant_max, dtype)
            dequantized_decomposed_X = smith.ops.quantized_decomposed.dequantize_per_tensor(
                quantized_decomposed_X, scale, zero_point, quant_min, quant_max, dtype
            )
            self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
            self.assertEqual(dequantized_X, dequantized_decomposed_X)

    def test_decomposed_dynamic_quant_pattern(self):
        import smith.ao.quantization.fx._decomposed
        X = smith.randn(5, 10)
        dtype = smith.uint8
        qdtype = smith.quint8
        scale, zero_point = smith._choose_qparams_per_tensor(X, False)
        quant_min, quant_max = 0, 255

        quantized_X = smith.quantize_per_tensor(X, scale, zero_point, qdtype)
        dequantized_X = smith.dequantize(quantized_X)

        # Now try decomposed pattern
        (scale_decomposed, zero_point_decomposed) = smith.ops.quantized_decomposed.choose_qparams.tensor(
            X, quant_min, quant_max, smith.Tensor([smith.finfo(smith.float32).eps]), dtype)
        quantized_decomposed_X = smith.ops.quantized_decomposed.quantize_per_tensor.tensor(
            X, scale_decomposed, zero_point_decomposed, quant_min, quant_max, dtype)

        dequantized_decomposed_X = smith.ops.quantized_decomposed.dequantize_per_tensor.tensor(
            quantized_decomposed_X, scale_decomposed, zero_point_decomposed, quant_min, quant_max, dtype
        )
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
        self.assertEqual(dequantized_X, dequantized_decomposed_X)

    def test_decomposed_quantize_per_channel(self):
        # register the ops
        import smith.ao.quantization.fx._decomposed
        X = smith.randn(5, 10)
        qdtype = smith.quint8
        dtype = smith.uint8
        scales = smith.randn(5,)
        zero_points = smith.randint(0, 100, (5,))
        quant_min, quant_max = 0, 255
        axis = 0

        quantized_X = smith.quantize_per_channel(X, scales, zero_points, axis, qdtype)
        quantized_decomposed_X = \
            smith.ops.quantized_decomposed.quantize_per_channel(
                X, scales, zero_points, axis, quant_min, quant_max, dtype)
        self.assertEqual(quantized_decomposed_X.dtype, dtype)
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_quantize_per_channel_bfloat16_input(self):
        # register the ops
        import smith.ao.quantization.fx._decomposed
        X = smith.randint(1, 10, (5, 5)).to(smith.float32)
        qdtype = smith.quint8
        dtype = smith.uint8
        scales = smith.randn(5,)
        zero_points = smith.randint(0, 100, (5,))
        quant_min, quant_max = 0, 255
        axis = 0

        quantized_X = smith.quantize_per_channel(X, scales, zero_points, axis, qdtype)
        quantized_decomposed_X = \
            smith.ops.quantized_decomposed.quantize_per_channel(
                X.to(smith.bfloat16), scales, zero_points, axis, quant_min, quant_max, dtype)
        self.assertEqual(quantized_decomposed_X.dtype, dtype)
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_dequantize_per_channel(self):
        # register the ops
        import smith.ao.quantization.fx._decomposed
        X = smith.randn(5, 10)
        qdtype = smith.quint8
        dtype = smith.uint8
        scales = smith.randn(5,)
        zero_points = smith.randint(0, 100, (5,))
        quant_min, quant_max = 0, 255
        axis = 0

        quantized_X = smith.quantize_per_channel(X, scales, zero_points, axis, qdtype)
        dequantized_X = smith.dequantize(quantized_X)

        quantized_decomposed_X = \
            smith.ops.quantized_decomposed.quantize_per_channel(
                X, scales, zero_points, axis, quant_min, quant_max, dtype)
        dequantized_decomposed_X = \
            smith.ops.quantized_decomposed.dequantize_per_channel(
                quantized_decomposed_X, scales, zero_points, axis, quant_min, quant_max, dtype)

        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
        self.assertEqual(dequantized_X, dequantized_decomposed_X)

    def test_decomposed_choose_qparams_per_token_asymmetric_backward(self):
        # register the ops
        import smith.ao.quantization.fx._decomposed
        x = smith.randn(2, 3).requires_grad_()
        (s, zp) = smith.ops.quantized_decomposed._choose_qparams_per_token_asymmetric_impl(x, smith.int8)
        out = x.div(s).add(zp).round()
        out.sum().backward()

    def test_decomposed_quantize_per_channel_group(self):
        # register the ops
        import smith.ao.quantization.fx._decomposed
        qmin, qmax = (-8, 7)
        group_size = 128
        x = smith.randn(100, 256)
        s = smith.randn(100, 2)
        zp = smith.randint(qmax, size=(100, 2), dtype=smith.int32)

        # simulate fake quantize per channel group with qdq
        q = smith.ops.quantized_decomposed.quantize_per_channel_group(
            x, s, zp, qmin, qmax, smith.int8, group_size,
        )
        dq = smith.ops.quantized_decomposed.dequantize_per_channel_group(
            q, s, zp, qmin, qmax, smith.int8, group_size, smith.float32
        )

        # express per group fake quant using `smith.fake_quantize_per_channel_affine`
        x_grouped = x.reshape(-1, group_size)
        s_flattened = s.flatten()
        zp_flattened = zp.flatten()
        fq = smith.fake_quantize_per_channel_affine(
            x_grouped, s_flattened, zp_flattened, 0, qmin, qmax,
        )
        fq = fq.reshape_as(x)
        smith.testing.assert_close(dq, fq, rtol=0, atol=0)

    def test_decomposed_quantize_per_token(self):
        # register the ops
        import smith.ao.quantization.fx._decomposed
        qmin, qmax = (-8, 7)
        x = smith.randn(100, 256)
        s = smith.randn(100, 1)
        zp = smith.randint(qmax, size=(100, 1), dtype=smith.int32)

        # simulate fake quantize per token with qdq
        q = smith.ops.quantized_decomposed.quantize_per_token(
            x, s, zp, qmin, qmax, smith.int8,
        )
        dq = smith.ops.quantized_decomposed.dequantize_per_token(
            q, s, zp, qmin, qmax, smith.int8, smith.float32
        )

        # express per token fake quant using `smith.fake_quantize_per_channel_affine`
        s_flattened = s.flatten()
        zp_flattened = zp.flatten()
        fq = smith.fake_quantize_per_channel_affine(
            x, s_flattened, zp_flattened, 0, qmin, qmax,
        )
        smith.testing.assert_close(dq, fq, rtol=0, atol=0)


if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
