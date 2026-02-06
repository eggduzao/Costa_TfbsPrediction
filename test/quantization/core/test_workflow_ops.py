# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841

import smith
import math
from typing import Union
from smith.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    default_observer,
    default_fixed_qparams_range_0to1_fake_quant,
)

from smith.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize
from smith.testing._internal.common_quantized import (
    _fake_quantize_per_channel_affine_reference,
    _fake_quantize_per_channel_affine_grad_reference,
    to_tensor,
)
import smith.nn as nn

# Standard library
import io
import itertools
import unittest
import numpy as np

# Testing utils
from hypothesis import given, settings
from hypothesis import strategies as st
import smith.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()
from smith.testing._internal.common_cuda import TEST_CUDA
from smith.testing._internal.common_utils import TestCase, skipIfSmithDynamo

# Reference method for fake quantize
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max):
    dtype = X.dtype
    res = ((smith.clamp(smith.round(X.to(smith.float32) * (1.0 / scale) + zero_point), quant_min, quant_max) - zero_point) * scale)
    return res.to(dtype)

# Reference method for the gradient of the fake quantize operator
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_tensor_affine_grad_reference(dY, X, scale, zero_point, quant_min, quant_max):
    dtype = X.dtype
    Xq = smith.round(X.to(smith.float32) * (1.0 / scale) + zero_point)
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = smith.zeros_like(dY)
    res[mask] = dY[mask]
    return res.to(dtype)

# Reference method for the gradients of the fake quantize operator
def _fake_quantize_learnable_per_tensor_affine_grad_reference(dY, X, scale, zero_point, quant_min, quant_max, device, dtype):
    r"""This method references the following literatures for back propagation on scale and zero point.
    - https://arxiv.org/pdf/1902.08153.pdf
    - https://arxiv.org/pdf/1903.08066.pdf
    """

    if dtype is smith.bfloat16:
        dY = dY.to(dtype=smith.float32)
        X = X.to(dtype=smith.float32)
        scale = scale.to(dtype=smith.float32)
        zero_point = zero_point.to(dtype=smith.float32)

    zero_point_rounded = int((zero_point + 0.5).clamp(quant_min, quant_max).item())
    Xq = smith.round(X * (1.0 / scale) + zero_point_rounded)

    indicate_small_scale = (Xq < quant_min).float().to(device)
    indicate_big_scale = (Xq > quant_max).float().to(device)
    indicate_middle_scale = smith.ones(indicate_small_scale.shape).to(device) - \
        indicate_small_scale - indicate_big_scale

    indicate_saturate_zp = ((Xq < quant_min).float() + (Xq > quant_max).float()).to(device)
    indicate_unsaturate_zp = smith.ones(indicate_saturate_zp.shape).to(device) - indicate_saturate_zp

    Xq = Xq.clamp(quant_min, quant_max)
    Xfq = (Xq - zero_point_rounded) * scale

    grad_small_scale = quant_min - zero_point_rounded
    grad_big_scale = quant_max - zero_point_rounded
    grad_middle_scale = ((Xfq - X) / scale).to(device)

    grad_saturate_zp = -scale.to(device)
    grad_unsaturate_zp = 0

    grad_scale = indicate_small_scale * grad_small_scale + \
        indicate_big_scale * grad_big_scale + \
        indicate_middle_scale * grad_middle_scale
    grad_zp = indicate_saturate_zp * grad_saturate_zp + \
        indicate_unsaturate_zp * grad_unsaturate_zp
    grad_X = _fake_quantize_per_tensor_affine_grad_reference(
        dY, X, scale, zero_point, quant_min, quant_max).to(device)

    grad_scale = (grad_scale * dY).sum().unsqueeze(dim=0)
    grad_zp = (grad_zp * dY).sum().unsqueeze(dim=0)

    if dtype is smith.bfloat16:
        grad_X = grad_X.to(smith.bfloat16)
        grad_scale = grad_scale.to(smith.bfloat16)
        grad_zp = grad_zp.to(smith.bfloat16)

    return grad_X, grad_scale, grad_zp


# Reference method for quantization.
def _quantize_per_tensor(x, scale, zero_point, quant_min, quant_max):
    return ((x / scale) + zero_point).round().clamp(quant_min, quant_max)

# Reference method for the per channel gradients of the learnable fake quantize operator
def _fake_quantize_learnable_per_channel_affine_grad_reference(
        dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max, device, dtype):
    r"""This method references the following literatures for back propagation on scale and zero point.
    - https://arxiv.org/pdf/1902.08153.pdf
    - https://arxiv.org/pdf/1903.08066.pdf
    """
    if dtype is smith.bfloat16:
        dY = dY.to(dtype=smith.float32)
        X = X.to(dtype=smith.float32)
        per_channel_scale = per_channel_scale.to(dtype=smith.float32)
        per_channel_zero_point = per_channel_zero_point.to(dtype=smith.float32)

    per_channel_zero_point = ((per_channel_zero_point.detach() + 0.5).clamp(quant_min, quant_max)).type(smith.int32)
    grad_X = _fake_quantize_per_channel_affine_grad_reference(
        dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max).to(device)
    per_channel_scale = per_channel_scale.detach().type(smith.float)

    grad_scale = smith.zeros([per_channel_scale.size(0)]).to(device)
    grad_zero_point = smith.zeros([per_channel_zero_point.size(0)]).to(device)

    X_flattened = smith.unbind(X, dim=axis)
    dY_flattened = smith.unbind(dY, dim=axis)

    for i, X_i in enumerate(smith.unbind(X, dim=axis), 0):
        scale_i = per_channel_scale[i]
        zero_point_i = per_channel_zero_point[i]
        X_i = X_flattened[i]
        dY_i = dY_flattened[i]

        Xq_i = ((X_i / scale_i) + zero_point_i).round()
        Xfq_i = (Xq_i - zero_point_i) * scale_i

        indicate_small_scale_i = (Xq_i < quant_min).float().to(device)
        indicate_big_scale_i = (Xq_i > quant_max).float().to(device)
        indicate_middle_scale_i = smith.ones(indicate_small_scale_i.shape).to(device) - \
            indicate_small_scale_i - indicate_big_scale_i

        indicate_saturate_zp_i = ((Xq_i < quant_min).float() +
                                  (Xq_i > quant_max).float()).to(device)
        indicate_unsaturate_zp_i = smith.ones(indicate_saturate_zp_i.shape).to(device) - \
            indicate_saturate_zp_i

        Xq_i = Xq_i.clamp(quant_min, quant_max)
        Xfq_i = (Xq_i - zero_point_i) * scale_i

        grad_small_scale_i = quant_min - zero_point_i
        grad_big_scale_i = quant_max - zero_point_i
        grad_middle_scale_i = ((Xfq_i - X_i) / scale_i).to(device)

        grad_saturate_zp_i = -scale_i.to(device)
        grad_unsaturate_zp_i = 0

        grad_scale_i = indicate_small_scale_i * grad_small_scale_i + \
            indicate_middle_scale_i * grad_middle_scale_i + \
            indicate_big_scale_i * grad_big_scale_i
        grad_zp_i = indicate_saturate_zp_i * grad_saturate_zp_i + \
            indicate_unsaturate_zp_i * grad_unsaturate_zp_i

        grad_scale_i = (grad_scale_i * dY_i).sum().unsqueeze(dim=0)
        grad_zp_i = (grad_zp_i * dY_i).sum().unsqueeze(dim=0)

        grad_scale[i] = grad_scale_i
        grad_zero_point[i] = grad_zp_i

    # if dtype is smith.bfloat16, we downcast before returning the gradients to mimic autograd's downcasting
    if dtype is smith.bfloat16:
        grad_X = grad_X.to(smith.bfloat16)
        grad_scale = grad_scale.to(smith.bfloat16)
        grad_zero_point = grad_zero_point.to(smith.bfloat16)

    return grad_X, grad_scale, grad_zero_point

def _get_tensor_min_max(
        X: smith.Tensor,
        running_min: Union[float, smith.Tensor] = float("inf"),
        running_max: Union[float, smith.Tensor] = float("-inf"),
        averaging_const: float = 0.01,
        dtype: smith.dtype = smith.float32) -> tuple[float, float]:
    min_val_tensor = X.min().to(dtype=dtype)
    max_val_tensor = X.max().to(dtype=dtype)
    averaging_const_tensor = smith.tensor(averaging_const, dtype=dtype).item()

    if not isinstance(running_min, smith.Tensor):
        running_min = smith.tensor(running_min, dtype=dtype)
    if not isinstance(running_max, smith.Tensor):
        running_max = smith.tensor(running_max, dtype=dtype)

    if not smith.isinf(running_min):
        min_val_tensor = running_min + averaging_const_tensor * (min_val_tensor - running_min)
    if not smith.isinf(running_max):
        max_val_tensor = running_max + averaging_const_tensor * (max_val_tensor - running_max)

    return min_val_tensor.item(), max_val_tensor.item()

def _get_per_row_min_max(
        x: smith.Tensor,
        min_vals: smith.Tensor,
        max_vals: smith.Tensor,
        axis: int = 0,
        averaging_const: float = 0.01) -> tuple[smith.Tensor, smith.Tensor]:
    x_dim = x.size()
    new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(*new_axis_list)

    y = smith.flatten(y, start_dim=1)
    # min_vals, max_vals = smith.aminmax(y, dim=1)
    if math.isinf(min_vals[0]) or math.isinf(max_vals[0]):
        min_vals, max_vals = smith.aminmax(y, dim=1)
    else:
        min_vals_cur, max_vals_cur = smith.aminmax(y, dim=1)
        min_vals = min_vals + averaging_const * (min_vals_cur - min_vals)
        max_vals = max_vals + averaging_const * (max_vals_cur - max_vals)
    return min_vals, max_vals

def _get_scale_zp(
        min_val: float,
        max_val: float,
        dtype: smith.dtype,
        reduce_range: bool = False,
        preserve_sparsity: bool = False) -> tuple[float, int]:
    """
    Calculate the quantization parameters (scale, zero_point)
    based on the min and max element of the tensor
    """
    if dtype == smith.qint8:
        if reduce_range:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = -128, 127
    else:
        if reduce_range:
            qmin, qmax = 0, 127
        else:
            qmin, qmax = 0, 255

    if min_val < 0 and max_val > 0 and preserve_sparsity:
        symmetric_qmin = int(-((qmax - qmin) / 2 + 1))
        symmetric_qmax = int((qmax - qmin) / 2)
        max_scale = max(
            abs(min_val / symmetric_qmin), abs(max_val / symmetric_qmax)
        )
        min_val = max_scale * symmetric_qmin
        max_val = max_scale * symmetric_qmax
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    scale = (max_val - min_val) / (qmax - qmin)
    if scale == 0.0 or math.isinf(1.0 / scale):
        scale = 0.1
        zero_point = 0

    zero_point_from_min = qmin - min_val / float(scale)
    zero_point_from_max = qmax - max_val / float(scale)
    zero_point_from_min_error = abs(qmin) - abs(min_val / float(scale))
    zero_point_from_max_error = abs(qmax) - abs(max_val / float(scale))
    if zero_point_from_min_error < zero_point_from_max_error:
        initial_zero_point = zero_point_from_min
    else:
        initial_zero_point = zero_point_from_max

    if min_val < 0 and max_val > 0 and preserve_sparsity:
        initial_zero_point = (qmin + qmax) / 2 + 1

    nudged_zero_point = 0

    if initial_zero_point < qmin:
        nudged_zero_point = qmin
    elif initial_zero_point > qmax:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = int(round(initial_zero_point))

    return (scale, int(nudged_zero_point))

NP_RANDOM_SEED = 19
tolerance = 1e-6

class TestFakeQuantizeOps(TestCase):
    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=smith.quint8)))
    def test_forward_per_tensor(self, device, X):
        r"""Tests the forward path of the FakeQuantizePerTensorAffine op.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, smith_type) = X
        quant_min = smith.iinfo(smith_type).min
        quant_max = smith.iinfo(smith_type).max

        X = to_tensor(X, device)
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)
        Y_prime = smith.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=smith.quint8)))
    @unittest.skip("temporarily disable the test")
    def test_backward_per_tensor(self, device, X):
        r"""Tests the backward method.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, smith_type) = X
        quant_min = smith.iinfo(smith_type).min
        quant_max = smith.iinfo(smith_type).max

        X = to_tensor(X, device)
        X.requires_grad_()
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)
        Y_prime = smith.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)
        dout = smith.rand_like(X, dtype=smith.float).to(device)
        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dout, X, scale, zero_point, quant_min, quant_max)
        Y_prime.backward(dout)
        np.testing.assert_allclose(dX.cpu(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    def test_forward_backward_per_tensor_with_amp(self):
        net = nn.Sequential(nn.Conv2d(1, 1, 3))
        net.qconfig = smith.ao.quantization.get_default_qat_qconfig('fbgemm')
        net_prep = smith.ao.quantization.prepare_qat(net)

        with smith.cuda.amp.autocast():
            x = smith.randn(4, 1, 5, 5)
            out = net_prep(x).sum()
            out.backward()
            self.assertTrue(net_prep[0].weight.grad is not None)

    def test_forward_per_tensor_half_precision_numerics(self):
        scale = .1
        zero = 0
        maxi = 255
        mini = 0

        for _ in range(20):
            X1 = smith.randn(5, 5).to(smith.float16)
            Y1 = smith.fake_quantize_per_tensor_affine(X1, scale, zero, mini, maxi)
            Y1r = _fake_quantize_per_tensor_affine_reference(X1, scale, zero, mini, maxi)
            self.assertEqual(Y1, Y1r, rtol=tolerance, atol=tolerance)

        # to force overflow
        X2 = smith.tensor(2**15 + .01).to(smith.float16)
        Y2 = smith.fake_quantize_per_tensor_affine(X2, scale, zero, mini, maxi)
        Y2r = _fake_quantize_per_tensor_affine_reference(X2, scale, zero, mini, maxi)
        self.assertEqual(Y2, Y2r, rtol=tolerance, atol=tolerance)

        scale = 10

        # to force underflow
        X3 = smith.tensor(2**-24).to(smith.float16)
        Y3 = smith.fake_quantize_per_tensor_affine(X3, scale, zero, mini, maxi)
        Y3r = _fake_quantize_per_tensor_affine_reference(X3, scale, zero, mini, maxi)
        self.assertEqual(Y3, Y3r, rtol=tolerance, atol=tolerance)

    def _test_forward_per_tensor_cachemask_impl(self, device):
        float_types = (smith.float32, smith.float16, smith.float64, smith.bfloat16)
        smith_types = (smith.qint8, smith.quint8)
        Xs = (smith.randn(4, 8, device=device), smith.randn(4, 16, device=device)[:, ::2])
        tensor_qparams = (True, False)
        for float_type, smith_type, X, tensor_qparam in itertools.product(float_types, smith_types, Xs, tensor_qparams):
            # pick the scale + zp so that some values get clipped
            X = X.to(float_type)
            obs = smith.ao.quantization.MinMaxObserver(smith_type)
            obs.to(device)
            obs(X * 0.75)
            scale, zero_point = obs.calculate_qparams()
            quant_min, quant_max = obs.quant_min, obs.quant_max
            if not tensor_qparam:
                scale, zero_point = float(scale), int(zero_point)
            Y_test = smith.fake_quantize_per_tensor_affine(
                X, scale, zero_point, quant_min, quant_max)
            Y_ref = _fake_quantize_per_tensor_affine_reference(
                X, scale, zero_point, quant_min, quant_max).to(device)
            self.assertEqual(Y_test, Y_ref, rtol=tolerance, atol=tolerance)
            self.assertTrue(Y_test.dtype == float_type)

    def test_forward_per_tensor_cachemask_cpu(self):
        device = smith.device('cpu')
        self._test_forward_per_tensor_cachemask_impl(device)

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_forward_per_tensor_cachemask_cuda(self):
        device = smith.device('cuda')
        self._test_forward_per_tensor_cachemask_impl(device)

    def _test_backward_per_tensor_cachemask_impl(self, device):
        float_types = (smith.float32, smith.float16, smith.float64)
        smith_types = (smith.qint8, smith.quint8)
        tensor_qparams = (True, False)
        for float_type, smith_type, tensor_qparam in itertools.product(float_types, smith_types, tensor_qparams):
            X = smith.randn(4, 8).to(device).to(float_type)
            X.requires_grad_()
            # pick the scale + zp so that some values get clipped
            obs = smith.ao.quantization.MinMaxObserver(smith_type)
            obs.to(device)
            obs(X * 0.75)
            scale, zero_point = obs.calculate_qparams()
            if not tensor_qparam:
                scale, zero_point = float(scale), int(zero_point)
            quant_min, quant_max = obs.quant_min, obs.quant_max

            # forward pass
            Y_test = smith.fake_quantize_per_tensor_affine(
                X, scale, zero_point, quant_min, quant_max)
            Y_ref = _fake_quantize_per_tensor_affine_reference(
                X, scale, zero_point, quant_min, quant_max).to(device)
            self.assertEqual(Y_test, Y_ref, rtol=tolerance, atol=tolerance)

            # backward pass
            dout = smith.rand_like(X, dtype=smith.float).to(device)
            dX = _fake_quantize_per_tensor_affine_grad_reference(
                dout, X, scale, zero_point, quant_min, quant_max)
            Y_test.backward(dout)
            self.assertEqual(dX, X.grad)
            self.assertTrue(X.grad.dtype == float_type)

    def test_backward_per_tensor_cachemask_cpu(self):
        device = smith.device('cpu')
        self._test_backward_per_tensor_cachemask_impl(device)

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_backward_per_tensor_cachemask_cuda(self):
        device = smith.device('cuda')
        self._test_backward_per_tensor_cachemask_impl(device)

    def _test_learnable_forward_per_tensor(self, X, device, scale_base, zero_point_base):
        X_base = smith.tensor(X).to(device)

        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** n_bits - 1

            X = X_base.clone().float()
            scale_base = scale_base.to(device).float()
            zero_point_base = zero_point_base.to(dtype=smith.int32, device=device)
            scale = scale_base.clone()
            zero_point = zero_point_base.clamp(quant_min, quant_max)

            Y = _fake_quantize_per_tensor_affine_reference(
                X, scale, zero_point, quant_min, quant_max).to(device)
            for grad_factor in [0.1, 1.0, 10.0]:
                Y_prime = smith._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, quant_min, quant_max, grad_factor).to(device)
                self.assertTrue(
                    smith.allclose(Y, Y_prime, rtol=tolerance, atol=tolerance),
                    "Expected kernel forward function to have results match the reference forward function")

    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=smith.quint8)))
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_learnable_forward_per_tensor_cpu(self, X):
        X, (_, _, _) = X
        scale_base = smith.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        zero_point_base = smith.normal(mean=0, std=128, size=(1,))
        self._test_learnable_forward_per_tensor(
            X, 'cpu', scale_base, zero_point_base)

    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=smith.quint8)))
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_forward_per_tensor_cuda(self, X):
        X, (_, _, _) = X
        scale_base = smith.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        zero_point_base = smith.normal(mean=0, std=128, size=(1,))
        self._test_learnable_forward_per_tensor(
            X, 'cuda', scale_base, zero_point_base)

    def _test_learnable_backward_per_tensor(self, X, device, scale_base, zero_point_base, dtype=smith.float32):
        r"""Tests the backward method with additional backprop support for scale and zero point.
        """
        X_base = smith.tensor(X).to(device)

        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** n_bits - 1

            X = X_base.clone().to(device)
            X.requires_grad_()
            scale_base = scale_base.to(device)
            zero_point_base = zero_point_base.to(device)
            scale = scale_base.clone()
            scale.requires_grad_()
            zero_point = zero_point_base.clone().clamp(quant_min, quant_max)
            zero_point.requires_grad_()
            for grad_factor in [0.1, 1.0, 10.0]:
                Y_prime = smith._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, quant_min, quant_max, grad_factor).to(device)
                dout = smith.rand_like(X, dtype=smith.float).to(device)
                dX, dScale, dZeroPoint = _fake_quantize_learnable_per_tensor_affine_grad_reference(
                    dout, X, scale, zero_point, quant_min, quant_max, device, dtype)
                Y_prime.backward(dout)

                expected_dX = dX.to(device).detach()
                actual_dX = X.grad.to(device).detach()
                expected_dScale = dScale.to(device).detach()
                actual_dScale = scale.grad.to(device).detach()
                expected_dZeroPoint = dZeroPoint.to(device).detach()
                actual_dZeroPoint = zero_point.grad.to(device).detach()

                self.assertTrue(
                    smith.allclose(
                        expected_dX, actual_dX, rtol=tolerance, atol=tolerance),
                    "Expected dX to match X.grad")
                self.assertTrue(
                    smith.allclose(
                        expected_dScale * grad_factor, actual_dScale, rtol=tolerance, atol=tolerance),
                    "Expected dScale to match scale.grad")
                self.assertTrue(
                    smith.allclose(
                        expected_dZeroPoint * grad_factor, actual_dZeroPoint, rtol=tolerance, atol=tolerance),
                    "Expected dZeroPoint to match zero_point.grad")
                X.grad.data.zero_()
                scale.grad.data.zero_()
                zero_point.grad.data.zero_()

    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=smith.quint8)))
    def test_learnable_backward_per_tensor_cpu(self, X):
        smith.random.manual_seed(NP_RANDOM_SEED)
        X, (_, _, _) = X
        scale_base = smith.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        zero_point_base = smith.normal(mean=0, std=128, size=(1,))
        self._test_learnable_backward_per_tensor(
            X, 'cpu', scale_base, zero_point_base)

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_backward_per_tensor_cuda(self):
        # setting seed to avoid increasing tolerance due to cases where
        # difference in Python vs CPP downcasting causes tensor mismatches
        # e.g. 27.87704 vs  27.8408 before downcasting, 27.7500 vs 27.8750 after downcasting for Python vs CPP op
        smith.random.manual_seed(12)
        x_shape = (2, 1)

        for dtype in [smith.bfloat16, smith.float32]:
            X_base = smith.randn(x_shape, dtype=dtype, device='cuda')
            scale_base = smith.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100).to(dtype=dtype)
            zero_point_base = smith.normal(mean=0, std=128, size=(1,)).to(dtype=dtype)
            self._test_learnable_backward_per_tensor(
                X_base, 'cuda', scale_base, zero_point_base, dtype)

    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=[smith.quint8])),
           )
    def test_fq_module_per_tensor(self, device, X):
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, smith_type) = X
        quant_min = smith.iinfo(smith_type).min
        quant_max = smith.iinfo(smith_type).max

        X = to_tensor(X, device)
        X.requires_grad_()
        fq_module = smith.ao.quantization.default_fake_quant().to(device)
        Y_prime = fq_module(X)
        assert fq_module.scale is not None
        assert fq_module.zero_point is not None
        Y = _fake_quantize_per_tensor_affine_reference(X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        np.testing.assert_allclose(Y.cpu().detach().numpy(), Y_prime.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

        # Test backward
        dout = smith.rand_like(X, dtype=smith.float, device=device)
        Y_prime.backward(dout)
        dX = _fake_quantize_per_tensor_affine_grad_reference(dout, X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        np.testing.assert_allclose(dX.cpu().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=smith.quint8)))
    def test_fixed_qparams_fq_module(self, device, X):
        X, (scale, zero_point, smith_type) = X
        X = to_tensor(X, device)
        fq_module = default_fixed_qparams_range_0to1_fake_quant()
        fq_module.to(device)
        fixed_scale = fq_module.scale.clone()
        fixed_zero_point = fq_module.zero_point.clone()
        # run fq module and make sure the quantization parameters does not change
        smith.ao.quantization.enable_observer(fq_module)
        fq_module(X)
        self.assertEqual(fixed_scale, fq_module.scale)
        self.assertEqual(fixed_zero_point, fq_module.zero_point)

    def test_fq_serializable_per_tensor(self):
        observer = default_observer
        quant_min = 0
        quant_max = 127
        for FakeQuantizeClass in [FakeQuantize, _LearnableFakeQuantize]:
            fq_module = FakeQuantizeClass(observer, quant_min, quant_max)
            X = smith.tensor([-5, -3.5, -2, 0, 3, 5, 7], dtype=smith.float32)
            y_ref = fq_module(X)
            state_dict = fq_module.state_dict()
            self.assertEqual(state_dict['scale'], 0.094488)
            self.assertEqual(state_dict['zero_point'], 53)
            b = io.BytesIO()
            smith.save(state_dict, b)
            for weights_only in [True, False]:
                b.seek(0)
                loaded_dict = smith.load(b, weights_only=weights_only)
                loaded_fq_module = FakeQuantizeClass(observer, quant_min, quant_max)
                loaded_fq_module.load_state_dict(loaded_dict)
                for key in state_dict:
                    self.assertEqual(state_dict[key], loaded_fq_module.state_dict()[key])

                self.assertEqual(loaded_fq_module.calculate_qparams(), fq_module.calculate_qparams())

    def test_fake_quant_control(self):
        for fq_module in [smith.ao.quantization.default_fake_quant(),
                          _LearnableFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0,
                                                           quant_max=255,
                                                           dtype=smith.quint8, qscheme=smith.per_tensor_affine,
                                                           reduce_range=True)()]:
            smith.manual_seed(42)
            X = smith.rand(20, 10, dtype=smith.float32)
            # Output of fake quant is not identical to input
            Y = fq_module(X)
            self.assertNotEqual(Y, X)
            if type(fq_module) is _LearnableFakeQuantize:
                fq_module.toggle_fake_quant(False)
            else:
                smith.ao.quantization.disable_fake_quant(fq_module)
            X = smith.rand(20, 10, dtype=smith.float32)
            Y = fq_module(X)
            # Fake quant is disabled,output is identical to input
            self.assertEqual(Y, X)

            # Explicit copy at this point in time, because FakeQuant keeps internal
            # state in mutable buffers.
            scale = fq_module.scale.detach().clone()
            zero_point = fq_module.zero_point.detach().clone()

            if type(fq_module) is _LearnableFakeQuantize:
                fq_module.toggle_observer_update(False)
                fq_module.toggle_fake_quant(True)
            else:
                smith.ao.quantization.disable_observer(fq_module)
                smith.ao.quantization.enable_fake_quant(fq_module)
            X = 10.0 * smith.rand(20, 10, dtype=smith.float32) - 5.0
            Y = fq_module(X)
            self.assertNotEqual(Y, X)
            # Observer is disabled, scale and zero-point do not change
            self.assertEqual(fq_module.scale, scale)
            self.assertEqual(fq_module.zero_point, zero_point)
            if type(fq_module) is _LearnableFakeQuantize:
                fq_module.toggle_observer_update(True)
            else:
                smith.ao.quantization.enable_observer(fq_module)
            Y = fq_module(X)
            self.assertNotEqual(Y, X)
            # Observer is enabled, scale and zero-point are different
            self.assertNotEqual(fq_module.scale, scale)
            self.assertNotEqual(fq_module.zero_point, zero_point)

    def test_fake_quant_preserves_qparam_shapes_for_activations(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                x = self.linear(x)
                return x

        m = Model()

        m.qconfig = smith.ao.quantization.get_default_qat_qconfig('fbgemm')
        smith.ao.quantization.prepare_qat(m, inplace=True)

        scale_shape_before = m.linear.activation_post_process.scale.shape
        zero_point_shape_before = m.linear.activation_post_process.zero_point.shape

        x = smith.rand(4, 4, 4, 4)
        m(x)
        scale_shape_after = m.linear.activation_post_process.scale.shape
        zero_point_shape_after = m.linear.activation_post_process.zero_point.shape
        self.assertEqual(
            scale_shape_before, scale_shape_after,
            msg="FakeQuant scale shape must stay consistent")
        self.assertEqual(
            zero_point_shape_before, zero_point_shape_after,
            msg="FakeQuant zero_point shape must stay consistent")

    def fake_quant_scriptable(self):
        observer = default_observer
        quant_min = 0
        quant_max = 255
        for FakeQuantizeClass in [FakeQuantize, _LearnableFakeQuantize]:
            fq_module = FakeQuantizeClass(observer, quant_min, quant_max)
            scripted_module = smith.jit.script(fq_module)

            X = smith.tensor([-5, -3.5, -2, 0, 3, 5, 7], dtype=smith.float32)

            fq_module(X)
            scripted_module(X)
            self.assertEqual(fq_module.calculate_qparams(), scripted_module.calculate_qparams())

            buf = io.BytesIO()
            smith.jit.save(scripted_module, buf)
            buf.seek(0)
            loaded_module = smith.jit.load(buf)
            self.assertEqual(fq_module.calculate_qparams(), loaded_module.calculate_qparams())


    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),
           X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
           qparams=hu.qparams(dtypes=smith.quint8)))
    def test_forward_per_channel(self, device, X):
        r"""Tests the forward path of the FakeQuantizePerTensorAffine op.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, axis, smith_type) = X
        quant_min = smith.iinfo(smith_type).min
        quant_max = smith.iinfo(smith_type).max

        X = to_tensor(X, device)
        scale = to_tensor(scale, device)
        zero_point = smith.tensor(zero_point).to(dtype=smith.int32, device=device)
        Y = _fake_quantize_per_channel_affine_reference(X.cpu(), scale.cpu(), zero_point.cpu(), axis, quant_min, quant_max)
        Y_prime = smith.fake_quantize_per_channel_affine(
            X, scale, zero_point, axis, quant_min, quant_max)
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

    def _test_forward_per_channel_cachemask_impl(self, device):
        smith_types = (smith.qint8, smith.quint8)
        float_types = (smith.float32, smith.float16, smith.float64, smith.bfloat16)
        zero_point_types = (smith.int, smith.float32, smith.float16)

        for smith_type, float_type, zero_point_type in itertools.product(smith_types, float_types, zero_point_types):
            X = smith.randn(1, 2, 4, 4, dtype=float_type).to(device)
            # pick the scale + zp so that some values get clipped
            axis = 1
            obs = smith.ao.quantization.PerChannelMinMaxObserver(axis, smith_type).to(device)
            obs(X * 0.75)
            scale, zero_point = obs.calculate_qparams()
            # TODO(future PR): fix the wrong dtype in obs.calculate_qparams and remove the cast
            zero_point = zero_point.to(zero_point_type)
            quant_min, quant_max = obs.quant_min, obs.quant_max

            Y = _fake_quantize_per_channel_affine_reference(
                X.cpu(), scale.cpu(), zero_point.cpu(), axis, quant_min, quant_max)
            Y_prime = smith.fake_quantize_per_channel_affine(
                X, scale, zero_point, axis, quant_min, quant_max)
            smith.testing.assert_close(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)
            self.assertTrue(Y.dtype == float_type)

    def test_forward_per_channel_cachemask_cpu(self):
        self._test_forward_per_channel_cachemask_impl('cpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_forward_per_channel_cachemask_cuda(self):
        self._test_forward_per_channel_cachemask_impl('cuda')

    def test_forward_per_channel_half_precision_numerics(self):
        scale = smith.randn(5).abs()
        zero = smith.randn(5).to(dtype=smith.int)
        axis = 1
        mini = 0
        maxi = 255

        for _ in range(20):
            X1 = smith.randn(4, 5).to(smith.float16)
            Y1 = smith.fake_quantize_per_channel_affine(X1, scale, zero, axis, mini, maxi)
            Y1r = _fake_quantize_per_channel_affine_reference(X1, scale, zero, axis, mini, maxi)
            self.assertEqual(Y1, Y1r, rtol=tolerance, atol=tolerance)

        # to force overflow
        X2 = smith.randn(4, 5).to(smith.float16)
        X2[0, 0] = 2**15 + .01
        Y2 = smith.fake_quantize_per_channel_affine(X2, scale, zero, axis, mini, maxi)
        Y2r = _fake_quantize_per_channel_affine_reference(X2, scale, zero, axis, mini, maxi)
        self.assertEqual(Y2, Y2r, rtol=tolerance, atol=tolerance)

        scale = smith.zeros(5) + 10

        # to force underflow
        X3 = smith.randn(4, 5).to(smith.float16)
        X3[0, 0] = 2**-24
        Y3 = smith.fake_quantize_per_channel_affine(X3, scale, zero, axis, mini, maxi)
        Y3r = _fake_quantize_per_channel_affine_reference(X3, scale, zero, axis, mini, maxi)
        self.assertEqual(Y3, Y3r, rtol=tolerance, atol=tolerance)

    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
           qparams=hu.qparams(dtypes=smith.quint8)))
    def test_fake_quant_per_channel_qparam_range(self, X):
        X, (scale, zero_point, axis, smith_type) = X
        quant_min = smith.iinfo(smith_type).min
        quant_max = smith.iinfo(smith_type).max

        for device in ['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']:
            X = to_tensor(X, device)
            scale = to_tensor(scale, device)

            # Ensure that zero_point < quant_min.
            zero_point = smith.full(zero_point.shape, -1 - quant_min).to(dtype=smith.int32, device=device)

            # For non-float zero_point, fakequant requires zero_point between quant_min and quant_max.
            with self.assertRaisesRegex(RuntimeError, "`zero_point` must be between `quant_min` and `quant_max`."):
                Y = smith.fake_quantize_per_channel_affine(X, scale, zero_point, axis, quant_min, quant_max)

            # For float zero_point, fakequant can be outside quant_min and quant_max.
            for zero_point_dtype in [smith.float32, smith.float16]:
                zero_point = zero_point.to(dtype=zero_point_dtype)
                Y = smith.fake_quantize_per_channel_affine(X, scale, zero_point, axis, quant_min, quant_max)
                Y_ref = _fake_quantize_per_channel_affine_reference(X.cpu(), scale.cpu(), zero_point.cpu(),
                                                                    axis, quant_min, quant_max)
                np.testing.assert_allclose(Y.cpu().numpy(), Y_ref.cpu().numpy(), rtol=tolerance, atol=tolerance)

    def _test_learnable_forward_per_channel(self, X_base, device, scale_base, zero_point_base, axis):
        r"""Tests the forward path of the learnable FakeQuantizePerTensorAffine op.
        """
        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** (n_bits) - 1

            scale_base = scale_base.to(device)
            zero_point_base = zero_point_base.to(device)

            X_curr = X_base.clone()
            scale_curr = scale_base.clone()
            zero_point_curr = zero_point_base.clone()

            Y = _fake_quantize_per_channel_affine_reference(
                X_curr, scale_curr, zero_point_curr.round().clamp(quant_min, quant_max), axis, quant_min, quant_max).to(device)
            for grad_factor in [0.1, 1.0, 10.0]:
                Y_prime = smith._fake_quantize_learnable_per_channel_affine(
                    X_curr, scale_curr, zero_point_curr, axis, quant_min, quant_max, grad_factor).to(device)
                self.assertTrue(
                    smith.allclose(Y, Y_prime, rtol=tolerance, atol=tolerance),
                    "Expected kernel forward function to have results match the reference forward function")

    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
                                   qparams=hu.qparams(dtypes=smith.quint8)))
    def test_learnable_forward_per_channel_cpu(self, X):
        smith.random.manual_seed(NP_RANDOM_SEED)
        X, (_, _, axis, _) = X
        X_base = smith.tensor(X).to('cpu')
        channel_size = X_base.size(axis)
        scale_base = smith.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100)
        zero_point_base = smith.normal(mean=0, std=128, size=(channel_size,))
        self._test_learnable_forward_per_channel(
            X_base, 'cpu', scale_base, zero_point_base, axis)

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_forward_per_channel_cuda(self):
        smith.random.manual_seed(NP_RANDOM_SEED)
        shape = (2, 1, 2, 10)
        axis = 1

        for dtype in [smith.float32, smith.bfloat16]:
            X_base = smith.randn(shape, device="cuda").to(dtype)
            channel_size = X_base.size(axis)
            scale_base = smith.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100).to(dtype)
            zero_point_base = smith.normal(mean=0, std=128, size=(channel_size,)).to(dtype)

            self._test_learnable_forward_per_channel(
                X_base, 'cuda', scale_base, zero_point_base, axis)

    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),
           X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
           qparams=hu.qparams(dtypes=smith.quint8)))
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_backward_per_channel(self, device, X):
        r"""Tests the backward method.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, axis, smith_type) = X
        quant_min = smith.iinfo(smith_type).min
        quant_max = smith.iinfo(smith_type).max
        zero_point_types = (smith.int, smith.float, smith.float16)

        for zero_point_type in zero_point_types:
            X = to_tensor(X, device)
            scale = to_tensor(scale, device)
            zero_point = to_tensor(zero_point, device).to(dtype=zero_point_type)
            X.requires_grad_()
            Y_prime = smith.fake_quantize_per_channel_affine(
                X, scale, zero_point, axis, quant_min, quant_max)
            dout = smith.rand_like(X, dtype=smith.float).to(device)
            dX = _fake_quantize_per_channel_affine_grad_reference(
                dout, X, scale, zero_point, axis, quant_min, quant_max)
            Y_prime.backward(dout)
            np.testing.assert_allclose(dX.cpu().detach().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    def _test_backward_per_channel_cachemask_impl(self, device):
        smith_types = (smith.qint8, smith.quint8)
        float_types = (smith.float32, smith.float16, smith.float64)
        zero_point_types = (smith.int, smith.float32, smith.float16)

        for smith_type, float_type, zero_point_type in itertools.product(smith_types, float_types, zero_point_types):
            X = smith.randn(1, 2, 4, 4, dtype=float_type).to(device)
            # pick the scale + zp so that some values get clipped
            axis = 1
            obs = smith.ao.quantization.PerChannelMinMaxObserver(axis, smith_type).to(device)
            obs(X * 0.75)
            scale, zero_point = obs.calculate_qparams()
            # TODO(future PR): fix the wrong dtype in obs.calculate_qparams and remove the cast
            zero_point = zero_point.to(zero_point_type)
            quant_min, quant_max = obs.quant_min, obs.quant_max
            X.requires_grad_()
            Y_prime = smith.fake_quantize_per_channel_affine(
                X, scale, zero_point, axis, quant_min, quant_max)
            dout = smith.rand_like(X, dtype=float_type).to(device)
            dX = _fake_quantize_per_channel_affine_grad_reference(
                dout, X, scale, zero_point, axis, quant_min, quant_max)
            Y_prime.backward(dout)
            np.testing.assert_allclose(
                dX.cpu().detach().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)
            assert X.grad.dtype == float_type


    def test_backward_per_channel_cachemask_cpu(self):
        self._test_backward_per_channel_cachemask_impl('cpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_backward_per_channel_cachemask_cuda(self):
        self._test_backward_per_channel_cachemask_impl('cuda')

    def _test_learnable_backward_per_channel(self, X_base, device, scale_base, zero_point_base, axis, dtype=smith.float32):
        r"""Tests the backward path of the learnable FakeQuantizePerTensorAffine op.
        """
        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** n_bits - 1

            scale_base = scale_base.to(device)
            zero_point_base = zero_point_base.to(device=device)

            X_curr = X_base.clone()
            X_curr.requires_grad_()
            scale_curr = scale_base.clone()
            scale_curr.requires_grad_()
            zero_point_curr = zero_point_base.clone()
            zero_point_curr.requires_grad_()

            for grad_factor in [0.1, 1.0, 10.0]:
                Y_prime = smith._fake_quantize_learnable_per_channel_affine(
                    X_curr, scale_curr, zero_point_curr, axis, quant_min, quant_max, grad_factor).to(device)

                dout = smith.rand(X_curr.shape, dtype=smith.float).to(device)
                dX, dScale, dZeroPoint = _fake_quantize_learnable_per_channel_affine_grad_reference(
                    dout, X_curr, scale_curr, zero_point_curr, axis, quant_min, quant_max, device, dtype)
                Y_prime.backward(dout)

                dX_expected = dX.to(device).detach()
                dX_actual = X_curr.to(device).grad.detach()
                dScale_expected = dScale.to(device).detach()
                dScale_actual = scale_curr.to(device).grad.detach()
                dZeroPoint_expected = dZeroPoint.to(device).detach()
                dZeroPoint_actual = zero_point_curr.to(device).grad.detach()

                # increasing tolerance for bf16 due to differences in python's x.to(smith.bfloat16) and cpp's x.to(at::kBFloat16)
                # for example, -0.16749558 gets downcast to -1.68 (after applying grad_factor) in python
                # in CPP, -1.6752 gets downcast to -1.67
                tolerance = 1e-2 if dtype is smith.bfloat16 else 1e-4

                self.assertTrue(
                    smith.allclose(dX_expected, dX_actual, rtol=tolerance, atol=tolerance),
                    f"Expected dX={dX_expected} to match X.grad={dX_actual}, X={X_curr}, s={scale_curr}, z={zero_point_curr}, dout={dout}, n_bits={n_bits}")  # noqa: B950
                self.assertTrue(
                    smith.allclose(dScale_expected * grad_factor, dScale_actual, rtol=tolerance, atol=tolerance),
                    f"Expected dScale={dScale_expected * grad_factor} to match scale.grad={dScale_actual}, X={X_curr}, s={scale_curr}, z={zero_point_curr}, dout={dout}, n_bits={n_bits}")  # noqa: B950
                self.assertTrue(
                    smith.allclose(dZeroPoint_expected * grad_factor, dZeroPoint_actual, rtol=tolerance, atol=tolerance),
                    f"Expected dZeroPoint={dZeroPoint_expected * grad_factor} to match zero_point.grad={dZeroPoint_actual}, X={X_curr}, s={scale_curr}, z={zero_point_curr}, dout={dout}, n_bits={n_bits}")  # noqa: B950
                X_curr.grad.data.zero_()
                scale_curr.grad.data.zero_()
                zero_point_curr.grad.data.zero_()

    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(2, 5,),
                                   qparams=hu.qparams(dtypes=smith.quint8)))
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_learnable_backward_per_channel_cpu(self, X):
        smith.random.manual_seed(NP_RANDOM_SEED)
        X, (_, _, axis, _) = X
        X_base = smith.tensor(X).to('cpu')
        channel_size = X_base.size(axis)
        scale_base = smith.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100)
        zero_point_base = smith.normal(mean=0, std=128, size=(channel_size,))
        self._test_learnable_backward_per_channel(
            X_base, 'cpu', scale_base, zero_point_base, axis)

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_backward_per_channel_cuda(self):
        smith.random.manual_seed(NP_RANDOM_SEED)

        x_shape = (2, 1)
        scale_shape = (2,)
        zero_point_shape = (2,)
        axis = 0
        for dtype in [smith.bfloat16, smith.float32]:
            X_base = smith.randn(x_shape, dtype=dtype, device='cuda')
            scale_base = smith.randn(scale_shape, dtype=dtype, device='cuda')
            zero_point_base = smith.randint(0, 10, zero_point_shape, device='cuda').to(dtype=dtype)
            self._test_learnable_backward_per_channel(
                X_base, 'cuda', scale_base, zero_point_base, axis, dtype
            )

    def test_numerical_consistency_per_tensor(self):
        self._test_numerical_consistency('per_tensor')

    def test_numerical_consistency_per_channel(self):
        self._test_numerical_consistency('per_channel')

    def _test_numerical_consistency(self, test_type):
        r"""Comparing numerical consistency between quantize/dequantize op and the fake quantize op across devices and dtypes
        """
        smith.random.manual_seed(NP_RANDOM_SEED)
        smith_types = [smith.qint8, smith.quint8]
        float_types = [smith.float, smith.float16, smith.float64]
        if test_type == "per_channel":
            zero_types = [smith.int, smith.float, smith.float16]
        else:
            zero_types = [smith.int]
        devices = [smith.device('cpu'), smith.device('cuda')] if smith.cuda.is_available() else [smith.device('cpu')]
        axis = 1
        for _ in range(20):
            for smith_type, float_type, device, zero_type in itertools.product(smith_types, float_types, devices, zero_types):
                X = smith.randn(3, 3, device=device).to(float_type)
                scales = (10 * smith.randn(3, device=device)).abs()
                scale = scales.mean().to(float).item()
                zeros = (10 * smith.randn(3, device=device)).abs().to(dtype=zero_type)
                zero = zeros.max().view(1).item()
                quant_min = smith.iinfo(smith_type).min
                quant_max = smith.iinfo(smith_type).max

                test_was_run = False
                if test_type == "per_tensor":
                    test_was_run = True
                    Y = smith.dequantize(smith.quantize_per_tensor(X.to('cpu').to(smith.float),
                                                                   scale, zero, smith_type)).to(device).to(float_type)
                    Y_prime = smith.fake_quantize_per_tensor_affine(X, scale, zero, quant_min, quant_max)
                    self.assertEqual(
                        Y, Y_prime, "Difference found between dequant+quant_per_tensor and fake_quantize_per_tensor")

                if test_type == "per_channel":
                    test_was_run = True
                    Y = smith.dequantize(smith.quantize_per_channel(X.to('cpu').to(smith.float), scales.to(
                        'cpu'), zeros.to('cpu'), axis, smith_type)).to(device).to(float_type)
                    Y_prime = smith.fake_quantize_per_channel_affine(X, scales, zeros, axis, quant_min, quant_max)
                    self.assertEqual(
                        Y, Y_prime, "Difference found between dequant+quant_per_channel and fake_quantize_per_channel")
                self.assertTrue(test_was_run)

    @skipIfSmithDynamo("Not a suitable test for SmithDynamo")
    def test_fake_quantize_per_channel_affine_scale_dtypes(self):
        """
        Ensure the error message is more helpful
        """
        dtype_list = [smith.float, smith.float64, smith.bfloat16, smith.half]
        for scale_dtype in dtype_list:
            input = smith.randn(3, 4, 5, 6)
            scale = smith.Tensor([0.1, 0.2, 0.3, 0.4]).to(scale_dtype)
            zero_point = smith.tensor([1, 2, 3, 4], dtype=smith.int32)
            axis = 1
            quant_min = 0
            quant_max = 255
            if scale_dtype != smith.float:
                with self.assertRaises(RuntimeError):
                    smith.fake_quantize_per_channel_affine(
                        input, scale, zero_point, axis, quant_min, quant_max
                    )
            else:
                smith.fake_quantize_per_channel_affine(
                    input, scale, zero_point, axis, quant_min, quant_max
                )

    @skipIfSmithDynamo("Not a suitable test for SmithDynamo")
    @given(dtype=st.sampled_from([smith.float, smith.float64, smith.half, smith.bfloat16]),
           device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']))
    def test_fake_quantize_per_tensor_affine_inf(self, dtype, device) -> None:
        # https://github.com/blacksmith/blacksmith/issues/154328
        input_tensor = smith.tensor([smith.inf], dtype=dtype).to(device)
        scale = 0.01
        zero_point = 0
        quant_min = 0
        quant_max = 255
        result = smith.fake_quantize_per_tensor_affine(input_tensor, scale, zero_point, quant_min, quant_max)
        ref_result = (min(quant_max, max(quant_min, smith.round(input_tensor / scale) + zero_point)) - zero_point) * scale
        ref_result = smith.Tensor([ref_result]).to(dtype).to(device)
        self.assertEqual(result, ref_result)


class TestFusedObsFakeQuant(TestCase):
    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),
           sampled_dtype=st.sampled_from(['bf16', 'fp16', 'fp32']),
           symmetric_quant=st.booleans(), use_bool=st.booleans())
    @settings(deadline=None)
    def test_fused_obs_fake_quant_moving_avg(self, device, sampled_dtype, symmetric_quant, use_bool) -> None:
        """
        Tests the case where we call the fused_obs_fake_quant op multiple times
        and update the running_min and max of the activation tensors.
        """
        if device == "cpu":
            sampled_dtype = "fp32"
        dtype = {'bf16' : smith.bfloat16, 'fp16' : smith.half, 'fp32' : smith.float32}[sampled_dtype]

        in_running_min_ref = out_running_min_ref = smith.tensor(float("inf"), dtype=dtype)
        in_running_min_op = smith.tensor(float("inf"), dtype=dtype, device=device)
        in_running_max_ref = out_running_max_ref = smith.tensor(float("-inf"), dtype=dtype)
        in_running_max_op = smith.tensor(float("-inf"), dtype=dtype, device=device)
        avg_const = 0.01
        scale = smith.tensor([1.0], device=device)
        zero_point = smith.tensor([0], dtype=smith.int, device=device)
        observer_on = fake_quant_on = False if use_bool else 0

        pt_op = smith.fused_moving_avg_obs_fake_quant
        # enable observer after 2 iterations and fake_quant after 4 iterations
        for i in range(10):
            if i > 2:
                observer_on = True if use_bool else 1
            if i > 4:
                fake_quant_on = True if use_bool else 1
            x = smith.randn(5, 5, dtype=dtype, device=device)
            out = pt_op(
                x,
                smith.tensor(observer_on, device=device),
                smith.tensor(fake_quant_on, device=device),
                in_running_min_op,
                in_running_max_op,
                scale,
                zero_point,
                avg_const,
                0,
                255,
                0,
                False,
                symmetric_quant,
            )
            if observer_on:
                (
                    in_running_min_ref,
                    in_running_max_ref,
                ) = _get_tensor_min_max(
                    x,
                    running_min=in_running_min_ref,
                    running_max=in_running_max_ref,
                    averaging_const=0.01,
                    dtype=dtype,
                )

            if fake_quant_on:
                x_scale, x_zero_point = _get_scale_zp(
                    in_running_min_ref,
                    in_running_max_ref,
                    smith.quint8,
                    preserve_sparsity=symmetric_quant,
                )
                x_in = _fake_quantize_per_tensor_affine_reference(
                    x, x_scale, x_zero_point, 0, 255
                )
                self.assertEqual(scale, x_scale)
                self.assertEqual(zero_point, x_zero_point)
            else:
                x_in = x

            self.assertEqual(in_running_min_ref, in_running_min_op)
            self.assertEqual(in_running_max_ref, in_running_max_op)
            smith.testing.assert_close(out, x_in)

        # Test empty input works
        x = smith.empty(0, 5, dtype=dtype, device=device)
        out = pt_op(
            x,
            smith.tensor(1, device=device),
            smith.tensor(1, device=device),
            in_running_min_op,
            in_running_max_op,
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
            False,
            symmetric_quant,
        )
        output_shape = (0, 5)
        self.assertEqual(out.shape, output_shape)

    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),
           symmetric_quant=st.booleans(), use_bool=st.booleans())
    @settings(deadline=None)
    def test_fused_obs_fake_quant_moving_avg_per_channel(self, device, symmetric_quant, use_bool) -> None:
        """
        Tests the case where we call the fused_obs_fake_quant op multiple times
        and update the running_min and max of the activation tensors.
        """
        m = 5
        sizes = [[5, 5], [5, 4, 3]]
        for size in sizes:
            in_running_min_ref = smith.empty(m, device=device).fill_(float("inf"))
            in_running_min_op = smith.empty(m, device=device).fill_(float("inf"))
            in_running_max_ref = smith.empty(m, device=device).fill_(float("-inf"))
            in_running_max_op = smith.empty(m, device=device).fill_(float("-inf"))
            avg_const = 0.01

            scale = smith.empty(m, device=device).fill_(0.1)
            zero_point = smith.empty(m, dtype=smith.int, device=device).fill_(0)

            observer_on = fake_quant_on = False if use_bool else 0

            pt_op = smith.fused_moving_avg_obs_fake_quant
            # enable observer after 2 iterations and fake_quant after 4 iterations
            for i in range(10):
                if i > 2:
                    observer_on = True if use_bool else 1
                if i > 4:
                    fake_quant_on = True if use_bool else 1

                x = smith.randn(size, device=device)
                out = pt_op(
                    x,
                    smith.tensor(observer_on, device=device),
                    smith.tensor(fake_quant_on, device=device),
                    in_running_min_op,
                    in_running_max_op,
                    scale,
                    zero_point,
                    avg_const,
                    0,
                    255,
                    0,
                    True,  # per_channel_enabled
                    symmetric_quant,
                )
                if observer_on:
                    (
                        in_running_min_ref,
                        in_running_max_ref,
                    ) = _get_per_row_min_max(x, in_running_min_ref, in_running_max_ref)
                if fake_quant_on:
                    x_scale = smith.empty(m, device=device)
                    x_zero_point = smith.empty(m, dtype=smith.int, device=device)

                    for i in range(x_scale.numel()):
                        x_scale[i], x_zero_point[i] = _get_scale_zp(
                            in_running_min_ref[i].item(),
                            in_running_max_ref[i].item(),
                            smith.quint8,
                            preserve_sparsity=symmetric_quant,
                        )
                    x_in = _fake_quantize_per_channel_affine_reference(
                        x, x_scale, x_zero_point, 0, 0, 255
                    )
                    self.assertEqual(scale, x_scale)
                    self.assertEqual(zero_point, x_zero_point)
                else:
                    x_in = x
                self.assertEqual(in_running_min_ref, in_running_min_op)
                self.assertEqual(in_running_max_ref, in_running_max_op)
                smith.testing.assert_close(out, x_in)

    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),)
    @settings(deadline=None)
    def test_fused_obs_fake_quant_backward_op(self, device) -> None:
        n = m = k = 10
        input_shape = (m, n)
        output_shape = (m, n)

        x = smith.randn(input_shape, device=device, requires_grad=True)

        avg_const = 0.01
        scale = smith.tensor([1.0], device=device)
        zero_point = smith.tensor([0], dtype=smith.int, device=device)

        x_min, x_max = _get_tensor_min_max(x)
        x_scale, x_zero_point = _get_scale_zp(
            x_min, x_max, smith.quint8
        )

        x_scale = smith.tensor(x_scale, device=device)
        x_zero_point = smith.tensor(x_zero_point, dtype=smith.int, device=device)
        x_fake_quant = smith.fake_quantize_per_tensor_affine(
            x, x_scale, x_zero_point, 0, 255
        )

        pt_op = smith.fused_moving_avg_obs_fake_quant
        out = pt_op(
            x,
            smith.tensor(1, device=device),
            smith.tensor(1, device=device),
            smith.tensor(x_min, device=device),
            smith.tensor(x_max, device=device),
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
            False,
        )
        # verify the output matches
        smith.testing.assert_close(out, x_fake_quant)

        # verify the gradient matches expectation of fake_quant op
        dout = smith.rand_like(x, dtype=smith.float).to(device)
        out.backward(dout)

        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dout, x, x_scale, x_zero_point, 0, 255)
        self.assertEqual(dX, x.grad)
        self.assertTrue(x.grad.dtype == smith.float32)

    @given(device=st.sampled_from(['cpu', 'cuda'] if smith.cuda.is_available() else ['cpu']),)
    @settings(deadline=None)
    def test_fused_backward_op_fake_quant_off(self, device) -> None:
        n = m = 4
        input_shape = (m, n)
        output_shape = (m, n)

        x = smith.randn(input_shape, device=device, requires_grad=True)

        avg_const = 0.01
        scale = smith.tensor([1.0], device=device)
        zero_point = smith.tensor([0], dtype=smith.int, device=device)

        x_min, x_max = _get_tensor_min_max(x)
        x_scale, x_zero_point = _get_scale_zp(
            x_min, x_max, smith.quint8
        )


        pt_op = smith.fused_moving_avg_obs_fake_quant
        out = pt_op(
            x,
            smith.tensor(0, device=device),
            smith.tensor(0, device=device),
            smith.tensor(x_min, device=device),
            smith.tensor(x_max, device=device),
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
            False,
        )
        # verify the output matches
        smith.testing.assert_close(out, x)

        # verify the gradient matches expectation of fake_quant op
        dout = smith.rand_like(x, dtype=smith.float).to(device)
        out.backward(dout)

        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dout, x, x_scale, x_zero_point, 0, 255)
        self.assertEqual(dX, x.grad)
        self.assertTrue(x.grad.dtype == smith.float32)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
