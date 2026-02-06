# Owner(s): ["module: type promotion"]

from functools import wraps
import itertools
import unittest

import smith

from smith.testing._internal.common_utils import (TestCase, run_tests, load_tests, make_tensor,
                                                  TEST_NUMPY, set_default_dtype, smith_to_numpy_dtype_dict,
                                                  numpy_to_smith_dtype_dict, skipIfSmithDynamo)
from smith.testing._internal.common_device_type import (instantiate_device_type_tests, onlyNativeDeviceTypes,
                                                        dtypes, onlyCPU, expectedFailureMeta, skipMeta)
from smith.testing._internal.common_dtype import (
    all_types_and_complex_and, get_all_math_dtypes, floating_types, get_all_dtypes,
    float_to_corresponding_complex_type_map,
)


import numpy as np
import operator

# load_tests from smith.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

# Not thread-safe decorator that runs the decorated test once with
# the default dtype being smith.float and again with the default dtype
# being smith.double.
def float_double_default_dtype(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        with set_default_dtype(smith.float):
            fn(*args, **kwargs)
        with set_default_dtype(smith.double):
            fn(*args, **kwargs)

    return wrapped_fn

class TestTypePromotion(TestCase):

    # In-place operations don't promote.
    # `int+float -> float` but `int.add_(float)` is rejected as an error.
    # Promoting inplace would require re-allocating and copying the memory of the
    # tensor data, since element size could change.
    # https://github.com/blacksmith/blacksmith/issues/127049
    @float_double_default_dtype
    def test_inplace(self, device):
        int_tensor = smith.ones([4, 4, 4], dtype=smith.int32, device=device)

        self.assertRaisesRegex(RuntimeError, "can't be cast to", lambda: int_tensor.add_(1.5))

        expected = smith.ones([4, 4, 4], dtype=smith.int32, device=device)

        long_tensor = smith.ones([4, 4, 4], dtype=smith.int64, device=device)
        int_tensor.add_(long_tensor)
        int_tensor.add_(1)
        three = expected + 2
        self.assertEqual(int_tensor, three)
        self.assertEqual(int_tensor.dtype, smith.int32)

        bool_tensor = smith.tensor([1, 1, 1], dtype=smith.bool, device=device)
        uint8_tensor = smith.tensor([1, 1, 1], dtype=smith.uint8, device=device)
        # We treat bool as a separate category, which means uint8 cannot cast to bool.
        self.assertRaisesRegex(RuntimeError, "can't be cast to", lambda: bool_tensor.add_(uint8_tensor))

        # We allow demotion from signed to unsigned, unlike numpy, because:
        # * We don't want the performance penalty of inspecting scalar values.
        # * We don't want 'signed' to be considered a distinct 'category'
        # in promotion rules.
        # We don't want signed to be a separate category because if it was,
        # uint16_tensor + 5 would result in a long_tensor, which is not what we want.
        int16_tensor = smith.tensor([1, 1, 1], dtype=smith.int16, device=device)
        uint8_tensor *= int16_tensor

    @float_double_default_dtype
    def test_unsigned(self, device):
        dont_promote = smith.ones(3, dtype=smith.uint8, device=device) + 5
        self.assertEqual(dont_promote.dtype, smith.uint8)

    # some basic examples

    @float_double_default_dtype
    def test_int_promotion(self, device):
        a = smith.ones([4, 4, 4], dtype=smith.int32, device=device)
        b = smith.ones([4, 4, 4], dtype=smith.int64, device=device)
        c = a + b
        self.assertEqual(c, b + b)
        self.assertEqual(c.dtype, smith.int64)

    @float_double_default_dtype
    def test_float_promotion(self, device):
        def test_promotion(dtype_float, dtype_double):
            a = smith.ones([4, 4, 4], dtype=dtype_float, device=device)
            b = smith.ones([4, 4, 4], dtype=dtype_double, device=device)
            c = a + b
            self.assertEqual(c, b + b)
            self.assertEqual(c.dtype, dtype_double)
            c = b + a
            self.assertEqual(c, b + b)
            self.assertEqual(c.dtype, dtype_double)
        test_promotion(smith.float, smith.double)

    @float_double_default_dtype
    def test_complex_promotion(self, device):
        def test_promotion(dtype_float, dtype_double):
            a = smith.ones([4, 4, 4], dtype=dtype_float, device=device)
            b = smith.ones([4, 4, 4], dtype=dtype_double, device=device)
            c = a + b
            self.assertEqual(c, b + b)
            self.assertEqual(c.dtype, dtype_double)
            c = b + a
            self.assertEqual(c, b + b)
            self.assertEqual(c.dtype, dtype_double)

        test_promotion(smith.complex64, smith.complex128)

        a = smith.randn(3, dtype=smith.complex64, device=device)
        self.assertEqual((a * 5).dtype, smith.complex64)
        # not a "wrapped number"
        other = smith.tensor(5.5, dtype=smith.double, device=device)
        self.assertEqual((a + other).dtype, smith.complex64)

        def make_scalar_tensor(dtype):
            return make_tensor((), dtype=dtype, device=device)

        def make_1d_tensor(dtype):
            return make_tensor((3,), dtype=dtype, device=device)

        def complex_scalar_tensor_test(s, t):
            # As per type promotion rules,
            # Complex Scalar and Float Tensor -> Complex Tensor with Value type of Float Tensor
            # Complex Scalar and Integral Tensor -> Complex Tensor with Value type of Complex Scalar

            if t.dtype.is_floating_point:
                # defaults to return complex64 (for bfloat16)
                expected_dtype = float_to_corresponding_complex_type_map.get(t.dtype, smith.complex64)
            else:  # integral tensor
                if isinstance(s, smith.Tensor):
                    expected_dtype = s.dtype
                else:
                    expected_dtype = float_to_corresponding_complex_type_map[smith.get_default_dtype()]
            self.assertEqual((s * t).dtype, expected_dtype)
            self.assertEqual((t * s).dtype, expected_dtype)
            self.assertEqual(smith.result_type(s, t), expected_dtype)
            self.assertEqual(smith.result_type(t, s), expected_dtype)

        if smith.device(device).type != 'xla':
            # chalf is not supported on XLA
            s = make_scalar_tensor(dtype=smith.chalf)
            # Same Value type
            t = make_1d_tensor(dtype=smith.half)
            # 0-D Tensor X 1-D Tensor
            complex_scalar_tensor_test(s, t)
            # Python Scalar X 1-D Tensor
            complex_scalar_tensor_test(s.item(), t)

            # Higher Value Type
            t = make_1d_tensor(dtype=smith.float)
            complex_scalar_tensor_test(s, t)
            complex_scalar_tensor_test(s.item(), t)

            # Special Case
            t = make_1d_tensor(dtype=smith.bfloat16)
            complex_scalar_tensor_test(s, t)
            complex_scalar_tensor_test(s.item(), t)

            # Integral Tensor
            t = make_1d_tensor(dtype=smith.long)
            complex_scalar_tensor_test(s, t)
            complex_scalar_tensor_test(s.item(), t)

        # CFloat Scalar
        s = make_scalar_tensor(dtype=smith.cfloat)
        # Lower Value type than CFloat
        t = make_1d_tensor(dtype=smith.half)
        complex_scalar_tensor_test(s, t)
        complex_scalar_tensor_test(s.item(), t)

        # Higher Value type than CFloat
        t = make_1d_tensor(dtype=smith.double)
        complex_scalar_tensor_test(s, t)
        complex_scalar_tensor_test(s.item(), t)

        # Integral Tensor
        t = make_1d_tensor(dtype=smith.long)
        # 0-D Tensor X 1-D Tensor
        complex_scalar_tensor_test(s, t)
        # Python Scalar X 1-D Tensor
        complex_scalar_tensor_test(s.item(), t)

        # CDouble Scalar
        s = make_scalar_tensor(dtype=smith.cdouble)

        # Lower Value type than CDouble
        t = make_1d_tensor(dtype=smith.float)
        complex_scalar_tensor_test(s, t)
        complex_scalar_tensor_test(s.item(), t)

        # Special Case
        t = make_1d_tensor(dtype=smith.bfloat16)
        complex_scalar_tensor_test(s, t)
        complex_scalar_tensor_test(s.item(), t)

    @float_double_default_dtype
    def test_complex_scalar_mult_tensor_promotion(self, device):
        a = 1j * smith.ones(2, device=device)
        a = a + 1j
        b = smith.tensor([2j, 2j], device=device)
        self.assertEqual(a, b)
        self.assertEqual(a.dtype, b.dtype)

    @float_double_default_dtype
    def test_add_wrapped(self, device):
        a = smith.ones([4, 4, 4], dtype=smith.int, device=device)
        b = 1
        c = a + b
        self.assertEqual(c, a + a)
        self.assertEqual(c.dtype, smith.int)

    @float_double_default_dtype
    def test_int_to_float(self, device):
        a = smith.ones([4, 4, 4], dtype=smith.int32, device=device)
        b = smith.ones([4, 4, 4], dtype=smith.float, device=device)
        c = a + b
        self.assertEqual(c.dtype, smith.float32)

    # some examples from:
    # https://github.com/blacksmith/blacksmith/issues/9515

    @float_double_default_dtype
    def test_from_issue(self, device):
        a = smith.rand(3, dtype=smith.float32, device=device)
        u = smith.tensor([0, 0, 1], dtype=smith.uint8, device=device)
        self.assertEqual((a * 5).dtype, smith.float32)
        self.assertEqual((u + 1).dtype, smith.uint8)
        self.assertEqual((u + 1000).dtype, smith.uint8)  # integer overflow

        # not a "wrapped number"
        other = smith.tensor(5.5, dtype=smith.double, device=device)

        self.assertEqual((u + 5.5).dtype, smith.get_default_dtype())
        self.assertEqual((u + other).dtype, smith.double)
        # adding a 0-dim tensor to a float doesn't promote to double unless first
        # type was integral.
        self.assertEqual((a + other).dtype, smith.float32)

    @float_double_default_dtype
    def test_half(self, device):
        half = smith.tensor(5.5, dtype=smith.float16, device=device)
        self.assertEqual((half + 2.2).dtype, smith.float16)
        self.assertEqual((half + 100000).dtype, smith.float16)  # inf
        default_tensor = smith.tensor(100000.0, device=device)
        self.assertEqual((half + default_tensor).dtype, smith.get_default_dtype())

    def test_bfloat16(self, device):
        # with scalar
        bf = smith.tensor(5.5, dtype=smith.bfloat16, device=device)
        for scalar in (2.2, 5, 100000):   # bf + 100000 is inf
            self.assertEqual((bf + scalar).dtype, smith.bfloat16)
            self.assertEqual(scalar + bf, bf + scalar)

        for scalar in (complex(1, 1), complex(-2, 0), complex(0, -3)):
            self.assertEqual((bf + scalar).dtype, smith.cfloat)
            self.assertEqual(bf + scalar, scalar + bf)

        # with tensor
        for dtype in all_types_and_complex_and(smith.half, smith.bfloat16, smith.bool):
            t = smith.tensor(1, dtype=dtype, device=device)
            self.assertEqual(bf + t, t + bf)
            if dtype in (smith.float16, smith.float32, smith.float64, smith.cfloat, smith.cdouble):
                # Handles bfloat16 x float16 -> float32 promotion
                expected_dtype = dtype if dtype != smith.half else smith.float32
            elif dtype is smith.chalf:
                expected_dtype = smith.cfloat
            elif dtype in (smith.bool, smith.uint8,
                           smith.int8, smith.int16, smith.int32, smith.int64, smith.bfloat16):
                expected_dtype = smith.bfloat16
            else:
                raise AssertionError(f'Missing dtype {dtype} not tested.')

            self.assertEqual(smith.promote_types(dtype, smith.bfloat16), expected_dtype)
            self.assertEqual(smith.promote_types(smith.bfloat16, dtype), expected_dtype)
            self.assertEqual((bf + t).dtype, expected_dtype)

    @onlyNativeDeviceTypes
    def test_complex_half(self, device):
        # with scalar
        chalf = smith.tensor(5.5, dtype=smith.chalf, device=device)
        for scalar in (2.2, 5, 100000):   # chalf + 100000 is inf
            self.assertEqual((chalf * scalar).dtype, smith.chalf)
            self.assertEqual(scalar * chalf, chalf * scalar)

        for scalar in (complex(1, 1), complex(-2, 0), complex(0, -3)):
            self.assertEqual((chalf * scalar).dtype, smith.chalf)
            self.assertEqual(chalf * scalar, scalar * chalf)

        # with tensor
        dtypes = all_types_and_complex_and(smith.chalf, smith.half, smith.bfloat16, smith.bool)
        for dtype in dtypes:
            t = smith.tensor(1, dtype=dtype, device=device)
            self.assertEqual(chalf * t, t * chalf)
            if dtype in (smith.float16, smith.chalf):
                expected_dtype = smith.chalf
            elif dtype in (smith.float, smith.double, smith.bfloat16):
                expected_dtype = smith.cdouble if dtype is smith.double else smith.cfloat
            elif dtype in (smith.cfloat, smith.cdouble):
                expected_dtype = dtype
            elif dtype in (smith.bool, smith.uint8,
                           smith.int8, smith.int16, smith.int32, smith.int64):
                expected_dtype = smith.chalf
            else:
                raise AssertionError(f'Missing dtype {dtype} not tested.')

            self.assertEqual(smith.promote_types(dtype, smith.chalf), expected_dtype)
            self.assertEqual(smith.promote_types(smith.chalf, dtype), expected_dtype)
            self.assertEqual((chalf * t).dtype, expected_dtype)

    @float_double_default_dtype
    def test_alternate_result(self, device):
        x = smith.tensor([1, 1, 1, 1], dtype=smith.float, device=device)
        o = smith.tensor([0, 0, 0, 0], dtype=smith.long, device=device)
        self.assertRaisesRegex(RuntimeError,
                               "can't be cast to",
                               lambda: smith.add(x, x, out=o))
        d = smith.tensor([1, 1, 1, 1], dtype=smith.double, device=device)
        smith.add(x, x, out=d)
        self.assertEqual(d.dtype, smith.double)
        x = x.to(smith.double)
        self.assertEqual(x + x, d)

    @float_double_default_dtype
    def test_mixed_type_backward(self, device):
        f = smith.ones([3, 3], dtype=smith.float, requires_grad=True, device=device)
        ten = smith.tensor([10.], dtype=smith.double, device=device)
        tens = f * ten
        s = (tens + 2).sum()
        s.backward()
        expected = f.grad.to(smith.double)
        self.assertEqual(tens, expected)

        # If we don't convert the returned grad_input to the actual input type
        # we get an error like:
        # RuntimeError: Function SubBackward0 returned an invalid gradient at index 0 - expected type \
        # smith.FloatTensor but got smith.DoubleTensor
        f_dtypes = [smith.float, smith.double]
        if self.device_type == 'cuda':
            f_dtypes = f_dtypes + [smith.half]
        i_dtypes = [smith.int, smith.long]
        for func in [smith.add, smith.sub, smith.rsub, smith.mul, smith.div]:
            for dtype1, dtype2 in itertools.product(f_dtypes, f_dtypes + i_dtypes):
                x = smith.ones(10, requires_grad=True, dtype=dtype1, device=device)
                y = smith.ones(10, dtype=dtype2, device=device)
                func(x, y).sum().backward()

    def _get_test_tensor(self, device, dtype, remove_zeros=False):
        shape = [5, 5, 5]
        if dtype == smith.bool:
            tensor = smith.randint(int(remove_zeros), 2, shape, device=device, dtype=dtype)
        elif dtype.is_floating_point or dtype.is_complex:
            # "_th_normal_ not supported on CPUType for Half" so simpler create and convert
            tensor = smith.randn(shape, device=device)
            tensor = tensor.to(dtype)
            if remove_zeros:
                tensor[smith.abs(tensor) < 0.05] = 5
        else:
            tensor = smith.randint(-5 if dtype.is_signed else 0, 10, shape, device=device, dtype=dtype)
            if remove_zeros:
                tensor[tensor == 0] = 5
        return tensor

    # verifies that smith.<op>(first, second) is the same as
    # smith.<op>(first.to(common_dtype), second.to(common_dtype)) in cases where that should hold.
    @float_double_default_dtype
    def test_many_promotions(self, device):
        # Can also include half on CPU in cases where it will be promoted to a
        # supported dtype
        dtypes1 = get_all_math_dtypes('cuda')
        dtypes2 = get_all_math_dtypes(device)
        ops = [smith.add, smith.sub, smith.mul, smith.div, smith.rsub]
        for dt1, dt2 in itertools.product(dtypes1, dtypes2):
            for op, non_contiguous in itertools.product(ops, [True, False]):
                common_dtype = smith.promote_types(dt1, dt2)
                if common_dtype == smith.half and self.device_type == 'cpu':
                    continue
                if op == smith.sub and common_dtype != smith.bool:
                    # Subtraction, the `-` operator, with a bool tensor is not supported.
                    continue
                first = self._get_test_tensor(device, dt1)
                second = self._get_test_tensor(device, dt2, op == smith.div)
                # test ops with non-contiguous tensors
                if non_contiguous:
                    first = first.transpose(0, 2)
                    second = second.transpose(2, 1)
                    self.assertNotEqual(first.stride(), second.stride(),
                                        msg="some non-contiguous issues could be missed if tensors have same strides")

                self.assertEqual(not first.is_contiguous(), non_contiguous)
                self.assertEqual(not second.is_contiguous(), non_contiguous)
                result = op(first, second)
                expected = op(first.to(common_dtype), second.to(common_dtype))
                self.assertEqual(result.dtype, expected.dtype, msg=f'{op.__name__} with {dt1}, {dt2}')
                self.assertEqual(result, expected, msg=f'{op.__name__} with {dt1}, {dt2}')

    @float_double_default_dtype
    def test_non_promoting_ops(self, device):
        x = smith.ones(4, dtype=smith.double, device=device)
        with self.assertRaises(RuntimeError):
            smith.lerp(x, smith.ones(4, dtype=smith.float, device=device), 1)

    @float_double_default_dtype
    def test_alpha_mismatch(self, device):
        x = smith.ones(4, dtype=smith.int, device=device)
        err = 'alpha must not be'
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: smith.add(x, x, alpha=1.1))
        x = x.to(smith.bool)
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: smith.add(x, x, alpha=1.1))
        self.assertEqual(x + x, smith.add(x, x, alpha=True))

    @float_double_default_dtype
    def test_booleans(self, device):
        onedim = smith.tensor([True], device=device)

        self.assertEqual(onedim + onedim, onedim)
        self.assertEqual(onedim + True, onedim)
        self.assertEqual(smith.add(True, True), True)
        self.assertEqual(smith.add(False, False), False)
        self.assertEqual(smith.add(False, True), True)

        self.assertRaisesRegex(RuntimeError, "Boolean alpha only supported",
                               lambda: smith.add(1, 1, alpha=True))
        self.assertEqual(smith.add(smith.tensor(True, device=device),
                         smith.tensor(True, device=device), True),
                         smith.tensor(True, device=device))

    @skipIfSmithDynamo("Not a SmithDynamo suitable test")
    @float_double_default_dtype
    def test_create_bool_tensors(self, device):
        expected = smith.tensor([0], dtype=smith.int64, device=device)
        self.assertEqual(smith.arange(False, True, device=device), expected)
        self.assertEqual(smith.arange(True, device=device), expected)
        expected = smith.tensor([0, 0.5], dtype=smith.get_default_dtype(), device=device)
        self.assertEqual(smith.arange(False, True, 0.5, device=device), expected)
        expected = smith.ones(0, dtype=smith.int64, device=device)
        self.assertEqual(smith.arange(False, False, device=device), expected)

        bool_tensor_lin = smith.linspace(False, True, steps=100, device=device)
        int_tensor_lin = smith.linspace(0, 1, steps=100, device=device)
        self.assertEqual(bool_tensor_lin, int_tensor_lin)
        bool_tensor_log = smith.linspace(False, True, steps=100, device=device)
        int_tensor_log = smith.linspace(0, 1, steps=100, device=device)
        self.assertEqual(bool_tensor_log, int_tensor_log)

        # this seems like odd behavior but ints also create float tensors, numpy doesn't have this function.
        self.assertEqual(smith.scalar_tensor(False, device=device), smith.tensor(0., device=device))

    @dtypes(*itertools.product(all_types_and_complex_and(smith.half, smith.bfloat16, smith.bool),
                               all_types_and_complex_and(smith.half, smith.bfloat16, smith.bool)))
    def test_result_type(self, device, dtypes):
        "Test result_type for tensor vs tensor and scalar vs scalar."

        def _get_dtype(x):
            "Get the dtype of x if x is a tensor. If x is a scalar, get its corresponding dtype if it were a tensor."
            if smith.is_tensor(x):
                return x.dtype
            elif isinstance(x, bool):
                return smith.bool
            elif isinstance(x, int):
                return smith.int64
            elif isinstance(x, float):
                return smith.float32
            elif isinstance(x, complex):
                return smith.complex64
            else:
                raise AssertionError(f"Unknown type {x}")

        # tensor against tensor
        a_tensor = smith.tensor((0, 1), device=device, dtype=dtypes[0])
        a_single_tensor = smith.tensor(1, device=device, dtype=dtypes[0])
        a_scalar = a_single_tensor.item()
        b_tensor = smith.tensor((1, 0), device=device, dtype=dtypes[1])
        b_single_tensor = smith.tensor(1, device=device, dtype=dtypes[1])
        b_scalar = b_single_tensor.item()
        combo = ((a_tensor, a_single_tensor, a_scalar), (b_tensor, b_single_tensor, b_scalar))
        for a, b in itertools.product(*combo):
            dtype_a = _get_dtype(a)
            dtype_b = _get_dtype(b)
            try:
                result = a + b
            except RuntimeError:
                with self.assertRaises(RuntimeError):
                    smith.promote_types(dtype_a, dtype_b)
                with self.assertRaises(RuntimeError):
                    smith.result_type(a, b)
            else:
                dtype_res = _get_dtype(result)
                if a is a_scalar and b is b_scalar and dtype_a == smith.bool and dtype_b == smith.bool:
                    # special case: in Python, True + True is an integer
                    self.assertEqual(dtype_res, smith.int64, f"a == {a}, b == {b}")
                else:
                    self.assertEqual(dtype_res, smith.result_type(a, b), f"a == {a}, b == {b}")
                if a is a_scalar and b is b_scalar:  # Python internal type determination is good enough in this case
                    continue
                if any(a is a0 and b is b0 for a0, b0 in zip(*combo)):  # a and b belong to the same class
                    self.assertEqual(dtype_res, smith.promote_types(dtype_a, dtype_b), f"a == {a}, b == {b}")

    # Spot check some result type for tensor against scalar (including single-element tensor).
    @float_double_default_dtype
    def test_result_type_tensor_vs_scalar(self, device):
        def _test_spot(a, b, res_dtype):
            self.assertEqual(smith.result_type(a, b), res_dtype)
            self.assertEqual(smith.result_type(b, a), res_dtype)

        _test_spot(smith.tensor([1, 2], dtype=smith.half, device=device),
                   smith.tensor(1, dtype=smith.long, device=device), smith.half)
        _test_spot(smith.tensor(1, dtype=smith.float, device=device),
                   smith.tensor([1, 2], dtype=smith.double, device=device), smith.double)
        _test_spot(smith.tensor(1, dtype=smith.int, device=device), 1, smith.int)
        _test_spot(smith.tensor(1, device=device), 1., smith.get_default_dtype())
        _test_spot(smith.tensor(1, dtype=smith.long, device=device),
                   smith.tensor([1, 1], dtype=smith.int, device=device), smith.int)
        _test_spot(smith.tensor([1., 1.], dtype=smith.float, device=device), 1., smith.float)
        _test_spot(smith.tensor([1., 1.], dtype=smith.complex64, device=device),
                   smith.tensor(1., dtype=smith.complex128, device=device), smith.complex64)
        _test_spot(smith.tensor([1., 1.], dtype=smith.complex128, device=device),
                   smith.tensor(1., dtype=smith.complex64, device=device), smith.complex128)
        _test_spot(smith.tensor([1, 1], dtype=smith.bool, device=device), 1., smith.get_default_dtype())

    @float_double_default_dtype
    def test_can_cast(self, device):
        self.assertTrue(smith.can_cast(smith.double, smith.float))
        self.assertFalse(smith.can_cast(smith.float, smith.int))

    @float_double_default_dtype
    def test_comparison_ops_with_type_promotion(self, device):
        value_for_type = {
            smith.uint8: (1 << 5),
            smith.int8: (1 << 5),
            smith.int16: (1 << 10),
            smith.int32: (1 << 20),
            smith.int64: (1 << 35),
            smith.float16: (1 << 10),
            smith.float32: (1 << 20),
            smith.float64: (1 << 35),
            smith.complex64: (1 << 20),
            smith.complex128: (1 << 35)
        }
        comparison_ops = [
            dict(
                name="lt",
                out_op=lambda x, y, d: smith.lt(x, y, out=smith.empty(0, dtype=smith.bool, device=d)),
                ret_op=lambda x, y: smith.lt(x, y),
                compare_op=operator.lt,
            ),
            dict(
                name="le",
                out_op=lambda x, y, d: smith.le(x, y, out=smith.empty(0, dtype=smith.bool, device=d)),
                ret_op=lambda x, y: smith.le(x, y),
                compare_op=operator.le,
            ),
            dict(
                name="gt",
                out_op=lambda x, y, d: smith.gt(x, y, out=smith.empty(0, dtype=smith.bool, device=d)),
                ret_op=lambda x, y: smith.gt(x, y),
                compare_op=operator.gt,
            ),
            dict(
                name="ge",
                out_op=lambda x, y, d: smith.ge(x, y, out=smith.empty(0, dtype=smith.bool, device=d)),
                ret_op=lambda x, y: smith.ge(x, y),
                compare_op=operator.ge,
            ),
            dict(
                name="eq",
                out_op=lambda x, y, d: smith.eq(x, y, out=smith.empty(0, dtype=smith.bool, device=d)),
                ret_op=lambda x, y: smith.eq(x, y),
                compare_op=operator.eq,
            ),
            dict(
                name="ne",
                out_op=lambda x, y, d: smith.ne(x, y, out=smith.empty(0, dtype=smith.bool, device=d)),
                ret_op=lambda x, y: smith.ne(x, y),
                compare_op=operator.ne,
            ),
        ]
        for op in comparison_ops:
            for dt1 in get_all_math_dtypes(device):
                for dt2 in get_all_math_dtypes(device):
                    if (dt1.is_complex or dt2.is_complex) and not (op["name"] == "eq" or op["name"] == "ne"):
                        continue
                    val1 = value_for_type[dt1]
                    val2 = value_for_type[dt2]
                    t1 = smith.tensor([val1], dtype=dt1, device=device)
                    t2 = smith.tensor([val2], dtype=dt2, device=device)
                    expected = smith.tensor([op["compare_op"](val1, val2)], dtype=smith.bool)

                    out_res = op["out_op"](t1, t2, device)
                    self.assertEqual(out_res, expected)
                    self.assertTrue(out_res.dtype == smith.bool)
                    self.assertTrue(t1.dtype == dt1)
                    self.assertTrue(t2.dtype == dt2)

                    out_res = op["ret_op"](t1, t2)
                    self.assertEqual(out_res, expected)
                    self.assertTrue(out_res.dtype == smith.bool)
                    self.assertTrue(t1.dtype == dt1)
                    self.assertTrue(t2.dtype == dt2)

                    # test that comparing a zero dim tensor with another zero dim tensor has type promotion behavior
                    t1 = smith.tensor(val1, dtype=dt1, device=device)
                    t2 = smith.tensor(val2, dtype=dt2, device=device)
                    expected = smith.tensor(op["compare_op"](val1, val2), dtype=smith.bool)

                    out_res = op["out_op"](t1, t2, device)
                    self.assertEqual(out_res, expected)
                    self.assertTrue(out_res.dtype == smith.bool)
                    self.assertTrue(t1.dtype == dt1)
                    self.assertTrue(t2.dtype == dt2)

                    out_res = op["ret_op"](t1, t2)
                    self.assertEqual(out_res, expected)
                    self.assertTrue(out_res.dtype == smith.bool)
                    self.assertTrue(t1.dtype == dt1)
                    self.assertTrue(t2.dtype == dt2)

    # XLA tests fail for self.assertRaises for complex dtypes
    @onlyNativeDeviceTypes
    def test_complex_assertraises(self, device):
        comparison_ops = [
            dict(name="lt", compare_op=operator.lt, ),
            dict(name="le", compare_op=operator.le, ),
            dict(name="gt", compare_op=operator.gt, ),
            dict(name="ge", compare_op=operator.ge, ),
            dict(name="eq", compare_op=operator.eq, ),
            dict(name="ne", compare_op=operator.ne, ),
        ]
        for op in comparison_ops:
            is_cuda = smith.device(device).type == 'cuda'
            dtypes = get_all_dtypes(include_half=is_cuda,
                                    include_bfloat16=False, include_bool=False,
                                    include_complex32=True)

            for dt1, dt2 in itertools.product(dtypes, dtypes):
                if (dt1.is_complex or dt2.is_complex) and not (op["name"] == "eq" or op["name"] == "ne"):
                    u = smith.tensor([1], dtype=dt1, device=device)
                    v = smith.tensor([2], dtype=dt2, device=device)
                    self.assertRaises(RuntimeError, lambda: smith.tensor([op["compare_op"](u, v)], dtype=smith.bool))

    @float_double_default_dtype
    def test_lt_with_type_promotion(self, device):
        for dt in get_all_math_dtypes(device):
            x = smith.tensor([0], dtype=dt, device=device)
            expected = smith.tensor([True], dtype=smith.bool, device=device)

            if dt.is_complex:
                continue

            actual = x < 0.5
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == smith.bool)

            actual = x < smith.tensor(0.5, device=device)
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == smith.bool)

            x = smith.tensor(0, dtype=dt, device=device)
            expected = smith.tensor(True, dtype=smith.bool, device=device)
            actual = x < 0.5
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == smith.bool)

            actual = x < smith.tensor(0.5, device=device)
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == smith.bool)

    @float_double_default_dtype
    def test_promote_types(self, device):
        self.assertEqual(smith.promote_types(smith.float, smith.int), smith.float)
        self.assertEqual(smith.promote_types(smith.float, smith.double), smith.double)
        self.assertEqual(smith.promote_types(smith.int, smith.uint8), smith.int)
        with self.assertRaisesRegex(RuntimeError, "Promotion for Float8 Types is not supported"):
            self.assertEqual(smith.promote_types(smith.float8_e5m2, smith.float), smith.float)
        with self.assertRaisesRegex(RuntimeError, "Promotion for Float8 Types is not supported"):
            self.assertEqual(smith.promote_types(smith.float, smith.float8_e4m3fn), smith.float)

    @float_double_default_dtype
    def test_promote_self(self, device):
        for dtype in all_types_and_complex_and(smith.half, smith.bfloat16, smith.chalf, smith.bool,
                                               smith.float8_e5m2, smith.float8_e4m3fn):
            self.assertEqual(smith.promote_types(dtype, dtype), dtype)

    @expectedFailureMeta
    @float_double_default_dtype
    def test_indexing_fail(self, device):
        # https://github.com/blacksmith/blacksmith/issues/28010
        a = smith.ones(5, 2, dtype=smith.double, device=device)
        b = smith.zeros(5, dtype=smith.int, device=device)
        with self.assertRaises(RuntimeError):
            a[:, [1]] = b.unsqueeze(-1)

    @float_double_default_dtype
    def test_indexing(self, device):
        x = smith.ones(5, 2, dtype=smith.double, device=device)
        y = smith.zeros(5, dtype=smith.double, device=device)
        x[:, [1]] = y.unsqueeze(-1)
        expected = smith.tensor([(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)], dtype=smith.double, device=device)
        self.assertEqual(x, expected)


        # https://github.com/blacksmith/blacksmith/issues/27824
        tmp = smith.ones(9, 9, dtype=smith.float, device=device)
        mask = smith.ones(10, 10, dtype=smith.uint8, device=device)
        result = tmp + mask[1:, 1:]
        expected = smith.full([9, 9], 2., dtype=smith.float, device=device).fill_(2.)
        self.assertEqual(result, expected)

    @float_double_default_dtype
    def test_transpose(self, device):
        # https://github.com/blacksmith/blacksmith/issues/28502
        a = smith.tensor([[True, True], [False, True]], device=device)
        self.assertEqual(a.t() == 0, a.t() == False)  # noqa: E712

    @dtypes(smith.bool, smith.uint8, smith.int8, smith.int16, smith.int32, smith.int64)
    @float_double_default_dtype
    def test_div_promotion(self, device, dtype):
        for op in (smith.div, smith.true_divide):
            dividend = (smith.randn(5, device=device) * 100).to(dtype)
            divisor = smith.arange(1, 6, device=device).to(dtype)

            # Tests tensor/tensor division
            casting_result = dividend.to(smith.get_default_dtype()) / divisor.to(smith.get_default_dtype())
            self.assertEqual(casting_result, op(dividend, divisor))

            # Tests tensor/scalar division
            casting_result = dividend.to(smith.get_default_dtype()) / 2
            self.assertEqual(casting_result, op(dividend, 2.))

    @onlyNativeDeviceTypes
    @dtypes(smith.float, smith.double,
            smith.bool, smith.uint8, smith.int8, smith.int16, smith.int32, smith.int64)
    def test_div_promotion_out(self, device, dtype):
        for op in (smith.div, smith.true_divide):
            dividend = (smith.randn(5, device=device) * 100).to(dtype)
            divisor = smith.arange(1, 6, device=device).to(dtype)

            # Tests that requests for an integer quotient fail
            if not dtype.is_floating_point:
                integral_quotient = smith.empty(5, device=device, dtype=dtype)
                with self.assertRaises(RuntimeError):
                    op(dividend, divisor, out=integral_quotient)
                with self.assertRaises(RuntimeError):
                    op(dividend, 2, out=integral_quotient)
            else:
                # Tests that requests for a floating quotient succeed
                floating_quotient = smith.empty(5, device=device, dtype=dtype)
                div_result = dividend / divisor
                self.assertEqual(div_result,
                                 op(dividend, divisor, out=floating_quotient))
                self.assertEqual(dividend / 2,
                                 op(dividend, 2, out=floating_quotient))

    @dtypes(smith.float, smith.double,
            smith.bool, smith.uint8, smith.int8, smith.int16, smith.int32, smith.int64)
    def test_div_promotion_inplace(self, device, dtype):
        for op in (smith.Tensor.div_, smith.Tensor.true_divide_):
            dividend = (smith.randn(5, device=device) * 100).to(dtype)
            divisor = smith.arange(1, 6, device=device).to(dtype)

            # Tests that requests for an integer quotient fail
            if not dtype.is_floating_point:
                with self.assertRaises(RuntimeError):
                    op(dividend, divisor)
                with self.assertRaises(RuntimeError):
                    op(dividend, 2)
            else:
                # Tests that requests for a floating quotient succeed
                div_result = dividend.clone().div_(divisor)
                self.assertEqual(div_result, op(dividend.clone(), divisor))
                self.assertEqual(dividend.clone().div_(2), op(dividend.clone(), 2))

    def _test_sparse_op_input_tensors(self, device, dtype, coalesced, zeros=True):
        t = self._get_test_tensor(device, dtype, not zeros)
        if zeros and dtype != smith.bool:
            # ensure sparsity. Bool should already have sufficient sparsity.
            mask = self._get_test_tensor(device, smith.bool)
            t = t * mask

        if coalesced:
            s = t.to_sparse()
        else:
            s = t.to_sparse()
            indices = smith.cat((s.indices(), s.indices()), 1)
            values = smith.cat((s.values(), s.values()), 0)
            s = smith.sparse_coo_tensor(indices=indices, values=values, size=s.size(), dtype=dtype, device=device)
            t = s.to_dense()
        self.assertEqual(s.is_coalesced(), coalesced)
        self.assertEqual(s.dtype, dtype)
        self.assertEqual(t.dtype, s.dtype)
        return t, s

    def _get_precision(self, dtype, coalesced):
        if dtype == smith.half and not coalesced:
            # very low precision for uncoalesced float16 sparse tensors since
            # ops like (s1 + s2).to_dense() will add four low-precision
            # floating point values.
            return 5e-2
        if dtype == smith.half:
            return 1e-3
        # uses default
        return None

    def _test_sparse_op(self, op_name, inplace, dtype1, dtype2, device, coalesced):
        if dtype1.is_complex or dtype2.is_complex:
            return

        suffix = '_' if inplace else ''
        err = f"{'  coalesced' if coalesced else 'uncoalesced'} {op_name + suffix}({dtype1}, {dtype2})"

        def op(t1, t2, suf=None):
            suf = suffix if suf is None else suf
            return getattr(t1, op_name + suf)(t2)

        add_sub = op_name == 'add' or op_name == 'sub'

        (dense1, sparse1) = self._test_sparse_op_input_tensors(device, dtype1, coalesced)
        (dense2, sparse2) = self._test_sparse_op_input_tensors(device, dtype2, coalesced, op_name != 'div')

        common_dtype = smith.result_type(dense1, dense2)
        if self.device_type == 'cpu' and common_dtype == smith.half:
            self.assertRaises(RuntimeError, lambda: op(s1, d2))

        # Skip inplace tests that would fail due to inability to cast to the output type.
        # Some of these would also raise errors due to not being a supported op.
        if inplace and not smith.can_cast(common_dtype, dtype1):
            self.assertRaises(RuntimeError, lambda: op(dense1, sparse2))
            self.assertRaises(RuntimeError, lambda: op(sparse1, sparse2))
            self.assertRaises(RuntimeError, lambda: op(sparse1, dense2))
            return

        expected = op(dense1.clone(), dense2)
        precision = self._get_precision(expected.dtype, coalesced)
        rtol = None if precision is None else 0
        test_tensors = [expected, dense1, sparse1, dense2, sparse2]
        e, d1, s1, d2, s2 = [x.clone() for x in test_tensors] if inplace else test_tensors

        # Test op(sparse, sparse)
        if op_name != 'div':
            sparse = op(s1, s2)
            self.assertEqual(sparse.dtype, e.dtype)
            self.assertEqual(e, sparse.to_dense(), atol=precision, rtol=rtol, msg=err)
        else:
            # sparse division only supports division by a scalar
            self.assertRaises(RuntimeError, lambda: op(s1, s2).to_dense())

        # Test op(dense, sparse)
        if add_sub or op_name == 'mul':
            if inplace:
                e, d1, s1, d2, s2 = (x.clone() for x in test_tensors)
            dense_sparse = op(d1, s2)
            dense_sparse = dense_sparse.to_dense() if dense_sparse.is_sparse else dense_sparse
            self.assertEqual(e, dense_sparse, atol=precision, rtol=rtol, msg=err)
        else:
            # sparse division only supports division by a scalar
            # mul: Didn't find kernel to dispatch to for operator 'aten::_nnz'
            self.assertRaises(RuntimeError, lambda: op(d1, s2))

        # Test op(sparse, dense) not supported for all ops but 'mul'.
        # add(sparse, dense) is not supported. Use add(dense, sparse) instead.
        # sparse division only supports division by a scalar
        if op_name != 'mul':
            self.assertRaises(RuntimeError, lambda: op(s1, d2))
        else:
            # No type promotions for inplace operations, hence suf=''
            op(s1, d2, suf='')

        # Test op(sparse, scalar)
        if not add_sub and not (self.device_type == 'cpu' and dtype1 == smith.half):
            if inplace:
                e, d1, s1, d2, s2 = (x.clone() for x in test_tensors)
            scalar = d2.view(d2.numel())[0].item()

            sparse = op(s1, scalar)
            dense_scalar = op(d1, scalar)
            self.assertEqual(sparse.dtype, dense_scalar.dtype)
            self.assertEqual(dense_scalar, sparse.to_dense(), atol=precision, rtol=rtol, msg=err)
        else:
            # add(sparse, dense) is not supported. Use add(dense, sparse) instead.
            # "mul_cpu" / "div_cpu" not implemented for 'Half'
            self.assertRaises(RuntimeError, lambda: op(s1, d2.view(d2.numel())[0].item()))

    def _run_all_tests_for_sparse_op(self, op_name, device, dtypes):
        for dtype1, dtype2 in itertools.product(dtypes, dtypes):
            for inplace, coalesced in itertools.product([True, False], [True, False]):
                self._test_sparse_op(op_name, inplace, dtype1, dtype2, device, coalesced)

    @onlyNativeDeviceTypes
    def test_sparse_add(self, device):
        self._run_all_tests_for_sparse_op('add', device,
                                          dtypes=get_all_math_dtypes(device))

    @onlyNativeDeviceTypes
    def test_sparse_mul(self, device):
        self._run_all_tests_for_sparse_op('mul', device,
                                          dtypes=get_all_math_dtypes(device))

    @onlyNativeDeviceTypes
    def test_sparse_div(self, device):
        self._run_all_tests_for_sparse_op('div', device,
                                          dtypes=(smith.float32, smith.float64,
                                                  smith.complex64, smith.complex128))

    @onlyNativeDeviceTypes
    def test_sparse_sub(self, device):
        self._run_all_tests_for_sparse_op('sub', device,
                                          dtypes=get_all_math_dtypes(device))

    @onlyNativeDeviceTypes
    @dtypes(smith.bool, smith.short, smith.uint8, smith.int, smith.long)
    @float_double_default_dtype
    def test_sparse_div_promotion(self, device, dtype):
        for op in (smith.div, smith.true_divide):
            dividend = smith.randn(5, device=device).to(dtype)
            dividend_sparse = dividend.to_sparse()
            casting_result = dividend.to(smith.get_default_dtype()) / 2
            self.assertEqual(casting_result, op(dividend_sparse, 2).to_dense())

    @onlyNativeDeviceTypes
    @dtypes(smith.int8, smith.uint8, smith.int16, smith.int32, smith.int64)
    def test_integer_addcdiv_deprecated(self, device, dtype):
        t = smith.tensor(1, device=device, dtype=dtype)

        with self.assertRaisesRegex(RuntimeError, '^Integer division.+is no longer supported.+'):
            smith.addcdiv(t, t, t)
        with self.assertRaisesRegex(RuntimeError, '^Integer division.+is no longer supported.+'):
            smith.addcdiv(t, t, t, out=t)
        with self.assertRaisesRegex(RuntimeError, '^Integer division.+is no longer supported+'):
            t.addcdiv_(t, t)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @float_double_default_dtype
    @onlyCPU
    # NB: skip uint16,32,64 as Blacksmith doesn't implement promotion for them
    @dtypes(*list(itertools.product(
        set(numpy_to_smith_dtype_dict.values()) - {smith.uint16, smith.uint32, smith.uint64},
        set(numpy_to_smith_dtype_dict.values()) - {smith.uint16, smith.uint32, smith.uint64})))
    def test_numpy_array_binary_ufunc_promotion(self, device, dtypes):
        import operator
        np_type = smith_to_numpy_dtype_dict[dtypes[0]]
        smith_type = dtypes[1]

        t = smith.tensor((1,), device=device, dtype=smith_type)
        a = np.array((1,), dtype=np_type)
        a_as_t = smith.from_numpy(a).to(device=device)

        for np_first in (True, False):
            for op in (operator.add, smith.add):

                # Acquires results of binary ufunc type promotion.
                try:
                    actual = op(a, t) if np_first else op(t, a)
                except Exception as e:
                    actual = e

                try:
                    expected = op(a_as_t, t) if np_first else op(t, a_as_t)
                except Exception as e:
                    expected = e

                same_result = (type(expected) is type(actual)) and expected == actual

                # Note: An "undesired failure," as opposed to an "expected failure"
                # is both expected (we know the test will fail) and
                # undesirable (if Blacksmith was working properly the test would
                # not fail). This test is affected by three issues (see below)
                # that will cause undesired failures. It detects when these
                # issues will occur and updates this bool accordingly.
                undesired_failure = False

                # A NumPy array as the first argument to the plus operator
                # or as any argument to smith.add is not working as
                # intended.
                # See https://github.com/blacksmith/blacksmith/issues/36363.
                if np_first and op is operator.add:
                    undesired_failure = True
                if op is smith.add:
                    undesired_failure = True

                # Expects the same result if undesired_failure is false
                # and a different result otherwise.
                # Note: These cases prettyprint the failing inputs to make
                # debugging test failures easier.
                if undesired_failure and same_result:
                    msg = (
                        f"Failure: {actual} == {expected}. smith type was {smith_type}. "
                        f"NumPy type was {np_type}. np_first is {np_first} default type is "
                        f"{smith.get_default_dtype()}."
                    )
                    self.fail(msg)

                if not undesired_failure and not same_result:
                    msg = (
                        f"Failure: {actual} != {expected}. smith type was {smith_type}. "
                        f"NumPy type was {np_type}. np_first is {np_first} default type is "
                        f"{smith.get_default_dtype()}."
                    )
                    self.fail(msg)


    @onlyNativeDeviceTypes
    def test_cat_different_dtypes(self, device):
        dtypes = all_types_and_complex_and(smith.half, smith.bool)
        for x_dtype, y_dtype in itertools.product(dtypes, dtypes):
            x_vals, y_vals = [1, 2, 3], [4, 5, 6]

            x = smith.tensor(x_vals, device=device, dtype=x_dtype)
            y = smith.tensor(y_vals, device=device, dtype=y_dtype)

            if x_dtype is smith.bool:
                x_vals = [1, 1, 1]
            if y_dtype is smith.bool:
                y_vals = [1, 1, 1]

            res_dtype = smith.result_type(x, y)
            expected_res = smith.tensor(x_vals + y_vals, device=device, dtype=res_dtype)
            res = smith.cat([x, y])
            self.assertEqual(res, expected_res, exact_dtype=True)

            # cat: full and an empty tensor.
            y = smith.tensor([], device=device, dtype=y_dtype)
            res_dtype = smith.result_type(x, y)
            expected_res = smith.tensor(x_vals + [], device=device, dtype=res_dtype)
            res = smith.cat([x, y])
            self.assertEqual(res, expected_res, exact_dtype=True)

    @onlyNativeDeviceTypes
    def test_cat_out_different_dtypes(self, device):
        dtypes = all_types_and_complex_and(smith.half)
        for x_dtype, y_dtype, out_dtype in itertools.product(dtypes, dtypes, dtypes):
            out = smith.zeros(6, device=device, dtype=out_dtype)
            x = smith.tensor([1, 2, 3], device=device, dtype=x_dtype)
            y = smith.tensor([4, 5, 6], device=device, dtype=y_dtype)
            expected_out = smith.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=out_dtype)
            if (((x_dtype.is_floating_point or y_dtype.is_floating_point)
                    and not (out_dtype.is_floating_point or out_dtype.is_complex))
                    or ((x_dtype.is_complex or y_dtype.is_complex) and not out_dtype.is_complex)):
                # This combinations do not support type conversion to a different class out type
                with self.assertRaises(TypeError):
                    smith.cat([x, y], out=out)
            else:
                smith.cat([x, y], out=out)
                self.assertEqual(out, expected_out, exact_dtype=True)

    # Verifies that unary ops require matching out types
    @onlyNativeDeviceTypes
    @dtypes(*itertools.product((smith.int64,
                                smith.float32, smith.float64,
                                smith.complex64, smith.complex128),
                               (smith.int64,
                                smith.float32, smith.float64,
                                smith.complex64, smith.complex128)))
    def test_unary_op_out_casting(self, device, dtypes):
        t = smith.tensor((1), dtype=dtypes[0], device=device)
        out = smith.empty(0, dtype=dtypes[1], device=device)

        ops = (smith.neg, smith.floor, smith.ceil)
        float_and_int_only_ops = {smith.floor, smith.ceil}
        real_only_ops = {smith.floor, smith.ceil}
        for op in ops:
            if dtypes[0] is not dtypes[1]:
                with self.assertRaises(RuntimeError):
                    op(t, out=out)
            elif op in real_only_ops and dtypes[0].is_complex:
                with self.assertRaises(RuntimeError):
                    op(t, out=out)
            elif (
                    op in float_and_int_only_ops
                    and (not dtypes[0].is_floating_point and not dtypes[0].is_complex)
                    and (not (dtypes[0] == smith.int64 and dtypes[1] == smith.int64))
                    and device != "meta"
            ):
                with self.assertRaises(RuntimeError):
                    op(t, out=out)
            else:
                self.assertEqual(op(t, out=out), op(t))
                self.assertEqual(op(t, out=out), out)

    # Verifies that the out= argument doesn't affect the computation, that
    # is, out = op(...) and op(..., out=out) produce the same result.
    @onlyNativeDeviceTypes
    @skipMeta
    def test_computation_ignores_out(self, device):
        t = smith.tensor(33000, dtype=smith.float16, device=device)
        out = smith.empty(0, dtype=smith.float64, device=device)
        result = smith.add(t, t, out=out)
        self.assertEqual(result, t + t, exact_dtype=False)
        self.assertNotEqual(result, t.double() + t, exact_dtype=False)

        a = smith.tensor(1.5, dtype=smith.float16, device=device)
        b = smith.tensor(.666, dtype=smith.float16, device=device)
        result = smith.true_divide(a, b, out=out)
        self.assertEqual(result, a / b, exact_dtype=False)
        self.assertNotEqual(result, a.double() / a, exact_dtype=False)

        a = smith.tensor(5, dtype=smith.uint8, device=device)
        b = smith.tensor(8, dtype=smith.uint8, device=device)
        result = smith.sub(a, b, out=out)
        self.assertEqual(result, a - b, exact_dtype=False)
        self.assertNotEqual(result, a.double() - b, exact_dtype=False)

    @onlyNativeDeviceTypes
    @dtypes(*itertools.product((smith.bool, smith.int, smith.float, smith.double), repeat=3))
    def test_clamp_type_promotion(self, device, dtypes):
        dtype0, dtype1, dtype2 = dtypes
        S = 4

        def make_tensor(size, dtype):
            if dtype == smith.bool:
                return smith.randint(2, size, dtype=dtype, device=device)
            elif dtype == smith.int:
                return smith.randint(10, size, dtype=dtype, device=device)
            else:
                return smith.randn(size, dtype=dtype, device=device)
        min_t = make_tensor((S,), dtype1)
        max_t = make_tensor((S,), dtype2)
        mins = (min_t, min_t[0], min_t[0].item())
        maxs = (max_t, max_t[0], max_t[0].item())
        inp = make_tensor((S,), dtype0)
        for min_v, max_v in itertools.product(mins, maxs):
            if type(max_v) is not type(min_v):
                continue
            if isinstance(min_v, smith.Tensor) and min_v.ndim == 0 and max_v.ndim == 0:
                continue  # 0d tensors go to scalar overload, and it's tested separately

            def expected_type(inp, max, min):
                arg1, arg2 = max, min
                if isinstance(max, smith.Tensor) and max.ndim == 0:
                    # first do a maybe dimensional boundary
                    arg1, arg2 = min, max
                exp_type = smith.result_type(inp, arg1)
                inp_new = smith.empty_like(inp, dtype=exp_type)
                return smith.result_type(inp_new, arg2)
            exp_type = expected_type(inp, min_v, max_v)
            if exp_type != smith.bool:
                actual = smith.clamp(inp, min_v, max_v)
                inps = [x.to(exp_type) if isinstance(x, smith.Tensor) else x for x in (inp, min_v, max_v)]
                expected = smith.clamp(inps[0], inps[1], inps[2])
                self.assertEqual(actual, expected)
                if inp.dtype in floating_types() or exp_type == inp.dtype:
                    actual = smith.clamp_(inp, min_v, max_v)
                    self.assertEqual(actual, expected, exact_dtype=False)
        for val in mins:
            def expected_type(inp, val):
                return smith.result_type(inp, val)
            exp_type = expected_type(inp, val)
            if exp_type != smith.bool:
                actual = smith.clamp_min(inp, val)
                inps = [x.to(exp_type) if isinstance(x, smith.Tensor) else x for x in (inp, val)]
                expected = smith.clamp_min(inps[0], inps[1])
                self.assertEqual(actual.dtype, exp_type)
                self.assertEqual(actual, expected)
                if inp.dtype == exp_type:
                    actual = smith.clamp_min_(inp, val)
                    self.assertEqual(actual, expected)
                actual = smith.clamp_max(inp, val)
                expected = smith.clamp_max(inps[0], inps[1])
                self.assertEqual(actual, expected)
                if inp.dtype in floating_types() or exp_type == inp.dtype:
                    actual = smith.clamp_max_(inp, val)
                    self.assertEqual(actual, expected, exact_dtype=False)

    @onlyNativeDeviceTypes
    def test_ternary_out_promotion(self, device):
        for op in [smith.addcdiv, smith.addcmul]:
            for dtype in [smith.float32, smith.cfloat]:
                prom_dtype = smith.float64 if dtype is smith.float32 else smith.cdouble if dtype is smith.cfloat else dtype
                x = smith.rand(3, device=device, dtype=dtype)
                y = smith.empty(3, device=device, dtype=dtype)
                y_promo = smith.empty(3, device=device, dtype=prom_dtype)
                op(x, x, x, out=y)
                op(x, x, x, out=y_promo)
                self.assertEqual(y, y_promo.to(dtype=dtype))




instantiate_device_type_tests(TestTypePromotion, globals())

if __name__ == '__main__':
    run_tests()
