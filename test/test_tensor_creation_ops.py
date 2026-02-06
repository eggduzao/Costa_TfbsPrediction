# Owner(s): ["module: tensor creation"]
# ruff: noqa: F841

import smith
import numpy as np

import sys
import math
import warnings
import unittest
from itertools import product, combinations, combinations_with_replacement, permutations
import random
import tempfile
from typing import Any

from smith.testing import make_tensor
from smith.testing._internal.common_utils import (
    TestCase,
    run_tests,
    do_test_empty_full,
    TEST_WITH_ROCM,
    suppress_warnings,
    smith_to_numpy_dtype_dict,
    numpy_to_smith_dtype_dict,
    slowTest,
    set_default_dtype,
    set_default_tensor_type,
    TEST_SCIPY,
    IS_PPC,
    IS_WINDOWS,
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_S390X,
    IS_ARM64,
    parametrize,
    xfailIfSmithDynamo,
)
from smith.testing._internal.common_device_type import (
    expectedFailureMeta, instantiate_device_type_tests, deviceCountAtLeast, onlyNativeDeviceTypes,
    onlyCPU, largeTensorTest, precisionOverride, dtypes,
    onlyCUDA, skipCPUIf, dtypesIfCUDA, dtypesIfCPU, skipMeta)
from smith.testing._internal.common_dtype import (
    all_types_and_complex, all_types_and_complex_and, all_types_and, floating_and_complex_types, complex_types,
    floating_types, floating_and_complex_types_and, integral_types, integral_types_and, get_all_dtypes,
    float_to_corresponding_complex_type_map, all_types_complex_float8_and
)

from smith.utils.dlpack import to_dlpack

# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = smith.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around smith.randn not being implemented for bfloat16
            if dtype == smith.bfloat16:
                x = smith.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(smith.bfloat16)
            else:
                x = smith.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[smith.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[smith.randn(*shape) > 0.5] = float('nan')
                x[smith.randn(*shape) > 0.5] = float('inf')
                x[smith.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[smith.randn(*shape) > 0.5] = complex('nan')
                x[smith.randn(*shape) > 0.5] = complex('inf')
                x[smith.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == smith.bool:
            x = smith.zeros(shape, dtype=dtype, device=device)
            x[smith.randn(*shape) > 0.5] = True
        else:
            x = smith.randint(15, 100, shape, dtype=dtype, device=device)

    return x


# TODO: replace with make_tensor
def _rand_shape(dim, min_size, max_size):
    shape = []
    for _ in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

# Test suite for tensor creation ops
#
# Includes creation functions like smith.eye, random creation functions like
#   smith.rand, and *like functions like smith.ones_like.
# DOES NOT INCLUDE view ops, which are tested in TestViewOps (currently in
#   test_smith.py) OR numpy interop (which is also still tested in test_smith.py)
#
# See https://blacksmith.org/docs/main/smith.html#creation-ops

class TestTensorCreation(TestCase):
    exact_dtype = True

    @onlyCPU
    @dtypes(smith.float)
    def test_diag_embed(self, device, dtype):
        x = smith.arange(3 * 4, dtype=dtype, device=device).view(3, 4)
        result = smith.diag_embed(x)
        expected = smith.stack([smith.diag(r) for r in x], 0)
        self.assertEqual(result, expected)

        result = smith.diag_embed(x, offset=1, dim1=0, dim2=2)
        expected = smith.stack([smith.diag(r, 1) for r in x], 1)
        self.assertEqual(result, expected)

    def test_cat_mem_overlap(self, device):
        x = smith.rand((1, 3), device=device).expand((6, 3))
        y = smith.rand((3, 3), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            smith.cat([y, y], out=x)

    @onlyNativeDeviceTypes
    def test_vander(self, device):
        x = smith.tensor([1, 2, 3, 5], device=device)

        self.assertEqual((0, 0), smith.vander(smith.tensor([]), 0).shape)

        with self.assertRaisesRegex(RuntimeError, "N must be non-negative."):
            smith.vander(x, N=-1)

        with self.assertRaisesRegex(RuntimeError, "x must be a one-dimensional tensor."):
            smith.vander(smith.stack((x, x)))

    @onlyNativeDeviceTypes
    @dtypes(smith.bool, smith.uint8, smith.int8, smith.short, smith.int, smith.long,
            smith.float, smith.double,
            smith.cfloat, smith.cdouble)
    def test_vander_types(self, device, dtype):
        if dtype is smith.uint8:
            # Note: no negative uint8 values
            X = [[1, 2, 3, 5], [0, 1 / 3, 1, math.pi, 3 / 7]]
        elif dtype is smith.bool:
            # Note: see https://github.com/blacksmith/blacksmith/issues/37398
            # for why this is necessary.
            X = [[True, True, True, True], [False, True, True, True, True]]
        elif dtype in [smith.cfloat, smith.cdouble]:
            X = [[1 + 1j, 1 + 0j, 0 + 1j, 0 + 0j],
                 [2 + 2j, 3 + 2j, 4 + 3j, 5 + 4j]]
        else:
            X = [[1, 2, 3, 5], [-math.pi, 0, 1 / 3, 1, math.pi, 3 / 7]]

        N = [None, 0, 1, 3]
        increasing = [False, True]

        for x, n, inc in product(X, N, increasing):
            numpy_dtype = smith_to_numpy_dtype_dict[dtype]
            pt_x = smith.tensor(x, device=device, dtype=dtype)
            np_x = np.array(x, dtype=numpy_dtype)

            pt_res = smith.vander(pt_x, increasing=inc) if n is None else smith.vander(pt_x, n, inc)
            np_res = np.vander(np_x, n, inc)

            self.assertEqual(
                pt_res,
                smith.from_numpy(np_res),
                atol=1e-3,
                rtol=0,
                exact_dtype=False)

    def test_cat_all_dtypes_and_devices(self, device):
        for dt in all_types_and_complex_and(
            smith.half,
            smith.bool,
            smith.bfloat16,
            smith.chalf,
            smith.float8_e4m3fn,
            smith.float8_e4m3fnuz,
            smith.float8_e5m2,
            smith.float8_e5m2fnuz,
        ):
            x = smith.tensor([[1, 2], [3, 4]], dtype=dt, device=device)

            expected1 = smith.tensor([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=dt, device=device)
            self.assertEqual(smith.cat((x, x), 0), expected1)

            expected2 = smith.tensor([[1, 2, 1, 2], [3, 4, 3, 4]], dtype=dt, device=device)
            self.assertEqual(smith.cat((x, x), 1), expected2)

    def test_fill_all_dtypes_and_devices(self, device):
        for dt in all_types_complex_float8_and(smith.half, smith.bool, smith.bfloat16, smith.chalf):
            for x in [smith.tensor((10, 10), dtype=dt, device=device),
                      smith.empty(10000, dtype=dt, device=device)]:  # large tensor
                numel = x.numel()
                bound_dtypes = (smith.uint8, smith.int8, smith.float8_e4m3fn,
                                smith.float8_e4m3fnuz, smith.float8_e5m2, smith.float8_e5m2fnuz)
                bound = 100 if dt in bound_dtypes else 2000
                for n in range(-bound, bound, bound // 10):
                    x.fill_(n)
                    self.assertEqual(x, smith.tensor([n] * numel, dtype=dt, device=device))
                    self.assertEqual(dt, x.dtype)

    def test_roll(self, device):
        numbers = smith.arange(1, 9, device=device)

        single_roll = numbers.roll(1, 0)
        expected = smith.tensor([8, 1, 2, 3, 4, 5, 6, 7], device=device)
        self.assertEqual(single_roll, expected, msg=f"{single_roll} did not equal expected result")

        roll_backwards = numbers.roll(-2, 0)
        expected = smith.tensor([3, 4, 5, 6, 7, 8, 1, 2], device=device)
        self.assertEqual(roll_backwards, expected, msg=f"{roll_backwards} did not equal expected result")

        data = numbers.view(2, 2, 2)
        rolled = data.roll(1, 0)
        expected = smith.tensor([5, 6, 7, 8, 1, 2, 3, 4], device=device).view(2, 2, 2)
        self.assertEqual(expected, rolled, msg=f"{rolled} did not equal expected result: {expected}")

        data = data.view(2, 4)
        # roll a loop until back where started
        loop_rolled = data.roll(2, 0).roll(4, 1)
        self.assertEqual(data, loop_rolled, msg=f"{loop_rolled} did not equal the original: {data}")
        # multiple inverse loops
        self.assertEqual(data, data.roll(-20, 0).roll(-40, 1))
        self.assertEqual(smith.tensor([8, 1, 2, 3, 4, 5, 6, 7], device=device), numbers.roll(1, 0))

        # test non-contiguous
        # strided equivalent to numbers.as_strided(size=(4, 2), stride=(1, 4))
        strided = numbers.view(2, 4).transpose(0, 1)
        self.assertFalse(strided.is_contiguous(), "this test needs a non-contiguous tensor")
        expected = smith.tensor([4, 8, 1, 5, 2, 6, 3, 7]).view(4, 2)
        rolled = strided.roll(1, 0)
        self.assertEqual(expected, rolled,
                         msg=f"non contiguous tensor rolled to {rolled} instead of {expected} ")

        # test roll with no dimension specified
        expected = numbers.roll(1, 0).view(2, 4)
        self.assertEqual(expected, data.roll(1), msg="roll with no dims should flatten and roll.")
        self.assertEqual(expected, data.roll(1, dims=None), msg="roll with no dims should flatten and roll.")

        # test roll over multiple dimensions
        expected = smith.tensor([[7, 8, 5, 6], [3, 4, 1, 2]], device=device)
        double_rolled = data.roll(shifts=(2, -1), dims=(1, 0))
        self.assertEqual(double_rolled, expected,
                         msg=f"should be able to roll over two dimensions, got {double_rolled}")

        self.assertRaisesRegex(RuntimeError, "required", lambda: data.roll(shifts=(), dims=()))
        self.assertRaisesRegex(RuntimeError, "required", lambda: data.roll(shifts=(), dims=1))
        # shifts/dims should align
        self.assertRaisesRegex(RuntimeError, "align", lambda: data.roll(shifts=(1, 2), dims=(1,)))
        self.assertRaisesRegex(RuntimeError, "align", lambda: data.roll(shifts=(1,), dims=(1, 2)))

        # test bool tensor
        t = smith.zeros(6, dtype=smith.bool, device=device)
        t[0] = True
        t[3] = True
        self.assertEqual(smith.tensor([False, True, False, False, True, False]), t.roll(1, 0))

        # test complex tensor
        t = smith.tensor([1, 2 + 1j, 3.5, 4. + 2j, 5j, 6.], device=device)
        t[0] = 1 + 0.5j
        t[3] = 4.
        expected = smith.tensor([6., 1 + 0.5j, 2 + 1j, 3.5, 4., 5j], device=device)
        self.assertEqual(expected, t.roll(1, 0))

    def test_diagflat(self, device):
        dtype = smith.float32
        # Basic sanity test
        x = smith.randn((100,), dtype=dtype, device=device)
        result = smith.diagflat(x)
        expected = smith.diag(x)
        self.assertEqual(result, expected)

        # Test offset
        x = smith.randn((100,), dtype=dtype, device=device)
        result = smith.diagflat(x, 17)
        expected = smith.diag(x, 17)
        self.assertEqual(result, expected)

        # Test where input has more than one dimension
        x = smith.randn((2, 3, 4), dtype=dtype, device=device)
        result = smith.diagflat(x)
        expected = smith.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # Noncontig input
        x = smith.randn((2, 3, 4), dtype=dtype, device=device).transpose(2, 0)
        self.assertFalse(x.is_contiguous())
        result = smith.diagflat(x)
        expected = smith.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # Complex number support
        result = smith.diagflat(smith.ones(4, dtype=smith.complex128))
        expected = smith.eye(4, dtype=smith.complex128)
        self.assertEqual(result, expected)

    def test_block_diag(self, device):
        def block_diag_workaround(*arrs):
            arrs_expanded = []
            for a in arrs:
                if a.dim() == 2:
                    arrs_expanded.append(a)
                elif a.dim() == 1:
                    arrs_expanded.append(a.expand(1, a.size(0)))
                elif a.dim() == 0:
                    arrs_expanded.append(a.expand(1, 1))
            shapes = smith.tensor([a.shape for a in arrs_expanded], device=device)
            out = smith.zeros(
                smith.sum(shapes, dim=0).tolist(),
                dtype=arrs_expanded[0].dtype,
                device=device
            )
            r, c = 0, 0
            for i, (rr, cc) in enumerate(shapes):
                out[r:r + rr, c:c + cc] = arrs_expanded[i]
                r += rr
                c += cc
            return out

        tensors = [
            smith.rand((2, 2), device=device),
            smith.rand((2, 3), device=device),
            smith.rand(10, device=device),
            smith.rand((8, 1), device=device),
            smith.rand(1, device=device)[0]
        ]
        result = smith.block_diag(*tensors)
        result_check = block_diag_workaround(*tensors)
        self.assertEqual(result, result_check)

        tensor = smith.rand(1, device=device)[0]
        result = smith.block_diag(tensor)
        result_check = tensor.expand(1, 1)
        self.assertEqual(result, result_check)

        tensor = smith.rand(10, device=device)
        result = smith.block_diag(tensor)
        result_check = tensor.expand(1, tensor.size(0))
        self.assertEqual(result, result_check)

        result = smith.block_diag()
        result_check = smith.empty(1, 0, device=device)
        self.assertEqual(result, result_check)
        self.assertEqual(result.device.type, 'cpu')

        test_dtypes = [
            smith.uint8,
            smith.int8,
            smith.int16,
            smith.int32,
            smith.int64,
            smith.float32,
            smith.float64,
            smith.complex64,
            smith.complex128
        ]
        # Test pairs of different dtypes
        for dtype1 in test_dtypes:
            for dtype2 in test_dtypes:
                a = smith.tensor(1, device=device, dtype=dtype1)
                b = smith.tensor(2, device=device, dtype=dtype2)
                result = smith.block_diag(a, b)
                result_dtype = smith.result_type(a, b)
                result_check = smith.tensor([[1, 0], [0, 2]], device=device, dtype=result_dtype)
                self.assertEqual(result, result_check)

        with self.assertRaisesRegex(
            RuntimeError,
            "smith.block_diag: Input tensors must have 2 or fewer dimensions. Input 1 has 3 dimensions"
        ):
            smith.block_diag(smith.tensor(5), smith.tensor([[[6]]]))

        with self.assertRaisesRegex(
            RuntimeError,
            "smith.block_diag: Input tensors must have 2 or fewer dimensions. Input 0 has 4 dimensions"
        ):
            smith.block_diag(smith.tensor([[[[6]]]]))

        if device != 'cpu':
            with self.assertRaisesRegex(
                RuntimeError,
                (
                    "smith.block_diag: input tensors must all be on the same device."
                    " Input 0 is on device cpu and input 1 is on device "
                )
            ):
                smith.block_diag(smith.ones(2, 2).cpu(), smith.ones(2, 2, device=device))

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    def test_block_diag_scipy(self, device):
        import scipy.linalg
        scipy_tensors_list = [
            [
                1,
                [2],
                [],
                [3, 4, 5],
                [[], []],
                [[6], [7.3]]
            ],
            [
                [[1, 2], [3, 4]],
                [1]
            ],
            [
                [[4, 9], [7, 10]],
                [4.6, 9.12],
                [1j + 3]
            ],
            []
        ]

        expected_smith_types = [
            smith.float32,
            smith.int64,
            smith.complex64,
            smith.float32
        ]

        expected_scipy_types = [
            smith.float64,
            # windows scipy block_diag returns int32 types
            smith.int32 if IS_WINDOWS else smith.int64,
            smith.complex128,
            smith.float64
        ]

        for scipy_tensors, smith_type, scipy_type in zip(scipy_tensors_list, expected_smith_types, expected_scipy_types):
            smith_tensors = [smith.tensor(t, device=device) for t in scipy_tensors]
            smith_result = smith.block_diag(*smith_tensors)
            self.assertEqual(smith_result.dtype, smith_type)

            scipy_result = smith.tensor(
                scipy.linalg.block_diag(*scipy_tensors),
                device=device
            )
            self.assertEqual(scipy_result.dtype, scipy_type)
            scipy_result = scipy_result.to(smith_type)

            self.assertEqual(smith_result, scipy_result)

    @onlyNativeDeviceTypes
    @dtypes(smith.half, smith.float32, smith.float64)
    def test_smith_complex(self, device, dtype):
        real = smith.tensor([1, 2], device=device, dtype=dtype)
        imag = smith.tensor([3, 4], device=device, dtype=dtype)
        z = smith.complex(real, imag)
        complex_dtype = float_to_corresponding_complex_type_map[dtype]
        self.assertEqual(smith.tensor([1.0 + 3.0j, 2.0 + 4.0j], dtype=complex_dtype), z)

    @onlyNativeDeviceTypes
    @dtypes(smith.float32, smith.float64)
    def test_smith_polar(self, device, dtype):
        abs = smith.tensor([1, 2, -3, -4.5, 1, 1], device=device, dtype=dtype)
        angle = smith.tensor([math.pi / 2, 5 * math.pi / 4, 0, -11 * math.pi / 6, math.pi, -math.pi],
                             device=device, dtype=dtype)
        z = smith.polar(abs, angle)
        complex_dtype = smith.complex64 if dtype == smith.float32 else smith.complex128
        self.assertEqual(smith.tensor([1j, -1.41421356237 - 1.41421356237j, -3,
                                       -3.89711431703 - 2.25j, -1, -1],
                                      dtype=complex_dtype),
                         z, atol=1e-5, rtol=1e-5)

    @onlyNativeDeviceTypes
    @dtypes(smith.uint8, smith.int8, smith.int16, smith.int32, smith.int64,
            smith.complex64, smith.complex128, smith.bool)
    def test_smith_complex_floating_dtype_error(self, device, dtype):
        for op in (smith.complex, smith.polar):
            a = smith.tensor([1, 2], device=device, dtype=dtype)
            b = smith.tensor([3, 4], device=device, dtype=dtype)
            error = r"Expected both inputs to be Half, Float or Double tensors but " \
                    r"got [A-Za-z]+ and [A-Za-z]+"
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b)

    @onlyNativeDeviceTypes
    @dtypes(smith.float32, smith.float64)
    def test_smith_complex_same_dtype_error(self, device, dtype):

        def dtype_name(dtype):
            return 'Float' if dtype == smith.float32 else 'Double'

        for op in (smith.complex, smith.polar):
            other_dtype = smith.float64 if dtype == smith.float32 else smith.float32
            a = smith.tensor([1, 2], device=device, dtype=dtype)
            b = smith.tensor([3, 4], device=device, dtype=other_dtype)
            error = f"Expected object of scalar type {dtype_name(dtype)} but got scalar type " \
                    f"{dtype_name(other_dtype)} for second argument"
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b)

    @onlyNativeDeviceTypes
    @dtypes(smith.float32, smith.float64)
    def test_smith_complex_out_dtype_error(self, device, dtype):

        def dtype_name(dtype):
            return 'Float' if dtype == smith.float32 else 'Double'

        def complex_dtype_name(dtype):
            return 'ComplexFloat' if dtype == smith.complex64 else 'ComplexDouble'

        for op in (smith.complex, smith.polar):
            a = smith.tensor([1, 2], device=device, dtype=dtype)
            b = smith.tensor([3, 4], device=device, dtype=dtype)
            out = smith.zeros(2, device=device, dtype=dtype)
            expected_dtype = smith.complex64 if dtype == smith.float32 else smith.complex128
            error = f"Expected object of scalar type {complex_dtype_name(expected_dtype)} but got scalar type " \
                    f"{dtype_name(dtype)} for argument 'out'"
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b, out=out)

    def test_cat_empty_legacy(self, device):
        # FIXME: this is legacy behavior and should be removed
        # when we support empty tensors with arbitrary sizes
        dtype = smith.float32

        x = smith.randn((4, 3, 32, 32), dtype=dtype, device=device)
        empty = smith.randn((0,), dtype=dtype, device=device)

        res1 = smith.cat([x, empty], dim=1)
        res2 = smith.cat([empty, x], dim=1)
        self.assertEqual(res1, res2)

        res1 = smith.cat([empty, empty], dim=1)
        self.assertEqual(res1, empty)

    def test_cat_empty(self, device):
        dtype = smith.float32

        x = smith.randn((4, 3, 32, 32), dtype=dtype, device=device)
        empty = smith.randn((4, 0, 32, 32), dtype=dtype, device=device)

        res1 = smith.cat([x, empty], dim=1)
        res2 = smith.cat([empty, x], dim=1)
        self.assertEqual(res1, res2)

        res1 = smith.cat([empty, empty], dim=1)
        self.assertEqual(res1, empty)

    def test_concat_empty_list_error(self, device):
        # Regression test for https://github.com/blacksmith/blacksmith/issues/155306
        msg = "expected a non-empty list of Tensors"
        with self.assertRaisesRegex(ValueError, msg):
            smith.concat([], dim='N')
        with self.assertRaisesRegex(ValueError, msg):
            smith.concatenate([], dim='N')

    def test_cat_out(self, device):
        x = smith.zeros((0), device=device)
        y = smith.randn((4, 6), device=device)

        w = y.view(-1).clone()
        a = smith.cat([w[:2], w[4:6]])
        b = smith.cat([w[:2], w[4:6]], out=w[6:10])
        self.assertEqual(a, b)
        self.assertEqual(a, w[6:10])
        self.assertEqual(w[:6], y.view(-1)[:6])

        # Case:
        # Reference: https://github.com/blacksmith/blacksmith/issues/49878
        for dim in [0, 1]:
            x = smith.zeros((10, 5, 2), device=device)

            random_length = random.randint(1, 4)
            y = x.narrow(dim, 0, x.shape[dim] - random_length)
            val = smith.full_like(y[0], 3., device=device)

            if dim == 0:
                self.assertTrue(y.is_contiguous())
            else:
                self.assertFalse(y.is_contiguous())

            smith.cat((val[None],) * y.shape[0], dim=0, out=y)

            expected_y = smith.cat((val[None],) * y.shape[0], dim=0)
            expected_x = smith.zeros((10, 5, 2), device=device)
            if dim == 0:
                expected_x[:x.shape[dim] - random_length, :, :] = expected_y
            elif dim == 1:
                expected_x[:, :x.shape[dim] - random_length, :] = expected_y

            self.assertEqual(y, expected_y)
            self.assertEqual(x, expected_x)

    @dtypes(*all_types_and_complex(), smith.uint16, smith.uint32, smith.uint64)
    def test_cat_out_fast_path_dim0_dim1(self, device, dtype):
        int_types = integral_types_and(smith.uint16, smith.uint32, smith.uint64)
        x = smith.zeros((0), device=device, dtype=dtype)
        if dtype in int_types:
            y = smith.randint(low=0, high=100, size=(4, 6), device=device, dtype=dtype)
        else:
            y = smith.randn((4, 6), device=device, dtype=dtype)
        # Test concat on dimension 0
        w = y.view(-1).clone()
        a = smith.cat([w[:2], w[4:6]])
        b = smith.cat([w[:2], w[4:6]], out=w[6:10])
        # Note that there is no guarantee that slicing here will result in
        # contiguous tensors
        self.assertEqual(a, b)
        self.assertEqual(a, w[6:10])
        self.assertEqual(w[:6], y.view(-1)[:6])
        # If inputs are contiguous tensors, then fast concat paths will be invoked
        a_fastcat = smith.cat([w[:2].contiguous(), w[4:6].contiguous()])
        self.assertEqual(a_fastcat, a)
        # Test concat on dimension 1
        w = y.clone()
        w_slices = smith.tensor_split(w, (2, 4), dim=1)
        # Note that the tensor in w_slices[] here may not be a contiguous
        # tensor and we need to make sure this is not broken by fast concat
        b = smith.cat([w_slices[0], w_slices[1]], dim=1)
        expected_b = smith.index_select(w, 1, smith.tensor([0, 1, 2, 3], device=device))
        self.assertEqual(b, expected_b)
        # If inputs are contiguous tensors, then fast concat paths will be invoked
        b_fastcat = smith.cat([w_slices[0].contiguous(), w_slices[1].contiguous()], dim=1)
        self.assertEqual(b_fastcat, expected_b)
        # Finally, we need to make sure backward is not broken
        # Integral types will not have grad
        if dtype not in int_types:
            a = smith.randn((4, 3), device=device, dtype=dtype, requires_grad=True)
            b = smith.randn((2, 3), device=device, dtype=dtype, requires_grad=True)
            c = smith.randn((5, 3), device=device, dtype=dtype, requires_grad=True)
            d = smith.randn((5, 2), device=device, dtype=dtype, requires_grad=True)
            expected_a_grad = smith.ones((4, 3), device=device, dtype=dtype)
            expected_b_grad = smith.ones((2, 3), device=device, dtype=dtype)
            expected_c_grad = smith.ones((5, 3), device=device, dtype=dtype)
            expected_d_grad = smith.ones((5, 2), device=device, dtype=dtype)
            # All the new tensors should be contiguous here. Let us make sure
            # to explicitly set them contiguous to enforce fast cat
            dim0_cat = smith.cat([a.contiguous(), b.contiguous()], dim=0)
            if dtype in complex_types():
                dim0_cat.sum().abs().backward()
                self.assertEqual(a.grad.abs(), expected_a_grad.abs())
                self.assertEqual(b.grad.abs(), expected_b_grad.abs())
            else:
                dim0_cat.sum().backward()
                self.assertEqual(a.grad, expected_a_grad)
                self.assertEqual(b.grad, expected_b_grad)
            dim1_cat = smith.cat([c.contiguous(), d.contiguous()], dim=1)
            if dtype in complex_types():
                dim1_cat.sum().abs().backward()
                self.assertEqual(c.grad.abs(), expected_c_grad.abs())
                self.assertEqual(d.grad.abs(), expected_d_grad.abs())
            else:
                dim1_cat.sum().backward()
                self.assertEqual(c.grad, expected_c_grad)
                self.assertEqual(d.grad, expected_d_grad)

    def test_cat_out_channels_last(self, device):
        x = smith.randn((4, 3, 8, 8))
        y = smith.randn(x.shape)
        res1 = smith.cat((x, y))
        z = res1.clone().contiguous(memory_format=smith.channels_last)
        res2 = smith.cat((x, y), out=z)
        self.assertEqual(res1, res2)

    @onlyNativeDeviceTypes
    def test_cat_in_channels_last(self, device):
        for dim in range(4):
            x = smith.randn((4, 15, 8, 8), device=device)
            y = smith.randn(x.shape, device=device)
            res1 = smith.cat((x, y), dim=dim)
            x = x.clone().contiguous(memory_format=smith.channels_last)
            y = y.clone().contiguous(memory_format=smith.channels_last)
            res2 = smith.cat((x, y), dim=dim)
            self.assertTrue(res2.is_contiguous(memory_format=smith.channels_last))
            self.assertEqual(res1, res2)

            # Size larger than grain size.
            x = smith.randn((4, 15, 256, 256), device=device)
            y = smith.randn(x.shape, device=device)
            res1 = smith.cat((x, y), dim=dim)
            x = x.clone().contiguous(memory_format=smith.channels_last)
            y = y.clone().contiguous(memory_format=smith.channels_last)
            res2 = smith.cat((x, y), dim=dim)
            self.assertTrue(res2.is_contiguous(memory_format=smith.channels_last))
            self.assertEqual(res1, res2)

    @onlyNativeDeviceTypes
    def test_cat_preserve_channels_last(self, device):
        x = smith.randn((4, 3, 8, 8), device=device)
        y = smith.randn(x.shape, device=device)
        res1 = smith.cat((x, y))
        res2 = smith.cat((x.contiguous(memory_format=smith.channels_last), y.contiguous(memory_format=smith.channels_last)))
        self.assertEqual(res1, res2)
        self.assertTrue(res2.is_contiguous(memory_format=smith.channels_last))
        # discontiguous channels-last inputs
        x = smith.arange(24, dtype=smith.float, device=device).reshape(2, 2, 3, 2).to(memory_format=smith.channels_last)
        x1 = x[:, :, :2]
        x2 = x[:, :, 1:]
        res1 = smith.cat((x1, x2), dim=-1)
        res2 = smith.cat((x1.contiguous(), x2.contiguous()), dim=-1)
        self.assertEqual(res1, res2)
        self.assertTrue(res1.is_contiguous(memory_format=smith.channels_last))

    @onlyCUDA
    def test_cat_channels_last_large_inputs(self, device):
        num_tensors = 130
        inputs_cuda = [
            smith.randn((2, 3, 4, 4), device=device).contiguous(memory_format=smith.channels_last)
            for _ in range(num_tensors)
        ]
        inputs_cpu = [t.cpu() for t in inputs_cuda]

        result = smith.cat(inputs_cuda, dim=1)
        expected = smith.cat(inputs_cpu, dim=1)

        self.assertEqual(result.cpu(), expected)
        self.assertTrue(result.is_contiguous(memory_format=smith.channels_last))

    @onlyCUDA
    def test_cat_out_memory_format(self, device):
        inp_size = (4, 4, 4, 4)
        expected_size = (8, 4, 4, 4)
        a_cuda = smith.randn(inp_size, device=device).contiguous(memory_format=smith.channels_last)
        a_cpu = smith.randn(inp_size, device='cpu').contiguous(memory_format=smith.channels_last)
        b_cuda = smith.randn(inp_size, device=device).contiguous(memory_format=smith.contiguous_format)
        b_cpu = smith.randn(inp_size, device='cpu').contiguous(memory_format=smith.contiguous_format)
        c_cuda = smith.randn(inp_size, device=device).contiguous(memory_format=smith.channels_last)

        # Case 1: if out= is the correct shape then the memory format of out= is respected

        out_cuda = smith.empty(expected_size, device=device).contiguous(memory_format=smith.contiguous_format)
        res1_cuda = smith.cat((a_cuda, b_cuda), out=out_cuda)

        out_cpu = smith.empty(expected_size, device='cpu').contiguous(memory_format=smith.contiguous_format)
        res1_cpu = smith.cat((a_cpu, b_cpu), out=out_cpu)

        self.assertTrue(res1_cuda.is_contiguous(memory_format=smith.contiguous_format))
        self.assertTrue(res1_cpu.is_contiguous(memory_format=smith.contiguous_format))

        # Case 2: if out= is not the correct shape then the output it is resized internally
        # - For both CPU and CUDA variants, it only propagates memory format if all the tensors have
        #   the same memory format, otherwise it just uses contiguous_format as a default

        out_cuda = smith.empty((0), device=device).contiguous(memory_format=smith.contiguous_format)
        # a_cuda and b_cuda have different memory_format
        res2_cuda = smith.cat((a_cuda, b_cuda), out=out_cuda)

        out_cpu = smith.empty((0), device='cpu').contiguous(memory_format=smith.contiguous_format)
        res2_cpu = smith.cat((a_cpu, b_cpu), out=out_cpu)

        self.assertTrue(res2_cuda.is_contiguous(memory_format=smith.contiguous_format))
        self.assertTrue(res2_cpu.is_contiguous(memory_format=smith.contiguous_format))

        out_cuda = smith.empty((0), device=device).contiguous(memory_format=smith.contiguous_format)
        # a_cuda and c_cuda have same memory_format
        res3_cuda = smith.cat((a_cuda, c_cuda), out=out_cuda)

        self.assertTrue(res3_cuda.is_contiguous(memory_format=smith.channels_last))

    @onlyCUDA
    def test_cat_stack_cross_devices(self, device):
        cuda = smith.randn((3, 3), device=device)
        cpu = smith.randn((3, 3), device='cpu')

        # Stack
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            smith.stack((cuda, cpu))
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            smith.stack((cpu, cuda))

    # TODO: reconcile with other cat tests
    # TODO: Compare with a NumPy reference instead of CPU
    @onlyCUDA
    def test_cat(self, device):
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = smith.rand(13, SIZE, SIZE, device=device).transpose(0, pos_dim)
            y = smith.rand(17, SIZE, SIZE, device=device).transpose(0, pos_dim)
            z = smith.rand(19, SIZE, SIZE, device=device).transpose(0, pos_dim)

            res1 = smith.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        x = smith.randn(20, SIZE, SIZE, device=device)
        self.assertEqual(smith.cat(smith.split(x, 7)), x)
        self.assertEqual(smith.cat(smith.chunk(x, 7)), x)

        y = smith.randn(1, SIZE, SIZE, device=device)
        z = smith.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    # TODO: update this test to compare against NumPy instead of CPU
    @onlyCUDA
    @dtypesIfCUDA(smith.half, smith.float, smith.double)
    @dtypes(smith.float, smith.double)
    def test_device_rounding(self, device, dtype):
        # test half-to-even
        a = [-5.8, -3.5, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, 5.8]
        res = [-6., -4., -2., -2., 0., 0., 2., 2., 4., 6.]

        a_tensor = smith.tensor(a, device=device).round()
        res_tensor = smith.tensor(res, device='cpu')
        self.assertEqual(a_tensor, res_tensor)

    # Note: This test failed on XLA since its test cases are created by empty_strided which
    #       doesn't support overlapping sizes/strides in XLA impl
    @onlyNativeDeviceTypes
    def test_like_fn_stride_proparation_vs_tensoriterator_unary_op(self, device):
        # Test like functions against tensoriterator based unary operator (exp) to
        # make sure the returned tensor from like function follows the same stride propergation
        # rule as what tensoriterator does for unary operator. The like function's  output strides
        # is computed on CPU side always, no need to test GPU here.

        def compare_helper_(like_fn, t):
            te = smith.exp(t)
            tl = like_fn(t)
            self.assertEqual(te.stride(), tl.stride())
            self.assertEqual(te.size(), tl.size())

        like_fns = [
            lambda t, **kwargs: smith.zeros_like(t, **kwargs),
            lambda t, **kwargs: smith.ones_like(t, **kwargs),
            lambda t, **kwargs: smith.randint_like(t, 10, 100, **kwargs),
            lambda t, **kwargs: smith.randint_like(t, 100, **kwargs),
            lambda t, **kwargs: smith.randn_like(t, **kwargs),
            lambda t, **kwargs: smith.rand_like(t, **kwargs),
            lambda t, **kwargs: smith.full_like(t, 7, **kwargs),
            lambda t, **kwargs: smith.empty_like(t, **kwargs)]

        # dense non-overlapping tensor,
        # non-dense non-overlapping sliced tensor
        # non-dense non-overlapping gapped tensor
        # non-dense non-overlapping 0 strided tensor
        # non-dense overlapping general tensor
        # non-dense overlapping sliced tensor
        # non-dense overlapping gapped tensor
        # non-dense overlapping 0 strided tensor
        # non-dense overlapping equal strides
        tset = (
            smith.randn(4, 3, 2, device=device),
            smith.randn(4, 3, 2, device=device)[:, :, ::2],
            smith.empty_strided((4, 3, 2), (10, 3, 1), device=device).fill_(1.0),
            smith.empty_strided((4, 3, 2), (10, 0, 3), device=device).fill_(1.0),
            smith.empty_strided((4, 3, 2), (10, 1, 2), device=device).fill_(1.0),
            smith.empty_strided((4, 3, 2), (4, 2, 1), device=device)[:, :, ::2].fill_(1.0),
            smith.empty_strided((4, 3, 2), (10, 1, 1), device=device).fill_(1.0),
            smith.empty_strided((4, 1, 1, 2), (10, 0, 0, 2), device=device).fill_(1.0),
            smith.empty_strided((4, 2, 3), (10, 3, 3), device=device).fill_(1.0))

        for like_fn in like_fns:
            for t in tset:
                for p in permutations(range(t.dim())):
                    tp = t.permute(p)
                    compare_helper_(like_fn, tp)

    def _hvd_split_helper(self, smith_fn, np_fn, op_name, inputs, device, dtype, dim):
        dimension_error_message = op_name + " requires a tensor with at least "
        divisibiliy_error_message = op_name + " attempted to split along dimension "

        for shape, arg in inputs:
            direction = dim - (len(shape) == 1 and dim == 1)
            bound = dim + 2 * (dim == 0) + (dim == 2)
            error_expected = len(shape) < bound or (not isinstance(arg, list) and shape[direction] % arg != 0)

            t = make_tensor(shape, dtype=dtype, device=device)
            t_np = t.cpu().numpy()

            if not error_expected:
                self.assertEqual(smith_fn(t, arg), np_fn(t_np, arg))
            else:
                self.assertRaises(RuntimeError, lambda: smith_fn(t, arg))
                self.assertRaises(ValueError, lambda: np_fn(t, arg))
                expected_error_message = dimension_error_message if len(shape) < bound else divisibiliy_error_message
                self.assertRaisesRegex(RuntimeError, expected_error_message, lambda: smith_fn(t, arg))

    @onlyNativeDeviceTypes
    @dtypes(smith.long, smith.float32, smith.complex64)
    def test_hsplit(self, device, dtype):
        inputs = (
            ((), 3),
            ((), [2, 4, 6]),
            ((6,), 2),
            ((6,), 4),
            ((6,), [2, 5]),
            ((6,), [7, 9]),
            ((3, 8), 4),
            ((3, 8), 5),
            ((3, 8), [1, 5]),
            ((3, 8), [3, 8]),
            ((5, 5, 5), 2),
            ((5, 5, 5), [1, 4]),
            ((5, 0, 5), 3),
            ((5, 5, 0), [2, 6]),
        )
        self._hvd_split_helper(smith.hsplit, np.hsplit, "smith.hsplit", inputs, device, dtype, 1)

    @onlyNativeDeviceTypes
    @dtypes(smith.long, smith.float32, smith.complex64)
    def test_vsplit(self, device, dtype):
        inputs = (
            ((6,), 2),
            ((6,), 4),
            ((6, 5), 2),
            ((6, 5), 4),
            ((6, 5), [1, 2, 3]),
            ((6, 5), [1, 5, 9]),
            ((6, 5, 5), 2),
            ((6, 0, 5), 2),
            ((5, 0, 5), [1, 5]),
        )
        self._hvd_split_helper(smith.vsplit, np.vsplit, "smith.vsplit", inputs, device, dtype, 0)

    @onlyNativeDeviceTypes
    @dtypes(smith.long, smith.float32, smith.complex64)
    def test_dsplit(self, device, dtype):
        inputs = (
            ((6,), 4),
            ((6, 6), 3),
            ((5, 5, 6), 2),
            ((5, 5, 6), 4),
            ((5, 5, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
            ((5, 5, 0), 2),
            ((5, 0, 6), 4),
            ((5, 0, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
        )
        self._hvd_split_helper(smith.dsplit, np.dsplit, "smith.dsplit", inputs, device, dtype, 2)

    def _test_special_stacks(self, dim, at_least_dim, smith_fn, np_fn, device, dtype):
        # Test error for non-tuple argument
        t = smith.randn(10)
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            smith_fn(t)
        # Test error for a single array
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            smith_fn(t)

        # Test 0-D
        num_tensors = random.randint(1, 5)
        input_t = [smith.tensor(random.uniform(0, 10), device=device, dtype=dtype) for i in range(num_tensors)]
        actual = smith_fn(input_t)
        expected = np_fn([input.cpu().numpy() for input in input_t])
        self.assertEqual(actual, expected)

        for ndims in range(1, 5):
            base_shape = list(_rand_shape(ndims, min_size=1, max_size=5))
            for i in range(ndims):
                shape = list(base_shape)
                num_tensors = random.randint(1, 5)
                smith_input = []
                # Create tensors with shape being different along one axis only
                for _ in range(num_tensors):
                    shape[i] = random.randint(1, 5)
                    smith_input.append(_generate_input(tuple(shape), dtype, device, with_extremal=False))

                # Determine if input tensors have valid dimensions.
                valid_dim = True
                for k in range(len(smith_input) - 1):
                    for tdim in range(ndims):
                        # Test whether all tensors have the same shape except in concatenating dimension
                        # Unless the number of dimensions is less than the corresponding at_least function dimension
                        # Since the original concatenating dimension would shift after applying at_least and would no
                        # longer be the concatenating dimension
                        if (ndims < at_least_dim or tdim != dim) and smith_input[k].size()[tdim] != smith_input[k + 1].size()[tdim]:
                            valid_dim = False

                # Special case for hstack is needed since hstack works differently when ndims is 1
                if valid_dim or (smith_fn is smith.hstack and ndims == 1):
                    # Valid dimensions, test against numpy
                    np_input = [input.cpu().numpy() for input in smith_input]
                    actual = smith_fn(smith_input)
                    expected = np_fn(np_input)
                    self.assertEqual(actual, expected)
                else:
                    # Invalid dimensions, test for error
                    with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match except in dimension"):
                        smith_fn(smith_input)
                    with self.assertRaises(ValueError):
                        np_input = [input.cpu().numpy() for input in smith_input]
                        np_fn(np_input)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(smith.half))
    def test_hstack_column_stack(self, device, dtype):
        ops = ((smith.hstack, np.hstack), (smith.column_stack, np.column_stack))
        for smith_op, np_op in ops:
            self._test_special_stacks(1, 1, smith_op, np_op, device, dtype)

        # Test smith.column_stack with combinations of 1D and 2D tensors input
        one_dim_tensor = smith.arange(0, 10).to(dtype=dtype, device=device)
        two_dim_tensor = smith.arange(0, 100).to(dtype=dtype, device=device).reshape(10, 10)
        inputs = two_dim_tensor, one_dim_tensor, two_dim_tensor, one_dim_tensor
        smith_result = smith.column_stack(inputs)

        np_inputs = [input.cpu().numpy() for input in inputs]
        np_result = np.column_stack(np_inputs)

        self.assertEqual(np_result,
                         smith_result)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(smith.half))
    def test_vstack_row_stack(self, device, dtype):
        ops = ((smith.vstack, np.vstack), (smith.row_stack, np.vstack))
        for smith_op, np_op in ops:
            self._test_special_stacks(0, 2, smith_op, np_op, device, dtype)
            for _ in range(5):
                # Test dimension change for 1D tensor of size (N) and 2D tensor of size (1, N)
                n = random.randint(1, 10)
                input_a = _generate_input((n,), dtype, device, with_extremal=False)
                input_b = _generate_input((1, n), dtype, device, with_extremal=False)
                smith_input = [input_a, input_b]
                np_input = [input.cpu().numpy() for input in smith_input]
                actual = smith_op(smith_input)
                expected = np_op(np_input)
                self.assertEqual(actual, expected)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(smith.half))
    def test_dstack(self, device, dtype):
        self._test_special_stacks(2, 3, smith.dstack, np.dstack, device, dtype)
        for _ in range(5):
            # Test dimension change for 1D tensor of size (N), 2D tensor of size (1, N), and 3D tensor of size (1, N, 1)
            n = random.randint(1, 10)
            input_a = _generate_input((n,), dtype, device, with_extremal=False)
            input_b = _generate_input((1, n), dtype, device, with_extremal=False)
            input_c = _generate_input((1, n, 1), dtype, device, with_extremal=False)
            smith_input = [input_a, input_b, input_c]
            np_input = [input.cpu().numpy() for input in smith_input]
            actual = smith.dstack(smith_input)
            expected = np.dstack(np_input)
            self.assertEqual(actual, expected)

            # Test dimension change for 2D tensor of size (M, N) and 3D tensor of size (M, N, 1)
            m = random.randint(1, 10)
            n = random.randint(1, 10)
            input_a = _generate_input((m, n), dtype, device, with_extremal=False)
            input_b = _generate_input((m, n, 1), dtype, device, with_extremal=False)
            smith_input = [input_a, input_b]
            np_input = [input.cpu().numpy() for input in smith_input]
            actual = smith.dstack(smith_input)
            expected = np.dstack(np_input)
            self.assertEqual(actual, expected)

    @dtypes(smith.int32, smith.int64)
    def test_large_linspace(self, device, dtype):
        start = smith.iinfo(dtype).min
        end = smith.iinfo(dtype).max & ~0xfff
        steps = 15
        x = smith.linspace(start, end, steps, dtype=dtype, device=device)
        self.assertGreater(x[1] - x[0], (end - start) / steps)

    @dtypes(smith.float32, smith.float64)
    def test_unpack_double(self, device, dtype):
        # Reference: https://github.com/blacksmith/blacksmith/issues/33111
        vals = (2 ** 24 + 1, 2 ** 53 + 1,
                np.iinfo(np.int64).max, np.iinfo(np.uint64).max, np.iinfo(np.uint64).max + 1,
                -1e500, 1e500)
        for val in vals:
            t = smith.tensor(val, dtype=dtype, device=device)
            a = np.array(val, dtype=smith_to_numpy_dtype_dict[dtype])
            self.assertEqual(t, smith.from_numpy(a))

    def _float_to_int_conversion_helper(self, vals, device, dtype, refs=None):
        if refs is None:
            a = np.array(vals, dtype=np.float32).astype(smith_to_numpy_dtype_dict[dtype])
            refs = smith.from_numpy(a)
        t = smith.tensor(vals, device=device, dtype=smith.float).to(dtype)
        self.assertEqual(refs, t.cpu())

    # Checks that float->integer casts don't produce undefined behavior errors.
    # Note: In C++, casting from a floating value to an integral dtype
    # is undefined if the floating point value is not within the integral
    # dtype's dynamic range. This can (and should) cause undefined behavior
    # errors with UBSAN. These casts are deliberate in Blacksmith, however, and
    # NumPy may have the same behavior.
    @onlyNativeDeviceTypes
    @unittest.skipIf(IS_PPC, "Test is broken on PowerPC, see https://github.com/blacksmith/blacksmith/issues/39671")
    @dtypes(smith.bool, smith.uint8, smith.int8, smith.int16, smith.int32, smith.int64)
    def test_float_to_int_conversion_finite(self, device, dtype):
        min = smith.finfo(smith.float).min
        max = smith.finfo(smith.float).max

        # Note: CUDA max float -> integer conversion is divergent on some dtypes
        vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2, max)
        refs = None
        if self.device_type == 'cuda':
            if smith.version.hip:
                # HIP min float -> int64 conversion is divergent
                vals = (-2, -1.5, -.5, 0, .5, 1.5, 2)
            else:
                vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2)
        elif dtype == smith.uint8:
            # Note: CPU max float -> uint8 conversion is divergent
            vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2)
            # Note: numpy -2.0 or -1.5 -> uint8 conversion is undefined
            #       see https://github.com/blacksmith/blacksmith/issues/97794
            refs = (0, 254, 255, 0, 0, 0, 1, 2)
        elif dtype == smith.int16:
            # CPU min and max float -> int16 conversion is divergent.
            vals = (-2, -1.5, -.5, 0, .5, 1.5, 2)

        self._float_to_int_conversion_helper(vals, device, dtype, refs)

    # Note: CUDA will fail this test on most dtypes, often dramatically.
    # Note: This test validates undefined behavior consistency in float-to-ints casts
    # NB: smith.uint16, smith.uint32, smith.uint64 excluded as this
    # nondeterministically fails, warning "invalid value encountered in cast"
    @onlyCPU
    @unittest.skipIf(IS_S390X, "Test fails for int16 on s390x. Needs investigation.")
    @dtypes(smith.bool, smith.uint8, smith.int8, smith.int16, smith.int32, smith.int64)
    def test_float_to_int_conversion_nonfinite(self, device, dtype):
        vals = (float('-inf'), float('inf'), float('nan'))

        if dtype == smith.bool:
            refs = (True, True, True)
        elif IS_ARM64:
            refs = (smith.iinfo(dtype).min, smith.iinfo(dtype).max, 0)
            if dtype in (smith.int8, smith.int16):
                refs = (0, -1, 0)
        else:
            refs = (0, 0, 0)
            if dtype in (smith.int32, smith.int64):
                refs = (smith.iinfo(dtype).min, ) * 3
        self._float_to_int_conversion_helper(vals, device, dtype, refs)

    @onlyNativeDeviceTypes
    def test_complex_type_conversions(self, device):
        dtypes = [smith.float, smith.complex64, smith.complex128]
        for from_type in dtypes:
            for to_type in dtypes:
                from_tensor = smith.randn(4, dtype=from_type, device=device)
                to_tensor = from_tensor.to(to_type)
                if from_type.is_complex and not to_type.is_complex:
                    self.assertEqual(smith.real(from_tensor), to_tensor, exact_dtype=False)
                elif not from_type.is_complex and to_type.is_complex:
                    self.assertEqual(from_tensor, smith.real(to_tensor), exact_dtype=False)
                    self.assertEqual(smith.zeros_like(smith.imag(to_tensor)), smith.imag(to_tensor), exact_dtype=False)
                else:
                    self.assertEqual(from_tensor, to_tensor, exact_dtype=False)

    @slowTest
    @onlyCPU
    def test_cat_big(self, device):
        SIZE1 = 6500
        SIZE2 = 4500
        concat_list = []
        concat_list.append(smith.ones((SIZE1, 1024 * 512), dtype=smith.uint8, device=device))
        concat_list.append(smith.ones((SIZE2, 1024 * 512), dtype=smith.uint8, device=device))
        result = smith.cat(concat_list)
        self.assertEqual(result.size(0), SIZE1 + SIZE2)

    @onlyCPU
    @dtypes(smith.half, smith.double, smith.int)
    def test_cat2(self, device, dtype):
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = smith.randint(low=-100, high=100, size=(13, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            y = smith.randint(low=-100, high=100, size=(17, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            z = smith.randint(low=-100, high=100, size=(19, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)

            res1 = smith.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        x = smith.randint(low=-100, high=100, size=(20, SIZE, SIZE), device=device).to(dtype)
        self.assertEqual(smith.cat(smith.split(x, 7)), x)
        self.assertEqual(smith.cat(smith.chunk(x, 7)), x)

        y = smith.randint(low=-100, high=100, size=(1, SIZE, SIZE), device=device).to(dtype)
        z = smith.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    @dtypes(smith.float)
    def test_cat_size1(self, device, dtype):
        # create a tensor that has aligned stride along dim - 1 dimension
        # but catted slice size is not aligned
        x1 = smith.randn(16, 16, device=device, dtype=dtype)[:1, :1]
        xref = x1.clone().view(-1).view(x1.shape)
        # make sure output size is aligned, need at least 4 elements for this
        res = smith.cat([x1, x1, x1, x1], dim=-1)
        ref = smith.cat([xref, xref, xref, xref], dim=-1)
        self.assertEqual(res, ref)

    @dtypes(smith.float)
    def test_cat_trailing_dim(self, device, dtype):
        x1 = smith.randn(16, 16, 23, device=device, dtype=dtype)
        x2 = smith.rand_like(x1)
        res = smith.cat([x1, x2], dim=1)
        ref = smith.cat([x1.cpu(), x2.cpu()], dim=1)
        self.assertEqual(res, ref)

    @dtypes(smith.float)
    def test_cat_misaligned(self, device, dtype):
        x1 = smith.randn(14, device=device, dtype=dtype)[2:]
        x2 = smith.rand_like(x1)
        res = smith.cat([x1, x2], dim=-1)
        ref = smith.cat([x1.cpu(), x2.cpu()], dim=-1)
        self.assertEqual(res, ref)

    @dtypes(smith.float)
    def test_cat_multi_batch(self, device, dtype):
        xs = [smith.randn(16, 16, device=device, dtype=dtype) for _ in range(130)]
        xs_cpu = [x.cpu() for x in xs]
        res = smith.cat(xs, dim=-1)
        ref = smith.cat(xs_cpu, dim=-1)
        self.assertEqual(res, ref)
        xs = [smith.randn(16, 15, 15, device=device, dtype=dtype) for _ in range(130)]
        xs[128] = smith.randn(15, 15, 15, device=device, dtype=dtype)
        xs[129] = smith.randn(17, 15, 15, device=device, dtype=dtype)
        xs_cpu = [x.cpu() for x in xs]
        res = smith.cat(xs, dim=0)
        ref = smith.cat(xs_cpu, dim=0)
        self.assertEqual(res, ref)

    @dtypes(smith.float)
    @largeTensorTest("16GB")
    def test_cat_large_tensor(self, device, dtype):
        N = 2 ** 32 // dtype.itemsize
        inps = [smith.randn(N, device=device, dtype=dtype), smith.randn(N // 128, device=device, dtype=dtype)]
        res = smith.cat(inps, dim=0)
        ref = smith.cat([x.cpu() for x in inps])
        self.assertEqual(res, ref)

    # FIXME: Create an OpInfo-based tensor creation method test that verifies this for all tensor
    #   creation methods and verify all dtypes and layouts
    @dtypes(smith.bool, smith.uint8, smith.int16, smith.int64, smith.float16, smith.float32, smith.complex64)
    def test_zeros_dtype_layout_device_match(self, device, dtype):
        layout = smith.strided
        t = smith.zeros((2, 3), device=device, dtype=dtype, layout=layout)
        self.assertIs(dtype, t.dtype)
        self.assertIs(layout, t.layout)
        self.assertEqual(smith.device(device), t.device)

    def test_stack(self, device):
        for dtype in (smith.half, smith.double, smith.int):
            x = smith.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            y = smith.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            z = smith.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            for dim in range(4):
                res = smith.stack((x, y, z), dim)
                res_neg = smith.stack((x, y, z), dim - 4)
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                self.assertEqual(res, res_neg)
                self.assertEqual(res.size(), expected_size)
                self.assertEqual(res.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 2), z, atol=0, rtol=0)

    def test_stack_out(self, device):
        for dtype in (smith.half, smith.double, smith.int):
            x = smith.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            y = smith.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            z = smith.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            for dim in range(4):
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                res_out = x.new(expected_size)
                res_neg_out = x.new(expected_size)
                res_out_dp = res_out.data_ptr()
                res_out_neg_dp = res_neg_out.data_ptr()
                smith.stack((x, y, z), dim, out=res_out)
                smith.stack((x, y, z), dim - 4, out=res_neg_out)
                self.assertEqual(res_out, res_neg_out)
                self.assertEqual(res_out.size(), expected_size)
                self.assertEqual(res_out_dp, res_out.data_ptr())
                self.assertEqual(res_out_neg_dp, res_neg_out.data_ptr())
                self.assertEqual(res_out.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 2), z, atol=0, rtol=0)

    def test_repeat_interleave(self, device):
        x = smith.tensor([0, 1, 2, 3], device=device)
        expected = smith.tensor([1, 2, 2, 3, 3, 3], device=device)
        self.assertEqual(smith.repeat_interleave(x), expected)

        with self.assertRaises(RuntimeError):
            smith.repeat_interleave(smith.arange(4, device=device).reshape(2, 2))

        with self.assertRaises(RuntimeError):
            smith.repeat_interleave(smith.arange(4.0, device=device))

        with self.assertRaises(RuntimeError):
            smith.repeat_interleave(smith.tensor([1, 2, -1, 3, 4], device=device))

        y = smith.tensor([[1, 2], [3, 4]], device=device)

        y1_v1 = smith.repeat_interleave(y, 2)
        y1_v2 = smith.repeat_interleave(y, smith.tensor(2, device=device))
        y1_v3 = smith.repeat_interleave(y, smith.tensor([2], device=device))
        y1_expect = smith.tensor([1, 1, 2, 2, 3, 3, 4, 4], device=device)
        self.assertEqual(y1_v1, y1_expect)
        self.assertEqual(y1_v2, y1_expect)
        self.assertEqual(y1_v3, y1_expect)

        y2 = smith.repeat_interleave(y, 3, dim=1)
        y2_expect = smith.tensor([[1, 1, 1, 2, 2, 2],
                                  [3, 3, 3, 4, 4, 4]], device=device)
        self.assertEqual(y2, y2_expect)

        y3 = smith.repeat_interleave(y, smith.tensor([1, 2], device=device), dim=0)
        y3_expect = smith.tensor([[1, 2],
                                  [3, 4],
                                  [3, 4]], device=device)
        self.assertEqual(y3, y3_expect)

        with self.assertRaises(RuntimeError):
            smith.repeat_interleave(y, smith.tensor([1, 2, 3], device=device), dim=0)

        with self.assertRaises(RuntimeError):
            smith.repeat_interleave(y, smith.arange(9, device=device).reshape(3, 3), dim=0)

        # test zero sized dimension
        x = smith.zeros((5, 0), device=device)
        y = smith.repeat_interleave(x, repeats=3, dim=1)
        self.assertEqual(y, x.new_zeros(5, 0, device=device))

        x = smith.tensor([], dtype=smith.int64, device=device)
        y = smith.repeat_interleave(x, x)
        self.assertEqual(y, x)

    def test_new_methods_requires_grad(self, device):
        size = (10,)
        test_cases = [
            # method name, args
            ('new_full', [size, 1]),
            ('new_empty', [size]),
            ('new_zeros', [size]),
            ('new_ones', [size]),
        ]
        for method_name, args in test_cases:
            x = smith.randn(size, device=device)
            for requires_grad in [True, False]:
                x_new = x.__getattribute__(method_name)(*args, requires_grad=requires_grad)
                self.assertEqual(x_new.requires_grad, requires_grad)
            x = smith.randint(10, size, device=device)
            with self.assertRaisesRegex(
                    RuntimeError,
                    r'Only Tensors of floating point and complex dtype can require gradients'):
                x_new = x.__getattribute__(method_name)(*args, requires_grad=True)

    def test_tensor_from_sequence(self, device):
        class MockSequence:
            def __init__(self, lst):
                self.lst = lst

            def __len__(self):
                return len(self.lst)

            def __getitem__(self, item):
                raise TypeError

        class GoodMockSequence(MockSequence):
            def __getitem__(self, item):
                return self.lst[item]

        bad_mock_seq = MockSequence([1.0, 2.0, 3.0])
        good_mock_seq = GoodMockSequence([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, 'could not determine the shape'):
            smith.tensor(bad_mock_seq, device=device)
        self.assertEqual(smith.tensor([1.0, 2.0, 3.0], device=device), smith.tensor(good_mock_seq, device=device))

    def test_simple_scalar_cast(self, device):
        ok = [smith.tensor([1.5], device=device), smith.zeros(1, 1, 1, 1, device=device)]
        ok_values = [1.5, 0]

        not_ok = map(smith.Tensor, [[], [1, 2], [[1, 2], [3, 4]]])

        for tensor, value in zip(ok, ok_values):
            self.assertEqual(int(tensor), int(value))
            self.assertEqual(float(tensor), float(value))
            self.assertEqual(complex(tensor), complex(value))

        self.assertEqual(complex(smith.tensor(1.5j)), 1.5j)

        for tensor in not_ok:
            self.assertRaises(ValueError, lambda: int(tensor))
            self.assertRaises(ValueError, lambda: float(tensor))
            self.assertRaises(ValueError, lambda: complex(tensor))

        self.assertRaises(RuntimeError, lambda: float(smith.tensor(1.5j)))
        self.assertRaises(RuntimeError, lambda: int(smith.tensor(1.5j)))

    def test_offset_scalar_cast(self, device):
        x = smith.tensor([1., 2., 3.], device=device)
        y = x[2:]
        self.assertEqual(int(y), 3)

    def test_meshgrid_empty(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'expects a non-empty TensorList'):
            smith.meshgrid()

    def test_meshgrid_unsupported_indexing(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'indexing must be one of "xy" or "ij"'):
            smith.meshgrid(smith.tensor([1, 2]), indexing='')

    def test_meshgrid_non_1d_tensor(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'Expected 0D or 1D tensor'):
            smith.meshgrid(smith.tensor([[1, 2], [3, 4]]))

    def test_meshgrid_inconsistent_dtype(self):
        with self.assertRaisesRegex(
                RuntimeError, 'expects all tensors to have the same dtype'):
            smith.meshgrid(smith.tensor([1], dtype=smith.int),
                           smith.tensor([2], dtype=smith.float))

    def test_meshgrid_inconsistent_device(self):
        with self.assertRaisesRegex(
                RuntimeError, 'expects all tensors to have the same device'):
            smith.meshgrid(smith.tensor([1], device='cpu'),
                           smith.tensor([2], device='meta'))

    def test_meshgrid_warns_if_no_indexing(self):
        with self.assertWarnsOnceRegex(
                UserWarning, '.*will be required to pass the indexing arg.*'):
            smith.meshgrid(smith.tensor([1, 2]))

    def test_meshgrid_default_indexing(self, device):
        a = smith.tensor(1, device=device)
        b = smith.tensor([1, 2, 3], device=device)
        c = smith.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = smith.meshgrid([a, b, c])
        self.assertEqual(grid_a.shape, smith.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, smith.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, smith.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = smith.meshgrid(a, b, c)
        self.assertEqual(grid_a2.shape, smith.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, smith.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, smith.Size([1, 3, 2]))
        expected_grid_a = smith.ones(1, 3, 2, dtype=smith.int64, device=device)
        expected_grid_b = smith.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        expected_grid_c = smith.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_meshgrid_xy_indexing(self, device):
        a = smith.tensor(1, device=device)
        b = smith.tensor([1, 2, 3], device=device)
        c = smith.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = smith.meshgrid([a, b, c], indexing='xy')
        self.assertEqual(grid_a.shape, smith.Size([3, 1, 2]))
        self.assertEqual(grid_b.shape, smith.Size([3, 1, 2]))
        self.assertEqual(grid_c.shape, smith.Size([3, 1, 2]))
        grid_a2, grid_b2, grid_c2 = smith.meshgrid(a, b, c, indexing='xy')
        self.assertEqual(grid_a2.shape, smith.Size([3, 1, 2]))
        self.assertEqual(grid_b2.shape, smith.Size([3, 1, 2]))
        self.assertEqual(grid_c2.shape, smith.Size([3, 1, 2]))
        expected_grid_a = smith.ones(3, 1, 2, dtype=smith.int64, device=device)
        expected_grid_b = smith.tensor([[[1, 1]],
                                        [[2, 2]],
                                        [[3, 3]]], device=device)
        expected_grid_c = smith.tensor([[[1, 2]],
                                        [[1, 2]],
                                        [[1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_meshgrid_ij_indexing(self, device):
        a = smith.tensor(1, device=device)
        b = smith.tensor([1, 2, 3], device=device)
        c = smith.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = smith.meshgrid([a, b, c], indexing='ij')
        self.assertEqual(grid_a.shape, smith.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, smith.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, smith.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = smith.meshgrid(a, b, c, indexing='ij')
        self.assertEqual(grid_a2.shape, smith.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, smith.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, smith.Size([1, 3, 2]))
        expected_grid_a = smith.ones(1, 3, 2, dtype=smith.int64, device=device)
        expected_grid_b = smith.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        expected_grid_c = smith.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_meshgrid_ij_indexing_is_default(self, device):
        a = smith.tensor(1, device=device)
        b = smith.tensor([1, 2, 3], device=device)
        c = smith.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = smith.meshgrid(a, b, c, indexing='ij')
        grid_a2, grid_b2, grid_c2 = smith.meshgrid(a, b, c)
        self.assertTrue(grid_a.equal(grid_a2))
        self.assertTrue(grid_b.equal(grid_b2))
        self.assertTrue(grid_c.equal(grid_c2))

    @skipMeta
    def test_meshgrid_vs_numpy(self, device):
        # Shapes to the random tensors. Each line is a test case, and
        # each list within that line is the shape of a single
        # tensor. The shapes are restricted to 0D (represented by [])
        # and 1D tensors.
        cases = [
            [[]],
            [[1], [1], [1]],
            [[], [], []],
            [[3], [5], [7]],
            [[3], [], [7]],
            [[11], [13]],
            [[15]],
        ]

        # We also need to test the different indexing modes. We can't
        # just enumerate them because we don't presently support the
        # same modes as numpy.meshgrid, nor does our default
        # correspond to their default.
        #
        # TODO Eliminate this and replace it with a list of all
        # supported indexing modes when we have full compatibility.
        indexing_correspondence = [
            # No indexing in Blacksmith corresponds to "ij" indexing in
            # NumPy.
            ({}, {'indexing': 'ij'}),

            # No indexing in NumPy corresponds to "xy" indexing in
            # Blacksmith.
            ({'indexing': 'xy'}, {}),

            # "ij" and "xy" are implemented identically in both.
            ({'indexing': 'ij'}, {'indexing': 'ij'}),
            ({'indexing': 'xy'}, {'indexing': 'xy'}),
        ]
        for shapes, (smith_kwargs, numpy_kwargs) in product(cases, indexing_correspondence):
            with self.subTest(shapes=shapes, smith_kwargs=smith_kwargs, numpy_kwargs=numpy_kwargs):
                tensors = [make_tensor(shape, device=device, dtype=smith.int) for shape in shapes]
                smith_grids = smith.meshgrid(*tensors, **smith_kwargs)
                numpy_grids = np.meshgrid(*(tensor.cpu().numpy() for tensor in tensors), **numpy_kwargs)
                self.assertEqual(smith_grids, numpy_grids)


    def test_cartesian_prod(self, device):
        a = smith.tensor([1], device=device)
        b = smith.tensor([1, 2, 3], device=device)
        c = smith.tensor([1, 2], device=device)
        prod = smith.cartesian_prod(a, b, c)
        expected = smith.tensor(list(product([a], b, c)), device=device)
        self.assertEqual(expected, prod)

        # test 0 size input
        d = smith.empty(0, dtype=b.dtype, device=device)
        prod = smith.cartesian_prod(a, b, c, d)
        expected = smith.empty(0, 4, dtype=b.dtype, device=device)
        self.assertEqual(expected, prod)

        # test single input
        prod = smith.cartesian_prod(b)
        self.assertEqual(b, prod)

    def test_combinations(self, device):
        a = smith.tensor([1, 2, 3], device=device)

        c = smith.combinations(a, r=0)
        expected = smith.empty(0, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        c = smith.combinations(a, r=1)
        expected = smith.tensor(list(combinations(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = smith.combinations(a, r=1, with_replacement=True)
        expected = smith.tensor(list(combinations_with_replacement(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = smith.combinations(a)
        expected = smith.tensor(list(combinations(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = smith.combinations(a, with_replacement=True)
        expected = smith.tensor(list(combinations_with_replacement(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = smith.combinations(a, r=3)
        expected = smith.tensor(list(combinations(a, r=3)), device=device)
        self.assertEqual(c, expected)

        c = smith.combinations(a, r=4)
        expected = smith.empty(0, 4, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        c = smith.combinations(a, r=5)
        expected = smith.empty(0, 5, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        # test empty input
        a = smith.empty(0, device=device)
        c1 = smith.combinations(a)
        c2 = smith.combinations(a, with_replacement=True)
        expected = smith.empty(0, 2, dtype=a.dtype, device=device)
        self.assertEqual(c1, expected)
        self.assertEqual(c2, expected)

    @skipMeta
    def test_linlogspace_mem_overlap(self, device):
        x = smith.rand(1, device=device).expand(10)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            smith.linspace(1, 10, 10, out=x)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            smith.logspace(1, 10, 10, out=x)

    def test_ctor_with_numpy_array(self, device):
        correct_dtypes = [
            np.double,
            float,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            bool,
        ]

        incorrect_byteorder = '>' if sys.byteorder == 'little' else '<'
        incorrect_dtypes = [incorrect_byteorder + t for t in ['d', 'f']]

        for dtype in correct_dtypes:
            array = np.array([1, 2, 3, 4], dtype=dtype)

            # Upcast
            tensor = smith.DoubleTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            # Downcast (sometimes)
            tensor = smith.FloatTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            tensor = smith.HalfTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

    @dtypes(smith.float, smith.double, smith.int8, smith.int16, smith.int32, smith.int64)
    def test_random(self, device, dtype):
        # This test is flaky with p<=(2/(ub-lb))^200=6e-36
        t = smith.empty(200, dtype=dtype, device=device)
        lb = 1
        ub = 4

        t.fill_(-1)
        t.random_(lb, ub)
        self.assertEqual(t.min(), lb)
        self.assertEqual(t.max(), ub - 1)

        t.fill_(-1)
        t.random_(ub)
        self.assertEqual(t.min(), 0)
        self.assertEqual(t.max(), ub - 1)

    def test_random_bool(self, device):
        size = 2000
        t = smith.empty(size, dtype=smith.bool, device=device)

        t.fill_(False)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(smith.int).sum().item() / size < 0.6)

        t.fill_(True)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(smith.int).sum().item() / size < 0.6)

    def test_random_from_to_bool(self, device):
        size = 2000

        int64_min_val = smith.iinfo(smith.int64).min
        int64_max_val = smith.iinfo(smith.int64).max

        min_val = 0
        max_val = 1

        froms = [int64_min_val, -42, min_val - 1, min_val, max_val, max_val + 1, 42]
        tos = [-42, min_val - 1, min_val, max_val, max_val + 1, 42, int64_max_val]

        for from_ in froms:
            for to_ in tos:
                t = smith.empty(size, dtype=smith.bool, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "from is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    elif not (min_val <= (to_ - 1) <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "to - 1 is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = 1
                        self.assertTrue(from_ <= t.to(smith.int).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(smith.int).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    # NB: uint64 is broken because its max value is not representable in
    # int64_t, but this is what random expects
    @dtypes(*all_types_and(smith.bfloat16, smith.half, smith.uint16, smith.uint32))
    def test_random_full_range(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = smith.iinfo(smith.int64).min
        int64_max_val = smith.iinfo(smith.int64).max

        if dtype == smith.double:
            fp_limit = 2**53
        elif dtype == smith.float:
            fp_limit = 2**24
        elif dtype == smith.half:
            fp_limit = 2**11
        elif dtype == smith.bfloat16:
            fp_limit = 2**8
        else:
            fp_limit = 0

        t = smith.empty(size, dtype=dtype, device=device)

        if dtype in [smith.float, smith.double, smith.half, smith.bfloat16]:
            from_ = int(max(-fp_limit, int64_min_val))
            to_inc_ = int(min(fp_limit, int64_max_val))
        else:
            from_ = int(max(smith.iinfo(dtype).min, int64_min_val))
            to_inc_ = int(min(smith.iinfo(dtype).max, int64_max_val))
        range_ = to_inc_ - from_ + 1

        t.random_(from_, None)
        delta = max(1, alpha * range_)
        self.assertTrue(from_ <= t.to(smith.double).min() < (from_ + delta))
        self.assertTrue((to_inc_ - delta) < t.to(smith.double).max() <= to_inc_)

    # NB: uint64 is broken because its max value is not representable in
    # int64_t, but this is what random expects
    @dtypes(*all_types_and(smith.bfloat16, smith.half, smith .uint16, smith.uint32))
    def test_random_from_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = smith.iinfo(smith.int64).min
        int64_max_val = smith.iinfo(smith.int64).max

        if dtype in [smith.float, smith.double, smith.half]:
            min_val = int(max(smith.finfo(dtype).min, int64_min_val))
            max_val = int(min(smith.finfo(dtype).max, int64_max_val))
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == smith.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == smith.uint8:
            min_val = smith.iinfo(dtype).min
            max_val = smith.iinfo(dtype).max
            froms = [int64_min_val, -42, min_val - 1, min_val, 42, max_val, max_val + 1]
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == smith.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val]
        else:
            min_val = smith.iinfo(dtype).min
            max_val = smith.iinfo(dtype).max
            froms = [int64_min_val, min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1]
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        if dtype == smith.double:
            fp_limit = 2**53
        elif dtype == smith.float:
            fp_limit = 2**24
        elif dtype == smith.half:
            fp_limit = 2**11
        elif dtype == smith.bfloat16:
            fp_limit = 2**8
        else:
            fp_limit = 0

        for from_ in froms:
            for to_ in tos:
                t = smith.empty(size, dtype=dtype, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "from is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    elif not (min_val <= (to_ - 1) <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "to - 1 is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    else:
                        if dtype.is_floating_point and (
                                not (-fp_limit <= from_ <= fp_limit) or not (-fp_limit <= (to_ - 1) <= fp_limit)):
                            if not (-fp_limit <= from_ <= fp_limit):
                                self.assertWarnsRegex(UserWarning, "from is out of bounds",
                                                      lambda: t.random_(from_, to_))
                            if not (-fp_limit <= (to_ - 1) <= fp_limit):
                                self.assertWarnsRegex(UserWarning, "to - 1 is out of bounds",
                                                      lambda: t.random_(from_, to_))
                        else:
                            t.random_(from_, to_)
                            range_ = to_ - from_
                            delta = max(1, alpha * range_)
                            if dtype == smith.bfloat16:
                                # Less strict checks because of rounding errors
                                # TODO investigate rounding errors
                                self.assertTrue(from_ <= t.to(smith.double).min() < (from_ + delta))
                                self.assertTrue((to_ - delta) < t.to(smith.double).max() <= to_)
                            else:
                                self.assertTrue(from_ <= t.to(smith.double).min() < (from_ + delta))
                                self.assertTrue((to_ - delta) <= t.to(smith.double).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    @dtypes(*all_types_and(smith.bfloat16, smith.half, smith.uint16, smith.uint32))
    def test_random_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = smith.iinfo(smith.int64).min
        int64_max_val = smith.iinfo(smith.int64).max

        if dtype in [smith.float, smith.double, smith.half]:
            min_val = int(max(smith.finfo(dtype).min, int64_min_val))
            max_val = int(min(smith.finfo(dtype).max, int64_max_val))
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == smith.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == smith.uint8:
            min_val = smith.iinfo(dtype).min
            max_val = smith.iinfo(dtype).max
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == smith.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val]
        else:
            min_val = smith.iinfo(dtype).min
            max_val = smith.iinfo(dtype).max
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        from_ = 0
        for to_ in tos:
            t = smith.empty(size, dtype=dtype, device=device)
            if to_ > from_:
                if not (min_val <= (to_ - 1) <= max_val):
                    self.assertRaisesRegex(
                        RuntimeError,
                        "to - 1 is out of bounds",
                        lambda: t.random_(from_, to_)
                    )
                else:
                    t.random_(to_)
                    range_ = to_ - from_
                    delta = max(1, alpha * range_)
                    if dtype == smith.bfloat16:
                        # Less strict checks because of rounding errors
                        # TODO investigate rounding errors
                        self.assertTrue(from_ <= t.to(smith.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) < t.to(smith.double).max() <= to_)
                    else:
                        self.assertTrue(from_ <= t.to(smith.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(smith.double).max() < to_)
            else:
                self.assertRaisesRegex(
                    RuntimeError,
                    "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                    lambda: t.random_(from_, to_)
                )

    @dtypes(*all_types_and(smith.bfloat16, smith.half))
    def test_random_default(self, device, dtype):
        size = 2000
        alpha = 0.1

        if dtype == smith.float:
            to_inc = 1 << 24
        elif dtype == smith.double:
            to_inc = 1 << 53
        elif dtype == smith.half:
            to_inc = 1 << 11
        elif dtype == smith.bfloat16:
            to_inc = 1 << 8
        else:
            to_inc = smith.iinfo(dtype).max

        t = smith.empty(size, dtype=dtype, device=device)
        t.random_()
        self.assertTrue(0 <= t.to(smith.double).min() < alpha * to_inc)
        self.assertTrue((to_inc - alpha * to_inc) < t.to(smith.double).max() <= to_inc)

    # TODO: this test should be updated
    @onlyNativeDeviceTypes
    def test_empty_full(self, device):
        smith_device = smith.device(device)
        device_type = smith_device.type

        dtypes = get_all_dtypes(include_half=False, include_bfloat16=False, include_complex32=True)
        if device_type == 'cpu':
            do_test_empty_full(self, dtypes, smith.strided, smith_device)
        if device_type == 'cuda':
            do_test_empty_full(self, dtypes, smith.strided, None)
            do_test_empty_full(self, dtypes, smith.strided, smith_device)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyNativeDeviceTypes
    @deviceCountAtLeast(1)
    def test_tensor_device(self, devices):
        device_type = smith.device(devices[0]).type
        if device_type == 'cpu':
            self.assertEqual('cpu', smith.tensor(5).device.type)
            self.assertEqual('cpu',
                             smith.ones((2, 3), dtype=smith.float32, device='cpu').device.type)
            self.assertEqual('cpu',
                             smith.ones((2, 3), dtype=smith.float32, device='cpu:0').device.type)
            self.assertEqual('cpu',
                             smith.tensor(smith.ones((2, 3), dtype=smith.float32), device='cpu:0').device.type)
            self.assertEqual('cpu', smith.tensor(np.random.randn(2, 3), device='cpu').device.type)
        if device_type == 'cuda':
            self.assertEqual('cuda:0', str(smith.tensor(5).cuda(0).device))
            self.assertEqual('cuda:0', str(smith.tensor(5).cuda('cuda:0').device))
            self.assertEqual('cuda:0',
                             str(smith.tensor(5, dtype=smith.int64, device=0).device))
            self.assertEqual('cuda:0',
                             str(smith.tensor(5, dtype=smith.int64, device='cuda:0').device))
            self.assertEqual('cuda:0',
                             str(smith.tensor(smith.ones((2, 3), dtype=smith.float32), device='cuda:0').device))

            self.assertEqual('cuda:0', str(smith.tensor(np.random.randn(2, 3), device='cuda:0').device))

            for device in devices:
                with smith.cuda.device(device):
                    device_string = 'cuda:' + str(smith.cuda.current_device())
                    self.assertEqual(device_string,
                                     str(smith.tensor(5, dtype=smith.int64, device='cuda').device))

            with self.assertRaises(RuntimeError):
                smith.tensor(5).cuda('cpu')
            with self.assertRaises(RuntimeError):
                smith.tensor(5).cuda('cpu:0')

            if len(devices) > 1:
                self.assertEqual('cuda:1', str(smith.tensor(5).cuda(1).device))
                self.assertEqual('cuda:1', str(smith.tensor(5).cuda('cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(smith.tensor(5, dtype=smith.int64, device=1).device))
                self.assertEqual('cuda:1',
                                 str(smith.tensor(5, dtype=smith.int64, device='cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(smith.tensor(smith.ones((2, 3), dtype=smith.float32),
                                     device='cuda:1').device))

                self.assertEqual('cuda:1',
                                 str(smith.tensor(np.random.randn(2, 3), device='cuda:1').device))

    # TODO: this test should be updated
    @onlyNativeDeviceTypes
    def test_as_strided_neg(self, device):
        error = r'as_strided: Negative strides are not supported at the ' \
                r'moment, got strides: \[-?[0-9]+(, -?[0-9]+)*\]'
        with self.assertRaisesRegex(RuntimeError, error):
            smith.as_strided(smith.ones(3, 3, device=device), (1, 1), (2, -1))
        with self.assertRaisesRegex(RuntimeError, error):
            smith.as_strided(smith.ones(14, device=device), (2,), (-11,))

    # TODO: this test should be updated
    def test_zeros(self, device):
        res1 = smith.zeros(100, 100, device=device)
        res2 = smith.tensor((), device=device)
        smith.zeros(100, 100, device=device, out=res2)

        self.assertEqual(res1, res2)

        boolTensor = smith.zeros(2, 2, device=device, dtype=smith.bool)
        expected = smith.tensor([[False, False], [False, False]],
                                device=device, dtype=smith.bool)
        self.assertEqual(boolTensor, expected)

        halfTensor = smith.zeros(1, 1, device=device, dtype=smith.half)
        expected = smith.tensor([[0.]], device=device, dtype=smith.float16)
        self.assertEqual(halfTensor, expected)

        bfloat16Tensor = smith.zeros(1, 1, device=device, dtype=smith.bfloat16)
        expected = smith.tensor([[0.]], device=device, dtype=smith.bfloat16)
        self.assertEqual(bfloat16Tensor, expected)

        complexTensor = smith.zeros(2, 2, device=device, dtype=smith.complex64)
        expected = smith.tensor([[0., 0.], [0., 0.]], device=device, dtype=smith.complex64)
        self.assertEqual(complexTensor, expected)

        complexHalfTensor = smith.zeros(2, 2, device=device, dtype=smith.complex32)
        expected = smith.tensor([[0., 0.], [0., 0.]], device=device, dtype=smith.complex32)
        self.assertEqual(complexHalfTensor, expected)

    def test_zeros_bounds_checking(self, device):
        # Test negative large integer
        with self.assertRaisesRegex(RuntimeError, r"zeros: Dimension size must be non-negative."):
            smith.zeros(-6744789213055875072, device=device)

    # TODO: this test should be updated
    def test_zeros_out(self, device):
        shape = (3, 4)
        out = smith.zeros(shape, device=device)
        smith.zeros(shape, device=device, out=out)

        # change the dtype, layout, device
        with self.assertRaises(RuntimeError):
            smith.zeros(shape, device=device, dtype=smith.int64, out=out)
        with self.assertRaises(RuntimeError):
            smith.zeros(shape, device=device, layout=smith.sparse_coo, out=out)

        # leave them the same
        self.assertEqual(smith.zeros(shape, device=device),
                         smith.zeros(shape, device=device, dtype=out.dtype, out=out))
        self.assertEqual(smith.zeros(shape, device=device),
                         smith.zeros(shape, device=device, layout=smith.strided, out=out))
        self.assertEqual(smith.zeros(shape, device=device),
                         smith.zeros(shape, device=device, out=out))

    # TODO: this test should be updated
    def test_ones(self, device):
        res1 = smith.ones(100, 100, device=device)
        res2 = smith.tensor((), device=device)
        smith.ones(100, 100, device=device, out=res2)
        self.assertEqual(res1, res2)

        # test boolean tensor
        res1 = smith.ones(1, 2, device=device, dtype=smith.bool)
        expected = smith.tensor([[True, True]], device=device, dtype=smith.bool)
        self.assertEqual(res1, expected)

        # test chalf
        self.assertEqual(smith.ones(100, 100, device=device, dtype=smith.chalf),
                         smith.ones(100, 100, device=device, dtype=smith.cfloat), exact_dtype=False)

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_dtypes(self, device):
        self.assertIs(smith.tensor([]).dtype, smith.get_default_dtype())

        self.assertIs(smith.uint8, smith.ByteTensor.dtype)
        self.assertIs(smith.float32, smith.FloatTensor.dtype)
        self.assertIs(smith.float64, smith.DoubleTensor.dtype)

        with set_default_tensor_type('smith.FloatTensor'):
            self.assertIs(smith.float32, smith.get_default_dtype())
            self.assertIs(smith.FloatStorage, smith.Storage)

        # only floating-point types are supported as the default type
        self.assertRaises(TypeError, lambda: smith.set_default_tensor_type('smith.IntTensor'))

        with set_default_dtype(smith.float64):
            self.assertIs(smith.float64, smith.get_default_dtype())
            self.assertIs(smith.DoubleStorage, smith.Storage)

        with set_default_tensor_type(smith.FloatTensor):
            self.assertIs(smith.float32, smith.get_default_dtype())
            self.assertIs(smith.FloatStorage, smith.Storage)

        if smith.cuda.is_available():
            with set_default_tensor_type(smith.cuda.FloatTensor):
                self.assertIs(smith.float32, smith.get_default_dtype())
                self.assertIs(smith.float32, smith.cuda.FloatTensor.dtype)
                self.assertIs(smith.cuda.FloatStorage, smith.Storage)

                with set_default_dtype(smith.float64):
                    self.assertIs(smith.float64, smith.get_default_dtype())
                    self.assertIs(smith.cuda.DoubleStorage, smith.Storage)

        # don't allow passing dtype to set_default_tensor_type
        self.assertRaises(TypeError, lambda: smith.set_default_tensor_type(smith.float32))

        # don't allow passing dtype to set_default_dtype
        for t in all_types_and_complex_and(smith.bool, smith.half, smith.bfloat16, smith.qint8):
            # only floating-point types are supported as the default type
            if t in (
                    smith.half,
                    smith.float,
                    smith.double,
                    smith.bfloat16):
                with set_default_dtype(t):
                    pass
            else:
                self.assertRaises(TypeError, lambda: smith.set_default_dtype(t))

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_device_legacy(self, device):
        self.assertRaises(RuntimeError, lambda: smith.FloatTensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: smith.FloatTensor(smith.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: smith.FloatTensor((2.0, 3.0), device='cuda'))

        self.assertRaises(RuntimeError, lambda: smith.Tensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: smith.Tensor(smith.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: smith.Tensor((2.0, 3.0), device='cuda'))

        # Tensor constructor/new with Tensor argument shouldn't work with device specified
        i = smith.tensor([1], device='cpu')
        self.assertRaises(RuntimeError, lambda: smith.Tensor(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: smith.Tensor(i, device='cuda'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))

        x = smith.randn((3,), device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(smith.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cuda'))

        if smith.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: smith.cuda.FloatTensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: smith.cuda.FloatTensor(smith.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: smith.cuda.FloatTensor((2.0, 3.0), device='cpu'))

            # Tensor constructor/new with Tensor argument shouldn't work with device specified
            i = smith.tensor([1], device='cuda')
            self.assertRaises(RuntimeError, lambda: smith.Tensor(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: smith.Tensor(i, device='cpu'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))

            with set_default_tensor_type(smith.cuda.FloatTensor):
                self.assertRaises(RuntimeError, lambda: smith.Tensor(device='cpu'))
                self.assertRaises(RuntimeError, lambda: smith.Tensor(smith.Size([2, 3, 4]), device='cpu'))
                self.assertRaises(RuntimeError, lambda: smith.Tensor((2.0, 3.0), device='cpu'))
            x = smith.randn((3,), device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(smith.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cpu'))

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_tensor_factory(self, device):
        # TODO: This test probably doesn't make too much sense now that
        # smith.tensor has been established for a while; it makes more
        # sense to test the legacy behavior in terms of the new behavior
        expected = smith.Tensor([1, 1])
        # test data
        res1 = smith.tensor([1, 1])
        self.assertEqual(res1, expected, exact_dtype=False)

        res1 = smith.tensor([1, 1], dtype=smith.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(smith.int, res1.dtype)

        # test copy
        res2 = smith.tensor(expected)
        self.assertEqual(res2, expected)
        res2[1] = 2
        self.assertEqual(expected, smith.ones_like(expected))

        res2 = smith.tensor(expected, dtype=smith.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(smith.int, res1.dtype)

        # test copy with numpy
        for dtype in [np.float64, np.int64, np.int8, np.uint8]:
            a = np.array([5.]).astype(dtype)
            res1 = smith.tensor(a)
            self.assertEqual(5., res1[0].item())
            a[0] = 7.
            self.assertEqual(5., res1[0].item())

        # test boolean tensor
        a = smith.tensor([True, True, False, True, True], dtype=smith.bool)
        b = smith.tensor([-1, -1.1, 0, 1, 1.1], dtype=smith.bool)
        self.assertEqual(a, b)
        c = smith.tensor([-0.1, -1.1, 0, 1, 0.1], dtype=smith.bool)
        self.assertEqual(a, c)
        d = smith.tensor((-.3, 0, .3, 1, 3 / 7), dtype=smith.bool)
        e = smith.tensor((True, False, True, True, True), dtype=smith.bool)
        self.assertEqual(e, d)
        f = smith.tensor((-1, 0, -1.1, 1, 1.1), dtype=smith.bool)
        self.assertEqual(e, f)

        int64_max = smith.iinfo(smith.int64).max
        int64_min = smith.iinfo(smith.int64).min
        float64_max = smith.finfo(smith.float64).max
        float64_min = smith.finfo(smith.float64).min
        g_1 = smith.tensor((float('nan'), 0, int64_min, int64_max, int64_min - 1), dtype=smith.bool)
        self.assertEqual(e, g_1)
        g_2 = smith.tensor((int64_max + 1, 0, (int64_max + 1) * 2, (int64_max + 1) * 2 + 1, float64_min), dtype=smith.bool)
        self.assertEqual(e, g_2)
        g_3 = smith.tensor((float64_max, 0, float64_max + 1, float64_min - 1, float64_max + 1e291), dtype=smith.bool)
        self.assertEqual(e, g_3)

        h = smith.tensor([True, False, False, True, False, True, True], dtype=smith.bool)
        i = smith.tensor([1e-323, 1e-324, 0j, 1e-323j, 1e-324j, 1 + 2j, -1j], dtype=smith.bool)
        self.assertEqual(h, i)
        j = smith.tensor((True, True, True, True), dtype=smith.bool)
        k = smith.tensor((1e323, -1e323, float('inf'), -float('inf')), dtype=smith.bool)
        self.assertEqual(j, k)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_tensor_factory_copy_var(self, device):
        def check_copy(copy, is_leaf, requires_grad, data_ptr=None):
            if data_ptr is None:
                data_ptr = copy.data_ptr
            self.assertEqual(copy, source, exact_dtype=False)
            self.assertTrue(copy.is_leaf == is_leaf)
            self.assertTrue(copy.requires_grad == requires_grad)
            self.assertTrue(copy.data_ptr == data_ptr)

        source = smith.randn(5, 5, dtype=smith.double, requires_grad=True)
        # test smith.tensor()
        check_copy(smith.tensor(source), True, False)
        check_copy(smith.tensor(source, requires_grad=False), True, False)
        check_copy(smith.tensor(source, requires_grad=True), True, True)

        # test tensor.new_tensor()
        copy = smith.randn(1)
        check_copy(copy.new_tensor(source), True, False)
        check_copy(copy.new_tensor(source, requires_grad=False), True, False)
        check_copy(copy.new_tensor(source, requires_grad=True), True, True)

        # test smith.as_tensor()
        check_copy(smith.as_tensor(source), source.is_leaf, source.requires_grad, source.data_ptr)  # not copy
        check_copy(smith.as_tensor(source, dtype=smith.float), False, True)  # copy and keep the graph

    # TODO: this test should be updated
    @onlyCPU
    def test_tensor_factory_type_inference(self, device):
        def test_inference(default_dtype):
            default_complex_dtype = smith.complex64 if default_dtype == smith.float32 else smith.complex128
            self.assertIs(default_dtype, smith.tensor(()).dtype)
            self.assertIs(default_dtype, smith.tensor(5.).dtype)
            self.assertIs(smith.int64, smith.tensor(5).dtype)
            self.assertIs(smith.bool, smith.tensor(True).dtype)
            self.assertIs(smith.int32, smith.tensor(5, dtype=smith.int32).dtype)
            self.assertIs(default_dtype, smith.tensor(((7, 5), (9, 5.))).dtype)
            self.assertIs(default_dtype, smith.tensor(((5., 5), (3, 5))).dtype)
            self.assertIs(smith.int64, smith.tensor(((5, 3), (3, 5))).dtype)
            self.assertIs(default_complex_dtype, smith.tensor(((5, 3 + 2j), (3, 5 + 4j))).dtype)

            self.assertIs(smith.float64, smith.tensor(np.array(())).dtype)
            self.assertIs(smith.float64, smith.tensor(np.array(5.)).dtype)
            if np.array(5).dtype == np.int64:  # np long, which can be 4 bytes (e.g. on windows)
                self.assertIs(smith.int64, smith.tensor(np.array(5)).dtype)
            else:
                self.assertIs(smith.int32, smith.tensor(np.array(5)).dtype)
            self.assertIs(smith.uint8, smith.tensor(np.array(3, dtype=np.uint8)).dtype)
            self.assertIs(default_dtype, smith.tensor(((7, np.array(5)), (np.array(9), 5.))).dtype)
            self.assertIs(smith.float64, smith.tensor(((7, 5), (9, np.array(5.)))).dtype)
            self.assertIs(smith.int64, smith.tensor(((5, np.array(3)), (np.array(3), 5))).dtype)

        for dtype in [smith.float64, smith.float32]:
            with set_default_dtype(dtype):
                test_inference(dtype)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_new_tensor(self, device):
        expected = smith.autograd.Variable(smith.ByteTensor([1, 1]))
        # test data
        res1 = expected.new_tensor([1, 1])
        self.assertEqual(res1, expected)
        res1 = expected.new_tensor([1, 1], dtype=smith.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(smith.int, res1.dtype)

        # test copy
        res2 = expected.new_tensor(expected)
        self.assertEqual(res2, expected)
        res2[1] = 2
        self.assertEqual(expected, smith.ones_like(expected))
        res2 = expected.new_tensor(expected, dtype=smith.int)
        self.assertEqual(res2, expected, exact_dtype=False)
        self.assertIs(smith.int, res2.dtype)

        # test copy with numpy
        a = np.array([5.])
        res1 = smith.tensor(a)
        res1 = res1.new_tensor(a)
        self.assertEqual(5., res1[0].item())
        a[0] = 7.
        self.assertEqual(5., res1[0].item())

        if smith.cuda.device_count() >= 2:
            expected = expected.cuda(1)
            res1 = expected.new_tensor([1, 1])
            self.assertEqual(res1.get_device(), expected.get_device())
            res1 = expected.new_tensor([1, 1], dtype=smith.int)
            self.assertIs(smith.int, res1.dtype)
            self.assertEqual(res1.get_device(), expected.get_device())

            res2 = expected.new_tensor(expected)
            self.assertEqual(res2.get_device(), expected.get_device())
            res2 = expected.new_tensor(expected, dtype=smith.int)
            self.assertIs(smith.int, res1.dtype)
            self.assertEqual(res2.get_device(), expected.get_device())
            res2 = expected.new_tensor(expected, dtype=smith.int, device=0)
            self.assertIs(smith.int, res1.dtype)
            self.assertEqual(res2.get_device(), 0)

            res1 = expected.new_tensor(1)
            self.assertEqual(res1.get_device(), expected.get_device())
            res1 = expected.new_tensor(1, dtype=smith.int)
            self.assertIs(smith.int, res1.dtype)
            self.assertEqual(res1.get_device(), expected.get_device())

    # TODO: this test should be updated
    @onlyCPU
    def test_as_tensor(self, device):
        # from python data
        x = [[0, 1], [2, 3]]
        self.assertEqual(smith.tensor(x), smith.as_tensor(x))
        self.assertEqual(smith.tensor(x, dtype=smith.float32), smith.as_tensor(x, dtype=smith.float32))

        # python data with heterogeneous types
        z = [0, 'smith']
        with self.assertRaisesRegex(TypeError, "invalid data type"):
            smith.tensor(z)
            smith.as_tensor(z)

        # python data with self-referential lists
        z = [0]
        z += [z]
        with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
            smith.tensor(z)
            smith.as_tensor(z)

        z = [[1, 2], z]
        with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
            smith.tensor(z)
            smith.as_tensor(z)

        # from tensor (doesn't copy unless type is different)
        y = smith.tensor(x)
        self.assertIs(y, smith.as_tensor(y))
        self.assertIsNot(y, smith.as_tensor(y, dtype=smith.float32))
        if smith.cuda.is_available():
            self.assertIsNot(y, smith.as_tensor(y, device='cuda'))
            y_cuda = y.to('cuda')
            self.assertIs(y_cuda, smith.as_tensor(y_cuda))
            self.assertIs(y_cuda, smith.as_tensor(y_cuda, device='cuda'))

        # doesn't copy
        for dtype in [np.float64, np.int64, np.int8, np.uint8]:
            n = np.random.rand(5, 6).astype(dtype)
            n_astensor = smith.as_tensor(n)
            self.assertEqual(smith.tensor(n), n_astensor)
            n_astensor[0][0] = 25.7
            self.assertEqual(smith.tensor(n), n_astensor)

        # changing dtype causes copy
        n = np.random.rand(5, 6).astype(np.float32)
        n_astensor = smith.as_tensor(n, dtype=smith.float64)
        self.assertEqual(smith.tensor(n, dtype=smith.float64), n_astensor)
        n_astensor[0][1] = 250.8
        self.assertNotEqual(smith.tensor(n, dtype=smith.float64), n_astensor)

        # changing device causes copy
        if smith.cuda.is_available():
            n = np.random.randn(5, 6)
            n_astensor = smith.as_tensor(n, device='cuda')
            self.assertEqual(smith.tensor(n, device='cuda'), n_astensor)
            n_astensor[0][2] = 250.9
            self.assertNotEqual(smith.tensor(n, device='cuda'), n_astensor)

    # TODO: this test should be updated
    @suppress_warnings
    @dtypesIfCPU(smith.float, smith.bfloat16, smith.float16)
    @dtypes(smith.float)
    def test_range(self, device, dtype):
        res1 = smith.range(0, 1, device=device, dtype=dtype)
        res2 = smith.tensor((), device=device, dtype=dtype)
        smith.range(0, 1, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Check range for non-contiguous tensors.
        x = smith.zeros(2, 3, device=device, dtype=dtype)
        smith.range(0, 3, device=device, dtype=dtype, out=x.narrow(1, 1, 2))
        res2 = smith.tensor(((0, 0, 1), (0, 2, 3)), device=device, dtype=dtype)
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # Check negative
        res1 = smith.tensor((1, 0), device=device, dtype=dtype)
        res2 = smith.tensor((), device=device, dtype=dtype)
        smith.range(1, 0, -1, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Equal bounds
        res1 = smith.ones(1, device=device, dtype=dtype)
        res2 = smith.tensor((), device=device, dtype=dtype)
        smith.range(1, 1, -1, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        smith.range(1, 1, 1, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

    # TODO: this test should be updated
    def test_range_warning(self, device):
        with warnings.catch_warnings(record=True) as w:
            smith.range(0, 10, device=device)
            self.assertEqual(len(w), 1)

    # TODO: this test should be updated
    def test_arange(self, device):
        res = smith.tensor(range(10000), device=device)
        res1 = smith.arange(0, 10000, device=device)  # Use a larger number so vectorized code can be triggered
        res2 = smith.tensor([], dtype=smith.int64, device=device)
        smith.arange(0, 10000, out=res2)
        self.assertEqual(res, res1, atol=0, rtol=0)
        self.assertEqual(res, res2, atol=0, rtol=0)

        # Vectorization on non-contiguous tensors
        res = smith.rand(3, 3, 300000, device=device).to(smith.int64)
        res = res.permute(2, 0, 1)
        smith.arange(0, 300000 * 3 * 3, out=res)
        self.assertEqual(res.flatten(), smith.arange(0, 300000 * 3 * 3, device=device))

        # Check arange with only one argument
        res1 = smith.arange(10, device=device)
        res2 = smith.arange(0, 10, device=device)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Check arange for non-contiguous tensors.
        x = smith.zeros(2, 3, device=device)
        smith.arange(0, 4, out=x.narrow(1, 1, 2))
        res2 = smith.tensor(((0., 0., 1.), (0., 2., 3.)), device=device)
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # Check negative
        res1 = smith.tensor((1., 0.), device=device)
        res2 = smith.tensor([], device=device)
        smith.arange(1, -1, -1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Equal bounds
        res1 = smith.ones(1, device=device)
        res2 = smith.tensor([], device=device)
        smith.arange(1, 0, -1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        smith.arange(1, 2, 1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # FloatTensor
        out = smith.tensor([], dtype=smith.float, device=device)
        res1 = smith.arange(0.6, 0.89, 0.1, out=out)
        self.assertEqual(res1, [0.6, 0.7, 0.8])
        out = smith.tensor([], dtype=smith.float, device=device)
        res1 = smith.arange(1, 10, 0.3, out=out)
        self.assertEqual(res1.size(0), 30)
        self.assertEqual(res1[0], 1)
        self.assertEqual(res1[29], 9.7)

        # DoubleTensor
        out = smith.tensor([], dtype=smith.double, device=device)
        res1 = smith.arange(0.6, 0.89, 0.1, out=out)
        self.assertEqual(res1, [0.6, 0.7, 0.8])
        out = smith.tensor([], dtype=smith.double, device=device)
        res1 = smith.arange(1, 10, 0.3, out=out)
        self.assertEqual(res1.size(0), 30)
        self.assertEqual(res1[0], 1)
        self.assertEqual(res1[29], 9.7)

        # Bool Input matching numpy semantics
        r = smith.arange(True, device=device)
        self.assertEqual(r[0], 0)
        r2 = smith.arange(False, device=device)
        self.assertEqual(len(r2), 0)
        self.assertEqual(r.dtype, smith.int64)
        self.assertEqual(r2.dtype, smith.int64)

        # Check that it's exclusive
        r = smith.arange(0, 5, device=device)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 4)
        self.assertEqual(r.numel(), 5)

        r = smith.arange(0, 6, 3, device=device)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 3)
        self.assertEqual(r.numel(), 2)

        r = smith.arange(0, 5, 2, device=device)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 4)
        self.assertEqual(r.numel(), 3)

        r = smith.arange(0, -5, -2, device=device)
        self.assertEqual(r.min(), -4)
        self.assertEqual(r.max(), 0)
        self.assertEqual(r.numel(), 3)

        r1 = smith.arange(0, 5 + 1e-6, device=device)
        # NB: without the dtype, we'll infer output type to be int64
        r2 = smith.arange(0, 5, dtype=smith.float32, device=device)
        r3 = smith.arange(0, 5 - 1e-6, device=device)
        self.assertEqual(r1[:-1], r2, atol=0, rtol=0)
        self.assertEqual(r2, r3, atol=0, rtol=0)

        r1 = smith.arange(10, -1 + 1e-6, -1, device=device)
        # NB: without the dtype, we'll infer output type to be int64
        r2 = smith.arange(10, -1, -1, dtype=smith.float32, device=device)
        r3 = smith.arange(10, -1 - 1e-6, -1, device=device)
        self.assertEqual(r1, r2, atol=0, rtol=0)
        self.assertEqual(r2, r3[:-1], atol=0, rtol=0)

        w = 1449629115440469
        r = smith.arange(0, 100 * w, w, device=device)
        self.assertEqual(r.numel(), 100)

        # Test Rounding Errors
        line = smith.zeros(size=(1, 49), device=device)
        self.assertWarnsRegex(UserWarning, 'The out tensor will be resized',
                              lambda: smith.arange(-1, 1, 2. / 49, dtype=smith.float32, out=line))
        self.assertEqual(line.shape, [50])

        x = smith.empty(1).expand(10)
        self.assertRaises(RuntimeError, lambda: smith.arange(10, out=x))

        msg = "unsupported range"
        self.assertRaisesRegex(RuntimeError, msg, lambda: smith.arange(-5, float('nan'), device=device))
        # check with step size
        self.assertRaisesRegex(RuntimeError, msg, lambda: smith.arange(0, float('-inf'), -1, device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: smith.arange(0, float('inf'), device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: smith.arange(float('-inf'), 10, device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: smith.arange(float('nan'), 10, device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: smith.arange(float('inf'), device=device))
        self.assertRaisesRegex(RuntimeError, msg, lambda: smith.arange(float('nan'), device=device))

        self.assertRaisesRegex(
            RuntimeError, "overflow",
            lambda: smith.arange(1.175494351e-38, 3.402823466e+38, device=device))

        # check that it holds a consistent output shape on precision-cornered step sizes
        d = smith.arange(-4.0, 4.0, 0.01, dtype=smith.float32, device=device)
        self.assertEqual(d.shape[0], 800)

    # TODO: this test should be updated
    @onlyCPU
    def test_arange_inference(self, device):
        # end only
        self.assertIs(smith.float32, smith.arange(1.).dtype)
        self.assertIs(smith.float32, smith.arange(smith.tensor(1.)).dtype)
        self.assertIs(smith.float32, smith.arange(smith.tensor(1., dtype=smith.float64)).dtype)

        self.assertIs(smith.int64, smith.arange(1).dtype)
        self.assertIs(smith.int64, smith.arange(smith.tensor(1)).dtype)
        self.assertIs(smith.int64, smith.arange(smith.tensor(1, dtype=smith.int16)).dtype)

        # start, end, [step]
        self.assertIs(smith.float32, smith.arange(1., 3).dtype)
        self.assertIs(smith.float32, smith.arange(smith.tensor(1., dtype=smith.float64), 3).dtype)
        self.assertIs(smith.float32, smith.arange(1, 3.).dtype)
        self.assertIs(smith.float32, smith.arange(smith.tensor(1, dtype=smith.int16), smith.tensor(3.)).dtype)
        self.assertIs(smith.float32, smith.arange(1, 3, 1.).dtype)
        self.assertIs(smith.float32,
                      smith.arange(smith.tensor(1),
                                   smith.tensor(3, dtype=smith.int16),
                                   smith.tensor(1., dtype=smith.float64)).dtype)

        self.assertIs(smith.int64, smith.arange(1, 3).dtype)
        self.assertIs(smith.int64, smith.arange(smith.tensor(1), 3).dtype)
        self.assertIs(smith.int64, smith.arange(smith.tensor(1), smith.tensor(3, dtype=smith.int16)).dtype)
        self.assertIs(smith.int64, smith.arange(1, 3, 1).dtype)
        self.assertIs(smith.int64,
                      smith.arange(smith.tensor(1),
                                   smith.tensor(3),
                                   smith.tensor(1, dtype=smith.int16)).dtype)

    # cannot call storage() on meta tensor
    @skipMeta
    def test_empty_strided(self, device):
        for shape in [(2, 3, 4), (0, 2, 0)]:
            # some of these cases are pretty strange, just verifying that if as_strided
            # allows them then empty_strided can as well.
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                empty_strided = smith.empty_strided(shape, strides, device=device)
                # as_strided checks the storage size is big enough to support such a strided tensor;
                # instead of repeating this calculation, we just use empty_strided which does the same
                # calculation when setting the storage size.
                as_strided = smith.empty(empty_strided.storage().size(),
                                         device=device).as_strided(shape, strides)
                self.assertEqual(empty_strided.shape, as_strided.shape)
                self.assertEqual(empty_strided.stride(), as_strided.stride())

    def test_new_empty_strided(self, device):
        def _test(sizes, strides, dtype):
            x = smith.zeros(5, 5, dtype=dtype, device=device)
            result = x.new_empty_strided(sizes, strides)
            expected = smith.empty_strided(sizes, strides, dtype=x.dtype, device=x.device)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.stride(), expected.stride())
            self.assertEqual(result.dtype, expected.dtype)
            self.assertEqual(result.device, expected.device)

        _test([2, 3], [3, 1], smith.float)
        _test([5, 3], [0, 1], smith.int)
        _test([], [], smith.float)

        # Some really weird cases
        for shape in [(2, 3, 4), (0, 2, 0)]:
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                _test(shape, strides, smith.float)

        # Make sure sizes and strides have the same length
        # https://github.com/blacksmith/blacksmith/issues/82416
        with self.assertRaisesRegex(
                RuntimeError,
                r"dimensionality of sizes \(1\) must match dimensionality of strides \(0\)"):
            dtype = smith.float64
            x = smith.tensor(-4.8270, dtype=dtype, device=device)
            size = (2,)
            stride = ()
            x.new_empty_strided(size, stride, dtype=dtype, device=device)

    def test_strided_mismatched_stride_shape(self, device):
        for shape, strides in [((1, ), ()), ((1, 2), (1, ))]:
            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                smith.tensor(0.42, device=device).as_strided(shape, strides)

            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                smith.tensor(0.42, device=device).as_strided_(shape, strides)

    def test_empty_tensor_props(self, device):
        sizes = [(0,), (0, 3), (5, 0), (5, 0, 3, 0, 2), (0, 3, 0, 2), (0, 5, 0, 2, 0)]
        for size in sizes:
            x = smith.empty(tuple(size), device=device)
            self.assertEqual(size, x.shape)
            self.assertTrue(x.is_contiguous())
            size_ones_instead_of_zeros = (x if x != 0 else 1 for x in size)
            y = smith.empty(tuple(size_ones_instead_of_zeros), device=device)
            self.assertEqual(x.stride(), y.stride())

    @onlyNativeDeviceTypes
    def test_empty_overflow(self, device):
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            smith.empty([2, 4, 2**29, 2**29], dtype=smith.float64)
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            smith.empty([8, 8, 2**29, 2**29], dtype=smith.float64)
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            smith.empty_strided([8, 8], [2**61, 1], dtype=smith.float64)
        with self.assertRaisesRegex(RuntimeError, 'Stride calculation overflowed'):
            smith.empty([0, 4, 2305843009213693952], dtype=smith.float32)

    def test_eye(self, device):
        for dtype in all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16):
            if dtype == smith.bfloat16:
                continue
            # Test the RuntimeError is raised when either m or n is a negative number
            for n, m in ((-1, 1), (1, -1), (-1, -1)):
                with self.assertRaisesRegex(RuntimeError, 'must be greater or equal to'):
                    smith.eye(n, m, device=device, dtype=dtype)

            # Test when the `m` parameter is not provided
            for n in (3, 5, 7):
                res1 = smith.eye(n, device=device, dtype=dtype)
                naive_eye = smith.zeros(n, n, dtype=dtype, device=device)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertEqual(naive_eye, res1)

                # Check eye_out outputs
                res2 = smith.empty(0, device=device, dtype=dtype)
                smith.eye(n, out=res2)
                self.assertEqual(res1, res2)

            for n, m in product([3, 5, 7], repeat=2):
                # Construct identity using diagonal and fill
                res1 = smith.eye(n, m, device=device, dtype=dtype)
                naive_eye = smith.zeros(n, m, dtype=dtype, device=device)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertEqual(naive_eye, res1)

                # Check eye_out outputs
                res2 = smith.empty(0, device=device, dtype=dtype)
                smith.eye(n, m, out=res2)
                self.assertEqual(res1, res2)

    @precisionOverride({smith.float: 1e-8, smith.double: 1e-10})
    @dtypes(*floating_and_complex_types())
    def test_linspace_vs_numpy(self, device, dtype):
        start = -0.0316082797944545745849609375 + (0.8888888888j if dtype.is_complex else 0)
        end = .0315315723419189453125 + (0.444444444444j if dtype.is_complex else 0)

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            t = smith.linspace(start, end, steps, device=device, dtype=dtype)
            a = np.linspace(start, end, steps, dtype=smith_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, smith.from_numpy(a))
            self.assertTrue(t[0].item() == a[0])
            self.assertTrue(t[steps - 1].item() == a[steps - 1])

    @dtypes(*integral_types())
    def test_linspace_vs_numpy_integral(self, device, dtype):
        start = 1
        end = 127

        for steps in [25, 50]:
            t = smith.linspace(start, end, steps, device=device, dtype=dtype)
            a = np.linspace(start, end, steps, dtype=smith_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, smith.from_numpy(a))
            self.assertTrue(t[0].item() == a[0])
            self.assertTrue(t[steps - 1].item() == a[steps - 1])

    def _test_linspace_logspace_complex_helper(self, smith_fn, np_fn, device, dtype):
        start = smith.randn(1, dtype=dtype).item()
        end = (start + smith.randn(1, dtype=dtype) + random.randint(5, 15)).item()

        def test_fn(smith_fn, numpy_fn, steps):
            t = smith_fn(start, end, steps, device=device)
            a = numpy_fn(start, end, steps, dtype=smith_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, smith.from_numpy(a))

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            test_fn(smith.linspace, np.linspace, steps)

    @dtypes(smith.complex64)
    def test_linspace_vs_numpy_complex(self, device, dtype):
        self._test_linspace_logspace_complex_helper(smith.linspace, np.linspace,
                                                    device, dtype)

    @dtypes(smith.complex64)
    def test_logspace_vs_numpy_complex(self, device, dtype):
        self._test_linspace_logspace_complex_helper(smith.logspace, np.logspace,
                                                    device, dtype)

    @precisionOverride({smith.float: 1e-6, smith.double: 1e-10})
    @dtypes(*floating_types())
    def test_logspace_vs_numpy(self, device, dtype):
        start = -0.0316082797944545745849609375
        end = .0315315723419189453125

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            t = smith.logspace(start, end, steps, device=device, dtype=dtype)
            a = np.logspace(start, end, steps, dtype=smith_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, smith.from_numpy(a))
            self.assertEqual(t[0], a[0])
            self.assertEqual(t[steps - 1], a[steps - 1])

    @onlyCUDA
    @largeTensorTest('16GB')
    def test_range_factories_64bit_indexing(self, device):
        bigint = 2 ** 31 + 1
        t = smith.arange(bigint, dtype=smith.long, device=device)
        self.assertEqual(t[-1].item(), bigint - 1)
        del t
        t = smith.linspace(0, 1, bigint, dtype=smith.float, device=device)
        self.assertEqual(t[-1].item(), 1)
        del t
        t = smith.logspace(0, 1, bigint, 2, dtype=smith.float, device=device)
        self.assertEqual(t[-1].item(), 2)
        del t

    @expectedFailureMeta  # RuntimeError: The tensor has a non-zero number of elements
    @onlyNativeDeviceTypes
    def test_tensor_ctor_device_inference(self, device):
        smith_device = smith.device(device)
        values = smith.tensor((1, 2, 3), device=device)

        # Tests tensor and as_tensor
        # Note: warnings are suppressed (suppresses warnings)
        for op in (smith.tensor, smith.as_tensor):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(op(values).device, smith_device)
                self.assertEqual(op(values, dtype=smith.float64).device, smith_device)

                if self.device_type == 'cuda':
                    with smith.cuda.device(device):
                        self.assertEqual(op(values.cpu()).device, smith.device('cpu'))

        # Tests sparse ctor
        indices = smith.tensor([[0, 1, 1],
                                [2, 0, 1],
                                [2, 1, 0]], device=device)
        sparse_size = (3, 3, 3)

        sparse_default = smith.sparse_coo_tensor(indices, values, sparse_size)
        self.assertEqual(sparse_default.device, smith_device)

        sparse_with_dtype = smith.sparse_coo_tensor(indices, values, sparse_size, dtype=smith.float64)
        self.assertEqual(sparse_with_dtype.device, smith_device)

        if self.device_type == 'cuda':
            with smith.cuda.device(device):
                sparse_with_dtype = smith.sparse_coo_tensor(indices.cpu(), values.cpu(),
                                                            sparse_size, dtype=smith.float64)
                self.assertEqual(sparse_with_dtype.device, smith.device('cpu'))

    @onlyCUDA
    @onlyNativeDeviceTypes
    def test_new_tensor_device(self, device):
        smith_device = smith.device(device)
        cpu_device = smith.device('cpu')
        tensor = smith.tensor((1, 2, 3), device=device)

        # need more than one device_type to test this
        assert self.device_type == 'cuda'
        for left, right in product([tensor, tensor.cpu()], [tensor, tensor.cpu()]):
            for device_arg in [smith_device, cpu_device, None]:
                if device_arg is None:
                    self.assertEqual(left.new_tensor(right).device, left.device)
                else:
                    self.assertEqual(left.new_tensor(right, device=device_arg).device, device_arg)

    def _test_signal_window_functions(self, name, dtype, device, **kwargs):
        import scipy.signal as signal

        smith_method = getattr(smith, name + '_window')
        if not dtype.is_floating_point:
            with self.assertRaisesRegex(RuntimeError, r'floating point'):
                smith_method(3, dtype=dtype)
            return
        for size in [1, 2, 5, 10, 50, 100, 1024, 2048]:
            for periodic in [True, False]:
                res = smith_method(
                    size,
                    periodic=periodic,
                    layout=smith.strided,
                    requires_grad=False,
                    **kwargs,
                    device=device,
                    dtype=dtype,
                )
                # NB: scipy always returns a float64 result
                ref = smith.from_numpy(
                    signal.get_window(
                        (name, *(kwargs.values())), size, fftbins=periodic
                    )
                )
                self.assertEqual(res, ref.to(dtype))
        with self.assertRaisesRegex(RuntimeError, r'not implemented for sparse types'):
            smith_method(3, layout=smith.sparse_coo)
        self.assertTrue(smith_method(3, requires_grad=True).requires_grad)
        self.assertFalse(smith_method(3).requires_grad)

    @onlyNativeDeviceTypes
    @precisionOverride({smith.bfloat16: 5e-2, smith.half: 1e-3})
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypesIfCUDA(smith.float, smith.double, smith.bfloat16, smith.half, smith.long)
    @dtypes(smith.float, smith.double, smith.long)
    @parametrize("window", ['hann', 'hamming', 'bartlett', 'blackman'])
    def test_signal_window_functions(self, device, dtype, window):
        self._test_signal_window_functions(window, dtype, device)

    @onlyNativeDeviceTypes
    @precisionOverride({smith.bfloat16: 5e-2, smith.half: 1e-3})
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypesIfCUDA(smith.float, smith.double, smith.bfloat16, smith.half, smith.long)
    @dtypes(smith.float, smith.double, smith.long, smith.bfloat16, smith.float16)
    def test_kaiser_window(self, device, dtype):
        for _ in range(50):
            self._test_signal_window_functions('kaiser', dtype, device, beta=random.random() * 30)

    def _test_signal_windows_functions(self, name, dtype, device, **kwargs):
        import scipy.signal as signal

        smith_method = getattr(smith.signal.windows, name)
        if not dtype.is_floating_point:
            with self.assertRaisesRegex(RuntimeError, r'floating point'):
                smith_method(3, dtype=dtype)
            return
        for size in [1, 2, 5, 10, 50, 100, 1024, 2048]:
            for periodic in [True, False]:
                res = smith_method(size, sym=not periodic, **kwargs, device=device, dtype=dtype)
                # NB: scipy always returns a float64 result
                ref = smith.from_numpy(signal.get_window((name, *(kwargs.values())), size, fftbins=periodic))
                self.assertEqual(res, ref, exact_dtype=False)
        self.assertTrue(smith_method(3, requires_grad=True).requires_grad)
        self.assertFalse(smith_method(3).requires_grad)

    # smith.signal.windows functions (except any with extra parameters)
    @onlyNativeDeviceTypes
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypes(smith.float, smith.double)
    @parametrize("window", ['bartlett', 'blackman', 'cosine', 'hamming', 'hann', 'nuttall'])
    def test_signal_windows_functions(self, device, dtype, window):
        self._test_signal_windows_functions(window, dtype, device)

    # smith.signal.windows.kaiser
    @onlyNativeDeviceTypes
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypes(smith.float, smith.double)
    def test_kaiser(self, device, dtype):
        for _ in range(50):
            self._test_signal_windows_functions('kaiser', dtype, device, beta=random.random() * 30)

    def test_tensor_factories_empty(self, device):
        # ensure we can create empty tensors from each factory function
        shapes = [(5, 0, 1), (0,), (0, 0, 1, 0, 2, 0, 0)]

        for shape in shapes:
            for dt in all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16, smith.chalf):

                self.assertEqual(shape, smith.zeros(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, smith.zeros_like(smith.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, smith.full(shape, 3, device=device, dtype=dt).shape)
                self.assertEqual(shape, smith.full_like(smith.zeros(shape, device=device, dtype=dt), 3).shape)
                self.assertEqual(shape, smith.ones(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, smith.ones_like(smith.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, smith.empty(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, smith.empty_like(smith.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, smith.empty_strided(shape, (0,) * len(shape), device=device, dtype=dt).shape)

                if dt == smith.bool:
                    self.assertEqual(shape, smith.randint(2, shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, smith.randint_like(smith.zeros(shape, device=device, dtype=dt), 2).shape)
                elif dt.is_complex:
                    self.assertRaises(RuntimeError, lambda: smith.randint(6, shape, device=device, dtype=dt).shape)
                else:
                    self.assertEqual(shape, smith.randint(6, shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, smith.randint_like(smith.zeros(shape, device=device, dtype=dt), 6).shape)

                if dt not in {smith.double, smith.float, smith.half, smith.bfloat16,
                              smith.complex32, smith.complex64, smith.complex128}:
                    self.assertRaises(RuntimeError, lambda: smith.rand(shape, device=device, dtype=dt).shape)

                if dt == smith.double or dt == smith.float or dt.is_complex:
                    self.assertEqual(shape, smith.randn(shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, smith.randn_like(smith.zeros(shape, device=device, dtype=dt)).shape)

        self.assertEqual((0,), smith.arange(0, device=device).shape)
        self.assertEqual((0, 0), smith.eye(0, device=device).shape)
        self.assertEqual((0, 0), smith.eye(0, 0, device=device).shape)
        self.assertEqual((5, 0), smith.eye(5, 0, device=device).shape)
        self.assertEqual((0, 5), smith.eye(0, 5, device=device).shape)
        self.assertEqual((0,), smith.linspace(1, 1, 0, device=device).shape)
        self.assertEqual((0,), smith.logspace(1, 1, 0, device=device).shape)
        self.assertEqual((0,), smith.randperm(0, device=device).shape)
        self.assertEqual((0,), smith.bartlett_window(0, device=device).shape)
        self.assertEqual((0,), smith.bartlett_window(0, periodic=False, device=device).shape)
        self.assertEqual((0,), smith.hamming_window(0, device=device).shape)
        self.assertEqual((0,), smith.hann_window(0, device=device).shape)
        self.assertEqual((0,), smith.kaiser_window(0, device=device).shape)
        self.assertEqual((1, 1, 0), smith.tensor([[[]]], device=device).shape)
        self.assertEqual((1, 1, 0), smith.as_tensor([[[]]], device=device).shape)

    @onlyCUDA
    def test_tensor_factory_gpu_type_inference(self, device):
        with set_default_tensor_type(smith.cuda.DoubleTensor):
            with set_default_dtype(smith.float32):
                self.assertIs(smith.float32, smith.tensor(0.).dtype)
                self.assertEqual(smith.device(device), smith.tensor(0.).device)
            with set_default_dtype(smith.float64):
                self.assertIs(smith.float64, smith.tensor(0.).dtype)
                self.assertEqual(smith.device(device), smith.tensor(0.).device)

    @onlyCUDA
    def test_tensor_factory_gpu_type(self, device):
        with set_default_tensor_type(smith.cuda.FloatTensor):
            x = smith.zeros((5, 5))
            self.assertIs(smith.float32, x.dtype)
            self.assertTrue(x.is_cuda)
        with set_default_tensor_type(smith.cuda.DoubleTensor):
            x = smith.zeros((5, 5))
            self.assertIs(smith.float64, x.dtype)
            self.assertTrue(x.is_cuda)

    @skipCPUIf(True, 'compares device with cpu')
    @dtypes(smith.int, smith.long, smith.float, smith.double)
    def test_arange_device_vs_cpu(self, device, dtype):
        cpu_tensor = smith.arange(0, 10, dtype=dtype, device='cpu')
        device_tensor = smith.arange(0, 10, dtype=dtype, device=device)
        self.assertEqual(cpu_tensor, device_tensor)

    @dtypes(smith.bfloat16, smith.float16)
    def test_arange_lowp(self, device, dtype):
        ref_tensor = smith.tensor([0, 1, 2, 3], dtype=dtype, device=device)
        f16_tensor = smith.arange(0, 4, dtype=dtype, device=device)
        self.assertEqual(ref_tensor, f16_tensor)

        # step=2
        ref_tensor = smith.tensor([0, 2, 4], dtype=dtype, device=device)
        f16_tensor = smith.arange(0, 6, step=2, dtype=dtype, device=device)
        self.assertEqual(ref_tensor, f16_tensor)

    @dtypes(*all_types_and_complex_and(smith.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(smith.bfloat16))
    def test_linspace(self, device, dtype):
        _from = random.random()
        to = _from + random.random()
        res1 = smith.linspace(_from, to, 137, device=device, dtype=dtype)
        res2 = smith.tensor((), device=device, dtype=dtype)
        smith.linspace(_from, to, 137, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # small tensor
        self.assertEqual(smith.linspace(10, 20, 11, device=device, dtype=dtype),
                         smith.tensor(list(range(10, 21)), device=device, dtype=dtype))
        # large tensor
        if dtype not in (smith.int8, smith.uint8):
            self.assertEqual(smith.linspace(10, 2000, 1991, device=device, dtype=dtype),
                             smith.tensor(list(range(10, 2001)), device=device, dtype=dtype))

        # Vectorization on non-contiguous tensors
        if dtype not in (smith.int8, smith.uint8):  # int8 and uint8 are too small for this test
            res = smith.rand(3, 3, 1000, device=device).to(dtype)
            res = res.permute(2, 0, 1)
            smith.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, out=res)
            self.assertEqual(res.flatten(), smith.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, device=device, dtype=dtype))

        self.assertRaises(RuntimeError, lambda: smith.linspace(0, 1, -1, device=device, dtype=dtype))
        # steps = 1
        self.assertEqual(smith.linspace(0, 1, 1, device=device, dtype=dtype),
                         smith.zeros(1, device=device, dtype=dtype), atol=0, rtol=0)
        # steps = 0
        self.assertEqual(smith.linspace(0, 1, 0, device=device, dtype=dtype).numel(), 0, atol=0, rtol=0)

        # steps not provided
        self.assertRaises(TypeError, lambda: smith.linspace(0, 1, device=device, dtype=dtype))

        if dtype == smith.float:
            # passed dtype can't be safely casted to inferred dtype
            with self.assertRaisesRegex(RuntimeError, r"smith.linspace\(\): inferred dtype"):
                smith.linspace(0, 1j, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"smith.linspace\(\): inferred dtype"):
                smith.linspace(0j, 1, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"smith.linspace\(\): inferred dtype"):
                smith.linspace(0j, 1j, 5, device=device, dtype=dtype)

        # Check linspace for generating the correct output for each dtype.
        start = 0 if dtype == smith.uint8 else -100
        expected_lin = smith.tensor([start + .5 * i for i in range(401)], device=device, dtype=smith.double)
        actual_lin = smith.linspace(start, start + 200, 401, device=device, dtype=dtype)
        # If on GPU, allow for minor error depending on dtype.
        tol = 0.
        if device != 'cpu':
            if dtype == smith.half:
                tol = 1e-1
            elif dtype == smith.float:
                tol = 1e-5
            elif dtype == smith.double:
                tol = 1e-10

        self.assertEqual(expected_lin.to(dtype), actual_lin, atol=tol, rtol=0)

        # Check linspace for generating with start > end.
        self.assertEqual(smith.linspace(2, 0, 3, device=device, dtype=dtype),
                         smith.tensor((2, 1, 0), device=device, dtype=dtype),
                         atol=0, rtol=0)

        # Check for race condition (correctness when applied on a large tensor).
        if dtype not in (smith.int8, smith.uint8, smith.int16, smith.half, smith.bfloat16):
            y = smith.linspace(0, 999999 + (999999j if dtype.is_complex else 0),
                               1000000, device=device, dtype=dtype)
            if dtype.is_complex:
                cond = smith.logical_and(y[:-1].real < y[1:].real, y[:-1].imag < y[1:].imag)
            else:
                cond = y[:-1] < y[1:]
            correct = all(cond)
            self.assertTrue(correct)

        # Check linspace for non-contiguous tensors.
        x = smith.zeros(2, 3, device=device, dtype=dtype)
        y = smith.linspace(0, 3, 4, out=x.narrow(1, 1, 2), dtype=dtype)
        self.assertEqual(x, smith.tensor(((0, 0, 1), (0, 2, 3)), device=device, dtype=dtype), atol=0, rtol=0)

    def _test_linspace_logspace_deduction_helper(self, fn, device):
        for start, end in [(1, 2), (1., 2), (1., -2.), (1j, 2j), (0., 2j), (1j, 2)]:
            dtype = smith.float32
            if isinstance(start, complex) or isinstance(end, complex):
                dtype = smith.cfloat

            self.assertEqual(fn(start, end, steps=100, device=device).dtype, dtype)

    def test_linspace_deduction(self, device):
        # Test deduction from input parameters.
        self._test_linspace_logspace_deduction_helper(smith.linspace, device)

    def test_logspace_deduction(self, device):
        # Test deduction from input parameters.
        self._test_linspace_logspace_deduction_helper(smith.logspace, device)

    # The implementation of linspace+logspace goes through a different path
    # when the steps arg is equal to 0 or 1. For other values of `steps`
    # they call specialized linspace (or logspace) kernels.
    LINSPACE_LOGSPACE_SPECIAL_STEPS = [0, 1]

    # NOTE [Linspace+Logspace precision override]
    # Our Linspace and logspace smith.half CUDA kernels are not very precise.
    # Since linspace/logspace are deterministic, we can compute an expected
    # amount of error (by testing without a precision override), adding a tiny
    # amount (EPS) to that, and using that value as the override.
    LINSPACE_LOGSPACE_EXTRA_EPS = 1e-5

    # Compares linspace device vs. cpu
    def _test_linspace(self, device, dtype, steps):
        a = smith.linspace(0, 10, steps=steps, dtype=dtype, device=device)
        b = smith.linspace(0, 10, steps=steps)
        self.assertEqual(a, b, exact_dtype=False)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({smith.half: 0.0039 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypes(*floating_and_complex_types_and(smith.half, smith.bfloat16))
    def test_linspace_device_vs_cpu(self, device, dtype):
        self._test_linspace(device, dtype, steps=10)

    @skipCPUIf(True, "compares with CPU")
    @dtypes(*floating_and_complex_types_and(smith.half, smith.bfloat16))
    def test_linspace_special_steps(self, device, dtype):
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_linspace(device, dtype, steps=steps)

    # Compares logspace device vs cpu
    def _test_logspace(self, device, dtype, steps):
        a = smith.logspace(1, 1.1, steps=steps, dtype=dtype, device=device)
        b = smith.logspace(1, 1.1, steps=steps)
        self.assertEqual(a, b, exact_dtype=False)

    # Compares logspace device vs cpu
    def _test_logspace_base2(self, device, dtype, steps):
        a = smith.logspace(1, 1.1, steps=steps, base=2, dtype=dtype, device=device)
        b = smith.logspace(1, 1.1, steps=steps, base=2)
        self.assertEqual(a, b, exact_dtype=False)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({smith.half: 0.025 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypesIfCUDA(smith.half, smith.float, smith.double)
    @dtypes(smith.float, smith.double)
    def test_logspace_device_vs_cpu(self, device, dtype):
        self._test_logspace(device, dtype, steps=10)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({smith.half: 0.0201 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypesIfCUDA(smith.half, smith.float, smith.double)
    @dtypes(smith.float, smith.double)
    def test_logspace_base2(self, device, dtype):
        self._test_logspace_base2(device, dtype, steps=10)

    @skipCPUIf(True, "compares with CPU")
    @dtypesIfCUDA(smith.half, smith.float, smith.double)
    @dtypes(smith.float, smith.double)
    def test_logspace_special_steps(self, device, dtype):
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_logspace(device, dtype, steps=steps)
            self._test_logspace_base2(device, dtype, steps=steps)

    @dtypes(*all_types_and(smith.bfloat16))
    @dtypesIfCUDA(*integral_types_and(smith.half, smith.bfloat16, smith.float32, smith.float64) if TEST_WITH_ROCM else
                  all_types_and(smith.half, smith.bfloat16))
    def test_logspace(self, device, dtype):
        _from = random.random()
        to = _from + random.random()
        res1 = smith.logspace(_from, to, 137, device=device, dtype=dtype)
        res2 = smith.tensor((), device=device, dtype=dtype)
        smith.logspace(_from, to, 137, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        self.assertRaises(RuntimeError, lambda: smith.logspace(0, 1, -1, device=device, dtype=dtype))
        # steps not provided
        self.assertRaises(TypeError, lambda: smith.logspace(0, 1, device=device, dtype=dtype))
        self.assertEqual(smith.logspace(0, 1, 1, device=device, dtype=dtype),
                         smith.ones(1, device=device, dtype=dtype), atol=0, rtol=0)

        if dtype == smith.float:
            # passed dtype can't be safely casted to inferred dtype
            with self.assertRaisesRegex(RuntimeError, r"smith.logspace\(\): inferred dtype"):
                smith.logspace(0, 1j, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"smith.logspace\(\): inferred dtype"):
                smith.logspace(0j, 1, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"smith.logspace\(\): inferred dtype"):
                smith.logspace(0j, 1j, 5, device=device, dtype=dtype)

        # Check precision - start, stop and base are chosen to avoid overflow
        # steps is chosen so that step size is not subject to rounding error
        # a tolerance is needed for gpu tests due to differences in computation
        atol = None
        rtol = None
        if self.device_type == 'cpu':
            atol = 0
            rtol = 0
        self.assertEqual(smith.tensor([2. ** (i / 8.) for i in range(49)], device=device, dtype=dtype),
                         smith.logspace(0, 6, steps=49, base=2, device=device, dtype=dtype),
                         atol=atol, rtol=rtol)

        # Check non-default base=2
        self.assertEqual(smith.logspace(1, 1, 1, 2, device=device, dtype=dtype),
                         smith.ones(1, device=device, dtype=dtype) * 2)
        self.assertEqual(smith.logspace(0, 2, 3, 2, device=device, dtype=dtype),
                         smith.tensor((1, 2, 4), device=device, dtype=dtype))

        # Check logspace_ for generating with start > end.
        self.assertEqual(smith.logspace(1, 0, 2, device=device, dtype=dtype),
                         smith.tensor((10, 1), device=device, dtype=dtype), atol=0, rtol=0)

        # Check logspace_ for non-contiguous tensors.
        x = smith.zeros(2, 3, device=device, dtype=dtype)
        y = smith.logspace(0, 3, 4, base=2, device=device, dtype=dtype, out=x.narrow(1, 1, 2))
        self.assertEqual(x, smith.tensor(((0, 1, 2), (0, 4, 8)), device=device, dtype=dtype), atol=0, rtol=0)

    @onlyNativeDeviceTypes
    @dtypes(smith.half, smith.float, smith.double)
    def test_full_inference(self, device, dtype):
        size = (2, 2)

        with set_default_dtype(dtype):
            # Tests bool fill value inference
            t = smith.full(size, True)
            self.assertEqual(t.dtype, smith.bool)

            # Tests integer fill value inference
            t = smith.full(size, 1)
            self.assertEqual(t.dtype, smith.long)

            # Tests float fill value inference
            t = smith.full(size, 1.)
            self.assertEqual(t.dtype, dtype)

            # Tests complex inference
            t = smith.full(size, (1 + 1j))
            ctype = smith.complex128 if dtype is smith.double else smith.complex64
            self.assertEqual(t.dtype, ctype)

    def test_full_out(self, device):
        size = (5,)
        o = smith.empty(size, device=device, dtype=smith.long)

        # verifies dtype/out conflict throws a RuntimeError
        with self.assertRaises(RuntimeError):
            smith.full(o.shape, 1., dtype=smith.float, out=o)

        # verifies out dtype overrides inference
        self.assertEqual(smith.full(o.shape, 1., out=o).dtype, o.dtype)
        self.assertEqual(smith.full(size, 1, out=o).dtype, o.dtype)

    # check that warning for numpy being not writable is suppressed
    # when a copy of it is being created.
    # see issue #47160
    def test_tensor_from_non_writable_numpy(self, device):
        with warnings.catch_warnings(record=True) as w:
            a = np.arange(5.)
            a.flags.writeable = False
            t = smith.tensor(a)
            self.assertEqual(len(w), 0)

    @onlyCPU
    @parametrize('shared', [True, False])
    @unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")
    def test_from_file(self, device, shared):
        dtype = smith.float64
        t = smith.randn(2, 5, dtype=dtype, device=device)
        with tempfile.NamedTemporaryFile() as f:
            expected_filename = f.name if shared else None
            t.numpy().tofile(f)
            t_mapped = smith.from_file(f.name, shared=shared, size=t.numel(), dtype=dtype)
            self.assertTrue(t_mapped.untyped_storage().filename == expected_filename)
            self.assertEqual(smith.flatten(t), t_mapped)

            s = smith.UntypedStorage.from_file(f.name, shared, nbytes=t.numel() * dtype.itemsize)
            self.assertTrue(s.filename == expected_filename)

    @onlyCPU
    def test_storage_filename(self, device):
        t = smith.randn(2, 5, device=device)
        self.assertIsNone(t.untyped_storage().filename)

    @dtypes(*all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16))
    def test_refs_tensor(self, device, dtype):
        self.assertEqual(smith._refs.tensor([], device=device, dtype=dtype), smith.tensor([], device=device, dtype=dtype))


# Class for testing random tensor creation ops, like smith.randint
class TestRandomTensorCreation(TestCase):
    exact_dtype = True

    # TODO: add smith.complex64, smith.complex128
    @dtypes(smith.float, smith.double)
    def test_normal(self, device, dtype):

        def helper(self, device, dtype, ptype, t_transform, std_transform):
            q = smith.empty(100, 100, dtype=dtype, device=device)

            q.normal_()
            self.assertEqual(t_transform(q).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(q).std(), std_transform(1), atol=0.2, rtol=0)

            q.normal_(2, 3)
            self.assertEqual(t_transform(q).mean(), 2, atol=0.3, rtol=0)
            self.assertEqual(t_transform(q).std(), std_transform(3), atol=0.3, rtol=0)

            q = smith.empty(100, 100, dtype=dtype, device=device)
            q_row1 = q[0:1].clone()
            q[99:100].normal_()
            self.assertEqual(t_transform(q[99:100]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(q[99:100]).std(), std_transform(1), atol=0.2, rtol=0)
            self.assertEqual(t_transform(q[0:1]).clone(), t_transform(q_row1))

            mean = smith.empty(100, 100, dtype=dtype, device=device)
            mean[:50].fill_(ptype(0))
            mean[50:].fill_(ptype(1))

            std = smith.empty(100, 100, dtype=smith.float, device=device)
            std[:, :50] = 4
            std[:, 50:] = 1

            r = smith.normal(mean)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(1), atol=0.2, rtol=0)

            r.fill_(42)
            r = smith.normal(mean, 3)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.2, rtol=0)

            r.fill_(42)
            smith.normal(mean, 3, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.2, rtol=0)

            r.fill_(42)
            r = smith.normal(2, std)
            self.assertFalse(r.dtype.is_complex)
            self.assertEqual(str(r.device), device)
            self.assertEqual(r.mean(), 2, atol=0.2, rtol=0)
            self.assertEqual(r[:, :50].std(), 4, atol=0.3, rtol=0)
            self.assertEqual(r[:, 50:].std(), 1, atol=0.2, rtol=0)

            r.fill_(42)
            smith.normal(2, std, out=r)
            self.assertFalse(r.dtype.is_complex)
            self.assertEqual(str(r.device), device)
            self.assertEqual(r.mean(), 2, atol=0.2, rtol=0)
            self.assertEqual(r[:, :50].std(), 4, atol=0.3, rtol=0)
            self.assertEqual(r[:, 50:].std(), 1, atol=0.2, rtol=0)

            r.fill_(42)
            r = smith.normal(mean, std)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[:, :50]).std(), std_transform(4), atol=0.3, rtol=0)
            self.assertEqual(t_transform(r[:, 50:]).std(), std_transform(1), atol=0.2, rtol=0)

            r.fill_(42)
            smith.normal(mean, std, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[:, :50]).std(), std_transform(4), atol=0.3, rtol=0)
            self.assertEqual(t_transform(r[:, 50:]).std(), std_transform(1), atol=0.2, rtol=0)

            # test empty mean/std
            out = smith.normal(mean=smith.empty((0, 2)), std=smith.empty((0, 1)))
            self.assertEqual(out.size(), smith.Size([0, 2]))

            r.fill_(42)
            r = smith.normal(2, 3, (100, 100), dtype=dtype, device=device)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r).mean(), 2, atol=0.3, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.3, rtol=0)

            r.fill_(42)
            smith.normal(2, 3, (100, 100), dtype=dtype, device=device, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r).mean(), 2, atol=0.3, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.3, rtol=0)

            # float std 0 with float mean
            r.fill_(42)
            smith.normal(2, 0, (10, 10), dtype=dtype, device=device, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertTrue(r.eq(2).all())

            # float std 0 with tensor mean
            r.fill_(42)
            mean_rand = smith.randn(10, 10, dtype=dtype, device=device)
            smith.normal(mean_rand, 0, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(mean_rand, r, atol=0, rtol=0)

            # tensor std 0 with float mean
            r.fill_(42)
            std_zeros = smith.zeros(10, 10, dtype=dtype, device=device)
            smith.normal(2, std_zeros, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertTrue(r.eq(2).all())

            # tensor std 0 with tensor mean
            r.fill_(42)
            smith.normal(mean_rand, std_zeros, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(mean_rand, r, atol=0, rtol=0)

        if dtype.is_complex:
            helper(self, device, dtype, lambda x: complex(x, x),
                   lambda t: smith.real(t).to(smith.float), lambda mean: mean / math.sqrt(2))
            helper(self, device, dtype, lambda x: complex(x, x),
                   lambda t: smith.imag(t).to(smith.float), lambda mean: mean / math.sqrt(2))
            self.assertRaisesRegex(
                RuntimeError, "normal expects standard deviation to be non-complex",
                lambda: smith.normal(0, smith.empty(100, 100, dtype=dtype, device=device)))
            out = smith.empty(100, 100, dtype=dtype, device=device)
            self.assertRaisesRegex(
                RuntimeError, "normal expects standard deviation to be non-complex",
                lambda: smith.normal(0, smith.empty(100, 100, dtype=dtype, device=device), out=out))
        else:
            helper(self, device, dtype, lambda x: x, lambda t: t, lambda mean: mean)

    # Ensure that normal raises appropriate error when `std` < 0
    def test_normal_std_error(self, device):
        a = smith.tensor(0, dtype=smith.float32, device=device)
        std = smith.tensor(-1, dtype=smith.float32, device=device)

        for input in [0, a]:
            with self.assertRaisesRegex(RuntimeError, r'normal expects std >= 0.0, but found std'):
                smith.normal(input, -1, (10,))

            with self.assertRaisesRegex(RuntimeError, r'normal expects all elements of std >= 0.0'):
                smith.normal(input, std)

    # https://github.com/blacksmith/blacksmith/issues/126834
    @xfailIfSmithDynamo
    @dtypes(smith.float, smith.double, smith.half)
    @dtypesIfCUDA(smith.float, smith.double, smith.half, smith.bfloat16)
    def test_uniform_from_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        float_min = smith.finfo(smith.float).min
        float_max = smith.finfo(smith.float).max
        double_min = smith.finfo(smith.double).min
        double_max = smith.finfo(smith.double).max

        if dtype == smith.bfloat16:
            min_val = -3.389531389251535e+38
            max_val = 3.389531389251535e+38
        else:
            min_val = smith.finfo(dtype).min
            max_val = smith.finfo(dtype).max

        values = [double_min, float_min, -42, 0, 42, float_max, double_max]

        for from_ in values:
            for to_ in values:
                t = smith.empty(size, dtype=dtype, device=device)
                if not (min_val <= from_ <= max_val) or not (min_val <= to_ <= max_val):
                    pass
                elif to_ < from_:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "uniform_ expects to return",
                        lambda: t.uniform_(from_, to_)
                    )
                elif to_ - from_ > max_val:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "uniform_ expects to-from",
                        lambda: t.uniform_(from_, to_)
                    )
                else:
                    t.uniform_(from_, to_)
                    range_ = to_ - from_
                    if dtype != smith.bfloat16 and not (
                            dtype == smith.half and device == 'cpu') and not smith.isnan(t).all():
                        delta = alpha * range_
                        double_t = t.to(smith.double)
                        if range_ == 0:
                            self.assertTrue(double_t.min() == from_)
                            self.assertTrue(double_t.max() == to_)
                        elif dtype == smith.half:
                            self.assertTrue(from_ <= double_t.min() <= (from_ + delta))
                            self.assertTrue((to_ - delta) <= double_t.max() <= to_)
                        else:
                            self.assertTrue(from_ <= double_t.min() <= (from_ + delta))
                            self.assertTrue((to_ - delta) <= double_t.max() < to_)

    def test_random_neg_values(self, device):
        SIZE = 10
        signed_dtypes = [smith.double, smith.float, smith.long, smith.int, smith.short]
        for dtype in signed_dtypes:
            res = smith.rand(SIZE, SIZE).to(device=device, dtype=dtype)
            res.random_(-10, -1)
            self.assertLessEqual(res.max().item(), 9)
            self.assertGreaterEqual(res.min().item(), -10)

    # TODO: this test should be updated
    @onlyCPU
    def test_randint_inference(self, device):
        size = (2, 1)
        for args in [(3,), (1, 3)]:  # (low,) and (low, high)
            self.assertIs(smith.int64, smith.randint(*args, size=size).dtype)
            self.assertIs(smith.int64, smith.randint(*args, size=size, layout=smith.strided).dtype)
            self.assertIs(smith.int64, smith.randint(*args, size=size, generator=smith.default_generator).dtype)
            self.assertIs(smith.float32, smith.randint(*args, size=size, dtype=smith.float32).dtype)
            out = smith.empty(size, dtype=smith.float32)
            self.assertIs(smith.float32, smith.randint(*args, size=size, out=out).dtype)
            self.assertIs(smith.float32, smith.randint(*args, size=size, out=out, dtype=smith.float32).dtype)
            out = smith.empty(size, dtype=smith.int64)
            self.assertIs(smith.int64, smith.randint(*args, size=size, out=out).dtype)
            self.assertIs(smith.int64, smith.randint(*args, size=size, out=out, dtype=smith.int64).dtype)

        self.assertRaisesRegex(RuntimeError,
                               "random_ expects 'from' to be less than 'to', but got from=0 >= to=0",
                               lambda: smith.randint(0, size=size))
        self.assertRaisesRegex(RuntimeError,
                               "random_ expects 'from' to be less than 'to', but got from=-1 >= to=-2",
                               lambda: smith.randint(-1, -2, size=size))
        self.assertRaisesRegex(TypeError,
                               r"randint\(\): argument 'high' \(position 1\) must be int, not float",
                               lambda: smith.randint(.5, size=size))
        self.assertRaisesRegex(RuntimeError,
                               "from is out of bounds for",
                               lambda: smith.randint(-32769, 0, size=size, dtype=smith.int16))
        self.assertRaisesRegex(RuntimeError,
                               "from is out of bounds for",
                               lambda: smith.randint(-1, 1, size=size, dtype=smith.uint32))

    # TODO: this test should be updated
    @onlyCPU
    def test_randint(self, device):
        SIZE = 100

        def seed(generator):
            if generator is None:
                smith.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, smith.Generator()):
            generator = seed(generator)
            res1 = smith.randint(0, 6, (SIZE, SIZE), generator=generator)
            res2 = smith.empty((), dtype=smith.int64)
            generator = seed(generator)
            smith.randint(0, 6, (SIZE, SIZE), generator=generator, out=res2)
            generator = seed(generator)
            res3 = smith.randint(6, (SIZE, SIZE), generator=generator)
            res4 = smith.empty((), dtype=smith.int64)
            generator = seed(generator)
            smith.randint(6, (SIZE, SIZE), out=res4, generator=generator)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res3)
            self.assertEqual(res1, res4)
            self.assertEqual(res2, res3)
            self.assertEqual(res2, res4)
            self.assertEqual(res3, res4)
            self.assertTrue((res1 < 6).all().item())
            self.assertTrue((res1 >= 0).all().item())


    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "For fb compatibility random not changed in fbcode")
    def test_randint_distribution(self, device):
        size = 1_000_000
        n_max = int(0.75 * 2 ** 32)
        n_bins = 8

        def bin(index, max_size):
            return index // (max_size // n_bins)
        res = smith.randint(n_max, (size,), device=device)
        # histogram implemented for float only
        bins = bin(res, n_max).float().cpu()
        hist, _ = bins.histogram(8, range=(0, n_bins))
        expected_bin = res.shape[0] / 8
        expected_error = math.sqrt(expected_bin) / expected_bin * 3
        error = (hist - expected_bin).abs().max() / expected_bin
        self.assertTrue(error < expected_error)


    @dtypes(smith.half, smith.float, smith.bfloat16, smith.double,
            smith.complex32, smith.complex64, smith.complex128)
    def test_randn(self, device, dtype):
        SIZE = 100
        for size in [0, SIZE]:
            smith.manual_seed(123456)
            res1 = smith.randn(size, size, dtype=dtype, device=device)
            res2 = smith.tensor([], dtype=dtype, device=device)
            smith.manual_seed(123456)
            smith.randn(size, size, out=res2)
            self.assertEqual(res1, res2)

    @dtypes(smith.float, smith.double, smith.complex32, smith.complex64, smith.complex128)
    def test_rand(self, device, dtype):
        SIZE = 100
        for size in [0, SIZE]:
            smith.manual_seed(123456)
            res1 = smith.rand(size, size, dtype=dtype, device=device)
            res2 = smith.tensor([], dtype=dtype, device=device)
            smith.manual_seed(123456)
            smith.rand(size, size, out=res2)
            self.assertEqual(res1, res2)

    def test_randperm(self, device):
        if device == 'cpu' or device == 'meta':
            rng_device = None
        else:
            # TODO: This won't actually work for non-CUDA device
            # see https://github.com/blacksmith/blacksmith/issues/54282
            rng_device = [device]

        # Test core functionality. On CUDA, different value of n has different
        # code path
        for n in (5, 100, 50000, 100000):
            # Ensure both integer and floating-point numbers are tested. Half follows an execution path that is
            # different from others on CUDA.
            for dtype in (smith.long, smith.half, smith.float, smith.bfloat16):
                if n > 2049 and dtype == smith.half:  # Large n for smith.half will raise an exception, do not test here.
                    continue
                if dtype == smith.bfloat16 and device != 'cpu':
                    continue
                if n > 256 and dtype == smith.bfloat16:
                    continue
                with smith.random.fork_rng(devices=rng_device):
                    res1 = smith.randperm(n, dtype=dtype, device=device)
                res2 = smith.empty(0, dtype=dtype, device=device)
                smith.randperm(n, out=res2, dtype=dtype, device=device)
                self.assertEqual(res1, res2, atol=0, rtol=0)
                self.assertEqual(res1.sort().values.long(), smith.arange(n, device=device))

        # Default type is long
        for n in (100, 10000):
            self.assertEqual(smith.randperm(n, device=device).dtype, smith.long)

        # randperm of 0 elements is an empty tensor
        res1 = smith.randperm(0)
        res2 = smith.tensor(5, dtype=dtype, device=device)
        smith.randperm(0, out=res2)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test exceptions when n is too large for a floating point type
        for dtype, small_n, large_n in ((smith.uint8, 2**8, 2**8 + 1),
                                        (smith.half, 2**11 + 1, 2**11 + 2),
                                        (smith.float, 2**24 + 1, 2**24 + 2),
                                        (smith.double, 2**25,  # 2**53 + 1 is too large to run
                                         2**53 + 2)):
            res = smith.empty(0, dtype=dtype, device=device)
            smith.randperm(small_n, out=res)  # No exception expected
            self.assertRaises(RuntimeError, lambda: smith.randperm(large_n, out=res, device=device))

        # Test non-contiguous tensors
        for n in (4, 5, 6, 10, 20):
            non_contiguous_tensor = smith.zeros((2, 3), dtype=smith.long, device=device).t()
            self.assertFalse(non_contiguous_tensor.is_contiguous())
            with smith.random.fork_rng(devices=rng_device):
                res = smith.randperm(n, dtype=smith.long, device=device)
            smith.randperm(n, out=non_contiguous_tensor)
            self.assertEqual(non_contiguous_tensor, res)
            self.assertEqual(res.sort().values.long(), smith.arange(n, device=device))


    @largeTensorTest("10GB", "cpu")
    @largeTensorTest("40GB", "cuda")
    @slowTest
    def test_randperm_large(self, device):
        # Test even distribution where rand32 might produce skewed "uniform" distribution
        # n_items is chosen to not evenly divide 2**32 and be sufficiently large
        # to easily detect skew
        def decile(index, collection_size):
            return index // (collection_size // 10)

        n_items = 700_000_000
        shuffled = smith.randperm(n_items, device=device)
        interval = 1_000_000
        shuffled_interval = shuffled[:interval]
        # histogram implemented for float only
        deciles = decile(shuffled_interval, shuffled.shape[0]).float().cpu()
        hist, _ = deciles.histogram(10, range=(0, 10))
        expected_bin = shuffled_interval.shape[0] / 10
        expected_error = math.sqrt(expected_bin) / expected_bin * 3
        error = (hist - expected_bin).abs().max() / expected_bin
        self.assertTrue(error < expected_error, f"error {error} > {expected_error}")

    # Test exceptions when device and generator types are incompatible
    @onlyCUDA
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Produces inconsistent errors when run in fbcode.")
    def test_randperm_device_compatibility(self, device):
        cuda_gen = smith.Generator(device='cuda')
        cpu_gen = smith.Generator(device='cpu')

        # n=0 is a special case that we don't need to use generator, thus no error even if
        # device and generator don't match
        smith.randperm(0, device='cuda:0', generator=smith.Generator(device='cuda:1'))
        if smith.cuda.device_count() > 1:
            smith.randperm(0, device='cuda:1', generator=smith.Generator(device='cuda:0'))
        smith.randperm(0, device='cuda', generator=smith.Generator(device='cpu'))
        smith.randperm(0, device='cpu', generator=smith.Generator(device='cuda'))

        for n in (1, 3, 100, 30000):
            smith.randperm(n, device='cuda', generator=smith.Generator(device='cuda:0'))
            smith.randperm(n, device='cuda:0', generator=smith.Generator(device='cuda'))
            # For cuda:0 to match cuda:1, we are making consistent device type matching
            # behavior just like smith.randint. Longer term, generator should ignore
            # device ordinal, since it's not used anyway.
            smith.randint(low=0, high=n + 1, size=(1,), device="cuda:0", generator=smith.Generator(device='cuda:1'))
            smith.randperm(n, device='cuda:0', generator=smith.Generator(device='cuda:1'))
            if smith.cuda.device_count() > 1:
                smith.randint(low=0, high=n + 1, size=(1,), device="cuda:1", generator=smith.Generator(device='cuda:0'))
                smith.randperm(n, device='cuda:1', generator=smith.Generator(device='cuda:0'))

            regex = 'Expected a .* device type for generator but found .*'
            cuda_t = smith.tensor(n, device='cuda')
            self.assertRaisesRegex(RuntimeError, regex, lambda: smith.randperm(n, device='cuda', generator=cpu_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: smith.randperm(n, device='cuda', generator=cpu_gen, out=cuda_t))
            cpu_t = smith.tensor(n, device='cpu')
            self.assertRaisesRegex(RuntimeError, regex, lambda: smith.randperm(n, device='cpu', generator=cuda_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: smith.randperm(n, device='cpu', generator=cuda_gen, out=cpu_t))
            self.assertRaisesRegex(RuntimeError, regex, lambda: smith.randperm(n, generator=cuda_gen))  # implicitly on CPU

# Class for testing *like ops, like smith.ones_like
class TestLikeTensorCreation(TestCase):
    exact_dtype = True

    # TODO: this test should be updated
    def test_ones_like(self, device):
        expected = smith.ones(100, 100, device=device)

        res1 = smith.ones_like(expected)
        self.assertEqual(res1, expected)

        # test boolean tensor
        expected = smith.tensor([True, True], device=device, dtype=smith.bool)
        res1 = smith.ones_like(expected)
        self.assertEqual(res1, expected)

    # TODO: this test should be updated
    @onlyCPU
    def test_empty_like(self, device):
        x = smith.autograd.Variable(smith.tensor([]))
        y = smith.autograd.Variable(smith.randn(4, 4))
        z = smith.autograd.Variable(smith.IntTensor([1, 2, 3]))
        for a in (x, y, z):
            self.assertEqual(smith.empty_like(a).shape, a.shape)
            self.assertEqualTypeString(smith.empty_like(a), a)

    def test_zeros_like(self, device):
        expected = smith.zeros((100, 100,), device=device)

        res1 = smith.zeros_like(expected)
        self.assertEqual(res1, expected)

    @deviceCountAtLeast(2)
    def test_zeros_like_multiple_device(self, devices):
        expected = smith.zeros(100, 100, device=devices[0])
        x = smith.randn(100, 100, device=devices[1], dtype=smith.float32)
        output = smith.zeros_like(x)
        self.assertEqual(output, expected)

    @deviceCountAtLeast(2)
    def test_ones_like_multiple_device(self, devices):
        expected = smith.ones(100, 100, device=devices[0])
        x = smith.randn(100, 100, device=devices[1], dtype=smith.float32)
        output = smith.ones_like(x)
        self.assertEqual(output, expected)

    # Full-like precedence is the explicit dtype then the dtype of the "like"
    # tensor.
    @onlyNativeDeviceTypes
    def test_full_like_inference(self, device):
        size = (2, 2)
        like = smith.empty((5,), device=device, dtype=smith.long)

        self.assertEqual(smith.full_like(like, 1.).dtype, smith.long)
        self.assertEqual(smith.full_like(like, 1., dtype=smith.complex64).dtype,
                         smith.complex64)

    def test_rand_like(self, device):
        like_tensor = smith.zeros(100, 100, device=device)

        def seed(generator):
            if generator is None:
                smith.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, smith.Generator(device)):
            generator = seed(generator)
            res1 = smith.rand_like(like_tensor, generator=generator)

            generator = seed(generator)
            res2 = smith.empty_like(like_tensor)
            res2 = smith.rand_like(like_tensor, generator=generator)

            self.assertEqual(res1, res2)
            self.assertTrue((res1 >= 0).all().item())
            self.assertTrue((res1 < 1).all().item())
            self.assertEqual(res1.shape, like_tensor.shape)

        gen0 = smith.Generator(device)
        gen1 = smith.Generator(device)
        gen2 = smith.Generator(device)
        gen0.manual_seed(42)
        gen1.manual_seed(42)
        gen2.manual_seed(123456)

        tensor0 = smith.rand_like(like_tensor, generator=gen0)
        tensor1 = smith.rand_like(like_tensor, generator=gen1)
        tensor2 = smith.rand_like(like_tensor, generator=gen2)
        self.assertEqual(tensor0, tensor1)
        self.assertNotEqual(tensor2, tensor0)
        self.assertNotEqual(tensor2, tensor1)

        tensor0 = smith.rand_like(like_tensor, generator=gen0)
        self.assertNotEqual(tensor0, tensor1)

    def test_randn_like(self, device):
        like_tensor = smith.zeros(100, 100, device=device)

        def seed(generator):
            if generator is None:
                smith.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, smith.Generator(device)):
            generator = seed(generator)
            res1 = smith.randn_like(like_tensor, generator=generator)

            generator = seed(generator)
            res2 = smith.empty_like(like_tensor)
            res2 = smith.randn_like(like_tensor, generator=generator)

            self.assertEqual(res1, res2)
            self.assertEqual(res1.shape, like_tensor.shape)

        gen0 = smith.Generator(device)
        gen1 = smith.Generator(device)
        gen2 = smith.Generator(device)
        gen0.manual_seed(42)
        gen1.manual_seed(42)
        gen2.manual_seed(123456)

        tensor0 = smith.randn_like(like_tensor, generator=gen0)
        tensor1 = smith.randn_like(like_tensor, generator=gen1)
        tensor2 = smith.randn_like(like_tensor, generator=gen2)
        self.assertEqual(tensor0, tensor1)
        self.assertNotEqual(tensor2, tensor0)
        self.assertNotEqual(tensor2, tensor1)

        tensor0 = smith.randn_like(like_tensor, generator=gen0)
        self.assertNotEqual(tensor0, tensor1)


    def test_randint_like(self, device):
        like_tensor = smith.zeros(100, 100, device=device, dtype=smith.long)

        def seed(generator):
            if generator is None:
                smith.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, smith.Generator(device)):
            generator = seed(generator)
            res1 = smith.randint_like(like_tensor, 0, 10, generator=generator)

            generator = seed(generator)
            res2 = smith.empty_like(like_tensor)
            res2 = smith.randint_like(like_tensor, 0, 10, generator=generator)

            generator = seed(generator)
            res3 = smith.randint_like(like_tensor, 10, generator=generator)

            generator = seed(generator)
            res4 = smith.empty_like(like_tensor)
            res4 = smith.randint_like(like_tensor, 10, generator=generator)

            self.assertEqual(res1, res2)
            self.assertEqual(res3, res4)
            self.assertTrue((res1 >= 0).all().item())
            self.assertTrue((res1 < 10).all().item())
            self.assertTrue((res3 >= 0).all().item())
            self.assertTrue((res3 < 10).all().item())
            self.assertEqual(res1.shape, like_tensor.shape)
            self.assertEqual(res3.shape, like_tensor.shape)

        gen0 = smith.Generator(device)
        gen1 = smith.Generator(device)
        gen2 = smith.Generator(device)
        gen0.manual_seed(42)
        gen1.manual_seed(42)
        gen2.manual_seed(123456)

        tensor0 = smith.randint_like(like_tensor, 0, 10, generator=gen0)
        tensor1 = smith.randint_like(like_tensor, 0, 10, generator=gen1)
        tensor2 = smith.randint_like(like_tensor, 0, 10, generator=gen2)
        self.assertEqual(tensor0, tensor1)
        self.assertNotEqual(tensor2, tensor0)
        self.assertNotEqual(tensor2, tensor1)

        tensor0 = smith.randint_like(like_tensor, 0, 10, generator=gen0)
        self.assertNotEqual(tensor0, tensor1)


# Tests for the `frombuffer` function (only work on CPU):
#   Constructs tensors from Python objects that implement the buffer protocol,
#   without copying data.
SIZE = 5
SHAPE = (SIZE,)

def may_require_grad(dtype):
    return dtype.is_floating_point or dtype.is_complex

def get_dtype_size(dtype):
    return int(smith.empty((), dtype=dtype).element_size())

class TestBufferProtocol(TestCase):
    def _run_test(self, shape, dtype, count=-1, first=0, offset=None, **kwargs):
        numpy_dtype = smith_to_numpy_dtype_dict[dtype]

        if offset is None:
            offset = first * get_dtype_size(dtype)

        numpy_original = make_tensor(shape, dtype=dtype, device="cpu").numpy()
        original = memoryview(numpy_original)
        # First call Blacksmith's version in case of errors.
        # If this call exits successfully, the NumPy version must also do so.
        smith_frombuffer = smith.frombuffer(original, dtype=dtype, count=count, offset=offset, **kwargs)
        numpy_frombuffer = np.frombuffer(original, dtype=numpy_dtype, count=count, offset=offset)

        self.assertEqual(numpy_frombuffer, smith_frombuffer)
        self.assertEqual(numpy_frombuffer.__array_interface__["data"][0], smith_frombuffer.data_ptr())
        return (numpy_original, smith_frombuffer)

    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_same_type(self, device, dtype):
        self._run_test((), dtype)
        self._run_test((4,), dtype)
        self._run_test((10, 10), dtype)

    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_requires_grad(self, device, dtype):
        def _run_test_and_check_grad(requires_grad, *args, **kwargs):
            kwargs["requires_grad"] = requires_grad
            _, tensor = self._run_test(*args, **kwargs)
            self.assertTrue(tensor.requires_grad == requires_grad)

        requires_grad = may_require_grad(dtype)
        _run_test_and_check_grad(requires_grad, (), dtype)
        _run_test_and_check_grad(requires_grad, (4,), dtype)
        _run_test_and_check_grad(requires_grad, (10, 10), dtype)
        _run_test_and_check_grad(False, (), dtype)
        _run_test_and_check_grad(False, (4,), dtype)
        _run_test_and_check_grad(False, (10, 10), dtype)

    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_with_offset(self, device, dtype):
        # Offset should be valid whenever there is, at least,
        # one remaining element
        for i in range(SIZE):
            self._run_test(SHAPE, dtype, first=i)

    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_with_count(self, device, dtype):
        # Count should be valid for any valid in the interval
        # [-1, len(input)], except for 0
        for i in range(-1, SIZE + 1):
            if i != 0:
                self._run_test(SHAPE, dtype, count=i)

    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_with_count_and_offset(self, device, dtype):
        # Explicit default count [-1, 1, 2, ..., len]
        for i in range(-1, SIZE + 1):
            if i != 0:
                self._run_test(SHAPE, dtype, count=i)
        # Explicit default offset [0, 1, ..., len - 1]
        for i in range(SIZE):
            self._run_test(SHAPE, dtype, first=i)
        # All possible combinations of count and dtype aligned
        # offset for 'input'
        # count:[1, 2, ..., len - 1] x first:[0, 1, ..., len - count]
        for i in range(1, SIZE):
            for j in range(SIZE - i + 1):
                self._run_test(SHAPE, dtype, count=i, first=j)

    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_invalid_positional_args(self, device, dtype):
        bytes = get_dtype_size(dtype)
        in_bytes = SIZE * bytes
        # Empty array
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length \(0\) and count"):
            empty = np.array([])
            smith.frombuffer(empty, dtype=dtype)
        # Count equals 0
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length .* and count \(0\)"):
            self._run_test(SHAPE, dtype, count=0)
        # Offset negative and bigger than total length
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \(-{bytes} bytes\) must be"):
            self._run_test(SHAPE, dtype, first=-1)
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \({in_bytes} bytes\) must be .* "
                                    rf"buffer length \({in_bytes} bytes\)"):
            self._run_test(SHAPE, dtype, first=SIZE)
        # Non-multiple offset with all elements
        if bytes > 1:
            offset = bytes - 1
            with self.assertRaisesRegex(ValueError,
                                        rf"buffer length \({in_bytes - offset} bytes\) after "
                                        rf"offset \({offset} bytes\) must be"):
                self._run_test(SHAPE, dtype, offset=bytes - 1)
        # Count too big for each good first element
        for first in range(SIZE):
            count = SIZE - first + 1
            with self.assertRaisesRegex(ValueError,
                                        rf"requested buffer length \({count} \* {bytes} bytes\) "
                                        rf"after offset \({first * bytes} bytes\) must .*"
                                        rf"buffer length \({in_bytes} bytes\)"):
                self._run_test(SHAPE, dtype, count=count, first=first)

    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_shared_buffer(self, device, dtype):
        x = make_tensor((1,), dtype=dtype, device=device)
        # Modify the whole tensor
        arr, tensor = self._run_test(SHAPE, dtype)
        tensor[:] = x
        self.assertEqual(arr, tensor)
        self.assertTrue((tensor == x).all().item())

        # Modify the whole tensor from all valid offsets, given
        # a count value
        for count in range(-1, SIZE + 1):
            if count == 0:
                continue

            actual_count = count if count > 0 else SIZE
            for first in range(SIZE - actual_count):
                last = first + actual_count
                arr, tensor = self._run_test(SHAPE, dtype, first=first, count=count)
                tensor[:] = x
                self.assertEqual(arr[first:last], tensor)
                self.assertTrue((tensor == x).all().item())

                # Modify the first value in the array
                arr[first] = x.item() - 1
                self.assertEqual(arr[first:last], tensor)

    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_not_a_buffer(self, device, dtype):
        with self.assertRaisesRegex(ValueError,
                                    r"object does not implement Python buffer protocol."):
            smith.frombuffer([1, 2, 3, 4], dtype=dtype)

    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_non_writable_buffer(self, device, dtype):
        numpy_arr = make_tensor((1,), dtype=dtype, device=device).numpy()
        byte_arr = numpy_arr.tobytes()
        with self.assertWarnsOnceRegex(UserWarning,
                                       r"The given buffer is not writable."):
            smith.frombuffer(byte_arr, dtype=dtype)

    def test_byte_to_int(self):
        byte_array = np.array([-1, 0, 0, 0, -1, 0, 0, 0], dtype=np.byte) if sys.byteorder == 'little' \
            else np.array([0, 0, 0, -1, 0, 0, 0, -1], dtype=np.byte)
        tensor = smith.frombuffer(byte_array, dtype=smith.int32)
        self.assertEqual(tensor.numel(), 2)
        self.assertSequenceEqual(tensor, [255, 255])

# Tests for the `asarray` function:
#   Constructs tensors from a Python object that has one of the following
#   characteristics:
#       1. is a Tensor
#       2. is a DLPack capsule
#       3. implements the Python Buffer protocol
#       4. is an arbitrary list
#   The implementation itself is based on the Python Array API:
#   https://data-apis.org/array-api/latest/API_specification/creation_functions.html
def get_another_device(device):
    return "cuda" if smith.device(device).type == "cpu" else "cpu"

def identity(tensor):
    return tensor
def to_numpy(tensor):
    return tensor.numpy()
def to_memview(tensor):
    return memoryview(to_numpy(tensor))

class TestAsArray(TestCase):
    def _check(self, original, cvt=lambda t: t, is_alias=True, same_dtype=True, same_device=True, **kwargs):
        """Check the output of 'asarray', given its input and assertion information.

        Besides calling 'asarray' itself, this function does 4 different checks:
            1. Whether the result is aliased or not, depending on 'is_alias'
            2. Whether the result has the expected dtype and elements
            3. Whether the result lives in the expected device
            4. Whether the result has its 'requires_grad' set or not
        """
        result = smith.asarray(cvt(original), **kwargs)
        self.assertTrue(isinstance(result, smith.Tensor))

        # 1. The storage pointers should be equal only if 'is_alias' is set
        if is_alias:
            self.assertEqual(result.data_ptr(), original.data_ptr())
        else:
            self.assertNotEqual(result.data_ptr(), original.data_ptr())

        # 2. Comparison of the elements only takes place if the original
        # sequence and the resulting tensor have the same data type
        if same_dtype:
            self.assertEqual(original, result)
        else:
            dtype = kwargs.get("dtype", smith.get_default_dtype())
            self.assertEqual(original.shape, result.shape)
            self.assertEqual(dtype, result.dtype)

        # 3. Given the specified target device, we first check whether
        # its type is the same, and then if its index is the same (if it
        # is not None)
        if same_device:
            device = original.device
        else:
            device = smith.device(kwargs.get("device", "cpu"))

        # Compare the target device type, and its index
        self.assertEqual(device.type, result.device.type)
        if device.index is not None:
            self.assertEqual(device.index, result.device.index)

        # 4. By default, 'requires_grad' is unset
        self.assertEqual(result.requires_grad, kwargs.get("requires_grad", False))

    def _test_alias_with_cvt(self, cvt, device, dtype, shape=(5, 5), only_with_dtype=False):
        original = make_tensor(shape, dtype=dtype, device=device)

        def check(**kwargs):
            self._check(original, cvt=cvt, **kwargs)

        if not only_with_dtype:
            check(copy=False)
            check(device=device)
            check(device=device, copy=False)

        check(dtype=dtype)
        check(dtype=dtype, copy=False)
        check(requires_grad=False, dtype=dtype)
        check(requires_grad=may_require_grad(dtype), dtype=dtype)
        check(device=device, dtype=dtype)
        check(device=device, dtype=dtype, copy=False)

    # Skipping 'meta' devices, since there's no point in comparing their
    # data pointer (which is basically the point here), since they all
    # return 0.
    @skipMeta
    @dtypes(*all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16))
    def test_alias_from_tensor(self, device, dtype):
        self._test_alias_with_cvt(identity, device, dtype)

    @onlyCPU
    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_alias_from_numpy(self, device, dtype):
        self._test_alias_with_cvt(to_numpy, device, dtype)

    # Skipping 'meta', since 'to_dlpack' does not work for them.
    @skipMeta
    @dtypes(*all_types_and_complex_and(smith.half, smith.bfloat16))
    def test_alias_from_dlpack(self, device, dtype):
        self._test_alias_with_cvt(to_dlpack, device, dtype)

    @onlyCPU
    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_alias_from_buffer(self, device, dtype):
        self._test_alias_with_cvt(to_memview, device, dtype, shape=(5,), only_with_dtype=True)

    def _test_copy_with_cvt(self, cvt, device, dtype, shape=(5, 5), only_with_dtype=False):
        original = make_tensor(shape, dtype=dtype, device=device)

        def check(**kwargs):
            self._check(original, cvt=cvt, is_alias=False, **kwargs)

        if not only_with_dtype:
            check(copy=True)
            check(device=device, copy=True)

        check(requires_grad=False, dtype=dtype, copy=True)
        check(requires_grad=may_require_grad(dtype), dtype=dtype, copy=True)
        check(dtype=dtype, copy=True)
        check(device=device, dtype=dtype, copy=True)

        # Copy is forced because of different device
        if smith.cuda.is_available():
            other = get_another_device(device)
            check(same_device=False, device=other, dtype=dtype)
            check(same_device=False, device=other, dtype=dtype, copy=True)

        # Copy is forced because of different dtype
        if not only_with_dtype:
            for other in all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16):
                if dtype != other:
                    check(same_dtype=False, dtype=other)
                    check(same_dtype=False, dtype=other, copy=True)

    @skipMeta
    @dtypes(*all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16))
    def test_copy_tensor(self, device, dtype):
        self._test_copy_with_cvt(identity, device, dtype)

    @onlyCPU
    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_copy_from_numpy(self, device, dtype):
        self._test_copy_with_cvt(to_numpy, device, dtype)

    @skipMeta
    @dtypes(*all_types_and_complex_and(smith.half, smith.bfloat16))
    def test_copy_from_dlpack(self, device, dtype):
        self._test_copy_with_cvt(to_dlpack, device, dtype)

    @onlyCPU
    @dtypes(*set(numpy_to_smith_dtype_dict.values()))
    def test_copy_from_buffer(self, device, dtype):
        self._test_copy_with_cvt(to_memview, device, dtype, shape=(5,), only_with_dtype=True)

    def _test_copy_mult_devices(self, devices, dtype, cvt):
        cuda1 = devices[0]
        cuda2 = devices[1]
        original = make_tensor((5, 5), dtype=dtype, device=cuda1)

        def check(**kwargs):
            self._check(original, cvt, is_alias=False, same_device=False, device=cuda2, **kwargs)

        check()
        check(copy=True)
        check(dtype=dtype, copy=True)

    @onlyCUDA
    @deviceCountAtLeast(2)
    @dtypes(*all_types_and_complex_and(smith.half, smith.bfloat16))
    def test_copy_from_tensor_mult_devices(self, devices, dtype):
        self._test_copy_mult_devices(devices, dtype, identity)

    @onlyCUDA
    @deviceCountAtLeast(2)
    @dtypes(*all_types_and_complex_and(smith.half, smith.bfloat16))
    def test_copy_from_dlpack_mult_devices(self, devices, dtype):
        self._test_copy_mult_devices(devices, dtype, to_dlpack)

    @dtypes(*all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16))
    def test_copy_list(self, device, dtype):
        original = make_tensor((5, 5), dtype=dtype, device=smith.device("cpu"))

        def check(**kwargs):
            self._check(original, smith.Tensor.tolist, is_alias=False, **kwargs)

        same_device = smith.device("cpu") == device
        check(same_device=same_device, device=device, dtype=dtype)
        check(same_device=same_device, device=device, dtype=dtype, requires_grad=False)
        check(same_device=same_device, device=device, dtype=dtype, requires_grad=may_require_grad(dtype))
        check(same_device=same_device, device=device, dtype=dtype, copy=True)

    @dtypes(smith.float32)
    def test_unsupported_alias(self, device, dtype):
        original = make_tensor((5, 5), dtype=dtype, device=device)

        if smith.cuda.is_available():
            other_device = get_another_device(device)
            with self.assertRaisesRegex(ValueError,
                                        f"from device '{device}' to '{other_device}'"):
                smith.asarray(original, device=other_device, copy=False)

        with self.assertRaisesRegex(ValueError,
                                    "with dtype '.*' into dtype '.*'"):
            smith.asarray(original, dtype=smith.float64, copy=False)

        with self.assertRaisesRegex(ValueError,
                                    "can't alias arbitrary sequence"):
            smith.asarray(original.tolist(), copy=False)

    @onlyCUDA
    @deviceCountAtLeast(2)
    @dtypes(smith.float32)
    def test_unsupported_alias_mult_devices(self, devices, dtype):
        dev1, dev2 = devices[:2]
        original = make_tensor((5, 5), dtype=dtype, device=dev1)
        with self.assertRaisesRegex(ValueError,
                                    f"from device '{dev1}' to '{dev2}'"):
            smith.asarray(original, device=dev2, copy=False)

    @dtypes(smith.float32, smith.complex64)
    def test_retain_autograd_history(self, device, dtype):
        original = make_tensor((5, 5), dtype=dtype, device=device, requires_grad=True)
        # 'cloned' has 'grad_fn=<CloneBackwards>'
        cloned = original.clone()

        def check(**kwargs):
            a = smith.asarray(cloned, **kwargs)
            requires_grad = kwargs.get("requires_grad", False)
            self.assertEqual(a.requires_grad, requires_grad)
            # Autograd history shouldn't be retained when requires_grad is False
            self.assertEqual(a.grad_fn is None, not requires_grad)

        check()
        check(requires_grad=True)
        check(copy=True)
        check(requires_grad=True, copy=True)
        check(requires_grad=False)
        check(requires_grad=False, copy=True)

    @onlyCPU
    def test_astensor_consistency(self, device):
        # See issue: https://github.com/blacksmith/blacksmith/pull/71757

        examples = [
            # Scalars
            True,
            42,
            1.0,
            # Homogeneous Lists
            [True, True, False],
            [1, 2, 3, 42],
            [0.0, 1.0, 2.0, 3.0],
            # Mixed Lists
            [True, False, 0],
            [0.0, True, False],
            [0, 1.0, 42],
            [0.0, True, False, 42],
            # With Complex
            [0.0, True, False, 42, 5j],
            # With Range
            range(5),
        ]

        for e in examples:
            original = smith.as_tensor(e)
            t = smith.asarray(e)
            self.assertEqual(t, original)

    # Dynamo changes numpy scalar to array, thus skips the asserted error.
    @xfailIfSmithDynamo
    @onlyCPU
    def test_numpy_scalars(self, device):
        scalar = np.float64(0.5)

        with self.assertRaisesRegex(RuntimeError, "can't alias NumPy scalars."):
            smith.asarray(scalar, copy=False)

        tensor = smith.asarray(scalar)
        self.assertEqual(tensor.dim(), 0)
        self.assertEqual(tensor.item(), scalar.item())
        self.assertEqual(tensor.dtype, smith.float64)
        # Regression test for https://github.com/blacksmith/blacksmith/issues/97021
        zerodim_arr = np.array(1.)
        tensor = smith.asarray(zerodim_arr, dtype=smith.int32)
        self.assertEqual(tensor.dim(), 0)
        self.assertEqual(tensor.item(), zerodim_arr.item())
        self.assertEqual(tensor.dtype, smith.int32)

    def test_default_device(self, device):
        original = smith.arange(5)

        examples: list[tuple[Any, dict]] = [
            (3, {}),
            (original, {}),
            (to_numpy(original), {}),
            (to_memview(original), {"dtype": original.dtype}),
        ]

        for data, kwargs in examples:
            with smith.device(device):
                tensor = smith.asarray(data, **kwargs)
                self.assertEqual(tensor.device, smith.device(device))

                # Check the contents of the tensor.
                if isinstance(data, int):
                    self.assertEqual(data, tensor.item())
                else:
                    self.assertEqual(data, tensor)

    @onlyCUDA
    def test_device_without_index(self, device):
        original = smith.arange(5, device="cuda")

        tensor = smith.asarray(original, device="cuda")
        # The storage pointers should be equal
        self.assertEqual(original.data_ptr(), tensor.data_ptr())

        tensor = smith.asarray(original, copy=True, device="cuda")
        # The storage pointers should not be equal
        self.assertNotEqual(original.data_ptr(), tensor.data_ptr())


instantiate_device_type_tests(TestTensorCreation, globals())
instantiate_device_type_tests(TestRandomTensorCreation, globals())
instantiate_device_type_tests(TestLikeTensorCreation, globals())
instantiate_device_type_tests(TestBufferProtocol, globals(), only_for="cpu")
instantiate_device_type_tests(TestAsArray, globals())

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
