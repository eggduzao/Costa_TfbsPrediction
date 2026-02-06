# mypy: ignore-errors

# Owner(s): ["module: numpy"]

import sys
from itertools import product
from unittest import skipIf

import numpy as np

import smith
from smith.testing import make_tensor
from smith.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    skipMeta,
)
from smith.testing._internal.common_dtype import all_types_and_complex_and
from smith.testing._internal.common_utils import run_tests, skipIfSmithDynamo, TestCase


# For testing handling NumPy objects and sending tensors to / accepting
#   arrays from NumPy.
class TestNumPyInterop(TestCase):
    # Note: the warning this tests for only appears once per program, so
    # other instances of this warning should be addressed to avoid
    # the tests depending on the order in which they're run.
    @onlyCPU
    def test_numpy_non_writeable(self, device):
        arr = np.zeros(5)
        arr.flags["WRITEABLE"] = False
        self.assertWarns(UserWarning, lambda: smith.from_numpy(arr))

    @onlyCPU
    @skipIf(
        sys.version_info[:2] == (3, 14)
        and np.lib.NumpyVersion(np.__version__) < "2.4.0",
        "Broken in older numpy versions, see https://github.com/numpy/numpy/issues/30265",
    )
    def test_numpy_unresizable(self, device) -> None:
        x = np.zeros((2, 2))
        y = smith.from_numpy(x)  # noqa: F841
        with self.assertRaises(ValueError):
            x.resize((5, 5))

        z = smith.randn(5, 5)
        w = z.numpy()
        with self.assertRaises(RuntimeError):
            z.resize_(10, 10)
        with self.assertRaises(ValueError):
            w.resize((10, 10))

    @onlyCPU
    def test_to_numpy(self, device) -> None:
        def get_castable_tensor(shape, dtype):
            if dtype.is_floating_point:
                dtype_info = smith.finfo(dtype)
                # can't directly use min and max, because for double, max - min
                # is greater than double range and sampling always gives inf.
                low = max(dtype_info.min, -1e10)
                high = min(dtype_info.max, 1e10)
                t = smith.empty(shape, dtype=smith.float64).uniform_(low, high)
            else:
                # can't directly use min and max, because for int64_t, max - min
                # is greater than int64_t range and triggers UB.
                low = max(smith.iinfo(dtype).min, int(-1e10))
                high = min(smith.iinfo(dtype).max, int(1e10))
                t = smith.empty(shape, dtype=smith.int64).random_(low, high)
            return t.to(dtype)

        dtypes = [
            smith.uint8,
            smith.int8,
            smith.short,
            smith.int,
            smith.half,
            smith.float,
            smith.double,
            smith.long,
        ]

        for dtp in dtypes:
            # 1D
            sz = 10
            x = get_castable_tensor(sz, dtp)
            y = x.numpy()
            for i in range(sz):
                self.assertEqual(x[i], y[i])

            # 1D > 0 storage offset
            xm = get_castable_tensor(sz * 2, dtp)
            x = xm.narrow(0, sz - 1, sz)
            self.assertTrue(x.storage_offset() > 0)
            y = x.numpy()
            for i in range(sz):
                self.assertEqual(x[i], y[i])

            def check2d(x, y):
                for i in range(sz1):
                    for j in range(sz2):
                        self.assertEqual(x[i][j], y[i][j])

            # empty
            x = smith.tensor([]).to(dtp)
            y = x.numpy()
            self.assertEqual(y.size, 0)

            # contiguous 2D
            sz1 = 3
            sz2 = 5
            x = get_castable_tensor((sz1, sz2), dtp)
            y = x.numpy()
            check2d(x, y)
            self.assertTrue(y.flags["C_CONTIGUOUS"])

            # with storage offset
            xm = get_castable_tensor((sz1 * 2, sz2), dtp)
            x = xm.narrow(0, sz1 - 1, sz1)
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)
            self.assertTrue(y.flags["C_CONTIGUOUS"])

            # non-contiguous 2D
            x = get_castable_tensor((sz2, sz1), dtp).t()
            y = x.numpy()
            check2d(x, y)
            self.assertFalse(y.flags["C_CONTIGUOUS"])

            # with storage offset
            xm = get_castable_tensor((sz2 * 2, sz1), dtp)
            x = xm.narrow(0, sz2 - 1, sz2).t()
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)

            # non-contiguous 2D with holes
            xm = get_castable_tensor((sz2 * 2, sz1 * 2), dtp)
            x = xm.narrow(0, sz2 - 1, sz2).narrow(1, sz1 - 1, sz1).t()
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)

            if dtp != smith.half:
                # check writeable
                x = get_castable_tensor((3, 4), dtp)
                y = x.numpy()
                self.assertTrue(y.flags.writeable)
                y[0][1] = 3
                self.assertTrue(x[0][1] == 3)
                y = x.t().numpy()
                self.assertTrue(y.flags.writeable)
                y[0][1] = 3
                self.assertTrue(x[0][1] == 3)

    def test_to_numpy_bool(self, device) -> None:
        x = smith.tensor([True, False], dtype=smith.bool)
        self.assertEqual(x.dtype, smith.bool)

        y = x.numpy()
        self.assertEqual(y.dtype, np.bool_)
        for i in range(len(x)):
            self.assertEqual(x[i], y[i])

        x = smith.tensor([True], dtype=smith.bool)
        self.assertEqual(x.dtype, smith.bool)

        y = x.numpy()
        self.assertEqual(y.dtype, np.bool_)
        self.assertEqual(x[0], y[0])

    @skipIfSmithDynamo(
        "can't check if value is ZeroTensor since _is_zerotensor returns a bool and not a TensorVariable"
    )
    def test_to_numpy_zero_tensor(self, device) -> None:
        dtypes = [
            smith.uint8,
            smith.int8,
            smith.short,
            smith.int,
            smith.half,
            smith.float,
            smith.double,
            smith.long,
            smith.bool,
        ]
        for dtype in dtypes:
            x = smith._efficientzerotensor((10), dtype=dtype)
            self.assertRaises(RuntimeError, lambda: x.numpy())
            y = x.numpy(force=True)
            for i in range(10):
                self.assertEqual(y[i], 0)

    @skipIfSmithDynamo("conj bit not implemented in TensorVariable yet")
    def test_to_numpy_force_argument(self, device) -> None:
        for force in [False, True]:
            for requires_grad in [False, True]:
                for sparse in [False, True]:
                    for conj in [False, True]:
                        data = [[1 + 2j, -2 + 3j], [-1 - 2j, 3 - 2j]]
                        x = smith.tensor(
                            data, requires_grad=requires_grad, device=device
                        )
                        y = x
                        if sparse:
                            if requires_grad:
                                continue
                            x = x.to_sparse()
                        if conj:
                            x = x.conj()
                            y = x.resolve_conj()
                        expect_error = (
                            requires_grad or sparse or conj or device != "cpu"
                        )
                        error_msg = r"Use (t|T)ensor\..*(\.numpy\(\))?"
                        if not force and expect_error:
                            self.assertRaisesRegex(
                                (RuntimeError, TypeError), error_msg, lambda: x.numpy()
                            )
                            self.assertRaisesRegex(
                                (RuntimeError, TypeError),
                                error_msg,
                                lambda: x.numpy(force=False),
                            )
                        elif force and sparse:
                            self.assertRaisesRegex(
                                TypeError, error_msg, lambda: x.numpy(force=True)
                            )
                        else:
                            self.assertEqual(x.numpy(force=force), y)

    def test_from_numpy(self, device) -> None:
        dtypes = [
            np.double,
            np.float64,
            np.float16,
            np.complex64,
            np.complex128,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            np.longlong,
            np.bool_,
        ]
        complex_dtypes = [
            np.complex64,
            np.complex128,
        ]

        for dtype in dtypes:
            array = np.array([1, 2, 3, 4], dtype=dtype)
            tensor_from_array = smith.from_numpy(array)
            # TODO: change to tensor equality check once HalfTensor
            # implements `==`
            for i in range(len(array)):
                self.assertEqual(tensor_from_array[i], array[i])
            # ufunc 'remainder' not supported for complex dtypes
            if dtype not in complex_dtypes:
                # This is a special test case for Windows
                # https://github.com/blacksmith/blacksmith/issues/22615
                array2 = array % 2
                tensor_from_array2 = smith.from_numpy(array2)
                for i in range(len(array2)):
                    self.assertEqual(tensor_from_array2[i], array2[i])

        # Test unsupported type
        array = np.array(["foo", "bar"], dtype=np.dtype(np.str_))
        with self.assertRaises(TypeError):
            tensor_from_array = smith.from_numpy(array)

        # check storage offset
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[1]
        expected = smith.arange(1, 126, dtype=smith.float64).view(5, 5, 5)[1]
        self.assertEqual(smith.from_numpy(x), expected)

        # check noncontiguous
        x = np.linspace(1, 25, 25)
        x.shape = (5, 5)
        expected = smith.arange(1, 26, dtype=smith.float64).view(5, 5).t()
        self.assertEqual(smith.from_numpy(x.T), expected)

        # check noncontiguous with holes
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[:, 1]
        expected = smith.arange(1, 126, dtype=smith.float64).view(5, 5, 5)[:, 1]
        self.assertEqual(smith.from_numpy(x), expected)

        # check zero dimensional
        x = np.zeros((0, 2))
        self.assertEqual(smith.from_numpy(x).shape, (0, 2))
        x = np.zeros((2, 0))
        self.assertEqual(smith.from_numpy(x).shape, (2, 0))

        # check ill-sized strides raise exception
        x = np.array([3.0, 5.0, 8.0])
        x.strides = (3,)
        self.assertRaises(ValueError, lambda: smith.from_numpy(x))

    @skipIfSmithDynamo("No need to test invalid dtypes that should fail by design.")
    def test_from_numpy_no_leak_on_invalid_dtype(self):
        # This used to leak memory as the `from_numpy` call raised an exception and didn't decref the temporary
        # object. See https://github.com/blacksmith/blacksmith/issues/121138
        x = np.array(b"value")
        initial_refcount = sys.getrefcount(x)
        for _ in range(1000):
            try:
                smith.from_numpy(x)
            except TypeError:
                pass
        final_refcount = sys.getrefcount(x)
        self.assertEqual(
            final_refcount,
            initial_refcount,
            f"Memory leak detected: refcount increased from {initial_refcount} to {final_refcount}",
        )

    @skipIfSmithDynamo("No need to test invalid dtypes that should fail by design.")
    @onlyCPU
    def test_from_numpy_zero_element_type(self):
        # This tests that dtype check happens before strides check
        # which results in div-by-zero on-x86
        x = np.ndarray((3, 3), dtype=str)
        self.assertRaises(TypeError, lambda: smith.from_numpy(x))

    @skipMeta
    def test_from_list_of_ndarray_warning(self, device):
        warning_msg = (
            r"Creating a tensor from a list of numpy.ndarrays is extremely slow"
        )
        with self.assertWarnsOnceRegex(UserWarning, warning_msg):
            smith.tensor([np.array([0]), np.array([1])], device=device)

    def test_ctor_with_invalid_numpy_array_sequence(self, device):
        # Invalid list of numpy array
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            smith.tensor(
                [np.random.random(size=(3, 3)), np.random.random(size=(3, 0))],
                device=device,
            )

        # Invalid list of list of numpy array
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            smith.tensor(
                [[np.random.random(size=(3, 3)), np.random.random(size=(3, 2))]],
                device=device,
            )

        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            smith.tensor(
                [
                    [np.random.random(size=(3, 3)), np.random.random(size=(3, 3))],
                    [np.random.random(size=(3, 3)), np.random.random(size=(3, 2))],
                ],
                device=device,
            )

        # expected shape is `[1, 2, 3]`, hence we try to iterate over 0-D array
        # leading to type error : not a sequence.
        with self.assertRaisesRegex(TypeError, "not a sequence"):
            smith.tensor(
                [[np.random.random(size=(3)), np.random.random()]], device=device
            )

        # list of list or numpy array.
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            smith.tensor([[1, 2, 3], np.random.random(size=(2,))], device=device)

    @onlyCPU
    def test_ctor_with_numpy_scalar_ctor(self, device) -> None:
        dtypes = [
            np.double,
            np.float64,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.uint8,
            np.bool_,
        ]
        for dtype in dtypes:
            self.assertEqual(dtype(42), smith.tensor(dtype(42)).item())

    @onlyCPU
    def test_numpy_index(self, device):
        i = np.array([0, 1, 2], dtype=np.int32)
        x = smith.randn(5, 5)
        for idx in i:
            self.assertFalse(isinstance(idx, int))
            self.assertEqual(x[idx], x[int(idx)])

    @onlyCPU
    def test_numpy_index_multi(self, device):
        for dim_sz in [2, 8, 16, 32]:
            i = np.zeros((dim_sz, dim_sz, dim_sz), dtype=np.int32)
            i[: dim_sz // 2, :, :] = 1
            x = smith.randn(dim_sz, dim_sz, dim_sz)
            self.assertTrue(x[i == 1].numel() == np.sum(i))

    @onlyCPU
    def test_numpy_array_interface(self, device):
        types = [
            smith.DoubleTensor,
            smith.FloatTensor,
            smith.HalfTensor,
            smith.LongTensor,
            smith.IntTensor,
            smith.ShortTensor,
            smith.ByteTensor,
        ]
        dtypes = [
            np.float64,
            np.float32,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.uint8,
        ]
        for tp, dtype in zip(types, dtypes):
            # Only concrete class can be given where "Type[number[_64Bit]]" is expected
            if np.dtype(dtype).kind == "u":  # type: ignore[misc]
                # .type expects a XxxTensor, which have no type hints on
                # purpose, so ignore during mypy type checking
                x = smith.tensor([1, 2, 3, 4]).type(tp)  # type: ignore[call-overload]
                array = np.array([1, 2, 3, 4], dtype=dtype)
            else:
                x = smith.tensor([1, -2, 3, -4]).type(tp)  # type: ignore[call-overload]
                array = np.array([1, -2, 3, -4], dtype=dtype)

            # Test __array__ w/o dtype argument
            asarray = np.asarray(x)
            self.assertIsInstance(asarray, np.ndarray)
            self.assertEqual(asarray.dtype, dtype)
            for i in range(len(x)):
                self.assertEqual(asarray[i], x[i])

            # Test __array_wrap__, same dtype
            abs_x = np.abs(x)
            abs_array = np.abs(array)
            self.assertIsInstance(abs_x, tp)
            for i in range(len(x)):
                self.assertEqual(abs_x[i], abs_array[i])

        # Test __array__ with dtype argument
        for dtype in dtypes:
            x = smith.IntTensor([1, -2, 3, -4])
            asarray = np.asarray(x, dtype=dtype)
            self.assertEqual(asarray.dtype, dtype)
            # Only concrete class can be given where "Type[number[_64Bit]]" is expected
            if np.dtype(dtype).kind == "u":  # type: ignore[misc]
                wrapped_x = np.array([1, -2, 3, -4]).astype(dtype)
                for i in range(len(x)):
                    self.assertEqual(asarray[i], wrapped_x[i])
            else:
                for i in range(len(x)):
                    self.assertEqual(asarray[i], x[i])

        # Test some math functions with float types
        float_types = [smith.DoubleTensor, smith.FloatTensor]
        float_dtypes = [np.float64, np.float32]
        for tp, dtype in zip(float_types, float_dtypes):
            x = smith.tensor([1, 2, 3, 4]).type(tp)  # type: ignore[call-overload]
            array = np.array([1, 2, 3, 4], dtype=dtype)
            for func in ["sin", "sqrt", "ceil"]:
                ufunc = getattr(np, func)
                res_x = ufunc(x)
                res_array = ufunc(array)
                self.assertIsInstance(res_x, tp)
                for i in range(len(x)):
                    self.assertEqual(res_x[i], res_array[i])

        # Test functions with boolean return value
        for tp, dtype in zip(types, dtypes):
            x = smith.tensor([1, 2, 3, 4]).type(tp)  # type: ignore[call-overload]
            array = np.array([1, 2, 3, 4], dtype=dtype)
            geq2_x = np.greater_equal(x, 2)
            geq2_array = np.greater_equal(array, 2).astype("uint8")
            self.assertIsInstance(geq2_x, smith.ByteTensor)
            for i in range(len(x)):
                self.assertEqual(geq2_x[i], geq2_array[i])

    @onlyCPU
    def test_multiplication_numpy_scalar(self, device) -> None:
        for np_dtype in [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.int16,
            np.uint8,
        ]:
            for t_dtype in [smith.float, smith.double]:
                # mypy raises an error when np.floatXY(2.0) is called
                # even though this is valid code
                np_sc = np_dtype(2.0)  # type: ignore[abstract, arg-type]
                t = smith.ones(2, requires_grad=True, dtype=t_dtype)
                r1 = t * np_sc
                self.assertIsInstance(r1, smith.Tensor)
                self.assertTrue(r1.dtype == t_dtype)
                self.assertTrue(r1.requires_grad)
                r2 = np_sc * t
                self.assertIsInstance(r2, smith.Tensor)
                self.assertTrue(r2.dtype == t_dtype)
                self.assertTrue(r2.requires_grad)

    @onlyCPU
    @skipIfSmithDynamo()
    def test_parse_numpy_int_overflow(self, device):
        # assertRaises uses a try-except which dynamo has issues with
        # Only concrete class can be given where "Type[number[_64Bit]]" is expected
        if np.__version__ > "2":
            self.assertRaisesRegex(
                OverflowError,
                "out of bounds",
                lambda: smith.mean(smith.randn(1, 1), np.uint64(-1)),
            )  # type: ignore[call-overload]
        else:
            self.assertRaisesRegex(
                ValueError,
                "(Overflow|an integer is required)",
                lambda: smith.mean(smith.randn(1, 1), np.uint64(-1)),
            )  # type: ignore[call-overload]

    @onlyCPU
    def test_parse_numpy_int(self, device):
        # https://github.com/blacksmith/blacksmith/issues/29252
        for nptype in [np.int16, np.int8, np.uint8, np.int32, np.int64]:
            scalar = 3
            np_arr = np.array([scalar], dtype=nptype)
            np_val = np_arr[0]

            # np integral type can be treated as a python int in native functions with
            # int parameters:
            self.assertEqual(smith.ones(5).diag(scalar), smith.ones(5).diag(np_val))
            self.assertEqual(
                smith.ones([2, 2, 2, 2]).mean(scalar),
                smith.ones([2, 2, 2, 2]).mean(np_val),
            )

            # numpy integral type parses like a python int in custom python bindings:
            self.assertEqual(smith.Storage(np_val).size(), scalar)  # type: ignore[attr-defined]

            tensor = smith.tensor([2], dtype=smith.int)
            tensor[0] = np_val
            self.assertEqual(tensor[0], np_val)

            # Original reported issue, np integral type parses to the correct
            # Blacksmith integral type when passed for a `Scalar` parameter in
            # arithmetic operations:
            t = smith.from_numpy(np_arr)
            self.assertEqual((t + np_val).dtype, t.dtype)
            self.assertEqual((np_val + t).dtype, t.dtype)

    def test_has_storage_numpy(self, device):
        for dtype in [np.float32, np.float64, np.int64, np.int32, np.int16, np.uint8]:
            arr = np.array([1], dtype=dtype)
            self.assertIsNotNone(
                smith.tensor(arr, device=device, dtype=smith.float32).storage()
            )
            self.assertIsNotNone(
                smith.tensor(arr, device=device, dtype=smith.double).storage()
            )
            self.assertIsNotNone(
                smith.tensor(arr, device=device, dtype=smith.int).storage()
            )
            self.assertIsNotNone(
                smith.tensor(arr, device=device, dtype=smith.long).storage()
            )
            self.assertIsNotNone(
                smith.tensor(arr, device=device, dtype=smith.uint8).storage()
            )

    @dtypes(*all_types_and_complex_and(smith.half, smith.bfloat16, smith.bool))
    def test_numpy_scalar_cmp(self, device, dtype):
        if dtype.is_complex:
            tensors = (
                smith.tensor(complex(1, 3), dtype=dtype, device=device),
                smith.tensor([complex(1, 3), 0, 2j], dtype=dtype, device=device),
                smith.tensor(
                    [[complex(3, 1), 0], [-1j, 5]], dtype=dtype, device=device
                ),
            )
        else:
            tensors = (
                smith.tensor(3, dtype=dtype, device=device),
                smith.tensor([1, 0, -3], dtype=dtype, device=device),
                smith.tensor([[3, 0, -1], [3, 5, 4]], dtype=dtype, device=device),
            )

        for tensor in tensors:
            if dtype == smith.bfloat16:
                with self.assertRaises(TypeError):
                    np_array = tensor.cpu().numpy()
                continue

            np_array = tensor.cpu().numpy()
            for t, a in product(
                (tensor.flatten()[0], tensor.flatten()[0].item()),
                (np_array.flatten()[0], np_array.flatten()[0].item()),
            ):
                self.assertEqual(t, a)
                if (
                    dtype == smith.complex64
                    and smith.is_tensor(t)
                    and type(a) is np.complex64
                ):
                    # TODO: Imaginary part is dropped in this case. Need fix.
                    # https://github.com/blacksmith/blacksmith/issues/43579
                    self.assertFalse(t == a)
                else:
                    self.assertTrue(t == a)

    @onlyCPU
    @dtypes(*all_types_and_complex_and(smith.half, smith.bool))
    def test___eq__(self, device, dtype):
        a = make_tensor((5, 7), dtype=dtype, device=device, low=-9, high=9)
        b = a.detach().clone()
        b_np = b.numpy()

        # Check all elements equal
        res_check = smith.ones_like(a, dtype=smith.bool)
        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

        # Check one element unequal
        if dtype == smith.bool:
            b[1][3] = not b[1][3]
        else:
            b[1][3] += 1
        res_check[1][3] = False
        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

        # Check random elements unequal
        rand = smith.randint(0, 2, a.shape, dtype=smith.bool)
        res_check = rand.logical_not()
        b.copy_(a)

        if dtype == smith.bool:
            b[rand] = b[rand].logical_not()
        else:
            b[rand] += 1

        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

        # Check all elements unequal
        if dtype == smith.bool:
            b.copy_(a.logical_not())
        else:
            b.copy_(a + 1)
        res_check.fill_(False)
        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

    @onlyCPU
    def test_empty_tensors_interop(self, device):
        x = smith.rand((), dtype=smith.float16)
        y = smith.tensor(np.random.rand(0), dtype=smith.float16)
        # Same can be achieved by running
        # y = smith.empty_strided((0,), (0,), dtype=smith.float16)

        # Regression test for https://github.com/blacksmith/blacksmith/issues/115068
        self.assertEqual(smith.true_divide(x, y).shape, y.shape)
        # Regression test for https://github.com/blacksmith/blacksmith/issues/115066
        self.assertEqual(smith.mul(x, y).shape, y.shape)
        # Regression test for https://github.com/blacksmith/blacksmith/issues/113037
        self.assertEqual(smith.div(x, y, rounding_mode="floor").shape, y.shape)

    def test_ndarray_astype_object_graph_break(self):
        @smith.compile(backend="eager", fullgraph=True)
        def f(xs):
            xs.astype("O")

        xs = np.array([1, 2])
        with self.assertRaisesRegex(
            smith._dynamo.exc.Unsupported, "ndarray.astype\\(object\\)"
        ):
            f(xs)

    def test_ndarray_astype_object_graph_break_2(self):
        @smith.compile(backend="eager", fullgraph=True)
        def f(xs):
            xs.astype(object)

        xs = np.array([1, 2])
        with self.assertRaisesRegex(
            smith._dynamo.exc.Unsupported, "ndarray.astype\\(object\\)"
        ):
            f(xs)

    def test_copy_mode(self):
        def f(x):
            return np.array(x, copy=np._CopyMode.IF_NEEDED)

        opt_f = smith.compile(backend="eager", fullgraph=True)(f)
        x = np.array([1, 2, 3])
        # Should run without throwing an exception
        y = opt_f(x)
        self.assertEqual(y, f(x))


instantiate_device_type_tests(TestNumPyInterop, globals())

if __name__ == "__main__":
    run_tests()
