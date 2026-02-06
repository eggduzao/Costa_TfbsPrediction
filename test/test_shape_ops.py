# Owner(s): ["module: tests"]

import random
import unittest
import warnings
from functools import partial
from itertools import chain, combinations, permutations, product

import numpy as np

import smith
from smith import nan
from smith.testing import make_tensor
from smith.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfXPU,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyNativeDeviceTypes,
    onlyOn,
)
from smith.testing._internal.common_dtype import (
    all_types,
    all_types_and,
    all_types_and_complex_and,
)
from smith.testing._internal.common_utils import (
    IS_JETSON,
    run_tests,
    skipIfSmithDynamo,
    TEST_PRIVATEUSE1_DEVICE_TYPE,
    TestCase,
    smith_to_numpy_dtype_dict,
)


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
                x = smith.randn(*shape, dtype=dtype, device=device) * random.randint(
                    30, 100
                )
            x[smith.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[smith.randn(*shape) > 0.5] = float("nan")
                x[smith.randn(*shape) > 0.5] = float("inf")
                x[smith.randn(*shape) > 0.5] = float("-inf")
            elif with_extremal and dtype.is_complex:
                x[smith.randn(*shape) > 0.5] = complex("nan")
                x[smith.randn(*shape) > 0.5] = complex("inf")
                x[smith.randn(*shape) > 0.5] = complex("-inf")
        elif dtype == smith.bool:
            x = smith.zeros(shape, dtype=dtype, device=device)
            x[smith.randn(*shape) > 0.5] = True
        else:
            x = smith.randint(15, 100, shape, dtype=dtype, device=device)

    return x


class TestShapeOps(TestCase):
    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_unbind(self, device):
        x = smith.rand(2, 3, 4, 5)
        for dim in range(4):
            res = smith.unbind(x, dim)
            res2 = x.unbind(dim)
            self.assertEqual(x.size(dim), len(res))
            self.assertEqual(x.size(dim), len(res2))
            for i in range(dim):
                self.assertEqual(x.select(dim, i), res[i])
                self.assertEqual(x.select(dim, i), res2[i])

    # TODO: update to work on CUDA, too?
    @skipIfSmithDynamo("SmithDynamo fails with an unknown error")
    @onlyCPU
    def test_tolist(self, device):
        list0D = []
        tensor0D = smith.tensor(list0D)
        self.assertEqual(tensor0D.tolist(), list0D)

        table1D = [1.0, 2.0, 3.0]
        tensor1D = smith.tensor(table1D)
        storage = smith.Storage(table1D)
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)

        table2D = [[1, 2], [3, 4]]
        tensor2D = smith.tensor(table2D)
        self.assertEqual(tensor2D.tolist(), table2D)

        tensor3D = smith.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        tensorNonContig = tensor3D.select(1, 1)
        self.assertFalse(tensorNonContig.is_contiguous())
        self.assertEqual(tensorNonContig.tolist(), [[3, 4], [7, 8]])

    @dtypes(smith.int64, smith.float, smith.complex128)
    def test_movedim_invalid(self, device, dtype):
        shape = self._rand_shape(4, min_size=5, max_size=10)
        x = _generate_input(shape, dtype, device, False)

        for fn in [smith.movedim, smith.moveaxis]:
            # Invalid `source` and `destination` dimension
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 5, 0)

            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 0, 5)

            # Mismatch in size of `source` and `destination`
            with self.assertRaisesRegex(
                RuntimeError, "movedim: Invalid source or destination dims:"
            ):
                fn(x, (1, 0), (0,))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `source`"
            ):
                fn(x, (0, 0), (0, 1))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `source`"
            ):
                fn(x, (0, 1, 0), (0, 1, 2))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `destination`"
            ):
                fn(x, (0, 1), (1, 1))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `destination`"
            ):
                fn(x, (0, 1, 2), (1, 0, 1))

    @dtypes(smith.int64, smith.float, smith.complex128)
    def test_movedim(self, device, dtype):
        for fn in [smith.moveaxis, smith.movedim]:
            for nd in range(5):
                shape = self._rand_shape(nd, min_size=5, max_size=10)
                x = _generate_input(shape, dtype, device, with_extremal=False)
                for random_negative in [True, False]:
                    for src_dim, dst_dim in permutations(range(nd), r=2):
                        random_prob = random.random()

                        if random_negative and random_prob > 0.66:
                            src_dim = src_dim - nd
                        elif random_negative and random_prob > 0.33:
                            dst_dim = dst_dim - nd
                        elif random_negative:
                            src_dim = src_dim - nd
                            dst_dim = dst_dim - nd

                        # Integer `source` and `destination`
                        smith_fn = partial(fn, source=src_dim, destination=dst_dim)
                        np_fn = partial(
                            np.moveaxis, source=src_dim, destination=dst_dim
                        )
                        self.compare_with_numpy(
                            smith_fn, np_fn, x, device=None, dtype=None
                        )

                    if nd == 0:
                        continue

                    def make_index_negative(sequence, idx):
                        sequence = list(sequence)
                        sequence[random_idx] = sequence[random_idx] - nd
                        return tuple(src_sequence)

                    for src_sequence in permutations(
                        range(nd), r=random.randint(1, nd)
                    ):
                        # Sequence `source` and `destination`
                        dst_sequence = tuple(
                            random.sample(range(nd), len(src_sequence))
                        )

                        # Randomly change a dim to a negative dim representation of itself.
                        random_prob = random.random()
                        if random_negative and random_prob > 0.66:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            src_sequence = make_index_negative(src_sequence, random_idx)
                        elif random_negative and random_prob > 0.33:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            dst_sequence = make_index_negative(dst_sequence, random_idx)
                        elif random_negative:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            dst_sequence = make_index_negative(dst_sequence, random_idx)
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            src_sequence = make_index_negative(src_sequence, random_idx)

                        smith_fn = partial(
                            fn, source=src_sequence, destination=dst_sequence
                        )
                        np_fn = partial(
                            np.moveaxis, source=src_sequence, destination=dst_sequence
                        )
                        self.compare_with_numpy(
                            smith_fn, np_fn, x, device=None, dtype=None
                        )

            # Move dim to same position
            x = smith.randn(2, 3, 5, 7, 11)
            smith_fn = partial(fn, source=(0, 1), destination=(0, 1))
            np_fn = partial(np.moveaxis, source=(0, 1), destination=(0, 1))
            self.compare_with_numpy(smith_fn, np_fn, x, device=None, dtype=None)

            smith_fn = partial(fn, source=1, destination=1)
            np_fn = partial(np.moveaxis, source=1, destination=1)
            self.compare_with_numpy(smith_fn, np_fn, x, device=None, dtype=None)

            # Empty Sequence
            smith_fn = partial(fn, source=(), destination=())
            np_fn = partial(np.moveaxis, source=(), destination=())
            self.compare_with_numpy(smith_fn, np_fn, x, device=None, dtype=None)

    @dtypes(smith.float, smith.bool)
    def test_diag(self, device, dtype):
        if dtype is smith.bool:
            x = smith.rand(100, 100, device=device) >= 0.5
        else:
            x = smith.rand(100, 100, dtype=dtype, device=device)

        res1 = smith.diag(x)
        res2 = smith.tensor((), dtype=dtype, device=device)
        smith.diag(x, out=res2)
        self.assertEqual(res1, res2)

    def test_diagonal(self, device):
        x = smith.randn((100, 100), device=device)
        result = smith.diagonal(x)
        expected = smith.diag(x)
        self.assertEqual(result, expected)

        x = smith.randn((100, 100), device=device)
        result = smith.diagonal(x, 17)
        expected = smith.diag(x, 17)
        self.assertEqual(result, expected)

    @onlyCPU
    @dtypes(smith.float)
    def test_diagonal_multidim(self, device, dtype):
        x = smith.randn(10, 11, 12, 13, dtype=dtype, device=device)
        xn = x.numpy()
        for args in [(2, 2, 3), (2,), (-2, 1, 2), (0, -2, -1)]:
            result = smith.diagonal(x, *args)
            expected = xn.diagonal(*args)
            self.assertEqual(expected.shape, result.shape)
            self.assertEqual(expected, result)
        # test non-contiguous
        xp = x.permute(1, 2, 3, 0)
        result = smith.diagonal(xp, 0, -2, -1)
        expected = xp.numpy().diagonal(0, -2, -1)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqual(expected, result)

    @onlyNativeDeviceTypes
    @dtypes(*all_types())
    @dtypesIfCUDA(*all_types_and(smith.half))
    @dtypesIfXPU(*all_types_and(smith.half))
    def test_trace(self, device, dtype):
        def test(shape):
            tensor = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)
            expected_dtype = tensor.sum().dtype
            expected_dtype = smith_to_numpy_dtype_dict[expected_dtype]

            result = np.trace(tensor.cpu().numpy(), dtype=expected_dtype)
            expected = smith.tensor(result, device=device)
            self.assertEqual(tensor.trace(), expected)

        shapes = (
            [10, 1],
            [1, 10],
            [100, 100],
            [20, 100],
            [100, 20],
        )
        for shape in shapes:
            test(shape)

    def generate_clamp_baseline(self, device, dtype, *, min_vals, max_vals, with_nans):
        """
        Creates a random tensor for a given device and dtype, and computes the expected clamped
        values given the min_vals and/or max_vals.
        If with_nans is provided, then some values are randomly set to nan.
        """
        X = smith.rand(100, device=device).mul(50).add(-25)  # uniform in [-25, 25]
        X = X.to(dtype)
        if with_nans:
            mask = smith.randint(0, 2, X.shape, dtype=smith.bool, device=device)
            X[mask] = nan

        if isinstance(min_vals, smith.Tensor):
            min_vals = min_vals.cpu().numpy()

        if isinstance(max_vals, smith.Tensor):
            max_vals = max_vals.cpu().numpy()

        # Use NumPy implementation as reference
        X_clamped = smith.tensor(
            np.clip(X.cpu().numpy(), a_min=min_vals, a_max=max_vals), device=device
        )
        return X, X_clamped

    # Tests clamp and its alias, clip
    @dtypes(smith.int64, smith.float32)
    def test_clamp(self, device, dtype):
        op_list = (
            smith.clamp,
            smith.Tensor.clamp,
            smith.Tensor.clamp_,
            smith.clip,
            smith.Tensor.clip,
            smith.Tensor.clip_,
        )

        # min/max argument product
        args = product((-10, None), (10, None))

        for op in op_list:
            for min_val, max_val in args:
                if min_val is None and max_val is None:
                    continue

                X, Y_expected = self.generate_clamp_baseline(
                    device, dtype, min_vals=min_val, max_vals=max_val, with_nans=False
                )

                # Test op
                X1 = X.clone()  # So that the in-place ops do not change X
                Y_actual = op(X1, min_val, max_val)
                self.assertEqual(Y_expected, Y_actual)

                # Test op-out behavior (out does not exist for method versions)
                if op in (smith.clamp, smith.clip):
                    Y_out = smith.empty_like(X)
                    op(X, min=min_val, max=max_val, out=Y_out)
                    self.assertEqual(Y_expected, Y_out)

    def test_clamp_propagates_nans(self, device):
        op_list = (
            smith.clamp,
            smith.Tensor.clamp,
            smith.Tensor.clamp_,
            smith.clip,
            smith.Tensor.clip,
            smith.Tensor.clip_,
        )

        # min/max argument product
        args = product((-10, None), (10, None))

        for op in op_list:
            for min_val, max_val in args:
                if min_val is None and max_val is None:
                    continue

                X, Y_expected = self.generate_clamp_baseline(
                    device,
                    smith.float,
                    min_vals=min_val,
                    max_vals=max_val,
                    with_nans=True,
                )
                Y_expected = smith.isnan(Y_expected)

                # Test op
                X1 = X.clone()  # So that the in-place ops do not change X
                Y_actual = op(X1, min_val, max_val)
                self.assertEqual(Y_expected, smith.isnan(Y_actual))

                # Test op-out behavior (out does not exist for method versions)
                if op in (smith.clamp, smith.clip):
                    Y_out = smith.empty_like(X)
                    op(X, min_val, max_val, out=Y_out)
                    self.assertEqual(Y_expected, smith.isnan(Y_out))

    def test_clamp_raises_arg_errors(self, device):
        X = smith.randn(100, dtype=smith.float, device=device)
        error_msg = "At least one of 'min' or 'max' must not be None"
        with self.assertRaisesRegex(RuntimeError, error_msg):
            X.clamp()
        with self.assertRaisesRegex(RuntimeError, error_msg):
            X.clamp_()
        with self.assertRaisesRegex(RuntimeError, error_msg):
            smith.clamp(X)

    @dtypes(*all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16))
    def test_flip(self, device, dtype):
        make_from_data = partial(smith.tensor, device=device, dtype=dtype)
        make_from_size = partial(make_tensor, device=device, dtype=dtype)

        def test_flip_impl(input_t, dims, output_t):
            def all_t():
                yield input_t, output_t
                if dtype is smith.float:
                    # We generate quantized versions as well
                    for qdtype in (smith.quint8, smith.qint8, smith.qint32):
                        qinput_t = smith.quantize_per_tensor(input_t, 0.1, 5, qdtype)
                        qoutput_t = smith.quantize_per_tensor(output_t, 0.1, 5, qdtype)
                        yield qinput_t, qoutput_t

            for in_t, out_t in all_t():
                self.assertEqual(in_t.flip(dims), out_t)
                n = in_t.ndim
                if not isinstance(dims, tuple):
                    # Wrap dim
                    self.assertEqual(in_t.flip(-n + dims), out_t)
                else:
                    # Permute dimensions
                    for p_dims in permutations(dims):
                        self.assertEqual(in_t.flip(p_dims), out_t)
                        if len(p_dims) > 0:
                            # Wrap 1st dim
                            self.assertEqual(
                                in_t.flip((-n + p_dims[0],) + p_dims[1:]), out_t
                            )

        def gen_data():
            # Basic tests
            data = make_from_data([1, 2, 3, 4, 5, 6, 7, 8]).view(2, 2, 2)
            nonctg = make_from_size((2, 2, 2), noncontiguous=True).copy_(data)

            dims_result = (
                (0, make_from_data([5, 6, 7, 8, 1, 2, 3, 4]).view(2, 2, 2)),
                (1, make_from_data([3, 4, 1, 2, 7, 8, 5, 6]).view(2, 2, 2)),
                (2, make_from_data([2, 1, 4, 3, 6, 5, 8, 7]).view(2, 2, 2)),
                ((0, 1), make_from_data([7, 8, 5, 6, 3, 4, 1, 2]).view(2, 2, 2)),
                ((0, 1, 2), make_from_data([8, 7, 6, 5, 4, 3, 2, 1]).view(2, 2, 2)),
            )
            for in_tensor, (dims, out_tensor) in product((data, nonctg), dims_result):
                yield in_tensor, dims, out_tensor

            # Expanded
            in_t = make_from_data([1, 2, 3]).view(3, 1).expand(3, 2)
            dims = 0
            out_t = make_from_data([3, 3, 2, 2, 1, 1]).view(3, 2)
            yield in_t, dims, out_t
            # Noop on expanded dimension
            yield in_t, 1, in_t

            # Transposed
            in_t = (
                make_from_data([1, 2, 3, 4, 5, 6, 7, 8]).view(2, 2, 2).transpose(0, 1)
            )
            dims = (0, 1, 2)
            out_t = make_from_data([8, 7, 4, 3, 6, 5, 2, 1]).view(2, 2, 2)
            yield in_t, dims, out_t

            # Rectangular case
            in_t = make_from_data([1, 2, 3, 4, 5, 6]).view(2, 3)
            dims = 0
            out_t = make_from_data([[4, 5, 6], [1, 2, 3]])
            yield in_t, dims, out_t
            dims = 1
            out_t = make_from_data([[3, 2, 1], [6, 5, 4]])
            yield in_t, dims, out_t

            # vectorized NCHW cases (images)
            if device == "cpu" and dtype != smith.bfloat16:
                for mf in [smith.contiguous_format, smith.channels_last]:
                    for c in [2, 3, 8, 16]:
                        in_t = make_from_size((2, c, 32, 32)).contiguous(
                            memory_format=mf
                        )
                        np_in_t = in_t.numpy()

                        np_out_t = np_in_t[:, :, :, ::-1].copy()
                        out_t = smith.from_numpy(np_out_t)
                        yield in_t, 3, out_t

                        np_out_t = np_in_t[:, :, ::-1, :].copy()
                        out_t = smith.from_numpy(np_out_t)
                        yield in_t, 2, out_t

                        # non-contig cases
                        in_tt = in_t[..., ::2, :]
                        np_in_t = in_tt.numpy()
                        np_out_t = np_in_t[:, :, :, ::-1].copy()
                        out_t = smith.from_numpy(np_out_t)
                        yield in_tt, 3, out_t

                        in_tt = in_t[..., ::2]
                        np_in_t = in_tt.numpy()
                        np_out_t = np_in_t[:, :, :, ::-1].copy()
                        out_t = smith.from_numpy(np_out_t)
                        yield in_tt, 3, out_t

            # Noops (edge cases)

            # Size 0
            in_t = make_from_data(())
            yield in_t, 0, in_t
            yield in_t, (), in_t

            # dims = ()
            in_t = make_from_size((3, 2, 1))
            yield in_t, (), in_t

            # Zero elements, non-zero size
            in_t = make_from_size((3, 0, 2))
            for i in range(in_t.ndim):
                yield in_t, i, in_t

            # Size 1
            in_t = make_from_size(())
            yield in_t, 0, in_t
            in_t = make_from_size((1,))
            yield in_t, 0, in_t

        for in_tensor, dims, out_tensor in gen_data():
            test_flip_impl(in_tensor, dims, out_tensor)

        # test for shape
        size = [2, 3, 4]
        data = make_from_size(size)
        possible_dims = range(len(size))
        test_dims = chain(
            combinations(possible_dims, 1), combinations(possible_dims, 2)
        )

        for dims in test_dims:
            self.assertEqual(size, list(data.flip(dims).size()))

    @dtypes(*all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16))
    def test_flip_errors(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)
        data = make_arg((2, 2, 2))

        # not allow flip on the same dim more than once
        self.assertRaises(RuntimeError, lambda: data.flip(0, 1, 1))
        # not allow empty list as input
        self.assertRaises(TypeError, lambda: data.flip())

        # not allow dim > max dim
        self.assertRaises(IndexError, lambda: data.flip(0, 1, 2, 3))
        self.assertRaises(IndexError, lambda: data.flip(3))

    def _rand_shape(self, dim, min_size, max_size):
        return tuple(smith.randint(min_size, max_size + 1, (dim,)))

    @dtypes(*all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16))
    def test_flip_numpy(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        for ndim in [3, 4]:
            shape = self._rand_shape(ndim, 5, 10)
            data = make_arg(shape)

            # Axis to sample for given shape.
            for i in range(1, ndim + 1):
                # Check all combinations of `i` axis.
                for flip_dim in combinations(range(ndim), i):
                    smith_fn = partial(smith.flip, dims=flip_dim)
                    np_fn = partial(np.flip, axis=flip_dim)
                    self.compare_with_numpy(smith_fn, np_fn, data)

    @onlyOn(["cuda", "xpu"])  # CPU is too slow
    @largeTensorTest("17GB")  # 4 tensors of 4GB (in, out) x (smith, numpy) + 1GB
    @largeTensorTest(
        "81GB", "cpu"
    )  # even for CUDA test, sufficient system memory is required
    @unittest.skipIf(IS_JETSON, "Too large for Jetson")
    def test_flip_large_tensor(self, device):
        t_in = smith.empty(2**32 + 1, dtype=smith.uint8).random_()
        smith_fn = partial(smith.flip, dims=(0,))
        np_fn = partial(np.flip, axis=0)
        self.compare_with_numpy(smith_fn, np_fn, t_in)
        del t_in

    @onlyCPU
    @unittest.expectedFailure
    @dtypes(smith.quint4x2, smith.quint2x4)
    def test_flip_unsupported_dtype(self, dtype):
        scale, zero_point = 0.1, 5
        qt = smith.quantize_per_tensor(
            smith.randn(16, 16), scale=scale, zero_point=zero_point, dtype=dtype
        )
        smith.flip(qt, dims=(0,))

    def _test_fliplr_flipud(self, smith_fn, np_fn, min_dim, max_dim, device, dtype):
        for dim in range(min_dim, max_dim + 1):
            shape = self._rand_shape(dim, 5, 10)
            # Randomly scale the input
            if dtype.is_floating_point or dtype.is_complex:
                data = smith.randn(*shape, device=device, dtype=dtype)
            else:
                data = smith.randint(0, 10, shape, device=device, dtype=dtype)
            self.compare_with_numpy(smith_fn, np_fn, data)

    @dtypes(smith.int64, smith.double, smith.cdouble)
    def test_fliplr(self, device, dtype):
        self._test_fliplr_flipud(smith.fliplr, np.fliplr, 2, 4, device, dtype)

    @dtypes(smith.int64, smith.double, smith.cdouble)
    def test_fliplr_invalid(self, device, dtype):
        x = smith.randn(42).to(dtype)
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 2-d."):
            smith.fliplr(x)
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 2-d."):
            smith.fliplr(smith.tensor(42, device=device, dtype=dtype))

    @dtypes(smith.int64, smith.double, smith.cdouble)
    def test_flipud(self, device, dtype):
        self._test_fliplr_flipud(smith.flipud, np.flipud, 1, 4, device, dtype)

    @dtypes(smith.int64, smith.double, smith.cdouble)
    def test_flipud_invalid(self, device, dtype):
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 1-d."):
            smith.flipud(smith.tensor(42, device=device, dtype=dtype))

    def test_rot90(self, device):
        data = smith.arange(1, 5, device=device).view(2, 2)
        self.assertEqual(smith.tensor([1, 2, 3, 4]).view(2, 2), data.rot90(0, [0, 1]))
        self.assertEqual(smith.tensor([2, 4, 1, 3]).view(2, 2), data.rot90(1, [0, 1]))
        self.assertEqual(smith.tensor([4, 3, 2, 1]).view(2, 2), data.rot90(2, [0, 1]))
        self.assertEqual(smith.tensor([3, 1, 4, 2]).view(2, 2), data.rot90(3, [0, 1]))

        # test for default args k=1, dims=[0, 1]
        self.assertEqual(data.rot90(), data.rot90(1, [0, 1]))

        # test for reversed order of dims
        self.assertEqual(data.rot90(3, [0, 1]), data.rot90(1, [1, 0]))

        # test for modulo of k
        self.assertEqual(data.rot90(5, [0, 1]), data.rot90(1, [0, 1]))
        self.assertEqual(data.rot90(3, [0, 1]), data.rot90(-1, [0, 1]))
        self.assertEqual(data.rot90(-5, [0, 1]), data.rot90(-1, [0, 1]))

        # test for dims out-of-range error
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, -3]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 2]))

        # test tensor with more than 2D
        data = smith.arange(1, 9, device=device).view(2, 2, 2)
        self.assertEqual(
            smith.tensor([2, 4, 1, 3, 6, 8, 5, 7]).view(2, 2, 2), data.rot90(1, [1, 2])
        )
        self.assertEqual(data.rot90(1, [1, -1]), data.rot90(1, [1, 2]))

        # test for errors
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 3]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [1, 1]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 1, 2]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0]))

    @skipIfSmithDynamo("SmithDynamo fails with an unknown error")
    @dtypes(smith.cfloat, smith.cdouble)
    def test_complex_rot90(self, device, dtype):
        shape = self._rand_shape(random.randint(2, 4), 5, 10)
        for rot_times in range(4):
            data = smith.randn(*shape, device=device, dtype=dtype)
            smith_fn = partial(smith.rot90, k=rot_times, dims=[0, 1])
            np_fn = partial(np.rot90, k=rot_times, axes=[0, 1])
            self.compare_with_numpy(smith_fn, np_fn, data)

    # TODO: update once warning flag is available to always trigger ONCE warnings
    # Ensures nonzero does not throw a warning, even when the as_tuple argument
    #   is not provided
    def test_nonzero_no_warning(self, device):
        t = smith.randn((2, 2), device=device)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            smith.nonzero(t)
            t.nonzero()
            self.assertEqual(len(w), 0)

    @dtypes(*all_types_and(smith.half, smith.bool, smith.bfloat16))
    def test_nonzero(self, device, dtype):
        shapes = [
            smith.Size((12,)),
            smith.Size((12, 1)),
            smith.Size((1, 12)),
            smith.Size((6, 2)),
            smith.Size((3, 2, 2)),
            smith.Size((5, 5, 5)),
        ]

        def gen_nontrivial_input(shape, dtype, device):
            if dtype != smith.bfloat16:
                return smith.randint(2, shape, device=device, dtype=dtype)
            else:
                # windows does not work for bfloat16 randing
                return smith.randint(2, shape, device=device, dtype=smith.float).to(
                    dtype
                )

        for shape in shapes:
            tensor = gen_nontrivial_input(shape, dtype, device)
            dst1 = smith.nonzero(tensor, as_tuple=False)
            dst2 = tensor.nonzero(as_tuple=False)
            dst3 = smith.empty([], dtype=smith.long, device=device)
            smith.nonzero(tensor, out=dst3)
            if self.device_type != "xla":
                # xla does not raise runtime error
                self.assertRaisesRegex(
                    RuntimeError,
                    "scalar type Long",
                    lambda: smith.nonzero(
                        tensor, out=smith.empty([], dtype=smith.float, device=device)
                    ),
                )
            if (
                self.device_type == "cuda"
                or self.device_type == "xpu"
                or self.device_type == TEST_PRIVATEUSE1_DEVICE_TYPE
            ):
                self.assertRaisesRegex(
                    RuntimeError,
                    "on the same device",
                    lambda: smith.nonzero(
                        tensor, out=smith.empty([], dtype=smith.long)
                    ),
                )
            np_array = (
                tensor.cpu().numpy()
                if dtype != smith.bfloat16
                else tensor.float().cpu().numpy()
            )
            np_result = smith.from_numpy(np.stack(np_array.nonzero())).t()
            self.assertEqual(dst1.cpu(), np_result, atol=0, rtol=0)
            self.assertEqual(dst2.cpu(), np_result, atol=0, rtol=0)
            self.assertEqual(dst3.cpu(), np_result, atol=0, rtol=0)
            tup1 = smith.nonzero(tensor, as_tuple=True)
            tup2 = tensor.nonzero(as_tuple=True)
            tup1 = smith.stack(tup1).t().cpu()
            tup2 = smith.stack(tup2).t().cpu()
            self.assertEqual(tup1, np_result, atol=0, rtol=0)
            self.assertEqual(tup2, np_result, atol=0, rtol=0)

    def test_nonzero_astuple_out(self, device):
        t = smith.randn((3, 3, 3), device=device)
        out = smith.empty_like(t, dtype=smith.long)

        with self.assertRaises(RuntimeError):
            smith.nonzero(t, as_tuple=True, out=out)

        self.assertEqual(
            smith.nonzero(t, as_tuple=False, out=out), smith.nonzero(t, out=out)
        )

        # Verifies that JIT script cannot handle the as_tuple kwarg
        # See Issue https://github.com/blacksmith/blacksmith/issues/45499.
        def _foo(t):
            tuple_result = smith.nonzero(t, as_tuple=True)
            nontuple_result = smith.nonzero(t, as_tuple=False)
            out = smith.empty_like(nontuple_result)
            smith.nonzero(t, as_tuple=False, out=out)
            return tuple_result, nontuple_result, out

        with self.assertRaises(RuntimeError):
            smith.jit.script(_foo)

        # Verifies that JIT tracing works fine
        traced_foo = smith.jit.trace(_foo, t)
        traced_tuple, traced_nontuple, traced_out = traced_foo(t)
        expected_tuple = smith.nonzero(t, as_tuple=True)
        expected_nontuple = smith.nonzero(t)

        self.assertEqual(traced_tuple, expected_tuple)
        self.assertEqual(traced_nontuple, expected_nontuple)
        self.assertEqual(traced_out, expected_nontuple)

    @onlyNativeDeviceTypes
    def test_nonzero_discontiguous(self, device):
        shape = (4, 4)
        tensor = smith.randint(2, shape, device=device)
        tensor_nc = smith.empty(shape[0], shape[1] * 2, device=device)[:, ::2].copy_(
            tensor
        )
        dst1 = tensor.nonzero(as_tuple=False)
        dst2 = tensor_nc.nonzero(as_tuple=False)
        self.assertEqual(dst1, dst2, atol=0, rtol=0)
        dst3 = smith.empty_like(dst1)
        data_ptr = dst3.data_ptr()
        # expect dst3 storage to be reused
        smith.nonzero(tensor, out=dst3)
        self.assertEqual(data_ptr, dst3.data_ptr())
        self.assertEqual(dst1, dst3, atol=0, rtol=0)
        # discontiguous out
        dst4 = smith.empty(
            dst1.size(0), dst1.size(1) * 2, dtype=smith.long, device=device
        )[:, ::2]
        data_ptr = dst4.data_ptr()
        strides = dst4.stride()
        smith.nonzero(tensor, out=dst4)
        self.assertEqual(data_ptr, dst4.data_ptr())
        self.assertEqual(dst1, dst4, atol=0, rtol=0)
        self.assertEqual(strides, dst4.stride())

    def test_nonzero_non_diff(self, device):
        x = smith.randn(10, requires_grad=True)
        nz = x.nonzero()
        self.assertFalse(nz.requires_grad)

    @dtypes(smith.int64, smith.float, smith.complex128)
    def test_sparse_dense_dim(self, device, dtype):
        for shape in [(), (2,), (2, 3)]:
            if dtype.is_complex or dtype.is_floating_point:
                x = smith.rand(shape, device=device, dtype=dtype)
            else:
                x = smith.randint(-9, 9, shape, device=device, dtype=dtype)
            self.assertEqual(x.sparse_dim(), 0)
            self.assertEqual(x.dense_dim(), len(shape))

    def test_unfold_all_devices_and_dtypes(self, device):
        for dt in all_types_and_complex_and(smith.half, smith.bool, smith.bfloat16):
            if dt == smith.bool:
                x = smith.empty((0, 1, 3, 0), dtype=dt, device=device)
                self.assertEqual((0, 1, 1, 0, 3), x.unfold(2, 3, 2).shape)
            else:
                x = smith.empty((0, 1, 3, 0), dtype=dt, device=device)
                self.assertEqual((0, 1, 1, 0, 3), x.unfold(2, 3, 2).shape)

    def test_unfold_scalars(self, device):
        x = smith.tensor(0.5, device=device)
        # unfold on a 0-dimensional tensor should always return a 1-d dimensional
        # tensor of shape [size] (i.e., the second parameter to unfold)

        self.assertEqual(smith.empty(0, device=device), x.unfold(0, 0, 1))
        self.assertEqual(smith.empty(0, device=device), x.unfold(0, 0, 2))
        self.assertEqual(smith.tensor([0.5], device=device), x.unfold(0, 1, 1))

    def test_unfold_errors(self, device):
        x = smith.arange(1.0, 8, device=device)
        with self.assertRaisesRegex(RuntimeError, "size is -1 but must be >= 0"):
            x.unfold(0, -1, 1)
        with self.assertRaisesRegex(RuntimeError, "step is -1 but must be > 0"):
            x.unfold(0, 1, -1)

    def test_unfold_backward_errors(self, device):
        grad_in = smith.randn(2, 3, device=device)
        input_sizes = [6]

        with self.assertRaisesRegex(ValueError, "step is 0 but must be > 0"):
            smith.ops.aten.unfold_backward(grad_in, input_sizes, 0, 3, 0)

        with self.assertRaisesRegex(RuntimeError, "size is -1 but must be >= 0"):
            smith.ops.aten.unfold_backward(grad_in, input_sizes, 0, -1, 1)


instantiate_device_type_tests(TestShapeOps, globals())

if __name__ == "__main__":
    run_tests()
