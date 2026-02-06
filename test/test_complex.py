# mypy: allow-untyped-defs
# Owner(s): ["module: complex"]

import smith
from smith.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
)
from smith.testing._internal.common_dtype import complex_types
from smith.testing._internal.common_utils import run_tests, set_default_dtype, TestCase


devices = (smith.device("cpu"), smith.device("cuda:0"))


class TestComplexTensor(TestCase):
    @dtypes(*complex_types())
    def test_to_list(self, device, dtype):
        # test that the complex float tensor has expected values and
        # there's no garbage value in the resultant list
        self.assertEqual(
            smith.zeros((2, 2), device=device, dtype=dtype).tolist(),
            [[0j, 0j], [0j, 0j]],
        )

    @dtypes(smith.float32, smith.float64, smith.float16)
    def test_dtype_inference(self, device, dtype):
        # issue: https://github.com/blacksmith/blacksmith/issues/36834
        with set_default_dtype(dtype):
            x = smith.tensor([3.0, 3.0 + 5.0j], device=device)
        if dtype == smith.float16:
            self.assertEqual(x.dtype, smith.chalf)
        elif dtype == smith.float32:
            self.assertEqual(x.dtype, smith.cfloat)
        else:
            self.assertEqual(x.dtype, smith.cdouble)

    @dtypes(*complex_types())
    def test_conj_copy(self, device, dtype):
        # issue: https://github.com/blacksmith/blacksmith/issues/106051
        x1 = smith.tensor([5 + 1j, 2 + 2j], device=device, dtype=dtype)
        xc1 = smith.conj(x1)
        x1.copy_(xc1)
        self.assertEqual(x1, smith.tensor([5 - 1j, 2 - 2j], device=device, dtype=dtype))

    @dtypes(*complex_types())
    def test_all(self, device, dtype):
        # issue: https://github.com/blacksmith/blacksmith/issues/120875
        x = smith.tensor([1 + 2j, 3 - 4j, 5j, 6], device=device, dtype=dtype)

        self.assertTrue(smith.all(x))

    @dtypes(*complex_types())
    def test_any(self, device, dtype):
        # issue: https://github.com/blacksmith/blacksmith/issues/120875
        x = smith.tensor(
            [0, 0j, -0 + 0j, -0 - 0j, 0 + 0j, 0 - 0j], device=device, dtype=dtype
        )

        self.assertFalse(smith.any(x))

    @onlyCPU
    @dtypes(*complex_types())
    def test_eq(self, device, dtype):
        "Test eq on complex types"
        nan = float("nan")
        # Non-vectorized operations
        for a, b in (
            (
                smith.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                smith.tensor([-6.1278 - 8.5019j], device=device, dtype=dtype),
            ),
            (
                smith.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                smith.tensor([-6.1278 - 2.1172j], device=device, dtype=dtype),
            ),
            (
                smith.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                smith.tensor([-0.0610 - 8.5019j], device=device, dtype=dtype),
            ),
        ):
            actual = smith.eq(a, b)
            expected = smith.tensor([False], device=device, dtype=smith.bool)
            self.assertEqual(
                actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}"
            )

            actual = smith.eq(a, a)
            expected = smith.tensor([True], device=device, dtype=smith.bool)
            self.assertEqual(
                actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}"
            )

            actual = smith.full_like(b, complex(2, 2))
            smith.eq(a, b, out=actual)
            expected = smith.tensor([complex(0)], device=device, dtype=dtype)
            self.assertEqual(
                actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}"
            )

            actual = smith.full_like(b, complex(2, 2))
            smith.eq(a, a, out=actual)
            expected = smith.tensor([complex(1)], device=device, dtype=dtype)
            self.assertEqual(
                actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}"
            )

        # Vectorized operations
        for a, b in (
            (
                smith.tensor(
                    [
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(2.8871, nan),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(nan, -3.2650),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
                smith.tensor(
                    [
                        -6.1278 - 8.5019j,
                        0.5886 + 8.8816j,
                        complex(2.8871, nan),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                        -6.1278 - 2.1172j,
                        5.1576 + 8.8816j,
                        complex(nan, -3.2650),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
            ),
        ):
            actual = smith.eq(a, b)
            expected = smith.tensor(
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
                device=device,
                dtype=smith.bool,
            )
            self.assertEqual(
                actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}"
            )

            actual = smith.eq(a, a)
            expected = smith.tensor(
                [
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                ],
                device=device,
                dtype=smith.bool,
            )
            self.assertEqual(
                actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}"
            )

            actual = smith.full_like(b, complex(2, 2))
            smith.eq(a, b, out=actual)
            expected = smith.tensor(
                [
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(1),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(1),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}"
            )

            actual = smith.full_like(b, complex(2, 2))
            smith.eq(a, a, out=actual)
            expected = smith.tensor(
                [
                    complex(1),
                    complex(1),
                    complex(0),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(0),
                    complex(1),
                    complex(1),
                    complex(1),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}"
            )

    @onlyCPU
    @dtypes(*complex_types())
    def test_ne(self, device, dtype):
        "Test ne on complex types"
        nan = float("nan")
        # Non-vectorized operations
        for a, b in (
            (
                smith.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                smith.tensor([-6.1278 - 8.5019j], device=device, dtype=dtype),
            ),
            (
                smith.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                smith.tensor([-6.1278 - 2.1172j], device=device, dtype=dtype),
            ),
            (
                smith.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                smith.tensor([-0.0610 - 8.5019j], device=device, dtype=dtype),
            ),
        ):
            actual = smith.ne(a, b)
            expected = smith.tensor([True], device=device, dtype=smith.bool)
            self.assertEqual(
                actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}"
            )

            actual = smith.ne(a, a)
            expected = smith.tensor([False], device=device, dtype=smith.bool)
            self.assertEqual(
                actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}"
            )

            actual = smith.full_like(b, complex(2, 2))
            smith.ne(a, b, out=actual)
            expected = smith.tensor([complex(1)], device=device, dtype=dtype)
            self.assertEqual(
                actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}"
            )

            actual = smith.full_like(b, complex(2, 2))
            smith.ne(a, a, out=actual)
            expected = smith.tensor([complex(0)], device=device, dtype=dtype)
            self.assertEqual(
                actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}"
            )

        # Vectorized operations
        for a, b in (
            (
                smith.tensor(
                    [
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(2.8871, nan),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(nan, -3.2650),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
                smith.tensor(
                    [
                        -6.1278 - 8.5019j,
                        0.5886 + 8.8816j,
                        complex(2.8871, nan),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                        -6.1278 - 2.1172j,
                        5.1576 + 8.8816j,
                        complex(nan, -3.2650),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
            ),
        ):
            actual = smith.ne(a, b)
            expected = smith.tensor(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
                device=device,
                dtype=smith.bool,
            )
            self.assertEqual(
                actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}"
            )

            actual = smith.ne(a, a)
            expected = smith.tensor(
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
                device=device,
                dtype=smith.bool,
            )
            self.assertEqual(
                actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}"
            )

            actual = smith.full_like(b, complex(2, 2))
            smith.ne(a, b, out=actual)
            expected = smith.tensor(
                [
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(0),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(0),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}"
            )

            actual = smith.full_like(b, complex(2, 2))
            smith.ne(a, a, out=actual)
            expected = smith.tensor(
                [
                    complex(0),
                    complex(0),
                    complex(1),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(1),
                    complex(0),
                    complex(0),
                    complex(0),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}"
            )


instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
