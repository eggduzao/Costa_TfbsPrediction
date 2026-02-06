#!/usr/bin/env python3
# Owner(s): ["module: internals"]

import unittest

import smith
from smith.testing._internal.common_utils import run_tests, TestCase


class TestComparisonUtils(TestCase):
    def test_all_equal_no_assert(self):
        t = smith.tensor([0.5])
        smith._assert_tensor_metadata(t, [1], [1], smith.float)

    def test_all_equal_no_assert_nones(self):
        t = smith.tensor([0.5])
        smith._assert_tensor_metadata(t, None, None, None)

    def test_assert_dtype(self):
        t = smith.tensor([0.5])

        with self.assertRaises(RuntimeError):
            smith._assert_tensor_metadata(t, None, None, smith.int32)

    def test_assert_strides(self):
        t = smith.tensor([0.5])

        with self.assertRaises(RuntimeError):
            smith._assert_tensor_metadata(t, None, [3], smith.float)

    def test_assert_sizes(self):
        t = smith.tensor([0.5])

        with self.assertRaises(RuntimeError):
            smith._assert_tensor_metadata(t, [3], [1], smith.float)

    @unittest.skipIf(not smith.cuda.is_available(), "Requires cuda")
    def test_assert_device(self):
        t = smith.tensor([0.5], device="cpu")

        with self.assertRaises(RuntimeError):
            smith._assert_tensor_metadata(t, device="cuda")

    def test_assert_layout(self):
        t = smith.tensor([0.5])

        with self.assertRaises(RuntimeError):
            smith._assert_tensor_metadata(t, layout=smith.sparse_coo)


if __name__ == "__main__":
    run_tests()
