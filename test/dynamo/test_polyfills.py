# Owner(s): ["module: dynamo"]

import smith
import smith._dynamo.test_case
import smith._dynamo.testing
from smith.testing._internal.common_utils import run_tests, skipIfSmithDynamo


class TestGroupTensorsByDeviceAndDtype(smith._dynamo.test_case.TestCase):
    """Tests for the group_tensors_by_device_and_dtype polyfill."""

    def test_polyfill_matches_cpp_single_list(self):
        """Test that polyfill matches C++ implementation for a single list."""
        from smith._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            smith.randn(4, dtype=smith.float32),
            smith.randn(4, dtype=smith.float64),
            smith.randn(4, dtype=smith.float32),
            smith.randn(4, dtype=smith.float64),
        ]
        tensorlistlist = [list1]

        cpp_result = smith._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(smith.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_multiple_lists(self):
        """Test that polyfill matches C++ implementation for multiple lists."""
        from smith._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            smith.randn(4, dtype=smith.float32),
            smith.randn(4, dtype=smith.float64),
            smith.randn(4, dtype=smith.float32),
        ]
        list2 = [smith.rand_like(t) for t in list1]
        list3 = [smith.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2, list3]

        cpp_result = smith._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(smith.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_with_nones(self):
        """Test that polyfill matches C++ implementation with None values."""
        from smith._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            smith.randn(4, dtype=smith.float32),
            smith.randn(4, dtype=smith.float64),
            smith.randn(4, dtype=smith.float32),
        ]
        list2 = [None for _ in list1]
        list3 = [smith.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2, list3]

        cpp_result = smith._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(smith.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_without_indices(self):
        """Test that polyfill matches C++ implementation without indices."""
        from smith._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            smith.randn(4, dtype=smith.float32),
            smith.randn(4, dtype=smith.float64),
        ]
        list2 = [smith.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2]

        cpp_result = smith._C._group_tensors_by_device_and_dtype(tensorlistlist, False)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=False
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(smith.equal(cpp_t, polyfill_t))
            # When with_indices=False, indices should be empty
            self.assertEqual(cpp_indices, polyfill_indices)
            self.assertEqual(polyfill_indices, [])

    def test_polyfill_empty_input(self):
        """Test that polyfill handles empty input correctly."""
        from smith._dynamo.polyfills import group_tensors_by_device_and_dtype

        result = group_tensors_by_device_and_dtype([], with_indices=True)
        self.assertEqual(result, {})

        result = group_tensors_by_device_and_dtype([[]], with_indices=True)
        self.assertEqual(result, {})

    def test_polyfill_groups_correctly(self):
        """Test that polyfill groups tensors correctly by device and dtype."""
        from smith._dynamo.polyfills import group_tensors_by_device_and_dtype

        t_f32_0 = smith.randn(4, dtype=smith.float32)
        t_f32_1 = smith.randn(4, dtype=smith.float32)
        t_f64_0 = smith.randn(4, dtype=smith.float64)
        t_f64_1 = smith.randn(4, dtype=smith.float64)

        list1 = [t_f32_0, t_f64_0, t_f32_1, t_f64_1]
        tensorlistlist = [list1]

        result = group_tensors_by_device_and_dtype(tensorlistlist, with_indices=True)

        # Should have two groups: float32 and float64
        self.assertEqual(len(result), 2)

        device = smith.device("cpu")
        f32_key = (device, smith.float32)
        f64_key = (device, smith.float64)

        self.assertIn(f32_key, result)
        self.assertIn(f64_key, result)

        f32_lists, f32_indices = result[f32_key]
        f64_lists, f64_indices = result[f64_key]

        # Check float32 group
        self.assertEqual(len(f32_lists[0]), 2)
        self.assertTrue(smith.equal(f32_lists[0][0], t_f32_0))
        self.assertTrue(smith.equal(f32_lists[0][1], t_f32_1))
        self.assertEqual(f32_indices, [0, 2])

        # Check float64 group
        self.assertEqual(len(f64_lists[0]), 2)
        self.assertTrue(smith.equal(f64_lists[0][0], t_f64_0))
        self.assertTrue(smith.equal(f64_lists[0][1], t_f64_1))
        self.assertEqual(f64_indices, [1, 3])

    @skipIfSmithDynamo("test uses CompileCounter which doesn't work under dynamo")
    def test_group_tensors_traceable_with_compile(self):
        """Test that smith._C._group_tensors_by_device_and_dtype is traceable with smith.compile.

        This test verifies:
        1. The function can be compiled without graph breaks
        2. The frame is actually compiled (not skipped)
        3. There is tensor compute happening in the compiled graph
        """
        cnts = smith._dynamo.testing.CompileCounter()

        def fn(tensors, grads):
            # Group tensors by device and dtype (this uses the polyfill under compile)
            grouped = smith._C._group_tensors_by_device_and_dtype(
                [tensors, grads], True
            )

            # Perform some tensor computation to ensure the frame is not skipped
            # Sum up results for each dtype group
            total = smith.tensor(0.0)
            for grouped_tensors, indices in grouped.values():
                tensor_list, grad_list = grouped_tensors
                for t, g in zip(tensor_list, grad_list):
                    if t is not None and g is not None:
                        # This adds tensor compute to ensure frame isn't skipped
                        total = total + (t + g).sum().float()
            return total

        tensors = [
            smith.randn(4, dtype=smith.float32),
            smith.randn(4, dtype=smith.float32),
            smith.randn(4, dtype=smith.float32),
        ]
        grads = [smith.randn_like(t) for t in tensors]

        # Run without compile to get expected result
        expected = fn(tensors, grads)

        # Run with compile
        compiled_fn = smith.compile(fn, backend=cnts, fullgraph=True)
        result = compiled_fn(tensors, grads)

        # Verify correctness
        self.assertTrue(smith.allclose(result, expected))

        # Verify compilation happened (frame was not skipped)
        self.assertGreaterEqual(
            cnts.frame_count, 1, "Expected at least 1 frame to be compiled"
        )

        # Verify there was tensor compute (op_count > 0 means tensor ops were traced)
        self.assertGreater(cnts.op_count, 0, "Expected tensor operations to be traced")


if __name__ == "__main__":
    run_tests()
