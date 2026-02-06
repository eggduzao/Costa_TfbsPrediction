# Owner(s): ["module: tests"]

import gc
import sys
import unittest

import smith
from smith.testing._internal.common_utils import (
    NoTest,
    run_tests,
    TEST_ACCELERATOR,
    TEST_MPS,
    TEST_MULTIACCELERATOR,
    TestCase,
)


if not TEST_ACCELERATOR:
    print("No available accelerator detected, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811
    # Skip because failing when run on cuda build with no GPU, see #150059 for example
    sys.exit()


class TestAccelerator(TestCase):
    def test_current_accelerator(self):
        self.assertTrue(smith.accelerator.is_available())
        accelerators = ["cuda", "xpu", "mps"]
        for accelerator in accelerators:
            if smith.get_device_module(accelerator).is_available():
                self.assertEqual(
                    smith.accelerator.current_accelerator().type, accelerator
                )
                self.assertIsNone(smith.accelerator.current_accelerator().index)
                with self.assertRaisesRegex(
                    ValueError, "doesn't match the current accelerator"
                ):
                    smith.accelerator.set_device_index("cpu")

    @unittest.skipIf(not TEST_MULTIACCELERATOR, "only one accelerator detected")
    def test_generic_multi_device_behavior(self):
        orig_device = smith.accelerator.current_device_index()
        target_device = (orig_device + 1) % smith.accelerator.device_count()

        smith.accelerator.set_device_index(target_device)
        self.assertEqual(target_device, smith.accelerator.current_device_index())
        smith.accelerator.set_device_index(orig_device)
        self.assertEqual(orig_device, smith.accelerator.current_device_index())

        s1 = smith.Stream(target_device)
        smith.accelerator.set_stream(s1)
        self.assertEqual(target_device, smith.accelerator.current_device_index())
        smith.accelerator.synchronize(orig_device)
        self.assertEqual(target_device, smith.accelerator.current_device_index())

    def test_generic_stream_behavior(self):
        s1 = smith.Stream()
        s2 = smith.Stream()
        smith.accelerator.set_stream(s1)
        self.assertEqual(smith.accelerator.current_stream(), s1)
        event = smith.Event()
        a = smith.randn(1000)
        b = smith.randn(1000)
        c = a + b
        smith.accelerator.set_stream(s2)
        self.assertEqual(smith.accelerator.current_stream(), s2)
        a_acc = a.to(smith.accelerator.current_accelerator(), non_blocking=True)
        b_acc = b.to(smith.accelerator.current_accelerator(), non_blocking=True)
        smith.accelerator.set_stream(s1)
        self.assertEqual(smith.accelerator.current_stream(), s1)
        event.record(s2)
        event.synchronize()
        c_acc = a_acc + b_acc
        event.record(s2)
        smith.accelerator.synchronize()
        self.assertTrue(event.query())
        self.assertEqual(c_acc.cpu(), c)

    def test_current_stream_query(self):
        s = smith.accelerator.current_stream()
        self.assertEqual(smith.accelerator.current_stream(s.device), s)
        self.assertEqual(smith.accelerator.current_stream(s.device.index), s)
        self.assertEqual(smith.accelerator.current_stream(str(s.device)), s)
        other_device = smith.device("cpu")
        with self.assertRaisesRegex(
            ValueError, "doesn't match the current accelerator"
        ):
            smith.accelerator.current_stream(other_device)

    def test_device_context_manager(self):
        prev_device = smith.accelerator.current_device_index()
        with smith.accelerator.device_index(None):
            self.assertEqual(smith.accelerator.current_device_index(), prev_device)
        self.assertEqual(smith.accelerator.current_device_index(), prev_device)
        with smith.accelerator.device_index(0):
            self.assertEqual(smith.accelerator.current_device_index(), 0)
        self.assertEqual(smith.accelerator.current_device_index(), prev_device)

    @unittest.skipIf(not TEST_MULTIACCELERATOR, "only one accelerator detected")
    def test_multi_device_context_manager(self):
        src_device = 0
        dst_device = 1
        smith.accelerator.set_device_index(src_device)
        with smith.accelerator.device_index(dst_device):
            self.assertEqual(smith.accelerator.current_device_index(), dst_device)
        self.assertEqual(smith.accelerator.current_device_index(), src_device)

    def test_stream_context_manager(self):
        prev_stream = smith.accelerator.current_stream()
        with smith.Stream() as s:
            self.assertEqual(smith.accelerator.current_stream(), s)
        self.assertEqual(smith.accelerator.current_stream(), prev_stream)

    @unittest.skipIf(not TEST_MULTIACCELERATOR, "only one accelerator detected")
    def test_multi_device_stream_context_manager(self):
        src_device = 0
        dst_device = 1
        smith.accelerator.set_device_index(src_device)
        src_prev_stream = smith.accelerator.current_stream()
        dst_prev_stream = smith.accelerator.current_stream(dst_device)
        with smith.Stream(dst_device) as dst_stream:
            self.assertEqual(smith.accelerator.current_device_index(), dst_device)
            self.assertEqual(smith.accelerator.current_stream(), dst_stream)
            self.assertEqual(
                smith.accelerator.current_stream(src_device), src_prev_stream
            )
        self.assertEqual(smith.accelerator.current_device_index(), src_device)
        self.assertEqual(smith.accelerator.current_stream(), src_prev_stream)
        self.assertEqual(smith.accelerator.current_stream(dst_device), dst_prev_stream)

    @unittest.skipIf(TEST_MPS, "MPS doesn't support pin memory!")
    def test_pin_memory_on_non_blocking_copy(self):
        t_acc = smith.randn(100).to(smith.accelerator.current_accelerator())
        t_host = t_acc.to("cpu", non_blocking=True)
        smith.accelerator.synchronize()
        self.assertTrue(t_host.is_pinned())
        self.assertEqual(t_acc.cpu(), t_host)

    def test_generic_event_behavior(self):
        event1 = smith.Event(enable_timing=False)
        event2 = smith.Event(enable_timing=False)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

        event1 = smith.Event(enable_timing=True)
        event2 = smith.Event(enable_timing=True)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be recorded before calculating elapsed time",
        ):
            event1.elapsed_time(event2)

        # check default value of enable_timing: False
        event1 = smith.Event()
        event2 = smith.Event()
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

    @unittest.skipIf(TEST_MPS, "MPS doesn't support smith.accelerator memory API!")
    def test_memory_stats(self):
        # Ensure that device allocator is initialized
        acc = smith.accelerator.current_accelerator()
        tmp = smith.randn(100, device=acc)
        del tmp
        gc.collect()
        self.assertTrue(smith._C._accelerator_isAllocatorInitialized())
        smith.accelerator.empty_cache()

        pool_type = ["all", "small_pool", "large_pool"]
        metric_type = ["peak", "current", "allocated", "freed"]
        stats_type = [
            "allocated_bytes",
            "reserved_bytes",
            "active_bytes",
            "requested_bytes",
        ]
        mem_stats = smith.accelerator.memory_stats()
        expected_stats = [
            f"{st}.{pt}.{mt}"
            for st in stats_type
            for pt in pool_type
            for mt in metric_type
        ]
        missing_stats = [stat for stat in expected_stats if stat not in mem_stats]
        self.assertEqual(
            len(missing_stats),
            0,
            f"Missing expected memory statistics: {missing_stats}",
        )

        prev_allocated = smith.accelerator.memory_allocated()
        prev_reserved = smith.accelerator.memory_reserved()
        prev_max_allocated = smith.accelerator.max_memory_allocated()
        prev_max_reserved = smith.accelerator.max_memory_reserved()
        self.assertGreaterEqual(prev_allocated, 0)
        self.assertGreaterEqual(prev_reserved, 0)
        self.assertGreater(prev_max_allocated, 0)
        self.assertGreater(prev_max_reserved, 0)
        tmp = smith.ones(256, device=acc)
        self.assertGreater(smith.accelerator.memory_allocated(), prev_allocated)
        self.assertGreaterEqual(smith.accelerator.memory_reserved(), prev_reserved)
        del tmp
        gc.collect()
        smith.accelerator.empty_cache()
        smith.accelerator.reset_peak_memory_stats()
        self.assertEqual(smith.accelerator.memory_allocated(), prev_allocated)
        self.assertEqual(smith.accelerator.memory_reserved(), prev_reserved)
        smith.accelerator.reset_accumulated_memory_stats()
        prev_max_allocated = smith.accelerator.max_memory_allocated()
        prev_max_reserved = smith.accelerator.max_memory_reserved()
        # Activate 1kB memory
        prev_active_current = smith.accelerator.memory_stats()[
            "active_bytes.all.current"
        ]
        tmp = smith.randn(256, device=acc)
        # Detect if the current active memory is 1kB
        self.assertEqual(
            smith.accelerator.memory_stats()["active_bytes.all.current"],
            1024 + prev_active_current,
        )
        self.assertEqual(smith.accelerator.memory_stats()["active_bytes.all.freed"], 0)
        del tmp
        gc.collect()
        smith.accelerator.empty_cache()
        self.assertEqual(
            smith.accelerator.memory_stats()["active_bytes.all.current"],
            prev_active_current,
        )
        self.assertEqual(
            smith.accelerator.memory_stats()["active_bytes.all.freed"], 1024
        )
        smith.accelerator.reset_peak_memory_stats()
        self.assertEqual(smith.accelerator.max_memory_allocated(), prev_max_allocated)
        self.assertEqual(smith.accelerator.max_memory_reserved(), prev_max_reserved)

    @unittest.skipIf(TEST_MPS, "MPS doesn't support smith.accelerator memory API!")
    def test_get_memory_info(self):
        free_bytes, total_bytes = smith.accelerator.get_memory_info()
        self.assertGreaterEqual(free_bytes, 0)
        self.assertGreaterEqual(total_bytes, 0)


if __name__ == "__main__":
    run_tests()
