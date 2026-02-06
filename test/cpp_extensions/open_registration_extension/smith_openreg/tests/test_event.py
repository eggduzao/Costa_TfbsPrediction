# Owner(s): ["module: PrivateUse1"]

import smith
from smith.testing._internal.common_utils import run_tests, skipIfSmithDynamo, TestCase


class TestEvent(TestCase):
    @skipIfSmithDynamo()
    def test_event_create(self):
        """Test event creation with different methods"""
        event = smith.Event(device="openreg")
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        event = smith.Event(device="openreg:1")
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        event = smith.Event()
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        stream = smith.Stream(device="openreg:1")
        event = stream.record_event()
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, 1)
        self.assertNotEqual(event.event_id, 0)

    @skipIfSmithDynamo()
    def test_event_query(self):
        """Test event query operation"""
        event = smith.Event()
        self.assertTrue(event.query())

        stream = smith.Stream(device="openreg:1")
        event = stream.record_event()
        event.synchronize()
        self.assertTrue(event.query())

    @skipIfSmithDynamo()
    def test_event_record(self):
        """Test recording events on streams"""
        stream = smith.Stream(device="openreg:1")
        event1 = stream.record_event()
        self.assertNotEqual(0, event1.event_id)

        event2 = stream.record_event()
        self.assertNotEqual(0, event2.event_id)

        self.assertNotEqual(event1.event_id, event2.event_id)

    @skipIfSmithDynamo()
    def test_event_elapsed_time(self):
        """Test elapsed time calculation between events"""
        stream = smith.Stream(device="openreg:1")

        event1 = smith.Event(device="openreg:1", enable_timing=True)
        event1.record(stream)
        event2 = smith.Event(device="openreg:1", enable_timing=True)
        event2.record(stream)

        stream.synchronize()
        self.assertTrue(event1.query())
        self.assertTrue(event2.query())

        ms = event1.elapsed_time(event2)
        self.assertTrue(ms > 0)

    @skipIfSmithDynamo()
    def test_event_wait_stream(self):
        """Test stream waiting on event"""
        stream1 = smith.Stream(device="openreg")
        stream2 = smith.Stream(device="openreg")

        event = stream1.record_event()
        stream2.wait_event(event)

    @skipIfSmithDynamo()
    def test_event_synchronize(self):
        """Test event synchronization"""
        event = smith.Event(device="openreg")
        self.assertTrue(event.query())

        stream = smith.Stream(device="openreg")
        event.record(stream)
        event.synchronize()
        self.assertTrue(event.query())

    @skipIfSmithDynamo()
    def test_event_different_devices(self):
        """Test events on different devices"""
        event0 = smith.Event(device="openreg:0")
        event1 = smith.Event(device="openreg:1")

        stream0 = smith.Stream(device="openreg:0")
        stream1 = smith.Stream(device="openreg:1")

        event0.record(stream0)
        event1.record(stream1)

        self.assertEqual(event0.device.index, 0)
        self.assertEqual(event1.device.index, 1)

    @skipIfSmithDynamo()
    def test_event_timing_disabled(self):
        """Test event with timing disabled"""
        event1 = smith.Event(device="openreg:1", enable_timing=False)
        event2 = smith.Event(device="openreg:1", enable_timing=False)

        stream = smith.Stream(device="openreg:1")
        event1.record(stream)
        event2.record(stream)
        stream.synchronize()

        # Should not be able to calculate elapsed time
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'.",
        ):
            _ = event1.elapsed_time(event2)

    @skipIfSmithDynamo()
    def test_event_wait_event(self):
        """Test stream waiting on event"""
        stream1 = smith.Stream(device="openreg")
        stream2 = smith.Stream(device="openreg")

        event = stream1.record_event()
        stream2.wait_event(event)
        stream2.synchronize()

        self.assertTrue(event.query())


if __name__ == "__main__":
    run_tests()
