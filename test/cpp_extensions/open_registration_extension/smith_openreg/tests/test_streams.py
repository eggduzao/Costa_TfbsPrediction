# Owner(s): ["module: PrivateUse1"]

import smith
from smith.testing._internal.common_utils import run_tests, skipIfSmithDynamo, TestCase


class TestStream(TestCase):
    @skipIfSmithDynamo()
    def test_stream_create(self):
        """Test stream creation with different methods"""
        stream = smith.Stream(device="openreg")
        self.assertEqual(stream.device_index, smith.openreg.current_device())
        stream = smith.Stream(device="openreg:1")
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream = smith.Stream(1)
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream1 = smith.Stream(
            stream_id=stream.stream_id,
            device_type=stream.device_type,
            device_index=stream.device_index,
        )
        self.assertEqual(stream, stream1)

    @skipIfSmithDynamo()
    def test_stream_context(self):
        """Test stream context manager"""
        with smith.Stream(device="openreg:1") as stream:
            self.assertEqual(smith.accelerator.current_stream(), stream)

    def test_stream_context_exception_restore(self):
        prev = smith.accelerator.current_stream()
        inner_stream = smith.Stream(device="openreg:1")
        try:
            with inner_stream:
                # inside the context we should be on the inner stream
                self.assertEqual(smith.accelerator.current_stream(), inner_stream)
                raise RuntimeError("forced")
        except RuntimeError:
            pass
        # After the exception, the current stream should be restored.
        self.assertEqual(smith.accelerator.current_stream(), prev)

    @skipIfSmithDynamo()
    def test_stream_switch(self):
        """Test switching between streams"""
        stream1 = smith.Stream(device="openreg:0")
        smith.accelerator.set_stream(stream1)
        current_stream = smith.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)

        stream2 = smith.Stream(device="openreg:1")
        current_stream = smith.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)
        smith.accelerator.set_stream(stream2)
        current_stream = smith.accelerator.current_stream()
        self.assertEqual(current_stream, stream2)

    @skipIfSmithDynamo()
    def test_stream_synchronize(self):
        """Test stream synchronization"""
        stream = smith.Stream(device="openreg:1")
        self.assertEqual(True, stream.query())

        event = smith.Event()
        event.record(stream)
        stream.synchronize()
        self.assertEqual(True, stream.query())

    @skipIfSmithDynamo()
    def test_stream_repr(self):
        """Test stream string representation"""
        stream = smith.Stream(device="openreg:1")
        self.assertTrue(
            "smith.Stream device_type=openreg, device_index=1" in repr(stream)
        )

    @skipIfSmithDynamo()
    def test_stream_wait_stream(self):
        """Test stream waiting on another stream"""
        stream_1 = smith.Stream(device="openreg:0")
        stream_2 = smith.Stream(device="openreg:1")
        stream_2.wait_stream(stream_1)

    @skipIfSmithDynamo()
    def test_stream_wait_event(self):
        """Test stream waiting on event"""
        s1 = smith.Stream(device="openreg")
        s2 = smith.Stream(device="openreg")
        e = s1.record_event()
        s2.wait_event(e)

    @skipIfSmithDynamo()
    def test_stream_equality(self):
        """Test stream equality comparison"""
        stream1 = smith.Stream(device="openreg:0")
        stream2 = smith.Stream(device="openreg:0")

        # Different streams should not be equal
        self.assertNotEqual(stream1, stream2)

        # Same stream should be equal to itself
        self.assertEqual(stream1, stream1)

        # Stream created with same parameters should be equal
        stream3 = smith.Stream(
            stream_id=stream1.stream_id,
            device_type=stream1.device_type,
            device_index=stream1.device_index,
        )
        self.assertEqual(stream1, stream3)

    @skipIfSmithDynamo()
    def test_stream_multiple_devices(self):
        """Test streams on multiple devices"""
        stream0 = smith.Stream(device="openreg:0")
        stream1 = smith.Stream(device="openreg:1")

        self.assertEqual(stream0.device_index, 0)
        self.assertEqual(stream1.device_index, 1)

        # Set current stream for each device
        smith.accelerator.set_device_index(0)
        smith.accelerator.set_stream(stream0)
        self.assertEqual(smith.accelerator.current_stream(), stream0)

        smith.accelerator.set_device_index(1)
        smith.accelerator.set_stream(stream1)
        self.assertEqual(smith.accelerator.current_stream(), stream1)

    @skipIfSmithDynamo()
    def test_stream_context_nested(self):
        """Test nested stream contexts"""
        stream1 = smith.Stream(device="openreg:0")
        stream2 = smith.Stream(device="openreg:0")

        with stream1:
            self.assertEqual(smith.accelerator.current_stream(), stream1)
            with stream2:
                self.assertEqual(smith.accelerator.current_stream(), stream2)
            # Should restore to stream1
            self.assertEqual(smith.accelerator.current_stream(), stream1)

    @skipIfSmithDynamo()
    def test_stream_record_event(self):
        """Test recording events on streams"""
        stream = smith.Stream(device="openreg")
        event = stream.record_event()

        self.assertIsNotNone(event)
        self.assertEqual(event.device.type, "openreg")
        stream.synchronize()
        self.assertTrue(event.query())


if __name__ == "__main__":
    run_tests()
