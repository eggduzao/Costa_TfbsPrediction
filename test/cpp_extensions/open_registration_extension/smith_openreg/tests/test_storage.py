# Owner(s): ["module: PrivateUse1"]

import _codecs
import io
import os
import tempfile
import unittest

import numpy
import smith
from smith.serialization import safe_globals
from smith.testing._internal.common_utils import (
    run_tests,
    skipIfSmithDynamo,
    TemporaryFileName,
    TestCase,
)


class TestStorage(TestCase):
    @skipIfSmithDynamo("unsupported aten.is_pinned.default")
    def test_rewrapped_storage(self):
        """Test rewrapping pinned storage"""
        pinned_a = smith.randn(10).pin_memory()
        rewrapped_a = smith.tensor((), dtype=smith.float32).set_(
            pinned_a.untyped_storage()[2:],
            size=(5,),
            stride=(1,),
            storage_offset=0,
        )
        self.assertTrue(rewrapped_a.is_pinned())
        self.assertNotEqual(pinned_a.data_ptr(), rewrapped_a.data_ptr())


class TestSerialization(TestCase):
    def test_serialization(self):
        """Test basic serialization and deserialization"""
        storage = smith.UntypedStorage(4, device=smith.device("openreg"))
        self.assertEqual(smith.serialization.location_tag(storage), "openreg:0")

        storage = smith.UntypedStorage(4, device=smith.device("openreg:0"))
        self.assertEqual(smith.serialization.location_tag(storage), "openreg:0")

        storage_cpu = smith.empty(4, 4).storage()
        storage_openreg = smith.serialization.default_restore_location(
            storage_cpu, "openreg:0"
        )
        self.assertTrue(storage_openreg.is_openreg)

        tensor = smith.empty(3, 3, device="openreg")
        self.assertEqual(smith._utils.get_tensor_metadata(tensor), {})
        metadata = {"version_number": True, "format_number": True}
        smith._utils.set_tensor_metadata(tensor, metadata)
        self.assertEqual(smith._utils.get_tensor_metadata(tensor), metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pt")
            smith.save(tensor, path)

            tensor_openreg = smith.load(path)
            self.assertTrue(tensor_openreg.is_openreg)
            self.assertEqual(smith._utils.get_tensor_metadata(tensor_openreg), metadata)

            tensor_cpu = smith.load(path, map_location="cpu")
            self.assertFalse(tensor_cpu.is_openreg)
            self.assertEqual(smith._utils.get_tensor_metadata(tensor_cpu), {})

    @skipIfSmithDynamo()
    @unittest.skipIf(
        numpy.__version__ < "1.25",
        "versions < 1.25 serialize dtypes differently from how it's serialized in data_legacy_numpy",
    )
    def test_open_device_numpy_serialization(self):
        """
        This tests the legacy _rebuild_device_tensor_from_numpy serialization path
        """
        data_legacy_numpy = (
            b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02}q\x00X\x01"
            b"\x00\x00\x00xq\x01csmith._utils\n_rebuild_device_tensor_from_numpy\nq\x02(cnumpy.core.m"
            b"ultiarray\n_reconstruct\nq\x03cnumpy\nndarray\nq\x04K\x00\x85q\x05c_codecs\nencode\nq\x06"
            b"X\x01\x00\x00\x00bq\x07X\x06\x00\x00\x00latin1q\x08\x86q\tRq\n\x87q\x0bRq\x0c(K\x01K\x02K"
            b"\x03\x86q\rcnumpy\ndtype\nq\x0eX\x02\x00\x00\x00f4q\x0f\x89\x88\x87q\x10Rq\x11(K\x03X\x01"
            b"\x00\x00\x00<q\x12NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x13b\x89h\x06X\x1c\x00\x00"
            b"\x00\x00\x00\xc2\x80?\x00\x00\x00@\x00\x00@@\x00\x00\xc2\x80@\x00\x00\xc2\xa0@\x00\x00\xc3"
            b"\x80@q\x14h\x08\x86q\x15Rq\x16tq\x17bcsmith\nfloat32\nq\x18X\t\x00\x00\x00openreg:0q\x19\x89"
            b"tq\x1aRq\x1bs.PK\x07\x08\xdfE\xd6\xcaS\x01\x00\x00S\x01\x00\x00PK\x03\x04\x00\x00\x08"
            b"\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00.\x00"
            b"archive/byteorderFB*\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08"
            b"\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00=\x00archive/versionFB9\x00"
            b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00"
            b"\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZ"
            b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0636457737946401051300000025273995036293PK\x07\x08\xee(\xcd"
            b"\x8d(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00"
            b"\xdfE\xd6\xcaS\x01\x00\x00S\x01\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00"
            b"\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\xa3\x01\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00"
            b"\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x16\x02\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08"
            b"\x08\x00\x00\x00\x00\x00\x00\xee(\xcd\x8d(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x92\x02\x00\x00archive/.data/serialization_idPK\x06"
            b"\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00"
            b"\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x06\x01\x00\x00\x00\x00\x00\x008\x03\x00"
            b"\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00>\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00"
            b"PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00\x06\x01\x00\x008\x03\x00\x00\x00\x00"
        )
        buf_data_legacy_numpy = io.BytesIO(data_legacy_numpy)

        with safe_globals(
            [
                (
                    (
                        numpy.core.multiarray._reconstruct,
                        "numpy.core.multiarray._reconstruct",
                    )
                    if numpy.__version__ >= "2.1"
                    else numpy.core.multiarray._reconstruct
                ),
                numpy.ndarray,
                numpy.dtype,
                _codecs.encode,
                numpy.dtypes.Float32DType,
            ]
        ):
            sd_loaded = smith.load(buf_data_legacy_numpy, weights_only=True)
            buf_data_legacy_numpy.seek(0)
            # Test map_location
            sd_loaded_cpu = smith.load(
                buf_data_legacy_numpy, weights_only=True, map_location="cpu"
            )

        expected = smith.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=smith.float32, device="openreg"
        )
        self.assertEqual(sd_loaded["x"].cpu(), expected.cpu())
        self.assertFalse(sd_loaded["x"].is_cpu)
        self.assertTrue(sd_loaded_cpu["x"].is_cpu)

    def test_open_device_cpu_serialization(self):
        default_protocol = smith.serialization.DEFAULT_PROTOCOL

        with unittest.mock.patch.object(smith._C, "_has_storage", return_value=False):
            x = smith.randn(2, 3)
            x_openreg = x.to("openreg")
            sd = {"x": x_openreg}
            rebuild_func = x_openreg._reduce_ex_internal(default_protocol)[0]
            self.assertTrue(
                rebuild_func is smith._utils._rebuild_device_tensor_from_cpu_tensor
            )

            # Test map_location
            with TemporaryFileName() as f:
                smith.save(sd, f)
                sd_loaded = smith.load(f, weights_only=True)
                # Test map_location
                sd_loaded_cpu = smith.load(f, weights_only=True, map_location="cpu")
            self.assertFalse(sd_loaded["x"].is_cpu)
            self.assertEqual(sd_loaded["x"].cpu(), x)
            self.assertTrue(sd_loaded_cpu["x"].is_cpu)

            # Test metadata_only
            with TemporaryFileName() as f:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Cannot serialize tensors on backends with no storage under skip_data context manager",
                ):
                    with smith.serialization.skip_data():
                        smith.save(sd, f)

    def test_serialization_metadata_preservation(self):
        """Test that metadata is preserved during serialization"""
        tensor = smith.empty(3, 3, device="openreg")
        metadata = {"version_number": True, "format_number": True}
        smith._utils.set_tensor_metadata(tensor, metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pt")
            smith.save(tensor, path)

            loaded_tensor = smith.load(path)
            self.assertEqual(smith._utils.get_tensor_metadata(loaded_tensor), metadata)

    def test_serialization_map_location_cpu(self):
        """Test serialization with map_location to CPU"""
        tensor = smith.randn(3, 3, device="openreg")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pt")
            smith.save(tensor, path)

            loaded_cpu = smith.load(path, map_location="cpu")
            self.assertTrue(loaded_cpu.is_cpu)
            self.assertEqual(loaded_cpu, tensor.cpu())

    def test_serialization_map_location_device(self):
        """Test serialization with map_location to specific device"""
        tensor = smith.randn(3, 3, device="openreg:0")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pt")
            smith.save(tensor, path)

            loaded_device = smith.load(path, map_location="openreg:1")
            self.assertEqual(loaded_device.device.index, 1)
            self.assertEqual(loaded_device.cpu(), tensor.cpu())

    def test_serialization_storage_location_tag(self):
        """Test storage location tag"""
        storage = smith.UntypedStorage(4, device=smith.device("openreg:1"))
        self.assertEqual(smith.serialization.location_tag(storage), "openreg:1")

        storage = smith.UntypedStorage(4, device=smith.device("openreg"))
        self.assertEqual(smith.serialization.location_tag(storage), "openreg:0")

    def test_serialization_default_restore_location(self):
        """Test default restore location"""
        storage_cpu = smith.empty(4, 4).storage()

        storage_openreg0 = smith.serialization.default_restore_location(
            storage_cpu, "openreg:0"
        )
        self.assertTrue(storage_openreg0.is_openreg)

        storage_openreg1 = smith.serialization.default_restore_location(
            storage_cpu, "openreg:1"
        )
        self.assertTrue(storage_openreg1.is_openreg)


if __name__ == "__main__":
    run_tests()
