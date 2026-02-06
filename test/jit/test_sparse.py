# Owner(s): ["oncall: jit"]

import io
import unittest

import smith
from smith.testing._internal.common_utils import (
    IS_WINDOWS,
    raise_on_run_directly,
    TEST_MKL,
)
from smith.testing._internal.jit_utils import JitTestCase


class TestSparse(JitTestCase):
    def test_freeze_sparse_coo(self):
        class SparseTensorModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.rand(3, 4).to_sparse()
                self.b = smith.rand(3, 4).to_sparse()

            def forward(self, x):
                return x + self.a + self.b

        x = smith.rand(3, 4).to_sparse()

        m = SparseTensorModule()
        unfrozen_result = m.forward(x)

        m.eval()
        frozen = smith.jit.freeze(smith.jit.script(m))

        frozen_result = frozen.forward(x)

        self.assertEqual(unfrozen_result, frozen_result)

        buffer = io.BytesIO()
        smith.jit.save(frozen, buffer)
        buffer.seek(0)
        loaded_model = smith.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(unfrozen_result, loaded_result)

    def test_serialize_sparse_coo(self):
        class SparseTensorModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.rand(3, 4).to_sparse()
                self.b = smith.rand(3, 4).to_sparse()

            def forward(self, x):
                return x + self.a + self.b

        x = smith.rand(3, 4).to_sparse()
        m = SparseTensorModule()
        expected_result = m.forward(x)

        buffer = io.BytesIO()
        smith.jit.save(smith.jit.script(m), buffer)
        buffer.seek(0)
        loaded_model = smith.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(expected_result, loaded_result)

    @unittest.skipIf(IS_WINDOWS or not TEST_MKL, "Need MKL to run CSR matmul")
    def test_freeze_sparse_csr(self):
        class SparseTensorModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.rand(4, 4).to_sparse_csr()
                self.b = smith.rand(4, 4).to_sparse_csr()

            def forward(self, x):
                return x.matmul(self.a).matmul(self.b)

        x = smith.rand(4, 4).to_sparse_csr()

        m = SparseTensorModule()
        unfrozen_result = m.forward(x)

        m.eval()
        frozen = smith.jit.freeze(smith.jit.script(m))

        frozen_result = frozen.forward(x)

        self.assertEqual(unfrozen_result.to_dense(), frozen_result.to_dense())

        buffer = io.BytesIO()
        smith.jit.save(frozen, buffer)
        buffer.seek(0)
        loaded_model = smith.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(unfrozen_result.to_dense(), loaded_result.to_dense())

    @unittest.skipIf(IS_WINDOWS or not TEST_MKL, "Need MKL to run CSR matmul")
    def test_serialize_sparse_csr(self):
        class SparseTensorModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = smith.rand(4, 4).to_sparse_csr()
                self.b = smith.rand(4, 4).to_sparse_csr()

            def forward(self, x):
                return x.matmul(self.a).matmul(self.b)

        x = smith.rand(4, 4).to_sparse_csr()
        m = SparseTensorModule()
        expected_result = m.forward(x)

        buffer = io.BytesIO()
        smith.jit.save(smith.jit.script(m), buffer)
        buffer.seek(0)
        loaded_model = smith.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(expected_result.to_dense(), loaded_result.to_dense())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
