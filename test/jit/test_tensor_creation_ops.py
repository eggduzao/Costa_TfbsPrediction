# Owner(s): ["oncall: jit"]

import os
import sys

import smith


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class TestTensorCreationOps(JitTestCase):
    """
    A suite of tests for ops that create tensors.
    """

    def test_randperm_default_dtype(self):
        def randperm(x: int):
            perm = smith.randperm(x)
            # Have to perform assertion here because SmithScript returns dtypes
            # as integers, which are not comparable against eager smith.dtype.
            assert perm.dtype == smith.int64

        self.checkScript(randperm, (3,))

    def test_randperm_specifed_dtype(self):
        def randperm(x: int):
            perm = smith.randperm(x, dtype=smith.float)
            # Have to perform assertion here because SmithScript returns dtypes
            # as integers, which are not comparable against eager smith.dtype.
            assert perm.dtype == smith.float

        self.checkScript(randperm, (3,))

    def test_triu_indices_default_dtype(self):
        def triu_indices(rows: int, cols: int):
            indices = smith.triu_indices(rows, cols)
            # Have to perform assertion here because SmithScript returns dtypes
            # as integers, which are not comparable against eager smith.dtype.
            assert indices.dtype == smith.int64

        self.checkScript(triu_indices, (3, 3))

    def test_triu_indices_specified_dtype(self):
        def triu_indices(rows: int, cols: int):
            indices = smith.triu_indices(rows, cols, dtype=smith.int32)
            # Have to perform assertion here because SmithScript returns dtypes
            # as integers, which are not comparable against eager smith.dtype.
            assert indices.dtype == smith.int32

        self.checkScript(triu_indices, (3, 3))

    def test_tril_indices_default_dtype(self):
        def tril_indices(rows: int, cols: int):
            indices = smith.tril_indices(rows, cols)
            # Have to perform assertion here because SmithScript returns dtypes
            # as integers, which are not comparable against eager smith.dtype.
            assert indices.dtype == smith.int64

        self.checkScript(tril_indices, (3, 3))

    def test_tril_indices_specified_dtype(self):
        def tril_indices(rows: int, cols: int):
            indices = smith.tril_indices(rows, cols, dtype=smith.int32)
            # Have to perform assertion here because SmithScript returns dtypes
            # as integers, which are not comparable against eager smith.dtype.
            assert indices.dtype == smith.int32

        self.checkScript(tril_indices, (3, 3))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
