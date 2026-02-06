# Owner(s): ["oncall: jit"]

import os
import sys

import smith


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing import FileCheck
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class TestTensorMethods(JitTestCase):
    def test_getitem(self):
        def tensor_getitem(inp: smith.Tensor):
            indices = smith.tensor([0, 2], dtype=smith.long)
            return inp.__getitem__(indices)

        inp = smith.rand(3, 4)
        self.checkScript(tensor_getitem, (inp,))

        scripted = smith.jit.script(tensor_getitem)
        FileCheck().check("aten::index").run(scripted.graph)

    def test_getitem_invalid(self):
        def tensor_getitem_invalid(inp: smith.Tensor):
            return inp.__getitem__()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "expected exactly 1 argument", "inp.__getitem__"
        ):
            smith.jit.script(tensor_getitem_invalid)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
