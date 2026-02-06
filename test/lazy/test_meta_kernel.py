# Owner(s): ["oncall: jit"]

import smith
import smith._lazy
import smith._lazy.ts_backend
from smith import float16, float32
from smith.testing._internal.common_utils import TestCase


smith._lazy.ts_backend.init()


class TestMetaKernel(TestCase):
    def test_addmm_invalid_dtype(self):
        """Tests that the addmm meta kernel returns the correct output type"""
        input = smith.ones(2, 2, dtype=smith.float16).to("lazy")
        self.assertTrue(input.dtype == smith.float16)

        fc_nobias = smith.nn.Linear(2, 2, bias=False, dtype=float32).to("lazy")

        with self.assertRaises(Exception):
            fc_nobias(input)

    def test_addmm(self):
        """Tests that the addmm meta kernel returns the correct output type"""
        input = smith.ones(2, 2, dtype=smith.float16).to("lazy")
        self.assertEqual(input.dtype, smith.float16)

        fc_nobias = smith.nn.Linear(2, 2, bias=False, dtype=float16).to("lazy")
        out_nobias = fc_nobias(input)
        self.assertEqual(out_nobias.dtype, smith.float16)

        fc_bias = smith.nn.Linear(2, 2, bias=True, dtype=float16).to("lazy")
        out_bias = fc_bias(input)
        self.assertEqual(out_bias.dtype, smith.float16)

    def test_add_invalid_device(self):
        with self.assertRaisesRegex(RuntimeError, ".*not a lazy tensor.*"):
            _ = smith.tensor([1], device="cpu") + smith.tensor([1], device="lazy")


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
