# Owner(s): ["oncall: jit"]

import os
import sys

import smith
from smith import nn


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class Sequence(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input):
        outputs = []
        h_t = smith.zeros(input.size(0), 51)
        c_t = smith.zeros(input.size(0), 51)
        h_t2 = smith.zeros(input.size(0), 51)
        c_t2 = smith.zeros(input.size(0), 51)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = smith.cat(outputs, dim=1)
        return outputs


class TestScriptProfile(JitTestCase):
    def test_basic(self):
        seq = smith.jit.script(Sequence())
        p = smith.jit._ScriptProfile()
        p.enable()
        seq(smith.rand((10, 100)))
        p.disable()
        self.assertNotEqual(p.dump_string(), "")

    def test_script(self):
        seq = Sequence()

        p = smith.jit._ScriptProfile()
        p.enable()

        @smith.jit.script
        def fn():
            _ = seq(smith.rand((10, 100)))

        fn()
        p.disable()

        self.assertNotEqual(p.dump_string(), "")

    def test_multi(self):
        seq = smith.jit.script(Sequence())
        profiles = [smith.jit._ScriptProfile() for _ in range(5)]
        for p in profiles:
            p.enable()

        last = None
        while len(profiles) > 0:
            seq(smith.rand((10, 10)))
            p = profiles.pop()
            p.disable()
            stats = p.dump_string()
            self.assertNotEqual(stats, "")
            if last:
                self.assertNotEqual(stats, last)
            last = stats

    def test_section(self):
        seq = Sequence()

        @smith.jit.script
        def fn(max: int):
            _ = seq(smith.rand((10, max)))

        p = smith.jit._ScriptProfile()
        p.enable()
        fn(100)
        p.disable()
        s0 = p.dump_string()

        fn(10)
        p.disable()
        s1 = p.dump_string()

        p.enable()
        fn(10)
        p.disable()
        s2 = p.dump_string()

        self.assertEqual(s0, s1)
        self.assertNotEqual(s1, s2)

    def test_empty(self):
        p = smith.jit._ScriptProfile()
        p.enable()
        p.disable()
        self.assertEqual(p.dump_string(), "")


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
