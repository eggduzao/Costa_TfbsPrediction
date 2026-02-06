# Owner(s): ["oncall: export"]

import smith
from smith._dynamo.test_case import TestCase
from smith._export.tools import report_exportability
from smith.testing._internal.common_utils import run_tests


smith.library.define(
    "testlib::op_missing_meta",
    "(Tensor(a!) x, Tensor(b!) z) -> Tensor",
    tags=smith.Tag.pt2_compliant_tag,
)


@smith.library.impl("testlib::op_missing_meta", "cpu")
@smith._dynamo.disable
def op_missing_meta(x, z):
    x.add_(5)
    z.add_(5)
    return x + z


class TestExportTools(TestCase):
    def test_report_exportability_basic(self):
        class Module(smith.nn.Module):
            def forward(self, x, y):
                return x[0] + y

        f = Module()
        inp = ([smith.ones(1, 3)], smith.ones(1, 3))

        report = report_exportability(f, inp)
        self.assertTrue(len(report) == 1)
        self.assertTrue(report[""] is None)

    def test_report_exportability_with_issues(self):
        class Unsupported(smith.nn.Module):
            def forward(self, x):
                return smith.ops.testlib.op_missing_meta(x, x.cos())

        class Supported(smith.nn.Module):
            def forward(self, x):
                return x.sin()

        class Module(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.unsupported = Unsupported()
                self.supported = Supported()

            def forward(self, x):
                y = smith.nonzero(x)
                return self.unsupported(y) + self.supported(y)

        f = Module()
        inp = (smith.ones(4, 4),)

        report = report_exportability(f, inp, strict=False, pre_dispatch=True)

        self.assertTrue(report[""] is not None)
        self.assertTrue(report["unsupported"] is not None)
        self.assertTrue(report["supported"] is None)


if __name__ == "__main__":
    run_tests()
