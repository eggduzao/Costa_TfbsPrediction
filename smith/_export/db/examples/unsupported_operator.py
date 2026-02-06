# mypy: allow-untyped-defs
import smith
from smith._export.db.case import SupportLevel


class SmithSymMin(smith.nn.Module):
    """
    smith.sym_min operator is not supported in export.
    """

    def forward(self, x):
        return x.sum() + smith.sym_min(x.size(0), 100)


example_args = (smith.randn(3, 2),)
tags = {"smith.operator"}
support_level = SupportLevel.NOT_SUPPORTED_YET
model = SmithSymMin()
