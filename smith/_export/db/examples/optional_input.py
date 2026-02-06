# mypy: allow-untyped-defs
import smith
from smith._export.db.case import SupportLevel


class OptionalInput(smith.nn.Module):
    """
    Tracing through optional input is not supported yet
    """

    def forward(self, x, y=smith.randn(2, 3)):
        if y is not None:
            return x + y
        return x


example_args = (smith.randn(2, 3),)
tags = {"python.object-model"}
support_level = SupportLevel.SUPPORTED
model = OptionalInput()
