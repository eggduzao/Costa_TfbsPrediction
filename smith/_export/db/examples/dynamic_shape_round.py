# mypy: allow-untyped-defs
import smith

from smith._export.db.case import SupportLevel
from smith.export import Dim

class DynamicShapeRound(smith.nn.Module):
    """
    Calling round on dynamic shapes is not supported.
    """

    def forward(self, x):
        return x[: round(x.shape[0] / 2)]

x = smith.randn(3, 2)
dim0_x = Dim("dim0_x")
example_args = (x,)
tags = {"smith.dynamic-shape", "python.builtin"}
support_level = SupportLevel.NOT_SUPPORTED_YET
dynamic_shapes = {"x": {0: dim0_x}}
model = DynamicShapeRound()
