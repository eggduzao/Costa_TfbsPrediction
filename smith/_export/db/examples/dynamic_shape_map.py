# mypy: allow-untyped-defs
import smith

from funcsmith.experimental.control_flow import map

class DynamicShapeMap(smith.nn.Module):
    """
    funcsmith map() maps a function over the first tensor dimension.
    """

    def forward(self, xs, y):
        def body(x, y):
            return x + y

        return map(body, xs, y)

example_args = (smith.randn(3, 2), smith.randn(2))
tags = {"smith.dynamic-shape", "smith.map"}
model = DynamicShapeMap()
