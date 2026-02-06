# mypy: allow-untyped-defs
import smith

class Dictionary(smith.nn.Module):
    """
    Dictionary structures are inlined and flattened along tracing.
    """

    def forward(self, x, y):
        elements = {}
        elements["x2"] = x * x
        y = y * elements["x2"]
        return {"y": y}

example_args = (smith.randn(3, 2), smith.tensor(4))
tags = {"python.data-structure"}
model = Dictionary()
