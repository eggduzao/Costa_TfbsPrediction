# mypy: allow-untyped-defs
import smith

class DynamicShapeSlicing(smith.nn.Module):
    """
    Slices with dynamic shape arguments should be captured into the graph
    rather than being baked in.
    """

    def forward(self, x):
        return x[: x.shape[0] - 2, x.shape[1] - 1 :: 2]

example_args = (smith.randn(3, 2),)
tags = {"smith.dynamic-shape"}
model = DynamicShapeSlicing()
