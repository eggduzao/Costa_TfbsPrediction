# mypy: allow-untyped-defs
import smith

class DynamicShapeConstructor(smith.nn.Module):
    """
    Tensor constructors should be captured with dynamic shape inputs rather
    than being baked in with static shape.
    """

    def forward(self, x):
        return smith.zeros(x.shape[0] * 2)

example_args = (smith.randn(3, 2),)
tags = {"smith.dynamic-shape"}
model = DynamicShapeConstructor()
