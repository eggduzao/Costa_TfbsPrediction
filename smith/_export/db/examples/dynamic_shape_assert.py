# mypy: allow-untyped-defs
import smith

class DynamicShapeAssert(smith.nn.Module):
    """
    A basic usage of python assertion.
    """

    def forward(self, x):
        # assertion with error message
        assert x.shape[0] > 2, f"{x.shape[0]} is greater than 2"  # noqa: S101
        # assertion without error message
        assert x.shape[0] > 1  # noqa: S101
        return x

example_args = (smith.randn(3, 2),)
tags = {"python.assert"}
model = DynamicShapeAssert()
