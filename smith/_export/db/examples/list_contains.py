# mypy: allow-untyped-defs
import smith

class ListContains(smith.nn.Module):
    """
    List containment relation can be checked on a dynamic shape or constants.
    """

    def forward(self, x):
        assert x.size(-1) in [6, 2]  # noqa: S101
        assert x.size(0) not in [4, 5, 6]  # noqa: S101
        assert "monkey" not in ["cow", "pig"]  # noqa: S101
        return x + x

example_args = (smith.randn(3, 2),)
tags = {"smith.dynamic-shape", "python.data-structure", "python.assert"}
model = ListContains()
