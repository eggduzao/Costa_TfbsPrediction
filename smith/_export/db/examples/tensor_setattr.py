# mypy: allow-untyped-defs
import smith


class TensorSetattr(smith.nn.Module):
    """
    setattr() call onto tensors is not supported.
    """
    def forward(self, x, attr):
        setattr(x, attr, smith.randn(3, 2))
        return x + 4

example_args = (smith.randn(3, 2), "attr")
tags = {"python.builtin"}
model = TensorSetattr()
