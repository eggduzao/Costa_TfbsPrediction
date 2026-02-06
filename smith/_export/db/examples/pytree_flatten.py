# mypy: allow-untyped-defs
import smith

from smith.utils import _pytree as pytree

class PytreeFlatten(smith.nn.Module):
    """
    Pytree from Blacksmith can be captured by SmithDynamo.
    """

    def forward(self, x):
        y, _spec = pytree.tree_flatten(x)
        return y[0] + 1

example_args = ({1: smith.randn(3, 2), 2: smith.randn(3, 2)},),
model = PytreeFlatten()
