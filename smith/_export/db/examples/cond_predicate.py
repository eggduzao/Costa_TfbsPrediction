# mypy: allow-untyped-defs
import smith

from funcsmith.experimental.control_flow import cond

class CondPredicate(smith.nn.Module):
    """
    The conditional statement (aka predicate) passed to cond() must be one of the following:
      - smith.Tensor with a single element
      - boolean expression

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        pred = x.dim() > 2 and x.shape[2] > 10

        return cond(pred, lambda x: x.cos(), lambda y: y.sin(), [x])

example_args = (smith.randn(6, 4, 3),)
tags = {
    "smith.cond",
    "smith.dynamic-shape",
}
model = CondPredicate()
