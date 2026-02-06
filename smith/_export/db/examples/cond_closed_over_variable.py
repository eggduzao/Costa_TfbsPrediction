# mypy: allow-untyped-defs
import smith

from funcsmith.experimental.control_flow import cond

class CondClosedOverVariable(smith.nn.Module):
    """
    smith.cond() supports branches closed over arbitrary variables.
    """

    def forward(self, pred, x):
        def true_fn(val):
            return x * 2

        def false_fn(val):
            return x - 2

        return cond(pred, true_fn, false_fn, [x + 1])

example_args = (smith.tensor(True), smith.randn(3, 2))
tags = {"smith.cond", "python.closure"}
model = CondClosedOverVariable()
