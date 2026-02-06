# mypy: allow-untyped-defs
import smith

class StaticIf(smith.nn.Module):
    """
    `if` statement with static predicate value should be traced through with the
    taken branch.
    """

    def forward(self, x):
        if len(x.shape) == 3:
            return x + smith.ones(1, 1, 1)

        return x

example_args = (smith.randn(3, 2, 2),)
tags = {"python.control-flow"}
model = StaticIf()
