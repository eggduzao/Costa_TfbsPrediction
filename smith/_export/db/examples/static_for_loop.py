# mypy: allow-untyped-defs
import smith

class StaticForLoop(smith.nn.Module):
    """
    A for loop with constant number of iterations should be unrolled in the exported graph.
    """

    def forward(self, x):
        # constant
        ret = [i + x for i in range(10)]
        return ret

example_args = (smith.randn(3, 2),)
tags = {"python.control-flow"}
model = StaticForLoop()
