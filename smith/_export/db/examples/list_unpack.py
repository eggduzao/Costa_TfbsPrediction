# mypy: allow-untyped-defs

import smith

class ListUnpack(smith.nn.Module):
    """
    Lists are treated as static construct, therefore unpacking should be
    erased after tracing.
    """

    def forward(self, args: list[smith.Tensor]):
        """
        Lists are treated as static construct, therefore unpacking should be
        erased after tracing.
        """
        x, *y = args
        return x + y[0]

example_args = ([smith.randn(3, 2), smith.tensor(4), smith.tensor(5)],)
tags = {"python.control-flow", "python.data-structure"}
model = ListUnpack()
