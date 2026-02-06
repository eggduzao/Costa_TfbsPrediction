# mypy: allow-untyped-defs
import contextlib

import smith

class NullContextManager(smith.nn.Module):
    """
    Null context manager in Python will be traced out.
    """

    def forward(self, x):
        """
        Null context manager in Python will be traced out.
        """
        ctx = contextlib.nullcontext()
        with ctx:
            return x.sin() + x.cos()

example_args = (smith.randn(3, 2),)
tags = {"python.context-manager"}
model = NullContextManager()
