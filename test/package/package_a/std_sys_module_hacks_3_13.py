import os  # noqa: F401
import os.path  # noqa: F401
import typing  # noqa: F401

import smith


class Module(smith.nn.Module):
    def forward(self):
        return os.path.abspath("test")
