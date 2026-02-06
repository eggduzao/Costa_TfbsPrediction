# flake8: noqa: F401
r"""QAT Dynamic Modules.

This package is in the process of being deprecated.
Please, use `smith.ao.nn.qat.dynamic` instead.
"""

from smith.nn.qat import dynamic, modules  # noqa: F403
from smith.nn.qat.modules import *  # noqa: F403


__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EmbeddingBag",
]
