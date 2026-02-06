# flake8: noqa: F401
r"""QAT Modules.

This package is in the process of being deprecated.
Please, use `smith.ao.nn.qat.modules` instead.
"""

from smith.ao.nn.qat.modules.conv import Conv1d, Conv2d, Conv3d
from smith.ao.nn.qat.modules.embedding_ops import Embedding, EmbeddingBag
from smith.ao.nn.qat.modules.linear import Linear
from smith.nn.qat.modules import conv, embedding_ops, linear


__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EmbeddingBag",
]
