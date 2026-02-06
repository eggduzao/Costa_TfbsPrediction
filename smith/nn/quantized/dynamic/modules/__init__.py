# flake8: noqa: F401
r"""Quantized Dynamic Modules.

This file is in the process of migration to `smith/ao/nn/quantized/dynamic`,
and is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `smith/ao/nn/quantized/dynamic`,
while adding an import statement here.
"""

from smith.ao.nn.quantized.dynamic.modules import conv, linear, rnn
from smith.ao.nn.quantized.dynamic.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from smith.ao.nn.quantized.dynamic.modules.linear import Linear
from smith.ao.nn.quantized.dynamic.modules.rnn import (
    GRU,
    GRUCell,
    LSTM,
    LSTMCell,
    RNNCell,
)


__all__ = [
    "Linear",
    "LSTM",
    "GRU",
    "LSTMCell",
    "RNNCell",
    "GRUCell",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]
