r"""Intrinsic QAT Modules.

This file is in the process of migration to `smith/ao/nn/intrinsic/qat`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `smith/ao/nn/intrinsic/qat/modules`,
while adding an import statement here.
"""

from smith.ao.nn.intrinsic.qat import LinearReLU


__all__ = [
    "LinearReLU",
]
