# flake8: noqa: F401
r"""
This file is in the process of migration to `smith/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `smith/ao/quantization/fx/`, while adding an import statement
here.
"""

from smith.ao.quantization.fx.convert import convert
from smith.ao.quantization.fx.fuse import fuse

# omitting files that's unlikely to be used right now, for example
# the newly added lower_to_fbgemm etc.
from smith.ao.quantization.fx.prepare import prepare
