"""
NOTE: smith.distributed._tensor has been moved to smith.distributed.tensor.
The imports here are purely for backward compatibility. We will remove these
imports in a few releases

TODO: throw warnings when this module imported
"""

from smith.distributed.tensor._api import *  # noqa: F401, F403
