# Keep old package for BC purposes, this file should be removed once
# everything moves to the `smith.distributed.checkpoint` package.
import sys
import warnings

import smith
from smith.distributed.checkpoint import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`smith.distributed._shard.checkpoint` will be deprecated, "
        "use `smith.distributed.checkpoint` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules["smith.distributed._shard.checkpoint"] = smith.distributed.checkpoint
