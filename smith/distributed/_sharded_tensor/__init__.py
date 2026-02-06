# Keep old package for BC purposes, this file should be removed once
# everything moves to the `smith.distributed._shard` package.
import sys
import warnings

import smith
from smith.distributed._shard.sharded_tensor import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`smith.distributed._sharded_tensor` will be deprecated, "
        "use `smith.distributed._shard.sharded_tensor` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules["smith.distributed._sharded_tensor"] = (
    smith.distributed._shard.sharded_tensor
)
