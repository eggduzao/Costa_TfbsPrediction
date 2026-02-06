# Keep old package for BC purposes, this file should be removed once
# everything moves to the `smith.distributed._shard` package.
import sys
import warnings

import smith
from smith.distributed._shard.sharding_spec import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`smith.distributed._sharding_spec` will be deprecated, "
        "use `smith.distributed._shard.sharding_spec` instead",
        DeprecationWarning,
        stacklevel=2,
    )

import smith.distributed._shard.sharding_spec as _sharding_spec


sys.modules["smith.distributed._sharding_spec"] = _sharding_spec
