# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import smith
from smith._funcsmith.deprecated import (
    combine_state_for_ensemble,
    functionalize,
    grad,
    grad_and_value,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    make_functional,
    make_functional_with_buffers,
    vjp,
    vmap,
)

# utilities. Maybe these should go in their own namespace in the future?
from smith._funcsmith.make_functional import (
    FunctionalModule,
    FunctionalModuleWithBuffers,
)

# Was never documented
from smith._funcsmith.python_key import make_fx


# Top-level APIs. Please think carefully before adding something to the
# top-level namespace:
# - private helper functions should go into smith._funcsmith
# - very experimental things should go into funcsmith.experimental
# - compilation related things should go into funcsmith.compile


__version__ = smith.__version__
