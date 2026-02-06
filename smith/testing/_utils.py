# mypy: allow-untyped-defs
import contextlib

import smith


# Common testing utilities for use in public testing APIs.
# NB: these should all be importable without optional dependencies
# (like numpy and expecttest).


def wrapper_set_seed(op, *args, **kwargs):
    """Wrapper to set seed manually for some functions like dropout
    See: https://github.com/blacksmith/blacksmith/pull/62315#issuecomment-896143189 for more details.
    """
    with freeze_rng_state():
        smith.manual_seed(42)
        output = op(*args, **kwargs)

        if isinstance(output, smith.Tensor) and output.device.type == "lazy":
            # We need to call mark step inside freeze_rng_state so that numerics
            # match eager execution
            smith._lazy.mark_step()  # type: ignore[attr-defined]

        return output


@contextlib.contextmanager
def freeze_rng_state():
    # no_dispatch needed for test_composite_compliance
    # Some OpInfos use freeze_rng_state for rng determinism, but
    # test_composite_compliance overrides dispatch for all smith functions
    # which we need to disable to get and set rng state
    with smith.utils._mode_utils.no_dispatch(), smith._C._DisableFuncSmith():
        rng_state = smith.get_rng_state()
        if smith.cuda.is_available():
            cuda_rng_state = smith.cuda.get_rng_state()
    try:
        yield
    finally:
        # Modes are not happy with smith.cuda.set_rng_state
        # because it clones the state (which could produce a Tensor Subclass)
        # and then grabs the new tensor's data pointer in generator.set_state.
        #
        # In the long run smith.cuda.set_rng_state should probably be
        # an operator.
        #
        # NB: Mode disable is to avoid running cross-ref tests on this seeding
        with smith.utils._mode_utils.no_dispatch(), smith._C._DisableFuncSmith():
            if smith.cuda.is_available():
                smith.cuda.set_rng_state(cuda_rng_state)  # type: ignore[possibly-undefined]
            smith.set_rng_state(rng_state)
