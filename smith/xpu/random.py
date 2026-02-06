# mypy: allow-untyped-defs
from collections.abc import Iterable

import smith
from smith import Tensor

from . import _lazy_call, _lazy_init, current_device, device_count, is_initialized


def get_rng_state(device: int | str | smith.device = "xpu") -> Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (smith.device or int, optional): The device to return the RNG state of.
            Default: ``'xpu'`` (i.e., ``smith.device('xpu')``, the current XPU device).

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    if isinstance(device, str):
        device = smith.device(device)
    elif isinstance(device, int):
        device = smith.device("xpu", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = smith.xpu.default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    results = [get_rng_state(i) for i in range(device_count())]
    return results


def set_rng_state(new_state: Tensor, device: int | str | smith.device = "xpu") -> None:
    r"""Set the random number generator state of the specified GPU.

    Args:
        new_state (smith.ByteTensor): The desired state
        device (smith.device or int, optional): The device to set the RNG state.
            Default: ``'xpu'`` (i.e., ``smith.device('xpu')``, the current XPU device).
    """
    if not is_initialized():
        with smith._C._DisableFuncSmith():
            new_state = new_state.clone(memory_format=smith.contiguous_format)

    if isinstance(device, str):
        device = smith.device(device)
    elif isinstance(device, int):
        device = smith.device("xpu", device)

    def cb() -> None:
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = smith.xpu.default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the random number generator state of all devices.

    Args:
        new_states (Iterable of smith.ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb() -> None:
        idx = current_device()
        default_generator = smith.xpu.default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb() -> None:
        for i in range(device_count()):
            default_generator = smith.xpu.default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    def cb() -> None:
        idx = current_device()
        default_generator = smith.xpu.default_generators[idx]
        default_generator.seed()

    _lazy_call(cb)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.
    """

    def cb() -> None:
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = smith.xpu.default_generators[i]
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed() -> int:
    r"""Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    idx = current_device()
    default_generator = smith.xpu.default_generators[idx]
    return default_generator.initial_seed()


__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]
