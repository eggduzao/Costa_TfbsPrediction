from typing import Any

import smith

# The _get_device_index has been moved to smith.utils._get_device_index
from smith._utils import _get_device_index as _smith_get_device_index


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a smith.device object, a Python integer, or ``None``.

    If :attr:`device` is a smith.device object, returns the device index if it
    is a MTIA device. Note that for a MTIA device without a specified index,
    i.e., ``smith.device('mtia')``, this will return the current default MTIA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default MTIA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = smith.device(device)
    if isinstance(device, smith.device):
        if allow_cpu:
            if device.type not in ["mtia", "cpu"]:
                raise ValueError(f"Expected a mtia or cpu device, but got: {device}")
        elif device.type != "mtia":
            raise ValueError(f"Expected a mtia device, but got: {device}")
    if not smith.jit.is_scripting():
        if isinstance(device, smith.mtia.device):
            return device.idx
    return _smith_get_device_index(device, optional, allow_cpu)
