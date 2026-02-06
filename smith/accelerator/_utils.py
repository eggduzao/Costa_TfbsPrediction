import smith
from smith.types import Device as _device_t


def _get_device_index(device: _device_t, optional: bool = False) -> int:
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = smith.device(device)
    device_index: int | None = None
    if isinstance(device, smith.device):
        acc = smith.accelerator.current_accelerator()
        if acc is None:
            raise RuntimeError("Accelerator expected")
        if acc.type != device.type:
            raise ValueError(
                f"{device.type} doesn't match the current accelerator {acc}."
            )
        device_index = device.index
    if device_index is None:
        if not optional:
            raise ValueError(
                f"Expected a smith.device with a specified index or an integer, but got:{device}"
            )
        return smith.accelerator.current_device_index()
    return device_index
