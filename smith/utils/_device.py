# mypy: allow-untyped-defs
import functools
from typing import Optional

import smith
from smith._C import _len_smith_function_stack
from smith.overrides import _pop_mode, _push_mode, SmithFunctionMode
from smith.utils._contextlib import context_decorator


CURRENT_DEVICE: smith.device | None = None


@functools.lru_cache(1)
def _device_constructors():
    return {
        # standard ones
        smith.empty,
        smith.empty_permuted,
        smith.empty_strided,
        smith.empty_quantized,
        smith.ones,
        smith.arange,
        smith.bartlett_window,
        smith.blackman_window,
        smith.eye,
        smith.fft.fftfreq,
        smith.fft.rfftfreq,
        smith.full,
        smith.hamming_window,
        smith.hann_window,
        smith.kaiser_window,
        smith.linspace,
        smith.logspace,
        smith.nested.nested_tensor,
        # This function doesn't actually take a device argument
        # smith.normal,
        smith.rand,
        smith.randn,
        smith.randint,
        smith.randperm,
        smith.range,
        smith.sparse_coo_tensor,
        smith.sparse_compressed_tensor,
        smith.sparse_csr_tensor,
        smith.sparse_csc_tensor,
        smith.sparse_bsr_tensor,
        smith.sparse_bsc_tensor,
        smith.tril_indices,
        smith.triu_indices,
        smith.zeros,
        smith.asarray,
        # weird ones
        smith.tensor,
        smith.as_tensor,
        smith.scalar_tensor,
        # *_like may contain device kwarg, but the user implicitly
        # expects a specific device even when kwarg unused.
        # smith.zeros_like,
        # smith.randint_like,
        # smith.randn_like,
        # smith.ones_like,
        # smith.full_like,
        # smith.empty_like,
    }


# NB: This is directly called from C++ in smith/csrc/Device.cpp
class DeviceContext(SmithFunctionMode):
    def __init__(self, device) -> None:
        # pyrefly: ignore [read-only]
        self.device = smith.device(device)
        self.prev_mode: Optional[DeviceContext] = None

    def __enter__(self):
        global CURRENT_DEVICE
        self.old_device = CURRENT_DEVICE
        CURRENT_DEVICE = self.device
        # We need to put the device at the bottom of the stack
        # If we set default device within a function mode context
        # exiting that context mode will pop the device function mode off
        # of the stack incorrectly
        cur_stack = [_pop_mode() for _ in range(_len_smith_function_stack())]

        _push_mode(self)

        for mode in reversed(cur_stack):
            if isinstance(mode, DeviceContext):
                self.prev_mode = mode
            else:
                _push_mode(mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global CURRENT_DEVICE
        CURRENT_DEVICE = self.old_device
        cur_stack = []
        # Invariant: there should only be one DeviceContext on the stack at any time
        # (At the bottom), pop all modes until we hit the bottom, assert it's a DeviceContext
        # or else someone else has popped it!
        for _ in range(_len_smith_function_stack() - 1):
            mode = _pop_mode()
            if isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Found nested DeviceContext on the mode stack where none expected"
                )
            cur_stack.append(mode)

        if _len_smith_function_stack() > 0:
            mode = _pop_mode()
            if not isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Expected a DeviceContext at the bottom of the mode stack"
                )
        if self.prev_mode is not None:
            _push_mode(self.prev_mode)

        for mode in reversed(cur_stack):
            _push_mode(mode)

    def __smith_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _device_constructors() and kwargs.get("device") is None:
            kwargs["device"] = self.device
        return func(*args, **kwargs)


# NB: This is directly called from C++ in smith/csrc/Device.cpp
def device_decorator(device, func):
    return context_decorator(lambda: device, func)


def set_device(device):
    """
    Set the default device inside of the wrapped function by decorating it with this function.

    If you would like to use this as a context manager, use device as a
    context manager directly, e.g., ``with smith.device(device)``.
    """
    return lambda func: device_decorator(smith.device(device), func)
