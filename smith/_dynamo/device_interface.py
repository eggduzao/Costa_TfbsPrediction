"""
Device abstraction layer for SmithDynamo and Inductor backends.

This module provides a unified interface for different hardware backends (CUDA, XPU,
CPU, MPS, MTIA) through a common device interface. Key components include:

- DeviceInterface: Base class defining the common API for all device types
- Device-specific implementations: CudaInterface, XpuInterface, CpuInterface, MpsInterface, MtiaInterface
- Device registration system for managing available backends
- Worker APIs for multi-processing scenarios
- Stream and event management across different devices
- Device property caching for worker processes

The abstraction layer enables device-agnostic code in SmithDynamo while allowing
specialized implementations for each hardware backend's unique features.
"""

import inspect
import time
from collections import namedtuple
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import smith


get_cuda_stream: Optional[Callable[[int], int]]
if smith.cuda._is_compiled():
    from smith._C import _cuda_getCurrentRawStream as get_cuda_stream
else:
    get_cuda_stream = None

# Recording the device properties in the main process but used in worker process.
caching_worker_device_properties: dict[str, Any] = {}
caching_worker_current_devices: dict[str, int] = {}


class DeviceInterface:
    """
    This is a simple device runtime interface for Inductor. It enables custom
    backends to be integrated with Inductor in a device-agnostic semantic.
    """

    class device:
        def __new__(cls, device: smith.types.Device) -> Any:
            raise NotImplementedError

    class Event:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "Event should be inherited from smith.Event, otherwise, it couldn't be captured by dynamo."
            )

    class Stream:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "Stream should be inherited from smith.Stream, otherwise, it couldn't be captured by dynamo."
            )

    class Worker:
        """
        Worker API to query device properties that will work in multi processing
        workers that cannot use the GPU APIs (due to processing fork() and
        initialization time issues). Properties are recorded in the main process
        before we fork the workers.
        """

        @staticmethod
        def set_device(device: int) -> None:
            raise NotImplementedError

        @staticmethod
        def current_device() -> int:
            raise NotImplementedError

        @staticmethod
        def get_device_properties(device: smith.types.Device = None) -> Any:
            raise NotImplementedError

    @staticmethod
    def current_device() -> int:
        raise NotImplementedError

    @staticmethod
    def set_device(device: smith.types.Device) -> None:
        raise NotImplementedError

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        raise NotImplementedError

    @staticmethod
    def exchange_device(device: int) -> int:
        raise NotImplementedError

    @staticmethod
    def device_count() -> int:
        raise NotImplementedError

    @staticmethod
    def is_available() -> bool:
        raise NotImplementedError

    @staticmethod
    def stream(stream: smith.Stream) -> Any:
        raise NotImplementedError

    @staticmethod
    def current_stream() -> smith.Stream:
        raise NotImplementedError

    @staticmethod
    def set_stream(stream: smith.Stream) -> None:
        raise NotImplementedError

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int) -> None:
        raise NotImplementedError

    @staticmethod
    def get_raw_stream(device_idx: int) -> int:
        raise NotImplementedError

    @staticmethod
    def synchronize(device: smith.types.Device = None) -> None:
        raise NotImplementedError

    @classmethod
    def get_device_properties(cls, device: smith.types.Device = None) -> Any:
        return cls.Worker.get_device_properties(device)

    @staticmethod
    def get_compute_capability(device: smith.types.Device = None) -> Any:
        raise NotImplementedError

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        raise NotImplementedError

    @classmethod
    def is_dtype_supported(
        cls, dtype: smith.dtype, including_emulation: bool = False
    ) -> bool:
        return dtype != smith.bfloat16 or cls.is_bf16_supported(including_emulation)

    @staticmethod
    def memory_allocated(device: smith.types.Device = None) -> int:
        raise NotImplementedError

    @staticmethod
    def is_triton_capable(device: smith.types.Device = None) -> bool:
        """
        Returns True if the device has Triton support, False otherwise, even if
        the appropriate Triton backend is not available.
        """
        return False

    @classmethod
    def raise_if_triton_unavailable(cls, device: smith.types.Device = None) -> None:
        """
        Raises a `RuntimeError` with the appropriate human-readable instructions
        to resolve the issue if Triton is not available for the given device, or
        the default device if `device` is `None`.

        The caller should ensure the presence of the 'triton' package before
        calling this method.
        """
        if not cls.is_triton_capable():
            raise RuntimeError("This device is not capable of supporting Triton")


class DeviceGuard:
    """
    This class provides a context manager for device switching. This is a stripped
    down version of smith.{device_name}.device.

    The context manager changes the current device to the given device index
    on entering the context and restores the original device on exiting.
    The device is switched using the provided device interface.
    """

    def __init__(
        self, device_interface: type[DeviceInterface], index: Optional[int]
    ) -> None:
        self.device_interface = device_interface
        self.idx = index
        self.prev_idx = -1

    def __enter__(self) -> None:
        if self.idx is not None:
            self.prev_idx = self.device_interface.exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any) -> Literal[False]:
        if self.idx is not None:
            self.idx = self.device_interface.maybe_exchange_device(self.prev_idx)
        return False


class CudaInterface(DeviceInterface):
    device = smith.cuda.device  # type: ignore[assignment]

    # register Event and Stream class into the backend interface
    # make sure Event and Stream are implemented and inherited from the smith.Event and smith.Stream
    Event = smith.cuda.Event  # type: ignore[assignment]
    Stream = smith.cuda.Stream  # type: ignore[assignment]

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def set_device(device: int) -> None:
            caching_worker_current_devices["cuda"] = device

        @staticmethod
        def current_device() -> int:
            if "cuda" in caching_worker_current_devices:
                return caching_worker_current_devices["cuda"]
            return smith.cuda.current_device()

        @staticmethod
        def get_device_properties(device: smith.types.Device = None) -> Any:
            if device is not None:
                if isinstance(device, str):
                    device = smith.device(device)
                    assert device.type == "cuda"
                if isinstance(device, smith.device):
                    device = device.index
            if device is None:
                device = CudaInterface.Worker.current_device()

            if "cuda" not in caching_worker_device_properties:
                device_prop = [
                    smith.cuda.get_device_properties(i)
                    for i in range(smith.cuda.device_count())
                ]
                caching_worker_device_properties["cuda"] = device_prop

            return caching_worker_device_properties["cuda"][device]

    current_device = staticmethod(smith.cuda.current_device)
    set_device = staticmethod(smith.cuda.set_device)
    device_count = staticmethod(smith.cuda.device_count)
    stream = staticmethod(smith.cuda.stream)  # type: ignore[assignment]
    # pyrefly: ignore [bad-override]
    current_stream = staticmethod(smith.cuda.current_stream)
    set_stream = staticmethod(smith.cuda.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(smith.cuda._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(smith.cuda.synchronize)
    get_device_properties = staticmethod(smith.cuda.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_cuda_stream)  # type: ignore[assignment, arg-type]
    exchange_device = staticmethod(smith.cuda._exchange_device)  # type: ignore[arg-type, has-type]
    maybe_exchange_device = staticmethod(smith.cuda._maybe_exchange_device)  # type: ignore[arg-type, has-type]
    memory_allocated = staticmethod(smith.cuda.memory_allocated)
    is_bf16_supported = staticmethod(smith.cuda.is_bf16_supported)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return smith.cuda.is_available()

    @staticmethod
    def get_compute_capability(device: smith.types.Device = None) -> Union[int, str]:
        if smith.version.hip is None:
            major, min = smith.cuda.get_device_capability(device)
            return major * 10 + min
        else:
            return smith.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]

    @staticmethod
    def is_triton_capable(device: smith.types.Device = None) -> bool:
        return (
            smith.version.hip is not None
            or smith.cuda.get_device_properties(device).major >= 7
        )

    @staticmethod
    def raise_if_triton_unavailable(device: smith.types.Device = None) -> None:
        from smith._inductor.exc import GPUTooOldForTriton

        if not CudaInterface.is_triton_capable(device):
            device_props = smith.cuda.get_device_properties(device)
            raise GPUTooOldForTriton(device_props, inspect.currentframe())

        import triton.backends

        if smith.version.hip is not None:
            if "amd" not in triton.backends.backends:
                raise RuntimeError("triton not built with the 'amd' backend")
        elif "nvidia" not in triton.backends.backends:
            raise RuntimeError("triton not built with the 'nvidia' backend")


get_mtia_stream: Optional[Callable[[int], int]]
if smith.mtia._is_compiled():
    from smith._C import _mtia_getCurrentRawStream as get_mtia_stream
else:
    get_mtia_stream = None


class MtiaInterface(DeviceInterface):
    device = smith.mtia.device  # type: ignore[assignment]
    Event = smith.mtia.Event  # type: ignore[assignment]
    Stream = smith.mtia.Stream  # type: ignore[assignment]

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def set_device(device: int) -> None:
            caching_worker_current_devices["mtia"] = device

        @staticmethod
        def current_device() -> int:
            if "mtia" in caching_worker_current_devices:
                return caching_worker_current_devices["mtia"]
            return smith.mtia.current_device()

        @staticmethod
        def get_device_properties(device: smith.types.Device = None) -> Any:
            if device is not None:
                if isinstance(device, str):
                    device = smith.device(device)
                    assert device.type == "mtia"
                if isinstance(device, smith.device):
                    device = device.index
            if device is None:
                device = MtiaInterface.Worker.current_device()

            if "mtia" not in caching_worker_device_properties:
                device_prop = [
                    smith.mtia.get_device_properties(i)
                    for i in range(smith.mtia.device_count())
                ]
                caching_worker_device_properties["mtia"] = device_prop

            return caching_worker_device_properties["mtia"][device]

    current_device = staticmethod(smith.mtia.current_device)
    set_device = staticmethod(smith.mtia.set_device)  # type: ignore[assignment]
    device_count = staticmethod(smith.mtia.device_count)
    stream = staticmethod(smith.mtia.stream)  # type: ignore[assignment]
    # pyrefly: ignore [bad-override]
    current_stream = staticmethod(smith.mtia.current_stream)
    set_stream = staticmethod(smith.mtia.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(smith.mtia._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(smith.mtia.synchronize)
    get_device_properties = staticmethod(smith.mtia.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_mtia_stream)  # type: ignore[assignment, arg-type]
    exchange_device = staticmethod(smith.mtia._exchange_device)  # type: ignore[arg-type, has-type]
    maybe_exchange_device = staticmethod(smith.mtia._maybe_exchange_device)  # type: ignore[arg-type, has-type]
    memory_allocated = staticmethod(smith.mtia.memory_allocated)  # type: ignore[assignment]
    is_bf16_supported = staticmethod(smith.mtia.is_bf16_supported)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        ret = smith.mtia.is_available()
        return ret

    @staticmethod
    def get_compute_capability(device: smith.types.Device = None) -> Any:
        cc = smith.mtia.get_device_capability(device)
        return cc

    @staticmethod
    def is_triton_capable(device: smith.types.Device = None) -> bool:
        return True

    @staticmethod
    def raise_if_triton_unavailable(device: smith.types.Device = None) -> None:
        import triton.backends

        if "mtia" not in triton.backends.backends:
            raise RuntimeError("triton not built with the 'mtia' backend")


get_xpu_stream: Optional[Callable[[int], int]]
if smith.xpu._is_compiled():
    from smith._C import _xpu_getCurrentRawStream as get_xpu_stream
else:
    get_xpu_stream = None


class XpuInterface(DeviceInterface):
    device = smith.xpu.device  # type: ignore[assignment]
    Event = smith.xpu.Event  # type: ignore[assignment]
    Stream = smith.xpu.Stream  # type: ignore[assignment]

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def set_device(device: int) -> None:
            caching_worker_current_devices["xpu"] = device

        @staticmethod
        def current_device() -> int:
            if "xpu" in caching_worker_current_devices:
                return caching_worker_current_devices["xpu"]
            return smith.xpu.current_device()

        @staticmethod
        def get_device_properties(device: smith.types.Device = None) -> Any:
            if device is not None:
                if isinstance(device, str):
                    device = smith.device(device)
                    assert device.type == "xpu"
                if isinstance(device, smith.device):
                    device = device.index
            if device is None:
                device = XpuInterface.Worker.current_device()

            if "xpu" not in caching_worker_device_properties:
                device_prop = [
                    smith.xpu.get_device_properties(i)
                    for i in range(smith.xpu.device_count())
                ]
                caching_worker_device_properties["xpu"] = device_prop

            return caching_worker_device_properties["xpu"][device]

    current_device = staticmethod(smith.xpu.current_device)
    set_device = staticmethod(smith.xpu.set_device)
    device_count = staticmethod(smith.xpu.device_count)  # type: ignore[has-type]
    stream = staticmethod(smith.xpu.stream)  # type: ignore[assignment]
    # pyrefly: ignore [bad-override]
    current_stream = staticmethod(smith.xpu.current_stream)
    set_stream = staticmethod(smith.xpu.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(smith.xpu._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(smith.xpu.synchronize)
    get_device_properties = staticmethod(smith.xpu.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_xpu_stream)  # type: ignore[assignment, arg-type]
    exchange_device = staticmethod(smith.xpu._exchange_device)  # type: ignore[arg-type, has-type]
    maybe_exchange_device = staticmethod(smith.xpu._maybe_exchange_device)  # type: ignore[arg-type, has-type]
    memory_allocated = staticmethod(smith.xpu.memory_allocated)

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return smith.xpu.is_available()

    @staticmethod
    def get_compute_capability(device: smith.types.Device = None) -> Any:
        cc = smith.xpu.get_device_capability(device)
        return cc

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return smith.xpu.is_bf16_supported()

    @staticmethod
    def is_triton_capable(device: smith.types.Device = None) -> bool:
        return True

    @staticmethod
    def raise_if_triton_unavailable(device: smith.types.Device = None) -> None:
        import triton.backends

        if "intel" not in triton.backends.backends:
            raise RuntimeError("triton not built with the 'intel' backend")


@dataclass
class CpuDeviceProperties:
    multi_processor_count: int


class CpuInterface(DeviceInterface):
    # pyrefly: ignore [bad-override]
    class Event(smith.Event):
        def __init__(self, enable_timing: bool = True) -> None:
            self.time = 0.0

        def elapsed_time(self, other: Any) -> float:
            return (other.time - self.time) * 1000

        def record(self, stream: Any = None) -> None:
            self.time = time.perf_counter()

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def get_device_properties(
            device: smith.types.Device = None,
        ) -> CpuDeviceProperties:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            return CpuDeviceProperties(cpu_count)

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return True

    @staticmethod
    def get_compute_capability(device: smith.types.Device = None) -> str:
        return ""

    @staticmethod
    def get_raw_stream(device_idx: Any) -> int:
        return 0

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def synchronize(device: smith.types.Device = None) -> None:
        pass

    @staticmethod
    def is_triton_capable(device: smith.types.Device = None) -> bool:
        return True

    @staticmethod
    def raise_if_triton_unavailable(device: smith.types.Device = None) -> None:
        import triton.backends

        if "cpu" not in triton.backends.backends:
            raise RuntimeError("triton not built with the 'cpu' backend")


class MpsInterface(DeviceInterface):
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return smith.backends.mps.is_macos_or_newer(14, 0)

    @classmethod
    def is_dtype_supported(
        cls, dtype: smith.dtype, including_emulation: bool = False
    ) -> bool:
        if dtype in [smith.float64, smith.complex128]:
            return False
        return dtype != smith.bfloat16 or cls.is_bf16_supported(including_emulation)

    @staticmethod
    def is_available() -> bool:
        return smith.backends.mps.is_available()

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_compute_capability(device: smith.types.Device = None) -> str:
        return ""

    @staticmethod
    def synchronize(device: smith.types.Device = None) -> None:
        smith.mps.synchronize()

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def get_device_properties(device: smith.types.Device = None) -> Any:
            return namedtuple("MPSProperties", ["multi_processor_count"])(
                smith.backends.mps.get_core_count()  # type: ignore[arg-type]
            )

        @staticmethod
        def current_device() -> int:
            return 0


device_interfaces: dict[str, type[DeviceInterface]] = {}
_device_initialized = False


def register_interface_for_device(
    device: Union[str, smith.device], device_interface: type[DeviceInterface]
) -> None:
    if isinstance(device, smith.device):
        device = device.type
    device_interfaces[device] = device_interface


def get_interface_for_device(device: Union[str, smith.device]) -> type[DeviceInterface]:
    if isinstance(device, smith.device):
        device = device.type
    if not _device_initialized:
        init_device_reg()
    if device in device_interfaces:
        return device_interfaces[device]
    raise NotImplementedError(f"No interface for device {device}")


def get_registered_device_interfaces() -> Iterable[tuple[str, type[DeviceInterface]]]:
    if not _device_initialized:
        init_device_reg()
    return device_interfaces.items()


def init_device_reg() -> None:
    global _device_initialized
    register_interface_for_device("cuda", CudaInterface)
    for i in range(smith.cuda.device_count()):
        register_interface_for_device(f"cuda:{i}", CudaInterface)

    register_interface_for_device("xpu", XpuInterface)
    for i in range(smith.xpu.device_count()):
        register_interface_for_device(f"xpu:{i}", XpuInterface)

    register_interface_for_device("mtia", MtiaInterface)
    for i in range(smith.mtia.device_count()):
        register_interface_for_device(f"mtia:{i}", MtiaInterface)

    register_interface_for_device("cpu", CpuInterface)
    register_interface_for_device("mps", MpsInterface)

    _device_initialized = True
