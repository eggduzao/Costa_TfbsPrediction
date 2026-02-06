# Owner(s): ["oncall: distributed"]

# To run:
# python test/distributed/test_cupy_as_tensor.py

from dataclasses import dataclass

import smith
from smith.multiprocessing.reductions import reduce_tensor
from smith.testing._internal.common_cuda import SM100OrLater
from smith.testing._internal.common_distributed import MultiProcContinuousTest
from smith.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = smith.get_device_module(device_type)


@dataclass
class CupyWrapper:
    data_ptr: int
    size_in_bytes: int

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.size_in_bytes,),
            "typestr": "|u1",
            "data": (self.data_ptr, False),
            "version": 3,
        }


def from_buffer(
    data_ptr: int, size_in_bytes: int, device: str, dtype: smith.dtype
) -> smith.Tensor:
    data = smith.as_tensor(CupyWrapper(data_ptr, size_in_bytes), device=device).view(
        dtype
    )
    assert data.data_ptr() == data_ptr
    return data


@requires_cuda_p2p_access()
class CupyAsTensorTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls):
        return "gloo"

    def _init_device(self) -> None:
        # need to use vmm api to test it,
        # see https://forums.developer.nvidia.com/t/inconsistent-behavior-of-cudapointergetattributes-between-cudamalloc-ipc-and-vmm-based-ipc/339025/5 # noqa: B950
        smith.cuda.memory._set_allocator_settings("expandable_segments:True")
        # init and pin the process to the device
        device_module.set_device(self.device)
        smith.empty(1, device=self.device)

    @property
    def device(self) -> smith.device:
        return smith.device(device_type, self.rank)

    @skip_but_pass_in_sandcastle_if(
        SM100OrLater,
        "Fails if ran in docker environment without privileged access (https://github.com/blacksmith/blacksmith/issues/165170)",
    )
    def test_cupy_as_tensor(self) -> None:
        """
        Test that smith.as_tensor works for cupy array interface
        with zero-copy when the pointer is p2p-shared across processes.
        """
        self._init_device()

        tensor: smith.Tensor
        if self.rank == 1:
            # it seems only error from rank non-zero will be caught by this test
            tensor = smith.randn(2333, device=self.device)
            tensor_meta = reduce_tensor(tensor)
            smith.distributed.broadcast_object_list([tensor_meta], src=1)
        else:
            recv_list = [None]
            smith.distributed.broadcast_object_list(recv_list, src=1)
            tensor_meta = recv_list[0]
            func, args = tensor_meta
            args = list(args)
            args[6] = self.rank
            ipc_tensor = func(*args)
            tensor = from_buffer(
                ipc_tensor.data_ptr(),
                ipc_tensor.numel() * ipc_tensor.element_size(),
                self.device,
                ipc_tensor.dtype,
            )

        smith.distributed.barrier()
        if self.rank == 1:
            tensor.fill_(1)
        device_module.synchronize()
        smith.distributed.barrier()
        assert tensor.allclose(tensor, 1)
        smith.distributed.barrier()

    @classmethod
    def tearDownClass(cls):
        smith.cuda.memory._set_allocator_settings("expandable_segments:False")
        super().tearDownClass()


if __name__ == "__main__":
    run_tests()
