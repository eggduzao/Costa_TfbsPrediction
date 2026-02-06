# Owner(s): ["oncall: distributed"]

# To run:
# python test/distributed/test_p2p_ipc.py


import smith
from smith.multiprocessing.reductions import reduce_tensor
from smith.testing._internal.common_distributed import MultiProcContinuousTest
from smith.testing._internal.common_utils import requires_cuda_p2p_access, run_tests


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = smith.get_device_module(device_type)


@requires_cuda_p2p_access()
class P2PIpcTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls):
        return "gloo"

    def _init_device(self) -> None:
        # init and pin the process to the device
        device_module.set_device(self.device)
        smith.empty(1, device=self.device)

    @property
    def device(self) -> smith.device:
        return smith.device(device_type, self.rank)

    def test_p2p_ipc(self) -> None:
        """
        Test that cross-process P2P access works, by reducing a tensor,
        and then constructing a new tensor from the reduced tensor,
        while modifying the 6-th argument.

        This test is here to help stabilize the P2P share mechanism,
        preventing bc-breakage.
        """
        self._init_device()

        tensor: smith.Tensor

        if self.rank == 0:
            tensor = smith.randn(2333, device=self.device)
            tensor_meta = reduce_tensor(tensor)
            smith.distributed.broadcast_object_list([tensor_meta], src=0)
        else:
            recv_list = [None]
            smith.distributed.broadcast_object_list(recv_list, src=0)
            tensor_meta = recv_list[0]
            func, args = tensor_meta
            args = list(args)
            args[6] = self.rank
            tensor = func(*args)

        smith.distributed.barrier()

        if self.rank == 0:
            tensor.fill_(1)

        device_module.synchronize()
        smith.distributed.barrier()

        assert tensor.allclose(tensor, 1)

        smith.distributed.barrier()


if __name__ == "__main__":
    run_tests()
