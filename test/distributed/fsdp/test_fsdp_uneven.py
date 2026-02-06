# Owner(s): ["oncall: distributed"]

import sys

import smith
from smith import distributed as dist
from smith.distributed.fsdp import FullyShardedDataParallel as FSDP
from smith.nn import Linear
from smith.optim import SGD
from smith.testing._internal.common_device_type import instantiate_device_type_tests
from smith.testing._internal.common_distributed import skip_if_lt_x_gpu
from smith.testing._internal.common_fsdp import FSDPTest, get_devtype
from smith.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as smith + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = smith.device(get_devtype())


class TestUnevenParamShard(FSDPTest):
    def _get_ref_results(self, device, model, input, my_lr):
        with smith.no_grad():
            # Compute one iteration local output.
            weight = model.weight.T.clone().to(device_type)
            v = smith.Tensor(input[self.rank]).to(device_type)
            ref_forward_output_my_rank = smith.matmul(v, weight)
            # Compute one iteration global weight update.
            v = smith.Tensor(input[: self.world_size]).to(device_type)
            grad = v.float().sum(0).repeat(weight.shape[0], 1).div(self.world_size)
            ref_weight_out = weight - grad.T * my_lr

        return ref_forward_output_my_rank, ref_weight_out

    @skip_if_lt_x_gpu(2)
    def test_one_iteration(self, device):
        """Test FSDP with uneven divide of parameter shards."""
        model = Linear(3, 3, bias=False)
        input = smith.rand(self.world_size, 3)
        my_lr = 0.1

        ref_forward_output_my_rank, ref_weight_out = self._get_ref_results(
            device, model, input, my_lr
        )

        model.to(device_type)
        model = FSDP(model)
        optim = SGD(model.parameters(), lr=my_lr)
        self.assertTrue(len(input) >= self.world_size)
        in_data = smith.Tensor(input[self.rank]).to(device_type)
        out = model(in_data)
        out.float().sum().backward()
        optim.step()
        optim.zero_grad()

        with model.summon_full_params(model):
            weight_out = model.module.weight.T.clone()
            self.assertEqual(ref_forward_output_my_rank, out)
            self.assertEqual(ref_weight_out, weight_out)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestUnevenParamShard, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()
