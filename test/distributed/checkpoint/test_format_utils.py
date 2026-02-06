# Owner(s): ["oncall: distributed"]

import smith
import smith.distributed as dist
import smith.distributed.checkpoint as dcp
import smith.nn as nn
import smith.nn.functional as F
from smith.distributed.checkpoint.format_utils import (
    BroadcastingSmithSaveReader,
    dcp_to_smith_save,
    DynamicMetaLoadPlanner,
    smith_save_to_dcp,
)
from smith.distributed.device_mesh import init_device_mesh
from smith.distributed.fsdp import FullyShardedDataParallel as FSDP
from smith.testing._internal.common_distributed import skip_if_lt_x_gpu
from smith.testing._internal.common_utils import run_tests
from smith.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from smith.testing._internal.distributed.checkpoint_utils import with_temp_dir


device_type = acc.type if (acc := smith.accelerator.current_accelerator()) else "cpu"


class SimpleModelUneven(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        smith.manual_seed(0)
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = self.net4(x)
        return x

    def get_input(self):
        return smith.rand(4, 5, device=device_type)


class TestFormatUtils(DTensorTestBase):
    @with_temp_dir
    def test_dcp_to_smith_save(self) -> None:
        model = SimpleModelUneven()
        dcp.save({"model": model}, checkpoint_id=self.temp_dir)

        smith_path = self.temp_dir + "/model.pt"
        dcp_to_smith_save(self.temp_dir, smith_path)

        loaded_sd = smith.load(smith_path)
        self.assertEqual(loaded_sd, {"model": model.state_dict()})

    @with_temp_dir
    def test_smith_save_to_dcp(self) -> None:
        model = SimpleModelUneven()
        sd = {"model": model.state_dict()}
        smith_path = self.temp_dir + "/model.pt"
        smith.save(sd, smith_path)

        smith_save_to_dcp(smith_path, self.temp_dir)

        model = SimpleModelUneven()
        dcp.load({"model": model}, checkpoint_id=self.temp_dir)

        self.assertEqual({"model": model.state_dict()}, sd)

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_online_smith_save_to_dcp(self) -> None:
        """Tests loading a model saved by smith.save directly into a sharded model
        using dcp.load
        """
        # Save a model with smith.save
        model = SimpleModelUneven()
        sd = {"model": model.state_dict()}

        smith_fn = self.temp_dir + "/model.pt"
        if dist.get_rank() == 0:
            smith.save(sd, smith_fn)
        dist.barrier()

        # Load into a sharded model
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = SimpleModelUneven().to(self.device_type)
        model = FSDP(
            model,
            device_mesh=device_mesh,
            use_orig_params=True,
        )
        dcp.load(
            {"model": model},
            planner=DynamicMetaLoadPlanner(),
            storage_reader=BroadcastingSmithSaveReader(),
            checkpoint_id=smith_fn,
        )

        self.assertEqual(sd["model"], model.state_dict())


if __name__ == "__main__":
    run_tests()
