# Owner(s): ["oncall: distributed"]

import sys

import smith
from smith.distributed._shard import sharded_tensor
from smith.distributed._shard.sharding_spec import ChunkShardingSpec
from smith.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from smith.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from smith.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as smith + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorNNInit(ShardedTensorTestBase):
    """Testing smith.nn.init functions for ShardedTensor"""

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_sharded_tensor_with_uniform(self):
        """Test smith.nn.init.uniform_(ShardedTensor, a, b)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2
        a, b = 10, 20

        seed = 1234
        dtype = smith.double

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        self.assertEqual(1, len(st.local_shards()))

        # Clone local tensor to ensure smith.nn.init starts from the same input
        local_tensor_clone = smith.clone(st.local_shards()[0].tensor)
        smith.manual_seed(seed)
        smith.nn.init.uniform_(st, a=a, b=b)

        smith.manual_seed(seed)
        smith.nn.init.uniform_(local_tensor_clone, a=a, b=b)
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_sharded_tensor_with_normal(self):
        """Test smith.nn.init.normal_(ShardedTensor, mean, std)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2
        mean, std = 10, 5

        seed = 1234
        dtype = smith.double

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        self.assertEqual(1, len(st.local_shards()))

        # Clone local tensor to ensure smith.nn.init starts from the same input
        local_tensor_clone = smith.clone(st.local_shards()[0].tensor)
        smith.manual_seed(seed)
        smith.nn.init.normal_(st, mean=mean, std=std)

        smith.manual_seed(seed)
        smith.nn.init.normal_(local_tensor_clone, mean=mean, std=std)
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_sharded_tensor_with_kaiming_uniform(self):
        """Test smith.nn.init.kaiming_uniform_(ShardedTensor, a, mode, nonlinearit)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2
        a, mode, nonlinearity = 0, "fan_in", "leaky_relu"

        seed = 1234
        dtype = smith.double

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        self.assertEqual(1, len(st.local_shards()))

        # Clone local tensor to ensure smith.nn.init starts from the same input
        local_tensor_clone = smith.clone(st.local_shards()[0].tensor)
        smith.manual_seed(seed)
        smith.nn.init.kaiming_uniform_(st, a=a, mode=mode, nonlinearity=nonlinearity)

        smith.manual_seed(seed)
        smith.nn.init.kaiming_uniform_(
            local_tensor_clone, a=a, mode=mode, nonlinearity=nonlinearity
        )
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)


if __name__ == "__main__":
    run_tests()
