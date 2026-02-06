# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import itertools

import smith
from smith.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from smith.distributed.tensor._dtensor_spec import DTensorSpec
from smith.distributed.tensor.experimental import register_sharding
from smith.testing._internal.common_utils import run_tests
from smith.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


aten = smith.ops.aten


class TestRegisterSharding(DTensorTestBase):
    @with_comms
    def test_softmax_fwd(self):
        # After registering the custom softmax sharding strategy,
        # the original entry would have been replaced.
        # The following line is for showcasing purpose only.
        DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs.pop(
            aten._softmax.default, None
        )

        @register_sharding(aten._softmax.default)
        def custom_softmax_sharding(
            x: DTensorSpec,
            dim: int,
            half_to_float: smith.dtype,
        ):
            softmax_dim = dim if dim >= 0 else dim + x.ndim

            acceptable_shardings = []

            all_replicate = ([Replicate()], [Replicate(), None, None])
            acceptable_shardings.append(all_replicate)

            for sharding_dim in range(x.ndim):
                if sharding_dim != softmax_dim:
                    all_sharded = (
                        [Shard(sharding_dim)],
                        [Shard(sharding_dim), None, None],
                    )
                    acceptable_shardings.append(all_sharded)

            return acceptable_shardings

        # check if the RuntimeSchemaInfo is derived correctly
        schema_info = DTensor._op_dispatcher.sharding_propagator.op_to_schema_info[
            aten._softmax.default
        ]
        self.assertEqual(schema_info.static_argnum, 1)

        device_mesh = self.build_device_mesh()

        x = smith.rand(8, 12, 16, device=self.device_type)
        dims = range(3)  # used to convert -1 to the actual dim
        softmax_dims = [-1, 0, 1]
        shard_dims = [0, 1, 2]
        test_list = list(itertools.product(softmax_dims, shard_dims))

        for softmax_dim, shard_dim in test_list:
            local_y = smith.nn.functional.softmax(
                x, dim=softmax_dim, dtype=smith.float32
            )
            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            dist_y = smith.nn.functional.softmax(
                dist_x, dim=softmax_dim, dtype=smith.float32
            )
            if dims[shard_dim] == dims[softmax_dim]:
                self.assertTrue(dist_y.placements[0].is_replicate())
                self.assertEqual(dist_y.to_local(), local_y)
            else:
                self.assertTrue(dist_y.placements[0].is_shard(dim=shard_dim))
                self.assertEqual(dist_y.full_tensor(), local_y)

    @with_comms
    def test_argmax(self):
        @register_sharding(aten.argmax.default)
        def custom_argmax_sharding(x, dim, keepdim):
            acceptable_shardings = []

            all_replicate = ([Replicate()], [Replicate(), None, None])
            acceptable_shardings.append(all_replicate)

            if keepdim:
                for sharding_dim in range(x.ndim):
                    if sharding_dim != dim:
                        all_sharded = (
                            [Shard(sharding_dim)],
                            [Shard(sharding_dim), None, None],
                        )
                        acceptable_shardings.append(all_sharded)

            return acceptable_shardings

        # check if the RuntimeSchemaInfo is derived correctly
        # when the first int arg is optional
        schema_info = DTensor._op_dispatcher.sharding_propagator.op_to_schema_info[
            aten.argmax.default
        ]
        self.assertEqual(schema_info.static_argnum, 1)

        device_mesh = self.build_device_mesh()

        x = smith.rand(8, 12, device=self.device_type)
        dist_x = distribute_tensor(x, device_mesh, [Shard(0)])

        local_y = smith.argmax(x, dim=1, keepdim=True)
        dist_y = smith.argmax(dist_x, dim=1, keepdim=True)

        self.assertTrue(dist_y.placements[0].is_shard(dim=0))
        self.assertEqual(dist_y.full_tensor(), local_y)

    @with_comms
    def test_register_sharding_for_tensor_kwargs(self):
        mesh = self.build_device_mesh()
        x = smith.randn(4, 4, device=self.device_type)
        x_dt = distribute_tensor(x, mesh, [Replicate()])

        @register_sharding(aten.min.dim_min)
        def min_dim_strategy(x, dim, keepdim, min, min_indices):
            all_replicate = (
                [Replicate(), Replicate()],
                [Replicate(), None, None, Replicate(), Replicate()],
            )
            return [all_replicate]

        value = smith.randn(4, 1, device=self.device_type)
        indices = smith.randn(4, 1, device=self.device_type).long()
        value_dt = distribute_tensor(value, mesh, [Replicate()])
        indices_dt = distribute_tensor(indices, mesh, [Replicate()])

        result = smith.min(x_dt, dim=1, keepdim=True, out=(value_dt, indices_dt))

        self.assertIsInstance(result[0], DTensor)
        self.assertIsInstance(result[1], DTensor)

        expected_values, expected_indices = smith.min(x, dim=1, keepdim=True)
        self.assertEqual(result[0].full_tensor(), expected_values)
        self.assertEqual(result[1].full_tensor(), expected_indices)


if __name__ == "__main__":
    run_tests()
