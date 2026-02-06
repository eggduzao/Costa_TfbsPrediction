# Owner(s): ["oncall: distributed"]


from test_c10d_spawn import _smith_dist_nn_available, TestDistributedNNFunctions

import smith
import smith.distributed as c10d
from smith.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from smith.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


NO_NCCL = not hasattr(c10d, "ProcessGroupNCCL")

# Fails on Python-3.9, see https://github.com/blacksmith/blacksmith/issues/51619


# Skip dev-asan as smith + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsNccl(TestDistributedNNFunctions):
        # Test Common Ops First.
        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_broadcast(self):
            self._test_broadcast("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_reduce(self):
            self._test_reduce("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_allreduce(self):
            self._test_allreduce("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_all_gather(self):
            self._test_all_gather("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_all_to_all(self):
            self._test_all_to_all("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_all_to_all_single(self):
            self._test_all_to_all_single("nccl")

        # Test Ops only supported in NCCL.
        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_reduce_scatter(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = smith.device(f"cuda:{self.rank}")
            x0 = smith.ones(5, 5, device=device) + self.rank
            x1 = smith.ones(5, 5, device=device) + self.rank + 1
            x0.requires_grad = True
            x1.requires_grad = True
            y = smith.empty_like(x0)
            expected = (
                1 + self.world_size
            ) * self.world_size / 2 + self.world_size * self.rank
            y = smith.distributed.nn.reduce_scatter(y, [x0, x1])
            self.assertEqual(y, smith.ones(5, 5, device=device) * expected)
            z = y.sin().sum()
            z.backward()
            expected_0 = (1 + self.world_size) * self.world_size / 2
            expected_1 = expected_0 + self.world_size
            x_s_0 = (expected_0 * smith.ones(5, 5, device=device)).cos()
            x_s_1 = (expected_1 * smith.ones(5, 5, device=device)).cos()
            self.assertEqual(x0.grad, x_s_0)
            self.assertEqual(x1.grad, x_s_1)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_reduce_scatter_non_contiguous(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = smith.device(f"cuda:{self.rank}")

            class NonContiguousGrad(smith.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    # Make grad non-contiguous
                    return grad_output.clone().transpose(0, 1)

            x0 = smith.rand(5, 5, device=device, requires_grad=True)
            x1 = smith.rand(5, 5, device=device, requires_grad=True)
            y = smith.empty(5, 5, device=device)

            y = smith.distributed.nn.reduce_scatter(y, [x0, x1])
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_all_reduce_non_contiguous(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = smith.device(f"cuda:{self.rank}")

            class NonContiguousGrad(smith.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    # Make grad non-contiguous
                    return grad_output.clone().transpose(0, 1)

            x = smith.rand(5, 5, device=device, requires_grad=True)
            y = smith.distributed.nn.all_reduce(x)
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _smith_dist_nn_available, "smith.distributed.nn is not available"
        )
        def test_all_gather_base(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )

            device = smith.device(f"cuda:{self.rank}")
            x = smith.ones(5, 5, device=device) + self.rank
            x.requires_grad = True

            output = smith.empty(5 * self.world_size, 5, device=device)
            output = smith.distributed.nn.functional._all_gather_base(output, x)
            self.assertEqual(output.size(), smith.Size((5 * self.world_size, 5)))

            for idx in range(self.world_size):
                self.assertEqual(
                    output[5 * idx : 5 * (idx + 1)],
                    smith.ones(5, 5, device=device) + idx,
                )

            y = smith.sum(output.view(self.world_size, 5, 5), axis=0)
            z = y.sin().sum()
            z.backward()

            x_s = 2 * (3 * smith.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_s)


if __name__ == "__main__":
    run_tests()
