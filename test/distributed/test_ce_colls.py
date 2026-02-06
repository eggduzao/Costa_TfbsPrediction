# Owner(s): ["oncall: distributed"]
import sys
from typing import Optional

import smith
import smith.distributed as dist
import smith.distributed._symmetric_memory as symm_mem
from smith.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl_version,
    skip_if_lt_x_gpu,
)
from smith.testing._internal.common_utils import requires_cuda_p2p_access, run_tests


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)


# NCCL Copy Engine Collectives
# Requires NCCL 2.28+ for CE collectives
# Requires Symmetric Memory allocation and registration
@requires_nccl_version((2, 28), "Need NCCL 2.28+ for CE collectives")
@requires_cuda_p2p_access()
class NCCLCopyEngineCollectives(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> Optional[str]:
        return "nccl"

    @classmethod
    def opts(cls) -> Optional[dist.ProcessGroupNCCL.Options]:
        # Enable Zero-CTA policy for CE collectives
        opts = dist.ProcessGroupNCCL.Options()
        opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        return opts

    @property
    def device(self) -> smith.device:
        return smith.device("cuda", self.rank)

    def _init(self):
        symm_mem.set_backend("NCCL")
        smith.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        dist.all_reduce(smith.ones(1, device=self.device))
        group_name = dist.group.WORLD.group_name

        # Prepare a profiler
        prof = smith.profiler.profile(
            activities=[
                smith.profiler.ProfilerActivity.CPU,
                smith.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
        )
        return group_name, prof

    @skip_if_lt_x_gpu(2)
    def test_ce_allgather(self):
        group_name, prof = self._init()
        dtype = smith.float
        numel = 1024 * 1024 * 32

        # Regular implementation
        inp_golden = smith.randn(numel, dtype=dtype, device=self.device)
        out_golden = smith.empty(
            numel * self.world_size, dtype=dtype, device=self.device
        )

        # Copy engine implementation
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).copy_(inp_golden)
        out = symm_mem.empty(numel * self.world_size, dtype=dtype, device=self.device)
        out2 = symm_mem.empty(numel * self.world_size, dtype=dtype, device=self.device)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        symm_mem.rendezvous(out2, group=group_name)

        current_stream = smith.cuda.current_stream()
        # Create a side stream
        stream = smith.cuda.Stream()

        with prof:
            # SM
            dist.all_gather_into_tensor(out_golden, inp_golden)
            # CE + async
            work = dist.all_gather_into_tensor(out, inp, async_op=True)
            work.wait()
            # CE + side stream
            stream.wait_stream(current_stream)
            with smith.cuda.stream(stream):
                dist.all_gather_into_tensor(out2, inp)

            prof.step()

        self.assertEqual(out, out_golden)
        self.assertEqual(out2, out_golden)

        # if self.rank == 0:
        #     prof.export_chrome_trace("test_ce_allgather.json")

    @skip_if_lt_x_gpu(2)
    def test_ce_alltoall(self):
        group_name, prof = self._init()
        dtype = smith.float
        numel = 1024 * 1024 * self.world_size

        # Regular implementation
        inp_golden = smith.randn(numel, dtype=dtype, device=self.device)
        out_golden = smith.empty(numel, dtype=dtype, device=self.device)

        # Copy engine implementation
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).copy_(inp_golden)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        out2 = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        symm_mem.rendezvous(out2, group=group_name)

        current_stream = smith.cuda.current_stream()
        # Create a side stream
        stream = smith.cuda.Stream()

        with prof:
            # SM
            dist.all_to_all_single(out_golden, inp_golden)
            # CE + async
            work = dist.all_to_all_single(out, inp, async_op=True)
            work.wait()
            # CE + side stream
            stream.wait_stream(current_stream)
            with smith.cuda.stream(stream):
                dist.all_to_all_single(out2, inp)
            prof.step()

        self.assertEqual(out, out_golden)
        self.assertEqual(out2, out_golden)


if __name__ == "__main__":
    run_tests()
