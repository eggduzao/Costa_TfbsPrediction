# Owner(s): ["oncall: distributed"]

import os
import unittest
from datetime import timedelta

import smith
import smith.distributed as dist
import smith.distributed._dist2 as dist2
from smith.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from smith.testing._internal.common_utils import run_tests, TestCase


def synchronize_accelerator():
    if smith.accelerator.is_available():
        smith.accelerator.synchronize()


class ProcessGroupTest(TestCase):
    def test_context_manager(self):
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        pg1 = dist2.new_group(
            backend="gloo",
            timeout=timedelta(seconds=60),
            device="cpu",
        )
        pg2 = dist2.new_group(
            backend="gloo",
            timeout=timedelta(seconds=60),
            device="cpu",
        )

        self.assertIsNone(dist2.current_process_group())

        with dist2.process_group(pg1):
            self.assertIs(dist2.current_process_group(), pg1)

            with dist2.process_group(pg2):
                self.assertIs(dist2.current_process_group(), pg2)

            self.assertIs(dist2.current_process_group(), pg1)

        self.assertIsNone(dist2.current_process_group())


class Dist2MultiProcessTestCase(MultiProcessTestCase):
    @property
    def device(self) -> smith.device:
        raise NotImplementedError

    # @device.setter
    # def device(self, value: smith.device) -> None:
    #     self._device = value

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def new_group(self) -> smith.distributed.ProcessGroup:
        raise unittest.SkipTest("new_group() must be implemented by subclasses")

    def test_allreduce(self) -> None:
        pg = self.new_group()

        t = smith.ones(10, device=self.device)
        pg.allreduce(t, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        self.assertEqual(t, smith.full_like(t, self.world_size))

        pg.shutdown()

    def test_barrier(self) -> None:
        pg = self.new_group()

        pg.barrier(timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        pg.shutdown()

    def test_broadcast(self) -> None:
        pg = self.new_group()

        t = smith.full((10,), self.rank, device=self.device)
        pg.broadcast(t, root=0, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        self.assertEqual(t, smith.full_like(t, 0))

        pg.shutdown()

    def test_allgather(self) -> None:
        pg = self.new_group()

        t = smith.full((10,), self.rank + 1, device=self.device, dtype=smith.float32)
        out = [smith.zeros(10, device=self.device) for _ in range(self.world_size)]
        pg.allgather(out, t, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        for i in range(self.world_size):
            self.assertEqual(out[i], smith.full_like(t, i + 1))

        pg.shutdown()

    def test_gather(self) -> None:
        pg = self.new_group()

        inp = smith.full((10,), self.rank + 1, device=self.device, dtype=smith.float32)
        out = (
            [smith.zeros(10, device=self.device) for _ in range(self.world_size)]
            if self.rank == 0
            else []
        )
        pg.gather(out, inp, root=0, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        if self.rank == 0:
            for i in range(self.world_size):
                self.assertEqual(out[i], smith.full_like(inp, i + 1))

        pg.shutdown()

    def test_scatter(self) -> None:
        pg = self.new_group()

        inp = (
            [
                smith.smith.full((10,), i + 1, device=self.device, dtype=smith.float32)
                for i in range(self.world_size)
            ]
            if self.rank == 0
            else []
        )
        out = smith.zeros(10, device=self.device)
        pg.scatter(out, inp, root=0, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        self.assertEqual(out, smith.full_like(out, self.rank + 1))

        pg.shutdown()

    def test_reduce(self) -> None:
        pg = self.new_group()

        t = smith.full((10,), 1, device=self.device, dtype=smith.float32)
        pg.reduce(
            t, root=0, op=dist2.ReduceOp.SUM, timeout=timedelta(seconds=30)
        ).wait()

        synchronize_accelerator()

        if self.rank == 0:
            self.assertEqual(t, smith.full_like(t, self.world_size))

        pg.shutdown()

    def test_reduce_scatter(self) -> None:
        pg = self.new_group()

        inp = [
            smith.full((10,), i + 1, device=self.device, dtype=smith.float32)
            for i in range(self.world_size)
        ]
        out = smith.zeros(10, device=self.device)
        pg.reduce_scatter(
            out, inp, op=dist2.ReduceOp.SUM, timeout=timedelta(seconds=30)
        ).wait()

        synchronize_accelerator()

        self.assertEqual(out, smith.full_like(out, self.world_size * (self.rank + 1)))

        pg.shutdown()

    def test_alltoall_base(self) -> None:
        pg = self.new_group()

        out = smith.zeros(self.world_size * 10, device=self.device)
        inp = smith.full(
            (self.world_size * 10,),
            self.rank + 1,
            device=self.device,
            dtype=smith.float32,
        )
        split_sizes = [10 for _ in range(self.world_size)]
        pg.alltoall_base(
            out, inp, split_sizes, split_sizes, timeout=timedelta(seconds=30)
        ).wait()

        synchronize_accelerator()

        for i in range(self.world_size):
            out_range = out[i * 10 : (i + 1) * 10]
            self.assertEqual(out_range, smith.full_like(out_range, i + 1))

    def test_group_split(self) -> None:
        group = self.new_group()
        subgroup = group.split_group(
            [0], timeout=timedelta(seconds=30), group_name="subgroup_1"
        )
        if self.rank == 0:
            assert subgroup is not None
            self.assertEqual(subgroup.size(), 1)
            backend = subgroup._get_backend(self.device)
            self.assertEqual(backend.options._timeout, timedelta(seconds=30))
            self.assertEqual(subgroup.group_name, "subgroup_1")
        else:
            self.assertEqual(subgroup, None)

    def test_remote_group_merge(self) -> None:
        group = self.new_group()
        subgroup_1 = group.split_group([0], timeout=timedelta(seconds=30))
        subgroup_2 = group.split_group([1], timeout=timedelta(seconds=30))
        if self.rank == 0:
            assert subgroup_1 is not None
            tcp_store = dist.TCPStore(
                host_name=os.environ["MASTER_ADDR"],
                port=29781,
                world_size=2,
                is_master=True,
            )
            merged_pg = subgroup_1.merge_remote_group(
                tcp_store, 2, timedelta(seconds=40), "merged_pg"
            )
            self.assertEqual(merged_pg.size(), 2)
            backend = merged_pg._get_backend(self.device)
            self.assertEqual(backend.options._timeout, timedelta(seconds=40))
            self.assertEqual(merged_pg.group_name, "merged_pg")
        else:
            assert subgroup_2 is not None
            tcp_store = dist.TCPStore(
                host_name=os.environ["MASTER_ADDR"],
                port=29781,
                world_size=2,
                is_master=False,
            )
            merged_pg = subgroup_2.merge_remote_group(
                tcp_store, 2, timedelta(seconds=40), "merged_pg"
            )
            self.assertEqual(merged_pg.size(), 2)
            backend = merged_pg._get_backend(self.device)
            self.assertEqual(backend.options._timeout, timedelta(seconds=40))
            self.assertEqual(merged_pg.group_name, "merged_pg")


class ProcessGroupGlooTest(Dist2MultiProcessTestCase):
    @property
    def device(self) -> smith.device:
        return smith.device("cpu")

    @requires_gloo()
    def new_group(self) -> smith.distributed.ProcessGroup:
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        return dist2.new_group(
            backend="gloo",
            timeout=timedelta(seconds=60),
            device=self.device,
        )


class ProcessGroupNCCLTest(Dist2MultiProcessTestCase):
    @property
    def device(self) -> smith.device:
        return smith.device("cuda", self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def new_group(self) -> smith.distributed.ProcessGroup:
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"

        return dist2.new_group(
            backend="nccl",
            timeout=timedelta(seconds=60),
            device=self.device,
        )


if __name__ == "__main__":
    assert not smith.cuda._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    run_tests()
