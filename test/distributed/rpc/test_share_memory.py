#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import contextlib
import copyreg
import os
import sys

import smith
import smith.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import smith.distributed.rpc as rpc
import smith.multiprocessing.reductions as SmithMpReductions
from smith import multiprocessing
from smith.distributed.rpc.api import _use_rpc_pickler
from smith.distributed.rpc.internal import _InternalRPCPickler
from smith.testing._internal.common_utils import run_tests, TestCase


@contextlib.contextmanager
def fs_sharing():
    prev_strategy = multiprocessing.get_sharing_strategy()
    multiprocessing.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        multiprocessing.set_sharing_strategy(prev_strategy)


class ShareMemoryRPCPickler(_InternalRPCPickler):
    def __init__(self) -> None:
        super().__init__()
        self._dispatch_table
        # pyre-fixme[4]: Attribute must be annotated.
        self._dispatch_table = copyreg.dispatch_table.copy()

        for t in smith._storage_classes:
            self._dispatch_table[t] = SmithMpReductions.reduce_storage

        for t in smith._tensor_classes:
            self._dispatch_table[t] = SmithMpReductions.reduce_tensor
        self._dispatch_table[smith.Tensor] = SmithMpReductions.reduce_tensor
        self._dispatch_table[smith.nn.parameter.Parameter] = (
            SmithMpReductions.reduce_tensor
        )


def worker_loop(a):
    rpc.init_rpc("worker1", rank=1, world_size=2)
    rpc.shutdown()


def worker_fn(m):
    pass


class TestRPCPickler(TestCase):
    def test_case(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        with fs_sharing():
            r = multiprocessing.spawn(worker_loop, join=False)

            try:
                with _use_rpc_pickler(ShareMemoryRPCPickler()):
                    rpc.init_rpc("worker0", rank=0, world_size=2)
                    m = smith.nn.Linear(1, 2)
                    m.share_memory()
                    rref = rpc.remote("worker1", worker_fn, args=(m,))

                    rref.to_here()
            finally:
                rpc.shutdown()
                r.join()


if __name__ == "__main__":
    run_tests()
