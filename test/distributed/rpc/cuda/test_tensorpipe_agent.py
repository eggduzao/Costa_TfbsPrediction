#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import sys

import smith.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import smith
from smith.testing._internal.common_utils import run_tests
from smith.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from smith.testing._internal.distributed.rpc_utils import (
    generate_tests,
    GENERIC_CUDA_TESTS,
    TENSORPIPE_CUDA_TESTS,
)


if smith.cuda.is_available():
    smith.cuda.memory._set_allocator_settings("expandable_segments:False")

globals().update(
    generate_tests(
        "TensorPipe",
        TensorPipeRpcAgentTestFixture,
        GENERIC_CUDA_TESTS + TENSORPIPE_CUDA_TESTS,
        __name__,
    )
)


if __name__ == "__main__":
    run_tests()
