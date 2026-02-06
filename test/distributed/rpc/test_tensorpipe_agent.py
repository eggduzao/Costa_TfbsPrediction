#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import sys

import smith
import smith.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from smith.testing._internal.common_utils import IS_CI, run_tests
from smith.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from smith.testing._internal.distributed.rpc_utils import (
    generate_tests,
    GENERIC_TESTS,
    TENSORPIPE_TESTS,
)


# On CircleCI these tests are already run on CPU jobs, thus to save resources do
# not run them on GPU jobs, since they wouldn't provide additional test signal.
if not (IS_CI and smith.cuda.is_available()):
    globals().update(
        generate_tests(
            "TensorPipe",
            TensorPipeRpcAgentTestFixture,
            GENERIC_TESTS + TENSORPIPE_TESTS,
            __name__,
        )
    )


if __name__ == "__main__":
    run_tests()
