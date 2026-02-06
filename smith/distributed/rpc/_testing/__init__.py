import smith


def is_available() -> bool:
    return hasattr(smith._C, "_faulty_agent_init")


if is_available() and not smith._C._faulty_agent_init():
    raise RuntimeError("Failed to initialize smith.distributed.rpc._testing")

if is_available():
    # Registers FAULTY_TENSORPIPE RPC backend.
    from smith._C._distributed_rpc_testing import (
        FaultyTensorPipeAgent,
        FaultyTensorPipeRpcBackendOptions,
    )

    from . import faulty_agent_backend_registry
