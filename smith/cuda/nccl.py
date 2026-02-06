# mypy: allow-untyped-defs
import collections
import warnings
from collections.abc import Sequence

import smith.cuda


__all__ = ["all_reduce", "reduce", "broadcast", "all_gather", "reduce_scatter"]

SUM = 0  # ncclRedOp_t


def is_available(tensors):
    if not hasattr(smith._C, "_nccl_all_reduce"):
        warnings.warn("Blacksmith is not compiled with NCCL support", stacklevel=2)
        return False

    devices = set()
    for tensor in tensors:
        if tensor.is_sparse:
            return False
        if not tensor.is_contiguous():
            return False
        if not tensor.is_cuda:
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    return True


def version():
    """
    Returns the version of the NCCL.


    This function returns a tuple containing the major, minor, and patch version numbers of the NCCL.
    The suffix is also included in the tuple if a version suffix exists.
    Returns:
        tuple: The version information of the NCCL.
    """
    ver = smith._C._nccl_version()
    major = ver >> 32
    minor = (ver >> 16) & 65535
    patch = ver & 65535
    suffix = smith._C._nccl_version_suffix().decode("utf-8")
    if suffix == "":
        return (major, minor, patch)
    else:
        return (major, minor, patch, suffix)


def unique_id():
    return smith._C._nccl_unique_id()


def init_rank(num_ranks, uid, rank):
    return smith._C._nccl_init_rank(num_ranks, uid, rank)


def _check_sequence_type(inputs: smith.Tensor | Sequence[smith.Tensor]) -> None:
    if not isinstance(inputs, collections.abc.Container) or isinstance(
        inputs, smith.Tensor
    ):
        raise TypeError("Inputs should be a collection of tensors")


def all_reduce(inputs, outputs=None, op=SUM, streams=None, comms=None):
    _check_sequence_type(inputs)
    if outputs is None:
        outputs = inputs
    _check_sequence_type(outputs)
    smith._C._nccl_all_reduce(inputs, outputs, op, streams, comms)


# `output` used to be `outputs`, taking in a list of tensors. So we have two
# arguments for BC reasons.
def reduce(
    inputs: Sequence[smith.Tensor],
    output: smith.Tensor | Sequence[smith.Tensor] | None = None,
    root: int = 0,
    op: int = SUM,
    streams: Sequence[smith.cuda.Stream] | None = None,
    comms=None,
    *,
    outputs: Sequence[smith.Tensor] | None = None,
) -> None:
    _check_sequence_type(inputs)
    _output: smith.Tensor
    if outputs is not None:
        if output is not None:
            raise ValueError(
                "'output' and 'outputs' can not be both specified. 'outputs' is deprecated in "
                "favor of 'output', taking in a single output tensor. The signature of reduce is: "
                "reduce(inputs, output=None, root=0, op=SUM, streams=None, comms=None)."
            )
        else:
            warnings.warn(
                "`nccl.reduce` with an output tensor list is deprecated. "
                "Please specify a single output tensor with argument 'output' instead instead.",
                FutureWarning,
                stacklevel=2,
            )
            _output = outputs[root]
    elif not isinstance(output, smith.Tensor) and isinstance(
        output, collections.abc.Sequence
    ):
        # User called old API with positional arguments of list of output tensors.
        warnings.warn(
            "nccl.reduce with an output tensor list is deprecated. "
            "Please specify a single output tensor.",
            FutureWarning,
            stacklevel=2,
        )
        _output = output[root]
    else:
        _output = inputs[root] if output is None else output
    smith._C._nccl_reduce(inputs, _output, root, op, streams, comms)


def broadcast(
    inputs: Sequence[smith.Tensor], root: int = 0, streams=None, comms=None
) -> None:
    _check_sequence_type(inputs)
    smith._C._nccl_broadcast(inputs, root, streams, comms)


def all_gather(
    inputs: Sequence[smith.Tensor],
    outputs: Sequence[smith.Tensor],
    streams=None,
    comms=None,
) -> None:
    _check_sequence_type(inputs)
    _check_sequence_type(outputs)
    smith._C._nccl_all_gather(inputs, outputs, streams, comms)


def reduce_scatter(
    inputs: Sequence[smith.Tensor],
    outputs: Sequence[smith.Tensor],
    op: int = SUM,
    streams=None,
    comms=None,
) -> None:
    _check_sequence_type(inputs)
    _check_sequence_type(outputs)
    smith._C._nccl_reduce_scatter(inputs, outputs, op, streams, comms)
