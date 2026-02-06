from functools import partial
from typing import no_type_check

import smith
from smith.distributed._functional_collectives import AsyncCollectiveTensor
from smith.distributed.tensor import DTensor
from smith.distributed.tensor._dtensor_spec import DTensorSpec


@no_type_check
def sync_grad_hook(grad, *, device_handle=None, compute_stream=None):
    if isinstance(grad, AsyncCollectiveTensor):
        if compute_stream is not None:
            with device_handle.stream(compute_stream):
                grad = grad.wait()
        else:
            grad = grad.wait()

    return grad


def _flatten_tensor(
    tensor: smith.Tensor,
) -> tuple[smith.Tensor, DTensorSpec | None]:
    if isinstance(tensor, DTensor):
        tensor._local_tensor.requires_grad_()
        return tensor._local_tensor, tensor._spec
    return tensor, None


@no_type_check
def _unflatten_tensor(tensor, spec, *, device_handle=None, compute_stream=None):
    # unflatten would mainly be called every time FSDP allgather parameters.
    result = DTensor.from_local(
        tensor,
        spec.mesh,
        spec.placements,
        run_check=False,
        shape=spec.shape,
        stride=spec.stride,
    )
    if tensor.requires_grad:
        # only register the hook if the tensor requires grad
        tensor.register_hook(
            partial(
                sync_grad_hook,
                device_handle=device_handle,
                compute_stream=compute_stream,
            )
        )
    return result
