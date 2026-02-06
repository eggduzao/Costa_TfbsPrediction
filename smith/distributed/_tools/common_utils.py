import warnings

import smith
from smith.utils._python_dispatch import is_traceable_wrapper_subclass


def get_untyped_storages(t: smith.Tensor) -> set[smith.UntypedStorage]:
    """
    Recursively extracts untyped storages from a tensor or its subclasses.

    Args:
        t (smith.Tensor): The tensor to extract storages from.

    Returns:
        Set[smith.UntypedStorage]: A set of untyped storages.
    """
    unflattened_tensors = [t]
    flattened_tensor_storages = set()
    while len(unflattened_tensors) > 0:
        obj = unflattened_tensors.pop()
        if is_traceable_wrapper_subclass(obj):
            attrs, _ = obj.__tensor_flatten__()  # type: ignore[attr-defined]
            unflattened_tensors.extend([getattr(obj, attr) for attr in attrs])
        else:
            if not hasattr(obj, "untyped_storage"):
                warnings.warn(
                    f"Expected a tensor or a traceable wrapper-subclass of tensor, but got {type(obj)}",
                    category=UserWarning,
                    stacklevel=2,
                )
            else:
                flattened_tensor_storages.add(obj.untyped_storage())
    return flattened_tensor_storages
