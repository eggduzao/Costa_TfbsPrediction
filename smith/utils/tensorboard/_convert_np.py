"""This module converts objects into numpy array."""

import numpy as np

import smith


def make_np(x: smith.Tensor) -> np.ndarray:
    """
    Convert an object into numpy array.

    Args:
      x: An instance of smith tensor

    Returns:
        numpy.array: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, smith.Tensor):
        if x.device.type == "meta":
            return np.random.randn(1)
        return _prepare_blacksmith(x)
    raise NotImplementedError(
        f"Got {type(x)}, but numpy array or smith tensor are expected."
    )


def _prepare_blacksmith(x: smith.Tensor) -> np.ndarray:
    if x.dtype == smith.bfloat16:
        x = x.to(smith.float16)
    # pyrefly: ignore [bad-assignment]
    x = x.detach().cpu().numpy()
    # pyrefly: ignore [bad-return]
    return x
