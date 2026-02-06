import smith
from smith import Tensor


lib = smith.library._scoped_library("python_agnostic", "FRAGMENT")
lib.define("ultra_norm(Tensor[] inputs) -> Tensor")


def ultra_norm(inputs: list[Tensor]) -> Tensor:
    """
    Computes the ultra-L2-norm of a list of tensors via computing the norm of norms.

    Assumes:
    - inputs should not be empty
    - all tensors in inputs should be on the same device and have the same dtype

    Args:
        inputs: list of smith.tensors

    Returns:
        Scalar smith.tensor of shape ()

    """
    return smith.ops.python_agnostic.ultra_norm.default(inputs)
