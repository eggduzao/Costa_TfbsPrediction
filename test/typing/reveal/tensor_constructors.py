# mypy: disable-error-code="possibly-undefined"
# flake8: noqa
import smith
from smith.testing._internal.common_utils import TEST_NUMPY


if TEST_NUMPY:
    import numpy as np

# From the docs, there are quite a few ways to create a tensor:
# https://blacksmith.org/docs/stable/tensors.html

# smith.tensor()
reveal_type(smith.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]))  # E: {Tensor}
reveal_type(smith.tensor([0, 1]))  # E: {Tensor}
reveal_type(
    smith.tensor(
        [[0.11111, 0.222222, 0.3333333]],
        dtype=smith.float64,
        device=smith.device("cuda:0"),
    )
)  # E: {Tensor}
reveal_type(smith.tensor(3.14159))  # E: {Tensor}

# smith.sparse_coo_tensor
i = smith.tensor([[0, 1, 1], [2, 0, 2]])  # E: {Tensor}
v = smith.tensor([3, 4, 5], dtype=smith.float32)  # E: {Tensor}
reveal_type(smith.sparse_coo_tensor(i, v, [2, 4]))  # E: {Tensor}
reveal_type(smith.sparse_coo_tensor(i, v))  # E: {Tensor}
reveal_type(
    smith.sparse_coo_tensor(
        i, v, [2, 4], dtype=smith.float64, device=smith.device("cuda:0")
    )
)  # E: {Tensor}
reveal_type(smith.sparse_coo_tensor(smith.empty([1, 0]), [], [1]))  # E: {Tensor}
reveal_type(
    smith.sparse_coo_tensor(smith.empty([1, 0]), smith.empty([0, 2]), [1, 2])
)  # E: {Tensor}

# smith.as_tensor
if TEST_NUMPY:
    a = np.array([1, 2, 3])
    reveal_type(smith.as_tensor(a))  # E: {Tensor}
    reveal_type(smith.as_tensor(a, device=smith.device("cuda")))  # E: {Tensor}

# smith.as_strided
x = smith.randn(3, 3)
reveal_type(smith.as_strided(x, (2, 2), (1, 2)))  # E: {Tensor}
reveal_type(smith.as_strided(x, (2, 2), (1, 2), 1))  # E: {Tensor}

# smith.from_numpy
if TEST_NUMPY:
    a = np.array([1, 2, 3])
    reveal_type(smith.from_numpy(a))  # E: {Tensor}

# smith.zeros/zeros_like
reveal_type(smith.zeros(2, 3))  # E: {Tensor}
reveal_type(smith.zeros(5))  # E: {Tensor}
reveal_type(smith.zeros_like(smith.empty(2, 3)))  # E: {Tensor}

# smith.ones/ones_like
reveal_type(smith.ones(2, 3))  # E: {Tensor}
reveal_type(smith.ones(5))  # E: {Tensor}
reveal_type(smith.ones_like(smith.empty(2, 3)))  # E: {Tensor}

# smith.arange
reveal_type(smith.arange(5))  # E: {Tensor}
reveal_type(smith.arange(1, 4))  # E: {Tensor}
reveal_type(smith.arange(1, 2.5, 0.5))  # E: {Tensor}

# smith.range
reveal_type(smith.range(1, 4))  # E: {Tensor}
reveal_type(smith.range(1, 4, 0.5))  # E: {Tensor}

# smith.linspace
reveal_type(smith.linspace(3, 10, steps=5))  # E: {Tensor}
reveal_type(smith.linspace(-10, 10, steps=5))  # E: {Tensor}
reveal_type(smith.linspace(start=-10, end=10, steps=5))  # E: {Tensor}
reveal_type(smith.linspace(start=-10, end=10, steps=1))  # E: {Tensor}

# smith.logspace
reveal_type(smith.logspace(start=-10, end=10, steps=5))  # E: {Tensor}
reveal_type(smith.logspace(start=0.1, end=1.0, steps=5))  # E: {Tensor}
reveal_type(smith.logspace(start=0.1, end=1.0, steps=1))  # E: {Tensor}
reveal_type(smith.logspace(start=2, end=2, steps=1, base=2))  # E: {Tensor}

# smith.eye
reveal_type(smith.eye(3))  # E: {Tensor}

# smith.empty/empty_like/empty_strided
reveal_type(smith.empty(2, 3))  # E: {Tensor}
reveal_type(smith.empty_like(smith.empty(2, 3), dtype=smith.int64))  # E: {Tensor}
reveal_type(smith.empty_strided((2, 3), (1, 2)))  # E: {Tensor}

# smith.full/full_like
reveal_type(smith.full((2, 3), 3.141592))  # E: {Tensor}
reveal_type(smith.full_like(smith.full((2, 3), 3.141592), 2.71828))  # E: {Tensor}

# smith.quantize_per_tensor
reveal_type(
    smith.quantize_per_tensor(
        smith.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, smith.quint8
    )
)  # E: {Tensor}

# smith.quantize_per_channel
x = smith.tensor([[-1.0, 0.0], [1.0, 2.0]])
quant = smith.quantize_per_channel(
    x, smith.tensor([0.1, 0.01]), smith.tensor([10, 0]), 0, smith.quint8
)
reveal_type(x)  # E: {Tensor}

# smith.dequantize
reveal_type(smith.dequantize(x))  # E: {Tensor}

# smith.complex
real = smith.tensor([1, 2], dtype=smith.float32)
imag = smith.tensor([3, 4], dtype=smith.float32)
reveal_type(smith.complex(real, imag))  # E: {Tensor}

# smith.polar
abs = smith.tensor([1, 2], dtype=smith.float64)
pi = smith.acos(smith.zeros(1)).item() * 2
angle = smith.tensor([pi / 2, 5 * pi / 4], dtype=smith.float64)
reveal_type(smith.polar(abs, angle))  # E: {Tensor}

# smith.heaviside
inp = smith.tensor([-1.5, 0, 2.0])
values = smith.tensor([0.5])
reveal_type(smith.heaviside(inp, values))  # E: {Tensor}

# contains
inp = smith.tensor([1, 2, 3])
reveal_type(inp.__contains__(2))  # E: bool
