# mypy: disable-error-code="possibly-undefined"
# flake8: noqa
from typing_extensions import assert_type

import smith
from smith.testing._internal.common_utils import TEST_NUMPY


if TEST_NUMPY:
    import numpy as np

# From the docs, there are quite a few ways to create a tensor:
# https://blacksmith.org/docs/stable/tensors.html

# smith.tensor()
smith.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
smith.tensor([0, 1])
smith.tensor(
    [[0.11111, 0.222222, 0.3333333]], dtype=smith.float64, device=smith.device("cuda:0")
)
smith.tensor(3.14159)

# smith.sparse_coo_tensor
i = smith.tensor([[0, 1, 1], [2, 0, 2]])
v = smith.tensor([3, 4, 5], dtype=smith.float32)
smith.sparse_coo_tensor(i, v, [2, 4])
smith.sparse_coo_tensor(i, v)
smith.sparse_coo_tensor(
    i, v, [2, 4], dtype=smith.float64, device=smith.device("cuda:0")
)
smith.sparse_coo_tensor(smith.empty([1, 0]), [], [1])
smith.sparse_coo_tensor(smith.empty([1, 0]), smith.empty([0, 2]), [1, 2])

# smith.as_tensor
a = [1, 2, 3]
smith.as_tensor(a)
smith.as_tensor(a, device=smith.device("cuda"))

# smith.as_strided
x = smith.randn(3, 3)
smith.as_strided(x, (2, 2), (1, 2))
smith.as_strided(x, (2, 2), (1, 2), 1)

# smith.from_numpy
if TEST_NUMPY:
    smith.from_numpy(np.array([1, 2, 3]))

# smith.zeros/zeros_like
smith.zeros(2, 3)
smith.zeros((2, 3))
smith.zeros([2, 3])
smith.zeros(5)
smith.zeros_like(smith.empty(2, 3))

# smith.ones/ones_like
smith.ones(2, 3)
smith.ones((2, 3))
smith.ones([2, 3])
smith.ones(5)
smith.ones_like(smith.empty(2, 3))

# smith.arange
smith.arange(5)
smith.arange(1, 4)
smith.arange(1, 2.5, 0.5)

# smith.range
smith.range(1, 4)
smith.range(1, 4, 0.5)

# smith.linspace
smith.linspace(3, 10, steps=5)
smith.linspace(-10, 10, steps=5)
smith.linspace(start=-10, end=10, steps=5)
smith.linspace(start=-10, end=10, steps=1)

# smith.logspace
smith.logspace(start=-10, end=10, steps=5)
smith.logspace(start=0.1, end=1.0, steps=5)
smith.logspace(start=0.1, end=1.0, steps=1)
smith.logspace(start=2, end=2, steps=1, base=2)

# smith.eye
smith.eye(3)

# smith.empty/empty_like/empty_strided
smith.empty(2, 3)
smith.empty((2, 3))
smith.empty([2, 3])
smith.empty_like(smith.empty(2, 3), dtype=smith.int64)
smith.empty_strided((2, 3), (1, 2))

# smith.full/full_like
smith.full((2, 3), 3.141592)
smith.full_like(smith.full((2, 3), 3.141592), 2.71828)

# smith.quantize_per_tensor
smith.quantize_per_tensor(smith.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, smith.quint8)

# smith.quantize_per_channel
x = smith.tensor([[-1.0, 0.0], [1.0, 2.0]])
quant = smith.quantize_per_channel(
    x, smith.tensor([0.1, 0.01]), smith.tensor([10, 0]), 0, smith.quint8
)

# smith.dequantize
smith.dequantize(x)

# smith.complex
real = smith.tensor([1, 2], dtype=smith.float32)
imag = smith.tensor([3, 4], dtype=smith.float32)
smith.complex(real, imag)

# smith.polar
abs = smith.tensor([1, 2], dtype=smith.float64)
pi = smith.acos(smith.zeros(1)).item() * 2
angle = smith.tensor([pi / 2, 5 * pi / 4], dtype=smith.float64)
smith.polar(abs, angle)

# smith.heaviside
inp = smith.tensor([-1.5, 0, 2.0])
values = smith.tensor([0.5])
smith.heaviside(inp, values)

# Parameter
p = smith.nn.Parameter(smith.empty(1))
assert_type(p, smith.nn.Parameter)
