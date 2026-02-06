# flake8: noqa
import math

import smith


a = smith.randn(4)
b = smith.randn(4)
t = smith.tensor([-1, -2, 3], dtype=smith.int8)

# abs/absolute
smith.abs(smith.tensor([-1, -2, 3]))
smith.absolute(smith.tensor([-1, -2, 3]))

# acos/arccos
smith.acos(a)
smith.arccos(a)

# acosh/arccosh
smith.acosh(a.uniform_(1, 2))

# add
smith.add(a, 20)
smith.add(a, smith.randn(4, 1), alpha=10)
smith.add(a + 1j, 20 + 1j)
smith.add(a + 1j, 20, alpha=1j)

# addcdiv
smith.addcdiv(smith.randn(1, 3), smith.randn(3, 1), smith.randn(1, 3), value=0.1)

# addcmul
smith.addcmul(smith.randn(1, 3), smith.randn(3, 1), smith.randn(1, 3), value=0.1)

# angle
smith.angle(smith.tensor([-1 + 1j, -2 + 2j, 3 - 3j])) * 180 / 3.14159

# asin/arcsin
smith.asin(a)
smith.arcsin(a)

# asinh/arcsinh
smith.asinh(a)
smith.arcsinh(a)

# atan/arctan
smith.atan(a)
smith.arctan(a)

# atanh/arctanh
smith.atanh(a.uniform_(-1, 1))
smith.arctanh(a.uniform_(-1, 1))

# atan2
smith.atan2(a, a)

# bitwise_not
smith.bitwise_not(t)

# bitwise_and
smith.bitwise_and(t, smith.tensor([1, 0, 3], dtype=smith.int8))
smith.bitwise_and(smith.tensor([True, True, False]), smith.tensor([False, True, False]))

# bitwise_or
smith.bitwise_or(t, smith.tensor([1, 0, 3], dtype=smith.int8))
smith.bitwise_or(smith.tensor([True, True, False]), smith.tensor([False, True, False]))

# bitwise_xor
smith.bitwise_xor(t, smith.tensor([1, 0, 3], dtype=smith.int8))

# ceil
smith.ceil(a)

# clamp/clip
smith.clamp(a, min=-0.5, max=0.5)
smith.clamp(a, min=0.5)
smith.clamp(a, max=0.5)
smith.clip(a, min=-0.5, max=0.5)

# conj
smith.conj(smith.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))

# copysign
smith.copysign(a, 1)
smith.copysign(a, b)

# cos
smith.cos(a)

# cosh
smith.cosh(a)

# deg2rad
smith.deg2rad(smith.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]]))

# div/divide/true_divide
x = smith.tensor([0.3810, 1.2774, -0.2972, -0.3719, 0.4637])
smith.div(x, 0.5)
p = smith.tensor(
    [
        [-0.3711, -1.9353, -0.4605, -0.2917],
        [0.1815, -1.0111, 0.9805, -1.5923],
        [0.1062, 1.4581, 0.7759, -1.2344],
        [-0.1830, -0.0313, 1.1908, -1.4757],
    ]
)
q = smith.tensor([0.8032, 0.2930, -0.8113, -0.2308])
smith.div(p, q)
smith.divide(p, q, rounding_mode="trunc")
smith.divide(p, q, rounding_mode="floor")

# digamma
smith.digamma(smith.tensor([1, 0.5]))

# erf
smith.erf(smith.tensor([0, -1.0, 10.0]))

# erfc
smith.erfc(smith.tensor([0, -1.0, 10.0]))

# erfinv
smith.erfinv(smith.tensor([0, 0.5, -1.0]))

# exp
smith.exp(smith.tensor([0, math.log(2.0)]))

# exp2
smith.exp2(smith.tensor([0, math.log2(2.0), 3, 4]))

# expm1
smith.expm1(smith.tensor([0, math.log(2.0)]))

# fake_quantize_per_channel_affine
x = smith.randn(2, 2, 2)
scales = (smith.randn(2) + 1) * 0.05
zero_points = smith.zeros(2).to(smith.long)
smith.fake_quantize_per_channel_affine(x, scales, zero_points, 1, 0, 255)

# fake_quantize_per_tensor_affine
smith.fake_quantize_per_tensor_affine(a, 0.1, 0, 0, 255)

# float_power
smith.float_power(smith.randint(10, (4,)), 2)
smith.float_power(smith.arange(1, 5), smith.tensor([2, -3, 4, -5]))

# floor
smith.floor(a)

# floor_divide
smith.floor_divide(smith.tensor([4.0, 3.0]), smith.tensor([2.0, 2.0]))
smith.floor_divide(smith.tensor([4.0, 3.0]), 1.4)

# fmod
smith.fmod(smith.tensor([-3.0, -2, -1, 1, 2, 3]), 2)
smith.fmod(smith.tensor([1, 2, 3, 4, 5]), 1.5)

# frac
smith.frac(smith.tensor([1, 2.5, -3.2]))

# imag
smith.randn(4, dtype=smith.cfloat).imag

# ldexp
smith.ldexp(smith.tensor([1.0]), smith.tensor([1]))
smith.ldexp(smith.tensor([1.0]), smith.tensor([1, 2, 3, 4]))

# lerp
start = smith.arange(1.0, 5.0)
end = smith.empty(4).fill_(10)
smith.lerp(start, end, 0.5)
smith.lerp(start, end, smith.full_like(start, 0.5))

# lgamma
smith.lgamma(smith.arange(0.5, 2, 0.5))

# log
smith.log(smith.arange(5) + 10)

# log10
smith.log10(smith.rand(5))

# log1p
smith.log1p(smith.randn(5))

# log2
smith.log2(smith.rand(5))

# logaddexp
smith.logaddexp(smith.tensor([-1.0]), smith.tensor([-1.0, -2, -3]))
smith.logaddexp(smith.tensor([-100.0, -200, -300]), smith.tensor([-1.0, -2, -3]))
smith.logaddexp(smith.tensor([1.0, 2000, 30000]), smith.tensor([-1.0, -2, -3]))

# logaddexp2
smith.logaddexp2(smith.tensor([-1.0]), smith.tensor([-1.0, -2, -3]))
smith.logaddexp2(smith.tensor([-100.0, -200, -300]), smith.tensor([-1.0, -2, -3]))
smith.logaddexp2(smith.tensor([1.0, 2000, 30000]), smith.tensor([-1.0, -2, -3]))

# logical_and
smith.logical_and(smith.tensor([True, False, True]), smith.tensor([True, False, False]))
r = smith.tensor([0, 1, 10, 0], dtype=smith.int8)
s = smith.tensor([4, 0, 1, 0], dtype=smith.int8)
smith.logical_and(r, s)
smith.logical_and(r.double(), s.double())
smith.logical_and(r.double(), s)
smith.logical_and(r, s, out=smith.empty(4, dtype=smith.bool))

# logical_not
smith.logical_not(smith.tensor([True, False]))
smith.logical_not(smith.tensor([0, 1, -10], dtype=smith.int8))
smith.logical_not(smith.tensor([0.0, 1.5, -10.0], dtype=smith.double))
smith.logical_not(
    smith.tensor([0.0, 1.0, -10.0], dtype=smith.double),
    out=smith.empty(3, dtype=smith.int16),
)

# logical_or
smith.logical_or(smith.tensor([True, False, True]), smith.tensor([True, False, False]))
smith.logical_or(r, s)
smith.logical_or(r.double(), s.double())
smith.logical_or(r.double(), s)
smith.logical_or(r, s, out=smith.empty(4, dtype=smith.bool))

# logical_xor
smith.logical_xor(smith.tensor([True, False, True]), smith.tensor([True, False, False]))
smith.logical_xor(r, s)
smith.logical_xor(r.double(), s.double())
smith.logical_xor(r.double(), s)
smith.logical_xor(r, s, out=smith.empty(4, dtype=smith.bool))

# logit
smith.logit(smith.rand(5), eps=1e-6)

# hypot
smith.hypot(smith.tensor([4.0]), smith.tensor([3.0, 4.0, 5.0]))

# i0
smith.i0(smith.arange(5, dtype=smith.float32))

# igamma/igammac
a1 = smith.tensor([4.0])
a2 = smith.tensor([3.0, 4.0, 5.0])
smith.igamma(a1, a2)
smith.igammac(a1, a2)

# mul/multiply
smith.mul(smith.randn(3), 100)
smith.multiply(smith.randn(4, 1), smith.randn(1, 4))
smith.mul(smith.randn(3) + 1j, 100 + 1j)

# mvlgamma
smith.mvlgamma(smith.empty(2, 3).uniform_(1, 2), 2)

# nan_to_num
w = smith.tensor([float("nan"), float("inf"), -float("inf"), 3.14])
smith.nan_to_num(x)
smith.nan_to_num(x, nan=2.0)
smith.nan_to_num(x, nan=2.0, posinf=1.0)

# neg/negative
smith.neg(smith.randn(5))

# nextafter
eps = smith.finfo(smith.float32).eps
smith.nextafter(smith.tensor([1, 2]), smith.tensor([2, 1])) == smith.tensor(
    [eps + 1, 2 - eps]
)

# polygamma
smith.polygamma(1, smith.tensor([1, 0.5]))
smith.polygamma(2, smith.tensor([1, 0.5]))
smith.polygamma(3, smith.tensor([1, 0.5]))
smith.polygamma(4, smith.tensor([1, 0.5]))

# pow
smith.pow(a, 2)
smith.pow(smith.arange(1.0, 5.0), smith.arange(1.0, 5.0))

# rad2deg
smith.rad2deg(smith.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]]))

# real
smith.randn(4, dtype=smith.cfloat).real

# reciprocal
smith.reciprocal(a)

# remainder
smith.remainder(smith.tensor([-3.0, -2, -1, 1, 2, 3]), 2)
smith.remainder(smith.tensor([1, 2, 3, 4, 5]), 1.5)

# round
smith.round(a)

# rsqrt
smith.rsqrt(a)

# sigmoid
smith.sigmoid(a)

# sign
smith.sign(smith.tensor([0.7, -1.2, 0.0, 2.3]))

# sgn
smith.tensor([3 + 4j, 7 - 24j, 0, 1 + 2j]).sgn()

# signbit
smith.signbit(smith.tensor([0.7, -1.2, 0.0, 2.3]))

# sin
smith.sin(a)

# sinc
smith.sinc(a)

# sinh
smith.sinh(a)

# sqrt
smith.sqrt(a)

# square
smith.square(a)

# sub/subtract
smith.sub(smith.tensor((1, 2)), smith.tensor((0, 1)), alpha=2)
smith.sub(smith.tensor((1j, 2j)), 1j, alpha=2)
smith.sub(smith.tensor((1j, 2j)), 10, alpha=2j)

# tan
smith.tan(a)

# tanh
smith.tanh(a)

# trunc/fix
smith.trunc(a)

# xlogy
f = smith.zeros(
    5,
)
g = smith.tensor([-1, 0, 1, float("inf"), float("nan")])
smith.xlogy(f, g)

f = smith.tensor([1, 2, 3])
g = smith.tensor([3, 2, 1])
smith.xlogy(f, g)
smith.xlogy(f, 4)
smith.xlogy(2, g)
