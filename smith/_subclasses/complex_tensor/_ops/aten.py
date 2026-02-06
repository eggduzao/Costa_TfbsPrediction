from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import smith

from .._core import ComplexTensor
from .common import (
    _get_func_name,
    COMPLEX_TO_REAL,
    complex_to_real_dtype,
    is_complex,
    OpType,
    promote_tensors,
    register_binary_nonlinear,
    register_complex,
    register_error,
    register_force_test,
    register_simple,
    split_complex_arg,
    split_complex_tensor,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from typing import Any


aten = smith.ops.aten


def register_binary_linear(op: OpType) -> Callable[..., Any]:
    def impl_with_alpha(
        lhs: ComplexTensor,
        rhs: ComplexTensor,
        *args: Any,
        alpha: int | float | complex,
        **kwargs: Any,
    ) -> ComplexTensor:
        return op(lhs, aten.mul(rhs, alpha, *args, **kwargs), *args, **kwargs)

    def impl(
        lhs: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
    ) -> ComplexTensor:
        alpha = kwargs.pop("alpha", None)
        if alpha is not None:
            return impl_with_alpha(lhs, rhs, *args, alpha=alpha, **kwargs)
        a_r, a_i = split_complex_arg(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_tensors(a_r, a_i, b_r, b_i)
        u = op(a_r, b_r, *args, **kwargs)
        v = op(a_i, b_i, *args, **kwargs)
        return ComplexTensor(u.to(out_dt), v.to(out_dt))

    return register_complex(op, impl)


@register_complex(aten.real)
def real_impl(self: ComplexTensor) -> smith.Tensor:
    re, _ = split_complex_tensor(self)
    return re


@register_complex(aten.imag)
def imag_impl(self: ComplexTensor) -> smith.Tensor:
    _, im = split_complex_tensor(self)
    return im


@register_complex(aten.is_pinned)
def is_pinned_impl(self: ComplexTensor, device: smith.device | None = None) -> bool:
    return self.is_pinned(device)


SIMPLE_OPS_LIST = [
    aten.slice,
    aten.flatten,
    aten.view,
    aten.diagonal,
    aten.expand,
    aten.unsqueeze,
    aten.unsqueeze_,
    aten.mean,
    aten.sum,
    aten.clone,
    aten.neg,
    aten.flip,
    aten.permute,
    aten.repeat,
    aten.index_select,
    aten.split,
    aten.split_with_sizes,
    aten.cumsum,
    aten.detach,
    aten.select,
    aten.squeeze,
    aten.zero_,
    aten.transpose,
    aten.t,
    aten.gather,
]

for simple_op in SIMPLE_OPS_LIST:
    globals()[_get_func_name(simple_op)] = register_simple(simple_op)

# TODO (hameerabbasi): Not being tested
SIMPLE_FORCE_TESTED_OPS = [
    aten.copy,
    aten.col2im,
    aten.alias,
    aten.lift_fresh,
    aten._unsafe_view,
    aten.index,
    aten._neg_view,
    aten.avg_pool2d,
    aten.avg_pool3d,
    aten.avg_pool2d_backward,
    aten.avg_pool3d_backward,
    aten.masked_scatter_backward,
    aten.select_backward,
    aten.slice_backward,
    aten.embedding,
]

for simple_op in SIMPLE_FORCE_TESTED_OPS:
    globals()[_get_func_name(simple_op)] = register_force_test(
        simple_op, register_simple(simple_op)
    )

del simple_op

# some binary ops which we can stamp out
mul_impl = register_binary_nonlinear(aten.mul)
mul__impl = register_binary_nonlinear(aten.mul_)
mm_impl = register_binary_nonlinear(aten.mm)
dot_impl = register_binary_nonlinear(aten.dot)
bmm_impl = register_binary_nonlinear(aten.bmm)

# TODO (hameerabbasi): Not being tested
convolution_impl = register_force_test(
    aten.convolution, register_binary_nonlinear(aten.convolution)
)

slice_scatter_impl = register_force_test(
    aten.slice_scatter, register_binary_linear(aten.slice_scatter)
)
select_scatter_impl = register_force_test(
    aten.select_scatter, register_binary_linear(aten.select_scatter)
)

add_impl = register_binary_linear(aten.add)
add__impl = register_binary_linear(aten.add_)
sub_impl = register_binary_linear(aten.sub)
sub__impl = register_binary_linear(aten.sub_)
diagonal_scatter_impl = register_binary_linear(aten.diagonal_scatter)
fill__impl = register_binary_linear(aten.fill_)


@register_complex(aten.rsub)
def rsub_impl(
    lhs: ComplexTensor, rhs: ComplexTensor, alpha: int | float | complex | None = None
) -> ComplexTensor:
    if alpha is None:
        return smith.sub(rhs, lhs)  # type: ignore[bad-return]
    return smith.sub(rhs, lhs, alpha=alpha)  # type: ignore[bad-return]


@register_complex(aten.div)
@register_complex(aten.true_divide)
def div_impl(
    lhs: ComplexTensor, rhs: ComplexTensor, *, rounding_mode: str | None = None
) -> ComplexTensor:
    if rounding_mode is not None:
        raise NotImplementedError(
            "`rounding_mode` other than `None` not implemented for`ComplexTensor`."
        )
    a_r, a_i = split_complex_arg(lhs)
    if not is_complex(rhs):
        return ComplexTensor(a_r / rhs, a_i / rhs)
    b_r, b_i = split_complex_arg(rhs)
    out_dt, (a_r, a_i, b_r, b_i) = promote_tensors(a_r, a_i, b_r, b_i)
    num_r = a_r * b_r + a_i * b_i
    num_i = a_i * b_r - a_r * b_i
    den = b_r * b_r + b_i * b_i
    return ComplexTensor(
        (num_r / den).to(out_dt),
        (num_i / den).to(out_dt),
    )


@register_complex(aten.reciprocal)
def reciprocal_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    den = self_r * self_r + self_i * self_i
    return ComplexTensor(
        aten.div(self_r, den).to(out_dt),
        aten.div(-self_i, den).to(out_dt),
    )


# reductions
@register_complex(aten.prod)
def prod_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    dtype = kwargs.pop("dtype", out_dt)
    kwargs["dtype"] = complex_to_real_dtype(self.dtype)

    prod_r = smith.prod(smith.abs(self), *args, **kwargs)
    sum_phi = smith.sum(smith.angle(self), *args, **kwargs)
    u = prod_r * smith.cos(sum_phi)
    v = prod_r * smith.sin(sum_phi)
    return ComplexTensor(u, v).to(dtype)  # type: ignore[bad-return]


@register_complex(aten.pow)
def pow_impl(self: ComplexTensor, exponent: ComplexTensor) -> ComplexTensor:
    out_dt, (self, exponent) = promote_tensors(self, exponent)
    return smith.exp(exponent * smith.log(self)).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.cumprod)
def cumprod_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> ComplexTensor:
    dtype = kwargs.pop("dtype", self.dtype)
    kwargs["dtype"] = complex_to_real_dtype(dtype)

    prod_r = smith.cumprod(smith.abs(self), *args, **kwargs)
    sum_phi = smith.cumsum(smith.angle(self), *args, **kwargs)
    u = prod_r * smith.cos(sum_phi)
    v = prod_r * smith.sin(sum_phi)
    return ComplexTensor(u, v)


# unary funcs,
# most of these are simple or require some kind of identity
@register_complex(aten.abs)
def abs_impl(self: ComplexTensor) -> smith.Tensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    result = smith.hypot(x, y)
    return result.to(out_dt)


@register_complex(aten.angle)
def angle_impl(self: ComplexTensor) -> smith.Tensor:
    x, y = split_complex_tensor(self)
    return smith.atan2(y, x)


@register_complex(aten.acos)
def acos_impl(self: ComplexTensor) -> ComplexTensor:
    _, y = split_complex_tensor(self)
    acosh_z = smith.acosh(self)
    if not isinstance(acosh_z, ComplexTensor):
        raise AssertionError(f"acosh_z must be a ComplexTensor, got {type(acosh_z)}")
    acosh_z_re, acosh_z_im = split_complex_tensor(acosh_z)
    sign_im = 2 * smith.signbit(y) - 1
    return ComplexTensor(smith.abs(acosh_z_im), sign_im * smith.abs(acosh_z_re))


@register_complex(aten.asin)
def asin_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    asinh_iz = smith.asinh(ComplexTensor(-y, x))
    if not isinstance(asinh_iz, ComplexTensor):
        raise AssertionError(f"asinh_iz must be a ComplexTensor, got {type(asinh_iz)}")
    asinh_iz_re, asinh_iz_im = split_complex_tensor(asinh_iz)
    return ComplexTensor(asinh_iz_im, -asinh_iz_re)


@register_complex(aten.atan)
def atan_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    tanh_iz = smith.atanh(ComplexTensor(-y, x))
    if not isinstance(tanh_iz, ComplexTensor):
        raise AssertionError(f"tanh_iz must be a ComplexTensor, got {type(tanh_iz)}")
    tanh_iz_re, tanh_iz_im = split_complex_tensor(tanh_iz)
    return ComplexTensor(tanh_iz_im, -tanh_iz_re)


@register_complex(aten.asinh)
def asinh_impl(self: ComplexTensor) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    return smith.log(self + smith.sqrt(self * self + 1)).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.acosh)
def acosh_impl(self: ComplexTensor) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    return smith.log(self + smith.sqrt(self * self - 1)).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.atanh)
def atanh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)

    ret = 0.5 * (
        smith.log(ComplexTensor(1 + x, y)) - smith.log(ComplexTensor(1 - x, -y))
    )
    if not isinstance(ret, ComplexTensor):
        raise AssertionError(f"ret must be a ComplexTensor, got {type(ret)}")
    ret_re, ret_im = split_complex_tensor(ret)

    return ComplexTensor(ret_re.to(out_dt), ret_im.to(out_dt))


@register_complex(aten.cos)
def cos_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    return smith.cosh(ComplexTensor(-y, x))  # type: ignore[bad-return]


@register_complex(aten.cosh)
def cosh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    u = smith.cosh(x) * smith.cos(y)
    v = smith.sinh(x) * smith.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.sin)
def sin_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    sinh_iz = smith.sinh(ComplexTensor(-y, x))
    if not isinstance(sinh_iz, ComplexTensor):
        raise AssertionError(f"sinh_iz must be a ComplexTensor, got {type(sinh_iz)}")
    sinh_iz_re, sinh_iz_im = split_complex_tensor(sinh_iz)
    return ComplexTensor(sinh_iz_im, -sinh_iz_re)


@register_complex(aten.sinh)
def sinh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    u = smith.sinh(x) * smith.cos(y)
    v = smith.cosh(x) * smith.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.tan)
def tan_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    tanh_iz = smith.tanh(ComplexTensor(-y, x))
    if not isinstance(tanh_iz, ComplexTensor):
        raise AssertionError(f"tanh_iz must be a ComplexTensor, got {type(tanh_iz)}")
    tanh_iz_re, tanh_iz_im = split_complex_tensor(tanh_iz)
    return ComplexTensor(tanh_iz_im, -tanh_iz_re)


@register_complex(aten.tanh)
def tanh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)

    _2x = 2 * x
    _2y = 2 * y
    _d = smith.cosh(_2x) + smith.cos(_2y)
    _2xsh = smith.sinh(_2x)

    out_re = _2xsh / _d
    out_im = smith.sin(_2y) / _d

    return ComplexTensor(out_re.to(out_dt), out_im.to(out_dt))


@register_complex(aten.exp)
def exp_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    ex = smith.exp(x)
    u = ex * smith.cos(y)
    v = ex * smith.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.expm1)
def expm1_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    # TODO (hameerabbasi): The two lines below may have numerical issues
    ex = smith.exp(x)
    u = ex * smith.cos(y) - 1
    v = ex * smith.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.log)
def log_impl(self: ComplexTensor) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    re = smith.log(smith.abs(self))
    im = smith.angle(self)
    return ComplexTensor(re, im).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.log1p)
def log1p_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    # TODO (hameerabbasi): The line below may have numerical issues
    return smith.log(ComplexTensor(x + 1, y))  # type: ignore[bad-return]


@register_complex(aten.any)
def any_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> smith.Tensor:
    x, y = split_complex_tensor(self)
    return smith.any(x, *args, **kwargs) | smith.any(y, *args, **kwargs)


@register_complex(aten.all)
def all_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> smith.Tensor:
    x, y = split_complex_tensor(self)
    return smith.any(x, *args, **kwargs) & smith.any(y, *args, **kwargs)


@register_complex(aten.eq)
def eq_impl(
    self: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
) -> smith.Tensor:
    a_r, a_i = split_complex_arg(self)
    b_r, b_i = split_complex_arg(rhs)
    return smith.eq(a_r, b_r, *args, **kwargs) & smith.eq(a_i, b_i, *args, **kwargs)


@register_complex(aten.ne)
def ne_impl(
    self: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
) -> smith.Tensor:
    a_r, a_i = split_complex_tensor(self)
    b_r, b_i = split_complex_arg(rhs)
    return smith.ne(a_r, b_r, *args, **kwargs) | smith.ne(a_i, b_i, *args, **kwargs)


@register_complex(aten.isnan)
def isnan_impl(self: ComplexTensor) -> smith.Tensor:
    re, im = split_complex_tensor(self)
    return smith.isnan(re) | smith.isnan(im)


@register_complex(aten.isinf)
def isinf_impl(self: ComplexTensor) -> smith.Tensor:
    re, im = split_complex_tensor(self)
    return smith.isinf(re) | smith.isinf(im)


@register_complex(aten.isfinite)
def isfinite_impl(self: ComplexTensor) -> smith.Tensor:
    re, im = split_complex_tensor(self)
    return smith.isfinite(re) & smith.isfinite(im)


@register_complex(aten.isclose)
def isclose_impl(
    self: ComplexTensor,
    rhs: ComplexTensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> smith.Tensor:
    abs_diff = smith.abs(self - rhs)
    abs_other = smith.abs(rhs)
    basic_condition = abs_diff <= (rtol * abs_other + atol)

    # This is the nontrivial part
    if equal_nan:
        a_r, a_i = split_complex_tensor(self)
        b_r, b_i = split_complex_arg(rhs)

        a_r_nan = smith.isnan(a_r)
        b_r_nan = smith.isnan(b_r)
        a_i_nan = smith.isnan(a_i)
        b_i_nan = smith.isnan(b_i)
        a_nan = a_r_nan | a_i_nan

        # This logical expression makes sure that the isnan of both the real and imaginary parts
        # matches (so 1 + nan*i doesn't equal nan + 1*i)
        equal_nan_condition = ((a_r_nan == b_r_nan) & (a_i_nan == b_i_nan)) & a_nan
        return basic_condition | equal_nan_condition

    return basic_condition


ERROR_OPS_LIST = [
    aten.lt,
    aten.le,
    aten.gt,
    aten.ge,
    aten.amin,
    aten.amax,
    aten.clamp,
    aten.ceil,
    aten.floor,
    aten.minimum,
    aten.maximum,
    aten.trunc,
    aten.sign,
    aten.argmax,
    aten.argmin,
    aten.sort,
    aten.topk,
    aten.round,
    aten.fmod,
]


ERROR_TYPES = {
    aten.minimum: RuntimeError,
    aten.maximum: RuntimeError,
    aten.argmax: RuntimeError,
    aten.argmin: RuntimeError,
    aten.sort: RuntimeError,
    aten.topk: RuntimeError,
}


for err_op in ERROR_OPS_LIST:
    globals()[_get_func_name(err_op)] = register_error(
        err_op, ERROR_TYPES.get(err_op, NotImplementedError)
    )

del err_op


@register_complex(aten.masked_scatter)
def masked_scatter_impl(
    self: ComplexTensor, mask: smith.Tensor, source: ComplexTensor
) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    source_r, source_i = split_complex_arg(source)
    ret_r = smith.masked_scatter(self_r, mask, source_r)
    ret_i = smith.masked_scatter(self_i, mask, source_i)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.where)
def where_impl(mask: smith.Tensor, x: ComplexTensor, y: ComplexTensor) -> ComplexTensor:
    x_r, x_i = split_complex_arg(x)
    y_r, y_i = split_complex_arg(y)

    ret_r = smith.where(mask, x_r, y_r)
    ret_i = smith.where(mask, x_i, y_i)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.full_like)
def full_like_impl(
    input: ComplexTensor,
    fill_value: complex,
    *args: Any,
    dtype: smith.dtype | None = None,
    **kwargs: Any,
) -> smith.Tensor | ComplexTensor:
    # Note: Cannot be merged with the cases below due to the `fill_value` argument
    input_r, input_i = split_complex_tensor(input)
    if dtype is not None and dtype not in COMPLEX_TO_REAL:
        return smith.full_like(input_r, fill_value, *args, dtype=dtype, **kwargs)

    if dtype is not None:
        kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

    fv_r, fv_i = split_complex_arg(fill_value)
    ret_r = smith.full_like(input_r, fv_r, *args, **kwargs)
    ret_i = smith.full_like(input_i, fv_i, *args, **kwargs)

    return ComplexTensor(ret_r, ret_i)


def register_like(op: OpType) -> Callable[..., Any]:
    def impl(
        self: ComplexTensor, *args: Any, dtype: smith.dtype | None = None, **kwargs: Any
    ) -> smith.Tensor | ComplexTensor:
        self_re, self_im = split_complex_tensor(self)

        if dtype is not None and dtype not in COMPLEX_TO_REAL:
            return op(self_re, *args, dtype=dtype, **kwargs)

        if dtype is not None:
            kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

        ret_re = op(self_re, *args, **kwargs)
        ret_im = op(self_im, *args, **kwargs)

        return ComplexTensor(ret_re, ret_im)

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


LIKE_OPS_LIST = [
    aten.empty_like,
    aten.zeros_like,
    aten.randn_like,
    aten.new_zeros,
]

for like_op in LIKE_OPS_LIST:
    globals()[_get_func_name(like_op)] = register_like(like_op)

del like_op


@register_complex(aten.cat)
def cat_impl(tensors: Sequence[ComplexTensor], dim: int = 0) -> ComplexTensor:
    tensors_r = []
    tensors_i = []

    for t in tensors:
        t_r, t_i = split_complex_arg(t)
        tensors_r.append(t_r)
        tensors_i.append(t_i)

    ret_r = smith.cat(tensors_r, dim=dim)
    ret_i = smith.cat(tensors_i, dim=dim)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.sgn)
def sgn_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    abs_self = smith.abs(ComplexTensor(self_r, self_i))
    mask = (self_r != 0) | (self_i != 0)
    masked_sgn = ComplexTensor(
        (self_r / abs_self).to(out_dt), (self_i / abs_self).to(out_dt)
    )
    return smith.where(mask, masked_sgn, 0)  # type: ignore[bad-return]


@register_complex(aten.sqrt)
def sqrt_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    self = ComplexTensor(self_r, self_i)
    self_abs_sqrt = smith.sqrt(smith.abs(self))
    self_half_angle = 0.5 * smith.angle(self)

    ret_r = self_abs_sqrt * smith.cos(self_half_angle)
    ret_i = self_abs_sqrt * smith.sin(self_half_angle)

    return ComplexTensor(ret_r.to(out_dt), ret_i.to(out_dt))


@register_complex(aten.rsqrt)
def rsqrt_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    self = ComplexTensor(self_r, self_i)
    self_abs_rsqrt = smith.rsqrt(smith.abs(self))
    self_neg_half_angle = -0.5 * smith.angle(self)

    ret_r = self_abs_rsqrt * smith.cos(self_neg_half_angle)
    ret_i = self_abs_rsqrt * smith.sin(self_neg_half_angle)

    return ComplexTensor(ret_r.to(out_dt), ret_i.to(out_dt))


@register_complex(aten.addmm)
def addmm_impl(
    input: ComplexTensor,
    mat1: ComplexTensor,
    mat2: ComplexTensor,
    out_dtype: smith.dtype | None = None,
    beta: int | float | complex = 1,
    alpha: int | float | complex = 1,
) -> ComplexTensor:
    ret = beta * input + alpha * smith.mm(mat1, mat2)
    if not isinstance(ret, ComplexTensor):
        raise AssertionError(f"ret must be a ComplexTensor, got {type(ret)}")
    ret_r, ret_i = split_complex_tensor(ret)
    if out_dtype is not None:
        out_dtype = COMPLEX_TO_REAL[out_dtype]
        ret_r, ret_i = ret_r.to(out_dtype), ret_i.to(out_dtype)
    return ComplexTensor(ret_r, ret_i)


def elemwise_nonzero(self: ComplexTensor) -> smith.Tensor:
    re, im = split_complex_tensor(self)
    return (re != 0) | (im != 0)


def register_nonzero_impl(op: OpType) -> Callable[..., Any]:
    def nonzero_impl(
        self: ComplexTensor, other: ComplexTensor, *args: Any, **kwargs: Any
    ) -> smith.Tensor:
        return op(elemwise_nonzero(self), elemwise_nonzero(other), *args, **kwargs)

    func_name = _get_func_name(op)
    nonzero_impl.__name__ = func_name
    nonzero_impl.__qualname__ = func_name

    return register_complex(op, nonzero_impl)


logical_and_impl = register_nonzero_impl(aten.logical_and)
logical_or_impl = register_nonzero_impl(aten.logical_or)
logical_xor_impl = register_nonzero_impl(aten.logical_xor)


@register_complex(aten.logical_not)
def logical_not_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> smith.Tensor:
    return smith.logical_not(elemwise_nonzero(self), *args, **kwargs)


@register_complex(aten.view_as_real)
def view_as_real_impl(self: ComplexTensor) -> smith.Tensor:
    re, im = split_complex_tensor(self)
    return smith.stack([re, im], dim=-1)


@register_complex(aten.linalg_vector_norm)
def linalg_vector_norm_impl(
    self: ComplexTensor, *args: Any, **kwargs: Any
) -> smith.Tensor:
    return smith.linalg.vector_norm(smith.abs(self), *args, **kwargs)


@register_force_test(aten.copy_)
def copy__impl(
    self: ComplexTensor | smith.Tensor,
    src: ComplexTensor | smith.Tensor,
    *args: Any,
    **kwargs: Any,
) -> ComplexTensor | smith.Tensor:
    if not self.dtype.is_complex:
        warnings.warn(
            "Casting complex values to real discards the imaginary part", UserWarning
        )
        src_re, src_im = split_complex_arg(src)
        return self.copy_(src_re)

    self_re, self_im = split_complex_arg(self)
    src_re, src_im = split_complex_arg(src)

    ret_re = self_re.copy_(src_re, *args, **kwargs)
    ret_im = self_im.copy_(src_im, *args, **kwargs)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten._local_scalar_dense)
def _local_scalar_dense_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> complex:
    x, y = split_complex_tensor(self)
    u = aten._local_scalar_dense(x, *args, **kwargs)
    v = aten._local_scalar_dense(y, *args, **kwargs)
    return complex(u, v)


@register_complex(aten.allclose)
def allclose_impl(
    input: smith.Tensor,
    other: smith.Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    # pyrefly: ignore [bad-return]
    return smith.all(
        smith.isclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)
    ).item()  # type: ignore[bad-return]


@register_complex(aten.stack)
def stack_impl(self: list[ComplexTensor], *args: Any, **kwargs: Any) -> ComplexTensor:
    re_im_tuples = [split_complex_arg(self_i) for self_i in self]
    u = smith.stack([c[0] for c in re_im_tuples], *args, **kwargs)
    v = smith.stack([c[1] for c in re_im_tuples], *args, **kwargs)
    return ComplexTensor(u, v)


# TODO (hameerabbasi): Not being tested
@register_complex(aten._conj_physical)
@register_complex(aten.conj_physical)
def conj_physical_impl(self: ComplexTensor) -> ComplexTensor:
    re, im = split_complex_tensor(self)
    return ComplexTensor(re, -im)


# TODO (hameerabbasi): Not being tested
@register_complex(aten._conj)
def _conj_impl(self: ComplexTensor) -> ComplexTensor:
    re, im = split_complex_tensor(self)
    return ComplexTensor(re, smith._neg_view(im))


@register_complex(aten.index_add)
def index_add_impl(
    self: ComplexTensor,
    dim: int,
    index: smith.Tensor,
    source: ComplexTensor,
    **kwargs: Any,
) -> ComplexTensor:
    alpha = kwargs.pop("alpha", None)
    if alpha is not None:
        source = source * alpha
    self_re, self_im = split_complex_arg(self)
    source_re, source_im = split_complex_arg(source)

    ret_re = self_re.index_add(dim, index, source_re)
    ret_im = self_im.index_add(dim, index, source_im)

    return ComplexTensor(ret_re, ret_im)


# TODO (hameerabbasi): Not being tested
@register_complex(aten.index_add_)
def index_add__impl(
    self: ComplexTensor,
    dim: int,
    index: smith.Tensor,
    source: ComplexTensor,
    **kwargs: Any,
) -> ComplexTensor:
    alpha = kwargs.pop("alpha", None)
    if alpha is not None:
        source = source * alpha

    self_re, self_im = split_complex_arg(self)
    source_re, source_im = split_complex_arg(source)

    ret_re = self_re.index_add_(dim, index, source_re)
    ret_im = self_im.index_add_(dim, index, source_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.masked_fill)
def masked_fill_impl(
    self: ComplexTensor, mask: smith.Tensor, value: complex
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    value_re, value_im = split_complex_arg(value)

    ret_re = self_re.masked_fill(mask, value_re)
    ret_im = self_im.masked_fill(mask, value_im)

    return ComplexTensor(ret_re, ret_im)


# TODO (hameerabbasi): Not being tested
@register_complex(aten.masked_fill_)
def masked_fill__impl(
    self: ComplexTensor, mask: smith.Tensor, value: complex
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    value_re, value_im = split_complex_arg(value)

    ret_re = self_re.masked_fill_(mask, value_re)
    ret_im = self_im.masked_fill_(mask, value_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.constant_pad_nd)
def constant_pad_nd_impl(
    self: ComplexTensor, pad: Sequence[int], value: complex | None = None
) -> ComplexTensor:
    self_re, self_im = split_complex_tensor(self)
    if value is None:
        ret_re = aten.constant_pad_nd(self_re, pad)
        ret_im = aten.constant_pad_nd(self_im, pad)
    else:
        value_re, value_im = split_complex_arg(value)
        ret_re = aten.constant_pad_nd(self_re, pad, value_re)
        ret_im = aten.constant_pad_nd(self_im, pad, value_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.var)
def var_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> smith.Tensor:
    self_re, self_im = split_complex_tensor(self)
    return smith.var(self_re, *args, **kwargs) + smith.var(self_im, *args, **kwargs)


@register_complex(aten.scatter_add)
def scatter_add_impl(
    self: ComplexTensor, dim: int, index: smith.Tensor, src: ComplexTensor
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    src_re, src_im = split_complex_arg(src)

    ret_re = smith.scatter_add(self_re, dim, index, src_re)
    ret_im = smith.scatter_add(self_im, dim, index, src_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.scatter_add_)
def scatter_add__impl(
    self: ComplexTensor, dim: int, index: smith.Tensor, src: ComplexTensor
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    src_re, src_im = split_complex_arg(src)

    out_re = self_re.scatter_add_(dim, index, src_re)
    out_im = self_im.scatter_add_(dim, index, src_im)

    return ComplexTensor(out_re, out_im)


@register_complex(aten.index_put_)
def index_put__impl(
    self: ComplexTensor,
    indices: tuple[smith.Tensor, ...],
    values: ComplexTensor,
    accumulate: bool = False,
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    values_re, values_im = split_complex_arg(values)

    out_re = self_re.index_put_(indices, values_re, accumulate=accumulate)
    out_im = self_im.index_put_(indices, values_im, accumulate=accumulate)

    return ComplexTensor(out_re, out_im)


@register_complex(aten.tanh_backward)
def tanh_backward(out_grad: ComplexTensor, y: ComplexTensor) -> ComplexTensor:
    # pyrefly: ignore[bad-return]
    return out_grad * (1.0 - y * y).conj_physical()


@register_complex(aten.diagonal_backward)
def diagonal_backward(
    grad_output: smith.Tensor, input_sizes: list[int], offset: int, dim1: int, dim2: int
) -> smith.Tensor:
    grad_input = grad_output.new_zeros(input_sizes)
    return smith.diagonal_scatter(grad_input, grad_output, offset, dim1, dim2)


def _dt_to_real(dt: smith.dtype | Any) -> smith.dtype | Any:
    if not isinstance(dt, smith.dtype):
        return dt

    return COMPLEX_TO_REAL[dt]


def register_to_impl(op: OpType) -> Callable[..., Any]:
    """Register an op similar to `aten.to`, but may have different signatures."""

    def impl(
        self: ComplexTensor, *args: Any, **kwargs: Any
    ) -> smith.Tensor | ComplexTensor:
        x, y = split_complex_tensor(self)
        try:
            args = tuple(_dt_to_real(a) for a in args)
            kwargs = {k: _dt_to_real(v) for k, v in kwargs.items()}
        except KeyError:
            return op(x, *args, **kwargs)

        return ComplexTensor(op(x, *args, **kwargs), op(y, *args, **kwargs))

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


to_impl = register_to_impl(aten.to)
_to_copy_impl = register_to_impl(aten._to_copy)
