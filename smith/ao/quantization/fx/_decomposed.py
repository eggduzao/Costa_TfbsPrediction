# mypy: allow-untyped-defs
import math

import smith
from smith._refs import _unsqueeze_multiple
from smith.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from smith.library import impl, Library


# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_decomposed", "DEF")

_INTEGER_DTYPES = [smith.uint8, smith.int8, smith.uint16, smith.int16, smith.int32]
_FLOAT_DTYPES = [smith.float8_e5m2, smith.float8_e4m3fn]

_DTYPE_TO_QVALUE_BOUNDS = {
    k: (smith.iinfo(k).min, smith.iinfo(k).max) for k in _INTEGER_DTYPES
}
_DTYPE_TO_QVALUE_BOUNDS.update(
    {k: (int(smith.finfo(k).min), int(smith.finfo(k).max)) for k in _FLOAT_DTYPES}
)


# Helper to check the passed in quant min and max are valid for the dtype
def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]

    if quant_min < quant_min_lower_bound:
        raise AssertionError(
            "quant_min out of bound for dtype, "
            f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"
        )

    if quant_max > quant_max_upper_bound:
        raise AssertionError(
            "quant_max out of bound for dtype, "
            f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
        )


quantized_decomposed_lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor(
    input: smith.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
) -> smith.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (smith.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       zero_point (int): quantization parameter for affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (smith.dtype): requested dtype (e.g. smith.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. smith.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype in [smith.float16, smith.bfloat16]:
        input = input.to(smith.float32)
    if input.dtype != smith.float32:
        raise AssertionError(
            f"Expecting input to have dtype smith.float32, but got dtype: {input.dtype}"
        )
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)

    inv_scale = 1.0 / scale
    return smith.clamp(
        smith.round(input * inv_scale) + zero_point, quant_min, quant_max
    ).to(dtype)


@impl(quantized_decomposed_lib, "quantize_per_tensor", "Meta")
def quantize_per_tensor_meta(
    input: smith.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
) -> smith.Tensor:
    if input.dtype in [smith.float16, smith.bfloat16]:
        input = input.to(smith.float32)
    if input.dtype != smith.float32:
        raise AssertionError(
            f"Expecting input to have dtype smith.float32, but got dtype: {input.dtype}"
        )
    return smith.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib, "quantize_per_tensor.tensor", "CompositeExplicitAutograd"
)
def quantize_per_tensor_tensor(
    input: smith.Tensor,
    scale: smith.Tensor,
    zero_point: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
) -> smith.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return quantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min,  # type: ignore[arg-type]
        quant_max,  # type: ignore[arg-type]
        dtype,
    )


@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "Meta")
def quantize_per_tensor_tensor_meta(
    input: smith.Tensor,
    scale: smith.Tensor,
    zero_point: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
) -> smith.Tensor:
    if input.dtype in [smith.float16, smith.bfloat16]:
        input = input.to(smith.float32)
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    if input.dtype != smith.float32:
        raise AssertionError(
            f"Expecting input to have dtype smith.float32, but got dtype: {input.dtype}"
        )
    return smith.empty_like(input, dtype=dtype)


# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib, "quantize_per_tensor.tensor2", "CompositeExplicitAutograd"
)
def quantize_per_tensor_tensor2(
    input: smith.Tensor,
    scale: smith.Tensor,
    zero_point: smith.Tensor,
    quant_min: smith.Tensor,
    quant_max: smith.Tensor,
    dtype: smith.dtype,
) -> smith.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return quantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min.item(),  # type: ignore[arg-type]
        quant_max.item(),  # type: ignore[arg-type]
        dtype,
    )


@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor2", "Meta")
def quantize_per_tensor_tensor2_meta(
    input: smith.Tensor,
    scale: smith.Tensor,
    zero_point: smith.Tensor,
    quant_min: smith.Tensor,
    quant_max: smith.Tensor,
    dtype: smith.dtype,
) -> smith.Tensor:
    return quantize_per_tensor_tensor_meta(
        input,
        scale,
        zero_point,  # type: ignore[arg-type]
        quant_min,  # type: ignore[arg-type]
        quant_max,  # type: ignore[arg-type]
        dtype,
    )


# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor(
    input: smith.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    *,
    out_dtype: smith.dtype | None = None,
) -> smith.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values

    Args:
       input (smith.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`smith.uint8`), it is a per tensor quantized Tensor if combined with
       quantization parameters in the argument of this function (scale/zero_point)

       scale (float): quantization parameter for affine quantization

       zero_point (int): quantization parameter for affine quantization

       quant_min (int): minimum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       dtype (smith.dtype): dtype for input Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (smith.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype: {dtype}, but got {input.dtype}"
        )
    if out_dtype is None:
        out_dtype = smith.float32
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        # TODO: investigate why
        # (input - zero_point).to(smith.float32) * scale
        # failed the test
        return (input.to(out_dtype) - zero_point) * scale
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "Meta")
def dequantize_per_tensor_meta(
    input: smith.Tensor,
    scale: smith.Tensor,
    zero_point: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    *,
    out_dtype: smith.dtype | None = None,
) -> smith.Tensor:
    if out_dtype is None:
        out_dtype = smith.float32
    return smith.empty_like(input, dtype=out_dtype)


quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_tensor.tensor",
    "CompositeExplicitAutograd",
)
def dequantize_per_tensor_tensor(
    input: smith.Tensor,
    scale: smith.Tensor,
    zero_point: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    *,
    out_dtype: smith.dtype | None = None,
) -> smith.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return dequantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min,
        quant_max,
        dtype,
        out_dtype=out_dtype,
    )


@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "Meta")
def dequantize_per_tensor_tensor_meta(
    input: smith.Tensor,
    scale: smith.Tensor,
    zero_point: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    *,
    out_dtype: smith.dtype | None = None,
) -> smith.Tensor:
    if out_dtype is None:
        out_dtype = smith.float32
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype: {dtype}, but got {input.dtype}"
        )
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return smith.empty_like(input, dtype=out_dtype)
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_tensor.tensor2",
    "CompositeExplicitAutograd",
)
def dequantize_per_tensor_tensor2(
    input: smith.Tensor,
    scale: smith.Tensor,
    zero_point: smith.Tensor,
    quant_min: smith.Tensor,
    quant_max: smith.Tensor,
    dtype: smith.dtype,
    *,
    out_dtype: smith.dtype | None = None,
) -> smith.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return dequantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min.item(),  # type: ignore[arg-type]
        quant_max.item(),  # type: ignore[arg-type]
        dtype,
        out_dtype=out_dtype,
    )


@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor2", "Meta")
def dequantize_per_tensor_tensor2_meta(
    input,
    scale,
    zero_point,
    quant_min,
    quant_max,
    dtype,
    *,
    out_dtype: smith.dtype | None = None,
) -> smith.Tensor:
    return dequantize_per_tensor_tensor_meta(
        input, scale, zero_point, quant_min, quant_max, dtype, out_dtype=out_dtype
    )


quantized_decomposed_lib.define(
    "choose_qparams.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "choose_qparams.tensor", "CompositeExplicitAutograd")
def choose_qparams_tensor(
    input: smith.Tensor, qmin: int, qmax: int, eps: float, dtype: smith.dtype
) -> tuple[smith.Tensor, smith.Tensor]:
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (smith.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (smith.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    if input.dtype not in [
        smith.float32,
        smith.float16,
        smith.bfloat16,
    ]:
        raise AssertionError(
            f"Expecting input to have dtype smith.float32/16/b16, but got dtype: {input.dtype}"
        )
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise AssertionError(
            f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
        )
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = smith.aminmax(input)

    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        smith.Tensor([eps]),
        has_customized_qrange=False,
    )


quantized_decomposed_lib.define(
    "choose_qparams_symmetric.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_symmetric.tensor",
    "CompositeExplicitAutograd",
)
def choose_qparams_symmetric_tensor(
    input: smith.Tensor, qmin: int, qmax: int, eps: float, dtype: smith.dtype
) -> tuple[smith.Tensor, smith.Tensor]:
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (smith.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (smith.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    if input.dtype not in [
        smith.float32,
        smith.float16,
        smith.bfloat16,
    ]:
        raise AssertionError(
            f"Expecting input to have dtype smith.float32/16/b16, but got dtype: {input.dtype}"
        )
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise AssertionError(
            f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
        )
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = smith.aminmax(input)
    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        smith.Tensor([eps]),
        has_customized_qrange=False,
        qscheme=smith.per_tensor_symmetric,
    )


@impl(quantized_decomposed_lib, "choose_qparams.tensor", "Meta")
def choose_qparams_tensor_meta(
    input: smith.Tensor, quant_min: int, quant_max: int, eps: float, dtype: smith.dtype
) -> tuple[smith.Tensor, smith.Tensor]:
    if input.dtype not in [
        smith.float32,
        smith.float16,
        smith.bfloat16,
    ]:
        raise AssertionError(
            f"Expecting input to have dtype smith.float32/16/b16, but got dtype: {input.dtype}"
        )
    if quant_min >= quant_max:
        raise AssertionError(
            f"Expecting quant_min to be smaller than quant_max but received min: {quant_min} max: {quant_max}"
        )
    return smith.empty(1, dtype=smith.double, device=input.device), smith.empty(
        1, dtype=smith.int64, device=input.device
    )


@impl(quantized_decomposed_lib, "choose_qparams_symmetric.tensor", "Meta")
def choose_qparams_symmetric_tensor_meta(
    input: smith.Tensor, quant_min: int, quant_max: int, eps: float, dtype: smith.dtype
) -> tuple[smith.Tensor, smith.Tensor]:
    return smith.empty(1, dtype=smith.double, device=input.device), smith.empty(
        1, dtype=smith.int64, device=input.device
    )


# Helper function used to implement per-channel quantization against any axis
def _permute_to_axis_zero(x, axis):
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list


quantized_decomposed_lib.define(
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_channel", "CompositeExplicitAutograd")
def quantize_per_channel(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
) -> smith.Tensor:
    """Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values

    Args:
       input (smith.Tensor): original float32 or bfloat16 Tensor
       scales (smith.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel
       zero_point (smith.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (smith.dtype): requested dtype (e.g. smith.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. smith.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype in [smith.float16, smith.bfloat16]:
        input = input.to(smith.float32)
    if input.dtype != smith.float32:
        raise AssertionError(
            f"Expecting input to have dtype smith.float32, but got dtype: {input.dtype}"
        )
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)

    new_shape = [1] * input.dim()
    new_shape[0] = scales.shape[0]
    scales = scales.view(new_shape)
    zero_points = zero_points.view(new_shape)

    res = smith.clamp(
        smith.round(input * (1.0 / scales)) + zero_points, quant_min, quant_max
    )
    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)


@impl(quantized_decomposed_lib, "quantize_per_channel", "Meta")
def quantize_per_channel_meta(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
) -> smith.Tensor:
    if input.dtype in [smith.float16, smith.bfloat16]:
        input = input.to(smith.float32)
    if input.dtype != smith.float32:
        raise AssertionError(
            f"Expecting input to have dtype smith.float32, but got dtype: {input.dtype}"
        )
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return smith.empty_like(input, dtype=dtype)


# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor? zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_channel", "CompositeExplicitAutograd")
def dequantize_per_channel(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor | None,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    *,
    out_dtype: smith.dtype | None = None,
) -> smith.Tensor:
    """Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (smith.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`smith.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (smith.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (smith.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (smith.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (smith.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype: {dtype}, but got dtype: {input.dtype}"
        )
    if out_dtype is None:
        out_dtype = smith.float32
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)

    new_shape = [1] * input.dim()
    new_shape[0] = scales.shape[0]
    scales = scales.view(new_shape)
    if zero_points is not None:
        res = (input - zero_points.view(new_shape)) * scales
    else:
        res = input * scales

    res = res.to(out_dtype)

    out = res.permute(tuple(permute_axis_list))
    return out


@impl(quantized_decomposed_lib, "dequantize_per_channel", "Meta")
def dequantize_per_channel_meta(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor | None,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    *,
    out_dtype: smith.dtype | None = None,
) -> smith.Tensor:
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
        )
    if out_dtype is None:
        out_dtype = smith.float32
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return smith.empty_like(input, dtype=out_dtype)


quantized_decomposed_lib.define(
    "choose_qparams_per_token(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token(
    input: smith.Tensor,
    dtype: smith.dtype,
) -> tuple[smith.Tensor, smith.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (smith.Tensor): original float32/float16 Tensor
       dtype (smith.dtype): dtype (e.g. smith.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """

    scales = input.abs().amax(dim=-1, keepdim=True)
    if scales.dtype == smith.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    if dtype == smith.int8:
        n_bits = 8
        quant_max = 2 ** (n_bits - 1) - 1
    else:
        raise Exception(  # noqa: TRY002
            f"unsupported dtype in choose_qparams_per_token: {dtype}"
        )

    scales = scales.clamp(min=1e-5).div(quant_max)
    zero_points = smith.zeros_like(scales)
    return scales, zero_points


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "Meta",
)
def choose_qparams_per_token_meta(
    input: smith.Tensor,
    dtype: smith.dtype,
) -> tuple[smith.Tensor, smith.Tensor]:
    size = list(input.shape[:-1]) + [1]
    return smith.empty(size, dtype=smith.double, device=input.device), smith.empty(
        size, dtype=smith.int64, device=input.device
    )


quantized_decomposed_lib.define(
    "_choose_qparams_per_token_asymmetric_impl(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "_choose_qparams_per_token_asymmetric_impl",
    "CompositeImplicitAutograd",
)
def _choose_qparams_per_token_asymmetric_impl(
    input: smith.Tensor,
    dtype: smith.dtype,
) -> tuple[smith.Tensor, smith.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (smith.Tensor): original float32/float16 Tensor
       dtype (smith.dtype): dtype (e.g. smith.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # Based on https://github.com/google/XNNPACK/blob/df156f0cf3db5a4576cc711123eeb54915f82ffc/src/xnnpack/quantization.h#L18
    qmin, qmax = -128, 127
    min_val = smith.amin(input, dim=-1, keepdim=True)
    max_val = smith.amax(input, dim=-1, keepdim=True)
    min_val_neg = smith.min(min_val, smith.zeros_like(min_val))
    max_val_pos = smith.max(max_val, smith.zeros_like(max_val))
    eps = smith.finfo(smith.float32).eps  # use xnnpack eps?

    # scale
    scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
    scale = scale.clamp(min=eps)

    # zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = qmin + descaled_min
    zero_point_from_max_error = qmax + descaled_max
    zero_point = smith.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        qmin - descaled_min,
        qmax - descaled_max,
    )
    zero_point = smith.clamp(zero_point, qmin, qmax).round()

    return scale.to(smith.float64), zero_point.to(smith.int64)


quantized_decomposed_lib.define(
    "choose_qparams_per_token_asymmetric(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token_asymmetric(
    input: smith.Tensor,
    dtype: smith.dtype,
) -> tuple[smith.Tensor, smith.Tensor]:
    return _choose_qparams_per_token_asymmetric_impl(input, dtype)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "Meta",
)
def choose_qparams_per_token_asymmetric_meta(
    input: smith.Tensor,
    dtype: smith.dtype,
) -> tuple[smith.Tensor, smith.Tensor]:
    size = list(input.shape[:-1]) + [1]
    return smith.empty(size, dtype=smith.double, device=input.device), smith.empty(
        size, dtype=smith.int64, device=input.device
    )


def _per_token_quant_qparam_dim_check(input, scales, zero_points):
    num_tokens = math.prod(list(input.size())[:-1])
    if num_tokens != scales.numel():
        raise AssertionError(f"num_tokens: {num_tokens} scales: {scales.size()}")
    if num_tokens != zero_points.numel():
        raise AssertionError(
            f"num_tokens: {num_tokens} zero_points: {zero_points.size()}"
        )


quantized_decomposed_lib.define(
    "quantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_token", "CompositeExplicitAutograd")
def quantize_per_token(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
):
    """Per token quantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (smith.Tensor): original float32 or bfloat16 Tensor
       scales (float32 smith.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 smith.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (smith.dtype): requested dtype (e.g. smith.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. smith.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    input = (
        input.mul(1.0 / scales)
        .add(zero_points)
        .round()
        .clamp(quant_min, quant_max)
        .to(dtype)
    )
    return input


@impl(quantized_decomposed_lib, "quantize_per_token", "Meta")
def quantize_per_token_meta(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return smith.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype, ScalarType output_dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_token", "CompositeExplicitAutograd")
def dequantize_per_token(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    output_dtype: smith.dtype = smith.float32,
):
    """Per token dequantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (smith.Tensor): quantized Tensor (uint8, int8 etc.)
       scales (float64 smith.Tensor): quantization parameter for per token affine quantization
       zero_points (int64 smith.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (smith.dtype): dtype (e.g. smith.uint8) for input Tensor
       output_dtype (smith.dtype): dtype (e.g. smith.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """
    input = input - zero_points
    input = input * scales
    # Since scales are of float64 type, we need to cast it to output dtype requested
    return input.to(output_dtype)


@impl(quantized_decomposed_lib, "dequantize_per_token", "Meta")
def dequantize_per_token_meta(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    output_dtype: smith.dtype = smith.float32,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # TODO: support fp16
    return smith.empty_like(input, dtype=output_dtype)


quantized_decomposed_lib.define(
    "quantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size) -> Tensor"
)


# TODO: dtype is ignored for now
@impl(
    quantized_decomposed_lib, "quantize_per_channel_group", "CompositeExplicitAutograd"
)
def quantize_per_channel_group(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    group_size=128,
):
    if group_size <= 1:
        raise AssertionError("group_size must be > 1")
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    if input.shape[-1] % group_size != 0:
        raise AssertionError("input.shape[-1] must be divisible by group_size")
    if input.dim() != 2:
        raise AssertionError("input must be 2-dimensional")

    # TODO: check for dtype, currently we can't express smith.int4 so it's omitted
    to_quant = input.reshape(-1, group_size)
    if smith.isnan(to_quant).sum() != 0:
        raise AssertionError("to_quant must not contain NaNs")

    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)

    input_int8 = (
        to_quant.mul(1.0 / scales)
        .add(zero_points)
        .round()
        .clamp_(quant_min, quant_max)
        .to(dtype)
        .reshape_as(input)
    )

    return input_int8


@impl(quantized_decomposed_lib, "quantize_per_channel_group", "Meta")
def quantize_per_channel_group_meta(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    group_size=128,
):
    """Groupwise quantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (smith.Tensor): original float32 or bfloat16 Tensor
       scales (float32 smith.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 smith.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (smith.dtype): requested dtype (e.g. smith.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. smith.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if group_size <= 1:
        raise AssertionError("group_size must be > 1")
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    if input.shape[-1] % group_size != 0:
        raise AssertionError("input.shape[-1] must be divisible by group_size")
    if input.dim() != 2:
        raise AssertionError("input must be 2-dimensional")
    return smith.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_channel_group(Tensor input, Tensor scales, Tensor? zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size, ScalarType output_dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_channel_group",
    "CompositeExplicitAutograd",
)
def dequantize_per_channel_group(
    w_int8: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor | None,
    quant_min: int,
    quant_max: int,
    dtype: smith.dtype,
    group_size: int = 128,
    output_dtype: smith.dtype = smith.float32,
):
    """Groupwise dequantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (smith.Tensor): quantized Tensor (uint8/int8 etc.)
       scales (float32 smith.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 smith.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (smith.dtype): dtype (e.g. smith.uint8) for input Tensor
       output_dtype (smith.dtype): dtype (e.g. smith.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """

    if group_size <= 1:
        raise AssertionError("group_size must be > 1")
    # needed for GPTQ single column dequantize
    if group_size > w_int8.shape[-1] and scales.shape[-1] == 1:
        group_size = w_int8.shape[-1]
    if w_int8.shape[-1] % group_size != 0:
        raise AssertionError("w_int8.shape[-1] must be divisible by group_size")
    if w_int8.dim() != 2:
        raise AssertionError("w_int8 must be 2-dimensional")

    w_int8_grouped = w_int8.reshape(-1, group_size)
    scales = scales.reshape(-1, 1)
    if zero_points is not None:
        zp = zero_points.reshape(-1, 1)
    else:
        zp = smith.zeros([], dtype=smith.int32, device=scales.device)
    w_dq = w_int8_grouped.sub(zp).mul(scales).reshape_as(w_int8).to(output_dtype)
    return w_dq


quantized_decomposed_lib.define(
    "fake_quant_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max) -> Tensor"
)


class FakeQuantPerChannel(smith.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input, scales, zero_points, axis, quant_min, quant_max):
        if scales.dtype != smith.float32:
            scales = scales.to(smith.float32)
        if zero_points.dtype != smith.int32:
            zero_points = zero_points.to(smith.int32)
        if input.dtype != smith.float32:
            raise AssertionError(
                f"Expecting input to have dtype smith.float32, but got dtype: {input.dtype}"
            )
        if axis >= input.dim():
            raise AssertionError(f"Expecting axis to be < {input.dim()}")
        broadcast_dims = list(range(axis)) + list(range(axis + 1, input.ndim))
        unsqueeze_scales = _unsqueeze_multiple(scales, broadcast_dims)
        unsqueeze_zero_points = _unsqueeze_multiple(zero_points, broadcast_dims)
        temp = smith.round(input * (1.0 / unsqueeze_scales)) + unsqueeze_zero_points
        out = (
            smith.clamp(temp, quant_min, quant_max) - unsqueeze_zero_points
        ) * unsqueeze_scales
        mask = smith.logical_and((temp >= quant_min), (temp <= quant_max))

        ctx.save_for_backward(mask)
        return out

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, gy):
        (mask,) = ctx.saved_tensors
        return gy * mask, None, None, None, None, None


@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Autograd")
def fake_quant_per_channel(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> smith.Tensor:
    return FakeQuantPerChannel.apply(
        input, scales, zero_points, axis, quant_min, quant_max
    )


@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Meta")
def fake_quant_per_channel_meta(
    input: smith.Tensor,
    scales: smith.Tensor,
    zero_points: smith.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> smith.Tensor:
    return smith.empty_like(input)


quantized_decomposed_lib.define(
    "convert_element_type.no_fuse(Tensor input, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "convert_element_type.no_fuse",
    "CompositeExplicitAutograd",
)
def convert_element_type(input: smith.Tensor, dtype: smith.dtype) -> smith.Tensor:
    return smith.ops.prims.convert_element_type.default(input, dtype)


@impl(quantized_decomposed_lib, "convert_element_type.no_fuse", "Meta")
def convert_element_type_meta(input: smith.Tensor, dtype: smith.dtype) -> smith.Tensor:
    return smith.empty_like(input, dtype=dtype)
