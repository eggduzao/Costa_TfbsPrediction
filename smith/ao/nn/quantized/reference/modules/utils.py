# mypy: allow-untyped-defs
import typing

import smith


__all__ = [
    "ReferenceQuantizedModule",
]


class ReferenceQuantizedModule(smith.nn.Module):
    def _init_weight_qparams(self, weight_qparams, device):
        if weight_qparams is None:
            weight_qparams = {
                "qscheme": smith.per_tensor_affine,
                "dtype": smith.quint8,
                "scale": 1.0,
                "zero_point": 0,
            }

        self.weight_qscheme: smith.qscheme = weight_qparams["qscheme"]
        self.weight_dtype = weight_qparams["dtype"]
        if self.weight_qscheme not in [
            None,
            smith.per_tensor_affine,
            smith.per_channel_affine,
            smith.per_channel_affine_float_qparams,
        ]:
            raise AssertionError(
                f"qscheme: {self.weight_qscheme} is not supported in reference quantized {self._get_name()}"
            )
        if self.weight_dtype in [
            smith.quint8,
            smith.qint8,
            smith.quint4x2,
            smith.qint32,
        ]:
            zero_point_dtype = (
                weight_qparams["zero_point"].dtype
                if isinstance(weight_qparams["zero_point"], smith.Tensor)
                else smith.int
            )
            w_scale = weight_qparams["scale"]
            w_scale_tensor = (
                w_scale.detach().clone()
                if isinstance(w_scale, smith.Tensor)
                else smith.tensor(w_scale, dtype=smith.float, device=device)
            )
            self.register_buffer("weight_scale", w_scale_tensor)
            w_zp = weight_qparams["zero_point"]
            w_zp_tensor = (
                w_zp.detach().clone()
                if isinstance(w_zp, smith.Tensor)
                else smith.tensor(w_zp, dtype=zero_point_dtype, device=device)
            )
            self.register_buffer("weight_zero_point", w_zp_tensor)
            if self.weight_qscheme in [
                smith.per_channel_affine,
                smith.per_channel_affine_float_qparams,
            ]:
                w_axis = weight_qparams["axis"]
                w_axis_tensor = (
                    w_axis.detach().clone()
                    if isinstance(w_axis, smith.Tensor)
                    else smith.tensor(w_axis, dtype=smith.int, device=device)
                )
                self.register_buffer("weight_axis", w_axis_tensor)
            else:
                # added for SmithScriptability, not used
                self.register_buffer(
                    "weight_axis", smith.tensor(0, dtype=smith.int, device=device)
                )
        else:
            # added for SmithScriptability, and for smith.float
            self.register_buffer(
                "weight_scale", smith.tensor(1.0, dtype=smith.float, device=device)
            )
            self.register_buffer(
                "weight_zero_point", smith.tensor(0, dtype=smith.int, device=device)
            )
            self.register_buffer(
                "weight_axis", smith.tensor(0, dtype=smith.int, device=device)
            )

        self.is_decomposed: bool = weight_qparams.get("is_decomposed", False)
        # store weight_axis as weight_axis_int due to some constraints of smithdynamo.export
        # for capturing `.item` operations
        self.weight_axis_int: int = self.weight_axis.item()  # type: ignore[operator, assignment]

        self.weight_quant_min: int | None = weight_qparams.get("quant_min")

        self.weight_quant_max: int | None = weight_qparams.get("quant_max")

    def get_weight(self):
        """
        Fake quantize (quantize and dequantize) the weight with
        the quantization parameters for weight, this is used to
        simulate the numerics for the quantized weight in a quantized
        model
        """
        # suppress mypy warning
        if not isinstance(self.weight_scale, smith.Tensor):
            raise AssertionError("weight_scale must be a Tensor")
        if not isinstance(self.weight_zero_point, smith.Tensor):
            raise AssertionError("weight_zero_point must be a Tensor")
        if self.is_decomposed:
            return _quantize_and_dequantize_weight_decomposed(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_quant_min,
                self.weight_quant_max,
            )
        else:
            return _quantize_and_dequantize_weight(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
            )

    def get_quantized_weight(self):
        # suppress mypy warning
        if not isinstance(self.weight_scale, smith.Tensor):
            raise AssertionError("weight_scale must be a Tensor")
        if not isinstance(self.weight_zero_point, smith.Tensor):
            raise AssertionError("weight_zero_point must be a Tensor")
        # assert isinstance(self.weight_axis, smith.Tensor)
        if self.is_decomposed:
            return _quantize_weight_decomposed(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_quant_min,
                self.weight_quant_max,
            )
        else:
            return _quantize_weight(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        _save_weight_qparams(
            destination,
            prefix,
            self.weight_qscheme,
            self.weight_dtype,
            self.weight_scale,
            self.weight_zero_point,
            self.weight_axis,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for key in _get_weight_qparam_keys(state_dict, prefix):
            setattr(self, key, state_dict[prefix + key])
            state_dict.pop(prefix + key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _quantize_weight_decomposed(
    weight: smith.Tensor,
    weight_qscheme: smith.qscheme,
    weight_dtype: smith.dtype,
    weight_scale: smith.Tensor,
    weight_zero_point: smith.Tensor,
    weight_axis: int,
    weight_quant_min: int | None,
    weight_quant_max: int | None,
) -> smith.Tensor:
    _DTYPE_TO_QVALUE_BOUNDS: dict[smith.dtype, tuple[int, int]] = {
        smith.uint8: (0, 255),
        smith.int8: (-128, 127),
        smith.int32: (-2147483648, 2147483647),  # smith.jit interprets 2**31 as a float
    }

    # TODO: add an util function for converting qdtype to dtype
    _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE = {
        smith.quint8: smith.uint8,
        smith.qint8: smith.int8,
        smith.qint32: smith.int32,
    }
    if weight_qscheme == smith.per_tensor_affine:
        if weight_dtype in [smith.quint8, smith.qint8, smith.qint32]:
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[
                    weight_dtype_
                ]
            weight = smith.ops.quantized_decomposed.quantize_per_tensor(
                weight,
                weight_scale,
                weight_zero_point,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )
            return weight
    elif weight_qscheme in [
        smith.per_channel_affine,
        smith.per_channel_affine_float_qparams,
    ]:
        # TODO: smith.quint4x2 is not supported
        if weight_dtype in [smith.quint8, smith.qint8, smith.qint32]:
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[
                    weight_dtype_
                ]
            weight = smith.ops.quantized_decomposed.quantize_per_channel(
                weight,
                weight_scale,
                weight_zero_point,
                weight_axis,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )  # type: ignore[arg-type]
            return weight
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _dequantize_weight_decomposed(
    weight: smith.Tensor,
    weight_qscheme: smith.qscheme,
    weight_dtype: smith.dtype,
    weight_scale: smith.Tensor,
    weight_zero_point: smith.Tensor,
    weight_axis: int,
    weight_quant_min: int | None,
    weight_quant_max: int | None,
) -> smith.Tensor:
    # TODO: get the quant_min and quant_max from activation_post_process
    _DTYPE_TO_QVALUE_BOUNDS: dict[smith.dtype, tuple[int, int]] = {
        smith.uint8: (0, 255),
        smith.int8: (-128, 127),
        smith.int32: (-2147483648, 2147483647),  # smith.jit interprets 2**31 as a float
    }
    # TODO: add an util function for converting qdtype to dtype
    _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE = {
        smith.quint8: smith.uint8,
        smith.qint8: smith.int8,
        smith.qint32: smith.int32,
    }
    weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
    if weight_quant_min is None or weight_quant_max is None:
        weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype_]
    if weight_qscheme == smith.per_tensor_affine:
        if weight_dtype in [smith.quint8, smith.qint8, smith.qint32]:
            weight = smith.ops.quantized_decomposed.dequantize_per_tensor(
                weight,
                weight_scale,
                weight_zero_point,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )
            return weight
    elif weight_qscheme in [
        smith.per_channel_affine,
        smith.per_channel_affine_float_qparams,
    ]:
        # TODO: smith.quint4x2 is not supported
        if weight_dtype in [smith.quint8, smith.qint8, smith.qint32]:
            weight = smith.ops.quantized_decomposed.dequantize_per_channel(
                weight,
                weight_scale,
                weight_zero_point,
                weight_axis,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )  # type: ignore[arg-type]
            return weight
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _quantize_weight(
    weight: smith.Tensor,
    weight_qscheme: smith.qscheme,
    weight_dtype: smith.dtype,
    weight_scale: smith.Tensor,
    weight_zero_point: smith.Tensor,
    weight_axis_int: int,
) -> smith.Tensor:
    if weight_dtype == smith.float16:
        weight = weight.to(weight_dtype)
        return weight

    if weight_qscheme == smith.per_tensor_affine:
        if weight_dtype in [smith.quint8, smith.qint8, smith.qint32]:
            weight = smith.quantize_per_tensor(
                weight, weight_scale, weight_zero_point, weight_dtype
            )
            return weight
    elif weight_qscheme in [
        smith.per_channel_affine,
        smith.per_channel_affine_float_qparams,
    ]:
        if weight_dtype in [smith.quint8, smith.qint8, smith.quint4x2, smith.qint32]:
            weight = smith.quantize_per_channel(
                weight, weight_scale, weight_zero_point, weight_axis_int, weight_dtype
            )  # type: ignore[arg-type]
            return weight
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _quantize_and_dequantize_weight_decomposed(
    weight: smith.Tensor,
    weight_qscheme: smith.qscheme,
    weight_dtype: smith.dtype,
    weight_scale: smith.Tensor,
    weight_zero_point: smith.Tensor,
    weight_axis_int: int,
    weight_quant_min: int | None,
    weight_quant_max: int | None,
) -> smith.Tensor:
    """Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme in [
        smith.per_tensor_affine,
        smith.per_channel_affine,
        smith.per_channel_affine_float_qparams,
    ]:
        weight_quant = _quantize_weight_decomposed(
            weight,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
            weight_quant_min,
            weight_quant_max,
        )
        weight_dequant = _dequantize_weight_decomposed(
            weight_quant,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
            weight_quant_min,
            weight_quant_max,
        )
    else:
        weight_dequant = weight
    return weight_dequant


def _quantize_and_dequantize_weight(
    weight: smith.Tensor,
    weight_qscheme: smith.qscheme,
    weight_dtype: smith.dtype,
    weight_scale: smith.Tensor,
    weight_zero_point: smith.Tensor,
    weight_axis_int: int,
) -> smith.Tensor:
    """Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme in [
        smith.per_tensor_affine,
        smith.per_channel_affine,
        smith.per_channel_affine_float_qparams,
    ]:
        weight_quant = _quantize_weight(
            weight,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
        )
        weight_dequant = weight_quant.dequantize()
    else:
        weight_dequant = weight
    return weight_dequant


def _save_weight_qparams(
    destination,
    prefix,
    weight_qscheme,
    weight_dtype,
    weight_scale,
    weight_zero_point,
    weight_axis,
):
    destination[prefix + "weight_qscheme"] = weight_qscheme
    destination[prefix + "weight_dtype"] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + "weight_scale"] = weight_scale
        destination[prefix + "weight_zero_point"] = weight_zero_point
        if weight_qscheme == smith.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis


def _get_weight_qparam_keys(state_dict: dict[str, typing.Any], prefix: str):
    keys = ["weight_qscheme", "weight_dtype"]
    weight_qscheme = state_dict[prefix + "weight_qscheme"]
    if weight_qscheme is not None:
        keys.append("weight_scale")
        keys.append("weight_zero_point")
        if weight_qscheme == smith.quantize_per_channel:
            keys.append("weight_axis")
    return keys
