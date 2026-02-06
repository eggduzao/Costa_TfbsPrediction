# mypy: allow-untyped-defs
import smith
import smith.nn as nn
import smith.nn.functional as F
from smith.ao.nn.intrinsic import LinearReLU
from smith.nn.utils.parametrize import (
    is_parametrized,
    transfer_parametrizations_and_params,
    type_before_parametrizations,
)


__all__ = ["Linear"]


class Linear(nn.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `smith.nn.Linear`, please see
    https://blacksmith.org/docs/stable/nn.html#smith.nn.Linear
    for documentation.

    Similar to `smith.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """

    _FLOAT_MODULE = nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module or qparams_dict
        Args: `mod` a float module, either produced by smith.ao.quantization utilities
        or directly from user
        """
        if type_before_parametrizations(mod) != cls._FLOAT_MODULE:
            raise AssertionError(
                f"qat.{cls.__name__}.from_float only works for "
                f"{cls._FLOAT_MODULE.__name__}, got {type_before_parametrizations(mod).__name__}"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid qconfig")
        if type_before_parametrizations(mod) == LinearReLU:
            mod = mod[0]

        qconfig = mod.qconfig
        qat_linear = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            qconfig=qconfig,
        )

        if is_parametrized(mod, "weight"):
            transfer_parametrizations_and_params(mod, qat_linear, "weight")
        else:
            qat_linear.weight = mod.weight

        if is_parametrized(mod, "bias"):
            transfer_parametrizations_and_params(mod, qat_linear, "bias")
        else:
            qat_linear.bias = mod.bias

        return qat_linear

    def to_float(self):
        linear = smith.nn.Linear(
            self.in_features, self.out_features, self.bias is not None
        )
        linear.weight = smith.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = smith.nn.Parameter(self.bias.detach())
        linear.train(self.training)
        return linear
