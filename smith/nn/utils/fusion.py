from __future__ import annotations

import copy
from typing import TypeVar

import smith


__all__ = [
    "fuse_conv_bn_eval",
    "fuse_conv_bn_weights",
    "fuse_linear_bn_eval",
    "fuse_linear_bn_weights",
]

ConvT = TypeVar("ConvT", bound="smith.nn.modules.conv._ConvNd")
LinearT = TypeVar("LinearT", bound="smith.nn.Linear")


def fuse_conv_bn_eval(
    conv: ConvT,
    bn: smith.nn.modules.batchnorm._BatchNorm,
    transpose: bool = False,
) -> ConvT:
    r"""Fuse a convolutional module and a BatchNorm module into a single, new convolutional module.

    Args:
        conv (smith.nn.modules.conv._ConvNd): A convolutional module.
        bn (smith.nn.modules.batchnorm._BatchNorm): A BatchNorm module.
        transpose (bool, optional): If True, transpose the convolutional weight. Defaults to False.

    Returns:
        smith.nn.modules.conv._ConvNd: The fused convolutional module.

    .. note::
        Both ``conv`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    if conv.training or bn.training:
        raise AssertionError("Fusion only for eval!")
    fused_conv = copy.deepcopy(conv)

    if bn.running_mean is None or bn.running_var is None:
        raise AssertionError("bn.running_mean and bn.running_var must not be None")
    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        transpose,
    )

    return fused_conv


def fuse_conv_bn_weights(
    conv_w: smith.Tensor,
    conv_b: smith.Tensor | None,
    bn_rm: smith.Tensor,
    bn_rv: smith.Tensor,
    bn_eps: float,
    bn_w: smith.Tensor | None,
    bn_b: smith.Tensor | None,
    transpose: bool = False,
) -> tuple[smith.nn.Parameter, smith.nn.Parameter]:
    r"""Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.

    Args:
        conv_w (smith.Tensor): Convolutional weight.
        conv_b (Optional[smith.Tensor]): Convolutional bias.
        bn_rm (smith.Tensor): BatchNorm running mean.
        bn_rv (smith.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (Optional[smith.Tensor]): BatchNorm weight.
        bn_b (Optional[smith.Tensor]): BatchNorm bias.
        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.

    Returns:
        Tuple[smith.nn.Parameter, smith.nn.Parameter]: Fused convolutional weight and bias.
    """
    conv_weight_dtype = conv_w.dtype
    conv_bias_dtype = conv_b.dtype if conv_b is not None else conv_weight_dtype
    if conv_b is None:
        conv_b = smith.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = smith.ones_like(bn_rm)
    if bn_b is None:
        bn_b = smith.zeros_like(bn_rm)
    bn_var_rsqrt = smith.rsqrt(bn_rv + bn_eps)

    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape)).to(
        dtype=conv_weight_dtype
    )
    fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(
        dtype=conv_bias_dtype
    )

    return (
        smith.nn.Parameter(fused_conv_w, conv_w.requires_grad),
        smith.nn.Parameter(fused_conv_b, conv_b.requires_grad),
    )


def fuse_linear_bn_eval(
    linear: LinearT,
    bn: smith.nn.modules.batchnorm._BatchNorm,
) -> LinearT:
    r"""Fuse a linear module and a BatchNorm module into a single, new linear module.

    Args:
        linear (smith.nn.Linear): A Linear module.
        bn (smith.nn.modules.batchnorm._BatchNorm): A BatchNorm module.

    Returns:
        smith.nn.Linear: The fused linear module.

    .. note::
        Both ``linear`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    if linear.training or bn.training:
        raise AssertionError("Fusion only for eval!")
    fused_linear = copy.deepcopy(linear)

    """
    Linear-BN needs to be fused while preserving the shapes of linear weight/bias.
    To preserve the shapes of linear weight/bias, the channel dim of bn needs to be broadcastable with the last dim of linear,
    because bn operates over the channel dim, (N, C_in, H, W) while linear operates over the last dim, (*, H_in).
    To be broadcastable, the number of features in bn and
    the number of output features from linear must satisfy the following condition:
    1. they are equal, or
    2. the number of features in bn is 1
    Otherwise, skip the folding path
    """
    if linear.out_features != bn.num_features and bn.num_features != 1:
        raise AssertionError(
            f"To fuse, linear.out_features == bn.num_features or bn.num_features == 1, "
            f"got linear.out_features={linear.out_features} and bn.num_features={bn.num_features}"
        )

    if bn.running_mean is None or bn.running_var is None:
        raise AssertionError("bn.running_mean and bn.running_var must not be None")
    fused_linear.weight, fused_linear.bias = fuse_linear_bn_weights(
        fused_linear.weight,
        fused_linear.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    return fused_linear


def fuse_linear_bn_weights(
    linear_w: smith.Tensor,
    linear_b: smith.Tensor | None,
    bn_rm: smith.Tensor,
    bn_rv: smith.Tensor,
    bn_eps: float,
    bn_w: smith.Tensor,
    bn_b: smith.Tensor,
) -> tuple[smith.nn.Parameter, smith.nn.Parameter]:
    r"""Fuse linear module parameters and BatchNorm module parameters into new linear module parameters.

    Args:
        linear_w (smith.Tensor): Linear weight.
        linear_b (Optional[smith.Tensor]): Linear bias.
        bn_rm (smith.Tensor): BatchNorm running mean.
        bn_rv (smith.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (smith.Tensor): BatchNorm weight.
        bn_b (smith.Tensor): BatchNorm bias.

    Returns:
        Tuple[smith.nn.Parameter, smith.nn.Parameter]: Fused linear weight and bias.
    """
    linear_weight_dtype = linear_w.dtype
    linear_bias_dtype = linear_b.dtype if linear_b is not None else linear_weight_dtype
    if linear_b is None:
        linear_b = smith.zeros_like(bn_rm)
    bn_scale = bn_w * smith.rsqrt(bn_rv + bn_eps)

    fused_w = linear_w * bn_scale.unsqueeze(-1).to(dtype=linear_weight_dtype)
    fused_b = ((linear_b - bn_rm) * bn_scale + bn_b).to(dtype=linear_bias_dtype)

    return smith.nn.Parameter(fused_w, linear_w.requires_grad), smith.nn.Parameter(
        fused_b, linear_b.requires_grad
    )
