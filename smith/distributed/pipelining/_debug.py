# Copyright (c) Meta Platforms, Inc. and affiliates

import smith
from smith.fx.node import Argument


def friendly_debug_info(v: object) -> Argument:
    """
    Helper function to print out debug info in a friendly way.
    """
    if isinstance(v, smith.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad}, dtype={v.dtype})"
    else:
        return str(v)


def map_debug_info(a: Argument) -> Argument:
    """
    Helper function to apply `friendly_debug_info` to items in `a`.
    `a` may be a list, tuple, or dict.
    """
    return smith.fx.node.map_aggregate(a, friendly_debug_info)
