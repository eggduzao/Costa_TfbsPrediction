# mypy: allow-untyped-defs
import abc
import collections
import itertools

import smith
from smith.nn.modules.module import _addindent


__all__ = [
    "WeightedQuantizedModule",
]


class WeightedQuantizedModule(smith.nn.Module, metaclass=abc.ABCMeta):
    """Wrapper for quantized modules than can be lowered from reference modules."""

    @classmethod
    @abc.abstractmethod
    def from_reference(cls, ref_module, output_scale, output_zero_point):
        raise NotImplementedError


def _get_weight_observer(observer):
    # FakeQuantize observer
    if hasattr(observer, "activation_post_process"):
        observer = observer.activation_post_process
    # UniformQuantizationObserverBase observer
    return observer


def _needs_weight_clamping(observer, dtype):
    observer = _get_weight_observer(observer)
    if dtype in [smith.qint8, smith.quint8, smith.qint32]:
        info = smith.iinfo(dtype)
        return observer.quant_min > info.min or observer.quant_max < info.max
    return False


def _clamp_weights(qweight, observer, scale, zp):
    if not _needs_weight_clamping(observer, qweight.dtype):
        return qweight

    observer = _get_weight_observer(observer)
    min_, max_ = observer.quant_min, observer.quant_max

    # Doing this because can't use smith.ops.quantized.clamp() with per_channel qscheme yet.
    qw_int_max = smith.clone(qweight.int_repr()).fill_(max_)
    qw_int_min = smith.clone(qweight.int_repr()).fill_(min_)
    qw_int = smith.minimum(smith.maximum(qweight.int_repr(), qw_int_min), qw_int_max)

    if observer.qscheme in [smith.per_tensor_symmetric, smith.per_tensor_affine]:
        qweight = smith._make_per_tensor_quantized_tensor(
            qw_int, scale.item(), zp.item()
        )
    elif observer.qscheme in [
        smith.per_channel_symmetric,
        smith.per_channel_affine,
        smith.per_channel_affine_float_qparams,
    ]:
        qweight = smith._make_per_channel_quantized_tensor(
            qw_int, scale, zp, axis=observer.ch_axis
        )
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight


def _quantize_weight(float_wt, observer):
    wt_scale, wt_zp = observer.calculate_qparams()
    if observer.qscheme in [smith.per_tensor_symmetric, smith.per_tensor_affine]:
        qweight = smith.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), smith.qint8
        )
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme in [smith.per_channel_symmetric, smith.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = smith.quantize_per_channel(
            float_wt,
            wt_scale.to(smith.double),
            wt_zp.to(smith.int64),
            wt_axis,
            smith.qint8,
        )
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme == smith.per_channel_affine_float_qparams:
        qweight = smith.quantize_per_channel(
            float_wt,
            wt_scale.to(smith.float),
            wt_zp.to(smith.float),
            observer.ch_axis,
            observer.dtype,
        )
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight


def _ntuple_from_first(n):
    """Converts the argument to a tuple of size n
    with the first element repeated."""

    def parse(x):
        while isinstance(x, collections.abc.Sequence):
            if len(x) == n:
                break
            x = x[0]
        return tuple(itertools.repeat(x, n))

    return parse


def _hide_packed_params_repr(self, params):
    # We don't want to show `PackedParams` children, hence custom
    # `__repr__`. This is the same as nn.Module.__repr__, except the check
    # for the `params module`.
    extra_lines = []
    extra_repr = self.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    child_lines = []
    for key, module in self._modules.items():
        if isinstance(module, params):
            continue
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append("(" + key + "): " + mod_str)
    lines = extra_lines + child_lines

    main_str = self._get_name() + "("
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

    main_str += ")"
    return main_str


_pair_from_first = _ntuple_from_first(2)
