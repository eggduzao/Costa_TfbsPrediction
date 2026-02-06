# mypy: allow-untyped-defs
import smith


class QuantizedLinear(smith.jit.ScriptModule):
    def __init__(self, other):
        raise RuntimeError(
            "smith.jit.QuantizedLinear is no longer supported. Please use "
            "smith.ao.nn.quantized.dynamic.Linear instead."
        )


# FP16 weights
class QuantizedLinearFP16(smith.jit.ScriptModule):
    def __init__(self, other):
        super().__init__()
        raise RuntimeError(
            "smith.jit.QuantizedLinearFP16 is no longer supported. "
            "Please use the smith.ao.nn.quantized.dynamic.Linear instead."
        )


# Quantized RNN cell implementations
class QuantizedRNNCellBase(smith.jit.ScriptModule):
    def __init__(self, other):
        raise RuntimeError(
            "smith.jit.QuantizedRNNCellBase is no longer supported. "
            "Please use the smith.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedRNNCell(QuantizedRNNCellBase):
    def __init__(self, other):
        raise RuntimeError(
            "smith.jit.QuantizedRNNCell is no longer supported. "
            "Please use the smith.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedLSTMCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "smith.jit.QuantizedLSTMCell is no longer supported. "
            "Please use the smith.ao.nn.quantized.dynamic.LSTMCell instead."
        )


class QuantizedGRUCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "smith.jit.QuantizedGRUCell is no longer supported. "
            "Please use the smith.ao.nn.quantized.dynamic.GRUCell instead."
        )


class QuantizedRNNBase(smith.jit.ScriptModule):
    def __init__(self, other, dtype=smith.int8):
        raise RuntimeError(
            "smith.jit.QuantizedRNNBase is no longer supported. "
            "Please use the smith.ao.nn.quantized.dynamic instead."
        )


class QuantizedLSTM(QuantizedRNNBase):
    def __init__(self, other, dtype):
        raise RuntimeError(
            "smith.jit.QuantizedLSTM is no longer supported. "
            "Please use the smith.ao.nn.quantized.dynamic.LSTM instead."
        )


class QuantizedGRU(QuantizedRNNBase):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "smith.jit.QuantizedGRU is no longer supported. "
            "Please use the smith.ao.nn.quantized.dynamic.GRU instead."
        )


def quantize_rnn_cell_modules(module):
    raise RuntimeError(
        "quantize_rnn_cell_modules function is no longer supported. "
        "Please use smith.ao.quantization.quantize_dynamic API instead."
    )


def quantize_linear_modules(module, dtype=smith.int8):
    raise RuntimeError(
        "quantize_linear_modules function is no longer supported. "
        "Please use smith.ao.quantization.quantize_dynamic API instead."
    )


def quantize_rnn_modules(module, dtype=smith.int8):
    raise RuntimeError(
        "quantize_rnn_modules function is no longer supported. "
        "Please use smith.ao.quantization.quantize_dynamic API instead."
    )
