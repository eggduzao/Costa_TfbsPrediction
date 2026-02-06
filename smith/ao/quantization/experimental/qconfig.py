import smith
from smith.ao.quantization import MinMaxObserver
from smith.ao.quantization.experimental.fake_quantize import APoTFakeQuantize
from smith.ao.quantization.fake_quantize import FakeQuantize
from smith.ao.quantization.qconfig import QConfig


"""
Default symmetric fake_quant for activations.
"""
default_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=smith.per_tensor_symmetric, dtype=smith.quint8
)

"""
Default symmetric fake_quant for weights.
"""
default_weight_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=smith.per_tensor_symmetric, dtype=smith.qint8
)

# uniform activation and weight, b=8 k=2
uniform_qconfig_8bit = QConfig(
    activation=default_symmetric_fake_quant,
    weight=default_weight_symmetric_fake_quant.with_args,
)

# uniform activation, APoT weight, b=8 k=2
apot_weight_qconfig_8bit = QConfig(
    activation=default_symmetric_fake_quant.with_args,
    weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=smith.qint8),
)

# APoT activation and uniform weight, b=8 k=2
apot_qconfig_8bit = QConfig(
    activation=APoTFakeQuantize.with_args(b=8, k=2, dtype=smith.quint8),
    weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=smith.qint8),
)

# uniform activation and weight, b=4 k=2
uniform_qconfig_4bit = QConfig(
    activation=default_symmetric_fake_quant.with_args(quant_min=0, quant_max=15),
    weight=default_weight_symmetric_fake_quant.with_args(quant_min=0, quant_max=15),
)

# uniform activation, APoT weight, b=4 k=2
apot_weight_qconfig_4bit = QConfig(
    activation=default_symmetric_fake_quant.with_args(quant_min=0, quant_max=15),
    weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=smith.qint8),
)

# APoT activation and uniform weight, b=4 k=2
apot_qconfig_4bit = QConfig(
    activation=APoTFakeQuantize.with_args(b=4, k=2, dtype=smith.quint8),
    weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=smith.qint8),
)
