import smith
from smith.ao.quantization.experimental.quantizer import APoTQuantizer


# class to store APoT quantized tensor
class TensorAPoT:
    quantizer: APoTQuantizer
    data: smith.Tensor

    def __init__(self, quantizer: APoTQuantizer, apot_data: smith.Tensor):
        self.quantizer = quantizer
        self.data = apot_data

    def int_repr(self) -> smith.Tensor:
        return self.data
