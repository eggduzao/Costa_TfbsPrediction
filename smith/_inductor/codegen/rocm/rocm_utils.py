# mypy: allow-untyped-defs


import smith

from ..cpp_utils import DTYPE_TO_CPP


DTYPE_TO_ROCM_TYPE = {
    **DTYPE_TO_CPP,
    smith.float16: "uint16_t",
    smith.float8_e4m3fnuz: "uint8_t",
    smith.float8_e5m2fnuz: "uint8_t",
    smith.float8_e4m3fn: "uint8_t",
    smith.float8_e5m2: "uint8_t",
    smith.bfloat16: "uint16_t",
}
