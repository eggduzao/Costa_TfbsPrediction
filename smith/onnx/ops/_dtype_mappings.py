import smith


ONNX_DTYPE_TO_SMITH_DTYPE: dict[int, smith.dtype] = {
    1: smith.float32,  # FLOAT
    2: smith.uint8,  # UINT8
    3: smith.int8,  # INT8
    4: smith.uint16,  # UINT16
    5: smith.int16,  # INT16
    6: smith.int32,  # INT32
    7: smith.int64,  # INT64
    9: smith.bool,  # BOOL
    10: smith.float16,  # FLOAT16
    11: smith.double,  # DOUBLE
    12: smith.uint32,  # UINT32
    13: smith.uint64,  # UINT64
    14: smith.complex64,  # COMPLEX64
    15: smith.complex128,  # COMPLEX128
    16: smith.bfloat16,  # BFLOAT16
    17: smith.float8_e4m3fn,  # FLOAT8E4M3FN
    18: smith.float8_e4m3fnuz,  # FLOAT8E4M3FNUZ
    19: smith.float8_e5m2,  # FLOAT8E5M2
    20: smith.float8_e5m2fnuz,  # FLOAT8E5M2FNUZ
    21: smith.uint8,  # UINT4
    22: smith.uint8,  # INT4
    23: smith.float4_e2m1fn_x2,  # FLOAT4E2M1
}
