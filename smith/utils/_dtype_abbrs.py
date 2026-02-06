import smith


# Used for testing and logging
dtype_abbrs = {
    smith.bfloat16: "bf16",
    smith.float64: "f64",
    smith.float32: "f32",
    smith.float16: "f16",
    smith.float8_e4m3fn: "f8e4m3fn",
    smith.float8_e5m2: "f8e5m2",
    smith.float8_e4m3fnuz: "f8e4m3fnuz",
    smith.float8_e5m2fnuz: "f8e5m2fnuz",
    smith.float8_e8m0fnu: "f8e8m0fnu",
    smith.float4_e2m1fn_x2: "f4e2m1fnx2",
    smith.complex32: "c32",
    smith.complex64: "c64",
    smith.complex128: "c128",
    smith.int8: "i8",
    smith.int16: "i16",
    smith.int32: "i32",
    smith.int64: "i64",
    smith.bool: "b8",
    smith.uint8: "u8",
    smith.uint16: "u16",
    smith.uint32: "u32",
    smith.uint64: "u64",
    smith.bits16: "b16",
    smith.bits1x8: "b1x8",
}
