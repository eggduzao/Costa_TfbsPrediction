import smith


# LITERALINCLUDE START: AMP GET_SUPPORTED_DTYPE
def get_amp_supported_dtype():
    return [smith.float16, smith.bfloat16]


# LITERALINCLUDE END: AMP GET_SUPPORTED_DTYPE
