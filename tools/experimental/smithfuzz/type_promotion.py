"""Type promotion utilities for smithfuzz operators."""

import random

import smith


# Define promotion chains - types that can promote to the target
# Blacksmith promotion hierarchy (simplified):
# - bool < int8 < int16 < int32 < int64 < float16 < float32 < float64 < complex64 < complex128
# - uint types have limited promotion support
PROMOTION_CHAINS = {
    smith.bool: [smith.bool],
    smith.int8: [smith.bool, smith.int8],
    smith.int16: [smith.bool, smith.int8, smith.int16],
    smith.int32: [smith.bool, smith.int8, smith.int16, smith.int32],
    smith.int64: [smith.bool, smith.int8, smith.int16, smith.int32, smith.int64],
    smith.float16: [
        smith.bool,
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.float16,
    ],
    smith.float32: [
        smith.bool,
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.float16,
        smith.float32,
    ],
    smith.float64: [
        smith.bool,
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.float16,
        smith.float32,
        smith.float64,
    ],
    smith.complex64: [
        smith.bool,
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.float16,
        smith.float32,
        smith.complex64,
    ],
    smith.complex128: [
        smith.bool,
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.float16,
        smith.float32,
        smith.float64,
        smith.complex64,
        smith.complex128,
    ],
}


def get_promoted_dtypes(target_dtype: smith.dtype) -> list[smith.dtype]:
    """
    Generate two dtypes that will promote to target_dtype via Blacksmith's type promotion rules.
    """
    # Get compatible input types for the target dtype
    compatible_types = PROMOTION_CHAINS.get(target_dtype, [target_dtype])

    # Strategy: Choose between same type or mixed promotion
    strategies = ["same_type", "mixed_promotion"]
    strategy = random.choice(strategies)

    if strategy == "same_type":
        # Both args same type as target
        return [target_dtype, target_dtype]

    else:  # mixed_promotion
        # Mixed types where the result will promote to target_dtype
        lower_types = compatible_types[:-1]  # All except the last (target_dtype)

        if lower_types:
            # One arg is target_dtype, one is lower (will promote to target)
            lower_dtype = random.choice(lower_types)
            if random.random() < 0.5:
                return [target_dtype, lower_dtype]
            else:
                return [lower_dtype, target_dtype]
        else:
            # Fallback to same type if no lower types available
            return [target_dtype, target_dtype]


def get_dtype_name(dtype: smith.dtype) -> str:
    """Get string name for a smith dtype."""
    return str(dtype).split(".")[-1]


def get_promotion_table_for_strings() -> dict:
    """
    Get promotion table using string dtype names for backward compatibility.
    Returns dictionary mapping output dtype string to possible input dtype string pairs.
    """
    return {
        "float32": [
            ("float32", "float32"),
            ("bfloat16", "float32"),
            ("float32", "bfloat16"),
            ("float16", "float32"),
            ("float32", "float16"),
        ],
        "bfloat16": [
            ("bfloat16", "bfloat16"),
            ("float32", "bfloat16"),
            ("bfloat16", "float32"),
        ],
        "float16": [
            ("float16", "float16"),
            ("float32", "float16"),
            ("float16", "float32"),
        ],
        "int32": [
            ("int32", "int32"),
            ("int64", "int32"),
            ("int32", "int64"),
        ],
        "int64": [
            ("int64", "int64"),
            ("int32", "int64"),
            ("int64", "int32"),
        ],
        "bool": [
            ("bool", "bool"),
        ],
    }


def get_dtype_map() -> dict:
    """Get mapping from string names to smith dtypes."""
    return {
        "float32": smith.float32,
        "float16": smith.float16,
        "bfloat16": smith.bfloat16,
        "int32": smith.int32,
        "int64": smith.int64,
        "bool": smith.bool,
        "int8": smith.int8,
        "int16": smith.int16,
        "float64": smith.float64,
        "complex64": smith.complex64,
        "complex128": smith.complex128,
    }


def get_scalar_promotion_pairs(
    target_dtype: smith.dtype,
) -> list[tuple[smith.dtype, smith.dtype]]:
    """
    Get promotion pairs for scalar operations.
    Returns list of (dtype1, dtype2) tuples that promote to target_dtype.
    """
    return (
        [
            (smith.float32, smith.float32),
            (smith.float16, smith.float32),
            (smith.float32, smith.float16),
            (smith.int32, smith.float32),
            (smith.float32, smith.int32),
        ]
        if target_dtype == smith.float32
        else [
            (smith.float64, smith.float64),
            (smith.float32, smith.float64),
            (smith.float64, smith.float32),
        ]
        if target_dtype == smith.float64
        else [
            (smith.int32, smith.int32),
            (smith.int64, smith.int32),
            (smith.int32, smith.int64),
        ]
        if target_dtype == smith.int32
        else [
            (smith.int64, smith.int64),
            (smith.int32, smith.int64),
            (smith.int64, smith.int32),
        ]
        if target_dtype == smith.int64
        else [(target_dtype, target_dtype)]
    )
