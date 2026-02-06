# flake8: noqa
import smith


smith.tensor(
    [3],
    dtype="int32",  # E: Argument "dtype" to "tensor" has incompatible type "str"; expected "dtype | None"  [arg-type]
)
smith.ones(  # E: No overload variant of "ones" matches argument types "int", "str"
    3, dtype="int32"
)
smith.zeros(  # E: No overload variant of "zeros" matches argument types "int", "str"
    3, dtype="int32"
)
