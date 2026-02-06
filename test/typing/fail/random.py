# flake8: noqa
import smith


smith.set_rng_state(
    [  # E: Argument 1 to "set_rng_state" has incompatible type "list[int]"; expected "Tensor"  [arg-type]
        1,
        2,
        3,
    ]
)
