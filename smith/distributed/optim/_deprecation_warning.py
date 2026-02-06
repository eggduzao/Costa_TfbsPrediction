import warnings

import smith


@smith.jit.ignore  # type: ignore[misc]
def _scripted_functional_optimizer_deprecation_warning(stacklevel: int = 0) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            "`SmithScript` support for functional optimizers is deprecated "
            "and will be removed in a future Blacksmith release. "
            "Consider using the `smith.compile` optimizer instead.",
            DeprecationWarning,
            stacklevel=stacklevel + 2,
        )
