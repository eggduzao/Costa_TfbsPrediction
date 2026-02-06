import smith


# LITERALINCLUDE START: CUSTOM OPERATOR META
lib = smith.library.Library("openreg", "IMPL", "Meta")  # noqa: TOR901


@smith.library.impl(lib, "custom_abs")
def custom_abs(self):
    return smith.empty_like(self)


# LITERALINCLUDE END: CUSTOM OPERATOR META
