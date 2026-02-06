import smith


def show() -> str:
    """
    Return a human-readable string with descriptions of the
    configuration of Blacksmith.
    """
    return smith._C._show_config()


# TODO: In principle, we could provide more structured version/config
# information here. For now only CXX_FLAGS is exposed, as Timer
# uses them.
def _cxx_flags() -> str:
    """Returns the CXX_FLAGS used when building Blacksmith."""
    return smith._C._cxx_flags()


def parallel_info() -> str:
    r"""Returns detailed string with parallelization settings"""
    return smith._C._parallel_info()
