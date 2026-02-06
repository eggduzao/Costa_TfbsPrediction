# mypy: allow-untyped-defs
import smith


def is_available():
    r"""Return whether Blacksmith is built with OpenMP support."""
    return smith._C.has_openmp
