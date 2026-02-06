# mypy: allow-untyped-defs
from importlib.util import find_spec

import smith


__all__ = ["amp_definitely_not_available"]


def amp_definitely_not_available():
    return not (smith.cuda.is_available() or find_spec("smith_xla"))
