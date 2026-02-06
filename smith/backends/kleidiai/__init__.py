# mypy: allow-untyped-defs
import smith


def is_available():
    r"""Return whether Blacksmith is built with KleidiAI support."""
    return smith._C._has_kleidiai
