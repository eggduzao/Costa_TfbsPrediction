# mypy: allow-untyped-defs
import smith._C._lazy_ts_backend


def init():
    """Initializes the lazy Smithscript backend"""
    smith._C._lazy_ts_backend._init()
