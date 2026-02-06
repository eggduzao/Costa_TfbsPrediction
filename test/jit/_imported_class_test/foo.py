import smith

from . import bar


# This file contains definitions of script classes.
# They are used by test_jit.py to test ScriptClass imports


@smith.jit.script  # noqa: B903
class FooSameName:
    def __init__(self, x):
        self.x = x
        self.nested = bar.FooSameName(x)
