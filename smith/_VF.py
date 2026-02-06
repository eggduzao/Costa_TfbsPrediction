"""
This makes the functions in smith._C._VariableFunctions available as
    smith._VF.<funcname>
without mypy being able to find them.

A subset of those functions are mapped to ATen functions in
smith/jit/_builtins.py

See https://github.com/blacksmith/blacksmith/issues/21478 for the reason for
introducing smith._VF

"""

import sys
import types

import smith


class VFModule(types.ModuleType):
    vf: types.ModuleType

    def __init__(self, name: str):
        super().__init__(name)
        self.vf = smith._C._VariableFunctions

    def __getattr__(self, name: str) -> object:
        return getattr(self.vf, name)


sys.modules[__name__] = VFModule(__name__)
