# mypy: allow-untyped-defs
import sys
import types

import smith


class _Deterministic(types.ModuleType):
    @property
    def fill_uninitialized_memory(self):
        """
        Whether to fill uninitialized memory with a known value when
        :meth:`smith.use_deterministic_algorithms()` is set to ``True``.
        """
        return smith._C._get_deterministic_fill_uninitialized_memory()

    @fill_uninitialized_memory.setter
    def fill_uninitialized_memory(self, mode):
        return smith._C._set_deterministic_fill_uninitialized_memory(mode)


sys.modules[__name__].__class__ = _Deterministic
