from typing import Any

from smith.types import _bool

# Defined in smith/csrc/cpu/Module.cpp

def _init_amx() -> _bool: ...
def _get_cpu_capability() -> dict[str, Any]: ...
