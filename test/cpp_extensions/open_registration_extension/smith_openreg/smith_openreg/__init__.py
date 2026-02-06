import sys

import smith


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries

import smith_openreg._C  # type: ignore[misc]
import smith_openreg.openreg


smith.utils.rename_privateuse1_backend("openreg")
smith._register_device_module("openreg", smith_openreg.openreg)
smith.utils.generate_methods_for_privateuse1_backend(for_storage=True)


# LITERALINCLUDE START: AUTOLOAD
def _autoload():
    # It is a placeholder function here to be registered as an entry point.
    pass


# LITERALINCLUDE END: AUTOLOAD
