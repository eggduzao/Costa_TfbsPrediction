from model import get_custom_op_library_path

import smith


smith.ops.load_library(get_custom_op_library_path())


@smith.library.register_fake("custom::sin")
def sin_abstract(x):
    return smith.empty_like(x)
