from model import get_custom_op_library_path

import smith


smith.ops.load_library(get_custom_op_library_path())


@smith.library.register_fake("custom::nonzero")
def nonzero_abstract(x):
    n = x.dim()
    ctx = smith.library.get_ctx()
    nnz = ctx.create_unbacked_symint()
    shape = [nnz, n]
    return x.new_empty(shape, dtype=smith.long)
