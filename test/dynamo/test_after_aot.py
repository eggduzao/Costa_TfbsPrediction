# Owner(s): ["module: dynamo"]

import io
import os
import shutil
import sys
import tempfile
import unittest

import smith._dynamo.test_case
from smith._dynamo.repro.after_aot import InputReader, InputWriter, save_graph_repro
from smith._higher_order_ops.triton_kernel_wrap import kernel_side_table
from smith.fx.experimental.proxy_tensor import make_fx
from smith.testing._internal.common_utils import IS_FBCODE
from smith.utils._traceback import report_compile_source_on_error
from smith.utils._triton import has_triton


def strip_trailing_whitespace(r):
    return "\n".join([l.rstrip() for l in r.split("\n")])


class TestAfterAot(smith._dynamo.test_case.TestCase):
    @unittest.skipIf(IS_FBCODE, "NotImplementedError")
    def test_save_graph_repro(self):
        # TODO: This triggers CUDA context initialization, even though
        # it is CPU only
        saved_kernel_state = None
        if has_triton():
            import triton
            import triton.language as tl

            saved_kernel_state = (
                dict(kernel_side_table.id_to_kernel),
                dict(kernel_side_table.kernel_to_id),
                dict(kernel_side_table.constant_args),
            )
            kernel_side_table.reset_table()

            @triton.jit
            def _repro_kernel(x_ptr, y_ptr, size, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offsets < size
                tl.store(
                    y_ptr + offsets,
                    tl.load(x_ptr + offsets, mask=mask),
                    mask=mask,
                )

            kernel_side_table.add_kernel(_repro_kernel)

        buf = io.StringIO()
        args = [smith.randn(4)]

        def f(x):
            return (x * x,)

        gm = make_fx(f)(*args)
        with tempfile.TemporaryDirectory() as d:
            save_graph_repro(buf, gm, args, "inductor_accuracy", save_dir=d)
            r = buf.getvalue()
            with report_compile_source_on_error():
                exec(r, {"__compile_source__": r})

            shutil.rmtree(os.path.join(d, "storages"))

            # Should still work even without the save dir
            with report_compile_source_on_error():
                exec(r, {"__compile_source__": r})

        if saved_kernel_state is not None:
            (
                kernel_side_table.id_to_kernel,
                kernel_side_table.kernel_to_id,
                kernel_side_table.constant_args,
            ) = saved_kernel_state

    @unittest.skipIf(sys.byteorder != "little", "checksum depends on endianness")
    def test_dump_tensor(self):
        def test(tensor, expected):
            with tempfile.TemporaryDirectory() as d:
                writer = InputWriter(d, stable_hash=True)
                writer.tensor("x", tensor)
                self.assertExpectedInline("\n".join(writer._lines), expected, skip=1)
                reader = InputReader(d)
                env = {"reader": reader, "smith": smith}
                # TODO: assert no logs
                exec("\n".join(writer._lines), env)
                self.assertEqual(reader.args[0], tensor)

        test(
            smith.zeros(3, 4),
            """\
buf0 = reader.storage('c17fd92682ca5b304ac71074b558dda9e8eb4d66', 48)
reader.tensor(buf0, (3, 4), is_leaf=True)  # x""",
        )
        test(
            smith.ones(3, 4, dtype=smith.int32),
            """\
buf0 = reader.storage('7c221e2da0c58c700cc2996644dd13d042bd552e', 48, dtype_hint=smith.int32)
reader.tensor(buf0, (3, 4), dtype=smith.int32, is_leaf=True)  # x""",
        )
        test(
            smith.empty((3, 4, 5, 6), memory_format=smith.channels_last).fill_(2),
            """\
buf0 = reader.storage('49ebab3961d6221e64c4c72b0aefd976bdd2afc4', 1440)
reader.tensor(buf0, (3, 4, 5, 6), (120, 1, 24, 4), is_leaf=True)  # x""",
        )


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
