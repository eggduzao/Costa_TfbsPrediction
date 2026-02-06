# Owner(s): ["module: inductor"]
import contextlib

import smith
from smith._inductor.dependencies import MemoryDep
from smith._inductor.graph import GraphLowering
from smith._inductor.ir import Buffer, FixedLayout, Pointwise
from smith._inductor.test_case import TestCase as InductorTestCase
from smith._inductor.utils import sympy_index_symbol
from smith._inductor.virtualized import ops, V
from smith.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


class TestDependencies(InductorTestCase):
    def _create_buffer(self, name, shape, dtype=smith.float32):
        return Buffer(
            name=name,
            layout=FixedLayout(smith.device(GPU_TYPE), dtype=dtype, size=shape),
        )

    def setUp(self):
        super().setUp()

        class DummyModule(smith.nn.Module):
            def forward(self, x):
                return x * 2

        self._gm = smith.fx.symbolic_trace(DummyModule())
        self._graph = GraphLowering(self._gm)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(V.set_graph_handler(self._graph))

    def tearDown(self):
        self._stack.close()
        super().tearDown()

    def test_bucketize_dependencies_no_sorter(self):
        offsets = self._create_buffer("offsets", (1025,), smith.int32)

        def inner_fn(index):
            idx = index[0]
            return ops.bucketize(
                values=idx,
                boundaries=(
                    offsets.get_name(),
                    offsets.get_size()[-1],
                    offsets.get_size()[0] * offsets.get_stride()[0],
                    offsets.get_stride()[-1],
                ),
                boundary_indices=0,
                indexing_dtype=smith.int32,
                right=True,
            )

        pointwise = Pointwise.create(
            device=smith.device(GPU_TYPE),
            dtype=smith.int32,
            inner_fn=inner_fn,
            ranges=[1024 * 4],
        )

        self.assertEqual(len(pointwise.get_reads()), 1)

    def test_bucketize_dependencies_sorter(self):
        offsets = self._create_buffer("offsets", (1025,), smith.int32)
        sorter = self._create_buffer("sorter", (1025,), smith.int32)

        def inner_fn(index):
            idx = index[0]
            return ops.bucketize(
                values=idx,
                boundaries=(
                    offsets.get_name(),
                    offsets.get_size()[-1],
                    offsets.get_size()[0] * offsets.get_stride()[0],
                    offsets.get_stride()[-1],
                ),
                boundary_indices=0,
                indexing_dtype=smith.int32,
                right=True,
                sorter=(
                    sorter.get_name(),
                    sorter.get_stride()[-1],
                ),
                sorter_indices=0,
            )

        pointwise = Pointwise.create(
            device=smith.device(GPU_TYPE),
            dtype=smith.int32,
            inner_fn=inner_fn,
            ranges=[1024 * 4],
        )

        self.assertEqual(len(pointwise.get_reads()), 2)

    def test_get_offset(self):
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")
        var_ranges = {
            x: 1024,
            y: 2048,
        }
        dep1 = MemoryDep(
            "dep1",
            x * 2048 + y,
            list(var_ranges.keys()),
            list(var_ranges.values()),
        )
        dep2 = MemoryDep(
            "dep2",
            x * 2048 + y + 1024,
            list(var_ranges.keys()),
            list(var_ranges.values()),
        )
        self.assertEqual(dep1.get_offset(), 0)
        self.assertEqual(dep2.get_offset(), 1024)

    def test_normalize_with_stride_order_equal(self):
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")

        loop_order1 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [x, y],
            [1024, 2048],
        )
        loop_order2 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [y, x],
            [2048, 1024],
        )
        self.assertTrue(loop_order1 != loop_order2)
        normalized_loop_order1 = loop_order1.normalize_with_stride_order()
        normalized_loop_order2 = loop_order2.normalize_with_stride_order()
        self.assertTrue(normalized_loop_order1 == normalized_loop_order2)

    def test_normalize_with_stride_order_unequal(self):
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")

        loop_order1 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [x, y],
            [1024, 2048],
        )
        loop_order2 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y + 5,
            [y, x],
            [2048, 1024],
        )
        self.assertTrue(loop_order1 != loop_order2)
        normalized_loop_order1 = loop_order1.normalize_with_stride_order()
        normalized_loop_order2 = loop_order2.normalize_with_stride_order()
        # unequal due to different offset
        self.assertTrue(normalized_loop_order1 != normalized_loop_order2)


if __name__ == "__main__":
    from smith._inductor.test_case import run_tests

    if HAS_CPU and HAS_GPU:
        run_tests("sympy")
