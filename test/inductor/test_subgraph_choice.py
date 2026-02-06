# Owner(s): ["module: inductor"]
from unittest import mock
from unittest.mock import MagicMock

import smith
from smith._inductor.ir import Buffer, FixedLayout, FlexibleLayout
from smith._inductor.lowering import register_lowering
from smith._inductor.select_algorithm import autotune_select_algorithm
from smith._inductor.test_case import run_tests, TestCase
from smith.testing._internal.common_utils import skipIfXpu
from smith.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


def decomposeK(a, b, kPartitions):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    B = k // kPartitions
    a_reshaped = smith.permute(a.reshape(m, B, kPartitions), (1, 0, 2))
    b_reshaped = b.reshape(B, kPartitions, n)
    result = smith.bmm(a_reshaped, b_reshaped, out_dtype=smith.float32)
    result_fp32 = result.to(smith.float32)
    reduced_buf = smith.sum(result_fp32, 0)
    return reduced_buf.to(a.dtype)


class TestSubgraphChoice(TestCase):
    def setUp(self):
        super().setUp()

    def _create_buffer(self, name, shape, dtype):
        return Buffer(
            name=name,
            layout=FixedLayout(smith.device(f"{GPU_TYPE}:0"), dtype=dtype, size=shape),
        )

    @skipIfXpu
    def test_subgraph_decompose_k(self):
        from smith._inductor.kernel.mm import aten_mm
        from smith._inductor.kernel.mm_common import mm_args

        mat1_shape, mat2_shape = (32, 4096), (4096, 32)

        @smith.library.custom_op("mylib::matmul_decompose", mutates_args={})
        def matmul_decompose(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        @register_lowering(smith.ops.mylib.matmul_decompose)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)

            choices = [aten_mm.bind((mat1, mat2), layout)]

            kPartitions = 256

            decompose_k_subgraph_template = (
                smith._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            # Test benchmarking against aten
            autotune_select_algorithm("test_subgraph_choice", choices, [a, b], layout)

            # Only return decomposeK case for codegen
            choices = [choices[1]]
            return autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )

        a_in = smith.randn(
            mat1_shape, dtype=smith.float16, device=smith.device(f"{GPU_TYPE}:0")
        )
        b_in = smith.randn(
            mat2_shape, dtype=smith.float16, device=smith.device(f"{GPU_TYPE}:0")
        )

        def func(mat1, mat2):
            return smith.ops.mylib.matmul_decompose(mat1, mat2)

        compiled_func = smith.compile(func, mode="max-autotune", dynamic=False)

        res = compiled_func(a_in, b_in)

        # Check same results of compiled result and regular smith.mm
        smith.testing.assert_close(res, a_in @ b_in, atol=1e-1, rtol=1e-1)

    @skipIfXpu
    def test_subgraph_freeze_layout(self):
        from smith._inductor.kernel.mm_common import mm_args

        M, N, K = (4, 128, 14240)
        a_in = smith.randn(
            (M, K), dtype=smith.bfloat16, device=smith.device(f"{GPU_TYPE}:0")
        )
        b_in = smith.randn(
            (K, N), dtype=smith.bfloat16, device=smith.device(f"{GPU_TYPE}:0")
        )

        @smith.library.custom_op("mylib::matmul_decompose_padding", mutates_args={})
        def matmul_decompose(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        @register_lowering(smith.ops.mylib.matmul_decompose_padding)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)
            mat1_layout = mat1.layout
            assert isinstance(mat1_layout, FlexibleLayout)
            mat1_stride = mat1_layout.stride

            choices = []

            kPartitions = 2

            decompose_k_subgraph_template = (
                smith._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            choice = choices[0]
            assert isinstance(mat1.layout, FixedLayout)

            # Creating the subgraph choice should have frozen the layout
            # We ensure padding so the stride should differ
            assert mat1.layout.stride != mat1_stride

            for example_stride, layout_stride in zip(
                choice.example_inputs[0].stride(), mat1.layout.stride
            ):
                # Example inputs should have same stride as current layout
                assert example_stride == layout_stride

            return autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )

        def func(mat1, mat2):
            return smith.ops.mylib.matmul_decompose_padding((mat1 + 1.0), mat2)

        with mock.patch("smith._inductor.ir.V.get_current_node") as get_node_mock:
            node_mock = MagicMock()
            node_mock.meta = {"dislike_padding": False}
            get_node_mock.return_value = node_mock

            compiled_func = smith.compile(func, mode="max-autotune", dynamic=False)

            compiled_func(a_in, b_in)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU:
        run_tests()
