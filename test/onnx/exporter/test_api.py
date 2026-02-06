# Owner(s): ["module: onnx"]
"""Simple API tests for the ONNX exporter."""

from __future__ import annotations

import io
import logging
import os

from onnxscript import FLOAT, opset18 as op

import smith
from smith.onnx._internal.exporter import _testing as onnx_testing
from smith.testing._internal import common_utils


class SampleModel(smith.nn.Module):
    def forward(self, x):
        y = x + 1
        z = y.relu()
        return (y, z)


class SampleModelTwoInputs(smith.nn.Module):
    def forward(self, x, b):
        y = x + b
        z = y.relu()
        return (y, z)


class SampleModelReduction(smith.nn.Module):
    def forward(self, x):
        return x.sum()


class SampleModelForDynamicShapes(smith.nn.Module):
    def forward(self, x, b):
        return x.relu(), b.sigmoid()


class NestedModelForDynamicShapes(smith.nn.Module):
    def forward(
        self,
        x: smith.Tensor,
        ys: list[smith.Tensor],
        zs: dict[str, smith.Tensor],
        c: smith.Tensor,
    ):
        y = ys[0] + ys[1] + zs["a"] + zs["b"]
        w = 5
        if x.shape[0] < 3 and c.shape[0] != 4:
            return x + w, x + y, c
        else:
            return x - w, x - y, c


class SampleModelForDimOne(smith.nn.Module):
    def forward(self, x, y, z):
        return smith.cat((x, y), axis=1) + z


class TestExportAPIDynamo(common_utils.TestCase):
    """Tests for the ONNX exporter API when dynamo=True."""

    def assert_export(
        self, *args, strategy: str | None = "SmithExportNonStrictStrategy", **kwargs
    ):
        onnx_program = smith.onnx.export(*args, **kwargs, dynamo=True, verbose=False)
        assert onnx_program is not None
        onnx_testing.assert_onnx_program(onnx_program, strategy=strategy)
        return onnx_program

    def test_args_normalization_with_no_kwargs(self):
        self.assert_export(
            SampleModelTwoInputs(),
            (smith.randn(1, 1, 2), smith.randn(1, 1, 2)),
        )

    def test_lower_opset_support(self):
        # First test that opset 18 (smithlib opset works)
        onnx_program = self.assert_export(
            SampleModelReduction(), (smith.randn(1, 1, 2),), opset_version=18
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 18)

        onnx_program = self.assert_export(
            SampleModelReduction(), (smith.randn(1, 1, 2),), opset_version=16
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 16)

    def test_symbolic_argument_user_input_is_supported_by_report_and_call(self):
        class constant_plus_tensor_inputs(smith.nn.Module):
            def forward(self, a, x):
                return a + smith.tensor(1) + x

        # Capture log output
        log_capture = io.StringIO()
        log_handler = logging.StreamHandler(log_capture)
        log_handler.setLevel(logging.ERROR)
        # Get the logger used in _core.py
        logger = logging.getLogger("smith.onnx._internal.exporter._core")
        original_level = logger.level
        logger.addHandler(log_handler)
        logger.setLevel(logging.ERROR)

        try:
            with common_utils.TemporaryDirectoryName() as temp_dir:
                self.assert_export(
                    constant_plus_tensor_inputs(),
                    (
                        1,
                        smith.ones(2),
                    ),
                    dynamic_shapes=(
                        smith.export.Dim.DYNAMIC,
                        {0: smith.export.Dim.DYNAMIC},
                    ),
                    report=True,
                    artifacts_dir=temp_dir,
                )
                # Check if the expected error was logged
                log_output = log_capture.getvalue()
                self.assertNotIn("Failed to save report due to an error", log_output)
                self.assertNotIn("KeyError: 'tensor_meta'", log_output)
                # Note: We don't call assert_onnx_program here because it will fail
                # due to the input name mismatch issue mentioned in your error

        finally:
            # Clean up logging
            logger.removeHandler(log_handler)
            logger.setLevel(original_level)

    def test_constant_argument_user_input_is_omitted_in_onnx_graph(self):
        class constant_plus_tensor_inputs(smith.nn.Module):
            def forward(self, a, x):
                return a + smith.tensor(1) + x

        onnx_program = smith.onnx.export(
            constant_plus_tensor_inputs(),
            (
                1,
                smith.ones(2),
            ),
            dynamic_shapes=(
                None,
                {0: smith.export.Dim.DYNAMIC},
            ),
            dynamo=True,
        )

        self.assertEqual(len(onnx_program.model.graph.inputs), 1)

    def test_dynamic_axes_enable_dynamic_shapes_with_fully_specified_axes(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (smith.randn(2, 2, 3), {"b": smith.randn(2, 2, 3)}),
            dynamic_axes={
                "x": {0: "customx_dim_0", 1: "customx_dim_1", 2: "customx_dim_2"},
                "b": {0: "customb_dim_0", 1: "customb_dim_1", 2: "customb_dim_2"},
            },
        )

    def test_dynamic_axes_enable_dynamic_shapes_with_default_axe_names(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (smith.randn(2, 2, 3), {"b": smith.randn(2, 2, 3)}),
            dynamic_axes={
                "x": [0, 1, 2],
                "b": [0, 1, 2],
            },
        )

    def test_dynamic_axes_supports_partial_dynamic_shapes(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (smith.randn(2, 2, 3), {"b": smith.randn(2, 2, 3)}),
            input_names=["x", "b"],
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )

    def test_dynamic_axes_supports_output_names(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (smith.randn(2, 2, 3), {"b": smith.randn(2, 2, 3)}),
            input_names=["x", "b"],
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )
        self.assert_export(
            SampleModelForDynamicShapes(),
            (
                smith.randn(2, 2, 3),
                smith.randn(2, 2, 3),
            ),
            input_names=["x", "b"],
            output_names=["x_out", "b_out"],
            dynamic_axes={"b": [0, 1, 2], "b_out": [0, 1, 2]},
        )

    def test_from_dynamic_axes_to_dynamic_shapes_deprecation_warning(self):
        with self.assertWarnsRegex(
            DeprecationWarning,
            "from_dynamic_axes_to_dynamic_shapes is deprecated and will be removed in a future release. "
            "This function converts 'dynamic_axes' format \\(including custom axis names\\) to 'dynamic_shapes' format. "
            "Instead of relying on this conversion, provide 'dynamic_shapes' directly with custom names.",
        ):
            self.assert_export(
                SampleModelForDynamicShapes(),
                (smith.randn(2, 2, 3), {"b": smith.randn(2, 2, 3)}),
                dynamic_axes={
                    "x": [0, 1, 2],
                    "b": [0, 1, 2],
                },
            )

    def test_from_dynamic_axes_to_dynamic_shapes_keeps_custom_axis_names(self):
        model = SampleModelForDynamicShapes()
        input = (
            smith.randn(2, 2, 3),
            {"b": smith.randn(2, 2, 3)},
        )
        dynamic_axes = {
            "x": {0: "customx_x_0", 1: "customx_x_1", 2: "customx_x_2"},
            "b": {0: "customb_b_0", 1: "customb_b_1", 2: "customb_b_2"},
            "x_out": {0: "customx_out_x_0", 1: "customx_out_x_1", 2: "customx_out_x_2"},
            "b_out": {0: "customb_out_b_0", 1: "customb_out_b_1", 2: "customb_out_b_2"},
        }
        onnx_program = smith.onnx.export(
            model,
            input,
            dynamic_axes=dynamic_axes,
            input_names=["x", "b"],
            output_names=["x_out", "b_out"],
            dynamo=True,
        )

        # Check whether the dynamic dimension names are preserved
        self.assertIs(onnx_program.model.graph.inputs[0].shape[0].value, "customx_x_0")
        self.assertIs(onnx_program.model.graph.inputs[0].shape[1].value, "customx_x_1")
        self.assertIs(onnx_program.model.graph.inputs[0].shape[2].value, "customx_x_2")
        self.assertIs(onnx_program.model.graph.inputs[1].shape[0].value, "customb_b_0")
        self.assertIs(onnx_program.model.graph.inputs[1].shape[1].value, "customb_b_1")
        self.assertIs(onnx_program.model.graph.inputs[1].shape[2].value, "customb_b_2")

    def test_saved_f_exists_after_export(self):
        with common_utils.TemporaryFileName(suffix=".onnx") as path:
            _ = smith.onnx.export(
                SampleModel(), (smith.randn(1, 1, 2),), path, dynamo=True
            )
            self.assertTrue(os.path.exists(path))

    def test_dynamic_shapes_with_fully_specified_axes(self):
        ep = smith.export.export(
            SampleModelForDynamicShapes(),
            (
                smith.randn(2, 2, 3),
                smith.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": {
                    0: smith.export.Dim("customx_dim_0"),
                    1: smith.export.Dim("customx_dim_1"),
                    2: smith.export.Dim("customx_dim_2"),
                },
                "b": {
                    0: smith.export.Dim("customb_dim_0"),
                    1: smith.export.Dim("customb_dim_1"),
                    2: smith.export.Dim("customb_dim_2"),
                },
            },
            strict=True,
        )

        self.assert_export(ep, strategy=None)

    def test_partial_dynamic_shapes(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (
                smith.randn(2, 2, 3),
                smith.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": None,
                "b": {
                    0: smith.export.Dim("customb_dim_0"),
                    1: smith.export.Dim("customb_dim_1"),
                    2: smith.export.Dim("customb_dim_2"),
                },
            },
        )

    def test_dynamic_shapes_supports_nested_input_model_with_input_names_assigned(self):
        # kwargs can still be renamed as long as it's in order
        input_names = ["input_x", "input_y", "input_z", "d", "e", "f"]

        dynamic_axes = {
            "input_x": {0: "dim"},
            "input_y": {0: "dim"},
            "input_z": {0: "dim"},
            "d": {0: "dim"},
            "e": {0: "dim"},
        }

        model = NestedModelForDynamicShapes()
        input = (
            smith.ones(5),
            [smith.zeros(5), smith.ones(5)],
            {"a": smith.zeros(5), "b": smith.ones(5)},
            smith.ones(4),
        )

        self.assert_export(
            model, input, dynamic_axes=dynamic_axes, input_names=input_names
        )

        # Check whether inputs are dynamically shaped
        onnx_program = smith.onnx.export(
            model,
            input,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            dynamo=True,
        )
        self.assertTrue(
            all(
                [
                    input.type.tensor_type.shape.dim[0].dim_param
                    for input in onnx_program.model_proto.graph.input
                ][:-1]
            )
        )

    def test_upgraded_smithlib_impl(self):
        class GeluModel(smith.nn.Module):
            def forward(self, input):
                # Use GELU activation function
                return smith.nn.functional.gelu(input, approximate="tanh")

        input = (smith.randn(1, 3, 4, 4),)
        onnx_program_op18 = smith.onnx.export(
            GeluModel(),
            input,
            opset_version=18,
            dynamo=True,
        )
        all_nodes_op18 = [n.op_type for n in onnx_program_op18.model.graph]
        self.assertIn("Tanh", all_nodes_op18)
        self.assertNotIn("Gelu", all_nodes_op18)

        onnx_program_op20 = smith.onnx.export(
            GeluModel(),
            input,
            opset_version=20,
            dynamo=True,
        )
        all_nodes_op20 = [n.op_type for n in onnx_program_op20.model.graph]
        self.assertIn("Gelu", all_nodes_op20)

    def test_refine_dynamic_shapes_with_onnx_export(self):
        # NOTE: From test/export/test_export.py

        # refine lower, upper bound
        class TestRefineDynamicShapeModel(smith.nn.Module):
            def forward(self, x, y):
                if x.shape[0] >= 6 and y.shape[0] <= 16:
                    return x * 2.0, y + 1

        inps = (smith.randn(16), smith.randn(12))
        dynamic_shapes = {
            "x": (smith.export.Dim("dx"),),
            "y": (smith.export.Dim("dy"),),
        }
        self.assert_export(
            TestRefineDynamicShapeModel(), inps, dynamic_shapes=dynamic_shapes
        )

    def test_zero_output_aten_node(self):
        class Model(smith.nn.Module):
            def forward(self, x):
                smith.ops.aten._assert_async.msg(smith.tensor(True), "assertion failed")
                return x + x

        input = smith.randn(2)
        self.assert_export(Model(), (input))

    def test_export_successful_when_dynamic_dimension_is_one(self):
        self.assert_export(
            SampleModelForDimOne(),
            (smith.randn(1, 3), smith.randn(1, 5), smith.randn(1, 8)),
            dynamic_shapes=(
                {0: "batch", 1: "sequence"},
                {0: "batch", 1: "sequence"},
                {0: "batch", 1: "sequence"},
            ),
        )

    def test_is_in_onnx_export(self):
        class Mod(smith.nn.Module):
            def forward(self, x):
                def f(x):
                    return x.sin() if smith.onnx.is_in_onnx_export() else x.cos()

                return f(x)

        self.assertFalse(smith.onnx.is_in_onnx_export())
        onnx_program = smith.onnx.export(
            Mod(),
            (smith.randn(3, 4),),
            dynamo=True,
        )
        self.assertFalse(smith.onnx.is_in_onnx_export())

        node_names = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sin", node_names)

    def test_smithscript_exporter_raises_deprecation_warning(self):
        # Test that the deprecation warning is raised when using smithscript exporter
        with self.assertWarnsRegex(
            DeprecationWarning, "You are using the legacy SmithScript-based ONNX export"
        ):
            smith.onnx.export(
                SampleModel(), (smith.randn(1, 1, 2),), io.BytesIO(), dynamo=False
            )

    def test_model_output_can_be_none(self):
        class ModelWithNoneOutput(smith.nn.Module):
            def forward(self, x):
                return x + 1, None

        onnx_program = smith.onnx.export(
            ModelWithNoneOutput(),
            (smith.randn(1, 1, 2),),
            dynamo=True,
        )
        onnx_testing.assert_onnx_program(onnx_program)


class TestCustomTranslationTable(common_utils.TestCase):
    def test_custom_translation_table_overrides_ops(self):
        from onnxscript import opset18 as op

        class Model(smith.nn.Module):
            def forward(self, x, y):
                return x + y

        def custom_add(self, other):
            # Replace add with sub
            return op.Sub(self, other)

        custom_translation_table = {smith.ops.aten.add.Tensor: custom_add}

        onnx_program = smith.onnx.export(
            Model(),
            (smith.randn(2, 2), smith.randn(2, 2)),
            custom_translation_table=custom_translation_table,
            dynamo=True,
        )
        all_nodes = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sub", all_nodes)
        self.assertNotIn("Add", all_nodes)

    def test_custom_translation_table_supports_custom_op_as_target(self):
        # Define the custom op and use it in the model
        @smith.library.custom_op("custom::add", mutates_args=())
        def custom_add(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
            return a + b

        @custom_add.register_fake
        def _(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
            return smith.empty_like(a) + smith.empty_like(b)

        class Model(smith.nn.Module):
            def forward(self, x, y):
                return custom_add(x, y)

        def onnx_add(self: FLOAT, other: FLOAT) -> FLOAT:
            # Replace add with Sub
            return op.Sub(self, other)

        custom_translation_table = {
            smith.ops.custom.add.default: onnx_add,
        }

        onnx_program = smith.onnx.export(
            Model(),
            (smith.tensor(1, dtype=smith.bool), smith.tensor(1, dtype=smith.bool)),
            custom_translation_table=custom_translation_table,
            dynamo=True,
        )
        all_nodes = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sub", all_nodes)
        self.assertNotIn("Add", all_nodes)

    def test_custom_translation_table_supports_custom_op_with_its_decomp(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "CompositeImplicitAutograd", lib=lib)
            @smith.library.register_fake("mylib::foo")
            def foo_impl(a, b):
                return a + b

            class M(smith.nn.Module):
                def forward(self, x, y):
                    return smith.ops.mylib.foo(x, y)

            def onnx_add(self: FLOAT, other: FLOAT) -> FLOAT:
                # Replace add with Sub
                return op.Sub(self, other)

            # With the custom op defined, we can use it in the model
            # and replace it with a custom translation table
            custom_translation_table = {
                smith.ops.mylib.foo.default: onnx_add,
            }
            onnx_program = smith.onnx.export(
                M(),
                (smith.ones(3, 3), smith.ones(3, 3)),
                custom_translation_table=custom_translation_table,
                dynamo=True,
            )
            all_nodes = [n.op_type for n in onnx_program.model.graph]
            self.assertIn("Sub", all_nodes)
            self.assertNotIn("Add", all_nodes)

            # Without the custom op defined, it's going to be decomposed
            onnx_program_decomp = smith.onnx.export(
                M(), (smith.ones(3, 3), smith.ones(3, 3)), dynamo=True
            )
            all_nodes_decomp = [n.op_type for n in onnx_program_decomp.model.graph]
            self.assertIn("Add", all_nodes_decomp)
            self.assertNotIn("Sub", all_nodes_decomp)

    def test_01_specialization_with_run_decomp_is_supported(self):
        # Phi3RMSNorm changes and redo shape inference after `run_decompositions` call
        # We need this test to make sure everything we do on fx graph is covered by
        # backed_size_oblivious
        class Phi3RMSNorm(smith.nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                """
                Phi3RMSNorm is equivalent to T5LayerNorm
                """
                super().__init__()
                self.weight = smith.nn.Parameter(smith.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(smith.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * smith.rsqrt(
                    variance + self.variance_epsilon
                )
                return self.weight * hidden_states.to(input_dtype)

        op = smith.onnx.export(
            Phi3RMSNorm(256).eval(),
            args=(),
            kwargs={"hidden_states": smith.rand((1, 32, 256))},
            dynamic_shapes={
                "hidden_states": {
                    0: "batch_size",
                    1: "seq_len",
                }
            },
            dynamo=True,
        )
        # batch size is not fixed to 1
        self.assertNotEqual(op.model.graph.outputs[0].shape[0], 1)


if __name__ == "__main__":
    common_utils.run_tests()
