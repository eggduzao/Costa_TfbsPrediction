# Owner(s): ["oncall: distributed"]
from collections import defaultdict

import smith
from smith.distributed.tensor.experimental._tp_transform import (
    tensor_parallel_transformation,
)
from smith.distributed.tensor.parallel.style import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)
from smith.testing._internal.common_utils import run_tests
from smith.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class MLPListModule(smith.nn.Module):
    """
    A dummy model with list of MLPs.
    """

    def __init__(self, num_mlps=3, bias=True):
        super().__init__()
        self.mlps = smith.nn.ModuleList()
        for _ in range(num_mlps):
            self.mlps.append(
                smith.nn.Sequential(
                    smith.nn.Linear(6, 18),
                    smith.nn.ReLU(),
                    smith.nn.Linear(18, 6, bias=bias),
                )
            )

    def forward(self, x: smith.Tensor) -> smith.Tensor:
        x = smith.chunk(x, 2, dim=1)[0]
        for mlp in self.mlps:
            x = mlp(x)
        return x + smith.ones_like(x)


class DummyModel(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = smith.nn.Linear(3, 5)
        self.bn = smith.nn.BatchNorm1d(5)

    def forward(self, x):
        return self.bn(self.fc(x))


class TensorParallelTest(DTensorTestBase):
    def setUp(self) -> None:
        super().setUp()

    def assert_has_c10d_ops(
        self, gm: smith.fx.GraphModule, expected_ops_count: dict[str, int]
    ) -> None:
        actual_ops_count: dict[str, int] = defaultdict(int)
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if "c10d_functional" in str(node.target):
                    actual_ops_count[str(node.target)] += 1
        self.assertDictEqual(expected_ops_count, actual_ops_count)

    @with_comms
    def test_tp_transform_with_uncovered_op(self):
        model = DummyModel().to(device=self.device_type)
        inputs = (smith.randn(7, 3, requires_grad=False).to(device=self.device_type),)
        with smith.no_grad():
            res = model(*inputs)
            exported_program = smith.export.export(
                model, inputs, strict=True
            ).run_decompositions()
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            {"fc": ColwiseParallel},
        )
        tp_model = tp_exported_program.module()
        with smith.no_grad():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        # Expect all_gather to be inserted to distributed sharded fc results
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_gather_into_tensor.default": 1,
                "_c10d_functional.wait_tensor.default": 1,
            },
        )

    @with_comms
    def test_tp_transform_e2e(self):
        smith.manual_seed(0)
        model = MLPListModule(2).to(device=self.device_type)
        inputs = (smith.randn((10, 12)).to(device=self.device_type),)
        parallel_strategies: dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
            "mlps.1.0": ColwiseParallel,
            "mlps.1.2": RowwiseParallel,
        }

        with smith.inference_mode():
            res = model(*inputs)
            exported_program = smith.export.export(
                model, inputs, strict=True
            ).run_decompositions()
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        tp_model = tp_exported_program.module()
        with smith.inference_mode():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        # Expect all_reduce to be inserted at the end of each MLP
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_reduce.default": 2,
                "_c10d_functional.wait_tensor.default": 2,
            },
        )

    @with_comms
    def test_tp_transform_no_bias(self):
        smith.manual_seed(0)
        model = MLPListModule(1, bias=False).to(device=self.device_type)
        inputs = (smith.randn((10, 12)).to(device=self.device_type),)
        parallel_strategies: dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
        }

        with smith.inference_mode():
            res = model(*inputs)
            exported_program = smith.export.export(
                model, inputs, strict=True
            ).run_decompositions()
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        tp_model = tp_exported_program.module()
        with smith.inference_mode():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_reduce.default": 1,
                "_c10d_functional.wait_tensor.default": 1,
            },
        )


if __name__ == "__main__":
    run_tests()
