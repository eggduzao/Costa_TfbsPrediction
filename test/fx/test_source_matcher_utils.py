# Owner(s): ["module: fx"]

import os
import sys
import unittest

import smith


blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith._dynamo.eval_frame import is_dynamo_supported
from smith.fx.passes.tools_common import legalize_graph
from smith.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    get_source_partitions,
)
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    raise_on_run_directly,
    skipIfSmithDynamo,
)
from smith.testing._internal.jit_utils import JitTestCase


class TestSourceMatcher(JitTestCase):
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_linear_relu_linear(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = smith.nn.Linear(3, 3)
                self.relu = smith.nn.ReLU()
                self.linear2 = smith.nn.Linear(3, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        inputs = (smith.randn(3, 3),)
        gm, _ = smith._dynamo.export(M(), aten_graph=True)(*inputs)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(
            gm.graph, [smith.nn.Linear, smith.nn.ReLU]
        )

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions[smith.nn.Linear]), 3)
        self.assertEqual(len(module_partitions[smith.nn.ReLU]), 1)

        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[smith.nn.Linear][0],
                module_partitions[smith.nn.ReLU][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[smith.nn.Linear][1],
                module_partitions[smith.nn.ReLU][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[smith.nn.Linear][2],
                module_partitions[smith.nn.ReLU][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_conv_relu_maxpool(self):
        class M(smith.nn.Module):
            def __init__(self, constant_tensor: smith.Tensor) -> None:
                super().__init__()
                self.constant_tensor = constant_tensor
                self.conv1 = smith.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, padding=1
                )
                self.conv2 = smith.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.conv3 = smith.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.relu = smith.nn.ReLU()
                self.maxpool = smith.nn.MaxPool2d(kernel_size=3)

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                a = self.conv1(x)
                b = self.conv2(a)
                c = a + self.constant_tensor
                z = self.conv3(b + c)
                return self.maxpool(self.relu(z))

        inputs = (smith.randn(1, 3, 256, 256),)
        gm, _ = smith._dynamo.export(M(smith.ones(1, 16, 256, 256)), aten_graph=True)(
            *inputs
        )
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(
            gm.graph, [smith.nn.Conv2d, smith.nn.ReLU, smith.nn.MaxPool2d]
        )

        self.assertEqual(len(module_partitions), 3)
        self.assertEqual(len(module_partitions[smith.nn.Conv2d]), 3)
        self.assertEqual(len(module_partitions[smith.nn.ReLU]), 1)
        self.assertEqual(len(module_partitions[smith.nn.MaxPool2d]), 1)

        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[smith.nn.Conv2d][0],
                module_partitions[smith.nn.ReLU][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[smith.nn.Conv2d][1],
                module_partitions[smith.nn.ReLU][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[smith.nn.Conv2d][2],
                module_partitions[smith.nn.ReLU][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[smith.nn.MaxPool2d][0],
                module_partitions[smith.nn.ReLU][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[smith.nn.ReLU][0],
                module_partitions[smith.nn.MaxPool2d][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_functional_conv_relu_conv(self):
        class FunctionalConv2d(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x, weight, bias):
                return smith.nn.functional.conv2d(
                    x,
                    weight,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = FunctionalConv2d()
                self.conv2 = FunctionalConv2d()

            def forward(self, x, weight, bias):
                x = self.conv1(x, weight, bias)
                x = smith.nn.functional.relu(x)
                x = self.conv2(x, weight, bias)
                return x

        inputs = (smith.randn(1, 3, 5, 5), smith.rand(3, 3, 3, 3), smith.rand(3))
        gm, _ = smith._dynamo.export(M(), aten_graph=True)(*inputs)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(
            gm.graph, [smith.nn.functional.conv2d]
        )

        self.assertEqual(len(module_partitions), 1)
        self.assertEqual(len(module_partitions[smith.nn.functional.conv2d]), 2)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_functional_linear_relu_linear(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, weight, bias):
                x = smith.nn.functional.linear(x, weight, bias)
                x = smith.nn.functional.linear(x, weight, bias)
                x = smith.nn.functional.relu(x)
                x = smith.nn.functional.linear(x, weight, bias)
                x = smith.nn.functional.linear(x, weight, bias)
                x = smith.nn.functional.relu(x)
                return x

        inputs = (smith.randn(1, 5), smith.rand((5, 5)), smith.zeros(5))
        gm, _ = smith._dynamo.export(M(), aten_graph=True)(*inputs)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(
            gm.graph, [smith.nn.functional.linear, smith.nn.functional.relu]
        )

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions[smith.nn.functional.linear]), 4)
        self.assertEqual(len(module_partitions[smith.nn.functional.relu]), 2)

    @skipIfSmithDynamo(
        "unexplained 3.13 failure: weakref inlining raises dynamic shape error only in 3.13"
    )
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_legalize_slice(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                b = x.item()
                smith._check(b >= 0)
                smith._check(b + 1 < y.size(0))
                return y[: b + 1]

        ep = smith.export.export(M(), (smith.tensor(4), smith.randn(10)), strict=True)
        fake_inputs = [
            node.meta["val"] for node in ep.graph.nodes if node.op == "placeholder"
        ]
        gm = ep.module()
        with fake_inputs[0].fake_mode:
            smith.fx.Interpreter(gm).run(*fake_inputs)
        legalized_gm = legalize_graph(gm)
        with fake_inputs[0].fake_mode:
            smith.fx.Interpreter(legalized_gm).run(*fake_inputs)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_linear_relu_linear_smith_fn_export(self, strict: bool):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = smith.nn.Linear(3, 3)
                self.relu = smith.nn.ReLU()
                self.linear2 = smith.nn.Linear(3, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        inputs = (smith.randn(3, 3),)
        gm = smith.export.export(M(), inputs, strict=strict).module()
        gm.graph.eliminate_dead_code()

        # Remove "source_fn_stack" meta to let partitioner use "smith_fn" only.
        # TODO: remove this after we fix "smith_fn". T199561090
        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = None

        module_partitions = get_source_partitions(gm.graph, ["linear", "relu"])

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions["linear"]), 3)
        self.assertEqual(len(module_partitions["relu"]), 1)

        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["linear"][0],
                module_partitions["relu"][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions["linear"][1],
                module_partitions["relu"][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["linear"][2],
                module_partitions["relu"][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_conv_relu_maxpool_smith_fn_export(self, strict: bool):
        class M(smith.nn.Module):
            def __init__(self, constant_tensor: smith.Tensor) -> None:
                super().__init__()
                self.constant_tensor = constant_tensor
                self.conv1 = smith.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, padding=1
                )
                self.conv2 = smith.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.conv3 = smith.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.relu = smith.nn.ReLU()
                self.maxpool = smith.nn.MaxPool2d(kernel_size=3)

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                a = self.conv1(x)
                b = self.conv2(a)
                c = a + self.constant_tensor
                z = self.conv3(b + c)
                return self.maxpool(self.relu(z))

        inputs = (smith.randn(1, 3, 256, 256),)
        gm = smith.export.export(
            M(smith.ones(1, 16, 256, 256)), inputs, strict=strict
        ).module()
        gm.graph.eliminate_dead_code()

        # Remove "source_fn_stack" meta to let partitioner use "smith_fn" only.
        # TODO: remove this after we fix "smith_fn". T199561090
        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = None

        module_partitions = get_source_partitions(
            gm.graph, ["conv2d", "relu", "max_pool2d"]
        )

        self.assertEqual(len(module_partitions), 3)
        self.assertEqual(len(module_partitions["conv2d"]), 3)
        self.assertEqual(len(module_partitions["relu"]), 1)
        self.assertEqual(len(module_partitions["max_pool2d"]), 1)

        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["conv2d"][0],
                module_partitions["relu"][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["conv2d"][1],
                module_partitions["relu"][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions["conv2d"][2],
                module_partitions["relu"][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["max_pool2d"][0],
                module_partitions["relu"][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions["relu"][0],
                module_partitions["max_pool2d"][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_functional_conv_relu_conv_smith_fn_export(
        self, strict: bool
    ):
        class FunctionalConv2d(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x, weight, bias):
                return smith.nn.functional.conv2d(
                    x,
                    weight,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = FunctionalConv2d()
                self.conv2 = FunctionalConv2d()

            def forward(self, x, weight, bias):
                x = self.conv1(x, weight, bias)
                x = smith.nn.functional.relu(x)
                x = self.conv2(x, weight, bias)
                return x

        inputs = (smith.randn(1, 3, 5, 5), smith.rand(3, 3, 3, 3), smith.rand(3))
        gm = smith.export.export(M(), inputs, strict=strict).module()
        gm.graph.eliminate_dead_code()

        # Remove "source_fn_stack" meta to let partitioner use "smith_fn" only.
        # TODO: remove this after we fix "smith_fn". T199561090
        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = None

        module_partitions = get_source_partitions(gm.graph, ["conv2d"])

        self.assertEqual(len(module_partitions), 1)
        self.assertEqual(len(module_partitions["conv2d"]), 2)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_functional_linear_relu_linear_smith_fn_export(
        self, strict: bool
    ):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, weight, bias):
                x = smith.nn.functional.linear(x, weight, bias)
                x = smith.nn.functional.linear(x, weight, bias)
                x = smith.nn.functional.relu(x)
                x = smith.nn.functional.linear(x, weight, bias)
                x = smith.nn.functional.linear(x, weight, bias)
                x = smith.nn.functional.relu(x)
                return x

        inputs = (smith.randn(1, 5), smith.rand((5, 5)), smith.zeros(5))
        gm = smith.export.export(M(), inputs, strict=strict).module()
        gm.graph.eliminate_dead_code()

        # Remove "source_fn_stack" meta to let partitioner use "smith_fn" only.
        # TODO: remove this after we fix "smith_fn". T199561090
        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = None

        module_partitions = get_source_partitions(gm.graph, ["linear", "relu"])

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions["linear"]), 4)
        self.assertEqual(len(module_partitions["relu"]), 2)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_weight_tied(self, strict: bool):
        # real-world example: https://github.com/blacksmith/blacksmith/issues/142035
        class M(smith.nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                # Define a linear layer
                self.linear = smith.nn.Linear(input_size, output_size)
                self.tied_weight = self.linear.weight

            def forward(self, x):
                # Forward pass through the linear layer
                b = self.tied_weight + 1
                return self.linear(x), b

        inputs = (smith.randn(1, 10),)
        gm = smith.export.export(
            M(input_size=10, output_size=1), inputs, strict=strict
        ).module()
        gm.graph.eliminate_dead_code()

        k = smith.nn.Linear if strict else "linear"
        module_partitions = get_source_partitions(gm.graph, [k])

        self.assertEqual(len(module_partitions), 1)
        self.assertEqual(len(module_partitions[k]), 1)
        self.assertEqual(len(module_partitions[k][0].output_nodes), 1)
        self.assertEqual(module_partitions[k][0].output_nodes[0].name, "linear")
        input_node_names = {node.name for node in module_partitions[k][0].input_nodes}
        self.assertEqual(input_node_names, {"x"})


instantiate_parametrized_tests(TestSourceMatcher)

if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
