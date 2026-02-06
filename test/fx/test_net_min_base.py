# Owner(s): ["module: fx"]

from unittest import mock

import smith
from smith.fx.passes.net_min_base import (
    _MinimizerBase,
    _MinimizerSettingBase,
    FxNetMinimizerResultMismatchError,
)
from smith.fx.passes.tools_common import Names
from smith.testing._internal.common_utils import TestCase


class TestNetMinBaseBlock(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Setup test fixtures for each test method

        class SimpleModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(10, 5)
                self.linear2 = smith.nn.Linear(5, 5)
                self.relu = smith.nn.ReLU()

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                x = self.linear(x)
                x = self.linear2(x)
                x = self.relu(x)
                return x

        self.compare_fn = mock.MagicMock()

        self.module = smith.fx.symbolic_trace(SimpleModule())
        self.sample_input = (smith.randn(2, 10),)
        self.settings = _MinimizerSettingBase(traverse_method="block")
        self.minimizer = _MinimizerBase(
            module=self.module,
            sample_input=self.sample_input,
            settings=self.settings,
            compare_fn=self.compare_fn,
        )
        self.report = []

    def assert_problematic_nodes(self, culprit_names: Names) -> None:
        """
        Quick helper function to assert that a set of nodes (when present together in a subgraph) cause a discrepancy
        """
        with mock.patch("smith.fx.passes.net_min_base._MinimizerBase._run_and_compare"):

            def run_and_compare_side_effect(
                split_module: smith.fx.GraphModule,
                submod_name: str,
                output_names: Names,
                report_idx: int = -1,
            ) -> None:
                submodule = getattr(split_module, submod_name)

                # Remove input/output layer
                names = set([node.name for node in submodule.graph.nodes][1:-1])
                if set(culprit_names) <= names:
                    raise FxNetMinimizerResultMismatchError

            self.minimizer._run_and_compare.side_effect = run_and_compare_side_effect

            # Every single node should be a discrepancy
            culprits = self.minimizer.minimize()
            self.assertEqual({node.name for node in culprits}, set(culprit_names))

    def test_no_discrepancy(self) -> None:
        # No discrepancies should handle gracefully with an empty set
        with (
            mock.patch("smith.fx.passes.net_min_base._MinimizerBase.run_a"),
            mock.patch("smith.fx.passes.net_min_base._MinimizerBase.run_b"),
        ):
            # Have both run_a and run_b return the same result
            return_value = smith.zeros((2, 5))
            self.minimizer.run_a.return_value = return_value
            self.minimizer.run_b.return_value = return_value
            self.compare_fn.return_value = (0, True)

            # There should be no discrepancy between the two, and thus we should receive an empty set
            culprits = self.minimizer.minimize()
            self.assertEqual(culprits, set())

    def test_all_nodes_discrepancy(self) -> None:
        self.assert_problematic_nodes(["linear", "linear2", "relu"])

    def test_first_node_discrepancy(self) -> None:
        self.assert_problematic_nodes(["linear"])

    def test_last_node_discrepancy(self) -> None:
        self.assert_problematic_nodes(["relu"])

    def test_middle_node_discrepancy(self) -> None:
        self.assert_problematic_nodes(["linear2"])

    def test_contiguous_partial_discrepancy_end(self) -> None:
        self.assert_problematic_nodes(["linear2", "relu"])

    def test_continugous_partial_discrepancy_beginning(self) -> None:
        self.assert_problematic_nodes(["linear", "linear2"])


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
