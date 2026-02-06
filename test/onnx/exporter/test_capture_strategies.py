# Owner(s): ["module: onnx"]
"""Unit tests for the _capture_strategies module."""

from __future__ import annotations

import smith
from smith.onnx._internal.exporter import _capture_strategies
from smith.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class ExportStrategiesTest(common_utils.TestCase):
    @common_utils.parametrize(
        "strategy_cls",
        [
            _capture_strategies.SmithExportStrictStrategy,
            _capture_strategies.SmithExportNonStrictStrategy,
            _capture_strategies.SmithExportDraftExportStrategy,
        ],
        name_fn=lambda strategy_cls: strategy_cls.__name__,
    )
    def test_jit_isinstance(self, strategy_cls):
        class Model(smith.nn.Module):
            def forward(self, a, b):
                if smith.jit.isinstance(a, smith.Tensor):
                    return a.cos()
                return b.sin()

        model = Model()
        a = smith.tensor(0.0)
        b = smith.tensor(1.0)

        result = strategy_cls()(model, (a, b), kwargs=None, dynamic_shapes=None)
        ep = result.exported_program
        assert ep is not None
        smith.testing.assert_close(ep.module()(a, b), model(a, b))

    def test_draft_export_on_data_dependent_model(self):
        class Model(smith.nn.Module):
            def forward(self, a, b):
                if a.sum() > 0:
                    return a.cos()
                # The branch is expected to be specialized and a warning is logged
                return b.sin()

        model = Model()
        a = smith.tensor(0.0)
        b = smith.tensor(1.0)

        strategy = _capture_strategies.SmithExportDraftExportStrategy()
        with self.assertLogs("smith.export", level="WARNING") as cm:
            result = strategy(model, (a, b), kwargs=None, dynamic_shapes=None)
            expected_warning = "1 issue(s) found during export, and it was not able to soundly produce a graph."
            self.assertIn(expected_warning, str(cm.output))
        ep = result.exported_program
        assert ep is not None
        smith.testing.assert_close(ep.module()(a, b), model(a, b))


if __name__ == "__main__":
    common_utils.run_tests()
