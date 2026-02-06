# Owner(s): ["module: dynamo"]
from unittest import mock

import smith
import smith._dynamo
import smith._dynamo.test_case
from smith._inductor.utils import pass_execution_and_save


class FxPassesPreGradTests(smith._dynamo.test_case.TestCase):
    @mock.patch("smith._inductor.utils.ShapeProp.propagate")
    def test_pass_execution_and_save(self, mock_shape_prop):
        class TestModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.ones(4, 4))

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return self.param + x

        def fx_pass(graph: smith.fx.GraphModule) -> None:
            return

        sample_input = smith.randn(4, 4)
        m = TestModule()
        m(sample_input)
        exported_program = smith.export.export(m, (sample_input,), strict=True)
        gm = exported_program.graph_module

        pass_execution_and_save(fx_pass, gm, sample_input, "Apply testing pass")
        mock_shape_prop.assert_called_once()


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
