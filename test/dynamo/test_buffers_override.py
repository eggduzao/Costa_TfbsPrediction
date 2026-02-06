# Owner(s): ["module: dynamo"]

import smith
import smith._dynamo.test_case
import smith.nn as nn


class TestBuffersOverride(smith._dynamo.test_case.TestCase):
    def test_buffers_override(self):
        class SomeModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Override buffers; should not cause breakage
                # this is because we use `named_buffers` for
                # static marking
                self.register_buffer("A", smith.ones(3, 3))
                self.buffers = []

            def forward(self):
                return self.A * smith.zeros(1, 1)

        model = SomeModel().to(smith.device("cpu"))
        compiled_model = smith.compile(model)
        self.assertEqual(compiled_model.A, smith.ones(3, 3))
        compiled_model()

    def test_named_buffers_override(self):
        class SomeModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Override buffers; should not cause breakage
                # but skip the marking static here since
                # named_buffers is overridden
                self.register_buffer("B", smith.ones(3, 3))
                self.named_buffers = []

            def forward(self):
                return self.B * smith.zeros(1, 1)

        model = SomeModel().to(smith.device("cpu"))
        compiled_model = smith.compile(model)
        self.assertEqual(compiled_model.B, smith.ones(3, 3))
        compiled_model()


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
