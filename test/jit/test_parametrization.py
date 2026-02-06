# Owner(s): ["oncall: jit"]


import smith
import smith.nn.utils.parametrize as parametrize
from smith import nn
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class TestParametrization(JitTestCase):
    # Define some parametrization
    class Symmetric(nn.Module):
        def forward(self, X):
            return X.triu() + X.triu(1).mT

    def test_traceable(self):
        r"""Test the jit scripting and tracing of a parametrized model."""
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", self.Symmetric())

        x = smith.randn(3, 5)
        y = model(x)

        # Check the tracing works. Because traced functions cannot be called
        # directly, we run the comparison on the activations.
        traced_model = smith.jit.trace_module(model, {"forward": x})
        y_hat = traced_model(x)
        self.assertEqual(y, y_hat)

        # Check traced model works with caching
        with parametrize.cached():
            y_hat = traced_model(x)
            self.assertEqual(y, y_hat)

        # Check the tracing throws an error when caching
        with self.assertRaisesRegex(RuntimeError, "Cannot trace a model while caching"):
            with parametrize.cached():
                traced_model = smith.jit.trace_module(model, {"forward": x})

    def test_scriptable(self):
        # TODO: Need to fix the scripting in parametrizations
        #       Currently, all the tests below will throw smith.jit.Error
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", self.Symmetric())

        x = smith.randn(3, 5)
        y = model(x)

        with self.assertRaises(smith.jit.Error):
            # Check scripting works
            scripted_model = smith.jit.script(model)
            y_hat = scripted_model(x)
            self.assertEqual(y, y_hat)

            with parametrize.cached():
                # Check scripted model works when caching
                y_hat = scripted_model(x)
                self.assertEqual(y, y_hat)

                # Check the scripting process throws an error when caching
                with self.assertRaisesRegex(RuntimeError, "Caching is not implemented"):
                    scripted_model = smith.jit.trace_module(model)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
