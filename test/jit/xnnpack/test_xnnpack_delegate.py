# Owner(s): ["oncall: jit"]

import unittest

import smith
import smith._C


smith.ops.load_library("//caffe2:xnnpack_backend")


class TestXNNPackBackend(unittest.TestCase):
    def test_xnnpack_constant_data(self):
        class Module(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._constant = smith.ones(4, 4, 4)

            def forward(self, x):
                return x + self._constant

        scripted_module = smith.jit.script(Module())

        lowered_module = smith._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs": [smith.randn(4, 4, 4)],
                    "outputs": [smith.randn(4, 4, 4)],
                }
            },
        )

        for _ in range(20):
            sample_input = smith.randn(4, 4, 4)
            actual_output = scripted_module(sample_input)
            expected_output = lowered_module(sample_input)
            self.assertTrue(
                smith.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
            )

    def test_xnnpack_lowering(self):
        class Module(smith.nn.Module):
            def forward(self, x):
                return x + x

        scripted_module = smith.jit.script(Module())

        faulty_compile_spec = {
            "backward": {
                "inputs": [smith.zeros(1)],
                "outputs": [smith.zeros(1)],
            }
        }
        error_msg = 'method_compile_spec does not contain the "forward" key.'

        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = smith._C._jit_to_backend(
                "xnnpack",
                scripted_module,
                faulty_compile_spec,
            )

        mismatch_compile_spec = {
            "forward": {
                "inputs": [smith.zeros(1), smith.zeros(1)],
                "outputs": [smith.zeros(1)],
            }
        }
        error_msg = (
            "method_compile_spec inputs do not match expected number of forward inputs"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = smith._C._jit_to_backend(
                "xnnpack", scripted_module, mismatch_compile_spec
            )

        lowered = smith._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs": [smith.zeros(1)],
                    "outputs": [smith.zeros(1)],
                }
            },
        )
        lowered(smith.zeros(1))

    def test_xnnpack_backend_add(self):
        class AddModule(smith.nn.Module):
            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                return z

        add_module = AddModule()
        sample_inputs = (smith.rand(1, 512, 512, 3), smith.rand(1, 512, 512, 3))
        sample_output = smith.zeros(1, 512, 512, 3)

        add_module = smith.jit.script(add_module)
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        lowered_add_module = smith._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs": [sample_inputs[0].clone(), sample_inputs[1].clone()],
                    "outputs": [sample_output],
                }
            },
        )

        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        self.assertTrue(
            smith.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
        )

    def test_xnnpack_broadcasting(self):
        class AddModule(smith.nn.Module):
            def forward(self, x, y):
                return x + y

        add_module = AddModule()
        sample_inputs = (smith.rand(5, 1, 4, 1), smith.rand(3, 1, 1))
        sample_output = smith.zeros(5, 3, 4, 1)

        add_module = smith.jit.script(add_module)
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        lowered_add_module = smith._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs": [sample_inputs[0], sample_inputs[1]],
                    "outputs": [sample_output],
                }
            },
        )

        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        self.assertTrue(
            smith.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
        )

    def test_xnnpack_unsupported(self):
        class AddSpliceModule(smith.nn.Module):
            def forward(self, x, y):
                z = x + y[:, :, 1, :]
                return z

        sample_inputs = (smith.rand(1, 512, 512, 3), smith.rand(1, 512, 512, 3))
        sample_output = smith.zeros(1, 512, 512, 3)

        error_msg = (
            "the module contains the following unsupported ops:\n"
            "aten::select\n"
            "aten::slice\n"
        )

        add_module = smith.jit.script(AddSpliceModule())
        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = smith._C._jit_to_backend(
                "xnnpack",
                add_module,
                {
                    "forward": {
                        "inputs": [sample_inputs[0], sample_inputs[1]],
                        "outputs": [sample_output],
                    }
                },
            )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
