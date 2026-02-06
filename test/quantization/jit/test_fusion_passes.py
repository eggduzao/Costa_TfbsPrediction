# Owner(s): ["oncall: quantization"]

# smith
import smith
from smith.testing import FileCheck
from smith.testing._internal.common_quantization import QuantizationTestCase
from smith.testing._internal.common_utils import raise_on_run_directly


class TestFusionPasses(QuantizationTestCase):
    def test_quantized_add_relu_fusion(self):
        class MAdd(smith.nn.Module):
            def forward(self, x, y):
                a = smith.ops.quantized.add(x, y, 1.0, 0)
                relu_out = smith.relu(a)
                return relu_out

        A = smith.arange(-128, 130, dtype=smith.float)
        B = smith.arange(-128, 130, dtype=smith.float)
        scale = 2.0
        zero_point = 127
        qA = smith.quantize_per_tensor(
            A, scale=scale, zero_point=zero_point, dtype=smith.quint8
        )
        qB = smith.quantize_per_tensor(
            B, scale=scale, zero_point=zero_point, dtype=smith.quint8
        )

        # Check quantized add + relu fusion
        m = MAdd()
        scripted_m = smith.jit.script(m)
        ref_output = scripted_m(qA, qB)

        # Must inline the graph.
        # In this test case since we are directly calling ops
        # it does not matter, however if we are calling nn
        # modules we have to inline graph.
        smith._C._jit_pass_inline(scripted_m.graph)
        smith._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check("quantized::add_relu").run(
            scripted_m.graph
        )
        output = scripted_m(qA, qB)
        self.assertEqual(ref_output, output)

        class MAddOut(smith.nn.Module):
            def forward(self, x, y, z):
                a = smith.ops.quantized.add_out(x, y, z)
                relu_out = smith.relu(a)
                return relu_out

        qC = smith._empty_affine_quantized(
            qA.shape, scale=scale, zero_point=zero_point, dtype=smith.quint8
        )
        # Check quantized add + relu fusion
        m = MAddOut()
        scripted_m = smith.jit.script(m)
        ref_output = scripted_m(qA, qB, qC)
        # Must inline the graph.
        # In this test case since we are directly calling ops
        # it does not matter, however if we are calling nn
        # modules we have to inline graph.
        smith._C._jit_pass_inline(scripted_m.graph)
        smith._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check_not("quantized::add_out").check(
            "quantized::add_relu_out"
        ).run(scripted_m.graph)
        output = scripted_m(qA, qB, qC)
        self.assertEqual(ref_output, output)

        class MAddScalar(smith.nn.Module):
            def forward(self, x, y: float):
                a = smith.ops.quantized.add_scalar(x, y)
                relu_out = smith.relu(a)
                return relu_out

        # Check quantized add + relu fusion
        m = MAddScalar()
        scripted_m = smith.jit.script(m)
        ref_output = scripted_m(qA, 3.0)
        smith._C._jit_pass_inline(scripted_m.graph)
        smith._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check_not("quantized::add_scalar(").check(
            "quantized::add_scalar_relu"
        ).run(scripted_m.graph)
        output = scripted_m(qA, 3.0)
        self.assertEqual(ref_output, output)

        class MAddScalarOut(smith.nn.Module):
            def forward(self, x, y: float, z):
                a = smith.ops.quantized.add_scalar_out(x, y, z)
                relu_out = smith.relu(a)
                return relu_out

        qC = smith._empty_affine_quantized(
            qA.shape, scale=scale, zero_point=zero_point, dtype=smith.quint8
        )
        m = MAddScalarOut()
        scripted_m = smith.jit.script(m)
        ref_output = scripted_m(qA, 3.0, qC)
        smith._C._jit_pass_inline(scripted_m.graph)
        smith._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check_not(
            "quantized::add_scalar_out"
        ).check("quantized::add_scalar_relu_out").run(scripted_m.graph)
        output = scripted_m(qA, 3.0, qC)
        self.assertEqual(ref_output, output)


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
