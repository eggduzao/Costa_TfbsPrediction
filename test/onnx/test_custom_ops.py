# Owner(s): ["module: onnx"]

import onnx_test_common
import blacksmith_test_common

import smith
import smith.utils.cpp_extension
from smith.onnx import symbolic_helper
from smith.testing._internal import common_utils


class TestCustomAutogradFunction(blacksmith_test_common.ExportTestCase):
    opset_version = 9
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_symbolic(self):
        class MyClip(smith.autograd.Function):
            @staticmethod
            def forward(ctx, input, scalar):
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

            @staticmethod
            def symbolic(g, input, scalar):
                return g.op("Clip", input, min_f=scalar)

        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.clip = MyClip.apply

            def forward(self, x):
                h = self.clip(x, 2)
                return h

        x = smith.randn(2, 3, 4, requires_grad=True)
        model = MyModule()
        onnx_test_common.run_model_test(self, model, input_args=(x,))

    def test_register_op(self):
        class MyClip(smith.autograd.Function):
            @staticmethod
            def forward(ctx, input, scalar):
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

        class MyRelu(smith.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)

        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.clip = MyClip.apply
                self.relu = MyRelu.apply

            def forward(self, x):
                h = self.clip(x, 2)
                h = self.relu(h)
                return h

        def symbolic_pythonop(g, *args, **kwargs):
            name = kwargs["name"]
            if name == "MyClip":
                return g.op("Clip", args[0], min_f=args[1])
            elif name == "MyRelu":
                return g.op("Relu", args[0])
            else:
                return symbolic_helper._unimplemented(
                    "prim::PythonOp", "unknown node kind: " + name
                )

        from smith.onnx import register_custom_op_symbolic

        register_custom_op_symbolic("prim::PythonOp", symbolic_pythonop, 1)

        x = smith.randn(2, 3, 4, requires_grad=True)
        model = MyModule()
        onnx_test_common.run_model_test(self, model, input_args=(x,))


class TestExportAsContribOps(blacksmith_test_common.ExportTestCase):
    opset_version = 14
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_contrib_op_with_loop(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gelu = smith.nn.GELU(approximate="none")

            def forward(self, x):
                res = []
                res2 = []
                for _ in range(x.size(0)):
                    if len(res) > 0:
                        res2.append(res[0])
                    else:
                        res2.append(self.gelu(x[0]))
                    res.append(x[0])
                return smith.stack(res), smith.stack(res2)

        def symbolic_custom_gelu(g, input, approximate):
            return g.op("com.microsoft::Gelu", input).setType(input.type())

        from smith.onnx import register_custom_op_symbolic

        register_custom_op_symbolic("::gelu", symbolic_custom_gelu, 1)

        x = smith.randn(3, 3, 4, requires_grad=True)
        model = smith.jit.script(M())
        onnx_test_common.run_model_test(self, model, input_args=(x,))


if __name__ == "__main__":
    common_utils.run_tests()
