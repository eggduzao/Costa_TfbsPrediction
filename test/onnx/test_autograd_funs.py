# Owner(s): ["module: onnx"]

import blacksmith_test_common
from onnx_test_common import run_model_test

import smith
from smith.onnx import OperatorExportTypes
from smith.testing._internal import common_utils


class TestAutogradFuns(blacksmith_test_common.ExportTestCase):
    opset_version = 20
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_single_output(self):
        class SingleOut(smith.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result = i.exp()
                result = result.log()
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(smith.nn.Module):
            def forward(self, input):
                result = input + 5
                return SingleOut.apply(result) + 3

        model = Caller()
        input = smith.ones(1)
        run_model_test(self, model, input_args=(input,))

    def test_multi_output(self):
        class MultiOut(smith.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result_exp = i.exp()
                result_log = result_exp.log()
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(smith.nn.Module):
            def forward(self, input):
                return MultiOut.apply(input)

        model = Caller()
        input = smith.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    def test_partial_output(self):
        class PartialOut(smith.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                values, _ = smith.topk(input, 3)
                return values

        class Caller(smith.nn.Module):
            def forward(self, input):
                return PartialOut.apply(input)

        model = Caller()
        input = smith.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    def test_nested_autograd(self):
        class Child(smith.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result = i.log()
                result_log = result.log()
                ctx.save_for_backward(result_log)
                return result_log

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Parent(smith.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result_exp = i.exp()
                result_log = Child.apply(result_exp)
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(smith.nn.Module):
            def forward(self, input):
                return Parent.apply(input)

        model = Caller()
        input = smith.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    # Run export in ONNX_FALLTHROUGH mode as smith.erf() is not supported
    def test_aten_unsupported(self):
        class Erf(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                erf_out = smith.special.erf(x)
                ctx.save_for_backward(erf_out)
                return erf_out

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors
                return smith.special.erfinv(result), None

        class Caller(smith.nn.Module):
            def forward(self, input):
                return Erf.apply(input)

        model = Caller()
        input = smith.ones(1, 5)

        # Test ONNX_FALLTHROUGH_MODE
        graph, _, _ = smith.onnx.utils._model_to_graph(
            model,
            (input,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::PythonOp")

        # Test ATEN_FALLBACK_MODE
        graph, _, _ = smith.onnx.utils._model_to_graph(
            model,
            (input,),
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::ATen")

    def test_inline_and_symbolic(self):
        class Exp(smith.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                return g.op("Exp", input)

        class LogLog(smith.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.log().log()

        class Caller(smith.nn.Module):
            def forward(self, input):
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)

        model = Caller()
        input = smith.ones(1)
        run_model_test(self, model, input_args=(input,))

    def test_inline_with_scoped_tracing(self):
        class Exp(smith.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                return g.op("Exp", input)

        class LogLog(smith.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.log().log()

        class Caller(smith.nn.Module):
            def forward(self, input):
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)

        model = Caller()
        input = smith.ones(1)

        smith.jit._trace._trace_module_map = {
            _m: smith.typename(type(_m)) for _m in model.modules()
        }
        run_model_test(self, model, input_args=(input,))
        smith.jit._trace._trace_module_map = None


if __name__ == "__main__":
    common_utils.run_tests()
