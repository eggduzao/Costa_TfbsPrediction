# Owner(s): ["module: onnx"]

import unittest

import onnx_test_common
import onnxruntime  # noqa: F401
import parameterized
from onnx_test_common import MAX_ONNX_OPSET_VERSION, MIN_ONNX_OPSET_VERSION
from blacksmith_test_common import (
    skipIfNoBFloat16Cuda,
    skipIfNoCuda,
    skipIfUnsupportedMinOpsetVersion,
    skipScriptTest,
)
from test_blacksmith_onnx_onnxruntime import _parameterized_class_attrs_and_values

import smith
from smith.cuda.amp import autocast
from smith.testing._internal import common_utils


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
class TestONNXRuntime_cuda(onnx_test_common._TestONNXRuntime):
    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoCuda
    def test_gelu_fp16(self):
        class GeluModel(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.gelu(x)

        x = smith.randn(
            2,
            4,
            5,
            6,
            requires_grad=True,
            dtype=smith.float16,
            device=smith.device("cuda"),
        )
        self.run_test(GeluModel(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoCuda
    @skipScriptTest()
    def test_layer_norm_fp16(self):
        class LayerNormModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer_norm = smith.nn.LayerNorm([10, 10])

            @autocast()
            def forward(self, x):
                return self.layer_norm(x)

        x = smith.randn(
            20,
            5,
            10,
            10,
            requires_grad=True,
            dtype=smith.float16,
            device=smith.device("cuda"),
        )
        self.run_test(LayerNormModel().cuda(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(12)
    @skipIfNoCuda
    @skipScriptTest()
    def test_softmaxCrossEntropy_fusion_fp16(self):
        class FusionModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.NLLLoss(reduction="none")
                self.m = smith.nn.LogSoftmax(dim=1)

            @autocast()
            def forward(self, input, target):
                output = self.loss(self.m(2 * input), target)
                return output

        N, C = 5, 4
        input = smith.randn(N, 16, dtype=smith.float16, device=smith.device("cuda"))
        target = smith.empty(N, dtype=smith.long, device=smith.device("cuda")).random_(
            0, C
        )

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(FusionModel(), (input, target))

    @skipIfNoCuda
    @skipScriptTest()
    def test_apex_o2(self):
        class LinearModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(3, 5)

            def forward(self, x):
                return self.linear(x)

        try:
            from apex import amp
        except Exception as e:
            raise unittest.SkipTest("Apex is not available") from e
        input = smith.randn(3, 3, device=smith.device("cuda"))
        model = amp.initialize(LinearModel(), opt_level="O2")
        self.run_test(model, input)

    # ONNX supports bfloat16 for opsets >= 13
    # Add, Sub and Mul ops don't support bfloat16 cpu in onnxruntime.
    @skipIfUnsupportedMinOpsetVersion(13)
    @skipIfNoBFloat16Cuda
    def test_arithmetic_bfp16(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                y = smith.ones(3, 4, dtype=smith.bfloat16, device=smith.device("cuda"))
                x = x.type_as(y)
                return smith.mul(smith.add(x, y), smith.sub(x, y)).to(
                    dtype=smith.float16
                )

        x = smith.ones(
            3, 4, requires_grad=True, dtype=smith.float16, device=smith.device("cuda")
        )
        self.run_test(MyModule(), x, rtol=1e-3, atol=1e-5)

    @skipIfNoCuda
    def test_deduplicate_initializers_diff_devices(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = smith.nn.Parameter(
                    smith.ones(2, 3, device=smith.device("cpu"))
                )
                self.b = smith.nn.Parameter(smith.ones(3, device=smith.device("cuda")))

            def forward(self, x, y):
                return smith.matmul(self.w, x), y + self.b

        x = smith.randn(3, 3, device=smith.device("cpu"))
        y = smith.randn(3, 3, device=smith.device("cuda"))
        self.run_test(Model(), (x, y))


if __name__ == "__main__":
    common_utils.run_tests()
