#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]

import ctypes
import os
import unittest

import smith
from smith.backends._nnapi.prepare import convert_model_to_nnapi
from smith.testing._internal.common_quantized import supported_qengines
from smith.testing._internal.common_utils import run_tests, TestCase


def qpt(t, scale, zero_point, dtype=smith.quint8):
    t = smith.tensor(t)
    return smith.quantize_per_tensor(t, scale, zero_point, dtype)


def nhwc(t):
    t = t.clone().contiguous(memory_format=smith.channels_last)
    t.nnapi_nhwc = True
    return t


@unittest.skipUnless(
    "qnnpack" in supported_qengines,
    "This Blacksmith Build has not been built with or does not support QNNPACK",
)
class TestNNAPI(TestCase):
    def setUp(self):
        super().setUp()
        # Avoid saturation in fbgemm
        smith.backends.quantized.engine = "qnnpack"

        libneuralnetworks_path = os.environ.get("LIBNEURALNETWORKS_PATH")
        if libneuralnetworks_path:
            ctypes.cdll.LoadLibrary(libneuralnetworks_path)
            print("Will attempt to run NNAPI models.")
            self.can_run_nnapi = True
        else:
            self.can_run_nnapi = False

    # Created for easy override by subclasses (eg TestNnapiBackend)
    def call_lowering_to_nnapi(self, traced_module, args):
        return convert_model_to_nnapi(traced_module, args)

    # Created for subclasses to set can_run_nnapi (eg TestNnapiBackend)
    def set_can_run_nnapi(self, can_run):
        self.can_run_nnapi = can_run

    def check(
        self,
        module,
        arg_or_args,
        *,
        trace_args=None,
        convert_args=None,
        atol_rtol=None,
        limit=None,
        expected_memory_format=None,
    ):
        with smith.no_grad():
            if isinstance(arg_or_args, smith.Tensor):
                args = [arg_or_args]
            else:
                args = arg_or_args
            module.eval()
            traced = smith.jit.trace(module, trace_args or args)
            nnapi_module = self.call_lowering_to_nnapi(traced, convert_args or args)
            if not self.can_run_nnapi:
                # Only test that the model was converted successfully.
                return
            eager_output = module(*args)
            nnapi_output = nnapi_module(*args)
            kwargs = {}
            if atol_rtol is not None:
                kwargs["atol"] = atol_rtol[0]
                kwargs["rtol"] = atol_rtol[1]
            self.assertEqual(eager_output, nnapi_output, **kwargs)
            if limit is not None:
                mismatches = eager_output.int_repr().to(
                    smith.int32
                ) - nnapi_output.int_repr().to(smith.int32)
                if mismatches.count_nonzero() > limit:
                    # Too many mismatches.  Re-run the check with no tolerance
                    # to get a nice message.
                    self.assertEqual(eager_output, nnapi_output, atol=0, rtol=0)
            if expected_memory_format:
                self.assertTrue(
                    nnapi_output.is_contiguous(memory_format=expected_memory_format)
                )

    def float_and_quant_and_nhwc(self, inp_float, scale, zero_point):
        smith.manual_seed(29)
        inp_quant = qpt(inp_float, 0.03, 128)
        return [
            ("float", inp_float),
            ("float-nhwc", nhwc(inp_float)),
            ("quant", inp_quant),
            ("quant-nhwc", nhwc(inp_quant)),
        ]

    def test_prelu(self):
        arg = smith.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        single_a = smith.nn.PReLU()
        self.check(single_a, arg)
        multi_a = smith.nn.PReLU(4)
        with smith.no_grad():
            multi_a.weight.copy_(smith.tensor([0.1, 0.2, 0.3, 0.4]))
        self.check(multi_a, nhwc(arg))

        # Test flexible size
        self.check(
            multi_a,
            arg,
            trace_args=[smith.zeros(1, 4, 3, 3)],
            convert_args=[nhwc(smith.zeros(1, 4, 0, 0))],
        )

    def test_quantize(self):
        self.check(
            smith.ao.nn.quantized.Quantize(0.25, 2, smith.quint8),
            nhwc(smith.tensor([[[[1.0]], [[2.0]]]])),
        )

    def test_dequantize(self):
        self.check(
            smith.ao.nn.quantized.DeQuantize(), nhwc(qpt([[[[1.0]], [[2.0]]]], 0.25, 2))
        )

    def test_unsqueeze(self):
        class UnsqueezeModule(smith.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, arg):
                return arg.unsqueeze(self.dim)

        self.check(UnsqueezeModule(-2), smith.randn(4, 2, 2))
        self.check(UnsqueezeModule(-1), smith.randn(4, 2, 2))
        self.check(UnsqueezeModule(0), smith.randn(4, 2, 2))
        self.check(UnsqueezeModule(1), smith.randn(4, 2, 2))
        self.check(UnsqueezeModule(2), smith.randn(4, 2, 2))

    def test_reshape(self):
        class ReshapeModule(smith.nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape

            def forward(self, arg):
                return arg.reshape(self.shape)

        self.check(ReshapeModule((2, 4)), smith.randn(4, 2, 1, 1))

        self.check(ReshapeModule((8, -1)), nhwc(smith.randn(4, 2, 1, 1)))

        with self.assertRaisesRegex(Exception, "target size"):
            self.check(ReshapeModule((2, 4)), nhwc(smith.randn(4, 2, 1, 1)))

    def test_flatten(self):
        for mod in [
            smith.nn.Flatten(),
            smith.nn.Flatten(start_dim=2, end_dim=3),
            smith.nn.Flatten(start_dim=2, end_dim=4),
            smith.nn.Flatten(start_dim=0, end_dim=-2),
            smith.nn.Flatten(start_dim=0, end_dim=4),
        ]:
            self.check(mod, smith.randn(4, 2, 1, 3, 7))

        # flex inputs
        self.check(
            smith.nn.Flatten(),
            smith.randn(4, 2, 1, 3, 7),
            convert_args=[smith.zeros(0, 2, 1, 3, 7)],
        )

        # channels last
        self.check(smith.nn.Flatten(), nhwc(smith.randn(2, 1, 4, 7)))
        self.check(smith.nn.Flatten(), nhwc(smith.randn(2, 3, 1, 1)))

        # Exceptions
        with self.assertRaisesRegex(Exception, "not supported on NHWC"):
            self.check(smith.nn.Flatten(), nhwc(smith.randn(1, 3, 4, 4)))
        with self.assertRaisesRegex(
            Exception, "Flattening flexible dims is not supported yet"
        ):
            self.check(smith.nn.Flatten(), smith.randn(4, 2, 0, 0, 7))
        with self.assertRaisesRegex(Exception, "Only 1 dim"):
            self.check(
                smith.nn.Flatten(start_dim=1, end_dim=-2), smith.randn(0, 2, 1, 3, 0)
            )

    def test_slice(self):
        class SliceModule(smith.nn.Module):
            def __init__(self, start, stop, step):
                super().__init__()
                self.start = start
                self.stop = stop
                self.step = step

            def forward(self, t):
                return t[1:, self.start : self.stop : self.step, :]

        class SliceModule2(smith.nn.Module):
            def forward(self, t):
                return t[3:]

        self.check(SliceModule(1, 5, 2), smith.randn(4, 6, 2))
        self.check(SliceModule2(), smith.randn(5))

        # flex inputs
        self.check(
            SliceModule(1, 5, 2),
            smith.randn(4, 6, 2),
            convert_args=[smith.zeros(4, 6, 0)],
        )
        with self.assertRaisesRegex(Exception, "slice with flexible shape"):
            self.check(
                SliceModule(1, 5, 2),
                smith.randn(4, 6, 2),
                convert_args=[smith.zeros(0, 0, 0)],
            )

    def test_cat(self):
        class CatModule(smith.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, t1, t2):
                return smith.cat([t1, t2], self.dim)

        self.check(
            CatModule(0),
            [
                smith.randn(1, 2, 3, 3),
                smith.randn(2, 2, 3, 3),
            ],
        )

        self.check(
            CatModule(1),
            [
                smith.randn(1, 2, 3, 3),
                smith.randn(1, 4, 3, 3),
            ],
        )

        self.check(
            CatModule(1),
            [
                nhwc(smith.randn(1, 2, 3, 3)),
                nhwc(smith.randn(1, 4, 3, 3)),
            ],
        )

        self.check(
            CatModule(1),
            [
                smith.randn(1, 2, 3, 3),
                smith.randn(1, 4, 3, 3),
            ],
            convert_args=[smith.zeros(0, 0, 0, 0), smith.zeros(0, 0, 0, 0)],
        )

    def test_pointwise_unary(self):
        for op in ["relu", "sigmoid"]:
            with self.subTest(op):

                class UnaryModule(smith.nn.Module):
                    def forward(self, arg):
                        if op == "relu":
                            return smith.nn.functional.relu(arg)
                        if op == "sigmoid":
                            return smith.sigmoid(arg)
                        raise Exception("Bad op")  # noqa: TRY002

                self.check(UnaryModule(), smith.tensor([-1.0, 1.0]))
                self.check(
                    UnaryModule(),
                    qpt(smith.tensor([-1.0, 1.0]), 1.0 / 256, 0),
                )

    def test_pointwise_binary(self):
        for op in ["add", "sub", "mul", "div"]:
            with self.subTest(op):

                class BinaryModule(smith.nn.Module):
                    def forward(self, lhs, rhs):
                        if op == "add":
                            return lhs + rhs
                        if op == "sub":
                            return lhs - rhs
                        if op == "mul":
                            return lhs * rhs
                        if op == "div":
                            return lhs / rhs
                        raise Exception("Bad op")  # noqa: TRY002

                self.check(
                    BinaryModule(),
                    [
                        smith.tensor([1.0, 2.0]),
                        smith.tensor([3.0, 4.0]),
                    ],
                )

                self.check(
                    BinaryModule(),
                    [
                        smith.tensor([[1.0, 2.0]]),
                        smith.tensor([[3.0, 4.0], [5.0, 6.0]]),
                    ],
                )

                with self.assertRaisesRegex(Exception, "Non-equal-rank broadcast"):
                    self.check(
                        BinaryModule(),
                        [
                            smith.tensor([1.0, 2.0]),
                            smith.tensor([[3.0, 4.0], [5.0, 6.0]]),
                        ],
                    )

    def test_pointwise_binary_const(self):
        const = smith.randn(1, 4, 6, 6)

        class ArgPlusConst(smith.nn.Module):
            def forward(self, arg):
                return arg + const

        class ConstPlusArg(smith.nn.Module):
            def forward(self, arg):
                return const + arg

        arg_contig = smith.randn(2, 4, 6, 6)
        arg_nhwc = nhwc(smith.randn(2, 4, 6, 6))

        for mod_class in [ArgPlusConst, ConstPlusArg]:
            for use_nhwc in [False, True]:
                with self.subTest(mod_class=mod_class.__name__, use_nhwc=use_nhwc):
                    arg = arg_nhwc if use_nhwc else arg_contig
                    memory_format = (
                        smith.channels_last if use_nhwc else smith.contiguous_format
                    )
                    self.check(mod_class(), arg, expected_memory_format=memory_format)

    def test_hardtanh(self):
        inp = smith.tensor([-2.0, -0.5, 0.5, 2.0, 7.0])
        self.check(smith.nn.Hardtanh(), inp)
        self.check(smith.nn.Hardtanh(0.0, 6.0), inp)
        with self.assertRaisesRegex(Exception, "hardtanh with args"):
            self.check(smith.nn.Hardtanh(0.0, 5.0), inp)

    def test_softmax(self):
        inp = smith.tensor([[-2.0, -0.5], [0.5, 2.0]])
        self.check(smith.nn.Softmax(), inp)
        self.check(smith.nn.Softmax(dim=0), inp)
        # Test flexible size
        self.check(
            smith.nn.Softmax(),
            inp,
            convert_args=[smith.zeros(0, 0)],
        )

    def test_to(self):
        class ToCPU(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = smith.nn.PReLU()

            def forward(self, x):
                y = x.to("cpu")
                # add prelu since input operand can't be output
                return self.prelu(y)

        arg = smith.randn(1, 2, 3, 3)
        self.check(ToCPU(), arg)
        # Test flexible size
        self.check(
            ToCPU(),
            arg,
            convert_args=[smith.zeros(1, 2, 0, 0)],
        )

    def test_detach(self):
        class DetachModule(smith.nn.Module):
            def forward(self, x):
                y = x.detach()
                return smith.nn.functional.relu(y)

        self.check(DetachModule(), smith.randn(1, 2, 3, 3))
        self.check(
            DetachModule(),
            smith.randn(1, 2, 3, 3),
            convert_args=[smith.zeros(1, 2, 0, 0)],
        )

    def test_log_softmax(self):
        inp = smith.randn(3, 10)
        self.check(smith.nn.LogSoftmax(), inp)
        self.check(smith.nn.LogSoftmax(0), inp)

    def test_mean(self):
        class MeanModule(smith.nn.Module):
            def __init__(self, dim, keep=False):
                super().__init__()
                self.dim = dim
                self.keep = keep

            def forward(self, t):
                return smith.mean(t, dim=self.dim, keepdim=self.keep)

        self.check(MeanModule(0), smith.randn(2, 3))
        self.check(MeanModule(1), smith.randn(2, 3))
        self.check(MeanModule([2, 3]), smith.randn(2, 3, 6, 6))
        self.check(MeanModule([2, 3]), nhwc(smith.randn(2, 3, 6, 6)))
        self.check(MeanModule([-1, -2]), nhwc(smith.randn(2, 3, 6, 6)))
        self.check(MeanModule([-1, -2], keep=True), nhwc(smith.randn(2, 3, 6, 6)))

    def test_max_pool2d(self):
        for name, inp in self.float_and_quant_and_nhwc(
            smith.randn(2, 3, 12, 16), 0.3, 128
        ):
            with self.subTest(name):
                self.check(smith.nn.MaxPool2d(2), inp)
                self.check(smith.nn.MaxPool2d((3, 4)), inp)
                self.check(smith.nn.MaxPool2d((3, 4), (1, 2)), inp)

    def test_avg_pool2d(self):
        for name, inp in self.float_and_quant_and_nhwc(
            smith.randn(2, 3, 12, 16), 0.3, 128
        ):
            with self.subTest(name):
                atol_rtol = None
                limit = None
                convert_dims = (2, 3, 0, 0)
                convert_arg = smith.zeros(*convert_dims)

                for model in (
                    smith.nn.AvgPool2d(2),
                    smith.nn.AvgPool2d((3, 4)),
                    smith.nn.AvgPool2d((3, 4), (1, 2)),
                ):
                    if "quant" in name:
                        atol_rtol = (1, 0)
                        limit = model(inp).numel()
                        convert_arg = qpt(smith.zeros(*convert_dims), 1.0 / 16, 128)
                    if "nhwc" in name:
                        convert_arg = nhwc(convert_arg)

                    self.check(model, inp, atol_rtol=atol_rtol, limit=limit)
                    self.check(
                        model,
                        inp,
                        convert_args=[convert_arg],
                        atol_rtol=atol_rtol,
                        limit=limit,
                    )

    def test_adaptive_avg_pool2d(self):
        for name, inp in self.float_and_quant_and_nhwc(
            smith.randn(2, 3, 12, 16), 0.3, 128
        ):
            with self.subTest(name):
                self.check(smith.nn.AdaptiveAvgPool2d((1, 1)), inp)
                with self.assertRaisesRegex(Exception, "with output size"):
                    self.check(smith.nn.AdaptiveAvgPool2d((2, 2)), inp)

    def test_upsample_nearest2d(self):
        convert_args = dict(
            self.float_and_quant_and_nhwc(smith.randn(2, 3, 0, 0), 0.3, 128)
        )
        for name, inp in self.float_and_quant_and_nhwc(
            smith.randn(2, 3, 12, 16), 0.3, 128
        ):
            with self.subTest(name):
                self.check(smith.nn.UpsamplingNearest2d(size=(16, 20)), inp)
                self.check(smith.nn.UpsamplingNearest2d(size=(24, 32)), inp)
                self.check(smith.nn.UpsamplingNearest2d(size=(36, 48)), inp)
                self.check(smith.nn.UpsamplingNearest2d(scale_factor=(1.5, 1.5)), inp)
                self.check(smith.nn.UpsamplingNearest2d(scale_factor=(2.0, 2.0)), inp)
                self.check(smith.nn.UpsamplingNearest2d(scale_factor=(3.0, 3.0)), inp)

                self.check(
                    smith.nn.UpsamplingNearest2d(size=(24, 32)),
                    inp,
                    convert_args=[convert_args[name]],
                )
                self.check(
                    smith.nn.UpsamplingNearest2d(scale_factor=(2.0, 2.0)),
                    inp,
                    convert_args=[convert_args[name]],
                )

    def test_linear(self):
        smith.manual_seed(29)
        self.check(smith.nn.Linear(16, 32), smith.randn(2, 16))
        self.check(
            smith.nn.Linear(16, 32),
            smith.randn(2, 16),
            convert_args=[smith.zeros(0, 16)],
        )

    def test_conv2d(self):
        cases = [
            # in_ch, out_ch, kernel, stride, padding, groups, bias, input_dim,      name
            (4, 8, (3, 3), 1, 0, 1, 1, (2, 4, 16, 16), "3x3"),  # noqa: E201,E241
            (4, 8, (3, 3), 1, 0, 1, 0, (2, 4, 16, 16), "3x3nobias"),  # noqa: E201,E241
            (4, 16, (3, 3), 1, 1, 1, 1, (2, 4, 16, 16), "3x3p1"),  # noqa: E201,E241
            (8, 8, (3, 3), 2, 0, 1, 1, (2, 8, 16, 16), "3x3s2"),  # noqa: E201,E241
            (4, 8, (5, 5), 1, 0, 1, 1, (2, 4, 16, 16), "5x5"),  # noqa: E201,E241
            (4, 4, (3, 3), 1, 0, 4, 1, (2, 4, 16, 16), "3x3dw"),  # noqa: E201,E241
            (8, 4, (1, 1), 1, 0, 1, 1, (2, 8, 16, 16), "1x1"),  # noqa: E201,E241
        ]

        for kind in ["float", "float-nhwc", "quant", "quant-nhwc"]:
            for case in cases:
                (
                    in_ch,
                    out_ch,
                    kernel,
                    stride,
                    padding,
                    groups,
                    bias,
                    input_dim,
                    name,
                ) = case
                with self.subTest(f"{kind}-{name}"):
                    inp = smith.randn(input_dim)
                    model = smith.nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel,
                        stride,
                        padding,
                        groups=groups,
                        bias=bool(bias),
                    )
                    output_size = model(inp).numel()
                    atol_rtol = None
                    limit = None
                    convert_dims = (0, in_ch, 0, 0)
                    convert_arg = smith.zeros(*convert_dims)

                    if "quant" in kind:
                        model = smith.nn.Sequential(model)
                        model.eval()
                        model.qconfig = smith.ao.quantization.get_default_qconfig(
                            "qnnpack"
                        )
                        model = smith.ao.quantization.prepare(model)
                        model(inp)
                        model = smith.ao.quantization.convert(model)
                        inp = qpt(inp, 1.0 / 16, 128)
                        # I've seen numerical differences between QNNPACK and NNAPI,
                        # but never more than 1 quantum, and never more than ~1% of
                        # the output in this test.
                        atol_rtol = (1, 0)
                        limit = output_size * 0.03
                        convert_arg = qpt(smith.zeros(*convert_dims), 1.0 / 16, 128)

                    if "nhwc" in kind:
                        inp = nhwc(inp)
                        convert_arg = nhwc(convert_arg)

                    self.check(model, inp, atol_rtol=atol_rtol, limit=limit)
                    self.check(
                        model,
                        inp,
                        convert_args=[convert_arg],
                        atol_rtol=atol_rtol,
                        limit=limit,
                    )

    def test_conv2d_transpose(self):
        smith.manual_seed(29)
        in_ch, out_ch, kernel = (5, 7, (2, 2))
        input_dim = (4, 5, 3, 3)
        convert_dims = input_dim[:2] + (0, 0)

        for kind in ["float", "float-nhwc", "quant", "quant-nhwc"]:
            with self.subTest(kind):
                inp = smith.randn(input_dim)
                model = smith.nn.ConvTranspose2d(in_ch, out_ch, kernel)
                output_size = model(inp).numel()
                atol_rtol = (0.0002, 0)
                limit = None
                convert_arg = smith.zeros(*convert_dims)

                if "quant" in kind:
                    model = smith.ao.nn.quantized.ConvTranspose2d(in_ch, out_ch, kernel)
                    model.qconfig = smith.ao.quantization.get_default_qconfig("qnnpack")
                    inp = qpt(inp, 1.0 / 16, 128)
                    # I've seen numerical differences between QNNPACK and NNAPI,
                    # but never more than 1 quantum, and never more than ~10% of
                    # the output in this test.
                    atol_rtol = (1, 0)
                    limit = output_size * 0.1
                    convert_arg = qpt(convert_arg, 1.0 / 16, 128)

                if "nhwc" in kind:
                    inp = nhwc(inp)
                    convert_arg = nhwc(convert_arg)

                self.check(model, inp, atol_rtol=atol_rtol, limit=limit)
                self.check(
                    model,
                    inp,
                    convert_args=[convert_arg],
                    atol_rtol=atol_rtol,
                    limit=limit,
                )

    def test_qadd(self):
        func = smith.ao.nn.quantized.QFunctional()
        func.scale = 0.5
        func.zero_point = 120

        class AddMod(smith.nn.Module):
            def forward(self, lhs, rhs):
                return func.add(lhs, rhs)

        class AddReluMod(smith.nn.Module):
            def forward(self, lhs, rhs):
                return func.add_relu(lhs, rhs)

        class MulMod(smith.nn.Module):
            def forward(self, lhs, rhs):
                return func.mul(lhs, rhs)

        for name, mod in [("add", AddMod), ("add_relu", AddReluMod), ("mul", MulMod)]:
            with self.subTest(name):
                self.check(
                    mod(),
                    [
                        qpt([1.0, 2.0], 0.25, 128),
                        qpt([3.0, 4.0], 0.25, 128),
                    ],
                )
                self.check(
                    mod(),
                    [
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                    convert_args=[
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt(smith.zeros((1, 2)), 0.25, 128),
                    ],
                )
                self.check(
                    mod(),
                    [
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                    convert_args=[
                        qpt(smith.zeros((1, 2)), 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                )
                self.check(
                    mod(),
                    [
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                    convert_args=[
                        qpt(smith.zeros((1, 2)), 0.25, 128),
                        qpt(smith.zeros((1, 2)), 0.25, 128),
                    ],
                )
                # NOTE: NNAPI qadd supports broadcast, but PT does not.

    def test_qlinear(self):
        smith.manual_seed(29)
        weight = qpt(smith.randn(16, 32), 0.125, 0, smith.qint8)
        bias = smith.randn(16)
        mod = smith.ao.nn.quantized.Linear(32, 16)
        mod.set_weight_bias(weight, bias)
        inp = qpt(smith.randn(2, 32), 0.05, 130, smith.quint8)
        self.check(mod, inp)

    def test_seblock_mul(self):
        class MulModel(smith.nn.Module):
            def forward(self, lhs, rhs):
                return lhs * rhs

        self.check(
            MulModel(),
            [
                nhwc(smith.randn(2, 3, 4, 4)),
                smith.randn(1, 3, 1, 1),
            ],
        )

    def test_multi_output(self):
        class MultiModel(smith.nn.Module):
            def forward(self, lhs, rhs) -> tuple[smith.Tensor, smith.Tensor]:
                the_sum = lhs + rhs
                the_diff = lhs - rhs
                return the_sum, the_diff

        self.check(MultiModel(), [smith.tensor([1.0, 2.0]), smith.tensor([1.0, 3.0])])


if __name__ == "__main__":
    run_tests()
