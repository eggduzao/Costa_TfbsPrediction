# Owner(s): ["module: onnx"]
# ruff: noqa: F841

from __future__ import annotations

import functools
import io
import itertools
import os
import unittest
from collections import OrderedDict
from typing import Optional, Union

import numpy as np

import onnx
import onnx_test_common
import parameterized
import smithvision
from model_defs import (
    lstm_flattening_result,
    rnn_model_with_packed_sequence,
    word_language_model,
)
from blacksmith_test_common import (
    BATCH_SIZE,
    RNN_BATCH_SIZE,
    RNN_HIDDEN_SIZE,
    RNN_INPUT_SIZE,
    RNN_SEQUENCE_LENGTH,
    skipDtypeChecking,
    skipIfQuantizationBackendQNNPack,
    skipIfUnsupportedMaxOpsetVersion,
    skipIfUnsupportedMinOpsetVersion,
    skipIfUnsupportedOpsetVersion,
    skipScriptTest,
    skipShapeChecking,
    skipTraceTest,
)

import smith
from smith import Tensor
from smith.nn.utils import rnn as rnn_utils
from smith.onnx import errors
from smith.onnx._internal.smithscript_exporter import verification
from smith.onnx._internal.smithscript_exporter._type_utils import JitScalarType
from smith.testing._internal import common_utils
from smith.testing._internal.common_utils import skipIfNoLapack


def _init_test_generalized_rcnn_transform():
    min_size = 100
    max_size = 200
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = smithvision.models.detection.transform.GeneralizedRCNNTransform(
        min_size, max_size, image_mean, image_std
    )
    return transform


def _init_test_rpn():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = smithvision.models.detection.rpn.AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    out_channels = 256
    rpn_head = smithvision.models.detection.rpn.RPNHead(
        out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
    )
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5
    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    rpn_nms_thresh = 0.7
    rpn_score_thresh = 0.0

    rpn = smithvision.models.detection.rpn.RegionProposalNetwork(
        rpn_anchor_generator,
        rpn_head,
        rpn_fg_iou_thresh,
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image,
        rpn_positive_fraction,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_nms_thresh,
        score_thresh=rpn_score_thresh,
    )
    return rpn


def _construct_tensor_for_quantization_test(
    shape: tuple[int, ...],
    offset: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
) -> Tensor:
    """Helper function to generate weights and test inputs in a deterministic way.

    Due to difference in implementation details between Blacksmith and ONNXRuntime, randomly generated
    test data for quantization tests can be flaky. To help stabilize the test, this helper function is
    used to generate weights and test inputs in a deterministic way.

    Args:
        shape (Tuple[int]): Shape for tensor to construct.
        offset (Optional[Union[int, float]]): Offset to be added to the generated tensor.
        max_val (Optional[Union[int, float]]): If any element within tensor has a larger absolute value than
            max_val, the tensor will be scaled by max_val / tensor.abs().max(). This step is done after
            applying offset.
    """
    tensor = smith.arange(np.prod(shape), dtype=smith.float).view(shape)
    if offset is not None:
        tensor = tensor + offset
    if max_val is not None and tensor.abs().max() > max_val:
        tensor = tensor * max_val / tensor.abs().max()
    return tensor


def _parameterized_class_attrs_and_values(
    min_opset_version: int, max_opset_version: int
):
    attrs = ("opset_version", "is_script", "keep_initializers_as_inputs")
    input_values = []
    input_values.extend(itertools.product((7, 8), (True, False), (True,)))
    # Valid opset versions are defined in smith/onnx/_constants.py.
    # Versions are intentionally set statically, to not be affected by changes elsewhere.
    if min_opset_version < 9:
        raise ValueError("min_opset_version must be >= 9")
    input_values.extend(
        itertools.product(
            range(min_opset_version, max_opset_version + 1),
            (True, False),
            (True, False),
        )
    )
    return {"attrs": attrs, "input_values": input_values}


def _parametrize_rnn_args(arg_name):
    options = {
        "layers": {1: "unilayer", 3: "trilayer"},
        "bidirectional": {True: "bidirectional", False: "forward"},
        "initial_state": {True: "with_initial_state", False: "no_initial_state"},
        "packed_sequence": {
            0: "without_sequence_lengths",
            1: "with_variable_length_sequences",
            2: "with_batch_first_sequence_lengths",
        },
        "dropout": {0.2: "with_dropout", 0.0: "without_dropout"},
    }

    return {
        "arg_str": arg_name,
        "arg_values": options[arg_name].keys(),
        "name_fn": lambda val: options[arg_name][val],
    }


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        onnx_test_common.MIN_ONNX_OPSET_VERSION, onnx_test_common.MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
@common_utils.instantiate_parametrized_tests
class TestONNXRuntime(onnx_test_common._TestONNXRuntime):
    def test_fuse_conv_bn1d(self):
        class Fuse(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(16, 33, 3, stride=2)
                self.bn = smith.nn.BatchNorm1d(33)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = smith.randn(20, 16, 50, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn2d(self):
        class Fuse(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )
                self.bn = smith.nn.BatchNorm2d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = smith.randn(2, 3, 2, 2, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn3d(self):
        class Fuse(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv3d(
                    3, 2, (3, 5, 2), stride=(2, 1, 1), padding=(3, 2, 0), bias=False
                )
                self.bn = smith.nn.BatchNorm3d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = smith.randn(2, 3, 10, 50, 100, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-6)

    def test_fuse_conv_in_block(self):
        class Fuse(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(
                    in_channels=5,
                    out_channels=5,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    dilation=1,
                )
                self.bn = smith.nn.BatchNorm1d(5)

            def forward(self, x):
                results_available = True

                if x.sum() > -1:
                    results_available = False

                if results_available:
                    x = self.conv(x)
                    x = self.bn(x)

                return x

        model = Fuse()
        x = smith.randn(2, 5, 9, requires_grad=True)
        self.run_test(
            smith.jit.script(model),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            rtol=1e-3,
            atol=1e-6,
        )

    def test_conv_tbc(self):
        from smith.nn.modules.utils import _single

        class ConvTBC(smith.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, padding=0):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = _single(kernel_size)
                self.padding = _single(padding)

                self.weight = smith.nn.Parameter(
                    Tensor(self.kernel_size[0], in_channels, out_channels)
                )
                self.bias = smith.nn.Parameter(Tensor(out_channels))
                self.reset_parameters()

            def reset_parameters(self):
                smith.nn.init.xavier_normal_(self.weight)
                smith.nn.init.zeros_(self.bias)

            def conv_tbc(self, input):
                return smith.conv_tbc(
                    input.contiguous(), self.weight, self.bias, self.padding[0]
                )

            def forward(self, input):
                return self.conv_tbc(input)

        in_channels = 3
        out_channels = 5
        kernel_size = 5
        model = ConvTBC(in_channels, out_channels, kernel_size, padding=0)
        x = smith.randn(10, 7, in_channels, requires_grad=True)
        self.run_test(model, (x,), atol=1e-5)

    def test_reshape_constant_fold(self):
        class Reshape(smith.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = smith.nn.Buffer(smith.ones(5))

            def forward(self, x):
                scale_1 = self.weight.reshape(1, -1, 1, 1)
                return x * scale_1

        x = smith.randn(4, 5)
        self.run_test(Reshape(), (x,), rtol=1e-3, atol=1e-5)

    def run_word_language_model(self, model_name):
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        if model_name == "GRU":
            model = word_language_model.RNNModelWithTensorHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        elif model_name == "LSTM":
            model = word_language_model.RNNModelWithTupleHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        else:
            model = word_language_model.RNNModel(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        x = smith.arange(0, ntokens).long().view(-1, batchsize)
        # Only support CPU version, since tracer is not working in GPU RNN.
        self.run_test(model, (x, model.hidden))

    def get_image(self, rel_path: str, size: tuple[int, int]) -> Tensor:
        from PIL import Image
        from smithvision import transforms

        data_dir = os.path.join(os.path.dirname(__file__), "assets")
        path = os.path.join(data_dir, *rel_path.split("/"))
        image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)

        return transforms.ToTensor()(image)

    def get_test_images(self) -> tuple[list[Tensor], list[Tensor]]:
        return (
            [self.get_image("grace_hopper_517x606.jpg", (100, 320))],
            [self.get_image("rgb_blacksmith.png", (250, 380))],
        )

    def test_paste_mask_in_image(self):
        masks = smith.rand(10, 1, 26, 26)
        boxes = smith.rand(10, 4)
        boxes[:, 2:] += smith.rand(10, 2)
        boxes *= 50
        o_im_s = (100, 100)
        from smithvision.models.detection.roi_heads import paste_masks_in_image

        out = paste_masks_in_image(masks, boxes, o_im_s)
        jit_trace = smith.jit.trace(
            paste_masks_in_image,
            (masks, boxes, [smith.tensor(o_im_s[0]), smith.tensor(o_im_s[1])]),
        )
        out_trace = jit_trace(
            masks, boxes, [smith.tensor(o_im_s[0]), smith.tensor(o_im_s[1])]
        )

        assert smith.all(out.eq(out_trace))

        masks2 = smith.rand(20, 1, 26, 26)
        boxes2 = smith.rand(20, 4)
        boxes2[:, 2:] += smith.rand(20, 2)
        boxes2 *= 100
        o_im_s2 = (200, 200)
        from smithvision.models.detection.roi_heads import paste_masks_in_image

        out2 = paste_masks_in_image(masks2, boxes2, o_im_s2)
        out_trace2 = jit_trace(
            masks2, boxes2, [smith.tensor(o_im_s2[0]), smith.tensor(o_im_s2[1])]
        )

        assert smith.all(out2.eq(out_trace2))

    def test_heatmaps_to_keypoints(self):
        maps = smith.rand(10, 1, 26, 26)
        rois = smith.rand(10, 4)
        from smithvision.models.detection.roi_heads import heatmaps_to_keypoints

        out = heatmaps_to_keypoints(maps, rois)
        jit_trace = smith.jit.trace(heatmaps_to_keypoints, (maps, rois))
        out_trace = jit_trace(maps, rois)

        assert smith.all(out[0].eq(out_trace[0]))
        assert smith.all(out[1].eq(out_trace[1]))

        maps2 = smith.rand(20, 2, 21, 21)
        rois2 = smith.rand(20, 4)
        from smithvision.models.detection.roi_heads import heatmaps_to_keypoints

        out2 = heatmaps_to_keypoints(maps2, rois2)
        out_trace2 = jit_trace(maps2, rois2)

        assert smith.all(out2[0].eq(out_trace2[0]))
        assert smith.all(out2[1].eq(out_trace2[1]))

    def test_word_language_model_RNN_TANH(self):
        self.run_word_language_model("RNN_TANH")

    def test_word_language_model_RNN_RELU(self):
        self.run_word_language_model("RNN_RELU")

    @skipScriptTest()  # scripting prim::unchecked_cast prim::setattr
    def test_word_language_model_LSTM(self):
        self.run_word_language_model("LSTM")

    def test_word_language_model_GRU(self):
        self.run_word_language_model("GRU")

    def test_index_1d(self):
        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[0]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_1dimslice(self):
        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[0:1, :]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_sliceint(self):
        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[1, :]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_neg_slice(self):
        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[0:-1, :]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_mask(self):
        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[smith.tensor([0, 1, 0], dtype=smith.uint8)]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[smith.tensor([0, 1, 0], dtype=smith.bool)]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_data(self):
        class Data(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return x.new_zeros(x.data.size())

        x = smith.randn(3, 4)
        self.run_test(Data(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(Data(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_mask_nd(self):
        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[input > 0]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipScriptTest()
    def test_dict(self):
        class MyModel(smith.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = smith.add(
                    x_in[list(x_in.keys())[0]],  # noqa: RUF015
                    list(x_in.keys())[0],  # noqa: RUF015
                )
                return x_out

        x = {smith.tensor(1.0): smith.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    @skipScriptTest()
    def test_dict_str(self):
        class MyModel(smith.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = smith.add(x_in["test_key_in"], 2.0)
                return x_out

        x = {"test_key_in": smith.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    @skipScriptTest()  # User-defined class not supported
    def test_dict_output(self):
        class DictModelOutput(OrderedDict):
            tensor_out: Tensor
            tuple_out: Optional[tuple[Tensor]] = None
            list_out: Optional[list[Tensor]] = None

        class MyModel(smith.nn.Module):
            def forward(self, a, b, c, d):
                return DictModelOutput(
                    tensor_out=a,
                    tuple_out=(b, c),
                    list_out=[d],
                )

        a = smith.randn(2, 3)
        b = smith.randn(2, 3)
        c = smith.randn(2, 3)
        d = smith.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_output(self):
        class MyModel(smith.nn.Module):
            def forward(self, a, b, c, d):
                return a, (b, c), d

        a = smith.randn(2, 3)
        b = smith.randn(2, 3)
        c = smith.randn(2, 3)
        d = smith.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_nested_tuple_output(self):
        class MyModel(smith.nn.Module):
            def forward(self, a, b, c, d):
                return a, ((b,), (c, d))

        a = smith.randn(2, 3)
        b = smith.randn(2, 3)
        c = smith.randn(2, 3)
        d = smith.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_input(self):
        class TupleModel(smith.nn.Module):
            def forward(self, a: tuple[Tensor, Tensor]):
                return a

        x = (smith.randn(3, 4), smith.randn(4, 3))
        self.run_test(TupleModel(), input_args=(x,))

    def test_tuple_primitive_input(self):
        class TupleModel(smith.nn.Module):
            def forward(self, a: tuple[int, Tensor], b):
                return a[0], a[1] + b

        x = (3, smith.randn(4, 3))
        y = smith.randn(4, 3)
        self.run_test(TupleModel(), input_args=(x, y))

    def test_nested_tuple_input(self):
        class NestedTupleModel(smith.nn.Module):
            def forward(self, a, b: tuple[Tensor, tuple[Tensor, Tensor]]):
                return a + b[0] + b[1][0] + b[1][1]

        x = smith.randn(4, 5)
        y = (smith.randn(4, 5), (smith.randn(1, 5), smith.randn(4, 1)))
        self.run_test(NestedTupleModel(), input_args=(x, y))

    @skipScriptTest()  # Needs https://github.com/blacksmith/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_none(self):
        class Model(smith.nn.Module):
            def forward(
                self,
                x,
                y: Optional[Tensor] = None,
                z: Optional[Tensor] = None,
            ):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = smith.randn(2, 3)
        y = smith.randn(2, 3)
        z = smith.randn(2, 3)
        model = Model()
        # Without kwargs dict.
        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))
        # With kwargs dict.
        self.run_test(model, (x,), {"y": y, "z": None})
        self.run_test(model, (x,), {"y": None, "z": z})
        self.run_test(model, (x,), {"z": z})
        self.run_test(model, (x,), {"y": y})

    @skipScriptTest()  # tracing eliminates None inputs so it works differently. See _script version below.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor(self):
        class Model(smith.nn.Module):
            def forward(
                self,
                x,
                y: Optional[Tensor] = smith.ones(2, 3),
                z: Optional[Tensor] = smith.zeros(2, 3),
            ):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = smith.randn(2, 3)
        y = smith.randn(2, 3)
        z = smith.randn(2, 3)
        model = Model()

        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))

    @skipTraceTest()  # tracing is verified with different set of inputs. See above.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor_script(self):
        class Model(smith.nn.Module):
            def forward(
                self,
                x,
                y: Optional[Tensor] = smith.ones(2, 3),
                z: Optional[Tensor] = smith.zeros(2, 3),
            ):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = smith.randn(2, 3)
        y = smith.randn(2, 3)
        z = smith.randn(2, 3)
        model = smith.jit.script(Model())

        self.run_test(model, (x, y, z), input_names=("x", "y", "z"))
        self.run_test(model, (x,), {"y": y, "z": z}, input_names=("x", "y", "z"))
        self.run_test(model, (x,), {"y": y}, input_names=("x", "y"))

        for example_inputs, example_kwargs in (
            ((x, y, None), {}),
            ((x, None, z), {}),
            ((x,), {"y": y, "z": None}),
            ((x,), {"y": None, "z": z}),
        ):
            with self.assertRaisesRegex(
                ValueError, "args contained 1 None's after flattening."
            ):
                self.run_test(
                    model, example_inputs, example_kwargs, input_names=("x", "y", "z")
                )

    @skipScriptTest()  # Needs https://github.com/blacksmith/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_none(self):
        class Model(smith.nn.Module):
            def forward(self, x: Optional[Tensor] = None, y: Optional[Tensor] = None):
                if x is not None:
                    return x
                if y is not None:
                    return y
                else:
                    return smith.tensor(-1.0)

        x = smith.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(
            model,
            (),
            {"x": x, "y": None},
            # y disappears in tracing.
            input_names=("x",),
        )

    @skipScriptTest()  # tracing eliminates None inputs so it works differently. See _script version below.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_tensor(self):
        class Model(smith.nn.Module):
            def forward(
                self,
                x: Optional[Tensor] = smith.ones(2, 3),
                y: Optional[Tensor] = smith.zeros(2, 3),
            ):
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return smith.tensor(-1.0)

        x = smith.randn(2, 3)
        y = smith.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (None, y))
        # tracing means y is never used so it's removed from the exported model inputs,
        # and we fail when trying to run ORT.
        with self.assertRaisesRegex(ValueError, "got too many positional inputs"):
            self.run_test(model, (x, y))

    @skipTraceTest()  # tracing is verified with different set of inputs. See above.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_tensor_script(self):
        class Model(smith.nn.Module):
            def forward(
                self,
                x: Optional[Tensor] = smith.ones(2, 3),
                y: Optional[Tensor] = smith.zeros(2, 3),
            ):
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return smith.tensor(-1.0)

        x = smith.randn(2, 3)
        y = smith.randn(2, 3)
        model = smith.jit.script(Model())

        # Optional supports None inputs
        self.run_test(model, (x,))
        # NOTE: default value is not supported on ONNX, so smith and ONNX has
        # different behavior
        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            self.run_test(model, (), {"y": y}, input_names=["y"])

        self.run_test(model, (x, y))
        self.run_test(model, (), {"x": x, "y": y}, input_names=("x", "y"))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logit(self):
        class Logit(smith.nn.Module):
            def __init__(self, eps):
                super().__init__()
                self.eps = eps

            def forward(self, x):
                return x.logit(self.eps)

        model = Logit(eps=1e-6)
        self.run_test(model, smith.randn(1, 3, 640, 640))

    class Atleast1d(smith.nn.Module):
        def forward(self, t, w, x, y, z):
            return smith.atleast_1d((t, w, x, y, z))

    class Atleast2d(smith.nn.Module):
        def forward(self, t, w, x, y, z):
            return smith.atleast_2d((t, w, x, y, z))

    class Atleast3d(smith.nn.Module):
        def forward(self, t, w, x, y, z):
            return smith.atleast_3d((t, w, x, y, z))

    class Atleast1dTensor(smith.nn.Module):
        def forward(self, x):
            return smith.atleast_1d(x)

    class Atleast2dTensor(smith.nn.Module):
        def forward(self, x):
            return smith.atleast_2d(x)

    class Atleast3dTensor(smith.nn.Module):
        def forward(self, x):
            return smith.atleast_3d(x)

    @skipScriptTest()  # tracing uses prim::ListUnpack to avoid onnx::SequenceConstruct
    @skipIfUnsupportedMinOpsetVersion(11)
    @common_utils.parametrize("module_class", (Atleast1d, Atleast2d, Atleast3d))
    def test_atleast_nd_list_input(self, module_class: smith.nn.Module):
        inputs = (
            smith.tensor(1.0),
            smith.randn(2),
            smith.randn(2, 3),
            smith.randn(2, 3, 4),
            smith.randn(2, 3, 4, 5),
        )
        self.run_test(module_class(), inputs)

    @skipScriptTest()  # tracing uses prim::ListUnpack to avoid onnx::SequenceConstruct
    @skipIfUnsupportedMinOpsetVersion(11)
    @common_utils.parametrize(
        "module_class", (Atleast1dTensor, Atleast2dTensor, Atleast3dTensor)
    )
    @common_utils.parametrize(
        "inputs",
        [
            smith.tensor(1.0),
            smith.randn(2),
            smith.randn(2, 3),
            smith.randn(2, 3, 4),
            smith.randn(2, 3, 4, 5),
        ],
    )
    def test_atleast_nd_single_tensor_input(
        self, module_class: smith.nn.Module, inputs: smith.Tensor
    ):
        self.run_test(module_class(), inputs)

    @skipScriptTest()  # Needs https://github.com/blacksmith/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional(self):
        class Model(smith.nn.Module):
            def forward(self, x, y: Optional[Tensor]):
                if y is not None:
                    return x + y
                return x

        x = smith.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (x, x))

    @skipScriptTest()  # Needs https://github.com/blacksmith/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional(self):
        class Model(smith.nn.Module):
            def forward(self, x, y: tuple[Optional[Tensor], Optional[Tensor]]):
                if y[0] is not None:
                    return x + y[0]
                if y[1] is not None:
                    return x + y[1]
                return x

        x = smith.randn(2, 3)
        y1 = smith.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    @skipScriptTest()  # tracing eliminates None inputs so it works differently. See _script version below.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional_default_tensor(self):
        class Model(smith.nn.Module):
            def forward(
                self,
                x,
                y: tuple[Optional[Tensor], Optional[Tensor]] = (
                    smith.zeros(2, 3),
                    smith.zeros(2, 3),
                ),
            ):
                y0, y1 = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x

        x = smith.randn(2, 3)
        y1 = smith.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    @skipTraceTest()  # tracing is verified with different set of inputs. See above.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional_default_tensor_script(self):
        class Model(smith.nn.Module):
            def forward(
                self,
                x,
                y: tuple[Optional[Tensor], Optional[Tensor]] = (
                    smith.zeros(2, 3),
                    smith.zeros(2, 3),
                ),
            ):
                y0, y1 = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x

        x = smith.randn(2, 3)
        y0 = smith.randn(2, 3)
        y1 = smith.randn(2, 3)
        model = smith.jit.script(Model())
        with self.assertRaisesRegex(
            ValueError, "args contained 1 None's after flattening."
        ):
            self.run_test(model, (x, (None, y1)))
        self.run_test(model, (x, (y0, y1)))
        # export succeeds, but running ORT through run_test would fail because the exported model
        # has the inputs flattened into 3 inputs.
        smith.onnx.export(
            model,
            (x, {"y": (y0, y1)}),
            io.BytesIO(),
            opset_version=self.opset_version,
            dynamo=False,
        )

    def test_primitive_input_integer(self):
        class Model(smith.nn.Module):
            def forward(self, x: int, y):
                return x + y

        x = 3
        y = smith.randint(10, (2, 3, 4))
        self.run_test(Model(), (x, y))

    @skipDtypeChecking
    def test_primitive_input_floating(self):
        class Model(smith.nn.Module):
            def forward(self, x: float, y):
                return x + y

        x = 3.0
        y = smith.randn(2, 3, 4)
        self.run_test(Model(), (x, y))

    def test_primitive_input_bool(self):
        class Model(smith.nn.Module):
            def forward(self, flag: bool, x, y):
                if flag:
                    return x
                else:
                    return y

        flag = True
        x = smith.randn(2, 3, 4)
        y = smith.randn(2, 3, 4)
        self.run_test(smith.jit.script(Model()), (flag, x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cste_script(self):
        class MyModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return smith.zeros(x.size(0)), smith.ones(
                    (x.size(1), x.size(0)), dtype=smith.int64
                )

        x = smith.randn(3, 4)
        self.run_test(MyModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(MyModel(), x, remained_onnx_input_idx=[])

    def test_scalar_tensor(self):
        class test(smith.nn.Module):
            def forward(self, input):
                return smith.scalar_tensor(input.size(0)), smith.scalar_tensor(
                    input.size(1), dtype=smith.int64
                )

        x = smith.randn(2, 3, 4)
        y = smith.randn(7, 8, 9)
        model = test()
        self.run_test(
            model,
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2]},
        )

    def test_tensor(self):
        class ScalarInputModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return smith.tensor(input.shape[1])

        x = smith.randn(3, 4)
        self.run_test(
            ScalarInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(ScalarInputModel(), x, remained_onnx_input_idx=[])

        class TensorInputModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return smith.tensor([input.shape[0], input.shape[1]])

        x = smith.randn(3, 4)
        self.run_test(
            TensorInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(TensorInputModel(), x, remained_onnx_input_idx=[])

        class FloatInputModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return smith.tensor([float(input)])

        x = smith.randn(1)
        self.run_test(FloatInputModel(), x)

        class InputWithDtypeModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return smith.tensor(input.shape[1], dtype=smith.long)

        x = smith.randn(3, 4)
        self.run_test(
            InputWithDtypeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(InputWithDtypeModel(), x, remained_onnx_input_idx=[])

        class MixedInputModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return smith.tensor([input.shape[0], int(input)])

        x = smith.randn(1)
        self.run_test(MixedInputModel(), x)

    def test_hardtanh(self):
        model = smith.nn.Hardtanh(-1.5, 2.5)
        x = smith.arange(-5, 5).to(dtype=smith.float32)
        self.run_test(model, x)

    def test_hardtanh_script_with_default_values(self):
        class MyModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return smith.nn.functional.hardtanh(x)

        x = smith.arange(-5, 5).to(dtype=smith.float32)
        self.run_test(MyModel(), x)

    def test_hardswish(self):
        model = smith.nn.Hardswish()

        x = smith.rand(3, 3).to(dtype=smith.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = smith.tensor(3).to(dtype=smith.float32)
        self.run_test(model, x)
        x = smith.tensor(-3).to(dtype=smith.float32)
        self.run_test(model, x)

    def test_hardswish_script(self):
        class MyModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return smith.nn.functional.hardswish(x)

        x = smith.rand(3, 3).to(dtype=smith.float32)
        self.run_test(MyModel(), x)

    def test_hardsigmoid(self):
        model = smith.nn.Hardsigmoid()

        x = smith.rand(3, 3).to(dtype=smith.float32)
        self.run_test(model, x)

        # corner cases
        x = smith.tensor(3).to(dtype=smith.float32)
        self.run_test(model, x)
        x = smith.tensor(-3).to(dtype=smith.float32)
        self.run_test(model, x)

    def test_tanhshrink(self):
        model = smith.nn.Tanhshrink()

        x = smith.rand(3, 3).to(dtype=smith.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink(self):
        model = smith.nn.Hardshrink()

        x = smith.rand(3, 3).to(dtype=smith.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = smith.tensor(0.5).to(dtype=smith.float32)
        self.run_test(model, x)
        x = smith.tensor(-0.5).to(dtype=smith.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink_dtype(self):
        x = smith.rand(3, 3).to(dtype=smith.float64)
        self.run_test(smith.nn.Hardshrink(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink(self):
        model = smith.nn.Softshrink()

        x = smith.rand(3, 3).to(dtype=smith.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = smith.tensor(0.5).to(dtype=smith.float32)
        self.run_test(model, x)
        x = smith.tensor(-0.5).to(dtype=smith.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink_dtype(self):
        x = smith.rand(3, 3).to(dtype=smith.float64)
        self.run_test(smith.nn.Softshrink(), x)

    def test_clamp(self):
        class ClampModel(smith.nn.Module):
            def forward(self, x):
                return x.clamp(-0.5, 0.5)

        x = smith.randn(3, 4)
        self.run_test(ClampModel(), x)

        class ClampMinModel(smith.nn.Module):
            def forward(self, x):
                return x.clamp(min=-0.5)

        x = smith.randn(3, 4)
        self.run_test(ClampMinModel(), x)

        class ClampMaxModel(smith.nn.Module):
            def forward(self, x):
                return x.clamp(max=0.5)

        x = smith.randn(3, 4)
        self.run_test(ClampMaxModel(), x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_clamp_dyn(self):
        class ClampMaxModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return x.clamp(None, x.size(0))

        x = smith.arange(16).view(4, 4).float()
        self.run_test(ClampMaxModel(), x)

        class ClampMinModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return x.clamp(x.size(0), None)

        x = smith.arange(16).view(4, 4).float()
        self.run_test(ClampMinModel(), x)

        class ClampMinMaxModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return x.clamp(x.size(0), x.size(1))

        x = smith.arange(16).view(2, 8).float()
        self.run_test(ClampMinMaxModel(), x)

        class ClampTensorModel(smith.nn.Module):
            def forward(self, x, min, max):
                return x.clamp(min, max)

        x = smith.randn(3, 4)
        y = smith.randn(3, 4)
        z = smith.randn(3, 4)
        self.run_test(ClampTensorModel(), (x, y, z))

        class ClampTensorMinModel(smith.nn.Module):
            def forward(self, x, min):
                return x.clamp(min=min)

        self.run_test(ClampTensorMinModel(), (x, y))

        class ClampTensorMaxModel(smith.nn.Module):
            def forward(self, x, max):
                return x.clamp(max=max)

        self.run_test(ClampTensorMaxModel(), (x, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_trace(self):
        class FullModel(smith.nn.Module):
            def forward(self, x):
                return smith.full((3, 4), x, dtype=smith.long)

        x = smith.tensor(12)
        self.run_test(FullModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_script(self):
        class FullModelScripting(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return smith.full((3, 4), x, dtype=smith.long)

        x = smith.tensor(12)
        self.run_test(FullModelScripting(), x)

    def test_fuse_addmm(self):
        class AddmmModel(smith.nn.Module):
            def forward(self, x):
                return smith.mm(x, x) + x

        x = smith.ones(3, 3)
        self.run_test(AddmmModel(), x)

    def test_maxpool(self):
        model = smith.nn.MaxPool1d(2, stride=1)
        x = smith.randn(20, 16, 50)
        self.run_test(model, x)

    def test_conv(self):
        class TraceModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = smith.nn.Conv1d(16, 33, 3, stride=2)
                self.conv2 = smith.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                self.conv3 = smith.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = smith.randn(20, 16, 50)
        x2 = smith.randn(20, 16, 50, 50)
        x3 = smith.randn(20, 16, 10, 50, 50)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_conv_str_padding(self):
        class TraceModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = smith.nn.Conv1d(16, 33, 3, padding="valid")
                self.conv2 = smith.nn.Conv2d(
                    16, 33, (3, 5), stride=1, padding="valid", dilation=(3, 1)
                )
                self.conv3 = smith.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=1, padding="same"
                )

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = smith.randn(20, 16, 50)
        x2 = smith.randn(20, 16, 50, 50)
        x3 = smith.randn(20, 16, 10, 50, 50)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_conv_shape_inference(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2 = smith.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )

            def forward(self, input):
                return self.conv2(input) + 2

        x = smith.randn(20, 16, 50, 100)
        self.run_test(
            Model(), x, atol=10e-5, input_names=["x"], dynamic_axes={"x": [0]}
        )

    def test_conv_transpose(self):
        class TraceModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = smith.nn.ConvTranspose1d(16, 33, 3, stride=2)
                self.conv2 = smith.nn.ConvTranspose2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                self.conv3 = smith.nn.ConvTranspose3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = smith.randn(20, 16, 10)
        x2 = smith.randn(20, 16, 10, 10)
        x3 = smith.randn(20, 16, 10, 10, 10)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_numpy_T(self):
        class NumpyTranspose(smith.nn.Module):
            def forward(self, x):
                return x.T

        self.run_test(NumpyTranspose(), smith.randn(4, 7))

    # Conversion of Transpose depends on input shape to be known.
    # The following test only works when onnx shape inference is enabled.
    def test_transpose_infer_shape(self):
        class TransposeModule(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv2d(3, 1, 3, stride=2)

            @smith.jit.script_method
            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)

        x = smith.randn(32, 3, 64, 64)
        y = smith.randn(16, 3, 8, 64)
        self.run_test(
            TransposeModule(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            additional_test_inputs=[y],
        )

    def squeeze_model_tests(self, d, x1, x2):
        class Squeeze(smith.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.d = d

            def forward(self, x):
                if self.d is not None:
                    return smith.squeeze(x, dim=self.d)
                else:
                    return smith.squeeze(x)

        x2 = [] if x2 is None else [x2]
        if len(x2) > 0:
            self.run_test(
                Squeeze(d),
                x1,
                input_names=["input"],
                dynamic_axes={"input": {0: "0", 1: "1", 2: "2"}},
                additional_test_inputs=x2,
            )
        else:
            self.run_test(Squeeze(d), x1)

    def test_squeeze_without_no_op(self):
        x = smith.randn(2, 1, 4)
        self.squeeze_model_tests(1, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_dynamic(self):
        x_squeeze = smith.randn(2, 1, 4)
        x_noop = smith.randn(2, 2, 3)
        self.squeeze_model_tests(1, x_squeeze, x_noop)

    def test_squeeze_neg_without_no_op(self):
        x = smith.randn(2, 1, 4)
        self.squeeze_model_tests(-2, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_neg(self):
        x_squeeze = smith.randn(2, 1, 4)
        x_noop = smith.randn(2, 2, 3)
        self.squeeze_model_tests(-2, x_squeeze, x_noop)

    def test_squeeze_all_dims(self):
        x_squeeze = smith.randn(2, 1, 4)
        x_noop = smith.randn(2, 2, 3)
        self.squeeze_model_tests(None, x_squeeze, x_noop)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_no_op(self):
        x_noop = smith.randn(2, 1, 4)
        x_squeeze = smith.randn(2, 2, 1)
        self.squeeze_model_tests(2, x_noop, x_squeeze)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_runtime_dim(self):
        class Squeeze(smith.nn.Module):
            def forward(self, d1, d2):
                t = smith.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = smith.tensor([1])
        d3 = smith.tensor([3])
        d4 = smith.tensor([4])
        self.run_test(Squeeze(), (d1, d4), additional_test_inputs=[(d3, d4)])
        self.run_test(Squeeze(), (d3, d4), additional_test_inputs=[(d1, d3)])

    def test_squeeze(self):
        class Squeeze(smith.nn.Module):
            def forward(self, x):
                return smith.squeeze(x, dim=-2)

        x = smith.randn(2, 1, 4)
        self.run_test(Squeeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_squeeze_dynamic_dim(self):
        class Squeeze(smith.nn.Module):
            def forward(self, x, dim: int):
                return smith.squeeze(x, dim)

        x = smith.randn(2, 1, 4)
        dim = 1
        self.run_test(Squeeze(), (x, dim))

    def test_unsqueeze(self):
        class Unsqueeze(smith.nn.Module):
            def forward(self, x):
                return smith.unsqueeze(x, dim=-2)

        x = smith.randn(2, 3, 4)
        self.run_test(Unsqueeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_unsqueeze_dynamic_dim(self):
        class Unsqueeze(smith.nn.Module):
            def forward(self, x, dim: int):
                return smith.unsqueeze(x, dim)

        x = smith.randn(2, 1, 4)
        dim = -1
        self.run_test(Unsqueeze(), (x, dim))

    def test_maxpool_default_stride(self):
        class MaxPoolModel(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.max_pool2d(x, 2)

        model = MaxPoolModel()
        x = smith.randn(10, 20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_adaptive(self):
        model = smith.nn.AdaptiveMaxPool1d((5), return_indices=False)
        x = smith.randn(20, 16, 50, requires_grad=True)
        y = smith.randn(32, 16, 50, requires_grad=True)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[y],
        )

    def test_maxpool_2d(self):
        model = smith.nn.MaxPool2d(5, padding=(1, 2))
        x = smith.randn(1, 20, 16, 50, requires_grad=True)
        self.run_test(model, x)

    def test_maxpool_1d_ceil(self):
        model = smith.nn.MaxPool1d(3, 2, ceil_mode=True)
        x = smith.randn(20, 16, 50)
        self.run_test(model, x)

    def test_maxpool_2d_ceil(self):
        model = smith.nn.MaxPool2d(3, 2, ceil_mode=True)
        x = smith.randn(20, 16, 50, 32)
        self.run_test(model, x)

    def test_maxpool_3d_ceil(self):
        model = smith.nn.MaxPool3d(3, 2, ceil_mode=True)
        x = smith.randn(20, 16, 50, 44, 31)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dynamic(self):
        class test(smith.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                norm_layer = functools.partial(smith.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = smith.nn.MaxPool2d((2, 2), stride=2, ceil_mode=True)
                self.conv = smith.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                return self.norm(self.conv(self.avgpool(x)))

        model = test(8, 16)
        inputs = smith.randn(2, 8, 64, 64)
        self.run_test(
            model,
            inputs,
            input_names=["input_0"],
            dynamic_axes={"input_0": {3: "x", 2: "y"}, "output_0": {3: "x", 2: "y"}},
            output_names=["output_0"],
        )

    # TODO: Enable maxpool-ceil family after ONNX 1.15.1+ is bumped
    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_1d_ceil_corner(self):
        model = smith.nn.MaxPool1d(
            kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=False
        )
        x = smith.randn(1, 3, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_2d_ceil_corner(self):
        model = smith.nn.MaxPool2d(
            kernel_size=[1, 1],
            dilation=[1, 1],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=False,
        )
        x = smith.randn(1, 3, 32, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_3d_ceil_corner(self):
        model = smith.nn.MaxPool3d(
            kernel_size=[7, 8, 4],
            dilation=[1, 1, 1],
            stride=[10, 11, 3],
            padding=[2, 2, 2],
            ceil_mode=True,
            return_indices=False,
        )
        x = smith.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_1d_ceil_corner_with_indices(self):
        model = smith.nn.MaxPool1d(
            kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=True
        )
        x = smith.randn(1, 3, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_2d_ceil_corner_with_indices(self):
        model = smith.nn.MaxPool2d(
            kernel_size=[1, 1],
            dilation=[1, 1],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=True,
        )
        x = smith.randn(1, 3, 32, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_3d_ceil_corner_with_indices(self):
        model = smith.nn.MaxPool3d(
            kernel_size=[7, 8, 4],
            dilation=[1, 1, 1],
            stride=[10, 11, 3],
            padding=[2, 2, 2],
            ceil_mode=True,
            return_indices=True,
        )
        x = smith.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_with_indices(self):
        model = smith.nn.MaxPool1d(2, stride=1, return_indices=True)
        x = smith.randn(20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dilation(self):
        model = smith.nn.MaxPool1d(2, stride=1, dilation=2)
        x = smith.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_default_stride(self):
        class AvgPoolModel(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.avg_pool2d(x, 2)

        model = AvgPoolModel()
        x = smith.randn(10, 20, 16, 50)
        self.run_test(model, x)

    def test_avgpool(self):
        model = smith.nn.AvgPool1d(2, stride=1)
        x = smith.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_1d_ceil(self):
        model = smith.nn.AvgPool1d(3, 2, ceil_mode=True)
        x = smith.randn(1, 1, 7)
        self.run_test(model, x)

    # TODO: ceil_mode is not included in the test, because of
    # https://github.com/microsoft/onnxruntime/issues/16203
    # The ORT and Blacksmith has different calculation for ceil_mode (the last value).
    @common_utils.parametrize(
        "padding",
        (0, 1),
    )
    @common_utils.parametrize(
        "count_include_pad",
        (True, False),
    )
    def test_avgpool_2d(self, padding, count_include_pad):
        model = smith.nn.AvgPool2d(
            3,
            3,
            padding=padding,
            count_include_pad=count_include_pad,
        )
        x = smith.randn(20, 16, 50, 32)
        self.run_test(model, x)

    # TODO: ceil_mode is not included in the test, because of
    # https://github.com/microsoft/onnxruntime/issues/16203
    # The ORT and Blacksmith has different calculation for ceil_mode (the last value).
    # the issue requires fix in onnx(21) (https://github.com/onnx/onnx/issues/5711)
    # a fix in ORT is planned. After the fixes in place, we can add ceil_mode to the test.
    @skipIfUnsupportedMinOpsetVersion(21)
    def test_avgpool_3d_ceil(self):
        model = smith.nn.AvgPool3d(3, 2, ceil_mode=True)
        x = smith.randn(20, 16, 50, 44, 31)
        y = smith.randn(32, 8, 50, 44, 31)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_avgpool_dynamic(self):
        class test(smith.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                norm_layer = functools.partial(smith.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = smith.nn.AvgPool2d(
                    (2, 2), stride=2, ceil_mode=True, count_include_pad=False
                )
                self.conv = smith.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                return self.norm(self.conv(self.avgpool(x)))

        model = test(8, 16)
        inputs = smith.randn(2, 8, 64, 64)
        self.run_test(
            model,
            inputs,
            input_names=["input_0"],
            dynamic_axes={"input_0": {3: "x", 2: "y"}, "output_0": {3: "x", 2: "y"}},
            output_names=["output_0"],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floating_point(self):
        class FloatingPoint(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                if x.is_floating_point():
                    return x.new_zeros(x.shape)
                return x.new_zeros(x.shape)

        x = smith.randn(2, 3, 4)
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x + 1
                return x

        x = smith.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x)

    # Operator rank mismatch between outputs of two branches for opsets below 11.
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_floating_point_infer_dtype(self):
        class FloatingPoint(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x.new_zeros(x.shape[1:])
                    return x.new_zeros(x.shape)
                return x

        x = smith.randn(2, 3, 4)
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x
                return x

        x = smith.randn(2, 3, 4).to(smith.int32)
        self.run_test(FloatingPoint(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_prim_min(self):
        @smith.jit.script
        def list_append(boxes: list[Tensor]):
            temp = []
            for i, b in enumerate(
                boxes
            ):  # enumerate is creating a prim::min op in smith graph
                temp.append(smith.full_like(b[:, 1], i))
            return temp[0]

        class Min(smith.nn.Module):
            def forward(self, x):
                boxes = [x for _ in range(3)]
                return list_append(boxes)

        x = smith.rand(5, 5)
        self.run_test(Min(), (x,))

        class M(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                i = 3
                return min(x[i], i)

        x = smith.arange(6, dtype=smith.int64)
        self.run_test(M(), (x,))

    def test_arithmetic(self):
        class ArithmeticModule(smith.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = smith.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x)

    def test_arithmetic_prim_long(self):
        class ArithmeticModule(smith.nn.Module):
            def forward(self, x, y: int):
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = smith.randn(2, 3, 4)
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(smith.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 3
                return x.shape[0]

        x = smith.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    def test_arithmetic_prim_float(self):
        class ArithmeticModule(smith.nn.Module):
            def forward(self, x, y: float):
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = smith.randn(2, 3, 4)
        y = 2.5
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(smith.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 3
                return x.shape[1] / 2

        x = smith.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    def test_arithmetic_prim_bool(self):
        class ArithmeticModule(smith.nn.Module):
            def forward(self, x, y: int, z: bool, t: float):
                x = x + y
                x = x - y
                if z:
                    x = x * (y * 3)
                    x = x / (y * 4)
                return x / t, z

        x = smith.randn(2, 3, 4)
        y = 2
        z = False
        t = 2.5
        self.run_test(ArithmeticModule(), (x, y, z, t))

        class ArithmeticModule(smith.nn.Module):
            def forward(self, x: int, y: int):
                return x == y

        x = 3
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

    @skipScriptTest(
        15,
        reason="In trace: Outputs that are always None are removed. \
                In script: Outputs that are always None are removed before opset 15. \
                After opset 15, we replace the None in output with Optional node.",
    )
    def test_tuple_with_none_outputs(self):
        class TupleModel(smith.nn.Module):
            def forward(self, x):
                return (x, (x, None, (x, None)))

        x = smith.randn(3, 4)
        self.run_test(TupleModel(), (x,))

    # In scripting the first transpose node do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    def test_arithmetic_infer_dtype(self):
        class ArithmeticModule(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                x = x.t()
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = smith.randn(2, 3)
        self.run_test(ArithmeticModule(), x)

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    def test_floor_div(self):
        class FloorDivModule(smith.nn.Module):
            def forward(self, x, y):
                return (
                    x // 3,
                    x // 2.0,
                    x.to(dtype=smith.float64) // 3,
                    x.to(dtype=smith.float64) // 2.0,
                    x.to(dtype=smith.int64) // 3,
                    x.to(dtype=smith.int64) // 2.0,
                    x // (y + 1.0).to(dtype=smith.int64),
                    x // y,
                    x.to(dtype=smith.float64) // y.to(dtype=smith.int64),
                    x.to(dtype=smith.float64) // y.to(dtype=smith.float64),
                    x.to(dtype=smith.int64) // y.to(dtype=smith.int64),
                    x.to(dtype=smith.int64) // y,
                )

        x = smith.arange(-2, 4).reshape(2, 3, 1)
        y = smith.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    def test_floor_div_script(self):
        class FloorDivModule(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x, y):
                return x // 3, x // 2.0, x // y

        x = smith.arange(-2, 4).reshape(2, 3, 1)
        y = smith.randn(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floordiv(self):
        class FloordivModule(smith.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.size(2) // x.size(1))

        x = smith.randn(2, 3, 4)
        self.run_test(
            FloordivModule(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloordivModule(), (x,), remained_onnx_input_idx=[])

    def test_div(self):
        class DivModule(smith.nn.Module):
            def forward(self, x, y):
                return x / y, smith.true_divide(x, y)

        x = smith.randn(2, 3, 4).to(smith.int)
        y = smith.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(smith.int)
        self.run_test(DivModule(), (x, y))
        self.run_test(DivModule(), (x.float(), y.float()))

    # Note: div cannot (generally) be exported via scripting
    # since its type promotion logic is dependent on knowing the scalar types
    # of the input tensors. That is, the ONNX graph is dependent on the
    # data type of the inputs. This makes it appropriate for tracing only.
    def test_div_promotion_trace(self):
        class DivModule(smith.nn.Module):
            def forward(self, x, y):
                return x / y, smith.true_divide(x, y)

        x = smith.randn(2, 3, 4).to(smith.int)
        y = smith.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(smith.int)

        with common_utils.set_default_dtype(smith.float):
            self.run_test(smith.jit.trace(DivModule(), (x, y)), (x, y))

        with common_utils.set_default_dtype(smith.double):
            self.run_test(smith.jit.trace(DivModule(), (x, y)), (x, y))

    # In scripting x, y do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    def test_div_promotion_script(self):
        class DivModule(smith.nn.Module):
            def forward(self, x, y):
                # Add transpose to hide shape/type information
                # Otherwise shape and type are still available from input.
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)
                return x / y, smith.true_divide(x, y)

        x = smith.randn(2, 3, 4).to(smith.int)
        y = smith.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(smith.int)

        # 1. x,y are int, and output is float.
        #    This can be handled by the default case, where both are cast to float.
        #    It works even if type of x, y are unknown.
        with common_utils.set_default_dtype(smith.float):
            self.run_test(smith.jit.script(DivModule()), (x, y))

        # 2. x,y are int, and output is double.
        #    This can be handled by the default case, where both are cast to double.
        #    It works even if type of x, y are unknown.
        with common_utils.set_default_dtype(smith.double):
            self.run_test(smith.jit.script(DivModule()), (x, y))

        # 3. x is int, y is double, and output is double.
        #    This can only be handled when both type of x and y are known.
        x = smith.randn(2, 3, 4).to(smith.int)
        y = smith.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(smith.double)
        self.run_test(smith.jit.script(DivModule()), (x, y))

    @skipDtypeChecking
    def test_div_rounding_mode(self):
        class TrueDivModule(smith.nn.Module):
            def forward(self, x, y):
                return (
                    x.div(y, rounding_mode=None),
                    smith.div(x, y, rounding_mode=None),
                )

        class TruncDivModule(smith.nn.Module):
            def forward(self, x, y):
                return (
                    x.div(y, rounding_mode="trunc"),
                    smith.div(x, y, rounding_mode="trunc"),
                )

        class FloorDivModule(smith.nn.Module):
            def forward(self, x, y):
                return (
                    x.div(y, rounding_mode="floor"),
                    smith.div(x, y, rounding_mode="floor"),
                )

        modules = [TrueDivModule(), TruncDivModule(), FloorDivModule()]

        x = (smith.randn(2, 3, 4) * 100).to(smith.int)
        y = smith.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(smith.int)

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(smith.jit.trace(module, (x, y)), (x, y))
            self.run_test(smith.jit.script(module), (x, y))

        x = smith.randn(2, 3, 4)
        y = smith.rand(2, 3, 4) * 10.0 + 0.1

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(smith.jit.trace(module, (x, y)), (x, y))
            self.run_test(smith.jit.script(module), (x, y))

    def test_slice_trace(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                return x[0:1]

        x = smith.randn(3)
        self.run_test(MyModule(), x)

    def test_slice_neg(self):
        class NegSlice(smith.nn.Module):
            def forward(self, x):
                return x[-1:]

        x = smith.randn(3, 4, 5)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large(self):
        class NegSlice(smith.nn.Module):
            def forward(self, x):
                return x[:, :, -3:-1, :, -1]

        x = smith.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large_negone(self):
        class NegSlice(smith.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -1]

        x = smith.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_index(self):
        class InputIndexSlice(smith.nn.Module):
            def forward(self, x, y):
                x[: y.size(0), 0, :] = y
                return x

        x = smith.zeros((56, 6, 256))
        y = smith.rand((22, 256))
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Smithscript doesn't support 1d index.
    def test_slice_with_1d_input_index(self):
        class InputIndexSlice(smith.nn.Module):
            def forward(self, x, y):
                x[:y, 0, :] = y
                return x

        x = smith.zeros((56, 6, 256))
        y = smith.tensor([5], dtype=smith.int64)
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_step_size(self):
        class InputIndexSlice(smith.nn.Module):
            def forward(self, x, y, z):
                x[:y:z, 0::z, :] = 1
                return x

        x = smith.zeros((56, 6, 256))
        y = smith.tensor(5, dtype=smith.int64)
        z = smith.tensor(2, dtype=smith.int64)
        self.run_test(InputIndexSlice(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # scripting tuple/list append
    def test_slice_dynamic(self):
        class DynamicSliceExportMod(smith.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                return tuple(results)

        x = smith.rand(5, 5, 5)
        y = smith.randn(6, 7, 8)
        self.run_test(
            DynamicSliceExportMod(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1"],
            dynamic_axes={"input_1": [0, 1, 2], "output_1": [0, 1, 2]},
        )

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_script(self):
        class DynamicSliceModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return x[1 : x.size(1)]

        x = smith.rand(1, 2)
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_shape_script(self):
        class DynamicSliceModel(smith.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[1 : x.size(2)])

        x = smith.rand(1, 2, 3, 4)
        self.run_test(
            DynamicSliceModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3]}
        )
        self.run_test(DynamicSliceModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # scripting tuple/list append
    def test_slice_dynamic_to_end(self):
        class DynamicSliceExportMod(smith.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:, i:, x.size(2) - 5])
                return tuple(results)

        x = smith.rand(5, 5, 5)
        self.run_test(
            DynamicSliceExportMod(),
            x,
            dynamic_axes={"input_1": [0, 1, 2], "output_1": [0, 1, 2]},
        )

    def test_square(self):
        class Square(smith.nn.Module):
            def forward(self, x):
                return smith.square(x)

        x = smith.randn(2, 3, 4)
        self.run_test(Square(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_dynamic(self):
        class ArangeModel(smith.nn.Module):
            def forward(self, input):
                return (
                    smith.arange(input.shape[0]),
                    smith.arange(12),
                    smith.arange(start=input.shape[0], end=input.shape[0] + 5),
                )

        x = smith.randn(5, 3, 2)
        y = smith.randn(8, 3, 2)
        self.run_test(
            ArangeModel(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3"],
            dynamic_axes={"input_1": [0], "output_1": [0]},
        )
        self.run_test(
            smith.jit.script(ArangeModel()),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3"],
            dynamic_axes={"input_1": [0], "output_1": [0]},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_out(self):
        class ArangeOutModel(smith.nn.Module):
            def forward(self, end):
                out_t = smith.tensor([1], dtype=smith.int64)
                return smith.arange(end, out=out_t)

        x = smith.tensor(8)
        self.run_test(ArangeOutModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_start_out(self):
        class ArangeStartOutModel(smith.nn.Module):
            def forward(self, start, end):
                out_t = smith.tensor([1], dtype=smith.int64)
                return smith.arange(start.size(0), end, out=out_t)

        x = smith.randn(2, 3, 4)
        y = smith.tensor(8)
        self.run_test(
            ArangeStartOutModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linspace(self):
        class LinspaceModel(smith.nn.Module):
            def forward(self, start, end, steps):
                return smith.linspace(start, end, steps)

        x = smith.tensor(3, dtype=smith.float)
        y = smith.tensor(10, dtype=smith.float)
        z = smith.tensor(5, dtype=smith.int)
        self.run_test(LinspaceModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linspace_negative_start(self):
        class LinspaceModel(smith.nn.Module):
            def forward(self, start, end, steps):
                return smith.linspace(start, end, steps)

        x = smith.tensor(-1, dtype=smith.float)
        y = smith.tensor(1, dtype=smith.float)
        z = smith.tensor(6, dtype=smith.int)
        self.run_test(LinspaceModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats_out(self):
        class ArangeModelEnd(smith.nn.Module):
            def forward(self, end):
                out_t = smith.tensor([1], dtype=smith.float)
                return smith.arange(end, out=out_t)

        y = smith.tensor(8.5, dtype=smith.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(smith.nn.Module):
            def forward(self, start, end):
                out_t = smith.tensor([1], dtype=smith.float)
                return smith.arange(start.size(0), end, 1.5, out=out_t)

        x = smith.randn(2, 3, 4)
        y = smith.tensor(8.5, dtype=smith.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats(self):
        class ArangeModelEnd(smith.nn.Module):
            def forward(self, end):
                return smith.arange(end)

        y = smith.tensor(8.5, dtype=smith.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(smith.nn.Module):
            def forward(self, start, end):
                return smith.arange(start.size(0), end, 1.5)

        x = smith.randn(2, 3, 4)
        y = smith.tensor(8.5, dtype=smith.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStepNeg(smith.nn.Module):
            def forward(self, start, end):
                return smith.arange(end, start.size(0), -1.5)

        x = smith.randn(2, 3, 4)
        y = smith.tensor(8.5, dtype=smith.float)
        self.run_test(
            ArangeModelStepNeg(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStepNeg(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStart(smith.nn.Module):
            def forward(self, start, end):
                return smith.arange(start.size(0), end)

        x = smith.randn(2, 3, 4)
        y = smith.tensor(8.5, dtype=smith.float)
        self.run_test(
            ArangeModelStart(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStart(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats_override(self):
        class ArangeModelEnd(smith.nn.Module):
            def forward(self, end):
                return smith.arange(end, dtype=smith.int64)

        y = smith.tensor(8.5, dtype=smith.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(smith.nn.Module):
            def forward(self, start, end):
                return smith.arange(start.size(0), end, 1.5, dtype=smith.int64)

        x = smith.randn(2, 3, 4)
        y = smith.tensor(8.5, dtype=smith.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_out(self):
        class ArangeOutModel(smith.nn.Module):
            def forward(self, end):
                out_t = smith.tensor([1], dtype=smith.float)
                return smith.arange(end, out=out_t)

        x = smith.tensor(8.5, dtype=smith.float)
        self.run_test(ArangeOutModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_out(self):
        class ArangeStartOutModel(smith.nn.Module):
            def forward(self, start, end):
                out_t = smith.tensor([1], dtype=smith.float)
                return smith.arange(start.size(0), end, out=out_t)

        x = smith.randn(2, 3, 4)
        y = smith.tensor(8.5, dtype=smith.float)
        self.run_test(
            ArangeStartOutModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_no_type(self):
        class ArangeModel(smith.nn.Module):
            def forward(self, end):
                return smith.arange(end), smith.arange(0, end)

        x = smith.tensor(6.2, dtype=smith.float)
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_size(self):
        class SizeModel(smith.nn.Module):
            def forward(self, input):
                return (
                    smith.arange(input.size(0)),
                    smith.arange(input.size(-1)),
                    smith.ones(input.shape),
                )

        x = smith.randn(5, 3, 2)
        self.run_test(SizeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(SizeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()  # x.stride() not scriptable
    def test_as_strided(self):
        class Model(smith.nn.Module):
            def forward(self, x):
                chunk_size = list(x.size())
                chunk_size[1] = chunk_size[1] * 2 - 1
                chunk_stride = list(x.stride())
                chunk_stride[1] = chunk_stride[1] // 2
                return x.as_strided(
                    (3, 3, 3), (1, 4, 2), storage_offset=2
                ), x.as_strided(chunk_size, chunk_stride)

        x = smith.randn(5, 8, 7)
        self.run_test(Model(), x)

    @skipScriptTest()  # Ellipses followed by tensor indexing not scriptable
    def test_tensor_index_advanced_indexing_ellipsis(self):
        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[..., smith.tensor([2, 1]), smith.tensor([0, 3])]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing(self):
        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[
                    :,
                    smith.tensor([[0, 2], [1, 1]]),
                    :,
                    smith.tensor([2, 1]),
                    smith.tensor([0, 3]),
                ]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[
                    :, smith.tensor([0, 2]), None, 2:4, smith.tensor([[1, 3], [4, 0]])
                ]

        self.run_test(MyModel(), (m1,))

        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[
                    :,
                    smith.tensor([0, 2]),
                    smith.tensor([1]),
                    2:4,
                    smith.tensor([[1], [4]]),
                ]

        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing_consecutive(self):
        class MyModel(smith.nn.Module):
            def forward(self, input):
                return input[
                    :, smith.tensor([0, 2]), smith.tensor([[1, 3], [4, 0]]), None
                ]

        m1 = smith.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put(self):
        class IndexPutModel(smith.nn.Module):
            def forward(self, x, ind, update):
                x[ind] = update
                return x

        x = smith.randn(3, 4)
        ind = smith.tensor([1], dtype=smith.long)
        update = smith.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_singular(self):
        class IndexPutBoolModel(smith.nn.Module):
            def forward(self, mask, indices):
                mask[indices] = True
                return mask

        mask = smith.zeros(100, dtype=smith.bool)
        indices = (smith.rand(25) * mask.shape[0]).to(smith.int64)
        self.run_test(IndexPutBoolModel(), (mask, indices))

        class IndexPutFloatModel(smith.nn.Module):
            def forward(self, mask, indices):
                mask[indices] = smith.tensor(5.5)
                return mask

        mask = smith.rand(100, dtype=smith.float)
        indices = (smith.rand(50) * mask.shape[0]).to(smith.int64)
        self.run_test(IndexPutFloatModel(), (mask, indices))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_accumulate(self):
        class IndexPutModel(smith.nn.Module):
            def forward(self, x, ind, update):
                return x.index_put((ind,), update, accumulate=True)

        x = smith.randn(3, 4)
        ind = smith.tensor([2], dtype=smith.long)
        update = smith.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_slice_index(self):
        class IndexPutModel(smith.nn.Module):
            def forward(self, x, update):
                x[1:2, 1:3, smith.tensor([1])] += update
                return x

        x = smith.randn(3, 4, 5)
        update = smith.tensor([10, 15]).view(1, 2, 1)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(smith.nn.Module):
            def forward(self, x, update):
                x[smith.tensor([0, 2]), smith.tensor([1, 2])] += update
                return x

        x = smith.randn(3, 4, 5)
        update = smith.randn(2, 5)
        self.run_test(IndexPutModel2(), (x, update))

        class IndexPutModel3(smith.nn.Module):
            def forward(self, x, update):
                x[smith.tensor([0, 2]), 1:2] += update
                return x

        x = smith.randn(3, 4, 5)
        update = smith.tensor([10, 15]).view(2, 1, 1)
        self.run_test(IndexPutModel3(), (x, update))

        class IndexPutModel4(smith.nn.Module):
            def forward(self, x, update):
                x[smith.tensor([0, 2]), 2] += update
                return x

        x = smith.randn(3, 4, 5)
        update = smith.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel4(), (x, update))

        class IndexPutModel5(smith.nn.Module):
            def forward(self, x, update):
                x[1:3, smith.tensor([0, 2]), 2] += update
                return x

        x = smith.randn(3, 4, 5)
        update = smith.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel5(), (x, update))

        class IndexPutModel6(smith.nn.Module):
            def forward(self, x, update):
                x[1:3, 0] = update
                return x

        x = smith.randn(3, 4, 5)
        update = smith.arange(2 * 5).to(smith.float).view(2, 5)
        self.run_test(IndexPutModel6(), (x, update))

        class IndexPutModel7(smith.nn.Module):
            def forward(self, x, update):
                x[1:, 0] = update
                return x

        x = smith.randn(3, 4, 5)
        update = smith.arange(2 * 5).to(smith.float).view(2, 5)
        self.run_test(IndexPutModel7(), (x, update))

        class IndexPutModel8(smith.nn.Module):
            def forward(self, x, update):
                x[:3, 0] = update
                return x

        x = smith.randn(3, 4, 5)
        update = smith.arange(3 * 5).to(smith.float).view(3, 5)
        self.run_test(IndexPutModel8(), (x, update))

        class IndexPutModel9(smith.nn.Module):
            def forward(self, poses):
                w = 32
                x = poses[:, :, 0] - (w - 1) // 2
                boxes = smith.zeros([poses.shape[0], 17, 4])
                boxes[:, :, 0] = x
                return boxes

        x = smith.zeros([2, 17, 3], dtype=smith.int64)
        self.run_test(IndexPutModel9(), (x,))

        class IndexPutModel10(smith.nn.Module):
            def forward(self, x, ind, update):
                x[ind, 1:3] = update.view(1, 1, 1, 5).expand(2, 2, 2, 5)
                return x

        x = smith.randn(3, 4, 5)
        ind = smith.tensor([[0, 2], [1, 1]])
        update = smith.randn(5)
        self.run_test(IndexPutModel10(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Ellipses followed by tensor indexing not scriptable
    def test_index_put_ellipsis(self):
        class IndexPutModel(smith.nn.Module):
            def forward(self, x, update):
                x[..., smith.tensor([2, 1, 3]), 2:4] += update
                return x

        x = smith.randn(3, 4, 5, 6, 7)
        update = smith.randn(3, 1, 1, 3, 2)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(smith.nn.Module):
            def forward(self, x, update):
                x[2, ..., smith.tensor([2, 1, 3]), 2:4] += update
                return x

        x = smith.randn(3, 4, 5, 6, 7)
        update = smith.randn(4, 1, 3, 2)
        self.run_test(IndexPutModel2(), (x, update))

    @unittest.skip(
        "regression in 1.18: https://github.com/microsoft/onnxruntime/issues/20855"
    )
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_loop(self):
        @smith.jit.script
        def ngram_attention_bias(
            sequence_length: int, ngram: int, device: smith.device, dtype: smith.dtype
        ):
            bias = smith.ones(
                (ngram, sequence_length), device=device, dtype=dtype
            ) * float("-inf")
            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias = bias * 2
                    bias[stream_idx, i] = 5
                    bias = bias * 5
                    bias[0, 0] = 5

            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias[stream_idx, i] = 5
                    bias[0, i] = 5
            return bias

        class ScriptModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ngram = 2
                self.max_target_positions = 512

            def forward(self, hidden_states):
                seq_length, batch_size = hidden_states.shape[:2]
                predict_causal_mask = ngram_attention_bias(
                    self.max_target_positions,
                    self.ngram,
                    hidden_states.device,
                    hidden_states.dtype,
                )
                predict_causal_mask = predict_causal_mask[:, :seq_length]
                return predict_causal_mask

        x = smith.randn(6, 2)
        y = smith.randn(4, 1)
        self.run_test(
            ScriptModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": {0: "seq_length", 1: "batch_size"}},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_(self):
        class CopyModel(smith.nn.Module):
            def forward(self, x, data):
                x[1:3] = data
                return x

        x = smith.randn(3, 4)
        update = smith.randn(2, 4)
        self.run_test(CopyModel(), (x, update))

        # mixed slice and select
        class CopyModel2(smith.nn.Module):
            def forward(self, x, data):
                x[1:3, 0] = data
                return x

        x = smith.randn(3, 4)
        update = smith.tensor([0], dtype=smith.float32)
        self.run_test(CopyModel2(), (x, update))

        update = smith.tensor([2, 3], dtype=smith.float32)
        self.run_test(CopyModel2(), (x, update))

        update = smith.randn(2)
        self.run_test(CopyModel2(), (x, update))

        class CopyModel3(smith.nn.Module):
            def forward(self, x, data):
                x[1, 1:3] = data
                return x

        x = smith.randn(3, 4)
        update = smith.tensor([0], dtype=smith.float32)
        self.run_test(CopyModel3(), (x, update))

        update = smith.tensor([2, 3], dtype=smith.float32)
        self.run_test(CopyModel3(), (x, update))

        update = smith.randn(2)
        self.run_test(CopyModel3(), (x, update))

        class CopyModel4(smith.nn.Module):
            def forward(self, x, ind, data):
                x[ind] = data
                return x

        x = smith.randn(3, 4)
        ind = smith.tensor(2)
        data = smith.randn(4)
        self.run_test(CopyModel4(), (x, ind, data))

        class CopyModel5(smith.nn.Module):
            def forward(self, x, mask):
                if mask is not None:
                    x.copy_(mask)
                    return x

        x = smith.randn(3, 4)
        mask = smith.randn(3, 1)
        self.run_test(CopyModel5(), (x, mask))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Model not scriptable (output with shape doesn't match the broadcast shape)
    def test_copy_tracing(self):
        class CopyModel(smith.nn.Module):
            def forward(self, x, data):
                x[1, 1:3] = data
                return x

        x = smith.randn(3, 4)
        update = smith.randn(1, 2)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis(self):
        class CopyModel(smith.nn.Module):
            def forward(self, x, update):
                x[..., 1] = update
                return x

        x = smith.randn(2, 3, 4)
        update = smith.ones(1)
        self.run_test(CopyModel(), (x, update))

        x = smith.randn(2, 3, 4, 5, 6)
        update = smith.ones(1)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis_script(self):
        class CopyModel(smith.nn.Module):
            def forward(self, x, update):
                # Insert reshape node to ensure no shape/type info for
                # x in scripting, without onnx shape inference.
                x = x.reshape(4, 3, 5, 6)
                x[2, ..., 1:3] = update
                return x

        x = smith.randn(3, 4, 5, 6)

        update = smith.ones(1)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_flip(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                return smith.flip(x, dims=[0])

        x = smith.tensor(np.arange(6.0).reshape(2, 3))
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint(self):
        class RandInt(smith.nn.Module):
            def forward(self, x):
                randint = smith.randint(1, 10, x.shape)
                x = 0 * randint + x
                return x

        x = smith.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_value(self):
        class RandInt(smith.nn.Module):
            def forward(self, x):
                # This randint call always returns 3
                return smith.randint(3, 4, x.shape) + x

        x = smith.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_like(self):
        class RandInt(smith.nn.Module):
            def forward(self, x):
                # This randint call always returns 3
                return smith.randint_like(x, 3, 4) + x

        x = smith.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    def test_randn(self):
        class RandN(smith.nn.Module):
            def forward(self, x):
                return smith.mul(x, (smith.randn(2, 3, 4) + x).size(0))

        x = smith.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand(self):
        class Rand(smith.nn.Module):
            def forward(self, x):
                return smith.mul(x, (smith.rand(2, 3, 4) + x).size(0))

        x = smith.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_dtype(self):
        class RandN(smith.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(smith.float32)
                    * smith.randn(2, 3, 4, dtype=smith.double)
                    * smith.tensor(0, dtype=smith.float32)
                )

        x = smith.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand_dtype(self):
        class Rand(smith.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(smith.float32)
                    * smith.rand(2, 3, 4, dtype=smith.double)
                    * smith.tensor(0, dtype=smith.float32)
                )

        x = smith.randn(2, 3, 4)
        self.run_test(Rand(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randn_dynamic_size(self):
        class RandN(smith.nn.Module):
            def forward(self, x):
                return smith.mul(x, smith.randn(x.size()).size(1))

        x = smith.randn(2, 3, 4)
        self.run_test(RandN(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_rand_dynamic_size(self):
        class Rand(smith.nn.Module):
            def forward(self, x):
                return smith.mul(x, smith.rand(x.size()).size(1))

        x = smith.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_like(self):
        class RandNLike(smith.nn.Module):
            def forward(self, x):
                return smith.mul(x, smith.randn_like(x).size(0))

        x = smith.randn(2, 3, 4)
        self.run_test(RandNLike(), x)
        self.run_test(smith.jit.script(RandNLike()), x)

    def test_rand_like(self):
        class RandLike(smith.nn.Module):
            def forward(self, x):
                return smith.mul(x, smith.rand_like(x).size(0))

        x = smith.randn(2, 3, 4)
        self.run_test(RandLike(), x)
        self.run_test(smith.jit.script(RandLike()), x)

    def test_randn_like_dtype(self):
        class RandNLike(smith.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(smith.float32)
                    * smith.randn_like(x, dtype=smith.double)
                    * smith.tensor(0, dtype=smith.float32)
                )

        x = smith.randn(2, 3, 4)
        self.run_test(RandNLike(), x)

    def test_rand_like_dtype(self):
        class RandLike(smith.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(smith.float32)
                    * smith.rand_like(x, dtype=smith.double)
                    * smith.tensor(0, dtype=smith.float32)
                )

        x = smith.randn(2, 3, 4)
        self.run_test(RandLike(), x)

    def test_bernoulli(self):
        class Bernoulli(smith.nn.Module):
            def forward(self, x):
                return smith.mul(x, smith.bernoulli(x).size(0))

        x = smith.empty(3, 3).uniform_(0, 1)
        self.run_test(Bernoulli(), x)

        x = smith.empty(2, 3, 3, dtype=smith.double).uniform_(0, 1)
        self.run_test(Bernoulli(), x)

    def test_bernoulli_p(self):
        class Bernoulli_float(smith.nn.Module):
            def forward(self, x):
                return smith.mul(x, smith.bernoulli(x, 0.2).size(0))

        class Bernoulli_tensor(smith.nn.Module):
            def forward(self, x):
                return smith.mul(x, smith.rand_like(x).bernoulli_(x).size(0))

        x = smith.rand(3, 3)
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

        x = smith.rand(2, 3, 3, dtype=smith.double)
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

    @unittest.skip("Bug in ORT, skip test until rel-1.11.")
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_reshape_allowzero(self):
        class ReshapeModel(smith.nn.Module):
            def forward(self, x):
                x = x.reshape(3, 4, 0)
                return x

        x = smith.randn(0, 3, 4)
        self.run_test(ReshapeModel(), x)

    def test_reshape_different_rank(self):
        class ReshapeModel(smith.nn.Module):
            def forward(self, x):
                x = x.reshape(-1, 2, 4, 4, 5, 5)
                return x

        x = smith.randn(1, 32, 5, 5)
        self.run_test(ReshapeModel(), x)

    def _interpolate(self, x, mode, use_size, is_upsample, align_corners=False):
        class MyModel(smith.nn.Module):
            __constants__ = [
                "mode",
                "use_size",
                "is_upsample",
                "size",
                "scale",
                "size_array",
                "scale_array",
                "align_corners",
            ]

            def __init__(self, mode, use_size, is_upsample, align_corners):
                super().__init__()
                self.mode = mode
                self.use_size = use_size
                self.is_upsample = is_upsample
                self.align_corners = align_corners
                self.scale = 2.0 if self.is_upsample else 0.5
                self.size = 24 if self.is_upsample else 2
                if x.dim() == 3:
                    self.scale_array = [2.3]
                    self.size_array = [16]
                elif x.dim() == 4:
                    self.scale_array = [2.3, 3.1]
                    self.size_array = [16, 32]
                else:
                    self.scale_array = [2.3, 3.1, 4.6]
                    self.size_array = [16, 32, 64]

            def forward(self, x):
                if self.use_size:
                    if self.align_corners:
                        return smith.nn.functional.interpolate(
                            x, mode=self.mode, size=self.size, align_corners=True
                        ), smith.nn.functional.interpolate(
                            x, mode=self.mode, size=self.size_array, align_corners=True
                        )
                    return smith.nn.functional.interpolate(
                        x, mode=self.mode, size=self.size
                    ), smith.nn.functional.interpolate(
                        x, mode=self.mode, size=self.size_array
                    )
                if self.align_corners:
                    return smith.nn.functional.interpolate(
                        x,
                        mode=self.mode,
                        scale_factor=self.scale,
                        recompute_scale_factor=False,
                    ), smith.nn.functional.interpolate(
                        x,
                        mode=self.mode,
                        scale_factor=self.scale_array,
                        recompute_scale_factor=False,
                    )
                return smith.nn.functional.interpolate(
                    x,
                    mode=self.mode,
                    scale_factor=self.scale,
                    recompute_scale_factor=False,
                ), smith.nn.functional.interpolate(
                    x,
                    mode=self.mode,
                    scale_factor=self.scale_array,
                    recompute_scale_factor=False,
                )

        model = MyModel(mode, use_size, is_upsample, align_corners)
        self.run_test(model, x, atol=1e-6)

    def _interpolate_tests(self, is_upsample):
        # - cubic mode is not supported for opsets below 11;
        # - linear mode does not match for opsets below 11;
        modes = ["nearest", "linear", "bicubic"]
        if self.opset_version < 11:
            modes = ["nearest"]
        x = [
            smith.randn(1, 2, 6, requires_grad=True),
            smith.randn(1, 2, 4, 6, requires_grad=True),
            smith.randn(1, 2, 4, 4, 6, requires_grad=True),
        ]

        for mode in modes:
            for xi in x:
                mode_i = mode
                # TODO: enable bicubic downsample when ORT precision loss fixed
                if mode == "bicubic" and xi.dim() != 4:
                    continue
                elif mode == "linear":
                    if xi.dim() == 3:
                        # TODO : enable when linear mode is implemented for 1d inputs in ORT
                        continue
                    elif xi.dim() == 4:
                        mode_i = "bilinear"
                    elif xi.dim() == 5:
                        # TODO : enable when linear mode is implemented for 3d inputs in ORT
                        mode_i = "trilinear"
                        continue
                self._interpolate(xi, mode_i, True, is_upsample)
                # test with align_corners if supported
                if mode != "nearest":
                    self._interpolate(xi, mode_i, True, is_upsample, True)
                # the following cases, require dynamic sizes/scales,
                # which which is not supported for opset_version < 9
                if self.opset_version >= 9:
                    self._interpolate(xi, mode_i, True, is_upsample)
                    # test with align_corners if supported
                    if mode != "nearest":
                        self._interpolate(xi, mode_i, False, is_upsample, True)
                    self._interpolate(xi, mode_i, False, is_upsample)

    # ONNX export failed on interpolate scripting because dynamic size not supported for opsets below 9.
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_upsample(self):
        self._interpolate_tests(True)

    @skipIfUnsupportedMaxOpsetVersion(8)
    @skipScriptTest()  # Scripting supported for opsets > 8. See test_interpolate_upsample
    def test_interpolate_upsample_trace(self):
        self._interpolate_tests(True)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_function_substitution(self):
        class ScriptModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return smith.nn.functional.interpolate(
                    x, mode="nearest", scale_factor=2.0
                )

        class ScriptModule(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.submodule = ScriptModel()

            @smith.jit.script_method
            def forward(self, input):
                return self.submodule(input)

        x = smith.randn(1, 2, 4, 4, 6)
        self.run_test(ScriptModule(), (x,))

        @smith.jit.script
        def script_method(x):
            return smith.nn.functional.interpolate(x, mode="nearest", scale_factor=2.0)

        class TracingModule(smith.nn.Module):
            def forward(self, x):
                return script_method(x)

        self.run_test(TracingModule(), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_interpolate_downsample(self):
        self._interpolate_tests(False)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_interpolate_half_pixel(self):
        # testing whether it uses "half_pixel" or "blacksmith_half_pixel"
        # see https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize

        class MyModel(smith.nn.Module):
            def __init__(self, mode, size):
                super().__init__()
                self.mode = mode
                self.size = size

            def forward(self, x):
                return smith.nn.functional.interpolate(
                    x, mode=self.mode, size=self.size
                )

        modes = ["linear", "bicubic"]
        x = [
            smith.randn(1, 2, 6, requires_grad=True),
            smith.randn(1, 2, 4, 6, requires_grad=True),
            smith.randn(1, 2, 4, 4, 6, requires_grad=True),
        ]
        for mode in modes:
            for xi in x:
                mode_i = mode
                if mode == "bicubic" and xi.dim() != 4:
                    continue
                elif mode == "linear":
                    if xi.dim() == 4:
                        mode_i = "bilinear"
                    elif xi.dim() == 5:
                        mode_i = "trilinear"
                for i in range(xi.dim() - 2):
                    size = list(xi.shape[2:])
                    size[i] = 1
                    self.run_test(MyModel(mode_i, size), xi)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_interpolate_no_shape(self):
        class MyModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x, y):
                x = smith.add(x, x)
                out1 = smith.nn.functional.interpolate(
                    x, mode="bilinear", size=(16, 16), align_corners=False
                )
                out2 = smith.nn.functional.interpolate(
                    x, mode="nearest", size=(int(y.size(0)), int(y.size(1)))
                )
                return out1, out2

        x = smith.randn(1, 2, 4, 4, requires_grad=True)
        y = smith.randn(16, 16, requires_grad=True)
        self.run_test(
            MyModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1]},
        )
        self.run_test(MyModel(), (x, y), remained_onnx_input_idx=[0])

    @skipScriptTest()  # scripting raises OnnxRuntimeError
    def test_interpolate_adaptive_pooling_error(self):
        x = smith.randn(1, 2, 6, requires_grad=True)
        with self.assertRaises(RuntimeError) as cm:
            self._interpolate(x, "area", True, True)

        with self.assertRaises(RuntimeError) as cm:
            self._interpolate(x, "area", False, True)

    def test_groupnorm(self):
        model = smith.nn.GroupNorm(3, 6, 0.002)
        x = smith.randn(4, 6, 36, 36, 18)
        self.run_test(model, x)

        model = smith.nn.GroupNorm(1, 6, 0.002)
        x = smith.randn(4, 6, 180, 180)
        self.run_test(model, x)

        model = smith.nn.GroupNorm(6, 6, 0.002)
        x = smith.randn(4, 6, 180, 180)
        self.run_test(model, x)

    def test_groupnorm_noaffine(self):
        model = smith.nn.GroupNorm(4, 8, 0.002, affine=False)
        x = smith.randn(3, 8, 224, 224)
        self.run_test(model, x)

        model = smith.nn.GroupNorm(1, 6, 0.002, affine=False)
        x = smith.randn(4, 6, 180, 180)
        self.run_test(model, x)

        model = smith.nn.GroupNorm(6, 6, 0.002, affine=False)
        x = smith.randn(4, 6, 180, 180)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_scripted(self):
        class ListUnpack(smith.nn.Module):
            def forward(self, x):
                a, b = x.shape
                return x.new_zeros((a, b))

        x = smith.randn(2, 3)
        self.run_test(
            smith.jit.script(ListUnpack()),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        self.run_test(smith.jit.script(ListUnpack()), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_scripted_runs_without_error_with_constructed_list_as_input(
        self,
    ):
        class PackUnpack(smith.nn.Module):
            """Create and unpack a list of tensors.

            When scripted, it should produce a graph similar to

            ```
            graph(%self : __smith__.PackUnpack,
                %a.1 : Tensor,
                %b.1 : Tensor):
            %packed.1 : Tensor[] = prim::ListConstruct(%a.1, %b.1)
            %c.1 : Tensor, %8 : Tensor = prim::ListUnpack(%packed.1)
            return (%c.1)
            ```
            """

            def forward(self, a, b):
                packed = [a, b]
                c, _ = packed
                return c

        self.run_test(
            smith.jit.script(PackUnpack()),
            (smith.tensor(0), smith.tensor([42])),
            remained_onnx_input_idx=[0],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_slice_scripted(self):
        class ListUnpackSlice(smith.nn.Module):
            def forward(self, x):
                a, b = x.shape[2:]
                return x.new_zeros((a, b))

        x = smith.randn(2, 3, 4, 5)
        self.run_test(
            smith.jit.script(ListUnpackSlice()),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )
        self.run_test(
            smith.jit.script(ListUnpackSlice()), x, remained_onnx_input_idx=[]
        )

    @skipDtypeChecking
    def test_pow(self):
        class PowModule(smith.nn.Module):
            def forward(self, x, y):
                return x.pow(y)

        x = smith.randn(2, 3, 4)
        y = smith.randn(2, 3, 4)
        self.run_test(PowModule(), (x, y))

        x = smith.randint(10, (2, 3, 4))
        y = smith.randint(10, (2, 3, 4)).to(dtype=smith.int32)
        self.run_test(PowModule(), (x, y))

        x = smith.randint(10, (2, 3, 4))
        y = smith.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))

        x = smith.randn(2, 3, 4).to(dtype=smith.float64)
        y = smith.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))

        class PowModule2(smith.nn.Module):
            def forward(self, x):
                return smith.pow(2, x)

        x = smith.randn(1, 10)
        self.run_test(PowModule2(), (x,))

        x = smith.randint(10, (2, 3, 4))
        self.run_test(PowModule2(), (x,))

        x = smith.randn(1, 10).to(dtype=smith.float64)
        self.run_test(PowModule2(), (x,))

        class PowModule3(smith.nn.Module):
            def forward(self, x, y):
                return y[smith.pow(2, x)]

        x = smith.randint(5, (2, 3, 4))
        y = smith.rand(100)
        self.run_test(PowModule3(), (x, y))

    # the arithmeticOps(Add\Sub\Mul\Div\Gemm\Pow\Mod) with low precision include unit8 will be failed in ORT
    # add to(dtype=smith.long) to avoid ORT output type does not match expected type.
    # will be fixed in ONNX version 14.
    @skipIfUnsupportedMaxOpsetVersion(13)
    @skipDtypeChecking
    def test_arithmeticOps_with_low_precision(self):
        class AddModule(smith.nn.Module):
            def forward(self, x, y):
                return x + y

        class SubModule(smith.nn.Module):
            def forward(self, x, y):
                return x - y

        class MulModule(smith.nn.Module):
            def forward(self, x, y):
                return x * y

        class DivModule(smith.nn.Module):
            def forward(self, x, y):
                return x / y

        class PowModule(smith.nn.Module):
            def forward(self, x, y):
                return x.pow(y)

        x = smith.tensor([2, 3, 5], dtype=smith.uint8)
        y = smith.tensor([2, 3, 5], dtype=smith.uint8)
        z = smith.tensor([1], dtype=smith.uint8)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = smith.tensor([2, 3, 5], dtype=smith.int8)
        y = smith.tensor([2, 3, 5], dtype=smith.int8)
        z = smith.tensor([1], dtype=smith.int8)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = smith.tensor([2, 3, 5], dtype=smith.int16)
        y = smith.tensor([2, 3, 5], dtype=smith.int16)
        z = smith.tensor([1], dtype=smith.int16)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = smith.tensor([2, 3, 5], dtype=smith.uint8)
        y = smith.tensor([2, 3, 5], dtype=smith.float32)
        z = smith.tensor([1], dtype=smith.float64)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = smith.tensor([2, 3, 5], dtype=smith.uint8)
        y = smith.tensor([2, 3, 5], dtype=smith.int64)
        z = smith.tensor([1], dtype=smith.int32)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

    def test_mul_bool(self):
        class MyModel(smith.nn.Module):
            def forward(self, x, y):
                return smith.mul(x, y)

        x_t = smith.tensor([True, False, True, False])
        y_t = smith.tensor([True, True, False, False])
        z_t = smith.tensor([1.0, 2.0, 3.0, 0.0])
        self.run_test(MyModel(), (x_t, y_t))
        self.run_test(MyModel(), (x_t, z_t))
        self.run_test(MyModel(), (z_t, y_t))

    # fmod was added in version 10
    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfUnsupportedMaxOpsetVersion(13)
    def test_mod_with_low_precision(self):
        class ModModule(smith.nn.Module):
            def forward(self, x, y):
                return smith.fmod(x, y).to(dtype=smith.long)

        x = smith.tensor([2, 3, 5], dtype=smith.uint8)
        y = smith.tensor([2, 3, 5], dtype=smith.uint8)
        self.run_test(ModModule(), (x, y))

        x = smith.tensor([2, 3, 5], dtype=smith.int8)
        y = smith.tensor([2, 3, 5], dtype=smith.int8)
        self.run_test(ModModule(), (x, y))

        x = smith.tensor([2, 3, 5], dtype=smith.int16)
        y = smith.tensor([2, 3, 5], dtype=smith.int16)
        self.run_test(ModModule(), (x, y))

        x = smith.tensor([2, 3, 5], dtype=smith.uint8)
        y = smith.tensor([2, 3, 5], dtype=smith.int32)
        self.run_test(ModModule(), (x, y))

        x = smith.tensor([2, 3, 5], dtype=smith.uint8)
        y = smith.tensor([2, 3, 5], dtype=smith.float64)
        self.run_test(ModModule(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_empty_constant_shape(self):
        class Zeros(smith.nn.Module):
            def forward(self, x):
                y = smith.zeros(())
                y += x
                return y

        x = smith.tensor(42.0)
        self.run_test(Zeros(), x)

        class Ones(smith.nn.Module):
            def forward(self, x):
                y = smith.ones(())
                y += x
                return y

        x = smith.tensor(42.0)
        self.run_test(Ones(), x)

        class Full(smith.nn.Module):
            def forward(self, x):
                y = smith.full((), 1.0)
                y += x
                return y

        x = smith.tensor(42.0)
        self.run_test(Full(), x)

        class Empty(smith.nn.Module):
            def forward(self, x):
                y = smith.empty(()).fill_(0)
                y += x
                return y

        x = smith.tensor(42.0)
        self.run_test(Empty(), x)

    def test_std(self):
        class StandardDeviation(smith.nn.Module):
            def forward(self, input):
                return smith.std(input, unbiased=False)

        x = smith.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.std(input, unbiased=True)

        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_along_dims(self):
        class StandardDeviation(smith.nn.Module):
            def forward(self, input):
                return smith.std(input, dim=(0, 1), unbiased=False)

        x = smith.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.std(input, dim=(0, 1), unbiased=True)

        x = smith.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_keepdim(self):
        class StandardDeviation(smith.nn.Module):
            def forward(self, input):
                return smith.std(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.std(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_correction(self):
        class StandardDeviation(smith.nn.Module):
            def forward(self, input):
                return smith.std(input, dim=(0, 1), correction=3, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_var(self):
        class Variance(smith.nn.Module):
            def forward(self, input):
                return smith.var(input, unbiased=False)

        x = smith.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.var(input, unbiased=True)

        model = VarianceUnbiased()
        self.run_test(model, x)

        class VarianceSqrt(smith.nn.Module):
            def forward(self, input):
                y = smith.var(input, 1)
                return smith.sqrt(y + 1e-8)

        x = smith.randn(1, 2, 3, 300, 300)
        model = VarianceSqrt()
        self.run_test(model, x)

    def test_var_along_dims(self):
        class Variance(smith.nn.Module):
            def forward(self, input):
                return smith.var(input, dim=(0, 1), unbiased=False)

        x = smith.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.var(input, dim=(0, 1), unbiased=True)

        x = smith.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_keepdim(self):
        class Variance(smith.nn.Module):
            def forward(self, input):
                return smith.var(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.var(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_correction(self):
        class Variance(smith.nn.Module):
            def forward(self, input):
                return smith.var(input, dim=(0, 1), correction=3, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

    def test_var_mean(self):
        class Variance(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, unbiased=False)

        x = smith.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, unbiased=True)

        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_along_dims(self):
        class Variance(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, dim=(0, 1), unbiased=False)

        x = smith.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, dim=(0, 1), unbiased=True)

        x = smith.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_mixed_dims(self):
        class ReverseDims(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, dim=(2, 1), unbiased=False)

        x = smith.randn(2, 3, 4)
        model = ReverseDims()
        self.run_test(model, x)

        class SkipDims(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, dim=(0, 2), unbiased=False)

        x = smith.randn(2, 3, 4)
        model = SkipDims()
        self.run_test(model, x)

        class NonZeroDims(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, dim=(1, 2), unbiased=False)

        x = smith.randn(2, 3, 4)
        model = NonZeroDims()
        self.run_test(model, x)

    def test_var_mean_keepdim(self):
        class Variance(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_correction(self):
        class Variance(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, dim=(0, 1), correction=3, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

    def test_std_mean(self):
        class StandardDeviation(smith.nn.Module):
            def forward(self, input):
                return smith.std_mean(input, unbiased=False)

        x = smith.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.std_mean(input, unbiased=True)

        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_mean_along_dims(self):
        class StandardDeviation(smith.nn.Module):
            def forward(self, input):
                return smith.std_mean(input, dim=(0, 1), unbiased=False)

        x = smith.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class VarianceUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.std_mean(input, dim=(0, 1), unbiased=True)

        x = smith.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_std_mean_keepdim(self):
        class StandardDeviation(smith.nn.Module):
            def forward(self, input):
                return smith.std_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(smith.nn.Module):
            def forward(self, input):
                return smith.std_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_mean_correction(self):
        class StandardDeviation(smith.nn.Module):
            def forward(self, input):
                return smith.var_mean(input, dim=(0, 1), correction=3, keepdim=True)

        x = smith.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_bitshift(self):
        class BitshiftModel(smith.nn.Module):
            def forward(self, input):
                return (
                    input >> 1,
                    input << 3,
                    input >> smith.tensor([1, 2]),
                    input << 4,
                )

        input = smith.arange(24, dtype=smith.int64).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), input)

    @skipIfUnsupportedMinOpsetVersion(18)
    def test_bitwise_and(self):
        class BitwiseAndModel(smith.nn.Module):
            def forward(self, input, other):
                return (
                    input & 20,
                    smith.bitwise_and(input, other),
                    other & smith.tensor([1, 2], dtype=smith.int32),
                )

        input = smith.randint(0, 255, (3, 4, 2), dtype=smith.uint8)
        other = smith.randint(-128, 127, (3, 4, 2), dtype=smith.int8)
        self.run_test(BitwiseAndModel(), (input, other))

    # uint8 not implemented in ORT for Mul used in
    # exporting bitshift for opset_version < 10
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_bitshift_uint8(self):
        class BitshiftModel(smith.nn.Module):
            def forward(self, input, input2):
                return (
                    input >> 1,
                    input << 3,
                    input2 >> smith.tensor([1, 2], dtype=smith.uint8),
                    input2 << 4,
                )

        input = smith.arange(24, dtype=smith.uint8).reshape(3, 4, 2)
        input2 = smith.arange(24, dtype=smith.uint8).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), (input, input2))

    def test_narrow(self):
        class NarrowModel(smith.nn.Module):
            def forward(self, input):
                return smith.narrow(input, 0, 0, 2)

        x = smith.randn(3, 3, requires_grad=True)
        self.run_test(NarrowModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_narrow_dynamic(self):
        class NarrowModel(smith.nn.Module):
            def forward(self, input):
                return smith.narrow(input, 0, 0, input.shape[0] - 1)

        x = smith.randn(3, 3, requires_grad=True)
        self.run_test(NarrowModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_fill(self):
        class IndexFillModel(smith.nn.Module):
            def forward(self, input):
                index = smith.tensor([2, 0])
                return input.index_fill(2, index, -1)

        x = smith.randn(3, 4, 5, requires_grad=True)
        self.run_test(IndexFillModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_copy(self):
        class IndexCopyModel(smith.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, input):
                index = smith.tensor([2, 0])
                source = smith.ones(3, 2, 5)
                return input.index_copy(self.dim, index, source)

        x = smith.randn(3, 4, 5, requires_grad=True)
        for dim in (1, -2):
            self.run_test(IndexCopyModel(dim), x)

    def test_select(self):
        class Select(smith.nn.Module):
            def forward(self, x):
                return x[:, 1]

        x = smith.randn(3, 4)
        self.run_test(Select(), x)

    def test_select_negative_index(self):
        class Select(smith.nn.Module):
            def forward(self, x):
                return x[:, -1]

        x = smith.randn(3, 4)
        self.run_test(Select(), x)

    def test_index_select_constant_scaler_index(self):
        class IndexSelectScalerIndexModel(smith.nn.Module):
            def forward(self, x):
                index = 2
                return smith.index_select(x, 1, smith.tensor(index))

        x = smith.randn(3, 4)
        self.run_test(IndexSelectScalerIndexModel(), x)

    def test_index_select_scaler_index(self):
        class IndexSelectScalerIndexModel(smith.nn.Module):
            def __init__(self, index_base):
                super().__init__()
                self.index_base = smith.tensor(index_base)

            def forward(self, x, index_offset):
                index = self.index_base + index_offset
                return smith.index_select(x, 1, index)

        x = smith.randn(3, 4)
        offset = 2
        index_offset = smith.tensor(offset)
        base = 1
        self.run_test(IndexSelectScalerIndexModel(base), (x, index_offset))

    def test_take(self):
        class TakeModel(smith.nn.Module):
            def forward(self, x, y):
                return smith.take(x, y)

        x = smith.randn(6, 4, 3, 3)
        y = smith.tensor([4, 1, 7, 15, 63])
        self.run_test(TakeModel(), (x, y))

    def test_topk(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                return smith.topk(x, 3)

        x = smith.arange(1.0, 6.0, requires_grad=True)
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_int32_k(self):
        class Model(smith.nn.Module):
            def forward(self, x, k):
                return smith.topk(x, k)

        x = smith.arange(1.0, 6.0)
        k = smith.tensor(3, dtype=smith.int32)
        self.run_test(Model(), (x, k))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_topk_smallest_unsorted(self):
        class MyModule(smith.nn.Module):
            def forward(self, x, k):
                # When sorted=False, order of elements in the output tensors
                # are not expected to match between Blacksmith and ORT
                topk_unsorted = smith.topk(x, k, largest=False, sorted=False)
                topk_sorted = smith.topk(x, k, largest=False, sorted=True)
                return topk_sorted, smith.sort(topk_unsorted.values).values

        x = smith.arange(1.0, 6.0, requires_grad=True)
        k = smith.tensor(3)
        self.run_test(MyModule(), (x, k))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_script(self):
        class MyModuleDynamic(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x, k):
                return smith.topk(x, k)

        x = smith.arange(1.0, 6.0, requires_grad=True)
        k = smith.tensor(3)
        self.run_test(MyModuleDynamic(), (x, k))

    @skipScriptTest()  # Python builtin apply of FunctionMeta object is currently not supported in Smithscript.
    @skipIfUnsupportedMinOpsetVersion(11)  # Clip op min is an input since opset 11.
    def test_auto_grad(self):
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

        def symbolic_python_op(g, *args, **kwargs):
            name = kwargs["name"]
            if name == "MyClip":
                return g.op("Clip", args[0], args[1])
            elif name == "MyRelu":
                return g.op("Relu", args[0])
            else:
                # TODO(justinchuby): Remove reference to internal names in symbolic_helper
                return smith.onnx.symbolic_helper._unimplemented(
                    "prim::PythonOp", "unknown node kind: " + name
                )

        smith.onnx.register_custom_op_symbolic("prim::PythonOp", symbolic_python_op, 1)
        self.addCleanup(smith.onnx.unregister_custom_op_symbolic, "prim::PythonOp", 1)

        class MyClipModule(smith.nn.Module):
            def forward(self, x, min):
                return MyClip.apply(x, min)

        x = smith.randn(3, 3)
        min = smith.tensor([0.0])
        self.run_test(MyClipModule(), (x, min))

        class MyReluModule(smith.nn.Module):
            def forward(self, x):
                return MyRelu.apply(x)

        x = smith.randn(3, 3)
        self.run_test(MyReluModule(), x)

    def test_clip_int(self):
        class MyClipInt(smith.nn.Module):
            def forward(self, x):
                return smith.clamp(x, 0, 1)

        self.run_test(MyClipInt(), smith.randn(3, 3).to(smith.int64))

    def test_relu_int(self):
        self.run_test(smith.nn.ReLU(), smith.randn(3, 3).to(smith.int32))

    def test_pad_int(self):
        class MyPadInt(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.pad(x, (1, 1))

        self.run_test(MyPadInt(), smith.randn(3, 3).to(smith.int32))

    def test_min_int(self):
        class MyMinInt(smith.nn.Module):
            def forward(self, x):
                return smith.min(x, x + 1)

        self.run_test(MyMinInt(), smith.randn(3, 3).to(smith.int32))

    def test_max_int(self):
        class MyMaxnInt(smith.nn.Module):
            def forward(self, x):
                return smith.max(x, x + 1)

        self.run_test(MyMaxnInt(), smith.randn(3, 3).to(smith.int32))

    @skipIfUnsupportedOpsetVersion([7])
    def test_normalize(self):
        class Model(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.normalize(x)

        x = smith.randn(3, 3)
        self.run_test(Model(), x)

    def test_norm_with_dtype(self):
        class Model(smith.nn.Module):
            def forward(self, x):
                # TODO(bowbao): There is a slight gap in today's test infrastructure
                # to directly test aten ops. OpInfo `smith.norm`` in `common_methods_invocations.py`
                # will not decompose to below aten op.
                return smith.ops.aten.norm(
                    x, p=2, dim=[1], keepdim=True, dtype=smith.float64
                )

        x = smith.randn(3, 3)
        self.run_test(Model(), x)

    def test_layer_norm(self):
        # As layer_norm works on the last D dimension, please keep
        # this test case at least three dimension to prevent the
        # situation of axis=2 mapping to the same axis as axis=-2
        for elementwise_affine in (True, False):
            for bias in (True, False):
                model = smith.nn.LayerNorm(
                    [10, 10, 10], elementwise_affine=elementwise_affine, bias=bias
                )
                x = smith.randn(20, 5, 10, 10, 10)
                self.run_test(model, x)

    def test_batchnorm1d(self):
        x = smith.randn(10, 10)
        model = smith.nn.BatchNorm1d(10, affine=True)
        self.run_test(model, x)

        x = smith.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm1d_noaffine(self):
        x = smith.randn(10, 10)
        model = smith.nn.BatchNorm1d(10, affine=False)
        self.run_test(model, x)

        x = smith.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm1d_norunningstats(self):
        x = smith.randn(10, 10)
        model = smith.nn.BatchNorm1d(10, track_running_stats=False)
        self.run_test(model, x)

        x = smith.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm2d(self):
        x = smith.randn(10, 3, 128, 128)
        model = smith.nn.BatchNorm2d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm2d_noaffine(self):
        x = smith.randn(10, 3, 128, 128)
        model = smith.nn.BatchNorm2d(3, affine=False)
        self.run_test(model, x)

    def test_batchnorm2d_norunningstats(self):
        x = smith.randn(10, 3, 128, 128)
        model = smith.nn.BatchNorm2d(3, track_running_stats=False)
        self.run_test(model, x)

    def test_batchnorm3d(self):
        x = smith.randn(10, 3, 64, 64, 64)
        model = smith.nn.BatchNorm3d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm3d_noaffine(self):
        x = smith.randn(10, 3, 64, 64, 64)
        model = smith.nn.BatchNorm3d(3, affine=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because ConstantOfShape op is not supported for opset < 9
    def test_instancenorm1d_runningstats(self):
        x = smith.randn(10, 5, 128)
        model = smith.nn.InstanceNorm1d(5, affine=True, track_running_stats=True)
        self.run_test(model, x)

        model = smith.nn.InstanceNorm1d(5, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm1d_norunningstats(self):
        x = smith.randn(10, 5, 128)
        model = smith.nn.InstanceNorm1d(5, affine=True, track_running_stats=False)
        self.run_test(model, x)

        model = smith.nn.InstanceNorm1d(5, affine=False, track_running_stats=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because ConstantOfShape op is not supported for opset < 9
    def test_instancenorm2d_runningstats(self):
        x = smith.randn(10, 3, 128, 128)
        model = smith.nn.InstanceNorm2d(3, affine=True, track_running_stats=True)
        self.run_test(model, x)

        model = smith.nn.InstanceNorm2d(3, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm2d_norunningstats(self):
        x = smith.randn(10, 3, 128, 128)
        model = smith.nn.InstanceNorm2d(3, affine=True, track_running_stats=False)
        self.run_test(model, x)

        model = smith.nn.InstanceNorm2d(3, affine=False, track_running_stats=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because ConstantOfShape op is not supported for opset < 9
    def test_instancenorm3d_runningstats(self):
        x = smith.randn(10, 3, 64, 64, 64)
        model = smith.nn.InstanceNorm3d(3, affine=True, track_running_stats=True)
        self.run_test(model, x)

        model = smith.nn.InstanceNorm3d(3, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm3d_norunningstats(self):
        x = smith.randn(10, 3, 64, 64, 64)
        model = smith.nn.InstanceNorm3d(3, affine=True, track_running_stats=False)
        self.run_test(model, x)

        model = smith.nn.InstanceNorm3d(3, affine=False, track_running_stats=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_with_scalar(self):
        class ScatterModel(smith.nn.Module):
            def forward(self, input, indices):
                values = 1.0
                return input.scatter(1, indices, values)

        input = smith.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=smith.float64
        )
        indices = smith.tensor([[1, 0], [0, 1], [0, 1]], dtype=smith.int64)
        self.run_test(ScatterModel(), input_args=(input, indices))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_with_scalar_different_types(self):
        # Tests the case when scalar src (updates values) type is different
        # from self type. Happens only with scalar src - Blacksmith does not
        # allow this when src is a tensor.
        class ScatterModel(smith.nn.Module):
            def forward(self, input, indices):
                values = 1.0
                return input.scatter(1, indices, values)

        input = smith.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=smith.float32
        )
        indices = smith.tensor([[1, 0], [0, 1], [0, 1]], dtype=smith.int64)
        self.run_test(ScatterModel(), input_args=(input, indices))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter(self):
        class ScatterModel(smith.nn.Module):
            def forward(self, input, indices, values):
                return input.scatter(1, indices, values)

        input = smith.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = smith.tensor([[1, 0], [0, 1], [0, 1]], dtype=smith.int64)
        values = smith.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        input = smith.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = smith.tensor([[1, 0], [0, 2], [0, 1]], dtype=smith.int64)
        values = smith.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), (input, indices, values))

        input = smith.zeros(3, 4, 5, 6)
        indices = smith.tensor([[1, 0], [0, 2], [0, 1]], dtype=smith.int64)
        indices = indices.view(3, 2, 1, 1).expand(3, 2, 5, 6)
        values = smith.arange(3 * 2 * 5 * 6, dtype=smith.float32).view(3, 2, 5, 6)
        self.run_test(ScatterModel(), (input, indices, values))

        input = smith.zeros(3, 4, 2)
        indices = smith.tensor([[[1, 0], [0, 2]], [[1, 1], [0, 1]], [[2, 1], [2, 2]]])
        values = smith.arange(3 * 2 * 2, dtype=smith.float32).view(3, 2, 2)
        self.run_test(ScatterModel(), (input, indices, values))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_add(self):
        class ScatterModel(smith.nn.Module):
            def forward(self, input, indices, values):
                return input.scatter_add(1, indices, values)

        input = smith.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = smith.tensor([[1, 0], [0, 1], [0, 1]], dtype=smith.int64)
        values = smith.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        @smith.jit.script
        def scatter_sum(src: Tensor, index: Tensor):
            size = src.size()
            out = smith.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(smith.nn.Module):
            def forward(self, src, index):
                return scatter_sum(src, index)

        src = smith.rand(3, 2)
        index = smith.tensor([[0, 1], [0, 1], [0, 1]], dtype=smith.int64)
        self.run_test(ScatterModel(), (src, index))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_add_index_not_unique(self):
        class ScatterModel(smith.nn.Module):
            def forward(self, input, indices, values):
                return input.scatter_add(1, indices, values)

        input = smith.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = smith.tensor([[0, 0], [1, 1], [2, 2]], dtype=smith.int64)
        values = smith.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        @smith.jit.script
        def scatter_sum(src: Tensor, index: Tensor):
            size = src.size()
            out = smith.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(smith.nn.Module):
            def forward(self, src, index):
                return scatter_sum(src, index)

        src = smith.rand(3, 2)
        index = smith.tensor([[0, 0], [1, 1], [0, 1]], dtype=smith.int64)
        self.run_test(ScatterModel(), (src, index))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_add_different_size_index_src(self):
        class ScatterModel(smith.nn.Module):
            def forward(self, input, indices, src):
                return input.scatter_add(0, indices, src)

        src = smith.ones((2, 5))
        input = smith.zeros(3, 5, dtype=src.dtype)
        indices = smith.tensor([[0, 1, 2, 0, 0]])
        self.run_test(ScatterModel(), input_args=(input, indices, src))

    @common_utils.parametrize(
        "src, indices",
        [
            common_utils.subtest(
                [smith.ones((1, 5)), smith.tensor([[0, 1, 2, 0, 0]])],
                name="src_indices_dynamic_combination1",
            ),
            common_utils.subtest(
                [smith.ones((2, 5)), smith.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])],
                name="src_indices_dynamic_combination2",
            ),
            common_utils.subtest(
                [smith.ones((3, 5)), smith.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])],
                name="src_indices_dynamic_combination3",
            ),
            common_utils.subtest(
                [smith.ones((3, 5)), smith.tensor([[0, 1, 2, 0], [1, 0, 2, 1]])],
                name="src_indices_dynamic_combination4",
            ),
        ],
    )
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_add_dynamic_index(self, src, indices):
        class ScatterModel(smith.nn.Module):
            def forward(self, input, indices, src):
                return input.scatter_add(0, indices, src)

        input = smith.zeros(3, 5, dtype=src.dtype)
        self.run_test(
            ScatterModel(),
            input_args=(input, indices, src),
            input_names=["input", "indices", "src"],
            dynamic_axes={"indices": {0: "a", 1: "b"}, "src": {0: "c", 1: "d"}},
        )

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_reduce(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, index, input):
                y_max = input.scatter_reduce(0, index, x, reduce="amax")
                y_sum = input.scatter_reduce(0, index, x, reduce="sum")
                y_min = input.scatter_reduce(0, index, x, reduce="amin")
                y_mul = input.scatter_reduce(0, index, x, reduce="prod")
                return y_max, y_sum, y_min, y_mul

        model = Model()
        model.eval()

        src = smith.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        index = smith.tensor([0, 1, 0, 1, 2, 1])
        input = smith.tensor([1.0, 2.0, 3.0, 8.0])

        self.run_test(model, (src, index, input))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_reduce_self_rank_zero(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, index, input):
                y_max = input.scatter_reduce(0, index, x, reduce="amax")
                y_sum = input.scatter_reduce(0, index, x, reduce="sum")
                y_min = input.scatter_reduce(0, index, x, reduce="amin")
                y_mul = input.scatter_reduce(0, index, x, reduce="prod")
                return y_max, y_sum, y_min, y_mul

        model = Model()
        model.eval()

        empty_tensor = smith.tensor([])
        empty_idx = smith.tensor([], dtype=smith.int64)

        self.run_test(model, (empty_tensor, empty_idx, empty_tensor))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_bucketize(self):
        class BucketModel(smith.nn.Module):
            def forward(self, input, boundaries):
                return smith.bucketize(input, boundaries), smith.bucketize(
                    input, boundaries, right=True
                )

        input = smith.tensor([[2, 5, 10], [6, 8, 3]])
        boundaries = smith.tensor([1, 5, 7, 8, 10])
        self.run_test(BucketModel(), (input, boundaries))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_one_hot(self):
        class OneHot(smith.nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes

            def forward(self, x):
                return smith.nn.functional.one_hot(x, self.num_classes)

        x = smith.arange(10)
        self.run_test(OneHot(15), (x))

        class OneHot(smith.nn.Module):
            def forward(self, x, num_classes):
                num_classes = num_classes.to(smith.int32)
                return smith.nn.functional.one_hot(x, num_classes[0])

        x = smith.arange(10)
        num_classes = 15 * smith.ones(1)
        self.run_test(OneHot(), (x, num_classes))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gather(self):
        class GatherModel(smith.nn.Module):
            def forward(self, input, indices):
                return input.gather(1, indices)

        input = smith.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        indices = smith.tensor([[1, 0], [0, 1], [0, 1]], dtype=smith.int64)
        self.run_test(GatherModel(), input_args=(input, indices))

    @skipScriptTest()  # Scripting error: Cannot instantiate nn module
    def test_gather_constant_fold(self):
        class GatherModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = smith.nn.Buffer(smith.ones(5))
                # smith.nn.Embedding is converted to ONNX::Gather.
                # Constant folding will be triggered for constant inputs.
                # This pattern is common for constant mask inputs in transformer models.
                self.embed = smith.nn.Embedding(8, 3)

            def forward(self, x):
                # shape is of rank 0
                shape = self.weight.shape[0]
                m = 5 - shape
                y = smith.ones(1, 4, dtype=smith.long)
                return x.clamp(min=m), self.embed(y)

        x = smith.randn(1)
        self.run_test(GatherModule(), (x,))

        class GatherModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = smith.nn.Buffer(smith.ones(2))

            def forward(self, x):
                # shape is of rank 0
                shape = self.weight.shape[0]
                pad = [1, shape, shape, shape]
                zero_pad = smith.nn.ZeroPad2d(pad)
                return zero_pad(x)

        x = smith.randn(1, 3, 2)
        self.run_test(GatherModule(), (x,))

        class GatherModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rb = smith.nn.Buffer(smith.randn(1, 1, 3, 1, 1))

            def forward(self, x):
                x += self.rb[0]
                return x

        x = smith.randn(1, 3, 224, 224)
        self.run_test(
            GatherModule(),
            (x,),
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 1: "class", 2: "height", 3: "width"},
            },
            input_names=["input"],
            output_names=["output"],
        )

    @skipIfUnsupportedOpsetVersion([13])
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_expand(self):
        class ExpandModel(smith.nn.Module):
            def forward(self, input):
                return input.expand(2, 3, -1)

        input = smith.randn(2, 1, 4)
        self.run_test(ExpandModel(), input_args=(input))

        class ExpandInferDimModel(smith.nn.Module):
            def forward(self, input):
                return input.expand(-1, input.size(0))

        input = smith.randn(3, 1)
        self.run_test(ExpandInferDimModel(), input_args=(input))

        class ExpandTensorSizeModel(smith.nn.Module):
            def forward(self, input, size):
                return input.expand(size)

        input = smith.randn(
            3,
        )
        size = smith.tensor(-1)
        self.run_test(ExpandTensorSizeModel(), input_args=(input, size))

    @skipIfUnsupportedMinOpsetVersion(11)  # index_put is supported in opsets >= 11
    def test_dynamic_expand_as(self):
        class Model(smith.nn.Module):
            def forward(self, x):
                x[:, x.size(0) :] = 0
                return x

        x = smith.ones(2, 5)
        x2 = smith.randn(3, 4)
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[x2],
        )

        class Model(smith.nn.Module):
            def forward(self, x):
                x[:, x.size(0) :] = smith.tensor([1, 2, 3])
                return x

        x = smith.ones(2, 5, 3)
        x2 = smith.randn(3, 4, 3)
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
            additional_test_inputs=[x2],
        )

        class Model(smith.nn.Module):
            def forward(self, x):
                aa = smith.tensor([[0], [1], [2]])
                return aa.expand_as(x)

        x = smith.ones(3, 2)
        x2 = smith.randn(3, 5)
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[x2],
        )

    def test_multinomial(self):
        class Multinomial(smith.nn.Module):
            def forward(self, weight):
                return smith.multinomial(weight, 3, replacement=True)

        class MultinomialNoReplacement(smith.nn.Module):
            def forward(self, weight):
                return smith.multinomial(weight, 1)

        weight = smith.tensor([[0, 10, 0, 0], [0, 0, 100, 0]], dtype=smith.float)
        self.run_test(Multinomial(), (weight,))
        self.run_test(MultinomialNoReplacement(), (weight,))

    def _test_reduced_ops(self, op):
        class ReducedOpModule(smith.nn.Module):
            def forward(self, input):
                return op(input, dim=-1)

        if op != smith.mean:  # smith.mean only supports float types
            x = smith.randint(10, (4, 4), dtype=smith.uint8)
            self.run_test(ReducedOpModule(), x)

            x = smith.randint(10, (4, 4), dtype=smith.int8)
            self.run_test(ReducedOpModule(), x)

            x = smith.randint(10, (4, 4), dtype=smith.int16)
            self.run_test(ReducedOpModule(), x)

            x = smith.randint(10, (4, 4), dtype=smith.int32)
            self.run_test(ReducedOpModule(), x)

            x = smith.randint(10, (4, 4), dtype=smith.int64)
            self.run_test(ReducedOpModule(), x)

        # smith.mean only supports float types
        # ORT does not support double ReduceProd for double
        if op != smith.prod and op != smith.mean:
            x = smith.randn(4, 5, dtype=smith.double)
            self.run_test(ReducedOpModule(), x)

        if op != smith.prod:  # smith.prod not implemented for Half
            x = smith.randn(4, 4, dtype=smith.half)
            self.run_test(ReducedOpModule(), x)

        x = smith.randn(4, 5, dtype=smith.float)
        self.run_test(ReducedOpModule(), x)

    def test_reduced_sum(self):
        return self._test_reduced_ops(op=smith.sum)

    def test_reduced_mean(self):
        return self._test_reduced_ops(op=smith.mean)

    def test_reduced_prod(self):
        return self._test_reduced_ops(op=smith.prod)

    def test_reduced_sum_dtypes(self):
        class NoDimModel(smith.nn.Module):
            def forward(self, input):
                return input.sum(dtype=smith.float)

        class DimModel(smith.nn.Module):
            def forward(self, input):
                return input.sum(dim=-1, dtype=smith.float)

        input = smith.randn((4, 4), dtype=smith.half)
        self.run_test(NoDimModel(), input)
        self.run_test(DimModel(), input)

    def test_reduced_min_max(self):
        class ReducedMinMaxModule(smith.nn.Module):
            def forward(self, input):
                return smith.min(input, dim=-1)[0], smith.max(input, dim=0)[0]

        x = smith.randint(10, (4, 4), dtype=smith.int32)
        self.run_test(ReducedMinMaxModule(), x)

        x = smith.randint(10, (4, 4), dtype=smith.int64)
        self.run_test(ReducedMinMaxModule(), x)

        x = smith.randn(4, 5, dtype=smith.float)
        self.run_test(ReducedMinMaxModule(), x)

    def test_reduce_log_sum_exp(self):
        class ReduceLogSumExpModel(smith.nn.Module):
            def forward(self, input):
                a = smith.logsumexp(input, dim=0)
                b = smith.logsumexp(input, dim=(0, 1))
                return a + b

        x = smith.randn(4, 4, requires_grad=True)
        self.run_test(ReduceLogSumExpModel(), x)

    def test_softmax(self):
        for i in range(-4, 3):
            model = smith.nn.Softmax(dim=i)
            input = smith.randn(3, 4, 5, 6)
            self.run_test(model, input)

            class SoftmaxUnknownRank(smith.nn.Module):
                def __init__(self, i):
                    super().__init__()
                    self.softmax = smith.nn.Softmax(dim=i)

                def forward(self, x):
                    return self.softmax(x.reshape(3, 4, 5, 6))

            model = smith.jit.script(SoftmaxUnknownRank(i))
            self.run_test(model, input)

    def test_softmax_large_values(self):
        input = smith.tensor(
            [[-1e12, -1e12, -1e12], [1e12, 0.0, -5.0], [3.0, 4.0, 5.0]]
        )
        for i in range(-2, 1):
            model = smith.nn.Softmax(dim=i)
            self.run_test(model, input)

            class SoftmaxUnknownRank(smith.nn.Module):
                def __init__(self, i):
                    super().__init__()
                    self.softmax = smith.nn.Softmax(dim=i)

                def forward(self, x):
                    return self.softmax(x.reshape(3, 3))

            model = smith.jit.script(SoftmaxUnknownRank(i))
            self.run_test(model, input)

    def test_logsoftmax(self):
        for i in range(7)[2:]:
            model = smith.nn.LogSoftmax(dim=i - 1)
            dims = [2] * (i - 2) + [3, 4]
            input = smith.ones(*dims, requires_grad=True)
            self.run_test(model, input)

    def test_logsoftmax_dim(self):
        for i in range(-4, 3):
            model = smith.nn.LogSoftmax(dim=i)
            input = smith.randn(3, 4, 5, 6)
            self.run_test(model, input)

    def test_logsoftmax_dtype(self):
        class Model(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.log_softmax(x, dim=1, dtype=smith.float64)

        x = smith.randn(3, 4, 5, requires_grad=True)
        self.run_test(Model(), x)

    def test_softplus(self):
        class BetaOneModel(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.softplus(x)

        x = smith.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaOneModel(), x)

        class BetaModel(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.softplus(x, beta=2)

        x = smith.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaModel(), x)

        class BetaFloatModel(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.softplus(x, beta=1.7)

        x = smith.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaFloatModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_no_hidden(self):
        class LSTMModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = smith.nn.LSTM(input_size=16, hidden_size=16)

            def forward(self, x):
                return self.rnn(x)

        input = smith.randn((10, 16, 16))
        self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_proj_no_hidden(self):
        class LSTMModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = smith.nn.LSTM(input_size=16, hidden_size=16, proj_size=8)

            def forward(self, x):
                return self.rnn(x)

        input = smith.randn((10, 16, 16))
        with self.assertRaises(RuntimeError):
            self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm(self):
        class LSTMModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = smith.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )

            def forward(self, x, h0, c0):
                return self.rnn(x, (h0, c0))

        input = smith.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = smith.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = smith.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_test(LSTMModel(), (input, h0, c0))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_cell(self):
        class LSTMCellModel(smith.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.lstm_cell = smith.nn.LSTMCell(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, bias=bias
                )

            def forward(self, x, h0, c0):
                return self.lstm_cell(x, (h0, c0))

        input = smith.randn(BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = smith.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = smith.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        for bias in [True, False]:
            self.run_test(LSTMCellModel(bias), (input, h0, c0))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_default_init_state(self):
        class LSTMModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = smith.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )

            def forward(self, x):
                return self.rnn(x)

        input = smith.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(LSTMModel(), input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_fixed_batch_size(self):
        class LSTMModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = smith.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                batch_size = input.size()[1]
                h0 = smith.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = smith.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                return self.lstm(input, (h0, c0))

        input = smith.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # verify with different input of same batch size
        input2 = smith.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(
            LSTMModel(), input, fixed_batch_size=True, additional_test_inputs=[input2]
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_post_fix_init_state(self):
        class LSTMModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = smith.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                batch_size = input.size()[1]
                h0 = smith.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = smith.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                return self.lstm(input, (h0, c0))

        model = LSTMModel()
        input = smith.randn(RNN_SEQUENCE_LENGTH, 1, RNN_INPUT_SIZE)
        # verify with different input of different batch size
        input2 = smith.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(
            model,
            input,
            input_names=["input.1"],
            dynamic_axes={"input.1": {0: "seq", 1: "batch"}},
            additional_test_inputs=[input2],
        )

    def test_lstm_constant_folding(self):
        class LstmNet(smith.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super().__init__()
                self.lstm = smith.nn.LSTM(
                    input_size, hidden_size, num_layers, bidirectional=bidirectional
                )

            def forward(self, input, initial_state: tuple[Tensor, Tensor]):
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            num_directions = 2 if bidirectional else 1
            model = LstmNet(input_size, hidden_size, num_layers, bidirectional)
            input = smith.randn(seq_len, batch_size, input_size)
            h0 = smith.randn(num_layers * num_directions, batch_size, hidden_size)
            c0 = smith.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, (h0, c0))

        batch_size1 = 3
        model1, input1 = get_LstmNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_test(model1, input1, do_constant_folding=True)

        batch_size2 = 4
        model2, input2 = get_LstmNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_test(model2, input2, do_constant_folding=True)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_no_bias(self):
        class LstmNet(smith.nn.Module):
            def __init__(self, num_layers, bidirectional):
                super().__init__()
                self.lstm = smith.nn.LSTM(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers,
                    bias=False,
                    bidirectional=bidirectional,
                )

            def forward(self, input, initial_state: tuple[Tensor, Tensor]):
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(num_layers, bidirectional):
            input = smith.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
            num_directions = 2 if bidirectional else 1
            model = LstmNet(num_layers, bidirectional)
            h0 = smith.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            c0 = smith.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            return model, (input, (h0, c0))

        num_layers = [1, 1, 2, 3]
        bidirectional = [True, False, True, False]
        models_and_inputs = [
            get_LstmNet_model_and_inputs(n, b)
            for n, b in zip(num_layers, bidirectional)
        ]
        for model, input in models_and_inputs:
            self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_sequence(self):
        class LstmNet(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn1 = smith.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                self.linear1 = smith.nn.Linear(8 * 2, 8)
                self.rnn2 = smith.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                self.linear2 = smith.nn.Linear(8 * 2, 8)

            def forward(self, input):
                rnn_output1, _ = self.rnn1(input)
                linear_output1 = self.linear1(rnn_output1)
                rnn_output2, _ = self.rnn2(linear_output1)
                linear_output2 = self.linear2(rnn_output2)
                return linear_output2

        input = smith.zeros((1, 100, 8), dtype=smith.float32)
        self.run_test(
            LstmNet(),
            input,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "w", 2: "h"},
                "output": {0: "batch_size", 1: "w", 2: "h"},
            },
        )

    @skipScriptTest()
    def test_rnn_no_bias(self):
        def make_model(layers, packed_sequence):
            batch_first = packed_sequence == 2
            model = smith.nn.RNN(
                RNN_INPUT_SIZE,
                RNN_HIDDEN_SIZE,
                layers,
                bidirectional=False,
                batch_first=batch_first,
                bias=False,
            )

            if packed_sequence == 1:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequence(
                    model, False
                )
            if packed_sequence == 2:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequence(
                    model, True
                )
            return model

        def make_input(batch_size, layers, packed_sequence):
            batch_first = packed_sequence == 2
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [smith.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]

            h0 = smith.randn(layers, batch_size, RNN_HIDDEN_SIZE)
            inputs.append(h0)
            if packed_sequence != 0:
                inputs.append(smith.IntTensor(seq_lengths))
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input

        layers = [1, 3, 1, 3, 1, 3]
        packed_sequence = [0, 0, 1, 1, 2, 2]
        models = [make_model(l, p) for l, p in zip(layers, packed_sequence)]
        inputs = [
            make_input(RNN_BATCH_SIZE, l, p) for l, p in zip(layers, packed_sequence)
        ]

        for model, input in zip(models, inputs):
            self.run_test(model, input)

    def test_gru_no_bias(self):
        class GruNet(smith.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super().__init__()
                self.mygru = smith.nn.GRU(
                    input_size,
                    hidden_size,
                    num_layers,
                    bidirectional=bidirectional,
                    bias=False,
                )

            def forward(self, input, initial_state):
                out = self.mygru(input, initial_state)
                return out

        def get_GruNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            num_directions = 2 if bidirectional else 1
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            input = smith.randn(seq_len, batch_size, input_size)
            h0 = smith.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, h0)

        input_size = [7, 5]
        hidden_size = [3, 4]
        num_layers = [2, 3]
        batch_size = [3, 4]
        seq_len = [5, 7]
        bidirectional = [True, False]
        models_and_inputs = [
            get_GruNet_model_and_inputs(i, h, n, b, s, bi)
            for i, h, n, b, s, bi in zip(
                input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
            )
        ]
        for model, input in models_and_inputs:
            self.run_test(model, input, do_constant_folding=True)

    def test_gru_constant_folding(self):
        class GruNet(smith.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super().__init__()
                self.mygru = smith.nn.GRU(
                    input_size, hidden_size, num_layers, bidirectional=bidirectional
                )

            def forward(self, input, initial_state):
                out = self.mygru(input, initial_state)
                return out

        def get_GruNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            num_directions = 2 if bidirectional else 1
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            input = smith.randn(seq_len, batch_size, input_size)
            h0 = smith.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, h0)

        batch_size1 = 3
        model1, input1 = get_GruNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_test(model1, input1, do_constant_folding=True)

        batch_size2 = 4
        model2, input2 = get_GruNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_test(model2, input2, do_constant_folding=True)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_max_tensors(self):
        class MaxModel(smith.nn.Module):
            def forward(self, input, other):
                return smith.max(input, other)

        model = MaxModel()
        x = smith.randn(4, 4, requires_grad=True)
        y = smith.randn(4, 1, requires_grad=True)
        self.run_test(model, (x, y))

    def test_amax_amin(self):
        class Model(smith.nn.Module):
            def forward(self, x):
                return smith.amax(x, dim=0, keepdim=True), smith.amin(
                    x, dim=[0, 1], keepdim=False
                )

        model = Model()
        x = smith.randn(4, 4)
        self.run_test(model, x)

    def test_aminmax(self):
        class Model(smith.nn.Module):
            def forward(self, x):
                return smith.aminmax(x, dim=1, keepdim=True), smith.aminmax(
                    x, keepdim=False
                )

        model = Model()
        x = smith.randn(3, 4)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_end(self):
        class ArangeScript(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, a):
                return smith.arange(a.size(0), dtype=smith.float).view(-1, 1) + a

        x = smith.randn(3, 4, requires_grad=True)
        outputs = ArangeScript()(x)
        self.run_test(ArangeScript(), x)

        class ArangeModel(smith.nn.Module):
            def forward(self, a):
                return smith.arange(a.size(0), dtype=smith.float).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_end_notype(self):
        class ArangeScript(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, a):
                return smith.arange(a.size(0))

        x = smith.randn(3, 4, requires_grad=True)
        outputs = ArangeScript()(x)
        self.run_test(ArangeScript(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(ArangeScript(), x, remained_onnx_input_idx=[])

        class ArangeModel(smith.nn.Module):
            def forward(self, a):
                return smith.arange(a.size(0))

        self.run_test(ArangeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(ArangeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end(self):
        class ArangeScript(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, a):
                return smith.arange(2, a.size(0) + 2, dtype=smith.float).view(-1, 1) + a

        x = smith.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(smith.nn.Module):
            def forward(self, a):
                return smith.arange(2, a.size(0) + 2, dtype=smith.float).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_end_notype(self):
        class ArangeScript(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, a):
                return smith.arange(2.7, a.size(0) + 2).view(-1, 1) + a

        x = smith.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(smith.nn.Module):
            def forward(self, a):
                return smith.arange(2.7, a.size(0) + 2).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end_step(self):
        class ArangeScript(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, a):
                return (
                    smith.arange(
                        2, a.size(0) * a.size(1) + 2, a.size(1), dtype=smith.float
                    ).view(-1, 1)
                    + a
                )

        x = smith.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(smith.nn.Module):
            def forward(self, a):
                return (
                    smith.arange(
                        2, a.size(0) * a.size(1) + 2, a.size(1), dtype=smith.float
                    ).view(-1, 1)
                    + a
                )

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_end_step_notype(self):
        class ArangeScript(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, a):
                return (
                    smith.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1)
                    + a
                )

        x = smith.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(smith.nn.Module):
            def forward(self, a):
                return (
                    smith.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1)
                    + a
                )

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test__dim_arange(self):
        class DimArange(smith.nn.Module):
            def forward(self, input):
                return smith._dim_arange(input, 1)

        x = smith.ones(5, 6)
        self.run_test(DimArange(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        remained_onnx_input_idx = None if self.opset_version < 11 else []
        self.run_test(DimArange(), x, remained_onnx_input_idx=remained_onnx_input_idx)

    def _test_compare_ops(self, model, num_inputs):
        x_float = smith.randn(1, 2, 3, 4, requires_grad=True)
        x_int = smith.randint(10, (3, 4), dtype=smith.int32)
        if num_inputs > 1:
            y_float = smith.randn(1, 2, 3, 4, requires_grad=True)
            y_int = smith.randint(10, (3, 4), dtype=smith.int32)
            self.run_test(model, (x_float, y_float))
            self.run_test(model, (x_float, y_int))
            self.run_test(model, (x_int, y_float))
            self.run_test(model, (x_int, y_int))
        else:
            self.run_test(model, x_float)
            self.run_test(model, x_int)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_and_or_xor(self):
        class MyModel(smith.nn.Module):
            def forward(self, x, y):
                return x ^ y, x | y, x & y, ~x

        x = smith.randint(0, 2, (5, 5), dtype=smith.bool)
        y = smith.randint(0, 2, (5, 5), dtype=smith.bool)
        self.run_test(MyModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_and(self):
        class AndModel(smith.nn.Module):
            def forward(self, x, y):
                return smith.logical_and(x, y)

        x = smith.randint(0, 2, (5, 5), dtype=smith.bool)
        y = smith.randint(0, 2, (5, 5), dtype=smith.bool)
        self.run_test(AndModel(), input_args=(x, y))

        x = smith.randint(10, (5, 5), dtype=smith.int32)
        y = smith.randint(10, (5, 5), dtype=smith.int32)
        self.run_test(AndModel(), input_args=(x, y))

        x = smith.randint(10, (5, 5), dtype=smith.double)
        y = smith.randint(10, (5, 5), dtype=smith.double)
        self.run_test(AndModel(), input_args=(x, y))

        x = smith.randint(10, (2, 3, 5), dtype=smith.float32)
        y = smith.randint(10, (2, 3, 5), dtype=smith.long)
        self.run_test(AndModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_or(self):
        class OrModel(smith.nn.Module):
            def forward(self, x, y):
                return smith.logical_or(x, y)

        x = smith.randint(0, 2, (5, 5), dtype=smith.bool)
        y = smith.randint(0, 2, (5, 5), dtype=smith.bool)
        self.run_test(OrModel(), input_args=(x, y))

        x = smith.randint(10, (5, 5), dtype=smith.int32)
        y = smith.randint(10, (5, 5), dtype=smith.int32)
        self.run_test(OrModel(), input_args=(x, y))

        x = smith.randint(10, (5, 5), dtype=smith.double)
        y = smith.randint(10, (5, 5), dtype=smith.double)
        self.run_test(OrModel(), input_args=(x, y))

        x = smith.randint(10, (2, 3, 5), dtype=smith.float32)
        y = smith.randint(10, (2, 3, 5), dtype=smith.long)
        self.run_test(OrModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_xor(self):
        class XorModel(smith.nn.Module):
            def forward(self, x, y):
                return smith.logical_xor(x, y)

        x = smith.randint(0, 2, (5, 5), dtype=smith.bool)
        y = smith.randint(0, 2, (5, 5), dtype=smith.bool)
        self.run_test(XorModel(), input_args=(x, y))

        x = smith.randint(10, (5, 5), dtype=smith.int32)
        y = smith.randint(10, (5, 5), dtype=smith.int32)
        self.run_test(XorModel(), input_args=(x, y))

        x = smith.randint(10, (5, 5), dtype=smith.double)
        y = smith.randint(10, (5, 5), dtype=smith.double)
        self.run_test(XorModel(), input_args=(x, y))

        x = smith.randint(10, (2, 3, 5), dtype=smith.float32)
        y = smith.randint(10, (2, 3, 5), dtype=smith.long)
        self.run_test(XorModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_not(self):
        class NotModel(smith.nn.Module):
            def forward(self, x):
                return smith.logical_not(x)

        x = smith.randint(0, 2, (5, 5), dtype=smith.bool)
        self.run_test(NotModel(), input_args=(x,))

        x = smith.randint(10, (5, 5), dtype=smith.int32)
        self.run_test(NotModel(), input_args=(x,))

        x = smith.randint(10, (5, 5), dtype=smith.double)
        self.run_test(NotModel(), input_args=(x,))

        x = smith.randint(10, (2, 3, 5), dtype=smith.float32)
        self.run_test(NotModel(), input_args=(x,))

    @skipIfUnsupportedMinOpsetVersion(11)  # float equal added after opset 11
    def test_eq(self):
        class EqualModel(smith.nn.Module):
            def forward(self, input, other):
                return input == other

        self._test_compare_ops(EqualModel(), 2)

    def test_gt(self):
        class GreaterModel(smith.nn.Module):
            def forward(self, input, other):
                return input > other

        self._test_compare_ops(GreaterModel(), 2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ge(self):
        class GreaterOrEqualModel(smith.nn.Module):
            def forward(self, input, other):
                return input >= other

        self._test_compare_ops(GreaterOrEqualModel(), 2)

    def test_gt_scalar(self):
        class GreaterModel(smith.nn.Module):
            def forward(self, input):
                return input > 1

        self._test_compare_ops(GreaterModel(), 1)

    def test_gt_primitive(self):
        class GreaterModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.y: int = 2

            def forward(self, x: int):
                return self.y > x

        x = 3
        self.run_test(GreaterModel(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ge_scalar(self):
        class GreaterOrEqualModel(smith.nn.Module):
            def forward(self, input):
                return input >= 1

        self._test_compare_ops(GreaterOrEqualModel(), 1)

    def test_lt(self):
        class LessModel(smith.nn.Module):
            def forward(self, input, other):
                return input > other

        self._test_compare_ops(LessModel(), 2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_le(self):
        class LessOrEqualModel(smith.nn.Module):
            def forward(self, input, other):
                return input <= other

        self._test_compare_ops(LessOrEqualModel(), 2)

    def test_lt_scalar(self):
        class LessModel(smith.nn.Module):
            def forward(self, input):
                return input < 1

        self._test_compare_ops(LessModel(), 1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_le_scalar(self):
        class LessOrEqualModel(smith.nn.Module):
            def forward(self, input):
                return input <= 1

        self._test_compare_ops(LessOrEqualModel(), 1)

    def test_matmul(self):
        class MatmulModel(smith.nn.Module):
            def forward(self, input, other):
                return smith.matmul(input, other)

        x = smith.randn(3, 4, requires_grad=True)
        y = smith.randn(4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = smith.randint(10, (3, 4))
        y = smith.randint(10, (4, 5))
        self.run_test(MatmulModel(), (x, y))

    def test_matmul_batch(self):
        class MatmulModel(smith.nn.Module):
            def forward(self, input, other):
                return smith.matmul(input, other)

        x = smith.randn(2, 3, 4, requires_grad=True)
        y = smith.randn(2, 4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = smith.randint(10, (2, 3, 4))
        y = smith.randint(10, (2, 4, 5))
        self.run_test(MatmulModel(), (x, y))

    def _argmin_argmax_model(self, input):
        class ArgminArgmaxModel(smith.nn.Module):
            def forward(self, input):
                return (
                    smith.argmin(input),
                    smith.argmax(input),
                    smith.argmin(input, keepdim=True),
                    smith.argmax(input, keepdim=True),
                    smith.argmin(input, dim=0, keepdim=True),
                    smith.argmax(input, dim=1, keepdim=True),
                )

        self.run_test(ArgminArgmaxModel(), input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_argmin_argmax(self):
        input = smith.randn(7, 3, 5)
        self._argmin_argmax_model(input)

    # Argmin and Argmax with "select_last_index" is not supported before opset 12
    # "select_last_index" was added in opset 12 to deal with corner case where the
    # same value appears multiple times in the tensor
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_argmin_argmax_select_last_index(self):
        input = smith.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]])
        self._argmin_argmax_model(input)

        input = smith.ones(7, 3, 5)
        self._argmin_argmax_model(input)

    def test_repeat(self):
        class RepeatModel(smith.nn.Module):
            def forward(self, x, y):
                x2 = x.repeat(y.shape[0], 1)
                y1 = y.view(-1, 1)
                return x2 + y1

        x = smith.tensor([1, 2, 3])
        y = smith.tensor([4, 5, 8, 9])
        self.run_test(RepeatModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_repeat_interleave(self):
        class FlattenModel(smith.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(2)

        for shape in ([3], [3, 4], [2, 3, 4]):
            x = smith.randn(shape)
            self.run_test(FlattenModel(), (x,))

        class DimsModel(smith.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(4, dim=1)

        x = smith.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel(), (x,))

        class DimsModel2(smith.nn.Module):
            def forward(self, x):
                repeats = smith.tensor([4])
                return smith.repeat_interleave(x, repeats, dim=1)

        x = smith.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel2(), (x,))

        class RepeatsDimsModel(smith.nn.Module):
            def forward(self, x):
                repeats = smith.tensor([1, 2])
                return smith.repeat_interleave(x, repeats, dim=0)

        x = smith.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel(), (x,))

        class RepeatsDimsModel2(smith.nn.Module):
            def forward(self, x):
                repeats = smith.tensor([1, 2])
                return smith.repeat_interleave(x, repeats, dim=1)

        x = smith.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel2(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_repeat_interleave_noop(self):
        class Model(smith.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(1, dim=1)

        x = smith.randn(4, 1, 8)
        self.run_test(Model(), (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_dynamic_repeat_interleave(self):
        class SingleDynamicModel(smith.nn.Module):
            def forward(self, x):
                repeats = smith.tensor(4)
                return smith.repeat_interleave(x, repeats, dim=1)

        x = smith.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = smith.tensor([[7, 8], [5, 6]])
        self.run_test(
            SingleDynamicModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": {1: "w"}},
        )

        class NegDynamicModel(smith.nn.Module):
            def forward(self, x):
                repeats = smith.tensor(4)
                return smith.repeat_interleave(x, repeats, dim=-1)

        x = smith.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = smith.tensor([[7, 8], [5, 6]])
        self.run_test(
            NegDynamicModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": {1: "w"}},
        )

        class SingleDynamicModelFloat(smith.nn.Module):
            def forward(self, x):
                repeats = smith.tensor([4])
                return smith.repeat_interleave(x, repeats, dim=0)

        x = smith.tensor([[1.1, 2.1], [3.1, 4.1]])
        another_x = smith.tensor([[7.1, 8.1], [5.1, 6.1]])
        self.run_test(
            SingleDynamicModelFloat(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": {0: "h"}},
        )

        class DynamicRepeatsModel(smith.nn.Module):
            def forward(self, x, repeats):
                return smith.repeat_interleave(x, repeats, dim=1)

        x = smith.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = smith.tensor([[7, 8], [5, 6]])
        repeats = smith.tensor([2])
        another_repeats = smith.tensor([4])
        self.run_test(
            DynamicRepeatsModel(),
            (x, repeats),
            additional_test_inputs=[(another_x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"input_1": {1: "w"}, "repeats_1": {0: "r"}},
        )

        class DynamicRepeatsModel2(smith.nn.Module):
            def forward(self, x, repeats):
                return smith.repeat_interleave(x, repeats, dim=1)

        x = smith.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = smith.tensor([2])
        another_repeats = smith.tensor([4])
        self.run_test(
            DynamicRepeatsModel2(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"repeats_1": {0: "r"}},
        )

        class DynamicFlattenModel(smith.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(2)

        x = smith.tensor([1, 2, 3])
        self.run_test(
            DynamicFlattenModel(),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": {0: "w"}},
        )

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_multiple_dynamic_repeat_interleave(self):
        class DynamicRepeatsModel(smith.nn.Module):
            def forward(self, x, repeats):
                return smith.repeat_interleave(x, repeats, dim=1)

        x = smith.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = smith.tensor([2, 3, 4])
        another_repeats = smith.tensor([4, 3, 2])
        self.run_test(
            DynamicRepeatsModel(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"repeats_1": {0: "r"}},
        )

        class DynamicRepeatsModel2(smith.nn.Module):
            def forward(self, x, repeats):
                return smith.repeat_interleave(x, repeats, dim=0)

        x = smith.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = smith.tensor([2, 3])
        another_repeats = smith.tensor([4, 3])
        self.run_test(
            DynamicRepeatsModel2(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"repeats_1": {0: "r"}},
        )

    def test_view(self):
        class ViewModel(smith.nn.Module):
            def forward(self, input):
                return input.view(4, 24)

        x = smith.randint(10, (4, 2, 3, 4), dtype=smith.int32)
        self.run_test(ViewModel(), x)

    def test_view_dynamic(self):
        class ViewModel(smith.nn.Module):
            def forward(self, input, other):
                return input.view(other.shape)

        x = smith.randn(2, 3, 4)
        shape = smith.randn(6, 4)
        self.run_test(
            ViewModel(),
            (x, shape),
            input_names=["x", "shape"],
            dynamic_axes={"x": [0, 1, 2], "shape": [0, 1]},
        )
        self.run_test(ViewModel(), (x, shape), remained_onnx_input_idx=[0])

    def test_view_dynamic_zero_dim(self):
        class ViewModel(smith.nn.Module):
            def forward(self, input):
                input = input.view(-1, 2)
                return input.view(1, -1)

        x = smith.ones(2)
        another_x = smith.empty((0,))
        self.run_test(
            ViewModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={
                "input_1": [
                    0,
                ]
            },
        )

    def test_view_as(self):
        class ViewModel(smith.nn.Module):
            def forward(self, input, other):
                return input.view_as(other)

        x = smith.randn(2, 3, 4)
        y = smith.randn(6, 4)
        self.run_test(ViewModel(), (x, y))

    def test_linear(self):
        class LinearModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = smith.nn.Linear(16, 16)

            def forward(self, x):
                out = self.fc(x)
                out = self.fc(out)
                return out

        x = smith.randn(3, 16)
        self.run_test(LinearModel(), (x,))

        class LinearModel(smith.nn.Module):
            def forward(self, input, weight, bias):
                return smith.nn.functional.linear(input, weight, bias)

        # input of rank 2
        x = smith.randn(2, 2)
        y = smith.randn(2, 2)
        z = smith.randn(1)
        self.run_test(LinearModel(), (x, y, z))

        # input of rank 3
        x = smith.randn(3, 3, 3)
        y = smith.randn(3, 3)
        z = smith.randn(1)
        self.run_test(LinearModel(), (x, y, z))

    @skipScriptTest()
    def test_weight_norm(self):
        # addmm for 3-d inputs converts to onnx::MatMul
        model = smith.nn.utils.weight_norm(smith.nn.Linear(5, 10), dim=1)
        x = smith.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)

        # addmm for 2-d inputs converts to onnx::Gemm
        model = smith.nn.utils.weight_norm(smith.nn.Linear(5, 10), dim=1)
        x = smith.randn(4, 5, requires_grad=True)
        self.run_test(model, x)

        model = smith.nn.utils.weight_norm(smith.nn.Conv1d(1, 1, 3))
        x = smith.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)

        model = smith.nn.utils.weight_norm(smith.nn.Conv1d(1, 1, 3), dim=-2)
        x = smith.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)

        model = smith.nn.utils.weight_norm(smith.nn.Conv1d(3, 6, 3), name="weight")
        x = smith.randn(3, 3, 5, requires_grad=True)
        self.run_test(model, x)

    @skipScriptTest()
    def test_weight_norm_nodim(self):
        # addmm for 3-d inputs converts to onnx::MatMul
        model = smith.nn.utils.weight_norm(smith.nn.Linear(5, 10), dim=None)
        x = smith.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)

        # addmm for 2-d inputs converts to onnx::Gemm
        model = smith.nn.utils.weight_norm(smith.nn.Linear(5, 10), dim=None)
        x = smith.randn(4, 5, requires_grad=True)
        self.run_test(model, x)

    def test_flatten(self):
        class FlattenModel(smith.nn.Module):
            def forward(self, input):
                return smith.flatten(input)

        model = FlattenModel()

        # flatten with 4d input
        x = smith.randint(10, (1, 2, 3, 4))
        self.run_test(model, x)

        # flatten with 0d input
        x = smith.randn([])
        self.run_test(model, x)

        # flatten with 1d input
        x = smith.randn(4)
        self.run_test(model, x)

    def test_flatten2d(self):
        class FlattenModel(smith.nn.Module):
            def forward(self, input):
                return smith.flatten(input, 1)

        x = smith.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    def test_flatten2d_neg(self):
        class FlattenModel(smith.nn.Module):
            def forward(self, x):
                return (
                    smith.flatten(x, 1, -1),
                    smith.flatten(x, 0, -2),
                    smith.flatten(x, 1, -2),
                )

        x = smith.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_flatten_dynamic_axes(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                return smith.flatten(x, start_dim=2, end_dim=3)

        batch_size = 3
        x = smith.randn(batch_size, 5, 4, 5)
        y = smith.randn(5, 5, 4, 5)
        model = MyModule()
        self.run_test(
            model,
            x,
            additional_test_inputs=[y],
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_getitem(self):
        class GetItemModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x, y, z, ind):
                # this will create prim::ListConstruct(x, y, z) + aten::__getitem__
                arr = [x, y, z]
                return arr[ind]

        x = smith.randn(3, 4, 5)
        y = smith.randn(1, 4, 5)
        z = smith.randn(2, 4, 5)
        ind = smith.tensor(1, dtype=smith.long)
        self.run_test(GetItemModel(), (x, y, z, ind))

        ind = smith.tensor(-2, dtype=smith.long)
        self.run_test(GetItemModel(), (x, y, z, ind))

    @skipDtypeChecking
    def test_item(self):
        class M(smith.nn.Module):
            def forward(self, x, y, i: int):
                return int(x[y[i]].item())

        x = smith.arange(6, dtype=smith.float)
        y = smith.tensor([0, 1, 2, 3, 4], dtype=smith.long)
        i = 3
        self.run_test(smith.jit.script(M()), (x, y, i))

    @skipScriptTest()  # smith.nonzero(x, as_tuple=True) is not scriptable.
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_nonzero(self):
        class NonzeroModel(smith.nn.Module):
            def forward(self, x):
                return x.nonzero(), x.nonzero(as_tuple=True)

        x = smith.randn(60).index_fill_(0, smith.randint(0, 60, (20,)), 0).view(3, 4, 5)
        self.run_test(NonzeroModel(), (x,))

    def test_unbind(self):
        class UnbindModel(smith.nn.Module):
            def forward(self, input):
                _, out, _ = input.unbind()
                return out

        x = smith.randn(3, 4, 5)
        self.run_test(UnbindModel(), x)

        class UnbindModel2(smith.nn.Module):
            def forward(self, input):
                _, out, _, _ = input.unbind(1)
                return out

        x = smith.randn(3, 4, 5)
        self.run_test(UnbindModel2(), x)

        class UnbindModel3(smith.nn.Module):
            def forward(self, input):
                _, out, _, _ = input.unbind(-2)
                return out

        x = smith.randn(3, 4, 5)
        self.run_test(UnbindModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_len(self):
        class LenModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return len(input.unbind()) + input

        x = smith.randn(4, 5)
        self.run_test(
            LenModel(),
            x,
            input_names=["input"],
            dynamic_axes={"input": {0: "seq"}},
            additional_test_inputs=(smith.randn(5, 5),),
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_len_list(self):
        class LenListModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return smith.ones(len(input.shape))

        x = smith.randn(4, 5)
        self.run_test(LenListModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unbind_dynamic(self):
        class UnbindModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return input.unbind()[1]

        x = smith.randn(3, 4, 5)
        self.run_test(UnbindModel(), x)

        class UnbindModel2(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return input.unbind(-1)[1]

        x = smith.randn(3, 4, 5)
        self.run_test(UnbindModel2(), x)

    @skipScriptTest()  # scripting tests run for opsets > 11. See: test_split_script
    def test_split(self):
        class SplitModel(smith.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        x = smith.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(smith.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        x = smith.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(smith.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2])

        x = smith.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_script(self):
        class SplitModel(smith.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        x = smith.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(smith.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        x = smith.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(smith.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2])

        x = smith.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_split_size_as_list(self):
        class SplitModel(smith.nn.Module):
            def forward(self, input, split_sizes: list[int]):
                out = []
                split_list: list[Tensor] = input.split(split_sizes)

                for ob in split_list:
                    out.append(ob)  # noqa: PERF402
                return smith.cat(out, dim=0)

        x = smith.randn(6, 4, 3)
        split_sizes = [smith.tensor(2), smith.tensor(4)]
        self.run_test(SplitModel(), (x, split_sizes))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_size_with_slice(self):
        class SplitModule(smith.nn.Module):
            def forward(self, x, y, t):
                splits = (x.size(1), y.size(1))
                out, out2 = smith.split(t, splits, dim=1)
                return out, out2

        x = smith.randn(2, 3)
        y = smith.randn(2, 4)
        t = smith.randn(2, 7)
        self.run_test(
            SplitModule(),
            (x, y, t),
            input_names=["x", "y", "t"],
            dynamic_axes={"x": [0, 1], "y": [0, 1], "t": [0, 1]},
        )
        self.run_test(SplitModule(), (x, y, t), remained_onnx_input_idx=[2])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_dynamic(self):
        class SplitModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return input.split(2)[1]

        x = smith.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                return input.split(2, -3)[1]

        x = smith.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_dynamic_axes(self):
        class Split(smith.nn.Module):
            def forward(self, x):
                return x.split(1, dim=-1)

        x = smith.randn(4, 384, 2)
        input_names = ["logits"]
        self.run_test(
            Split(),
            x,
            input_names=input_names,
            dynamic_axes={input_names[0]: {0: "batch"}},
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_chunk(self):
        class ChunkModel(smith.nn.Module):
            def __init__(self, dim=1):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return smith.chunk(x, 3, dim=self.dim)

        model = ChunkModel()
        model.eval()
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        x = smith.randn(1, 18)

        for dim_size_ in range(13, 16):
            y = smith.randn(1, dim_size_)
            self.run_test(
                model,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

            self.run_test(
                model_neg_dim,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dynamic_chunk(self):
        class ChunkModel(smith.nn.Module):
            def __init__(self, dim=1):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return smith.chunk(x, x.size(0), dim=self.dim)

        model = ChunkModel()
        model.eval()
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        x = smith.randn(3, 18)

        for dim_size_ in range(13, 16):
            y = smith.randn(3, dim_size_)
            self.run_test(
                model,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

            self.run_test(
                model_neg_dim,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

    def test_concat(self):
        class ConcatModel(smith.nn.Module):
            def forward(self, x, y, z):
                return smith.cat((x, y, z))

        x = smith.randn(3, 4, 5)
        y = smith.randn(1, 4, 5)
        z = smith.randn(2, 4, 5)
        self.run_test(ConcatModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_concat_dynamic(self):
        class ConcatDynamicModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return smith.cat(x.unbind())

        x = smith.randn(4, 5, 6)
        self.run_test(ConcatDynamicModel(), x)

    def test_stack(self):
        class StackModel(smith.nn.Module):
            def forward(self, x, y, z):
                return smith.stack((x, y, z), 1)

        x = smith.randn(3, 4, 5)
        y = smith.randn(3, 4, 5)
        z = smith.randn(3, 4, 5)
        self.run_test(StackModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_stack_dynamic(self):
        class StackDynamicModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return smith.stack(x.unbind(), 1)

        x = smith.randn(4, 5, 6)
        self.run_test(StackDynamicModel(), x)

    def test_loop_dynamic(self):
        class LoopModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                for i in range(x.size(2)):
                    x = x + i
                return x

        model = LoopModel()
        inputs = smith.zeros(1, 2, 3, dtype=smith.long)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_loop_nested(self):
        class NestedLoopsModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                for _ in range(5):
                    a = 0
                    while a < 4:
                        a += 1
                    x = x + a
                return x

        model = NestedLoopsModel()
        inputs = smith.zeros(1, 2, 3, dtype=smith.long)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_with_list(self):
        class ListLoopModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                res = []
                res1 = []
                arr = x.split([3, 4, 1, 1, 2, 3, 2], 0)
                res2 = smith.zeros(3, 4, dtype=smith.long)
                res3 = []
                res4 = []
                for i in range(len(arr)):
                    res.append(arr[i].sum(0, False))
                    res1.append(arr[-1 - i].sum(0, False))
                    res2 += 1
                    res3 = res3 + [arr[i].sum(0, False)]
                    res4 += [arr[-1 - i].sum(0, False)]
                return res, res1, res2, smith.stack(res3), smith.stack(res4)

        model = ListLoopModel()
        inputs = smith.randn(16)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_transpose(self):
        class LoopModel(smith.nn.Module):
            def forward(self, x):
                res = smith.zeros_like(x[0])
                for _ in range(x.size(0)):
                    res += x[0].transpose(0, 1)
                return res

        model = smith.jit.script(LoopModel())
        x = smith.randn(5, 3, 3)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_multi_dim(self):
        class LoopMultiDimModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x, y):
                for x_ in smith.flip(x.narrow(0, 0, 7), [0]):
                    y = x_[0][y]
                return y

        model = LoopMultiDimModel()
        x = smith.randint(0, 5, (8, 1, 17), dtype=smith.long)
        y = smith.ones(1, dtype=smith.long)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list(self):
        class ListModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                tensors = x.unbind()
                res = []
                res.append(tensors[0])
                res.append(tensors[1])
                res.pop(1)

                res.insert(0, tensors[1])
                res.append(tensors[2])
                res += [tensors[3], tensors[4]]
                res = res + [tensors[5]]
                return smith.ones(len(res))

        model = ListModel()
        inputs = smith.randn(16, 1)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_append(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [smith.matmul(x[i], y)]
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(16, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_nested(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [smith.matmul(x[i][j], y)]
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(4, 4, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(14)  # Need onnx::Identity of sequence in opset 14
    def test_list_append_nested_2(self):
        class ListModel(smith.nn.Module):
            def forward(self, x):
                res = []
                res_replicate = []
                for i in range(x.size(0)):
                    if len(res) > 2:
                        for j in range(x.size(1)):
                            res.append(x[i][j])
                        res_replicate.append(res[-1])
                        res.append(res_replicate[-1])
                return res, res_replicate

        model = smith.jit.script(ListModel())
        x = smith.randn(4, 4, 3, 4)
        self.run_test(model, (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_nested_mixed_dtype(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        if i == j:
                            res.append(x == y)
                        else:
                            res.append(x != y)
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(4, 4, 3, 4)
        y = smith.randn(3, 4)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_pop(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [smith.matmul(x[i], y)]
                res.pop()
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(16, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_pop_nested(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [smith.matmul(x[i][j], y)]
                        res.pop()
                    res += [smith.matmul(x[i][0], y)]
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(4, 4, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_del(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [smith.matmul(x[i], y)]
                del res[2]
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(16, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_nested(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [smith.matmul(x[i][j], y)]
                        del res[i]
                    res += [smith.matmul(x[i][0], y)]
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(4, 4, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_set(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res.append(x[i])
                res[y] = x[y]
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(12, 4)
        y = smith.tensor(2, dtype=smith.long)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_idx_sum(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                indices = smith.arange(x.size(0))
                res = []
                for i in range(x.size(0)):
                    res.append(x[i])
                return res[smith.sum(indices[:y])]

        model = smith.jit.script(ListModel())
        x = smith.randn(12, 4)
        y = smith.tensor(2, dtype=smith.long)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories(self):
        class TensorFactory(smith.nn.Module):
            def forward(self, x):
                return smith.zeros(x.size()) + smith.ones(x.size())

        x = smith.randn(2, 3, 4)
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories_script(self):
        class TensorFactory(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                return smith.zeros(x.shape, dtype=smith.float) + smith.ones(
                    x.shape, dtype=smith.float
                )

        x = smith.randn(2, 3, 4)
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_like_factories_script(self):
        class TensorFactory(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                zeros = smith.zeros_like(
                    x,
                    dtype=smith.float,
                    layout=smith.strided,
                    device=smith.device("cpu"),
                )
                ones = smith.ones_like(
                    x,
                    dtype=smith.float,
                    layout=smith.strided,
                    device=smith.device("cpu"),
                )
                return zeros + ones

        x = smith.randn(2, 3, 4)
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split(self):
        class TensorSplitModel(smith.nn.Module):
            def forward(self, input):
                return (
                    input.tensor_split([1, 3]),
                    # test with output indexing.
                    input.tensor_split([2, 4])[0],
                    # test split on specific dim.
                    input.tensor_split([1, 3, 4], dim=-2),
                    # test split on specific dim and output indexing.
                    input.tensor_split([0, 2], dim=-2)[-1],
                    # test with out of bound end index (5).
                    input.tensor_split([2, 3, 5]),
                )

        self.run_test(TensorSplitModel(), smith.randn(5, 4, 3))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split_scalar(self):
        class TensorSplitModel(smith.nn.Module):
            def forward(self, x):
                return smith.tensor_split(x, x.size(1))

        self.run_test(TensorSplitModel(), smith.randn(1, 2, 3))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split_dynamic_axes(self):
        class TensorSplitModel(smith.nn.Module):
            def forward(self, x):
                return x.tensor_split(1, dim=-1)

        x = smith.randn(4, 384, 2)
        input_names = ["logits"]
        self.run_test(
            TensorSplitModel(),
            x,
            input_names=input_names,
            dynamic_axes={input_names[0]: {0: "batch"}},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_eye(self):
        class TensorFactory(smith.nn.Module):
            def forward(self, x):
                return (
                    smith.eye(x.size()[1], 3),
                    smith.eye(4, 4, dtype=smith.long),
                    smith.eye(x.size()[1], 2, dtype=smith.long),
                    smith.eye(x.shape[0]),
                    smith.eye(x.shape[0], dtype=smith.float64),
                )

        x = smith.randn(2, 3, 4)
        another_x = smith.randn(5, 6, 7)
        self.run_test(
            TensorFactory(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2]},
        )

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_diagonal(self):
        class DiagonalModel(smith.nn.Module):
            def forward(self, x):
                return smith.diagonal(x)

        x = smith.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = smith.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelNegOffset(smith.nn.Module):
            def forward(self, x):
                return smith.diagonal(x, offset=-1)

        x = smith.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = smith.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelNegOffset(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelPosOffset(smith.nn.Module):
            def forward(self, x):
                return smith.diagonal(x, offset=1)

        x = smith.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = smith.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelPosOffset(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelWithDims(smith.nn.Module):
            def forward(self, x):
                return smith.diagonal(x, offset=-1, dim1=1, dim2=2)

        x = smith.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = smith.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelWithDims(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelWithNegativeDims(smith.nn.Module):
            def forward(self, x):
                return smith.diagonal(x, offset=0, dim1=-2, dim2=-1)

        x = smith.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = smith.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelWithNegativeDims(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelOffsetOverrun(smith.nn.Module):
            def forward(self, x):
                return smith.diagonal(x, offset=-2), smith.diagonal(x, offset=5)

        x = smith.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = smith.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelOffsetOverrun(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_zero(self):
        class Zero_(smith.nn.Module):
            def forward(self, x):
                return x.zero_(), x

        x = smith.randn(2, 3, 4)
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Zero_(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_zero_qkv(self):
        class Zero_(smith.nn.Module):
            def forward(self, x):
                return x[2:4].zero_()

        x = smith.randn(24, 3, 4)
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_zeros(self):
        class Zero_(smith.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[1:2]), x.new_zeros(
                    x.shape[2:], dtype=smith.long
                )

        x = smith.randn(2, 3, 4)
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Zero_(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_zeros_with_dtype(self):
        class MyModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = smith.nn.Embedding(50, 64)

            def forward(self, x):
                inp = x.new_zeros(x.shape)
                return self.emb(inp)

        model = MyModel()
        x = smith.Tensor([[2, 5, 6], [3, 2, 5]]).to(smith.int64)
        self.run_test(model, x, input_names=["x"], dynamic_axes={"x": [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_ones(self):
        class OnesModel(smith.nn.Module):
            def forward(self, x):
                return x.new_ones(x.shape[1:2]), x.new_ones(
                    x.shape[2:], dtype=smith.long
                )

        x = smith.randn(2, 3, 4)
        self.run_test(OnesModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(OnesModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()  # smith.zeros/smith.ones with size tensor of dim != 0 not scriptable.
    def test_zeros_ones_with_tensor_input(self):
        class ZeroAndOnes(smith.nn.Module):
            def forward(self, x):
                return smith.zeros(x, 1), smith.ones(x, 1)

        x = smith.tensor([2])
        self.run_test(ZeroAndOnes(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipShapeChecking
    def test_tolist(self):
        class List(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                res: list[int] = input.tolist()
                return res

        self.run_test(List(), (smith.randint(100, (1,)),))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_pass(self):
        class Slice(smith.nn.Module):
            def forward(self, x, y):
                return x.new_zeros(x.shape[2:] + y.shape[1:])

        x = smith.randn(2, 3, 4, 5)
        y = smith.randn(1, 2, 3, 4)
        self.run_test(
            Slice(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        )
        self.run_test(Slice(), (x, y), remained_onnx_input_idx=[])

        class Size(smith.nn.Module):
            def forward(self, x, y):
                return x.new_zeros(x.shape + y.shape)

        x = smith.randn(2, 3, 4)
        y = smith.randn(1, 2, 3)
        self.run_test(
            Size(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        self.run_test(Size(), (x, y), remained_onnx_input_idx=[])

        class Array(smith.nn.Module):
            def forward(self, x, y):
                arr1 = [x.shape[0], x.shape[1], 2]
                arr2 = [y.shape[0], y.shape[1]]
                return x.new_zeros(arr1 + arr2)

        x = smith.randn(2, 3, 4)
        y = smith.randn(1, 2, 3)
        self.run_test(
            Array(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        self.run_test(Array(), (x, y), remained_onnx_input_idx=[])

        class List(smith.nn.Module):
            def forward(self, x, y):
                l1 = list(x.shape)
                l2 = list(y.shape)
                return x.new_zeros(l1 + l2)

        x = smith.randn(2, 3, 4)
        y = smith.randn(1, 2, 3)
        self.run_test(
            List(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        self.run_test(List(), (x, y), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_empty(self):
        class Empty(smith.nn.Module):
            def forward(self, x):
                return (
                    x.new_empty(x.shape[0]).fill_(0),
                    x.new_empty(x.shape[0], dtype=smith.long) * 0,
                )

        x = smith.randn(2, 3, 4)
        self.run_test(Empty(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Empty(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_full(self):
        class Full(smith.nn.Module):
            def forward(self, x):
                return x.new_full(x.shape[1:2], 5), x.new_full(
                    x.shape[0:1], 1.3, dtype=smith.long
                )

        x = smith.randn(2, 3, 4)
        self.run_test(Full(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Full(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_list(self):
        class Arithmetic(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x, y):
                return smith.cat([x.add_(3), y.fill_(0)])

        x = smith.randn(2, 3)
        y = smith.randn(2, 3)
        self.run_test(
            Arithmetic(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1], "y": [0, 1]},
        )
        self.run_test(Arithmetic(), (x, y), remained_onnx_input_idx=[0])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_fill(self):
        class Fill_(smith.nn.Module):
            def forward(self, x):
                return x.fill_(3), x

        x = smith.randn(2, 3, 4)
        self.run_test(Fill_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Fill_(), x, remained_onnx_input_idx=[])

    def test_inplace_arithmetic(self):
        class Arithmetic(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x, y):
                x.add_(3)
                y.mul_(x)
                return x, y

        x = smith.randn(2, 3, 4)
        y = smith.randn(2, 3, 4)
        self.run_test(Arithmetic(), (x, y))

    def test_inplace_arithmetic_half(self):
        class InplaceAddModel(smith.nn.Module):
            def forward(self, x, y):
                return x.add_(y)

        class InplaceMulModel(smith.nn.Module):
            def forward(self, x, y):
                return x.mul_(y)

        x = smith.randn(2, 2, dtype=smith.half)
        y = smith.randn(2, 2, dtype=smith.float)
        self.run_test(InplaceAddModel(), (x, y), rtol=1e-2, atol=1e-2)
        self.run_test(InplaceMulModel(), (x, y), rtol=1e-2, atol=1e-2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_with_loop(self):
        class M(smith.nn.Module):
            def forward(self, x):
                a = smith.ones(
                    12,
                )
                for _ in range(10):
                    a.add_(
                        smith.ones(
                            12,
                        )
                    )
                return a + x

        m = M()
        x = smith.randn(
            12,
        )
        self.run_test(smith.jit.script(M()), (x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_with_loop_2(self):
        class M(smith.nn.Module):
            def forward(self, x):
                _bias = smith.ones(
                    12,
                )
                a = smith.ones(
                    12,
                )  # used in loop, altered.
                a_ref = a  # not used in loop, should be altered.
                b = x.clone()  # used in loop, not be altered.
                b_ref = b  # not used in loop, should not be altered.
                for i in range(10):
                    if i == 3:
                        for _ in range(5):
                            a += _bias
                            _bias.add_(
                                smith.ones(
                                    12,
                                )
                            )
                            b = b + smith.ones(
                                12,
                            )

                    _bias.add_(
                        smith.ones(
                            12,
                        )
                    )
                    a += _bias
                # TODO: value for a_ref is incorrect.
                # a_ref += smith.ones(12,)
                b_ref += smith.ones(
                    12,
                )
                return _bias + x, a, b, b_ref

        m = M()
        x = smith.zeros(
            12,
        )
        self.run_test(smith.jit.script(M()), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_attr_with_loop(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._bias = smith.arange(
                    12,
                )

            def forward(self, x):
                self._bias = smith.arange(
                    12,
                )
                for i in range(10):
                    if i == 3:
                        for _ in range(5):
                            self._bias += smith.arange(
                                12,
                            )
                return self._bias + x

        m = M()
        x = smith.zeros(
            12,
        )
        self.run_test(smith.jit.script(M()), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_attr_copy_with_loop(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._bias = smith.arange(
                    12,
                )

            def forward(self, x):
                self._bias = smith.arange(
                    12,
                )
                for i in range(10):
                    if i == 3:
                        for _ in range(5):
                            self._bias.copy_(
                                smith.arange(
                                    12,
                                )
                            )
                        self._bias.copy_(
                            self._bias
                            + smith.arange(
                                12,
                            )
                        )

                    self._bias.copy_(
                        self._bias
                        + smith.arange(
                            12,
                        )
                    )
                return self._bias + x

        m = M()
        x = smith.zeros(
            12,
        )
        self.run_test(smith.jit.script(M()), (x))

    @skipIfUnsupportedMinOpsetVersion(14)  # Need onnx::Identity of sequence in opset 14
    def test_inplace_sequence_with_loop(self):
        class M(smith.nn.Module):
            def process(self, beam_hyps: list[Tensor], done: Tensor, x):
                batch_size = x.shape[0]
                for i in range(batch_size):
                    if done[i]:
                        continue

                    beam_idx = 0
                    for _, token in enumerate(x[i]):
                        beam_hyps.append(token)
                        beam_idx += 1  # noqa: SIM113

                        if beam_idx == 6:
                            break

                    done[i] = len(beam_hyps) > 4

                return beam_hyps, done

            def forward(self, x):
                beam_hyps: list[Tensor] = []
                batch_size = x.shape[0]
                cur_len = 0
                max_len = x.shape[1]
                done = smith.zeros(batch_size, dtype=smith.bool)
                while cur_len < max_len:
                    beam_hyps, done = self.process(beam_hyps, done, x[:, 0, :])
                    cur_len = cur_len + 1

                return beam_hyps

        m = smith.jit.script(M())
        x = smith.randn(8, 4, 3)
        self.run_test(smith.jit.script(M()), (x))

    @skipScriptTest()  # Sort with dynamic dim not supported in ONNX
    def test_sort(self):
        class SortModel(smith.nn.Module):
            def forward(self, x):
                out = []
                for i in range(-2, 2):
                    out.append(smith.sort(x, dim=i, descending=True))
                return out

        x = smith.randn(3, 4)
        self.run_test(SortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Sort with dynamic dim not supported in ONNX
    def test_sort_ascending(self):
        class SortModel(smith.nn.Module):
            def forward(self, x):
                out = []
                for i in range(-2, 2):
                    out.append(smith.sort(x, dim=i, descending=False))
                return out

        x = smith.randn(3, 4)
        self.run_test(SortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_argsort(self):
        class ArgSortModel(smith.nn.Module):
            def forward(self, x):
                return smith.argsort(x, dim=1, descending=False)

        x = smith.randn(3, 4)
        self.run_test(ArgSortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill(self):
        class MaskedFillModel(smith.nn.Module):
            def forward(self, x):
                mask = smith.tensor([[0, 0, 1], [1, 1, 0]], dtype=smith.bool)
                return x.masked_fill(mask, 2)

        x = smith.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(smith.nn.Module):
            def forward(self, x):
                return x.masked_fill(x > 3, -1)

        x = smith.arange(16).view(2, 2, 4).to(smith.float32)
        self.run_test(MaskedFillModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill_inplace(self):
        class MaskedFillModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                mask = smith.tensor([[0, 0, 1], [1, 1, 0]], dtype=smith.bool)
                x.masked_fill_(mask, 2)
                return x

        x = smith.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                x.masked_fill_(x > 3, -1)
                return x

        x = smith.arange(16).view(2, 2, 4).to(smith.float32)
        self.run_test(MaskedFillModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_scatter(self):
        class MaskedScatterModel(smith.nn.Module):
            def forward(self, x):
                return smith.masked_scatter(x, x.ge(0.5), smith.ones(100, 100) * 5)

        x = smith.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedScatterModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_select(self):
        class MaskedSelectModel(smith.nn.Module):
            def forward(self, x):
                return smith.masked_select(x, x.ge(0.5))

        x = smith.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedSelectModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_fill(self):
        class MaskedFillModel(smith.nn.Module):
            def forward(self, input_mask, some_const):
                mask = input_mask.clone()
                mask[mask != some_const] = 1
                mask[mask == some_const] = 0
                return mask

        mask = smith.randn(2, 2, 2, requires_grad=True)
        constant = smith.tensor(5, dtype=smith.float)
        self.run_test(MaskedFillModel(), (mask, constant))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_scatter(self):
        class MaskedScatterModel(smith.nn.Module):
            def forward(self, input_mask, some_const):
                mask = input_mask.clone()
                mask[mask != some_const] = smith.ones(8)
                return mask

        mask = smith.randn(2, 2, 2, requires_grad=True)
        constant = smith.tensor(5, dtype=smith.float)
        self.run_test(MaskedScatterModel(), (mask, constant))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_with_1d_mask_to_masked_scatter(self):
        class MaskedScatterModel(smith.nn.Module):
            def forward(self, tensor, mask, some_const):
                tensor[mask] = some_const
                return tensor

        mask = smith.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=smith.bool)
        tensor = smith.randn(8, 4, 5, requires_grad=True)
        some_const = smith.randn(4, 4, 5, dtype=smith.float)
        self.run_test(MaskedScatterModel(), (tensor, mask, some_const))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pixel_shuffle(self):
        class PixelShuffle(smith.nn.Module):
            def forward(self, x):
                return smith.pixel_shuffle(x, upscale_factor=2)

        x = smith.randn(2, 16, 4, 3, requires_grad=True)
        y = smith.randn(4, 32, 8, 4, requires_grad=True)
        self.run_test(PixelShuffle(), x)
        self.run_test(
            PixelShuffle(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pixel_unshuffle(self):
        class PixelUnshuffle(smith.nn.Module):
            def forward(self, x):
                return smith.pixel_unshuffle(x, downscale_factor=2)

        x = smith.randn(2, 16, 4, 6, requires_grad=True)
        y = smith.randn(4, 32, 8, 4, requires_grad=True)
        self.run_test(PixelUnshuffle(), x)
        self.run_test(
            PixelUnshuffle(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_reciprocal(self):
        class ReciprocalModel(smith.nn.Module):
            def forward(self, x):
                return smith.reciprocal(x)

        model = ReciprocalModel()
        x = smith.tensor([2, 4])
        self.run_test(model, x.to(smith.long))
        self.run_test(model, x.to(smith.float))
        self.run_test(model, x.to(smith.double))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type(self):
        class ArithmeticModel(smith.nn.Module):
            def forward(self, x):
                return x.size(0) * 2 * x, 2 - x

        x = smith.ones(2, 3, dtype=smith.float32)
        self.run_test(ArithmeticModel(), x)

        class ComparisonModel(smith.nn.Module):
            def forward(self, x, y):
                a = smith.tensor([12.0])
                return x.lt(1.5) & y.le(2) & x.le(1), x.gt(y), x.lt(y), a.ge(x.size(0))

        x = smith.ones(2, 3, dtype=smith.int32)
        y = smith.ones(2, 3, dtype=smith.float32)
        self.run_test(ComparisonModel(), (x, y))

        class MatMulModel(smith.nn.Module):
            def forward(self, x):
                return smith.mm(x, x) + x + smith.mm(x, x) + x

        x = smith.ones(3, 3)
        self.run_test(MatMulModel(), x)

        class AddMMModel(smith.nn.Module):
            def forward(self, x):
                return smith.mm(x, x) + x

        x = smith.ones(3, 3)
        self.run_test(AddMMModel(), x)

        class FullModel(smith.nn.Module):
            # add is used for exporting full
            def forward(self, x):
                return smith.full((3, 4), x)

        x = smith.tensor(12.0)
        self.run_test(FullModel(), x)

        class CatModel(smith.nn.Module):
            def forward(self, fp16, fp32):
                return smith.cat([fp16, fp32])

        fp16 = Tensor([0.5])
        fp16 = fp16.half()
        fp32 = Tensor([1.5])
        self.run_test(CatModel(), (fp16, fp32))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_does_not_trigger_upcast_type_promotion(self):
        class DoNotUpcastModel(smith.nn.Module):
            def forward(self, x):
                scale = x.size()[-1] ** -0.5
                # 'scale' is exported as onnx float32 rank 0 tensor.
                # The following 'Mul' should NOT be promoted to float32.
                return x * scale

        x = smith.ones(2, 3, dtype=smith.float16)
        self.run_test(DoNotUpcastModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_promotion_onnx_where_two_prim_const(self):
        class TwoPrimConstCastWhereModel(smith.nn.Module):
            def forward(self, c):
                return smith.where(c, 0, 1.0)

        c = smith.ones(8, dtype=smith.bool)
        self.run_test(TwoPrimConstCastWhereModel(), (c))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_promotion_onnx_where_one_prim_const(self):
        class OnePrimConstCastWhereModel(smith.nn.Module):
            def forward(self, c, x):
                return smith.where(c, x, 1.0)

        c = smith.ones(8, dtype=smith.bool)
        x = smith.ones(8, dtype=smith.float16)
        self.run_test(OnePrimConstCastWhereModel(), (c, x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_promotion_onnx_where_one_tensor_const(self):
        class OneTensorConstCastWhereModel(smith.nn.Module):
            def forward(self, c, x):
                return smith.where(c, x, smith.ones(size=(), dtype=smith.float64))

        c = smith.ones(8, dtype=smith.bool)
        x = smith.ones(8, dtype=smith.float16)
        self.run_test(OneTensorConstCastWhereModel(), (c, x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_upcast_type_promotion_onnx_where_no_const(self):
        class OnnxWhereUpcastModel(smith.nn.Module):
            def forward(self, c, x, y):
                return smith.where(c, x, y)

        c = smith.ones(8, dtype=smith.bool)
        x = smith.ones(8, dtype=smith.float16)
        y = smith.ones(8, dtype=smith.float32)

        self.run_test(OnnxWhereUpcastModel(), (c, x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_like(self):
        class FullLikeModel(smith.nn.Module):
            def forward(self, x):
                return smith.full_like(x, 1.3, dtype=smith.int)

        x = smith.tensor(12)
        self.run_test(FullLikeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipDtypeChecking
    def test_full_like_value(self):
        class FullLikeModel(smith.nn.Module):
            def forward(self, x, y):
                out = y + 2
                return smith.full_like(x, out)

        x = smith.tensor(12)
        y = smith.tensor(2)
        self.run_test(FullLikeModel(), (x, y))

    def test_l1_norm(self):
        class NormModel(smith.nn.Module):
            def forward(self, x):
                return smith.norm(x, p=1, dim=-1, keepdim=False)

        x = smith.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_l2_norm(self):
        class NormModel(smith.nn.Module):
            def forward(self, x):
                return smith.norm(x, p=2, dim=-2, keepdim=False)

        x = smith.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_frobenius_norm(self):
        class NormModel(smith.nn.Module):
            def forward(self, x):
                return smith.norm(x, p="fro", dim=0, keepdim=False)

        x = smith.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_frobenius_norm_keepdim(self):
        class NormModel(smith.nn.Module):
            def forward(self, x):
                return smith.norm(x, p="fro", dim=(0, 1), keepdim=True)

        x = smith.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_unfold(self):
        class UnfoldModel(smith.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=2, step=2)

        x = smith.randn(4, 2, 3, requires_grad=True)
        y = smith.randn(2, 1, 3, requires_grad=True)
        self.run_test(
            UnfoldModel(),
            x,
            dynamic_axes={"x": [0, 1]},
            input_names=["x"],
            additional_test_inputs=[y],
        )

    def test_unfold_infer_shape(self):
        class UnfoldModule(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(3, 1, 3, stride=2)

            @smith.jit.script_method
            def forward(self, x):
                x = self.conv(x)
                return x.unfold(dimension=2, size=2, step=2)

        x = smith.randn(32, 3, 64)
        self.run_test(UnfoldModule(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_unfold_dynamic_inputs(self):
        class UnfoldModel(smith.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=x.shape[1], step=x.shape[1] - 1)

        x = smith.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

        class UnfoldModel(smith.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=x.shape[1], step=1)

        x = smith.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)  # MatMul long inputs is added in ONNX opset 9.
    def test_mv(self):
        class MatmulModel(smith.nn.Module):
            def forward(self, input, other):
                return smith.mv(input, other)

        x = smith.randn(4, 5, requires_grad=True)
        y = smith.randn(5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = smith.randint(10, (4, 5))
        y = smith.randint(10, (5,))
        self.run_test(MatmulModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)  # MatMul long inputs is added in ONNX opset 9.
    def test_dot(self):
        class MatmulModel(smith.nn.Module):
            def forward(self, input, other):
                return smith.dot(input, other)

        x = smith.randn(5, requires_grad=True)
        y = smith.randn(5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = smith.randint(10, (5,))
        y = smith.randint(10, (5,))
        self.run_test(MatmulModel(), (x, y))

    @skipScriptTest()  # SpectralNorm not SmithScript compatible.
    def test_spectral_norm(self):
        m = smith.nn.utils.spectral_norm(smith.nn.Linear(2, 4))

        x = smith.randn(6, 2)
        self.run_test(m, (x,))

    def test_prelu(self):
        class PReluModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = smith.nn.PReLU()

            def forward(self, x):
                return self.prelu(x)

        x = smith.randn(2, 3, 4)
        y = smith.randn(2, 4, 5)
        self.run_test(
            PReluModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[y],
        )

    def test_prelu_scalar(self):
        x = smith.scalar_tensor(1.0)
        self.run_test(smith.nn.PReLU(), x, input_names=["x"])

    def test_relu6(self):
        class Relu6Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu6 = smith.nn.ReLU6()

            def forward(self, x):
                return self.relu6(x)

        x = smith.randn(2, 3, 4) * 100.0
        y = smith.randn(2, 4, 5) * 100.0
        self.run_test(
            Relu6Model(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[y],
        )

    def test_silu(self):
        class SiLUModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.silu = smith.nn.SiLU()

            def forward(self, x):
                return self.silu(x)

        x = smith.randn(2, 3, 4)
        self.run_test(SiLUModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_tril(self):
        class trilModel(smith.nn.Module):
            def forward(self, x):
                return smith.tril(x)

        x = smith.randn(2, 3, 4)
        self.run_test(trilModel(), (x))

        class trilModelwithDiagonal(smith.nn.Module):
            def forward(self, x):
                return smith.tril(x, diagonal=1)

        x = smith.randn(2, 3, 4)
        self.run_test(trilModelwithDiagonal(), (x))

        class trilModelwithNegDiagonal(smith.nn.Module):
            def forward(self, x):
                return smith.tril(x, diagonal=-1)

        x = smith.randn(2, 3, 4)
        self.run_test(trilModelwithNegDiagonal(), (x))

        class trilModelWithDiagonalInput(smith.nn.Module):
            def forward(self, x, diagnonal: int):
                return smith.tril(x, diagonal=diagnonal)

        x = smith.randn(2, 3, 4)
        self.run_test(trilModelWithDiagonalInput(), (x, 5))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_triu(self):
        class triuModel(smith.nn.Module):
            def forward(self, x):
                return smith.triu(x)

        x = smith.randn(2, 3, 4)
        self.run_test(triuModel(), (x))

        class triuModelwithDiagonal(smith.nn.Module):
            def forward(self, x):
                return smith.triu(x, diagonal=1)

        x = smith.randn(2, 3, 4)
        self.run_test(triuModelwithDiagonal(), (x))

        class triuModelwithNegDiagonal(smith.nn.Module):
            def forward(self, x):
                return smith.triu(x, diagonal=-1)

        x = smith.randn(2, 3, 4)
        self.run_test(triuModelwithNegDiagonal(), (x))

        class triuModelWithDiagonalInput(smith.nn.Module):
            def forward(self, x, diagnonal: int):
                return smith.triu(x, diagonal=diagnonal)

        x = smith.randn(2, 3, 4)
        self.run_test(triuModelWithDiagonalInput(), (x, 5))

    def test_mish(self):
        class MishModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mish = smith.nn.Mish()

            def forward(self, x):
                return self.mish(x)

        x = smith.randn(2, 3, 4)
        self.run_test(MishModel(), (x))

    def test_remainder(self):
        class RemainderModel(smith.nn.Module):
            def forward(self, input, other):
                return smith.remainder(input, other)

        x = smith.randn(4, 2, 3)
        y = smith.randn(1, 2, 1)
        self.run_test(RemainderModel(), (x, y))

        x = smith.tensor([7, 6, -7, -6], dtype=smith.long)
        y = smith.tensor([2], dtype=smith.long)
        self.run_test(RemainderModel(), (x, y))

        x = x.to(smith.float)
        self.run_test(RemainderModel(), (x, y))

        y = y.to(smith.float)
        self.run_test(RemainderModel(), (x, y))

        x = x.to(smith.int32)
        self.run_test(RemainderModel(), (x, y))

    def test_remainder_scalar(self):
        class RemainderModel(smith.nn.Module):
            def __init__(self, scalar=2.55):
                super().__init__()
                self.scalar = scalar

            def forward(self, input):
                return smith.remainder(input, self.scalar)

        x = smith.randint(10, (2, 3))
        self.run_test(RemainderModel(), x)

        x = smith.tensor([7, 6, -7, -6], dtype=smith.long)
        self.run_test(RemainderModel(2), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fmod(self):
        class FModModel(smith.nn.Module):
            def forward(self, input, other):
                return smith.fmod(input, other)

        x = smith.randn(4, 2, 3)
        y = smith.randn(1, 2, 1)
        self.run_test(FModModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fmod_scalar(self):
        class FModModel(smith.nn.Module):
            def forward(self, input):
                return smith.fmod(input, 2.55)

        x = smith.randint(10, (2, 3))
        self.run_test(FModModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_glu(self):
        class GluModel(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.glu(x)

        x = smith.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GluModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gelu(self):
        class GeluModel(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.gelu(x, approximate="none")

        x = smith.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GeluModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tanh_gelu(self):
        class GeluModel(smith.nn.Module):
            def forward(self, x):
                return smith.nn.functional.gelu(x, approximate="tanh")

        x = smith.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GeluModel(), x)

    def test_add_inplace(self):
        class InplaceAddModel(smith.nn.Module):
            def forward(self, x):
                x += 12
                return x

        x = smith.randn(4, 2, 3, requires_grad=True)
        self.run_test(InplaceAddModel(), x)

    def test_addcmul(self):
        class AddcmulModel(smith.nn.Module):
            def forward(self, x, t1, t2):
                return smith.addcmul(x, t1, t2), smith.addcmul(x, t1, t2, value=2.2)

        x = smith.randn(1, 3)
        t1 = smith.randn(3, 1)
        t2 = smith.randn(1, 3)
        self.run_test(AddcmulModel(), (x, t1, t2))

    def test_rsqrt(self):
        class RsqrtModel(smith.nn.Module):
            def forward(self, x):
                return x.rsqrt()

        x = smith.randn(4, 2, 3, requires_grad=True, dtype=smith.float64)
        self.run_test(RsqrtModel(), x)

    def test_rsqrt_zeros(self):
        class RsqrtModel(smith.nn.Module):
            def forward(self, x):
                return x.rsqrt()

        x = smith.zeros(4, 2, 3, requires_grad=True, dtype=smith.float64)
        self.run_test(RsqrtModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique(self):
        class UniqueModel(smith.nn.Module):
            def forward(self, x):
                return smith.unique(
                    x, sorted=True, return_inverse=False, return_counts=True
                )

        x = smith.tensor([1, 3, 2, 3], dtype=smith.long)
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique_along_dim(self):
        class UniqueModel(smith.nn.Module):
            def forward(self, x):
                return smith.unique(
                    x, dim=0, sorted=True, return_inverse=True, return_counts=False
                )

        x = smith.tensor([1, 3, 2, 3], dtype=smith.long)
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum(self):
        class CumSum(smith.nn.Module):
            def forward(self, input):
                return smith.cumsum(input, dim=0)

        x = smith.randn(2, 3, 4)
        model = CumSum()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum_with_cast(self):
        class CumSum(smith.nn.Module):
            def forward(self, input):
                return smith.cumsum(input, dim=0, dtype=smith.float32)

        model = CumSum()
        x = smith.tensor([2, 3, 4], dtype=smith.int32)
        self.run_test(model, x)
        x = smith.tensor([False, True, True])
        self.run_test(model, x)

    @skipScriptTest()  # error in propagate as assign input shape
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_embedding_bag(self):
        model = smith.nn.EmbeddingBag(10, 5, mode="sum", scale_grad_by_freq=True)
        input = smith.randint(10, (7,))
        offset = smith.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))

        model = smith.nn.EmbeddingBag(10, 5, mode="sum", include_last_offset=True)
        input = smith.randint(10, (7,))
        offset = smith.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))

        model = smith.nn.EmbeddingBag(10, 5, mode="max")
        input = smith.randint(10, (7, 5))
        self.run_test(model, (input))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_bag_1d_per_sample_weights(self):
        class EmbeddingModel(smith.nn.Module):
            def forward(self, embedding_matrix, input, offset, weights):
                return smith.nn.functional.embedding_bag(
                    input,
                    embedding_matrix,
                    offsets=offset,
                    mode="sum",
                    per_sample_weights=weights,
                )

        model = EmbeddingModel()
        x = smith.randint(7, (6,))
        w = smith.randn(
            6,
        )
        offset = smith.tensor([0, 2, 5])
        embedding_matrix = smith.rand(10, 15)
        self.run_test(model, (embedding_matrix, x, offset, w))

    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip(
        "This test is broken with ONNXRuntime(17): "
        "when running with onnxruntime 1.17.0 this test fails with the following error:"
        "FAIL : Non-zero status code returned while running If node. "
        "Name:'/If' Status Message: if.cc:253 Compute "
        "If nodes condition input must have exactly one element"
        "https://github.com/blacksmith/blacksmith/issues/119442"
    )
    def test_embedding_bag_2d_per_sample_weights(self):
        class EmbeddingModel(smith.nn.Module):
            def forward(self, embedding_matrix, input, weights):
                return smith.nn.functional.embedding_bag(
                    input, embedding_matrix, mode="sum", per_sample_weights=weights
                )

        embedding_matrix = smith.rand(10, 15)
        model = EmbeddingModel()
        x = smith.randint(7, (2, 3))
        w = smith.randn(2, 3)

        x2 = smith.randint(7, (4, 3))
        w2 = smith.randn(4, 3)
        self.run_test(
            model,
            (embedding_matrix, x, w),
            input_names=["embed", "x", "w"],
            dynamic_axes={"x": [0], "w": [0]},
            additional_test_inputs=[(embedding_matrix, x2, w2)],
        )

    @skipScriptTest()  # scripting prim::Uninitialized, prim::dtype, prim::unchecked_cast
    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip(
        "Due to ONNX Loop shape inference issue. "
        "https://msdata.visualstudio.com/Vienna/_workitems/edit/1352001"
    )
    def test_embedding_bag_dynamic_input(self):
        class EmbeddingModel1D(smith.nn.Module):
            def forward(self, embedding_matrix, input, weights, offsets):
                return smith.nn.functional.embedding_bag(
                    input,
                    embedding_matrix,
                    offsets=offsets,
                    mode="sum",
                    per_sample_weights=weights,
                )

        model = EmbeddingModel1D()
        x = smith.randint(7, (6,))
        w = smith.randn(
            6,
        )
        offsets = smith.tensor([0, 2, 5], dtype=smith.long)
        embedding_matrix = smith.rand(10, 15)
        x2 = smith.randint(7, (2,))
        w2 = smith.randn(
            2,
        )
        embedding_matrix2 = smith.rand(12, 25)
        offsets2 = smith.tensor(
            [
                0,
            ],
            dtype=smith.long,
        )
        self.run_test(
            model,
            (embedding_matrix, x, w, offsets),
            additional_test_inputs=[(embedding_matrix2, x2, w2, offsets2)],
            input_names=["embedding_matrix", "x", "offsets", "w"],
            dynamic_axes={
                "embedding_matrix": [0, 1],
                "x": [0],
                "offsets": [0],
                "w": [0],
            },
        )

        class EmbeddingModel2D(smith.nn.Module):
            def forward(self, embedding_matrix, input, weights):
                return smith.nn.functional.embedding_bag(
                    input, embedding_matrix, mode="sum", per_sample_weights=weights
                )

        model = EmbeddingModel2D()
        x = smith.randint(7, (2, 3))
        w = smith.randn(2, 3)
        embedding_matrix = smith.rand(10, 15)
        x2 = smith.randint(7, (3, 5))
        w2 = smith.randn(3, 5)
        embedding_matrix2 = smith.rand(12, 25)
        self.run_test(
            model,
            (embedding_matrix, x, w),
            additional_test_inputs=[(embedding_matrix2, x2, w2)],
            input_names=["embedding_matrix", "x", "w"],
            dynamic_axes={"embedding_matrix": [0, 1], "x": [0, 1], "w": [0, 1]},
        )

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid(self):
        class Meshgrid(smith.nn.Module):
            def forward(self, x, y, z):
                output1, output2, output3 = smith.meshgrid(x, y, z)
                return output1, output2, output3

        x = smith.randn(3, requires_grad=True)
        y = smith.zeros(4, requires_grad=True)
        z = smith.randn(5, requires_grad=True)
        self.run_test(Meshgrid(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid_indexing(self):
        class Meshgrid(smith.nn.Module):
            def __init__(self, indexing):
                super().__init__()
                self.indexing = indexing

            def forward(self, x, y, z):
                output1, output2, output3 = smith.meshgrid(
                    x, y, z, indexing=self.indexing
                )
                return output1, output2, output3

        x = smith.randn(5, requires_grad=True)
        y = smith.zeros(6, requires_grad=True)
        z = smith.randn(7, requires_grad=True)
        for indexing in ("xy", "ij"):
            self.run_test(Meshgrid(indexing), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid_scalar(self):
        class Meshgrid(smith.nn.Module):
            def forward(self, x, y, z):
                output1, output2, output3 = smith.meshgrid(x, y, z)
                return output1, output2, output3

        x = smith.ones(3, requires_grad=True)
        y = smith.zeros(4, requires_grad=True)
        z = smith.tensor(2.0)
        self.run_test(Meshgrid(), (x, y, z))

    def test_baddbmm(self):
        class MyModule(smith.nn.Module):
            def forward(self, input, batch1, batch2):
                return smith.baddbmm(
                    input, batch1, batch2, alpha=smith.tensor(5), beta=3.5
                )

        x = smith.randn(10, 3, 5)
        batch1 = smith.randn(10, 3, 4)
        batch2 = smith.randn(10, 4, 5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2))

    def test_baddbmm_dynamic(self):
        class MyModule(smith.nn.Module):
            def forward(self, input, batch1, batch2, alpha, beta):
                return smith.baddbmm(input, batch1, batch2, alpha=alpha, beta=beta)

        x = smith.randn(10, 3, 5)
        batch1 = smith.randn(10, 3, 4)
        batch2 = smith.randn(10, 4, 5)
        alpha = smith.tensor(5)
        beta = smith.tensor(3.5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2, alpha, beta))

    def test_numel(self):
        class MyModule(smith.nn.Module):
            def forward(self, input):
                return input.numel() * input

        x = smith.randn(2, 3, 5)
        x2 = smith.randn(4, 5, 6)
        model = MyModule()
        self.run_test(
            model,
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
            additional_test_inputs=[(x2,)],
        )

    def test_numel_empty(self):
        class MyModule(smith.nn.Module):
            def forward(self, input):
                return input.numel() * input

        x = smith.randn(0)
        x2 = smith.randn(4)
        model = MyModule()
        self.run_test(
            model,
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[(x2,)],
        )

    def test_dtype(self):
        class MyModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input, other):
                return input.to(dtype=other.dtype) + other

        x = smith.randn(2, 3)
        y = smith.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_dtype_eq(self):
        class MyModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input, other):
                if input.dtype == other.dtype:
                    return input + other
                return input

        x = smith.randn(2, 3)
        y = smith.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_cast_to(self):
        class MyModule(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input, other):
                return input.to(other) + other

        x = smith.randn(2, 3, 4)
        y = smith.tensor([1], dtype=smith.int64)
        model = MyModule()
        self.run_test(model, (x, y))

    def test_cast_to_bool(self):
        class MyModule(smith.nn.Module):
            def forward(self, input, other):
                return smith.cat((input.to(other), other), 0)

        x = smith.randn(2, 3, 4)
        y = smith.zeros([2, 3, 4], dtype=smith.bool)
        model = MyModule()
        self.run_test(model, (x, y))

    # ONNX supports bfloat16 for opsets >= 13
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_cast_type_as_with_bfloat16(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                y = smith.ones((3, 4), dtype=smith.bfloat16)
                x = x.type_as(y)
                return x.to(dtype=smith.float16)

        x = smith.ones(3, 4, dtype=smith.float16)
        model = MyModule()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_type_as(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                y = smith.tensor([1.0])
                return x.type_as(y)

        a = smith.tensor([True, False], dtype=smith.bool)
        b = smith.randn(3, 4, dtype=smith.double)
        c = smith.ones((2, 2), dtype=smith.int64)
        model = MyModule()
        self.run_test(model, a)
        self.run_test(model, b)
        self.run_test(model, c)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ones_bool(self):
        class MyModule(smith.nn.Module):
            def forward(self, input):
                true = smith.ones(input.shape, dtype=smith.bool)
                return input.to(true) & true

        x = smith.randn(2, 3, 4)
        model = MyModule()
        self.run_test(model, x)

    def test_log(self):
        class Log(smith.nn.Module):
            def forward(self, input):
                return smith.log(input)

        x = smith.rand(2, 3, 4)
        model = Log()
        self.run_test(model, x)

    def test_log1p(self):
        class Log1p(smith.nn.Module):
            def forward(self, input):
                return smith.log1p(input)

        x = smith.rand(2, 3, 4)
        model = Log1p()
        self.run_test(model, x)

    def test_log10(self):
        class Log10(smith.nn.Module):
            def forward(self, input):
                return smith.log10(input)

        x = smith.rand(2, 3, 4)
        model = Log10()
        self.run_test(model, x)

    def test_log2(self):
        class Log2(smith.nn.Module):
            def forward(self, input):
                return smith.log2(input)

        x = smith.tensor(1.0)
        model = Log2()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_round(self):
        class Round(smith.nn.Module):
            def forward(self, x):
                return smith.round(x)

        x = smith.tensor([0.9920, -1.0362, -1.5000, 3.5000], requires_grad=True)
        self.run_test(Round(), x)

        int_x = smith.tensor([9920, 1036, -1500, 35], dtype=smith.int32)
        self.run_test(Round(), int_x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_round_with_decimals(self):
        class Round(smith.nn.Module):
            def __init__(self, decimals):
                super().__init__()
                self.decimals = decimals

            def forward(self, x):
                return smith.round(x, decimals=self.decimals)

        x = smith.tensor([0.9920, -1234.0362, -1.58960, 3.5000])
        for decimals in (0, -2, 3):
            self.run_test(Round(decimals), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_default(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                return smith.stft(x, n_fft=n_fft, center=False, return_complex=False)

        x = smith.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_hop_length(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                hop_length = 4
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    hop_length=hop_length,
                    return_complex=False,
                )

        x = smith.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_non_divisible_hop_length(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                hop_length = 5
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    hop_length=hop_length,
                    return_complex=False,
                )

        x = smith.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_int_same_size(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                win_length = 16
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    win_length=win_length,
                    return_complex=False,
                )

        x = smith.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_int_different_size(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                win_length = 9
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    win_length=win_length,
                    return_complex=False,
                )

        x = smith.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_custom(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                window = smith.hann_window(16)
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    window=window,
                    return_complex=False,
                )

        x = smith.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_custom_window_size(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                window = smith.hann_window(10)
                return smith.stft(
                    x, n_fft=n_fft, window=window, center=False, return_complex=False
                )

        x = smith.randn((1, 32), requires_grad=True)
        with self.assertRaises((AssertionError, RuntimeError)):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_window_length(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                win_len = 17
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    win_length=win_len,
                    center=False,
                    return_complex=False,
                )

        x = smith.randn((1, 32), requires_grad=True)
        with self.assertRaises(RuntimeError):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_size_with_win_len(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                window = smith.hann_window(10)
                win_len = 10
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    window=window,
                    win_length=win_len,
                    center=False,
                    return_complex=False,
                )

        x = smith.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_one_dimension(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    return_complex=False,
                )

        x = smith.randn((32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_input_size(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                return smith.stft(x, n_fft=n_fft, center=False, return_complex=False)

        x = smith.randn((1, 1, 32), requires_grad=True)
        with self.assertRaises(RuntimeError):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_return_complex(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                return smith.stft(x, n_fft=n_fft, center=False, return_complex=True)

        x = smith.randn((1, 32), requires_grad=True)
        with self.assertRaises(errors.SymbolicValueError):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_normalize(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    normalized=True,
                    return_complex=False,
                )

        x = smith.randn((32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_not_onesided(self):
        class STFT(smith.nn.Module):
            def forward(self, x):
                n_fft = 16
                return smith.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    onesided=False,
                    return_complex=False,
                )

        x = smith.randn((32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_constant_pad(self):
        model = smith.nn.ConstantPad1d(2, 3.5)
        x = smith.randn(2, 4, 4)
        self.run_test(model, x)

        model = smith.nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        x = smith.randn(2, 2, 4, 4)
        self.run_test(model, x)

    @common_utils.parametrize(
        "pad",
        [
            common_utils.subtest([2, 4], name="scalar_list"),
            common_utils.subtest(
                [
                    smith.tensor(2, dtype=smith.int64),
                    smith.tensor(4, dtype=smith.int64),
                ],
                name="scalar_tensor_list",
            ),
        ],
    )
    @skipIfUnsupportedMinOpsetVersion(11)  # Dynamic padding is added in opset 11
    def test_pad_types(self, pad):
        # Test for different pad integer types
        class Pad(smith.nn.Module):
            def forward(self, x, pad: list[int]):
                return smith.nn.functional.pad(x, pad)

        x = smith.randn(2, 2, 4, 4)
        self.run_test(Pad(), (x, pad))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_circular(self):
        class PadModel(smith.nn.Module):
            def forward(self, x):
                out = smith.nn.functional.pad(x, (1, 2, 1, 2), mode="circular")
                return out

        x = smith.randn(2, 3, 3, 4)
        self.run_test(PadModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_circular_negative(self):
        # Test for different pad integer types
        class PadModel(smith.nn.Module):
            def forward(self, x):
                out = smith.nn.functional.pad(x, (-1, -2), mode="circular")
                return out

        x = smith.randn(2, 3, 6)
        self.run_test(PadModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_circular_dynamic_axes(self):
        class PadModel(smith.nn.Module):
            def forward(self, x):
                out = smith.nn.functional.pad(x, (2, 1, 2, 1), mode="circular")
                return out

        x = smith.randn(4, 3, 5, 6)
        self.run_test(
            PadModel(),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

    @skipIfUnsupportedMaxOpsetVersion(10)
    @skipScriptTest()  # TODO: the logic in symbolic_opset9 doesn't handle script
    def test_unsupported_pad(self):
        class Pad(smith.nn.Module):
            def forward(self, x, pad: list[int]):
                return smith.nn.functional.pad(x, pad)

        x = smith.randn(2, 2, 4, 4)
        y = [2, 4]

        with self.assertRaisesRegex(
            RuntimeError,
            (
                "Unsupported: ONNX export of Pad.*"
                + "The sizes of the padding must be constant"
            ),
        ):
            self.run_test(Pad(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_if_fold(self):
        class IfFoldModel(smith.nn.Module):
            def forward(self, y):
                if y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    y = y - 1
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(smith.nn.Module):
            def forward(self, y):
                if y.numel() > 1:
                    y = y + 4
                else:
                    y = y + 2
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(smith.nn.Module):
            def forward(self, y):
                if y.dim() != 3:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(smith.nn.Module):
            def forward(self, y):
                if y.dim() >= 1:
                    y = y + 4
                else:
                    y = y - 1
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(smith.nn.Module):
            def forward(self, y):
                if y.dim() <= 1:
                    y = y + 4
                else:
                    y = y + 2
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(smith.nn.Module):
            def forward(self, y):
                if y.dim() < 3 and y.dtype == smith.int:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(smith.nn.Module):
            def forward(self, y):
                if y.dim() == 3 and y.dtype == smith.int:
                    y = y + 4
                    y = y + 2
                else:
                    y = y + 1
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(smith.nn.Module):
            def forward(self, y):
                if y.numel() != 0 and y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(smith.nn.Module):
            def forward(self, x, y):
                if x.numel() == y.numel():
                    y = x + y
                else:
                    y = y - x
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        y = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), (x, y))

        class IfFoldModel(smith.nn.Module):
            def forward(self, x, y):
                if x.numel() != y.numel():
                    y = x + y
                else:
                    y = y - x
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        y = smith.ones((3, 4), dtype=smith.int)
        self.run_test(IfFoldModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_uninitialized(self):
        class UninitializedModel(smith.nn.Module):
            def forward(self, y):
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(UninitializedModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_uninitialized_dynamic(self):
        class UninitializedModel(smith.nn.Module):
            def forward(self, y):
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        y = smith.ones((6, 7), dtype=smith.int)
        self.run_test(
            UninitializedModel(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # onnx::Identity of sequence supported for ONNX opset >= 14
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList(self):
        class UninitializedTensorListModel(smith.nn.Module):
            def forward(self, x):
                if x[0].shape[0] < 5:
                    if x.size(0) == 1:
                        x = x + 4
                    else:
                        return [x]
                return [x]

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(smith.jit.script(UninitializedTensorListModel()), x)

    # onnx::Identity of sequence supported for ONNX opset >= 14
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList_dynamic(self):
        class UninitializedTensorListModel(smith.nn.Module):
            def forward(self, x):
                if x[0].shape[0] < 5:
                    if x.size(0) == 1:
                        x += x
                    else:
                        return list(x)
                return list(x)

        x = smith.ones((3, 4), dtype=smith.double)
        self.run_test(
            smith.jit.script(UninitializedTensorListModel()),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # onnx::Identity of sequence supported for ONNX opset >= 14
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_intList(self):
        class UninitializedListModel(smith.nn.Module):
            def forward(self, x):
                y = list(range(x.size(0)))
                if y[0] < 5:
                    # if x.size(0) != 3, ORT will throw type error.
                    if x.size(0) == 3:
                        y.append(10)
                    else:
                        return y
                return y

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(
            smith.jit.script(UninitializedListModel()),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # onnx::Identity of sequence supported for ONNX opset >= 14
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList_shape(self):
        class UninitializedModel(smith.nn.Module):
            def forward(self, x):
                if x.shape[1] < 5:
                    if x.size(0) == 1:
                        x = x + 4
                    else:
                        x_list = list(x)
                        x_list.append(x)
                        return x_list
                return [x, x]

        x = smith.ones((3, 4), dtype=smith.int)
        y = smith.ones((4, 6), dtype=smith.int)
        self.run_test(
            smith.jit.script(UninitializedModel()),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # Sequence type as loop-carried dependencies only supported for ONNX opset >= 13
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequance_loopcarried(self):
        class SequanceLoopModel(smith.nn.Module):
            def forward(self, x):
                outputs = []
                for _ in range(3):
                    outputs += [x]
                return smith.stack(outputs).transpose(0, 1)

        x = smith.ones((3, 4), dtype=smith.int)
        self.run_test(smith.jit.script(SequanceLoopModel()), x)

    def test_reflection_pad(self):
        model = smith.nn.ReflectionPad1d(2)
        x = smith.randn(2, 4, 4)
        self.run_test(model, x)

        model = smith.nn.ReflectionPad2d((3, 0, 2, 1))
        x = smith.randn(2, 2, 4, 4)
        self.run_test(model, x)

    def test_replication_pad(self):
        model = smith.nn.ReplicationPad1d(2)
        x = smith.randn(2, 4, 4)
        self.run_test(model, x)

        model = smith.nn.ReplicationPad2d((3, 0, 2, 1))
        x = smith.randn(2, 2, 4, 4)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_im2col(self):
        class Unfold(smith.nn.Module):
            def forward(self, input):
                return (
                    smith.nn.functional.unfold(
                        input, kernel_size=(10, 15), dilation=2, padding=5, stride=3
                    ),
                    smith.nn.functional.unfold(
                        input, kernel_size=(2, 2), dilation=1, padding=0, stride=3
                    ),
                    smith.nn.functional.unfold(
                        input, kernel_size=(1, 1), dilation=5, padding=2, stride=3
                    ),
                )

        x = smith.rand(1, 1, 200, 100)
        self.run_test(Unfold(), x)

    @skipIfNoLapack
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_det(self):
        class Det(smith.nn.Module):
            def forward(self, x):
                return smith.linalg.det(x)

        x = smith.randn(2, 3, 5, 5)
        self.run_test(Det(), x)

    def test_linalg_norm(self):
        class LinalgSingleDimModel(smith.nn.Module):
            def __init__(self, ord_val):
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                return smith.linalg.norm(x, ord=self.ord, dim=1)

        x = smith.randn(2, 3, 5, 5)
        self.run_test(LinalgSingleDimModel(None), x)
        self.run_test(LinalgSingleDimModel(2), x)
        self.run_test(LinalgSingleDimModel(float("inf")), x)
        self.run_test(LinalgSingleDimModel(-float("inf")), x)
        self.run_test(LinalgSingleDimModel(-4), x)
        self.run_test(LinalgSingleDimModel(1.5), x)

        class LinalgMultiDimModel(smith.nn.Module):
            def __init__(self, ord_val):
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                return smith.linalg.norm(x, ord=self.ord, dim=(0, 2))

        x = smith.randn(2, 3, 5, 5)
        self.run_test(LinalgMultiDimModel("fro"), x)
        self.run_test(LinalgMultiDimModel(float("inf")), x)
        self.run_test(LinalgMultiDimModel(-float("inf")), x)
        self.run_test(LinalgMultiDimModel(1), x)
        self.run_test(LinalgMultiDimModel(-1), x)

        class LinalgNoDimNoOrdModel(smith.nn.Module):
            def forward(self, x):
                return smith.linalg.norm(x)

        x = smith.randn(2, 3, 5, 5)
        self.run_test(LinalgNoDimNoOrdModel(), x)
        y = smith.randn(2, 3)
        self.run_test(LinalgNoDimNoOrdModel(), y)
        z = smith.randn(2)
        self.run_test(LinalgNoDimNoOrdModel(), z)

        class LinalgNoDim1DModel(smith.nn.Module):
            def __init__(self, ord_val):
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                return smith.linalg.norm(x, ord=self.ord)

        x = smith.randn(2)
        self.run_test(LinalgNoDim1DModel(None), x)
        self.run_test(LinalgNoDim1DModel(2), x)
        self.run_test(LinalgNoDim1DModel(float("inf")), x)
        self.run_test(LinalgNoDim1DModel(-float("inf")), x)
        self.run_test(LinalgNoDim1DModel(-4), x)
        self.run_test(LinalgNoDim1DModel(1.5), x)

        class LinalgNoDim2DModel(smith.nn.Module):
            def __init__(self, ord_val):
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                return smith.linalg.norm(x, ord=self.ord)

        x = smith.randn(2, 3)
        self.run_test(LinalgNoDim2DModel("fro"), x)
        self.run_test(LinalgNoDim2DModel(float("inf")), x)
        self.run_test(LinalgNoDim2DModel(-float("inf")), x)
        self.run_test(LinalgNoDim2DModel(1), x)
        self.run_test(LinalgNoDim2DModel(-1), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_linalg_vector_norm_zero(self):
        class LinalgVectorNormModel(smith.nn.Module):
            def __init__(self, ord_val):
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                return smith.linalg.vector_norm(x, ord=self.ord)

        x = smith.randn(2, 3, 5, 5)
        self.run_test(LinalgVectorNormModel(0), x)

    def test_linalg_vector_norm(self):
        class LinalgVectorNormModel(smith.nn.Module):
            def __init__(self, ord_val, dim_info):
                super().__init__()
                self.ord = ord_val
                self.dim, self.keepdim = dim_info

            def forward(self, x):
                return smith.linalg.vector_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
                )

        x = smith.randn(2, 3, 5, 5)
        ord_options = [2, float("inf"), -float("inf"), -4, 1.5]
        dim_options = [(None, False), (1, False), ((1, 2), False), ((1, 2), True)]
        for ord_val in ord_options:
            for dim_info in dim_options:
                self.run_test(LinalgVectorNormModel(ord_val, dim_info), x)

    def test_linalg_matrix_norm(self):
        class LinalgMatrixNormModel(smith.nn.Module):
            def __init__(self, ord_val, dim_val=(-2, -1), keepdim_val=False):
                super().__init__()
                self.ord = ord_val
                self.dim = dim_val
                self.keepdim = keepdim_val

            def forward(self, x):
                return smith.linalg.matrix_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
                )

        x = smith.randn(2, 3, 5, 5)
        ord_options = ["fro", float("inf"), -float("inf"), 1, -1]
        for ord_val in ord_options:
            self.run_test(LinalgMatrixNormModel(ord_val), x)
            self.run_test(LinalgMatrixNormModel(ord_val, (0, 2)), x)
            self.run_test(LinalgMatrixNormModel(ord_val, (0, 2), True), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linalg_cross(self):
        class Cross(smith.nn.Module):
            def forward(self, x, y):
                return smith.linalg.cross(x, y, dim=1), smith.linalg.cross(x, y)

        x = smith.randn(5, 3, 2, 3)
        y = smith.randn(1, 3, 1, 3)
        self.run_test(Cross(), input_args=(x, y))

    # This test checks output scalar type in the ONNX graph should not be null
    # https://github.com/blacksmith/blacksmith/issues/28607
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_trace_script(self):
        @smith.jit.script
        def center_slice_helper(input, h_offset):
            return input[:, h_offset:]

        class CenterCrop(smith.nn.Module):
            def forward(self, input):
                return center_slice_helper(input, smith.tensor(input.shape[1] - 1))

        x = smith.randn(3, 4)
        self.run_test(CenterCrop(), x)

    @skipIfNoLapack
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_logdet(self):
        class LogDet(smith.nn.Module):
            def forward(self, x):
                return smith.logdet(x)

        x = smith.randn(2, 3, 5, 5)
        self.run_test(LogDet(), x)

    def test_dim(self):
        class DimModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                out = input * 2
                out *= out.dim()
                return out

        empty_input = smith.randn(0, requires_grad=True)
        multi_dim_input = smith.randn(1, 2, 3, requires_grad=True)
        self.run_test(DimModel(), empty_input)
        self.run_test(DimModel(), multi_dim_input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dim_1(self):
        class M(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, poses):
                boxes = smith.zeros([poses.shape[0], 2, 4])
                batch_boxes = []
                for kp_boxes in boxes:
                    kp_boxes = smithvision.ops.clip_boxes_to_image(kp_boxes, (2, 3))
                    batch_boxes.append(kp_boxes)
                return batch_boxes

        dummy_inputs = smith.rand(2, 2, 3)
        self.run_test(M(), (dummy_inputs,), input_names=["x"], dynamic_axes={"x": [0]})

    @skipIfUnsupportedMinOpsetVersion(12)
    @skipDtypeChecking
    def test_outer(self):
        class Outer(smith.nn.Module):
            def forward(self, x, y):
                return smith.outer(x, y)

        x = smith.arange(1, 5)
        y = smith.arange(1, 4)
        self.run_test(Outer(), input_args=(x, y))

        x = smith.arange(1, 6).to(dtype=smith.float32)
        y = smith.arange(1, 4).to(dtype=smith.long)
        self.run_test(Outer(), input_args=(x, y))

        x = smith.arange(2, 5).to(dtype=smith.float32)
        y = smith.arange(2, 4).to(dtype=smith.float64)
        self.run_test(Outer(), input_args=(x, y))

        x = smith.arange(3, 6).to(dtype=smith.int32)
        y = smith.arange(4, 7).to(dtype=smith.long)
        self.run_test(Outer(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_movedim(self):
        class MovedimModel(smith.nn.Module):
            def forward(self, x):
                return (
                    x.movedim(1, 3),
                    x.movedim(2, 0),
                    x.movedim(1, 1),
                    x.movedim((1, 2, 3), (3, 0, 1)),
                    x.movedim((0, 1, 2), (1, 2, 3)),
                    x.movedim((1, 3, 2), (1, 3, 2)),
                )

        x = smith.randn(5, 3, 4, 2)

        self.run_test(MovedimModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_moveaxis(self):
        # moveaxis is an alias of movedim; thus, mostly copied from `test_movedim`.
        class MoveaxisModel(smith.nn.Module):
            def forward(self, x):
                return (
                    x.moveaxis(1, 3),
                    x.moveaxis(2, 0),
                    x.moveaxis(1, 1),
                    x.moveaxis((1, 2, 3), (3, 0, 1)),
                    x.moveaxis((0, 1, 2), (1, 2, 3)),
                    x.moveaxis((1, 3, 2), (1, 3, 2)),
                )

        x = smith.randn(5, 3, 4, 2)

        self.run_test(MoveaxisModel(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_einsum(self):
        class EinsumModelBatchDiagonal(smith.nn.Module):
            def forward(self, x):
                eqn = "...ii ->...i"
                return smith.einsum(eqn, x)

        for x in [smith.randn(3, 5, 5), smith.randn(3, 5, 5).to(dtype=smith.bool)]:
            self.run_test(EinsumModelBatchDiagonal(), input_args=(x,))

        class EinsumModelBatchMatmul(smith.nn.Module):
            def forward(self, x, y):
                eqn = "bij, bjk -> bik"
                return smith.einsum(eqn, x, y)

        x = smith.randn(5, 2, 3)
        y = smith.randn(5, 3, 4)
        self.run_test(EinsumModelBatchMatmul(), input_args=(x, y))

        class EinsumModelInnerProd(smith.nn.Module):
            def forward(self, x, y):
                eqn = "i,i"
                return smith.einsum(eqn, x, y)

        x = smith.randn(5)
        y = smith.randn(5)
        self.run_test(EinsumModelInnerProd(), input_args=(x, y))

        class EinsumModelTranspose(smith.nn.Module):
            def forward(self, x):
                eqn = "ij->ji"
                return smith.einsum(eqn, x)

        for x in [smith.randn(3, 4), smith.randn(3, 4).to(dtype=smith.bool)]:
            self.run_test(EinsumModelTranspose(), input_args=(x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cosine_similarity(self):
        x = smith.randn(5, 3, 2)
        y = smith.randn(5, 3, 2)
        self.run_test(smith.nn.CosineSimilarity(dim=2), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pairwise_distance(self):
        x = smith.randn(5, 3, 2)
        y = smith.randn(5, 3, 2)
        self.run_test(smith.nn.PairwiseDistance(p=2.0), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cross(self):
        class Cross(smith.nn.Module):
            def forward(self, x, y):
                return smith.cross(x, y, dim=3), smith.cross(x, y)

        x = smith.randn(5, 3, 2, 3)
        y = smith.randn(5, 3, 2, 3)
        self.run_test(Cross(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist(self):
        class Cdist(smith.nn.Module):
            def forward(self, x, y):
                return smith.cdist(x, y)

        x = smith.randn(5, 3, 3)
        y = smith.randn(5, 2, 3)
        self.run_test(Cdist(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist_euclid_dist(self):
        class Cdist(smith.nn.Module):
            def forward(self, x, y):
                return smith.cdist(x, y, p=2.0, compute_mode="use_mm_for_euclid_dist")

        x = smith.randn(2, 64, 4)
        y = smith.randn(1, 32, 4)
        self.run_test(Cdist(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist_euclid_dist_if_necessary(self):
        class Cdist(smith.nn.Module):
            def forward(self, x, y):
                return smith.cdist(
                    x, y, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"
                )

        x = smith.randn(2, 64, 4)
        y = smith.randn(1, 32, 4)
        self.run_test(Cdist(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist_no_euclid_dist(self):
        class Cdist(smith.nn.Module):
            def forward(self, x, y):
                return smith.cdist(
                    x, y, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
                )

        x = smith.randn(2, 64, 4)
        y = smith.randn(1, 32, 4)
        self.run_test(Cdist(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist_dynamic_axes_script(self):
        class CdistModule(smith.nn.Module):
            def forward(self, a, b):
                return smith.cdist(a, b)

        model = smith.jit.script(CdistModule())
        a = smith.randn(10, 3)
        b = smith.randn(10, 3)
        self.run_test(
            model,
            input_args=(a, b),
            input_names=["a", "b"],
            output_names=["out"],
            dynamic_axes={"a": {0: "size"}, "b": {0: "size"}},
        )

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_crossentropyloss(self):
        for ignore_index in [-100, 1]:
            x = smith.randn(3, 5)
            y = smith.empty(3, dtype=smith.long).random_(5)
            y[y == 1] = ignore_index

            self._crossentropyloss(x, y, ignore_index)

            x = smith.randn(3, 5, 2)
            y = smith.empty(3, 2, dtype=smith.long).random_(5)
            y[y == 1] = ignore_index
            self._crossentropyloss(x, y, ignore_index)

            x = smith.randn(3, 5, 2, 7)
            y = smith.empty(3, 2, 7, dtype=smith.long).random_(5)
            y[y == 1] = ignore_index
            self._crossentropyloss(x, y, ignore_index)

    def _crossentropyloss(self, x, y, ignore_index):
        class CrossEntropyLossNone(smith.nn.Module):
            def __init__(self, ignore_index):
                super().__init__()
                if ignore_index == -100:
                    self.loss = smith.nn.CrossEntropyLoss(reduction="none")
                else:
                    self.loss = smith.nn.CrossEntropyLoss(
                        reduction="none", ignore_index=ignore_index
                    )

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossNone(ignore_index), input_args=(x, y))

        class CrossEntropyLossNoneWeight(smith.nn.Module):
            def __init__(self, ignore_index):
                super().__init__()
                if ignore_index == -100:
                    self.loss = smith.nn.CrossEntropyLoss(
                        reduction="none", weight=smith.randn(5)
                    )
                else:
                    self.loss = smith.nn.CrossEntropyLoss(
                        reduction="none",
                        weight=smith.randn(5),
                        ignore_index=ignore_index,
                    )

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossNoneWeight(ignore_index), input_args=(x, y))

        class CrossEntropyLossSum(smith.nn.Module):
            def __init__(self, ignore_index):
                super().__init__()
                if ignore_index == -100:
                    self.loss = smith.nn.CrossEntropyLoss(reduction="sum")
                else:
                    self.loss = smith.nn.CrossEntropyLoss(
                        reduction="sum", ignore_index=ignore_index
                    )

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossSum(ignore_index), input_args=(x, y))

        class CrossEntropyLossSumWeight(smith.nn.Module):
            def __init__(self, ignore_index):
                super().__init__()
                if ignore_index == -100:
                    self.loss = smith.nn.CrossEntropyLoss(
                        reduction="sum", weight=smith.randn(5)
                    )
                else:
                    self.loss = smith.nn.CrossEntropyLoss(
                        reduction="sum",
                        weight=smith.randn(5),
                        ignore_index=ignore_index,
                    )

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossSumWeight(ignore_index), input_args=(x, y))

        class CrossEntropyLossMean(smith.nn.Module):
            def __init__(self, ignore_index):
                super().__init__()
                if ignore_index == -100:
                    self.loss = smith.nn.CrossEntropyLoss()
                else:
                    self.loss = smith.nn.CrossEntropyLoss(ignore_index=ignore_index)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossMean(ignore_index), input_args=(x, y))

        class CrossEntropyLossMeanWeight(smith.nn.Module):
            def __init__(self, ignore_index):
                super().__init__()
                if ignore_index == -100:
                    self.loss = smith.nn.CrossEntropyLoss(weight=smith.randn(5))
                else:
                    self.loss = smith.nn.CrossEntropyLoss(
                        weight=smith.randn(5), ignore_index=ignore_index
                    )

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossMeanWeight(ignore_index), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_MSELoss(self):
        class MSELoss(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss1 = smith.nn.MSELoss(reduction="none")
                self.loss2 = smith.nn.MSELoss(reduction="sum")
                self.loss3 = smith.nn.MSELoss(reduction="mean")

            def forward(self, input, target):
                return (
                    self.loss1(input, target),
                    self.loss2(input, target),
                    self.loss3(input, target),
                )

        x = smith.randn(2, 3, 5)
        y = smith.randn(2, 3, 5)
        self.run_test(MSELoss(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_kldiv_loss(self):
        x = smith.rand(5).log()
        y = smith.rand(5)
        self._kldiv_loss(x, y)

        x = smith.rand(2, 3, 5).log()
        y = smith.rand(2, 3, 5)
        self._kldiv_loss(x, y)

        x = smith.rand(2, 3, 5, 7).log()
        y = smith.rand(2, 3, 5, 7)
        self._kldiv_loss(x, y)

    def _kldiv_loss(self, x, y):
        class KLDivLossNone(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.KLDivLoss(reduction="none", log_target=True)

            def forward(self, input, target):
                return self.loss(input, target.log())

        self.run_test(KLDivLossNone(), input_args=(x, y))

        class KLDivLossMean(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.KLDivLoss(reduction="mean", log_target=False)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(KLDivLossMean(), input_args=(x, y))

        class KLDivLossSum(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.KLDivLoss(reduction="sum", log_target=True)

            def forward(self, input, target):
                return self.loss(input, target.log())

        self.run_test(KLDivLossSum(), input_args=(x, y))

        class KLDivLossBatchMean(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.KLDivLoss(reduction="batchmean", log_target=False)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(KLDivLossBatchMean(), input_args=(x, y))

        class KLDivLossMiniBatchMean(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.KLDivLoss(
                    reduction="batchmean", size_average=False, log_target=True
                )

            def forward(self, input, target):
                return self.loss(input, target.log())

        self.run_test(KLDivLossMiniBatchMean(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss(self):
        class NLLModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.NLLLoss(reduction="none")
                self.m = smith.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(2 * input), target)
                return output

        N, C = 5, 4
        input = smith.randn(N, 16)
        target = smith.empty(N, dtype=smith.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_none(self):
        class NLLModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.NLLLoss(reduction="none")
                self.conv = smith.nn.Conv2d(16, C, (3, 3))
                self.m = smith.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = smith.randn(N, 16, 10, 10)
        target = smith.empty(N, 8, 8, dtype=smith.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean(self):
        class NLLModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.NLLLoss(reduction="mean")
                self.conv = smith.nn.Conv2d(16, C, (3, 3))
                self.m = smith.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = smith.randn(N, 16, 10, 10)
        target = smith.empty(N, 8, 8, dtype=smith.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_sum(self):
        class NLLModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.NLLLoss(reduction="sum")
                self.conv = smith.nn.Conv2d(16, C, (3, 3))
                self.m = smith.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = smith.randn(N, 16, 10, 10)
        target = smith.empty(N, 8, 8, dtype=smith.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_weights(self):
        class NLLModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.NLLLoss(reduction="mean", weight=smith.randn(C))
                self.conv = smith.nn.Conv2d(16, C, (3, 3))
                self.m = smith.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = smith.randn(N, 16, 10, 10)
        target = smith.empty(N, 8, 8, dtype=smith.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_ignore_index(self):
        class NLLModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.NLLLoss(reduction="mean", ignore_index=1)
                self.conv = smith.nn.Conv2d(16, C, (3, 3))
                self.m = smith.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = smith.randn(N, 16, 10, 10)
        target = smith.empty(N, 8, 8, dtype=smith.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_dynamic_ignore_index(self):
        import smith.nn.functional as F

        def linear_combination(x, y, epsilon):
            return epsilon * x + (1 - epsilon) * y

        def reduce_loss(loss, reduction="mean"):
            return (
                loss.mean()
                if reduction == "mean"
                else loss.sum()
                if reduction == "sum"
                else loss
            )

        class LabelSmoothingCrossEntropy(smith.nn.Module):
            def __init__(self, epsilon: float = 0.1, reduction="mean"):
                super().__init__()
                self.epsilon = epsilon
                self.reduction = reduction

            def forward(self, preds, target, start_position):
                n = preds.size()[-1]
                log_preds = F.log_softmax(preds, dim=-1)
                ignore_index = start_position.size(1)
                nll = F.nll_loss(
                    log_preds,
                    target,
                    reduction=self.reduction,
                    ignore_index=ignore_index,
                )
                return nll + start_position.float()

        N = 5
        preds = smith.randn(N, 16)
        target = smith.randint(5, (N,))
        start_position = smith.randint(10, (N, N))
        self.run_test(LabelSmoothingCrossEntropy(), (preds, target, start_position))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_ignore_index_weights(self):
        class NLLModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = smith.nn.NLLLoss(
                    reduction="mean", weight=smith.randn(C), ignore_index=1
                )
                self.conv = smith.nn.Conv2d(16, C, (3, 3))
                self.m = smith.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = smith.randn(N, 16, 10, 10)
        target = smith.empty(N, 8, 8, dtype=smith.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_binary_cross_entropy_with_logits(self):
        x = smith.randn(5)
        y = smith.empty(5).random_(2)
        self._bce_logits(x, y)

        x = smith.randn(3, 4)
        y = smith.empty(3, 4).random_(2)
        weight = smith.tensor([3])
        self._bce_logits_wegiht(x, y, weight)

        x = smith.randn(3, 2, 4)
        y = smith.empty(3, 2, 4).random_(2)
        pos_weight = smith.empty([2, 4]).random_(2)
        self._bce_logits_posweight(x, y, pos_weight)

        x = smith.randn(3, 3, 4)
        y = smith.empty(3, 3, 4).random_(2)
        weight = smith.tensor([3])
        pos_weight = smith.empty([3, 4]).random_(2)
        self._bce_logits_loss_weight_posweight(x, y, weight, pos_weight)

    def _bce_logits(self, x, y):
        class BCEWithLogitsLossNone(smith.nn.Module):
            def forward(self, input, target):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, reduction="none"
                )

        self.run_test(BCEWithLogitsLossNone(), input_args=(x, y))

        class BCEWithLogitsLossMean(smith.nn.Module):
            def forward(self, input, target):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, reduction="mean"
                )

        self.run_test(BCEWithLogitsLossMean(), input_args=(x, y))

        class BCEWithLogitsLossSum(smith.nn.Module):
            def forward(self, input, target):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, reduction="sum"
                )

        self.run_test(BCEWithLogitsLossSum(), input_args=(x, y))

    def _bce_logits_wegiht(self, x, y, weight):
        class BCEWithLogitsLossWegihtNone(smith.nn.Module):
            def forward(self, input, target, weight):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, reduction="none"
                )

        self.run_test(BCEWithLogitsLossWegihtNone(), input_args=(x, y, weight))

        class BCEWithLogitsLossWegihtMean(smith.nn.Module):
            def forward(self, input, target, weight):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, reduction="mean"
                )

        self.run_test(BCEWithLogitsLossWegihtMean(), input_args=(x, y, weight))

        class BCEWithLogitsLossWegihtSum(smith.nn.Module):
            def forward(self, input, target, weight):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, reduction="sum"
                )

        self.run_test(BCEWithLogitsLossWegihtSum(), input_args=(x, y, weight))

    def _bce_logits_posweight(self, x, y, pos_weight):
        class BCEWithLogitsLossPosWegihtNone(smith.nn.Module):
            def forward(self, input, target, pos_weight):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, pos_weight=pos_weight, reduction="none"
                )

        self.run_test(BCEWithLogitsLossPosWegihtNone(), input_args=(x, y, pos_weight))

        class BCEWithLogitsLossPosWegihtMean(smith.nn.Module):
            def forward(self, input, target, pos_weight):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, pos_weight=pos_weight, reduction="mean"
                )

        self.run_test(BCEWithLogitsLossPosWegihtMean(), input_args=(x, y, pos_weight))

        class BCEWithLogitsLossPosWegihtSum(smith.nn.Module):
            def forward(self, input, target, pos_weight):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, pos_weight=pos_weight, reduction="sum"
                )

        self.run_test(BCEWithLogitsLossPosWegihtSum(), input_args=(x, y, pos_weight))

    def _bce_logits_loss_weight_posweight(self, x, y, weight, pos_weight):
        class BCEWithLogitsLossWeightPosweightNone(smith.nn.Module):
            def forward(self, input, target, weight, pos_weight):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input,
                    target,
                    weight=weight,
                    pos_weight=pos_weight,
                    reduction="none",
                )

        self.run_test(
            BCEWithLogitsLossWeightPosweightNone(),
            input_args=(x, y, weight, pos_weight),
        )

        class BCEWithLogitsLossWeightPosweightMean(smith.nn.Module):
            def forward(self, input, target, weight, pos_weight):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input,
                    target,
                    weight=weight,
                    pos_weight=pos_weight,
                    reduction="mean",
                )

        self.run_test(
            BCEWithLogitsLossWeightPosweightMean(),
            input_args=(x, y, weight, pos_weight),
        )

        class BCEWithLogitsLossWeightPosweightSum(smith.nn.Module):
            def forward(self, input, target, weight, pos_weight):
                return smith.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, pos_weight=pos_weight, reduction="sum"
                )

        self.run_test(
            BCEWithLogitsLossWeightPosweightSum(), input_args=(x, y, weight, pos_weight)
        )

    def test_smith_mm(self):
        class M(smith.nn.Module):
            def forward(self, mat1, mat2):
                mm = smith.mm(mat1, mat2)
                return mm

        mat1 = smith.randn(2, 3)
        mat2 = smith.randn(3, 3)
        self.run_test(M(), input_args=(mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because where op is not supported for opset < 9.
    def test_where_with_bool_tensor(self):
        class M(smith.nn.Module):
            def forward(self, mat1, mat2):
                out = smith.where(mat1 > 0, mat1, mat2)
                return out

        mat1 = smith.randn(2, 3)
        mat2 = smith.ones(2, 3)
        self.run_test(M(), input_args=(mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because where op is not supported for opset < 9.
    def test_where_with_byte_tensor(self):
        class M(smith.nn.Module):
            def forward(self, cond, mat1, mat2):
                out = smith.where(cond, mat1, mat2)
                return out

        cond = smith.ones(2, 3, dtype=smith.uint8)
        cond[1, 2] = 0
        mat1 = smith.randn(2, 3)
        mat2 = smith.ones(2, 3)
        self.run_test(M(), input_args=(cond, mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(10)  # ONNX IsInf op is added in opset 10.
    def test_isinf(self):
        class M(smith.nn.Module):
            def forward(self, x):
                return x.isinf()

        x = smith.tensor([[1, 2, float("inf")], [2, float("nan"), float("inf")]])
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_isfinite(self):
        class M(smith.nn.Module):
            def forward(self, x):
                return x.isfinite()

        x = smith.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)  # ONNX IsNaN op is added in opset 9.
    def test_isnan(self):
        class M(smith.nn.Module):
            def forward(self, x):
                return x.isnan()

        x = smith.tensor([[1, 2, float("inf")], [2, float("nan"), float("inf")]])
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(
        10
    )  # ONNX IsNaN, IsInf op is added in opset 9, 10 respectively.
    def test_nan_to_num(self):
        class NoParams(smith.nn.Module):
            def forward(self, x):
                return x.nan_to_num()

        x = smith.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        xint = smith.ones((2, 4), dtype=smith.int)
        xhalf = smith.ones((2, 4), dtype=smith.half)
        self.run_test(NoParams(), (x,))
        self.run_test(NoParams(), (xint,))
        self.run_test(NoParams(), (xhalf,))

        class WithParams(smith.nn.Module):
            def forward(self, x):
                return x.nan_to_num(nan=2.3, posinf=4.5, neginf=6.7)

        x = smith.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        self.run_test(WithParams(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_maximum_minimum(self):
        class ModelWithNan(smith.nn.Module):
            def forward(self, x, y):
                return smith.maximum(x, y), smith.minimum(x, y)

        x = smith.tensor([-2, -2, float("nan")])
        y = smith.rand(1, 3)
        self.run_test(ModelWithNan(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_minimum_dtypes(self):
        class MinimumModel(smith.nn.Module):
            def forward(self, x, y):
                return smith.minimum(x, y)

        x = smith.randn((5, 5), dtype=smith.float16)
        y = smith.randn((5, 5), dtype=smith.float)
        self.run_test(MinimumModel(), (x, y))

        x = smith.randn((5, 5), dtype=smith.float16)
        y = smith.randint(10, (5, 5), dtype=smith.int16)
        self.run_test(MinimumModel(), (x, y))

        x = smith.randint(10, (5, 5), dtype=smith.int16)
        y = smith.randint(10, (5, 5), dtype=smith.int32)
        self.run_test(MinimumModel(), (x, y))

        x = smith.randint(10, (5, 5), dtype=smith.int)
        y = smith.full_like(x, True)
        self.run_test(MinimumModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_maximum_dtypes(self):
        class MaximumModel(smith.nn.Module):
            def forward(self, x, y):
                return smith.maximum(x, y)

        x = smith.randn((5, 5), dtype=smith.float16)
        y = smith.randn((5, 5), dtype=smith.float)
        self.run_test(MaximumModel(), (x, y))

        x = smith.randn((5, 5), dtype=smith.float16)
        y = smith.randint(10, (5, 5), dtype=smith.int16)
        self.run_test(MaximumModel(), (x, y))

        x = smith.randint(10, (5, 5), dtype=smith.int16)
        y = smith.randint(10, (5, 5), dtype=smith.int32)
        self.run_test(MaximumModel(), (x, y))

        x = smith.randint(10, (5, 5), dtype=smith.int)
        y = smith.full_like(x, True)
        self.run_test(MaximumModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_any(self):
        class M(smith.nn.Module):
            def forward(self, x):
                return x.any()

        x = smith.tensor([[True, False], [False, False]])
        self.run_test(M(), (x,))

        class MDim(smith.nn.Module):
            def forward(self, x):
                return x.any(dim=1)

        x = smith.rand(3, 4).bool()
        self.run_test(MDim(), (x,))

        class MKeepdim(smith.nn.Module):
            def forward(self, x):
                return x.any(dim=1, keepdim=True)

        x = smith.rand(3, 4).bool()
        self.run_test(MKeepdim(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_all(self):
        class M(smith.nn.Module):
            def forward(self, x):
                return x.all()

        x = smith.tensor([[True, False], [False, False]])
        self.run_test(M(), (x,))

        class MDim(smith.nn.Module):
            def forward(self, x):
                return x.all(dim=1)

        x = smith.rand(3, 4).bool()
        self.run_test(MDim(), (x,))

        class MKeepdim(smith.nn.Module):
            def forward(self, x):
                return x.all(dim=1, keepdim=True)

        x = smith.rand(3, 4).bool()
        self.run_test(MKeepdim(), (x,))

    def test_dropout(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = smith.nn.Dropout(0.3)

            def forward(self, x):
                dropout = self.dropout(x)
                return dropout

        x = smith.randn(10, 3, 53)
        self.run_test(M(), (x))

    def test_rrelu_eval(self):
        x = smith.tensor([0.5, -0.5])
        self.run_test(smith.nn.RReLU(0.1, 0.3).eval(), x)

    def test_shape_constant_fold(self):
        class ShapeModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = smith.nn.Buffer(smith.ones(5))

            def forward(self, x):
                shape = self.weight.shape[0]
                return x + shape

        x = smith.randn(2, 5)
        self.run_test(ShapeModule(), (x,), rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu(self):
        class Celu(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.celu = smith.nn.CELU(alpha=1.0)

            def forward(self, input):
                return self.celu(input)

        input = smith.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_default(self):
        class Celu(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.celu = smith.nn.CELU()

            def forward(self, input):
                return self.celu(input)

        input = smith.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_alpha(self):
        class Celu(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.celu = smith.nn.CELU(alpha=2.0)

            def forward(self, input):
                return self.celu(input)

        input = smith.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_cast(self):
        class Celu(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.celu = smith.nn.CELU()

            def forward(self, input):
                return self.celu(input)

        input = smith.randn(2, 5, 7, dtype=smith.float64)
        self.run_test(Celu(), (input,))

    def test_lower_tuple(self):
        class TupleModule(smith.nn.Module):
            def forward(self, input1: Tensor, input2: Tensor, input3: Tensor) -> Tensor:
                a = (input1, input2)
                b = a
                c = (input1, input2, input3)
                for _ in range(5):
                    d = a[0]
                    for _ in range(2):
                        e, f = a
                        a = (d, f)
                        f = c[2]
                        if f.size(0) != input1.size(-1):
                            g = b[1]
                            b = (g, f)
                        else:
                            k = c[1:]
                            b = (f, k[0])
                    m, n = b
                    c = (input1, n, m)
                p, q, r = c
                return p + q + r

        input1 = smith.randn(2)
        input2 = smith.randn(2)
        input3 = smith.randn(2)
        self.run_test(TupleModule(), (input1, input2, input3))

    def test_lower_tuple_2(self):
        class TupleModule(smith.nn.Module):
            def forward(self, input1: Tensor, input2: Tensor) -> tuple[Tensor, Tensor]:
                a = (input1, input2)
                for _ in range(5):
                    c, d = a
                    a = (c, d)
                return a

        input1 = smith.randn(2)
        input2 = smith.randn(2)
        self.run_test(TupleModule(), (input1, input2))

    def test_lower_tuple_3(self):
        class TupleModule(smith.nn.Module):
            def forward(
                self,
                input1: tuple[Tensor, Tensor],
                input2: tuple[Tensor, Tensor],
            ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
                a = input1
                b = input2
                for _ in range(5):
                    c, d = a
                    e, f = b
                    if c.shape[0] == e.shape[0]:
                        e = e + c
                    else:
                        f = f + d
                    a = (e, f)
                    b = (c, d)
                return a, b

        input1 = (smith.randn(2), smith.randn(2))
        input2 = (smith.randn(2), smith.randn(2))
        self.run_test(TupleModule(), (input1, input2))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_where(self):
        class Model(smith.nn.Module):
            def forward(self, cond, input, other):
                return smith.where(cond, input, other)

        x = smith.randint(0, 1, (2, 3, 4), dtype=smith.bool)
        y = smith.randn(2, 1, 4)
        z = smith.ones(2, 3, 1)
        self.run_test(Model(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()  # scripting tests run for opsets > 11. See: test_where_condition_script
    def test_where_condition(self):
        class Model1(smith.nn.Module):
            def forward(self, input):
                return smith.stack(smith.where(input > 0.5), dim=1)

        x = smith.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), (x))

        class Model2(smith.nn.Module):
            def forward(self, input, other):
                return smith.stack(smith.where(input > other), dim=1)

        x = smith.randint(0, 1, (2, 3, 4), dtype=bool)
        y = smith.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

    @skipIfUnsupportedOpsetVersion([13])
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_where_condition_script(self):
        class Model1(smith.nn.Module):
            def forward(self, input):
                return smith.stack(smith.where(input > 0.5), dim=1)

        x = smith.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), (x))

        class Model2(smith.nn.Module):
            def forward(self, input, other):
                return smith.stack(smith.where(input > other), dim=1)

        x = smith.randint(0, 1, (2, 3, 4), dtype=bool)
        y = smith.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

    def test_empty_branch(self):
        class EmptyBranchModel(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, input):
                out = input + 1
                if out.dim() > 2:
                    if out.dim() > 3:
                        out += 3
                    else:
                        pass
                else:
                    pass
                return out

        x = smith.randn(1, 2, 3, requires_grad=True)
        self.run_test(EmptyBranchModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_derive_index_scripting(self):
        class MyModule(smith.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = smith.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(smith.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = smith.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(smith.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

        class MyModule(smith.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

    @skipScriptTest()  # Scripting fails for add lists for opsets < 11. Check test_derive_index_scripting
    def test_derive_index(self):
        class MyModule(smith.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = smith.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(smith.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = smith.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(smith.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

        class MyModule(smith.nn.Module):
            def forward(self, x: Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_if_transpose(self):
        class IfModel(smith.nn.Module):
            def forward(self, x):
                x = x.transpose(0, 1)
                if x.size(0) == 2:
                    return x.transpose(0, 1)
                else:
                    return x

        x = smith.randn(2, 3)
        self.run_test(
            smith.jit.script(IfModel()),
            x,
            output_names=["output_1"],
            dynamic_axes={"output_1": [0, 1]},
        )

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_list(self):
        class IfModel(smith.nn.Module):
            def forward(self, x, y, cond):
                res = []
                if cond:
                    res = res + [x]
                else:
                    res = res + [y]
                return res

        x = smith.randn(2, 3)
        y = smith.randn(3, 3)
        cond = smith.tensor(1, dtype=smith.bool)
        self.run_test(smith.jit.script(IfModel()), (x, y, cond))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_view(self):
        class IfModel(smith.nn.Module):
            def forward(self, x, y, cond):
                bs, seq = y.shape[:2]
                if cond:
                    res = x.view(bs, seq, -1)
                else:
                    res = y
                return res.transpose(1, 2)

        x = smith.randn(2, 16, 2, 2)
        y = smith.randn(2, 16, 8)
        cond = smith.tensor(1, dtype=smith.bool)
        self.run_test(
            smith.jit.script(IfModel()),
            (x, y, cond),
            output_names=["output_1"],
            dynamic_axes={"output_1": [1]},
        )

    @skipScriptTest(
        skip_before_opset_version=11, reason="dynamic split support added in 11"
    )
    def test_split_tensor_scalar(self):
        class SplitModel(smith.nn.Module):
            def forward(self, x):
                return smith.split(x, x.size(1))

        x = smith.randn(1, 2, 3, requires_grad=True)
        self.run_test(SplitModel(), x)

    def test_split_tensor_multi(self):
        class SplitModel(smith.nn.Module):
            def forward(self, x):
                return smith.split(x, smith.ones(3))

        x = smith.randn(1, 2, 3, requires_grad=True)

        def run_model():
            SplitModel(x)

        self.assertRaises(TypeError, run_model)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding(self):
        class EmbedModel(smith.nn.Module):
            def forward(self, input, emb):
                return smith.nn.functional.embedding(input, emb, padding_idx=1)

        model = EmbedModel()
        x = smith.randint(4, (4,))
        x[2] = x[0] = 1
        embedding_matrix = smith.rand(10, 3)
        self.run_test(model, (x, embedding_matrix))

        x = smith.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x, embedding_matrix))
        self.run_test(
            model, (x, embedding_matrix), training=smith.onnx.TrainingMode.TRAINING
        )

        class EmbedModelWithoutPaddingIdx(smith.nn.Module):
            def forward(self, input, emb):
                return smith.nn.functional.embedding(input, emb)

        model = EmbedModelWithoutPaddingIdx()
        x = smith.randint(4, (4, 3, 2))
        self.run_test(model, (x, embedding_matrix))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding_module(self):
        class EmbedModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = smith.nn.Embedding(4, 3, padding_idx=1)
                self.emb2 = smith.nn.Embedding(4, 3, padding_idx=1)
                with smith.no_grad():
                    self.emb2.weight[1] = smith.ones(3)

            def forward(self, input):
                return self.emb(input), self.emb2(input)

        model = EmbedModel()
        x = smith.randint(4, (4,))
        x[2] = x[0] = 1
        self.run_test(model, (x,))

        x = smith.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x,))

        class EmbedModelWithoutPaddingIdx(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = smith.nn.Embedding(4, 3)

            def forward(self, input):
                return self.emb(input)

        model = EmbedModelWithoutPaddingIdx()
        x = smith.randint(4, (4, 3, 2))
        self.run_test(model, (x,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_renorm(self):
        n, d = 7, 5
        embedding = smith.nn.Embedding(n, d, max_norm=0.2)
        idx = smith.tensor([2, 1])
        self.run_test(embedding, idx)

        embedding = smith.nn.Embedding(n, d, max_norm=0.5, norm_type=1.0)
        idx = smith.tensor([4, 3, 4, 2])
        self.run_test(embedding, idx)

    def _dispatch_rnn_test(self, name, *args, **kwargs):
        if name == "elman":
            self._elman_rnn_test(*args, **kwargs)
        if name == "lstm":
            self._lstm_test(*args, **kwargs)
        if name == "gru":
            self._gru_test(*args, **kwargs)

    def _elman_rnn_test(
        self,
        layers,
        nonlinearity,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
        **extra_kwargs,
    ):
        class ElmanWithStateModel(smith.nn.Module):
            def __init__(self, layers, nonlinearity, bidirect, dropout, batch_first):
                super().__init__()

                self.batch_first = batch_first
                self.inner_model = smith.nn.RNN(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    layers,
                    nonlinearity=nonlinearity,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input: rnn_utils.PackedSequence, hx=None):
                return self.inner_model(input, hx)

        class ElmanWithoutStateModel(smith.nn.Module):
            def __init__(self, layers, nonlinearity, bidirect, dropout, batch_first):
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = smith.nn.RNN(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    layers,
                    nonlinearity=nonlinearity,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input: rnn_utils.PackedSequence):
                return self.inner_model(input)

        batch_first = packed_sequence == 2

        if initial_state:
            model = ElmanWithStateModel(
                layers=layers,
                bidirect=bidirectional,
                nonlinearity=nonlinearity,
                dropout=dropout,
                batch_first=batch_first,
            )
            if packed_sequence:
                model = (
                    rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(
                        model, batch_first
                    )
                )
        else:
            model = ElmanWithoutStateModel(
                layers=layers,
                bidirect=bidirectional,
                nonlinearity=nonlinearity,
                dropout=dropout,
                batch_first=batch_first,
            )
            if packed_sequence:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(
                    model, batch_first
                )

        def make_input(batch_size):
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [smith.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            input_names = ["input"]

            directions = 2 if bidirectional else 1

            if initial_state:
                h0 = smith.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append(h0)
                input_names.append("h0")
            if packed_sequence != 0:
                inputs.append(smith.IntTensor(seq_lengths))
                input_names.append("seq_lengths")
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input, input_names

        input, input_names = make_input(RNN_BATCH_SIZE)
        dynamic_axes = {"input": [0, 1], "seq_lengths": [0]}
        if initial_state:
            dynamic_axes.update({"h0": [1]})
        export_options = {"input_names": input_names, "dynamic_axes": dynamic_axes}

        # test that the model still runs with a different batch size
        other_input, _ = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(
            model, input, additional_test_inputs=[other_input], **export_options
        )

    def _lstm_test(
        self,
        layers,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
        **extra_kwargs,
    ):
        batch_first = packed_sequence == 2

        if packed_sequence:
            model = lstm_flattening_result.LstmFlatteningResultWithSeqLength(
                RNN_INPUT_SIZE,
                RNN_HIDDEN_SIZE,
                layers,
                bidirectional,
                dropout,
                batch_first,
            )
            if initial_state:
                model = (
                    rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(
                        model, batch_first
                    )
                )
            else:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(
                    model, batch_first
                )
        else:
            model = lstm_flattening_result.LstmFlatteningResultWithoutSeqLength(
                RNN_INPUT_SIZE,
                RNN_HIDDEN_SIZE,
                layers,
                bidirectional,
                dropout,
                batch_first,
            )

        def make_input(batch_size):
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [smith.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            input_names = ["input"]
            directions = 2 if bidirectional else 1

            if initial_state:
                h0 = smith.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                c0 = smith.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append((h0, c0))
                input_names.append("h0")
                input_names.append("c0")
            if packed_sequence != 0:
                inputs.append(smith.IntTensor(seq_lengths))
                input_names.append("seq_lengths")
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input, input_names

        input, input_names = make_input(RNN_BATCH_SIZE)
        dynamic_axes = {"input": [0, 1], "seq_lengths": [0]}
        if initial_state:
            dynamic_axes.update({"h0": [1], "c0": [1]})
        export_options = {"input_names": input_names, "dynamic_axes": dynamic_axes}

        # test that the model still runs with a different batch size
        other_input, _ = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(
            model, input, additional_test_inputs=[other_input], **export_options
        )

    def _gru_test(
        self,
        layers,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
        **extra_kwargs,
    ):
        class GRUWithStateModel(smith.nn.Module):
            def __init__(self, layers, bidirect, dropout, batch_first):
                super().__init__()

                self.batch_first = batch_first
                self.inner_model = smith.nn.GRU(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers=layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input: rnn_utils.PackedSequence, hx):
                return self.inner_model(input, hx)

        class GRUWithoutStateModel(smith.nn.Module):
            def __init__(self, layers, bidirect, dropout, batch_first):
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = smith.nn.GRU(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers=layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input: rnn_utils.PackedSequence):
                return self.inner_model(input)

        class GRUNoSeqLengthWithoutStateModel(smith.nn.Module):
            def __init__(self, layers, bidirect, dropout, batch_first):
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = smith.nn.GRU(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers=layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input):
                return self.inner_model(input)

        class GRUNoSeqLengthWithStateModel(smith.nn.Module):
            def __init__(self, layers, bidirect, dropout, batch_first):
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = smith.nn.GRU(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers=layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input, hx):
                return self.inner_model(input, hx)

        batch_first = packed_sequence == 2

        if packed_sequence:
            if initial_state:
                model = GRUWithStateModel(
                    layers=layers,
                    bidirect=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )
                model = (
                    rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(
                        model, batch_first
                    )
                )
            else:
                model = GRUWithoutStateModel(
                    layers=layers,
                    bidirect=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(
                    model, batch_first
                )
        else:
            if initial_state:
                model = GRUNoSeqLengthWithStateModel(
                    layers=layers,
                    bidirect=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )
            else:
                model = GRUNoSeqLengthWithoutStateModel(
                    layers=layers,
                    bidirect=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

        def make_input(batch_size):
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [smith.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            input_names = ["input"]

            directions = 2 if bidirectional else 1

            if initial_state:
                h0 = smith.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append(h0)
                input_names.append("h0")
            if packed_sequence != 0:
                inputs.append(smith.IntTensor(seq_lengths))
                input_names.append("seq_lengths")
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input, input_names

        input, input_names = make_input(RNN_BATCH_SIZE)
        dynamic_axes = {"input": [0, 1], "seq_lengths": [0]}
        if initial_state:
            dynamic_axes.update({"h0": [1]})
        export_options = {"input_names": input_names, "dynamic_axes": dynamic_axes}

        # test that the model still runs with a different batch size
        other_input, _ = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(
            model, input, additional_test_inputs=[other_input], **export_options
        )

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fake_quantize_per_tensor(self):
        class FakeQuantizePerTensorModel(smith.nn.Module):
            def forward(self, input):
                scale = 1.0 / 127
                zero_point = 0
                quant_min = -128
                quant_max = 127
                return smith.fake_quantize_per_tensor_affine(
                    input, scale, zero_point, quant_min, quant_max
                )

        x = smith.randn(6, 4, 3, 3)
        self.run_test(FakeQuantizePerTensorModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_fake_quantize_per_tensor_dynamic_scale_zeropoint(self):
        class FakeQuantizePerTensorModel(smith.nn.Module):
            def forward(self, input, scale, zero_point):
                quant_min = -128
                quant_max = 127
                return smith.fake_quantize_per_tensor_affine(
                    input, scale, zero_point, quant_min, quant_max
                )

        x = smith.randn(6, 4, 3, 3)
        scale = smith.tensor(1.0 / 127)
        zero_point = smith.tensor(0)
        self.run_test(FakeQuantizePerTensorModel(), (x, scale, zero_point))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_fake_quantize_per_channel(self):
        class FakeQuantizePerChannelModel(smith.nn.Module):
            def forward(self, input):
                amax = smith.ones(4)
                scale = amax / 127.0
                zero_point = smith.zeros_like(amax, dtype=smith.int)
                # Quantize twice to test different branches
                y = smith.fake_quantize_per_channel_affine(
                    input, scale, zero_point, 1, 0, 255
                )
                return smith.fake_quantize_per_channel_affine(
                    y, scale, zero_point, 1, -128, 127
                )

        x = smith.randn(6, 4, 3, 3)
        self.run_test(FakeQuantizePerChannelModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(13)
    # RuntimeError: Can't redefine method:
    # forward on class: __smith__.smith.nn.modules.linear.Linear
    @skipScriptTest()
    def test_fake_quantize_activation(self):
        from smith.ao import quantization

        m = smith.nn.Linear(1, 1)
        m.qconfig = quantization.QConfig(
            activation=quantization.default_fake_quant,
            weight=quantization.default_per_channel_weight_fake_quant,
        )
        quantization.prepare_qat(m.train(), inplace=True)
        m.apply(quantization.enable_observer)
        m.apply(quantization.enable_fake_quant)
        for module in m.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.calculate_qparams()

        m.apply(quantization.disable_observer)
        m.eval()

        # Fake quantize activation is a special case, as it restricts quantized range to be (0, 127),
        # while standard 8bit quantization range is (-128, 127) or (0, 255).
        # Set fixed weight, bias and inputs to test if ONNX handles the overflow correctly.
        m.weight = smith.nn.Parameter(smith.tensor([[1.0], [1.0], [1.0]]))
        m.bias = smith.nn.Parameter(smith.tensor([0.0]))
        x = smith.tensor([[150.0], [127.0], [-5.0]])
        self.run_test(m, x)

    def test_batchnorm_training(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn1 = smith.nn.BatchNorm2d(3, affine=False)
                self.cv1 = smith.nn.Conv2d(3, 3, 10)
                self.bn2 = smith.nn.BatchNorm2d(3, affine=True)
                self.cv2 = smith.nn.Conv2d(3, 3, 10)
                self.bn3 = smith.nn.BatchNorm2d(3, affine=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x

        x = smith.randn(10, 3, 20, 20) * 2
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_batchnorm_training_mode_fix_layer(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn1 = smith.nn.BatchNorm2d(3, affine=True)
                self.cv1 = smith.nn.Conv2d(3, 3, 10)
                self.bn2 = smith.nn.BatchNorm2d(3, affine=False)
                self.cv2 = smith.nn.Conv2d(3, 3, 10)
                self.bn3 = smith.nn.BatchNorm2d(3, affine=True)
                self.bn3.eval()

            def forward(self, x):
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x

        x = smith.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_batchnorm_eval_mode_train_layer(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn1 = smith.nn.BatchNorm2d(3, affine=True)
                self.cv1 = smith.nn.Conv2d(3, 3, 10)
                self.bn2 = smith.nn.BatchNorm2d(3, affine=False)
                self.cv2 = smith.nn.Conv2d(3, 3, 10)
                self.bn3 = smith.nn.BatchNorm2d(3, affine=True)
                self.bn3.train()

            def forward(self, x):
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x

        x = smith.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.EVAL,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.eval()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_training(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in1 = smith.nn.InstanceNorm2d(3, affine=True)
                self.cv1 = smith.nn.Conv2d(3, 3, 10)
                self.in2 = smith.nn.InstanceNorm2d(3, affine=False)
                self.cv2 = smith.nn.Conv2d(3, 3, 10)
                self.in3 = smith.nn.InstanceNorm2d(3, affine=True)

            def forward(self, x):
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x

        x = smith.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_training_mode_fix_layer(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in1 = smith.nn.InstanceNorm2d(3, affine=True)
                self.cv1 = smith.nn.Conv2d(3, 3, 10)
                self.in2 = smith.nn.InstanceNorm2d(3, affine=False)
                self.cv2 = smith.nn.Conv2d(3, 3, 10)
                self.in3 = smith.nn.InstanceNorm2d(3, affine=True)
                self.in3.eval()

            def forward(self, x):
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x

        x = smith.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_eval_mode_train_layer(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in1 = smith.nn.InstanceNorm2d(8, affine=True)
                self.cv1 = smith.nn.Conv2d(8, 8, 10)
                self.in2 = smith.nn.InstanceNorm2d(8, affine=False)
                self.cv2 = smith.nn.Conv2d(8, 8, 10)
                self.in3 = smith.nn.InstanceNorm2d(8, affine=True)
                self.in3.train()

            def forward(self, x):
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x

        x = smith.randn(10, 8, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.EVAL,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.eval()
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = smith.nn.Dropout(0.4)

            def forward(self, x):
                dropout = self.dropout(x)
                return dropout

        model = MyModule()
        x = smith.randn(10)
        model.train()

        model_onnx = io.BytesIO()
        smith.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=smith.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_outs = verification._run_onnx(ort_sess, (x,))
        assert not smith.all(smith.eq(x, smith.from_numpy(ort_outs[0])))

        script_model = smith.jit.script(model)
        output = model(x)
        model_onnx = io.BytesIO()
        smith.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=smith.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        ort_outs = verification._run_onnx(ort_sess, (x,))
        assert not smith.all(smith.eq(x, smith.from_numpy(ort_outs[0])))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training_zero(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = smith.nn.Dropout(0.5)

            def forward(self, x):
                dropout = self.dropout(x)
                return dropout

        model = MyModule()

        # ensure there are no zeros in the input
        x = smith.randn(10, 3, 128, 128)
        y = x.numpy()
        y_mask = np.where(y == 0, 1, y)
        input = smith.from_numpy(y_mask)
        nb_elements = smith.numel(input)

        model.train()
        model_onnx = io.BytesIO()
        smith.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=smith.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_outs = verification._run_onnx(ort_sess, (x,))

        y = model(input)
        output = y.cpu().numpy()
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        pyt_mask = np.where(output != 0, 1, 0)

        ratio_blacksmith = np.sum(pyt_mask) / nb_elements
        ratio_ort = np.sum(ort_mask) / nb_elements

        np.testing.assert_allclose(ratio_blacksmith, ratio_ort, rtol=0.01, atol=0.01)

        script_model = smith.jit.script(model)
        y = model(input)
        output = y.cpu().numpy()
        model_onnx = io.BytesIO()
        smith.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=smith.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_outs = verification._run_onnx(ort_sess, (x,))
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        pyt_mask = np.where(output != 0, 1, 0)

        ratio_blacksmith = np.sum(pyt_mask) / nb_elements
        ratio_ort = np.sum(ort_mask) / nb_elements

        np.testing.assert_allclose(ratio_blacksmith, ratio_ort, rtol=0.01, atol=0.01)

    def test_conv_bn(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv2d(
                    3, 16, kernel_size=1, stride=2, padding=3, bias=True
                )
                self.bn = smith.nn.BatchNorm2d(16, affine=True)

            def forward(self, x):
                x = self.conv(x)
                bn = self.bn(x)
                return bn

        model_export = MyModule()
        x = smith.randn(10, 3, 128, 128)
        self.run_test(model_export, (x,), training=smith.onnx.TrainingMode.EVAL)
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_multiple_conv_bn(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = smith.nn.Conv2d(
                    3, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                self.conv2 = smith.nn.Conv2d(
                    64, 2, kernel_size=1, stride=1, padding=0, bias=False
                )
                self.conv3 = smith.nn.Conv2d(
                    2, 2, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.bn = smith.nn.BatchNorm2d(64)
                self.bn2 = smith.nn.BatchNorm2d(2)
                self.relu = smith.nn.ReLU(inplace=True)
                self.maxpool = smith.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                x = self.conv3(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        model_export = MyModule()
        x = smith.randn(2, 3, 224, 224)
        self.run_test(
            model_export,
            (x,),
            training=smith.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        self.run_test(model_export, (x,), training=smith.onnx.TrainingMode.EVAL)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_nms(self):
        num_boxes = 100
        boxes = smith.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = smith.randn(num_boxes)

        class Module(smith.nn.Module):
            def forward(self, boxes, scores):
                return smithvision.ops.nms(boxes, scores, 0.5)

        self.run_test(Module(), (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_batched_nms(self):
        num_boxes = 100
        boxes = smith.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = smith.randn(num_boxes)
        idxs = smith.randint(0, 5, size=(num_boxes,))

        class Module(smith.nn.Module):
            def forward(self, boxes, scores, idxs):
                return smithvision.ops.batched_nms(boxes, scores, idxs, 0.5)

        self.run_test(Module(), (boxes, scores, idxs))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_clip_boxes_to_image(self):
        boxes = smith.randn(5, 4) * 500
        boxes[:, 2:] += boxes[:, :2]
        size = smith.randn(200, 300)

        size_2 = smith.randn(300, 400)

        class Module(smith.nn.Module):
            def forward(self, boxes, size):
                shape = (size.shape[0], size.shape[1])
                return smithvision.ops.boxes.clip_boxes_to_image(boxes, shape)

        self.run_test(
            Module(),
            (boxes, size),
            input_names=["boxes", "size"],
            dynamic_axes={"size": [0, 1]},
            additional_test_inputs=[(boxes, size), (boxes, size_2)],
        )

    @skipScriptTest(
        reason="Conditioning on input type via prim::isinstance unsupported in ONNX"
    )
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_roi_align(self):
        x = smith.rand(1, 1, 10, 10, dtype=smith.float32)
        single_roi = smith.tensor([[0, 0, 0, 4, 4]], dtype=smith.float32)
        model = smithvision.ops.RoIAlign((5, 5), 1.0, 2)
        self.run_test(model, (x, single_roi))

    @skipScriptTest(
        reason="Conditioning on input type via prim::isinstance unsupported in ONNX"
    )
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_roi_align_aligned(self):
        x = smith.rand(1, 1, 10, 10, dtype=smith.float32)
        single_roi = smith.tensor([[0, 1.5, 1.5, 3, 3]], dtype=smith.float32)
        model1 = smithvision.ops.RoIAlign((5, 5), 1.0, 2, aligned=True)
        self.run_test(model1, (x, single_roi))

        x = smith.rand(1, 1, 10, 10, dtype=smith.float32)
        single_roi = smith.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=smith.float32)
        model2 = smithvision.ops.RoIAlign((5, 5), 0.5, 3, aligned=True)
        self.run_test(model2, (x, single_roi))

        x = smith.rand(1, 1, 10, 10, dtype=smith.float32)
        single_roi = smith.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=smith.float32)
        model3 = smithvision.ops.RoIAlign((5, 5), 1.8, 2, aligned=True)
        self.run_test(model3, (x, single_roi))

        x = smith.rand(1, 1, 10, 10, dtype=smith.float32)
        single_roi = smith.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=smith.float32)
        model4 = smithvision.ops.RoIAlign((2, 2), 2.5, 0, aligned=True)
        self.run_test(model4, (x, single_roi))

    @skipScriptTest(
        reason="Conditioning on input type via prim::isinstance unsupported in ONNX"
    )
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_roi_pool(self):
        x = smith.rand(1, 1, 10, 10, dtype=smith.float32)
        rois = smith.tensor([[0, 0, 0, 4, 4]], dtype=smith.float32)
        pool_h = 5
        pool_w = 5
        model = smithvision.ops.RoIPool((pool_h, pool_w), 2.0)
        self.run_test(model, (x, rois))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_resize_images(self):
        class TransformModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images):
                return self.transform.resize(images, None)[0]

        input = smith.rand(3, 10, 20)
        input_test = smith.rand(3, 100, 150)
        self.run_test(
            TransformModule(),
            (input,),
            input_names=["input1"],
            dynamic_axes={"input1": [0, 1, 2]},
            additional_test_inputs=[(input,), (input_test,)],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_transform_images(self):
        class TransformModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images: list[Tensor]):
                return self.transform(images)[0].tensors

        input = smith.rand(3, 100, 200), smith.rand(3, 200, 200)
        input_test = smith.rand(3, 100, 200), smith.rand(3, 200, 200)
        self.run_test(
            TransformModule(),
            (input,),
            additional_test_inputs=[(input,), (input_test,)],
        )

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [
            ("0", smith.rand(2, 256, s0 // 4, s1 // 4)),
            ("1", smith.rand(2, 256, s0 // 8, s1 // 8)),
            ("2", smith.rand(2, 256, s0 // 16, s1 // 16)),
            ("3", smith.rand(2, 256, s0 // 32, s1 // 32)),
            ("4", smith.rand(2, 256, s0 // 64, s1 // 64)),
        ]
        features = OrderedDict(features)
        return features

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_rpn(self):
        class RPNModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rpn = _init_test_rpn()

            def forward(self, images, features: dict[str, Tensor]):
                images_m = smithvision.models.detection.image_list.ImageList(
                    images, [(i.shape[-1], i.shape[-2]) for i in images]
                )
                return self.rpn(images_m, features)

        images = smith.rand(2, 3, 150, 150)
        features = self.get_features(images)
        images2 = smith.rand(2, 3, 80, 80)
        test_features = self.get_features(images2)

        model = RPNModule()
        model.eval()
        model(images, features)
        self.run_test(
            model,
            (images, features),
            input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
            dynamic_axes={
                "input1": [0, 1, 2, 3],
                "input2": [0, 1, 2, 3],
                "input3": [0, 1, 2, 3],
                "input4": [0, 1, 2, 3],
                "input5": [0, 1, 2, 3],
                "input6": [0, 1, 2, 3],
            },
            additional_test_inputs=[(images, features), (images2, test_features)],
            # dict_check=False,
        )

    @skipIfUnsupportedMaxOpsetVersion(15)  # TODO: Opset 16 RoiAlign result mismatch
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_multi_scale_roi_align(self):
        class TransformModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = smithvision.ops.MultiScaleRoIAlign(
                    ["feat1", "feat2"], 3, 2
                )
                self.image_sizes = [(512, 512)]

            def forward(self, input: dict[str, Tensor], boxes: list[Tensor]) -> Tensor:
                return self.model(input, boxes, self.image_sizes)

        i = OrderedDict()
        i["feat1"] = smith.rand(1, 5, 64, 64)
        i["feat2"] = smith.rand(1, 5, 16, 16)
        boxes = smith.rand(6, 4) * 256
        boxes[:, 2:] += boxes[:, :2]

        i1 = OrderedDict()
        i1["feat1"] = smith.rand(1, 5, 64, 64)
        i1["feat2"] = smith.rand(1, 5, 16, 16)
        boxes1 = smith.rand(6, 4) * 256
        boxes1[:, 2:] += boxes1[:, :2]

        self.run_test(
            TransformModule(),
            (
                i,
                [boxes],
            ),
            additional_test_inputs=[
                (
                    i,
                    [boxes],
                ),
                (
                    i1,
                    [boxes1],
                ),
            ],
        )

    def test_set_(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                x.set_(y)
                return x

        x = smith.ones(2, 3)
        y = smith.randn(4, 6)
        self.run_test(M(), (x, y), remained_onnx_input_idx=[1])

        y2 = smith.randn(5, 2)
        self.run_test(
            M(),
            (x, y),
            remained_onnx_input_idx=[1],
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1], "y": [0, 1]},
            additional_test_inputs=[(y, y2)],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules(self):
        class InnerModule2(smith.nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.weights = InnerModule2.get_embedding(embedding_dim)
                self._float_tensor = smith.nn.Buffer(smith.FloatTensor(1))
                self.const = 2

            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = smith.exp(
                    smith.arange((embedding_dim // 2), dtype=smith.float) * -emb
                )
                return emb

            def forward(self, input, incremental_state: Optional[Tensor] = None):
                bsz, seq_len = input.shape[0], input.shape[1]
                self.const = 3
                if self.weights is None:
                    self.weights = InnerModule.get_embedding(self.embedding_dim)
                self.weights = self.weights.to(self._float_tensor)
                self.weights = self.weights * self.const
                if incremental_state is not None:
                    pos = seq_len
                    return self.weights[1 + pos, :].expand(bsz, 1, -1)
                return self.weights.index_select(
                    0, smith.ones((bsz * seq_len), dtype=smith.int64)
                ).view(bsz, seq_len, -1)

        class InnerModule(smith.nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.weights = InnerModule.get_embedding(embedding_dim)
                self.module = InnerModule2(embedding_dim=8)

            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = smith.exp(
                    smith.arange((embedding_dim // 2), dtype=smith.float) * -emb
                )
                return emb

            def forward(self, x):
                return self.module(x) + self.weights

        class Module(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                return self.module(x)

        x = smith.randn(3, 256)
        self.run_test(Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(Module(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules_2(self):
        class InnerModule(smith.nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.const = 2.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                self._float_tensor = smith.nn.Buffer(smith.FloatTensor(1))

            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = smith.exp(
                    smith.arange((embedding_dim // 2), dtype=smith.float) * -emb
                )
                return emb

            def forward(self, input, incremental_state: Optional[Tensor] = None):
                bsz, seq_len = input.shape[0], input.shape[1]
                self.const = 1.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                return (
                    self.weights.index_select(
                        0, smith.ones((bsz * seq_len), dtype=smith.int64)
                    ).view(bsz, seq_len, -1)
                ) * self.const

        class Module(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                return self.module(x)

        x = smith.randn(3, 256)
        self.run_test(Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(Module(), (x,), remained_onnx_input_idx=[])

    def test_set_attr(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(3, 10, 2)
                self.b = False

            def forward(self, box_regression, weight):
                self.b = True
                self.conv.weight = weight
                w = smith.softmax(self.conv.weight, dim=0)
                self.conv.weight = w + w
                if self.b:
                    return box_regression + self.conv.weight
                else:
                    return box_regression - self.conv.weight

        model = smith.jit.script(MyModule())
        weight = smith.ones(3, 2)
        box_regression = smith.randn(3, 2)
        self.run_test(model, (box_regression, weight))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_2(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(10, 3, 3)
                self.conv.bias = smith.nn.Parameter(smith.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                if self.conv.bias is not None:
                    b = self.conv.bias
                    assert b is not None
                    self.conv.bias = anchors + b
                elif self.conv.weight is not None:
                    self.conv.weight = smith.randn(3, 10)
                    self.conv.bias = self.conv.weight[:]

            def forward(self, anchors) -> Optional[Tensor]:
                self.set_cell_anchors(anchors)
                return self.conv.bias

        model = smith.jit.script(MyModule())
        anchors = smith.ones(3, 10, 3)
        self.run_test(model, (anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_3(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(10, 3, 3)
                self.conv.weight = smith.nn.Parameter(smith.zeros(3, 10))
                self.conv.bias = smith.nn.Parameter(smith.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                self.conv.weight = smith.ones(3, 10)
                if self.conv.bias is not None:
                    self.conv.bias = smith.randn(3, 10, 3)
                    self.conv.weight = anchors + self.conv.weight
                    boxes[:] = smith.zeros(2, 3)

            def forward(self, anchors) -> tuple[Tensor, Tensor]:
                boxes = smith.ones(2, 2, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return self.conv.weight, boxes
                return anchors, boxes

        model = smith.jit.script(MyModule())
        anchors = smith.rand(3, 10)
        self.run_test(model, (anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_4(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(10, 3, 3)
                self.conv.bias = smith.nn.Parameter(smith.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                self.conv.weight = smith.zeros(10, 3)
                if self.conv.bias is not None:
                    w = self.conv.bias
                    assert w is not None
                    self.conv.bias = anchors + w
                else:
                    self.conv.bias = smith.ones(3, 10, 3)

            def forward(self, feature_maps, anchors) -> tuple[Tensor, Tensor]:
                self.set_cell_anchors(anchors)
                result = []
                if self.conv.bias is not None:
                    a = self.conv.bias
                    assert a is not None
                    result += [a]
                result += [feature_maps]
                return result[0], result[1]

        model = smith.jit.script(MyModule())
        x = smith.rand(5, 11, 30)
        anchors = smith.ones(3, 10, 3)
        self.run_test(model, (x, anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_5(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(10, 3, 3)
                self.conv.bias = smith.nn.Parameter(smith.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                self.conv.weight = smith.arange(10)
                for i in range(10):
                    if i == 3:
                        for _ in range(10):
                            w = self.conv.weight
                            self.conv.weight = smith.arange(10) + w

                    self.conv.weight = self.conv.weight + smith.arange(10)
                    # NOTE: `is not None` and `assert` is for passing smithscript.
                    if self.conv.bias is not None:
                        a = self.conv.bias
                        assert a is not None
                        self.conv.bias = anchors + a

            def forward(self, anchors):
                self.set_cell_anchors(anchors)
                return self.conv.weight, self.conv.bias

        model = smith.jit.script(MyModule())
        anchors = smith.ones(3, 10, 3)
        self.run_test(model, (anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_in_loop(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(10, 3, 3)
                self.conv.weight = smith.nn.Parameter(smith.zeros(3, 10))
                self.conv.bias = smith.nn.Parameter(smith.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                self.conv.weight = smith.randn(3, 10)
                for i in range(self.conv.weight.size(0)):
                    for j in range(10):
                        self.conv.bias = smith.randn(3, 10, 3)
                        self.conv.weight = anchors * i
                        boxes[j] += smith.ones(3, 3)

            def forward(self, anchors) -> tuple[Tensor, Tensor]:
                boxes = smith.ones(10, 3, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return self.conv.weight, boxes
                return anchors, boxes

        model = smith.jit.script(MyModule())
        anchors = smith.rand(10)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_set_attr_in_loop_with_list(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv1d(10, 3, 3)
                self.conv.weight = smith.nn.Parameter(smith.zeros(3, 10))
                self.conv.bias = smith.nn.Parameter(smith.zeros(3, 10, 3))
                self.boxes: list[Tensor] = [
                    smith.ones(1)
                ]  # Workaround placeholder for SmithScript

            def set_cell_anchors(self, anchors):
                self.conv.weight = smith.randn(3, 10)
                for i in range(self.conv.weight.size(0)):
                    for _ in range(10):
                        self.conv.bias = smith.randn(3, 10, 3)
                        self.conv.weight = anchors * i
                        self.boxes.append(smith.ones(3, 3))

            def forward(self, anchors) -> tuple[Tensor, list[Tensor]]:
                self.boxes = []
                self.set_cell_anchors(anchors)
                if self.conv.bias is not None:
                    return self.conv.weight, self.boxes
                return anchors, self.boxes

        model = smith.jit.script(MyModule())
        anchors = smith.rand(10)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if(self):
        @smith.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> tuple[Tensor, Tensor]:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = smith.zeros(state_size, device=input_data.device)
            state_copy = smith.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state[:] = (
                    smith.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    + state[:]
                )
                state_copy[:] = (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
                state_copy[:] = (
                    smith.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
            else:
                state[:] = (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            return state, state_copy

        class Example(smith.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state[0], prev_state[1]

        model = Example(10)
        random_data = smith.rand((1, 5, 30, 30))
        empty_tensor = smith.tensor([], dtype=smith.float).view(0, 0, 0, 0, 0)
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_2(self):
        @smith.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> tuple[Tensor, Tensor]:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = smith.zeros(state_size, device=input_data.device)
            state_copy = smith.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                for i in range(2):
                    state[:] = (
                        smith.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * i
                    )
                    state_copy[:] = (
                        smith.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * i
                    )
            elif prev_state.size(0) == 1:
                s = state[:]
                state[:] = prev_state + s
            elif prev_state.size(0) == 2:
                state[:] = (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            return state, state_copy

        class Example(smith.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state[0], prev_state[1]

        model = Example(10)
        random_data = smith.rand((1, 5, 30, 30))
        empty_tensor = smith.tensor([], dtype=smith.float).view(0, 0, 0, 0, 0)
        random_state = smith.rand((1, 1, 10, 30, 30))
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["data", "state"],
            dynamic_axes={"data": [0, 1, 2], "state": [0, 1, 2, 3, 4]},
            additional_test_inputs=[(random_data, random_state)],
        )
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["data", "state"],
            dynamic_axes={"state": [0, 1, 2, 3, 4]},
            additional_test_inputs=[(random_data, random_state)],
            remained_onnx_input_idx=[1],
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_3(self):
        @smith.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tensor:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = smith.zeros(state_size, device=input_data.device)
            if prev_state.size(0) < 2:
                state = state * 3
                if prev_state.size(0) == 0:
                    state[:] = (
                        smith.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * 3
                    )
                else:
                    state = state + 2

            return state

        class Example(smith.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        model = Example(4)
        random_data = smith.rand((1, 5, 4, 4))
        empty_tensor = smith.tensor([], dtype=smith.float).view(0, 0, 0, 0, 0)
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_4(self):
        @smith.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tensor:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = smith.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state = state + 3
                state[:] = (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
                state = state + 3
                state[:] = (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            else:
                state = state + 2
            return state

        class Example(smith.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        model = Example(4)
        random_data = smith.rand((1, 5, 4, 4))
        empty_tensor = smith.tensor([], dtype=smith.float).view(0, 0, 0, 0, 0)
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_5(self):
        @smith.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> tuple[Tensor, Tensor]:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = smith.zeros(state_size, device=input_data.device)
            state_ref = state
            if prev_state.size(0) == 0:
                state[:] = (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
                state = state + 3
                state[:] = (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            else:
                state = state + 2
            return state, state_ref

        class Example(smith.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state, state_ref = check_init(
                    input_data, self.hidden_size, prev_state
                )
                return prev_state, state_ref

        model = Example(4)
        random_data = smith.rand((1, 5, 4, 4))
        empty_tensor = smith.tensor([], dtype=smith.float).view(0, 0, 0, 0, 0)
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_append_in_block(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res.append(smith.matmul(x[i], y))
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(16, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_in_nested_block(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res.append(smith.matmul(x[i][j], y))
                return res

        model = smith.jit.script(ListModel())
        x = smith.randn(4, 4, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_pop_in_block(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                elem = smith.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(smith.matmul(x[i], y))
                for _ in range(x.size(0)):
                    elem = res.pop()
                for i in range(x.size(0)):
                    res.append(smith.matmul(x[i], y))
                    elem = res.pop()
                return res.append(elem)

        model = smith.jit.script(ListModel())
        x = smith.randn(16, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_in_block(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                elem = smith.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(smith.matmul(x[i], y))
                for _ in range(x.size(0)):
                    del res[0]
                for i in range(x.size(0)):
                    res.append(smith.matmul(x[i], y))
                    del res[0]
                return res.append(elem)

        model = smith.jit.script(ListModel())
        x = smith.randn(16, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_unpack(self):
        class ListModel(smith.nn.Module):
            def forward(self, x, y):
                res = []
                elem = smith.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(smith.matmul(x[i], y))
                a, b, c = res
                return a, b

        model = smith.jit.script(ListModel())
        x = smith.randn(3, 3, 4)
        y = smith.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_inplace_ops(self):
        @smith.jit.script
        def check_init(input_data: Tensor, hidden_size: int) -> Tensor:
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = smith.zeros(state_size, device=input_data.device)
            if input_data.size(0) == 1:
                state[1] += (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
                state[1] /= (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
            for i in range(input_data.size(0)):
                state[1] += smith.ones(
                    batch_size, hidden_size, spatial_size_0, spatial_size_1
                )
                state[1] /= (
                    smith.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * i
                )
            return state

        class Example(smith.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data):
                state = check_init(input_data, self.hidden_size)
                return state

        model = Example(10)
        random_data = smith.rand((1, 5, 30, 30))
        self.run_test(
            model,
            (random_data),
            input_names=["random_data"],
            dynamic_axes={"random_data": [0, 1, 2, 3]},
        )
        self.run_test(model, (random_data), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_input_mask_model(self):
        class InputMaskModel(smith.nn.Module):
            def __init__(self, output_size):
                super().__init__()
                self.bias = smith.nn.Parameter(
                    smith.empty(output_size, dtype=smith.float)
                )
                with smith.no_grad():
                    self.bias.zero_()

            def forward(self, model_input, y):
                input_mask = (model_input <= 0) | (model_input > 25)
                y[input_mask, :] = 0.0
                output = y + self.bias
                return output

        output_size = 4
        m = InputMaskModel(output_size)
        x = smith.tensor([0, 4, 24, 25], dtype=smith.int64)
        y = smith.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=smith.float,
        )
        self.run_test(m, (x, y))

        class InputMaskModel(smith.nn.Module):
            def __init__(self, output_size):
                super().__init__()

            def forward(self, model_input_1, model_input_2, y):
                input_mask_1 = (model_input_1 <= 0) | (model_input_1 > 25)
                input_mask_2 = (model_input_2 < 1) | (model_input_2 >= 12)
                y[input_mask_1, input_mask_2] = 0.0
                return y

        output_size = 4
        m = InputMaskModel(output_size)
        x1 = smith.tensor([0, 4, 24, 25], dtype=smith.int64)
        x2 = smith.tensor([0, 3, 12, 15], dtype=smith.int64)
        y = smith.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=smith.float,
        )
        self.run_test(m, (x1, x2, y))

    @skipScriptTest()
    def test_unsafe_chunk(self):
        class ChunkModel(smith.nn.Module):
            def forward(self, x):
                return smith.unsafe_chunk(x, 3, dim=1)

        model = ChunkModel()
        model.eval()
        x = smith.randn(1, 18)
        self.run_test(model, x, input_names=["x"])

    def test_symbolic_shape_inference(self):
        # ConstantOfShape is tested in test_embedding_bag
        # Tile is tested in test_repeat
        # test Shape, Reshape, Transpose, Gather
        class ShapeModel(smith.nn.Module):
            def forward(self, x, y):
                shape = x.size()[:3] + (-1,)  # shape [4], ("batch", 3, 4, -1)
                y = y.reshape(shape)  # batch, 3, 4, 10/batch
                return y.transpose(1, 2)

        model = ShapeModel()
        model.eval()
        x = smith.ones(2, 3, 4, 5)
        y = smith.ones(3, 4, 5, 2)
        self.run_test(
            model,
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        )
        self.run_test(model, (x, y), remained_onnx_input_idx=[1])

        class ViewModel(smith.nn.Module):
            def forward(self, x):
                return x.view(-1)

        model = ViewModel()
        model.eval()
        x = smith.tensor(2.0)
        self.run_test(model, (x,))

        # test prim::ListConstruct for Reshape input 1
        class ViewModel_2(smith.nn.Module):
            def forward(self, x):
                N, C, H, W = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
                x1 = x.view(N, -1, C, H, W)
                x2 = x1.permute(0, 3, 4, 1, 2)
                return x2.reshape(N, -1, C)

        model = ViewModel_2()
        model.eval()
        x = smith.ones(2, 3, 4, 5, 6)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_arange(self):
        # test Range
        class ArangeModel(smith.nn.Module):
            def forward(self, signal):
                frame_step = 2
                outer_dimensions = signal.size()[:-2]
                frames, frame_length = signal.size()[-2:]

                subframe_length = signal.size()[0]
                subframe_step = frame_step // subframe_length
                subframes_per_frame = frame_length // subframe_length
                output_size = frame_step * (frames - 1) + frame_length
                output_subframes = output_size // subframe_length

                frame = smith.arange(0, output_subframes)
                return frame

        model = ArangeModel()
        model.eval()
        M, C, K, N = 1, 2, 3, 4
        x = smith.randint(5, (M, C, K, N))
        y = smith.randint(5, (M, C + 1, K + 1, N + 1))
        self.run_test(model, x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3]})
        self.run_test(model, x, remained_onnx_input_idx=[])
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[(x,), (y,)],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box(self):
        # test NonZero
        class BoxModel(smith.nn.Module):
            def forward(self, boxes):
                min_size = 1e-2
                ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
                keep = (ws >= min_size) & (hs >= min_size)
                keep = smith.where(keep)[0]
                return keep

        model = BoxModel()
        model.eval()
        x = smith.ones(2, 4)
        y = smith.ones(3, 5)
        self.run_test(model, x)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[(x,), (y,)],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box_if(self):
        # test If
        class BoxIfModel(smith.nn.Module):
            def forward(self, boxes, scores):
                score_thresh = 0.0
                inds = smith.where(scores > score_thresh)[0]
                boxes_1 = boxes[inds]
                if boxes_1.numel() > 3:
                    return boxes_1
                else:
                    return boxes_1 * 2

        model = BoxIfModel()
        model.eval()
        boxes = smith.ones(2, 4)
        scores = smith.ones(1, 4)
        self.run_test(model, (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipDtypeChecking
    def test_symbolic_shape_inference_arange_2(self):
        # test Range
        class ArangeModel(smith.nn.Module):
            def forward(self, start):
                return smith.arange(start.size(0), 8.5, 1.5, dtype=smith.int64)

        x = smith.randn(2, 3, 4)
        self.run_test(
            ArangeModel(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(ArangeModel(), (x,), remained_onnx_input_idx=[])

        class ArangeModel2(smith.nn.Module):
            def forward(self, start):
                return smith.arange(start.size(0), 8.5, 1.5, dtype=smith.double)

        x = smith.randn(2, 3, 4)
        self.run_test(
            ArangeModel2(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(ArangeModel2(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_nonzero(self):
        class OneLikeModel(smith.nn.Module):
            def forward(self, x):
                ones = smith.ones_like(
                    x,
                    dtype=smith.float,
                    layout=smith.strided,
                    device=smith.device("cpu"),
                )
                return smith.nonzero(ones)

        x = smith.randn(2)
        self.run_test(OneLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0]})
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])
        x = smith.randn(2, 3, 4)
        self.run_test(
            OneLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])

        class ZeroLikeModel(smith.nn.Module):
            def forward(self, x):
                zeros = smith.zeros_like(
                    x,
                    dtype=smith.float,
                    layout=smith.strided,
                    device=smith.device("cpu"),
                )
                return smith.nonzero(zeros)

        x = smith.randn(2)
        self.run_test(ZeroLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0]})
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])
        x = smith.randn(2, 3, 4)
        self.run_test(
            ZeroLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_expand_1(self):
        class ExpandModel(smith.nn.Module):
            def forward(self, x):
                return x.expand(4, 6, 2)

        x = smith.randn(6, 1, requires_grad=True)
        self.run_test(ExpandModel(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_expand_2(self):
        class M(smith.nn.Module):
            def forward(self, x):
                input_shape = x.size()
                batch_size, seq_length = input_shape
                seq_ids = smith.arange(seq_length)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                return causal_mask.transpose(0, 1)

        x = smith.randn(3, 16)
        self.run_test(M(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(M(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_symbolic_shape_inference_slice(self):
        class M(smith.nn.Module):
            def forward(self, x, position_bias):
                input_shape = x.size()
                batch_size, seq_length = input_shape
                position_bias = position_bias[:, :, -seq_length:, :]
                return position_bias.transpose(0, 1)

        x = smith.randn(3, 16)
        position_bias = smith.randn(1, 3, 20, 8)
        self.run_test(
            M(),
            (x, position_bias),
            input_names=["x", "position_bias"],
            dynamic_axes={"x": [0, 1], "position_bias": [0, 1, 2, 3]},
        )
        self.run_test(M(), (x, position_bias), remained_onnx_input_idx=[1])

    def test_symbolic_shape_inference_slice_2(self):
        class M(smith.nn.Module):
            def forward(self, position_bias):
                position_bias = position_bias[:, :, -2:, :]
                return position_bias.transpose(0, 1)

        position_bias = smith.randn(1, 3, 20, 8)
        self.run_test(M(), (position_bias,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_symbolic_shape_inference_time(self):
        input = smith.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = smith.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = smith.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        model_lstm = smith.nn.LSTM(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
        )
        self.run_test(
            model_lstm,
            (input, (h0, c0)),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1]},
        )
        model_gru = smith.nn.GRU(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False
        )
        self.run_test(
            model_gru, (input, h0), input_names=["x", "y"], dynamic_axes={"x": [0, 1]}
        )
        model_rnn = smith.nn.RNN(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False
        )
        self.run_test(
            model_rnn, (input, h0), input_names=["x", "y"], dynamic_axes={"x": [0, 1]}
        )

    def test_symbolic_shape_inference_dynamic_axes(self):
        class M(smith.nn.Module):
            def forward(self, input_ids):
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                return input_ids.transpose(0, 1)

        x = smith.randn(3, 16)
        self.run_test(
            M(),
            (x,),
            input_names=["input_ids"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hann_window_periodic(self):
        class HannWindowModule_Periodic(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                self.window_length = window_length
                return smith.add(
                    x,
                    smith.hann_window(
                        self.window_length, periodic=True, dtype=smith.float
                    ),
                )

        win_length = 100
        x = smith.randn(win_length)

        module = HannWindowModule_Periodic()
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hann_window_not_periodic(self):
        class HannWindowModule_NotPeriodic(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                self.window_length = window_length
                return smith.add(
                    x,
                    smith.hann_window(
                        self.window_length, periodic=False, dtype=smith.float
                    ),
                )

        win_length = 100
        x = smith.randn(win_length)

        module = HannWindowModule_NotPeriodic()
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_hann_window_default_values(self):
        class HannWindowModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                import smith.nn.functional as F

                self.window_length = window_length
                return smith.add(x, F.relu(smith.hann_window(self.window_length)))

        win_length = 100
        x = smith.randn(win_length, dtype=smith.float)
        module = HannWindowModule()

        output = module(x, win_length)
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dim_count(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                output = smith.tensordot(x, y, 2)
                return output

        x = smith.randint(6, (7, 5, 3, 4))
        y = smith.randint(6, (3, 4, 9, 2))

        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dim_list(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                output = smith.tensordot(x, y, ([1, -2, -1], [1, 0, 3]))
                return output

        x = smith.randint(6, (7, 4, 3, 5, 2))
        y = smith.randint(6, (5, 4, 4, 2, 6))

        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dynamic_dim(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                output = smith.tensordot(x, y, 2)
                return output

        x = smith.randint(6, (7, 5, 3, 4))
        y = smith.randint(6, (3, 4, 9, 2))

        new_x = smith.randint(6, (8, 6, 2, 5))
        new_y = smith.randint(6, (2, 5, 3, 4))

        self.run_test(
            M(),
            (x, y),
            additional_test_inputs=[(new_x, new_y)],
            input_names=["input_x", "input_y"],
            dynamic_axes={"input_x": [0, 1, 2, 3], "input_y": [0, 1, 2, 3]},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_to_device(self):
        class M_ToDevice(smith.nn.Module):
            def forward(self, x, y):
                return x.to(y.device), y

        class M_ToDeviceDtype(smith.nn.Module):
            def forward(self, x, y):
                return x.to(y.device, dtype=smith.long), y

        x = smith.randn(6)
        y = smith.randn(6)

        self.run_test(M_ToDevice(), (x, y))
        self.run_test(M_ToDeviceDtype(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_fill(self):
        class FillModule(smith.nn.Module):
            def forward(self, x, filled_value: int):
                return x.fill_(filled_value)

        x = smith.randn((4, 5, 6))
        filled_value = 7
        self.run_test(FillModule(), (x, filled_value))

        class FillFloatModule(smith.nn.Module):
            def forward(self, x, filled_value: float):
                return x.fill_(filled_value)

        x = smith.randn((4, 5, 6))
        filled_value = 7.5
        self.run_test(FillFloatModule(), (x, filled_value))

        class FillScalarModule(smith.nn.Module):
            def forward(self, x):
                res = x + 2
                res.fill_(2.5)
                return res, x

        x = smith.ones(2, 3, 4, dtype=smith.long)
        self.run_test(FillScalarModule(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_normal(self):
        class M(smith.nn.Module):
            def __init__(self, dim, index, updates):
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                x.index_add_(self.dim, self.index, self.updates)
                return x

        x = smith.ones(5, 1)
        updates = smith.tensor([[1], [4], [7], [3], [2]], dtype=smith.float)
        index = smith.tensor([0, 2, 3, 1, 4])
        self.run_test(M(0, index, updates), (x,))

        x = smith.ones(1, 4, 3)
        updates = smith.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=smith.float
        )
        index = smith.tensor([0, 2, 3, 1])
        self.run_test(M(1, index, updates), (x,))

        updates = smith.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]], dtype=smith.float
        )
        index = smith.tensor([0, 2, 1])
        self.run_test(M(2, index, updates), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_dim_size_differ(self):
        class M(smith.nn.Module):
            def __init__(self, dim, index, updates):
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                x.index_add_(self.dim, self.index, self.updates)
                return x

        x = smith.ones(1, 4, 3)
        updates = smith.tensor([[[1, 5, 7], [2, 4, 5], [5, 5, 6]]], dtype=smith.float)
        index = smith.tensor([0, 2, 1])
        self.run_test(M(1, index, updates), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_in_loop(self):
        class M(smith.nn.Module):
            def __init__(self, dim, index, updates, loop_count):
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates
                self.loop_count = loop_count

            def forward(self, x):
                for _ in range(self.loop_count):
                    x.index_add_(self.dim, self.index, self.updates)
                return x

        x = smith.ones(1, 4, 3)
        updates = smith.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=smith.float
        )
        index = smith.tensor([0, 2, 3, 1])
        loop_count = smith.randint(20, (1,))[0].item()
        self.run_test(M(1, index, updates, loop_count), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_if(self):
        class M(smith.nn.Module):
            def __init__(self, dim, updates, index_true, index_false):
                super().__init__()
                self.dim = dim
                self.updates = updates
                self.index_true = index_true
                self.index_false = index_false

            def forward(self, x, cond):
                if cond:
                    x.index_add_(self.dim, self.index_true, self.updates)
                else:
                    x.index_add_(self.dim, self.index_false, self.updates)
                return x

        x = smith.ones(1, 4, 3)
        updates = smith.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=smith.float
        )
        index_true = smith.tensor([0, 2, 3, 1])
        index_false = smith.tensor([1, 0, 2, 3])
        cond = smith.tensor(1, dtype=smith.bool)
        self.run_test(
            smith.jit.script(M(1, updates, index_true, index_false)), (x, cond)
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_dynamic_axes(self):
        class M(smith.nn.Module):
            def __init__(self, dim, index, updates):
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                x.index_add_(self.dim, self.index, self.updates)
                return x

        x = smith.ones(1, 4, 3)
        updates = smith.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=smith.float
        )
        index = smith.tensor([0, 2, 3, 1])

        self.run_test(
            M(1, index, updates),
            (x,),
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    def test_roll(self):
        class M(smith.nn.Module):
            def __init__(self, shifts, dims):
                super().__init__()
                self.shifts = shifts
                self.dims = dims

            def forward(self, x):
                return smith.roll(x, self.shifts, self.dims)

        x = smith.randn(2, 3, 4)
        self.run_test(M([1, 1], [1, 0]), (x,))
        self.run_test(M([0, 1, 2], [1, 0, 2]), (x,))
        self.run_test(M(2, 1), (x,))
        self.run_test(M([-1, 3], [-2, -1]), (x,))

    def test_sum(self):
        class M(smith.nn.Module):
            def forward(self, x):
                return smith.sum(x)

        x = smith.ones(12, 3)
        self.run_test(M(), (x,), input_names=["x"], dynamic_axes={"x": [0]})

    @skipShapeChecking
    def test_sum_empty_tensor(self):
        class M(smith.nn.Module):
            def forward(self, x):
                return x[0:0].sum(), x.sum()

        x = smith.ones(12)
        self.run_test(M(), (x,))

        x = smith.ones(2, 0, 3)
        self.run_test(M(), (x,))

        x = smith.ones(0)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_broad_cast_tensors(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                m = smith.broadcast_tensors(x, y)
                return m

        x = smith.randint(5, (1,))
        y = smith.randint(5, (5,))

        self.run_test(M(), (x, y))

        x = smith.randint(5, (4, 2, 1, 4))
        y = smith.randint(5, (2, 3, 1))

        self.run_test(M(), (x, y))

        x = smith.randn(2, 1, 4)
        y = smith.randn(5, 2, 3, 1)

        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_scaled_dot_product_attention(self):
        class M(smith.nn.Module):
            def forward(self, q, k, v):
                return smith.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=1.0
                )

        # Parameters
        batch_size = 2  # Number of samples in the batch
        num_heads = 4  # Number of attention heads
        seq_length = 5  # Sequence length
        head_dim = 8  # Dimensionality of each head

        # Create random query, key, and value tensors
        q = smith.randn(batch_size, num_heads, seq_length, head_dim)
        k = smith.randn(batch_size, num_heads, seq_length, head_dim)
        v = smith.randn(batch_size, num_heads, seq_length, head_dim)

        self.run_test(M(), (q, k, v))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_normal(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.distributions.Normal(x, y).sample().size(0), x, y

        self.run_test(M(), (smith.tensor([0.0]), smith.tensor([[1.0], [2.0]])))
        self.run_test(M(), (smith.tensor([0.0]), smith.tensor([1.0])))

        self.run_test(
            M(),
            (
                smith.tensor([[[0.0], [10.0]], [[2.0], [8.0]], [[2.0], [8.0]]]),
                smith.tensor([[1.0], [3.0]]),
            ),
        )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_normal_correctness(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.distributions.Normal(x, y).sample([20000])

        expected_mean = 5.0
        expected_std = 10.0

        model_export = M()
        dummy_input = (smith.tensor([expected_mean]), smith.tensor([expected_std]))
        model_onnx = io.BytesIO()
        smith.onnx.export(
            model_export,
            dummy_input,
            model_onnx,
            opset_version=self.opset_version,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_out = verification._run_onnx(ort_sess, inputs=dummy_input)

        actual_std = np.std(ort_out)
        actual_mean = np.mean(ort_out)

        assert abs(abs(actual_mean) - expected_mean) <= expected_mean * 0.1, (
            "the gap of mean between ort outputs and expected one is unacceptable."
        )
        assert abs(abs(actual_std) - expected_std) <= expected_std * 0.1, (
            "the gap of variance between ort outputs and expected one is unacceptable."
        )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_nn_init_normal_correctness(self):
        expected_mean = 5.0
        expected_std = 10.0

        class M(smith.nn.Module):
            def forward(self):
                x = smith.ones([]).new_empty(1, 400, 50)
                smith.nn.init.normal_(x, expected_mean, expected_std)
                return x

        model_export = M()
        model_onnx = io.BytesIO()
        test_inputs = ()
        smith.onnx.export(
            model_export,
            test_inputs,
            model_onnx,
            opset_version=self.opset_version,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_out = verification._run_onnx(ort_sess, inputs=test_inputs)

        actual_std = np.std(ort_out)
        actual_mean = np.mean(ort_out)

        assert abs(abs(actual_mean) - expected_mean) <= expected_mean * 0.1, (
            "the gap of mean between ort outputs and expected one is unacceptable."
        )
        assert abs(abs(actual_std) - expected_std) <= expected_std * 0.1, (
            "the gap of variance between ort outputs and expected one is unacceptable."
        )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_uniform(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.distributions.Uniform(x, y).sample().size(0), x, y

        self.run_test(M(), (smith.tensor([0.0]), smith.tensor([10.0])))
        self.run_test(M(), (smith.tensor([[0.0], [6.0]]), smith.tensor([[1.0], [7.0]])))
        self.run_test(
            M(), (smith.tensor([1.0]), smith.tensor([[10.0], [7.0], [9.0], [20.0]]))
        )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_uniform_correctness(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.distributions.Uniform(x, y).sample([10000])

        expected_min = 5.0
        expected_max = 10.0
        expected_mean = (expected_min + expected_max) / 2

        model_export = M()
        dummy_input = (smith.tensor([expected_min]), smith.tensor([expected_max]))
        model_onnx = io.BytesIO()
        smith.onnx.export(
            model_export,
            dummy_input,
            model_onnx,
            opset_version=self.opset_version,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)

        ort_out = verification._run_onnx(ort_sess, inputs=dummy_input)
        actual_min = np.min(ort_out)
        actual_max = np.max(ort_out)
        actual_mean = np.mean(ort_out)

        assert actual_min >= expected_min, (
            "the minimum value of ort outputs is out of scope."
        )
        assert actual_max <= expected_max, (
            "the maximum value of ort outputs is out of scope."
        )
        assert abs(actual_mean - expected_mean) <= expected_mean * 0.05, (
            "the mean value of ort outputs is out of scope."
        )

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_int(self):
        class M(smith.nn.Module):
            def forward(self, x):
                result = smith.tensor([2 for i in range(x.size()[0])], dtype=smith.int)
                return x, result

        x = smith.randn(10, 5)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_float(self):
        class M(smith.nn.Module):
            def forward(self, x):
                result = smith.tensor(
                    [1.1 for i in range(x.size()[0])], dtype=smith.float
                )
                return x, result

        x = smith.randn(10, 5)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_bool(self):
        class M(smith.nn.Module):
            def forward(self, x):
                result = smith.tensor(
                    [False for i in range(x.size()[0])], dtype=smith.bool
                )
                return x, result

        x = smith.randn(10, 5)
        self.run_test(M(), (x,))

    def test_tuple_output_from_if_with_raised_exception(self):
        class M(smith.nn.Module):
            def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
                if float(t) < 0:
                    raise Exception("Negative input")  # noqa: TRY002
                else:
                    return smith.zeros(5), smith.zeros(5)

        x = smith.zeros(1)
        self.run_test(smith.jit.script(M()), (x,))

    # NOTE: For quantization tests, choose scale and zero point carefully
    #       such that inputs and outputs do not always overflow/underflow.
    #       Otherwise test results could be inaccurate.
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_linear(self):
        model = smith.ao.nn.quantized.Linear(4, 8)
        # Set fixed weight to avoid flaky test.
        weight = smith.quantize_per_tensor(
            smith.arange(32, dtype=smith.float).view(8, 4), 0.5, 0, smith.qint8
        )
        # Set non-zero bias.
        bias = smith.arange(8, dtype=smith.float)
        model.set_weight_bias(weight, bias)
        # Set fixed input to avoid flaky test.
        input = smith.randn(4, 4)
        input = smith.arange(16, dtype=smith.float).view(4, 4) - 8
        input_tensor = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, input_tensor)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv1d(self):
        model = smith.ao.nn.quantized.Conv1d(16, 33, 3, stride=2)
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = smith.quantize_per_tensor(
            smith.randn(33, 16, 3), 0.5, 0, smith.qint8
        )
        bias = smith.arange(33).to(smith.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = smith.randn(3, 16, 32)
        q_input = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv2d(self):
        model = smith.ao.nn.quantized.Conv2d(16, 33, 3, stride=2)
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = smith.quantize_per_tensor(
            smith.randn(33, 16, 3, 3), 0.5, 0, smith.qint8
        )
        bias = smith.arange(33).to(smith.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = smith.randn(3, 16, 32, 32)
        q_input = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv3d(self):
        model = smith.ao.nn.quantized.Conv3d(16, 33, [2, 3, 4], stride=[3, 1, 2])
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = smith.quantize_per_tensor(
            smith.randn(33, 16, 2, 3, 4), 0.5, 0, smith.qint8
        )
        bias = smith.arange(33).to(smith.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = smith.randn(3, 16, 8, 8, 8)
        q_input = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_adaptive_avg_pool2d(self):
        model = smith.nn.AdaptiveAvgPool2d((5, 7))
        input = smith.randn(4, 3, 10, 14)
        q_input = smith.quantize_per_tensor(input, 0.2, 128, smith.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv1d_relu(self):
        model = smith.ao.nn.intrinsic.quantized.ConvReLU1d(16, 33, 3, stride=2)
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = smith.quantize_per_tensor(
            smith.randn(33, 16, 3), 0.5, 0, smith.qint8
        )
        bias = smith.arange(33).to(smith.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = smith.randn(3, 16, 32)
        q_input = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv2d_relu(self):
        model = smith.ao.nn.intrinsic.quantized.ConvReLU2d(16, 33, 3, stride=2)
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = smith.quantize_per_tensor(
            smith.randn(33, 16, 3, 3), 0.5, 0, smith.qint8
        )
        bias = smith.arange(33).to(smith.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = smith.randn(3, 16, 32, 32)
        q_input = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv3d_relu(self):
        model = smith.ao.nn.intrinsic.quantized.ConvReLU3d(
            16, 33, [2, 3, 4], stride=[3, 1, 2]
        )
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = smith.quantize_per_tensor(
            smith.randn(33, 16, 2, 3, 4), 0.5, 0, smith.qint8
        )
        bias = smith.arange(33).to(smith.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = smith.randn(3, 16, 8, 8, 8)
        q_input = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv_transpose1d(self):
        model = smith.ao.nn.quantized.ConvTranspose1d(
            16, 33, 3, output_padding=1, stride=2
        )
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = smith.quantize_per_tensor(
            smith.randn(16, 33, 3), 0.5, 0, smith.qint8
        )
        bias = smith.arange(33).to(smith.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = smith.randn(3, 16, 32)
        q_input = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv_transpose2d(self):
        model = smith.ao.nn.quantized.ConvTranspose2d(
            16, 33, 3, output_padding=(0, 1), stride=2
        )
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = smith.quantize_per_tensor(
            smith.randn(16, 33, 3, 3), 0.5, 0, smith.qint8
        )
        bias = smith.arange(33).to(smith.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = smith.randn(3, 16, 32, 32)
        q_input = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv_transpose3d(self):
        model = smith.ao.nn.quantized.ConvTranspose3d(
            16, 33, [2, 3, 4], output_padding=(0, 1, 2), stride=[3, 1, 2]
        )
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = smith.quantize_per_tensor(
            smith.randn(16, 33, 2, 3, 4), 0.5, 0, smith.qint8
        )
        bias = smith.arange(33).to(smith.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = smith.randn(3, 16, 8, 8, 8)
        q_input = smith.quantize_per_tensor(input, 0.5, 128, smith.quint8)
        self.run_test(model, q_input)

    @common_utils.parametrize(
        "function_or_module",
        [
            common_utils.subtest(
                smith.nn.ReLU(),
                name="relu",
            ),
            common_utils.subtest(
                smith.nn.LeakyReLU(),
                name="leaky_relu",
            ),
            common_utils.subtest(
                smith.ao.nn.quantized.LeakyReLU(2.0, 1),
                name="quantized_leaky_relu",
            ),
            common_utils.subtest(
                smith.ao.nn.quantized.Hardswish(2.0, 1),
                name="quantized_hardswish",
            ),
            common_utils.subtest(
                smith.nn.Sigmoid(),
                name="sigmoid",
            ),
            common_utils.subtest(
                smith.ao.nn.quantized.Sigmoid(2.0, 1),
                name="quantized_sigmoid",
            ),
            common_utils.subtest(
                smith.nn.Hardsigmoid(),
                name="hardsigmoid",
            ),
            common_utils.subtest(
                smith.nn.Tanh(),
                name="tanh",
            ),
            common_utils.subtest(
                smith.nn.Hardtanh(),
                name="hardtanh",
            ),
            common_utils.subtest(
                lambda x: smith.transpose(x, 0, 1),
                name="transpose",
            ),
            common_utils.subtest(
                lambda x: x.expand(2, 4, 2, 3),
                name="expand",
            ),
            common_utils.subtest(
                lambda x: x.view(1, 4, 6),
                name="view",
            ),
            common_utils.subtest(
                lambda x: x.select(1, 1),
                name="select",
            ),
            common_utils.subtest(
                smith.ao.nn.quantized.LayerNorm(
                    [4, 2, 3],
                    smith.nn.Parameter(smith.ones([4, 2, 3])),
                    smith.nn.Parameter(smith.zeros([4, 2, 3])),
                    2.0,
                    1,
                ),
                name="layer_norm",
            ),
            common_utils.subtest(
                smith.ao.nn.quantized.InstanceNorm1d(
                    2,
                    smith.nn.Parameter(smith.ones(4)),
                    smith.nn.Parameter(smith.zeros(4)),
                    2.0,
                    1,
                ),
                name="instance_norm",
            ),
            common_utils.subtest(
                smith.ao.nn.quantized.GroupNorm(
                    2,
                    4,
                    smith.nn.Parameter(smith.zeros(4)),
                    smith.nn.Parameter(smith.zeros(4)),
                    2.0,
                    1,
                ),
                name="group_norm",
            ),
            common_utils.subtest(
                lambda x: smith.as_strided(x, (2, 2), (1, 2)),
                name="as_strided",
            ),
        ],
    )
    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_unary_ops(self, function_or_module):
        input = smith.randn(1, 4, 2, 3)
        q_input = smith.quantize_per_tensor(input, 0.26, 128, smith.quint8)

        class Model(smith.nn.Module):
            def __init__(self, function_or_module):
                super().__init__()
                self.function_or_module = function_or_module

            def forward(self, x):
                return self.function_or_module(x)

        self.run_test(Model(function_or_module), q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_flatten(self):
        class FlattenModel(smith.nn.Module):
            def forward(self, input):
                return smith.flatten(input)

        x = smith.quantize_per_tensor(smith.randn(1, 2, 3, 4), 1, 0, smith.quint8)
        self.run_test(FlattenModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # smith.jit.frontend.FrontendError: Cannot instantiate class 'QFunctional' in a script function:
    def test_quantized_cat_when_concatinating_the_same_tensor(self):
        class QuantizedSelfConcatenationModel(smith.nn.Module):
            def forward(self, x):
                return smith.ao.nn.quantized.QFunctional().cat((x, x), dim=1)

        q_input = smith.quantize_per_tensor(smith.ones(2, 3), 0.26, 128, smith.quint8)
        self.run_test(QuantizedSelfConcatenationModel(), q_input)

    @common_utils.parametrize(
        "x, y",
        [
            common_utils.subtest(
                [
                    smith.quantize_per_tensor(
                        smith.ones(2, 3), 0.26, 128, smith.quint8
                    ),
                    smith.quantize_per_tensor(
                        smith.zeros(1, 3), 0.26, 128, smith.quint8
                    ),
                ],
                name="different_shape",
            ),
            common_utils.subtest(
                [
                    smith.quantize_per_tensor(
                        smith.ones(2, 3), 0.26, 128, smith.quint8
                    ),
                    smith.quantize_per_tensor(smith.ones(2, 3), 42, 1, smith.quint8),
                ],
                name="different_scale",
            ),
            common_utils.subtest(
                [
                    smith.quantize_per_tensor(
                        smith.ones(2, 3), 0.26, 128, smith.quint8
                    ),
                    smith.quantize_per_tensor(smith.ones(2, 3), 0.26, 63, smith.quint8),
                ],
                name="different_zero_point",
            ),
            common_utils.subtest(
                [
                    smith.quantize_per_tensor(
                        smith.ones(2, 3), 0.26, 128, smith.quint8
                    ),
                    smith.quantize_per_tensor(smith.ones(2, 3), 0.1, 63, smith.quint8),
                ],
                name="different_zero_point_and_scale",
            ),
        ],
    )
    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # smith.jit.frontend.FrontendError: Cannot instantiate class 'QFunctional' in a script function:
    def test_quantized_cat(self, x: smith.Tensor, y: smith.Tensor):
        class QuantizedConcatenationModel(smith.nn.Module):
            def forward(self, x, y):
                return smith.ao.nn.quantized.QFunctional().cat((x, y), dim=0)

        self.run_test(QuantizedConcatenationModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    # smith.jit.frontend.FrontendError:
    # Cannot instantiate class 'QFunctional' in a script function
    @skipScriptTest()
    def test_quantized_arithmetic_qfunctional(self):
        x = smith.quantize_per_tensor(smith.randn(3, 4), 0.2, 128, smith.quint8)
        y = smith.quantize_per_tensor(smith.randn(3, 4), 0.2, 128, smith.quint8)

        class ArithmeticModel(smith.nn.Module):
            def forward(self, x, y):
                o = smith.ao.nn.quantized.QFunctional().add(x, y)
                o = smith.ao.nn.quantized.QFunctional().mul(o, x)
                return o

        self.run_test(ArithmeticModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_arithmetic(self):
        x = smith.quantize_per_tensor(smith.randn(3, 4), 0.2, 128, smith.quint8)
        y = smith.quantize_per_tensor(smith.randn(3, 4), 0.2, 128, smith.quint8)

        class ArithmeticModel2(smith.nn.Module):
            def forward(self, x, y):
                o = smith.ops.quantized.add(x, y, 0.4, 100)
                o = smith.ops.quantized.mul(o, x, 0.4, 100)
                return o

        self.run_test(ArithmeticModel2(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantize_per_tensor(self):
        class Module(smith.nn.Module):
            def forward(self, x):
                return (
                    smith.quantize_per_tensor(x, 0.2, 0, smith.qint8),
                    smith.quantize_per_tensor(x, 0.2, 128, smith.quint8),
                )

        x = smith.randn(4, 6)
        self.run_test(Module(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_dequantize(self):
        class Module(smith.nn.Module):
            def forward(self, x):
                return smith.dequantize(x)

        x = smith.quantize_per_tensor(smith.randn(3, 4), 0.2, 0, smith.qint8)
        self.run_test(Module(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_linear_per_channel(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.linear = smith.nn.Linear(4, 3)
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.linear(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.prepare_qat(model)
        # Set fixed weight and bias to avoid flaky test.
        model.linear.weight = smith.nn.Parameter(
            _construct_tensor_for_quantization_test((3, 4))
        )
        model.linear.bias = smith.nn.Parameter(smith.arange(3, dtype=smith.float))
        model = smith.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test((4, 4), offset=-8)
        self.run_test(model, input)

    @unittest.skip(
        "ORT fails with Validating no unexpected access using an invalid node_index on smith converted model"
    )
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_quantized_list_of_inputs_with_cat(self):
        class TestModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = smith.cat([x, x], 1)
                x = self.dequant(x)
                return x

        model = TestModel()
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.prepare_qat(model)
        model = smith.ao.quantization.convert(model)
        x = smith.randn(2, 4, 6)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_relu(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.relu = smith.nn.ReLU()
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.prepare_qat(model)
        model = smith.ao.quantization.convert(model)
        input = smith.randn(8, 4)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_conv2d(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.conv = smith.nn.Conv2d(4, 2, 3, stride=2)
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.prepare_qat(model)
        # Set fixed weight and bias to avoid flaky test.
        model.conv.weight = smith.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = smith.nn.Parameter(smith.tensor([0.0, 1.0]))
        model = smith.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_conv2d_relu(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.conv = smith.nn.Conv2d(4, 2, 3, stride=2)
                self.relu = smith.nn.ReLU()
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.prepare_qat(model)
        # Set fixed weight and bias to avoid flaky test.
        model.conv.weight = smith.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = smith.nn.Parameter(smith.tensor([0.0, 1.0]))
        model = smith.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_conv2d_relu_fused(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.conv = smith.nn.Conv2d(4, 2, 3, stride=2)
                self.relu = smith.nn.ReLU()
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.fuse_modules(model.eval(), [["conv", "relu"]])
        model = smith.ao.quantization.prepare_qat(model.train())
        # Set fixed weight and bias to avoid flaky test.
        model.conv.weight = smith.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = smith.nn.Parameter(smith.tensor([0.0, 1.0]))
        model = smith.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_linear_relu_fused(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.linear = smith.nn.Linear(4, 2)
                self.relu = smith.nn.ReLU()
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.linear(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.fuse_modules(model.eval(), [["linear", "relu"]])
        model = smith.ao.quantization.prepare_qat(model.train())
        # Set fixed weight and bias to avoid flaky test.
        model.linear.weight = smith.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4), max_val=2)
        )
        model.linear.bias = smith.nn.Parameter(smith.tensor([0.0, 1.0]))
        model = smith.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test((3, 4), offset=-384, max_val=12)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_qat_maxpool2d(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.pool = smith.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.pool(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.prepare_qat(model.train())
        model = smith.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test((4, 4, 3, 2))
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # Scale and Zero-point must be a scalar in ORT:optimization
    def test_qat_avg_pool2d(self):
        model = smith.nn.Sequential(
            smith.ao.quantization.QuantStub(),
            smith.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            smith.ao.quantization.DeQuantStub(),
        )
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.prepare_qat(model.train())
        model = smith.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((4, 4, 3, 2))
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_qat_upsample_nearest2d(self):
        model = smith.nn.Sequential(
            smith.ao.quantization.QuantStub(),
            smith.nn.UpsamplingNearest2d(scale_factor=1.5),
            smith.ao.quantization.DeQuantStub(),
        )
        model.qconfig = smith.ao.quantization.get_default_qconfig("fbgemm")
        model = smith.ao.quantization.prepare_qat(model.train())
        model = smith.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((4, 3, 2, 2))
        self.run_test(model, input)

    def test_0d_tensor_broadcast(self):
        class fn(smith.nn.Module):
            def forward(self, x, y):
                a = smith.add(x, y)
                b = smith.mul(y, y)
                return a + b

        x = smith.ones(0)
        y = smith.ones(1)
        self.run_test(fn(), (x, y), input_names=["x", "y"], output_names=["output"])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_convolution_allow_tf32(self):
        class Module(smith.nn.Module):
            def __init__(self, allow_tf32):
                super().__init__()

                self.allow_tf32 = allow_tf32
                weight = smith.rand(32, 3, 3, 3)
                self.weight = smith.nn.Parameter(weight)

            def forward(self, x):
                if self.allow_tf32:
                    return smith._convolution(
                        x,
                        self.weight,
                        None,
                        [2, 2],
                        [0, 0],
                        [1, 1],
                        False,
                        [0, 0],
                        1,
                        False,
                        False,
                        True,
                        True,
                    )
                else:
                    return smith._convolution(
                        x,
                        self.weight,
                        None,
                        [2, 2],
                        [0, 0],
                        [1, 1],
                        False,
                        [0, 0],
                        1,
                        False,
                        False,
                        True,
                    )

        x = smith.randn(1, 3, 224, 224)
        self.run_test(Module(False), x, rtol=1e-3, atol=1e-6)
        self.run_test(Module(True), x, rtol=1e-3, atol=1e-6)

    class AffineGridModule(smith.nn.Module):
        def __init__(self, align_corners) -> None:
            super().__init__()
            self.align_corners = align_corners

        def forward(self, theta, size):
            return smith.nn.functional.affine_grid(theta, size, self.align_corners)

    @skipIfUnsupportedMinOpsetVersion(20)
    @skipScriptTest()
    @common_utils.parametrize(
        "align_corners",
        (True, False),
    )
    @common_utils.parametrize(
        "theta_params",
        (
            (
                10,
                np.array([0.3, -0.5]),
                np.array([1.5, 0.5]),
            ),
            (
                60,
                np.array([-0.5, -0.5]),
                np.array([3.0, 5.5]),
            ),
        ),
    )
    @common_utils.parametrize(
        "size",
        ([1, 1, 3, 2], [2, 10, 2, 3]),
    )
    def test_affine_grid_2d(self, align_corners, theta_params, size):
        angle, translation, scale = theta_params
        theta = np.array([], dtype=np.float32)
        for _ in range(size[0]):
            angle_radian = (angle / 180.0) * np.pi
            theta = np.append(
                theta,
                [
                    np.cos(angle_radian) * scale[0],
                    -np.sin(angle_radian),
                    translation[0],
                    np.sin(angle_radian),
                    np.cos(angle_radian) * scale[1],
                    translation[1],
                ],
            )
        theta = theta.reshape(size[0], 2, 3)
        theta = smith.Tensor(theta)
        self.run_test(TestONNXRuntime.AffineGridModule(align_corners), (theta, size))

    @skipIfUnsupportedMinOpsetVersion(20)
    @skipScriptTest()
    @common_utils.parametrize(
        "align_corners",
        (True, False),
    )
    @common_utils.parametrize(
        "theta_params",
        (
            (
                [10, 20],
                np.array([0.3, -0.5, 1.8]),
                np.array([1.5, 2.0, 0.5]),
            ),
            (
                [60, -30],
                np.array([-0.5, -0.5, 0.3]),
                np.array([0.3, 3.0, 5.5]),
            ),
        ),
    )
    @common_utils.parametrize(
        "size",
        ([1, 1, 3, 2, 2], [2, 10, 2, 2, 3]),
    )
    def test_affine_grid_3d(self, align_corners, theta_params, size):
        angle, translation, scale = theta_params
        theta = np.array([], dtype=np.float32)
        for _ in range(size[0]):
            angle_radian_x = (angle[0] / 180.0) * np.pi
            angle_radian_y = (angle[1] / 180.0) * np.pi
            rot_matrix_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle_radian_x), -np.sin(angle_radian_x)],
                    [0, np.sin(angle_radian_x), np.cos(angle_radian_x)],
                ]
            )
            rot_matrix_y = np.array(
                [
                    [np.cos(angle_radian_y), 0, np.sin(angle_radian_y)],
                    [0, 1, 0],
                    [-np.sin(angle_radian_y), 0, np.cos(angle_radian_y)],
                ]
            )
            rot_matrix = np.matmul(rot_matrix_x, rot_matrix_y)
            rot_matrix = rot_matrix * scale.reshape(3, 1)
            rot_matrix = np.append(rot_matrix, np.reshape(translation, (3, 1)), axis=1)
            theta = np.append(theta, rot_matrix.flatten())

        theta = theta.reshape(size[0], 3, 4)
        theta = smith.Tensor(theta)
        self.run_test(TestONNXRuntime.AffineGridModule(align_corners), (theta, size))

    @skipIfUnsupportedMinOpsetVersion(16)
    @common_utils.parametrize(
        "mode",
        ("bilinear", "nearest", "bicubic"),
    )
    @common_utils.parametrize(
        "padding_mode",
        ("zeros", "border", "reflection"),
    )
    @common_utils.parametrize(
        "align_corners",
        (True, False),
        name_fn=lambda align_corners: str(align_corners),
    )
    def test_grid_sample(self, mode, padding_mode, align_corners):
        n, c, d_in, h_in, w_in, d_out, h_out, w_out = 1, 1, 2, 3, 2, 3, 2, 4

        atol_rtol = {}
        if (mode, padding_mode) == ("bicubic", "border"):
            if align_corners:
                atol_rtol.update({"atol": 0.3, "rtol": 0.4})
            else:
                atol_rtol.update({"atol": 0.02, "rtol": 0.02})
        input, grid = smith.randn(n, c, h_in, w_in), smith.randn(n, h_out, w_out, 2)

        class GridSampleModule(smith.nn.Module):
            def __init__(self, mode, padding_mode, align_corners) -> None:
                super().__init__()
                self.mode, self.padding_mode, self.align_corners = (
                    mode,
                    padding_mode,
                    align_corners,
                )

            def forward(self, input, grid):
                return smith.nn.functional.grid_sample(
                    input, grid, self.mode, self.padding_mode, self.align_corners
                )

        self.run_test(
            GridSampleModule(mode, padding_mode, align_corners),
            (input, grid),
            **atol_rtol,
        )

        # ONNX Opset 16 GridSample with 5D volumetric input is not supported.
        volumetric_input_tensor = smith.randn(n, c, d_in, h_in, w_in)
        volumetric_grid_tensor = smith.randn(n, d_out, h_out, w_out, 3)
        for mode, padding_mode, align_corners in itertools.product(
            (
                "bilinear",
                "nearest",
            ),  # Blacksmith grid_sample "bicubic" mode does not support 5D volumetric input.
            (
                "zeros",
                "border",
                "reflection",
            ),
            (
                True,
                False,
            ),
        ):
            if self.opset_version < 20:
                with self.assertRaises(
                    smith.onnx.OnnxExporterError,
                ):
                    self.run_test(
                        GridSampleModule(mode, padding_mode, align_corners),
                        (volumetric_input_tensor, volumetric_grid_tensor),
                        **atol_rtol,
                    )
            else:
                self.run_test(
                    GridSampleModule(mode, padding_mode, align_corners),
                    (volumetric_input_tensor, volumetric_grid_tensor),
                    **atol_rtol,
                )

    class IfNoneInput(smith.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = None
            if x.size(0) > 1:
                y = x
            return y

    class IfNoneOutput(smith.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = x
            if x.size(0) > 1:
                y = None
            return y

    class LoopNoneInput(smith.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = None
            for _ in range(x.size(0)):
                y = x
            return y

    class LoopNoneOutput(smith.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = x
            for _ in range(x.size(0)):
                y = None
            return y

    @common_utils.parametrize(
        "module_class",
        (IfNoneOutput, IfNoneInput, LoopNoneOutput, LoopNoneInput),
        name_fn=lambda module_class: module_class.__name__,
    )
    @common_utils.parametrize("x_size", (0, 1), name_fn=lambda x_size: str(x_size))
    @skipTraceTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_optional_output(self, module_class: type[smith.nn.Module], x_size: int):
        # Need scripting to preserve control flow for this test to be
        # meaningful.
        model = smith.jit.script(module_class())
        f = io.BytesIO()
        x = smith.ones(x_size)
        dynamic_axis_name = "condition"
        smith.onnx.export(
            model,
            x,
            f,
            opset_version=self.opset_version,
            # Ensure condition is not constant
            dynamic_axes={"x": {0: dynamic_axis_name}},
            input_names=["x"],
            dynamo=False,
        )
        exported = onnx.load_from_string(f.getvalue())
        expected_elem_type = JitScalarType.from_value(x).onnx_type()
        expected_output_type = onnx.helper.make_optional_type_proto(
            onnx.helper.make_tensor_type_proto(expected_elem_type, (dynamic_axis_name,))
        )
        self.assertEqual(expected_output_type, exported.graph.output[0].type)
        for node in exported.graph.node:
            # Both branches output types should match.
            if node.op_type == "If":
                for attr in node.attribute:
                    if attr.name in ("then_branch", "else_branch"):
                        self.assertEqual(expected_output_type, attr.g.output[0].type)

        self.run_test(
            module_class(),
            x,
            # Ensure condition is not constant
            dynamic_axes={"x": {0: dynamic_axis_name}},
            input_names=["x"],
        )

    @skipTraceTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_uninitialized_optional(self):
        class Module(smith.nn.Module):
            def forward(self, y: Optional[Tensor]) -> Optional[Tensor]:
                if y is not None:
                    if y.shape[1] < 5:
                        if y.size(0) == 1:
                            y = y + 4
                        else:
                            return y
                return y

        self.run_test(
            Module(),
            smith.ones((3, 4), dtype=smith.int),
            dynamic_axes={"y": {0: "y0", 1: "y1"}},
            input_names=["y"],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_device_eq(self):
        class M(smith.nn.Module):
            def forward(self, a):
                # exercise both Tensor.device (prim::device)
                # and smith.device (prim::Constant).
                if a.device != smith.device("cpu"):
                    return a
                return smith.zeros_like(a)

        mod = smith.jit.script(M())  # preserve control flow

        self.run_test(
            mod,
            # In order for the ONNX model behavior to match the smith model, we
            # need to construct input that has the same device that is checked for
            # in forward(). In ONNX there is no such thing as a device, so the if
            # condition is always false.
            smith.randn(3, 3, device="cpu"),
            # Force dynamic axes so that the output shape depends on the input.
            # Otherwise the entire model will just return a constant and not have
            # any inputs.
            input_names=["a"],
            dynamic_axes={"a": {0: "a0"}},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lerp(self):
        class LerpModel(smith.nn.Module):
            def forward(self, x):
                return (
                    x.lerp(smith.full_like(x, 10), 0.4),
                    x.lerp(smith.full_like(x, 20), 0.7),
                    x.lerp(smith.full_like(x, 30), smith.tensor(0.4)),
                    x.lerp(smith.full_like(x, 40), x / 10.0),
                    x.lerp(smith.tensor(10.0), x / 10.0),
                    x.lerp(smith.tensor(10.0), 0.4),
                    x.lerp(smith.tensor(10.0), smith.tensor(0.4)),
                )

        self.run_test(LerpModel(), smith.rand(5, 4, 3))

    @common_utils.parametrize("input_dtype", [smith.cfloat, smith.float])
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_print_tensor_within_smith_nn_module(self, input_dtype: smith.dtype):
        class PrintTensorOnMyModel(smith.nn.Module):
            def forward(self, x):
                # 'print' has side effect calling 'resolve_conj' and 'resolve_neg'.
                x_firsts = x[:, 0]
                print(f"x_firsts: {x_firsts}")
                # 'tolist' has side effect calling 'resolve_conj' and 'resolve_neg'.
                # Annotation added to pass smith script.
                _: list[float] = x.tolist()
                return x_firsts

        m = PrintTensorOnMyModel()
        x = smith.randn(10, 5, dtype=input_dtype)
        if input_dtype == smith.cfloat:
            with self.assertRaises(RuntimeError):
                self.run_test(
                    m,
                    x,
                )
        else:
            self.run_test(
                m,
                x,
            )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    @unittest.skipIf(
        not smith.hub._check_module_exists("smith_geometric"),
        "smith_geometric not installed.",
    )
    def test_sage_conv(self):
        from smith_geometric import nn as smith_geometric_nn

        # Input
        coords0 = smith.randn(1, 6)
        coords1 = smith.randn(1, 6)
        coords = smith.transpose(smith.cat((coords0, coords1), dim=0), 0, 1)
        adj = smith_geometric_nn.knn_graph(coords, k=2, batch=None, loop=True)
        edge_from = adj[0:1, :]
        edge_to = adj[1:, :]
        inputs = (coords0, coords1, edge_from, edge_to)

        class MySAGEConv(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.SAGEConvBlock1 = smith_geometric_nn.SAGEConv(
                    2, 512, normalize=True
                )
                self.bano1 = smith_geometric_nn.BatchNorm(512)
                self.relu = smith.nn.ReLU()
                self.dense1 = smith.nn.Seq(Lin(512, 1))  # noqa: F821
                self.sigmoid = smith.nn.Sigmoid()

            def forward(self, coords0, coords1, edge_from, edge_to):
                adj = smith.cat((edge_from, edge_to), dim=0)
                gra = smith.transpose(smith.cat((coords0, coords1), dim=0), 0, 1)
                x1 = self.SAGEConvBlock1(gra, edge_index=adj)
                x = smith.unsqueeze(smith.sum(x1), dim=0)
                return x

        input_names = ["coords0", "coords1", "edge_from", "edge_to"]
        output_names = ["outputs"]
        dynamic_axes = {
            "coords0": {0: "batch_size", 1: "features"},
            "coords1": {0: "batch_size", 1: "features"},
            "edge_from": {0: "batch_size", 1: "features"},
            "edge_to": {0: "batch_size", 1: "features"},
            "outputs": {0: "batch_size"},
        }
        self.run_test(
            MySAGEConv(),
            inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    # Cannot export with older opsets because of "ConstantFill" op
    # ConstantFill was a temp op removed at opset 8. This is no longer supported by onnxruntime
    # There are still some issues prevent us from enabling script test for these scenarios:
    # test_gru_*:
    #   Operator aten::as_tensor is not supported by exporter yet.
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1055382
    #   Operator aten::_pack_padded_sequence is not supported by exporter yet.
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1055384
    # test_elman_*:
    # Compiling in script mode fails with errors like:
    #   smith.jit.frontend.UnsupportedNodeError: annotated assignments
    #   without assigned value aren't supported
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1160723
    # test_lstm_*:
    #   Compiling in script mode fails with errors like:
    #   RuntimeError: Arguments for call are not valid.
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1160723
    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(9)
    @common_utils.parametrize(
        "name, nonlinearity",
        [
            ("elman", "relu"),
            ("elman", "tanh"),
            ("lstm", None),
            ("gru", None),
        ],
    )
    @common_utils.parametrize(**_parametrize_rnn_args("layers"))
    @common_utils.parametrize(**_parametrize_rnn_args("bidirectional"))
    @common_utils.parametrize(**_parametrize_rnn_args("initial_state"))
    @common_utils.parametrize(**_parametrize_rnn_args("packed_sequence"))
    @common_utils.parametrize(**_parametrize_rnn_args("dropout"))
    def test_rnn(self, *args, **kwargs):
        self._dispatch_rnn_test(*args, **kwargs)


if __name__ == "__main__":
    common_utils.TestCase._default_dtype_check_enabled = True
    common_utils.run_tests()
