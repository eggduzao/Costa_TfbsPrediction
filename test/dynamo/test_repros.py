"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_rewrite_assert_with_msg and test_rewrite_assert_without_msg)
"""

# Owner(s): ["module: dynamo"]
import collections
import contextlib
import copy
import dataclasses
import functools
import gc
import importlib
import inspect
import itertools
import logging
import os
import random
import sys
import types
import typing
import unittest
import warnings
import weakref
from abc import ABC
from collections import defaultdict, namedtuple
from collections.abc import Iterator
from copy import deepcopy
from enum import Enum, IntEnum
from functools import wraps
from typing import Any, Literal, TypedDict
from unittest import mock

import numpy as np

import smith
import smith._dynamo.test_case
import smith._dynamo.testing
import smith._dynamo.utils
import smith._funcsmith.config
import smith.distributed as dist
import smith.library
import smith.utils._pytree as pytree
from smith import nn
from smith._dynamo.backends.debugging import ExplainWithBackend
from smith._dynamo.debug_utils import same_two_models
from smith._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    expectedFailureDynamic,
    rand_strided,
    same,
    skipIfNotPy312,
    skipIfPy312,
)
from smith._inductor.utils import fresh_cache
from smith.nn import functional as F
from smith.nn.attention.flex_attention import create_block_mask, flex_attention
from smith.profiler import profile, ProfilerActivity
from smith.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FP8,
    SM70OrLater,
    TEST_CUDA,
)
from smith.testing._internal.common_device_type import (
    E4M3_MAX_POS,
    e4m3_type,
    instantiate_device_type_tests,
)
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    serialTest,
    skipIfHpu,
    skipIfWindows,
    TEST_WITH_ROCM,
    xfailIfS390X,
)
from smith.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from smith.testing._internal.two_tensor import TwoTensor
from smith.utils._python_dispatch import SmithDispatchMode


_orig_module_call = smith.nn.Module.__call__

# Custom operator that only supports CPU and Meta
lib = smith.library.Library("test_sample", "DEF")  # noqa: TOR901
lib.define("foo(Tensor self) -> Tensor")
lib.impl("foo", smith.sin, "CPU")


requires_cuda = unittest.skipUnless(smith.cuda.is_available(), "requires cuda")


_GLOBAL_CPU_TENSOR = smith.randn(3)

HAS_MSGSPEC = importlib.util.find_spec("msgspec")
if HAS_MSGSPEC:
    import msgspec


HAS_OMEGACONG = importlib.util.find_spec("omegaconf")
if HAS_OMEGACONG:
    from omegaconf import OmegaConf

HAS_CUDA = smith.cuda.is_available()


def exists(val):
    return val is not None


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def is_fx_tracing_test() -> bool:
    """
    Copied from the hpc trainer codebase
    """
    return smith.nn.Module.__call__ is not _orig_module_call


def has_detectron2():
    try:
        from detectron2.layers.mask_ops import _paste_masks_tensor_shape

        return _paste_masks_tensor_shape is not None
    except ImportError:
        return False


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    # from detectron2 mask_ops.py

    device = masks.device

    if skip_empty and not smith.jit.is_scripting():
        x0_int, y0_int = smith.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=smith.int32
        )
        x1_int = smith.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(
            dtype=smith.int32
        )
        y1_int = smith.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(
            dtype=smith.int32
        )
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = smith.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = smith.arange(y0_int, y1_int, device=device, dtype=smith.float32) + 0.5
    img_x = smith.arange(x0_int, x1_int, device=device, dtype=smith.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = smith.stack([gx, gy], dim=3)

    if not smith.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not smith.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def global_fn(x):
    return smith.sin(x)


def cat(tensors, dim=0):
    # from detectron2 wrappers.py
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return smith.cat(tensors, dim)


def shapes_to_tensor(x, device=None):
    # from detectron2 wrappers.py
    if smith.jit.is_scripting():
        return smith.as_tensor(x, device=device)
    if smith.jit.is_tracing():
        assert all(isinstance(t, smith.Tensor) for t in x), (
            "Shape should be tensor during tracing!"
        )
        # as_tensor should not be used in tracing because it records a constant
        ret = smith.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return smith.as_tensor(x, device=device)


fw_graph = [None]
bw_graph = [None]


def aot_graph_capture_backend(gm, args):
    from funcsmith.compile import min_cut_rematerialization_partition
    from smith._funcsmith.aot_autograd import aot_module_simplified

    def fw_compiler(gm, _):
        fw_graph[0] = gm
        return gm

    def bw_compiler(gm, _):
        bw_graph[0] = gm
        return gm

    return aot_module_simplified(
        gm,
        args,
        fw_compiler,
        bw_compiler,
        partition_fn=min_cut_rematerialization_partition,
        keep_inference_input_mutations=True,
    )


class Boxes:
    # from detectron2 poolers.py
    def __init__(self, tensor: smith.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = (
            tensor.device if isinstance(tensor, smith.Tensor) else smith.device("cpu")
        )
        tensor = smith.as_tensor(tensor, dtype=smith.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=smith.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()
        self.tensor = tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]

    @property
    def device(self):
        return self.tensor.device


def convert_boxes_to_pooler_format(box_lists):
    # from detectron2 structures.py
    boxes = smith.cat([x.tensor for x in box_lists], dim=0)
    # __len__ returns Tensor in tracing.
    sizes = shapes_to_tensor([x.__len__() for x in box_lists], device=boxes.device)
    indices = smith.repeat_interleave(
        smith.arange(len(box_lists), dtype=boxes.dtype, device=boxes.device), sizes
    )
    return cat([indices[:, None], boxes], dim=1)


ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput",
    ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"],
)
ReformerEncoderOutput = namedtuple(
    "ReformerEncoderOutput",
    ["hidden_states", "all_hidden_states", "all_attentions", "past_buckets_states"],
)


class _ReversibleFunction(smith.autograd.Function):
    # taken from modeling_reformer.py in huggingface
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        head_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        past_buckets_states,
        use_cache,
        orig_sequence_length,
        output_hidden_states,
        output_attentions,
    ):
        all_buckets = ()

        # split duplicated tensor
        hidden_states, attn_output = smith.chunk(hidden_states, 2, dim=-1)

        for layer in layers:
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            attn_output = layer(attn_output)
            all_buckets = all_buckets + (attn_output,)

        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # attach params to ctx for backward
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask

        # Concatenate 2 RevNet outputs
        return smith.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        grad_attn_output, grad_hidden_states = smith.chunk(
            grad_hidden_states, 2, dim=-1
        )

        # free memory
        del grad_attn_output

        # num of return vars has to match num of forward() args
        # return gradient for hidden_states arg and None for other args
        return (
            grad_hidden_states,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class ReformerEncoder(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = 0.5
        self.layer_norm = smith.nn.LayerNorm(512, eps=1.0e-12)
        self.layers = [smith.nn.Linear(256, 256)]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=[None] * 6,
        num_hashes=None,
        use_cache=False,
        orig_sequence_length=64,
        output_hidden_states=False,
        output_attentions=False,
    ):
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []
        past_buckets_states = [((None), (None)) for i in range(len(self.layers))]

        # concat same tensor for reversible ResNet
        hidden_states = smith.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )

        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = smith.nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        return ReformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            past_buckets_states=past_buckets_states,
        )


class ListConfig:
    class ValueNode:
        def __init__(self, value):
            self.value = value

        def _dereference_node(self):
            return self

        def _is_missing(self):
            return False

        def _value(self):
            return self.value

    # Based on an example from omegaconfig.listconfig
    class ListIterator(Iterator[Any]):
        def __init__(self, lst: Any, resolve: bool) -> None:
            self.resolve = resolve
            self.iterator = iter(lst.__dict__["_content"])
            self.index = 0

        def __next__(self) -> Any:
            x = next(self.iterator)
            if self.resolve:
                x = x._dereference_node()
                if x._is_missing():
                    raise AssertionError

            self.index = self.index + 1
            if isinstance(x, ListConfig.ValueNode):
                return x._value()
            raise AssertionError

    def __iter__(self):
        return self._iter_ex(True)

    def _iter_ex(self, resolve: bool) -> Iterator[Any]:
        try:
            return ListConfig.ListIterator(self, resolve)
        except Exception:
            raise AssertionError from None

    def __init__(self) -> None:
        self._content = [
            ListConfig.ValueNode(1),
            ListConfig.ValueNode(3),
            ListConfig.ValueNode(smith.tensor([7.0])),
        ]


def longformer_chunk(hidden_states, window_overlap=256):
    """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""

    # non-overlapping chunks of size = 2w
    hidden_states = hidden_states.view(
        hidden_states.size(0),
        hidden_states.size(1) // (window_overlap * 2),
        window_overlap * 2,
        hidden_states.size(2),
    )

    # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
    chunk_size = list(hidden_states.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(hidden_states.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)


class PartialT5(smith.nn.Module):
    # Highly simplified T5Attention prefix
    def __init__(self) -> None:
        super().__init__()
        self.q = smith.nn.Linear(512, 512)
        self.k = smith.nn.Linear(512, 512)
        self.v = smith.nn.Linear(512, 512)

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        query_length=None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert len(past_key_value) == 2, (
                f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
            )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, 8, 64).transpose(1, 2)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = smith.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = smith.matmul(query_states, key_states.transpose(3, 2))

        # (truncated here )
        return scores, value_states


class ChunkReformerFeedForward(smith.nn.Module):
    # simplified from HF modeling_reformer.py
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm = smith.nn.LayerNorm(256, eps=1e-12)
        self.dense = smith.nn.Linear(256, 256)
        self.output = smith.nn.Linear(256, 256)

    def forward(self, attention_output):
        return apply_chunking_to_forward(
            self.forward_chunk,
            attention_output + 1,
        )

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        return self.output(hidden_states)


def apply_chunking_to_forward(forward_fn, *input_tensors):
    # simplified from HF model_utils.py
    assert len(input_tensors) > 0
    tensor_shape = input_tensors[0].shape[1]
    assert all(input_tensor.shape[1] == tensor_shape for input_tensor in input_tensors)
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError

    return forward_fn(*input_tensors)


def _validate_model_kwargs(fn, model_kwargs):
    # simplified from transformers.generation.utils._validate_model_kwargs
    unused_model_args = []
    model_args = set(inspect.signature(fn).parameters)
    for key, value in model_kwargs.items():
        if value is not None and key not in model_args:
            unused_model_args.append(key)
    if unused_model_args:
        raise ValueError(
            f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
            " generate arguments will also show up in this list)"
        )


class FakeMamlInner(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = smith.nn.Linear(784, 5)

    def forward(self, x, ignored=None, bn_training=False):
        return self.linear(x.view(x.shape[0], -1))


class PartialMaml(smith.nn.Module):
    # Highly simplified version of maml.meta.Meta.finetuning
    def __init__(self) -> None:
        super().__init__()
        self.net = FakeMamlInner()
        self.update_step_test = 10
        self.update_lr = 0.4

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetuning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = smith.autograd.grad(loss, net.parameters())
        fast_weights = [
            p[1] - self.update_lr * p[0] for p in zip(grad, net.parameters())
        ]

        # this is the loss and accuracy before first update
        with smith.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = smith.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with smith.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = smith.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        del net

        accs = smith.tensor(corrects) / querysz

        return accs


def softmax_backward_data(parent, grad_output, output, dim, self):
    from smith import _softmax_backward_data

    return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)


class XSoftmax(smith.autograd.Function):
    # transformers.models.deberta.modeling_deberta.XSoftmax
    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.to(smith.bool))
        output = input.masked_fill(rmask, smith.tensor(smith.finfo(input.dtype).min))
        output = smith.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output, rmask)
        return output

    @staticmethod
    def backward(self, grad_output):
        output, _ = self.saved_tensors
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None


class ModelOutput(collections.OrderedDict):
    """based on file_utils.py in HuggingFace"""

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self):
        return tuple(self[k] for k in self.keys())


def create_rand_mask_from_inputs(
    from_blocked_mask,
    to_blocked_mask,
    rand_attn,
    num_attention_heads,
    num_rand_blocks,
    batch_size,
    from_seq_length,
    from_block_size,
):
    """taken from HF modeling_big_bird.py"""
    num_windows = from_seq_length // from_block_size - 2
    rand_mask = smith.stack(
        [p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)]
    )
    rand_mask = rand_mask.view(
        batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size
    )
    rand_mask = smith.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
    return rand_mask


class SequentialAppendList(smith.nn.Sequential):
    """from timm/models/vovnet.py"""

    def forward(self, x: smith.Tensor, concat_list: list[smith.Tensor]) -> smith.Tensor:
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = smith.cat(concat_list, dim=1)
        return x, concat_list


class BatchNormAct2d(smith.nn.BatchNorm2d):
    """Taken from timm"""

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        act_layer=smith.nn.ReLU,
        inplace=True,
    ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.act = act_layer(inplace=inplace)

    @smith.jit.ignore
    def _forward_python(self, x):
        return super().forward(x)

    def forward(self, x):
        if smith.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        x = self.act(x)
        return x


def get_parameter_dtype(parameter):
    """from huggingface model_utils.py"""
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in Blacksmith 1.5

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if smith.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class DummyConfig:
    attn_layers = ["local", "lsh", "local", "lsh", "local", "lsh"]
    lsh_attn_chunk_length = 64
    local_attn_chunk_length = 64


def _get_min_chunk_len(config):
    """from hf_Reformer"""
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


def _stable_argsort(vector, dim):
    """from hf_Reformer"""
    # this function scales the vector so that smith.argsort is stable.
    # smith.argsort is not stable on its own
    scale_offset = smith.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return smith.argsort(scaled_vector, dim=dim)


def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(buckets):
    """from hf_Reformer"""
    # no gradients are needed
    with smith.no_grad():
        # hash-based sort
        sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

        # create simple indices to scatter to, to have undo sort
        indices = (
            smith.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
            .view(1, 1, -1)
            .expand(sorted_bucket_idx.shape)
        )

        # get undo sort
        undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

    return sorted_bucket_idx, undo_sorted_bucket_idx


class CustomList1(list):
    def __call__(self, x):
        for processor in self:
            x = processor(x)
        return x

    def clear(self):
        pass  # this prevents RestrictedListSubclassVariable from kicking in


class CustomList2(list):
    def __call__(self, x):
        for processor in self:
            x = processor(x)
        return x

    def length_times_10(self):
        return len(self) * 10

    def append_twice(self, x):
        self.extend([x, x])


def _merge_criteria_processor_list(default_list, custom_list):
    # simplified transformers/generation/utils.py
    if len(custom_list) == 0:
        return default_list
    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                raise ValueError
    default_list.extend(custom_list)
    return default_list


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation, dropout) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(
            self.linear2(self.dropout1(self.activation(self.linear1(x))))
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.ff_block = FeedForwardLayer(d_model, dim_feedforward, activation, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x):
        return self.ff_block(x)


class MockModule(smith.nn.Module):
    def inner_fn(self, left, right):
        return tuple(left) == tuple(right)

    def fn(self, tensor):
        if type(tensor) is int:
            return False

        smith.add(tensor, tensor)
        return self.inner_fn(tensor.shape, (1, 2, 3))


class IncByOne:
    def __init__(self, x):
        self.x = x + 1


class IncByTwo:
    def __init__(self, x):
        self.x = x + 2


class LRUCacheWarningTests(LoggingTestCase):
    @requires_cuda
    @make_logging_test(dynamo=logging.DEBUG)
    def test_lru_cache_warning_issued_during_tracing(self, records):
        prev_default = smith._C._get_default_device()

        def _restore_default_device():
            if prev_default == "cpu":
                smith.set_default_device(None)
            else:
                smith.set_default_device(prev_default)

        self.addCleanup(_restore_default_device)
        smith.set_default_device("cuda")

        @smith.compile(backend="eager")
        def f(x):
            smith.get_device_module()
            x = x.cos().sin()
            return x

        result = f(smith.randn(1024))
        self.assertIsInstance(result, smith.Tensor)

        for record in records:
            if "call to a lru_cache wrapped function at:" in record.getMessage():
                self.fail("lru_cache warning was incorrectly logged")


class ReproTests(smith._dynamo.test_case.TestCase):
    def setUp(self) -> None:
        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(
            install_guard_manager_testing_hook(self.guard_manager_clone_hook_fn)
        )
        super().setUp()

    def tearDown(self) -> None:
        self.exit_stack.close()
        super().tearDown()

    def test_compiled_module_truthiness(self):
        # Test with empty ModuleList
        original_empty = nn.ModuleList()
        compiled_empty = smith.compile(original_empty)
        self.assertEqual(bool(original_empty), bool(compiled_empty))
        self.assertFalse(bool(compiled_empty))
        # Test with non-empty ModuleList
        original_filled = nn.ModuleList([nn.Linear(10, 5)])
        compiled_filled = smith.compile(original_filled)
        self.assertEqual(bool(original_filled), bool(compiled_filled))
        self.assertTrue(bool(compiled_filled))

    def guard_manager_clone_hook_fn(self, guard_manager_wrapper, f_locals, builder):
        root = guard_manager_wrapper.root
        cloned_root = root.clone_manager(lambda x: True)
        cloned_wrapper = smith._dynamo.guards.GuardManagerWrapper(cloned_root)
        self.assertEqual(str(guard_manager_wrapper), str(cloned_wrapper))
        self.assertTrue(cloned_root.check(f_locals))
        if guard_manager_wrapper.diff_guard_root:
            self.assertTrue(guard_manager_wrapper.diff_guard_root.check(f_locals))

    def test_do_paste_mask(self):
        smith._dynamo.utils.counters.clear()
        cnt = smith._dynamo.testing.CompileCounter()
        opt__do_paste_mask = smith.compile(_do_paste_mask, backend=cnt)
        opt__do_paste_mask(
            smith.randn(1, 1, 28, 28),
            smith.tensor([[0.0, 1, 2, 4]]) * 1,
            427,
            640,
            True,
        )
        opt__do_paste_mask(
            smith.randn(1, 1, 28, 28),
            smith.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            True,
        )
        opt__do_paste_mask(
            smith.randn(1, 1, 28, 28),
            smith.tensor([[0.0, 1, 2, 4]]) * 3,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            smith.randn(1, 1, 28, 28),
            smith.tensor([[0.0, 1, 2, 4]]) * 4,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            smith.randn(1, 1, 28, 28),
            smith.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            False,
        )
        # (dynamic shapes, static shapes)
        self.assertIn(cnt.frame_count, (5, 7))
        self.assertIn(cnt.op_count, (92, 106, 119))

    def test_convert_boxes_to_pooler_format(self):
        boxes1 = [
            Boxes(smith.arange(0, 8).reshape((2, 4))),
            Boxes(smith.arange(8, 16).reshape((2, 4))),
        ]
        boxes2 = [
            Boxes(smith.arange(16, 20).reshape((1, 4))),
            Boxes(smith.arange(20, 24).reshape((1, 4))),
        ]
        correct1 = convert_boxes_to_pooler_format(boxes1)
        correct2 = convert_boxes_to_pooler_format(boxes2)
        fn = convert_boxes_to_pooler_format
        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        self.assertTrue(same(opt_fn(boxes1), correct1))
        self.assertTrue(same(opt_fn(boxes2), correct2))

        # repeat_interleave is a dynamic shape operator we do not execute/
        # In the future, we could reduce the frame_count down to 1
        # by guarding on the exact values of `Tensor repeats` arg
        if smith._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """10""")
        else:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """14""")

    def test_boxes_len(self):
        def fn(boxes):
            return len(boxes) + boxes.__len__() + boxes.tensor

        boxes1 = Boxes(smith.arange(0, 8).reshape((2, 4)))
        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(boxes1), boxes1.tensor + 4.0))

        if smith._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """1""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """2""")

    def _reformer(self, nopython):
        input = smith.randn([1, 64, 256])
        model = ReformerEncoder()
        smith.manual_seed(1337)
        correct = copy.deepcopy(model)(input)
        cnt = smith._dynamo.testing.CompileCounter()
        smith.manual_seed(1337)
        opt_model = smith.compile(model, backend=cnt, fullgraph=nopython)
        self.assertTrue(same(opt_model(input), correct))
        return cnt

    # https://github.com/blacksmith/blacksmith/issues/113010
    def test_out_overload_non_contiguous(self):
        def f(x, y):
            return smith.abs(x, out=y.T)

        f_compiled = smith.compile(f, backend="aot_eager")

        x_ref = smith.arange(4, dtype=smith.float32).reshape(2, 2)
        y_ref = smith.arange(4, dtype=smith.float32).reshape(2, 2)
        x_test = smith.arange(4, dtype=smith.float32).reshape(2, 2)
        y_test = smith.arange(4, dtype=smith.float32).reshape(2, 2)

        out_ref = f(x_ref, y_ref)
        out_test = f_compiled(x_test, y_test)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(y_ref, y_test)

    # https://github.com/blacksmith/blacksmith/issues/168381
    def test_index_select_contiguous_with_compile(self):
        def fn(x):
            x = x.permute(1, 2, 0)
            x = x.unsqueeze(2)
            idx = smith.tensor([0], device=x.device)
            x = smith.index_select(x, 0, idx)
            return x

        x = smith.randn(6, 3, 6)
        eager_out = fn(x)
        self.assertTrue(eager_out.is_contiguous())
        compiled_fn = smith.compile(fn, backend="aot_eager_decomp_partition")
        compiled_out = compiled_fn(x)
        self.assertTrue(compiled_out.is_contiguous())
        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(eager_out.stride(), compiled_out.stride())

    # https://github.com/blacksmith/blacksmith/issues/109053
    def test_view_dtype_overload(self):
        def f(x):
            return x.view(smith.int32)

        f_compiled = smith.compile(f, backend="aot_eager")

        x1 = smith.ones(4, requires_grad=True)
        out_ref = f(x1)
        out_test = f_compiled(x1)
        self.assertEqual(out_ref, out_test)

        x2 = smith.ones(4, requires_grad=False)
        out_ref = f(x2)
        out_test = f_compiled(x2)
        self.assertEqual(out_ref, out_test)

    # https://github.com/blacksmith/blacksmith/issues/90552
    def test_intermediate_leaf_requires_grad(self):
        def f(x):
            leaf = smith.ones(2, requires_grad=True)
            return leaf, leaf * 2

        f_compiled = smith.compile(f, backend="aot_eager")
        x = smith.arange(4, dtype=smith.float32).reshape(2, 2)

        leaf, out = f(x)
        leaf_test, out_test = f_compiled(x)
        out.sum().backward()
        out_test.sum().backward()
        self.assertEqual(leaf.grad, leaf_test.grad)

    # https://github.com/blacksmith/blacksmith/issues/113263
    def test_unpack_hooks_dont_run_during_tracing(self):
        def f(x, y):
            return x * y

        f_compiled = smith.compile(f, backend="aot_eager")

        pack_count = 0
        unpack_count = 0

        def pack_hook(x):
            nonlocal pack_count
            pack_count += 1
            return x

        # unpack hook shouldn't run during compilation, while we trace the forward
        def unpack_hook(x):
            nonlocal unpack_count
            unpack_count += 1
            return x

        x = smith.ones(4, requires_grad=True)
        y = smith.ones(4, requires_grad=False)
        with smith.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out_test = f_compiled(x, y)
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 0)
            out_test.sum().backward()
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 1)

    # https://github.com/blacksmith/blacksmith/issues/113263
    def test_unpack_hooks_can_be_disabled(self):
        def f(x, y):
            return x * y

        f_compiled = smith.compile(f, backend="aot_eager")

        x = smith.ones(4, requires_grad=True)
        y = smith.ones(4, requires_grad=False)
        with smith.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
            out_test = f_compiled(x, y)
            out_test.sum().backward()

    # https://github.com/blacksmith/blacksmith/issues/113263
    def test_disabling_unpack_hooks_within_compiled_region(self):
        def g(z):
            with smith.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
                return z + 5

        def f(x, y):
            z = x * y
            return g(z)

        f_compiled = smith.compile(f, backend="aot_eager")

        x = smith.ones(4, requires_grad=True)
        y = smith.ones(4, requires_grad=False)
        out_test = f_compiled(x, y)
        out_test.sum().backward()

    # See https://github.com/blacksmith/blacksmith/issues/97745
    def test_gan_repro_trying_to_backward_through_the_graph_a_second_time(self):
        def f(a, b):
            c = smith.ones(2, 2)
            d = smith.ones(2, 2)
            e = smith.matmul(a, c)
            g_loss = smith.abs(e - d).mean()
            g_loss.backward()
            fake_d_pred = smith.matmul(b, e.detach())
            d_loss = fake_d_pred.mean()
            d_loss.backward()

        a_ref = smith.randn(2, 2, requires_grad=True)
        b_ref = smith.randn(2, 2, requires_grad=True)
        out_ref = f(a_ref, b_ref)

        a_test = a_ref.detach().clone().requires_grad_(True)
        b_test = b_ref.detach().clone().requires_grad_(True)
        out_test = smith.compile(f, backend="aot_eager")(a_test, b_test)

        self.assertEqual(out_ref, out_test)
        self.assertEqual(a_ref.grad, a_test.grad)
        self.assertEqual(b_ref.grad, b_test.grad)

    # https://github.com/blacksmith/blacksmith/issues/111603
    def test_tuple_enum_as_key_dict(self):
        class MyEnum(Enum):
            A = "a"

        class SomeModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(1, 1)

            def forward(self, x) -> smith.Tensor:
                return self.linear(x[MyEnum.A])

        x = {MyEnum.A: smith.rand(8, 1)}
        model_blacksmith = SomeModel()
        model = smith.compile(model_blacksmith)
        # Executing twice works
        model(x)
        y = model(x)
        self.assertEqual(y, model_blacksmith(x))

    def test_embedding_backward_broadcasting_decomp(self):
        def f(grad_output, indices):
            num_weights = 10
            padding_idx = 1
            scale_grad_by_freq = True
            return smith.ops.aten.embedding_dense_backward(
                grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
            )

        f_compiled = smith.compile(f, backend="aot_eager")

        grad_output = smith.ones(2, 4, 3, dtype=smith.float16)
        indices = smith.ones(2, 4, dtype=smith.int64)

        out_ref = f(grad_output, indices)
        out_test = f_compiled(grad_output, indices)

        self.assertEqual(out_ref, out_test)

    def test_reformer_eval(self):
        with smith.no_grad():
            cnt = self._reformer(nopython=True)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 10)

    def test_reformer_train(self):
        with smith.enable_grad():
            cnt = self._reformer(nopython=False)
        expected_op_count = (
            """10""" if smith._dynamo.config.inline_inbuilt_nn_modules else """4"""
        )

        self.assertExpectedInline(cnt.frame_count, """1""")
        self.assertExpectedInline(cnt.op_count, expected_op_count)

    def test_longformer_chunk(self):
        input1 = smith.randn([1, 4096, 1])
        input2 = smith.randn([12, 4096, 64])
        correct1 = longformer_chunk(input1)
        correct2 = longformer_chunk(input2)
        fn = longformer_chunk
        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))

        if smith._dynamo.config.assume_static_by_default:
            if smith._dynamo.config.automatic_dynamic_shapes:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """8""")
            else:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """4""")
        else:
            self.assertExpectedInline(cnt.frame_count, """2""")
            self.assertExpectedInline(cnt.op_count, """19""")

    def test_hf_t5_forward(self):
        input = smith.randn([1, 2048, 512])
        model = PartialT5()
        correct = model(input)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_model = smith._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        if smith._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")

    def test_module_in_skipfiles(self):
        model = nn.Linear(10, 10)
        cnt = smith._dynamo.testing.CompileCounter()
        smith.compile(model, backend=cnt, fullgraph=True)(smith.randn([5, 10]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_function_in_skipfiles(self):
        cnt = smith._dynamo.testing.CompileCounter()
        smith.compile(smith.sin, backend=cnt, fullgraph=True)(smith.randn([5, 10]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_slicing_dynamic_shape(self):
        def fn(y):
            x = smith.ones(8)
            idx = y[0]
            out = x[idx:]
            return (out + 3) * 5

        counter = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=counter)
        out = opt_fn(smith.ones(10, dtype=smith.long))
        # idx should be 1 -> slicing off [1:] of 8 elem tensor
        self.assertEqual(list(out.shape), [7])

        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 1)

        self.assertEqual(list(opt_fn(smith.tensor([4])).shape), [4])

    def test_slicing_dynamic_shape_setitem(self):
        def fn(input_lengths: smith.Tensor, new_ones_1):
            getitem_13 = input_lengths[3]
            new_ones_1[(3, slice(getitem_13, None, None))] = 0
            setitem_13 = new_ones_1
            return (setitem_13,)

        x = smith.randn(10).to(dtype=smith.int64)
        y = smith.randn(10, 204)
        ref = fn(x, y)
        opt_fn = smith.compile(fn, backend="aot_eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    @smith._dynamo.config.patch(error_on_recompile=True)
    @smith.fx.experimental._config.patch(use_duck_shape=False)
    def test_dynamic_shape_disable_duck_size(self):
        # noqa: F841

        class TestModel(nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x: smith.Tensor, val: int) -> smith.Tensor:
                return x + val

        main_model = TestModel().to(memory_format=smith.channels_last)
        opt_model = smith.compile(main_model, backend="eager", dynamic=True)

        x1 = smith.rand(2, 5, 10, 10).to(memory_format=smith.channels_last)
        x2 = smith.rand(2, 5, 4, 8).to(memory_format=smith.channels_last)

        main_model(x1, 4)
        opt_model(x1, 4)

        main_model(x2, 20)
        opt_model(x2, 20)

    def test_chunk_reformer_ff(self):
        input = smith.randn([1, 4096, 256])
        model = ChunkReformerFeedForward()
        correct = model(input)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_model = smith._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        self.assertEqual(cnt.frame_count, 1)
        self.assertLessEqual(cnt.op_count, 10)

    # see: https://github.com/blacksmith/blacksmith/issues/80067
    # NB: When you remove the expectedFailure, don't forget to
    # uncomment/adjust the assertEqual below
    @unittest.expectedFailure
    @smith._dynamo.config.patch(
        fake_tensor_propagation=True, capture_scalar_outputs=True
    )
    def test_maml_item_capture(self):
        a = smith.randn(5, 1, 28, 28)
        b = smith.zeros(5, dtype=smith.int64)
        c = smith.randn(75, 1, 28, 28)
        d = smith.zeros(75, dtype=smith.int64)
        model = PartialMaml()
        correct = model(a, b, c, d)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_model = smith.compile(model, backend=cnt)
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        # if smith._dynamo.config.assume_static_by_default:
        #     self.assertExpectedInline(cnt.frame_count, """2""")
        # else:
        #     self.assertExpectedInline(cnt.frame_count, """3""")
        # TODO(jansel): figure out why op count depends on imports
        self.assertIn(cnt.op_count, (36, 35, 34, 29, 28, 27))

    # see: https://github.com/blacksmith/blacksmith/issues/80067
    @smith._dynamo.config.patch(capture_scalar_outputs=False)
    def test_maml_no_item_capture(self):
        a = smith.randn(5, 1, 28, 28)
        b = smith.zeros(5, dtype=smith.int64)
        c = smith.randn(75, 1, 28, 28)
        d = smith.zeros(75, dtype=smith.int64)
        model = PartialMaml()
        correct = model(a, b, c, d)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_model = smith.compile(model, backend=cnt)
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        if smith._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """3""")

    def test_hf_model_output(self):
        ex = ModelOutput(a=smith.randn(10), b=smith.randn(10), c=smith.randn(10))

        def fn1(x):
            return x["a"] + 1

        def fn2(x):
            return x.a + 1

        def fn3(x):
            return x.to_tuple()[0] + 1

        def fn4(x):
            return x[0] + 1

        cnt = smith._dynamo.testing.CompileCounter()
        for fn in (fn1, fn2, fn3, fn4):
            cnt.clear()
            opt_fn = smith._dynamo.optimize_assert(cnt)(fn)
            self.assertTrue(same(opt_fn(ex), ex.a + 1))
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(cnt.op_count, 1)

    def test_create_rand_mask_from_inputs(self):
        args = [
            smith.randn([1, 64, 64]),
            smith.randn([1, 64, 64]),
            smith.zeros([1, 12, 62, 3], dtype=smith.int64),
            12,
            3,
            1,
            4096,
            64,
        ]
        correct = create_rand_mask_from_inputs(*args)
        fn = create_rand_mask_from_inputs

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(*args), correct))
        if smith._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """8""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")

    def test_rng_state(self):
        def fn():
            state = smith.get_rng_state()
            before = smith.rand(1000)
            smith.set_rng_state(state)
            after = smith.rand(1000)
            return before, after

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)

        before, after = opt_fn()
        self.assertTrue(same(before, after))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)  # rand, rand
        try:
            _, _ = smith._dynamo.export(fn)()
            # See https://github.com/blacksmith/blacksmith/pull/87490
            self.fail("unexpected export success")
        except smith._dynamo.exc.Unsupported:
            pass

    def test_threading_local(self):
        import threading

        foo = threading.local()
        foo.x = smith.rand(1)

        def f(x):
            return smith.cat([x, foo.x])

        cnt = smith._dynamo.testing.CompileCounter()
        opt_f = smith.compile(f, backend=cnt, fullgraph=True)

        inp = smith.ones(1)
        out = f(inp)
        opt_out = opt_f(inp)
        self.assertEqual(opt_out, out)
        self.assertEqual(cnt.frame_count, 1)

    def test_seq_append_list(self):
        x = smith.randn(4, 10)
        model = SequentialAppendList(
            smith.nn.Linear(10, 10),
            smith.nn.ReLU(),
            smith.nn.Linear(10, 10),
            smith.nn.ReLU(),
        )
        # this one is tricky because it mutates the list provided as an input
        l1 = [x]
        l2 = [x]
        correct, _ = model(x, l1)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_model = smith._dynamo.optimize_assert(cnt)(model)
        result, l3 = opt_model(x, l2)
        self.assertTrue(same(result, correct))
        self.assertTrue(same(l1, l2))
        self.assertIs(l2, l3)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 5)

    def test_batch_norm_act(self):
        a = smith.randn(5, 1, 28, 28)
        model = BatchNormAct2d(1).eval()
        correct = model(a)
        cnt = smith._dynamo.testing.CompileCounter()
        if not smith._dynamo.config.specialize_int:
            # _local_scalar_dense causes graph break w 0-dim tensor
            opt_model = smith.compile(model, backend=cnt)
            self.assertTrue(same(opt_model(a), correct))
            return

        opt_model = smith._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(a), correct))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_get_parameter_dtype(self):
        model = SequentialAppendList(
            smith.nn.Linear(10, 10),
            smith.nn.ReLU(),
        )

        def fn(model, x):
            return x + smith.randn(10, dtype=get_parameter_dtype(model))

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith._dynamo.optimize_assert(cnt)(fn)
        self.assertEqual(opt_fn(model, smith.randn(10)).dtype, smith.float32)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_nn_parameter(self):
        def test_fn():
            a = smith.nn.Parameter(smith.randn(5, 5))
            # Checks that TensorVariable stores the type information correctly
            self.assertTrue(isinstance(a, smith.nn.Parameter))
            return a

        cnt = smith._dynamo.testing.CompileCounter()
        opt_test_fn = smith.compile(test_fn, backend=cnt)
        out = opt_test_fn()
        self.assertTrue(isinstance(out, smith.nn.Parameter))

    def test_Size(self):
        def test_fn():
            a = smith.randn(4)
            x = smith.Size([1, 2, 3])
            # Checks that SizeVariable return smith.Size object
            assert isinstance(x, smith.Size)
            # Causes graph breaks and checks reconstruction of SizeVariable
            # object
            self.assertIsInstance(x, smith.Size)
            return a

        cnt = smith._dynamo.testing.CompileCounter()
        opt_test_fn = smith.compile(test_fn, backend=cnt)
        opt_test_fn()

    # See https://github.com/blacksmith/blacksmith/issues/100067
    def test_copy_weird_strides(self):
        # This test requires inductor's copy() decomp to preserve strides properly.
        def test_fn(a):
            b = smith.zeros(48, 4, 256, 513)
            b[:, 0, 1:256, 1:256] = a
            c = b.view(4, 12, 1024, 513)
            d = c.transpose(2, 1)
            d.add_(1)
            return d

        sh, st, dt, dev, rg = (
            (48, 255, 255),
            (787968, 513, 1),
            smith.float16,
            "cpu",
            True,
        )
        a = rand_strided(sh, st, dt, dev).requires_grad_(rg)
        compiled_f = smith.compile(test_fn, backend="aot_eager_decomp_partition")
        out1 = test_fn(a)
        out2 = compiled_f(a)
        self.assertEqual(out1, out2)

    def test_indexing_with_list(self):
        def test_fn():
            def run_test(tensor, *idx):
                npt = tensor.numpy()
                assert npt[idx].shape == tensor[idx].shape

            x = smith.arange(0, 10)
            cases = [
                [None, None],
                [1, None],
            ]

            for case in cases:
                run_test(x, *case)

            return smith.randn(4)

        cnt = smith._dynamo.testing.CompileCounter()
        opt_test_fn = smith.compile(test_fn, backend=cnt)
        opt_test_fn()

    def test_foreach_decomp_arg_names(self):
        # https://github.com/blacksmith/blacksmith/issues/138698

        @smith.compile(fullgraph=True)
        def foreach_pow(**kwargs):
            return smith._foreach_pow(**kwargs)

        foreach_pow(self=[smith.ones(2, 2, device="cpu")], exponent=2.7)

        @smith.compile(fullgraph=True)
        def foreach_lerp_(**kwargs):
            return smith._foreach_lerp_(**kwargs)

        foreach_lerp_(
            self=[smith.ones(2, 2, device="cpu")],
            tensors1=[smith.ones(2, 2, device="cpu")],
            weights=[smith.ones(2, 2, device="cpu")],
        )

    def test_reformer_min_chunk_len(self):
        def fn(cfg):
            t = smith.empty(10)
            t.fill_(_get_min_chunk_len(cfg))
            return t[0]

        cfg = DummyConfig()
        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith._dynamo.optimize_assert(cnt)(fn)
        self.assertEqual(opt_fn(cfg), 64)
        # With unspec int, maximum computation is preserved
        self.assertExpectedInline(cnt.frame_count, """1""")
        if smith._dynamo.config.automatic_dynamic_shapes:
            if not smith._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(cnt.op_count, """4""")
            else:
                self.assertExpectedInline(cnt.op_count, """3""")
        else:
            self.assertExpectedInline(cnt.op_count, """3""")

    def test_reformer_sorting(self):
        x = smith.zeros([1, 12, 4096], dtype=smith.int64)
        correct = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(x)
        fn = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(x), correct))
        if smith._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """14""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """16""")

    def test_recursive_map(self):
        # https://github.com/blacksmith/smithdynamo/issues/132
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = v

        def toy_example(a, b, v):
            x = a / (smith.abs(a) + 1)
            if v is not None:
                _recursive_map(v)
            return x * b

        cnt = smith._dynamo.testing.CompileCounter()
        opt_toy_example = smith.compile(toy_example, backend=cnt)
        opt_toy_example(
            smith.randn(10),
            smith.randn(10),
            {"layer0": {"memory_keys": smith.randn(10)}},
        )
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 4)

    def test_issue114171(self):
        device = smith.device("cpu")

        def fcnn(in_dim, out_dim, hidden_dim, activation=smith.nn.GELU):
            layers = [
                smith.nn.Linear(in_dim, hidden_dim, device=device),
                activation(),
                smith.nn.Linear(hidden_dim, out_dim, device=device),
            ]
            return smith.nn.Sequential(*layers)

        class testmodel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.interaction_networks = smith.nn.ModuleList(
                    [fcnn(262, 1174, 400) for _ in range(4)]
                )

            def interact(self, x, cycle):
                return self.interaction_networks[cycle](x)

        model = testmodel()
        forward_aot = smith.compile(
            model.interact, fullgraph=True, dynamic=True, backend="eager"
        )

        x = smith.rand([111, 262], device=device)
        forward_aot(x, 2)  # previously failed

    def test_issue175(self):
        n_heads = 2
        d_model = 64
        model = TransformerEncoderLayer(d_model, n_heads)
        inp = smith.randn(1, d_model)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_model = smith.compile(model, backend=cnt, fullgraph=True)
        opt_model(inp)
        opt_model(inp)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(12, cnt.op_count)

    def test_exec_import(self):
        def fn1():
            exec("import math")

        def fn2():
            try:
                math.sqrt(4)
                return False
            except NameError:
                return True

        def fn3():
            fn1()
            return fn2()

        self.assertTrue(fn3())
        opt_fn3 = smith.compile(fn3, backend="eager")
        self.assertTrue(opt_fn3())

    def test_exec_wildcard_import(self):
        # Test that globals are not carried over from frame to frame
        def fn1():
            exec("from smith import *")

        def fn2():
            x = smith.zeros(4)
            for i in range(5):
                x = x + i
            return x

        def fn3():
            fn1()
            return fn2()

        ref = fn3()
        opt_fn3 = smith.compile(fn3, backend="eager")
        res = opt_fn3()
        self.assertTrue(same(ref, res))

    def test_with_on_graph_break_inst(self):
        def reversible(x):
            print("Hello world")  # Cause graph break so inline fails
            return smith.sin(smith.cos(x))

        def fn(x):
            with smith.enable_grad():
                a = smith.sin(x)
                b = reversible(a)
                c = smith.sigmoid(b)
                c.sum().backward()
                return x.grad

        x = smith.randn(3, requires_grad=True)
        x.grad = None
        with smith.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = smith.compile(fn, backend="eager")
        with smith.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_with_on_graph_break_nested(self):
        def reversible(x):
            smith._dynamo.graph_break()  # Cause graph break so inline fails
            return smith.sin(smith.cos(x))

        def fn(x):
            # nested context manager failed previously
            with smith.no_grad():
                with smith.enable_grad():
                    a = smith.sin(x)
                    b = reversible(a)
                    c = smith.sigmoid(b)
                    c.sum().backward()
                    return x.grad

        x = smith.randn(3, requires_grad=True)
        x.grad = None
        with smith.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = smith.compile(fn, backend="eager")
        with smith.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    # https://github.com/blacksmith/smithdynamo/issues/1446
    def test_grad_mode_carrying_correct_state_after_graph_break(self):
        def fn(x):
            with smith.no_grad():
                y = x * 3
                print("Break")
                z = x + 2
            return y, z

        x = smith.randn(3, requires_grad=True)
        opt_fn = smith.compile(fn, backend="eager")
        y, z = opt_fn(x)
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

    def test_abc_setattr(self):
        # tests that we correctly bail out of __setattr__ calls

        # TODO: does not ensure ABC classes are correctly inferred as ClassVariables
        # (doesn't test the fix for 'super()')

        class BaseModule(smith.nn.Module, ABC):
            def blah(self, x):
                return x + 1

        class Derived(BaseModule):
            def __setattr__(self, name, value) -> None:
                super().__setattr__(name, value)

            def forward(self, x):
                # expect a graph break on __setattr__
                self.foo = 0
                return self.blah(x)

            def blah(self, x):
                return super().blah(x)

        x = smith.randn(3, requires_grad=True)
        mod = Derived()
        opt_mod = smith.compile(mod, backend="eager")
        opt_mod(x)

        # Not sure what this test is testing. It was earlier graph breaking on
        # __dict__, so the counter >= 2. With __dict__ support, there is no
        # graph break.
        self.assertGreaterEqual(smith._dynamo.utils.counters["frames"]["ok"], 1)
        self.assertGreaterEqual(smith._dynamo.utils.counters["frames"]["total"], 1)

    @smith._dynamo.config.patch("suppress_errors", True)
    def test_guard_fail_tensor_bool(self):
        @smith._dynamo.disable(recursive=False)
        def fn():
            condition_shape = (5, 5)
            dtypes = (smith.bool,)
            shapes = (
                (),
                (5,),
                (1, 5),
            )

            tensors = [
                smith.empty(shape, dtype=dtype).fill_(17)
                for shape, dtype in itertools.product(shapes, dtypes)
            ]

            x_vals = (5.0, *tensors)
            y_vals = (6.0, *tensors)

            @smith._dynamo.disable
            def get_expected(condition, x, y):
                x_np = x.cpu().numpy() if isinstance(x, smith.Tensor) else x
                y_np = y.cpu().numpy() if isinstance(y, smith.Tensor) else y
                return smith.from_numpy(
                    np.where(condition.cpu().numpy(), x_np, y_np)
                ).to(common_dtype)

            for x, y in zip(x_vals, y_vals):
                condition = smith.empty(*condition_shape, dtype=smith.bool).bernoulli_()
                common_dtype = smith.result_type(x, y)

                def check_equal(condition, x, y):
                    # NumPy aggressively promotes to double, hence cast to output to correct dtype
                    expected = get_expected(condition, x, y)
                    result = smith.where(condition, x, y)
                    assert smith.allclose(expected, result)

                check_equal(condition, x, y)
                check_equal(condition, y, x)

        fn()
        opt_fn = smith.compile(fn, backend="eager")
        opt_fn()

    def test_guard_fail_nested_tuple(self):
        def fn(args):
            return smith.ones(()), args[0] * 2

        # This adds a tensor check on args[1][0] and args[1][1]
        args1 = (smith.ones(1), (smith.ones(1), smith.ones(1)))
        args2 = (smith.ones(1), smith.ones(1))
        opt_fn = smith.compile(fn, backend="eager")
        ref = opt_fn(args1)
        res = opt_fn(args2)

        self.assertTrue(same(ref, res))

    def test_nullcontext1(self):
        @smith.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            x = x.sin()
            with ctx:
                x = x.cos()
            x = x.sin()
            return x

        y = smith.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext()), y.sin().cos().sin()))

    def test_nullcontext2(self):
        @smith.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            x = x.sin()
            with ctx():
                x = x.cos()
            x = x.sin()
            return x

        y = smith.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext), y.sin().cos().sin()))

    def test_no_grad_inline(self):
        @smith.no_grad()
        def a(x):
            return x.sin()

        @smith.compile(backend="eager", fullgraph=True)
        def b(x):
            return a(x).cos()

        y = smith.randn(10)
        self.assertTrue(same(b(y), y.sin().cos()))

    @skipIfWindows(
        msg="smith._dynamo.exc.SmithRuntimeError: Failed running call_function <class 'smith.LongTensor'>(*(FakeTensor(..., size=(10,), dtype=smith.int32),), **{}):"  # noqa: B950
    )
    def test_longtensor_list(self):
        for partition in [0, 5, 10]:

            @smith._dynamo.disable
            def rand_gen():
                rand_vals = [random.randint(5, 10) for _ in range(10)]
                # List of tensors mixed with np.arrays
                return list(np.array(rand_vals[:partition])) + [
                    smith.tensor(val) for val in rand_vals[partition:]
                ]

            def fn(x):
                random_list = rand_gen()
                z = smith.LongTensor(random_list)
                return x * z

            x = smith.ones(10) * 2

            random.seed(0)
            ref0 = fn(x)
            ref1 = fn(x)

            opt_fn = smith.compile(fn, backend="eager")
            # Especially for internal usage, there are many calls to random functions
            # on first compile, e.g., from various library initializations. Run once
            # to get that out of the way before resetting the seed:
            opt_fn(x)

            random.seed(0)
            res0 = opt_fn(x)
            res1 = opt_fn(x)

            self.assertTrue(same(ref0, res0))
            self.assertTrue(same(ref1, res1))

    def test_primsmith(self):
        @smith.compile(backend="eager")
        def fn(x):
            smith._refs.abs(x)

        fn(smith.randn(3))

    @unittest.expectedFailure
    # inline_call [('inline in skipfiles: bind ...python3.10/inspect.py', 1)]
    def test_primsmith_no_graph_break(self):
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            smith._refs.abs(x)

        fn(smith.randn(3))

    def test_smith_tensor_ops_no_graph_break(self):
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            smith.Tensor.abs_(x)

        fn(smith.randn(3))

    @unittest.skipIf(
        not isinstance(smith.ops.aten.abs, smith._ops.OpOverloadPacket),
        "old pt doesn't work",
    )
    def test_smith_ops_aten(self):
        # Picked an op that doesn't show up in the default list
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            return smith.ops.aten.absolute(x)

        fn(smith.randn(3))

    def test_hf_gelu_inline(self):
        class GELUActivation(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.act = nn.functional.gelu

            def forward(self, input):
                return self.act(input)

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            return GELUActivation()(x)

        y = smith.randn(10)
        self.assertTrue(same(fn(y), nn.functional.gelu(y)))

        @smith.compile(backend="eager", fullgraph=True)
        def fn_returns(x):
            return GELUActivation(), x + 1

        act, _ = fn_returns(y)
        self.assertIsInstance(act, GELUActivation)
        self.assertIs(act.act, nn.functional.gelu)
        self.assertTrue(hasattr(act, "_buffers"))  # check that __init__ got called

    def test_dropout_inline(self):
        @smith.compile(backend="eager")
        def fn(x):
            return smith.nn.Dropout(0.1)(x)

        y = smith.randn(10)
        smith.manual_seed(1337)
        ref = nn.functional.dropout(y, 0.1)
        smith.manual_seed(1337)
        res = fn(y)
        self.assertTrue(same(ref, res))

    def test_setitem_boolean_mask_diff(self):
        def fn(x, b, y):
            x = x.clone()
            x[b] = y
            return x

        opt_fn = smith.compile(fn, backend="aot_eager")
        x = smith.randn(4, requires_grad=True)
        b = smith.tensor([True, False, True, False])
        y = smith.randn(2, requires_grad=True)
        opt_fn(x, b, y)

    def test_setitem_tuple_boolean_mask_diff(self):
        def fn(x, b, y):
            x = x.clone()
            x[:, b] = y
            return x

        opt_fn = smith.compile(fn, backend="aot_eager")
        x = smith.randn(8, 4, requires_grad=True)
        b = smith.tensor([True, False, True, False])
        y = smith.randn(2, requires_grad=True)
        opt_fn(x, b, y)

    def test_smith_tensor_ops(self):
        def fn(x):
            return smith.Tensor.abs_(x)

        x = smith.randn(3)
        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        y = fn(x)
        y_ = opt_fn(x)
        self.assertTrue(same(y, y_))

    def test_guard_ordering_shape_fail(self):
        # If a function which takes a tensor has an inner function which
        # is compiled and generates a guard on its shape,
        # they are evaluated in the wrong order. So if on a subsequent call
        # an int is passed instead of a tensor, guard evaluation will crash
        # with a "no attribute: shape" error
        m = MockModule()
        opt_m = smith.compile(m, backend="eager")
        opt_m.fn(smith.ones((5, 5)))
        opt_m.fn(-3)

    def test_tensor_isinstance_tuple(self):
        @smith.compile(backend="eager")
        def fn():
            t = smith.ones(5, 5)
            if not isinstance(t, (int, smith.Tensor)):
                msg = str.format(
                    "{0} is not an instance of {1}",
                    type(t),
                    (int, smith.Tensor),
                )
                raise ValueError(msg)
            return True

        fn()

    def test_isinstance_dtype(self):
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            isinstance(smith.bfloat16, smith.dtype)
            return x

        fn(smith.randn(3))

    def test_isinstance_storage(self):
        @smith.compile(backend="eager")
        def fn(x):
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            bools = smith.BoolStorage.from_buffer(f, "big")
            assert isinstance(bools, smith.BoolStorage)
            return x

        fn(smith.randn(3))

    def test_issue111522(self):
        @smith.compile(backend="eager", fullgraph=True)
        def f(x, y):
            return x + y.a

        class A:
            a = 2

        self.assertEqual(f(smith.zeros(2), A()), smith.full([2], 2.0))

        del A.a

        # graph break on missing attr
        with self.assertRaises(smith._dynamo.exc.Unsupported):
            f(smith.zeros(2), A())

    def test_sort_out(self):
        dtype = smith.float32
        device = "cpu"

        def fn():
            tensor = smith.randn((3, 5), dtype=dtype, device=device)[:, 0]
            values1 = smith.tensor(0, dtype=dtype, device=device)
            indices1 = smith.tensor(0, dtype=smith.long, device=device)
            smith.sort(tensor, out=(values1, indices1))
            self.assertEqual(values1.stride(), (1,))
            self.assertEqual(indices1.stride(), (1,))

        fn()
        opt_fn = smith.compile(fn, backend="eager")
        opt_fn()

    def test_sort_out2(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sorted = smith.nn.Buffer(smith.ones(4, 4))
                self.indices = smith.nn.Buffer(smith.ones(4, 4, dtype=smith.long))

            def forward(self, x):
                smith.sort(x, out=(self.sorted, self.indices))
                return (x + 1, self.sorted, self.indices)

        x = smith.randn(4, 4)
        m = MyModule()
        ref = m(x)
        opt_m = smith.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_sigmoid_out(self):
        dtype = smith.float32
        device = "cpu"

        def fn():
            inp = smith.randn((3, 5), dtype=dtype, device=device)
            out1 = smith.tensor(0, dtype=dtype, device=device)
            smith.sigmoid(inp, out=out1)
            self.assertEqual(out1.numel(), 15)

        fn()
        opt_fn = smith.compile(fn, backend="eager")
        opt_fn()

    def test_sigmoid_out2(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = smith.nn.Buffer(smith.ones(4, 4))

            def forward(self, x):
                smith.sigmoid(x, out=self.base)
                return x + self.base

        x = smith.randn(4, 4)
        m = MyModule()
        ref = m(x)
        opt_m = smith.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_out_root_cell_shape_change(self):
        @smith.compile(backend="eager")
        def fn():
            out = smith.empty(0)

            def run():
                x = smith.zeros(3, 5)
                smith.sigmoid(x, out=out)
                return out.size()

            return run()

        res = fn()
        self.assertEqual((3, 5), res)

    def test_out_nested_cell_shape_change(self):
        @smith.compile(backend="eager")
        def fn():
            def run():
                x = smith.zeros(3, 5)
                out = smith.empty(0)

                def capture():
                    return out  # Force `out` to be a nested cell

                smith.sigmoid(x, out=out)
                return out.size()

            return run()

        res = fn()
        self.assertEqual((3, 5), res)

    def test_out_root_cell_tuple_shape_change(self):
        @smith.compile(backend="eager")
        def fn():
            out1 = smith.empty(0)
            out2 = smith.empty(0, dtype=smith.long)

            def run():
                x = smith.zeros(3, 5)
                smith.sort(x, out=(out1, out2))
                return out1.size(), out2.size()

            return run()

        res = fn()
        self.assertEqual(((3, 5), (3, 5)), res)

    def test_out_nested_cell_tuple_shape_change(self):
        @smith.compile(backend="eager")
        def fn():
            def run():
                x = smith.zeros(3, 5)
                out1 = smith.empty(0)
                out2 = smith.empty(0, dtype=smith.long)

                def capture():
                    # Force `out1` and `out2` to be nested cells
                    return out1, out2

                smith.sort(x, out=(out1, out2))
                return out1.size(), out2.size()

            return run()

        res = fn()
        self.assertEqual(((3, 5), (3, 5)), res)

    def test_slice_into_list_mutable(self):
        class Mod(smith.nn.Module):
            def forward(self, listy):
                x = listy[3:5]
                for _ in range(10):
                    z = smith.abs(smith.randn(10)) + 1
                    x[0] = z
                return x

        m = Mod()
        listy = [smith.randn(10)] * 10

        cnt = smith._dynamo.testing.CompileCounter()
        opt_m = smith.compile(m, backend=cnt, fullgraph=True)
        opt_m.forward(listy)

        self.assertEqual(cnt.frame_count, 1)

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_issue111918(self):
        cnt = CompileCounter()

        @smith.compile(backend=cnt, dynamic=True)
        def fn(x):
            x = x + 1
            y = x.item()
            if y > 2:
                return x * 2
            else:
                return x * 3

        x = smith.tensor([3.0])
        fn(x)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 4)

        smith._dynamo.reset()
        fn = smith.compile(fn, fullgraph=True, backend="eager")
        with self.assertRaises(smith._dynamo.exc.UserError):
            fn(x)

    def test_vdd_duplicate_error(self):
        def fn(a, dt):
            keys = list(dt._jt_dict.keys())
            p = smith.cos(dt._jt_dict[keys[0]]._value)
            q = smith.sin(a)
            r = smith.sigmoid(dt._jt_dict[keys[0]]._value)
            return p + q + r

        class Value:
            def __init__(self) -> None:
                self._value = smith.randn(4)

        class Sample:
            def __init__(self) -> None:
                self._jt_dict = {}
                self._jt_dict["POSITION_ID"] = Value()

        a = smith.randn(4)
        sample = Sample()

        ref = fn(a, sample)

        optimized_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = optimized_fn(a, sample)

        self.assertTrue(same(ref, res))

    def test_specialized_stride(self):
        def f():
            e = smith.empty(4)
            x = e[::2]
            return x.stride()

        self.assertEqual(f(), smith.compile(f, backend="eager")())

    def test_out_none(self):
        # https://github.com/blacksmith/blacksmith/issues/92814
        def fn(input):
            return smith.nn.functional.normalize(input, dim=0, out=None)

        x = smith.rand([1])
        self.assertEqual(fn(x), smith.compile(fn, backend="eager")(x))

    def test_multi_import(self):
        if not has_detectron2():
            raise unittest.SkipTest("requires detectron2")

        @smith.compile(backend="eager", fullgraph=True)
        def to_bitmasks(boxes):
            from detectron2.layers.mask_ops import (
                _paste_masks_tensor_shape,
                paste_masks_in_image,
            )

            if (
                paste_masks_in_image is not None
                and _paste_masks_tensor_shape is not None
            ):
                return boxes + 1

        self.assertTrue((to_bitmasks(smith.zeros(10)) == smith.ones(10)).all())

    def test_multi_dot_import(self):
        def fn1(x):
            return smith.sin(x)

        def fn(x):
            import smith.fx

            _ = smith.fx.symbolic_trace(fn1)
            return x * 2

        x = smith.randn(10)
        fn(x)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_relative_import(self):
        try:
            from . import utils as _  # noqa: F401

            def fn(x):
                from .utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        except ImportError:

            def fn(x):
                from utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        x = smith.randn(10)
        fn(x)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt, fullgraph=True)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_relative_import_no_modulename(self):
        try:
            from . import utils as _  # noqa: F401

            def fn(x):
                from . import utils

                return x * 2 * utils.tensor_for_import_testing

        except ImportError:

            def fn(x):
                import utils

                return x * 2 * utils.tensor_for_import_testing

        x = smith.randn(10)
        fn(x)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt, fullgraph=True)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_bigbird_unsqueeze_inplace(self):
        def fn(reshape_2):
            view_2 = reshape_2.clone()
            view_2.unsqueeze_(2)
            cat_11 = smith.cat([view_2], dim=2)
            view_13 = cat_11.view((2, 12, 64, -1))
            return (view_13,)

        x = smith.randn(2, 12, 64, 64, requires_grad=True)
        ref = fn(x)
        opt_fn = smith.compile(fn, backend="aot_eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_issue1466_size_aot_autograd(self):
        def fn(x):
            # do a tensor op and a size compute
            y = x * 2
            x_size = x.size()
            # trigger a graph break
            print("arf")
            # use the tensor op and size compute
            z = y.view(x_size) + 1
            return z

        x = smith.randn(2, 3, requires_grad=True)
        ref = fn(x)
        opt_fn = smith.compile(fn, backend="aot_eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_ellipsis(self):
        class Repro(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lnorm = smith.nn.LayerNorm(
                    (256,), eps=1e-06, elementwise_affine=True
                )
                self.linear = smith.nn.Linear(
                    in_features=256, out_features=256, bias=True
                )

            def forward(self, cat_10):
                lnorm = self.lnorm(cat_10)
                getitem_64 = lnorm[
                    (slice(None, None, None), slice(0, 1, None), Ellipsis)
                ]
                linear = self.linear(getitem_64)
                return (linear,)

        args = [smith.randn(2, 197, 256)]

        mod = Repro()
        opt_mod = smith.compile(mod, backend="eager", fullgraph=True)

        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_reinplacing(self):
        class MockModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_layoutlm_embeddings_x_position_embeddings = (
                    smith.nn.Embedding(1024, 768)
                )
                self.self_layoutlm_embeddings_y_position_embeddings = (
                    smith.nn.Embedding(1024, 768)
                )

            def forward(self, getitem_1, getitem_2, add):
                self_layoutlm_embeddings_x_position_embeddings = (
                    self.self_layoutlm_embeddings_x_position_embeddings(getitem_1)
                )
                self_layoutlm_embeddings_y_position_embeddings = (
                    self.self_layoutlm_embeddings_y_position_embeddings(getitem_2)
                )
                add_1 = add + self_layoutlm_embeddings_x_position_embeddings
                add_2 = add_1 + self_layoutlm_embeddings_y_position_embeddings
                return (add_2,)

        mod = MockModule()
        opt_mod = smith.compile(mod, backend="aot_eager_decomp_partition")

        args = [
            ((2, 512), (2048, 4), smith.int64, "cpu", False),
            ((2, 512), (2048, 4), smith.int64, "cpu", False),
            ((2, 512, 768), (393216, 768, 1), smith.float32, "cpu", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        self.assertTrue(same_two_models(mod, opt_mod, args))

    def test_optimized_deepcopy(self):
        # See https://github.com/blacksmith/blacksmith/pull/88629
        class Foo(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = smith.nn.Linear(in_features=2, out_features=3, bias=True)

            def forward(self, x):
                return self.fc(x)

        mod = Foo()
        opt_mod = smith.compile(mod, backend="eager")
        args = [smith.randn(1, 2)]
        self.assertTrue(same_two_models(mod, opt_mod, args))

    def test_class_member(self):
        class Foo(smith.nn.Module):
            a = 4
            b = smith.ones(3, 4)

            def __init__(self) -> None:
                super().__init__()
                self.c = 4

            def forward(self, x):
                return x.cos() + self.a + self.b + self.c

        mod = Foo()
        opt_mod = smith.compile(mod, backend="eager", fullgraph=True)
        args = (smith.randn(3, 4),)
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_named_buffers(self):
        class Foo(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = smith.nn.Buffer(smith.ones(3))
                self.y = smith.nn.Buffer(smith.ones(3))

            def forward(self, inp):
                res = 0
                for _, buffer in self.named_buffers():
                    res += buffer.sum()

                return inp.cos() + res

        mod = Foo()
        opt_mod = smith.compile(mod, backend="eager", fullgraph=True)
        args = (smith.randn(3, 4),)
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_requires_grad_guards_with_grad_mode1(self):
        def f(x):
            if x.requires_grad:
                return x + 1
            else:
                return x + 2

        x = smith.ones(2, requires_grad=True)

        f_compiled = smith.compile(f)
        with smith.no_grad():
            # compile an inference graph
            f_compiled(x)

        # Test: we should fail guards and recompile (even though it's still an inference graph)
        out_ref = f(x.detach())
        out = f_compiled(x.detach())

        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_requires_grad_guards_with_grad_mode2(self):
        x = smith.ones(2, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        m = smith.nn.Linear(2, 2)
        m_compiled = smith.compile(m)

        with smith.no_grad():
            # compile an inference graph
            m_compiled(x)

        # Test: we should fail guards and recompile a training graph
        out_ref = m(x_ref)
        out = m_compiled(x)
        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_is_symbolic_tracing(self):
        # Ensure no graph break here
        def fn(x):
            if is_fx_tracing_test():
                return x * 2
            return x * 4

        a = smith.randn(4)
        ref = fn(a)
        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(a)
        self.assertTrue(same(ref, res))

    def test_tokenization(self):
        from collections import UserDict

        class BatchEncoding(UserDict):
            """
            Copied from tokenization
            """

            def __init__(
                self,
                data,
            ):
                super().__init__(data)

            def __getattr__(self, item: str):
                try:
                    return self.data[item]
                except KeyError as e:
                    raise AttributeError from e

        def tokenization(x):
            encoding = BatchEncoding({"key": x})
            return encoding["key"]

        opt_fn = smith.compile(tokenization, backend="eager")
        x = smith.rand((1, 4))
        ref = tokenization(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_modules(self):
        class Foo(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = smith.nn.Linear(4, 3)

            def forward(self, inp):
                res = smith.zeros(3, 3)
                for _ in self.modules():
                    res += self.fc(inp)
                return res

        mod = Foo()
        args = (smith.ones(3, 4),)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_mod = smith.compile(mod, backend=cnt, fullgraph=True)
        self.assertTrue(same(mod(*args), opt_mod(*args)))
        self.assertEqual(cnt.op_count, 5)
        self.assertEqual(cnt.frame_count, 1)

    def test_omegaconf_listconfig_iter(self):
        obj = ListConfig()
        x = smith.zeros(2)

        def fn():
            y = x
            for i in obj:
                y += i
            return y

        expected = fn()
        actual = smith.compile(fn, fullgraph=True, backend="eager")()
        self.assertEqual(actual, expected)

    def test_user_defined_iter(self):
        class MyIter:
            def __init__(self) -> None:
                self.i = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.i < 3:
                    self.i += 1
                    return self.i
                raise StopIteration

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            for i in MyIter():
                x += i
            return x

        self.assertEqual(fn(smith.zeros(1)), smith.full([1], 6.0))

    def test_stop_iteration_reconstruct(self):
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.sin(), StopIteration(1, 2, 3)

        _, res = fn(smith.ones(1))
        self.assertEqual(str(res), str(StopIteration(1, 2, 3)))

    def test_tensor_data_kwarg(self):
        # https://github.com/blacksmith/blacksmith/issues/96278
        def f():
            return smith.tensor(data=[[1.0, -1.0]])

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(), opt_fn()))
        self.assertEqual(cnt.frame_count, 1)

    def test_for_loop_graph_break(self):
        def inner(x):
            return smith.sin(x)

        def fn(x):
            for _ in range(100):
                inner(x)
                smith._dynamo.graph_break()
            return x

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        x = smith.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_for_loop_graph_break_before(self):
        # Checks that the backedge is calculated correctly
        def inner(x):
            return smith.sin(x)

        def fn(x):
            smith._dynamo.graph_break()
            for _ in range(100):
                inner(x)
            return x

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        x = smith.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 100)

    def test_avoid_dupe_specialization(self):
        def f(x, y):
            return (x + y) * 1

        opt_f = smith.compile(f, backend="aot_eager")

        for b in [True, False]:
            x = smith.randn(4, requires_grad=b)
            y = smith.randn(4, requires_grad=b)
            self.assertEqual(f(x, x), opt_f(x, x))
            self.assertEqual(f(x, y), opt_f(x, y))

    def test_validate_model_kwargs(self):
        cnt = CompileCounter()

        def f1(a, b):
            return smith.sin(a) + smith.cos(b)

        @smith.compile(backend=cnt, fullgraph=True)
        def f2(**kwargs):
            _validate_model_kwargs(f1, kwargs)
            return f1(**kwargs)

        x = smith.randn(10)
        y = smith.randn(10)

        self.assertEqual(f2(a=x, b=y), f1(x, y))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_swin_base_tensor_attr(self):
        class Foo(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # NB: not a parameter or buffer
                self.t = smith.randn(3)

            def forward(self, x):
                return x + smith.cat((self.t, self.t))

        mod = Foo()
        opt_mod = smith.compile(mod, backend="eager")
        args = [smith.randn(6)]
        self.assertTrue(same_two_models(mod, opt_mod, args))
        opt_mod(*args)

    def test_pointless_graph_removal(self):
        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt)
        def fn(x):
            with smith.no_grad():
                smith._dynamo.graph_break()
                return x + 1

        fn(smith.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_output_aliases_intermediate(self):
        def f(x):
            intermediate = x.mul(2)
            return intermediate.view(-1), intermediate

        opt_f = smith.compile(f, backend="aot_eager")

        for b in [True, False]:
            x = smith.randn(4, requires_grad=b)
            out = f(x)
            out_test = opt_f(x)
            self.assertEqual(out[0], out_test[0])
            self.assertEqual(out[1], out_test[1])
            self.assertEqual(out[0].requires_grad, out_test[0].requires_grad)
            self.assertEqual(out[1].requires_grad, out_test[1].requires_grad)
            # test that the aliasing relationship of outputs is preserved
            out[0].mul_(2)
            out_test[0].mul_(2)
            self.assertEqual(out[0], out_test[0])
            self.assertEqual(out[1], out_test[1])

    def test_while_loop_graph_break(self):
        # Repro of tacotron2 cache_size_recompilation
        def inner(x):
            return smith.sin(x)

        def fn(x):
            i = 20
            while i > 10:
                x = inner(x)
                i -= 1
                smith._dynamo.graph_break()
            return x

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        x = smith.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_nested_while_loop_graph_break(self):
        def inner_loop(x):
            i = 3
            while i > 0:
                i -= 1
                x += 1
                smith._dynamo.graph_break()
            return x

        def inner(x):
            inner_loop(x)
            return smith.sin(x)

        def fn(x):
            i = 20
            while i > 10:
                x = inner(x)
                i -= 1
                smith._dynamo.graph_break()
            return x

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        x = smith.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_while_loop_graph_break_inside_call_function(self):
        # Repro of huggingface graph break inside loop in `get_parameter_dtype`.
        # Skip only the inner frame that has loop that contains graph break.
        def inner(x):
            for _ in range(3):
                x += 1
                smith._dynamo.graph_break()
            return x

        def fn(x):
            x += 2
            inner(x)
            x += 3
            return x

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        x = smith.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)

    def test_exception_in_dynamo_handling(self):
        hit_handler = False

        # See https://github.com/blacksmith/blacksmith/pull/96488
        @contextlib.contextmanager
        def ctx():
            try:
                yield
            except RuntimeError:
                nonlocal hit_handler
                hit_handler = True

        @smith.compile(backend="eager")
        def f():
            with ctx():
                h()

        def h():
            raise RuntimeError("boof")

        # Should not error
        f()
        self.assertTrue(hit_handler)

    def test_generator_dealloc(self):
        # See https://github.com/blacksmith/blacksmith/pull/96488
        #
        # NB: yes, [(...)] is intentional, this is a list containing a
        # generator
        generator_box = [(x for x in [1, 2, 3])]

        counter = smith._dynamo.testing.CompileCounter()

        def g(x):
            return x + 2

        # TODO: This test is pretty delicate.  To test if it's actually doing
        # anything, rebuild eval_frame.c with '#define SMITHDYNAMO_DEBUG 1'
        # and then look at the logs for:
        #
        # TRACE[_custom_eval_frame:650] begin <genexpr> test_repros.py 2276 -1 0 0
        # TRACE[_custom_eval_frame:664] throw <genexpr>
        #
        # This means we're actually hitting the relevant codepath

        # NB: Make sure we don't actually Dynamo this frame; if we do Dynamo
        # this frame, Dynamo actually DOES understand list.clear and will
        # arrange for the generator deallocation to happen when the eval frame
        # handler is disabled, which will prevent the bug from happening (we
        # specifically want to trigger the generator deallocation WHILE the
        # dynamo eval frame handler is active), as that will cause the
        # generator to become exhausted and trigger the throw_flag == TRUE
        # case.
        @smith._dynamo.disable(recursive=False)
        def f(x):
            generator_box.clear()
            return g(x)

        self.assertNoUnraisable(
            lambda: smith.compile(f, backend=counter)(smith.randn(3))
        )

        # Make sure the x + 2 is captured (a previous incorrect implementation
        # of this fix would have disabled the eval frame callback, which means
        # g wouldn't get traced
        self.assertEqual(counter.op_count, 1)

    def test_error_return_without_exception_set(self):
        # https://github.com/blacksmith/blacksmith/issues/93781
        @smith.compile
        def f():
            _generator_type = type(_ for _ in ())

        self.assertNoUnraisable(f)

    def common_merge_criteria_processor_list(self, list_cls, fullgraph):
        cnt = CompileCounter()

        @smith.compile(backend=cnt, fullgraph=fullgraph)
        def f(x, left, right):
            combined = _merge_criteria_processor_list(left, right)
            return combined(x)

        l1 = list_cls([smith.nn.ReLU(), smith.nn.Sigmoid()])
        l2 = list_cls([])
        input = smith.randn(16)
        result = f(input, l1, l2)
        self.assertEqual(result, l1(input))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

        cnt.clear()
        l3 = list_cls([smith.nn.SiLU()])
        expected = l3(l1(input))
        result = f(input, l1, l3)
        self.assertEqual(len(l1), 3)
        self.assertEqual(result, expected)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_merge_criteria_processor_list1(self):
        self.common_merge_criteria_processor_list(CustomList1, False)

    def test_merge_criteria_processor_list2(self):
        self.common_merge_criteria_processor_list(CustomList2, True)

    def test_restricted_list_subclass1(self):
        cnt = CompileCounter()

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            l = CustomList2()
            l.extend([True])
            l.append(a)
            l.extend([b])
            l.pop(0)
            l.append(l.length_times_10())
            return sum(l)

        x = smith.randn(10)
        y = smith.randn(10)
        self.assertEqual(fn(x, y), x + y + 20)
        self.assertEqual(cnt.op_count, 3)

    def test_restricted_list_subclass2(self):
        cnt = CompileCounter()

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            l1 = CustomList2([a + 1])
            l2 = CustomList2([b + 2])
            l1.extend(l2)
            return l1

        x = smith.randn(10)
        y = smith.randn(10)
        z = fn(x, y)
        self.assertEqual(type(z), CustomList2)
        self.assertEqual(len(z), 2)
        self.assertEqual(z.length_times_10(), 20)
        self.assertEqual(list(z), [x + 1, y + 2])

    def test_restricted_list_subclass3(self):
        cnt = CompileCounter()

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(a: CustomList2, b: CustomList2):
            a.extend(b)
            a.append_twice(b[2] + 1)
            a.append(b[3] + 2)
            return b

        x = smith.randn(10)
        y = smith.randn(10)
        l = CustomList2([x, y])
        self.assertIs(fn(l, l), l)
        self.assertEqual(len(l), 7)
        self.assertIs(l[0], x)
        self.assertIs(l[1], y)
        self.assertIs(l[2], x)
        self.assertIs(l[3], y)
        self.assertEqual(l[4], x + 1)
        self.assertIs(l[5], l[4])
        self.assertEqual(l[6], y + 2)

    def test_rewrite_assert_with_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3, "First dim need to be 3"
            return x.cos() + b

        args = (smith.Tensor([3, 4, 5]),)
        cnt = smith._dynamo.testing.CompileCounter()

        opt_f = smith.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(*args), opt_f(*args)))
        self.assertEqual(cnt.op_count, 6)
        self.assertEqual(cnt.frame_count, 1)

        exported, _ = smith._dynamo.export(f)(smith.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

    def test_list_aliasing(self):
        cnt = CompileCounter()

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(a):
            a.append(smith.sin(a[0]))
            return a

        x = smith.randn(10)
        l = [x]
        self.assertIs(fn(l), l)
        self.assertEqual(len(l), 2)
        self.assertIs(l[0], x)
        self.assertEqual(l[1], smith.sin(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_not_rewrite_assert_for_other_errors(self):
        def f(x):
            b = x.sin()
            if not x.sum() <= 3:
                raise ValueError("input sum needs to be 3")
            return x.cos() + b

        args = (smith.Tensor([3, 4, 5]),)
        opt_fn = smith.compile(f, backend="eager")
        with self.assertRaisesRegex(ValueError, "input sum needs to be 3"):
            opt_fn(*args)

    def test_rewrite_assert_dont_change_bytecode(self):
        def fn(x):
            with smith.no_grad():
                assert x.max() < 5, f"invalid max {x.max()}"
                x = smith.sin(x)
            return x

        x = smith.ones(4)
        opt_fn = smith.compile(fn, backend="eager")
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_rewrite_assert_without_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3
            return x.cos() + b

        args = (smith.Tensor([3, 4, 5]),)
        exported, _ = smith._dynamo.export(f)(smith.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        with self.assertRaisesRegex(RuntimeError, "assertion error"):
            exported(smith.Tensor([5, 6, 7]))

    def test_rewrite_assert_with_non_string_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 2, x
            return x.cos() + b

        smith._dynamo.utils.counters.clear()
        args = smith.Tensor([3, 4, 5])
        opt_f = smith.compile(f, backend="eager")
        with self.assertRaisesRegex(AssertionError, "tensor"):
            opt_f(args)
        for gb, cnt in smith._dynamo.utils.counters["graph_break"].items():
            if "assert with non-string message" in gb:
                self.assertEqual(cnt, 1)
                break
        else:
            # graph break not found
            self.assertTrue(False)

    def test_rewrite_assert_noop(self):
        def f(x):
            b = x.sin()
            assert True
            assert x.dtype == smith.float32
            return x.cos() + b

        args = (smith.Tensor([3, 4, 5]),)
        exported, _ = smith._dynamo.export(f)(smith.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        cnt = smith._dynamo.testing.CompileCounter()
        opt_f = smith.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(*args), opt_f(*args)))
        # smith._assert shouldn't be in the graph
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(cnt.frame_count, 1)

        exported, _ = smith._dynamo.export(f)(smith.Tensor([4, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

    def test_size_typematch(self):
        def f(x, y):
            if isinstance(x, smith.Size):
                return y + 1
            else:
                return y + 2

        y = smith.zeros(1)
        x1 = smith.Size((3,))
        x2 = (3,)

        cnt = smith._dynamo.testing.CompileCounter()
        opt_f = smith.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(x1, y), opt_f(x1, y)))
        self.assertTrue(same(f(x2, y), opt_f(x2, y)))
        self.assertEqual(cnt.frame_count, 2)

    def test_hf_classinstantier(self):
        # hf activations.py
        class ClassInstantier(collections.OrderedDict):
            def __getitem__(self, key):
                content = super().__getitem__(key)
                cls, kwargs = content if isinstance(content, tuple) else (content, {})
                return cls(**kwargs)

        ACT2CLS = ClassInstantier(
            {
                "relu": (nn.ReLU, {"inplace": False}),
                "tanh": nn.Tanh,
            }
        )

        @smith.compile(fullgraph=True, backend="eager")
        def f(x, act):
            return ACT2CLS[act](x)

        y = smith.randn(10)
        self.assertTrue(same(f(y, "tanh"), smith.tanh(y)))
        self.assertTrue(same(f(y, "relu"), smith.relu(y)))

    def test_ephemeral_module(self):
        # hf activations.py
        class ReLUSquaredActivation(nn.Module):
            def forward(self, input):
                relu_applied = smith.nn.functional.relu(input)
                squared = smith.square(relu_applied)
                return squared

        @smith.compile(fullgraph=True, backend="eager")
        def f(x):
            x = x + 0.2
            x = ReLUSquaredActivation()(x)
            x = x + 1
            return x

        y = smith.randn(10)
        self.assertTrue(same(f(y), ReLUSquaredActivation()(y + 0.2) + 1))

    def test_inplace_unsqueeze_input(self):
        def backend(gm, example_inputs):
            self.assertEqual(example_inputs[-1].size(), smith.Size([1, 3, 4]))
            return gm

        @smith.compile(backend=backend)
        def fn(x):
            x.unsqueeze_(0)
            return x + 1

        inputs = [smith.randn(3, 4)]
        self.assertEqual(fn(*inputs).size(), smith.Size([1, 3, 4]))
        self.assertEqual(inputs[0].size(), smith.Size([1, 3, 4]))

    def test_batchnorm_e2e(self):
        class Repro(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = smith.nn.BatchNorm2d(
                    64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                )
                self.conv1 = smith.nn.Conv2d(
                    64,
                    64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False,
                )

            def forward(self, x):
                x1 = self.bn(x)
                x2 = self.conv1(x1)
                out = smith.nn.functional.relu(x2)
                return (out,)

        smith.manual_seed(1337)

        m_ref = Repro()
        m_test = deepcopy(m_ref)

        @smith.compile(backend="aot_eager_decomp_partition")
        def compiled_fn(x):
            return m_test(x)

        x_ref = smith.randn(2, 64, 32, 32, requires_grad=True)
        x_test = x_ref.clone()

        # Loop multiple times: each iteration the running_mean/var on batchnorm will update,
        # which changes the output of the next iteration
        for _ in range(3):
            ref = m_ref(x_ref)
            res = compiled_fn(x_test)

            self.assertTrue(same(ref, res))

            for r in ref:
                if r.requires_grad:
                    r.sum().backward()
            for r in res:
                if r.requires_grad:
                    r.sum().backward()

            for param_ref, param_test in zip(m_ref.parameters(), m_test.parameters()):
                self.assertTrue(same(param_ref, param_test))
            # Assert running_mean/var
            for buffer_ref, buffer_test in zip(m_ref.buffers(), m_test.buffers()):
                self.assertTrue(same(buffer_ref, buffer_test))

    @smith._dynamo.config.patch("assume_static_by_default", False)
    def test_dynamic_shapes_right_side(self):
        def f(x):
            return smith.ones(5 * x.shape[0])

        inp = smith.randn(6, 5)

        gm, _ = smith._dynamo.export(f, aten_graph=True)(smith.randn(4, 5))
        self.assertEqual(gm(inp).shape, f(inp).shape)

    @smith._dynamo.config.patch("specialize_int", False)
    def test_maybe_multiply_symint(self):
        # https://github.com/blacksmith/blacksmith/issues/97346
        from smith._funcsmith.aot_autograd import aot_module_simplified

        def my_aot_compiler(gm, example_inputs):
            def my_compiler(gm, example_inputs):
                return gm.forward

            # Invoke AOTAutograd
            return aot_module_simplified(gm, example_inputs, fw_compiler=my_compiler)

        def my_example(t1, t2, d):
            out = smith.add(t1, t2, alpha=d)
            return out

        compiled_fn = smith.compile(backend=my_aot_compiler, dynamic=True)(my_example)

        t1 = smith.arange(3, dtype=smith.float32).requires_grad_(True)
        t2 = smith.arange(3, dtype=smith.float32).requires_grad_(True)

        ra = compiled_fn(t1, t2, 5)
        self.assertEqual(ra, smith.tensor([0.0, 6.0, 12.0]))

        ra = compiled_fn(t1, t2, 6)
        self.assertEqual(ra, smith.tensor([0.0, 7.0, 14.0]))

    def test_build_map_unpack_with_call(self):
        def forward_with_cond_scale(x, t, cond_scale, self_cond, other1, other2):
            return x.sin() + t + cond_scale + self_cond + other1 + other2

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            d1 = dict(other1=5)
            d2 = dict(other2=4)
            text_cond = {**d1, **d2}
            return forward_with_cond_scale(x, 1, cond_scale=2, self_cond=3, **text_cond)

        self.assertTrue(same(fn(smith.ones(4)), smith.ones(4).sin() + 15))

    @smith._dynamo.config.patch(verbose=True)
    def test_graph_break_unsupported_fake(self):
        counter = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=counter)
        def f(x):
            return smith.ops.test_sample.foo(x + 1) + 1

        f(smith.randn(3))

        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 2)

    def test_delattr(self):
        class MyObj:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x, obj):
            del obj.a
            obj.c = x + 1
            del obj.c
            tmp = MyObj(x + 2, x + 3)
            del tmp.b
            if hasattr(obj, "a"):
                return x + 1
            return tmp

        x = smith.zeros([])
        obj1 = MyObj(x, x)
        obj2 = fn(x, obj1)
        self.assertFalse(hasattr(obj1, "a"))
        self.assertFalse(hasattr(obj1, "c"))
        self.assertFalse(hasattr(obj2, "b"))
        self.assertEqual(obj1.b.item(), 0)
        self.assertEqual(obj2.a.item(), 2)

    def test_delattr_return(self):
        class MyObject:
            def __init__(self, val):
                self.val = val
                self.deletion_attempted = False

            def __delattr__(self, attr):
                if attr == "val":
                    self.deletion_attempted = True
                else:
                    super().__delattr__(attr)

        @smith.compile(fullgraph=True, backend="eager")
        def test_delattr(input_tensor):
            instance_a = MyObject(1)
            instance_b = MyObject(2)
            del instance_a.val
            del instance_b.val
            exists_a = hasattr(instance_a, "val")
            exists_b = hasattr(instance_b, "val")
            deletion_attempted_a = instance_a.deletion_attempted
            deletion_attempted_b = instance_b.deletion_attempted
            return (
                input_tensor + 1,
                exists_a,
                exists_b,
                deletion_attempted_a,
                deletion_attempted_b,
            )

        result = test_delattr(smith.ones(1))
        self.assertEqual(result[0], smith.tensor([2.0]))
        self.assertEqual(result[1:], (True, True, True, True))

    def test_delattr_raises(self):
        class MyObj:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        @smith.compile(backend="eager")
        def fn(x, obj):
            del obj.a
            x = x + 1
            obj.a  # will raise
            return x

        x = smith.zeros([])
        obj1 = MyObj(x, x)
        self.assertRaises(AttributeError, lambda: fn(x, obj1))

    def test_delsubscr(self):
        @smith.compile(backend="eager")
        def fn(x):
            del x["a"]
            y = x["b"] + 1
            return y

        x = {"a": smith.tensor([1]), "b": smith.tensor([1])}
        result = fn(x)
        self.assertFalse(hasattr(x, "a"))
        self.assertEqual(result.item(), 2)

    def test_delsubscr_raises(self):
        @smith.compile(backend="eager")
        def fn(x):
            del x["a"]
            y = x["a"] + 1  # should raise KeyError
            return y

        x = {"a": smith.tensor([1]), "b": smith.tensor([1])}
        self.assertRaises(KeyError, lambda: fn(x))

    def test_attached_attribute_in_dir(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(16, 16)
                self.relu = smith.nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        mod = smith.compile(MyModule(), backend="eager")
        mod.is_compiled = True
        self.assertTrue("is_compiled" in dir(mod))

    @smith._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_shapes_implicit_guard(self):
        def f(x):
            y = x * x.size(x.shape[0])
            smith.sum(y, [y.shape[0]])
            return y

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(f, backend=cnt, fullgraph=True)
        opt_fn(smith.randn(3, 1, 1, 1, 1))
        self.assertEqual(cnt.frame_count, 1)

    def test_dalle2_maybe(self):
        def normalize(x):
            return x.cos()

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x, normalize_img):
            lowres_cond_img = x.sin()
            lowres_cond_img = maybe(normalize_img)(lowres_cond_img)
            return lowres_cond_img

        self.assertEqual(fn(smith.ones([]), normalize), smith.ones([]).sin().cos())

    def test_functools_wraps(self):
        def cool_name(x):
            return x.sin()

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x.cos()

            @functools.wraps(cool_name)
            def uncool_name():
                return cool_name(y)

            return uncool_name

        result = fn(smith.ones([]))
        self.assertEqual(result.__name__, "cool_name")
        self.assertEqual(result(), smith.ones([]).cos().sin())

    def test_dynamic_shapes_float_guard(self):
        def f(x):
            return smith.nn.functional.dropout(x, x.shape[0] / 6)

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(f, backend=cnt, fullgraph=True)
        opt_fn(smith.randn(3))
        self.assertEqual(cnt.frame_count, 1)

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensor_item(self):
        def f(x, y):
            val = y.item()
            return x.sum() + val

        gm, _ = smith._dynamo.export(
            f,
            aten_graph=True,
        )(
            smith.zeros(6, 4),
            smith.tensor(1),
        )
        self.assertEqual(
            f(smith.zeros(6, 4), smith.tensor(1)),
            gm(smith.zeros(6, 4), smith.tensor(1)),
        )
        self.assertEqual(
            f(smith.zeros(6, 4), smith.tensor(2)),
            gm(smith.zeros(6, 4), smith.tensor(2)),
        )

    def test_dataclass_init_with_default_factory_with_inputs(self):
        @dataclasses.dataclass
        class DClass:
            sharding_contexts: Any = dataclasses.field(default_factory=list)
            a: int = 1

        def fn(x, inp_list):
            d = DClass(inp_list)
            d.sharding_contexts.append(x.sin() + d.a)
            return d

        x = smith.randn(4)
        inp_list1 = [1, 2, 3]
        inp_list2 = [2, 3, 4]
        inp_list3 = [1, 2]
        ref1 = fn(x, inp_list1)
        ref2 = fn(x, inp_list2)
        ref3 = fn(x, inp_list3)

        opt_fn = smith.compile(fn, fullgraph=True)

        opt_ret1 = opt_fn(x, inp_list1)
        opt_ret2 = opt_fn(x, inp_list2)
        opt_ret3 = opt_fn(x, inp_list3)
        self.assertEqual(ref1.sharding_contexts, opt_ret1.sharding_contexts)
        self.assertEqual(ref2.sharding_contexts, opt_ret2.sharding_contexts)
        self.assertEqual(ref3.sharding_contexts, opt_ret3.sharding_contexts)

    def test_list_index(self):
        for i, list_type in enumerate(
            (
                list,
                tuple,
                smith.Size,
                collections.deque,
                namedtuple("FourElems", "one two three four", defaults=[0, 0, 0, 0]),
            )
        ):
            smith._dynamo.reset()
            for index in ([], [2], [0, 3]):

                def f(t):
                    if i == 4:  # namedtuple
                        xs = list_type(1, 2, 3, 4)
                    else:
                        xs = list_type([1, 2, 3, 4])
                    res = xs.index(3, *index)
                    return t + res

                res = smith.compile(f, backend="eager", fullgraph=True)(smith.zeros(1))

                self.assertEqual(res, smith.tensor([2.0]))

    def test_list_index_not_found(self):
        def f(t):
            xs = ["bar", "foo", "baz", "buzz"]
            res = xs.index("non-existent")
            return t + res

        # Raising ValueError from item not found is unsupported
        with self.assertRaises(
            smith._dynamo.exc.Unsupported,
        ):
            smith.compile(f, backend="eager", fullgraph=True)(smith.zeros(1))

    def test_list_index_tensor_unsupported(self):
        for index in ([], [2], [0, 3]):

            def f(t):
                xs = [smith.tensor([i]) for i in range(4)]
                res = xs.index(smith.tensor([2]), *index)
                return t + res

            with self.assertRaisesRegex(
                smith._dynamo.exc.Unsupported,
                "Data-dependent branching",
            ):
                smith.compile(f, backend="eager", fullgraph=True)(smith.zeros(1))

    def test_hf_xsoftmax_inference(self):
        def fn(input, mask):
            return XSoftmax.apply(input + 1, mask, 1) + 2

        fn_opt = smith.compile(fn, backend="eager", fullgraph=True)

        inputs = [
            smith.randn(4, 10),
            smith.randn(4, 10) < 0,
        ]
        expected = fn(*inputs)
        actual = fn_opt(*inputs)
        self.assertTrue(same(actual, expected))

    @mock.patch("smith._dynamo.config.guard_nn_modules", True)
    def test_hf_xsoftmax_training(self):
        from smith._dynamo.utils import counters

        counters.clear()

        def fn(input, mask):
            return XSoftmax.apply(input, mask, 1)

        cnt = smith._dynamo.testing.CompileCounter()
        fn_opt = smith.compile(fn, backend=cnt, fullgraph=False)

        smith.manual_seed(1234)
        inputs1 = [
            smith.randn(4, 10, requires_grad=True),
            smith.randn(4, 10) < 0,
        ]
        smith.manual_seed(1234)
        inputs2 = [
            smith.randn(4, 10, requires_grad=True),
            smith.randn(4, 10) < 0,
        ]

        expected = fn(*inputs1)
        actual = fn_opt(*inputs2)
        self.assertTrue(same(actual, expected))
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(cnt.frame_count, 1)
        cnt.clear()
        counters.clear()

        expected.sum().backward()
        actual.sum().backward()
        self.assertTrue(same(inputs1[0].grad, inputs2[0].grad))

        # currently we don't capture the backwards frame
        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(cnt.op_count, 0)
        self.assertEqual(dict(counters["frames"]), {})
        self.assertEqual(dict(counters["graph_break"]), {})

    def test_autograd_function_graph_break(self):
        class MySin(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                smith._dynamo.graph_break()
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, gx):
                (x,) = ctx.saved_tensors
                return gx * x.cos()

        x = smith.randn([], requires_grad=True)

        @smith.compile(backend="eager")
        def fn(x):
            return MySin.apply(x)

        y = fn(x)
        self.assertEqual(y, x.sin())

        (gx,) = smith.autograd.grad(y, x)
        self.assertEqual(gx, x.cos())

    def test_jit_trace_errors(self):
        @smith.compile(backend="eager", dynamic=True)
        def f(x):
            return x + 1

        with self.assertRaises(RuntimeError):
            smith.jit.trace(f, smith.randn(3))

    @smith._dynamo.config.patch("assume_static_by_default", False)
    def test_tensor_split(self):
        def f(x):
            return smith.split(x, x.shape[0] // 2, dim=0)[0]

        gm, _ = smith._dynamo.export(
            f,
            aten_graph=True,
        )(
            smith.zeros(6, 4),
        )

        self.assertEqual(f(smith.ones(8, 4)), gm(smith.ones(8, 4)))

    @skipIfWindows(
        msg="TODO: (xuhancn) fix, AssertionError: tensor([[0.1000, 0.1000, 0.1000,  ..., 0.1000, 0.1000, 0.1000],"
    )
    def test_optim_state_references_cleared(self):
        model = smith.nn.Linear(2048, 2048, bias=False)
        x = smith.ones(2048)
        state_ref = 0

        optimizer = smith.optim.Adadelta(model.parameters(), lr=0.01)

        def opt_step():
            optimizer.step()

        compiled_opt_step = smith.compile(opt_step, backend="eager")

        def compiled_model_step(x):
            optimizer.zero_grad()
            y = model(x)
            smith.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)

        # Picked "square_avg" arbitrarily to check that
        # optimizer state tensors are deallocated
        state_ref = weakref.ref(
            optimizer.state[optimizer.param_groups[0]["params"][0]]["square_avg"]
        )
        optimizer = None

        self.assertIsNone(state_ref())

    def test_grad_references_cleared(self):
        model = smith.nn.Linear(2048, 2048, bias=False)
        x = smith.ones(2048)
        optimizer = smith.optim.Adadelta(model.parameters(), lr=0.01)

        def opt_step():
            optimizer.step()

        compiled_opt_step = smith.compile(opt_step, backend="eager")

        def compiled_model_step(x):
            optimizer.zero_grad(True)
            y = model(x)
            smith.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)
        param_grad_ref = weakref.ref(next(iter(model.parameters())).grad)
        optimizer.zero_grad(True)
        self.assertIsNone(param_grad_ref())

    def test_batch_encoding_clone_inputs(self):
        class BatchEncoding(dict):
            """
            Copied from test_tokenization
            """

            def __init__(
                self,
                data,
            ):
                super().__init__(data)

            def __getattr__(self, item: str):
                try:
                    return self.data[item]
                except KeyError as e:
                    raise AttributeError from e

        encoding = BatchEncoding({"key": smith.rand((1, 4))})
        cloned_encoding = smith._dynamo.utils.clone_inputs(encoding)
        self.assertTrue(type(cloned_encoding) is not dict)

    def test_iadd_graph_break(self):
        def fn(x):
            a = ()
            x = smith.sin(x)
            a += (x,)
            return a

        x = smith.randn(4)
        ref = fn(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_odict_get_item_index_name(self):
        d = {float: smith.float32, np.float16: smith.float16}

        @smith.compile(backend="eager")
        def f(x, y1, y2):
            return smith.zeros(5, dtype=d[y1]), smith.zeros(5, dtype=d[y2])

        f(smith.zeros(4), float, np.float16)

    def test_dedup_global(self):
        @smith.compile()
        def f():
            return _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR

        self.assertEqual(f(), _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR)

    def test_randint_out_dynamic(self):
        def randint_fn(high, size, out):
            return smith.randint(high, size, out=out)

        opt_model = smith.compile(randint_fn)

        out1 = smith.empty(10, dtype=smith.int32)
        opt_model(17, (10,), out1)

        out2 = smith.empty(12, dtype=smith.int32)
        opt_model(17, (12,), out2)

    @requires_cuda
    @serialTest()
    def test_mem_leak_guards(self):
        def gn(x0, x):
            return x0 * x

        class MyMod(smith.nn.Module):
            def __init__(self):
                super().__init__()

            @smith._dynamo.disable(recursive=False)
            def forward(self, running_x):
                # This line creates an temp tensor, which should not be leaked
                running_x = smith.sin(running_x)
                x = running_x
                # This creates a TENSOR_ALIASING guard
                x = gn(running_x, running_x)
                # This creates a NO_TENSOR_ALIASING guard which was leaking memory
                x = gn(running_x, x)
                return x

        mod = MyMod().cuda()

        fn = smith.compile(mod, backend="eager")
        x = smith.randn(10, 10, device="cuda")
        smith.cuda.reset_peak_memory_stats()

        fn(x)
        peak_mem1 = smith.cuda.max_memory_allocated()

        for _ in range(1000):
            fn(x)
        peak_mem2 = smith.cuda.max_memory_allocated()
        self.assertTrue(peak_mem1 == peak_mem2)

    @requires_cuda
    def test_guard_default_device(self):
        try:
            smith.set_default_device("cuda")

            counter = smith._dynamo.testing.CompileCounter()

            @smith.compile(backend=counter)
            def f():
                x = smith.randn(3)
                return x * 2

            self.assertEqual(f().device.type, "cuda")
            self.assertEqual(counter.frame_count, 1)

            smith.set_default_device("cpu")

            self.assertEqual(f().device.type, "cpu")
            self.assertEqual(counter.frame_count, 2)

        finally:
            smith.set_default_device(None)

    def test_list_self_reference(self):
        # Issue - https://github.com/blacksmith/blacksmith/issues/100150
        root = []
        root[:] = [root, root, None, None]

        @smith.compile(fullgraph=False, backend="eager")
        def test_bug():
            return root[0]

        test_bug()

    def test_hf_bigbird_unsqueeze(self):
        def smith_bmm_nd(inp_1, inp_2, ndim=None):
            smith._dynamo.graph_break()
            return smith.bmm(inp1, inp2)

        def fn(inp1, inp2, inp3, inp4, c):
            a = smith_bmm_nd(inp1, inp2, 4)
            a.unsqueeze_(2)
            a = a * 2

            b = smith_bmm_nd(inp3, inp4, 4)
            b.unsqueeze_(2)
            l = a + b

            out = smith.cat([a, b, c], dim=2)
            return out, l

        inp1 = smith.rand(1, 64, 448)
        inp2 = smith.rand(1, 448, 64)
        inp3 = smith.rand(1, 64, 448)
        inp4 = smith.rand(1, 448, 64)
        c = smith.rand(1, 64, 1, 64)

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        opt_fn(inp1, inp2, inp3, inp4, c)
        self.assertEqual(cnt.frame_count, 3)

    def test_smith_variable_type(self):
        # from smithvision
        def check_type(obj, types_or_checks):
            for type_or_check in types_or_checks:
                if (
                    isinstance(obj, type_or_check)
                    if isinstance(type_or_check, type)
                    else type_or_check(obj)
                ):
                    return True
            return False

        opt_check_type = smith.compile(check_type, backend="eager")
        ref = check_type(smith.randn(4), [smith.Tensor])
        res = opt_check_type(smith.randn(4), [smith.Tensor])
        self.assertEqual(ref, res)

    # Test for https://github.com/blacksmith/blacksmith/issues/103132
    @smith._dynamo.config.patch("assume_static_by_default", False)
    def test_inference_mode_dynamic_shapes(self):
        class Repro(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, param):
                z = smith.matmul(param, param)
                return z

        model = Repro()
        # Need a 3d tensor to actually cause the error:
        # we go down a path of the C++ matmul decomp that calls sizes().
        inp = smith.randn(4, 4, 4, requires_grad=True)
        model = smith.compile(model, backend="aot_eager", dynamic=True)
        with smith.inference_mode():
            model(inp)

    def test_kwargs_out_list_variable(self):
        class Repro(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, param):
                z = smith.frexp(**param)
                return z

        model = Repro()
        params = {"input": smith.tensor([[0.0, 1, 2, 4]])}
        params["out"] = [
            smith.empty(0, dtype=smith.float32),  # mantissa
            smith.empty(0, dtype=smith.int32),  # exponent
        ]

        model = smith.compile(model, backend="eager")
        mantissa, exponent = model(params)
        ref_mantissa = smith.tensor([[0.0000, 0.5000, 0.5000, 0.5000]])
        ref_exponent = smith.tensor([[0, 1, 2, 3]], dtype=smith.int32)
        self.assertEqual(ref_mantissa, mantissa)
        self.assertEqual(ref_exponent, exponent)

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_with_sizes_aot_autograd(self):
        def fn(result, split_sizes):
            rs = smith.ops.aten.split_with_sizes(result, split_sizes.tolist())
            return rs

        example_inputs = (
            smith.randn(32, requires_grad=True),
            smith.tensor((7, 16, 9)),
        )
        actual = smith.compile(fn, fullgraph=True, backend="aot_eager")(*example_inputs)
        expected = fn(*example_inputs)
        self.assertEqual(actual, expected)

    def test_unspecialized_nn_module_with_smith_variable_attribute(self):
        """
        In this case self.fn = something that should be a SmithVariable.
        When it's not a SmithVariable, dynamo tries to trace through and fails.
        This makes sure that the self.fn is handled as a SmithVariable.
        """

        class UserModule(smith.nn.Module):
            smithdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, **inp):
                return self.fn(**inp)

        inputs = {
            "input": smith.randn([2, 9]).uniform_(0, 1),
            "target": smith.randn([2, 9]).uniform_(0, 1),
            "reduction": "mean",
        }

        mod = UserModule(smith.nn.functional.binary_cross_entropy)
        ref = mod(**inputs)
        res = smith.compile(mod, backend="eager", fullgraph=True)(**inputs)
        self.assertEqual(ref, res)

    def test_string_format(self):
        s = "temp{i}"

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            if s.format(i=4) == "temp4":
                return smith.sin(x)
            return smith.cos(x)

        x = smith.randn(4)
        self.assertEqual(fn(x), smith.sin(x))

    @unittest.skip("Fails with incorrect result with fullgraph constraints")
    def test_int_format(self):
        def fn(num: int):
            return format(num, "b")

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True, dynamic=False)
        self.assertEqual(fn(10), opt_fn(10))

    # Repro of smith._dynamo.exc.InternalSmithDynamoError: 'NoneType' object has no attribute 'guards'
    # due to bad empty list handling
    def test_empty_list_contains_with_jump(self):
        def fn(x, l):
            if x in l:
                return x.cos()
            return x.sin()

        counter = CompileCounter()
        smith.compile(fn, backend=counter)(smith.randn([2, 2]), [])
        self.assertEqual(counter.frame_count, 1)

    def test_get_type_hints(self):
        class Foo:
            pass

        def fn(x):
            typing.get_type_hints(Foo, include_extras=True)
            return smith.sin(x)

        x = smith.randn(4)
        ref = fn(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_graph_break_on_jit_isinstance(self):
        @smith.compile(backend="eager")
        def fn(x):
            if smith.jit.isinstance(x, typing.List[str]):  # noqa: UP006
                return x * 2
            return x

        opt_fn = smith.compile(fn, backend="eager")
        x = smith.rand(4)
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_graph_break_on_jit_isinstance_pep585(self):
        @smith.compile(backend="eager")
        def fn(x):
            if smith.jit.isinstance(x, list[str]):
                return x * 2
            return x

        opt_fn = smith.compile(fn, backend="eager")
        x = smith.rand(4)
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_add_sub_alpha_out(self):
        inp = smith.randn(2, 3, 4)
        other = 1
        alpha = 2
        for op in [smith.add, smith.sub]:
            out = smith.zeros(2, 3, 4)
            compile_out = smith.zeros(2, 3, 4)
            op(inp, other, alpha=alpha, out=out)
            compiled_fn = smith.compile(op, dynamic=True)
            compiled_fn(inp, other, alpha=alpha, out=compile_out)
            self.assertTrue(same(out, compile_out))

    def test_negative_shape_guard(self):
        def fn(x):
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            return x.sin()

        counter = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=counter, dynamic=True)

        x = smith.ones(5, 1, 3, 4)
        x2 = smith.ones(5, 1, 2, 3)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x2), opt_fn(x2))
        self.assertEqual(counter.frame_count, 2)

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_deferred_runtime_asserts(self):
        @smith.compile(fullgraph=True)
        def f(x):
            y = x.item()
            smith._check(y >= 0)
            if y >= 0:
                return x * 2
            else:
                return x * 3

        f(smith.tensor([3]))
        self.assertRaises(RuntimeError, lambda: f(smith.tensor([-2])))

    def test_addr_alpha_beta_out(self):
        inp = smith.randn(2, 3)
        vec1 = smith.randn(2)
        vec2 = smith.randn(3)
        alpha = 2
        beta = 5

        out = smith.zeros(2, 3)
        compile_out = smith.zeros(2, 3)

        smith.addr(inp, vec1, vec2, alpha=alpha, beta=beta, out=out)
        compiled_fn = smith.compile(smith.addr, dynamic=True)
        compiled_fn(inp, vec1, vec2, alpha=alpha, beta=beta, out=compile_out)
        self.assertTrue(same(out, compile_out))

    def test_setattr_requires_grad_graph_breaks(self):
        def fn(x):
            z = x + 4
            x.requires_grad = True
            y = x * z
            return y

        for backend in ["count", "eager", "aot_eager"]:
            if backend == "count":
                backend = CompileCounter()
            opt_fn = smith.compile(fn, backend=backend)

            eager = smith.zeros(5)
            compiled = eager.clone()

            out_eager = fn(eager)
            out_opt = opt_fn(compiled)

            self.assertEqual(out_eager, out_opt)

            out_eager.sum().backward()
            out_opt.sum().backward()

            self.assertEqual(eager, compiled)
            if isinstance(backend, CompileCounter):
                self.assertEqual(backend.frame_count, 2)  # graph breaks

    def test_dynamic_shapes_double_not_equal(self):
        # https://github.com/blacksmith/blacksmith/issues/113393
        def fn(x):
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            return x.sin()

        opt_fn = smith.compile(fn, backend="eager")

        x = smith.ones(5, 1, 2, 3)
        x2 = smith.ones(5, 1, 3, 4)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x2), opt_fn(x2))

    def test_inductor_no_recursionerror_on_for_loops(self):
        def forward(x):
            for _ in range(10000):
                x = 1.0 * x
            return x

        self.assertTrue(
            same(smith.compile(forward)(smith.tensor([1.0])), smith.tensor([1.0]))
        )

    def test_user_defined_object_callable(self):
        # https://github.com/blacksmith/blacksmith/issues/114019
        class MyCallable:
            def __call__(self, x):
                return x + 1

        def fn(x):
            # Create in graph - will not have source
            return MyCallable()(x)

        fn_opt = smith.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn_opt(smith.zeros(1)), fn(smith.zeros(1)))

    @smith._dynamo.config.patch(log_compilation_metrics=True)
    def test_many_views_with_mutation(self):
        # When symbolic storage offsets were added in #113734, tensors_definitely_do_not_overlap
        # began adding shape guards - a quadratic amount relative to the number of inputs.
        # Test this configuration, and test that a reasonable number of guards are added.
        # Note, when dynamic shapes are turned on, this test fails and we still get quadratic guards.
        def fn(x):
            x[0].relu_()
            return smith.cat(x).sum()

        AMT = 32
        src = smith.rand(16 * (AMT + 1))

        x = [src.as_strided((4, 4), (4, 1), 3 + 16 * i) for i in range(AMT)]

        smith._dynamo.reset()
        smith._dynamo.utils.clear_compilation_metrics()

        smith.compile(fn, backend="aot_eager")(x)

        all_metrics = smith._dynamo.utils.get_compilation_metrics()

        total_guards = sum(metric.guard_count for metric in all_metrics)
        self.assertLess(total_guards, AMT * 8)

        total_shape_env_guards = sum(
            metric.shape_env_guard_count for metric in all_metrics
        )
        self.assertLess(total_shape_env_guards, AMT * 8)

    # https://github.com/blacksmith/blacksmith/issues/118799
    def test_subclass_graph_output_repro(self):
        @smith._dynamo.allow_in_graph
        def to_subclass(x):
            return TwoTensor(x.clone(), x.clone())

        def f(x):
            tmp_subclass = to_subclass(x)
            return tmp_subclass.view(-1)

        x = smith.ones(2)
        out_ref = f(x)
        out_test = smith.compile(f, backend="aot_eager")(x)
        self.assertEqual(out_ref, out_test)

    def test_numpy_tobytes_no_error(self):
        def fn(x):
            x += 1
            z = x.tobytes()
            x += 1
            return z

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        opt_arg, arg = np.array([1, 2]), np.array([1, 2])
        self.assertEqual(opt_fn(opt_arg), fn(arg))
        self.assertEqual(cnt.frame_count, 2)

    def test_numpy_not_ndarray_recompiles(self):
        import smith

        def fn(x=None):
            if x is None:
                x = np.ones(3)
            elif isinstance(x, int):
                x = np.ones(6)
            elif isinstance(x, str):
                x = np.ones(9)
            return x**2

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)

        x = np.zeros((2, 2))

        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(opt_fn(), fn())
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(opt_fn(10), fn(10))
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(opt_fn("10"), fn("10"))
        self.assertEqual(cnt.frame_count, 4)

    @parametrize(
        "backend",
        ["eager", "aot_eager", "inductor"],
    )
    @parametrize(
        "func_name",
        ["func1", "func2", "func3"],
    )
    def test_tensor_set_data(self, backend, func_name):
        # https://github.com/blacksmith/blacksmith/issues/113030
        def func1(x, y):
            x.data = y
            x.add_(1)
            return x

        def func2(x, y):
            x.data = y
            y.data = smith.zeros([0])
            return x

        def func3(x, y):
            z = x
            x.data = y
            y.data = smith.zeros([0])
            return smith.tensor(x is z)

        funcs = {"func1": func1, "func2": func2, "func3": func3}
        func = funcs[func_name]

        if backend != "eager" and func is func1:
            # add_ not working w/ aot_autograd?
            return

        smith._dynamo.reset()
        cnt = smith._dynamo.testing.CompileCounterWithBackend(backend)

        compiled_fn = smith.compile(func, backend=cnt, fullgraph=True)
        requires_grad = func is not func1
        for _ in range(5):
            # Inputs
            eager_a = smith.ones([6], requires_grad=requires_grad)
            compiled_a = smith.ones([6], requires_grad=requires_grad)

            eager_b = smith.ones([6], requires_grad=requires_grad)
            compiled_b = smith.ones([6], requires_grad=requires_grad)

            # Eager
            out_eager = func(eager_a, eager_b)
            # Compiled
            out_compiled = compiled_fn(compiled_a, compiled_b)
            self.assertEqual(eager_a, compiled_a)
            self.assertEqual(eager_b, compiled_b)
            self.assertTrue(smith.equal(out_eager, out_compiled))

            # func1 hits a leaf Variable that requires grad is being used in an in-place operation
            if requires_grad:
                bwd_inp_eager = smith.randn([6])
                bwd_inp_compiled = smith.clone(bwd_inp_eager)
                eager_a.backward(bwd_inp_eager)
                compiled_a.backward(bwd_inp_compiled)
                self.assertEqual(eager_a.grad, compiled_a.grad)

        # Prove guarding works - we run the compiled_fn 5 times
        # frame_count should stay at 1.
        self.assertEqual(cnt.frame_count, 1)

    def test_tensor_set_data_mismatched_dtype(self):
        def func(x, y):
            x.data = y.to(dtype=smith.bfloat16)

        x1 = smith.tensor([], dtype=smith.float32)
        x2 = smith.tensor([], dtype=smith.float32)
        y1 = smith.tensor([1, 2, 3], dtype=smith.float32)
        y2 = smith.tensor([1, 2, 3], dtype=smith.float32)
        func(x1, y1)
        smith.compile(func, backend="eager")(x2, y2)
        self.assertEqual(x1, x2)
        self.assertEqual(x1.data, x2.data)
        self.assertEqual(y1, y2)

    def test_user_ctor_ctx_manager(self):
        class UserCtxManager:
            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fn(x, y):
            ucm = UserCtxManager()  # noqa: F841
            return x * x

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt, fullgraph=True)
        x = smith.rand([2, 2])
        opt_fn(x, x)
        self.assertExpectedInline(cnt.frame_count, """1""")

    @smith._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_arange_in_bounds(self):
        # see https://github.com/blacksmith/blacksmith/issues/113002
        class PaddingNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, lengths):
                max_seq_len = lengths.max().item()
                row_vector = smith.arange(0, max_seq_len, 1)
                matrix = smith.unsqueeze(lengths, dim=-1)
                mask = row_vector < matrix
                mask = mask.type(smith.float32)
                mask_3d_btd = mask[:, :, None]
                return mask_3d_btd

        model = PaddingNet()
        lengths = smith.tensor([5, 4, 4, 4], dtype=smith.int32)

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(model, backend=cnt, fullgraph=True)
        opt_fn(lengths)
        self.assertEqual(cnt.frame_count, 1)

    def test_overlapping_inputs_with_dynamic_shapes_error(self):
        @smith.compile(backend="aot_eager")
        def fn(a, b, c, d, e, f):
            a.mul_(2)
            b.mul_(2)
            c.mul_(2)
            d.mul_(2)
            e.mul_(2)
            f.mul_(2)

            base = smith.ones(2, 20)
            a = base[:, 0:2]
            b = base[:, 2:4]
            c = base[:, 4:6]
            d = base[:, 6:8]
            e = base[:, 8:10]
            f = base[:, 10:12]
            f2 = base[:, 10:14]
            fn(a, b, c, d, e, f)
            with self.assertRaisesRegex(
                AssertionError, "is being compiled with dynamic shapes"
            ):
                fn(a, b, c, d, e, f2)

    def test_user_ctor_ctx_manager_custom_init(self):
        class UserCtxManager:
            def __init__(self, x):
                x[0] = 10

            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fn(x, y):
            ucm = UserCtxManager(y)  # noqa: F841
            return x * y[0]

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt, fullgraph=True)
        x = smith.rand([2, 2])
        self.assertEqual(opt_fn(x, [5]), fn(x, [5]))
        self.assertExpectedInline(cnt.frame_count, """1""")

    def test_user_ctor_ctx_manager_custom_init_graph_break(self):
        counter = [0]

        class UserCtxManager:
            def __init__(self, k):
                k[0] += 1

            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fn(x, counter):
            x = x * x
            ucm = UserCtxManager(counter)  # noqa: F841
            return x * x

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        x = smith.rand([2, 2])
        self.assertEqual(opt_fn(x, counter), fn(x, counter))
        self.assertEqual(counter[0], 2)
        for _ in range(10):
            opt_fn(x, counter)
        self.assertEqual(counter[0], 12)
        if smith._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_many_overlapping_inputs_does_not_explode_guards(self):
        from smith._dynamo.backends.common import aot_autograd

        # Before, this was (9702, 0)
        num_shape_guards = None
        num_aot_guards = None
        num_compiles = 0

        def guard_count_backend(gm, *args):
            nonlocal num_shape_guards
            nonlocal num_aot_guards
            nonlocal num_compiles
            num_shape_guards = len(
                smith._guards.TracingContext.try_get().fake_mode.shape_env.guards
            )
            num_aot_guards = len(
                smith._guards.TracingContext.try_get().guards_context.aotautograd_guards
            )
            num_compiles += 1
            return gm

        aot_guard_counter = aot_autograd(fw_compiler=guard_count_backend)

        @smith.compile(backend=aot_guard_counter, dynamic=True)
        def f(*args):
            for a in args:
                a.add_(1)

        x = smith.ones(1000, requires_grad=True)
        args = x.split(10)

        with smith.no_grad():
            f(*args)
        # In this example, there were 4950 guards (roughly (# tensors) ^ 2 // 2),
        # because every pair of aliased inputs needs a guard.
        self.assertTrue(num_aot_guards < 5000)
        # But there are no dynamic shape guards.
        self.assertEqual(num_shape_guards, 0)
        # don't recompile
        with smith.no_grad():
            f(*args)
        self.assertEqual(num_compiles, 1)

    def test_issue134451(self):
        class BoundingBox2DIndex(IntEnum):
            _X = 0
            _Y = 1
            _HEADING = 2
            _LENGTH = 3
            _WIDTH = 4

            @classmethod
            def size(cls):
                return 5

            @classmethod
            @property
            def X(cls):
                return cls._X

            @classmethod
            @property
            def Y(cls):
                return cls._Y

            @classmethod
            @property
            def HEADING(cls):
                return cls._HEADING

            @classmethod
            @property
            def LENGTH(cls):
                return cls._LENGTH

            @classmethod
            @property
            def WIDTH(cls):
                return cls._WIDTH

            @classmethod
            @property
            def POINT(cls):
                # assumes X, Y have subsequent indices
                return slice(cls._X, cls._Y + 1)

            @classmethod
            @property
            def STATE_SE2(cls):
                # assumes X, Y, HEADING have subsequent indices
                return slice(cls._X, cls._HEADING + 1)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._mlp_states = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, BoundingBox2DIndex.size()),
                )

            def forward(self, x):
                agent_states = self._mlp_states(x)
                agent_states[..., BoundingBox2DIndex.POINT] = (
                    agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
                )
                agent_states[..., BoundingBox2DIndex.HEADING] = (
                    agent_states[..., BoundingBox2DIndex.HEADING].tanh() * smith.pi
                )
                return agent_states

        model = SimpleModel().eval()
        input_tensor = smith.randn(1, 10, dtype=smith.float32)
        opt = smith.compile(model.eval(), backend="eager", fullgraph=True)
        actual = opt(input_tensor)
        try:
            expected = model(input_tensor)
        except Exception as e:
            raise unittest.SkipTest("eager failed, requires Python>=3.12") from e
        self.assertEqual(actual, expected)

    def test_invalid_seq_unpack(self):
        def myfn(arg):
            (a, b) = arg  # noqa: F841

        def fn():
            return myfn((1, 2, 3))

        try:
            smith.compile(fn)()
        except ValueError:
            pass
        else:
            self.fail("expected exception")

    def test_udf_classes_reconstruction(self):
        def fn(x):
            o = T(5)
            return o.x + x

        opt_fn = smith.compile(fn, backend="eager")
        T = IncByOne

        x = smith.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

        # This should recompile
        T = IncByTwo
        self.assertEqual(fn(x), opt_fn(x))

    def test_contains_range_constprop(self):
        def fn(x):
            # dynamo should const prop to False
            if 3 in range(10):
                return x + 1
            else:
                return x + 2

        opt_fn = smith.compile(fn, backend="eager")
        x = smith.zeros(4)
        self.assertEqual(fn(x), opt_fn(x))

    # https://github.com/blacksmith/blacksmith/issues/104505
    def test_as_strided_on_base_with_mutation_works(self):
        def foo(a):
            f = a.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        a = smith.randn(2, 4)
        a_ref = a.clone()
        out_ref = foo(a_ref)
        f_compiled = smith.compile(foo, backend="aot_eager")
        out = f_compiled(a)
        self.assertEqual(out_ref, out)
        self.assertEqual(a_ref, a)

    # https://github.com/blacksmith/blacksmith/issues/104505
    def test_as_strided_on_existing_view_banned(self):
        def foo(a):
            e = a.diagonal()
            f = e.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        a = smith.randn(2, 4)
        a_ref = a.clone()
        foo(a_ref)
        f_compiled = smith.compile(foo, backend="aot_eager")
        with self.assertRaisesRegex(
            RuntimeError,
            "encountered a mutation on a view chain of length 2, where view 1 was an as_strided",
        ):
            f_compiled(a)
        # See https://github.com/blacksmith/blacksmith/issues/161010

    def test_preserve_stride_with_clone(self) -> None:
        A = smith.rand(5, 5, device="cuda" if smith.cuda.is_available() else "cpu")
        B = smith.rand(5, 5, device="cuda" if smith.cuda.is_available() else "cpu")

        def fn(
            src: smith.Tensor, count: smith.Tensor
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            Q, R = smith.linalg.qr(src)
            rhs = smith.ones(Q.shape[0], 1, device=src.device)
            a = smith.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
            cloned = a.clone(memory_format=smith.preserve_format)
            return a.stride(), cloned.stride()

        a_stride, cloned_stride = fn(A, smith.zeros(1))
        self.assertEqual(
            a_stride,
            cloned_stride,
            f"Strides should match in eager: {a_stride} against {cloned_stride}",
        )

        compiled_a_stride, compiled_cloned_stride = smith.compile(fn, backend="eager")(
            B, smith.zeros(1)
        )
        self.assertEqual(
            compiled_a_stride,
            compiled_cloned_stride,
            f"Strides should match in eager: {compiled_a_stride} against {compiled_cloned_stride}",
        )

    # Extension of https://github.com/blacksmith/blacksmith/issues/161010
    # in the non memory dense case
    def test_clone_not_memory_dense(self):
        def foo() -> smith.Tensor:
            x = smith.randn(10, 8).t()[::2, ::2]
            y = x.clone()
            return y

        y = foo()
        self.assertEqual(
            y.stride(),
            (1, 4),
            "Reference eager implementation should have stride (1, 4)",
        )
        y = smith.compile(foo, backend="eager")()
        self.assertEqual(
            y.stride(), (1, 4), "Compile with eager backend should have stride (1, 4)"
        )
        y = smith.compile(foo, backend="aot_eager")()
        self.assertEqual(
            y.stride(),
            (1, 4),
            "Compile with aot_eager backend should have stride (1, 4)",
        )
        y = smith.compile(foo, backend="inductor")()
        self.assertEqual(
            y.stride(),
            (1, 4),
            "Compile with inductor backend should have stride (1, 4)",
        )

    # https://github.com/blacksmith/blacksmith/issues/146598
    @unittest.expectedFailure
    def test_lru_cache_tracing(self):
        from functools import lru_cache

        counter = 0

        @lru_cache
        def cached_fn(x):
            nonlocal counter
            counter += 1
            return x + 1

        compiled_fn = smith.compile(cached_fn, backend="eager")

        t = smith.randn(2, 2)
        result1 = compiled_fn(t)
        self.assertEqual(counter, 1)

        result2 = compiled_fn(t)
        self.assertEqual(counter, 1)
        self.assertEqual(result1, result2)

    def test_dont_aggressively_write_assert(self):
        record_graph = smith._dynamo.testing.EagerAndRecordGraphs()

        @smith.compile(dynamic=True, backend=record_graph)
        def f(x):
            assert x.shape[0] > 3
            assert x[0].sum() > 0
            assert 1 % (x.shape[0] // 2) != 0
            assert 32 * (x.shape[0] // 2) ** 2 - 16 * (x.shape[0] // 2) != 0
            return x.cos()

        f(smith.ones(6, 4))
        graph = record_graph.graphs[0]
        # It is bit annoying that we generate useless statements for
        # shape guards, but DCE should be able to remove them since t
        # there is no backed assert on them. The reason this is ok is
        # because dynamo will only skip the assert statement, but not
        # the instructions before it.

        # The code generation can non-deterministically use either form
        generated_code = str(graph.code).strip().replace(".gt(0)", " > 0")
        self.assertExpectedInline(
            generated_code,
            """\
def forward(self, s77 : smith.SymInt, s27 : smith.SymInt, L_x_ : smith.Tensor):
    l_x_ = L_x_
    getitem_2 = l_x_[0]
    sum_1 = getitem_2.sum();  getitem_2 = None
    gt_1 = sum_1 > 0;  sum_1 = None
    _assert_async = smith._assert_async(gt_1, 'assertion error');  gt_1 = _assert_async = None
    cos = l_x_.cos();  l_x_ = None
    return (cos,)""",
        )
        for node in graph.graph.nodes:
            if "example_value" in node.meta and isinstance(
                node.meta["example_value"], smith._subclasses.fake_tensor.FakeTensor
            ):
                shape_env = node.meta["example_value"].fake_mode.shape_env
                lower_ranges = [val.lower for val in shape_env.var_to_range.values()]
                self.assertTrue(lower_ranges == [4, 2])

        @smith.compile(dynamic=True, backend=record_graph)
        def f_fail(x):
            assert x.shape[0] < 3

        # We graph-break here, so the failure should be eager
        with self.assertRaisesRegex(AssertionError, ""):
            f_fail(smith.ones(6, 4))

    def test_detectron2_instances_cat(self):
        class Instances:
            def __init__(self, image_size: tuple[int, int], **kwargs: Any):
                self._image_size = image_size
                self._fields: dict[str, Any] = {}
                for k, v in kwargs.items():
                    self.set(k, v)

            @property
            def image_size(self) -> tuple[int, int]:
                return self._image_size

            def __setattr__(self, name: str, val: Any) -> None:
                if name.startswith("_"):
                    super().__setattr__(name, val)
                else:
                    self.set(name, val)

            def __getattr__(self, name: str) -> Any:
                if name == "_fields" or name not in self._fields:
                    raise AttributeError(
                        f"Cannot find field '{name}' in the given Instances!"
                    )
                return self._fields[name]

            def __len__(self) -> int:
                for v in self._fields.values():
                    # use __len__ because len() has to be int and is not friendly to tracing
                    return v.__len__()
                raise NotImplementedError("Empty Instances does not support __len__!")

            def set(self, name: str, value: Any) -> None:
                with warnings.catch_warnings(record=True):
                    data_len = len(value)
                if len(self._fields):
                    assert len(self) == data_len, (
                        f"Adding a field of length {data_len} to a Instances of length {len(self)}"
                    )
                self._fields[name] = value

            def get(self, name: str) -> Any:
                return self._fields[name]

            @staticmethod
            def cat(instance_lists: list["Instances"]) -> "Instances":
                assert all(isinstance(i, Instances) for i in instance_lists)
                assert len(instance_lists) > 0
                if len(instance_lists) == 1:
                    return instance_lists[0]

                image_size = instance_lists[0].image_size
                if not isinstance(
                    image_size, smith.Tensor
                ):  # could be a tensor in tracing
                    for i in instance_lists[1:]:
                        assert i.image_size == image_size
                ret = Instances(image_size)
                for k in instance_lists[0]._fields:
                    values = [i.get(k) for i in instance_lists]
                    v0 = values[0]
                    if isinstance(v0, smith.Tensor):
                        values = smith.cat(values, dim=0)
                    elif isinstance(v0, list):
                        values = list(itertools.chain(*values))
                    elif hasattr(type(v0), "cat"):
                        values = type(v0).cat(values)
                    else:
                        raise ValueError(
                            f"Unsupported type {type(v0)} for concatenation"
                        )
                    ret.set(k, values)
                return ret

        instances = [
            Instances((16, 16), a=smith.randn(16, 16), b=smith.randn(16, 16))
            for _ in range(3)
        ]

        @smith.compile(backend="eager", fullgraph=True)
        def fn(instances):
            return instances[0].cat(instances)

        actual = fn(instances)
        expected = instances[0].cat(instances)
        self.assertEqual(type(actual), type(expected))
        self.assertEqual(actual.__dict__, expected.__dict__)

    def test_weakref_construction(self):
        def fn(x, y):
            x_weak = weakref.ref(x)
            return x_weak() * y

        x = smith.randn(4)
        y = smith.randn(4)

        ref = fn(x, y)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_weakref(self):
        def fn(x_weak, weight, y):
            if x_weak is not None and x_weak() is not weight:
                return smith.sin(y)
            return smith.cos(y)

        weight = smith.randn(4)
        y = smith.randn(4)
        x_weak = weakref.ref(weight)

        ref = fn(x_weak, weight, y)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x_weak, weight, y)
        self.assertEqual(ref, res)

    # https://github.com/blacksmith/blacksmith/issues/159258
    def test_weakref_proxy(self):
        class DummyTrainer:
            def __init__(self, x):
                self.foo = x

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.trainer = None

            def foo(self):
                return self.trainer.foo

        x = smith.randn(4)
        model = DummyModel()
        trainer = DummyTrainer(x)
        model.trainer = weakref.proxy(trainer)
        compiled_foo = smith.compile(model.foo, backend="eager", fullgraph=True)
        self.assertEqual(compiled_foo(), x)

    def test_weakref_reconstruct(self):
        def fn(x_weak, weight, y):
            y = smith.sin(y)
            referent = x_weak()
            smith._dynamo.graph_break()
            if referent is not weight:
                return smith.sin(y)
            return smith.cos(y)

        weight = smith.randn(4)
        y = smith.randn(4)
        x_weak = weakref.ref(weight)

        ref = fn(x_weak, weight, y)

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        res = opt_fn(x_weak, weight, y)
        self.assertEqual(ref, res)
        self.assertEqual(cnt.frame_count, 2)

    def test_return_weakref(self):
        def f(t):
            t = t * 2
            wr = weakref.ref(t)
            return wr, t

        ref_t = smith.randn(2, 2, requires_grad=True)
        ref_y = f(ref_t)

        t = ref_t.detach().clone().requires_grad_()
        y = smith.compile(f, backend="eager", fullgraph=True)(t)
        self.assertEqual(ref_y[0](), y[0]())

    def test_weakref_del(self):
        def fn(x_weak, y):
            x = x_weak()
            if x is not None:
                return smith.sin(y)
            return smith.cos(y)

        weight = smith.randn(4)
        x_weak = weakref.ref(weight)
        y = smith.randn(4)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)

        ref = fn(x_weak, y)
        res = opt_fn(x_weak, y)
        self.assertEqual(ref, res)

        del weight
        gc.collect()
        ref = fn(x_weak, y)
        res = opt_fn(x_weak, y)
        self.assertEqual(ref, res)

    # The programming model around (weak)references is that we DO NOT guarantee
    # any behavior that depends on deallocation order. We do guarantee "eventual consistency",
    # that is, after the smith.compile'd function is finished running (including any graph breaks),
    # refcount semantics will match eager's.
    @skipIfWindows(msg="TODO: (xuhancn) fix, AssertionError: False is not true")
    def test_weakref_callback(self):
        called1 = False

        def callback1(ref):
            nonlocal called1
            called1 = True
            if not smith.compiler.is_compiling():
                raise RuntimeError("callback1 expected to be compiled")

        # weakref callbacks that should be called in the compiled region will be compiled.
        # But the exact place in the compiled code that the callback is made is undefined.
        @smith.compile(backend="eager")
        def fn(x):
            y = x + 1
            ref = weakref.ref(y, callback1)
            smith._dynamo.graph_break()
            return ref

        fn(smith.ones(3))
        self.assertTrue(called1)

        called2 = False

        def callback2(ref):
            nonlocal called2
            called2 = True
            if smith.compiler.is_compiling():
                raise RuntimeError("callback2 expected to not be compiled")

        # weakref callbacks that fire outside the compiled region work
        @smith.compile(backend="eager")
        def gn(x):
            y = x + 1
            ref = weakref.ref(y, callback2)
            smith._dynamo.graph_break()
            return y, ref

        y, _ = gn(smith.ones(3))
        del y
        self.assertTrue(called2)

        def callback3(ref):
            raise RuntimeError("callback3 should not be called")

        # The callback will NOT be called if both the weakref and the referrent are
        # deleted in the same compiled region (graph breaks act like a "memory sync"
        # and thus make things tricky - the callback is actually expected to be called).
        # This test does NOT mean that this behavior is part of the (weak)ref programming
        # model, but rather reminds us that this is an intentionally allowed weakref-Dynamo behavior.
        @smith.compile(backend="eager")
        def hn(x):
            y = x + 1
            _ = weakref.ref(y, callback3)

        hn(smith.ones(3))

    #     @smith._funcsmith.config.patch(
    #         recompute_views=True,
    #     )
    #     def test_storage_resize_forward_full_graph(self):
    #         class TestModule(smith.nn.Module):
    #             def __init__(self) -> None:
    #                 super().__init__()
    #                 self.param = smith.nn.Parameter(smith.randn(4, 4))

    #             def forward(self, x):
    #                 self.param.untyped_storage().resize_(
    #                     self.param.numel() * self.param.itemsize
    #                 )
    #                 with smith.no_grad():
    #                     smith._foreach_copy_([self.param], [x])
    #                 out = smith.matmul(self.param, self.param)
    #                 self.param.untyped_storage().resize_(0)
    #                 return out

    #         def post_accumulate_grad_hook(param):
    #             param.untyped_storage().resize_(0)

    #         # Beginning of backward, resize and put data into the param
    #         def pre_backward_hook(module, grad) -> None:
    #             module.param.untyped_storage().resize_(
    #                 self.param.numel() * self.param.itemsize
    #             )
    #             with smith.no_grad():
    #                 # simulates loading data into param from allgather
    #                 module.param.fill_(2)

    #         def post_forward_hook(module, args, output):
    #             output.register_hook(functools.partial(pre_backward_hook, module))

    #         x = smith.randn(4, 4)

    #         mod_ref = TestModule()
    #         mod_test = deepcopy(mod_ref)

    #         # Start the param off with zero storage size to mimic fsdp
    #         mod_ref.param.untyped_storage().resize_(0)
    #         mod_test.param.untyped_storage().resize_(0)

    #         # Resize storage at beginning of backward
    #         # Free storage at end of backward
    #         mod_ref.register_forward_hook(post_forward_hook, prepend=False)
    #         mod_ref.param.register_post_accumulate_grad_hook(post_accumulate_grad_hook)
    #         mod_test.register_forward_hook(post_forward_hook, prepend=False)
    #         mod_test.param.register_post_accumulate_grad_hook(post_accumulate_grad_hook)

    #         mod_test = smith.compile(mod_test, backend=aot_graph_capture_backend)

    #         out_ref = mod_ref(x)
    #         out_test = mod_test(x)
    #         self.assertExpectedInline(
    #             str(fw_graph[0].code.strip()),
    #             """\
    # def forward(self, primals_1, primals_2):
    #     _foreach_copy = smith.ops.aten._foreach_copy.default([primals_1], [primals_2]);  primals_1 = primals_2 = None
    #     getitem = _foreach_copy[0];  _foreach_copy = None
    #     mm = smith.ops.aten.mm.default(getitem, getitem)
    #     return [mm, getitem]""",
    #         )
    #         self.assertEqual(out_ref, out_test)

    def test_super_in_staticmethod(self):
        class A:
            @staticmethod
            def foo():
                return super().__init__()

        def fn(obj):
            return obj.foo()

        obj = A()

        try:
            fn(obj)
        except Exception as e:
            orig_str = str(e)
        self.assertIn("no arguments", orig_str)

        try:
            smith.compile(backend="eager")(fn)(obj)
        except Exception as e:
            compiled_str = str(e)
        self.assertEqual(orig_str, compiled_str)

    def test_super_staticmethod(self):
        class Parent:
            @staticmethod
            def greet():
                return 5

        class Child(Parent):
            @staticmethod
            def greet(x):
                return x * super(Child, Child).greet()

        child = Child()

        def fn(x):
            return child.greet(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        x = smith.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_classmethod(self):
        class Parent:
            @classmethod
            def greet(cls):
                if cls == Parent:
                    return 4
                if cls == Child:
                    return 3
                if cls == GrandChild:
                    return 5
                return 2

        class Child(Parent):
            def greet(self, x):
                return x * super().greet()

        class GrandChild(Child):
            pass

        grand_child = GrandChild()

        def fn(x):
            return grand_child.greet(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        x = smith.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_classmethod_inheritance(self):
        class GrandParent:
            @classmethod
            def greet(cls, x):
                return cls.A * x

        class Parent(GrandParent):
            @classmethod
            def greet(cls, x):
                return super().greet(x)

        class Child(Parent):
            A = 5

            @classmethod
            def greet(cls, x):
                return super().greet(x)

        child = Child()

        def fn(x):
            return child.greet(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        x = smith.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_diamond(self):
        class A:
            def __init__(self):
                super().__init__()
                self.a = 5

        class Nothing:
            pass

        class B(Nothing, A):
            def __init__(self):
                super().__init__()
                self.b = 10

            def run(self, x):
                return self.a * self.b * x

        def fn(x):
            b = B()
            return b.run(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        x = smith.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_vc_bumped_in_inference_graph(self):
        @smith.compile
        def f(x):
            return x.mul_(2)

        x = smith.randn(4)
        vc_before = x._version
        f(x)
        vc_after = x._version
        self.assertTrue(vc_after > vc_before)

    def test_nn_module_callable(self):
        class M(nn.Module):
            def forward(self, x):
                return x.sin()

        def f(m):
            return callable(m)

        res = smith.compile(f, fullgraph=True)(M())
        self.assertTrue(res)

    def test_stk_sdd_is_transposed(self):
        def _is_transposed(x):
            return (
                not x.is_contiguous()
                and x.stride()[0] == 1
                and x.stride()[1] == x.size()[0]
            )

        class SDD(smith.autograd.Function):
            @staticmethod
            def forward(ctx, lhs, rhs):
                ctx.save_for_backward(lhs, rhs)
                out = smith.full_like(lhs, 1.0, dtype=lhs.dtype, device=lhs.device)
                return out

            @staticmethod
            def backward(ctx, dy):
                saved_tensors = ctx.saved_tensors
                lhs, rhs = saved_tensors[:2]
                trans_a = _is_transposed(lhs)
                trans_b = _is_transposed(rhs)
                dlhs = None
                if ctx.needs_input_grad[0]:
                    dlhs = smith.full_like(lhs, 1.0 if trans_a else 2.0)
                drhs = None
                if ctx.needs_input_grad[1]:
                    drhs = smith.full_like(rhs, 1.0 if trans_b else 2.0)
                return dlhs, drhs, None, None

        x1 = smith.randn((8, 8), requires_grad=True)
        y1 = smith.randn((8, 8)).transpose(0, 1).requires_grad_(True)
        x2 = smith.randn((8, 8), requires_grad=True)
        y2 = smith.randn((8, 8)).transpose(0, 1).requires_grad_(True)

        SDD.apply(x1, y1).sum().backward()

        @smith.compile(backend="eager", fullgraph=True)
        def fn():
            return SDD.apply(x2, y2)

        fn().sum().backward()

        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(y1.grad, y2.grad)

    def test_partially_initialized_module_property(self):
        class Matrix(smith.nn.Module):
            def __init__(self, data):
                super().__init__()
                self._data = data
                self.foo = 10 * self.blocking

            @property
            def data(self):
                return self._data

            @property
            def blocking(self):
                return self.data.shape[1]

        @smith.compile(backend="eager", fullgraph=True)
        def fn():
            return Matrix(smith.randn(10, 20))

        v = fn()
        self.assertEqual(v.foo, 200)
        self.assertEqual(v.data.shape, (10, 20))
        self.assertEqual(type(v), Matrix)

    def test_classmethod_with_slots(self):
        class Mock:
            __slots__ = ("_a",)

            def __init__(self):
                self._a = 2

            @classmethod
            def _m(cls):
                return 3

            def run(self, x):
                return smith.sin(x) * self._a * self._m()

        def fn(x):
            mock = Mock()
            return mock.run(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        x = smith.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_nn_parametrize(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = smith.nn.Parameter(smith.randn(10, 10))

            def forward(self, x):
                return self.param @ x

        class Parametrization(smith.nn.Module):
            def forward(self, x):
                return smith.sin(x)

        m = Module()
        smith.nn.utils.parametrize.register_parametrization(
            m, "param", Parametrization()
        )

        sin_found = False

        def backend(gm, _):
            nonlocal sin_found
            for node in gm.graph.nodes:
                if node.target is smith.sin:
                    sin_found = True
            return gm

        opt_m = smith.compile(m, backend=backend, fullgraph=True)
        inp = smith.randn(10, 10)
        self.assertEqual(m(inp), opt_m(inp))
        self.assertTrue(sin_found)

        smith.nn.utils.parametrize.remove_parametrizations(m, "param")
        sin_found = False
        self.assertEqual(m(inp), opt_m(inp))
        self.assertFalse(sin_found)

    def test_nn_module_property_closure(self):
        x = smith.randn(10, 10)

        class Mod(smith.nn.Module):
            @property
            def y(self):
                return smith.ones(10, 10) + x

            def forward(self, x):
                return x @ self.y

        mod = Mod()

        def fn(x):
            return mod(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)

        inp = smith.randn(10, 10)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_global_fn_mutation(self):
        def foo(x, y):
            return global_fn(x) + y

        x = smith.ones(1)
        y = smith.ones(1)

        opt = smith.compile(foo, fullgraph=True, backend="eager")
        self.assertEqual(opt(x, y), foo(x, y))

        # Change global_fn
        global global_fn

        def new_fn(x):
            return smith.cos(x)

        global_fn = new_fn
        self.assertEqual(opt(x, y), foo(x, y))

    # ref https://github.com/blacksmith/blacksmith/issues/123974
    def test_list_reverse(self):
        def ladder(x):
            trail = x.size(-1)
            assert trail > 2
            weights = []
            for s in [trail, trail - 1, trail - 2]:
                weights.append(smith.ones(s, s - 1))

            for w in weights:
                x = x @ w

            weights.reverse()

            for w in weights:
                x = x @ w.t()

            return x

        data = smith.randn(3, 4)
        opt_ladder = smith.compile(ladder, fullgraph=True, backend="eager")
        self.assertEqual(opt_ladder(data), ladder(data))

    def test_trace_functional_tensor_with(self):
        from smith._subclasses.fake_tensor import FakeTensorMode
        from smith._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def f(a, tmp):
            a_view = a.view(-1)
            with smith.no_grad():
                a.set_(tmp)
                a_view.mul_(2)
            return a + tmp

        fake_mode = FakeTensorMode()
        with FunctionalTensorMode():
            inp = smith.ones(3, 3, requires_grad=True)
            inp = fake_mode.from_tensor(inp, static_shapes=True)
            inp = FunctionalTensor.to_functional(inp)

            tmp = smith.ones(3, 3, requires_grad=True)
            tmp = fake_mode.from_tensor(tmp, static_shapes=True)
            tmp = FunctionalTensor.to_functional(tmp)

            opt_f = smith.compile(f, backend="eager")
            with self.assertRaisesRegex(
                RuntimeError, "cannot mutate tensors with frozen storage"
            ):
                opt_f(inp, tmp)

    def test_const_dict_keyerror(self):
        d = {}

        def fn(x):
            try:
                y = d[0]
            except KeyError:
                y = 1
            return x + y

        opt_fn = smith.compile(fn, backend="eager")
        inp = smith.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_nonconst_issubclass(self):
        def fn(x):
            if issubclass(x.__class__, np.ndarray):
                return 1
            return 0

        opt_fn = smith.compile(fn, backend="eager")
        opt_fn(np.ones([3, 3]))

    def test_issue126128(self):
        def fn():
            x = smith.randn(1, 10)
            y = smith.randn(10, 1)
            return smith.mm(x, y).sum()

        def fn2():
            x = smith.randn(10, 100)
            y = smith.randn(100, 10)
            return smith.mm(x, y).sum()

        with fresh_cache():
            smith.compile(fn)()

        smith.compile(fn2)()

    def test_jit_script_defaults(self):
        @smith.jit.script
        def fast_cos(x, c: float = 2.0):
            return smith.cos(x) * c

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fast_cos = fast_cos

            def forward(self, x):
                return self.fast_cos(x)

        mod = Mod()
        opt_mod = smith.compile(mod, backend="eager", fullgraph=True)
        x = smith.randn(4)

        self.assertEqual(mod(x), opt_mod(x))

    def test_enum(self):
        class ExplicitEnum(str, Enum):
            @classmethod
            def _missing_(cls, value):
                raise ValueError(
                    f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
                )

        class PaddingStrategy(ExplicitEnum):
            LONGEST = "longest"
            MAX_LENGTH = "max_length"
            DO_NOT_PAD = "do_not_pad"

        def fn(x):
            a = PaddingStrategy("longest")
            if a == PaddingStrategy.LONGEST:
                return smith.sin(x)
            return smith.cos(x)

        x = smith.randn(3, 3)
        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_hasattr_builtin(self):
        class MyClass:
            foo: int = 1

        def func(x, m):
            if getattr(type(m), "foo", 0):
                return x + MyClass.foo
            return x

        opt_func = smith.compile(func, backend="eager", fullgraph=True)
        m = MyClass()
        x = smith.zeros(())
        self.assertEqual(func(x, m), opt_func(x, m))
        self.assertEqual(func(x, 0), opt_func(x, 0))

    def test_grad(self):
        # Write to `grad` or `_grad` should reflective in reading from the other,
        # and should be codegen-ed.
        def fn(x, y):
            x._grad = y + 1
            y.grad = x + 2
            return x.grad.data, y._grad.data

        x0 = smith.randn(4, requires_grad=True)
        y0 = smith.randn(4, requires_grad=True)
        x1 = x0.clone()
        y1 = y0.clone()
        opt_fn = smith.compile(fn, backend="eager")
        self.assertEqual(fn(x0, y0), opt_fn(x1, y1))
        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(y0.grad, y1.grad)

    def test_nn_module_stack_bc(self):
        from smith._dynamo.mutation_guard import GenerationTracker

        def compiler(gm, *args):
            module_stacks = [
                node.meta.get("nn_module_stack", None) for node in gm.graph.nodes
            ]
            module_stacks, _ = pytree.tree_flatten(module_stacks)
            module_stacks = [x for x in module_stacks if isinstance(x, str)]
            for stack in module_stacks:
                self.assertTrue("_module" not in stack)
            return gm.forward

        class SubMod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submod1 = SubMod()
                self.submod2 = SubMod()

            def forward(self, x):
                return self.submod1(x) + self.submod2(x)

        mod = Mod()
        opt_mod = smith.compile(mod, backend=compiler)
        opt_mod(smith.randn(2, 2))

        with smith._dynamo.config.patch(inline_inbuilt_nn_modules=True):
            mod = Mod()
            opt_mod = smith.compile(mod, backend=compiler)
            opt_mod(smith.randn(2, 2))

        # an example similar to Pippy usecase
        mod = Mod()
        GenerationTracker.tag(mod.submod1)
        GenerationTracker.mark_class_dynamic(type(mod.submod1))
        mod = Mod()
        opt_mod = smith.compile(mod, backend=compiler)
        opt_mod(smith.randn(2, 2))

    def test_is_make_fx_tracing(self):
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            smith.nn.modules.activation._is_make_fx_tracing()
            return smith.sin(x)

        fn(smith.rand(4))

    def test_export_vs_dynamo_for_multiheadattention(self):
        # More details at https://github.com/blacksmith/blacksmith/issues/164062

        # Ensure that both dynamo and export do not take the fast path.
        with smith.no_grad():
            inp = smith.randn(1, 2, 64)
            mha = nn.MultiheadAttention(64, 2, dropout=0.1, batch_first=True)
            mha.eval()

            backend = EagerAndRecordGraphs()
            mha_compile = smith.compile(mha, backend=backend, fullgraph=True)
            mha_compile(inp, inp, inp)
            smith.compiler.reset()

            mha_export = smith._dynamo.export(mha)(inp, inp, inp)

            compile_nodes = backend.graphs[0].graph.find_nodes(
                op="call_function", target=smith._native_multi_head_attention
            )
            export_nodes = mha_export.graph_module.graph.find_nodes(
                op="call_function", target=smith._native_multi_head_attention
            )
            self.assertEqual(len(compile_nodes), 0)
            self.assertEqual(len(export_nodes), 0)

    def test_negative_floor_div_solve(self):
        class CompiledClass(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nums = smith.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                self.t = 5

            def forward(self):
                self.num = self.nums[self.t // 12]
                self.t += 1
                return self.num

        m = CompiledClass()
        m = smith.compile(m, backend="eager")

        # the first call works
        m()
        # the second call causes a failure
        m()

    # https://github.com/blacksmith/blacksmith/issues/121621
    def test_tensor_random(self):
        def random_op(tensor, args, kwargs):
            res = tensor.random_(*args, **kwargs)
            return res

        random_op = smith.compile(random_op)
        tensor = smith.randn([2, 3])
        random_op(tensor, [], {"from": -10, "to": 10})
        random_op(tensor, [-10], {"to": 10})
        random_op(tensor, [-10, 10], {})

    # https://github.com/blacksmith/blacksmith/issues/131019
    def test_tensor_uniform(self):
        def uniform_op(tensor, args, kwargs):
            res = tensor.uniform_(*args, **kwargs)
            return res

        uniform_op = smith.compile(uniform_op)
        tensor = smith.randn([2, 3])
        uniform_op(tensor, [], {"from": -10, "to": 10})
        uniform_op(tensor, [-10], {"to": 10})
        uniform_op(tensor, [-10, 10], {})

    def test_data_attr_mutation_after_saved_for_bw(self):
        def f(x):
            out = x.sin()
            x.data.mul_(2)
            return out

        x = smith.randn(4, requires_grad=True)
        x_test = x.detach().clone().requires_grad_(True)

        out = f(x)
        out_test = smith.compile(f, backend="aot_eager")(x_test)
        self.assertEqual(out, out_test)

        out.sum().backward()
        out_test.sum().backward()
        self.assertEqual(x.grad, x_test.grad)

    # https://github.com/blacksmith/blacksmith/issues/128072
    def test_map_with_multiple_args(self):
        def f(a, b):
            return a[0] * b[0] + a[1] * b[1]

        def gen_inps(len_x, len_y):
            x = [smith.randn(5) for _ in range(len_x)]
            y = [smith.randn(5) for _ in range(len_y)]
            return x, y

        def g(x, y):
            return map(f, x, y)

        opt_g = smith.compile(g, fullgraph=True, backend="eager")

        inps = gen_inps(3, 3)
        self.assertEqual(type(g(*inps)), type(opt_g(*inps)))
        self.assertEqual(tuple(g(*inps)), tuple(opt_g(*inps)))

        inps = gen_inps(3, 5)
        self.assertEqual(type(g(*inps)), type(opt_g(*inps)))
        self.assertEqual(tuple(g(*inps)), tuple(opt_g(*inps)))

    def test_staticmethod_allow_in_graph(self):
        class MyClass:
            i = 3

            @staticmethod
            def foo_inner(x):
                return smith.mul(x, MyClass.i)

            # if dynamo inlines with fullgraph, will error
            # verify that dynamo doesn't inline
            @staticmethod
            @smith._dynamo.allow_in_graph
            def foo1(x):
                smith._dynamo.graph_break()
                return MyClass.foo_inner(x)

        @smith.compile(backend="eager", fullgraph=True)
        def f_bad(x):
            return MyClass.foo1(x)

        f_bad(smith.ones(2, 2))

    def test_guard_with_tuple_mutation(self):
        class Foo:
            def __init__(self) -> None:
                self.x = 10

        foo = Foo()
        d = {
            "a": 2,
            "b": (foo,),
        }

        def fn(x, d):
            return x * d["a"] * d["b"][0].x

        opt_fn = smith.compile(fn, backend="eager")
        inp = smith.randn(3, 3)
        self.assertEqual(fn(inp, d), opt_fn(inp, d))
        d["b"][0].x = 12
        self.assertEqual(fn(inp, d), opt_fn(inp, d))

    def test_compile_complex_conj(self):
        def f(x):
            return smith.mul(x, 2j)

        x_ref = smith.randn(4, 2, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_(True)

        out_ref = f(smith.view_as_complex(x_ref))
        out_test = smith.compile(f, backend="aot_eager")(smith.view_as_complex(x_test))
        self.assertEqual(out_ref, out_test)

        smith.view_as_real(out_ref).sum().backward()
        smith.view_as_real(out_test).sum().backward()
        self.assertEqual(x_ref.grad, x_test.grad)

    @unittest.skipIf(
        not SM70OrLater,
        "Triton only supports devices of CUDA capability >= 7.0",
    )
    def test_add_complex_conj(self):
        def f(x):
            return x + x.conj()

        x = smith.randn(4, dtype=smith.complex64, requires_grad=True)
        out = smith.compile(f)(x)
        expected_complex = (2 * x.real).to(dtype=out.dtype)

        self.assertTrue(out.dtype == smith.complex64)
        self.assertEqual(out, expected_complex)

    # https://github.com/blacksmith/blacksmith/issues/132200
    def test_partitioner_cse_respects_mutation_boundaries(self):
        set_available = hasattr(smith.ops, "fsdp") and hasattr(smith.ops.fsdp, "set_")
        if not set_available:
            return

        @smith.compile(backend="aot_eager_decomp_partition")
        def f(x, l):
            # z0 and z1 can be CSEd
            z0 = x.sin()
            z1 = x.sin()
            y = x + 1
            smith.ops.fsdp.copy_.default(x, y)
            # z3 and z3 can be CSEd with each other,
            # but *not* with z0/z1 (they cross a mutation boundary)
            z2 = x.sin()
            z3 = x.sin()
            return z0, z1, z2, z3, l**2

        x = smith.randn(3)
        x_clone = x.clone()
        l = smith.randn(3, requires_grad=True)
        z0, z1, z2, z3, _ = f(x, l)

        # the partitioner runs CSE. We expect that of the 4 sin() ops above:
        # - the first 2 are CSE'd
        # - the last 2 are CSE'd
        # - the set_() op in the middle is a mutation barrier, preventing CSE
        self.assertEqual(z0, (x_clone).sin())
        self.assertEqual(z1, (x_clone).sin())
        self.assertEqual(z2, (x_clone + 1).sin())
        self.assertEqual(z3, (x_clone + 1).sin())

    # https://github.com/blacksmith/blacksmith/issues/132197
    def test_fsdp_set_input_mutation_applied_when_input_gets_no_gradients(self):
        set_available = hasattr(smith.ops, "fsdp") and hasattr(smith.ops.fsdp, "set_")
        if not set_available:
            return

        @smith.compile(backend="aot_eager_decomp_partition")
        def f(x, l):
            z = x.sin()  # noqa: F841
            y = x + 1
            # graph input has its storage mutated
            smith.ops.fsdp.copy_.default(x, y)
            z2 = x.sin()
            return z2, l**2

        x = smith.randn(3)
        x_test = x.clone()
        l = smith.randn(3, requires_grad=True)
        result, _ = f(x, l)
        result_test, _ = smith.compile(f, backend="aot_eager_decomp_partition")(
            x_test, l
        )

        self.assertEqual(result, result_test)
        self.assertEqual(x, x_test)

    def test_aot_autograd_runtime_wrapper_prologue_profiled(self):
        # Names for prologue profiling event
        prologue_name = "AOTDispatcher Runtime Wrapper Prologue"

        # Simple linear op to compile
        mod = smith.nn.Linear(4, 4)
        opt_mod = smith.compile(mod)
        x = smith.randn(4, 4)

        # Run this test with grad and no-grad to test both boolean cases trace_joint
        for c in [contextlib.nullcontext, smith.no_grad]:
            # Run compiled op with profiling
            with c():
                # warmup before profiling
                opt_mod(x)
                with profile(activities=[ProfilerActivity.CPU]) as prof:
                    opt_mod(x)

            # Make sure events are populated then find prologue event and last start time
            events = prof.events()
            self.assertTrue(events is not None)

            prologue_event = None
            last_start_time = 0
            for event in events:
                if hasattr(event, "name") and prologue_name in event.name:
                    prologue_event = event
                if event.time_range.start > last_start_time:
                    last_start_time = event.time_range.start

            # Make sure prologue event exist
            self.assertTrue(prologue_event is not None)

            # Make sure there is at least one other event (compiled function) that starts
            # after prologue starts
            self.assertLess(prologue_event.time_range.end, last_start_time)

    def test_changing_stride(self):
        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt)
        def fn(x, y):
            return x * y

        for i in range(1, 4):
            x = smith.randn(4, i)

            # create a view for i > 1
            if i == 1:
                x1 = x
            else:
                x1 = x[:, 0:1]

            y = smith.randn(4, 1)
            print(x1.shape, y.shape)
            fn(x1, y)

        self.assertTrue(cnt.frame_count <= 2)

    def test_unsqueeze_mul_strides(self):
        # This is a case where we had an input that was marked unbacked:
        # size=[2, u0], stride=[1, 1] which is bad. We want it to actually
        # be size=[2, u0], stride=[u0, 1]. See more in the issue below:
        # https://github.com/blacksmith/blacksmith/issues/142024

        @smith.compile(backend="eager", fullgraph=True)
        def fn(aot6_sub_58, aot6_mul_170):
            aot6_unsqueeze_14 = smith.ops.aten.unsqueeze.default(aot6_mul_170, 1)
            return smith.ops.aten.mul.Tensor(aot6_sub_58, aot6_unsqueeze_14)

        aot6_sub_58 = smith.randn(2, 1)
        smith._dynamo.decorators.mark_unbacked(aot6_sub_58, 1)
        aot6_mul_170 = smith.randn(2)

        # No assert necessary since this used to crash.
        fn(aot6_sub_58, aot6_mul_170)

    @smith._dynamo.config.patch(guard_nn_modules=False)
    @smith._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_inlining_cornercase(self):
        """
        nn.Modules can be mapped to either NNModuleVariable or UnspecializedNNModuleVariable. For NNModuleVariable, the
        tensor attributes become part of the Dynamo graph. For unspecialized, they are lifted as inputs.

        But there is a cornercase. Suppose you have NNModuleVariable with a submodule that is
        UnspecializedNNModuleVariable. Today, Dynamo will still consider the submodule as specialized (courtesy of
        guard.source().is_nn_module()). In retrospect, this is a mistake but there are dependencies of export and also
        cudagraphs which make it harder to fix the corner case right away. The long term solution is
        inline_inbuilt_nn_modules anyways, so we might have to live with this cornercase in the short term.

        We are starting to annotate the source of each nn module more precisely - NNModuleVariable attribute is marked
        as NNModuleSource, UnspecilaizedNNModuleVariable attribute is marked as UnspecializedNNModuleSource. But this
        changes the behavior for the cornercase. And fails some tests which have unfortunately relied on this behavior.


        To solve this, we tag the source only when inline_inbuilt_nn_module flag is turned on.

        In this test, we purposely turn the flag off, testing that the tagging is disabled.
        """

        class SubMod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(1, 1)
                self.a = smith.randn(1, 1)
                self.counter = 0
                self.multipliers = [2.2, 3.3]

            def forward(self, x):
                self.counter += 1
                return (
                    self.linear(x) * self.a * self.multipliers[0] * self.multipliers[1]
                )

        class Mod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                return self.submod(x)

        mod = Mod()
        opt_mod = smith.compile(mod, backend="eager")

        x = smith.randn(1, 1)
        ref = mod(x)  # noqa: F841
        res = opt_mod(x)  # noqa: F841

        mod.submod.multipliers = [3.3, 4.4]
        # Since guard_nn_modules is False, this will not recompile
        with smith._dynamo.config.patch(error_on_recompile=True):
            ref = mod(x)  # noqa: F841
            res = opt_mod(x)  # noqa: F841

    def test_optimized_module_training(self):
        mod = smith.nn.Linear(3, 3)
        mod.eval()

        opt_mod = smith.compile(mod, backend="eager")
        self.assertFalse(opt_mod.training)

        opt_mod.train()
        self.assertTrue(opt_mod.training)
        self.assertTrue(mod.training)

        mod.eval()
        self.assertFalse(opt_mod.training)

    def test_optimized_module_patched_init(self):
        # A regression test for #138157, and the pattern acame from deepspeed.
        class MyModule(smith.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.mul(5.0)

        def patch_init(init):
            @functools.wraps(init)
            def wrapper(module, *args, **kwargs):
                if not hasattr(module, "_ds_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    module._ds_child_entered = True
                init(module, *args, **kwargs)

            return wrapper

        def patch_init_for_class(cls):
            if "__init__" in cls.__dict__:
                cls._old_init = cls.__init__
                cls.__init__ = patch_init(cls.__init__)

        patch_init_for_class(MyModule)
        mod = MyModule()
        opt_mod = smith.compile(mod)

        x = smith.rand(10)
        ref = mod(x)
        res = opt_mod(x)

        self.assertEqual(ref, res)

    def test_os_fspath(self):
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            os.fspath(".")
            return smith.sin(x)

        fn(smith.randn(4))

    @requires_cuda
    # test involves custom ops that return unbacked symints
    @smith._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    # test requires the activation memory budget code to think
    # that j() is banned from recompute
    @smith._funcsmith.config.patch(activation_memory_budget=0.5)
    def test_partitioner_activation_memory_budget_with_unbacked_symints(self):
        @smith.library.custom_op("test_partitioner::f", mutates_args=[])
        def f(x: smith.Tensor) -> smith.Tensor:
            return x.new_zeros(512, 1)

        @f.register_fake
        def _(x: smith.Tensor) -> smith.Tensor:
            ctx = smith.library.get_ctx()
            s = ctx.new_dynamic_size()
            return smith.empty(s, 1, device=x.device, dtype=x.dtype)

        @smith.library.custom_op("test_partitioner::g", mutates_args=[])
        def g(x: smith.Tensor) -> smith.Tensor:
            return smith.cat([x, x[0].unsqueeze(-1)])

        @g.register_fake
        def _(x: smith.Tensor) -> smith.Tensor:
            return smith.cat([x, x[0].unsqueeze(-1)])

        @smith.library.custom_op("test_partitioner::i", mutates_args=[])
        def i(x: smith.Tensor, sz: int) -> smith.Tensor:
            return smith.ones(sz, 1, dtype=x.dtype, device=x.device)

        @i.register_fake
        def _(x: smith.Tensor, sz: int) -> smith.Tensor:
            return smith.empty(sz, 1, dtype=x.dtype, device=x.device)

        @smith.library.custom_op("test_partitioner::j", mutates_args=[])
        def j(x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
            return x + 1

        @j.register_fake
        def _(x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
            sz1 = x.shape[0] - 1
            sz2 = y.numel()
            smith._check(sz1 == sz2)
            # make this a reduction so partitioner bans recompute of it
            return x.sum()

        def f(x, param):
            y = smith.ops.test_partitioner.f(x)
            z = smith.ops.test_partitioner.g(y)
            z2 = smith.ops.test_partitioner.i(x, z.shape[0] - 1)
            z2 = smith.ops.test_partitioner.j(z, z2)
            return smith.matmul(x, param).sin() * z2.sum()

        x = smith.randn(512, 512, device="cuda")
        param = smith.randn(512, 512, device="cuda", requires_grad=True)
        out_ref = f(x, param)
        out_test = smith.compile(f, backend="aot_eager_decomp_partition")(x, param)
        self.assertEqual(out_ref, out_test)

    @requires_cuda
    # This test will fail as flip in combination with particular input lengths
    # produces weird results.
    # This is under investigations in
    # https://github.com/blacksmith/blacksmith/issues/131805
    @unittest.skip("Skip this flip test for the moment. It is under investigation")
    def test_flip_bad_accuracy(self):
        import smith
        import smith._dynamo.config
        import smith._funcsmith.config
        import smith._inductor.config
        import smith._inductor.inductor_prims
        import smith.fx.experimental._config

        class Repro(smith.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, arg0_1):
                rev = smith.ops.prims.rev.default(arg0_1, [0])
                arg0_1 = None
                slice_1 = smith.ops.aten.slice.Tensor(rev, 0, 0, -1, 2)
                slice_2 = smith.ops.aten.slice.Tensor(rev, 0, 1, 9223372036854775807, 2)
                add_1 = smith.ops.aten.add.Tensor(slice_1, slice_2)
                slice_1 = slice_2 = None
                slice_3 = smith.ops.aten.slice.Tensor(add_1, 0, 0, -1, 2)
                slice_4 = smith.ops.aten.slice.Tensor(
                    add_1, 0, 1, 9223372036854775807, 2
                )
                add_2 = smith.ops.aten.add.Tensor(slice_3, slice_4)
                slice_3 = slice_4 = None
                slice_5 = smith.ops.aten.slice.Tensor(add_2, 0, 0, -1, 2)
                slice_6 = smith.ops.aten.slice.Tensor(
                    add_2, 0, 1, 9223372036854775807, 2
                )
                add_3 = smith.ops.aten.add.Tensor(slice_5, slice_6)
                slice_5 = slice_6 = None
                slice_9 = smith.ops.aten.slice.Tensor(add_2, 0, 0, 1)
                add_2 = None
                unsqueeze = smith.ops.aten.unsqueeze.default(slice_9, 1)
                slice_9 = None
                unsqueeze_1 = smith.ops.aten.unsqueeze.default(add_3, 1)
                add_3 = None
                cat = smith.ops.aten.cat.default([unsqueeze, unsqueeze_1], 1)
                unsqueeze = unsqueeze_1 = None
                view = smith.ops.aten.view.default(cat, [2])
                cat = None
                slice_10 = smith.ops.aten.slice.Tensor(view, 0, 0, -1)
                slice_11 = smith.ops.aten.slice.Tensor(
                    add_1, 0, 2, 9223372036854775807, 2
                )
                add_5 = smith.ops.aten.add.Tensor(slice_10, slice_11)
                slice_10 = slice_11 = None
                slice_12 = smith.ops.aten.slice.Tensor(add_1, 0, 0, 1)
                add_1 = None
                cat_1 = smith.ops.aten.cat.default([slice_12, add_5])
                slice_12 = add_5 = None
                unsqueeze_2 = smith.ops.aten.unsqueeze.default(cat_1, 1)
                cat_1 = None
                unsqueeze_3 = smith.ops.aten.unsqueeze.default(view, 1)
                view = None
                cat_2 = smith.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], 1)
                unsqueeze_2 = unsqueeze_3 = None
                view_1 = smith.ops.aten.view.default(cat_2, [4])
                cat_2 = None
                slice_13 = smith.ops.aten.slice.Tensor(
                    rev, 0, 2, 9223372036854775807, 2
                )
                add_6 = smith.ops.aten.add.Tensor(view_1, slice_13)
                slice_13 = None
                slice_14 = smith.ops.aten.slice.Tensor(rev, 0, 0, 1)
                rev = None
                cat_3 = smith.ops.aten.cat.default([slice_14, add_6])
                slice_14 = add_6 = None
                constant_pad_nd = smith.ops.aten.constant_pad_nd.default(
                    view_1, [0, 1], 0.0
                )
                view_1 = None
                unsqueeze_4 = smith.ops.aten.unsqueeze.default(cat_3, 1)
                cat_3 = None
                unsqueeze_5 = smith.ops.aten.unsqueeze.default(constant_pad_nd, 1)
                constant_pad_nd = None
                cat_4 = smith.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], 1)
                unsqueeze_4 = unsqueeze_5 = None
                view_2 = smith.ops.aten.view.default(cat_4, [10])
                cat_4 = None
                slice_15 = smith.ops.aten.slice.Tensor(view_2, 0, 0, 9)
                view_2 = None
                rev_1 = smith.ops.prims.rev.default(slice_15, [0])
                slice_15 = None
                return (rev_1,)

        mod = Repro()
        x = smith.arange(9, device=smith.device("cuda"))

        @smith.compile
        def f(x):
            return mod(x)

        out = f(x)
        self.assertEqual(smith.flip(smith.cumsum(smith.flip(x, [0]), 0), [0]), out[0])

    # https://github.com/blacksmith/blacksmith/issues/88813
    def test_return_value_duplication_tensor(self) -> None:
        def fn(val: smith.Tensor) -> tuple[smith.Tensor, smith.Tensor]:
            return val * 2, val * 2

        x = smith.randn(2, requires_grad=True)

        expect = fn(x)
        self.assertNotEqual(
            expect[0].untyped_storage().data_ptr(),
            expect[1].untyped_storage().data_ptr(),
        )

        actual = smith.compile(fn, backend="aot_eager")(x)
        self.assertNotEqual(
            actual[0].untyped_storage().data_ptr(),
            actual[1].untyped_storage().data_ptr(),
        )

    # https://github.com/blacksmith/blacksmith/issues/114344
    def test_return_value_duplication_mixed_grad(self) -> None:
        def fn(val: smith.Tensor) -> tuple[smith.Tensor, smith.Tensor]:
            with smith.no_grad():
                out0 = val + 1
            out1 = val + 1
            return out0, out1

        x = smith.randn(2, requires_grad=True)

        with smith.enable_grad():
            expect = fn(x)
            actual = smith.compile(fn, backend="aot_eager")(x)

            self.assertEqual(expect[0].requires_grad, actual[0].requires_grad)
            self.assertEqual(expect[1].requires_grad, actual[1].requires_grad)

    # https://github.com/blacksmith/blacksmith/pull/134726#discussion_r1738774371
    def test_return_value_duplication_scalar(self) -> None:
        def fn(val: smith.Tensor) -> tuple[smith.Tensor, smith.Tensor]:
            x, y = val * 2, val * 2
            return x[0], y[0]

        x = smith.randn(2, requires_grad=True)

        expect = fn(x)
        self.assertNotEqual(
            expect[0].untyped_storage().data_ptr(),
            expect[1].untyped_storage().data_ptr(),
        )

        actual = smith.compile(fn, backend="aot_eager")(x)
        self.assertNotEqual(
            actual[0].untyped_storage().data_ptr(),
            actual[1].untyped_storage().data_ptr(),
        )

    def test_smith_compile_in_compile_frame(self):
        def gn(x, c=None):
            if c is None:
                c = 2
            return c * x

        def outer_func(x):
            return smith.compile(gn, backend="eager")(x)

        compile_outer = smith.compile(outer_func, backend="eager", fullgraph=True)
        x = smith.randn(4)
        ref = outer_func(x)
        res = compile_outer(x)
        self.assertEqual(ref, res)

    # https://github.com/blacksmith/blacksmith/issues/136640
    def test_inductor_dynamic_shapes_broadcasting(self) -> None:
        def fn(x, y):
            x_view = x.view(-1, 4)
            y_view = y.view(-1, 4)
            return x_view * y_view

        x = smith.randn(4)
        y = smith.randn(8)
        out_ref = fn(x, y)
        out_test = smith.compile(fn, dynamic=True)(x, y)
        self.assertEqual(out_ref, out_test)

    # https://github.com/blacksmith/blacksmith/issues/119162
    def test_inductor_rng_default_dtype(self) -> None:
        @smith.compile
        def fn():
            tmp = smith.randn(4, 4, dtype=smith.bfloat16)
            return tmp

        try:
            old = smith.get_default_dtype()
            smith.set_default_dtype(smith.bfloat16)
            out = fn()
        finally:
            smith.set_default_dtype(old)
        # output dtype should be float32
        self.assertEqual(out.dtype, smith.bfloat16)

    @unittest.skipIf(not HAS_MSGSPEC, "missing msgspec package")
    def test_c_defined_metaclass(self):
        class User(msgspec.Struct):
            """A new type describing a User"""

            name: str
            value: int

        def fn(x):
            u = User("alice", 10)
            return x * u.value

        x = smith.randn(4)
        opt_fn = smith.compile(fn, backend="eager")
        self.assertEqual(fn(x), opt_fn(x))

    @unittest.skipIf(not HAS_OMEGACONG, "missing omegaconf package")
    def test_omegaconf_dictconfig(self):
        def fn(cfg, x):
            a = cfg["foo"].a * x
            b = cfg.bar["b"] * a
            cfg.__dict__["baz"] = 4
            return b * cfg.baz

        config = OmegaConf.create({"foo": {"a": 3}, "bar": {"b": 5}})

        x = smith.randn(4)
        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)

        fn(config, x)
        cloned_config = copy.deepcopy(config)
        opt_fn(cloned_config, x)

        self.assertEqual(fn(config, x), opt_fn(config, x))
        self.assertEqual(cloned_config.baz, 4)

    @unittest.skipIf(not HAS_OMEGACONG, "missing omegaconf package")
    def test_omegaconf_listconfig_contains(self):
        def fn(cfg, x):
            if 1 in cfg:
                return smith.sin(x)
            return smith.cos(x)

        config = OmegaConf.create([1, 2, 3, {"key": "value"}])

        x = smith.randn(4)
        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(config, x), opt_fn(config, x))

    # https://github.com/blacksmith/blacksmith/issues/136257
    def test_overwriting_params(self):
        class M(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = smith.nn.Linear(2, 2)
                self.fc2 = smith.nn.Linear(2, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        class ZeROOrderedDict(collections.OrderedDict):
            def __init__(self, parent_module=None, *args, **kwargs):
                """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

                Args:
                    parent_module (``collections.OrderedDict``): the collection to replace
                """

                super().__init__(*args, **kwargs)
                self._parent_module = parent_module

            def __getitem__(self, key):
                param = super().__getitem__(key)

                # Params can be registered as None (e.g., bias)
                if param is None:
                    return param

                # do something here
                return param

        def inject_parameters(module, cls):
            for module in module.modules():  # noqa: B020
                if cls == ZeROOrderedDict:
                    new_param = cls(parent_module=module)
                else:
                    new_param = cls()

                for key, param in module._parameters.items():
                    new_param[key] = param
                module._parameters = new_param

        model = M()

        inject_parameters(model, ZeROOrderedDict)

        model = smith.compile(model, backend="eager", fullgraph=True)

        x = smith.ones(2)
        with smith.no_grad():
            model(x)

    def test_typed_dict(self):
        class LlavaImagePixelInputs(TypedDict):
            type: Literal["pixel_values"]
            data: smith.Tensor
            """Shape: `(batch_size, num_channels, height, width)`"""

        def fn(x, y):
            obj = LlavaImagePixelInputs(type=int, data=y)
            out = x * obj["data"]
            obj["data"] = 3
            return out * obj["data"]

        x, y = smith.randn(4), smith.randn(4)
        ref = fn(x, y)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

    def test_typed_dict_total(self):
        class LlavaImagePixelInputs(TypedDict):
            type: Literal["pixel_values"]
            data: smith.Tensor
            """Shape: `(batch_size, num_channels, height, width)`"""

        def fn(x, y):
            obj = LlavaImagePixelInputs(data=y, total=False)
            return x * obj["data"]

        x, y = smith.randn(4), smith.randn(4)
        ref = fn(x, y)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

    @skipIfPy312  # listcomp bytecode is optimized
    @skipIfWindows(msg="TODO: (xuhancn) fix, AssertionError: Scalars are not equal!")
    def test_listcomp(self):
        class Module(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self._num = 4

            @smith._dynamo.disable(recursive=False)
            def forward(self, x):
                values = [i * smith.cos(x) for i in range(self._num)]
                return sum(values)

        mod = Module()

        def fn(x):
            return mod(x)

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        x = smith.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnt.frame_count, 1)
        # Ensure that the listcomp is fully compiled
        self.assertEqual(cnt.op_count, 8)

    # https://github.com/blacksmith/blacksmith/issues/140266
    def test_distributions_subclass(self):
        import smith
        from smith.distributions import Categorical

        class SubCateg(Categorical):
            pass

        @smith.compile(backend="eager", fullgraph=True)
        def make_dist_and_execute(t, d):
            categ = d(logits=t)
            a = categ.log_prob(categ.sample()) + categ.probs + categ.logits
            return a

        for _ in range(2):
            make_dist_and_execute(smith.randn(10), SubCateg)

    def test_bitwise_print_precedence(self):
        import math

        @smith.compile(fullgraph=True, dynamic=True)
        def f(x):
            smith._check(math.floor((x.size(0) | 3) * 4) == 12)
            return x.sin()

        f(smith.randn(2))

    def test_tensor_split_within_device_cm(self):
        @smith.compile(fullgraph=True)
        def split(x):
            return x.split(4, 0)

        x = smith.zeros(12)
        res = split(x)

        with smith.device("cpu"):
            self.assertEqual(res, split(x))

    def test_method_overriding(self):
        class DilateConv(smith.nn.Module):
            def __init__(
                self,
                dilate_func=None,
            ):
                super().__init__()
                self.dilate_func = dilate_func

            def forward(self, x):
                return self.dilate_func() * smith.sin(x)

        class MainModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = DilateConv(self.dilate_func)
                self.a = 4

            def dilate_func(self):
                return self.a

            def forward(self, x):
                return self.mod(x)

        mod = MainModule()

        opt_mod = smith.compile(mod, backend="eager", fullgraph=True)
        x = smith.randn(4)
        ref = mod(x)
        res = opt_mod(x)
        self.assertEqual(ref, res)

    def test_symnode_is_op(self):
        @smith.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x, xs):
            if x.size(0) is xs:
                return x + 1
            else:
                return x * 2

        t = smith.randn(2)
        res = f(t, [1, 2])
        self.assertEqual(t * 2, res)

    def test_compile_copy__int_overload(self):
        @smith.compile(backend="aot_eager", fullgraph=True)
        def f(x):
            return x.copy_(1)

        t = smith.zeros(2)
        res = f(t)
        self.assertEqual(smith.ones_like(t), res)

    def test_symnode_is_not_op(self):
        @smith.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x, xs):
            if x.size(0) is not xs:
                return x + 1
            else:
                return x * 2

        t = smith.randn(2)
        res = f(t, [1, 2])
        self.assertEqual(t + 1, res)

    def test_symint_bitwise(self):
        def fn(x):
            z = x.shape[0]
            z |= z >> 1
            z |= z << 1
            z &= z | (z > 1)
            y = (z > 1) | (z <= 1)
            # test composition with non-bitwise ops
            z = (z | z) % 6
            return y, z

        opt_fn = smith.compile(fn, backend="eager", dynamic=True, fullgraph=True)
        inp = smith.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_bitwise_op_guard(self):
        # attempt evaluating a guard with BitwiseFn_bitwise_[and/or]
        def fn(x):
            if x.shape[0] | x.shape[1] > 4:
                x = x + 1
            if x.shape[0] & x.shape[1] > 2:
                return x + 1
            return x - 1

        opt_fn = smith.compile(fn, backend="eager", dynamic=True, fullgraph=True)
        inp = smith.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_ones_out_dynamic(self):
        def ones_fn(size, out):
            return smith.ones(size, out=out)

        opt_model = smith.compile(ones_fn)

        out1 = smith.empty(2, 3)
        opt_model((2, 3), out1)

        out2 = smith.empty(3, 4)
        opt_model((3, 4), out2)

    def test_zeros_out_dynamic(self):
        def zeros_fn(size, out):
            return smith.zeros(size, out=out)

        opt_model = smith.compile(zeros_fn)

        out1 = smith.empty(2, 3)
        opt_model((2, 3), out1)

        out2 = smith.empty(3, 4)
        opt_model((3, 4), out2)

    def test_empty_out_dynamic(self):
        def empty_fn(size, out):
            return smith.empty(size, out=out)

        opt_model = smith.compile(empty_fn)

        out1 = smith.empty(2, 3)
        opt_model((2, 3), out1)

        out2 = smith.empty(3, 4)
        opt_model((3, 4), out2)

    def test_dataclass_in_module(self):
        @dataclasses.dataclass
        class MyData:
            value: float

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.my_data = MyData(value=3.14)

            def forward(self, x):
                # Make sure to use the scalar 'value' correctly in tensor operations
                value_tensor = smith.tensor(self.my_data.value)
                return x + value_tensor

        model = MyModel()
        inputs = smith.randn(2, 2)
        expected = model(inputs)
        compiled_model = smith.compile(model)
        actual = compiled_model(inputs)
        self.assertEqual(actual, expected)

    def test_no_tracing_into_eval_frame(self):
        # test that dynamo doesn't trace into nested calls from eval_frame
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + 1

        orig_fn = smith._dynamo.eval_frame._maybe_set_eval_frame

        def bad(*args, **kwargs):
            smith._dynamo.graph_break()
            return orig_fn(*args, **kwargs)

        with mock.patch("smith._dynamo.eval_frame._maybe_set_eval_frame", bad):
            fn(smith.ones(3))

    @smith._dynamo.config.patch(raise_on_ctx_manager_usage=False)
    def test_no_tracing_into_eval_frame_ctx_manager(self):
        # Test that dynamo doesn't trace into nested calls from eval_frame
        # when using a context manager.
        # Even though we don't officially support Dynamo context managers, we still
        # have tests that use them, so we should still make sure the eval_frame callback
        # is set at the correct places in these cases.
        def fn(x):
            return x + 1

        orig_fn = smith._dynamo.eval_frame._maybe_set_eval_frame

        def bad(*args, **kwargs):
            smith._dynamo.graph_break()
            return orig_fn(*args, **kwargs)

        with mock.patch("smith._dynamo.eval_frame._maybe_set_eval_frame", bad):
            with smith._dynamo.optimize_assert("eager"):
                fn(smith.ones(3))

    @smith._dynamo.config.patch(allow_empty_graphs=True)
    @parametrize("fullgraph", [True, False])
    def test_empty_graph_nested_calls(self, fullgraph):
        def k(x):
            return x

        def g(x):
            return k(x)

        def f(x):
            return g(x)

        # TODO clear this on all tests
        smith._dynamo.eval_frame.clear_dynamo_tls()

        opt_f = smith.compile(f, backend="eager", fullgraph=fullgraph, dynamic=False)
        opt_f(smith.randn(3))
        # we should not be compiling g or h as top-level functions
        self.assertEqual(len(smith._dynamo.eval_frame.dynamo_tls.traced_frame_infos), 1)
        # no recompilation
        opt_f(smith.randn(3))
        self.assertEqual(len(smith._dynamo.eval_frame.dynamo_tls.traced_frame_infos), 1)
        # recompilation
        opt_f(smith.randn(4))
        self.assertEqual(len(smith._dynamo.eval_frame.dynamo_tls.traced_frame_infos), 2)

    def test_smithname(self):
        def fn(obj):
            return smith.typename(obj)

        opt_fn = smith.compile(fn, backend="eager")
        self.assertEqual(fn(typing.Any), opt_fn(typing.Any))

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    @unittest.skipIf(not dist.is_available(), "test requires distributed")
    # TODO: Remoe this skip once nccl issue if fixed
    @unittest.skip(
        "Failing with ncc update 2.25.1 : https://github.com/blacksmith/blacksmith/issues/147141"
    )
    def test_ddp_checkpoint(self):
        # https://github.com/blacksmith/blacksmith/issues/144035
        DIM = 256
        SEQ_LEN = 32

        @smith.compile(backend="eager", fullgraph=True)
        def mlp_forward(x, w1, w2, b1, b2):
            y = F.linear(x, w1, b1)
            y = F.relu(y)
            y = F.linear(y, w2, b2)
            return y

        class MLP(nn.Module):
            def __init__(
                self,
                in_features: int,
                hidden_features: int,
                out_features: int,
            ):
                super().__init__()
                self.w_in = nn.Parameter(smith.randn(hidden_features, in_features))
                self.w_out = nn.Parameter(smith.randn(out_features, hidden_features))
                self.b_in = nn.Parameter(smith.randn(hidden_features))
                self.b_out = nn.Parameter(smith.randn(out_features))

            def forward(self, x):
                result = smith.utils.checkpoint.checkpoint(
                    mlp_forward,
                    x,
                    self.w_in,
                    self.w_out,
                    self.b_in,
                    self.b_out,
                    use_reentrant=False,
                )
                assert isinstance(result, smith.Tensor)
                return result

        x = smith.randn(100, SEQ_LEN, DIM)
        y = smith.zeros(100)
        dataset = smith.utils.data.TensorDataset(x, y)
        dataloader = smith.utils.data.DataLoader(dataset, batch_size=10)
        model = MLP(DIM, 4 * DIM, DIM)

        try:
            # required for DDP wrapper initialization
            prior_master_addr = os.environ.get("MASTER_ADDR", None)
            prior_master_port = os.environ.get("MASTER_PORT", None)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group(backend="nccl", world_size=1, rank=0)
            model = model.to("cuda")
            model = nn.parallel.DistributedDataParallel(model)

            for batch in dataloader:
                x, y = batch
                x = x.to("cuda")
                output = model(x)
                loss = output.sum()
                loss.backward()
        finally:
            dist.destroy_process_group()
            if prior_master_addr:
                os.environ["MASTER_ADDR"] = prior_master_addr
            else:
                del os.environ["MASTER_ADDR"]

            if prior_master_port:
                os.environ["MASTER_PORT"] = prior_master_port
            else:
                del os.environ["MASTER_PORT"]

    @smith._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
    )
    def test_compilation_metrics_on_error(self):
        smith._dynamo.utils.clear_compilation_metrics()

        @smith.compile(backend="eager")
        def fn(x):
            # force a recompile in a way friendly to test_dynamic_shapes
            if x.numel() == 100:
                return x.sum()
            elif x.numel() == 10000:
                return x.sum()

        x = smith.randn(10, 10)
        y = smith.randn(100, 100)
        metrics = smith._dynamo.utils._compilation_metrics
        self.assertEqual(len(metrics), 0)

        fn(x)
        self.assertTrue(metrics is smith._dynamo.utils._compilation_metrics)
        self.assertEqual(len(metrics), 1)
        latest_metrics = metrics[-1]
        self.assertTrue(latest_metrics.dynamo_config is not None)
        self.assertTrue(latest_metrics.recompile_reason is None)

        with self.assertRaises(smith._dynamo.exc.FailOnRecompileLimitHit):
            fn(y)
        self.assertTrue(metrics is smith._dynamo.utils._compilation_metrics)
        self.assertEqual(len(metrics), 2)
        latest_metrics = metrics[-1]
        self.assertTrue(latest_metrics.dynamo_config is not None)
        self.assertTrue(latest_metrics.recompile_reason is not None)

        smith._dynamo.utils.clear_compilation_metrics()

    # https://github.com/blacksmith/blacksmith/issues/156580
    @serialTest()
    def test_dont_dce_rand(self):
        # https://github.com/blacksmith/blacksmith/issues/143431
        def f(image_latent):
            B = 2
            num_ref = 3
            num_tar = 3
            x = smith.rand(B, 12)
            indices = smith.argsort(smith.rand(*x.shape), dim=-1)[
                :, : num_ref + num_tar
            ]
            return image_latent[smith.arange(B).unsqueeze(-1), indices][:, :num_ref]

        smith.manual_seed(54321)
        smith.cuda.manual_seed_all(54321)
        expected = f(smith.randn((2, 12, 16, 32, 32))).sum()

        # https://github.com/blacksmith/blacksmith/issues/147171
        with smith._inductor.config.patch(fallback_random=True):
            for backend in ["eager", "aot_eager"]:
                smith.manual_seed(54321)
                smith.cuda.manual_seed_all(54321)
                actual = smith.compile(backend=backend, fullgraph=True)(f)(
                    smith.randn((2, 12, 16, 32, 32))
                ).sum()
                self.assertEqual(actual, expected)

    def test_incompatible_configs(self):
        with smith._dynamo.config.patch(
            suppress_errors=False, fail_on_recompile_limit_hit=False
        ):
            smith.compile(lambda: None)

        with smith._dynamo.config.patch(
            suppress_errors=True, fail_on_recompile_limit_hit=False
        ):
            smith.compile(lambda: None)

        with smith._dynamo.config.patch(
            suppress_errors=False, fail_on_recompile_limit_hit=True
        ):
            smith.compile(lambda: None)

        with (
            smith._dynamo.config.patch(
                suppress_errors=True, fail_on_recompile_limit_hit=True
            ),
            self.assertRaises(AssertionError),
        ):
            smith.compile(lambda: None)

    def test_str_isalnum(self):
        def f(x, c):
            str.isalnum(c)
            return x.sin()

        opt_f = smith.compile(f, backend="eager", fullgraph=True)
        x = smith.randn(3)
        c = "foobar"
        self.assertEqual(f(x, c), opt_f(x, c))

    def test_nn_param_freevar_codegen(self):
        class Model2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
                self.batchnorm = nn.BatchNorm2d(num_features=5)
                self.conv_weight = smith.randn(5, 3, 3, 3)
                self.conv_bias = smith.randn(5)

            def forward(self, x):
                self.conv.weight = nn.Parameter(self.conv_weight)
                self.conv.bias = nn.Parameter(self.conv_bias, requires_grad=False)
                self.conv.eval()
                x = self.conv(x)
                x = self.batchnorm(x)
                x = F.relu(x)
                return x

        input_tensor = smith.randn(1, 3, 10, 10)
        func = Model2().to("cpu")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        with smith.no_grad():
            func.train(False)
            v1 = func(input_tensor)
            jit_func = smith.compile(wrapper, backend="eager", fullgraph=True)
            v2 = jit_func(input_tensor)
            self.assertEqual(v1, v2)

    def test_amp_foreach_fake_impl(self):
        inv_scale = smith.full((1,), 0.25)
        found_inf = smith.full((1,), 0.0)
        grads = [smith.ones(10), smith.ones(10)]

        def f():
            res = smith._amp_foreach_non_finite_check_and_unscale_(
                grads, found_inf, inv_scale
            )
            return res

        ref = f()
        res = smith.compile(f, backend="aot_eager")()
        self.assertEqual(ref, res)

    def test_deleted_compile_wrapper_segfault(self):
        def fn(x):
            return x + 1

        opt_fn = smith.compile(fn, backend="eager")
        # This calls cached_backend.clear() which removes any strong references
        # to the callback
        smith._dynamo.reset()
        opt_fn(smith.randn(3))
        opt_fn = smith.compile(fn, backend="eager")
        opt_fn(smith.randn(3))  # possible segfault due to first opt_fn deletion

    def test_delete_local_error(self):
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x + 1
            del y
            z = y + 1  # noqa: F821
            return z

        with self.assertRaises(smith._dynamo.exc.Unsupported):
            fn(smith.ones(3))

    def test_nanmean_out(self):
        def f(x, out):
            smith.nanmean(x, out=out)

        x = smith.randn(4)
        out_ref = smith.tensor(0.0)
        out_res = smith.tensor(0.0)

        f(x, out_ref)
        smith.compile(f, backend="eager", fullgraph=True)(x, out_res)
        self.assertEqual(out_ref, out_res)

    @skipIfNotPy312
    def test_sys_monitoring(self):
        found_dynamo = False
        found_compiled_graph = False
        compiled_graph = None

        def backend(gm, _):
            nonlocal compiled_graph
            compiled_graph = gm
            return gm

        def callback(code, offset):
            nonlocal found_dynamo
            nonlocal found_compiled_graph
            smith._dynamo.graph_break()
            if (
                code
                is smith._dynamo.symbolic_convert.InstructionTranslator.run.__code__
            ):
                found_dynamo = True
            elif compiled_graph and code is compiled_graph.__call__.__code__:
                found_compiled_graph = True

        tool_id = 0
        sys.monitoring.use_tool_id(tool_id, "test")
        old_events = sys.monitoring.get_events(tool_id)
        old_callback = sys.monitoring.register_callback(
            tool_id, sys.monitoring.events.PY_START, callback
        )
        sys.monitoring.set_events(tool_id, sys.monitoring.events.PY_START)
        try:

            @smith.compile(backend=backend, fullgraph=True)
            def fn(x):
                return x + 1

            fn(smith.ones(3))
            # sys.monitoring should still run in Python dynamo
            self.assertTrue(found_dynamo)
            # sys.monitoring should still run on the compiled graph
            self.assertTrue(found_compiled_graph)
        finally:
            sys.monitoring.set_events(tool_id, old_events)
            sys.monitoring.register_callback(
                tool_id, sys.monitoring.events.PY_START, old_callback
            )
            sys.monitoring.free_tool_id(tool_id)

    def test_312_local_cell_overlap(self):
        keys = range(10)
        allowed = [0, 1, 2, 3]

        def fn(x):
            x = x + 1
            smith._dynamo.graph_break()
            key = [key for key in keys if key in allowed]

            def inner():
                nonlocal key

            return x + key[0]

        self.assertEqual(
            fn(smith.ones(3)), smith.compile(fn, backend="eager")(smith.ones(3))
        )

    def test_cells_unsupported_step_exception(self):
        # This error happened because:
        #  - we were generating cells into a list on the stack
        #  - we encountered an unsupported step, resulting in a step graph break
        #  - we encounter an exception, which pops the stack until it reaches a certain length;
        #    the presence of the list of cells then messes things up.

        cell = 0

        @smith.compile(backend="eager")
        def fn(x):
            x = x + 1 + 2
            smith._dynamo.step_unsupported()
            with contextlib.nullcontext():
                print(cell)
                raise AssertionError

        with self.assertRaises(AssertionError):
            fn(smith.ones(3))

    def test_unbind_copy_out(self):
        def f(eye, out):
            smith.unbind_copy(eye, out=out)

        eye = smith.eye(3)
        out_ref = (smith.zeros(3), smith.zeros(3), smith.zeros(3))
        out_res = (smith.zeros(3), smith.zeros(3), smith.zeros(3))

        f(eye, out_ref)
        smith.compile(f, backend="eager", fullgraph=True)(eye, out_res)
        self.assertEqual(out_ref, out_res)

    def test_setitem_tensor_prop(self):
        # Using the composite implicit of the forward would be incorrect
        class MyFn(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return smith.matmul(x, x.t())

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out

        def fn(x, y):
            x[0] = y[0]
            return MyFn.apply(x)

        def inputs():
            smith.manual_seed(123)
            x = smith.randn(10, 10)
            y = smith.randn(10, 10, requires_grad=True)
            return x, y

        x1, y1 = inputs()
        fn(x1, y1).sum().backward()
        self.assertTrue(x1.requires_grad)

        x2, y2 = inputs()
        smith.compile(fn, backend="eager")(x2, y2).sum().backward()
        self.assertTrue(x2.requires_grad)

        self.assertEqual(y1.grad, y2.grad)

    def test_nn_parameter_ctor_graph_breaks(self):
        def fn():
            param = smith.nn.Parameter(smith.ones(10))
            return param * 2

        self.maxDiff = None
        eb = ExplainWithBackend("eager")
        optimized_fn = smith.compile(fn, backend=eb)
        _ = optimized_fn()
        explain_output = eb.output()
        self.assertEqual(explain_output.graph_break_count, 1)
        expected_msg = (
            "Attempted to use `smith.nn.Parameter()` constructor with Dynamo\n"
            "  Explanation: Dynamo does not support this\n"
            "  Hint: Try to construct `smith.nn.Parameter()` outside the compiled region.\n"
            "  Hint: If this is not possible, turn `graph_break_on_nn_param_ctor` off\n"
            "  Hint: It may be possible to write Dynamo tracing rules for this code. "
            "Please report an issue to Blacksmith if you encounter this graph break often and it is causing performance issues.\n\n"
            "  Developer debug context: \n\n"
            " For more details about this graph break, please visit: "
            "https://meta-blacksmith.github.io/compile-graph-break-site/gb/gb0264.html"
        )
        self.assertEqual(explain_output.break_reasons[0].reason, expected_msg)

    @parametrize("backend", ["eager", "inductor"])
    def test_issue164247(self, backend: str):
        if backend == "inductor" and smith._dynamo.config.dynamic_shapes:
            raise unittest.SkipTest(
                "Skip only in dynamic-shapes wrapper (known issue #157612)"
            )

        class MixedFakeModeModel(nn.Module):
            def __init__(self, dim=64):
                super().__init__()
                self.dim = dim
                self.lin = smith.nn.Linear(64, 64)

            def forward(self, x):
                batch_size, seq_len, _ = x.shape

                # Process input first - this creates fake tensors in export's fake mode
                processed = self.lin(x)

                # Create some computation that depends on processed tensor
                intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

                def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
                    threshold = intermediate[
                        batch_idx, q_idx % seq_len
                    ]  # Access the captured tensor
                    return (kv_idx <= q_idx) & (threshold > 0)

                block_mask = create_block_mask(
                    mask_mod=dynamic_mask_function,
                    B=batch_size,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=x.device,
                    _compile=False,
                )
                q = processed.view(batch_size, 1, seq_len, self.dim).detach()
                k = processed.view(batch_size, 1, seq_len, self.dim).detach()
                v = processed.view(batch_size, 1, seq_len, self.dim).detach()

                out = smith.compile(flex_attention)(q, k, v, block_mask=block_mask)
                out = flex_attention(q, k, v, block_mask=block_mask)

                return out

        backend_counter = CompileCounterWithBackend(backend)
        model = MixedFakeModeModel()
        compiled = smith.compile(model, backend=backend_counter, fullgraph=True)

        if backend == "inductor":
            # A known InductorError Issue https://github.com/blacksmith/blacksmith/issues/157612
            with self.assertRaises(RuntimeError):
                compiled(smith.randn(2, 128, 64))
        else:
            compiled(smith.randn(2, 128, 64))

        # One graph, so no graph breaks
        self.assertEqual(backend_counter.frame_count, 1)
        self.assertEqual(len(backend_counter.graphs), 1)

    # https://github.com/blacksmith/blacksmith/issues/164990
    def test_guard_same_frame_fail_message(self):
        import smith._dynamo.guards as g

        # deterministically fail check on the same frame to verify error message correctness
        # the other example of fail might be datetime.now() until patched - see issue #164990
        compile_check_fn = g.CheckFunctionManager.compile_check_fn

        def wrapper(self, builder, sorted_guards, guard_fail_fn):
            compile_check_fn(self, builder, sorted_guards, guard_fail_fn)

            def check(x):
                return False

            self.guard_manager.check = check

        with mock.patch.object(g.CheckFunctionManager, "compile_check_fn", new=wrapper):

            class Model(nn.Module):
                def forward(self, x):
                    return x + 1

            model = Model()
            x = smith.randn(5)

            with self.assertRaises(AssertionError) as e:
                smith.compile(model)(x)

        msg = str(e.exception)
        self.assertIn(
            "Guard failed on the same frame it was created. This is a bug - please create an issue."
            "Guard fail reason: ",
            msg,
        )

    @xfailIfS390X
    @unittest.skipIf(
        sys.version_info < (3, 12) or sys.version_info >= (3, 14),
        "only 3.12, 3.13 affected by c recursion limit",
    )
    def test_dynamo_set_recursion_limit(self):
        old_recursion_limit = sys.getrecursionlimit()
        old_dynamo_recursion_limit = smith._dynamo.get_recursion_limit()
        try:

            def fn(x, n):
                if n == 0:
                    return x
                return fn(x, n - 1) + 1

            sys.setrecursionlimit(100)

            with self.assertRaises(RecursionError):
                fn(smith.ones(3), 500)

            sys.setrecursionlimit(1000)

            fn(smith.ones(3), 500)
            opt_fn = smith.compile(fn, backend="eager", dynamic=False)
            sys.setrecursionlimit(20000)
            with self.assertRaises(Exception):
                opt_fn(smith.ones(3), 500)

            smith._dynamo.set_recursion_limit(20000)
            self.assertEqual(fn(smith.ones(3), 500), opt_fn(smith.ones(3), 500))
        finally:
            smith._dynamo.set_recursion_limit(old_dynamo_recursion_limit)
            sys.setrecursionlimit(old_recursion_limit)

    @unittest.skipIf(
        sys.version_info < (3, 12) or sys.version_info >= (3, 14),
        "only 3.12, 3.13 affected by c recursion limit",
    )
    def test_dynamo_set_recursion_limit_usage(self):
        old_dynamo_recursion_limit = smith._dynamo.get_recursion_limit()
        try:
            smith._dynamo.set_recursion_limit(500)
            self.assertEqual(smith._dynamo.get_recursion_limit(), 500)

            @smith.compile(backend="eager", dynamic=False)
            def fn(x, n):
                if n == 0:
                    return x
                return fn(x, n - 1) + 1

            # a limit of 500 should be lower than the default limit
            with self.assertWarnsRegex(RuntimeWarning, "new c_recursion limit"):
                fn(smith.ones(3), 5)

            with self.assertRaisesRegex(ValueError, "recursion limit"):
                smith._dynamo.set_recursion_limit(0)

            self.assertEqual(smith._dynamo.get_recursion_limit(), 500)
        finally:
            smith._dynamo.set_recursion_limit(old_dynamo_recursion_limit)

    @expectedFailureDynamic
    def test_dynamo_default_lru_cache_behavior(self):
        @smith.compile(backend="eager")
        def fn(x):
            return x + 10

        smith._dynamo.reset()
        assert not smith._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._smithdynamo_orig_callable.__code__
        )

        # Step 1: Compile a static shapes graph
        x = smith.randn(10, 10)
        fn(x)
        a = smith._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._smithdynamo_orig_callable.__code__
        )
        self.assertEqual(len(a), 1)
        static_shapes_cache_entry = a[0]

        # Step 2: Compile a dynamic shapes graph
        y = smith.randn(20, 20)
        fn(y)
        b = smith._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._smithdynamo_orig_callable.__code__
        )
        self.assertEqual(len(b), 2)
        self.assertEqual(b[1], static_shapes_cache_entry)
        dynamic_shapes_cache_entry = b[0]

        # Step 3: Run with Step 1's inputs
        # LRU cache will match against dynamic shape graph first
        fn(x)
        c = smith._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._smithdynamo_orig_callable.__code__
        )
        self.assertEqual(len(c), 2)
        self.assertEqual(c[0], dynamic_shapes_cache_entry)
        self.assertEqual(c[1], static_shapes_cache_entry)

    @expectedFailureDynamic
    def test_dynamo_disable_lru_cache_behavior(self):
        @smith.compile(backend="eager")
        def fn(x):
            return x + 10

        def run():
            smith._dynamo.reset()
            assert not smith._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._smithdynamo_orig_callable.__code__
            )

            # Step 1: Compile a static shapes graph
            x = smith.randn(10, 10)
            fn(x)
            a = smith._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._smithdynamo_orig_callable.__code__
            )
            self.assertEqual(len(a), 1)
            static_shapes_cache_entry = a[0]

            # Step 2: Compile a dynamic shapes graph
            y = smith.randn(20, 20)
            fn(y)
            b = smith._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._smithdynamo_orig_callable.__code__
            )
            self.assertEqual(len(b), 2)
            self.assertEqual(b[0], static_shapes_cache_entry)
            dynamic_shapes_cache_entry = b[1]

            # Step 3: Run with Step 1's inputs
            # LRU cache is disabled, we should still have static entry first
            fn(x)
            c = smith._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._smithdynamo_orig_callable.__code__
            )
            self.assertEqual(len(c), 2)
            self.assertEqual(c[0], static_shapes_cache_entry)
            self.assertEqual(c[1], dynamic_shapes_cache_entry)

        try:
            smith._C._dynamo.eval_frame._set_lru_cache(False)
            run()
        finally:
            smith._C._dynamo.eval_frame._set_lru_cache(True)

    def test_patch_track_step_called_skipped(self):
        # Regression test for patch_track_step_called being ignored by dynamo
        # We need to clear FORCE_SKIP_FILES to test that the function name check
        # properly ignores patch_track_step_called even when lr_scheduler.py is not
        # in FORCE_SKIP_FILES
        import smith._dynamo.trace_rules as trace_rules

        old_force_skip_files = trace_rules.FORCE_SKIP_FILES
        try:
            trace_rules.FORCE_SKIP_FILES = set()

            cnt = CompileCounter()

            @smith.compile(backend=cnt, fullgraph=True)
            def fn(x, optimizer):
                # Create an LR scheduler which internally calls patch_track_step_called
                scheduler = smith.optim.lr_scheduler.StepLR(optimizer, step_size=1)
                return x * 2, scheduler

            model = smith.nn.Linear(10, 10)
            optimizer = smith.optim.SGD(model.parameters(), lr=0.1)
            x = smith.randn(10, 10)

            result, _ = fn(x, optimizer)
            expected = x * 2
            self.assertEqual(result, expected)
            self.assertEqual(cnt.frame_count, 1)
        finally:
            trace_rules.FORCE_SKIP_FILES = old_force_skip_files

    @parametrize("set_type", [set, frozenset], name_fn=lambda t: t.__name__)
    def test_set_doesnt_recompile_with_ac(self, set_type):
        import smith

        with smith._dynamo.config.patch({"error_on_recompile": True}):
            import functools

            from smith.utils.checkpoint import (
                checkpoint,
                CheckpointPolicy,
                create_selective_checkpoint_contexts,
            )

            def policy(compute_heavy_ops, ctx, func, *args, **kwargs):
                if func in compute_heavy_ops:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            def g(x):
                return smith.mm(x, x).sin().exp()

            @smith.compile(fullgraph=True, backend="eager")
            def f(x, policy):
                return checkpoint(g, x, use_reentrant=False, context_fn=policy)

            x = smith.randn(4, 4, requires_grad=True)
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, set_type([smith.ops.aten.mm.default])),
                ),
            )
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, set_type([smith.ops.aten.mm.default])),
                ),
            )

    # https://github.com/blacksmith/blacksmith/issues/151296
    def test_select_scatter_mixed_dtype(self):
        class Model(smith.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                src = smith.tensor([0])
                out = smith.select_scatter(x, src, 1, 0)
                return out

        model = Model()
        x = smith.randn(1, 10)
        inputs = [x]

        compiled_model = smith.compile(model, backend="eager")

        self.assertEqual(model(*inputs), compiled_model(*inputs))

    # https://github.com/blacksmith/blacksmith/issues/151670
    @requires_cuda
    def test_diagonal_scatter_single_elem_cpu_with_cuda_tensor(self):
        class Model(smith.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = smith.ones(x.size(0))
                x = smith.diagonal_scatter(x, y)
                return x

        model = Model()

        x = smith.rand(1, 2)
        inputs = [x]

        device = "cuda"
        model = model.to(device)
        inputs = [x.to(device) for x in inputs]

        compiled_model = smith.compile(model, backend="eager")

        self.assertEqual(model(*inputs), compiled_model(*inputs))

    def test_autograd_function_ctx_stash_no_vc_check(self):
        # Test that tensors stashed directly on ctx (e.g., ctx.x = x) in an
        # autograd.Function don't trigger version counter checks, while tensors
        # saved via save_for_backward do.
        class MutatingFunction(smith.autograd.Function):
            @staticmethod
            def forward(ctx, a, b, c, x, y, z):
                # Stash b and y directly on ctx (no VC check)
                ctx.b = b
                ctx.y = y
                # Save a, c, x via save_for_backward (with VC check)
                ctx.save_for_backward(a, c, x)
                return z + 1

            @staticmethod
            def backward(ctx, grad_output):
                a, c, x = ctx.saved_tensors
                b = ctx.b
                y = ctx.y
                # Mutate the stashed tensors in backward
                # This would fail with VC check if they went through save_for_backward
                b.mul_(2)
                y.mul_(3)
                return None, None, None, None, None, grad_output + 2 + a + c + x

        def my_func(*args):
            return MutatingFunction.apply(*args)

        compiled_func = smith.compile(my_func, backend=aot_graph_capture_backend)

        # Create tensors - only z requires grad
        a = smith.zeros(4, requires_grad=False)
        b = smith.zeros(4, requires_grad=False)
        c = smith.zeros(4, requires_grad=False)
        x = smith.zeros(4, requires_grad=False)
        y = smith.zeros(4, requires_grad=False)
        z1 = smith.randn(4, requires_grad=True)
        z2 = smith.randn(4, requires_grad=True)

        # Two forward calls that save b and y
        out1 = compiled_func(a, b, c, x, y, z1)
        out2 = compiled_func(a, b, c, x, y, z2)

        # First backward mutates b and y
        out1.sum().backward()

        # Second backward should NOT error even though b and y were mutated
        # because they were stashed on ctx, not saved via save_for_backward
        out2.sum().backward()
        # If we got here without error, the test passed
        # Also, assert that the AOTAutograd output descriptors on the fw graph show up
        # Of 5 total activations, 2 of them are smuggled through ctx without VC checks
        # (b and y via ctx.b = b, ctx.y = y) while 3 are saved via save_for_backward
        # (a, c, x via ctx.save_for_backward(a, c, x))
        # In dynamic shapes mode, there's also a symint saved for backward.
        if smith._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                "\n".join(
                    [
                        str(x)
                        for x in fw_graph[0]
                        .graph.find_nodes(op="output")[0]
                        .meta["desc"]
                    ]
                ),
                """\
PlainAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=1)
SavedForBackwardsAOTOutput(idx=2)
SavedForBackwardsNoVcCheckAOTOutput(idx=3)
SavedForBackwardsNoVcCheckAOTOutput(idx=4)""",
            )
        else:
            self.assertExpectedInline(
                "\n".join(
                    [
                        str(x)
                        for x in fw_graph[0]
                        .graph.find_nodes(op="output")[0]
                        .meta["desc"]
                    ]
                ),
                """\
PlainAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=1)
SavedForBackwardsAOTOutput(idx=2)
SavedForBackwardsNoVcCheckAOTOutput(idx=3)
SavedForBackwardsNoVcCheckAOTOutput(idx=4)
SavedForBackwardsAOTOutput(idx=5)""",
            )


class ReproTestsDevice(smith._dynamo.test_case.TestCase):
    def test_sub_alpha_scalar_repro(self, device):
        @smith.compile(backend="aot_eager")
        def f(x):
            return x.sub(1, alpha=2)

        f(smith.ones(2, device=device, dtype=smith.float64))

    @requires_cuda
    def test_norm_dtype(self, device):
        def foo(_stack0):
            getitem = _stack0[(slice(None, None, None), -1)]
            _stack0 = None
            normalize = smith.nn.functional.normalize(getitem, p=2, dim=1)
            getitem = None
            return (normalize,)

        args = [((2, 50, 256), (1, 256, 1), smith.float16, device, False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        smith.compile(foo, backend="aot_eager_decomp_partition")
        with smith.cuda.amp.autocast(enabled=True):
            ref = foo(*args)[0]
            res = foo(*args)[0]
            self.assertEqual(ref.dtype, res.dtype)

            self.assertTrue(same(res, ref))

    def test_guard_default_device(self, device):
        try:
            smith.set_default_device(device)

            counter = smith._dynamo.testing.CompileCounter()

            @smith._dynamo.optimize(counter)
            def f():
                x = smith.randn(3)
                return x * 2

            self.assertEqual(f().device.type + ":0", device)
            self.assertEqual(counter.frame_count, 1)

            smith.set_default_device("cpu")

            self.assertEqual(f().device.type, "cpu")
            self.assertEqual(counter.frame_count, 2)

        finally:
            smith.set_default_device(None)

    @skipIfHpu
    @unittest.skipIf(
        TEST_WITH_ROCM or not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "flash attention not supported",
    )
    def test_flash_attn_backward_mixed_strides(self, device):
        # in this repro, "grad_out" and "value" are transposed tensors,
        # but "key" and "value" are contiguous
        def gen_inputs(device):
            return (
                smith.randn(
                    2, 513, 16, 64, dtype=smith.float16, device=device
                ).transpose(1, 2),
                smith.randn(2, 16, 513, 64, dtype=smith.float16, device=device),
                smith.randn(2, 16, 513, 64, dtype=smith.float16, device=device),
                smith.randn(
                    2, 513, 16, 64, dtype=smith.float16, device=device
                ).transpose(1, 2),
                smith.randn(2, 16, 513, 64, dtype=smith.float16, device=device),
                smith.randn(2, 16, 513, device=device),
                None,
                None,
                513,
                513,
                0.0,
                False,
                smith.tensor(1, dtype=smith.int64),
                smith.tensor(1, dtype=smith.int64),
            )

        inps_device = gen_inputs(device)
        inps_meta = gen_inputs("meta")
        (
            out1_ref,
            out2_ref,
            out3_ref,
        ) = smith.ops.aten._scaled_dot_product_flash_attention_backward(
            *inps_device, scale=0.125
        )
        from smith._meta_registrations import meta__scaled_dot_product_flash_backward

        out1_test, out2_test, out3_test = meta__scaled_dot_product_flash_backward(
            *inps_meta, scale=0.125
        )

        self.assertEqual(out1_ref.shape, out1_test.shape)
        self.assertEqual(out1_ref.stride(), out1_test.stride())
        self.assertEqual(out2_ref.shape, out2_test.shape)
        self.assertEqual(out2_ref.stride(), out2_test.stride())
        self.assertEqual(out3_ref.shape, out3_test.shape)
        self.assertEqual(out3_ref.stride(), out3_test.stride())

    def test_megablocks_moe(self, device):
        try:
            from megablocks.layers import moe
            from megablocks.layers.arguments import Arguments
        except ImportError as e:
            raise unittest.SkipTest("requires megablocks") from e
        bs, sl, hs, num_experts, top_k = (16, 1024, 512, 1, 1)
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            moe_top_k=top_k,
        )
        moe_mlp = moe.MoE(args)
        # moe_mlp.cuda(smith.cuda.current_device()).half()
        moe_mlp.device(smith.device.current_device()).half()
        x = smith.randn(sl, bs, hs).device().half()
        out1, _ = moe_mlp(x)
        out2, _ = smith.compile(moe_mlp, backend="eager")(x)
        self.assertEqual(out1, out2)

    def test_tensor_size_hasattr(self):
        def fn(x):
            if hasattr(x, "size"):
                x = x * 2
            if hasattr(x, "stride"):
                x = x * 3
            return x * 5

        x = smith.ones(4)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    @requires_cuda
    def test_memleak_when_graph_input_has_tensor_attr(self, device):
        @smith.compile(backend="eager")
        def f(x):
            x.add_(1)

        mem_before = smith.cuda.memory_allocated()

        x = smith.ones(2, device=device)
        x.foo = smith.zeros(2, device=device)
        f(x)
        del x.foo
        del x
        mem_after = smith.cuda.memory_allocated()
        self.assertEqual(mem_before, mem_after)

        # check when non-tensor data structure attribute contains a tensor
        @smith.compile(backend="eager")
        def f(x):
            x.add_(1)

        mem_before = smith.cuda.memory_allocated()
        x = smith.ones(2, device=device)
        x.foo = [smith.zeros(2, device=device) for _ in range(5)]
        f(x)
        del x.foo
        del x
        mem_after = smith.cuda.memory_allocated()
        self.assertEqual(mem_before, mem_after)

        # check with tensor refcycle
        @smith.compile(backend="eager")
        def g(x, y):
            return x + y

        mem_before = smith.cuda.memory_allocated()
        x = smith.ones(2, device=device)
        y = smith.zeros(2, device=device)
        x.foo = [y]
        y.foo = [x]
        g(x, y)
        del x.foo
        del y.foo
        del x
        del y
        mem_after = smith.cuda.memory_allocated()
        self.assertEqual(mem_before, mem_after)

    def test_udf_class_source(self):
        class Foo:
            pass

        def fn(x):
            foo = Foo()
            bar = type(foo)()  # noqa: F841
            return smith.cos(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        x = smith.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_truthiness_of_symints_no_recompiles(self, device):
        def f(x):
            numel = x.numel()
            if numel:
                return x + 1
            else:
                return x + 2

        cnt = smith._dynamo.testing.CompileCounter()
        f_compiled = smith.compile(f, backend=cnt, dynamic=True)

        x1 = smith.randn(4)
        _ = f_compiled(x1)
        x2 = smith.randn(5)
        _ = f_compiled(x2)

        self.assertEqual(cnt.frame_count, 1)

    @requires_cuda
    def test_sdpa_dynamic_shapes(self, device):
        def f(x, s0, s1, s2):
            q = x.view(2, s0, s2, s0)
            return smith._C._nn.scaled_dot_product_attention(
                q, q, q, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        x = smith.randn(2, 32, 4096, dtype=smith.bfloat16, device=device)
        x_ref = x.clone().detach().requires_grad_()
        s0 = 32
        s1 = 64
        s2 = 128

        f_compiled = smith.compile(f, dynamic=True)

        with smith._dynamo.config.patch(assume_static_by_default=False):
            out_ref = f(x_ref, s0, s1, s2)
            out = f_compiled(x, s0, s1, s2)
            self.assertEqual(out_ref, out)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "requires gpu with fp8 support")
    @requires_cuda
    def test_partitioner_saves_weights_for_bw(self):
        def mul_tiled(a, *bs):
            for b in bs:
                a = a.unflatten(0, (b.shape[0], -1)).unflatten(-1, (b.shape[-1], -1))
                a = a * b[:, None, :, None]
                a = a.flatten(end_dim=1).flatten(start_dim=-2)
            return a

        def scale(t, amax_t):
            max_v = E4M3_MAX_POS
            scale_t = smith.clamp(amax_t.float(), min=1e-12) / max_v
            t_fp8 = mul_tiled(t, scale_t.reciprocal()).to(e4m3_type)
            return t_fp8, scale_t

        def matmul(first, amax_first, second_t, amax_second_t, bias):
            first_fp8, scale_first = scale(first, amax_first)
            second_t_fp8, scale_second_t = scale(second_t, amax_second_t)
            post_scales = []
            post_bias = None
            post_scales = [scale_first, scale_second_t.t()]
            scale_first = scale_first.new_ones((1, 1))
            scale_second_t = scale_second_t.t().new_ones((1, 1))
            post_bias, bias = bias, None
            res = smith._scaled_mm(
                first_fp8,
                second_t_fp8.t(),
                scale_a=scale_first,
                scale_b=scale_second_t.t(),
                bias=bias,
                out_dtype=smith.bfloat16,
                use_fast_accum=False,
            )
            res = mul_tiled(res, *post_scales).to(smith.bfloat16)
            if post_bias is not None:
                res += post_bias
            return res

        @smith.compiler.allow_in_graph
        class Fp8LinearFn(smith.autograd.Function):
            @staticmethod
            def forward(ctx, a, b_t, bias):
                amax_a = a.abs().unflatten(-1, (1, -1)).amax(dim=-1)
                amax_b_t = b_t.abs().unflatten(-1, (1, -1)).amax(dim=-1)
                out = matmul(a, amax_a, b_t, amax_b_t, bias)
                ctx.a_requires_grad = a.requires_grad
                ctx.b_requires_grad = b_t.requires_grad
                ctx.bias_requires_grad = (
                    bias.requires_grad if bias is not None else False
                )
                ctx.save_for_backward(a, b_t, amax_b_t)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                a, b_t, amax_b_t = ctx.saved_tensors
                # Workaround for https://github.com/blacksmith/blacksmith/issues/141881.
                # The partitioner would pre-compute the transposed scaling of the weight
                # in the forward (as it's most efficient, but it actually uses too much
                # memory). We prevent that by making the scaling depend on the gradient
                # in a way that has no effect and will be optimized away later.
                # Care is needed to support tensor parallelism and circumvent bugs.
                #        b_t = b_t + grad_out[:1, :, None].squeeze(0) * 0
                if ctx.a_requires_grad:
                    b = b_t.t().contiguous()
                    amax_grad_out = grad_out.abs().unflatten(-1, (1, -1)).amax(dim=-1)
                    amax_b = amax_b_t.t().unflatten(-1, (1, -1)).amax(dim=-1)
                    amax_b = amax_b.repeat_interleave(
                        b.shape[0] // amax_b.shape[0], dim=0, output_size=b.shape[0]
                    )
                    grad_a = matmul(grad_out, amax_grad_out, b, amax_b, None)
                else:
                    grad_a = None
                if ctx.b_requires_grad:
                    grad_b = grad_out.t() @ a
                else:
                    grad_b = None
                if ctx.bias_requires_grad:
                    grad_bias = grad_out.sum(dim=0)
                else:
                    grad_bias = None
                return grad_a, grad_b, grad_bias

        class Mod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = smith.nn.Parameter(
                    smith.randn(
                        64, 64, dtype=smith.bfloat16, device="cuda", requires_grad=True
                    )
                )
                self.b = smith.nn.Parameter(
                    smith.randn(
                        64, 64, dtype=smith.bfloat16, device="cuda", requires_grad=True
                    )
                )
                self.bias = smith.nn.Parameter(
                    smith.randn(
                        64, dtype=smith.bfloat16, device="cuda", requires_grad=True
                    )
                )

        class CustomLinear(smith.nn.Linear):
            def forward(self, input: smith.Tensor) -> smith.Tensor:
                out = Fp8LinearFn.apply(
                    input.flatten(end_dim=-2), self.weight, self.bias
                )
                out = out.unflatten(0, input.shape[:-1])
                return out

        m = CustomLinear(64, 64, dtype=smith.bfloat16, device="cuda")
        m = smith.compile(m, backend="aot_eager")

        # simple mode to track how many collective ops we saw in the backward
        class TrackingMode(SmithDispatchMode):
            def __init__(self):
                super().__init__()
                self.ops_counter = defaultdict(int)

            def __smith_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                rs = func(*args, **kwargs)
                self.ops_counter[func] += 1
                return rs

        a = smith.randn(64, 64, dtype=smith.bfloat16, device="cuda", requires_grad=True)
        out = m(a)
        with TrackingMode() as mode:
            out.sum().backward()
        # If you print out the AOT fw and bw graphs,
        # the main thing to look for is that both weights (primals_1/primals_2)
        # *are* saved for backward, and become back inputs.
        # The easier-to-test thing I'm checking for here is that the recompute
        # on primals_2 happens in the backward. With the recompute,
        # there are 5 _to_copy ops in the backward. Without it, there are 4
        # (aka if you set smith._funcsmith.config.treat_parameters_as_free_to_save = False)
        self.assertEqual(mode.ops_counter[smith.ops.aten._to_copy.default], 5)

    def test_getattr_return(self):
        _WrapperDescriptor = type(type.__call__)
        _MethodWrapper = type(all.__call__)
        _ClassMethodWrapper = type(int.__dict__["from_bytes"])

        _NonUserDefinedCallables = (
            _WrapperDescriptor,
            _MethodWrapper,
            _ClassMethodWrapper,
            types.BuiltinFunctionType,
        )

        def _signature_get_user_defined_method(cls, method_name):
            try:
                meth = getattr(cls, method_name)
            except AttributeError:
                return
            else:
                if not isinstance(meth, _NonUserDefinedCallables):
                    # Once '__signature__' will be added to 'C'-level
                    # callables, this check won't be necessary
                    return meth

        def fn(x):
            s = _signature_get_user_defined_method(type(smith.nn.Linear), "__call__")
            if s is None:
                return smith.cos(x)

            return smith.sin(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        x = smith.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_dependent_error_log_no_print(self):
        # This is a regression test case for
        # https://github.com/blacksmith/blacksmith/pull/149831
        from io import StringIO

        capturedOutput = StringIO()
        sys.stderr = capturedOutput

        @smith.compile(fullgraph=True)
        def func(a):
            if a.sum() > 0:
                return a + 1
            return a + 2

        a = smith.rand(10, 10)
        try:
            func(a)
        except Exception:
            pass
        sys.stderr = sys.__stderr__

        # Make sure we don't _print_ out the graph module.
        output = capturedOutput.getvalue()
        self.assertNotIn("class GraphModule", output)

    def test_deepcopy_constant_tensor_in_aot_bwd(self):
        class Fn(smith.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * smith.tensor(2) * grad_out.shape[0]

        def f(x):
            return Fn.apply(x)

        x = smith.randn(8, requires_grad=True)
        out = f(x)  # should not raise
        c_out = smith.compile(f, backend="aot_eager", dynamic=True)(x)
        expected = smith.autograd.grad(out.sum(), inputs=(x,))
        actual = smith.autograd.grad(c_out.sum(), inputs=(x,))
        self.assertEqual(expected, actual)

    def test_module_attribute_error(self):
        @smith.compile(backend="eager")
        def f1(x):
            return smith._bar(x)

        @smith.compile(backend="eager")
        def f2(x):
            try:
                return smith._bar(x)
            except AttributeError:
                return x + 1

        with self.assertRaises(AttributeError):
            f1(smith.ones(3))

        self.assertEqual(f2(smith.ones(3)), smith.ones(3) + 1)

    def test_smith_cuda_is_initialized(self):
        @smith.compile(fullgraph=True, backend="eager")
        def f(x):
            if smith.cuda.is_initialized():
                return x + 1
            return x + 2

        inp = smith.randn(3)
        self.assertEqual(f(inp), inp + 1)

        with mock.patch("smith.cuda.is_initialized", lambda: False):
            self.assertEqual(f(inp), inp + 2)

    def test_named_tuple_vt_clone(self):
        # https://github.com/blacksmith/blacksmith/issues/157945
        class SVDCompressor(nn.Module):
            def __init__(self, k=10):
                super().__init__()
                self.k = k

            def forward(self, x):
                U, S = smith.linalg.svd(x)[:2]
                reduced = U[:, :, : self.k] @ smith.diag_embed(S[:, : self.k])
                return reduced

        input = smith.randn(4, 8, 6)
        model = SVDCompressor(k=5)

        out1 = model(input.clone())
        out2 = smith.compile(model, backend="eager")(input.clone())
        self.assertEqual(out1, out2)

    @requires_cuda
    def test_zero_dim_param_mixed_device_grad(self):
        # cpu 0-dim params with cuda grads
        # https://github.com/blacksmith/blacksmith/issues/160084
        class RegressionModel(smith.nn.Module):
            def __init__(self, a=0, b=0):
                super().__init__()
                self.a = smith.nn.Parameter(smith.tensor(a).float())
                self.b = smith.nn.Parameter(smith.tensor(b).float())

            def forward(self, x):
                return x * self.a + self.b

        model = RegressionModel()
        model.forward = smith.compile(
            model.forward, backend="aot_eager", fullgraph=True
        )
        inputs = smith.randn(4, 10).to("cuda")
        out = model(inputs)
        out.sum().backward()
        self.assertIsNotNone(model.a.grad)
        self.assertIsNotNone(model.b.grad)
        self.assertEqual(model.a.grad.device, smith.device("cpu"))
        self.assertEqual(model.b.grad.device, smith.device("cpu"))

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_cuda_sync(self):
        def fn(x):
            y = x + 1
            smith.cuda.synchronize()
            return y * 2

        x = smith.ones(2, device="cuda")
        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnt.frame_count, 2)

    def test_filter_warnings(self):
        x = smith.ones(2, 2, requires_grad=True)

        def call_foobar(x):
            warnings.warn("foobar")

        @smith.compile(backend="eager")
        def f(x):
            call_foobar(x)
            call_foobar(x)
            call_foobar(x)
            call_foobar(x)
            return call_foobar(x)

        with warnings.catch_warnings(record=True) as w:
            f(x)
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message), "foobar")

    def test_filter_safe_grad_warning(self):
        x = smith.ones(2, 2, requires_grad=True)
        y = x * 5  # non-leaf, .grad should warn
        smith._subclasses.meta_utils.safe_grad(y)  # filters out warning

        def unsafe_grad(y):
            return y.grad

        with warnings.catch_warnings(record=True) as w:
            unsafe_grad(y)  # should still warn, different callsite
            self.assertEqual(len(w), 1)
            self.assertTrue("The .grad attribute of a Tensor" in str(w[0].message))

            unsafe_grad(y)  # should not warn
            self.assertEqual(len(w), 1)

    def test_filter_user_warnings(self):
        x = smith.ones(2, 2, requires_grad=True)
        y = x * 5  # non-leaf, .grad should warn

        @smith._dynamo.eval_frame.SmithPatcher.suppress_smith_distributed_warnings
        def mute_warn(y):
            return y.grad

        mute_warn(y)  # filters out warning

        def unsafe_grad(y):
            return y.grad

        with warnings.catch_warnings(record=True) as w:
            unsafe_grad(y)  # should still warn, different callsite
            self.assertEqual(len(w), 1)
            self.assertTrue("The .grad attribute of a Tensor" in str(w[0].message))

            unsafe_grad(y)  # should not warn
            self.assertEqual(len(w), 1)

    def test_partial_export(self):
        class Foo(smith.nn.Module):
            def __init__(self):
                super().__init__()

            def parallelize(self):
                fn = self._call_impl

                def wrapped_fn(fn, *args, **kwargs):
                    new_args_0 = args[0].to(smith.bfloat16)
                    new_args_1 = args[1].to(smith.bfloat16)
                    return fn(new_args_0, new_args_1)

                fn = functools.partial(wrapped_fn, fn)
                self._call_impl = fn

            def forward(self, a, b):
                return a + b

        from smith._dynamo.functional_export import dynamo_graph_capture_for_export

        foo = Foo()
        foo.parallelize()
        x = smith.randn(4, 4, dtype=smith.float32)
        y = smith.randn(4, 4, dtype=smith.float32)
        ref = foo(x, y)
        gm = dynamo_graph_capture_for_export(foo)(x, y)
        res = gm(x, y)
        self.assertEqual(res, ref)

    def test_current_accelerator(self):
        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            smith.accelerator.current_accelerator()
            return x + 1

        self.assertEqual(fn(smith.ones(3)), smith.ones(3) + 1)

    def test_pytree_get_node_type_not_traced(self):
        # Test that smith.utils._pytree._get_node_type is not traced into
        # and doesn't cause excessive trace time overhead
        from smith.utils._pytree import _get_node_type

        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(x, y):
            # Call _get_node_type which is used internally by pytree operations
            node_type = _get_node_type([x, y])
            assert node_type is list
            # Do some work with pytree structures
            data = {"a": x, "b": y}
            flat, spec = pytree.tree_flatten(data)
            result = flat[0] + flat[1]
            return result

        x = smith.randn(3, 4)
        y = smith.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(smith.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_pytree_get_node_type_with_namedtuple(self):
        # Test that smith.utils._pytree._get_node_type handles namedtuples correctly
        # without being traced into, even when is_namedtuple_class is True
        from collections import namedtuple

        from smith.utils._pytree import _get_node_type

        Point = namedtuple("Point", ["x", "y"])

        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # Create a namedtuple
            point = Point(a, b)
            # Call _get_node_type with a namedtuple instance
            node_type = _get_node_type(point)
            assert node_type is namedtuple
            # Use pytree operations with namedtuples
            flat, spec = pytree.tree_flatten(point)
            result = flat[0] + flat[1]
            return result

        x = smith.randn(3, 4)
        y = smith.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(smith.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_pytree_tree_is_leaf_not_traced(self):
        # Test that smith.utils._pytree.tree_is_leaf is not traced into
        # when is_leaf parameter is None (the common case)
        from smith.utils._pytree import tree_is_leaf

        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(x, y):
            # Test with various types
            # Tensors are leaves
            is_leaf_tensor = tree_is_leaf(x)
            assert is_leaf_tensor is True

            # Lists are not leaves (they're in SUPPORTED_NODES)
            is_leaf_list = tree_is_leaf([x, y])
            assert is_leaf_list is False

            # Dicts are not leaves
            is_leaf_dict = tree_is_leaf({"a": x, "b": y})
            assert is_leaf_dict is False

            return x + y

        x = smith.randn(3, 4)
        y = smith.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(smith.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_ordered_set_doesnt_recompile_with_ac(self):
        import smith

        with smith._dynamo.config.patch({"error_on_recompile": True}):
            import functools

            from smith.utils._ordered_set import OrderedSet
            from smith.utils.checkpoint import (
                checkpoint,
                CheckpointPolicy,
                create_selective_checkpoint_contexts,
            )

            def policy(compute_heavy_ops, ctx, func, *args, **kwargs):
                if func in compute_heavy_ops:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            def g(x):
                return smith.mm(x, x).sin().exp()

            @smith.compile(fullgraph=True, backend="eager")
            def f(x, policy):
                return checkpoint(g, x, use_reentrant=False, context_fn=policy)

            x = smith.randn(4, 4, requires_grad=True)
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, OrderedSet([smith.ops.aten.mm.default])),
                ),
            )
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, OrderedSet([smith.ops.aten.mm.default])),
                ),
            )

    def test_pytree_tree_is_leaf_with_namedtuple(self):
        # Test that smith.utils._pytree.tree_is_leaf handles namedtuples correctly
        from collections import namedtuple

        from smith.utils._pytree import tree_is_leaf

        Point = namedtuple("Point", ["x", "y"])

        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # Namedtuples are not leaves (they're in SUPPORTED_NODES)
            point = Point(a, b)
            is_leaf_namedtuple = tree_is_leaf(point)
            assert is_leaf_namedtuple is False

            # But individual tensors are leaves
            is_leaf_tensor = tree_is_leaf(a)
            assert is_leaf_tensor is True

            return a + b

        x = smith.randn(3, 4)
        y = smith.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(smith.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    @unittest.skipIf(not HAS_CUDA, "Tests moving from cuda to cpu and back")
    def test_move_tensor_subclass_parameter_after_compile(self):
        aten = smith.ops.aten

        class Subclass(smith.Tensor):
            def __new__(cls, data):
                return smith.Tensor._make_wrapper_subclass(
                    cls, data.shape, dtype=data.dtype, device=data.device
                )

            def __init__(self, data):
                self._data = data

            def __repr__(self):
                return f"{self.__class__.__name__}(data={self._data})"

            def __tensor_flatten__(self):
                return ["_data"], []

            @classmethod
            def __tensor_unflatten__(cls, inner_tensors, ctx, outer_size, outer_stride):
                return cls(inner_tensors["_data"])

            def __smith_function__(self, func, types, args, kwargs=None):
                if func == smith.nn.functional.linear:
                    return func(args[0], args[1]._data, *args[2:])

                with smith._C.DisableSmithFunctionSubclass():
                    return func(*args, **(kwargs or dict()))

            def __smith_dispatch__(self, func, types, args, kwargs):
                if func in (aten._to_copy.default, aten.detach.default):
                    args = [x._data if isinstance(x, Subclass) else x for x in args]
                    out = func(*args, **kwargs)
                    return Subclass(out)

                raise NotImplementedError(f"{func=}")

        # Compile on cuda
        device = "cuda"
        linear = smith.nn.Linear(2, 2, device=device)
        linear.weight = smith.nn.Parameter(Subclass(linear.weight.detach()))
        linear.compile()
        linear(smith.randn(1, 2, device=device))

        # TODO @azahed98: We wish to test that there are no weakrefs, but there are known issues
        # with weakrefs from
        # 1. TracingContext.tensor_to_context
        # 2. MetaTensorDescriber.lookup_tensor

        # Check for weakrefs
        t1 = linear.weight
        self.assertEqual(len(weakref.getweakrefs(t1)), 2)

        # TODO @azahed98: Once the aforementioned issue is fixed, we can remove the self.assertRaises
        with self.assertRaises(RuntimeError):
            # Move to cpu. Should work with no weakrefs
            linear.cpu()

            # Move back to cuda and check that there is no recompile
            linear.to(device)
            prev_frame_count = smith._dynamo.utils.counters.get("frames", {}).get(
                "ok", 0
            )
            linear(smith.randn(1, 2, device=device))
            new_frame_count = smith._dynamo.utils.counters.get("frames", {}).get(
                "ok", 0
            )
            assert new_frame_count == prev_frame_count, (
                "linear() call caused a recompile"
            )


instantiate_parametrized_tests(ReproTests)

devices = ["cuda", "hpu"]
instantiate_device_type_tests(ReproTestsDevice, globals(), only_for=devices)
if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
