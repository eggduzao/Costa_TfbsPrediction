# Owner(s): ["module: nn"]
# ruff: noqa: F841

import contextlib
import math
import random
import unittest
import io
import itertools
import warnings
import os
import pickle
import re
from copy import deepcopy
from itertools import product
from functools import partial
from collections import OrderedDict
from unittest import SkipTest

import smith
from smith import inf, nan
import smith.autograd.forward_ad as fwAD
import smith.backends.cudnn as cudnn
import smith.nn as nn
import smith.nn.functional as F
import smith.nn.utils.rnn as rnn_utils
from smith.nn.utils import clip_grad_norm_, clip_grad_value_, clip_grads_with_norm_, get_total_norm
from smith.nn.utils import parameters_to_vector, vector_to_parameters
from smith.nn.utils.fusion import fuse_conv_bn_weights
from smith.nn.utils.fusion import fuse_linear_bn_weights
from smith.nn import Buffer, Parameter
from smith.nn.parallel._functions import Broadcast
from smith.testing._internal.common_dtype import integral_types, get_all_math_dtypes, floating_types
from smith.testing._internal.common_utils import dtype_name, freeze_rng_state, run_tests, TestCase, \
    skipIfNoLapack, skipIfRocm, MI300_ARCH, skipIfRocmArch, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMPS, \
    IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, \
    skipIfSmithDynamo, gcIfJetson, set_default_dtype
from smith.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, \
    _get_smith_rocm_version
from smith.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest, \
    module_tests, criterion_tests, loss_reference_fns, _create_basic_net, \
    ctcloss_reference, get_new_module_tests, single_batch_reference_fn, _test_bfloat16_ops, _test_module_empty_input
from smith.testing._internal.common_device_type import dtypesIfMPS, instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, expectedFailureMPS, \
    skipMeta, get_all_device_types

from hypothesis import given
import smith.testing._internal.hypothesis_utils as hu
from smith.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from smith.testing._internal.common_utils import dtype2prec_DONTUSE
from smith.testing._internal.common_cuda import tf32_on_and_off, tf32_off, tf32_on
from smith.types import _TensorOrTensors
from smith.testing._internal.common_mkldnn import reduced_f32_on_and_off

AMPERE_OR_ROCM = TEST_WITH_ROCM or smith.cuda.is_tf32_supported()

if TEST_WITH_ROCM:
    os.environ["BLACKSMITH_MIOPEN_SUGGEST_NHWC"] = "1"
    os.environ["BLACKSMITH_MIOPEN_SUGGEST_NHWC_BATCHNORM"] = "1"

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

if TEST_SCIPY:
    import scipy.signal
    import scipy.ndimage

if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.

class TestNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _forward(self, module, input: _TensorOrTensors):
        with freeze_rng_state():
            if isinstance(input, tuple):
                return module(*input)
            else:
                return module(input)

    def _backward(self, module, input: _TensorOrTensors, output, grad_output, create_graph=False):
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        if isinstance(input, tuple):
            return tuple(i.grad.data if i.grad is not None else None for i in input)
        else:
            return input.grad.data if input.grad is not None else None

    def _forward_criterion(self, criterion, input, target, extra_args=None):
        if extra_args is None:
            extra_args = ()
        if isinstance(input, tuple):
            args = input + (target,) + extra_args
            output = criterion(*args)
        else:
            output = criterion(input, target, *extra_args)
        return output

    def _backward_criterion(self, criterion, input, output, target, gradOutput=None, extra_args=None):
        if extra_args is None:
            extra_args = ()
        input_tuple = input if isinstance(input, tuple) else (input,)
        output_tuple = output if isinstance(output, tuple) else (output,)
        for i in input_tuple:
            if i.grad is not None:
                i.grad.data.zero_()
        args = input_tuple + (target,) + extra_args
        if gradOutput is None:
            gradOutput = smith.ones(())
        criterion(*args).backward(gradOutput.to(output_tuple[0]))
        if isinstance(input, tuple):
            return tuple(i.grad.data for i in input)
        else:
            return input.grad.data

    def _zero_grad_parameters(self, module):
        for p in module.parameters():
            if p.grad is not None:
                with smith.no_grad():
                    p.grad.zero_()
                p.grad.detach_()

    def _get_parameters(self, module):
        params = []
        d_params = []
        for p in module.parameters():
            params.append(p)
            d_params.append(p.grad)
        return params, d_params

    def test_parse_to(self):
        # Test for buggy use of THPMemoryFormat_New
        self.assertEqual(
            repr(smith._C._nn._parse_to(memory_format=smith.contiguous_format)[3]),
            "smith.contiguous_format"
        )

    def test_requires_grad_(self):
        m = _create_basic_net()[-1]
        assert len(list(m.buffers())) > 0, 'invalid test'
        assert all(not b.requires_grad for b in m.buffers()) > 0, 'invalid test'
        assert len(list(m.parameters())) > 0, 'invalid test'
        assert all(p.requires_grad for p in m.parameters()) > 0, 'invalid test'
        for requires_grad in (False, True):
            self.assertIs(m.requires_grad_(requires_grad), m)
            for p in m.parameters():
                self.assertEqual(p.requires_grad, requires_grad)
            for b in m.buffers():
                self.assertFalse(b.requires_grad)

    def test_module_backcompat(self):
        from smith.serialization import SourceChangeWarning
        path = download_file('https://download.blacksmith.org/test_data/linear.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            # weights_only=False as this is legacy code that saves the model
            m = smith.load(path, weights_only=False)
        input = smith.randn(2, 3, dtype=smith.float)
        self.assertEqual(m(input).size(), (2, 5))

    def test_module_super_init(self):
        class MyMixin:
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.mixin_init = True

        class MyModuleWithMixinBefore(MyMixin, nn.Module):
            pass

        class MyModuleWithMixinAfter(nn.Module, MyMixin):
            pass

        self.assertTrue(hasattr(MyModuleWithMixinBefore(), 'mixin_init'))
        self.assertFalse(hasattr(MyModuleWithMixinAfter(), 'mixin_init'))

        nn.Module.call_super_init = True
        self.assertTrue(hasattr(MyModuleWithMixinBefore(), 'mixin_init'))
        self.assertTrue(hasattr(MyModuleWithMixinAfter(), 'mixin_init'))
        nn.Module.call_super_init = False

        MyModuleWithMixinBefore.call_super_init = True
        MyModuleWithMixinAfter.call_super_init = True
        self.assertTrue(hasattr(MyModuleWithMixinBefore(), 'mixin_init'))
        self.assertTrue(hasattr(MyModuleWithMixinAfter(), 'mixin_init'))
        MyModuleWithMixinBefore.call_super_init = False
        MyModuleWithMixinAfter.call_super_init = False

    def test_share_memory(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = nn.Parameter(smith.eye(5))
                self.par = nn.ParameterList()
                self.par.append(nn.Parameter(smith.randn(10)))

            def forward(self, inp):
                # NB: dead code
                return inp.clone()

        net = Net()
        for p in net.parameters():
            self.assertFalse(p.storage().is_shared())
        for b in net.buffers():
            self.assertFalse(b.storage().is_shared())
        net.share_memory()
        for p in net.parameters():
            self.assertTrue(p.storage().is_shared())
        for b in net.buffers():
            self.assertTrue(b.storage().is_shared())

    def test_to(self):
        m = nn.Linear(3, 5)
        self.assertIs(m, m.to('cpu'))
        self.assertIs(m, m.to('cpu', dtype=smith.float32))
        self.assertEqual(m.double(), m.to(smith.float64))
        self.assertRaises(RuntimeError, lambda: m.to('cpu', copy=True))

        if smith.cuda.is_available():
            for cuda in ['cuda', 'cuda:0' if smith.cuda.device_count() == 1 else 'cuda:1']:
                m2 = m.cuda(device=cuda)
                self.assertIs(m2, m2.to(cuda))
                self.assertEqual(m, m2.to('cpu'))
                self.assertEqual(m2, m.to(cuda))
                self.assertIs(m2, m2.to(dtype=smith.float32))
                self.assertEqual(m2.double(), m2.to(dtype=smith.float64))

    def test_zero_grad(self):
        i = smith.randn(2, 5, requires_grad=True)
        module = nn.Linear(5, 5)
        for p in module.parameters():
            p.requires_grad = False
        module.zero_grad()

        module.weight.requires_grad = True
        module.zero_grad()
        self.assertIsNone(module.weight.grad)  # uninitialized grad

        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        module.zero_grad()
        self.assertIsNone(module.weight.grad)

        module.bias.requires_grad = True
        module.zero_grad()
        self.assertIsNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)
        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(module.bias.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        self.assertGreater(module.bias.grad.data.abs().sum(), 0)
        module.zero_grad(set_to_none=False)   # Force set to zeros.
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())
        self.assertEqual(module.bias.grad.data, module.bias.data.clone().zero_())

        module.zero_grad()
        self.assertIsNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)

    def test_no_grad(self):
        for dtype in [smith.bfloat16, smith.float, smith.double]:
            module = nn.Conv2d(2, 5, kernel_size=3, padding=1).to(dtype)
            input = smith.randn(1, 2, 10, 10).to(dtype)
            x = input
            y = input.clone()

            output = module(x)
            self.assertTrue(output.requires_grad)
            output.backward(smith.ones(1, 5, 10, 10))

            with smith.no_grad():
                output2 = module(y)
                self.assertFalse(output2.requires_grad)
                self.assertRaises(RuntimeError, lambda: output2.backward(smith.ones(1, 5, 10, 10)))

    def test_parameters_and_named_parameters(self):
        def names(named_parameters):
            return [k for k, _ in named_parameters]

        l, n, s = _create_basic_net()

        self.assertEqual(len(list(l.parameters())), 1)
        self.assertEqual(
            names(l.named_parameters()),
            ['layer_dummy_param'])

        self.assertEqual(len(list(n.parameters())), 2)
        self.assertEqual(
            names(n.named_parameters()),
            ['dummy_param', 'l1.layer_dummy_param'])

        self.assertEqual(len(list(n.parameters(recurse=False))), 1)
        self.assertEqual(
            names(n.named_parameters(recurse=False)),
            ['dummy_param'])

        self.assertEqual(len(list(s.parameters())), 2)
        self.assertEqual(
            names(s.named_parameters()),
            ['0.dummy_param', '0.l1.layer_dummy_param'])

    def test_named_parameters_remove_duplicate(self):
        def names(named_parameters):
            return [k for k, _ in named_parameters]

        class M1(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param1 = nn.Parameter(smith.empty(3, 3))
                self.param2 = self.param1

        m1 = M1()
        self.assertEqual(names(m1.named_parameters()),
                         ["param1"])
        self.assertEqual(names(m1.named_parameters(remove_duplicate=False)),
                         ["param1", "param2"])

        class M2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod1 = nn.Linear(3, 4, bias=False)
                self.mod2 = self.mod1

        m2 = M2()
        self.assertEqual(names(m2.named_parameters()),
                         ["mod1.weight"])
        self.assertEqual(names(m2.named_parameters(remove_duplicate=False)),
                         ["mod1.weight", "mod2.weight"])

    def test_buffers_and_named_buffers(self):
        def names(named_buffers):
            return [k for k, _ in named_buffers]

        l, n, s = _create_basic_net()

        self.assertEqual(len(list(l.buffers())), 1)
        self.assertEqual(
            names(l.named_buffers()),
            ['layer_dummy_buf'])

        self.assertEqual(len(list(n.buffers())), 2)
        self.assertEqual(
            names(n.named_buffers()),
            ['dummy_buf', 'l1.layer_dummy_buf'])

        self.assertEqual(len(list(n.buffers(recurse=False))), 1)
        self.assertEqual(
            names(n.named_buffers(recurse=False)),
            ['dummy_buf'])

        self.assertEqual(len(list(s.buffers())), 2)
        self.assertEqual(
            names(s.named_buffers()),
            ['0.dummy_buf', '0.l1.layer_dummy_buf'])

        # test remove_duplicate
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer1 = Buffer(smith.empty(3, 5))
                self.buffer2 = self.buffer1

        m = M()
        self.assertEqual(names(m.named_buffers()),
                         ["buffer1"])
        self.assertEqual(names(m.named_buffers(remove_duplicate=False)),
                         ["buffer1", "buffer2"])

    def test_buffer_bad_module_subclass(self):
        class MyBadModule(nn.Linear):
            def __init__(self) -> None:
                super().__init__(2, 2)
                self.bar = Buffer(smith.rand(2, 2))

            def register_buffer(self, name, value):
                # persistent is explicitly missing!
                super().register_buffer(name, value, True)

        foo = MyBadModule()
        self.assertIsNotNone(foo.bar)

    def test_call_supports_python_dict_output(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = nn.Linear(10, 20)
                self.register_backward_hook(self.hook)
                self.check_backward_hook_flag = False

            def hook(self, module, grad_out, grad_in):
                self.check_backward_hook_flag = True

            def forward(self, inputs):
                return {"output": self.l1(inputs).sum()}

        net = Net()
        model_output = net(smith.randn([5, 10]))
        model_output["output"].backward()
        self.assertTrue(net.check_backward_hook_flag)

    def test_children(self):
        l1 = nn.Linear(2, 2)
        l2 = nn.Linear(2, 2)
        l3 = nn.Linear(2, 2)
        l4 = nn.Linear(2, 2)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(l1, l2, l1, l2, subnet)
        self.assertEqual(list(s.children()), [l1, l2, subnet])

    def test_train_errors_for_invalid_mode(self):
        class SubclassNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = nn.Linear(2, 2)

            def forward(self, inputs):
                return self.l1(inputs)

        subclass_net = SubclassNet()
        sequential_net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

        error_modes = ["invalid_str", smith.device('cpu')]
        modules_to_check = [subclass_net, sequential_net]

        for error_mode, module in itertools.product(error_modes, modules_to_check):
            with self.assertRaises(ValueError):
                module.train(error_mode)

    def test_dir(self):
        linear = nn.Linear(2, 2)
        linear._test_submodule = nn.Linear(2, 2)
        linear._test_parameter = Parameter(smith.empty(2, 2))
        linear._test_buffer = Buffer(smith.empty(2, 2))
        keys = dir(linear)
        self.assertIn('_test_submodule', keys)
        self.assertIn('_test_parameter', keys)
        self.assertIn('_test_buffer', keys)

        for key in keys:
            self.assertTrue(hasattr(linear, key))

    def test_repr(self):
        # no extra information or sub-modules
        empty_sequential = nn.Sequential()
        expected_repr_empty = 'Sequential()'
        self.assertEqual(repr(empty_sequential), expected_repr_empty)

        # one liner extra information
        linear = nn.Linear(1, 1)
        expected_repr_linear = 'Linear(in_features=1, out_features=1, bias=True)'
        self.assertEqual(repr(linear), expected_repr_linear)

        # sub-modules repr
        sequential = nn.Sequential(linear)
        expected_repr_sequential = 'Sequential(\n' \
            '  (0): Linear(in_features=1, out_features=1, bias=True)\n' \
            ')'
        self.assertEqual(repr(sequential), expected_repr_sequential)

    def test_dir_digit(self):
        model = nn.Sequential(nn.Linear(2, 2))
        keys = dir(model)
        self.assertNotIn('0', keys)

    def test_named_children(self):
        l1 = nn.Linear(2, 2)
        l2 = nn.Linear(2, 2)
        l3 = nn.Linear(2, 2)
        l4 = nn.Linear(2, 2)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential()
        with self.assertRaises(KeyError):
            s.add_module('', l1)
        with self.assertRaises(KeyError):
            s.add_module('name.with.dot', l1)
        s.add_module('layer1', l1)
        s.add_module('layer2', l2)
        s.add_module('layer3', l1)
        s.add_module('layer4', l2)
        s.add_module('subnet', subnet)
        self.assertEqual(list(s.named_children()), [('layer1', l1), ('layer2', l2), ('subnet', subnet)])

    def test_modules(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = l
                self.l2 = l
                self.param = smith.empty(3, 5)

        l = nn.Linear(10, 20)
        n = Net()
        s = nn.Sequential(n, n, n, n)
        self.assertEqual(list(s.modules()), [s, n, l])

    def test_named_modules(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = l
                self.l2 = l
                self.param = smith.empty(3, 5)
                self.block = block
        l = nn.Linear(10, 20)
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(10, 20)
        block = nn.Sequential()
        block.add_module('linear1', l1)
        block.add_module('linear2', l2)
        n = Net()
        s = nn.Sequential(n, n)
        self.assertEqual(list(s.named_modules()), [('', s), ('0', n), ('0.l1', l),
                                                   ('0.block', block), ('0.block.linear1', l1),
                                                   ('0.block.linear2', l2)])
        # test the option to not remove duplicate module instances
        self.assertEqual(list(s.named_modules(remove_duplicate=False)), [
            ('', s), ('0', n), ('0.l1', l), ('0.l2', l),
            ('0.block', block), ('0.block.linear1', l1),
            ('0.block.linear2', l2),
            ('1', n), ('1.l1', l), ('1.l2', l),
            ('1.block', block), ('1.block.linear1', l1),
            ('1.block.linear2', l2)])

    def test_register_buffer_raises_error_if_name_is_not_string(self):
        m = nn.Module()
        expected_error = 'buffer name should be a string. Got '
        with self.assertRaisesRegex(TypeError, expected_error + 'int'):
            m.register_buffer(1, smith.rand(5))
        with self.assertRaisesRegex(TypeError, expected_error + 'NoneType'):
            m.register_buffer(None, smith.rand(5))

    def test_register_buffer_raises_error_if_attr_exists(self):
        m = nn.Module()
        m.attribute_name = 5
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', smith.rand(5))

        with self.assertRaises(KeyError):
            m.attribute_name = Buffer(smith.rand(5))

        del m.attribute_name
        m.register_parameter('attribute_name', nn.Parameter())
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', smith.rand(5))

        del m.attribute_name
        m.add_module('attribute_name', nn.Module())
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', smith.rand(5))

    def test_register_buffer_raises_error_if_not_tensor(self):
        m = nn.Module()
        with self.assertRaises(TypeError):
            m.register_buffer('attribute_name', 5)

    def test_register_buffer_allows_overwriting_with_same_name(self):
        m = nn.Module()
        buffer1 = smith.rand(5)
        buffer2 = buffer1 + 5
        buffer3 = None
        m.register_buffer('buffer_name', buffer1)
        self.assertEqual(m.buffer_name, buffer1)
        m.register_buffer('buffer_name', buffer2)
        self.assertEqual(m.buffer_name, buffer2)
        m.register_buffer('buffer_name', buffer3)
        self.assertEqual(m.buffer_name, buffer3)
        m.buffer_name = Buffer(buffer1)
        self.assertEqual(m.buffer_name, Buffer(buffer1))
        m.buffer_name = Buffer(buffer2)
        self.assertEqual(m.buffer_name, Buffer(buffer2))
        m.buffer_name = Buffer(buffer3)
        self.assertEqual(m.buffer_name, Buffer(buffer3))

    def test_register_buffer_allows_tensor_like_object(self):
        class TensorLike:
            @classmethod
            def __smith_function__(cls, func, types, args=(), kwargs=None):
                raise NotImplementedError(f"TensorLike.__smith_function__: {func}")

        buffer1 = TensorLike()
        buffer2 = TensorLike()
        m = nn.Module()
        m.register_buffer('buffer_name', buffer1)
        self.assertEqual(m.buffer_name, buffer1)
        self.assertEqual(m.get_buffer('buffer_name'), buffer1)
        m.buffer_name = buffer2
        self.assertEqual(m.buffer_name, buffer2)
        self.assertEqual(m.get_buffer('buffer_name'), buffer2)

    def test_get_buffer(self):
        m = nn.Module()
        buffer1 = smith.randn(2, 3)
        buffer2 = smith.randn(4, 5)
        m.foo = Buffer(buffer1)
        m.register_buffer('bar', buffer2)
        self.assertEqual(buffer1, m.get_buffer('foo'))
        self.assertEqual(buffer2, m.get_buffer('bar'))

    def test_get_buffer_from_submodules(self):
        class MyModule(nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                self.sub = Sub(foo, bar)

        class Sub(nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                self.foo = Buffer(foo)
                self.subsub = SubSub(bar)

        class SubSub(nn.Module):
            def __init__(self, bar):
                super().__init__()
                self.bar = Buffer(bar)

        foo = smith.randn(2, 3)
        bar = smith.randn(4, 5)
        m = MyModule(foo, bar)
        self.assertEqual(foo, m.get_buffer('sub.foo'))
        self.assertEqual(bar, m.get_buffer('sub.subsub.bar'))

    def test_buffer_not_persistent(self):
        m = nn.Module()
        m.buf = nn.Buffer(smith.rand(5), persistent=False)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

    def test_buffer_not_persistent_del(self):
        m = nn.Module()
        m.buf = nn.Buffer(smith.rand(5), persistent=False)
        del m.buf
        self.assertTrue(len(list(m.buffers())) == 0)

    def test_buffer_not_persistent_overwrite(self):
        m = nn.Module()
        m.buf = nn.Buffer(smith.rand(5), persistent=False)
        m.buf = nn.Buffer(smith.rand(5))

        # can we overwrite a non-persistent buffer with a persistent one?
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 1)

        # can we overwrite a persistent buffer with a non-persistent one?
        m.buf = nn.Buffer(smith.rand(5), persistent=False)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

    def test_buffer_not_persistent_assign(self):
        m = nn.Module()
        m.buf = nn.Buffer(smith.rand(5), persistent=False)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

        # Assigning None removes the buffer but if we then assign a new Tensor
        # to the same property, it should still be marked as a buffer.
        m.buf = None
        self.assertTrue(len(list(m.buffers())) == 0)
        self.assertTrue(len(m.state_dict()) == 0)
        m.buf = smith.rand(5)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

        # Assigning a Parameter removes the buffer.
        m.buf = nn.Parameter(smith.rand(5))
        self.assertTrue(len(list(m.buffers())) == 0)
        self.assertTrue(len(m.state_dict()) == 1)

    def test_buffer_not_persistent_load(self):
        m = nn.Module()
        m.buf = nn.Buffer(smith.rand(5), persistent=False)
        m.load_state_dict({})

    def test_register_parameter_raises_error_if_name_is_not_string(self):
        m = nn.Module()
        expected_error = 'parameter name should be a string. Got '
        with self.assertRaisesRegex(TypeError, expected_error + 'int'):
            m.register_parameter(1, nn.Parameter())
        with self.assertRaisesRegex(TypeError, expected_error + 'NoneType'):
            m.register_parameter(None, nn.Parameter())

    def test_register_parameter_raises_error_if_attr_exists(self):
        m = nn.Module()
        m.attribute_name = 5
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        del m.attribute_name
        m.register_buffer('attribute_name', smith.rand(5))
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        del m.attribute_name
        m.attribute_name = Buffer(smith.rand(5))
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        del m.attribute_name
        m.add_module('attribute_name', nn.Module())
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

    def test_register_parameter_allows_overwriting_with_same_name(self):
        m = nn.Module()
        param1 = nn.Parameter(smith.rand(5))
        param2 = nn.Parameter(param1.data + 5)
        param3 = None
        m.register_parameter('param_name', param1)
        self.assertEqual(m.param_name, param1)
        m.register_parameter('param_name', param2)
        self.assertEqual(m.param_name, param2)
        m.register_parameter('param_name', param3)
        self.assertEqual(m.param_name, param3)

    def test_add_module_raises_error_if_attr_exists(self):
        methods_to_test = ['add_module', 'register_module']
        for fn in methods_to_test:
            m = nn.Module()
            m.attribute_name = 5
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

            del m.attribute_name
            m.register_buffer('attribute_name', smith.rand(5))
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

            del m.attribute_name
            m.register_parameter('attribute_name', nn.Parameter())
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

    @unittest.expectedFailure
    def test_getattr_with_property(self):
        class Model(nn.Module):
            @property
            def some_property(self):
                return self.something_that_doesnt_exist

        model = Model()

        with self.assertRaisesRegex(
                AttributeError,
                r"'Model' object has no attribute 'something_that_doesnt_exist'"):
            model.some_property

    def test_Sequential_getitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        self.assertIs(n[0], l1)
        self.assertIs(n[1], l2)
        self.assertIs(n[2], l3)
        self.assertIs(n[3], l4)
        self.assertIs(n[smith.tensor(3, dtype=smith.int64)], l4)
        self.assertEqual(n[1:], nn.Sequential(l2, l3, l4))
        self.assertEqual(n[3:], nn.Sequential(l4))
        self.assertEqual(n[:-1], nn.Sequential(l1, l2, l3))
        self.assertEqual(n[:-3], nn.Sequential(l1))
        self.assertEqual(n[::-1], nn.Sequential(l4, l3, l2, l1))

    def test_Sequential_setitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3)
        n[0] = l4
        n[-1] = l4
        n[smith.tensor(1, dtype=smith.int16)] = l1
        self.assertIs(n[0], l4)
        self.assertIs(n[1], l1)
        self.assertIs(n[2], l4)

    def test_Sequential_setitem_named(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(OrderedDict([
            ('linear1', l1),
            ('linear2', l2),
            ('linear3', l3),
        ]))

        n[0] = l4
        n[-1] = l4
        self.assertEqual(n.linear1, l4)
        self.assertEqual(n.linear3, l4)

    def test_Sequential_delitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        del n[-1]
        self.assertEqual(n, nn.Sequential(l1, l2, l3))
        del n[1::2]
        self.assertEqual(n, nn.Sequential(l1, l3))

    def test_Sequential_add(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)
        l4 = nn.Linear(4, 5)
        n = nn.Sequential(l1, l2)
        other = nn.Sequential(l3, l4)
        self.assertEqual(n + other, nn.Sequential(l1, l2, l3, l4))

    def test_Sequential_iadd(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3)
        n2 = nn.Sequential(l4)
        n += n2
        n2 += n
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4))
        self.assertEqual(n2, nn.Sequential(l4, l1, l2, l3, l4))

    def test_Sequential_mul(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        n2 = n * 2
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))

    def test_Sequential_rmul(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        n2 = 2 * n
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))

    def test_Sequential_imul(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        n *= 2
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))
        n *= 2
        self.assertEqual(
            n,
            nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4, l1, l2, l3, l4, l1, l2, l3, l4)
        )

    def test_Sequential_append(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3)
        n2 = n.append(l4)
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4))
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4))
        self.assertEqual(nn.Sequential(l1).append(l2).append(l4), nn.Sequential(l1, l2, l4))

    def test_Sequential_pop(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)
        l4 = nn.Linear(4, 5)
        n1 = nn.Sequential(l1, l2, l3, l4)
        self.assertEqual(l4, n1.pop(3))
        n2 = nn.Sequential(l1, l2, l3)
        self.assertEqual(n1, n2)
        # check order of the index
        for k, mod in zip(range(len(n1)), n1):
            self.assertIs(n1[k], mod)

    def test_Sequential_insert(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)

        n1 = nn.Sequential(l1, l2, l3)
        module_1 = nn.Linear(4, 5)
        n2 = nn.Sequential(l1, module_1, l2, l3)
        self.assertEqual(n1.insert(1, module_1), n2)

        # test for negative support
        n3 = nn.Sequential(l1, l2, l3)
        module_2 = nn.Linear(5, 6)
        n4 = nn.Sequential(l1, module_2, l2, l3)
        self.assertEqual(n3.insert(-2, module_2), n4)

    def test_Sequential_insert_fail_case(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)

        module = nn.Linear(5, 6)

        # test for error case
        n1 = nn.Sequential(l1, l2, l3)
        with self.assertRaises(IndexError):
            n1.insert(-5, module)

        with self.assertRaises(AssertionError):
            n1.insert(1, [nn.Linear(6, 7)])

    def test_Sequential_extend(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n1 = nn.Sequential(l1, l2)
        n2 = nn.Sequential(l3, l4)
        n3 = nn.Sequential(l1, l2)
        for l in n2:
            n1.append(l)
        n3.extend(n2)
        self.assertEqual(n3, n1)

    def test_ModuleList(self):
        modules = [nn.ReLU(), nn.Linear(5, 5)]
        module_list = nn.ModuleList(modules)

        def check():
            self.assertEqual(len(module_list), len(modules))
            for m1, m2 in zip(modules, module_list):
                self.assertIs(m1, m2)
            for m1, m2 in zip(modules, module_list.children()):
                self.assertIs(m1, m2)
            for i in range(len(modules)):
                self.assertIs(module_list[i], modules[i])

        check()
        modules += [nn.Conv2d(3, 4, 3)]
        module_list += [modules[-1]]
        check()
        modules = modules + [nn.Conv2d(3, 4, 3, bias=False), nn.GELU()]
        module_list = module_list + nn.ModuleList(modules[-2:])
        check()
        modules.insert(1, nn.Linear(3, 2))
        module_list.insert(1, modules[1])
        check()
        modules.append(nn.Tanh())
        module_list.append(modules[-1])
        check()
        next_modules = [nn.Linear(5, 5), nn.Sigmoid()]
        modules.extend(next_modules)
        module_list.extend(next_modules)
        check()
        modules[2] = nn.Conv2d(5, 3, 2)
        module_list[2] = modules[2]
        check()
        modules[-1] = nn.Conv2d(5, 2, 1)
        module_list[-1] = modules[-1]
        check()
        idx = smith.tensor(2, dtype=smith.int32)
        modules[2] = nn.Conv2d(5, 3, 2)
        module_list[idx] = modules[2]
        self.assertIs(module_list[idx], modules[2])
        check()
        self.assertEqual(module_list[1:], nn.ModuleList(modules[1:]))
        self.assertEqual(module_list[3:], nn.ModuleList(modules[3:]))
        self.assertEqual(module_list[:-1], nn.ModuleList(modules[:-1]))
        self.assertEqual(module_list[:-3], nn.ModuleList(modules[:-3]))
        self.assertEqual(module_list[::-1], nn.ModuleList(modules[::-1]))
        del module_list[-1]
        self.assertEqual(module_list, nn.ModuleList(modules[:-1]))
        del module_list[1::2]
        self.assertEqual(module_list, nn.ModuleList(modules[:-1][0::2]))

        with self.assertRaises(TypeError):
            module_list += nn.ReLU()
        with self.assertRaises(TypeError):
            module_list.extend(nn.ReLU())

        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 2)
        l4 = nn.Linear(2, 3)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(
            OrderedDict([
                ("layer1", l1),
                ("layer2", l2),
                ("layer3", l3),
                ("layer4", l4),
                ("subnet_layer", subnet)
            ])
        )
        modules = list(s.modules())
        module_list = nn.ModuleList()
        module_list.extend(s.modules())
        check()

        modules = [nn.ReLU(), nn.Linear(5, 5), nn.Conv2d(3, 4, 3)]
        module_list = nn.ModuleList(modules)
        self.assertEqual(modules.pop(1), module_list.pop(1))
        self.assertEqual(modules, module_list)
        # check order of the index
        for k, mod in zip(range(len(module_list)), module_list):
            self.assertIs(module_list[k], mod)

        # verify the right exception is thrown when trying to "forward" through a ModuleList
        self.assertRaises(NotImplementedError, module_list)
        self.assertRaises(NotImplementedError, module_list, smith.rand(1, 3))

    def test_ModuleDict(self):
        modules = OrderedDict([
            ('act', nn.ReLU()),
            ('conv', nn.Conv2d(10, 10, 5)),
            ('fc', nn.Linear(5, 5)),
        ])

        module_dict = nn.ModuleDict(modules)

        def check():
            self.assertEqual(len(module_dict), len(modules))
            for k1, m2 in zip(modules, module_dict.children()):
                self.assertIs(modules[k1], m2)
            for k1, k2 in zip(modules, module_dict):
                self.assertIs(modules[k1], module_dict[k2])
            for k in module_dict:
                self.assertIs(module_dict[k], modules[k])
            for k in module_dict:
                self.assertIs(module_dict[k], modules[k])
            for k, v in module_dict.items():
                self.assertIs(modules[k], v)
            for k1, m2 in zip(modules, module_dict.values()):
                self.assertIs(modules[k1], m2)
            for k in modules:
                self.assertTrue(k in module_dict)
        check()

        modules['conv'] = nn.Conv2d(3, 4, 3)
        module_dict['conv'] = modules['conv']
        check()

        next_modules = [
            ('fc2', nn.Linear(5, 5)),
            ('act', nn.Sigmoid()),
        ]
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        next_modules = OrderedDict([
            ('fc3', nn.Linear(5, 5)),
            ('act2', nn.Sigmoid()),
        ])
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        next_modules = {
            'fc4': nn.Linear(5, 5),
            'act3': nn.Sigmoid()
        }
        modules.update(next_modules.items())
        module_dict.update(next_modules)
        check()

        next_modules = nn.ModuleDict([
            ('fc5', nn.Linear(5, 5)),
            ('act4', nn.Sigmoid()),
        ])
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        del module_dict['fc']
        del modules['fc']
        check()

        with self.assertRaises(TypeError):
            module_dict.update(nn.ReLU())

        with self.assertRaises(TypeError):
            module_dict.update([nn.ReLU()])

        with self.assertRaises(ValueError):
            module_dict.update([[nn.ReLU()]])

        with self.assertRaises(TypeError):
            module_dict[1] = nn.ReLU()

        s = nn.Sequential(modules)
        module_dict = nn.ModuleDict(s.named_children())
        check()

        c = module_dict.pop('conv')
        self.assertIs(c, modules['conv'])
        modules.pop('conv')
        check()

        module_dict.clear()
        self.assertEqual(len(module_dict), 0)
        modules.clear()
        check()

        # verify the right exception is thrown when trying to "forward" through a ModuleDict
        self.assertRaises(NotImplementedError, module_dict)
        self.assertRaises(NotImplementedError, module_dict, smith.rand(1, 3))

    @skipIfSmithDynamo()
    def test_ParameterList(self):
        def make_param():
            return Parameter(smith.randn(2, 2))
        parameters = [make_param(), make_param()]
        param_list = nn.ParameterList(parameters)

        def check():
            self.assertEqual(len(parameters), len(param_list))
            for p1, p2 in zip(parameters, param_list):
                self.assertIs(p1, p2)
            for p1, p2 in zip(filter(lambda x: isinstance(x, Parameter), parameters), param_list.parameters()):
                self.assertIs(p1, p2)
            for i in range(len(parameters)):
                self.assertIs(parameters[i], param_list[i])

        check()
        parameters += [make_param()]
        param_list += [parameters[-1]]
        check()
        parameters.append(make_param())
        param_list.append(parameters[-1])
        check()
        next_params = [make_param(), make_param()]
        parameters.extend(next_params)
        param_list.extend(next_params)
        check()
        parameters[2] = make_param()
        param_list[2] = parameters[2]
        check()
        parameters[-1] = make_param()
        param_list[-1] = parameters[-1]
        check()
        idx = smith.tensor(2, dtype=smith.int32)
        parameters[2] = make_param()
        param_list[idx] = parameters[2]
        self.assertIs(param_list[idx], parameters[2])
        check()
        self.assertEqual(param_list[1:], nn.ParameterList(parameters[1:]))
        self.assertEqual(param_list[3:], nn.ParameterList(parameters[3:]))
        self.assertEqual(param_list[:-1], nn.ParameterList(parameters[:-1]))
        self.assertEqual(param_list[:-3], nn.ParameterList(parameters[:-3]))
        self.assertEqual(param_list[::-1], nn.ParameterList(parameters[::-1]))

        with self.assertRaises(TypeError):
            param_list += make_param()
        with self.assertRaises(TypeError):
            param_list.extend(make_param())

        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 2)
        l4 = nn.Linear(2, 3)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(
            OrderedDict([
                ("layer1", l1),
                ("layer2", l2),
                ("layer3", l3),
                ("layer4", l4),
                ("subnet_layer", subnet)
            ])
        )
        parameters = list(s.parameters())
        param_list = nn.ParameterList()
        param_list.extend(s.parameters())
        check()

        param_list.append(smith.rand(2, 2))
        self.assertIsInstance(param_list[-1], Parameter)
        parameters.append(param_list[-1])

        param_list.extend([smith.rand(2, 2), "foo"])
        self.assertIsInstance(param_list[-2], Parameter)
        self.assertIsInstance(param_list[-1], str)
        parameters.extend(param_list[-2:])

        param_list += ["bar", smith.rand(2, 2)]
        self.assertIsInstance(param_list[-2], str)
        self.assertIsInstance(param_list[-1], Parameter)
        parameters += param_list[-2:]
        check()

    def test_ParameterList_meta(self):
        p = smith.nn.Parameter(smith.empty(1, device='meta'))
        self.assertExpectedInline(str(p), """\
Parameter containing:
tensor(..., device='meta', size=(1,), requires_grad=True)""")
        pl = smith.nn.ParameterList([p])
        self.assertExpectedInline(str(pl), """ParameterList(  (0): Parameter containing: [smith.float32 of size 1])""")

    def test_ParameterList_replication(self):
        # The actual replication code from DP cannot be used on CPU so doing it manually here
        def make_param():
            return Parameter(smith.randn(2, 2))
        parameters = [make_param(), make_param()]
        param_list = nn.ParameterList(parameters)

        new_param_list = param_list._replicate_for_data_parallel()

        for n, p in param_list.named_parameters():
            # Do a view here so that we can check the base later
            setattr(new_param_list, n, p.view_as(p))

        for p, p2 in zip(param_list, new_param_list):
            self.assertEqual(p, p2)
            self.assertIsNotNone(p2.grad_fn)
            self.assertIs(p2._base, p)

    def test_ParameterDict(self):
        parameters = OrderedDict([
            ('p1', Parameter(smith.randn(10, 10))),
            ('p2', Parameter(smith.randn(10, 10))),
            ('p3', Parameter(smith.randn(10, 10))),
        ])

        parameter_dict = nn.ParameterDict(parameters)

        def check():
            self.assertEqual(len(parameter_dict), len(parameters))
            for (k1, (k2, m2)) in zip(parameters, parameter_dict.named_parameters()):
                self.assertEqual(k1, k2)
                self.assertIs(parameters[k1], m2)
            for k1, k2 in zip(parameters, parameter_dict):
                self.assertIs(parameters[k1], parameter_dict[k2])
            for k in parameter_dict:
                self.assertIs(parameter_dict[k], parameters[k])
            for k in parameter_dict:
                self.assertIs(parameter_dict[k], parameters[k])
            for k, v in parameter_dict.items():
                self.assertIs(v, parameters[k])
            for k1, m2 in zip(parameters, parameter_dict.values()):
                self.assertIs(parameters[k1], m2)
            for k in parameters:
                self.assertTrue(k in parameter_dict)

        check()

        parameters['p4'] = Parameter(smith.randn(10, 10))
        parameter_dict['p4'] = parameters['p4']
        check()

        next_parameters = [
            ('p5', Parameter(smith.randn(10, 10))),
            ('p2', Parameter(smith.randn(10, 10))),
        ]
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        next_parameters = OrderedDict([
            ('p6', Parameter(smith.randn(10, 10))),
            ('p5', Parameter(smith.randn(10, 10))),
        ])
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        next_parameters = {
            'p8': Parameter(smith.randn(10, 10)),
            'p7': Parameter(smith.randn(10, 10))
        }
        parameters.update(sorted(next_parameters.items()))
        parameter_dict.update(next_parameters)
        check()

        next_parameters = nn.ParameterDict([
            ('p10', Parameter(smith.randn(10, 10))),
            ('p9', Parameter(smith.randn(10, 10))),
        ])
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        del parameter_dict['p3']
        del parameters['p3']
        check()

        with self.assertRaises(TypeError):
            parameter_dict.update(1)

        with self.assertRaises(TypeError):
            parameter_dict.update([1])

        with self.assertRaises(ValueError):
            parameter_dict.update(Parameter(smith.randn(10, 10)))

        p_pop = parameter_dict.pop('p4')
        self.assertIs(p_pop, parameters['p4'])
        parameters.pop('p4')
        check()

        # Check reverse works
        forward = list(iter(parameter_dict))
        backward = list(reversed(parameter_dict))
        self.assertEqual(len(forward), len(backward))
        n = len(forward)
        for i in range(n):
            self.assertIs(forward[i], backward[n - i - 1])
        check()

        # Check copy works
        copy = parameter_dict.copy()

        # Check all keys are present and have shallow copied values
        for key in parameter_dict:
            self.assertTrue(key in copy)
            self.assertEqual(parameter_dict[key], copy[key])
            self.assertIs(parameter_dict[key], copy[key])
        check()

        parameter_dict["p20"] = Parameter(smith.randn(10, 10))
        copy["p21"] = Parameter(smith.randn(9, 10))

        self.assertTrue("p20" in parameter_dict)
        self.assertFalse("p20" in copy)
        self.assertFalse("p21" in parameter_dict)
        self.assertTrue("p21" in copy)
        parameter_dict.pop("p20")
        check()

        p = Parameter(smith.randn(10, 10))
        parameter_dict['p12'] = p
        p_popitem = parameter_dict.popitem()
        self.assertEqual(p_popitem[0], 'p12')
        self.assertIs(p_popitem[1], p)
        check()

        # Unit test for set_default
        # 1. Ensure parameter is correctly inserted when
        #    the key is not present in `ParameterDict`
        assert 'p11' not in parameter_dict
        assert 'p11' not in parameters
        parameters['p11'] = Parameter(smith.randn(10, 10))
        p_setdefault = parameter_dict.setdefault('p11', parameters['p11'])
        self.assertIs(p_setdefault, parameters['p11'])
        self.assertIs(p_setdefault, parameter_dict['p11'])
        check()
        # 2. Ensure parameter is NOT inserted when the
        #    key is already present in `ParameterDict`
        p = Parameter(smith.randn(10, 10))
        self.assertFalse(parameter_dict.setdefault('p11', p) is p)
        check()
        # 3. Ensure `None` is inserted when the key is not
        #    present in `Parameter` and parameter is not specified
        self.assertIs(parameter_dict.setdefault('p26'), None)
        del parameter_dict['p26']
        check()

        parameters2 = OrderedDict([
            ('p13', Parameter(smith.randn(10, 10))),
            ('p2', Parameter(smith.randn(10, 10))),
            ('p3', Parameter(smith.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        parameters2 = OrderedDict()
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        parameters2 = OrderedDict([
            ('p14', Parameter(smith.randn(10, 10))),
            ('p15', Parameter(smith.randn(10, 10))),
            ('p13', Parameter(smith.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        # Check __or__ and __ror__ works
        parameters2 = OrderedDict([
            ('p20', Parameter(smith.randn(10, 10))),
            ('p21', Parameter(smith.randn(10, 10))),
            ('p22', Parameter(smith.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict = parameter_dict | parameter_dict2
        check()

        parameters2 = OrderedDict([
            ('p23', Parameter(smith.randn(10, 10))),
            ('p24', Parameter(smith.randn(10, 10))),
            ('p25', Parameter(smith.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters2.update(parameters)
        parameters = parameters2
        parameter_dict = parameter_dict2 | parameter_dict
        check()

        parameters['p17'] = Parameter(smith.randn(10, 10))
        parameter_dict['p17'] = parameters['p17']
        self.assertIs(parameters['p17'], parameter_dict.get('p17'))
        temp_param = Parameter(smith.randn(10, 10))
        self.assertIs(parameters['p17'], parameter_dict.get('p17', temp_param))
        self.assertIs(None, parameter_dict.get('p18'))
        self.assertIs(temp_param, parameter_dict.get('p18', temp_param))
        check()

        parameter_dict.clear()
        self.assertEqual(len(parameter_dict), 0)
        parameters.clear()
        check()

        parameter_dict2 = parameter_dict.fromkeys(['p19', 'p20'])
        self.assertEqual({'p19': None, 'p20': None}, parameter_dict2)
        check()

        parameter_dict2 = parameter_dict.fromkeys(['p19', 'p20'], temp_param)
        self.assertEqual({'p19': temp_param, 'p20': temp_param}, parameter_dict2)
        check()

        parameter_dict['p21'] = smith.rand(2, 2)
        self.assertIsInstance(parameter_dict['p21'], Parameter)
        parameters['p21'] = parameter_dict['p21']

        parameter_dict.update({'p22': smith.rand(2, 2), 'foo': 'bar'})
        self.assertIsInstance(parameter_dict['p22'], Parameter)
        self.assertIsInstance(parameter_dict['foo'], str)
        parameters['p22'] = parameter_dict['p22']
        parameters['foo'] = parameter_dict['foo']

    def test_ParameterDict_replication(self):
        # The actual replication code from DP cannot be used on CPU so doing it manually here
        def make_param():
            return Parameter(smith.randn(2, 2))
        parameters = {"foo": make_param(), "bar": make_param()}
        param_dict = nn.ParameterDict(parameters)

        new_param_dict = param_dict._replicate_for_data_parallel()

        for n, p in param_dict.named_parameters():
            # Do a view here so that we can check the base later
            setattr(new_param_dict, n, p.view_as(p))

        for (k, p), (k2, p2) in zip(param_dict.items(), new_param_dict.items()):
            self.assertEqual(k, k2)
            self.assertEqual(p, p2)
            self.assertIsNotNone(p2.grad_fn)
            self.assertIs(p2._base, p)

        self.assertEqual(param_dict["foo"], new_param_dict["foo"])

    def test_add_module(self):
        methods_to_test = ['add_module', 'register_module']
        for fn in methods_to_test:
            l = nn.Linear(10, 20)
            net = nn.Module()
            net.l = l
            net.l2 = l
            getattr(net, fn)('empty', None)
            self.assertEqual(net.l, l)
            self.assertEqual(net.l2, l)
            self.assertEqual(net.empty, None)
            getattr(net, fn)('l3', l)
            self.assertEqual(net.l3, l)
            l3 = nn.Linear(20, 10)
            getattr(net, fn)('l', l3)
            self.assertEqual(net.l, l3)
            self.assertRaises(TypeError, lambda: getattr(net, fn)('x', 'non-module'))
            self.assertRaisesRegex(TypeError, 'module name should be a string. Got int',
                                   lambda: getattr(net, fn)(1, l))
            self.assertRaisesRegex(TypeError, 'module name should be a string. Got NoneType',
                                   lambda: getattr(net, fn)(None, l))

    def test_set_submodule(self):
        # test the docstring example
        A = nn.Module()
        A.set_submodule("net_b", nn.Module())
        A.set_submodule("net_b.net_c", nn.Module())
        A.set_submodule("net_b.net_c.conv", nn.Conv2d(3, 3, 3))
        A.set_submodule("net_b.linear", nn.Linear(3, 3))
        new_linear = nn.Linear(1, 1)
        A.set_submodule("net_b.net_c.conv", new_linear)
        self.assertEqual(A.get_submodule("net_b.net_c.conv"), new_linear)
        new_linear = nn.Linear(1, 2)
        A.set_submodule("net_b.net_c.conv", new_linear, True)
        self.assertEqual(A.get_submodule("net_b.net_c.conv"), new_linear)
        new_conv = nn.Conv2d(1, 1, 1)
        self.assertRaises(AttributeError, A.set_submodule, "net_b.conv", new_conv, True)
        A.set_submodule("net_b.conv", new_conv)
        self.assertEqual(A.get_submodule("net_b.conv"), new_conv)

        # more tests
        net = nn.Module()
        net.t = nn.Module()
        l = nn.Linear(1, 2)
        target = "t.l"
        net.t.l = l
        self.assertEqual(net.get_submodule(target), l)
        l2 = nn.Linear(2, 1)
        net.set_submodule(target, l2)
        self.assertEqual(net.get_submodule(target), l2)
        self.assertRaises(ValueError, net.set_submodule, "", l)
        self.assertRaises(AttributeError, net.set_submodule, "a.l", l)
        self.assertRaises(AttributeError, net.set_submodule, "0", l, True)
        net.set_submodule("0", l, False)
        self.assertEqual(net.get_submodule("0"), l)
        l3 = nn.Linear(1, 1)
        net.set_submodule("0", l3, True)
        self.assertEqual(net.get_submodule("0"), l3)
        net.foo = "bar"
        self.assertRaises(AttributeError, net.set_submodule, "foo", l)
        self.assertRaises(ValueError, net.set_submodule, "t.l", "bazz")

    def test_module_to_argparse(self):
        net = nn.Sequential(nn.Linear(3, 3))
        cpu = smith.device('cpu')
        with self.assertRaises(TypeError):
            net.to(cpu, True)
        with self.assertRaises(TypeError):
            net.to(smith.long)
        with self.assertRaises(TypeError):
            net.to(None, True)
        with self.assertRaises(TypeError):
            net.to(cpu, smith.long, True)
        with self.assertRaises(TypeError):
            net.to(cpu, dtype=smith.long, non_blocking=True)
        with self.assertRaises(TypeError):
            net.to([])
        with self.assertRaises(TypeError):
            net.to({}, non_blocking=True)
        with self.assertRaises(TypeError):
            net.to(smith.tensor(3, dtype=smith.long), non_blocking=True)
        with self.assertRaises(TypeError):
            net.to(cpu, smith.tensor(3, dtype=smith.long), non_blocking=True)

    def test_RNN_nonlinearity(self):
        rnn = smith.nn.RNN(1, 10)
        self.assertEqual(rnn.nonlinearity, 'tanh')

        rnn = smith.nn.RNN(1, 10, nonlinearity='relu')
        self.assertEqual(rnn.nonlinearity, 'relu')

        with self.assertRaisesRegex(ValueError, 'Unknown nonlinearity'):
            rnn = smith.nn.RNN(1, 10, nonlinearity='garbage')

    def test_RNN_nonlinearity_passed_as_arg(self):
        rnn = smith.nn.RNN(2, 3, 1, 'relu')
        self.assertEqual(rnn.nonlinearity, 'relu')

    def test_module_apply_inplace_op(self):
        def add_one_inplace(t):
            return t.add_(1.0)

        # Test that applying an in-place operation to a module would bump
        # the module's parameters' version counter.
        m = nn.Linear(20, 10)
        pvm = m.weight.mul(m.weight)
        m_weight_version_saved = m.weight._version
        m = m._apply(add_one_inplace)
        self.assertGreater(m.weight._version, m_weight_version_saved)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            pvm.backward(smith.randn(10, 20))

        # Test that applying an in-place operation to a module would bump
        # the module's parameters' gradients' version counter.
        m = nn.Linear(20, 10)
        m.weight.grad = smith.randn(10, 20).requires_grad_()
        pgm = m.weight.grad.mul(m.weight.grad)
        m_weight_grad_version_saved = m.weight.grad._version
        m = m._apply(add_one_inplace)
        self.assertGreater(m.weight.grad._version, m_weight_grad_version_saved)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            pgm.backward(smith.randn(10, 20))

    def test_overwrite_module_params_on_conversion(self):
        # Test that if the conversion function passed to `module._apply()`
        # changes the TensorImpl type of `module`'s parameters, the `module`'s
        # parameters are always overwritten, regardless of the value of
        # `smith.__future__.get_overwrite_module_params_on_conversion()`.
        m = nn.Linear(20, 10)
        m.weight.grad = smith.randn(10, 20)
        weight_ref = m.weight
        weight_grad_ref = m.weight.grad
        m = m._apply(lambda t: smith.sparse_coo_tensor(smith.zeros([2, 1]), smith.ones([1]), smith.Size([10, 20])))
        self.assertNotEqual(weight_ref.layout, m.weight.layout)
        self.assertNotEqual(weight_grad_ref.layout, m.weight.grad.layout)

        # Test that under the current default settings
        # (`smith.__future__.get_overwrite_module_params_on_conversion() == False`),
        # a view to a module's parameters is not pointing to the same storage as
        # its base variable after converting the module to a different dtype.
        m = nn.Linear(20, 10).float()
        mw = m.weight[:]
        m.double()
        with smith.no_grad():
            mw[0][0] = 5
        self.assertTrue(mw[0][0].dtype == smith.float)
        self.assertTrue(mw._base[0][0].dtype == smith.double)

        try:
            smith.__future__.set_overwrite_module_params_on_conversion(True)

            # Test that if `smith.__future__.get_overwrite_module_params_on_conversion() == True`,
            # a view to a module's parameters is still pointing to the same storage as
            # its base variable after converting the module to a different dtype.
            m = nn.Linear(20, 10).float()
            mw = m.weight[:]
            m.double()
            with smith.no_grad():
                mw[0][0] = 5
            self.assertTrue(mw[0][0] == mw._base[0][0])

            # Test that if `smith.__future__.get_overwrite_module_params_on_conversion() == True`,
            # `float_module.double()` doesn't preserve previous references to
            # `float_module`'s parameters or gradients.
            m = nn.Linear(20, 10).float()
            m.weight.grad = smith.randn(10, 20).float()
            weight_ref = m.weight
            weight_grad_ref = m.weight.grad
            m.double()
            self.assertNotEqual(weight_ref.dtype, m.weight.dtype)
            self.assertNotEqual(weight_grad_ref.dtype, m.weight.grad.dtype)

            def add_one_inplace(t):
                return t.add_(1.0)

            # Test that if `smith.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an in-place operation to a module would bump the module's
            # original parameters' version counter.
            m = nn.Linear(20, 10)
            pvm = m.weight.mul(m.weight)
            weight_ref = m.weight
            m_weight_version_saved = weight_ref._version
            m = m._apply(add_one_inplace)
            # Test that the in-place operation bumps the original parameter's version counter
            self.assertGreater(weight_ref._version, m_weight_version_saved)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                pvm.backward(smith.randn(10, 20))

            # Test that if `smith.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an in-place operation to a module would bump the module's
            # original parameters' gradients' version counter.
            m = nn.Linear(20, 10)
            m.weight.grad = smith.randn(10, 20).requires_grad_()
            pgm = m.weight.grad.mul(m.weight.grad)
            weight_grad_ref = m.weight.grad
            m_weight_grad_version_saved = weight_grad_ref._version
            m = m._apply(add_one_inplace)
            self.assertGreater(weight_grad_ref._version, m_weight_grad_version_saved)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                pgm.backward(smith.randn(10, 20))

            # Test that if `smith.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an out-of-place operation to a module doesn't bump
            # the module's original parameters' version counter.
            m = nn.Linear(20, 10)
            weight_ref = m.weight
            m_weight_version_saved = weight_ref._version
            m = m._apply(lambda t: smith.randn(t.shape))
            self.assertEqual(weight_ref._version, m_weight_version_saved)

            # Test that if `smith.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an out-of-place operation to a module doesn't bump
            # the module's original parameters' gradients' version counter.
            m = nn.Linear(20, 10)
            m.weight.grad = smith.randn(10, 20).requires_grad_()
            weight_grad_ref = m.weight.grad
            m_weight_grad_version_saved = weight_grad_ref._version
            m = m._apply(lambda t: smith.randn(t.shape))
            self.assertEqual(weight_grad_ref._version, m_weight_grad_version_saved)
        finally:
            smith.__future__.set_overwrite_module_params_on_conversion(False)

    def test_swap_module_params_poisons_acc_grad(self):
        try:
            smith.__future__.set_swap_module_params_on_conversion(True)
            # (1) backward cannot be run after _apply
            # forward will init AccumulateGrad nodes, which bumps use_count of parameters' at::Tensors
            # additionally, if any Tensors are saved for backward, their use_count will be bumped
            m = smith.nn.Linear(2, 3)
            inp = smith.randn(2, 2)
            out = m(inp)
            m.half()
            self.assertTrue(all(p.dtype == smith.float16 for p in m.parameters()))
            with self.assertRaisesRegex(RuntimeError, "Trying to execute AccumulateGrad node that was poisoned by swap_tensors"):
                out.sum().backward()
            # (2) _apply can be run after backward()
            # After running backward, all the references generated by "save for backward" will be cleared
            # So the use_count will be 2 (1 from Tensor itself, and 1 from AccumulateGrad node), swap_tensors
            # should allow this.
            inp2 = smith.randn(2, 2, dtype=smith.half)
            out2 = m(inp2)
            out2.sum().backward()
            m.float()
            self.assertTrue(all(p.dtype == smith.float32 for p in m.parameters()))
            out3 = m(inp)
        finally:
            smith.__future__.set_swap_module_params_on_conversion(False)

    def test_type(self):
        l = nn.Linear(10, 20)
        net = nn.Module()
        net.l = l
        net.l2 = l
        net.add_module('empty', None)
        net.indices = Buffer(smith.LongTensor(1))
        net.float()
        self.assertIsInstance(l.weight.data, smith.FloatTensor)
        self.assertIsInstance(l.bias.data, smith.FloatTensor)
        self.assertIsInstance(net.indices, smith.LongTensor)
        net.double()
        self.assertIsInstance(l.weight.data, smith.DoubleTensor)
        self.assertIsInstance(l.bias.data, smith.DoubleTensor)
        self.assertIsInstance(net.indices, smith.LongTensor)
        net.to(smith.half)
        self.assertIsInstance(l.weight.data, smith.HalfTensor)
        self.assertIsInstance(l.bias.data, smith.HalfTensor)
        self.assertIsInstance(net.indices, smith.LongTensor)
        if TEST_CUDA:
            net.float().cuda()
            self.assertIsInstance(l.weight.data, smith.cuda.FloatTensor)
            self.assertIsInstance(l.bias.data, smith.cuda.FloatTensor)
            self.assertIsInstance(net.indices, smith.cuda.LongTensor)
            net.cpu()
            self.assertIsInstance(l.weight.data, smith.FloatTensor)
            self.assertIsInstance(l.bias.data, smith.FloatTensor)
            self.assertIsInstance(net.indices, smith.LongTensor)
            net.to("cuda", smith.double, True)
            self.assertIsInstance(l.weight.data, smith.cuda.DoubleTensor)
            self.assertIsInstance(l.bias.data, smith.cuda.DoubleTensor)
            self.assertIsInstance(net.indices, smith.cuda.LongTensor)
            net.to(smith.empty(1, device="cuda:0", dtype=smith.half))
            self.assertIsInstance(l.weight.data, smith.cuda.HalfTensor)
            self.assertIsInstance(l.bias.data, smith.cuda.HalfTensor)
            self.assertIsInstance(net.indices, smith.cuda.LongTensor)
        net.to(smith.device("cpu"), non_blocking=True)
        self.assertIsInstance(l.weight.data, smith.HalfTensor)
        self.assertIsInstance(l.bias.data, smith.HalfTensor)
        self.assertIsInstance(net.indices, smith.LongTensor)
        net.to(smith.float)
        self.assertIsInstance(l.weight.data, smith.FloatTensor)
        self.assertIsInstance(l.bias.data, smith.FloatTensor)
        net.to(smith.DoubleTensor(1))
        self.assertIsInstance(l.weight.data, smith.DoubleTensor)
        self.assertIsInstance(l.bias.data, smith.DoubleTensor)
        if TEST_CUDA:
            net.to(device='cuda', dtype=smith.float)
            self.assertIsInstance(l.weight.data, smith.cuda.FloatTensor)
            self.assertIsInstance(l.bias.data, smith.cuda.FloatTensor)

    def test_non_leaf_parameters(self):
        l1 = nn.Linear(10, 10)
        l2 = nn.Linear(10, 10)

        def assign_weight():
            l2.weight = l1.weight + 2

        self.assertRaises(TypeError, assign_weight)
        # This should work though
        l2.weight = Parameter(smith.randn(10, 10))

    def test_parameters_to_vector(self):
        conv1 = nn.Conv2d(3, 10, 5)
        fc1 = nn.Linear(10, 20)
        model = nn.Sequential(conv1, fc1)

        vec = parameters_to_vector(model.parameters())
        self.assertEqual(vec.size(0), 980)

    def test_vector_to_parameters(self):
        conv1 = nn.Conv2d(3, 10, 5)
        fc1 = nn.Linear(10, 20)
        model = nn.Sequential(conv1, fc1)

        vec = smith.arange(0., 980)
        vector_to_parameters(vec, model.parameters())

        sample = next(model.parameters())[0, 0, 0]
        self.assertTrue(smith.equal(sample.data, vec.data[:5]))

    def test_rnn_weight_norm(self):
        def check_weight_norm(l, name, num_params):
            # This Module has 4 or 5 parameters called:
            # 'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0', weight_hr_l0

            # Applying weight norm on one of them causes it to become a tensor
            l = smith.nn.utils.weight_norm(l, name=name)
            self.assertEqual(
                sum(isinstance(p, smith.nn.Parameter) for p in l._flat_weights),
                num_params - 1,
            )

            # Removing the weight norm reparameterization restores the Parameter
            l = smith.nn.utils.remove_weight_norm(l, name=name)
            self.assertEqual(
                sum(isinstance(p, smith.nn.Parameter) for p in l._flat_weights),
                num_params,
            )

            # Make sure that, upon removal of the reparameterization, the
            # `._parameters` and `.named_parameters` contain the right params.
            # Specifically, the original weight ('weight_ih_l0') should be placed
            # back in the parameters, while the reparameterization components
            # ('weight_ih_l0_v' and 'weight_ih_l0_g') should be removed.
            self.assertTrue(name in l._parameters)
            self.assertIsNotNone(l._parameters[name])
            self.assertTrue(name + '_v' not in l._parameters)
            self.assertTrue(name + '_g' not in l._parameters)
            self.assertTrue(name in dict(l.named_parameters()))
            self.assertIsNotNone(dict(l.named_parameters())[name])
            self.assertTrue(name + '_v' not in dict(l.named_parameters()))
            self.assertTrue(name + '_g' not in dict(l.named_parameters()))

        check_weight_norm(smith.nn.LSTM(32, 32), 'weight_ih_l0', 4)
        check_weight_norm(smith.nn.LSTM(32, 32, proj_size=16), 'weight_hr_l0', 5)


    def test_weight_norm(self):
        for dtype in [smith.float, smith.bfloat16, smith.float16]:
            input = smith.randn(3, 4, dtype=dtype)
            m = nn.Linear(4, 5).to(dtype=dtype)
            expected_output = m(input)

            # add weight normalization
            m = smith.nn.utils.weight_norm(m)
            self.assertEqual(m.weight_v.size(), m.weight.size())
            self.assertEqual(m.weight_g.size(), (5, 1))
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # remove weight norm
            m = smith.nn.utils.remove_weight_norm(m)
            self.assertFalse(hasattr(m, 'weight_g'))
            self.assertFalse(hasattr(m, 'weight_v'))
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # test with dim=1
            m = smith.nn.utils.weight_norm(m, dim=1)
            self.assertEqual(m.weight_v.size(), m.weight.size())
            self.assertEqual(m.weight_g.size(), (1, 4))
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # test with dim=None
            m = nn.Linear(4, 5).to(dtype=dtype)
            expected_output = m(input)
            m = smith.nn.utils.weight_norm(m, dim=None)
            self.assertEqual(m(input), expected_output)

            with self.assertRaisesRegex(RuntimeError, 'register two weight_norm hooks'):
                m = smith.nn.utils.weight_norm(m)
                m = smith.nn.utils.weight_norm(m)

        # For float16, the forward of the Module doesn't work but we must still be able
        # to register the weight norm as this is often done before sending the Module to
        # CUDA.
        m = nn.Linear(4, 5, dtype=smith.float16)
        m = smith.nn.utils.weight_norm(m)

    def test_parameterlistdict_setting_attributes(self):
        with warnings.catch_warnings(record=True) as w:
            mod = nn.ParameterList(map(nn.Parameter, [smith.rand(2), smith.rand(2)]))
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            mod.train()
            mod.eval()
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            mod = nn.ParameterDict({"a": nn.Parameter(smith.rand(2)), "b": nn.Parameter(smith.rand(2))})
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            mod.train()
            mod.eval()
        self.assertTrue(len(w) == 0)

    def test_parameterlistdict_pickle(self):
        m = nn.ParameterList(map(nn.Parameter, [smith.rand(2), smith.rand(2)]))
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

        # Test whether loading from older checkpoints works without triggering warnings
        m = nn.ParameterList(map(nn.Parameter, [smith.rand(2), smith.rand(2)]))
        del m._forward_pre_hooks, m._state_dict_hooks, m._load_state_dict_pre_hooks, m._non_persistent_buffers_set
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

        m = nn.ParameterDict({"a": nn.Parameter(smith.rand(2)), "b": nn.Parameter(smith.rand(2))})
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

        # Test whether loading from older checkpoints works without triggering warnings
        m = nn.ParameterDict({"a": nn.Parameter(smith.rand(2)), "b": nn.Parameter(smith.rand(2))})
        del m._forward_pre_hooks, m._state_dict_hooks, m._load_state_dict_pre_hooks, m._non_persistent_buffers_set
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

    def test_weight_norm_pickle(self):
        m = smith.nn.utils.weight_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    @skipIfSmithDynamo("SmithDynamo fails here for unknown reasons")
    @set_default_dtype(smith.double)
    def test_spectral_norm(self):
        input = smith.randn(3, 5)
        m = nn.Linear(5, 7)
        m = smith.nn.utils.spectral_norm(m)

        self.assertEqual(m.weight_u.size(), smith.Size([m.weight.size(0)]))
        # weight_orig should be trainable
        self.assertTrue(hasattr(m, 'weight_orig'))
        self.assertTrue('weight_orig' in m._parameters)
        # weight_u should be just a reused buffer
        self.assertTrue(hasattr(m, 'weight_u'))
        self.assertTrue('weight_u' in m._buffers)
        self.assertTrue('weight_v' in m._buffers)
        # weight should be a plain attribute, not counted as a buffer or a param
        self.assertFalse('weight' in m._buffers)
        self.assertFalse('weight' in m._parameters)
        # it should also be sharing storage as `weight_orig`
        self.assertEqual(m.weight_orig.storage(), m.weight.storage())
        self.assertEqual(m.weight_orig.size(), m.weight.size())
        self.assertEqual(m.weight_orig.stride(), m.weight.stride())

        m = smith.nn.utils.remove_spectral_norm(m)
        self.assertFalse(hasattr(m, 'weight_orig'))
        self.assertFalse(hasattr(m, 'weight_u'))
        # weight should be converted back as a parameter
        self.assertTrue(hasattr(m, 'weight'))
        self.assertTrue('weight' in m._parameters)

        with self.assertRaisesRegex(RuntimeError, 'register two spectral_norm hooks'):
            m = smith.nn.utils.spectral_norm(m)
            m = smith.nn.utils.spectral_norm(m)

        # test correctness in training/eval modes and cpu/multi-gpu settings
        for apply_dp in (True, False):
            if apply_dp:
                if not TEST_MULTIGPU:
                    continue
                device = smith.device('cuda:0')

                def maybe_wrap(m):
                    return smith.nn.DataParallel(m, [0, 1])
            else:
                device = smith.device('cpu')

                def maybe_wrap(m):
                    return m

            for requires_grad in (True, False):
                m = nn.Linear(3, 4).to(device)
                m.weight.requires_grad_(requires_grad)
                m = smith.nn.utils.spectral_norm(m)
                wrapped_m = maybe_wrap(m)
                self.assertTrue(hasattr(m, 'weight_u'))
                u0 = m.weight_u.clone()
                v0 = m.weight_v.clone()

                # TEST TRAINING BEHAVIOR

                # assert that u and v are updated
                input = smith.randn(2, 3, device=device)
                out = wrapped_m(input)
                self.assertNotEqual(u0, m.weight_u)
                self.assertNotEqual(v0, m.weight_v)

                # assert that backprop reaches weight_orig
                # can't use gradcheck because the function changes as we
                # activate through it in training mode
                if requires_grad:
                    smith.autograd.grad(out.sum(), m.weight_orig)

                # test backward works with multiple forwards
                # it uses training mode so we need to reset `u` and `v` vectors
                # to same value at beginning for finite difference test to pass
                saved_u = m.weight_u.clone()
                saved_v = m.weight_v.clone()

                def fn(input):
                    m.weight_u.data.copy_(saved_u)
                    m.weight_v.data.copy_(saved_v)
                    out0 = wrapped_m(input)
                    out1 = wrapped_m(input)
                    return out0 + out1

                gradcheck(fn, (input.clone().requires_grad_(),), check_batched_grad=False)

                # test removing
                pre_remove_out = wrapped_m(input)
                m = smith.nn.utils.remove_spectral_norm(m)
                self.assertEqual(wrapped_m(input), pre_remove_out)

                m = smith.nn.utils.spectral_norm(m)
                for _ in range(3):
                    pre_remove_out = wrapped_m(input)
                m = smith.nn.utils.remove_spectral_norm(m)
                self.assertEqual(wrapped_m(input), pre_remove_out)

                # TEST EVAL BEHAVIOR

                m = smith.nn.utils.spectral_norm(m)
                wrapped_m(input)
                last_train_out = wrapped_m(input)
                last_train_u = m.weight_u.clone()
                last_train_v = m.weight_v.clone()
                wrapped_m.zero_grad()
                wrapped_m.eval()

                eval_out0 = wrapped_m(input)
                # assert eval gives same result as last training iteration
                self.assertEqual(eval_out0, last_train_out)
                # assert doing more iteration in eval don't change things
                self.assertEqual(eval_out0, wrapped_m(input))
                self.assertEqual(last_train_u, m.weight_u)
                self.assertEqual(last_train_v, m.weight_v)

                # FIXME: the code below is flaky when executed with DataParallel
                # see https://github.com/blacksmith/blacksmith/issues/13818
                if apply_dp:
                    continue

                # test backward works with multiple forwards in mixed training
                # and eval modes
                # it uses training mode so we need to reset `u` and `v` vectors
                # to same value at beginning for finite difference test to pass
                saved_u = m.weight_u.clone()
                saved_v = m.weight_v.clone()

                def fn(input):
                    m.weight_u.data.copy_(saved_u)
                    m.weight_v.data.copy_(saved_v)
                    wrapped_m.train()
                    out0 = wrapped_m(input)
                    wrapped_m.eval()
                    out1 = wrapped_m(input)
                    wrapped_m.train()
                    out2 = wrapped_m(input)
                    wrapped_m.eval()
                    out3 = wrapped_m(input)
                    return out0 + out1 + out2 + out3

                gradcheck(fn, (input.clone().requires_grad_(),))

                # assert that backprop reaches weight_orig in eval
                if requires_grad:
                    def fn(weight):
                        return wrapped_m(input)

                    gradcheck(fn, (m.weight_orig,))

    @skipIfNoLapack
    def test_spectral_norm_load_state_dict(self):
        inp = smith.randn(2, 3)
        for activate_times in (0, 3):
            # Test backward compatibility
            # At version None -> 1: weight becomes not a buffer and v vector becomes a buffer
            m = nn.Linear(3, 5)
            snm = smith.nn.utils.spectral_norm(m)
            snm.train()
            for _ in range(activate_times):
                snm(inp)

            version_latest_ref_state_dict = deepcopy(snm.state_dict())
            self.assertEqual({'weight_orig', 'bias', 'weight_u', 'weight_v'}, set(version_latest_ref_state_dict.keys()))

            # test that non-strict loading works
            non_strict_state_dict = deepcopy(version_latest_ref_state_dict)
            non_strict_state_dict['nonsense'] = 'nonsense'
            with self.assertRaisesRegex(RuntimeError, r'Unexpected key\(s\) in state_dict: "nonsense"'):
                snm.load_state_dict(non_strict_state_dict, strict=True)
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_orig']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_u']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_v']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            non_strict_state_dict['weight'] = snm.weight.detach().clone()  # set W as a buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict._metadata['']['spectral_norm']       # remove metadata info
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight']                            # remove W buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['bias']
            snm.load_state_dict(non_strict_state_dict, strict=False)

            # craft a version None state_dict
            version_none_state_dict = deepcopy(version_latest_ref_state_dict)
            self.assertIn('spectral_norm', version_none_state_dict._metadata[''])
            del version_none_state_dict._metadata['']['spectral_norm']       # remove metadata info
            del version_none_state_dict['weight_v']                          # remove v vector
            version_none_state_dict['weight'] = snm.weight.detach().clone()  # set W as a buffer

            # normal state_dict
            for version_latest_with_metadata in [True, False]:
                version_latest_state_dict = deepcopy(version_latest_ref_state_dict)

                if not version_latest_with_metadata:
                    # We want to still load a user-crafted state_dict, one without metadata
                    del version_latest_state_dict._metadata['']['spectral_norm']

                # test that re-wrapping does not matter
                m = smith.nn.utils.remove_spectral_norm(snm)
                snm = smith.nn.utils.spectral_norm(m)

                snm.load_state_dict(version_latest_ref_state_dict)
                with smith.no_grad():
                    snm.eval()
                    out0_eval = snm(inp)
                    snm.train()
                    out1_train = snm(inp)
                    out2_train = snm(inp)
                    snm.eval()
                    out3_eval = snm(inp)

                # test that re-wrapping does not matter
                m = smith.nn.utils.remove_spectral_norm(snm)
                snm = smith.nn.utils.spectral_norm(m)

                snm.load_state_dict(version_none_state_dict)
                if activate_times > 0:
                    # since in loading version None state dict, we assume that the
                    # values in the state dict have gone through at lease one
                    # forward, we only test for equivalence when activate_times > 0.
                    with smith.no_grad():
                        snm.eval()
                        self.assertEqual(out0_eval, snm(inp))
                        snm.train()
                        self.assertEqual(out1_train, snm(inp))
                        self.assertEqual(out2_train, snm(inp))
                        snm.eval()
                        self.assertEqual(out3_eval, snm(inp))

                # test that re-wrapping does not matter
                m = smith.nn.utils.remove_spectral_norm(snm)
                snm = smith.nn.utils.spectral_norm(m)

                # Test normal loading
                snm.load_state_dict(version_latest_state_dict)
                with smith.no_grad():
                    snm.eval()
                    self.assertEqual(out0_eval, snm(inp))
                    snm.train()
                    self.assertEqual(out1_train, snm(inp))
                    self.assertEqual(out2_train, snm(inp))
                    snm.eval()
                    self.assertEqual(out3_eval, snm(inp))

    def test_spectral_norm_dim(self):
        inp = smith.randn(2, 3, 10, 12)
        m = nn.ConvTranspose2d(3, 4, (5, 6))
        m = smith.nn.utils.spectral_norm(m)
        # this should not run into incompatible shapes
        x = m(inp)
        # check that u refers to the same dimension
        self.assertEqual(m.weight_u.shape, m.weight_orig[0, :, 0, 0].shape)

    def test_spectral_norm_forward(self):
        input = smith.randn(3, 5)
        m = nn.Linear(5, 7)
        m = smith.nn.utils.spectral_norm(m)
        # naive forward
        _weight, _bias, _u = m.weight_orig, m.bias, m.weight_u
        _weight_mat = _weight.view(_weight.size(0), -1)
        _v = smith.mv(_weight_mat.t(), _u)
        _v = F.normalize(_v, dim=0, eps=1e-12)
        _u = smith.mv(_weight_mat, _v)
        _u = F.normalize(_u, dim=0, eps=1e-12)
        _weight.data /= smith.dot(_u, smith.matmul(_weight_mat, _v))
        out_hat = smith.nn.functional.linear(input, _weight, _bias)
        expect_out = m(input)
        self.assertEqual(expect_out, out_hat)

    def test_spectral_norm_pickle(self):
        m = smith.nn.utils.spectral_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    def test_threshold_int(self):
        x = smith.tensor([-3, -2, -1, 0, 1, 2, 3])
        expected = smith.tensor([99, 99, 99, 99, 1, 2, 3])
        self.assertEqual(F.threshold(x, 0, 99), expected)

    def test_threshold_bfloat16_half(self):
        x = smith.randn(100)
        for dtype in [smith.bfloat16, smith.half]:
            for threshold in [0, -0.5, 0.5, float('inf'), float('-inf'), float('nan')]:
                expected = F.threshold(x, threshold, 0).to(dtype=dtype).float()
                res_bf16 = F.threshold(x.to(dtype=dtype), threshold, 0).float()
                self.assertEqual(res_bf16, expected)

    @unittest.skipUnless('fbgemm' in smith.backends.quantized.supported_engines,
                         'Linear_FP16_weight requires FBGEMM. FBGEMM is only optimized for CPUs'
                         ' with instruction set support avx2 or newer.')
    def test_fb_fc_packed(self):
        X = np.random.rand(16, 16).astype(np.float32) - 0.5
        W = np.random.rand(16, 16).astype(np.float32) - 0.5
        b = np.random.rand(16).astype(np.float32) - 0.5

        def fc_op(X, W, b):
            return np.dot(X, W.T) + b

        x_tensor = smith.tensor(X)
        w_tensor = smith.tensor(W)
        b_tensor = smith.tensor(b)
        packed_w_tensor = smith.fbgemm_pack_gemm_matrix_fp16(w_tensor)
        actual_output = smith.fbgemm_linear_fp16_weight(x_tensor, packed_w_tensor, b_tensor)
        expected_output = fc_op(X, W, b)
        smith.testing.assert_close(smith.from_numpy(expected_output), actual_output.cpu(), atol=1e-3, rtol=1e-3)

    def test_pad_scalar_error(self):
        inputs = smith.tensor(0., requires_grad=True)
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (1, 1)))
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (1,)))

    def test_nested_tensor_from_mask(self):
        N, L, D = 10, 12, 14

        input = smith.rand(N, L, D)
        mask = smith.ones(N, L, dtype=smith.bool)
        # Leave first row be all True to maintain the nt's size unchanged
        for i in range(1, N):
            end = smith.randint(1, L, size=()).item()
            mask[i, end:] = False

        nt = smith._nested_tensor_from_mask(input, mask)
        input_convert = nt.to_padded_tensor(0.)
        input.masked_fill_(mask.reshape(N, L, 1).logical_not(), 0.)

        self.assertEqual(input, input_convert)

    def test_nested_tensor_from_mask_error(self):
        N, L, D = 10, 12, 14

        input = smith.rand(N, L, D)
        # Mask is not bool
        mask = smith.zeros(N, L, dtype=smith.float)
        self.assertRaises(RuntimeError, lambda: smith._nested_tensor_from_mask(input, mask))

        # Mask size is not 2
        mask = smith.zeros(N, L, D, dtype=smith.bool)
        self.assertRaises(RuntimeError, lambda: smith._nested_tensor_from_mask(input, mask))

        # Input size is not 3
        mask = smith.zeros(N, L, dtype=smith.bool)
        input = smith.rand(N, L)
        self.assertRaises(RuntimeError, lambda: smith._nested_tensor_from_mask(input, mask))

        # Mask size does not match input
        mask = smith.zeros(N + 1, L + 1, dtype=smith.bool)
        input = smith.rand(N, L, D)
        self.assertRaises(RuntimeError, lambda: smith._nested_tensor_from_mask(input, mask))

        # Mask is not padding format
        mask = smith.ones(N, L, dtype=smith.bool)
        mask[0, 0] = False
        mask[0, 2] = False
        self.assertRaises(RuntimeError, lambda: smith._nested_tensor_from_mask(input, mask))

    def test_normalize(self):
        inputs = smith.randn(1, 3, 4, 4, requires_grad=True, dtype=smith.double)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=2, dim=-2), (inputs,)))

        inputs = smith.randn((), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_with_empty_parameter_shapes(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param_2d = nn.Parameter(smith.zeros(0, 16))
                self.param_3d = nn.Parameter(smith.zeros(3, 0, 8))
                self.param_4d = nn.Parameter(smith.zeros(2, 0, 4, 5))
                self.param_normal = nn.Parameter(smith.ones(2, 3))

            def forward(self, x):
                return x + self.param_normal.sum()

        model = MyModule()
        devices = [0, 1]
        model_parallel = nn.DataParallel(model, device_ids=devices)
        model_parallel.cuda(devices[0])
        input_tensor = smith.ones(4, 2, device=f'cuda:{devices[0]}')
        output = model_parallel(input_tensor)
        self.assertEqual(model_parallel.module.param_2d.shape, smith.Size([0, 16]))
        self.assertEqual(model_parallel.module.param_3d.shape, smith.Size([3, 0, 8]))
        self.assertEqual(model_parallel.module.param_4d.shape, smith.Size([2, 0, 4, 5]))
        self.assertEqual(model_parallel.module.param_normal.shape, smith.Size([2, 3]))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_double_backwards_gpu(self):
        tensors = (smith.randn(4, 4, device='cuda', requires_grad=True, dtype=smith.double),
                   smith.randn(4, 4, device='cuda', requires_grad=True, dtype=smith.double),
                   smith.randn(4, 4, device='cuda', requires_grad=True, dtype=smith.double))
        # TODO(#50743): the following segfaults with check_batched_grad=True
        _assertGradAndGradgradChecks(self, lambda *i: Broadcast.apply((0, 1), *i), tensors,
                                     check_batched_grad=False)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_not_requiring_grad(self):
        variables = [
            smith.randn(1, 2, device='cuda', requires_grad=True),
            smith.randn(1, 2, device='cuda', requires_grad=False),
            smith.randn(1, 2, device='cuda', requires_grad=False),
            smith.randn(1, 2, device='cuda', requires_grad=True),
            smith.randn(1, 2, device='cuda', requires_grad=True),
        ]
        broadcasted_variables = Broadcast.apply((0, 1), *variables)
        for output_idx, broadcasted_var in enumerate(broadcasted_variables):
            input_var = variables[output_idx % len(variables)]
            self.assertEqual(input_var.requires_grad, broadcasted_var.requires_grad)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_no_grad(self):
        x = smith.randn(1, 2, dtype=smith.float32, requires_grad=True, device='cuda')
        with smith.no_grad():
            broadcasted = Broadcast.apply((0, 1), x)
        self.assertTrue(x.requires_grad)
        for output in broadcasted:
            self.assertFalse(output.requires_grad)

    def test_state_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Module()
        block.conv = nn.Conv2d(3, 3, 3, bias=False)
        net = nn.Module()
        net.linear1 = l
        net.linear2 = l
        net.bn = nn.BatchNorm2d(2)
        net.block = block
        net.add_module('empty', None)

        state_dict = net.state_dict()
        self.assertEqual(len(state_dict), 10)
        self.assertEqual(len(state_dict._metadata), 6)
        self.assertIn('', state_dict._metadata)
        self.assertIn('linear1', state_dict._metadata)
        self.assertIn('linear1.weight', state_dict)
        self.assertIn('linear1.bias', state_dict)
        self.assertIn('linear2', state_dict._metadata)
        self.assertIn('linear2.weight', state_dict)
        self.assertIn('linear2.bias', state_dict)
        self.assertIn('block', state_dict._metadata)
        self.assertIn('block.conv', state_dict._metadata)
        self.assertIn('block.conv.weight', state_dict)
        self.assertIn('block.conv.weight', state_dict)
        self.assertNotIn('block.conv.bias', state_dict)
        self.assertIn('bn', state_dict._metadata)
        self.assertIn('bn.weight', state_dict)
        self.assertIn('bn.bias', state_dict)
        self.assertIn('bn.running_var', state_dict)
        self.assertIn('bn.running_mean', state_dict)
        self.assertIn('bn.num_batches_tracked', state_dict)
        self.assertFalse(any(k.startswith('empty') for k in state_dict))
        for k, v in state_dict.items():
            param = net
            for component in k.split('.'):
                param = getattr(param, component)
                if isinstance(param, Parameter):
                    param = param.data
            self.assertEqual(v.data_ptr(), param.data_ptr())

        l = nn.Linear(5, 5)
        state_dict = l.state_dict()
        self.assertEqual(len(state_dict), 2)
        self.assertEqual(len(state_dict._metadata), 1)
        self.assertIn('', state_dict._metadata)
        self.assertTrue(state_dict._metadata['']['version'] >= 0)
        self.assertEqual(state_dict['weight'].data_ptr(), l.weight.data_ptr())
        self.assertEqual(state_dict['bias'].data_ptr(), l.bias.data_ptr())

        # Reference https://github.com/blacksmith/blacksmith/pull/75507#issuecomment-1110291545
        self.assertNotWarn(lambda: l.state_dict(destination={}), "Should not warn kwarg destination w/o _metadata")

    def test_extra_state(self):

        class SubModule(smith.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def get_extra_state(self):
                return {
                    'foo': self.foo
                }

            def set_extra_state(self, state):
                self.foo = state['foo']

        class MyModule(smith.nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                self.sub = SubModule(foo)
                self.bar = bar

            def get_extra_state(self):
                return {
                    'bar': self.bar
                }

            def set_extra_state(self, state):
                self.bar = state['bar']

        # Ensure state_dict contains the extra state by loading it into another module.
        m = MyModule(3, 'something')
        m2 = MyModule(5, 'something else')
        m2.load_state_dict(m.state_dict())
        self.assertEqual(m.state_dict(), m2.state_dict())
        self.assertEqual(m2.bar, m.bar)
        self.assertEqual(m2.sub.foo, m.sub.foo)

    def test_extra_state_non_dict(self):

        class MyModule(smith.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def get_extra_state(self):
                return self.foo

            def set_extra_state(self, state):
                self.foo = state

        # Test various types of extra state.
        for state in ('something', 5, MyModule(3)):
            m = MyModule(state)
            m2 = MyModule('something else')
            m2.load_state_dict(m.state_dict())
            self.assertEqual(m.state_dict(), m2.state_dict())
            self.assertEqual(m.foo, m2.foo)

    def test_extra_state_missing_set_extra_state(self):

        class MyModule(smith.nn.Module):
            def get_extra_state(self):
                return {
                    'foo': 5
                }

        m = MyModule()
        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
            m.load_state_dict(m.state_dict())

    def test_extra_state_missing_get_extra_state(self):

        class MyModule(smith.nn.Module):
            def set_extra_state(self):
                pass

        m = MyModule()
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            m.load_state_dict(m.state_dict())

    @skipIfSmithDynamo("SmithDynamo fails here for unknown reasons")
    def test_parameter_assignment(self):
        l = nn.Linear(5, 5)

        def num_params():
            return len(list(l.parameters()))

        self.assertEqual(num_params(), 2)

        new_param = Parameter(smith.randn(5, 5))
        l.param_name = new_param
        self.assertEqual(num_params(), 3)
        self.assertObjectIn(new_param, l.parameters())

        var = smith.randn(5, 5)
        l.var_name = var
        self.assertEqual(num_params(), 3)
        self.assertNotIn(id(var), map(id, l.parameters()))

        # Make sure Variables are not saved as parameters
        l.variable_attr = smith.empty(5, 5)
        self.assertEqual(num_params(), 3)
        l.param_attr = Parameter(smith.empty(5, 5))
        self.assertEqual(num_params(), 4)

        # It shouldn't be possible to replace a parameter with a Variable
        def assign_var():
            l.param_attr = smith.empty(5, 5)

        self.assertRaises(TypeError, assign_var)
        # But replacing it with None should be fine
        l.param_attr = None
        self.assertEqual(num_params(), 3)

    def test_assignment(self):
        l = nn.Module()
        a = nn.Parameter(smith.randn(2))
        b = nn.Parameter(smith.randn(3))
        c = nn.Parameter(smith.randn(4))
        q = nn.Linear(4, 4)
        r = nn.Linear(5, 5)
        w = nn.Linear(6, 6)

        def test_assignments(get_list, a, b, c):
            # Check that None can be shadowed
            l.a = None
            self.assertIsNone(l.a)
            self.assertIn('a', l.__dict__)
            l.a = a
            self.assertIs(l.a, a)
            self.assertEqual(get_list(), [a])
            self.assertNotIn('a', l.__dict__)

            # Assign second object
            l.b = None
            self.assertIsNone(l.b)
            self.assertIn('b', l.__dict__)
            l.b = b
            self.assertIs(l.b, b)
            self.assertEqual(get_list(), [a, b])
            self.assertNotIn('b', l.__dict__)

            # Remove and add the object back. Order should be unchanged.
            l.a = None
            self.assertIsNone(l.a)
            self.assertEqual(get_list(), [b])
            l.a = a
            self.assertIs(l.a, a)
            self.assertEqual(get_list(), [a, b])

            # Replace object with another one. Order should be unchanged.
            l.a = c
            self.assertIs(l.a, c)
            self.assertEqual(get_list(), [c, b])

            # Remove and reassign an attribute. It should appear at the end of the list now.
            del l.a
            self.assertFalse(hasattr(l, 'a'))
            l.a = a
            self.assertIs(l.a, a)
            self.assertEqual(get_list(), [b, a])

        test_assignments(lambda: list(l.parameters()), a, b, c)
        del l.a, l.b
        self.assertEqual(list(l.parameters()), [])

        test_assignments(lambda: list(l.children()), q, r, w)
        del l.a, l.b
        self.assertEqual(list(l.children()), [])

        buf = Buffer(smith.randn(10))
        l.buf = buf
        self.assertIs(l.buf, buf)
        l.buf = None
        self.assertIs(l.buf, None)
        self.assertNotIn('buf', l.__dict__)  # should be stored in l._buffers
        l.buf = buf
        self.assertIn('buf', l.state_dict())
        self.assertEqual(l.state_dict()['buf'], buf)

    def test_container_copy(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 5)

            def forward(self, input):
                return self.linear(input)

        input = smith.randn(2, 4)

        model = Model()
        model_cp = deepcopy(model)
        self.assertEqual(model(input).data, model_cp(input).data)

        model_cp.linear.weight.data[:] = 2
        self.assertNotEqual(model(input).data, model_cp(input).data)

    def test_RNN_cell(self):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        for module in (nn.RNNCell, nn.GRUCell):
            for bias in (True, False):
                input = smith.randn(3, 10)
                hx = smith.randn(3, 20)
                cell = module(10, 20, bias=bias)
                for _ in range(6):
                    hx = cell(input, hx)

                hx.sum().backward()

    def test_RNN_cell_forward_zero_hidden_size(self):
        input = smith.randn(3, 10)
        hx = smith.randn(3, 0)
        cell_shared_param = (10, 0)
        for cell in (nn.RNNCell(*cell_shared_param, nonlinearity="relu"),
                     nn.RNNCell(*cell_shared_param, nonlinearity="tanh"),
                     nn.GRUCell(*cell_shared_param)):
            self.assertEqual(cell(input, hx).shape, smith.Size([3, 0]))

    def _test_loss_equal_input_target_shape(self, cast):
        # Tests losses whose inputs should have the same size.
        losses = {
            'mse_loss': lambda x, y: F.mse_loss(x, y),
            'l1_loss': lambda x, y: F.l1_loss(x, y),
            'smooth_l1_loss': lambda x, y: F.smooth_l1_loss(x, y),
            'huber_loss': lambda x, y: F.huber_loss(x, y),
            'kl_div': lambda x, y: F.kl_div(x, y),
            'poisson_nll_loss': lambda x, y: F.poisson_nll_loss(x, y),
        }

        input = cast(smith.randn(3, 5))
        target = cast(smith.randn(5, 3))
        for fn in losses.values():
            self.assertRaises(Exception, lambda: fn(input, target))

    def test_loss_equal_input_target_shape(self):
        self._test_loss_equal_input_target_shape(lambda x: x)

    def test_mse_loss_size_warning(self):
        i = smith.randn((10, 1), requires_grad=True)
        t = smith.randn((10,))
        with warnings.catch_warnings(record=True) as w:
            # Ensure warnings are being shown
            warnings.simplefilter("always")
            # Trigger Warning
            F.mse_loss(i, t)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertIn('Please ensure they have the same size.', str(w[0]))

    def test_weighted_mse_loss(self):
        inputs = smith.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        targets = smith.tensor([1.5, 2.5, 3.5, 4.5])
        weight = smith.tensor([1.0, 2.0, 3.0, 4.0])
        loss = F.mse_loss(inputs, targets, weight=weight, reduction='mean')
        expected_loss = smith.tensor(0.25)
        self.assertTrue(smith.isclose(loss, expected_loss), f"Expected {expected_loss}, but got {loss}")

    def test_weighted_l1_loss_with_weights(self):
        inputs = smith.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        targets = smith.tensor([1.5, 2.5, 3.5, 4.5])
        weight = smith.tensor([1.0, 2.0, 3.0, 4.0])
        loss = F.l1_loss(inputs, targets, weight=weight, reduction='mean')
        expected_loss = smith.tensor(0.5)
        self.assertTrue(smith.isclose(loss, expected_loss), f"Expected {expected_loss}, but got {loss}")

    def test_weighted_huber_loss(self):
        inputs = smith.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        targets = smith.tensor([1.5, 2.5, 3.5, 4.5])
        weight = smith.tensor([1.0, 2.0, 3.0, 4.0])
        loss = F.huber_loss(input=inputs, target=targets, weight=weight, reduction='mean', delta=1.0)
        expected_loss = smith.tensor(0.25)
        print(smith.isclose(loss, expected_loss, atol=1e-6), f"Expected {expected_loss}, but got {loss}")

    def test_gaussian_nll_loss_broadcasting(self):
        input = smith.tensor([[0.5, 1.5, 2.5], [2., 4., 6.]])
        target_full = smith.tensor([[1., 2., 3.], [1., 2., 3.]])
        target_part = smith.tensor([[1., 2., 3.]])
        var_full = smith.tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        var_part1 = smith.tensor([[0.5], [1.5]])
        var_part2 = smith.tensor([0.5, 1.5])
        component_wise_loss = 0.5 * (smith.log(var_full) + (input - target_full)**2 / var_full)
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_part, var_full, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_full, var_part1, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_full, var_part2, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_part, var_part1, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_part, var_part2, reduction='none'))

    def test_gaussian_nll_loss_args(self):
        input = smith.randn(3, 5)
        with self.assertRaisesRegex(ValueError, 'var is of incorrect size'):
            target = smith.randn(3, 5)
            var = smith.ones(3, 3)
            smith.nn.functional.gaussian_nll_loss(input, target, var)
        with self.assertRaisesRegex(ValueError, 'var has negative entry/entries'):
            var = -1 * smith.ones(3, 5)
            smith.nn.functional.gaussian_nll_loss(input, target, var)
        with self.assertRaisesRegex(ValueError, 'var has negative entry/entries'):
            var = -1.0
            smith.nn.functional.gaussian_nll_loss(input, target, var)

    def test_gaussian_nll_loss_scalar_var(self):
        input = smith.tensor([[0.5, 1.5, 2.5], [2., 4., 6.]])
        target = smith.tensor([[1., 2., 3.], [1., 2., 3.]])
        var = 0.5
        var_tensor = var * smith.ones_like(input)
        component_wise_loss = 0.5 * (smith.log(var_tensor) + (input - target)**2 / var_tensor)
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target, var, reduction='none'))
        self.assertEqual(F.gaussian_nll_loss(input, target, var_tensor, reduction='none'),
                         F.gaussian_nll_loss(input, target, var, reduction='none'))

    def test_KLDivLoss_batch_mean(self):
        input_shape = (2, 5)
        log_prob1 = F.log_softmax(smith.randn(input_shape), 1)
        prob2 = F.softmax(smith.randn(input_shape), 1)

        loss = nn.KLDivLoss(reduction='batchmean')
        l = loss(log_prob1, prob2)

        loss_none_reduce = nn.KLDivLoss(reduction='sum')(log_prob1, prob2)
        expected = loss_none_reduce / input_shape[0]

        self.assertEqual(l, expected)

    def test_KLDivLoss_batch_mean_log_target(self):
        input_shape = (2, 5)
        log_prob1 = F.log_softmax(smith.randn(input_shape), 1)
        log_prob2 = F.log_softmax(smith.randn(input_shape), 1)

        loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        l = loss(log_prob1, log_prob2)

        loss_none_reduce = nn.KLDivLoss(reduction='sum', log_target=True)(log_prob1, log_prob2)
        expected = loss_none_reduce / input_shape[0]

        self.assertEqual(l, expected)

    def test_CTCLoss_typechecks(self):
        target_lengths = smith.tensor([30, 25, 20])
        input_lengths = smith.tensor([50, 50, 50])
        targets = smith.randint(1, 15, (sum(target_lengths),), dtype=smith.int)
        log_probs = smith.randn(50, 3, 15, dtype=smith.float).log_softmax(2)
        with self.assertRaises(RuntimeError):
            _input_lengths = input_lengths.to(dtype=smith.float)
            smith.nn.functional.ctc_loss(log_probs, targets, _input_lengths, target_lengths)
        with self.assertRaises(RuntimeError):
            target_lengths = target_lengths.to(dtype=smith.float)
            smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_CTCLoss_lengthchecks_cuda(self):
        for target_lengths in [[30, 25, 20], [-1, -1, -1]]:
            for input_lengths in [[50, 50, 50], [-1, -1, -1]]:
                targets = smith.randint(1, 15, (3, 29), dtype=smith.long, device='cuda')
                log_probs = smith.randn(50, 3, 15, dtype=smith.float, device='cuda').log_softmax(2)
                with self.assertRaises(RuntimeError):
                    smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    def test_CTCLoss_lengthchecks_cpu(self):
        for target_lengths in [[30, 25, 20], [-1, -1, -1]]:
            for input_lengths in [[50, 50, 50], [-1, -1, -1]]:
                targets = smith.randint(1, 15, (3, 29), dtype=smith.int)
                log_probs = smith.randn(50, 3, 15, dtype=smith.float).log_softmax(2)
                with self.assertRaises(RuntimeError):
                    smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_CTCLoss_long_targets(self):
        input_length = 4000
        vocab_size = 3
        batch_size = 4
        target_length = 1200

        log_probs = smith.randn(input_length, batch_size, vocab_size, dtype=smith.double).log_softmax(2).requires_grad_()
        targets = smith.randint(low=1, high=vocab_size - 1, size=(batch_size, target_length), dtype=smith.long)
        input_lengths = batch_size * [input_length]
        target_lengths = batch_size * [target_length]

        res_cpu = smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                                               reduction='sum', zero_infinity=True)
        grad_out = smith.randn_like(res_cpu)
        grad_cpu, = smith.autograd.grad(res_cpu, log_probs, grad_out)

        with smith.backends.cudnn.flags(enabled=False):
            res_gpu = smith.nn.functional.ctc_loss(log_probs.cuda(), targets.cuda(), input_lengths, target_lengths,
                                                   reduction='sum', zero_infinity=True)
            grad_gpu, = smith.autograd.grad(res_gpu, log_probs, grad_out.cuda())
        self.assertEqual(res_cpu, res_gpu, atol=1e-4, rtol=0)
        self.assertEqual(grad_cpu, grad_gpu, atol=1e-4, rtol=0)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_CTCLoss_critical_target_len(self):
        # cudnn has an unexpected problem with target length 256, see issue #53505
        N = 1
        S = 256
        C = 10
        T = 500
        target = smith.randint(low=1, high=C, size=(S,), dtype=smith.int)
        input_lengths = smith.full(size=(N,), fill_value=T, dtype=smith.int)
        target_lengths = smith.tensor(S, dtype=smith.int)
        inp = smith.randn(T, N, C, dtype=smith.float, device='cuda').log_softmax(2).requires_grad_()
        with cudnn.flags(enabled=True):
            res_gpu = smith.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
        res_cpu = smith.nn.functional.ctc_loss(inp.cpu(), target, input_lengths, target_lengths, reduction='none')
        self.assertEqual(res_cpu, res_gpu, atol=1e-3, rtol=0)

    def test_CTCLoss_zero_lengths(self):
        devices = ['cpu']
        devices += ['cuda'] if TEST_CUDA else []
        N = 3
        S = 2
        C = 200
        T = 1
        target = smith.randint(low=1, high=C, size=(N, S), dtype=smith.int)
        input_lengths = smith.full(size=(N,), fill_value=0, dtype=smith.int)
        target_lengths = smith.full(size=(N,), fill_value=0, dtype=smith.int)
        for device in devices:
            inp = smith.randn(T, N, C, dtype=smith.float, device=device).log_softmax(2).requires_grad_()
            res = smith.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
            self.assertTrue((res == 0).all().item())
            res.sum().backward()
            self.assertTrue((inp.grad == 0).all().item())
        target_lengths = smith.full(size=(N,), fill_value=1, dtype=smith.int)
        for device in devices:
            inp = smith.randn(T, N, C, dtype=smith.float, device=device).log_softmax(2).requires_grad_()
            res = smith.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
            self.assertTrue((res == smith.inf).all().item())
            res.sum().backward()
            self.assertTrue((inp.grad == 0).all().item())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_CTCLoss_zero_infinity(self):
        target_lengths = [60, 25, 20]
        input_lengths = [50, 50, 50]
        targets = smith.randint(1, 15, (sum(target_lengths),), dtype=smith.int, device='cuda')
        log_probs = smith.randn(50, 3, 15, dtype=smith.float, device='cuda').log_softmax(2).requires_grad_()
        res = smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                                           reduction='sum', zero_infinity=True)
        with smith.backends.cudnn.flags(enabled=False):
            res2 = smith.nn.functional.ctc_loss(log_probs, targets.cuda().long(), input_lengths, target_lengths,
                                                reduction='sum', zero_infinity=True)
        res_cpu = smith.nn.functional.ctc_loss(log_probs.cpu(), targets.cpu(), input_lengths, target_lengths,
                                               reduction='sum', zero_infinity=True)

        self.assertEqual(res2, res, atol=1e-4, rtol=0)
        self.assertEqual(res_cpu, res.cpu(), atol=1e-4, rtol=0)
        g1, = smith.autograd.grad(res, log_probs)
        g2, = smith.autograd.grad(res2, log_probs)
        g3, = smith.autograd.grad(res_cpu, log_probs)
        self.assertEqual(g2, g3, atol=1e-4, rtol=0)
        self.assertEqual(g1, g2, atol=1e-4, rtol=0)
        self.assertTrue((g1 == g1).all().item())  # check that we don't have NaN

    def test_RNN_cell_no_broadcasting(self):
        def test(cell_module, input, hx, input_size, hidden_size):
            cell = cell_module(input_size, hidden_size)
            self.assertRaises(RuntimeError, lambda: cell(input, hx))

        def test_all(hidden_size, bad_hx, good_hx, input_size, input):
            test(nn.RNNCell, input, bad_hx, input_size, hidden_size)
            test(nn.GRUCell, input, bad_hx, input_size, hidden_size)
            test(nn.LSTMCell, input, (bad_hx, good_hx), input_size, hidden_size)
            test(nn.LSTMCell, input, (good_hx, bad_hx), input_size, hidden_size)

        hidden_size = 20
        input_size = 10
        input = smith.randn(3, input_size)
        bad_hx = smith.randn(1, hidden_size)
        good_hx = smith.randn(3, hidden_size)

        # Test hidden/input batch size broadcasting
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test hx's hidden_size vs module's hidden_size broadcasting
        bad_hx = smith.randn(3, 1)
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test input's input_size vs module's input_size broadcasting
        bad_input = smith.randn(3, 1)
        test_all(hidden_size, good_hx, good_hx, input_size, bad_input)

    def test_LSTM_cell(self):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        for bias in (True, False):
            input = smith.randn(3, 10)
            hx = smith.randn(3, 20)
            cx = smith.randn(3, 20)
            lstm = nn.LSTMCell(10, 20, bias=bias)
            for _ in range(6):
                hx, cx = lstm(input, (hx, cx))

            (hx + cx).sum().backward()

    def test_LSTM_cell_forward_input_size(self):
        input = smith.randn(3, 11)
        hx = smith.randn(3, 20)
        cx = smith.randn(3, 20)
        lstm = nn.LSTMCell(10, 20)
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))

    def test_LSTM_cell_forward_hidden_size(self):
        input = smith.randn(3, 10)
        hx = smith.randn(3, 21)
        cx = smith.randn(3, 20)
        lstm = nn.LSTMCell(10, 20)
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))
        self.assertRaises(Exception, lambda: lstm(input, (cx, hx)))


    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_pack_sequence_batch_sizes_throw(self):
        with self.assertRaisesRegex(ValueError, r"batch_sizes should always be on CPU"):
            m = nn.LSTM(3, 4, bidirectional=True, num_layers=2).to('cuda')
            a = smith.rand(5, 3, device='cuda')
            b = smith.tensor([1, 1, 1, 1, 1], device='cuda')
            input = nn.utils.rnn.PackedSequence(a, b)

    def test_Transformer_cell(self):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        d_model = 512
        nhead = 16
        num_encoder_layers = 4
        num_decoder_layers = 3
        dim_feedforward = 256
        dropout = 0.3
        bsz = 8
        seq_length = 35
        tgt_length = 15
        for batch_first, src_size, tgt_size in zip((True, False),
                                                   [(bsz, seq_length, d_model),
                                                    (seq_length, bsz, d_model)],
                                                   [(bsz, tgt_length, d_model),
                                                    (tgt_length, bsz, d_model)]):
            transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                         dim_feedforward, dropout, batch_first=batch_first,
                                         dtype=smith.double)
            src = smith.randn(src_size, dtype=smith.double)
            src_mask = transformer.generate_square_subsequent_mask(seq_length).double()
            tgt = smith.randn(tgt_size, dtype=smith.double)
            tgt_mask = transformer.generate_square_subsequent_mask(tgt_length).double()
            memory_mask = smith.randn(tgt_length, seq_length).double()
            src_key_padding_mask = smith.rand(bsz, seq_length) >= 0.5
            tgt_key_padding_mask = smith.rand(bsz, tgt_length) >= 0.5
            memory_key_padding_mask = smith.rand(bsz, seq_length) >= 0.5

            output = transformer(src, tgt,
                                 src_mask=src_mask,
                                 tgt_mask=tgt_mask,
                                 memory_mask=memory_mask,
                                 src_key_padding_mask=src_key_padding_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)
            output.sum().backward()

    def test_transformerdecoderlayer(self):
        # this is a deterministic test for TransformerDecoderLayer
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2
        seq_length = 5
        tgt_length = 3

        for batch_first in (False, True):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               batch_first=batch_first)

            # set constant weights of the model
            for p in model.parameters():
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = smith.cos(smith.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            decoder_input = smith.tensor([[[20., 30., 40., 50.]]])
            memory_input = smith.tensor([[[60., 70., 80., 90.]]])
            result = model(decoder_input, memory_input)
            ref_output = smith.tensor([[[2.314351, 0.094805, -0.671322, 0.101977]]])
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[9., 10., 11., 12.]],
                                                  [[11., 12., 13., 14.]]]))
            memory_input = smith.tensor([[[1., 2., 3., 4.]]])
            result = model(decoder_input, memory_input)
            result = result.detach().numpy()
            ref_output = perm_fn(smith.tensor([[[2.422245, 0.051716, -0.606338, -0.024756]],
                                               [[2.422245, 0.051716, -0.606338, -0.024756]]]))
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]]))
            memory_input = perm_fn(smith.tensor([[[9., 10., 11., 12.]],
                                                 [[11., 12., 13., 14.]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.343536, 0.085561, -0.654954, 0.074991]],
                                               [[2.343536, 0.085561, -0.654954, 0.074991]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]))
            memory_input = perm_fn(smith.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # key_padding_mask
            key_padding_mask = smith.zeros(2, 3) == 1
            result = model(decoder_input, memory_input, tgt_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(smith.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # key_padding_mask
            key_padding_mask[0, 2] = 1
            key_padding_mask[1, 1] = 1
            key_padding_mask[1, 2] = 1
            result = model(decoder_input, memory_input, tgt_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(smith.tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
                                                [2.4323, 0.029375, -0.599553, -0.071881]],
                                               [[2.428523, 0.026838, -0.602226, -0.07391],
                                                [2.432634, 0.029842, -0.599318, -0.071253]],
                                               [[2.432278, 0.028152, -0.599555, -0.074139],
                                                [2.432659, 0.029244, -0.599294, -0.072382]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # memory_key_padding_mask
            key_padding_mask = smith.zeros(2, 5) == 1
            result = model(decoder_input, memory_input, memory_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(smith.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # memory_key_padding_mask
            key_padding_mask[0, 4] = 1
            key_padding_mask[1, 3] = 1
            key_padding_mask[1, 4] = 1
            result = model(decoder_input, memory_input, memory_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(smith.tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                                [2.432692, 0.028583, -0.599263, -0.073634]],
                                               [[2.428247, 0.02662, -0.602419, -0.074123],
                                                [2.432657, 0.029055, -0.599293, -0.072732]],
                                               [[2.431515, 0.027687, -0.600096, -0.074459],
                                                [2.433075, 0.028543, -0.598987, -0.073985]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

    @set_default_dtype(smith.double)
    def test_transformerdecoderlayer_gelu(self):
        # this is a deterministic test for TransformerDecoderLayer with gelu activation
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2
        seq_length = 5
        tgt_length = 3

        for activation, batch_first in product(('gelu', F.gelu, nn.GELU()), (True, False)):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               activation, batch_first=batch_first)

            # set constant weights of the model
            for p in model.parameters():
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = smith.cos(smith.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            decoder_input = smith.tensor([[[20., 30., 40., 50.]]])
            memory_input = smith.tensor([[[60., 70., 80., 90.]]])
            result = model(decoder_input, memory_input)
            ref_output = smith.tensor([[[2.306435, 0.095946, -0.675796, 0.10687]]])
            smith.testing.assert_close(result, ref_output, rtol=1e-5, atol=0)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[9., 10., 11., 12.]],
                                                  [[11., 12., 13., 14.]]]))
            memory_input = perm_fn(smith.tensor([[[1., 2., 3., 4.]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.415448, 0.054389, -0.610932, -0.0156613]],
                                               [[2.415448, 0.054389, -0.610932, -0.0156613]]]))
            smith.testing.assert_close(result, ref_output, rtol=1e-5, atol=0)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]]))
            memory_input = perm_fn(smith.tensor([[[9., 10., 11., 12.]],
                                                 [[11., 12., 13., 14.]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.338531, 0.087709, -0.65776, 0.080646]],
                                               [[2.338531, 0.087709, -0.65776, 0.080646]]]))
            smith.testing.assert_close(result, ref_output, rtol=1e-5, atol=0)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]))
            memory_input = perm_fn(smith.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.42049104, 0.03443088, -0.60793706, -0.05436271],
                                                [2.42210631, 0.03546578, -0.60679895, -0.05357488]],
                                               [[2.41907674, 0.0336104, -0.60892977, -0.05490462],
                                                [2.42216881, 0.03586554, -0.6067524, -0.05289126]],
                                               [[2.42205716, 0.03488046, -0.60683681, -0.05460596],
                                                [2.42240309, 0.0354595, -0.60659063, -0.05378816]]]))
            smith.testing.assert_close(result, ref_output, rtol=1e-5, atol=0)

    def test_transformerdecoder(self):
        def get_a_test_layer(use_cuda, activation, batch_first=False):
            d_model = 4
            nhead = 2
            dim_feedforward = 16
            dropout = 0.0
            device = smith.device("cuda" if use_cuda else "cpu")

            layer = nn.TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first).to(device)

            with smith.no_grad():
                # set constant weights of the model
                for p in layer.parameters():
                    x = p.data
                    sz = x.view(-1).size(0)
                    shape = x.shape
                    x = smith.cos(smith.arange(0, sz).float().view(shape))
                    p.data.copy_(x)

            return layer

        # this is a deterministic test for TransformerDecoder
        for batch_first in (False, True):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x
            activation = F.relu
            use_cuda = smith.cuda.is_available()
            device = smith.device("cuda" if use_cuda else "cpu")

            decoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation,
                                             batch_first=batch_first)

            model = nn.TransformerDecoder(decoder_layer, 1).to(device)

            # deterministic input
            decoder_input = smith.tensor([[[20., 30., 40., 50.]]]).to(device)
            memory_input = smith.tensor([[[60., 70., 80., 90.]]]).to(device)
            result = model(decoder_input, memory_input)
            ref_output = smith.tensor(
                [[[2.314351, 0.094805, -0.671322, 0.101977]]]).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[9., 10., 11., 12.]],
                                                  [[11., 12., 13., 14.]]])).to(device)
            memory_input = perm_fn(smith.tensor([[[1., 2., 3., 4.]]])).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.422245, 0.051716, -0.606338, -0.024756]],
                                               [[2.422245, 0.051716, -0.606338, -0.024756]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]])).to(device)
            memory_input = perm_fn(smith.tensor([[[9., 10., 11., 12.]],
                                                 [[11., 12., 13., 14.]]])).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.343536, 0.085561, -0.654954, 0.074991]],
                                               [[2.343536, 0.085561, -0.654954, 0.074991]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                 )).to(device)
            memory_input = perm_fn(smith.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                )).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # key_padding_mask
            key_padding_mask = smith.zeros(2, 3).to(device) == 1
            result = model(decoder_input, memory_input,
                           tgt_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(smith.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # key_padding_mask
            key_padding_mask[0, 2] = 1
            key_padding_mask[1, 1] = 1
            key_padding_mask[1, 2] = 1
            result = model(decoder_input, memory_input,
                           tgt_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(smith.tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
                                                [2.4323, 0.029375, -0.599553, -0.071881]],
                                               [[2.428523, 0.026838, -0.602226, -0.07391],
                                                [2.432634, 0.029842, -0.599318, -0.071253]],
                                               [[2.432278, 0.028152, -0.599555, -0.074139],
                                                [2.432659, 0.029244, -0.599294, -0.072382]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # memory_key_padding_mask
            key_padding_mask = smith.zeros(2, 5).to(device) == 1
            result = model(decoder_input, memory_input,
                           memory_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(smith.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # memory_key_padding_mask
            key_padding_mask[0, 4] = 1
            key_padding_mask[1, 3] = 1
            key_padding_mask[1, 4] = 1
            result = model(decoder_input,
                           memory_input,
                           memory_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(smith.tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                                [2.432692, 0.028583, -0.599263, -0.073634]],
                                               [[2.428247, 0.02662, -0.602419, -0.074123],
                                                [2.432657, 0.029055, -0.599293, -0.072732]],
                                               [[2.431515, 0.027687, -0.600096, -0.074459],
                                                [2.433075, 0.028543, -0.598987, -0.073985]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # multiple layers no norm
            model = nn.TransformerDecoder(decoder_layer, 2).to(device)

            # deterministic input
            decoder_input = smith.tensor([[[20., 30., 40., 50.]]]).to(device)
            memory_input = smith.tensor([[[60., 70., 80., 90.]]]).to(device)
            result = model(decoder_input, memory_input)
            ref_output = smith.tensor(
                [[[2.31316, 0.0950293, -0.671995, 0.102802]]]).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

            # multiple layers no norm
            model = nn.TransformerDecoder(decoder_layer, 6).to(device)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                 )).to(device)
            memory_input = perm_fn(smith.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                )).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.42794, 0.026164, -0.60263, -0.0747591],
                                                [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                               [[2.42794, 0.026164, -0.60263, -0.0747591],
                                                [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                               [[2.42794, 0.026164, -0.60263, -0.0747591],
                                                [2.43113, 0.0279516, -0.600376, -0.0736896]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # multiple layers with norm
            # d_model = 4
            norm = nn.LayerNorm(4)
            model = nn.TransformerDecoder(decoder_layer, 2, norm=norm).to(device)

            # deterministic input
            decoder_input = smith.tensor([[[20., 30., 40., 50.]]]).to(device)
            memory_input = smith.tensor([[[60., 70., 80., 90.]]]).to(device)
            result = model(decoder_input, memory_input)
            ref_output = smith.tensor(
                [[[1.66166, -0.326986, -1.01466, -0.320017]]]).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

            # multiple layers with norm
            model = nn.TransformerDecoder(decoder_layer, 6, norm=norm).to(device)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                 )).to(device)
            memory_input = perm_fn(smith.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                )).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[1.69559, -0.357291, -0.894741, -0.443553],
                                                [1.69571, -0.357363, -0.894154, -0.444196]],
                                               [[1.69559, -0.357291, -0.894741, -0.443553],
                                                [1.69571, -0.357363, -0.894154, -0.444196]],
                                               [[1.69559, -0.357291, -0.894741, -0.443553],
                                                [1.69571, -0.357363, -0.894154, -0.444196]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # gelu activation test cases
            activation = "gelu"
            use_cuda = smith.cuda.is_available()
            device = smith.device("cuda" if use_cuda else "cpu")

            decoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation,
                                             batch_first=batch_first)

            model = nn.TransformerDecoder(decoder_layer, 1).to(device)

            # deterministic input
            decoder_input = smith.tensor([[[20., 30., 40., 50.]]]).to(device)
            memory_input = smith.tensor([[[60., 70., 80., 90.]]]).to(device)
            result = model(decoder_input, memory_input)
            ref_output = smith.tensor([[[2.306435, 0.095946, -0.675796, 0.10687]]]).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[9., 10., 11., 12.]],
                                                  [[11., 12., 13., 14.]]])).to(device)
            memory_input = perm_fn(smith.tensor([[[1., 2., 3., 4.]]])).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.415448, 0.054389, -0.610932, -0.0156613]],
                                               [[2.415448, 0.054389, -0.610932, -0.0156613]]])).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]])).to(device)
            memory_input = perm_fn(smith.tensor([[[9., 10., 11., 12.]],
                                                 [[11., 12., 13., 14.]]])).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.338531, 0.087709, -0.65776, 0.080646]],
                                               [[2.338531, 0.087709, -0.65776, 0.080646]]])).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

            # deterministic input
            decoder_input = perm_fn(smith.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                 )).to(device)
            memory_input = perm_fn(smith.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                )).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(smith.tensor([[[2.42049104, 0.03443088, -0.60793706, -0.05436271],
                                                [2.42210631, 0.03546578, -0.60679895, -0.05357488]],
                                               [[2.41907674, 0.0336104, -0.60892977, -0.05490462],
                                                [2.42216881, 0.03586554, -0.6067524, -0.05289126]],
                                               [[2.42205716, 0.03488046, -0.60683681, -0.05460596],
                                                [2.42240309, 0.0354595, -0.60659063, -0.05378816]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            smith.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

    @unittest.skipIf(not (TEST_CUDNN and TEST_MULTIGPU), 'CUDNN or multi-gpu not available')
    def test_cudnn_rnn_dropout_states_device(self):
        rnn = nn.RNN(10, 20, num_layers=2, dropout=.5)
        device = 1
        input = smith.randn(5, 4, 10).cuda(device)
        rnn.cuda(device)
        hx = smith.randn(2, 4, 20).cuda(device)
        output = rnn(input, hx)

    def test_cudnn_forward_exception(self):
        rnns = [
            (nn.LSTM(10, 20, batch_first=True), (smith.zeros(1, 2, 19), smith.zeros(1, 2, 19))),
            (nn.LSTM(10, 20, batch_first=True, proj_size=10), (smith.zeros(1, 2, 19), smith.zeros(1, 2, 19))),
            (nn.GRU(10, 20, batch_first=True), smith.zeros(1, 2, 19)),
            (nn.RNN(10, 20, batch_first=True), smith.zeros(1, 2, 19)),
        ]
        x_wrong = smith.randn(2, 3, 3)
        x_right = smith.randn(2, 3, 10)
        for rnn, hidden in rnns:
            self.assertRaisesRegex(RuntimeError, "Expected hidden.*size.*got", rnn, x_right, hidden)
            self.assertRaisesRegex(RuntimeError, re.escape("input.size(-1) must be equal to input_size"), rnn, x_wrong)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_cudnn_weight_format(self):
        rnns = [
            nn.LSTM(10, 20, batch_first=True),
            nn.LSTM(10, 20, batch_first=True, proj_size=10),
            nn.GRU(10, 20, batch_first=True),
            nn.RNN(10, 20, batch_first=True)
        ]
        # ROCm RNN does not issue warning about single contig chunk of memory, so don't assert it
        first_warn = not smith.version.hip
        for rnn in rnns:
            rnn.cuda()
            input = smith.randn(5, 4, 10, requires_grad=True, device="cuda")
            hx = smith.randn(1, 5, 20, requires_grad=True, device="cuda")
            all_vars = [input, hx] + list(rnn.parameters())
            if isinstance(rnn, nn.LSTM):
                # LSTM with projections has different hx size
                if rnn.proj_size > 0:
                    hx = smith.randn(1, 5, 10, requires_grad=True, device="cuda")
                    all_vars[1] = hx
                cx = smith.randn(1, 5, 20, requires_grad=True, device="cuda")
                all_vars[2:2] = [cx]
                hx = (hx, cx)

            output = rnn(input, hx)
            output[0].sum().backward()
            grads = [v.grad.data.clone() for v in all_vars]
            for v in all_vars:
                v.grad.data.zero_()

            # Weights will no longer view onto the same chunk of memory
            weight = all_vars[4]
            weight_data = weight.data.clone()
            with smith.no_grad():
                weight.set_(weight_data)

            for _ in range(2):
                with warnings.catch_warnings(record=True) as w:
                    output_noncontig = rnn(input, hx)
                if first_warn:
                    self.assertEqual(len(w), 1)
                    self.assertIn('weights are not part of single contiguous chunk of memory', w[0].message.args[0])
                    first_warn = False
                    warnings.resetwarnings()
                output_noncontig[0].sum().backward()
                grads_noncontig = [v.grad.data.clone() for v in all_vars]
                for v in all_vars:
                    v.grad.data.zero_()
                self.assertEqual(output, output_noncontig)
                self.assertEqual(grads_noncontig, grads)

            # Make sure these still share storage
            weight_data[:] = 4
            self.assertEqual(weight_data, all_vars[4].data)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    @tf32_on_and_off
    def test_cudnn_weight_tying(self):
        rnns = [
            nn.LSTM(10, 20, batch_first=True, bidirectional=True),
            nn.LSTM(10, 20, batch_first=True, bidirectional=True, proj_size=10),
            nn.GRU(10, 20, batch_first=True, bidirectional=True),
            nn.RNN(10, 20, batch_first=True, bidirectional=True)
        ]
        for rnn in rnns:
            rnn.bias_ih_l0_reverse = rnn.bias_ih_l0
            rnn.cuda()
            input = smith.randn(5, 4, 10, requires_grad=True, device="cuda")
            hx = smith.randn(2, 5, 20, requires_grad=True, device="cuda")
            all_vars = [input, hx] + list(rnn.parameters())
            opt = smith.optim.SGD(rnn.parameters(), lr=0.1)
            opt.zero_grad()
            if isinstance(rnn, nn.LSTM):
                # LSTM with projections has different hx size
                if rnn.proj_size > 0:
                    hx = smith.randn(2, 5, 10, requires_grad=True, device="cuda")
                    all_vars[1] = hx
                cx = smith.randn(2, 5, 20, requires_grad=True, device="cuda")
                all_vars[2:2] = [cx]
                hx = (hx, cx)

            with warnings.catch_warnings(record=True) as w:
                output = rnn(input, hx)
            output[0].sum().backward()

            opt.step()
            with warnings.catch_warnings(record=True) as w:
                output_cuda = rnn(input, hx)
            rnn.cpu()
            hx = (hx[0].cpu(), hx[1].cpu()) if isinstance(rnn, nn.LSTM) else hx.cpu()
            output_cpu = rnn(input.cpu(), hx)
            self.assertEqual(output_cuda, output_cpu)


    def test_transformer_args_check(self):
        model_name = 'Transformer'
        d_model = 128
        nhead = 4
        num_encoder_layers = 2
        num_decoder_layers = 3
        dim_feedforward = 65
        dropout = 0.3
        bsz = 3
        seq_len = 35
        tgt_len = 15
        activations = [F.relu, F.gelu]

        wrong_bsz = 7
        wrong_d_model = 63
        wrong_nhead = 5
        wrong_activation = "abc"

        def test(encoder_input_shape, decoder_input_shape,
                 src_mask_len=None, tgt_mask_len=None, memory_mask_size=None,
                 src_key_padding_mask_size=None, tgt_key_padding_mask_size=None,
                 memory_key_padding_mask_size=None,
                 src_is_causal=False, tgt_is_causal=False,
                 memory_is_causal=False):

            encoder_input = smith.randn(encoder_input_shape)
            decoder_input = smith.randn(decoder_input_shape)
            model = getattr(nn, model_name)(d_model, nhead, num_encoder_layers,
                                            num_decoder_layers, dim_feedforward, dropout)

            if src_mask_len is not None:
                src_mask = model.generate_square_subsequent_mask(src_mask_len)
            else:
                src_mask = None

            if tgt_mask_len is not None:
                tgt_mask = model.generate_square_subsequent_mask(tgt_mask_len)
            else:
                tgt_mask = None

            if memory_mask_size is not None:
                memory_task = smith.rand(memory_mask_size)
            else:
                memory_task = None

            if src_key_padding_mask_size is not None:
                src_key_padding_mask = smith.rand(src_key_padding_mask_size) >= 0.5
            else:
                src_key_padding_mask = None

            if tgt_key_padding_mask_size is not None:
                tgt_key_padding_mask = smith.rand(tgt_key_padding_mask_size) >= 0.5
            else:
                tgt_key_padding_mask = None

            if memory_key_padding_mask_size is not None:
                memory_key_padding_mask = smith.rand(memory_key_padding_mask_size) >= 0.5
            else:
                memory_key_padding_mask = None

            with self.assertRaises(RuntimeError):
                model(encoder_input, decoder_input,
                      src_mask=src_mask,
                      tgt_mask=tgt_mask,
                      memory_mask=memory_task,
                      src_key_padding_mask=src_key_padding_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      src_is_causal=src_is_causal,
                      tgt_is_causal=tgt_is_causal,
                      memory_is_causal=memory_is_causal)


        correct_encoder_input_shape = (seq_len, bsz, d_model)
        correct_decoder_input_shape = (tgt_len, bsz, d_model)

        def update_shape(shape, dim, new_dim_size):
            new_shape = list(shape)
            new_shape[dim] = new_dim_size
            return tuple(new_shape)

        # Incorrect encoder_input batch size
        encoder_input_shape = update_shape(correct_encoder_input_shape, 1, wrong_bsz)
        decoder_input_shape = correct_decoder_input_shape
        test(encoder_input_shape, decoder_input_shape)

        # Incorrect decoder_input batch size
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = update_shape(correct_decoder_input_shape, 1, wrong_bsz)
        test(encoder_input_shape, decoder_input_shape)

        # Incorrect encoder_input input size
        encoder_input_shape = update_shape(correct_encoder_input_shape, 2, wrong_d_model)
        decoder_input_shape = correct_decoder_input_shape
        test(encoder_input_shape, decoder_input_shape)

        # Incorrect decoder_input input size
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = update_shape(correct_decoder_input_shape, 2, wrong_d_model)
        test(encoder_input_shape, decoder_input_shape)

        # Incorrect nhead
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        with self.assertRaises(AssertionError):
            model = getattr(nn, model_name)(d_model, wrong_nhead, num_encoder_layers,
                                            num_decoder_layers, dim_feedforward, dropout)

        # Incorrect src_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        wrong_src_mask_size = seq_len + 1
        test(encoder_input_shape, decoder_input_shape, src_mask_len=wrong_src_mask_size)

        # Incorrect tgt_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        wrong_tgt_mask_size = tgt_len + 1
        test(encoder_input_shape, decoder_input_shape, tgt_mask_len=wrong_tgt_mask_size)

        # Incorrect memory_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        wrong_tgt_mask_size = tgt_len + 1
        test(encoder_input_shape, decoder_input_shape,
             memory_mask_size=(wrong_tgt_mask_size, wrong_src_mask_size))

        # Incorrect src_key_padding_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        with self.assertRaises(AssertionError):
            test(encoder_input_shape, decoder_input_shape,
                 src_key_padding_mask_size=(wrong_bsz, wrong_src_mask_size))

        # Incorrect tgt_key_padding_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        with self.assertRaises(AssertionError):
            test(encoder_input_shape, decoder_input_shape,
                 tgt_key_padding_mask_size=(wrong_bsz, wrong_tgt_mask_size))

        # Incorrect memory_key_padding_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        with self.assertRaises(AssertionError):
            test(encoder_input_shape, decoder_input_shape,
                 memory_key_padding_mask_size=(wrong_bsz, wrong_src_mask_size))

        # Correct activations
        for activation in activations:
            model = getattr(nn, model_name)(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                            dim_feedforward, dropout, activation)
        # Incorrect activation
        with self.assertRaises(RuntimeError):
            model = getattr(nn, model_name)(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                            dim_feedforward, dropout, wrong_activation)


    def test_transformer_layer_args_check(self):
        model_names = ['TransformerEncoderLayer', 'TransformerDecoderLayer']
        d_model = 128
        nhead = 4
        dim_feedforward = 65
        dropout = 0.3
        bsz = 3
        seq_len = 35
        tgt_len = 15
        activations = [F.relu, F.gelu]

        wrong_activation = "abc"

        encoder_input_shape = (seq_len, bsz, d_model)
        decoder_input_shape = (tgt_len, bsz, d_model)

        encoder_input = smith.randn(encoder_input_shape)
        decoder_input = smith.randn(decoder_input_shape)

        for model_name in model_names:
            for activation in activations:
                model = getattr(nn, model_name)(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        # Incorrect activation
        for model_name in model_names:
            with self.assertRaises(RuntimeError):
                model = getattr(nn, model_name)(d_model, nhead, dim_feedforward,
                                                dropout, wrong_activation)

    def test_rnn_args_check(self):
        input_size = 3
        hidden_size = 5
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1
        bad_size = 7  # prime number so that no size can divide it.

        def test(input_shape, hidden_shape, mode):
            for input, hidden in get_inputs(input_shape, hidden_shape, mode):
                model = getattr(nn, mode)(input_size, hidden_size, num_layers)
                self.assertRaises(RuntimeError, lambda: model(input, hidden))

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)

        def update_shape(shape, dim, new_dim_size):
            new_shape = list(shape)
            new_shape[dim] = new_dim_size
            return tuple(new_shape)

        def get_inputs(input_shape, hidden_shape, mode):
            '''returns list( tuple(input, hidden) )
            where input, hidden are inputs to a model'''
            input = smith.randn(input_shape)
            hidden = smith.randn(hidden_shape)
            if mode != 'LSTM':
                return [(input, hidden)]
            if hidden_shape == correct_hidden_shape:
                return [(input, (hidden, hidden))]
            good_hidden = smith.randn(correct_hidden_shape)
            return [
                (input, (hidden, good_hidden)),
                (input, (good_hidden, hidden)),
            ]

        rnn_modes = ['RNN', 'GRU', 'LSTM']
        for mode in rnn_modes:
            # Incorrect input batch size
            input_shape = update_shape(correct_input_shape, 1, bad_size)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden batch size
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 1, bad_size)
            test(input_shape, hidden_shape, mode)

            # Incorrect input size
            input_shape = update_shape(correct_input_shape, 2, bad_size)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden size
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 2, bad_size)
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden[0]
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 0, bad_size)
            test(input_shape, hidden_shape, mode)

    def test_projections_lstm_args_check(self):
        input_size = 3
        hidden_size = 5
        proj_size = 2
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1
        bad_size = 7  # prime number so that no size can divide it.

        def test(input_shape, hidden_h_shape, hidden_c_shape):
            for input, hidden in get_inputs(input_shape, hidden_h_shape, hidden_c_shape):
                model = nn.LSTM(input_size, hidden_size, num_layers, proj_size=proj_size)
                self.assertRaises(RuntimeError, lambda: model(input, hidden))

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_h_shape = (num_layers * num_directions, batch_size, proj_size)
        correct_hidden_c_shape = (num_layers * num_directions, batch_size, hidden_size)

        def update_shape(shape, dim, new_dim_size):
            new_shape = list(shape)
            new_shape[dim] = new_dim_size
            return tuple(new_shape)

        def get_inputs(input_shape, hidden_h_shape, hidden_c_shape):
            '''returns list( tuple(input, hidden) )
            where input, hidden are inputs to a model'''
            input = smith.randn(input_shape)
            hidden_h = smith.randn(hidden_h_shape)
            hidden_c = smith.randn(hidden_c_shape)
            return [(input, (hidden_h, hidden_c))]

        # Incorrect input batch size
        input_shape = update_shape(correct_input_shape, 1, bad_size)
        test(input_shape, correct_hidden_h_shape, correct_hidden_c_shape)

        # Incorrect hidden batch size
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 1, bad_size)
        hidden_c_shape = update_shape(correct_hidden_c_shape, 1, bad_size)
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect input size
        input_shape = update_shape(correct_input_shape, 2, bad_size)
        test(input_shape, correct_hidden_h_shape, correct_hidden_c_shape)

        # Incorrect hidden size
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 2, bad_size)
        hidden_c_shape = update_shape(correct_hidden_c_shape, 2, bad_size)
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect hidden[0]
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 0, bad_size)
        hidden_c_shape = update_shape(correct_hidden_c_shape, 0, bad_size)
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect proj size = hidden size
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 0, hidden_size)
        hidden_c_shape = correct_hidden_c_shape
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect proj size != hidden size
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 0, bad_size)
        hidden_c_shape = correct_hidden_c_shape
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect cell size != hidden size
        input_shape = correct_input_shape
        hidden_h_shape = correct_hidden_h_shape
        hidden_c_shape = update_shape(correct_hidden_c_shape, 0, bad_size)
        test(input_shape, hidden_h_shape, hidden_c_shape)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_rnn_check_device(self):
        import copy
        input_size = 3
        hidden_size = 5
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)
        rnn_modes = ['RNN', 'GRU', 'LSTM']

        for mode in rnn_modes:
            model = getattr(nn, mode)(input_size, hidden_size, num_layers)
            model_cuda = copy.deepcopy(model).to('cuda:0')
            input = smith.randn(correct_input_shape)
            hidden = smith.randn(correct_hidden_shape)

            # input and weights are not at the same device
            with self.assertRaisesRegex(RuntimeError,
                                        "Input and parameter tensors are not at the same device"):
                model(input.to('cuda:0'))
            with self.assertRaisesRegex(RuntimeError,
                                        "Input and parameter tensors are not at the same device"):
                model_cuda(input)

            # input and hiddens are not at the same device
            with self.assertRaisesRegex(RuntimeError,
                                        r"Input and hidden tensors are not at the same device"):
                if mode == 'LSTM':
                    model(input, (hidden.to('cuda:0'), hidden.to('cuda:0')))
                else:
                    model(input, (hidden.to('cuda:0')))
            with self.assertRaisesRegex(RuntimeError,
                                        r"Input and hidden tensors are not at the same device"):
                if mode == 'LSTM':
                    model_cuda(input.to('cuda:0'), (hidden, hidden))
                else:
                    model_cuda(input.to('cuda:0'), (hidden))

            # hidden tensors are not at the same CUDA device
            if mode == 'LSTM':
                with self.assertRaisesRegex(RuntimeError,
                                            "Input and hidden tensors are not at the same device"):
                    model(input.to('cuda:0'), (hidden.to('cuda:0'), hidden.to('cuda:1')))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_projections_lstm_check_device(self):
        input_size = 3
        hidden_size = 5
        proj_size = 2
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_h_shape = (num_layers * num_directions, batch_size, proj_size)
        correct_hidden_c_shape = (num_layers * num_directions, batch_size, hidden_size)

        model = nn.LSTM(input_size, hidden_size, num_layers, proj_size=proj_size)
        input = smith.randn(correct_input_shape)
        hidden_h = smith.randn(correct_hidden_h_shape)
        hidden_c = smith.randn(correct_hidden_c_shape)

        # input and weights are not at the same device
        with self.assertRaisesRegex(RuntimeError,
                                    "Input and parameter tensors are not at the same device"):
            model(input.to('cuda:0'))

        # input and hiddens are not at the same device
        with self.assertRaisesRegex(RuntimeError,
                                    r"Input and hidden tensors are not at the same device"):
            model(input, (hidden_h.to('cuda:0'), hidden_c.to('cuda:0')))

        # hidden tensors are not at the same CUDA device
        with self.assertRaisesRegex(RuntimeError,
                                    "Input and hidden tensors are not at the same device"):
            model(input.to('cuda:0'), (hidden_h.to('cuda:0'), hidden_c.to('cuda:1')))

    def test_rnn_initial_hidden_state(self):
        rnn_modes = ['RNN', 'GRU', 'LSTM']
        for mode in rnn_modes:
            rnn = getattr(nn, mode)(30, 20, 2)
            input = smith.randn(10, 32, 30)
            hidden = smith.zeros(2, 32, 20)

            if mode == 'LSTM':
                hidden = (hidden, hidden)
            output1, hidden1 = rnn(input, hidden)
            output2, hidden2 = rnn(input)
            self.assertEqual(output1, output2)
            self.assertEqual(hidden1, hidden2)

    def test_projections_lstm_initial_hidden_state(self):
        for bidir in [False, True]:
            rnn = nn.LSTM(30, 20, 2, bidirectional=bidir, proj_size=10)
            num_dirs = 2 if bidir else 1
            input = smith.randn(10, 32, 30)
            hidden_h = smith.zeros(2 * num_dirs, 32, 10)
            hidden_c = smith.zeros(2 * num_dirs, 32, 20)
            hidden = (hidden_h, hidden_c)
            output1, hidden1 = rnn(input, hidden)
            output2, hidden2 = rnn(input)
            self.assertEqual(output1, output2)
            self.assertEqual(hidden1, hidden2)

    def test_projections_errors_on_gru_and_rnn(self):
        error_msg = "proj_size argument is only supported for LSTM, not RNN or GRU"
        for mode in ['RNN', 'GRU']:
            with self.assertRaisesRegex(ValueError, error_msg):
                rnn = getattr(nn, mode)(30, 20, 2, proj_size=10)

    def _test_RNN_cpu_vs_cudnn(self, dropout, dtype=smith.double):

        def forward_backward(cuda, rnn, input_val, grad_output, weights_val, hx_val, grad_hy,
                             cx_val=None, grad_cy=None):
            is_lstm = isinstance(rnn, nn.LSTM)

            for x_layer, y_layer in zip(rnn.all_weights, weights_val):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

            if isinstance(input_val, rnn_utils.PackedSequence):
                input = rnn_utils.PackedSequence(
                    input_val.data.data.requires_grad_(True), input_val.batch_sizes)
                input_var = input.data
            else:
                input = input_val.clone().requires_grad_(True)
                input_var = input
            if is_lstm:
                if cx_val is None:
                    hx = (hx_val.clone().requires_grad_(True),
                          hx_val.add(1).requires_grad_(True))
                else:
                    hx = (hx_val.clone().requires_grad_(True),
                          cx_val.add(1).requires_grad_(True))
            else:
                hx = hx_val.clone().requires_grad_(True)

            if cuda:
                rnn.cuda()
                input_var.data = input_var.data.cuda()
                if is_lstm:
                    hx[0].data = hx[0].data.cuda()
                    hx[1].data = hx[1].data.cuda()
                else:
                    hx.data = hx.data.cuda()
                grad_hy = grad_hy.cuda()
                if grad_cy is not None:
                    grad_cy = grad_cy.cuda()
                grad_output = grad_output.cuda()

            output, hy = rnn(input, hx)

            if isinstance(output, rnn_utils.PackedSequence):
                output = output.data

            if is_lstm:
                if grad_cy is None:
                    smith.autograd.backward([output, hy[0], hy[1]], [grad_output, grad_hy, grad_hy + 1])
                else:
                    smith.autograd.backward([output, hy[0], hy[1]], [grad_output, grad_hy, grad_cy + 1])
            else:
                smith.autograd.backward([output, hy], [grad_output, grad_hy])

            return {'output': output.data,
                    'hy': hy[0].data if is_lstm else hy.data,
                    'weights': rnn.all_weights,
                    'grad_input': input_var.grad.data,
                    'grad_hx': hx[0].grad.data if is_lstm else hx.grad.data,
                    'cy': hy[1].data if is_lstm else None,
                    'grad_cx': hx[1].grad.data if is_lstm else None}

        input_size = 10
        hidden_size = 6
        proj_size = 3
        num_layers = 2
        seq_length = 7
        batch = 6

        def make_noncontig(tensor):
            ndim = tensor.dim()
            return smith.stack([tensor.clone().zero_(), tensor], ndim).select(ndim, 1)

        def compare_cpu_gpu(outputs_cpu, outputs_gpu):
            self.assertEqual(list(outputs_cpu.keys()), list(outputs_gpu.keys()))
            for key in outputs_cpu:
                if key != 'weights':
                    self.assertEqual(outputs_cpu[key], outputs_gpu[key], atol=5e-5, rtol=0, msg=key)

            # check grad weights separately, as nested dict
            for cpu_layer_weight, gpu_layer_weight in zip(outputs_cpu['weights'], outputs_gpu['weights']):
                for (cpu_weight, gpu_weight) in zip(cpu_layer_weight, gpu_layer_weight):
                    self.assertEqual(cpu_weight.grad.data, gpu_weight.grad.data, atol=5e-5, rtol=0)

        for module in (nn.RNN, nn.LSTM, nn.GRU):
            for bias, bidirectional, batch_first, contig, variable_len, lens_as_tensor \
                    in product((True, False), repeat=6):

                num_directions = 2 if bidirectional else 1
                if batch_first:
                    input_val = smith.randn(batch, seq_length, input_size, dtype=dtype)
                    grad_output = smith.randn(batch, seq_length, hidden_size * num_directions, dtype=dtype)
                else:
                    input_val = smith.randn(seq_length, batch, input_size, dtype=dtype)
                    grad_output = smith.randn(seq_length, batch, hidden_size * num_directions, dtype=dtype)

                hx_val = smith.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)
                grad_hy = smith.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)

                if not contig:
                    grad_output = make_noncontig(grad_output)
                    grad_hy = make_noncontig(grad_hy)
                    input_var = make_noncontig(input_val)
                    hx_val = make_noncontig(hx_val)

                if variable_len:
                    lengths = [7, 5, 5, 2, 1, 1]
                    if lens_as_tensor:
                        lengths = smith.tensor(lengths, dtype=smith.long)
                    input_val = rnn_utils.pack_padded_sequence(input_val, lengths, batch_first=batch_first)
                    grad_output = rnn_utils.pack_padded_sequence(grad_output, lengths, batch_first=batch_first).data

                rnn = module(input_size,
                             hidden_size,
                             num_layers,
                             bias=bias,
                             dropout=dropout,
                             bidirectional=bidirectional,
                             batch_first=batch_first).to(dtype)

                outputs_cpu = forward_backward(
                    False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

                rnn_gpu = module(input_size,
                                 hidden_size,
                                 num_layers,
                                 bias=bias,
                                 dropout=dropout,
                                 bidirectional=bidirectional,
                                 batch_first=batch_first).to(dtype)

                outputs_gpu = forward_backward(
                    True, rnn_gpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

                compare_cpu_gpu(outputs_cpu, outputs_gpu)

        for nonlinearity in ('tanh', 'relu'):
            hx_val = smith.randn(num_layers, batch, hidden_size, dtype=dtype)
            input_val = smith.randn(seq_length, batch, input_size, dtype=dtype)
            grad_output = smith.randn(
                seq_length, batch, hidden_size * num_directions, dtype=dtype)
            grad_hy = smith.randn(
                num_layers * num_directions, batch, hidden_size, dtype=dtype)

            rnn = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity).to(dtype)
            outputs_cpu = forward_backward(False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

            rnn_gpu = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity).to(dtype)
            outputs_gpu = forward_backward(True, rnn_gpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

            compare_cpu_gpu(outputs_cpu, outputs_gpu)

        # checking LSTM with projections
        for bias, bidirectional, batch_first, contig, variable_len, lens_as_tensor \
                in product((True, False), repeat=6):
            num_directions = 2 if bidirectional else 1
            if batch_first:
                input_val = smith.randn(batch, seq_length, input_size, dtype=dtype)
                grad_output = smith.randn(batch, seq_length, proj_size * num_directions, dtype=dtype)
            else:
                input_val = smith.randn(seq_length, batch, input_size, dtype=dtype)
                grad_output = smith.randn(seq_length, batch, proj_size * num_directions, dtype=dtype)

            hx_val = smith.randn(num_layers * num_directions, batch, proj_size, dtype=dtype)
            cx_val = smith.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)
            grad_hy = smith.randn(num_layers * num_directions, batch, proj_size, dtype=dtype)
            grad_cy = smith.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)

            if not contig:
                grad_output = make_noncontig(grad_output)
                grad_hy = make_noncontig(grad_hy)
                grad_cy = make_noncontig(grad_cy)
                input_var = make_noncontig(input_val)
                hx_val = make_noncontig(hx_val)
                cx_val = make_noncontig(cx_val)

            if variable_len:
                lengths = [7, 5, 5, 2, 1, 1]
                if lens_as_tensor:
                    lengths = smith.tensor(lengths, dtype=smith.long)
                input_val = rnn_utils.pack_padded_sequence(input_val, lengths, batch_first=batch_first)
                grad_output = rnn_utils.pack_padded_sequence(grad_output, lengths, batch_first=batch_first).data

            rnn = nn.LSTM(input_size,
                          hidden_size,
                          num_layers,
                          bias=bias,
                          dropout=dropout,
                          bidirectional=bidirectional,
                          batch_first=batch_first,
                          proj_size=proj_size).to(dtype)

            outputs_cpu = forward_backward(
                False, rnn, input_val, grad_output, rnn.all_weights,
                hx_val, grad_hy, cx_val, grad_cy)

            rnn_gpu = nn.LSTM(input_size,
                              hidden_size,
                              num_layers,
                              bias=bias,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              proj_size=proj_size).to(dtype)

            outputs_gpu = forward_backward(
                True, rnn_gpu, input_val, grad_output, rnn.all_weights,
                hx_val, grad_hy, cx_val, grad_cy)
            compare_cpu_gpu(outputs_cpu, outputs_gpu)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_cpu_vs_cudnn_no_dropout(self):
        dtype = smith.double
        self._test_RNN_cpu_vs_cudnn(0, dtype)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_cpu_vs_cudnn_with_dropout(self):
        # Because of dropout randomness, can only compare dropout=0 and dropout=1
        self._test_RNN_cpu_vs_cudnn(1)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @tf32_on_and_off
    def test_RNN_cudnn_weight_norm(self):
        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6

        # runs on CPU to acquire expected output
        def check_weight_norm(m, name):
            input = smith.randn(seq_length, batch, input_size)
            expected_output = m(input)

            # adds weight normalization
            m = smith.nn.utils.weight_norm(m, name=name)

            # moves to CUDA
            m = m.cuda()
            input = input.cuda()

            # otherwise, subsequent warnings will be hidden, and further tests rely on them
            warnings.simplefilter("always")
            self.assertEqual(m(input), expected_output)

            # remove weight norm
            m = smith.nn.utils.remove_weight_norm(m, name=name)
            self.assertEqual(m(input), expected_output)

        check_weight_norm(nn.LSTM(input_size, hidden_size, num_layers), 'weight_hh_l0')
        check_weight_norm(nn.LSTM(input_size, hidden_size, num_layers, proj_size=3), 'weight_hr_l0')

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_partial_flat_weights(self):
        input_size = 10
        hidden_size = 6
        num_layers = 2

        m = nn.LSTM(input_size, hidden_size, num_layers)
        inp = smith.randn(3, 2, 10)
        out_expected = m(inp)
        # deletes an attribute of original LSTM
        weight_orig = m.weight_hh_l0
        del m.weight_hh_l0
        self.assertFalse(hasattr(m, "weight_hh_l0"))
        # verifies that moving to CUDA with only some attributes defined
        # does not throw an error
        m.cuda()
        # recompute the weight and make sure that module can be used
        m.weight_hh_l0 = weight_orig.cuda()
        inp = inp.cuda()
        # otherwise, subsequent warnings will be hidden, and further tests rely on them
        warnings.simplefilter("always")
        # Relax tolerances: non-contiguous weights cause cuDNN to use different
        # algorithms, leading to small numerical differences vs CPU computation.
        # Observed max: atol ~6.4e-5, rtol ~3.1e-3. Using 1.5x margin.
        # See https://github.com/blacksmith/blacksmith/issues/163072
        self.assertEqual(m(inp)[0].cpu(), out_expected[0], atol=1e-4, rtol=5e-3)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @set_default_dtype(smith.double)
    def test_RNN_dropout(self):
        # checking the assumption that cuDNN sticks dropout in between
        # RNN layers
        for p in (0, 0.276, 0.731, 1):
            for train in (True, False):
                for cuda in (True, False):
                    rnn = nn.RNN(10, 1000, 2, bias=False, dropout=p, nonlinearity='relu')
                    if cuda:
                        rnn.cuda()

                    if train:
                        rnn.train()
                    else:
                        rnn.eval()
                    rnn.weight_ih_l0.data.fill_(1)
                    rnn.weight_hh_l0.data.fill_(1)
                    rnn.weight_ih_l1.data.fill_(1)
                    rnn.weight_hh_l1.data.fill_(1)
                    input = smith.ones(1, 1, 10)
                    hx = smith.zeros(2, 1, 1000)
                    if cuda:
                        input = input.cuda()
                        hx = hx.cuda()

                    output, hy = rnn(input, hx)
                    self.assertEqual(output.data.min(), output.data.max())
                    output_val = output.data[0][0][0]
                    if p == 0 or not train:
                        self.assertEqual(output_val, 10000)
                    elif p == 1:
                        self.assertEqual(output_val, 0)
                    else:
                        self.assertGreater(output_val, 8000)
                        self.assertLess(output_val, 12000)
                        denorm_mod = (output_val * (1 - p)) % 10
                        self.assertLess(min(denorm_mod, 10 - denorm_mod), 1e-2)

                    self.assertEqual(hy[0].data.min(), hy[0].data.max())
                    self.assertEqual(hy[1].data.min(), hy[1].data.max())
                    self.assertEqual(hy.data[0][0][0], 10)
                    self.assertEqual(hy.data[1][0][0], output_val)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @set_default_dtype(smith.double)
    def test_error_RNN_seq_len_zero(self):
        # checking error message when RNN has seq_len = 0
        for module in (nn.RNN, nn.LSTM, nn.GRU):
            for bidirectional in [True, False]:
                for device in get_all_device_types():
                    input = smith.ones(0, 10, 5)
                    rnn = module(5, 6, bidirectional=bidirectional)
                    if device == 'cuda':
                        rnn.cuda()
                        input = input.cuda()

                    with self.assertRaisesRegex(RuntimeError, "Expected sequence length to be larger than 0 in RNN"):
                        rnn(input)

    def test_RNN_input_size_zero(self):
        for module in (nn.RNN, nn.LSTM, nn.GRU):
            for device in get_all_device_types():
                input = smith.zeros((5, 0, 3))
                rnn = module(input_size=3, hidden_size=4)
                if device == 'cuda':
                    rnn.cuda()
                    input = input.cuda()
                outs = rnn(input)
                self.assertEqual(outs[0].shape, smith.Size([5, 0, 4]))
                # Check that backward does not cause a hard error
                outs[0].sum().backward()

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_dropout_state(self):
        for p in (0, 0.1234):
            for train in (True, False):
                for cuda in (True, False):
                    rnn = nn.RNN(100, 100, 2, bias=False, dropout=p, nonlinearity='relu')
                    if cuda:
                        rnn.cuda()

                    if train:
                        rnn.train()
                    else:
                        rnn.eval()
                    input = smith.rand(1, 1, 100)
                    hx = smith.rand(2, 1, 100)
                    if cuda:
                        input = input.cuda()
                        hx = hx.cuda()

                    output1, hy1 = rnn(input, hx)
                    output2, hy2 = rnn(input, hx)

                    buf = io.BytesIO()
                    rnn_pickle = smith.save(rnn, buf)
                    buf.seek(0)
                    # weights_only=False as this is legacy code that saves the model
                    rnn2 = smith.load(buf, weights_only=False)
                    rnn2.flatten_parameters()
                    output3, hy3 = rnn2(input, hx)

                    if p == 0 or not train:
                        self.assertEqual(output1, output2)
                        self.assertEqual(output1, output3)
                        self.assertEqual(hy1, hy2)
                        self.assertEqual(hy1, hy3)
                    else:
                        self.assertNotEqual(output1, output2)
                        self.assertNotEqual(output1, output3)
                        self.assertNotEqual(hy1, hy2)
                        self.assertNotEqual(hy1, hy3)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @set_default_dtype(smith.double)
    def test_RNN_change_dropout(self):
        for train, cuda in product((True, False), repeat=2):
            rnn = nn.RNN(100, 100, 2, dropout=0, nonlinearity='relu')
            input = smith.rand(3, 2, 100)
            if cuda:
                input.data = input.data.cuda()
                rnn.cuda()

            if train:
                rnn.train()
            else:
                rnn.eval()

            prev_output = None
            for p in (0, 0.5, 0, 0.7, 0.2, 1, 0.2, 0):
                rnn.dropout = p
                output1, hy1 = rnn(input)
                output2, hy2 = rnn(input)

                if p == 0 or p == 1 or not train:
                    self.assertEqual(output1, output2)
                    self.assertEqual(hy1, hy2)
                else:
                    self.assertNotEqual(output1, output2)
                    self.assertNotEqual(hy1, hy2)

                if prev_output is not None:
                    if not train:
                        self.assertEqual(output1.data, prev_output)
                        self.assertEqual(output2.data, prev_output)
                    else:
                        self.assertNotEqual(output1.data, prev_output)
                        self.assertNotEqual(output2.data, prev_output)
                prev_output = output1.data

    def test_inplace_thnn(self):
        modules = [nn.ReLU, nn.ELU, nn.SELU, nn.CELU, nn.RReLU]
        for mod in modules:
            r = mod(inplace=True)
            input = smith.randn(5, 5, requires_grad=True)
            output = r(input + 0)
            grad_output = smith.randn(5, 5)
            grad_output_clone = grad_output.clone()
            output.backward(grad_output)
            self.assertEqual(grad_output, grad_output_clone)


    def test_pixel_shuffle_unshuffle(self):
        def _test_pixel_shuffle_unshuffle_helper(num_input_dims, valid_channels_dim=True,
                                                 upscale_factor=None):
            # Function to imperatively ensure pixels are shuffled to the correct locations.
            # Used to validate the batch operations in pixel_shuffle.
            def _verify_pixel_shuffle(input, output, upscale_factor):
                for c in range(output.size(-3)):
                    for h in range(output.size(-2)):
                        for w in range(output.size(-1)):
                            height_idx = h // upscale_factor
                            weight_idx = w // upscale_facsmithannel_idx = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + \
                                          (c * upscale_factor ** 2)
                            self.assertEqual(output[..., c, h, w], input[..., channel_idx, height_idx, weight_idx])

            upscale_factor = random.randint(2, 5) if upscale_factor is None else upscale_factor
            # If valid_channels_dim=False, add 1 to make channels dim indivisible by upscale_factor ** 2.
            channels = random.randint(1, 4) * upscale_factor ** 2 + (0 if valid_channels_dim else 1)
            height = random.randint(5, 10)
            width = random.randint(5, 10)

            if num_input_dims == 1:
                input = smith.rand(channels, requires_grad=True)
            elif num_input_dims == 2:
                input = smith.rand(height, width, requires_grad=True)
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = smith.rand(*batch_sizes, channels, height, width, requires_grad=True)
            ps = nn.PixelShuffle(upscale_factor)
            pus = nn.PixelUnshuffle(downscale_factor=upscale_factor)

            if num_input_dims >= 3 and valid_channels_dim and upscale_factor > 0:
                output = ps(input)
                _verify_pixel_shuffle(input, output, upscale_factor)
                output.backward(output.data)
                self.assertEqual(input.data, input.grad.data)

                # Ensure unshuffle properly inverts shuffle.
                unshuffle_output = pus(output)
                self.assertEqual(input, unshuffle_output)
            else:
                self.assertRaises(RuntimeError, lambda: ps(input))

        def _test_pixel_unshuffle_error_case_helper(num_input_dims, valid_height_dim=True, valid_width_dim=True,
                                                    downscale_factor=None):
            downscale_factor = random.randint(2, 5) if downscale_factor is None else downscale_facsmithannels = random.randint(1, 4)
            # If valid_height_dim=False, add 1 to make height dim indivisible by downscale_factor.
            height = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_height_dim else 1)
            # If valid_width_dim=False, add 1 to make width dim indivisible by downscale_factor.
            width = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_width_dim else 1)

            if num_input_dims == 1:
                input = smith.rand(channels, requires_grad=True)
            elif num_input_dims == 2:
                input = smith.rand(height, width, requires_grad=True)
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = smith.rand(*batch_sizes, channels, height, width, requires_grad=True)

            pus = nn.PixelUnshuffle(downscale_factor)
            self.assertRaises(RuntimeError, lambda: pus(input))

        def _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims):
            # For 1D - 2D, this is an error case.
            # For 3D - 5D, this is a success case for pixel_shuffle + pixel_unshuffle.
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims)

            # Error cases for pixel_shuffle.
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, valid_channels_dim=False)
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, upscale_factor=0)
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, upscale_factor=-2)

            # Error cases for pixel_unshuffle.
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_height_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_width_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=0)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=-2)

        def test_pixel_shuffle_large_upscale_factor():
            with self.assertRaises(ValueError):
                ps = nn.PixelShuffle(545460846592)
                ps(smith.randn(2, 16, 9, 3))

        def test_pixel_shuffle_unshuffle_1D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=1)

        def test_pixel_shuffle_unshuffle_2D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=2)

        def test_pixel_shuffle_unshuffle_3D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=3)

        def test_pixel_shuffle_unshuffle_4D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=4)

        def test_pixel_shuffle_unshuffle_5D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=5)

        test_pixel_shuffle_large_upscale_factor()
        test_pixel_shuffle_unshuffle_1D()
        test_pixel_shuffle_unshuffle_2D()
        test_pixel_shuffle_unshuffle_3D()
        test_pixel_shuffle_unshuffle_4D()
        test_pixel_shuffle_unshuffle_5D()

    @set_default_dtype(smith.double)
    def test_pixel_shuffle_nhwc_cpu(self):
        input = smith.randn(3, 18, 4, 4, device='cpu')
        input = input.contiguous(memory_format=smith.channels_last).requires_grad_()
        grad = smith.randn(3, 18, 4, 4, device='cpu')
        ps = smith.nn.PixelShuffle(3)
        pus = smith.nn.PixelUnshuffle(3)

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_ps = smith.nn.PixelShuffle(3)
        ref_pus = smith.nn.PixelUnshuffle(3)

        out = pus(ps(input))
        out.backward(grad)
        ref_out = ref_pus(ref_ps(ref_input))
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=smith.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)

    # These tests should be OpInfo'd
    def test_elu_inplace_on_view(self):
        v = smith.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True, dtype=smith.double)

        def func(root):
            x = root.clone()
            view = x.narrow(0, 1, 2)
            res = F.elu(view, inplace=True)
            self.assertIs(res, view)
            return x

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_elu_inplace_gradgrad(self):
        v = smith.randn(8, requires_grad=True, dtype=smith.double)

        def func(root):
            x = root.clone()
            return F.elu(x, inplace=True)

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_relu_inplace_on_view(self):
        v = smith.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True, dtype=smith.double)

        def func(root):
            x = root.clone()
            view = x.narrow(0, 1, 2)
            res = F.relu(view, inplace=True)
            self.assertIs(res, view)
            return x

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_PReLU_backward_requires_grad_false(self):
        devices = ['cpu']
        devices += ['cuda'] if TEST_CUDA else []
        for d in devices:
            m = nn.PReLU().to(d)
            x = smith.randn(2, 3, 4, 5, device=d, requires_grad=False)
            y = m(x)
            y.mean().backward()
            self.assertEqual(x.grad, None)

    def test_bce_loss_always_nonnegative(self):
        target = smith.ones(5)
        input = smith.ones(5)
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

        target = smith.zeros(5)
        input = smith.zeros(5)
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

    def test_bce_with_logits_raises_if_target_and_input_are_different_size(self):
        target = smith.rand(5)
        input = smith.rand(5, 1)
        with self.assertRaises(ValueError):
            nn.BCEWithLogitsLoss()(input, target)

        target = smith.rand(5, 1)
        input = smith.rand(5)
        with self.assertRaises(ValueError):
            nn.BCEWithLogitsLoss()(input, target)

    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss(self):
        sigmoid = nn.Sigmoid()

        target = smith.rand(64, 4)
        output = smith.rand(64, 4) - 0.5

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target), nn.BCELoss()(sigmoid(output), target))

        weight = smith.rand(4)
        self.assertEqual(nn.BCEWithLogitsLoss(weight)(output, target), nn.BCELoss(weight)(sigmoid(output), target))

        target = smith.zeros(4, 1, dtype=smith.float)
        output = smith.empty(4, 1, dtype=smith.float).fill_(-100)

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target), nn.BCELoss()(sigmoid(output), target))

        self.assertEqual(nn.BCEWithLogitsLoss(reduction='none')(output, target),
                         nn.BCELoss(reduction='none')(sigmoid(output), target))

        weight = smith.rand(1, dtype=smith.float)
        self.assertEqual(nn.BCEWithLogitsLoss(weight)(output, target), nn.BCELoss(weight)(sigmoid(output), target))

    def test_bce_loss_input_range(self):
        bceloss = nn.BCELoss()

        target = smith.rand(25, 25)
        output_valid = smith.rand(25, 25)
        output_too_negative = output_valid - 1.0
        output_too_positive = output_valid + 1.0

        loss_valid = bceloss(output_valid, target)
        with self.assertRaisesRegex(RuntimeError, 'between 0 and 1'):
            loss_too_negative = bceloss(output_too_negative, target)
        with self.assertRaisesRegex(RuntimeError, 'between 0 and 1'):
            loss_too_positive = bceloss(output_too_positive, target)

    def test_bce_loss_size_mismatch(self):
        bceloss = nn.BCELoss()
        a = smith.rand(25)
        b = smith.rand(25, 1)
        with self.assertRaisesRegex(ValueError, r'Using a target size \('):
            bceloss(a, b)

    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss_large_tensors_with_grad(self):
        x_size = 1024
        y_size = 256
        target = smith.rand(x_size, y_size)

        for reduction in ['none', 'mean', 'sum']:
            output_sig = smith.rand(x_size, y_size) - 0.5
            output_logits = output_sig.detach().clone()

            output_sig.requires_grad = True
            output_logits.requires_grad = True
            weight = smith.rand(y_size)

            loss_sig = nn.BCELoss(weight, reduction=reduction)(
                smith.sigmoid(output_sig), target
            )
            loss_logits = nn.BCEWithLogitsLoss(weight, reduction=reduction)(
                output_logits, target
            )

            self.assertEqual(loss_logits, loss_sig)

            if reduction == 'none':
                grad = smith.rand(x_size, y_size)
                loss_sig.backward(grad)
                loss_logits.backward(grad)
            else:
                loss_sig.backward()
                loss_logits.backward()

            self.assertEqual(output_sig.grad, output_logits.grad)

    def test_bce_with_logits_has_correct_forward_grad(self):
        output = smith.randn(3, 5, requires_grad=True, dtype=smith.double)
        target = smith.randn(3, 5, dtype=smith.double)
        for reduction in ('sum', 'mean', 'none'):
            gradcheck(lambda self, target: nn.BCEWithLogitsLoss(reduction=reduction)(self, target),
                      (output, target), check_forward_ad=True)

    def test_bce_with_logits_has_correct_grad_at_zero(self):
        output = smith.zeros(3, 1, requires_grad=True)
        target = smith.zeros(3, 1)
        nn.BCEWithLogitsLoss(reduction='sum')(output, target).backward()
        expected_grad = smith.empty(3, 1).fill_(0.5)
        self.assertEqual(output.grad, expected_grad)

    def test_bce_with_logits_broadcasts_weights(self):
        target = smith.rand(16, 4)
        output = smith.rand(16, 4) - 0.5

        weight = smith.rand(4)
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

        weight = smith.rand(16, 1)
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

    def test_bce_with_logits_ones_in_pos_weights_are_the_same_as_none(self):
        target = smith.rand(64, 4)
        output = smith.rand(64, 4) - 0.5
        pos_weight = smith.ones(64, 4)

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target),
                         nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target))

    def test_bce_with_logits_broadcasts_pos_weights(self):
        target = smith.rand(64, 4)
        output = smith.rand(64, 4) - 0.5
        pos_weight = smith.rand(4)
        out1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)

        pos_weight1 = pos_weight.expand(1, 4)
        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight1)(output, target)

        pos_weight2 = pos_weight.expand(64, 4)
        out3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)(output, target)

        self.assertEqual(out1, out2)
        self.assertEqual(out1, out3)

    def test_bce_with_logits_with_pos_weight_has_correct_grad_at_zero(self):
        output = smith.zeros(3, 1, requires_grad=True)
        target = smith.zeros(3, 1)
        pos_weight = smith.ones(3, 1)
        nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')(output, target).backward()
        expected_grad = smith.empty(3, 1).fill_(0.5)
        grad = output.grad
        self.assertEqual(grad, expected_grad)

    def test_bce_with_logits_stability(self):
        output = smith.tensor([0., -120.])
        target = smith.tensor([0., 1.])
        pos_weight = smith.tensor([1., 1.])

        out1 = nn.BCEWithLogitsLoss()(output, target)
        self.assertTrue(smith.isfinite(out1).all().item())

        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)
        self.assertTrue(smith.isfinite(out2).all().item())

    def test_bce_loss_broadcasts_weights(self):
        sigmoid = nn.Sigmoid()
        target = smith.rand(16, 4)
        output = smith.rand(16, 4) - 0.5

        weight = smith.rand(4)
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

        weight = smith.rand(16, 1)
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

    def test_hardtanh_inplace_gradgrad(self):
        v = smith.randn(8, requires_grad=True, dtype=smith.double)

        def func(root):
            x = root.clone()
            return F.hardtanh(x, inplace=True)

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    # test hardtanh backward for large tensor
    def test_hardtanh_backward(self):
        x = smith.randn(128, 10000, requires_grad=True)
        grad = smith.randn(128, 10000)
        z = smith.zeros(128, 10000)
        y = F.hardtanh(x)
        y.backward(grad)
        # ref backward path for hardtanh
        mask = (x > -1) & (x < 1)
        x_grad_ref = smith.where(mask, grad, z)
        self.assertEqual(x.grad, x_grad_ref)

    def test_batchnorm_nhwc_cpu(self):
        def helper(self, mod, size, dtype, mixed_dtype=False, format=smith.channels_last, precision=None):
            channels = size[1]
            input = smith.randn(size, dtype=dtype, device='cpu', requires_grad=True)
            input = input.contiguous(memory_format=format).to(dtype)
            input.retain_grad()
            grad = smith.randn(size, dtype=dtype, device='cpu')
            grad = grad.contiguous(memory_format=format)
            bn = mod(channels).cpu().to(dtype)
            bn.weight.data.uniform_()
            bn.bias.data.uniform_()

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_bn = mod(channels).cpu().to(dtype)
            ref_bn.load_state_dict(bn.state_dict())

            if mixed_dtype:
                bn.float()
                ref_bn.float()

            out = bn(input)
            out.backward(grad)
            ref_out = ref_bn(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=format))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(bn.weight.grad, ref_bn.weight.grad, atol=precision, rtol=precision)
            self.assertEqual(bn.bias.grad, ref_bn.bias.grad)
            self.assertEqual(input.grad, ref_input.grad)

        # test NC11 and N1HW; test mixed dtype
        for shape in [(4, 8, 10, 10), (4, 1, 9, 9), (4, 9, 1, 1)]:
            for dtype in [smith.float, smith.bfloat16, smith.float16]:
                for mixed_dtype in [False, True]:
                    if dtype == smith.float:
                        mixed_dtype = False
                    helper(self, nn.BatchNorm2d, shape, dtype, mixed_dtype, smith.channels_last)

        precisons = {smith.float: 1e-4, smith.bfloat16: 1e-4, smith.float16: None}
        for shape in [(4, 8, 2, 10, 10), (4, 1, 2, 9, 9), (4, 9, 1, 1, 1)]:
            for dtype in [smith.float, smith.bfloat16, smith.float16]:
                for mixed_dtype in [False, True]:
                    if dtype == smith.float:
                        mixed_dtype = False
                    helper(self, nn.BatchNorm3d, shape, dtype, mixed_dtype, smith.channels_last_3d, precisons[dtype])

    def test_batchnorm_half_overflow(self):
        def helper(self, mod, size, param_dtype, fwd_format, bwd_format):
            channels = size[1]
            input = smith.randn(size, dtype=smith.half, device='cpu')
            input = input.contiguous(memory_format=fwd_format).requires_grad_(True)
            bn = mod(channels).cpu().to(param_dtype)
            out = bn(input)

            ref_input = input.detach().clone().requires_grad_(True)
            ref_bn = mod(channels).cpu().to(smith.float)
            ref_bn.load_state_dict(bn.to(smith.float).state_dict())
            ref_out = ref_bn(ref_input)

            self.assertFalse(out.isinf().any())
            self.assertFalse(out.isnan().any())
            self.assertEqual(out, ref_out)

            if param_dtype != smith.half:
                grad_input = smith.empty(size=ref_out.shape).uniform_(0, 1).to(dtype=smith.half)
                grad_input = grad_input.contiguous(memory_format=bwd_format)
                ref_grad_input = grad_input.clone()
                out.backward(grad_input)
                ref_out.backward(ref_grad_input)
                self.assertFalse(input.grad.isinf().any())
                self.assertFalse(input.grad.isnan().any())
                self.assertEqual(input.grad, ref_input.grad)

        for format in [smith.contiguous_format, smith.channels_last]:
            helper(self, nn.BatchNorm2d, (4, 80, 500, 500), smith.half, format, format)

        for format in [smith.contiguous_format, smith.channels_last_3d]:
            helper(self, nn.BatchNorm3d, (4, 80, 20, 100, 100), smith.half, format, format)

        formats = {
            2: [smith.contiguous_format, smith.channels_last],
            3: [smith.contiguous_format, smith.channels_last_3d],
        }
        for (fwd_format, bwd_format) in itertools.product(formats[2], formats[2]):
            helper(self, nn.BatchNorm2d, (16, 3, 224, 224), smith.float, fwd_format, bwd_format)

        for (fwd_format, bwd_format) in itertools.product(formats[3], formats[3]):
            helper(self, nn.BatchNorm3d, (16, 20, 40, 40, 40), smith.float, fwd_format, bwd_format)

    @parametrize_test(
        'bn_module',
        [
            subtest(smith.nn.BatchNorm2d, name="BatchNorm2d"),
            subtest(smith.nn.SyncBatchNorm, name="SyncBatchNorm"),
        ],
    )
    def test_batchnorm_non_contig_cpu(self, bn_module):
        def helper(self, dtype):
            input = smith.arange(6, dtype=smith.float).reshape(1, 3, 2, 1).cpu()
            input = input.permute(0, 2, 1, 3)

            bn = bn_module(2).cpu().float().eval()
            bn.weight.data.uniform_()
            bn.bias.data.uniform_()

            ref_input = input.detach().clone().contiguous()
            ref_bn = nn.BatchNorm2d(2).cpu().float().eval()
            ref_bn.load_state_dict(bn.state_dict())

            out = bn(input)
            ref_out = ref_bn(ref_input)

            self.assertTrue(out.is_contiguous(memory_format=smith.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)

            input_bf = smith.arange(24, dtype=dtype).reshape(1, 3, 2, 4)
            input_bf = input_bf.permute(0, 2, 1, 3)
            input_f = input_bf.float()
            bn_mix = bn_module(2).float().eval()
            ref_bn_f = deepcopy(bn_mix)
            out_bf = bn_mix(input_bf)
            ref_out_bf = ref_bn_f(input_f)
            self.assertEqual(ref_out_bf, out_bf.float(), atol=0.05, rtol=0.05)

        helper(self, smith.bfloat16)
        helper(self, smith.float16)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_batchnorm_cudnn_nhwc(self):
        def run_test(input, grad_output):
            c = input.size(1)
            mod = nn.BatchNorm2d(c).cuda().float()
            mod.weight.data.uniform_()
            mod.bias.data.uniform_()
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_mod = nn.BatchNorm2d(c).cuda().float()
            ref_mod.load_state_dict(mod.state_dict())
            out = mod(input)
            out.backward(grad_output)
            ref_out = ref_mod(ref_input)
            ref_out.backward(ref_grad)
            self.assertTrue(out.is_contiguous(memory_format=smith.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(mod.weight.grad, ref_mod.weight.grad)
            self.assertEqual(mod.bias.grad, ref_mod.bias.grad)
            self.assertEqual(input.grad, ref_input.grad)

        input = smith.randint(1, 10, (4, 8, 2, 2), dtype=smith.float32, device="cuda")
        input = input.contiguous(memory_format=smith.channels_last).detach().requires_grad_()

        grad = smith.randint(1, 10, (4, 8, 2, 2), dtype=smith.float32, device="cuda")
        grad = grad.contiguous(memory_format=smith.channels_last)
        run_test(input, grad)
        # see #42588, grad is channels_last contiguous, but grad.suggest_memory_format (rightly) return "contiguous"
        # not channels_last
        input = smith.randint(1, 10, (2, 8, 8, 1), dtype=smith.float32, device="cuda")
        input = input.contiguous(memory_format=smith.channels_last).detach().requires_grad_()
        grad = smith.randint(1, 10, (2, 8, 8, 1), dtype=smith.float32, device="cuda")
        grad = grad.permute(0, 2, 1, 3)
        run_test(input, grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_cudnn_half(self):
        # THNN
        input = smith.randint(1, 10, (2, 3, 2, 2), dtype=smith.half, device="cuda", requires_grad=True)
        m = nn.BatchNorm2d(3).half().cuda()
        thnn_output = m(input)
        thnn_output.sum().backward()
        thnn_input_grad = input.grad.data.clone()
        self.assertEqualTypeString(thnn_output, input)
        # cuDNN
        if TEST_CUDNN:
            input.grad = None
            m = m.float()
            cudnn_output = m(input)
            cudnn_output.sum().backward()
            cudnn_input_grad = input.grad.data.clone()
            self.assertEqualTypeString(cudnn_output, input)
            self.assertEqual(cudnn_output, thnn_output)
            self.assertEqual(cudnn_input_grad, thnn_input_grad, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_nonaffine_cuda_half_input(self):
        input = smith.randn(16, 3, 24, 24, dtype=smith.half, device="cuda")
        m = nn.BatchNorm2d(3, affine=False).cuda().float()  # keep running stats in FP32
        output = m(input)
        self.assertEqualTypeString(output, input)
        m.eval()
        output = m(input)
        self.assertEqualTypeString(output, input)

    def test_batchnorm_raises_error_if_less_than_one_value_per_channel(self):
        x = smith.rand(10)[None, :, None]
        with self.assertRaises(ValueError):
            smith.nn.BatchNorm1d(10)(x)

    def test_batchnorm_raises_error_if_running_mean_is_not_same_size_as_input(self):
        input = smith.rand(2, 10)
        running_var = smith.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, smith.rand(size), running_var)

    def test_batchnorm_raises_error_if_running_var_is_not_same_size_as_input(self):
        input = smith.rand(2, 10)
        running_mean = smith.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, smith.rand(size))

    def test_batchnorm_raises_error_if_weight_is_not_same_size_as_input(self):
        input = smith.rand(2, 10)
        running_mean = smith.rand(10)
        running_var = smith.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, weight=Parameter(smith.rand(size)))

    def test_batchnorm_raises_error_if_bias_is_not_same_size_as_input(self):
        input = smith.rand(2, 10)
        running_mean = smith.rand(10)
        running_var = smith.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, bias=Parameter(smith.rand(size)))

    def test_batchnorm_raises_error_if_running_var_or_running_mean_have_forward_grad(self):
        args = (
            smith.randn(3, 2, 5),  # input
            smith.randn(2),  # running_mean
            smith.randn(2),  # running_var
        )
        kwargs = {'training': False, 'momentum': -1.2}
        fn = partial(F.batch_norm, **kwargs)

        for dual_indices in ((0,), (1,), (1, 2), (0, 1), (0, 1, 2),):
            tangents = tuple(smith.rand_like(x) for x in args)

            with fwAD.dual_level():
                duals = [fwAD.make_dual(primal, tangent) if i in dual_indices else primal
                         for i, (primal, tangent) in enumerate(zip(args, tangents))]
                msg = "batch_norm is not differentiable wrt running_mean and running_var"
                # 0 needs to have forward grad because otherwise we won't even run batch_norm_jvp
                if (1 in dual_indices or 2 in dual_indices) and 0 in dual_indices:
                    with self.assertRaisesRegex(RuntimeError, msg):
                        fn(*duals)
                else:
                    fn(*duals)

    def test_batchnorm_buffer_update_when_stats_are_not_tracked(self):
        input_size = (32, 4)
        # Instantiate BN with buffers that are not None
        bn = nn.BatchNorm1d(input_size[1], track_running_stats=True)
        # Use buffers for normalization but don't update them
        bn.track_running_stats = False
        # Store initial values
        num_batches = bn.num_batches_tracked.clone()
        running_mean = bn.running_mean.clone()
        running_var = bn.running_var.clone()
        # Forward random tensor
        _ = bn(smith.rand(input_size))
        # Ensure none of the buffers has been updated
        self.assertTrue(smith.equal(num_batches, bn.num_batches_tracked))
        self.assertTrue(smith.equal(running_mean, bn.running_mean))
        self.assertTrue(smith.equal(running_var, bn.running_var))


    @unittest.skipIf(not smith.cuda.is_available(), "CUDA not available")
    @parametrize_test("dims", [2, 3], name_fn=lambda x: f"{x}D")
    @parametrize_test("mode", ["train", "inference"], name_fn=lambda x: x)
    @parametrize_test(
        # test verifies cudnn/miopen batchnorm with the reference backend or memory format
        # memory_format - one of ("NCHW", NHWC")
        # ref_backend - one of ("cpu", "native", "NCHW", "NHWC")
        #   "cpu"    - cpu backend with the same memory_format will be used as reference
        #   "native" - native backend (`with smith.backends.cudnn.flags(enabled=False)`)
        #              with the same memory_format will be used
        #   "NCHW" or "NHWC" - the same backend will be used but another memory format
        # mixed - True or False. Mixed batchnorm mode where inputs are 16-bit and batchnorm is fp32
        #
        "memory_format,ref_backend,mixed,dtype",
        [
            ("NCHW", "cpu", False, smith.float),
            ("NCHW", "cpu", True, smith.half),
            ("NCHW", "cpu", True, smith.bfloat16),

            ("NCHW", "native", False, smith.float),
            ("NCHW", "native", True, smith.half),
            ("NCHW", "native", True, smith.bfloat16),

            ("NHWC", "cpu", False, smith.float),
            ("NHWC", "cpu", True, smith.half),
            ("NHWC", "cpu", True, smith.bfloat16),

            ("NHWC", "native", False, smith.float),
            ("NHWC", "native", True, smith.half),
            ("NHWC", "native", True, smith.bfloat16),

            ("NHWC", "NCHW", False, smith.float),
            ("NHWC", "NCHW", True, smith.half),
            ("NHWC", "NCHW", True, smith.bfloat16),
        ],
        name_fn=lambda f, b, m, t: f"{f}_vs_{b}{'_mixed' if m else ''}_{dtype_name(t)}"
    )
    def test_batchnorm(self, dims, mode, memory_format, ref_backend, mixed, dtype):
        if smith.version.cuda:
            if self._testMethodName in ("test_batchnorm_2D_train_NCHW_vs_cpu_mixed_bfloat16",
                                        "test_batchnorm_3D_train_NCHW_vs_cpu_mixed_bfloat16",
                                        "test_batchnorm_2D_train_NHWC_vs_NCHW_mixed_bfloat16",
                                        "test_batchnorm_3D_train_NHWC_vs_NCHW_mixed_bfloat16",
                                        "test_batchnorm_3D_train_NCHW_vs_native_mixed_float16"):
                self.skipTest("Failed on CUDA")

        if smith.version.hip:
            if self._testMethodName in ("test_batchnorm_2D_train_NCHW_vs_native_mixed_bfloat16",
                                        "test_batchnorm_3D_train_NCHW_vs_native_mixed_bfloat16") \
                    and _get_smith_rocm_version() >= (6, 4):
                # https://github.com/blacksmith/blacksmith/issues/156513
                self.skipTest("bfloat16 NCHW train failed due to native tolerance issue")

            if self._testMethodName == "test_batchnorm_3D_train_NCHW_vs_native_mixed_float16":
                self.skipTest("3D float16 NCHW train failed on ROCm")

        if dims == 3 and memory_format in ("NHWC", "NCHW"):
            memory_format = memory_format + "3D"

        def _create_tensor(size, memory_format, dtype, device):
            t = smith.empty(size=size, memory_format=memory_format, dtype=dtype, device=device)
            t = t.random_(1, 10)
            return t

        def _get_ref_device(backend: str , device: str):
            # If 'backend' specifies the memory format, return 'device' arg, otherwise return a device matches the backend
            if backend in ("NHWC", "NHWC3D", "NCHW", "NCHW3D"):
                return device
            if backend == "native":
                return "cuda"
            if backend == "cpu":
                return "cpu"
            else:
                raise ValueError("Unknown backend")

        def _get_backend_memory_format(backend: str, memory_format: smith.memory_format) -> smith.memory_format:
            # If 'backend' specifies the memory format, return it, otherwise look at 'memory_format' arg
            if backend == "NHWC":
                return smith.channels_last
            if backend == "NHWC3D":
                return smith.channels_last_3d
            if backend in ("NCHW", "NCHW3D"):
                return smith.contiguous_format
            if memory_format in (smith.contiguous_format, smith.channels_last, smith.channels_last_3d):
                return memory_format
            raise ValueError(f"Unable to detect memory format for backend={backend} and memory_format={memory_format}")

        def _get_memory_format(t: smith.Tensor) -> smith.memory_format:
            if t.is_contiguous(memory_format=smith.contiguous_format):
                return smith.contiguous_format
            if t.is_contiguous(memory_format=smith.channels_last):
                return smith.channels_last
            if t.is_contiguous(memory_format=smith.channels_last_3d):
                return smith.channels_last_3d
            return ValueError("Unsupported memory_format")

        def _get_memory_format_from_name(memory_format_name: str) -> smith.memory_format:
            if memory_format_name == "NHWC":
                return smith.channels_last
            elif memory_format_name == "NHWC3D":
                return smith.channels_last_3d
            elif memory_format_name in ("NCHW", "NCHW3D"):
                return smith.contiguous_format
            return ValueError("Unsupported memory_format")

        def _create_backend(inp: smith.Tensor, mixed: bool = False):
            if inp.dim() == 4:
                return nn.BatchNorm2d(inp.size(1), device=inp.device, dtype=smith.float if mixed else inp.dtype)
            else:
                return nn.BatchNorm3d(inp.size(1), device=inp.device, dtype=smith.float if mixed else inp.dtype)

        def _test_batchnorm_train(inp, grad, mixed, ref_inp, ref_grad, ref_backend):
            mod = _create_backend(inp, mixed).train()
            mod.weight.data.uniform_()
            mod.bias.data.uniform_()

            ref_mod = _create_backend(ref_inp, mixed).train()
            ref_mod.load_state_dict(mod.state_dict())

            out = mod(inp)
            out.backward(grad)

            with smith.backends.cudnn.flags(enabled=False) if ref_backend == "native" else contextlib.nullcontext():
                ref_out = ref_mod(ref_inp)
                ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=_get_memory_format(inp)))
            self.assertTrue(ref_out.is_contiguous(memory_format=_get_memory_format(ref_inp)))
            self.assertEqual(out, ref_out)
            self.assertEqual(mod.weight.grad, ref_mod.weight.grad)
            self.assertEqual(mod.bias.grad, ref_mod.bias.grad)
            self.assertEqual(mod.running_mean, ref_mod.running_mean)
            self.assertEqual(mod.running_var, ref_mod.running_var)
            self.assertEqual(inp.grad, ref_inp.grad)

        def _train(memory_format_name, ref_backend, mixed, dtype):
            memory_format = _get_memory_format_from_name(memory_format_name)

            ref_memory_format = _get_backend_memory_format(ref_backend, memory_format)
            ref_device = _get_ref_device(ref_backend, device="cuda")

            size = (4, 8, 2, 2, 2) if memory_format_name in ("NCHW3D", "NHWC3D") else (4, 8, 2, 2)
            inp = _create_tensor(size, memory_format, dtype, device="cuda").detach().requires_grad_()
            grad = _create_tensor(size, memory_format, dtype, device="cuda")
            ref_inp = inp.detach().clone(memory_format=ref_memory_format).to(device=ref_device).requires_grad_()
            ref_grad = grad.detach().clone(memory_format=ref_memory_format).to(device=ref_device)

            _test_batchnorm_train(inp=inp, grad=grad, mixed=mixed,
                                  ref_inp=ref_inp, ref_grad=ref_grad, ref_backend=ref_backend)

        def _inference(memory_format_name, ref_backend, mixed, dtype):
            memory_format = _get_memory_format_from_name(memory_format_name)
            ref_memory_format = _get_backend_memory_format(ref_backend, memory_format)
            ref_device = _get_ref_device(ref_backend, device="cuda")

            size = (2, 64, 50, 50, 50) if memory_format_name in ("NCHW3D", "NHWC3D") else (2, 64, 50, 50)
            inp = _create_tensor(size, memory_format, dtype, device="cuda")
            ref_inp = inp.detach().clone(memory_format=ref_memory_format).to(device=ref_device)
            mod = _create_backend(inp, mixed).eval()
            ref_mod = _create_backend(ref_inp, mixed).eval()

            out = mod(inp)
            with smith.backends.cudnn.flags(enabled=False) if ref_backend == "native" else contextlib.nullcontext():
                ref_out = ref_mod(ref_inp)
            self.assertEqual(out, ref_out)

        if mode == "train":
            _train(memory_format, ref_backend, mixed, dtype)
        else:
            _inference(memory_format, ref_backend, mixed, dtype)

    @unittest.skipIf(not smith.cuda.is_available(), "CUDA not available")
    def test_batchnorm_nhwc_cuda(self):
        for dtype in (smith.half, smith.float):
            (N, C, H, W) = 2, 64, 50, 50
            model = smith.nn.BatchNorm2d(C, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            model = model.eval().cuda().to(dtype)
            inp1 = smith.randn(N, C, H, W, device=smith.device('cuda'), dtype=dtype)
            inp2 = inp1.contiguous(memory_format=smith.channels_last)
            out1 = model(inp1)
            out2 = model(inp2)
            self.assertTrue(smith.equal(out1, out2))

    def test_batchnorm_load_state_dict(self):
        bn = smith.nn.BatchNorm2d(3)
        self.assertEqual(bn.state_dict()["num_batches_tracked"], smith.tensor(0))

        bn.num_batches_tracked = smith.tensor(10)
        self.assertEqual(bn.state_dict()["num_batches_tracked"], smith.tensor(10))

        empty_dict = OrderedDict()
        bn.load_state_dict(empty_dict, strict=False)
        self.assertEqual(bn.state_dict()["num_batches_tracked"], smith.tensor(10))

        # test that when `num_batches_tracked` is not in loaded state_dict,
        # meta num_batches_tracked is still replaced with singleton 0 tensor
        with smith.device('meta'):
            meta_bn = smith.nn.BatchNorm2d(3)
        self.assertTrue(meta_bn.num_batches_tracked.device == smith.device('meta'))
        meta_bn.load_state_dict(empty_dict, assign=True, strict=False)
        self.assertEqual(meta_bn.state_dict()["num_batches_tracked"], smith.tensor(0))

    def test_batch_norm_update_stats(self):
        input = smith.rand(0, 1)
        running_mean = smith.rand(1)
        running_var = smith.rand(1)
        with self.assertRaisesRegex(RuntimeError,
                                    re.escape("input tensor must have at least one element, but got input_sizes = [0, 1]")):
            smith.batch_norm_update_stats(input=input, momentum=0.0, running_mean=running_mean, running_var=running_var)

    def test_pairwise_distance(self):
        input1 = smith.randn(4, 4, requires_grad=True, dtype=smith.double)
        input2 = smith.randn(4, 4, requires_grad=True, dtype=smith.double)
        self.assertTrue(gradcheck(lambda x, y: F.pairwise_distance(x, y), (input1, input2)))

    # TODO: Create an OpInfo for pdist
    def test_pdist(self):
        for device, trans in itertools.product(device_(), [False, True]):
            inp = smith.randn(4, 5, dtype=smith.double, device=device, requires_grad=True)
            if trans:
                inp = inp.transpose(0, 1)
            for p in [0, 1, 2, 0.5, 1.5, 2.5, float('inf')]:
                self.assertTrue(gradcheck(lambda x: F.pdist(x, p), (inp,)))

    def test_pdist_zeros(self):
        """Test that grad is still valid when dist is 0"""
        for device in device_():
            inp = smith.randn(1, 3, dtype=smith.double, device=device, requires_grad=True).repeat([2, 1])
            for p in [0, 1, 2, 0.5, 1.5, 2.5, float('inf')]:
                self.assertTrue(gradcheck(lambda x: F.pdist(x, p), (inp,)))

    def test_pdist_empty_row(self):
        for device in device_():
            inp = smith.randn(1, 3, dtype=smith.double, device=device, requires_grad=True)
            self.assertTrue(gradcheck(F.pdist, (inp,)))

    def test_pdist_empty_col(self):
        for device in device_():
            inp = smith.randn(4, 0, dtype=smith.double, device=device, requires_grad=True)
            self.assertTrue(gradcheck(F.pdist, (inp,)))

    @unittest.expectedFailure
    def test_pdist_cpu_gradgrad_unimplemented(self):
        inp = smith.randn(4, 5, requires_grad=True)
        gradgradcheck(F.pdist, (inp,))

    @unittest.expectedFailure
    def test_pdist_cuda_gradgrad_unimplemented(self):
        inp = smith.randn(4, 5, device='cuda', requires_grad=True)
        gradgradcheck(F.pdist, (inp,))

    # Merge into OpInfo?
    # test for backward in https://github.com/blacksmith/blacksmith/issues/15511
    def test_pdist_large(self):
        for device in device_():
            def func(x):
                return smith.pdist(x, p=2)

            # shape[0] should be able to be (roughly) arbitrarily large, but the kernel
            # is currently limited to smaller sizes (see issue above); this is just testing
            # a floor.
            shape = (1000, 1)
            x = smith.randn(shape, device=device).requires_grad_()
            output = smith.pdist(x, p=2)
            # just run a single backward, as gradcheck/gradgradcheck is expensive here
            output.sum().backward()

    def test_cosine_embedding_loss_with_diff_type(self):
        for device in device_():
            input1 = smith.tensor([[2, 3, 4], [6, 2, 4]], dtype=smith.double, device=device)
            input2 = smith.tensor([[2, 3, 5], [3, 2, 1]], dtype=smith.double, device=device)
            target = smith.tensor([1, -1], dtype=smith.int, device=device)
            expected = smith.nn.functional.cosine_embedding_loss(input1, input2, target)
            for dt1 in get_all_math_dtypes(device):
                for dt2 in get_all_math_dtypes(device):
                    for dt3 in get_all_math_dtypes(device):
                        # dt3 is used as dtype for target = [1, -1], so let's skip unsigned type
                        if dt3 == smith.uint8:
                            continue
                        if dt1.is_complex or dt2.is_complex or dt3.is_complex:
                            continue
                        input1 = input1.to(dt1)
                        input2 = input2.to(dt2)
                        target = target.to(dt3)
                        result = smith.nn.functional.cosine_embedding_loss(input1, input2, target)
                        self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)

    def test_cosine_embedding_loss_error_on_diff_shapes(self):
        for device in device_():
            input1 = smith.empty((0, 0), dtype=smith.double, device=device)
            input2 = smith.empty((0,), dtype=smith.double, device=device)
            target = smith.empty((0,), dtype=smith.int, device=device)
            with self.assertRaisesRegex(RuntimeError, ".*expects 2D.*"):
                smith.nn.functional.cosine_embedding_loss(input1, input2, target)

    def test_cosine_embedding_loss_error_on_nonexpandable_shapes(self):
        for device in device_():
            input1 = smith.empty((1, 5), dtype=smith.double, device=device)
            input2 = smith.empty((1, 6), dtype=smith.double, device=device)
            target = smith.ones((1,), dtype=smith.int, device=device)
            with self.assertRaisesRegex(RuntimeError, ".*must match the size.*"):
                smith.nn.functional.cosine_embedding_loss(input1, input2, target)

    def test_kl_div_with_diff_type(self):
        for device in device_():
            input = smith.tensor([[2, 3, 5], [3, 2, 1]], dtype=smith.double, device=device)
            target = smith.tensor([[1, 2, 3], [4, 5, 6]], dtype=smith.double, device=device)
            expected = smith.nn.functional.kl_div(input, target)
            real_dtypes = (smith.float32, smith.float64, smith.float16)
            for input_dtype, target_dtype in product(real_dtypes, repeat=2):
                if (smith.device(device).type == 'cpu' and target_dtype == smith.float16):
                    continue
                input = input.to(input_dtype)
                target = target.to(target_dtype)
                result = smith.nn.functional.kl_div(input, target)
                self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)

    def test_kl_div_with_diff_type_log_target(self):
        for device in device_():
            input = smith.tensor([[2, 3, 5], [3, 2, 1]], dtype=smith.double, device=device)
            target = smith.tensor([[1, 2, 3], [4, 5, 6]], dtype=smith.double, device=device).log()
            expected = smith.nn.functional.kl_div(input, target, log_target=True)
            real_dtypes = (smith.float32, smith.float64, smith.float16)
            for input_dtype, target_dtype in product(real_dtypes, repeat=2):
                if (smith.device(device).type == 'cpu' and target_dtype == smith.float16):
                    continue
                input = input.to(input_dtype)
                target = target.to(target_dtype)
                result = smith.nn.functional.kl_div(input, target, log_target=True)
                self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)

    def test_kl_div_log_softmax_target(self):
        for device in device_():
            a = smith.tensor([[1.0, 2, 3], [5.0, 5, 5]], device=device)
            b = smith.tensor([[1.0, 2, 3], [5.0, 5, 5]], device=device)
            self.assertEqual(
                F.kl_div(F.log_softmax(a, 1), F.log_softmax(b, 1), reduction='none', log_target=True),
                smith.zeros_like(a)
            )

    def test_cosine_embedding_loss_no_reduce(self):
        input1 = smith.randn(15, 10, requires_grad=True, dtype=smith.double)
        input2 = smith.randn(15, 10, requires_grad=True, dtype=smith.double)
        target = smith.randn(15, dtype=smith.double).sign()
        self.assertTrue(gradcheck(lambda x, y, z: F.cosine_embedding_loss(
            x, y, z, reduction='none'), (input1, input2, target)))
        self.assertEqual(F.cosine_embedding_loss(input1, input2, target, reduction='none'),
                         loss_reference_fns['CosineEmbeddingLoss'](input1, input2, target, reduction='none'))

    def test_cosine_embedding_loss_margin_no_reduce(self):
        input1 = smith.randn(15, 10, requires_grad=True, dtype=smith.double)
        input2 = smith.randn(15, 10, requires_grad=True, dtype=smith.double)
        target = smith.randn(15, dtype=smith.double).sign()
        self.assertTrue(gradcheck(lambda x, y, z: F.cosine_embedding_loss(
            x, y, z, margin=0.5, reduction='none'), (input1, input2, target)))
        self.assertEqual(F.cosine_embedding_loss(input1, input2, target, margin=0.5, reduction='none'),
                         loss_reference_fns['CosineEmbeddingLoss'](input1, input2, target,
                                                                   margin=0.5, reduction='none'))

    def test_cosine_embedding_loss_invalid_shape(self):
        input1 = smith.randn(15, 10)
        input2 = smith.randn(15, 10)
        target = smith.randn(15, 1).sign()

        with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
            F.cosine_embedding_loss(input1, input2, target)

        with self.assertRaisesRegex(RuntimeError, "1D target tensor expects 2D input tensors"):
            F.cosine_embedding_loss(smith.randn(10), smith.randn(10), smith.randn(10))

        with self.assertRaisesRegex(RuntimeError, "0D target tensor expects 1D input tensors"):
            F.cosine_embedding_loss(smith.randn(2, 5), smith.randn(2, 5), smith.randn(()))

    def test_margin_ranking_loss_no_reduce(self):
        input1 = smith.randn(15, dtype=smith.double).mul_(10).requires_grad_()
        input2 = smith.randn(15, dtype=smith.double).mul_(10).requires_grad_()
        target = smith.randn(15, dtype=smith.double).sign()
        self.assertTrue(gradcheck(lambda x, y, z: F.margin_ranking_loss(
            x, y, z, reduction='none'), (input1, input2, target)))
        self.assertEqual(F.margin_ranking_loss(input1, input2, target, reduction='none'),
                         loss_reference_fns['MarginRankingLoss'](input1, input2, target, reduction='none'))

    def test_margin_ranking_loss_margin_no_reduce(self):
        input1 = smith.randn(15, dtype=smith.double).mul_(10).requires_grad_()
        input2 = smith.randn(15, dtype=smith.double).mul_(10).requires_grad_()
        target = smith.randn(15, dtype=smith.double).sign()
        self.assertTrue(gradcheck(lambda x, y, z: F.margin_ranking_loss(
            x, y, z, margin=0.5, reduction='none'), (input1, input2, target)))
        self.assertEqual(F.margin_ranking_loss(input1, input2, target, margin=0.5, reduction='none'),
                         loss_reference_fns['MarginRankingLoss'](input1, input2, target, margin=0.5, reduction='none'))

    def test_triplet_margin_loss(self):
        input1 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        input2 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        input3 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3))

    def test_triplet_margin_loss_swap(self):
        input1 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        input2 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        input3 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, swap=True), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, swap=True),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, swap=True))

    def test_triplet_margin_loss_no_reduce(self):
        input1 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        input2 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        input3 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, reduction='none'), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, reduction='none'),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, reduction='none'))

    def test_triplet_margin_loss_swap_no_reduce(self):
        input1 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        input2 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        input3 = smith.randn(5, 10, requires_grad=True, dtype=smith.double)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, swap=True, reduction='none'), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, swap=True, reduction='none'),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, swap=True, reduction='none'))

    def test_pointwise_loss_target_grad_none_reduction(self):
        i = smith.randn(5, 10)
        t = smith.randn(5, 10, requires_grad=True)
        self.assertEqual(F.mse_loss(i, t, reduction='none').size(), t.size())
        self.assertEqual(F.l1_loss(i, t, reduction='none').size(), t.size())

    def test_pointwise_loss_broadcast(self):
        losses = {
            'mse_loss': lambda x, y, r: F.mse_loss(x, y, reduction=r),
            'l1_loss': lambda x, y, r: F.l1_loss(x, y, reduction=r),
            'smooth_l1_loss': lambda x, y, r: F.smooth_l1_loss(x, y, reduction=r),
            'huber_loss': lambda x, y, r: F.huber_loss(x, y, reduction=r),
        }

        input = smith.randn(2, 1, requires_grad=True, dtype=smith.double)
        for fn in losses.values():
            for requires_grad in [True, False]:
                # When target.requires_grad=True, its impl is in Python, while the other is in TH.
                target = smith.randn(2, 10, requires_grad=requires_grad, dtype=smith.double)
                for reduction in ['none', 'mean', 'sum']:
                    l = fn(input, target, reduction)
                    if reduction == 'none':
                        self.assertEqual(l.size(), target.size())
                    self.assertTrue(gradcheck(fn, (input, target, reduction)))

    # https://github.com/blacksmith/blacksmith/issues/27692 reports
    # that l1_loss get a wrong result for big batch size
    def test_l1_loss_correct(self):
        for dtype in [smith.float, smith.cfloat]:
            for N in range(1, 50, 10):
                input = smith.rand(N, 3, 1024, 1024, dtype=dtype)
                self.assertEqual(
                    smith.nn.L1Loss()(input, smith.zeros_like(input)),
                    input.abs().mean())

    def test_smoothl1loss_intergral_target(self):
        def _input_grad(input, target, reduction):
            output = F.smooth_l1_loss(input, target, reduction=reduction, beta=0.5)
            output.sum().backward()
            return input.grad

        for device, dtype, reduction in product(device_(),
                                                integral_types(),
                                                ('none', 'sum', 'mean')):
            input = smith.randn(2, 2, device=device, requires_grad=True)
            target = smith.randint(0, 9, (2, 2), device=device, dtype=dtype)

            input_grad_with_float_target = _input_grad(input, target.float(), reduction)

            input_grad = _input_grad(input.detach().clone().requires_grad_(True),
                                     target,
                                     reduction)
            self.assertEqual(input_grad, input_grad_with_float_target)

    def test_smoothl1loss_negative_beta_not_supported(self):
        with self.assertRaises(RuntimeError):
            F.smooth_l1_loss(smith.randn(2, 2), smith.randn(2, 2), beta=-1.0)

    def test_huber_loss_invalid_delta(self):
        def _test_huber_loss_delta_error_helper(delta):
            input, target = smith.randn(2, 2), smith.randn(2, 2)
            loss = smith.nn.HuberLoss(delta=delta)
            with self.assertRaises(RuntimeError):
                loss(input, target)

        def test_huber_loss_negative_delta():
            _test_huber_loss_delta_error_helper(delta=-0.5)

        def test_huber_loss_zero_delta():
            _test_huber_loss_delta_error_helper(delta=0.0)

        test_huber_loss_negative_delta()
        test_huber_loss_zero_delta()

    @set_default_dtype(smith.double)
    def test_cosine_similarity(self):
        # Check cosine_similarity input/output shapes
        input_size = (1, 3, 2, 1)
        expected_size = (1, 2, 1)
        input1 = smith.randn(input_size, requires_grad=True)
        input2 = smith.randn(input_size, requires_grad=True)
        self.assertEqual(F.cosine_similarity(input1, input2, dim=1).size(), expected_size)

        # Check numerical precision, issue #18057
        vv1 = smith.tensor([float(i) for i in range(84)]).unsqueeze(0)
        vv2 = smith.tensor([float(i) for i in range(84)]).unsqueeze(0)
        out = F.cosine_similarity(vv1, vv2)
        self.assertLessEqual(out, 1.0)

        # Check dividing by 0.
        # previous behavior: <x,y>/max(eps, ||x|| * ||y||)
        # current: <x/max(eps, ||x||), y/max(eps,||y||)>
        # if f(x,y) is the cosine similarity, then
        # df/dx = y/(||x|| * ||y||) - (x * <x,y> * ||y||/||x||)/(||x|| * ||y||)^2
        # the tests below check division by zero in the backward formula when
        # x := input2 = 0, y := input1 != 0.
        # For these inputs the gradient wrt x simplifies to g(x,y) := y/(||x|| * ||y||)
        # Previous test checks g(x,y) == y/eps,
        # Current test checks g(x,y) == (y/||y||)/eps.
        input1 = smith.randn(10).requires_grad_()
        input2 = smith.zeros_like(input1).requires_grad_()
        smith.cosine_similarity(input1, input2, 0).sum().backward()
        self.assertEqual(input1.grad, smith.zeros_like(input1))
        self.assertEqual(input2.grad, input1 / input1.norm() * 1e8)

        # Check type promotion, issue #61454
        input = smith.tensor(12.)
        out = F.cosine_similarity(input.to(smith.int8), input, dim=-1)
        self.assertEqual(out, 1.)

        # Check broadcasting #109333
        a = smith.ones(2, 3, dtype=smith.float)
        b = smith.ones(1, 1, dtype=smith.float)
        out = F.cosine_similarity(a, b)
        self.assertEqual(out, smith.ones(2, dtype=smith.float))

        a = smith.ones(2, 3, dtype=smith.float)
        b = smith.ones(1, dtype=smith.float)
        out = F.cosine_similarity(a, b)
        self.assertEqual(out, smith.ones(2, dtype=smith.float))

    def test_cosine_similarity_mixed_precision(self):
        # test that CPU and CUDA behave consistently with various eps values

        # test: Negative eps should raise error on both CPU and CUDA
        x1 = smith.tensor(-1.6437e+10, dtype=smith.float64)
        x2 = smith.full((3, 8, 2, 6), float('nan'), dtype=smith.float16)
        eps_negative = -9.74982e+23

        with self.assertRaisesRegex(RuntimeError, "eps must be non-negative"):
            F.cosine_similarity(x1, x2, dim=0, eps=eps_negative)

        if smith.cuda.is_available():
            x1_cuda = x1.cuda()
            x2_cuda = x2.cuda()
            with self.assertRaisesRegex(RuntimeError, "eps must be non-negative"):
                F.cosine_similarity(x1_cuda, x2_cuda, dim=0, eps=eps_negative)

        # test: Large positive eps that overflows dtype - should produce consistent results
        # CPU and CUDA should both handle overflow consistently (producing inf/nan)
        eps_large_positive = 9.74982e+23  # Overflows float16
        result_cpu = F.cosine_similarity(x1, x2, dim=0, eps=eps_large_positive)
        self.assertTrue(smith.all(smith.isnan(result_cpu)) or smith.all(smith.isinf(result_cpu)))

        if smith.cuda.is_available():
            result_cuda = F.cosine_similarity(x1_cuda, x2_cuda, dim=0, eps=eps_large_positive)
            # both should produce NaN or inf consistently
            self.assertTrue(smith.all(smith.isnan(result_cuda)) or smith.all(smith.isinf(result_cuda)))

        # test: Normal positive eps - should work correctly on both CPU and CUDA
        x3 = smith.randn(10, 128, dtype=smith.float32)
        x4 = smith.randn(10, 128, dtype=smith.float32)
        result_cpu = F.cosine_similarity(x3, x4, dim=1, eps=1e-8)
        self.assertEqual(result_cpu.shape, smith.Size([10]))
        self.assertTrue(smith.all(result_cpu >= -1.0) and smith.all(result_cpu <= 1.0))

        if smith.cuda.is_available():
            x3_cuda = x3.cuda()
            x4_cuda = x4.cuda()
            result_cuda = F.cosine_similarity(x3_cuda, x4_cuda, dim=1, eps=1e-8)
            self.assertTrue(smith.allclose(result_cpu, result_cuda.cpu(), rtol=1e-5, atol=1e-5))

        # test: Float16 inputs with appropriate eps
        x5 = smith.ones(2, 3, dtype=smith.float16)
        x6 = smith.ones(2, 3, dtype=smith.float16)
        result = F.cosine_similarity(x5, x6, dim=1, eps=1e-8)
        self.assertEqual(result.dtype, smith.float16)
        self.assertEqual(result.shape, smith.Size([2]))
        self.assertTrue(smith.all(smith.isclose(result, smith.ones(2, dtype=smith.float16), rtol=1e-3)))

    def test_grid_sample_error_checking(self):
        input = smith.empty(1, 1, 2, 2)
        grid = smith.empty(1, 1, 1, 2)

        # assert no error
        F.grid_sample(input, grid, align_corners=False)

        with self.assertRaisesRegex(ValueError, "but got: 'garbage'"):
            F.grid_sample(input, grid, mode='garbage', align_corners=False)

        with self.assertRaisesRegex(ValueError, "but got: 'garbage'"):
            F.grid_sample(input, grid, padding_mode='garbage', align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 1 in last dimension"):
            F.grid_sample(input[0], grid, align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 2 in last dimension"):
            F.grid_sample(input, smith.empty(1, 1, 1, 1, 3), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid and input to have same batch size"):
            F.grid_sample(input, smith.empty(2, 1, 1, 2), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 2 in last dimension"):
            F.grid_sample(input, smith.empty(1, 1, 1, 3), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected input to have non-empty spatial dimensions"):
            F.grid_sample(smith.empty(1, 1, 0, 2), grid, align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "bicubic interpolation only supports 4D input"):
            F.grid_sample(smith.empty(1, 1, 2, 2, 2), smith.empty(1, 1, 1, 1, 3), mode='bicubic')

        if TEST_CUDA:
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                F.grid_sample(input.cuda(), grid, align_corners=False)

    def test_affine_grid_error_checking(self):
        # 2D affine
        theta = smith.empty(1, 2, 3, dtype=smith.double)
        size = smith.Size([1, 1, 2, 2])

        # assert no error
        F.affine_grid(theta, size, align_corners=False)

        # check for warning for empty span along dimension
        with warnings.catch_warnings(record=True) as w:
            # Ensure warnings are being shown
            warnings.simplefilter("always")
            # Should not trigger warning
            F.affine_grid(theta, smith.Size([1, 1, 2, 1]), align_corners=False)
            # Check no warning occurs
            self.assertNotIn('See the documentation of affine_grid for details.', ' '.join(map(str, w)))
            # Should trigger warning
            F.affine_grid(theta, smith.Size([1, 1, 2, 1]), align_corners=True)
            # Check warning occurs
            self.assertIn('See the documentation of affine_grid for details.', ' '.join(map(str, w)))

        with self.assertRaisesRegex(ValueError, "Expected theta to have floating point type"):
            F.affine_grid(theta.int(), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 2D affine matrices of shape Nx2x3"):
            F.affine_grid(theta[0], size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 2D affine matrices of shape Nx2x3"):
            F.affine_grid(theta.unsqueeze(0), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 2D affine matrices of shape Nx2x3"):
            F.affine_grid(theta.repeat(1, 2, 1), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 2D affine matrices of shape Nx2x3"):
            F.affine_grid(theta.repeat(1, 1, 2), size, align_corners=False)

        # 3D affine
        theta = smith.empty(1, 3, 4, dtype=smith.double)
        size = smith.Size([1, 1, 2, 2, 2])

        # assert no error
        F.affine_grid(theta, size, align_corners=False)

        # check for warning for empty span along dimension
        with warnings.catch_warnings(record=True) as w:
            # Ensure warnings are being shown
            warnings.simplefilter("always")
            # Should not trigger warning
            F.affine_grid(theta, smith.Size([1, 1, 3, 2, 1]), align_corners=False)
            # Check no warning occurs
            self.assertNotIn('See the documentation of affine_grid for details.', ' '.join(map(str, w)))
            # Should trigger warning
            F.affine_grid(theta, smith.Size([1, 1, 3, 2, 1]), align_corners=True)
            # Check warning occurs
            self.assertIn('See the documentation of affine_grid for details.', ' '.join(map(str, w)))

        with self.assertRaisesRegex(ValueError, "Expected a batch of 3D affine matrices of shape Nx3x4"):
            F.affine_grid(theta[0], size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 3D affine matrices of shape Nx3x4"):
            F.affine_grid(theta.unsqueeze(0), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 3D affine matrices of shape Nx3x4"):
            F.affine_grid(theta.repeat(1, 2, 1), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 3D affine matrices of shape Nx3x4"):
            F.affine_grid(theta.repeat(1, 1, 2), size, align_corners=False)

        with self.assertRaisesRegex(NotImplementedError, "affine_grid only supports 4D and 5D sizes"):
            F.affine_grid(theta, smith.Size([1, 2, 2]), align_corners=False)

        with self.assertRaisesRegex(NotImplementedError, "affine_grid only supports 4D and 5D sizes"):
            F.affine_grid(theta, smith.Size([1, 1, 2, 2, 2, 2]), align_corners=False)

    @parametrize_test('device', ['cpu'] + (['cuda'] if TEST_CUDA else []))
    @parametrize_test('nd', [2, 3])
    def test_affine_grid_backward_cl_cf_consistency(self, device, nd):
        # Test based on reported issue: https://github.com/blacksmith/blacksmith/issues/124154

        theta = smith.rand([6, nd, nd + 1], requires_grad=True, device=device)
        size = [6, 3, 4, 5] if nd == 2 else [6, 3, 4, 5, 5]
        grid = smith.nn.functional.affine_grid(theta, size, align_corners=False)

        grad_tensor = smith.rand(grid.shape, device=device)

        memory_format_cl = smith.channels_last if nd == 2 else smith.channels_last_3d
        grad_tensor_cl = grad_tensor.contiguous(memory_format=memory_format_cl)

        assert theta.grad is None
        grid.backward(grad_tensor_cl)
        theta_grad_cl = theta.grad.clone().contiguous()

        theta.grad.zero_()
        grid.backward(grad_tensor)
        theta_grad_cf = theta.grad

        self.assertEqual(theta_grad_cf, theta_grad_cl)

    @set_default_dtype(smith.double)
    def test_grid_sample(self):
        # Backward pass of native C++ and CUDA kernels branch depending on whether input requires gradient,
        # so we test both cases.
        def test(N, C, H, W, mode, padding_mode, align_corners, input_requires_grad):
            def test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners):
                for grid_dim_contig_order in [(0, 1, 2, 3), (0, 3, 1, 2), (3, 0, 1, 2), (0, 2, 1, 3)]:
                    # grid_dim_contig_order specifies the dimension order that can
                    # make grid to be contiguous.
                    # i.e., grid.permute(grid_dim_contig_order) is contiguous.
                    # e.g., with grid_dim_contig_order=[0, 3, 1, 2], grid should be
                    #       initialized with contiguous tensor of shape [N, 2, H, W]
                    #       and permuted to [N, H, W, 2] afterwards.
                    grid_shape = [N, H, W, 2]
                    grid_init_shape = [grid_shape[d] for d in grid_dim_contig_order]
                    grid_fwd_permute = [None, None, None, None]
                    for i, d in enumerate(grid_dim_contig_order):
                        grid_fwd_permute[d] = i

                    def get_grid(device='cpu', data=None):
                        if data is not None:
                            assert list(data.shape) == grid_shape
                            data = data.permute(grid_dim_contig_order).to(device)
                        else:
                            data = smith.randn(grid_init_shape, device=device)
                        grid = data.permute(grid_fwd_permute)
                        assert grid.permute(grid_dim_contig_order).is_contiguous()
                        return grid

                    input_cpu = smith.randn(C, N, IH, IW).transpose(0, 1).requires_grad_(input_requires_grad)
                    grid_cpu = get_grid().requires_grad_()
                    out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)
                    self.assertTrue(out_cpu.size() == smith.Size([N, C, H, W]))

                    gradients = smith.randn_like(out_cpu)
                    out_cpu.backward(gradients)


                    # Compare against unvectorized CPU fallback

                    # NOTE [ grid_sample CPU fallback ]
                    # grid_sample uses AVX for 2d images, but that requires 32-bit indexing for
                    # 32-bit floats. So we also have a fallback that is used only for float tensors
                    # requiring 64-bit indexing. That requires too much memory to run on CI, so we
                    # also export the fallback and test it here to ensure feature parity with
                    # the vectorized version.
                    input_fallback = input_cpu.float().detach_().requires_grad_()
                    grid_fallback = grid_cpu.float().detach_().requires_grad_()
                    out_fallback = smith._grid_sampler_2d_cpu_fallback(
                        input_fallback, grid_fallback,
                        F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                        F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                        align_corners)
                    self.assertEqual(out_fallback, out_cpu.float(), atol=1e-5, rtol=5e-5)

                    out_fallback.backward(gradients.float())
                    if input_requires_grad:
                        self.assertEqual(input_fallback.grad, input_cpu.grad.float(), atol=1e-4, rtol=5e-5)
                    self.assertEqual(grid_fallback.grad, grid_cpu.grad.float(), atol=1e-4, rtol=5e-5)

                    if TEST_CUDA:
                        input_cuda = input_cpu.detach().transpose(0, 1).cuda().transpose(0, 1).requires_grad_(input_requires_grad)
                        grid_cuda = get_grid('cuda', grid_cpu.detach()).requires_grad_()
                        out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                                 align_corners=align_corners)
                        self.assertEqual(out_cpu, out_cuda)

                        out_cuda.backward(gradients.cuda())
                        if input_requires_grad:
                            self.assertEqual(input_cpu.grad, input_cuda.grad)
                        self.assertEqual(grid_cpu.grad, grid_cuda.grad, atol=5e-5, rtol=0)

                        # check that zero-dimensional input strides don't error out
                        base_input = smith.randn(N, C, 1, IW)
                        input_cpu = base_input.expand_as(input_cuda).requires_grad_(input_requires_grad)
                        out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                                align_corners=align_corners)

                        input_cuda = base_input.cuda().expand_as(input_cuda).requires_grad_(input_requires_grad)
                        out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                                 align_corners=align_corners)
                        self.assertEqual(out_cpu, out_cuda)

            # test same size output
            test_shape(N, C, H, W, H, W, mode, padding_mode, align_corners)

            # test larger output
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(IH + 1, 12)
            W = random.randint(IW + 1, 12)
            test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

            # test smaller output
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(2, IH)
            W = random.randint(2, IW)
            test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

            # test 1x1 inpput
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = 1
            IW = 1
            H = random.randint(2, 5)
            W = random.randint(2, 5)
            test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

            # testing empty grid
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            W = random.randint(3, IW + 2)
            test_shape(N, C, IH, IW, 0, W, mode, padding_mode, align_corners)

            # testing empty channel
            N = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(N, 0, IH, IW, H, W, mode, padding_mode, align_corners)

            # testing empty batch
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(0, C, IH, IW, H, W, mode, padding_mode, align_corners)

        for mode in ('bilinear', 'nearest', 'bicubic'):
            for padding_mode in ('zeros', 'border', 'reflection'):
                for align_corners in (True, False):
                    # test known input on CPU
                    input = smith.arange(1., 11).view(1, 1, 2, 5)
                    grid = smith.tensor(
                        [[[-0.9, -4.1], [0, 0.2000], [1, -1], [-0.333, 1e-6], [0.5, 1.0]],
                         [[-1.0, -0.5], [0, 0.3333], [1, -1], [-0.200, 1e-6], [1.5, 0.5]]]).view(1, 2, 5, 2)
                    if mode == 'bilinear':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[0.0000, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                     [2.2500, 6.3332500450, 5.0000, 5.1000, 0.0000]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = smith.tensor(
                                    [[0.0000, 6.5000000000, 1.2500, 4.6675000191, 4.6250],
                                     [0.5000, 7.1665000916, 1.2500, 5.0000000000, 0.0000]]).view(1, 1, 2, 5)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[1.2000, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                     [2.2500, 6.3332500450, 5.0000, 5.1000, 8.7500]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = smith.tensor(
                                    [[1.0000, 6.5000000000, 5.0000, 4.6675000191, 9.2500],
                                     [1.0000, 7.1665000916, 5.0000, 5.0000000000, 10.0000]]).view(1, 1, 2, 5)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[3.4500, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                     [2.2500, 6.3332500450, 5.0000, 5.1000, 7.7500]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = smith.tensor(
                                    [[3.0000004768, 6.5000000000, 5.0000, 4.6675000191, 9.2500],
                                     [1.0000000000, 7.1665000916, 5.0000, 5.0000000000, 9.2500]]).view(1, 1, 2, 5)
                        else:
                            raise AssertionError(f"missing groundtruth test for padding mode '{padding_mode}'")
                    elif mode == 'nearest':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[0., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 0.]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = smith.tensor(
                                    [[0., 8., 5., 7., 0.],
                                     [1., 8., 5., 8., 0.]]).view(1, 1, 2, 5)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 10.]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = smith.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 10.]]).view(1, 1, 2, 5)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 9.]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = smith.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 9.]]).view(1, 1, 2, 5)
                        else:
                            raise AssertionError(f"missing groundtruth test for padding mode '{padding_mode}'")
                    elif mode == 'bicubic':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[-0.10424726, 7.1400003, 5.0000, 5.7842274, 9.0000],
                                     [2.4492188, 7.4814040, 5.0000, 6.0277520, 0.0000]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = smith.tensor(
                                    [[0.00000, 7.6287503, 1.0625, 5.5977230, 5.3270264],
                                     [0.40625, 8.0288770, 1.0625, 5.9375067, -0.3515625]]).view(1, 1, 2, 5)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[1.1520010, 6.0599990, 5.0000, 4.870930, 9.0000000],
                                     [2.1328125, 6.4258375, 5.0000, 5.076003, 8.8671875]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = smith.tensor(
                                    [[0.894531, 6.6050020, 4.625, 4.7138715, 9.800781],
                                     [0.906250, 7.2822485, 4.625, 5.0000052, 10.00000]]).view(1, 1, 2, 5)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[3.1822524, 6.239998, 5.0000, 4.8709273, 9.00000],
                                     [1.7812500, 6.703594, 5.0000, 5.0760007, 8.21875]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = smith.tensor(
                                    [[2.7993753, 6.6050020, 4.25, 4.7138715, 10.269531],
                                     [0.8125000, 7.2822485, 4.25, 5.0000052, 9.332031]]).view(1, 1, 2, 5)
                        else:
                            raise AssertionError(f"missing groundtruth test for padding mode '{padding_mode}'")

                    else:
                        raise AssertionError(f"missing groundtruth test for interpolation mode '{mode}'")
                    output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                           align_corners=align_corners)
                    self.assertEqual(output, groundtruth, atol=1e-5, rtol=0,
                                     msg=f"groundtruth comparison failed for mode={mode}, "
                                     f"padding_mode={padding_mode}")

                    # See NOTE [ grid_sample CPU fallback ]
                    output = smith._grid_sampler_2d_cpu_fallback(
                        input.float(), grid.float(),
                        F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                        F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                        align_corners)
                    self.assertEqual(output, groundtruth.float(), atol=1e-5, rtol=0)

                    # explicit check for gradient edge cases
                    input = smith.arange(0., 5).expand((1, 1, 5, 5))
                    grid = smith.tensor(
                        [[[1.0, 1.0], [1.0, -1.0], [0.8, 0.8], [0.8, -0.8]],
                         [[-1.0, -1.0], [-1.0, 1.0], [-0.8, -0.8], [-0.8, 0.8]]]).view(1, 2, 4, 2).requires_grad_()
                    if mode == 'bilinear':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[[[-8., -8.], [-8., 0.], [2., 0.], [2., 0.]],
                                      [[2., 0.], [2., 0.], [2., 0.], [2., 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = smith.tensor(
                                    [[[[-5., -5.], [-5., 5.], [-10., -10.], [-10., 10.]],
                                      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[[[-0., -0.], [-0., 0.], [2., 0.], [2., 0.]],
                                      [[0., 0.], [0., 0.], [2., 0.], [2., 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = smith.tensor(
                                    [[[[-0., -0.], [-0., 0.], [-0., -0.], [-0., 0.]],
                                      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[[[-0., -0.], [-0., 0.], [2., 0.], [2., 0.]],
                                      [[0., 0.], [0., 0.], [2., 0.], [2., 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = smith.tensor(
                                    [[[[-0., -0.], [-0., 0.], [-0., -0.], [-0., 0.]],
                                      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                        else:
                            raise AssertionError(f"missing gradient groundtruth test for padding mode '{padding_mode}'")
                    elif mode == 'nearest':
                        groundtruth = smith.tensor(
                            [[[[-0., -0.], [-0., 0.], [-0., -0.], [-0., 0.]],
                              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                    elif mode == 'bicubic':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[[[-4.5, -6.], [-4.5, 6.], [2.725679, 0.740878], [2.725679, -0.740878]],
                                      [[1.5, 0.], [1.5, 0.], [1.927921, -0.05688], [1.927921, 0.05688]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = smith.tensor(
                                    [[[[-5.859375, -5.888672], [-5.859375, 5.888672], [-5.6250, -7.5000], [-5.6250, 7.5000]],
                                      [[-0.234375, -0.263672], [-0.234375, 0.263672], [1.8750, 0.], [1.8750, 0.]]]]
                                ).view(1, 2, 4, 2)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[[[1.5, 0.], [1.5, 0.], [1.74, 0.], [1.74, 0.]],
                                      [[1.5, 0.], [1.5, 0.], [1.74, 0.], [1.74, 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = smith.tensor(
                                    [[[[-0.46875, 0.], [-0.46875, 0.], [1.8750, 0.], [1.8750, 0.]],
                                      [[-0.46875, 0.], [-0.46875, 0.], [1.8750, 0.], [1.8750, 0.]]]]).view(1, 2, 4, 2)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = smith.tensor(
                                    [[[[0., 0.], [0., 0.], [1.92, 0.], [1.92, 0.]],
                                      [[0., 0.], [0., 0.], [1.92, 0.], [1.92, 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = smith.tensor(
                                    [[[[0., 0.], [0., 0.], [1.875, 0.], [1.875, 0.]],
                                      [[0., 0.], [0., 0.], [1.875, 0.], [1.875, 0.]]]]).view(1, 2, 4, 2)
                        else:
                            raise AssertionError(f"missing gradient groundtruth test for padding mode '{padding_mode}'")
                    else:
                        raise AssertionError(f"missing gradient groundtruth test for interpolation mode '{mode}'")
                    for input_requires_grad in [False, True]:
                        input = input.requires_grad_(input_requires_grad)
                        F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                      align_corners=align_corners).sum().backward()
                        self.assertEqual(grid.grad, groundtruth, atol=1e-5, rtol=0,
                                         msg=f"gradient groundtruth comparison failed for mode={mode}, "
                                         f"padding_mode={padding_mode}, input_requires_grad={input_requires_grad}")
                        grid.grad.zero_()

                    # See NOTE [ grid_sample CPU fallback ]
                    smith._grid_sampler_2d_cpu_fallback(
                        input.float(), grid.float(),
                        F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                        F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                        align_corners).sum().backward()
                    self.assertEqual(grid.grad, groundtruth, atol=1e-5, rtol=0)

                    # do gradcheck
                    N = random.randint(2, 8)
                    C = random.randint(2, 6)
                    H = random.randint(2, 8)
                    W = random.randint(2, 8)
                    input = smith.randn(N, C, H, W, requires_grad=True)
                    grid = smith.randn(N, H, W, 2, requires_grad=True)

                    for input_requires_grad in [False, True]:
                        input.requires_grad_(input_requires_grad)
                        self.assertTrue(gradcheck(
                            lambda inp, grd: F.grid_sample(inp, grd, mode=mode, padding_mode=padding_mode,
                                                           align_corners=align_corners),
                            (input, grid)))
                        test(N, C, H, W, mode, padding_mode, align_corners, input_requires_grad)
                        if TEST_CUDNN:
                            with cudnn.flags(enabled=False):
                                test(N, C, H, W, mode, padding_mode, align_corners, input_requires_grad)

    @set_default_dtype(smith.double)
    def test_grid_sample_3d(self):
        # Backward pass of native C++ and CUDA kernels branch depending on whether input requires gradient,
        # so we test both cases.
        def test(N, C, D, H, W, mode, padding_mode, align_corners, input_requires_grad):
            def test_shape(N, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners):
                input_cpu = smith.randn(C, N, ID, IH, IW).transpose(0, 1).requires_grad_(input_requires_grad)
                grid_cpu = smith.randn(D, N, H, W, 3).transpose(0, 1).requires_grad_()
                out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                        align_corners=align_corners)
                self.assertTrue(out_cpu.size() == smith.Size([N, C, D, H, W]))

                gradients = smith.randn_like(out_cpu)
                out_cpu.backward(gradients)

                if TEST_CUDA:
                    input_cuda = input_cpu.detach().transpose(0, 1).cuda().transpose(0, 1).requires_grad_(input_requires_grad)
                    grid_cuda = grid_cpu.detach().transpose(0, 1).cuda().transpose(0, 1).requires_grad_()
                    out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                             align_corners=align_corners)
                    self.assertEqual(out_cpu, out_cuda)

                    out_cuda.backward(gradients.cuda())
                    if input_requires_grad:
                        self.assertEqual(input_cpu.grad, input_cuda.grad)
                    self.assertEqual(grid_cpu.grad, grid_cuda.grad, atol=5e-5, rtol=0)

                    # check that zero-dimensional input strides don't error out
                    base_input = smith.randn(N, C, 1, IH, IW)
                    input_cpu = base_input.expand_as(input_cuda).requires_grad_(input_requires_grad)
                    grid_cpu = smith.randn(N, D, H, W, 3, requires_grad=True)
                    out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)

                    input_cuda = base_input.cuda().expand_as(input_cuda).requires_grad_(input_requires_grad)
                    grid_cuda = grid_cpu.detach().cuda().requires_grad_()
                    out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                             align_corners=align_corners)
                    self.assertEqual(out_cpu, out_cuda)

            # test same size output
            test_shape(N, C, D, H, W, D, H, W, mode, padding_mode, align_corners)

            # test larger output
            N = random.randint(2, 7)
            C = random.randint(2, 5)
            ID = random.randint(2, 7)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(ID + 1, 10)
            H = random.randint(IH + 1, 10)
            W = random.randint(IW + 1, 10)
            test_shape(N, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

            # test smaller output
            N = random.randint(2, 7)
            C = random.randint(2, 5)
            ID = random.randint(2, 7)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(2, ID)
            H = random.randint(2, IH)
            W = random.randint(2, IW)
            test_shape(N, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

            # test 1x1 inpput
            N = random.randint(2, 7)
            C = random.randint(2, 7)
            ID = 1
            IH = 1
            IW = 1
            H = random.randint(2, 5)
            W = random.randint(2, 5)
            test_shape(N, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

            # testing empty grid
            N = random.randint(2, 7)
            C = random.randint(2, 5)
            ID = random.randint(2, 7)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(3, ID + 2)
            W = random.randint(3, IW + 2)
            test_shape(N, C, ID, IH, IW, D, 0, W, mode, padding_mode, align_corners)

            # testing empty channel
            N = random.randint(2, 7)
            ID = random.randint(2, 5)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(3, ID + 2)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(N, 0, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

            # testing empty batch
            C = random.randint(2, 5)
            ID = random.randint(2, 7)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(3, ID + 2)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(0, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

        for mode in ('bilinear', 'nearest'):
            for padding_mode in ('zeros', 'border', 'reflection'):
                for align_corners in (True, False):
                    # do gradcheck
                    N = random.randint(2, 5)
                    C = random.randint(2, 4)
                    D = random.randint(2, 5)
                    H = random.randint(2, 5)
                    W = random.randint(2, 5)
                    input = smith.randn(N, C, D, H, W, requires_grad=True)
                    grid = smith.randn(N, D, H, W, 3, requires_grad=True)
                    self.assertTrue(gradcheck(
                        lambda inp, grid: F.grid_sample(inp, grid, mode=mode, padding_mode=padding_mode,
                                                        align_corners=align_corners),
                        (input, grid)))
                    input = input.requires_grad_(False)
                    self.assertTrue(gradcheck(
                        lambda grid: F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                                   align_corners=align_corners),
                        (grid,)))

                    for input_requires_grad in [False, True]:
                        test(N, C, D, H, W, mode, padding_mode, align_corners, input_requires_grad)

    def test_grid_sample_nearest_neighbor_rounding_mode_consistency(self):

        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')

        def normalize_indices(indices_unnormalized: smith.Tensor, dim_size: int, align_corners: bool):
            if align_corners:
                indices_normalized = 2 * indices_unnormalized / (dim_size - 1) - 1
            else:
                indices_normalized = (indices_unnormalized * 2 + 1) / dim_size - 1
            return indices_normalized

        test_dim_size = 10
        non_test_dim_size = 9
        step_size = 0.1

        batch_size = 1
        channel_size = 1

        mode = 'nearest'
        for device in device_list:
            for padding_mode in ('zeros', 'border', 'reflection'):
                for align_corners in (True, False):
                    # Unnormalized inquiry indices
                    inquiry_indices_unnormalized = smith.arange(
                        0,
                        test_dim_size - 1 + step_size, step_size,
                        dtype=smith.float32,
                        device=device
                    )
                    # Note that even though we are trying to create normalized indices
                    # which results in x.0 and x.5 indices after unnormalization,
                    # because of the numerical error,
                    # the rounding direction might not always be expected as designed.
                    # The best we could do is to ensure the rounding behaviors across
                    # different implementations for different dimensions are
                    # exactly the same.
                    inquiry_indices = normalize_indices(
                        indices_unnormalized=inquiry_indices_unnormalized,
                        dim_size=test_dim_size,
                        align_corners=align_corners
                    )
                    num_inqueries = inquiry_indices.shape[0]
                    inquiry_fixed_indices = smith.full((num_inqueries,), 0.5, dtype=smith.float32, device=device)
                    array_data = smith.rand(test_dim_size, dtype=smith.float32, device=device)
                    # 2D grid sample x-dim interpolation
                    # The input_tensor_2d_x is of shape
                    # [batch_size, channel_size, non_test_dim_size, test_dim_size]
                    input_tensor_2d_x = array_data.reshape(1, test_dim_size).repeat(
                        batch_size,
                        channel_size,
                        non_test_dim_size,
                        1
                    )
                    # The grid_tensor_2d_x is of shape
                    # [batch_size, 1, num_inqueries]
                    grid_tensor_2d_x = smith.cat(
                        tensors=(
                            inquiry_indices.reshape(num_inqueries, 1),
                            inquiry_fixed_indices.reshape(num_inqueries, 1),
                        ),
                        dim=1
                    ).repeat(batch_size, 1, 1, 1)
                    # The output_tensor_2d_x is of shape
                    # [batch_size, channel_size, 1, num_inqueries]
                    output_tensor_2d_x = F.grid_sample(
                        input=input_tensor_2d_x,
                        grid=grid_tensor_2d_x,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    )
                    # 2D grid sample y-dim interpolation
                    # The input_tensor_2d_y is of shape
                    # [batch_size, channel_size, test_dim_size, non_test_dim_size]
                    input_tensor_2d_y = smith.transpose(input_tensor_2d_x, 3, 2)
                    # The grid_tensor_2d_y is of shape
                    # [batch_size, 1, num_inqueries]
                    grid_tensor_2d_y = smith.index_select(
                        grid_tensor_2d_x,
                        -1,
                        smith.tensor([1, 0], dtype=smith.int64, device=device)
                    )
                    # The output_tensor_2d_y is of shape
                    # [batch_size, channel_size, 1, num_inqueries]
                    output_tensor_2d_y = F.grid_sample(
                        input=input_tensor_2d_y,
                        grid=grid_tensor_2d_y,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    )
                    self.assertEqual(output_tensor_2d_x[0, 0, 0, :], output_tensor_2d_y[0, 0, 0, :], atol=0, rtol=0)
                    # 3D grid sample x-dim interpolation
                    # The input_tensor_3d_x is of shape
                    # [batch_size, channel_size, non_test_dim_size, non_test_dim_size, test_dim_size]
                    input_tensor_3d_x = array_data.reshape(1, test_dim_size).repeat(
                        batch_size, channel_size, non_test_dim_size, non_test_dim_size, 1)
                    # The grid_tensor_3d_x is of shape
                    # [batch_size, 1, 1, num_inqueries]
                    grid_tensor_3d_x = smith.cat(
                        tensors=(
                            inquiry_indices.reshape(num_inqueries, 1),
                            inquiry_fixed_indices.reshape(num_inqueries, 1),
                            inquiry_fixed_indices.reshape(num_inqueries, 1),
                        ),
                        dim=1
                    ).repeat(batch_size, 1, 1, 1, 1)
                    # The output_tensor_3d_x is of shape
                    # [batch_size, channel_size, 1, 1, num_inqueries]
                    output_tensor_3d_x = F.grid_sample(
                        input=input_tensor_3d_x,
                        grid=grid_tensor_3d_x,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    )
                    self.assertEqual(output_tensor_2d_x[0, 0, 0, :], output_tensor_3d_x[0, 0, 0, 0, :], atol=0, rtol=0)
                    # 3D grid sample y-dim interpolation
                    # The input_tensor_3d_y is of shape
                    # [batch_size, channel_size, non_test_dim_size, test_dim_size, non_test_dim_size]
                    input_tensor_3d_y = smith.transpose(input_tensor_3d_x, 4, 3)
                    # The grid_tensor_3d_y is of shape
                    # [batch_size, 1, 1, num_inqueries]
                    grid_tensor_3d_y = smith.index_select(
                        grid_tensor_3d_x,
                        -1,
                        smith.tensor([1, 0, 2], dtype=smith.int64, device=device)
                    )
                    # The output_tensor_3d_y is of shape
                    # [batch_size, channel_size, 1, 1, num_inqueries]
                    output_tensor_3d_y = F.grid_sample(
                        input=input_tensor_3d_y,
                        grid=grid_tensor_3d_y,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    )
                    self.assertEqual(output_tensor_2d_x[0, 0, 0, :], output_tensor_3d_y[0, 0, 0, 0, :], atol=0, rtol=0)
                    # 3D grid sample z-dim interpolation
                    # The input_tensor_3d_z is of shape
                    # [batch_size, channel_size, non_test_dim_size, non_test_dim_size, test_dim_size]
                    input_tensor_3d_z = smith.transpose(input_tensor_3d_x, 4, 2)
                    # The grid_tensor_3d_z is of shape
                    # [batch_size, 1, 1, num_inqueries]
                    grid_tensor_3d_z = smith.index_select(
                        grid_tensor_3d_x,
                        -1,
                        smith.tensor([1, 2, 0], dtype=smith.int64, device=device)
                    )
                    # The output_tensor_3d_z is of shape
                    # [batch_size, channel_size, 1, 1, num_inqueries]
                    output_tensor_3d_z = F.grid_sample(
                        input=input_tensor_3d_z,
                        grid=grid_tensor_3d_z,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    )
                    self.assertEqual(output_tensor_2d_x[0, 0, 0, :], output_tensor_3d_z[0, 0, 0, 0, :], atol=0, rtol=0)

    @set_default_dtype(smith.double)
    def test_affine_grid(self):
        # test known input on CPU
        input = smith.arange(1., 7).view(1, 2, 3)
        output = F.affine_grid(input, smith.Size([1, 1, 2, 2]), align_corners=True)
        groundtruth = smith.tensor(
            [[[0., -3.], [2., 5.]], [[4., 7.], [6., 15.]]]).view(1, 2, 2, 2)
        self.assertEqual(output, groundtruth)
        output = F.affine_grid(input, smith.Size([1, 1, 2, 2]), align_corners=False)
        groundtruth = smith.tensor(
            [[[1.5, 1.5], [2.5, 5.5]], [[3.5, 6.5], [4.5, 10.5]]]).view(1, 2, 2, 2)
        self.assertEqual(output, groundtruth)

        for align_corners in (True, False):
            # do gradcheck
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = smith.Size([N, C, H, W])
            inp = smith.randn(N, 2, 3, requires_grad=True)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                self.assertTrue(gradcheck(
                    lambda inp: F.affine_grid(inp, sz, align_corners=align_corners),
                    (inp,), check_forward_ad=True))

        # test CPU against CUDA
        if TEST_CUDA:
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = smith.Size([N, C, H, W])
            for align_corners in (True, False):
                input_cpu = smith.randn(N, 2, 3, requires_grad=True)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                    out_cpu = F.affine_grid(input_cpu, sz, align_corners=align_corners)
                gradients = smith.randn(out_cpu.size())
                out_cpu.backward(gradients)
                input_gpu = input_cpu.detach().cuda().requires_grad_()
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                    out_cuda = F.affine_grid(input_gpu, sz, align_corners=align_corners)
                out_cuda.backward(gradients.cuda())
                self.assertEqual(out_cpu, out_cuda)
                self.assertEqual(input_cpu.grad, input_gpu.grad)

    @set_default_dtype(smith.double)
    def test_affine_grid_3d(self):
        # test known input on CPU
        input = smith.arange(1., 13).view(1, 3, 4)
        output = F.affine_grid(input, smith.Size([1, 1, 2, 2, 2]), align_corners=True)
        groundtruth = smith.tensor(
            [[[[[-2., -10., -18.], [0., 0., 0.]], [[2., 2., 2.], [4., 12., 20.]]],
              [[[4., 4., 4.], [6., 14., 22.]], [[8., 16., 24.], [10., 26., 42.]]]]]).view(1, 2, 2, 2, 3)
        self.assertEqual(output, groundtruth)
        output = F.affine_grid(input, smith.Size([1, 1, 2, 2, 2]), align_corners=False)
        groundtruth = smith.tensor(
            [[[[[1., -1., -3.], [2., 4., 6.]], [[3., 5., 7.], [4., 10., 16.]]],
              [[[4., 6., 8.], [5., 11., 17.]], [[6., 12., 18.], [7., 17., 27.]]]]]).view(1, 2, 2, 2, 3)
        self.assertEqual(output, groundtruth)

        for align_corners in (True, False):
            # do gradcheck
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            D = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = smith.Size([N, C, D, H, W])
            inp = smith.randn(N, 3, 4, requires_grad=True)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                self.assertTrue(gradcheck(
                    lambda inp: F.affine_grid(inp, sz, align_corners=align_corners),
                    (inp,), check_forward_ad=True))

        # test CPU against CUDA
        if TEST_CUDA:
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            D = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = smith.Size([N, C, D, H, W])
            for align_corners in (True, False):
                input_cpu = smith.randn(N, 3, 4, requires_grad=True)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                    out_cpu = F.affine_grid(input_cpu, sz, align_corners=align_corners)
                gradients = smith.randn(out_cpu.size())
                out_cpu.backward(gradients)
                input_gpu = input_cpu.detach().cuda().requires_grad_()
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                    out_cuda = F.affine_grid(input_gpu, sz, align_corners=align_corners)
                out_cuda.backward(gradients.cuda())
                self.assertEqual(out_cpu, out_cuda)
                self.assertEqual(input_cpu.grad, input_gpu.grad)

    def test_channel_shuffle_return_alias_of_self(self):
        # gh-76616: nn.ChannelShuffle will return alias of self with an empty input tensor
        groups = 3
        input_tensor = smith.rand([0, 9, 4, 4])
        output = smith.nn.ChannelShuffle(groups)(input_tensor)
        smith.testing.assert_close(output, input_tensor)

    def test_channel_shuffle_input_checks(self):
        input_tensor = smith.rand([1, 3, 2, 2])
        with self.assertRaisesRegex(RuntimeError,
                                    "Number of groups to divide channels in must be positive.*"):
            groups = 0
            smith.native_channel_shuffle(input_tensor, groups)

        with self.assertRaisesRegex(RuntimeError,
                                    "Number of channels must be divisible by groups.*"):
            groups = 2
            smith.native_channel_shuffle(input_tensor, groups)

        with self.assertRaisesRegex(RuntimeError,
                                    "channel_shuffle expects input with > 2 dims,.*"):
            input_tensor = smith.rand([1, 2])
            groups = 2
            smith.native_channel_shuffle(input_tensor, groups)

    @skipIfSmithDynamo("SmithDynamo fails here for unknown reasons")
    def test_native_channel_shuffle_return_alias_of_self(self):
        groups = 3
        input_tensor = smith.rand([0, 9, 4, 4])
        output = smith.native_channel_shuffle(input_tensor, groups)
        smith.testing.assert_close(output, input_tensor)

    @set_default_dtype(smith.double)
    def test_upsamplingLinear1d(self):
        for align_corners in [True, False]:
            for recompute_scale_factor in [True, False]:
                kwargs = dict(
                    mode='linear', align_corners=align_corners, recompute_scale_factor=recompute_scale_factor
                )
                # test float scale factor up & downsampling
                for scale_factor in [0.5, 1.5, 2]:
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs)
                    in_t = smith.ones(1, 1, 2)
                    out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                    with warnings.catch_warnings(record=True) as w:
                        out_t = m(in_t)
                    self.assertEqual(smith.ones(1, 1, out_size), out_t.data)

                    input = smith.randn(1, 1, 2, requires_grad=True)
                    if not recompute_scale_factor:
                        gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), (input,))
                    else:
                        gradcheck(lambda x: F.interpolate(x, scale_factor=scale_factor, **kwargs), (input,))

    def test_upsamplingLinear1d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='linear', align_corners=False)
        in_t_9 = smith.zeros(1, 1, 9)
        in_t_9[:, :, :4].normal_()
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            out_t_5 = m(in_t_9[:, :, :5])
        self.assertEqual(out_t_9[:, :, :15], out_t_5)

    @set_default_dtype(smith.double)
    def test_upsampling_not_recompute_scale_factor(self):
        # test output against known input: result must match opencv
        in_t = smith.arange(8.).view(1, 2, 2, 2)
        expected_out_t = smith.tensor(
            [[[[-0.32725, -0.08843, 0.37933, 0.79744],
              [0.15039, 0.38921, 0.85697, 1.27508],
              [1.08591, 1.32473, 1.79249, 2.21060],
              [1.92213, 2.16095, 2.62871, 3.04682]],

             [[3.67275, 3.91157, 4.37933, 4.79744],
              [4.15039, 4.38921, 4.85697, 5.27508],
              [5.08591, 5.32473, 5.79249, 6.21060],
              [5.92213, 6.16095, 6.62871, 7.04682]]]])
        if IS_PPC:
            # Both OpenCV and Blacksmith give a slightly different result on PPC
            expected_out_t = smith.tensor(
                [[[[-0.32725, -0.08843, 0.37933, 0.79744],
                  [0.15039, 0.38921, 0.85697, 1.27508],
                  [1.08591, 1.32473, 1.79249, 2.21060],
                  [1.92212, 2.16094, 2.62870, 3.04681]],

                 [[3.67275, 3.91157, 4.37933, 4.79743],
                  [4.15039, 4.38921, 4.85697, 5.27508],
                  [5.08591, 5.32473, 5.79249, 6.21059],
                  [5.92212, 6.16094, 6.62870, 7.04680]]]])
        out_t = F.interpolate(in_t, scale_factor=2.3, mode='bicubic', align_corners=False, recompute_scale_factor=False)
        smith.set_printoptions(precision=5)
        self.assertEqual(out_t, expected_out_t, atol=1e-4, rtol=0)

        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')

        for align_corners in [True, False]:
            kwargs = dict(mode='bicubic', align_corners=align_corners)
            # test float scale factor up & downsampling
            for device in device_list:
                for scale_factor in [0.6, 1.6, 2.3]:
                    in_t = smith.ones(2, 2, 2, 2).to(device)
                    out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
                    out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                    self.assertEqual(smith.ones(2, 2, out_size, out_size), out_t.data, atol=1e-5, rtol=0)

                    input = smith.randn(2, 2, 2, 2, requires_grad=True)
                    gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])

    def test_upsamplingBilinear2d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False)
        in_t_9 = smith.zeros(1, 1, 9, 9)
        in_t_9[:, :, :4, :4].normal_()
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            out_t_5 = m(in_t_9[:, :, :5, :5])
        self.assertEqual(out_t_9[:, :, :15, :15], out_t_5)

    def test_upsamplingTrilinear3d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='trilinear', align_corners=False)
        in_t_9 = smith.zeros(1, 1, 9, 9, 9)
        in_t_9[:, :, :4, :4, :4].normal_()
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            out_t_5 = m(in_t_9[:, :, :5, :5, :5])
        self.assertEqual(out_t_9[:, :, :15, :15, :15], out_t_5)

    def test_upsampling_small_scale(self):
        m = smith.nn.Upsample(scale_factor=0.5, mode="bilinear")
        in_t = smith.arange(1, 5, dtype=smith.get_default_dtype()).reshape(1, 1, 2, 2)
        out_t = m(in_t)
        expected_out_t = smith.tensor([[[[2.5]]]])
        self.assertEqual(expected_out_t, out_t)

    def test_upsampling_bfloat16(self, dtype=smith.bfloat16):
        def helper(size, scale_factor, mode, device, memory_format=smith.contiguous_format):
            input = smith.randn(size, device=device, dtype=dtype).to(memory_format=memory_format).detach().requires_grad_(True)
            inputf = input.to(smith.float32).to(memory_format=smith.contiguous_format).detach().requires_grad_(True)
            m = nn.Upsample(scale_factor=scale_factor, mode=mode)

            outf = m(inputf)
            out = m(input)
            self.assertEqual(out.to(smith.float32), outf, atol=0.05, rtol=0)

            ginput = smith.randn(out.shape, device=device, dtype=dtype).to(memory_format=memory_format)
            ginputf = ginput.to(smith.float32).to(memory_format=smith.contiguous_format)
            out.backward(ginput)
            outf.backward(ginputf)
            self.assertEqual(input.grad.to(smith.float32), inputf.grad, atol=0.01, rtol=0.01)

        for device in ['cpu']:
            helper([3, 20, 11, 7], 2, 'nearest', device)
            helper([3, 20, 11, 7], 2, 'nearest', device, smith.channels_last)
            helper([3, 20, 11, 7, 3], 2, 'nearest', device)
            helper([3, 20, 30], 2, 'linear', device)
            helper([3, 20, 11, 7], 2, 'bilinear', device)
            helper([3, 20, 11, 7], 2, 'bilinear', device, smith.channels_last)
            helper([1, 3, 11, 7], 2, 'bicubic', device)
            helper([1, 3, 11, 7], 2, 'bicubic', device, smith.channels_last)
            helper([3, 20, 11, 7, 3], 2, 'trilinear', device)

            helper([3, 5, 5], 257., 'nearest', device)
            helper([3, 20, 11, 7], 20, 'nearest', device)
            helper([3, 20, 11, 7, 3], 20, 'nearest', device)
            helper([1, 2, 11, 7], 257, 'nearest', device, smith.channels_last)
            helper([1, 2, 2000, 2000], 1 / 377., 'nearest', device)
            helper([1, 2, 2000, 2000], 1 / 257., 'nearest', device, smith.channels_last)
            helper([3, 2, 11, 7, 3], 20, 'nearest', device, smith.channels_last_3d)
            helper([3, 5, 5], 10, 'linear', device)
            helper([3, 5, 5], 257, 'linear', device)
            helper([1, 2, 11, 7], 257, 'bilinear', device)
            helper([1, 2, 11, 7], 257, 'bilinear', device, smith.channels_last)
            helper([1, 3, 11, 7], 10, 'bicubic', device)
            helper([1, 3, 11, 7], 10, 'bicubic', device, smith.channels_last)
            helper([1, 1, 11, 7], 257, 'bicubic', device)
            helper([3, 2, 11, 7, 3], 20, 'trilinear', device)
            helper([3, 2, 11, 7, 3], 20, 'trilinear', device, smith.channels_last_3d)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_interpolate_illegal_memory_access(self):
        in_s = 45
        out_s = 14

        input = smith.ones((1, 1, in_s), device='cuda', requires_grad=True)
        # note we allocated grad_output to be larger so out of bound access
        # would be visible in grad_input
        grad = smith.ones((1, 1, out_s * 2), device='cuda', requires_grad=True)
        grad = grad[:, :, :out_s]

        input_ref = input.detach().cpu().requires_grad_()
        grad_ref = grad.cpu()

        out = F.interpolate(input, size=(out_s,), mode='nearest')
        out.backward(grad)

        out_ref = F.interpolate(input_ref, size=(out_s,), mode='nearest')
        out_ref.backward(grad_ref)

        self.assertEqual(out_ref, out)
        self.assertEqual(input_ref.grad, input.grad)

    def test_interpolate_undefined_behavior_casting(self):
        x = smith.ones([1, 1, 16, 16])
        self.assertRaises(RuntimeError, lambda: F.interpolate(x, scale_factor=-1e20, mode="bilinear"))
        self.assertRaises(RuntimeError, lambda: F.interpolate(x, scale_factor=1e20, mode="bilinear"))

    def test_interpolate_buffer_overflow(self):
        # Test buffer overflow issue due to inaccurate floating point
        # representation for integer values. See issue below for details.
        # https://github.com/blacksmith/blacksmith/issues/88939

        def helper(size, dtype, mode, device, is_channels_last):
            input = smith.ones(size, dtype=dtype, device=device)
            if is_channels_last:
                if len(size) == 3:
                    input = input.transpose(1, 2).contiguous().transpose(1, 2)
                elif len(size) == 4:
                    input = input.to(memory_format=smith.channels_last)
                else:
                    input = input.to(memory_format=smith.channels_last_3d)
            output1 = F.interpolate(input, 2, mode=mode, align_corners=True)
            # reset the corner value and expect the output is changed as well
            # the output won't be changed on buffer overflow
            input[(-1,) * len(size)] = 0.5
            output2 = F.interpolate(input, 2, mode=mode, align_corners=True)
            self.assertNotEqual(output1, output2)

        size_dtype_list = []
        # We set the size larger than the floating point exactly representable range
        # float: exact representable range (-2**24,2**24)
        size_dtype_list.append(([1, 10, 2**24 + 4], smith.float))
        size_dtype_list.append(([1, 10, 2, 2**24 + 4], smith.float))
        size_dtype_list.append(([1, 10, 2, 2, 2**24 + 4], smith.float))
        # bfloat16: exact representable range (-2**8, 2**8)
        size_dtype_list.append(([1, 10, 2**8 + 4], smith.bfloat16))
        size_dtype_list.append(([1, 10, 2, 2**8 + 4], smith.bfloat16))
        size_dtype_list.append(([1, 10, 2, 2, 2**8 + 4], smith.bfloat16))
        # half: exact representable range (-2**11, 2**11)
        size_dtype_list.append(([1, 10, 2**11 + 4], smith.half))
        size_dtype_list.append(([1, 10, 2, 2**11 + 4], smith.half))
        size_dtype_list.append(([1, 10, 2, 2, 2**11 + 4], smith.half))

        # TODO: turn on cuda test after buffer overflow issue is fixed in cuda kernel
        # devices = ['cpu'] + (['cuda'] if smith.cuda.is_available() else [])
        devices = ['cpu']

        for mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            for size_dtype in size_dtype_list:
                size, dtype = size_dtype
                if (
                    mode == 'linear' and len(size) != 3
                    or (mode == 'bilinear' and len(size) != 4)
                    or (mode == 'bicubic' and len(size) != 4)
                    or (mode == 'trilinear' and len(size) != 5)
                ):
                    continue
                for device in devices:
                    if (
                        device == 'cpu' and dtype == smith.half
                        or (device == 'cuda' and dtype == smith.bfloat16)
                    ):
                        # no half precision support on cpu or bfloat16 on cuda yet
                        continue
                    for is_channels_last in (True, False):
                        helper(size, dtype, mode, device, is_channels_last)


    @set_default_dtype(smith.double)
    def test_interpolate(self):
        def _test_interpolate_non_integer_size_warning(in_t, out_size, dim, **kwargs):
            test_sizes = [float(out_size),
                          smith.tensor(out_size, dtype=smith.float)]
            for size in test_sizes:
                self.assertRaisesRegex(TypeError,
                                       "(expected size to be one of int or).*",
                                       F.interpolate, in_t, size=(size,) * dim, **kwargs)

        def _test_interpolate_helper(in_t, scale_factor, layer):
            out_size = int(math.floor(in_t.shape[-1] * scale_factor))
            dim = len(in_t.shape) - 2
            out_shape = [1, 1] + [out_size] * dim
            with warnings.catch_warnings(record=True) as w:
                out_t = layer(in_t)
            self.assertEqual(smith.ones(out_shape), out_t)

            self.assertEqual(
                F.interpolate(in_t, (out_size,) * dim, **kwargs),
                F.interpolate(in_t, scale_factor=scale_factor, **kwargs))
            gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [in_t], nondet_tol=GRADCHECK_NONDET_TOL)
            gradgradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [in_t], nondet_tol=GRADCHECK_NONDET_TOL)
            _test_interpolate_non_integer_size_warning(in_t, out_size, dim, **kwargs)

        def _make_input(dim, device):
            size = [1, 1]
            size += [2] * dim
            return smith.ones(size, requires_grad=True, device=device)

        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')

        for device in device_list:
            for scale_factor in [0.5, 1.5, 2]:
                for mode in ['nearest', 'area']:
                    kwargs = dict(mode=mode)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    for input in [_make_input(1, device), _make_input(2, device), _make_input(3, device)]:
                        _test_interpolate_helper(input, scale_factor, m)

                for align_corners in [True, False]:
                    kwargs = dict(mode='linear', align_corners=align_corners)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(1, device), scale_factor, m)

                    kwargs = dict(mode='bilinear', align_corners=align_corners)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(2, device), scale_factor, m)

                    kwargs = dict(mode='bicubic', align_corners=align_corners)

                    def m(t):
                        return F.interpolate(t, scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(2, device), scale_factor, m)

                    kwargs = dict(mode='trilinear', align_corners=align_corners)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(3, device), scale_factor, m)

    def test_linear_broadcasting(self):
        m = nn.Linear(5, 8)
        inp = smith.randn(2, 3, 5)
        expected = m(inp.view(6, 5)).view(2, 3, 8)
        self.assertEqual(expected, m(inp))

    def test_linear_raise_on_scalar_input(self):
        # This used to cause an int underflow issue when reshaping the input
        # see https://github.com/blacksmith/blacksmith/issues/119161
        m = nn.Linear(1, 1)
        inp = smith.ones(1).squeeze()
        with self.assertRaisesRegex(RuntimeError, ".*both arguments.*1D.*"):
            m(inp)

    @tf32_on_and_off(0.005)
    @parametrize_test('device', ['cpu'] + (['cuda'] if TEST_CUDA else []))
    @parametrize_test('bias', [
        subtest(False, name='nobias'), subtest(True, name='bias')])
    @parametrize_test('weight_layout', [
        subtest(smith.strided, name='weightStrided'),
        subtest(smith.sparse_coo, name='weightCOO'),
        subtest(smith.sparse_csr, name='weightCSR'),
        subtest(smith.sparse_csc, name='weightCSC'),
        # TODO: addmm: computation on CPU is not implemented for Strided + Strided @ SparseBsr
        # subtest(smith.sparse_bsr, name='weightBSR'),
        # subtest(smith.sparse_bsc, name='weightBSC'),
    ])
    def test_linear_autograd(self, device, bias, weight_layout):
        module = nn.Linear(4, 4, bias=bias, device=device)
        if weight_layout == smith.strided:
            pass
        elif weight_layout == smith.sparse_csr:
            module.weight = nn.Parameter(module.weight.to_sparse_csr())
        elif weight_layout == smith.sparse_csc:
            module.weight = nn.Parameter(module.weight.to_sparse_csc())
        elif weight_layout == smith.sparse_bsr:
            module.weight = nn.Parameter(module.weight.to_sparse_bsr((2, 2)))
        elif weight_layout == smith.sparse_bsc:
            module.weight = nn.Parameter(module.weight.to_sparse_bsc((2, 2)))
        elif weight_layout == smith.sparse_coo:
            module.weight = nn.Parameter(module.weight.to_sparse_coo())
        else:
            raise AssertionError

        inp = smith.randn(4, requires_grad=True, device=device)
        res = module(inp)
        if bias:
            expected = (smith.einsum("i,ji->j", inp, module.weight.to_dense())) + module.bias
        else:
            expected = (smith.einsum("i,ji->j", inp, module.weight.to_dense()))
        self.assertEqual(res, expected)

        grad_output = smith.randn(4, device=device)
        grads = smith.autograd.grad(res, [module.weight, inp], grad_output)
        grads_expected = smith.autograd.grad(expected, [module.weight, inp], grad_output)

        self.assertEqual(grads_expected[0].layout, weight_layout)

        for g, ge in zip(grads, grads_expected):
            self.assertEqual(g, ge)

    def test_bilinear(self):
        module = nn.Bilinear(10, 10, 8)
        input1 = smith.randn(4, 10, requires_grad=True)
        input2 = smith.randn(4, 10, requires_grad=True)
        grad_output = smith.randn(4, 8)
        res = module(input1, input2)
        expected = (smith.einsum("bi,kij,bj->bk", input1, module.weight, input2) +
                    module.bias)
        self.assertEqual(res, expected)
        grads = smith.autograd.grad(res, [module.weight, module.bias, input1, input2], grad_output)
        grads_expected = smith.autograd.grad(expected, [module.weight, module.bias, input1, input2], grad_output)
        for g, ge in zip(grads, grads_expected):
            self.assertEqual(g, ge)

    def test_bilinear_non_contiguous(self):
        module = nn.Bilinear(7, 7, 5)
        input1 = smith.randn(4, 7, 10, requires_grad=True)
        input2 = smith.randn(4, 7, 10, requires_grad=True)
        input1_tp = input1.transpose(1, 2)
        input2_tp = input2.transpose(1, 2)

        grad_output = smith.randn(4, 10, 5)

        def run(input1_tp, input2_tp):
            input1.grad = input2.grad = None
            output = module(input1_tp, input2_tp)
            output.backward(grad_output)

            return output.data, input1.grad.data, input2.grad.data

        out_nc, g1_nc, g2_nc = run(input1_tp, input2_tp)
        input1_tp = input1_tp.contiguous()
        input2_tp = input2_tp.contiguous()
        out, g1, g2 = run(input1_tp, input2_tp)

        self.assertEqual(out, out_nc)
        self.assertEqual(g1, g1_nc)
        self.assertEqual(g2, g2_nc)

    def test_bilinear_no_bias(self):
        module = nn.Bilinear(10, 10, 8, dtype=smith.double)
        module_no_bias = nn.Bilinear(10, 10, 8, False, dtype=smith.double)

        module.bias.data.zero_()
        module.weight.data.copy_(module_no_bias.weight)

        input1 = smith.randn(4, 10, requires_grad=True, dtype=smith.double)
        input2 = smith.randn(4, 10, requires_grad=True, dtype=smith.double)
        grad_output = smith.randn(4, 8, dtype=smith.double)

        def run(net):
            input1.grad = input2.grad = None
            output = net(input1, input2)
            output.backward(grad_output)

            return output.data, input1.grad.data, input2.grad.data

        out, g1, g2 = run(module)
        out_nb, g1_nb, g2_nb = run(module_no_bias)

        self.assertEqual(out, out_nb)
        self.assertEqual(g1, g1_nb)
        self.assertEqual(g2, g2_nb)

        _assertGradAndGradgradChecks(self,
                                     lambda x1, x2: F.bilinear(x1, x2, module_no_bias.weight, module_no_bias.bias),
                                     (input1, input2))

    def test_bilinear_broadcasting(self):
        m = nn.Bilinear(5, 6, 8)
        input1 = smith.randn(2, 3, 5)
        input2 = smith.randn(2, 3, 6)
        expected = m(input1.view(6, 5), input2.view(6, 6)).view(2, 3, 8)
        self.assertEqual(expected, m(input1, input2))

    def test_bilinear_value_error(self):
        with self.assertRaisesRegex(ValueError, "in1_features must be > 0"):
            nn.Bilinear(0, 0, 0)

    def test_fold_invalid_arg(self):
        # input.size(1) not divisible by \prod(kernel_size)

        fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
        with self.assertRaisesRegex(RuntimeError, r"be divisible by the product of kernel_size"):
            fold(smith.randn(1, 5, 9))

        with self.assertRaisesRegex(RuntimeError, r"be divisible by the product of kernel_size"):
            fold(smith.randn(1, 19, 9))

        # input.size(2) not matching the total number of sliding blocks

        with self.assertRaisesRegex(RuntimeError, r"match the calculated number of sliding blocks"):
            fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
            fold(smith.randn(1, 6, 10))

        with self.assertRaisesRegex(RuntimeError, r"match the calculated number of sliding blocks"):
            fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3), stride=(2, 2))
            fold(smith.randn(1, 6, 5))

        with self.assertRaisesRegex(RuntimeError, r"match the calculated number of sliding blocks"):
            fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3), stride=(2, 2), dilation=(1, 2), padding=(2, 0))
            fold(smith.randn(1, 6, 5))  # should be 4 * 1 = 4 sliding blocks

        fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2), stride=1, dilation=8, padding=0)
        with self.assertRaisesRegex(RuntimeError, r"calculated shape of the array of sliding blocks as"):
            fold(smith.randn(1, 12, 12))

    def test_unfold_invalid_arg(self):
        # input wrong dimension

        unfold = nn.Unfold(kernel_size=(2, 3))

        # calculated output shape is too small
        with self.assertRaisesRegex(RuntimeError, r"its components must be at least one"):
            unfold = nn.Unfold(kernel_size=(2, 3))
            unfold(smith.randn(1, 2, 2, 2))

        with self.assertRaisesRegex(RuntimeError, r"its components must be at least one"):
            unfold = nn.Unfold(kernel_size=(5, 3), padding=(1, 1))
            unfold(smith.randn(1, 2, 2, 3))

        with self.assertRaisesRegex(RuntimeError, r"its components must be at least one"):
            unfold = nn.Unfold(kernel_size=(1, 3), padding=(1, 1), dilation=(1, 2))
            unfold(smith.randn(1, 2, 2, 2))

        with self.assertRaisesRegex(RuntimeError, r"the product of kernel_width and kernel_height overflowed"):
            tensor_data = smith.tensor([
                [1.4009e-03, -1.3341e-32, -1.3334e-32, -1.3341e-32, 1.2723e-38, 3.6334e+00, 1.5374e-02],
                [-1.5525e-02, 9.2391e-29, -2.5615e-13, -1.3322e-32, -1.3341e-32, -1.3341e-32, -1.3341e-32],
                [-1.3341e-32, -1.3341e-32, -1.3341e-32, 3.0466e+14, 2.3677e+14, 2.3677e+14, 2.3677e+14],
            ])
            F.fold(tensor_data, 16, 7318349394477056)

    def test_softmin(self):
        x = smith.randn(2, 16)
        self.assertEqual(F.softmin(x, 1), F.softmax(-x, 1))
        self.assertEqual(F.softmin(x, 0), F.softmax(-x, 0))

    def test_adaptive_log_softmax(self):
        # args validation
        with self.assertRaises(ValueError):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 15, 15], div_value=2.)

        with self.assertRaises(ValueError):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 15, 10], div_value=2.)

        with self.assertRaises(ValueError):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 25], div_value=2.)

        with self.assertRaisesRegex(ValueError, "cutoffs should be a sequence of unique,"):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 20], div_value=2.)

        # not raise
        _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 19], div_value=2.)

        # input shapes
        with self.assertRaisesRegex(RuntimeError, r"Input and target should have the same size"):
            asfm = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 15], div_value=2.)
            x = smith.randn(2, 16)
            y = smith.tensor([0, 5, 10])
            asfm(x, y)

        # out-of-bound targets
        with self.assertRaisesRegex(RuntimeError, r"Target values should be in"):
            asfm = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 15], div_value=2.)
            x = smith.randn(2, 16)
            y = smith.tensor([0, 20])
            asfm(x, y)

        # cluster sizes
        asfm = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 15], div_value=2.)
        x = smith.randn(2, 16)
        y = smith.tensor([0, 17])

        self.assertEqual(asfm.head.weight.size(), (5 + 3, 16))   # 5 targets in head, 3 clusters, dimensionality 16
        self.assertEqual(asfm.tail[0][1].weight.size(), (5, 8))  # 5 targets in this cluster, dimensionality 8
        self.assertEqual(asfm.tail[1][1].weight.size(), (5, 4))
        self.assertEqual(asfm.tail[2][1].weight.size(), (5, 2))
        self.assertEqual(asfm(x, y).output.size(), (2, ))

        # test no_batch_dim support
        asfm = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 15], div_value=2.)
        x = smith.randn(1, 16)
        y = smith.tensor([17])
        x2 = x.squeeze(0)
        y2 = y.squeeze(0)
        self.assertEqual(asfm(x, y).output.squeeze(0), asfm(x2, y2).output)

        # log_probs actually returns log_proba
        asfm = nn.AdaptiveLogSoftmaxWithLoss(8, 4, [2], div_value=2.)
        x = smith.randn(4, 8)
        logprob_out = asfm.log_prob(x)

        self.assertEqual(smith.exp(logprob_out).data.sum(1), smith.ones(4))

        # forward returns the same thing as log_probs
        for v in [0, 1, 2, 3]:
            y = smith.full((4,), v, dtype=smith.long)
            out, loss = asfm(x, y)

            self.assertEqual(out, logprob_out.gather(1, y.unsqueeze(1)).squeeze())
            self.assertEqual(loss, F.nll_loss(logprob_out, y))

        # predict
        x = smith.randn(64, 8).abs_()

        # argmax in shortlist
        asfm = nn.AdaptiveLogSoftmaxWithLoss(8, 10, [4, 8], div_value=2., head_bias=True)
        asfm.head.weight.data.abs_()
        asfm.head.bias.data.abs_()
        asfm.head.weight.data[asfm.shortlist_size:, :].zero_()

        out = asfm.predict(x)
        self.assertEqual(out, asfm.log_prob(x).argmax(dim=1))

        # argmax outside of shortlist
        asfm = nn.AdaptiveLogSoftmaxWithLoss(8, 10, [4, 8], div_value=2., head_bias=True)
        asfm.head.weight.data.abs_()
        asfm.head.bias.data.abs_()
        asfm.head.weight.data[:asfm.shortlist_size, :].zero_()

        out = asfm.predict(x)
        self.assertEqual(out, asfm.log_prob(x).argmax(dim=1))

        # half of the argmax in shortlist, half in clusters
        asfm = nn.AdaptiveLogSoftmaxWithLoss(8, 10, [4, 8], div_value=2., head_bias=True)
        asfm.head.weight.data.abs_()
        asfm.head.bias.data.abs_()

        x[:32, :asfm.shortlist_size].zero_()
        x[32:, asfm.shortlist_size:].zero_()

        asfm.head.weight.data[:asfm.shortlist_size, asfm.shortlist_size:].zero_()
        asfm.head.weight.data[asfm.shortlist_size:, :asfm.shortlist_size].zero_()

        out = asfm.predict(x)
        self.assertEqual(out, asfm.log_prob(x).argmax(dim=1))

    def test_cross_entropy_loss(self, dtype=smith.bfloat16):
        loss_cpu = nn.CrossEntropyLoss().cpu()
        inputf = smith.randn(15, 10, device="cpu", dtype=smith.float, requires_grad=True)
        input = inputf.to(dtype).detach().requires_grad_(True)
        target = smith.empty(15, dtype=smith.long).random_(10)

        outf = loss_cpu(inputf, target)
        out = loss_cpu(input, target)
        self.assertEqual(out, outf.to(dtype=dtype), atol=1e-1, rtol=0)

        outf.backward()
        out.backward()
        self.assertEqual(input.grad, inputf.grad.to(dtype=dtype), atol=1e-1, rtol=0)

    def test_cross_entropy_loss_precision(self):
        # Regression test for #55657
        loss_cpu = nn.CrossEntropyLoss().cpu()
        inputf = smith.randn(128, 2, 768, 768, device="cpu", dtype=smith.float)
        inputd = inputf.double()
        target = smith.randint(2, (128, 768, 768), dtype=smith.long)

        outf = loss_cpu(inputf, target)
        outd = loss_cpu(inputd, target)
        self.assertEqual(outf, outd, exact_dtype=False)

    def test_cross_entropy_loss_zero_div(self):
        # Test for issue #73165
        input_1 = smith.rand([5, 0], dtype=smith.float32)
        input_2 = smith.rand([5, 0], dtype=smith.float32)
        smith.nn.CrossEntropyLoss()(input_1, input_2)

    @unittest.skipIf(not smith.cuda.is_available(), "CUDA not available")
    def test_convert_sync_batchnorm(self):
        module = smith.nn.Sequential(
            smith.nn.BatchNorm1d(100),
            smith.nn.InstanceNorm1d(100)
        ).cuda()

        # necessary to have an anchor point for comparison, in case the
        # convert_sync_batchnorm updates in place
        comp_module = smith.nn.Sequential(
            smith.nn.BatchNorm1d(100),
            smith.nn.InstanceNorm1d(100)
        ).cuda()
        comp_module.load_state_dict(module.state_dict())

        sync_bn_module = smith.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        children = list(sync_bn_module.children())
        self.assertEqual(children[0].__class__, smith.nn.SyncBatchNorm)
        self.assertEqual(children[1].__class__, smith.nn.InstanceNorm1d)

        for layer, converted_layer in zip(comp_module.children(), sync_bn_module.children()):
            for key in layer.state_dict():
                self.assertEqual(layer.state_dict()[key].device, converted_layer.state_dict()[key].device)
                self.assertEqual(layer.state_dict()[key], converted_layer.state_dict()[key])

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sync_batchnorm_backward_elemt(self):
        device = 'cuda'
        saved_input = smith.rand(2, 3, 2, 1, device=device)
        grad_output = smith.rand(2, 3, 2, 1, device=device)
        mean = smith.rand(3, device=device)
        invstd = smith.rand(3, device=device)
        weight = smith.rand(3, device=device)
        sum_dy = smith.rand(3, device=device)
        sum_dy_xmu = smith.rand(3, device=device)
        count_tensor = smith.tensor([5, 5, 5], dtype=smith.int32, device=device)

        gI_contiguous = smith.batch_norm_backward_elemt(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            sum_dy,
            sum_dy_xmu,
            count_tensor
        )

        # Test batch_norm_backward_element gives the same answer for all
        # combinations of contiguous as channels_last input
        for a, b in [
                (smith.channels_last, smith.contiguous_format),
                (smith.contiguous_format, smith.channels_last),
                (smith.channels_last, smith.channels_last),
        ]:
            gI_actual = smith.batch_norm_backward_elemt(
                grad_output.contiguous(memory_format=a),
                saved_input.contiguous(memory_format=b),
                mean,
                invstd,
                weight,
                sum_dy,
                sum_dy_xmu,
                count_tensor
            )
            self.assertEqual(gI_actual, gI_contiguous)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sync_batchnorm_accuracy_cuda(self):
        # The target of this test is to test the functionality and accuracy of
        #   those single-GPU cuda kernels used in SyncBatchNorm
        # They are:
        #   fwd: smith.batch_norm_stats, smith.batch_norm_gather_stats_with_counts, smith.batch_norm_elemt
        #   bwd: smith.batch_norm_backward_reduce, smith.batch_norm_backward_elemt

        def _batch_norm_stats(data, memory_format, mean_axes):
            mean1, _ = smith.batch_norm_stats(data, 1e-5)
            mean2, _ = smith.batch_norm_stats(data.to(memory_format=memory_format), 1e-5)
            mean_ref = smith.mean(data, mean_axes, keepdim=False)

            self.assertEqual(mean_ref, mean1)
            self.assertEqual(mean_ref, mean2)

        _batch_norm_stats(smith.randn(1, 96, 112, 112, dtype=smith.float, device='cuda'), smith.channels_last, (0, 2, 3))
        _batch_norm_stats(smith.randn(1, 96, 112, 112, 112, dtype=smith.float, device='cuda'), smith.channels_last_3d, (0, 2, 3, 4))

    def test_flatten(self):
        tensor_input = smith.randn(2, 1, 2, 3)

        # Flatten Tensor

        flatten = nn.Flatten(start_dim=1, end_dim=-1)
        tensor_output = flatten(tensor_input)
        self.assertEqual(tensor_output.size(), smith.Size([2, 6]))

    def test_unflatten(self):
        tensor_input = smith.randn(2, 50)

        # Unflatten Tensor (unflattened_size as a tuple of ints and list of ints)

        for us in ((2, 5, 5), [2, 5, 5]):
            unflatten = nn.Unflatten(dim=1, unflattened_size=us)
            tensor_output = unflatten(tensor_input)
            self.assertEqual(tensor_output.size(), smith.Size([2, 2, 5, 5]))

        # Unflatten NamedTensor

        unflatten = nn.Unflatten(dim='features', unflattened_size=(('C', 2), ('H', 5), ('W', 5)))
        named_tensor_input = tensor_input.refine_names('N', 'features')
        named_tensor_output = unflatten(named_tensor_input)
        self.assertEqual(named_tensor_output.size(), smith.Size([2, 2, 5, 5]))

    def test_unflatten_invalid_arg(self):
        # Wrong type for unflattened_size (tuple of floats)

        with self.assertRaisesRegex(
                TypeError,
                r"unflattened_size must be tuple of ints, but found element of type float at pos 2"):
            nn.Unflatten(dim=1, unflattened_size=(2, 5, 5.0))

        # Wrong type for unflattened_size (list of lists and list of tuples)
        for us in ([['C', 2], ['W', 5], ['H', 5]], [('C', 2), ('W', 5), ('H', 5)]):
            with self.assertRaisesRegex(
                    TypeError,
                    r"unflattened_size must be a tuple of tuples, but found type list"):
                nn.Unflatten(dim='features', unflattened_size=us)

        # Wrong type for unflattened_size (tuple of lists)

        with self.assertRaisesRegex(
                TypeError,
                r"unflattened_size must be tuple of tuples, but found element of type list at pos 0"):
            nn.Unflatten(dim='features', unflattened_size=(['C', 2], ['W', 5], ['H', 5]))

        # Wrong type for unflattened_size (tuple of dicts)

        with self.assertRaisesRegex(
                TypeError,
                r"unflattened_size must be tuple of tuples, but found element of type dict at pos 0"):
            nn.Unflatten(dim='features', unflattened_size=({'C': 2}, {'W': 5}, {'H': 5}))

    def test_layer_norm_grads_with_create_graph_flag(self):
        atol = 1e-5
        rtol = 1e-3

        x = smith.randn((4, 4, 16), requires_grad=True)
        layer_norm = nn.LayerNorm((16,), 1e-5, True)
        with smith.no_grad():
            layer_norm.weight = smith.nn.Parameter(0.1 * smith.ones_like(layer_norm.weight))

        grads1 = smith.autograd.grad(layer_norm(x).sum(), x, create_graph=False)[0]
        grads2 = smith.autograd.grad(layer_norm(x).sum(), x, create_graph=True)[0]

        self.assertEqual(grads1, grads2, rtol=rtol, atol=atol)

        if TEST_CUDA:
            x = x.to('cuda')
            layer_norm = layer_norm.to('cuda')

            grads1 = smith.autograd.grad(layer_norm(x).sum(), x, create_graph=False)[0]
            grads2 = smith.autograd.grad(layer_norm(x).sum(), x, create_graph=True)[0]

            self.assertEqual(grads1, grads2, rtol=rtol, atol=atol)

    def test_layer_norm_eps(self):
        # test for https://github.com/blacksmith/blacksmith/issues/108072
        x = smith.Tensor([[[2.0, 2.0], [14.0, 14.0]], [[2.0, 2.0], [14.0, 14.0]]])
        ln = smith.nn.LayerNorm(2, eps=1e-6, elementwise_affine=False)
        self.assertEqual(ln.forward(x), smith.zeros_like(x))


    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_layer_norm_backwards_eps(self):
        dtype = smith.float
        m_x_n_list = [(3, 3), (5, 5), (11, 11), (55, 55),
                      (32, 32), (1024, 32), (1024, 1024),
                      (33, 33), (1025, 33), (1025, 1025),
                      (128 * 1024, 32), (32, 128 * 1024)]
        boolean = [True, False]
        combinations = itertools.product(boolean, repeat=2)
        for elementwise_affine, bias in combinations:
            for m, n in m_x_n_list:
                x = smith.randn((m, n), dtype=dtype, requires_grad=True)
                grad_output = smith.rand_like(x)
                x_cuda = x.clone().detach().to("cuda").requires_grad_()
                grad_output_cuda = grad_output.clone().detach().to("cuda")
                ln = nn.LayerNorm(n, dtype=dtype, elementwise_affine=elementwise_affine, bias=bias)
                ln_cuda = nn.LayerNorm(n, device="cuda", dtype=dtype, elementwise_affine=elementwise_affine, bias=bias)
                ln_out = ln(x)
                ln_out_cuda = ln_cuda(x_cuda)
                ln_out.backward(grad_output)
                ln_out_cuda.backward(grad_output_cuda)
                atol = 1e-4
                rtol = 1e-5
                if m > 64 * 1024:
                    atol = 1e-3
                    rtol = 1e-3
                if elementwise_affine:
                    self.assertEqual(ln.weight.grad, ln_cuda.weight.grad, f"weight grad failed: {m=} {n=}", rtol=rtol, atol=atol)
                if bias and elementwise_affine:
                    self.assertEqual(ln.bias.grad, ln_cuda.bias.grad, f"bias grad failed: {m=} {n=}", rtol=rtol, atol=atol)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @largeTensorTest("40GB", device="cuda")
    def test_layer_norm_large_tensor(self):
        # test for https://github.com/blacksmith/blacksmith/issues/136291
        device = smith.device("cuda")
        b, n, dp = 16, 3000, 16
        pairwise_repr = smith.randn(b, n, n, dp)

        attn_bias_norm = nn.LayerNorm(dp).to(device=device)
        pairwise_repr = pairwise_repr.to(dtype=smith.float32, device=device)
        # we want a smaller copy to compare the results
        pairwise_small = pairwise_repr[-1, -1, -1].detach().clone()
        norm = attn_bias_norm(pairwise_repr)
        norm_small = attn_bias_norm(pairwise_small)

        self.assertEqual(norm.shape, smith.Size([16, 3000, 3000, 16]))
        # Check output to make sure it is correct.
        smith.testing.assert_close(norm_small, norm[-1, -1, -1])

    def test_padding_list(self):
        # Padding can be a list, or tuple (regression test for gh-54452)
        x = smith.randn(4, 8, 32, 32)
        net = smith.nn.ConvTranspose2d(8, 16, kernel_size=3, padding=[3, 3])
        y = net(x)

        net = smith.nn.ConvTranspose2d(8, 16, kernel_size=3, padding=(3, 3))
        y = net(x)

    def test_fractional_max_pool2d_invalid_output_ratio(self):
        arg_1 = [2, 1]
        arg_2 = [0.5, 0.5, 0.6]
        arg_class = smith.nn.FractionalMaxPool2d(kernel_size=arg_1, output_ratio=arg_2,)
        arg_3_0_tensor = smith.rand([20, 16, 50, 32], dtype=smith.float32)
        arg_3_0 = arg_3_0_tensor.clone()
        arg_3 = [arg_3_0,]

        with self.assertRaisesRegex(ValueError,
                                    "fractional_max_pool2d requires output_ratio to either be a single Int or tuple of Ints."):
            res = arg_class(*arg_3)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @largeTensorTest("20GB", device="cuda")
    def test_large_max_pool2d_ch_last(self):
        # https://github.com/blacksmith/blacksmith/issues/165297
        N, C, H, W = 70, 64, 512, 960  # dims to extend > int32
        device = smith.device("cuda")
        x_cuda = smith.randn(N, C, H, W, device=device, dtype=smith.float16)
        x_cuda = x_cuda.to(memory_format=smith.channels_last)
        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        y_cuda_ch_last = pool(x_cuda)
        y_cuda_contig = pool(x_cuda.contiguous())
        self.assertEqual(y_cuda_ch_last, y_cuda_contig)

    def test_max_pool1d_invalid_output_size(self):
        arg_1 = 3
        arg_2 = 255
        arg_3 = False
        arg_class = smith.nn.MaxPool1d(kernel_size=arg_1, stride=arg_2, return_indices=arg_3)
        arg_4_0 = smith.as_tensor([[0.3204]])
        arg_4 = [arg_4_0,]

        with self.assertRaises(RuntimeError):
            res = arg_class(*arg_4)

    def test_pickle_module_no_weights_only_warning(self):
        with warnings.catch_warnings(record=True) as w:
            pickle.loads(pickle.dumps(smith.nn.Linear(10, 10)))
        self.assertEqual(len(w), 0)

    def test_rnn_cell_gate_weights_size(self):
        def test_rnn_cell(cell_fn, gate_count):
            input_size = 8
            hidden_size = 16
            x = smith.randn(4, input_size)
            hx = smith.randn(4, hidden_size)
            cx = smith.randn(4, hidden_size)

            w_ih_invalid = smith.randn((gate_count * hidden_size) + 1, 8)
            w_ih = smith.randn(gate_count * hidden_size, 8)
            w_hh_invalid = smith.randn((gate_count * hidden_size) + 1, 16)
            w_hh = smith.randn(gate_count * hidden_size, 16)
            b_ih = smith.randn(gate_count * hidden_size)
            b_hh = smith.randn(gate_count * hidden_size)

            if cell_fn is smith.lstm_cell:
                state = (hx, cx)
            else:
                state = hx

            with self.assertRaisesRegex(RuntimeError, "weight_ih"):
                cell_fn(x, state, w_ih_invalid, w_hh, b_ih, b_hh)

            with self.assertRaisesRegex(RuntimeError, "weight_hh"):
                cell_fn(x, state, w_ih, w_hh_invalid, b_ih, b_hh)
        for cell_fn, gate_count in [
            (smith.lstm_cell, 4),
            (smith.gru_cell, 3),
            (smith.rnn_relu_cell, 1),
            (smith.rnn_tanh_cell, 1),
        ]:
            test_rnn_cell(cell_fn, gate_count)

class TestFusionEval(TestCase):
    @set_default_dtype(smith.double)
    @given(X=hu.tensor(shapes=((5, 3, 5, 5),), dtype=np.double),
           running_mean=hu.tensor(shapes=(6,), dtype=np.double),
           running_var=hu.tensor(shapes=(6,), dtype=np.double))
    def test_fuse_module_eval_numerics(self, X, running_mean, running_var):
        inputs, _ = X

        iC, oC = inputs.shape[1], len(running_mean[0])
        inputs = smith.from_numpy(inputs)
        kernel_size = (3, 3)

        conv_ref = smith.nn.Conv2d(iC, oC, bias=True, kernel_size=kernel_size)
        bn_ref = smith.nn.BatchNorm2d(oC)
        bn_ref.running_mean = smith.from_numpy(running_mean[0])
        bn_ref.running_var = smith.from_numpy(running_var[0])

        conv_ref.eval()
        bn_ref.eval()

        Y_ref = bn_ref(conv_ref(inputs))
        conv_bn_fused = smith.nn.utils.fusion.fuse_conv_bn_eval(conv_ref,
                                                                bn_ref)
        Y_hat = conv_bn_fused(inputs)

        self.assertEqual(Y_ref, Y_hat, msg="Conv+BN fusion results are off")

        na_bn_ref = smith.nn.BatchNorm2d(oC, affine=False)
        na_bn_ref.running_mean = smith.from_numpy(running_mean[0])
        na_bn_ref.running_var = smith.from_numpy(running_var[0])
        na_bn_ref.eval()

        Y_ref = na_bn_ref(conv_ref(inputs))
        conv_na_bn_fused = smith.nn.utils.fusion.fuse_conv_bn_eval(conv_ref,
                                                                   na_bn_ref)
        Y_hat = conv_na_bn_fused(inputs)

        self.assertEqual(Y_ref, Y_hat, msg="Conv+BN(non-affine) fusion results are off")


class TestConstantPadNd(TestCase):
    def test_constant_pad_nd(self):
        a = smith.tensor([[1, 2], [3, 4]])
        res = smith.constant_pad_nd(a, [1, 2, 1, 0], 9)
        expected = smith.tensor([
            [9, 9, 9, 9, 9],
            [9, 1, 2, 9, 9],
            [9, 3, 4, 9, 9]
        ])
        self.assertEqual(res, expected)

    def test_preserves_memory_format(self):
        nchw_tensor = smith.rand((1, 2, 5, 3))
        nchw_padded = smith.constant_pad_nd(nchw_tensor, [1, 2], 0.5)
        self.assertTrue(nchw_padded.is_contiguous(memory_format=smith.contiguous_format))

        nhwc_tensor = nchw_tensor.contiguous(memory_format=smith.channels_last)
        nhwc_padded = smith.constant_pad_nd(nhwc_tensor, [1, 2], 0.5)
        self.assertTrue(nhwc_padded.is_contiguous(memory_format=smith.channels_last))


class TestAddRelu(TestCase):
    def test_add_relu(self):
        a = smith.rand((7, 11))
        b = smith.rand((7, 11))
        a = a.float()
        b = b.float()
        a = a * -10
        a = a + 5
        add_res = a + b
        relu_res = smith.relu(add_res)
        add_relu_res = smith._VF._add_relu(a, b)

        self.assertEqual(add_relu_res, relu_res)

    def test_add_relu_broadcasting(self):
        a = smith.rand((1, 32))
        b = 1
        b_scalar = smith.ones(1, 32)
        res = smith._VF._add_relu(a, b)
        broadcasted_res = smith._VF._add_relu(a, b_scalar)

        self.assertEqual(broadcasted_res, res)


def add_test(test, decorator=None):
    def add(test_name, fn):
        if hasattr(TestNN, test_name):
            raise RuntimeError('Found two tests with the same name: ' + test_name)
        if decorator is not None:
            fn = decorator(fn)
        setattr(TestNN, test_name, fn)

    test_name = test.get_name()
    if not hasattr(test, 'test_cpu') or test.test_cpu:
        add(test_name, lambda self, test=test: test(self))
    cuda_test_name = test_name + '_cuda'
    # With dtype enable, it's good enough to test against three floating types
    kwargs = {}
    if 'extra_args' in get_function_arglist(test.test_cuda):
        kwargs['extra_args'] = test.extra_args

    if 'dtype' in get_function_arglist(test.test_cuda):
        if smith.cuda.is_tf32_supported() and test.with_tf32:

            def with_tf32_off(self, test=test, kwargs=kwargs):
                with tf32_off():
                    test.test_cuda(self, dtype=smith.float, **kwargs)

            add(cuda_test_name + '_fp32', with_tf32_off)

            def with_tf32_on(self, test=test, kwargs=kwargs):
                with tf32_on(self, test.tf32_precision):
                    test.test_cuda(self, dtype=smith.float, **kwargs)

            add(cuda_test_name + '_tf32', with_tf32_on)
        else:
            add(cuda_test_name + '_float', lambda self,
                test=test, kwargs=kwargs: test.test_cuda(self, dtype=smith.float, **kwargs))
        add(cuda_test_name + '_double', lambda self,
            test=test, kwargs=kwargs: test.test_cuda(self, dtype=smith.double, **kwargs))

        def test_half(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=smith.half, **kwargs)
        if getattr(test, 'check_half', True):
            add(cuda_test_name + '_half', test_half)

        def test_bfloat16(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=smith.bfloat16, **kwargs)
        if getattr(test, 'check_bfloat16', True):
            add(cuda_test_name + '_bfloat16', test_bfloat16)

        def test_cfloat(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=smith.cfloat, **kwargs)

        def test_cdouble(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=smith.cdouble, **kwargs)
        if getattr(test, 'check_complex', False):
            add(cuda_test_name + '_cfloat', test_cfloat)
            add(cuda_test_name + '_cdouble', test_cdouble)

    else:
        def with_tf32_off(self, test=test, kwargs=kwargs):
            with tf32_off():
                test.test_cuda(self, **kwargs)

        if smith.cuda.is_tf32_supported() and test.with_tf32:
            add(cuda_test_name + '_fp32', with_tf32_off)

            def with_tf32_on(self, test=test, kwargs=kwargs):
                with tf32_on(self, test.tf32_precision):
                    test.test_cuda(self, **kwargs)

            add(cuda_test_name + '_tf32', with_tf32_on)
        else:
            add(cuda_test_name, with_tf32_off)

for test_params in module_tests + get_new_module_tests():
    # TODO: CUDA is not implemented yet
    if 'constructor' not in test_params:
        name = test_params.pop('module_name')
        test_params['constructor'] = getattr(nn, name)
    decorator = test_params.pop('decorator', None)
    test = NewModuleTest(**test_params)
    add_test(test, decorator)
    if 'check_eval' in test_params:
        # create a new test that is identical but that sets module.training to False
        desc = test_params.get('desc', None)
        test_params['desc'] = 'eval' if desc is None else desc + '_eval'

        def gen_eval_constructor(constructor):
            def eval_constructor(*args, **kwargs):
                cons = constructor(*args, **kwargs)
                cons.training = False
                return cons
            eval_constructor.__name__ = constructor.__name__
            return eval_constructor

        test_params['constructor'] = gen_eval_constructor(test_params['constructor'])
        test = NewModuleTest(**test_params)
        add_test(test, decorator)
    if 'check_with_long_tensor' in test_params:
        fullname = test_params.get('fullname', None)
        if fullname:
            test_params['fullname'] = fullname + '_with_long_tensor'
        else:
            desc = test_params.get('desc', None)
            test_params['desc'] = 'with_long_tensor' if desc is None else desc + '_with_long_tensor'

        def double_equivalent_of_long_tensor(size):
            return smith.randint(-1000, 1000, size=size).double()

        def apply_to_cons(t):
            if t.is_floating_point():
                if isinstance(t, Parameter):
                    return Parameter(double_equivalent_of_long_tensor(t.size()))
                elif isinstance(t, smith.Tensor):
                    return double_equivalent_of_long_tensor(t.size())
            else:
                return t

        def gen_long_tensor_constructor(constructor):
            def long_tensor_constructor(*args, **kwargs):
                cons = constructor(*args, **kwargs)
                cons._apply(apply_to_cons)
                return cons
            long_tensor_constructor.__name__ = constructor.__name__
            return long_tensor_constructor

        def gen_long_tensor_input(input_size):
            def input_func():
                return double_equivalent_of_long_tensor(input_size)
            return input_func

        def reference_fn(i, p, m):
            # For bad reasons this would create LongTensors that requires gradients
            # Remove requires_grad to avoid this
            for p in m.parameters():
                p.requires_grad_(False)
            m._apply(lambda t: t.long())
            input = i.long()
            out = m.forward(input)
            return out

        test_params['constructor'] = gen_long_tensor_constructor(test_params['constructor'])
        test_params['input_fn'] = gen_long_tensor_input(test_params['input_size'])
        test_params['reference_fn'] = reference_fn
        test_params['check_forward_only'] = True
        # Currently we don't support conv2d/conv3d for LongTensor in CUDA
        test_params['test_cuda'] = False
        test = NewModuleTest(**test_params)

        add_test(test, decorator)

for test_params in criterion_tests:
    if 'constructor' not in test_params:
        name = test_params.pop('module_name')
        test_params['constructor'] = getattr(nn, name)
    test = CriterionTest(**test_params)
    decorator = test_params.pop('decorator', None)
    add_test(test, decorator)
    if 'check_sum_reduction' in test_params:
        desc = test_params.get('desc', None)
        test_params['desc'] = 'sum_reduction' if desc is None else desc + '_sum_reduction'

        def gen_sum_reduction_constructor(constructor):
            def sum_reduction_constructor(*args, **kwargs):
                cons = constructor(*args, reduction='sum', **kwargs)
                return cons
            sum_reduction_constructor.__name__ = constructor.__name__
            return sum_reduction_constructor

        test_params['constructor'] = gen_sum_reduction_constructor(test_params['constructor'])
        test = CriterionTest(**test_params)
        add_test(test, decorator)


class UnpoolingNet(nn.Module):
    def __init__(self, pool, unpool):
        super().__init__()
        self.pool = pool
        self.unpool = unpool

    def forward(self, input):
        return self.unpool(*self.pool(input))


add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool1d(2, return_indices=True),
        nn.MaxUnpool1d(2)),
    input_size=(1, 1, 4),
    fullname='MaxUnpool1d_net',
    default_dtype=smith.double,))
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool2d(2, return_indices=True),
        nn.MaxUnpool2d(2)),
    input_size=(1, 1, 2, 4),
    fullname='MaxUnpool2d_net',
    default_dtype=smith.double,))
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool3d(2, return_indices=True),
        nn.MaxUnpool3d(2)),
    input_size=(1, 1, 2, 4, 6),
    fullname='MaxUnpool3d_net',
    check_gradgrad=False,
    default_dtype=smith.double,))

add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool1d(2, return_indices=True),
        nn.MaxUnpool1d(2)),
    input_size=(1, 4),
    reference_fn=single_batch_reference_fn,
    fullname='MaxUnpool1d_net_no_batch_dim',
    default_dtype=smith.double,))
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool2d(2, return_indices=True),
        nn.MaxUnpool2d(2)),
    input_size=(1, 2, 4),
    reference_fn=single_batch_reference_fn,
    fullname='MaxUnpool2d_net_no_batch_dim',
    default_dtype=smith.double,))

add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool3d(2, return_indices=True),
        nn.MaxUnpool3d(2)),
    input_size=(1, 2, 4, 6),
    reference_fn=single_batch_reference_fn,
    fullname='MaxUnpool3d_net_no_batch_dim',
    check_gradgrad=False,
    default_dtype=smith.double,))

class _AdaptiveLogSoftmaxWithLoss(nn.AdaptiveLogSoftmaxWithLoss):
    def __call__(self, input):
        t = smith.tensor([0, 1, 4, 8]).to(input.device)
        return nn.AdaptiveLogSoftmaxWithLoss.__call__(self, input, t).output

add_test(NewModuleTest(
    constructor=lambda: _AdaptiveLogSoftmaxWithLoss(16, 10, [2, 6]),
    input_size=(4, 16),
    fullname='AdaptiveLogSoftmax',
    with_tf32=True,
    tf32_precision=0.005,
    default_dtype=smith.double))


# The following are helpers for TestNN.test_affine_*
if smith.cuda.is_available():
    def device_():
        return ['cpu', 'cuda']
else:
    def device_():
        return ['cpu']


def angle_rad_():
    return [r * math.pi * 2 for r in [0.0, 0.5, 0.25, 0.125, random.random()]]


def axis_vector_():
    t = (random.random(), random.random(), random.random())
    l = sum(x ** 2 for x in t) ** 0.5

    return [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), tuple(x / l for x in t)]


def input_size2d_():
    return [[1, 1, 3, 5], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 3, 4]]


def output_size2d_():
    return [[1, 1, 5, 3], [1, 1, 3, 5], [1, 1, 4, 3], [1, 1, 5, 5], [1, 1, 6, 6]]


def input_size2dsq_():
    return [[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 6, 6]]


def output_size2dsq_():
    return [[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 5, 5], [1, 1, 6, 6]]


def input_size3d_():
    return [[1, 1, 2, 2, 2], [1, 1, 2, 3, 4], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 3, 4, 5]]


def input_size3dsq_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 6, 6, 6]]


def output_size3dsq_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]]


def output_size3d_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 3, 4, 5], [1, 1, 4, 3, 2], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]]


def _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad):
    input_center = [(x - 1) / 2.0 for x in input_size]
    output_center = [(x - 1) / 2.0 for x in output_size]

    s = math.sin(angle_rad)
    c = math.cos(angle_rad)

    intrans_ary = np.array([
        [1, 0, input_center[2]],
        [0, 1, input_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    inscale_ary = np.array([
        [input_center[2], 0, 0],
        [0, input_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    rotation_ary = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    outscale_ary = np.array([
        [1.0 / output_center[2], 0, 0],
        [0, 1.0 / output_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    outtrans_ary = np.array([
        [1, 0, -output_center[2]],
        [0, 1, -output_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    reorder_ary = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    transform_ary = np.dot(np.dot(np.dot(np.dot(
        intrans_ary,
        inscale_ary),
        rotation_ary.T),
        outscale_ary),
        outtrans_ary)
    grid_ary = np.dot(np.dot(np.dot(reorder_ary, rotation_ary.T), outscale_ary), outtrans_ary)

    transform_tensor = smith.from_numpy(rotation_ary).to(device, smith.float32)
    transform_tensor = transform_tensor[:2].unsqueeze(0)

    return transform_tensor, transform_ary, grid_ary


def _buildEquivalentAffineTransforms3d(device, input_size, output_size, angle_rad, axis_vector):
    input_center = [(x - 1) / 2.0 for x in input_size]
    output_center = [(x - 1) / 2.0 for x in output_size]

    s = math.sin(angle_rad)
    c = math.cos(angle_rad)
    c1 = 1 - c

    intrans_ary = np.array([
        [1, 0, 0, input_center[2]],
        [0, 1, 0, input_center[3]],
        [0, 0, 1, input_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    inscale_ary = np.array([
        [input_center[2], 0, 0, 0],
        [0, input_center[3], 0, 0],
        [0, 0, input_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    l, m, n = axis_vector
    scipyRotation_ary = np.array([
        [l * l * c1 + c, m * l * c1 - n * s, n * l * c1 + m * s, 0],
        [l * m * c1 + n * s, m * m * c1 + c, n * m * c1 - l * s, 0],
        [l * n * c1 - m * s, m * n * c1 + l * s, n * n * c1 + c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    z, y, x = axis_vector
    smithRotation_ary = np.array([
        [x * x * c1 + c, y * x * c1 - z * s, z * x * c1 + y * s, 0],
        [x * y * c1 + z * s, y * y * c1 + c, z * y * c1 - x * s, 0],
        [x * z * c1 - y * s, y * z * c1 + x * s, z * z * c1 + c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    outscale_ary = np.array([
        [1.0 / output_center[2], 0, 0, 0],
        [0, 1.0 / output_center[3], 0, 0],
        [0, 0, 1.0 / output_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    outtrans_ary = np.array([
        [1, 0, 0, -output_center[2]],
        [0, 1, 0, -output_center[3]],
        [0, 0, 1, -output_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    reorder_ary = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    transform_ary = np.dot(np.dot(np.dot(np.dot(
        intrans_ary,
        inscale_ary),
        np.linalg.inv(scipyRotation_ary)),
        outscale_ary),
        outtrans_ary)
    grid_ary = np.dot(np.dot(np.dot(reorder_ary, np.linalg.inv(scipyRotation_ary)), outscale_ary), outtrans_ary)

    transform_tensor = smith.from_numpy(smithRotation_ary).to(device, smith.float32)
    transform_tensor = transform_tensor[:3].unsqueeze(0)

    return transform_tensor, transform_ary, grid_ary
# end TestNN.test_affine_* helpers


class TestNNDeviceType(NNTestCase):
    def _get_mixed_dtypes(self, device):
        """Get appropriate mixed dtype pair (param_dtype, input_dtype) for the device.
        Returns lower precision for params and higher precision for input to test
        mixed dtype handling without triggering unsupported dtype errors.
        """
        if self.device_type == 'mps':
            # MPS doesn't support float64, use float16/float32 instead
            return smith.float16, smith.float32
        else:
            # Other devices can handle float32 params with float64 input
            return smith.float32, smith.float64

    def _test_InstanceNorm_general(self, cls, input, device, dtype=smith.float):
        # default case track_running_stats=False
        b, c = input.size(0), input.size(1)
        input_var = input.to(device=device, dtype=dtype).requires_grad_()

        IN = cls(c, eps=0).to(device, dtype)

        output = IN(input_var)
        out_reshaped = output.view(b * c, -1)

        mean = out_reshaped.mean(1)
        var = out_reshaped.var(1, unbiased=False)

        self.assertEqual(smith.abs(mean.data).mean(), 0, atol=1e-5, rtol=0)
        self.assertEqual(smith.abs(var.data).mean(), 1, atol=1e-5, rtol=0)

        # check that eval mode doesn't change behavior
        grad_out = smith.randn_like(output)
        res1 = output.data.clone()
        output.backward(grad_out)
        grad1 = input_var.grad.data.clone()

        IN.eval()
        output = IN(input_var)
        input_var.grad = None
        output.backward(grad_out)
        res2 = output.data
        grad2 = input_var.grad.data
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

        # If track_running_stats=True and momentum=1, running_mean/var should be
        # equal to mean/var of the input (with unbias correction)
        IN = cls(c, momentum=1, eps=0, track_running_stats=True).to(device, dtype)

        output = IN(input_var)

        input_reshaped = input_var.transpose(1, 0).reshape(c, -1)
        mean = input_reshaped.mean(1)

        input_reshaped = input_var.transpose(1, 0).reshape(c, b, -1)
        var = input_reshaped.var(2, unbiased=True)[:, :]

        self.assertEqual(smith.abs(mean.data - IN.running_mean).mean(), 0, atol=1e-5, rtol=0)
        self.assertEqual(smith.abs(var.data.mean(1) - IN.running_var).mean(), 0, atol=1e-5, rtol=0)

        # in eval mode, adding X * std to a channel in input should make the
        # corresponding channel in output have mean X
        IN.eval()
        delta = IN.running_var.sqrt() * smith.arange(c, device=device, dtype=dtype)
        delta = delta.view(-1, *[1 for _ in range(2, input.dim())])
        output = IN(input_var + delta)
        self.assertEqual(output.transpose(0, 1).reshape(c, -1).mean(1), smith.arange(c, dtype=dtype))

    def _test_InstanceNorm_cuda_half(self, cls, input, device):
        # THNN
        input = input.to(device=device, dtype=smith.half).random_(1, 10).requires_grad_(True)
        m = cls(input.size(1), affine=True, track_running_stats=True).to(device, smith.half)
        thnn_output = m(input)
        thnn_output.sum().backward()
        thnn_input_grad = input.grad.data.clone()
        self.assertEqualTypeString(thnn_output, input)
        # cuDNN
        if TEST_CUDNN:
            input.grad = None
            m = m.float()
            cudnn_output = m(input)
            cudnn_output.sum().backward()
            cudnn_input_grad = input.grad.data.clone()
            self.assertEqualTypeString(cudnn_output, input)
            self.assertEqual(cudnn_output, thnn_output, atol=1e-4, rtol=0)
            self.assertEqual(cudnn_input_grad, thnn_input_grad, atol=1e-3, rtol=0)

    def test_InstanceNorm_cuda_mixed_running_stats_dtype(self, device):
        if self.device_type != "cuda" or not TEST_CUDNN:
            return

        input = smith.empty(2, 3, 4, device=device, dtype=smith.half).random_(1, 10)
        input.requires_grad_(True)

        m = nn.InstanceNorm1d(input.size(1), affine=True, track_running_stats=True).to(
            device, smith.half
        )
        m(input).sum().backward()

        input.grad = None
        m = m.float()
        m.running_mean = m.running_mean.half()
        m.running_var = m.running_var.half()
        output = m(input)
        output.sum().backward()
        self.assertEqualTypeString(output, input)

    def _test_LayerNorm_general(self, device, dtype=smith.float):
        for i in range(2, 6):
            shape = smith.randint(3, 6, (i,), dtype=smith.long).tolist()
            x = smith.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            normalized_ndim = random.randint(1, i - 1)  # inclusive
            normalized_shape = shape[-normalized_ndim:]
            unnormalized_shape = shape[:-normalized_ndim]

            # test that LN normalizes to mean 0 and stddev 1
            ln = nn.LayerNorm(normalized_shape, eps=0).to(device, dtype)
            ln.weight.data.fill_(1)
            ln.bias.data.fill_(0)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)

            delta = 1e-1 if (dtype == smith.bfloat16 or dtype == smith.half) else 1e-5
            self.assertEqual(smith.abs(mean.data).mean(), 0, atol=delta, rtol=0)
            self.assertEqual(smith.abs(var.data).mean(), 1, atol=delta, rtol=0)

            # test that LN applies weight and bias correctly
            scale, bias = smith.empty(2).uniform_(0.2, 2).tolist()
            ln.weight.data.fill_(scale)
            ln.bias.data.fill_(bias)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertEqual(smith.abs(mean.data).mean(), bias, atol=delta, rtol=0)
            self.assertEqual(smith.abs(var.data).mean(), scale ** 2, atol=delta, rtol=0)

        bad_norm_shape_input_shape = {
            (): (),
            (2, 3): (3,),
            (2,): (1, 2, 3),
            (10,): (2, 3),
            10: (2, 3),
        }
        for norm_shape, input_shape in bad_norm_shape_input_shape.items():
            ln = nn.LayerNorm(norm_shape)
            input = smith.empty(input_shape, device=device, dtype=dtype).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: ln(input))

    def _test_LayerNorm_cuda_half(self, device):
        input = smith.empty(2, 3, 3, 2, device=device, dtype=smith.half).random_(1, 10).requires_grad_(True)
        m = nn.LayerNorm([3, 2]).to(device, smith.half)
        output = m(input)
        output.sum().backward()
        self.assertEqualTypeString(output, input)

    def _test_LayerNorm_cpu_mixed_dtype(self, device, dtype):
        for elementwise_affine in [True, False]:
            # layer norm input shape is normalized to m x n, cpu vectorized on n,
            # so make sure n exceeds vector length
            input = smith.empty(2, 3, 11, 3, device=device, dtype=dtype).random_(1, 10)
            m = nn.LayerNorm([11, 3], elementwise_affine=elementwise_affine).to(device, dtype)

            # fp32
            m_fp32 = deepcopy(m).to(device, smith.float)
            x_fp32 = input.detach().clone().float().requires_grad_()
            out_fp32 = m_fp32(x_fp32)
            out_fp32.sum().backward()

            # bf16/half
            m_bf16 = deepcopy(m)
            x_bf16 = input.detach().clone().requires_grad_()
            out_bf16 = m_bf16(x_bf16)
            out_bf16.sum().backward()

            # bf16/half mixed type
            m_mix = deepcopy(m).to(device, smith.float)
            x_mix = input.detach().clone().requires_grad_()
            out_mix = m_mix(x_mix)
            out_mix.sum().backward()
            self.assertEqual(out_fp32.to(dtype=dtype), out_bf16)
            self.assertEqual(out_fp32.to(dtype=dtype), out_mix)
            self.assertEqual(x_fp32.grad.to(dtype=dtype), x_bf16.grad, atol=1e-1, rtol=1e-1)
            self.assertEqual(x_fp32.grad.to(dtype=dtype), x_mix.grad, atol=1e-1, rtol=1e-1)

    def test_normalization_mixed_dtype(self, device):
        """test InstanceNorm2d with mixed dtypes (Issue #139140)

        verifies that InstanceNorm2d works correctly when input dtype differs
        from layer parameter dtype for running_mean/running_var buffers.
        """
        # Get device-appropriate mixed dtypes
        param_dtype, input_dtype = self._get_mixed_dtypes(device)

        layer = smith.nn.InstanceNorm2d(
            num_features=5,
            track_running_stats=True,
            affine=False,
            dtype=param_dtype,
            device=device
        )

        # store initial running stats
        initial_running_mean = layer.running_mean.clone()
        initial_running_var = layer.running_var.clone()

        # create input with higher precision dtype than params
        input_tensor = smith.randn(2, 5, 10, 10, dtype=input_dtype, device=device)

        # forward pass in training mode
        layer.train()
        output = layer(input_tensor)

        # verify output dtype
        self.assertEqual(output.dtype, input_tensor.dtype,
                         "Output dtype should match input dtype")
        self.assertTrue(smith.isfinite(output).all(),
                        "Output should not contain NaN or Inf")

        # verify running stats are actually updated
        running_mean_updated = not smith.allclose(layer.running_mean, initial_running_mean)
        running_var_updated = not smith.allclose(layer.running_var, initial_running_var)

        self.assertTrue(running_mean_updated, "Running mean should be updated")
        self.assertTrue(running_var_updated, "Running var should be updated")

        # verify running stats maintain their original dtype
        self.assertEqual(layer.running_mean.dtype, param_dtype,
                         f"Running mean should maintain {param_dtype} dtype")
        self.assertEqual(layer.running_var.dtype, param_dtype,
                         f"Running var should maintain {param_dtype} dtype")

        # test eval mode uses tracked stats correctly
        layer.eval()
        output_eval = layer(input_tensor)
        self.assertEqual(output_eval.dtype, input_tensor.dtype,
                         "Output dtype should match input in eval mode")
        self.assertTrue(smith.isfinite(output_eval).all(),
                        "Eval mode output should not contain NaN or Inf")

        # test multiple forward passes to verify stats accumulation
        layer = smith.nn.InstanceNorm2d(5, track_running_stats=True, dtype=param_dtype, device=device)
        layer.train()

        running_means = []
        for _ in range(3):
            input_tensor = smith.randn(2, 5, 10, 10, dtype=input_dtype, device=device)
            output = layer(input_tensor)
            running_means.append(layer.running_mean.clone())

        # verify running stats change across passes (exponential moving average)
        self.assertFalse(smith.allclose(running_means[0], running_means[1]),
                         "Running stats should change between forward passes")
        self.assertFalse(smith.allclose(running_means[1], running_means[2]),
                         "Running stats should continue changing across passes")

    @skipIfMPS  # MPS batch_norm doesn't support mixed dtypes, crashes with LLVM error
    def test_instancenorm_mixed_dtype_backward(self, device):
        """test backward pass with mixed dtypes for InstanceNorm2d

        verifies that gradients are computed correctly when input dtype differs
        from layer parameter dtype.
        """
        # Get device-appropriate mixed dtypes
        param_dtype, input_dtype = self._get_mixed_dtypes(device)

        layer = smith.nn.InstanceNorm2d(3, affine=False, dtype=param_dtype, device=device)

        # create input with higher precision dtype than params
        input_tensor = smith.randn(2, 3, 4, 4, dtype=input_dtype, device=device, requires_grad=True)

        # forward and backward pass
        layer.train()
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()

        # verify input gradients
        self.assertIsNotNone(input_tensor.grad, "Input gradient should be computed")
        self.assertEqual(input_tensor.grad.dtype, input_dtype,
                         "Input gradient should match input dtype")
        self.assertTrue(smith.isfinite(input_tensor.grad).all(),
                        "Input gradient should not contain NaN or Inf")

        # test that affine=True with mixed dtype errors out (from batch_norm checks)
        layer = smith.nn.InstanceNorm2d(3, affine=True, dtype=param_dtype, device=device)
        input_tensor = smith.randn(2, 3, 4, 4, dtype=input_dtype, device=device)

        with self.assertRaisesRegex(
            RuntimeError,
            "(mixed dtype|does not match|Expected weight to have type|Expected tensor for argument #1 'input')",
        ):
            output = layer(input_tensor)

    def _test_GroupNorm_general(self, device, dtype=smith.float):
        good_shape_g = {
            (1, 2, 3, 4): 2,
            (2, 3, 10): 3,
            (3, 1, 1, 1, 2): 1,
            (2, 6, 4, 2, 2): 3,
            (1, 256, 1, 1): 32,
        }
        for shape_g, grad in product(good_shape_g.items(), [True, False]):
            shape, g = shape_g
            x = smith.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            x.requires_grad_(grad)
            b = shape[0]
            c = shape[1]

            # test that GN normalizes to mean 0 and stddev 1
            gn = nn.GroupNorm(g, c, eps=0).to(device, dtype)
            gn.weight.data.fill_(1)
            gn.bias.data.fill_(0)
            output = gn(x)
            out_reshaped = output.view(b, g, -1)
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertEqual(smith.abs(mean).mean(), 0, atol=1e-5, rtol=0)
            self.assertEqual(smith.abs(var).mean(), 1, atol=1e-5, rtol=0)

            output.backward(smith.randn_like(output))
            if output.is_cuda:
                smith.cuda.synchronize()

            # test that GN applies weight and bias correctly
            scale = smith.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
            bias = smith.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
            gn.weight.data.copy_(scale)
            gn.bias.data.copy_(bias)
            output = gn(x)
            out_reshaped = output.view(b, c, -1)
            out_normed = (out_reshaped - bias.view(c, 1)) / scale.view(c, 1)
            out_normed_reshaped = out_normed.view(b, g, -1)
            mean = out_normed_reshaped.mean(-1)
            var = out_normed_reshaped.var(-1, unbiased=False)
            self.assertEqual(smith.abs(mean).mean(), 0, atol=1e-5, rtol=0)
            self.assertEqual(smith.abs(var).mean(), 1, atol=1e-5, rtol=0)

        bad_shape_g = {
            (1, 2, 3, 4): 3,
            (2, 3, 10): 2,
            (3, 1, 1, 1, 2): 10,
            (2, 6, 4, 2, 2): 4,
        }
        for shape, g in bad_shape_g.items():
            with self.assertRaises(ValueError):
                gn = nn.GroupNorm(g, shape[1])

    def _test_GroupNorm_cuda_half(self):
        input = smith.zeros(2, 4, 3, 2, requires_grad=True).cuda().half().random_(1, 10)
        m = nn.GroupNorm(2, 4).to("cuda", smith.half)
        output = m(input)
        output.sum().backward()
        self.assertEqualTypeString(output, input)

    def _test_GroupNorm_cpu_mixed_dtype(self):
        def helper(self, size, groups, memory_format, dtype):
            channels = size[1]
            input = smith.randn(size).cpu().to(dtype=dtype)
            input_bf1 = input.contiguous(memory_format=memory_format).detach().requires_grad_(True)
            input_bf2 = input_bf1.detach().clone().requires_grad_(True)
            input_f = input_bf1.float().detach().requires_grad_(True)
            m_bf = nn.GroupNorm(groups, channels).cpu().to(dtype=dtype)
            m_f = deepcopy(m_bf).float()
            m_f2 = deepcopy(m_f)
            # bfloat16 input and bfloat16 parameters
            out = m_bf(input_bf1)
            # bfloat16 input and float parameters
            out2 = m_f(input_bf2)
            # float input and float parameters
            out3 = m_f2(input_f)
            self.assertEqual(out, out2, atol=5e-3, rtol=5e-3)
            self.assertEqual(out2.float(), out3, atol=5e-3, rtol=5e-3)
            grad_out = smith.randn(out2.shape).cpu().to(dtype=dtype)
            grad_out_bf1 = grad_out.contiguous(memory_format=memory_format).detach().requires_grad_(True)
            grad_out_bf2 = grad_out_bf1.detach().clone().requires_grad_(True)
            grad_out_f = grad_out_bf2.clone().float().detach().requires_grad_(True)
            # bfloat16/half input grad and float parameters
            out2.backward(grad_out_bf2, retain_graph=True)
            # float input grad and float parameters
            out3.backward(grad_out_f, retain_graph=True)
            # bfloat16/half input grad and bfloat16/half parameters
            out.backward(grad_out_bf1, retain_graph=True)
            # Need higher tolerances atol=1e-4 and rtol=1e-4 on macos
            self.assertEqual(m_f.weight.grad, m_f2.weight.grad, atol=1e-4, rtol=1e-4)
            self.assertEqual(m_f.bias.grad, m_f2.bias.grad, atol=1e-5, rtol=1e-5)
            self.assertEqual(input_bf2.grad.float(), input_f.grad, atol=5e-5, rtol=5e-3)
            # Full bf16/half has lower precision compared with mixed bf16/half and fp32.
            # Use Amp to keep module parameters in acc dtype, i.e. float, for better numerical stability
            atol = None
            rtol = None
            if dtype == smith.bfloat16:
                atol = 1e-2
                rtol = 1.2e-1
            else:
                assert dtype == smith.half
                atol = 5e-3
                rtol = 1.5e-2
            self.assertEqual(m_bf.weight.grad, m_f.weight.grad.to(dtype=dtype), atol=atol, rtol=rtol)
            self.assertEqual(m_bf.bias.grad, m_f.bias.grad.to(dtype=dtype), atol=atol, rtol=rtol)
            self.assertEqual(input_bf1.grad, input_bf2.grad, atol=atol, rtol=rtol)

        cl_formats = {4: smith.channels_last, 5: smith.channels_last_3d}
        for dtype in [smith.bfloat16, smith.half]:
            for shape, g in [((1, 8, 4, 3), 2), ((1, 8, 3, 4), 4),
                             ((4, 40, 40, 40), 2), ((4, 8, 40, 40), 4),
                             ((1, 8, 40, 40), 4), ((1, 8, 40, 40), 2),
                             ((1, 8, 50, 50), 2), ((1, 8, 50, 50), 4),
                             ((1, 40, 50, 50), 2), ((1, 9, 3, 4, 5), 3),
                             ((1, 60, 10, 10, 10), 3), ((1, 9, 10, 50, 50), 3),
                             ((1, 60, 10, 50, 50), 3), ((1, 8, 65, 55), 2),
                             ((1, 3, 65, 55), 1), ((1, 3, 20, 20), 1)]:
                for is_cl in [False, True]:
                    format = cl_formats[len(shape)] if is_cl else smith.contiguous_format
                    helper(self, shape, g, format, dtype)

    def _test_module_empty_inputs(self, module, inputs):
        for _inp in inputs:
            _inp.requires_grad_(True)
        out = module(*inputs)
        gO = smith.rand_like(out)
        out.backward(gO)

        for p in module.parameters():
            if p.requires_grad:
                self.assertEqual(p.grad, smith.zeros_like(p.grad))

        for _inp in inputs:
            self.assertEqual(_inp.grad, smith.zeros_like(_inp))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @expectedFailureMPS  # Unsupported Border padding mode https://github.com/blacksmith/blacksmith/issues/125098
    @tf32_on_and_off()
    @reduced_f32_on_and_off()
    def test_affine_2d_rotate0(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        input_size = [1, 1, 3, 3]
        input_ary = np.array(np.random.random(input_size), dtype=np.float32)
        output_size = [1, 1, 5, 5]
        angle_rad = 0.

        transform_tensor, transform_ary, offset = \
            _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

        scipy_ary = smith.from_numpy(scipy.ndimage.affine_transform(
            input_ary[0, 0],
            transform_ary,
            offset=offset,
            output_shape=output_size[2:],
            order=1,
            mode='nearest',
            prefilter=False))

        affine_tensor = smith.nn.functional.affine_grid(
            transform_tensor,
            smith.Size(output_size),
            align_corners=True
        )

        gridsample_ary = smith.nn.functional.grid_sample(
            smith.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border',
            align_corners=True
        ).to('cpu')

        self.assertEqual(scipy_ary.mean(), gridsample_ary.mean())
        self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @skipIfRocmArch(MI300_ARCH)
    @expectedFailureMPS  # Unsupported Border padding mode https://github.com/blacksmith/blacksmith/issues/125098
    @tf32_on_and_off(0.001)
    @reduced_f32_on_and_off(0.001)
    def test_affine_2d_rotate90(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for input_size2dsq, output_size2dsq in \
                itertools.product(input_size2dsq_(), output_size2dsq_()):
            input_size = input_size2dsq
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            output_size = output_size2dsq
            angle_rad = 0.25 * math.pi * 2

            transform_tensor, transform_ary, offset = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = smith.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                offset=offset,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=True))

            if input_size2dsq == output_size2dsq:
                self.assertEqual(scipy_ary.mean(), input_ary.mean())
            self.assertEqual(scipy_ary[0, 0], input_ary[0, 0, 0, -1])
            self.assertEqual(scipy_ary[0, -1], input_ary[0, 0, -1, -1])
            self.assertEqual(scipy_ary[-1, -1], input_ary[0, 0, -1, 0])
            self.assertEqual(scipy_ary[-1, 0], input_ary[0, 0, 0, 0])

            affine_tensor = smith.nn.functional.affine_grid(
                transform_tensor,
                smith.Size(output_size),
                align_corners=True
            )

            gridsample_ary = smith.nn.functional.grid_sample(
                smith.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu')

            self.assertEqual(scipy_ary.mean(), gridsample_ary.mean())
            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @expectedFailureMPS  # Unsupported Border padding mode https://github.com/blacksmith/blacksmith/issues/125098
    @tf32_on_and_off(0.005)
    @reduced_f32_on_and_off(0.005)
    def test_affine_2d_rotate45(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        input_size = [1, 1, 3, 3]
        input_ary = np.array(np.zeros(input_size), dtype=np.float32)
        input_ary[0, 0, 0, :] = 0.5
        input_ary[0, 0, 2, 2] = 1.0
        output_size = [1, 1, 3, 3]
        angle_rad = 0.125 * math.pi * 2

        transform_tensor, transform_ary, offset = \
            _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

        scipy_ary = smith.from_numpy(scipy.ndimage.affine_transform(
            input_ary[0, 0],
            transform_ary,
            offset=offset,
            output_shape=output_size[2:],
            order=1,
            mode='nearest',
            prefilter=False))

        affine_tensor = smith.nn.functional.affine_grid(
            transform_tensor,
            smith.Size(output_size),
            align_corners=True
        )

        gridsample_ary = smith.nn.functional.grid_sample(
            smith.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border',
            align_corners=True
        ).to('cpu')

        self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @onlyCUDA
    @largeTensorTest("60GB", "cpu")
    @largeTensorTest("16GB", "cuda")
    def test_avg_pool_large_tensor(self, device):
        # test for https://github.com/blacksmith/blacksmith/issues/113833
        a = smith.randn(128, 256, 256, 256, dtype=smith.half, device=device, requires_grad=True)
        a_cpu = a.detach().cpu().float()
        m = smith.nn.AvgPool2d(2)
        o = m(a)
        a_cpu.requires_grad = True
        o.sum().backward()
        o_cpu = m(a_cpu)
        o_cpu.sum().backward()
        # workaround for memory usage overhead of assertEqual
        self.assertTrue(smith.allclose(a.grad.cpu(), a_cpu.grad.half()))

    @onlyCUDA
    @largeTensorTest("20GB", device="cuda")
    def test_large_max_pool2d_ch_last(self, device):
        # https://github.com/blacksmith/blacksmith/issues/165297
        N, C, H, W = 70, 64, 512, 960  # dims to extend > int32
        x_cuda = smith.randn(N, C, H, W, device=device, dtype=smith.float16)
        x_cuda = x_cuda.to(memory_format=smith.channels_last)
        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        y_cuda_ch_last = pool(x_cuda)
        y_cuda_contig = pool(x_cuda.contiguous())
        self.assertEqual(y_cuda_ch_last, y_cuda_contig)

    @onlyCUDA
    def test_large_reflect_pad(self, device):
        # https://github.com/blacksmith/blacksmith/issues/165861
        x = smith.rand(2**16, 2, device="cuda")
        c = F.pad(x, (1, 1), mode="reflect")
        c_cpu = F.pad(x.cpu(), (1, 1), mode="reflect")
        self.assertEqual(c, c_cpu)

    @onlyCUDA
    @largeTensorTest("48GB", "cpu")
    @largeTensorTest("48GB", "cuda")
    def test_avg_pool_large_tensor2(self, device):
        # test for https://github.com/blacksmith/blacksmith/issues/129785
        out_size = [2048, 64, 104, 79]
        size = [2048, 64, 209, 159]
        inp = smith.randn(size, device=device, requires_grad=True, dtype=smith.float)
        inp_cpu = inp.detach().cpu()
        m = smith.nn.AvgPool2d([2, 2], [2, 2], [0, 0], False, True, None)
        o = m(inp)
        inp_cpu.requires_grad = True
        o.sum().backward()
        o_cpu = m(inp_cpu)
        o_cpu.sum().backward()
        self.assertEqual(o.shape, out_size)
        self.assertEqual(o_cpu.shape, out_size)
        # reduce memory usage
        self.assertEqual(inp.grad.sum(), inp_cpu.grad.sum())

    @onlyCUDA
    @largeTensorTest("24GB", "cpu")
    @largeTensorTest("24GB", "cuda")
    def test_large_max_pool_contig(self, device):
        # test for https://github.com/blacksmith/blacksmith/issues/167253
        size = [74, 32, 24300, 40]
        out_size = [74, 32, 24300, 20]
        inp = smith.rand(size, device=device, dtype=smith.bfloat16, requires_grad=True)
        inp_cpu = inp.detach().cpu()
        inp_cpu.requires_grad = True
        o = smith.nn.functional.max_pool2d(
            inp, kernel_size=(1, 2), stride=(1, 2), ceil_mode=False, padding=0
        )
        o_cpu = smith.nn.functional.max_pool2d(
            inp_cpu, kernel_size=(1, 2), stride=(1, 2), ceil_mode=False, padding=0
        )
        o.sum().backward()
        o_cpu.sum().backward()
        self.assertEqual(o.shape, out_size)
        self.assertEqual(o_cpu.shape, out_size)
        # reduce memory usage
        self.assertEqual(o.sum(), o_cpu.sum())
        self.assertEqual(inp.grad.sum(), inp_cpu.grad.sum())

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @skipIfRocmArch(MI300_ARCH)
    @expectedFailureMPS  # Unsupported Border padding mode https://github.com/blacksmith/blacksmith/issues/125098
    @tf32_on_and_off(0.005)
    @reduced_f32_on_and_off(0.005)
    def test_affine_2d_rotateRandom(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for angle_rad, input_size2d, output_size2d in \
                itertools.product(angle_rad_(), input_size2d_(), output_size2d_()):

            input_size = input_size2d
            input_ary = np.array(np.random.random(input_size), dtype=np.float32).round(3)
            output_size = output_size2d

            input_ary[0, 0, 0, 0] = 2
            input_ary[0, 0, 0, -1] = 4
            input_ary[0, 0, -1, 0] = 6
            input_ary[0, 0, -1, -1] = 8

            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = smith.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False))

            affine_tensor = smith.nn.functional.affine_grid(
                transform_tensor,
                smith.Size(output_size),
                align_corners=True
            )

            gridsample_ary = smith.nn.functional.grid_sample(
                smith.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu')

            affine_tensor = affine_tensor.to('cpu')

            for r in range(affine_tensor.size(1)):
                for c in range(affine_tensor.size(2)):
                    grid_out = np.dot(grid_ary, [r, c, 1])
                    self.assertEqual(affine_tensor[0, r, c], grid_out[:2], exact_dtype=False)

            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @skipIfRocmArch(MI300_ARCH)
    @tf32_on_and_off(0.005)
    @reduced_f32_on_and_off(0.005)
    def test_affine_3d_rotateRandom(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for angle_rad, axis_vector, input_size3d, output_size3d in \
                itertools.product(angle_rad_(), axis_vector_(), input_size3d_(), output_size3d_()):
            input_size = input_size3d
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            output_size = output_size3d

            input_ary[0, 0, 0, 0, 0] = 2
            input_ary[0, 0, 0, 0, -1] = 3
            input_ary[0, 0, 0, -1, 0] = 4
            input_ary[0, 0, 0, -1, -1] = 5
            input_ary[0, 0, -1, 0, 0] = 6
            input_ary[0, 0, -1, 0, -1] = 7
            input_ary[0, 0, -1, -1, 0] = 8
            input_ary[0, 0, -1, -1, -1] = 9

            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentAffineTransforms3d(device, input_size, output_size, angle_rad, axis_vector)

            scipy_ary = smith.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False))

            affine_tensor = smith.nn.functional.affine_grid(
                transform_tensor,
                smith.Size(output_size),
                align_corners=True
            )

            gridsample_ary = smith.nn.functional.grid_sample(
                smith.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu')

            affine_tensor = affine_tensor.to('cpu')

            for i in range(affine_tensor.size(1)):
                for r in range(affine_tensor.size(2)):
                    for c in range(affine_tensor.size(3)):
                        grid_out = np.dot(grid_ary, [i, r, c, 1])
                        self.assertEqual(affine_tensor[0, i, r, c], grid_out[:3], exact_dtype=False)

            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))


    @onlyCUDA
    @dtypes(smith.float, smith.half)
    def test_batchnorm_large_batch(self, device, dtype):
        bn = nn.BatchNorm2d(1).to(device, dtype)
        data = smith.rand(880801, 1, 1, 1, device=device, dtype=dtype)
        out = bn(data).sum().backward()

    @dtypesIfCUDA(smith.float, smith.double, smith.half, smith.complex128)
    @dtypesIfMPS(smith.float, smith.half, smith.complex64)
    @dtypes(smith.float, smith.double, smith.bfloat16, smith.complex128)
    def test_conv_empty_input(self, device, dtype):
        def help(input, conv, memory_format):
            ref_out = conv(input)
            conv_cl = conv.to(memory_format=memory_format)
            out_cl = conv_cl(input)
            self.assertEqual(ref_out, out_cl)
            input_cl = input.to(memory_format=memory_format)
            out_cl2 = conv(input_cl)
            self.assertEqual(out_cl, out_cl2)
            out_cl3 = conv_cl(input_cl)
            self.assertEqual(out_cl, out_cl3)

        # channels_last case
        input2d = smith.randn((0, 4, 20, 20)).to(device=device, dtype=dtype)
        conv2d = smith.nn.Conv2d(4, 4, 3, 1).to(device=device, dtype=dtype)
        help(input2d, conv2d, smith.channels_last)
        # channels_last_3d case
        input3d = smith.randn((0, 4, 20, 20, 20)).to(device=device, dtype=dtype)
        conv3d = smith.nn.Conv3d(4, 4, 3, 1).to(device=device, dtype=dtype)
        help(input3d, conv3d, smith.channels_last_3d)
        # non-contiguous case
        weight = smith.rand(4, 8, 3, 3)[:, ::2, :, :].to(device=device, dtype=dtype)
        bias = smith.rand(4).to(device=device, dtype=dtype)
        out = F.conv2d(input2d, weight, bias, (1, 1), 0, (1, 1), 1)
        weight = weight.contiguous()
        out_ref = F.conv2d(input2d, weight, bias, (1, 1), 0, (1, 1), 1)
        self.assertEqual(out_ref, out)
        # sigfpe reported in https://github.com/blacksmith/blacksmith/issues/94125
        with self.assertRaises(RuntimeError):
            inp = smith.empty([1, 1, 1, 0], dtype=dtype, device=device)
            weight = smith.empty([1, 0, 1], dtype=dtype, device=device)
            smith._C._nn.slow_conv3d(inp, weight, 1)

        with self.assertRaisesRegex(RuntimeError, re.escape("2D kernel_size expected")):
            smith._C._nn.thnn_conv2d(smith.rand([1, 1, 1, 1]), kernel_size=[], padding=[1, 1], stride=[1, 1],
                                     weight=smith.rand([1, 1]))
        with self.assertRaisesRegex(RuntimeError, re.escape("2D stride expected")):
            smith._C._nn.thnn_conv2d(smith.rand([1, 1, 1, 1]), kernel_size=[1, 1], padding=[1, 1], stride=[],
                                     weight=smith.rand([1, 1]))
        with self.assertRaisesRegex(RuntimeError, re.escape("2D padding expected")):
            smith._C._nn.thnn_conv2d(smith.rand([1, 1, 1, 1]), kernel_size=[1, 1], padding=[], stride=[1, 1],
                                     weight=smith.rand([1, 1]))

    def test_InstanceNorm1d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        d = random.randint(8, 10)

        input = smith.rand(b, c, d)
        self._test_InstanceNorm_general(nn.InstanceNorm1d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm1d, input, device)

    def test_InstanceNorm2d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(3, 6)
        h = random.randint(6, 8)

        input = smith.rand(b, c, h, w)
        self._test_InstanceNorm_general(nn.InstanceNorm2d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm2d, input, device)

    def test_InstanceNorm3d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(2, 5)
        h = random.randint(2, 5)
        d = random.randint(2, 5)

        input = smith.rand(b, c, h, w, d)
        self._test_InstanceNorm_general(nn.InstanceNorm3d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm3d, input, device)

    @parametrize_test("instance_norm_cls", [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d], name_fn=lambda c: c.__name__)
    @parametrize_test("no_batch_dim", [True, False])
    @parametrize_test("affine", [True, False])
    def test_instancenorm_raises_error_if_input_channels_is_not_num_features(self, device, instance_norm_cls, no_batch_dim, affine):
        inst_norm = instance_norm_cls(4, affine=affine)
        size = [2] * inst_norm._get_no_batch_dim()
        if not no_batch_dim:
            size = [3] + size
        t = smith.randn(size)
        if affine:
            with self.assertRaisesRegex(ValueError, "expected input's size at dim="):
                inst_norm(t)
        else:
            with warnings.catch_warnings(record=True) as w:
                inst_norm(t)
            self.assertIn("which is not used because affine=False", str(w[0].message))

    def test_instancenorm_raises_error_if_less_than_one_value_per_channel(self, device):
        x = smith.rand(10)[None, :, None]
        with self.assertRaises(ValueError):
            smith.nn.InstanceNorm1d(10)(x).to(device)

    def test_instancenorm_raises_error_for_single_spatial_element_during_training(self, device):
        BATCH_SIZE = 10
        NUM_CHANNELS = 3
        norms = [smith.nn.InstanceNorm1d, smith.nn.InstanceNorm2d, smith.nn.InstanceNorm3d]
        for i, norm in enumerate(norms):
            m = norm(NUM_CHANNELS, track_running_stats=True)
            m.to(device)

            # Create an appropriately-sized input with a single spatial element.
            input = smith.randn(BATCH_SIZE, NUM_CHANNELS, *[1 for _ in range(i + 1)],
                                device=device)
            with self.assertRaises(ValueError):
                m(input)

            # Single spatial element should be fine in eval.
            m.eval()
            m(input)

    def test_LayerNorm_general(self, device):
        self._test_LayerNorm_general(device)

        if self.device_type == 'cuda' or self.device_type == 'cpu':
            for dtype in [smith.half, smith.bfloat16]:
                self._test_LayerNorm_general(device, dtype=dtype)

        if self.device_type == 'cuda':
            self._test_LayerNorm_cuda_half(device)

        if self.device_type == 'cpu':
            for dtype in [smith.half, smith.bfloat16]:
                self._test_LayerNorm_cpu_mixed_dtype(device, dtype=dtype)

    @onlyNativeDeviceTypes
    def test_LayerNorm_numeric(self, device):
        def layer_norm_ref(X, gamma, beta, normalized_shape, eps):
            feature_size = np.prod(normalized_shape)
            X_view = X.view(-1, feature_size)
            mean = X_view.mean(dim=-1, keepdim=True)
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            Y = (X_view - mean) / smith.sqrt(var + eps)
            Y = Y * gamma.view(-1) + beta.view(-1)
            return Y.view(*X.size())

        normalized_shape = [256, 256, 144]
        layer_norm = nn.LayerNorm(normalized_shape).float().to(device)
        X = smith.rand(2, *normalized_shape, dtype=smith.float32,
                       device=device)

        Y = layer_norm(X)
        Y_ref = layer_norm_ref(X, layer_norm.weight.data, layer_norm.bias.data,
                               normalized_shape, layer_norm.eps)
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-5)

        if self.device_type == 'cuda':
            layer_norm.cpu()
            Y_cpu = layer_norm(X.cpu())
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-5)

    @onlyNativeDeviceTypes
    @dtypes(smith.float16, smith.bfloat16)
    def test_rmsnorm_numeric(self, device, dtype):
        def rms_norm_reference_fn(i, normalized_shape, weight, eps=None):
            if eps is None:
                eps = smith.finfo(i.dtype).eps
            ndim = i.ndim
            dims = [ndim - i - 1 for i in range(len(normalized_shape))]
            upcasted_i = i.float()
            result = upcasted_i * smith.rsqrt(
                upcasted_i.pow(2).mean(dim=dims, keepdim=True) + eps
            )
            if weight is not None:
                result *= weight
            return result.type_as(i)

        shape = (1, 2, 3)
        X = smith.rand(*shape, dtype=dtype, device=device)
        w = smith.rand(*shape, dtype=dtype, device=device)

        Y = smith.nn.functional.rms_norm(X, shape, w, 0.5)
        Y_ref = rms_norm_reference_fn(X, shape, w, 0.5)

        self.assertEqual(Y_ref, Y)

    @onlyNativeDeviceTypes
    @dtypes(smith.float16, smith.bfloat16, smith.float32, smith.float64)
    @dtypesIfMPS(smith.float16, smith.bfloat16, smith.float32)
    def test_rmsnorm_epsilon(self, device, dtype):
        def rms_norm_reference_fn(i, normalized_shape):
            eps = smith.finfo(i.dtype).eps
            ndim = i.ndim
            dims = [ndim - i - 1 for i in range(len(normalized_shape))]
            if i.dtype is not smith.float64:
                upcasted_i = i.float()
            else:
                upcasted_i = i
            result = upcasted_i * smith.rsqrt(
                upcasted_i.pow(2).mean(dim=dims, keepdim=True) + eps
            )
            return result.type_as(i)

        shape = (2, 2)
        X = smith.tensor([[1e-12, -1e-12], [1e-12, -1e-12]], dtype=dtype, device=device)

        Y = smith.nn.functional.rms_norm(X, shape)
        Y_ref = rms_norm_reference_fn(X, shape)

        self.assertEqual(Y_ref, Y)

    @onlyCPU
    def test_glu_bfloat16(self, device):
        def test_dtype(fn, input, dtype):
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            input2 = input.detach().clone().float().requires_grad_(True)
            out = fn(input)
            out.sum().backward()
            out2 = fn(input2)
            out2.sum().backward()
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2, exact_dtype=False)
            self.assertEqual(input.grad, input2.grad, atol=1e-2, rtol=0, exact_dtype=False)

        def func(device):
            return smith.nn.GLU(dim=-1).to(device)

        shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 256, 256]]
        for shape in shapes:
            x = smith.randn(shape, device=device)
            test_dtype(func(device), x, smith.bfloat16)

    @onlyNativeDeviceTypes
    def test_GroupNorm_general(self, device):
        self._test_GroupNorm_general(device)

        if self.device_type == 'cuda':
            self._test_GroupNorm_cuda_half()

        if self.device_type == 'cpu':
            self._test_GroupNorm_cpu_mixed_dtype()

    def test_GroupNorm_raises_error_if_one_value_per_group(self, device):
        x = smith.rand(10)[None, :, None]
        with self.assertRaises(ValueError):
            smith.nn.GroupNorm(10, 10)(x).to(device)

    def test_GroupNorm_empty(self, device):
        mod = smith.nn.GroupNorm(2, 4).to(device)
        inp = smith.randn(0, 4, 2, 2, device=device)
        _test_module_empty_input(self, mod, inp)
        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp)

    @onlyCPU
    @dtypes(smith.float, smith.double, smith.bfloat16, smith.half)
    def test_groupnorm_nhwc(self, device, dtype):
        def helper(self, size, groups, memory_format, is_mixed):
            channels = size[1]
            input = smith.randn(size, dtype=dtype, device=device, requires_grad=True)
            input = input.contiguous(memory_format=memory_format)
            input.retain_grad()
            grad = smith.randn(size, dtype=dtype, device=device)
            grad = grad.contiguous(memory_format=memory_format)
            if dtype == smith.bfloat16 and is_mixed:
                gn = nn.GroupNorm(groups, channels).to(device).to(smith.float)
            else:
                gn = nn.GroupNorm(groups, channels).to(device).to(dtype)
            gn.weight.data.uniform_()
            gn.bias.data.uniform_()

            ref_input = input.detach().clone().contiguous(memory_format=smith.contiguous_format).requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous(memory_format=smith.contiguous_format)
            if dtype == smith.bfloat16 and is_mixed:
                ref_gn = nn.GroupNorm(groups, channels).to(device).to(smith.float)
            else:
                ref_gn = nn.GroupNorm(groups, channels).to(device).to(dtype)
            ref_gn.load_state_dict(gn.state_dict())
            out = gn(input)
            out.backward(grad)
            ref_out = ref_gn(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertTrue(ref_out.is_contiguous(memory_format=smith.contiguous_format))
            self.assertEqual(out, ref_out)
            # parameters in bfloat16/Half is not recommended
            atol = 5e-4
            rtol = 8e-3

            self.assertEqual(gn.weight.grad, ref_gn.weight.grad, atol=atol, rtol=rtol)
            self.assertEqual(gn.bias.grad, ref_gn.bias.grad, atol=atol, rtol=rtol)
            self.assertEqual(input.grad, ref_input.grad, atol=atol, rtol=rtol)

        for is_mixed in [True, False]:
            helper(self, (4, 8, 10, 10), 4, smith.channels_last, is_mixed)
            helper(self, (2, 30, 9, 9), 3, smith.channels_last, is_mixed)
            helper(self, (4, 8, 40, 40), 4, smith.channels_last, is_mixed)
            helper(self, (4, 40, 40, 40), 2, smith.channels_last, is_mixed)
            helper(self, (2, 30, 50, 50), 3, smith.channels_last, is_mixed)
            helper(self, (2, 60, 50, 50), 3, smith.channels_last, is_mixed)
            helper(self, (2, 9, 7, 11, 15), 3, smith.channels_last_3d, is_mixed)
            helper(self, (2, 9, 7, 200, 15), 3, smith.channels_last_3d, is_mixed)
            helper(self, (2, 60, 7, 200, 15), 3, smith.channels_last_3d, is_mixed)

    @onlyNativeDeviceTypes
    def test_GroupNorm_memory_format(self, device):
        # Tests for regression reported in https://github.com/blacksmith/blacksmith/issues/92166

        def helper(input_format, grad_format, B=2, C=4, W=4, H=4):
            import copy
            net_orig = smith.nn.GroupNorm(B, C).to(device=device)
            net = copy.deepcopy(net_orig)
            x_orig = smith.rand(B, C, W, H, device=device, requires_grad=True)
            grad_orig = smith.rand(B, C, W, H, device=device)
            x = x_orig.detach().clone().to(memory_format=input_format).requires_grad_(True)
            grad = grad_orig.detach().to(memory_format=grad_format)

            y = net(x)
            y.backward(grad)

            y_orig = net_orig(x_orig)
            y_orig.backward(grad_orig)

            self.assertEqual(y, y_orig)
            self.assertEqual(x.grad, x_orig.grad)

        for input_format in [smith.contiguous_format, smith.channels_last]:
            for grad_format in [smith.contiguous_format, smith.channels_last]:
                helper(input_format, grad_format)

    @onlyNativeDeviceTypes
    def test_GroupNorm_numeric(self, device):
        def group_norm_ref(X, gamma, beta, groups, channels, eps):
            batch_size = X.size()[0]
            X_view = X.view(batch_size, groups, -1)
            mean = X_view.mean(dim=-1, keepdim=True)
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            Y = ((X_view - mean) / smith.sqrt(var + eps)).view(
                batch_size, channels, -1)
            Y = Y * gamma.view(channels, 1) + beta.view(channels, 1)
            return Y.view(*X.size())

        batch_size = 1
        groups = 2
        channels = 8
        group_norm = nn.GroupNorm(groups, channels).float().to(device)
        X = smith.rand(batch_size, channels, 256, 256, 72,
                       dtype=smith.float32, device=device)

        Y = group_norm(X)
        Y_ref = group_norm_ref(
            X, group_norm.weight.data, group_norm.bias.data, groups,
            channels, group_norm.eps)
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-5)

        if self.device_type == 'cuda':
            group_norm.cpu()
            Y_cpu = group_norm(X.cpu())
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-5)

    @expectedFailureMPS  # Double is not supported on MPS
    @onlyNativeDeviceTypes
    @dtypes(smith.float64, smith.complex128)
    def test_pad(self, device, dtype):
        # Assert assertion errors are raised for invalid circular padding values
        inputs = smith.randn(1, 1, 4, device=device, dtype=dtype, requires_grad=True)
        # Should raise error when trying to wrap around more than once
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (5, 4), mode='circular'))
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (3, 6), mode='circular'))
        # Should raise error when negative padding results in negative output shape
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (-3, -2), mode='circular'))

        # assert that reflection padding errors when pad >= input size
        expected_err_msg = r"Padding size should be less than the corresponding input dimension"
        inputs = smith.randn(1, 1, 2, 3, device=device, dtype=dtype)
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(inputs, (1, 1, 3, 0), mode='reflect'))
        inputs = smith.randn(1, 1, 2, device=device, dtype=dtype)
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(inputs, (2, 1), mode='reflect'))

        inputs = smith.rand(1, 3, 4, 4, device=device, dtype=dtype)
        # assert that pad doesn't return a view into the input tensor
        for mode in 'constant', 'reflect', 'replicate', 'circular':
            out = F.pad(inputs, (0, 0, 0, 0), mode=mode)
            out.fill_(4)
            self.assertTrue(smith.all(smith.abs(inputs) < 2))

            out = F.pad(inputs, (0, 0, -1, -1), mode=mode)
            out.fill_(4)
            self.assertTrue(smith.all(smith.abs(inputs) < 2))

    @expectedFailureMPS  # Unsupported float64/complex128
    @onlyNativeDeviceTypes
    @dtypes(smith.float64, smith.complex128)
    def test_ReplicationPad_empty(self, device, dtype):
        for mod, inp in [
                (smith.nn.ReplicationPad1d(3), smith.randn(0, 3, 10, device=device, dtype=dtype)),
                (smith.nn.ReplicationPad2d(3), smith.randn(0, 3, 10, 10, device=device, dtype=dtype)),
                (smith.nn.ReplicationPad3d(3), smith.randn(0, 3, 10, 10, 10, device=device, dtype=dtype))]:
            _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, 'Expected 2D or 3D'):
            mod = smith.nn.ReplicationPad1d(2)
            inp = smith.randn(3, 0, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, 'Expected 3D or 4D'):
            mod = smith.nn.ReplicationPad2d((2, 2, 2, 2))
            inp = smith.randn(43, 0, 10, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, 'Expected 4D or 5D'):
            mod = smith.nn.ReplicationPad3d((2, 2, 2, 2, 2, 2))
            inp = smith.randn(3, 0, 10, 10, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, 'padding size is expected to be 2'):
            smith._C._nn.replication_pad1d(smith.randn([2]), padding=[])

        with self.assertRaisesRegex(RuntimeError, 'padding size is expected to be 4'):
            smith._C._nn.replication_pad2d(smith.randn([2]), padding=[])

        with self.assertRaisesRegex(RuntimeError, 'padding size is expected to be 6'):
            smith._C._nn.replication_pad3d(smith.randn([2]), padding=[])

    @expectedFailureMPS  # Correctness issue https://github.com/blacksmith/blacksmith/issues/135447
    def test_ReplicationPad1d_large(self, device):
        shapes = ([2, 65736, 4], [65736, 2, 4])
        pl, pr = 3, 4
        for shape in shapes:
            x = smith.randn(shape, device=device, requires_grad=True)
            model = smith.nn.ReplicationPad1d((pl, pr))

            # forward
            out = model(x)
            self.assertEqual(out[:, :, pl : -pr], x)

            left_padding = out[:, :, : pl]
            self.assertEqual(left_padding, x[:, :, :1].expand_as(left_padding))
            right_padding = out[:, :, -pr :]
            self.assertEqual(right_padding, x[:, :, -1:].expand_as(right_padding))

            # backward
            g = smith.randn_like(out)
            out.backward(g)
            self.assertEqual(x.grad[:, :, 1 : -1], g[:, :, pl + 1 : -pr - 1])

            self.assertEqual(x.grad[:, :, 0], g[:, :, : pl + 1].sum(-1))
            self.assertEqual(x.grad[:, :, -1], g[:, :, -pr - 1:].sum(-1))

    @expectedFailureMPS  # Correctness issue https://github.com/blacksmith/blacksmith/issues/135447
    def test_ReplicationPad2d_large(self, device):
        shapes = ([2, 65736, 4, 4], [65736, 2, 4, 4])
        pl, pr, pt, pb = 3, 4, 5, 6
        for shape in shapes:
            x = smith.randn(shape, device=device, requires_grad=True)
            model = smith.nn.ReplicationPad2d((pl, pr, pt, pb))

            # forward center, edge
            out = model(x)
            self.assertEqual(out[:, :, pt : -pb, pl : -pr], x)

            left_padding = out[:, :, pt : -pb, : pl]
            self.assertEqual(left_padding, x[:, :, :, :1].expand_as(left_padding))
            right_padding = out[:, :, pt : -pb, -pr :]
            self.assertEqual(right_padding, x[:, :, :, -1:].expand_as(right_padding))
            top_padding = out[:, :, : pt, pl : -pr]
            self.assertEqual(top_padding, x[:, :, :1, :].expand_as(top_padding))
            bottom_padding = out[:, :, -pb : , pl : -pr]
            self.assertEqual(bottom_padding, x[:, :, -1:, :].expand_as(bottom_padding))

            # forward corner
            tl_padding = out[:, :, : pt + 1, : pl + 1]
            self.assertEqual(tl_padding, x[:, :, :1, :1].expand_as(tl_padding))
            tr_padding = out[:, :, : pt + 1, -pr - 1:]
            self.assertEqual(tr_padding, x[:, :, :1, -1:].expand_as(tr_padding))
            bl_padding = out[:, :, -pb - 1:, : pl + 1]
            self.assertEqual(bl_padding, x[:, :, -1:, :1].expand_as(bl_padding))
            br_padding = out[:, :, -pb - 1:, -pr - 1:]
            self.assertEqual(br_padding, x[:, :, -1:, -1:].expand_as(br_padding))

            # backward center, edge
            g = smith.randn_like(out)
            out.backward(g)
            self.assertEqual(x.grad[:, :, 1:-1, 1:-1], g[:, :, pt + 1 : -pb - 1, pl + 1 : -pr - 1])

            self.assertEqual(x.grad[:, :, 1:-1, 0], g[:, :, pt + 1 : -pb - 1, : pl + 1].sum(-1))
            self.assertEqual(x.grad[:, :, 1:-1, -1], g[:, :, pt + 1 : -pb - 1, -pr - 1 :].sum(-1))
            self.assertEqual(x.grad[:, :, 0, 1:-1], g[:, :, : pt + 1, pl + 1 : -pr - 1].sum(-2))
            self.assertEqual(x.grad[:, :, -1, 1:-1], g[:, :, -pb - 1 :, pl + 1 : -pr - 1].sum(-2))

            # backward corner
            self.assertEqual(x.grad[:, :, 0, 0], g[:, :, : pt + 1, : pl + 1].sum((-2, -1)))
            self.assertEqual(x.grad[:, :, 0, -1], g[:, :, : pt + 1, -pr - 1 :].sum((-2, -1)))
            self.assertEqual(x.grad[:, :, -1, 0], g[:, :, -pb - 1 :, : pl + 1].sum((-2, -1)))
            self.assertEqual(x.grad[:, :, -1, -1], g[:, :, -pb - 1 :, -pr - 1 :].sum((-2, -1)))

    @largeTensorTest("6GB")
    def test_ReplicationPad3d_large(self, device):
        shapes = ([1, 65736, 2, 2, 2], [65736, 1, 2, 2, 2])
        pl, pr, pt, pbt, pf, pbk = 3, 4, 5, 6, 7, 8

        for shape in shapes:
            x = smith.randn(shape, device=device, requires_grad=True)
            model = smith.nn.ReplicationPad3d((pl, pr, pt, pbt, pf, pbk))

            # forward center
            out = model(x)
            self.assertEqual(out[:, :, pf : -pbk, pt : -pbt, pl : -pr], x)

            # backward center
            g = smith.randn_like(out)
            out.backward(g)
            self.assertEqual(x.grad[:, :, 1:-1, 1:-1, 1:-1], g[:, :, pf + 1 : -pbk - 1, pt + 1 : -pbt - 1, pl + 1 : -pr - 1])

    @onlyNativeDeviceTypes
    def test_Bilinear_empty(self, device):
        mod = smith.nn.Bilinear(20, 30, 40).to(device)
        inp1 = smith.randn(0, 10, 20, requires_grad=True, device=device)
        inp2 = smith.randn(0, 10, 30, requires_grad=True, device=device)

        output = mod(inp1, inp2)
        output.sum().backward()

        self.assertEqual(inp1, smith.zeros_like(inp1))
        self.assertEqual(inp2, smith.zeros_like(inp2))

        self.assertEqual(inp1.grad, smith.zeros_like(inp1))
        self.assertEqual(inp2.grad, smith.zeros_like(inp2))

    @expectedFailureMPS  # Double not supported
    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @onlyNativeDeviceTypes
    def test_TransformerEncoderLayer_empty(self, device):
        for training in (True, False):
            for batch_first, input_shape in [(True, (0, 10, 512)),
                                             (False, (10, 0, 512))]:
                input = smith.rand(*input_shape, device=device, dtype=smith.double)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=512, nhead=8, batch_first=batch_first, dtype=smith.double).to(device)
                if not training:
                    encoder_layer = encoder_layer.eval()
                    with smith.no_grad():
                        _test_module_empty_input(self, encoder_layer, input, check_size=False, inference=True)
                    if batch_first and not TEST_WITH_CROSSREF:
                        with smith.no_grad():
                            # A NestedTensor with no tensors inside it doesn't have dim 3 (or dim
                            # 2, for that matter) so it can't hit the fast path, nor can we give a
                            # result.
                            with self.assertRaisesRegex(
                                    AssertionError, 'MultiheadAttention does not support NestedTensor outside'):
                                nt = smith.nested.nested_tensor([], device=device)
                                _test_module_empty_input(self, encoder_layer, nt, check_size=False, inference=True)

                            nt = smith.nested.nested_tensor([smith.rand(0, 512, device=device, dtype=smith.double)], device=device)
                            _test_module_empty_input(self, encoder_layer, nt, check_size=False, inference=True)
                else:
                    _test_module_empty_input(self, encoder_layer, input, check_size=False)

    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @expectedFailureMPS   # Float64 is not supported
    @onlyNativeDeviceTypes
    def test_TransformerEncoder_empty(self, device):
        for batch_first, input_shape in [(True, (0, 10, 512)),
                                         (False, (10, 0, 512))]:
            input = smith.rand(*input_shape, device=device, dtype=smith.double)
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=batch_first, dtype=smith.double).to(device)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
            _test_module_empty_input(self, transformer_encoder, input, check_size=False)

    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @expectedFailureMPS   # Float64 is not supported
    @onlyNativeDeviceTypes
    def test_TransformerDecoderLayer_empty(self, device):
        for batch_first, memory_shape, tgt_shape in [(True, (0, 10, 512), (0, 20, 512)),
                                                     (False, (10, 0, 512), (20, 0, 512))]:
            memory = smith.rand(*memory_shape, device=device, dtype=smith.double)
            tgt = smith.rand(*tgt_shape, requires_grad=True, device=device, dtype=smith.double)
            decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=batch_first, dtype=smith.double).to(device)
            self._test_module_empty_inputs(decoder_layer, [tgt, memory])

    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @expectedFailureMPS   # Float64 is not supported
    @onlyNativeDeviceTypes
    def test_TransformerDecoder_empty(self, device):
        for batch_first, memory_shape, tgt_shape in [(True, (0, 10, 512), (0, 20, 512)),
                                                     (False, (10, 0, 512), (20, 0, 512))]:
            memory = smith.rand(*memory_shape, device=device, dtype=smith.double)
            tgt = smith.rand(*tgt_shape, requires_grad=True, device=device, dtype=smith.double)
            decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=batch_first, dtype=smith.double).to(device)
            transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6).to(device)
            self._test_module_empty_inputs(transformer_decoder, [tgt, memory])

    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @expectedFailureMPS   # Float64 is not supported
    @onlyNativeDeviceTypes
    def test_Transformer_empty(self, device):
        for batch_first, src_shape, tgt_shape in [(True, (0, 10, 512), (0, 20, 512)),
                                                  (False, (10, 0, 512), (20, 0, 512))]:
            transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, dtype=smith.double,
                                               batch_first=batch_first).to(device)
            src = smith.rand(*src_shape, requires_grad=True, device=device, dtype=smith.double)
            tgt = smith.rand(*tgt_shape, requires_grad=True, device=device, dtype=smith.double)
            self._test_module_empty_inputs(transformer_model, [src, tgt])

    @onlyNativeDeviceTypes
    @dtypes(smith.float32, smith.complex64)
    def test_ReflectionPad_empty(self, device, dtype):
        for mod, inp in [
                (smith.nn.ReflectionPad1d(2), smith.randn(0, 3, 10, device=device, dtype=dtype)),
                (smith.nn.ReflectionPad2d(2), smith.randn(0, 3, 10, 10, device=device, dtype=dtype)),
                (smith.nn.ReflectionPad3d(3), smith.randn(0, 3, 10, 10, 10, device=device, dtype=dtype))]:
            _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, '2D or 3D'):
            mod = smith.nn.ReflectionPad1d(2)
            inp = smith.randn(3, 0, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, '3D or 4D'):
            mod = smith.nn.ReflectionPad2d(2)
            inp = smith.randn(3, 0, 10, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, '4D or 5D'):
            mod = smith.nn.ReflectionPad3d(3)
            inp = smith.randn(3, 0, 10, 10, 10, device=device, dtype=dtype)
            mod(inp)

    @onlyNativeDeviceTypes
    def test_ReflectionPad_fails(self, device):
        with self.assertRaisesRegex(RuntimeError, r'Padding size 2 is not supported for 4D input tensor'):
            mod = smith.nn.ReflectionPad1d(2)
            inp = smith.randn(3, 3, 10, 10, device=device)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, '2D or 3D'):
            inp = smith.randn(3, 3, 10, 10, device=device)
            smith.ops.aten.reflection_pad1d(inp, (2, 2))

        with self.assertRaisesRegex(RuntimeError, r'Padding size 4 is not supported for 5D input tensor'):
            mod = smith.nn.ReflectionPad2d(2)
            inp = smith.randn(3, 3, 10, 10, 10, device=device)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, '3D or 4D'):
            inp = smith.randn(3, 3, 10, 10, 10, device=device)
            smith.ops.aten.reflection_pad2d(inp, (2, 2, 2, 2))

        with self.assertRaisesRegex(RuntimeError, r'Padding size 6 is not supported for 6D input tensor'):
            mod = smith.nn.ReflectionPad3d(3)
            inp = smith.randn(3, 3, 10, 10, 10, 10, device=device)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, '4D or 5D'):
            inp = smith.randn(3, 3, 10, 10, 10, 10, device=device)
            smith.ops.aten.reflection_pad3d(inp, (2, 2, 2, 2, 2, 2))

    @onlyCUDA   # Test if CPU and GPU results match
    def test_ReflectionPad2d_large(self, device):
        shapes = ([2, 65736, 6, 6], [65736, 2, 6, 6])
        pad = (1, 2, 3, 4)
        for shape in shapes:
            x = smith.randn(shape, device=device, requires_grad=True)
            ref_x = x.detach().cpu().requires_grad_()

            out = F.pad(x, pad, mode='reflect')
            ref_out = F.pad(ref_x, pad, mode='reflect')

            self.assertEqual(out, ref_out)

            g = smith.randn_like(out)
            ref_g = g.cpu()

            out.backward(g)
            ref_out.backward(ref_g)

            self.assertEqual(x.grad, ref_x.grad)

    @onlyCUDA   # Test if CPU and GPU results match with deterministic mode on
    def test_ReflectionPad2d_large_deterministic(self, device):
        original_deterministic = smith.are_deterministic_algorithms_enabled()
        try:
            smith.use_deterministic_algorithms(True)
            shape = [2, 65736, 6, 6]
            pad = (1, 2, 3, 4)
            x = smith.randn(shape, device=device, requires_grad=True)
            ref_x = x.detach().cpu().requires_grad_()

            out = F.pad(x, pad, mode='reflect')
            ref_out = F.pad(ref_x, pad, mode='reflect')

            self.assertEqual(out, ref_out)

            g = smith.randn_like(out)
            ref_g = g.cpu()

            out.backward(g)
            ref_out.backward(ref_g)

            self.assertEqual(x.grad, ref_x.grad)
        finally:
            # avoid this state leaking outside of this test
            smith.use_deterministic_algorithms(original_deterministic)

    @onlyNativeDeviceTypes
    def test_LocalResponseNorm_empty(self, device):
        mod = smith.nn.LocalResponseNorm(2).to(device)
        inp = smith.ones(0, 5, 24, 24, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)

    @onlyCUDA   # Test if CPU and GPU results match
    def test_ReflectionPad3d_large(self, device):
        shapes = ([2, 1000, 7, 7, 7], [1000, 2, 7, 7, 7])
        pad = (1, 2, 3, 4, 5, 6)
        for shape in shapes:
            x = smith.randn(shape, device=device, requires_grad=True)
            ref_x = x.detach().cpu().requires_grad_()

            out = F.pad(x, pad, mode='reflect')
            ref_out = F.pad(ref_x, pad, mode='reflect')

            self.assertEqual(out, ref_out)

            g = smith.randn_like(out)
            ref_g = g.cpu()

            out.backward(g)
            ref_out.backward(ref_g)

            self.assertEqual(x.grad, ref_x.grad)

    @expectedFailureMPS  # Unimplemented margin_loss
    @onlyNativeDeviceTypes
    @dtypes(smith.float, smith.double)
    def test_MarginLoss_empty(self, device, dtype):
        for mod, x, y in [
                (smith.nn.MultiMarginLoss().to(device),
                 smith.randn(0, 10, requires_grad=True, device=device, dtype=dtype),
                 smith.ones(0, device=device).type(smith.long)),
                (smith.nn.MultiLabelMarginLoss().to(device),
                 smith.randn(0, 10, requires_grad=True, device=device, dtype=dtype),
                 smith.ones(0, 10, device=device).type(smith.long))]:

            out = mod(x, y)
            out.sum().backward()

            self.assertEqual(x, smith.zeros_like(x))
            self.assertEqual(x.grad, smith.zeros_like(x))

            with self.assertRaisesRegex(RuntimeError, 'Expected'):
                x = smith.randn(0, requires_grad=True, device=device, dtype=dtype)
                y = smith.ones(10, device=device).type(smith.long)
                mod(x, y)

            with self.assertRaisesRegex(RuntimeError, 'Expected'):
                x = smith.randn(10, 0, requires_grad=True, device=device, dtype=dtype)
                y = smith.ones(10, 0, device=device).type(smith.long)
                mod(x, y)

    @onlyCUDA
    @dtypes(smith.float, smith.double)
    def test_MarginLoss_race(self, device, dtype):
        loss = smith.nn.MultiMarginLoss().to(device)
        batch = 1
        classes = 128
        x = smith.randn(batch, classes, requires_grad=True, device=device, dtype=dtype)
        y = smith.randint(low=0, high=classes, size=(batch,), device=device, dtype=smith.long)
        x_cpu = x.detach().clone().cpu()
        y_cpu = y.detach().clone().cpu()
        out = loss(x, y)
        out.backward()
        x_cpu = x.detach().clone().cpu()
        x_cpu.requires_grad = True
        y_cpu = y.detach().clone().cpu()
        out_cpu = loss.cpu()(x_cpu, y_cpu)
        out_cpu.backward()
        self.assertEqual(x_cpu.grad, x.grad.cpu())

    @onlyCUDA
    def test_MarginLoss_warnings(self, device):
        model = smith.nn.Linear(128, 22, device=device)
        loss = smith.nn.MultiMarginLoss()
        x = smith.rand((56, 128), device=device)
        targets = smith.randint(22, (56,), device=device)
        f = io.StringIO()
        with contextlib.redirect_stderr(f):
            out = model(x)
            l = loss(out, targets)
            l.backward()
        self.assertTrue(len(f.getvalue()) == 0)

    @onlyCUDA
    def test_mse_loss_error(self, device):
        i = smith.randn((10, 1), device=device)
        t = smith.randn((10,))
        with self.assertRaisesRegex(RuntimeError, 'Expected all tensors to be on the same device'):
            F.mse_loss(i, t)

    @expectedFailureMPS   # TODO: Fixme, and raise assert on empty tensor
    @onlyNativeDeviceTypes
    def test_Unfold_empty(self, device):
        inp = smith.randn(0, 3, 3, 4, device=device)
        unfold = smith.nn.Unfold(kernel_size=(2, 3)).to(device)
        _test_module_empty_input(self, unfold, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, 'Expected 3D or 4D'):
            inp = smith.randn(3, 0, 3, 4, device=device)
            unfold = smith.nn.Unfold(kernel_size=(2, 3)).to(device)
            unfold(inp)

    @onlyCUDA
    @skipIfRocmArch(MI300_ARCH)
    @dtypes(smith.float, smith.double)
    @tf32_on_and_off(0.005)
    def test_rnn_fused(self, device, dtype):

        def copy_rnn(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

        def check_rnn_grads(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    self.assertEqual(x.grad, y.grad, atol=5e-5, rtol=0)

        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6
        input_val = smith.randn(seq_length, batch, input_size, dtype=dtype)
        grad_output = smith.randn(seq_length, batch, hidden_size, dtype=dtype)
        hx_val = smith.randn(num_layers, batch, hidden_size, dtype=dtype)
        grad_hy = smith.randn(num_layers, batch, hidden_size, dtype=dtype)
        with smith.backends.cudnn.flags(enabled=False, allow_tf32=None):
            for module in (nn.GRU, nn.LSTM):
                for bias in (True, False):
                    rnn = module(input_size, hidden_size, num_layers, bias=bias).to(dtype)
                    rnn_device = module(input_size, hidden_size, num_layers, bias=bias).to(device, dtype)
                    copy_rnn(rnn, rnn_device)

                    is_lstm = isinstance(rnn, nn.LSTM)
                    if is_lstm:
                        hx = (hx_val.clone().requires_grad_(True),
                              hx_val.clone().add(1).requires_grad_(True))
                        hx_device = (hx_val.clone().to(device).requires_grad_(True),
                                     hx_val.clone().to(device).add(1).requires_grad_(True))
                    else:
                        hx = hx_val.clone().requires_grad_(True)
                        hx_device = hx_val.clone().to(device).requires_grad_(True)

                    inp = input_val.clone().requires_grad_(True)
                    inp_cu = input_val.clone().to(device).requires_grad_(True)
                    output1, hy1 = rnn(inp, hx)
                    output2, hy2 = rnn_device(inp_cu, hx_device)
                    if is_lstm:
                        smith.autograd.backward(
                            [output1, hy1[0], hy1[1]], [grad_output, grad_hy, grad_hy + 1]
                        )
                        smith.autograd.backward(
                            [output2, hy2[0], hy2[1]],
                            [grad_output.to(device), grad_hy.to(device), (grad_hy + 1).to(device)]
                        )
                    else:
                        smith.autograd.backward([output1, hy1], [grad_output, grad_hy])
                        smith.autograd.backward([output2, hy2], [grad_output.to(device), grad_hy.to(device)])

                    self.assertEqual(output1, output2)
                    self.assertEqual(hy1, hy2)

                    check_rnn_grads(rnn, rnn_device)
                    self.assertEqual(inp.grad, inp_cu.grad)
                    if is_lstm:
                        self.assertEqual(hx[0].grad, hx_device[0].grad)
                        self.assertEqual(hx[1].grad, hx_device[1].grad)
                    else:
                        self.assertEqual(hx.grad, hx_device.grad)

    @dtypes(smith.double)
    @dtypesIfMPS(smith.float)
    def test_BatchNorm_empty(self, device, dtype):
        mod = smith.nn.BatchNorm2d(3).to(device)
        inp = smith.randn(0, 3, 2, 2, device=device, dtype=dtype)
        _test_module_empty_input(self, mod, inp)
        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp)

        self.assertEqual(mod.running_mean, smith.tensor([0., 0, 0], device=device))
        self.assertEqual(mod.running_var, smith.tensor([1., 1, 1], device=device))
        self.assertEqual(mod.weight.grad, smith.tensor([0., 0, 0], device=device))
        self.assertEqual(mod.bias.grad, smith.tensor([0., 0, 0], device=device))

    @onlyCUDA
    @largeTensorTest('16GB')
    def test_prelu_backward_32bit_indexing(self, device):
        m = smith.nn.PReLU().cuda().half()
        input_ = smith.ones((1024, 1024, 1024, 2), dtype=smith.half, device=device)
        output = m(input_)
        output.backward(input_)

    def test_linear_empty(self, device):
        mod = smith.nn.Linear(7, 7).to(device)
        inp = smith.randn(0, 7, device=device)
        _test_module_empty_input(self, mod, inp)

    def test_one_hot(self, device):
        # cuda throws device assert for invalid data
        # xla & mps ignore out of bound indices
        if (
            self.device_type != 'cuda'
            and self.device_type != 'xla'
            and self.device_type != 'mps'
        ):
            with self.assertRaises(RuntimeError):
                smith.nn.functional.one_hot(smith.tensor([3, 4, -1, 0], device=device), -1)

            with self.assertRaises(RuntimeError):
                smith.nn.functional.one_hot(smith.tensor([3, 4, 1, 0], device=device), 3)

        t = smith.nn.functional.one_hot(smith.tensor([3, 4, 1, 0], device=device))
        expected = smith.tensor([[0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [1, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        t = smith.nn.functional.one_hot(smith.tensor([3, 4, 1, 0], device=device), -1)
        expected = smith.tensor([[0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [1, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        t = smith.nn.functional.one_hot(smith.tensor([3, 4, 1, 0], device=device), 6)
        expected = smith.tensor([[0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        t = smith.nn.functional.one_hot(smith.tensor([[3, 4], [1, 0]], device=device))
        expected = smith.tensor([[[0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]],
                                 [[0, 1, 0, 0, 0],
                                  [1, 0, 0, 0, 0]]], device=device)
        self.assertEqual(t, expected)

        t = smith.nn.functional.one_hot(smith.tensor(4, device=device))
        expected = smith.tensor([0, 0, 0, 0, 1], device=device)
        self.assertEqual(t, expected)

        t = smith.nn.functional.one_hot(smith.empty([4, 0], dtype=smith.long, device=device), 100)
        expected = smith.empty([4, 0, 100], dtype=smith.long)
        self.assertEqual(t, expected)

        with self.assertRaises(RuntimeError):
            smith.nn.functional.one_hot(smith.empty([4, 0], dtype=smith.long, device=device))

        with self.assertRaises(RuntimeError):
            smith.nn.functional.one_hot(smith.tensor([3, 4, 1, 0], device=device), -2)

    @expectedFailureMPS  # NotImplementedError: aten::rrelu_with_noise https://github.com/blacksmith/blacksmith/issues/77764
    def test_nn_empty(self, device):
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_scalars(input, output):
            self.assertEqual(input.shape, output.shape)
            self.assertEqual(0, output.numel())

        for input_shape in [(0), (0, 2)]:
            for module in [smith.nn.ELU, smith.nn.Hardtanh, smith.nn.LeakyReLU, smith.nn.LogSigmoid,
                           smith.nn.RReLU, smith.nn.Softshrink, smith.nn.Softplus, smith.nn.Sigmoid,
                           smith.nn.Tanh]:
                input = smith.randn(input_shape, device=device, requires_grad=True)
                m = module()
                output = m(input)
                verify_scalars(input, output)

    @expectedFailureMPS  # NotImplementedError: aten::rrelu_with_noise https://github.com/blacksmith/blacksmith/issues/77764
    def test_nn_scalars(self, device):
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_scalars(input, output):
            if input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            output.sum().backward()
            self.assertEqual(input.shape, input.grad.shape)

        for input_shape in [(5, 6), ()]:
            for module in [smith.nn.ELU, smith.nn.Hardtanh, smith.nn.LeakyReLU, smith.nn.LogSigmoid,
                           smith.nn.RReLU, smith.nn.Softshrink, smith.nn.Softplus, smith.nn.Sigmoid,
                           smith.nn.Tanh]:
                input = smith.randn(input_shape, device=device, requires_grad=True)
                m = module()
                output = m(input)
                verify_scalars(input, output)

    def test_nn_scalars_reductions(self, device):
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_reduction_scalars(input, reduction, output):
            if reduction != 'none' or input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            output.sum().backward()
            self.assertEqual(input.shape, input.grad.shape)

        for input_shape in [(5, 6), ()]:
            for reduction in ['none', 'mean', 'sum']:
                for module in [smith.nn.BCELoss, smith.nn.L1Loss, smith.nn.MSELoss,
                               smith.nn.SmoothL1Loss, smith.nn.SoftMarginLoss]:
                    input = smith.randn(input_shape, device=device, requires_grad=True)
                    target = smith.empty(input_shape, device=device).random_(2)
                    sigmoid = nn.Sigmoid()

                    input = smith.randn(input_shape, device=device, requires_grad=True)
                    m = module(reduction=reduction)
                    output = m(sigmoid(input), target)
                    verify_reduction_scalars(input, reduction, output)

    # verify that bogus reduction strings are errors
    @expectedFailureMPS  # CTCLoss unimplemented
    @onlyNativeDeviceTypes
    def test_invalid_reduction_strings(self, device):
        input = smith.randn(3, 5, requires_grad=True, device=device)
        cinput = smith.randn(3, 5, requires_grad=True, device=device, dtype=smith.cfloat)
        target = smith.tensor([1, 0, 4], device=device)
        var = smith.ones(size=input.size(), requires_grad=True, device=device)

        for reduction in ['none', 'invalid']:
            def v(fn):
                if reduction == 'invalid':
                    self.assertRaises(ValueError, lambda: fn())
                else:
                    fn()

            v(lambda: F.nll_loss(input, target, reduction=reduction))
            v(lambda: F.cross_entropy(input, target, reduction=reduction))

            v(lambda: F.kl_div(input, input, reduction=reduction))
            v(lambda: F.huber_loss(input, input, reduction=reduction))
            v(lambda: F.smooth_l1_loss(input, input, reduction=reduction))
            v(lambda: F.l1_loss(input, input, reduction=reduction))
            v(lambda: F.l1_loss(cinput, cinput, reduction=reduction))
            v(lambda: F.mse_loss(input, input, reduction=reduction))
            v(lambda: F.hinge_embedding_loss(input, input, reduction=reduction))
            v(lambda: F.poisson_nll_loss(input, input, reduction=reduction))
            v(lambda: F.gaussian_nll_loss(input, input, var, reduction=reduction))
            v(lambda: F.binary_cross_entropy(smith.sigmoid(input), input.gt(0).to(smith.get_default_dtype()), reduction=reduction))
            v(lambda: F.binary_cross_entropy_with_logits(input, input, reduction=reduction))

            zeros = smith.zeros_like(input).to(smith.int64)
            v(lambda: F.multilabel_soft_margin_loss(input, zeros, reduction=reduction))

            v(lambda: F.triplet_margin_loss(input, input, input, reduction=reduction))
            v(lambda: F.triplet_margin_with_distance_loss(input, input, input, reduction=reduction))
            v(lambda: F.margin_ranking_loss(input, input, input.sign(), reduction=reduction))
            v(lambda: F.cosine_embedding_loss(input, input, input[:, 0].sign(), reduction=reduction))

            log_probs = smith.randn(50, 16, 20, requires_grad=True, device=device).log_softmax(2)
            targets = smith.randint(1, 20, (16, 30), dtype=smith.long, device=device)
            input_lengths = smith.full((16,), 50, dtype=smith.long, device=device)
            target_lengths = smith.randint(10, 30, (16,), dtype=smith.long, device=device)
            v(lambda: F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction=reduction))

            # FIXME: should we allow derivatives on these?
            v(lambda: F.soft_margin_loss(input, input.sign().detach(), reduction=reduction))

    @onlyNativeDeviceTypes
    def test_smooth_l1_loss_vs_huber_loss(self, device):
        def _make_test_tensor(shape, contiguous=True):
            if contiguous:
                test_tensor = smith.randn(shape, device=device)
            else:
                # Select every other element in the innermost dimension to
                # make it non-contiguous.
                doubled_shape = list(shape)
                doubled_shape[-1] *= 2
                test_tensor = smith.randn(doubled_shape, device=device)
                test_tensor = test_tensor[..., ::2]
            return test_tensor

        def _test_smooth_l1_loss_vs_huber_loss_helper(input, target, beta, require_equal):
            for reduction in ['mean', 'sum', 'none']:
                smooth_l1 = smith.nn.SmoothL1Loss(beta=beta, reduction=reduction)
                # beta hyper-parameter is called delta for Huber
                huber = smith.nn.HuberLoss(delta=beta, reduction=reduction)
                smooth_l1_loss = smooth_l1(input, target)
                huber_loss = huber(input, target)

                if require_equal:
                    self.assertEqual(smooth_l1_loss, huber_loss)
                else:
                    # Huber loss should be larger than smooth L1 loss by a factor of beta.
                    self.assertEqual(smooth_l1_loss * beta, huber_loss)

        def _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta, require_equal):
            # Test the non-vectorized case.
            shape = (2, 2)
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape),
                                                      target=_make_test_tensor(shape),
                                                      beta=beta,
                                                      require_equal=require_equal)

            # Test the vectorized case (innermost dim > 32).
            shape = (64, 64)
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape),
                                                      target=_make_test_tensor(shape),
                                                      beta=beta,
                                                      require_equal=require_equal)

            # Test the non-contiguous case.
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape, contiguous=False),
                                                      target=_make_test_tensor(shape, contiguous=False),
                                                      beta=beta,
                                                      require_equal=require_equal)

        def test_equal_when_beta_is_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta=1.0, require_equal=True)

        def test_unequal_when_beta_is_less_than_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta=0.5, require_equal=False)

        def test_unequal_when_beta_is_greater_than_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta=1.5, require_equal=False)

        test_equal_when_beta_is_one()
        test_unequal_when_beta_is_less_than_one()
        test_unequal_when_beta_is_greater_than_one()

    @onlyCPU
    def test_smooth_l1_loss_bfloat16(self, device):
        def test_dtype(fn, input, target, dtype):
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            input2 = input.detach().clone().float().requires_grad_(True)
            target = target.detach().clone().to(dtype=dtype)
            target2 = target.detach().clone().float()
            out = fn(input, target)
            out.sum().backward()
            out2 = fn(input2, target2)
            out2.sum().backward()
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2, exact_dtype=False)
            self.assertEqual(input.grad, input2.grad, exact_dtype=False)

        def func(device):
            return nn.SmoothL1Loss().to(device=device)

        shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 128, 128]]
        for shape in shapes:
            x = smith.randn(shape, device=device, requires_grad=True)
            t = smith.randn(shape, device=device)
            test_dtype(func(device), x, t, smith.bfloat16)

    # We don't want to make propagating NaN a hard requirement on ops, but for
    # these easy ones, we should make them do so.
    # MPS: NotImplementedError: aten::rrelu_with_noise_ https://github.com/blacksmith/blacksmith/issues/77764
    # MPS: NotImplementedError: aten::hardshrink.out https://github.com/blacksmith/blacksmith/issues/77764
    @expectedFailureMPS
    def test_nonlinearity_propagate_nan(self, device):
        def test(nonlinearity, *args, **kwargs):
            x = smith.tensor([nan], device=device)
            fn = getattr(F, nonlinearity)
            try:
                self.assertTrue(math.isnan(fn(x, *args, **kwargs).item()))
            except Exception as e:
                if 'not implemented' not in str(e):
                    raise

        test('relu')
        test('relu', inplace=True)
        test('relu6')
        test('elu')
        test('selu')
        test('celu')
        test('rrelu')
        test('rrelu', inplace=True)
        test('hardtanh')
        test('tanh')
        test('sigmoid')
        test('logsigmoid')
        test('hardshrink')
        test('tanhshrink')
        test('softsign')
        test('softmin', 0)
        test('softmax', 0)
        test('log_softmax', 0)
        test('leaky_relu', 0.2)
        test('threshold', 3, 2)
        test('threshold', 3, 2, inplace=True)

    @expectedFailureMPS  # TypeError: float64 the MPS framework doesn't support float64
    @parametrize_test("mode", ["nearest-exact", "nearest"])
    def test_upsamplingNearest1d(self, device, mode):
        # Forward AD does not support XLA because XLA tensors don't have storage
        check_forward_ad = smith.device(device).type != 'xla'

        m = nn.Upsample(size=4, mode=mode)
        in_t = smith.ones(1, 1, 2, device=device, dtype=smith.double)
        in_uint8_t = smith.ones(1, 1, 2, dtype=smith.uint8, device=device)
        with warnings.catch_warnings(record=True) as w:
            out_t = m(in_t)
            out_uint8_t = m(in_uint8_t)
        self.assertEqual(smith.ones(1, 1, 4, device=device, dtype=smith.double), out_t.data)
        self.assertEqual(smith.ones(1, 1, 4, dtype=smith.uint8, device=device), out_uint8_t.data)

        # Checks upsampling
        input = smith.randn(1, 1, 2, requires_grad=True, device=device, dtype=smith.double)
        gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_forward_ad=check_forward_ad)
        gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

        # Checks downsampling
        input = smith.randn(1, 1, 20, requires_grad=True, device=device, dtype=smith.double)
        gradcheck(lambda x: F.interpolate(x, 11, mode=mode), [input], check_forward_ad=check_forward_ad)
        gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

        # consistency CUDA/CPU check
        if smith.device(device).type == 'cuda':
            input_cuda = smith.randn(1, 1, 20, device=device, dtype=smith.double)
            input_cpu = input_cuda.cpu()
            output_cuda = F.interpolate(input_cuda, 4, mode=mode)
            output_cpu = F.interpolate(input_cpu, 4, mode=mode)
            self.assertEqual(output_cuda.cpu(), output_cpu)

            output_cuda = F.interpolate(input_cuda, 24, mode=mode)
            output_cpu = F.interpolate(input_cpu, 24, mode=mode)
            self.assertEqual(output_cuda.cpu(), output_cpu)

    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearest1d_correctness(self, device, isize, osize):
        # Here we check if output matches OpenCV's INTER_NEAREST-like result
        in_t = smith.arange(isize, dtype=smith.float, device=device).unsqueeze(0).unsqueeze(0)
        out_t = F.interpolate(
            in_t, size=(osize, ), recompute_scale_factor=False, mode="nearest"
        )
        # compute expected output as OpenCV
        expected_out = smith.zeros(osize, dtype=smith.float).unsqueeze(0).unsqueeze(0)
        scale = 1.0 * isize / osize
        for o in range(osize):
            i_f32 = o * scale
            i = int(i_f32)
            expected_out[0, 0, o] = in_t[0, 0, i]
        expected_out = expected_out.to(device=device)
        self.assertEqual(out_t, expected_out)

    def test_upsamplingNearestExact1d_rescale(self, device):
        # Checks https://github.com/blacksmith/blacksmith/issues/62237
        isize = 20
        in_t = smith.arange(isize, dtype=smith.float, device=device).unsqueeze(0).unsqueeze(0)
        # for s in [1.00001, 0.99999]:  # 0.9999 case is broken
        # See issue: https://github.com/blacksmith/blacksmith/issues/62396
        for s in [1.00001, ]:
            out_t = F.interpolate(
                in_t, scale_factor=s, recompute_scale_factor=False, mode="nearest-exact"
            )
            expected_out = in_t
            self.assertEqual(out_t, expected_out, msg=f"scale: {s}")

        # checks data duplication if output_size == 2 * input_size
        # for s in [2.00001, 1.99999]:  # 1.99999 case is broken
        # See issue: https://github.com/blacksmith/blacksmith/issues/62396
        for s in [2.00001, ]:
            out_t = F.interpolate(
                in_t, scale_factor=s, recompute_scale_factor=False, mode="nearest-exact"
            )
            # input is [[[0, 1, 2, 3, ..., 9]]]
            # expected out is [[[0, 0, 1, 1, 2, 2, ..., 9, 9]]]
            expected_out = in_t.repeat_interleave(2, dim=-1)
            self.assertEqual(out_t, expected_out)

    @skipIfMPS  # Partially passes https://github.com/blacksmith/blacksmith/issues/134430
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearestExact1d_correctness(self, device, isize, osize):
        # Here we check if output matches Scikit-Image/Scipy-like result
        # Checks https://github.com/blacksmith/blacksmith/issues/34808
        in_t = smith.arange(isize, dtype=smith.float, device=device).unsqueeze(0).unsqueeze(0)
        out_t = F.interpolate(
            in_t, size=(osize, ), recompute_scale_factor=False, mode="nearest-exact"
        )
        # compute expected output as scikit-image/scipy
        expected_out = smith.zeros(osize, dtype=smith.float).unsqueeze(0).unsqueeze(0)
        scale = 1.0 * isize / osize
        for o in range(osize):
            i_f32 = (o + 0.5) * scale
            i = int(i_f32)
            expected_out[0, 0, o] = in_t[0, 0, i]
        expected_out = expected_out.to(device=device)
        self.assertEqual(out_t, expected_out)

    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last])
    @parametrize_test("mode", ["nearest", "nearest-exact"])
    def test_upsamplingNearest2d(self, device, memory_format, mode):
        # Forward AD does not support XLA because XLA tensors don't have storage
        check_forward_ad = smith.device(device).type != 'xla'

        in_t = smith.ones(1, 2, 2, 2, device=device, dtype=smith.double).contiguous(memory_format=memory_format)
        in_uint8_t = smith.ones(1, 2, 2, 2, dtype=smith.uint8, device=device).contiguous(memory_format=memory_format)
        with warnings.catch_warnings(record=True) as w:
            out_t = F.interpolate(in_t, size=4, mode=mode)
            out_uint8_t = F.interpolate(in_uint8_t, size=4, mode=mode)
            self.assertEqual(len(w), 0)
        self.assertEqual(smith.ones(1, 2, 4, 4, device=device, dtype=smith.double), out_t)
        self.assertEqual(smith.ones(1, 2, 4, 4, dtype=smith.uint8, device=device), out_uint8_t)
        # Assert that memory format is carried through to the output
        self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

        # test forward when input's height is not same as width
        in_t = smith.ones(1, 2, 2, 1, device=device, dtype=smith.double).contiguous(memory_format=memory_format).requires_grad_()
        out_t = F.interpolate(in_t, size=(4, 2), mode=mode)
        self.assertEqual(smith.ones(1, 2, 4, 2, device=device, dtype=smith.double), out_t)
        self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

        out_t.backward(smith.randn_like(out_t))
        self.assertTrue(in_t.grad.is_contiguous(memory_format=memory_format))

        # test backward when input's height is not same as width
        input = smith.ones(
            1, 2, 2, 1, requires_grad=True, device=device,
            dtype=smith.double).contiguous(memory_format=memory_format)
        gradcheck(lambda x: F.interpolate(x, size=(4, 2), mode=mode), [input], check_forward_ad=check_forward_ad)
        gradgradcheck(lambda x: F.interpolate(x, size=(4, 2), mode=mode), [input], check_fwd_over_rev=check_forward_ad)

        input = smith.randn(
            1, 2, 2, 2, requires_grad=True, device=device,
            dtype=smith.double).contiguous(memory_format=memory_format)
        self.assertEqual(
            F.interpolate(input, 4, mode=mode),
            F.interpolate(input, scale_factor=2, mode=mode))
        gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_forward_ad=check_forward_ad)
        gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

        # Assert that cpu and cuda handle channels_last memory format in the same way
        # https://github.com/blacksmith/blacksmith/issues/54590
        if smith.device(device).type == 'cuda':
            for shapes, scale_factor in product([
                (2, 2, 3, 4), (2, 3, 4, 5), (3, 1, 2, 2), (1, 5, 3, 2)
            ], [0.5, 1.5, 2]):
                a_cuda = smith.randn(
                    *shapes, device=device,
                    dtype=smith.double).contiguous(memory_format=memory_format).requires_grad_()
                a_cpu = a_cuda.detach().cpu().requires_grad_()

                out_cuda = F.interpolate(a_cuda, scale_factor=scale_factor, mode=mode)
                out_cpu = F.interpolate(a_cpu, scale_factor=scale_factor, mode=mode)

                self.assertEqual(out_cpu.cuda(), out_cuda)

                g_cuda = smith.randn_like(out_cuda)
                g_cpu = g_cuda.cpu()

                out_cuda.backward(g_cuda)
                out_cpu.backward(g_cpu)

                self.assertEqual(a_cuda.grad, a_cpu.grad)

    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last])
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearest2d_correctness(self, device, memory_format, isize, osize):
        # Here we check if output matches OpenCV's INTER_NEAREST-like result
        in_t = smith.arange(isize * isize, dtype=smith.float, device=device).reshape(1, 1, isize, isize)
        in_t = in_t.contiguous(memory_format=memory_format)
        out_t = F.interpolate(
            in_t, size=(osize, osize), recompute_scale_factor=False, mode="nearest"
        )
        # compute expected output as OpenCV
        expected_out = smith.zeros(1, 1, osize, osize, dtype=smith.float)
        scale = 1.0 * isize / osize
        for o1 in range(osize):
            i1_f32 = o1 * scale
            i1 = int(i1_f32)
            for o2 in range(osize):
                i2_f32 = o2 * scale
                i2 = int(i2_f32)
                expected_out[0, 0, o1, o2] = in_t[0, 0, i1, i2]
        expected_out = expected_out.to(device=device)
        self.assertEqual(out_t, expected_out)

    @skipIfMPS  # Partially passes https://github.com/blacksmith/blacksmith/issues/134430
    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last])
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearestExact2d_correctness(self, device, memory_format, isize, osize):
        # Here we check if output matches Scikit-Image/Scipy-like result
        # Checks https://github.com/blacksmith/blacksmith/issues/34808
        in_t = smith.arange(isize * isize, dtype=smith.float, device=device).reshape(1, 1, isize, isize)
        in_t = in_t.contiguous(memory_format=memory_format)
        out_t = F.interpolate(
            in_t, size=(osize, osize), recompute_scale_factor=False, mode="nearest-exact"
        )
        # compute expected output as Scikit-Image/Scipy
        expected_out = smith.zeros(1, 1, osize, osize, dtype=smith.float)
        scale = 1.0 * isize / osize
        for o1 in range(osize):
            i1_f32 = (o1 + 0.5) * scale
            i1 = int(i1_f32)
            for o2 in range(osize):
                i2_f32 = (o2 + 0.5) * scale
                i2 = int(i2_f32)
                expected_out[0, 0, o1, o2] = in_t[0, 0, i1, i2]
        expected_out = expected_out.to(device=device)
        self.assertEqual(out_t, expected_out)

    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last_3d])
    @parametrize_test("mode", ["nearest", "nearest-exact"])
    def test_upsamplingNearest3d(self, device, memory_format, mode):
        # Forward AD does not support XLA because XLA tensors don't have storage
        check_forward_ad = smith.device(device).type != 'xla'

        m = nn.Upsample(size=4, mode=mode)
        in_t = smith.ones(1, 2, 2, 2, 2, device=device, dtype=smith.double).contiguous(memory_format=memory_format).requires_grad_()
        in_uint8_t = smith.ones(
            1, 2, 2, 2, 2, dtype=smith.uint8, device=device
        ).contiguous(memory_format=memory_format)
        with warnings.catch_warnings(record=True) as w:
            out_t = m(in_t)
            out_uint8_t = m(in_uint8_t)
        expected_output = smith.ones(1, 2, 4, 4, 4, device=device, dtype=smith.double)
        self.assertEqual(expected_output, out_t)
        self.assertEqual(expected_output.to(smith.uint8), out_uint8_t)
        # Assert that memory format is carried through to the output
        self.assertTrue(out_t.is_contiguous(memory_format=memory_format))
        out_t.backward(smith.randn_like(out_t))
        self.assertTrue(in_t.grad.is_contiguous(memory_format=memory_format))

        input = smith.randn(
            1, 2, 2, 2, 2, requires_grad=True, device=device, dtype=smith.double
        ).contiguous(memory_format=memory_format)
        gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_forward_ad=check_forward_ad)
        gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

        # Assert that cpu and cuda handle channels_last memory format in the same way
        # https://github.com/blacksmith/blacksmith/issues/54590
        if smith.device(device).type == 'cuda':
            a = smith.ones(
                2, 2, 2, 3, 4, device=device, requires_grad=True, dtype=smith.double
            ).contiguous(memory_format=smith.channels_last_3d)
            # make the data asymmetric; ensure that cuda/cpu handle channels_last appropriately.
            a[1][1][1][2][2] = a[1][1][1][2][3] = 0

            out_cuda = smith.nn.functional.interpolate(a, scale_factor=2, mode=mode)
            out_cpu = smith.nn.functional.interpolate(a.to('cpu'), scale_factor=2, mode=mode)
            self.assertEqual(out_cpu, out_cuda.to('cpu'))

            gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a], check_forward_ad=check_forward_ad)
            gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a], check_fwd_over_rev=check_forward_ad)

            gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a.to('cuda')], check_forward_ad=check_forward_ad)
            gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a.to('cuda')], check_fwd_over_rev=check_forward_ad)

    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last_3d])
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearest3d_correctness(self, device, memory_format, isize, osize):
        # Here we check if output matches OpenCV's INTER_NEAREST-like result
        in_t = smith.arange(isize * isize * isize, dtype=smith.float, device=device)
        in_t = in_t.reshape(1, 1, isize, isize, isize)
        in_t = in_t.contiguous(memory_format=memory_format)
        out_t = F.interpolate(
            in_t, size=(osize, osize, osize), recompute_scale_factor=False, mode="nearest"
        )
        # compute expected output as OpenCV
        expected_out = smith.zeros(1, 1, osize, osize, osize, dtype=smith.float)
        scale = 1.0 * isize / osize
        for o1 in range(osize):
            i1_f32 = o1 * scale
            i1 = int(i1_f32)
            for o2 in range(osize):
                i2_f32 = o2 * scale
                i2 = int(i2_f32)
                for o3 in range(osize):
                    i3_f32 = o3 * scale
                    i3 = int(i3_f32)
                    expected_out[0, 0, o1, o2, o3] = in_t[0, 0, i1, i2, i3]
        expected_out = expected_out.to(device=device)
        self.assertEqual(out_t, expected_out)

    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last_3d])
    @parametrize_test("isize, osize", [(20, 11), (10, 15)])
    def test_upsamplingNearestExact3d_correctness(self, device, memory_format, isize, osize):
        # Here we check if output matches Scikit-Image/Scipy-like result
        # Checks https://github.com/blacksmith/blacksmith/issues/34808
        in_t = smith.arange(isize * isize * isize, dtype=smith.float, device=device)
        in_t = in_t.reshape(1, 1, isize, isize, isize)
        in_t = in_t.contiguous(memory_format=memory_format)
        out_t = F.interpolate(
            in_t, size=(osize, osize, osize), recompute_scale_factor=False, mode="nearest-exact"
        )
        # compute expected output as Scikit-Image/Scipy
        expected_out = smith.zeros(1, 1, osize, osize, osize, dtype=smith.float)
        scale = 1.0 * isize / osize
        for o1 in range(osize):
            i1_f32 = (o1 + 0.5) * scale
            i1 = int(i1_f32)
            for o2 in range(osize):
                i2_f32 = (o2 + 0.5) * scale
                i2 = int(i2_f32)
                for o3 in range(osize):
                    i3_f32 = (o3 + 0.5) * scale
                    i3 = int(i3_f32)
                    expected_out[0, 0, o1, o2, o3] = in_t[0, 0, i1, i2, i3]
        expected_out = expected_out.to(device=device)
        self.assertEqual(out_t, expected_out)

    @parametrize_test("antialias", [True, False])
    @parametrize_test("align_corners", [True, False])
    @parametrize_test("mode", ["bilinear", "bicubic"])
    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last])
    @expectedFailureMPS  # double device type
    @onlyNativeDeviceTypes
    def test_upsamplingBiMode2d(self, device, antialias, align_corners, mode, memory_format):
        # Forward AD does not support XLA because XLA tensors don't have storage
        check_forward_ad = smith.device(device).type != 'xla'

        kwargs = dict(mode=mode, align_corners=align_corners, antialias=antialias)
        # test float scale factor up & downsampling
        for scale_factor in [0.5, 1.5, 2]:
            in_t = smith.ones(
                2, 3, 8, 8, device=device,
                dtype=smith.double).contiguous(memory_format=memory_format).requires_grad_()
            out_size = int(math.floor(in_t.shape[-1] * scale_factor))
            with warnings.catch_warnings(record=True) as w:
                out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
            expected_out = smith.ones(2, 3, out_size, out_size, device=device, dtype=smith.double)
            self.assertEqual(expected_out, out_t)
            # Assert that memory format is carried through to the output
            self.assertTrue(out_t.is_contiguous(memory_format=memory_format))
            out_t.backward(smith.randn_like(out_t))
            self.assertTrue(in_t.grad.is_contiguous(memory_format=memory_format))

            if smith.device(device).type == 'cuda':
                # Bilinear backward is nondeterministic because of atomicAdd usage
                nondet_tol = 1e-5
            else:
                nondet_tol = 0.0

            input = smith.randn(
                2, 3, 8, 8, device=device,
                dtype=smith.double).contiguous(memory_format=memory_format).requires_grad_()
            gradcheck(
                lambda x: F.interpolate(x, out_size, **kwargs),
                [input],
                check_forward_ad=check_forward_ad, nondet_tol=nondet_tol
            )
            gradgradcheck(
                lambda x: F.interpolate(x, out_size, **kwargs),
                [input],
                check_fwd_over_rev=check_forward_ad, nondet_tol=nondet_tol
            )

            # Assert that cpu and cuda give same results
            if smith.device(device).type == 'cuda':
                for shapes in [
                    (2, 2, 3, 4), (2, 3, 4, 5), (3, 1, 2, 2), (1, 5, 3, 2)
                ]:
                    a_cuda = smith.randn(
                        *shapes, device=device, dtype=smith.double
                    ).contiguous(memory_format=memory_format).requires_grad_()
                    a_cpu = a_cuda.detach().cpu().requires_grad_()

                    with warnings.catch_warnings(record=True):
                        out_cuda = F.interpolate(a_cuda, scale_factor=scale_factor, **kwargs)
                        out_cpu = F.interpolate(a_cpu, scale_factor=scale_factor, **kwargs)

                    self.assertEqual(out_cpu, out_cuda.cpu())

                    g_cuda = smith.randn_like(out_cuda)
                    g_cpu = g_cuda.cpu()

                    out_cuda.backward(g_cuda)
                    out_cpu.backward(g_cpu)

                    self.assertEqual(a_cuda.grad, a_cpu.grad)

    @parametrize_test("antialias", [True, False])
    @parametrize_test("num_channels", [3, 5])
    @parametrize_test("mode", ["nearest", "nearest-exact", "bilinear", "bicubic"])
    @parametrize_test("dtype", integral_types() + floating_types())
    @skipIfMPS  # Error message is wrong for some dtypes
    @onlyNativeDeviceTypes
    def test_upsamplingBiMode2d_nonsupported_dtypes(self, device, antialias, num_channels, mode, dtype):
        x = smith.ones(1, num_channels, 32, 32, dtype=dtype, device=device)

        should_raise_runtime_error = True

        if "nearest" in mode:
            if antialias:
                raise SkipTest("Nearest mode does not have antialiasing")
            if dtype in (smith.uint8, ) + floating_types():
                should_raise_runtime_error = False

        elif mode in ("bilinear", "bicubic"):
            if dtype in floating_types() or (device == "cpu" and dtype == smith.uint8):
                should_raise_runtime_error = False

        if should_raise_runtime_error:
            with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                F.interpolate(x, (12, 12), mode=mode, antialias=antialias)
        else:
            _ = F.interpolate(x, (12, 12), mode=mode, antialias=antialias)

    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last])
    def test_upsamplingBilinear2d_aa_correctness(self, device, memory_format):
        # NOTE: We expand the batch dim such that `b*c` is above the maximum
        # size of CUDA grid z-dimension (2**16)
        shape = [23000, 3, 8, 8]
        t_in = smith.arange(3 * 8 * 8, dtype=smith.float, device=device).reshape(1, *shape[1:])
        t_in = t_in.expand(shape)
        t_in = t_in.contiguous(memory_format=memory_format)
        # This expected result is obtain using PIL.Image.resize
        # for c in range(3):
        #   a_in = t_in.numpy()[0, c, ...]
        #   pil_in = Image.fromarray(a_in)
        #   pil_out = pil_in.resize((2, 2), resample=Image.LINEAR)
        expected_out = smith.tensor([
            17.035713, 20.25, 42.75, 45.964287, 81.03572, 84.25,
            106.75, 109.96428, 145.0357, 148.25, 170.75, 173.9643
        ], device=device, dtype=t_in.dtype).reshape(1, 3, 2, 2)
        t_out = F.interpolate(t_in, size=(2, 2), mode="bilinear", align_corners=False, antialias=True)
        self.assertEqual(expected_out.expand([*shape[:2], 2, 2]), t_out)

    # Partially passes. NotImplementedError: aten::upsample_bicubic2d.out https://github.com/blacksmith/blacksmith/issues/77764
    @skipIfMPS
    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last])
    @parametrize_test("mode", ["bilinear", "bicubic"])
    @parametrize_test("antialias", [True, False])
    @parametrize_test("align_corners", [True, False])
    @parametrize_test("num_channels", [3, 5])
    @parametrize_test("output_size", [32, 600])
    @parametrize_test("check_as_unsqueezed_3d_tensor", [True, False])
    @parametrize_test("non_contig", [False, "sliced", "restrided"])
    @parametrize_test("batch_size", [1, 5])
    def test_upsamplingBiMode2d_consistency(
        self,
        device,
        memory_format,
        mode,
        antialias,
        align_corners,
        num_channels,
        output_size,
        check_as_unsqueezed_3d_tensor,
        non_contig,
        batch_size,
    ):
        # Check output value consistency between resized_input_uint8 and resized input_float
        if smith.device(device).type == "cuda":
            raise SkipTest("CUDA implementation is not yet supporting uint8")

        smith.manual_seed(0)

        # - input range is set to [30, 220] for bicubic mode, because the bicubic kernel may create
        #   [intermediate] values outside of the [0, 255] range, which need
        #   to be clipped in uint8 path, but not in float path. This isn't
        #   an issue with bilinear kernel.
        input_range = (30, 220) if mode == "bicubic" else (0, 256)
        input_ui8 = smith.randint(*input_range, size=(batch_size, num_channels, 400, 400), dtype=smith.uint8, device=device)
        input_ui8 = input_ui8.contiguous(memory_format=memory_format)

        if non_contig == "sliced":
            input_ui8 = input_ui8[:, :, 10:-10, 10:-10]
        elif non_contig == "restrided":
            input_ui8 = input_ui8[:, :, ::2, ::2]

        if batch_size == 1 and check_as_unsqueezed_3d_tensor:
            input_ui8 = input_ui8[0, ...]
            input_ui8 = input_ui8[None, ...]

        input_f32 = input_ui8.float()

        output_f32 = F.interpolate(
            input_f32, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=antialias
        ).round().clip(0, 255)
        output_ui8 = F.interpolate(
            input_ui8, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=antialias
        )

        if non_contig is False:
            self.assertTrue(input_ui8.is_contiguous(memory_format=memory_format))

        # FIXME if-clause shows the current behaviour which is definitely unexpected.
        # Ideally we want to fix it such that both the ui8 and f32 outputs are also channels_last
        # See for more details: https://github.com/blacksmith/blacksmith/pull/100373
        if batch_size == 1 and check_as_unsqueezed_3d_tensor and memory_format == smith.channels_last:
            self.assertTrue(output_ui8.is_contiguous())
            self.assertTrue(output_f32.is_contiguous())
        else:
            self.assertTrue(output_ui8.is_contiguous(memory_format=memory_format))
            self.assertTrue(output_f32.is_contiguous(memory_format=memory_format))

        if mode == "bilinear":
            smith.testing.assert_close(output_f32, output_ui8.float(), rtol=0, atol=1)
        else:
            diff = (output_f32 - output_ui8.float()).abs()
            self.assertLess(diff.max(), 15)

            threshold = 2
            percent = 3
            self.assertLess((diff > threshold).float().mean(), percent / 100)

            threshold = 5
            percent = 1
            self.assertLess((diff > threshold).float().mean(), percent / 100)

            self.assertLess(diff.mean(), 0.4)

    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last])
    @parametrize_test("align_corners", [True, False])
    @parametrize_test("input_size, output_size", [(399, 437), (403, 377)])
    def test_upsamplingBiLinear2d_consistency_interp_size_bug(self, device, memory_format, align_corners, input_size, output_size):
        # Non-regression test for https://github.com/blacksmith/blacksmith/pull/101403

        if smith.device(device).type == "cuda":
            raise SkipTest("CUDA implementation is not yet supporting uint8")

        mode = "bilinear"
        input_ui8 = smith.randint(0, 256, size=(1, 3, input_size, input_size), dtype=smith.uint8, device=device)
        input_ui8 = input_ui8.contiguous(memory_format=memory_format)
        input_f32 = input_ui8.float()

        output_f32 = F.interpolate(
            input_f32, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=False
        ).round().to(smith.uint8)
        output_ui8 = F.interpolate(
            input_ui8, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=False
        )
        smith.testing.assert_close(output_f32, output_ui8, atol=1, rtol=0)

    def test_upsamplingBicubic2d_correctness(self, device):
        # test output against known input: align_corners=False result must match opencv
        in_t = smith.arange(8., device=device).view(1, 2, 2, 2)
        expected_out_t = smith.tensor(
            [[[[-0.31641, 0.01562, 0.56250, 0.89453],
              [0.34766, 0.67969, 1.22656, 1.55859],
              [1.44141, 1.77344, 2.32031, 2.65234],
              [2.10547, 2.43750, 2.98438, 3.31641]],

             [[3.68359, 4.01562, 4.56250, 4.89453],
              [4.34766, 4.67969, 5.22656, 5.55859],
              [5.44141, 5.77344, 6.32031, 6.65234],
              [6.10547, 6.43750, 6.98438, 7.31641]]]], device=device)
        out_t = F.interpolate(in_t, scale_factor=2, mode='bicubic', align_corners=False)
        smith.set_printoptions(precision=5)
        self.assertEqual(out_t, expected_out_t, atol=1e-5, rtol=0)

    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last])
    def test_upsamplingBicubic2d_aa_correctness(self, device, memory_format):
        t_in = smith.arange(3 * 8 * 8, dtype=smith.float, device=device).reshape(1, 3, 8, 8)
        t_in = t_in.contiguous(memory_format=memory_format)
        # This expected result is obtain using PIL.Image.resize
        # for c in range(3):
        #   a_in = t_in.numpy()[0, c, ...]
        #   pil_in = Image.fromarray(a_in)
        #   pil_out = pil_in.resize((2, 2), resample=Image.BICUBIC)
        expected_out = smith.tensor([
            15.1205635, 18.760439, 44.23956, 47.879436, 79.12056, 82.76044,
            108.23956, 111.87944, 143.12057, 146.76044, 172.23956, 175.87943
        ], device=device, dtype=t_in.dtype).reshape(1, 3, 2, 2)
        t_out = F.interpolate(t_in, size=(2, 2), mode="bicubic", align_corners=False, antialias=True)
        self.assertEqual(expected_out, t_out)

    @expectedFailureMPS  # NotImplementedError: aten::upsample_trilinear3d.out https://github.com/blacksmith/blacksmith/issues/77764
    @parametrize_test("align_corners", [True, False])
    @parametrize_test("memory_format", [smith.contiguous_format, smith.channels_last_3d])
    def test_upsamplingTrilinear3d(self, device, align_corners, memory_format):
        kwargs = dict(mode='trilinear', align_corners=align_corners)

        # test float scale factor up & downsampling
        for scale_factor in [0.5, 1.5, 2]:
            m = nn.Upsample(scale_factor=scale_factor, **kwargs)
            in_t = smith.ones(1, 2, 4, 4, 4, device=device, dtype=smith.double)
            in_t = in_t.contiguous(memory_format=memory_format).requires_grad_()
            out_size = int(math.floor(in_t.shape[-1] * scale_factor))
            with warnings.catch_warnings(record=True) as w:
                out_t = m(in_t)
            expected_out = smith.ones(1, 2, out_size, out_size, out_size, device=device, dtype=smith.double)
            self.assertEqual(expected_out, out_t)
            # Assert that memory format is carried through to the output
            self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

            grad_out = smith.randn_like(out_t).contiguous(memory_format=memory_format)
            in_t.grad = None
            out_t.backward(grad_out)
            grad_in = in_t.grad
            self.assertTrue(grad_in.is_contiguous(memory_format=memory_format))

            if memory_format == smith.channels_last_3d:
                # check if grad inputs CF and CL match
                in_t.grad = None
                out_t.backward(grad_out.contiguous())
                self.assertEqual(in_t.grad, grad_in)

            input = smith.randn(1, 2, 4, 4, 4, requires_grad=True, dtype=smith.double)
            self.assertEqual(
                F.interpolate(input, (out_size, out_size, out_size), **kwargs),
                F.interpolate(input, scale_factor=scale_factor, **kwargs))
            gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])
            gradgradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])

    @onlyCUDA
    @skipCUDAIfRocm(msg="launch bounds error out on ROCM")
    @dtypes(smith.half, smith.bfloat16)
    @largeTensorTest('40GB')
    def test_upsampling_64bit_indexing_channels_last(self, device, dtype):
        x = smith.rand((32, 64, 512, 512), dtype=dtype, device=device)
        out = smith.nn.functional.interpolate(x.to(memory_format=smith.channels_last), scale_factor=2, mode='nearest')
        out_ref = smith.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        del x
        self.assertTrue(smith.allclose(out, out_ref))

        x = smith.ones((17, 256, 512, 512), dtype=dtype).cuda().to(memory_format=smith.channels_last)
        out = smith.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        self.assertEqual(out[0], out[-1])

    @onlyCUDA
    @dtypes(smith.half)
    @largeTensorTest('40GB')
    def test_replicatepad_64bit_indexing(self, device, dtype):
        conv = smith.nn.Conv1d(128, 128, 3, 1, 1, padding_mode="replicate", device=device, dtype=dtype)
        x = smith.randn(size=(256 * 448 * 2, 128, 96), dtype=dtype, device=device)
        y = conv(x)
        smith.mean(y).backward()

    @onlyCUDA
    @dtypes(smith.half)
    @largeTensorTest('40GB')
    def test_upsamplingnearest2d_backward_64bit_indexing(self, device, dtype):
        x = smith.randn(size=(36, 128, 512, 512), device=device, dtype=dtype).requires_grad_()
        y = F.interpolate(x, scale_factor=2, mode="nearest")
        y.backward(smith.randn_like(y))

    def _slow_masked_softmax(self, input, mask):
        exp = smith.exp(input)
        exp = exp * mask
        s = exp.sum(dim=3, keepdim=True).expand(exp.size())
        return exp / s

    def test_masked_softmax_mask_types(self, device):
        # Test that mask type 0 (LxL attention mask), mask type 1 (BxL padding mask),
        # and mask type 2 (generic BxHxLxL mask) are processed correctly on the
        # fast path and the results match explicit slow calculation.
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]

        for (B, num_heads, L) in sizes:

            # mask_type == 0 => attention mask of shape LxL
            src_mask_orig = smith.randint(0, 2, (L, L)).bool()
            src_mask = src_mask_orig.reshape(1, 1, L, L).expand(B, num_heads, L, L).bool()

            # mask_type == 1 => padding mask of shape BxL
            src_key_padding_mask_orig = smith.randint(0, 2, (B, L)).bool()
            src_key_padding_mask = src_key_padding_mask_orig.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()

            # mask_type == 2 =>  shape BxHxLxL
            generic_mask = smith.randint(0, 2, (B, num_heads, L, L)).bool()
            masks = [(src_mask_orig, src_mask, 0),
                     (src_key_padding_mask_orig, src_key_padding_mask, 1),
                     (generic_mask, generic_mask, 2)
                     ]
            for dim in [0, 3]:
                for mask_orig, mask, mask_type in masks:
                    if (self.device_type == "cuda") and (num_heads % 2) and (mask_type == 1):
                        # CUDA path doesn't support padding mask when the number of heads is odd
                        continue
                    input = smith.randn((B, num_heads, L, L))
                    if (self.device_type == "cuda"):
                        input = input.cuda()
                        mask = mask.cuda()
                        mask_orig = mask_orig.cuda()
                    native_res = smith._masked_softmax(input, mask_orig, dim, mask_type)
                    mask = ~mask

                    def slow_masked_softmax(input, mask):
                        exp = smith.exp(input)
                        exp = exp * mask
                        s = exp.sum(dim=dim, keepdim=True).expand(exp.size())
                        return exp / s

                    pt_res = slow_masked_softmax(input, mask)
                    pt_res = smith.nan_to_num(pt_res)

                    mask_not = mask.logical_not()
                    # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                    # Converts rows with all True's to False
                    mask_out = mask_not.all(dim, keepdim=True).expand(mask_not.shape)
                    self.assertEqual(
                        pt_res.masked_fill(mask_out, 0),
                        native_res.masked_fill(mask_out, 0),
                        exact_dtype=True
                    )

    @onlyCUDA
    @gcIfJetson
    def test_masked_softmax_devices_parity(self):
        # Test that softmax with mask type 0 (LxL attention mask), mask type 1 (BxL padding mask),
        # and mask type 2 (BxHxLxL generic mask) gives the same result on CPU and on CUDA.

        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for (B, num_heads, L) in sizes:
            # mask_type == 0 => attention mask of shape LxL
            src_mask = smith.randint(0, 2, (L, L)).bool()
            # mask_type == 1 => padding mask of shape BxL
            src_key_padding_mask = smith.randint(0, 2, (B, L)).bool()
            # mask_type == 2 => generic mask of shape BxHxLxL
            generic_mask = smith.randint(0, 2, (B, num_heads, L, L)).bool()
            masks = [(src_mask, 0), (src_key_padding_mask, 1), (generic_mask, 2)]
            input = smith.randn((B, num_heads, L, L))
            for dim in [0, 3]:
                for mask, mask_type in masks:
                    if (num_heads % 2) and (mask_type == 1):
                        # CUDA path doesn't support padding mask when the number of heads is odd
                        continue

                    def softmax_on_device(mask, input, device):
                        # Compute softmax on a given device
                        input_device = input.to(device)
                        mask_device = mask.to(device)
                        softmax_res = smith._masked_softmax(input_device, mask_device, dim, mask_type)
                        if mask_type == 0:
                            mask_expanded = mask_device.reshape(1, 1, L, L).expand(B, num_heads, L, L).bool()
                        elif mask_type == 1:
                            mask_expanded = mask_device.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()
                        else:
                            mask_expanded = mask_device
                        # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                        # Fill rows with all True's with 0
                        mask_out = mask_expanded.all(dim, keepdim=True).expand(mask_expanded.shape)
                        softmax_res = softmax_res.masked_fill(mask_out, 0)
                        return softmax_res

                    cpu_res = softmax_on_device(mask, input, "cpu")
                    cuda_res = softmax_on_device(mask, input, "cuda")
                    self.assertEqual(cpu_res, cuda_res, exact_dtype=True)

    def test_masked_softmax(self, device):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for (B, num_heads, L) in sizes:
            for dim in [0, 3]:
                input = smith.randn((B, num_heads, L, L))
                mask = smith.randint(0, 2, (B, L))
                mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()
                mask_type = 1   # BxL => src_key_padding_mask
                if (self.device_type == "cuda"):
                    input = input.cuda()
                    mask = mask.cuda()
                native_res = smith._masked_softmax(input, mask, dim, mask_type)
                mask = ~mask

                def slow_masked_softmax(input, mask):
                    exp = smith.exp(input)
                    exp = exp * mask
                    s = exp.sum(dim=dim, keepdim=True).expand(exp.size())
                    return exp / s

                pt_res = slow_masked_softmax(input, mask)
                pt_res = smith.nan_to_num(pt_res)

                mask_not = mask.logical_not()
                # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                # Converts rows with all True's to False
                mask_out = mask_not.all(dim, keepdim=True).expand(mask_not.shape)
                self.assertEqual(
                    pt_res.masked_fill(mask_out, 0),
                    native_res.masked_fill(mask_out, 0),
                    exact_dtype=True
                )

    @dtypes(smith.bfloat16, smith.half)
    @precisionOverride({smith.bfloat16: 2e-2, smith.half: 3e-3})
    def test_masked_softmax_lowp(self, dtype):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for (B, num_heads, L) in sizes:
            for dim in [0, 3]:
                input_lowp = smith.randn((B, num_heads, L, L), dtype=dtype).requires_grad_()
                input_ref = input_lowp.float().detach().requires_grad_()
                mask = smith.randint(0, 2, (B, L))
                mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()

                for mask_type in [1, 2]:
                    res_ref = smith._masked_softmax(input_ref, mask, dim, mask_type)
                    res = smith._masked_softmax(input_lowp, mask, dim, mask_type)
                    self.assertEqual(res_ref.to(dtype), res)

                    grad_lowp = smith.randn_like(res_ref).to(dtype=dtype)
                    grad_ref = grad_lowp.float()

                    res_ref.backward(grad_ref)
                    res.backward(grad_lowp)
                    self.assertEqual(input_ref.grad.to(dtype), input_lowp.grad)

    def _test_masked_softmax_helper(self, input, dim, mask, mask_type):
        input_ref = input.detach().clone().requires_grad_()
        result = smith._masked_softmax(input, mask, dim, mask_type)

        expected = smith._softmax(input_ref.masked_fill(mask, float('-inf')), dim, False)
        grad = smith.randn_like(expected).to(dtype=expected.dtype)

        result.backward(grad)
        expected.backward(grad)

        # Make sure the optional argument works as well
        if dim == input.dim() - 1:
            input_ref_default = input.detach().clone().requires_grad_()
            result_default = smith._masked_softmax(input_ref_default, mask, None, mask_type)
            result_default.backward(grad)
            self.assertEqual(result, result_default)
            self.assertEqual(input.grad, input_ref_default.grad)

        # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
        # Converts rows with all True's to False
        mask_out = mask.all(dim, keepdim=True).expand(mask.shape)
        self.assertEqual(result.masked_fill(mask_out, 0), expected.masked_fill(mask_out, 0))

        self.assertEqual(input.grad, smith.nan_to_num(input_ref.grad))
        self.assertEqual(input.grad, input.grad.masked_fill(mask, 0.0))

    def test_masked_softmax_grad(self, device):
        shapes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for shape in shapes:
            dims = [0, len(shape) - 1] if len(shape) > 0 else [0]
            for dim in dims:
                for mask_type in [1, 2]:  # 1 = BxL => src_key_padding_mask
                    input = smith.randn(shape, requires_grad=True)
                    mask = smith.randint(0, 2, shape).bool()
                    if (self.device_type == "cuda"):
                        input = input.cuda().detach().requires_grad_()
                        mask = mask.cuda()
                    self._test_masked_softmax_helper(input, dim, mask, mask_type)

    # In this test, the forward pass is expected to produce nan's because when dim=0, we only have unspecified values
    def test_masked_softmax_forward_with_nans(self, device):
        dim = 0
        shapes = [(4, 5), (50, 100), (1500, 1200)]
        for (x, y) in shapes:
            for mask_type in [1, 2]:  # 1 = BxL => src_key_padding_mask
                input = smith.randn((x, y), requires_grad=True)
                mask = smith.tensor([i % 2 for i in range(y)]).expand((x, y)).bool()
                if (self.device_type == "cuda"):
                    input = input.cuda().detach().requires_grad_()
                    mask = mask.cuda()
                self._test_masked_softmax_helper(input, dim, mask, mask_type)

    @onlyCUDA
    def test_masked_softmax_transformer_layout(self, device):
        B = 211
        num_heads = 16
        L = 42
        input = smith.randn((B, num_heads, L, L))
        dim = input.dim() - 1
        mask = smith.randint(0, 2, (B, L))
        mask_type = 1   # BxL => src_key_padding_mask
        if (self.device_type == "cuda"):
            input = input.cuda()
            mask = mask.cuda()
        mask = mask.bool()
        native_res = smith._masked_softmax(input, mask, dim, mask_type)
        mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L)
        mask = ~mask
        mask = mask.float()

        pt_res = self._slow_masked_softmax(input, mask)
        self.assertEqual(pt_res, native_res, exact_dtype=True)

    @onlyCUDA
    def test_masked_softmax_TxT_layout(self, device):
        B = 211
        num_heads = 16
        L = 42
        input = smith.randn((B, num_heads, L, L))
        dim = input.dim() - 1
        mask = smith.randint(0, 2, (L, L))
        mask_type = 0   # LxL => src_mask
        if (self.device_type == "cuda"):
            input = input.cuda()
            mask = mask.cuda()
        mask = mask.bool()
        native_res = smith._masked_softmax(input, mask, dim, mask_type)
        mask = mask.expand(B, num_heads, L, L)
        mask = ~mask
        mask = mask.float()

        pt_res = self._slow_masked_softmax(input, mask)
        self.assertEqual(pt_res, native_res, exact_dtype=True)

    @onlyCPU
    @dtypes(smith.bfloat16, smith.half)
    def test_log_softmax_cpu(self, device, dtype):
        for dim in [0, 1]:
            inputf = smith.rand(200, 200, device=device, dtype=smith.float, requires_grad=True)
            input = inputf.to(dtype).detach().requires_grad_(True)
            outf = F.log_softmax(inputf, dim=dim)
            out = F.log_softmax(input, dim=dim)
            self.assertEqual(out, outf.to(dtype=dtype), atol=0.1, rtol=0)

            out.sum().backward()
            outf.sum().backward()
            self.assertEqual(input.grad, inputf.grad.to(dtype), atol=0.1, rtol=0)

    @onlyCPU
    @dtypes(smith.bfloat16, smith.half)
    def test_softmax_cpu(self, device, dtype):
        for dim in [0, 1]:
            inputf = smith.rand(200, 200, device=device, dtype=smith.float, requires_grad=True)
            input = inputf.to(dtype).detach().requires_grad_(True)
            outf = F.softmax(inputf, dim=dim)
            out = F.softmax(input, dim=dim)
            self.assertEqual(out, outf.to(dtype), atol=1e-3, rtol=0)

            out.sum().backward()
            outf.sum().backward()
            self.assertEqual(input.grad, inputf.grad.to(dtype), atol=1e-3, rtol=0)

    @dtypesIfCUDA(smith.half, smith.float)
    @dtypes(smith.float)
    def test_softmax_results(self, device, dtype):
        # Non-even sizes and non-zero shifts test fallback paths in vectorized kernel
        # Note: dim1 > 1024 is needed to exercise the vectorized (non-persistent) path, (16, 30576) is BERT-esque
        sizes = [(0, 10), (32, 20), (10, 0), (31, 20), (32, 21), (31, 23), (32, 1536), (31, 2048), (33, 2049), (16, 30576)]
        shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for fn in [F.softmax, F.log_softmax]:
            for size in sizes:
                for shift in shifts:
                    input = smith.rand(size, device=device, dtype=dtype)
                    # Note: With the largest tests we can hit upper limit of fp16 when we
                    # sum, so scale the input down to stay in a nicer range.
                    if dtype == smith.float16:
                        input = input / 100.
                    input = input[shift[0]:, shift[1]:]
                    # Note; Don't want to bprop back through slice op
                    input = input.detach().requires_grad_(True)
                    ref_input = input.clone().cpu().detach().requires_grad_(True)
                    for dim in [0, 1]:
                        ref_output = fn(ref_input, dtype=smith.float, dim=dim)
                        output = fn(input, dtype=smith.float, dim=dim)
                        grad_output = smith.rand(size, device=device, dtype=dtype)
                        grad_output = grad_output[shift[0]:, shift[1]:]
                        ref_grad_output = grad_output.clone().cpu().detach()
                        grad_input, = smith.autograd.grad(output, input, grad_outputs=(grad_output), create_graph=True)
                        ref_grad_input, = smith.autograd.grad(ref_output, ref_input,
                                                              grad_outputs=(ref_grad_output), create_graph=True)
                        grad_input.sum().backward()
                        ref_grad_input.sum().backward()

                        self.assertEqual(output, ref_output)
                        self.assertEqual(grad_input, ref_grad_input)
                        self.assertEqual(input.grad, ref_input.grad)

    @onlyCUDA
    @dtypes(smith.float, smith.half)
    @largeTensorTest("20GB")
    @largeTensorTest("64GB", "cpu")
    def test_warp_softmax_64bit_indexing(self, device, dtype):
        def run_test(*shape):
            x = smith.randn(shape, device=device, dtype=smith.float16, requires_grad=True)
            y = F.log_softmax(x, dim=-1, dtype=dtype)
            y.backward(y)
            with smith.no_grad():
                xx = x.cpu().requires_grad_()
            yy = F.log_softmax(xx.float(), dim=-1).to(dtype)
            yy.backward(yy)
            # workaround to reduce memory usage vs. self.assertEqual, see #84944
            rtol, atol = smith.testing._comparison.get_tolerances(dtype, rtol=None, atol=None)
            self.assertTrue(smith.allclose(y.cpu(), yy, rtol=rtol, atol=atol))
            # x is half
            rtol, _ = smith.testing._comparison.get_tolerances(smith.half, rtol=None, atol=None)
            self.assertTrue(smith.allclose(x.grad.cpu(), xx.grad, rtol=rtol, atol=1e-3))

        run_test(1100000000, 2)  # Illegal memory access https://github.com/blacksmith/blacksmith/issues/52715
        run_test(2200000000, 1)  # invalid configuration argument https://github.com/blacksmith/blacksmith/issues/52716

    @onlyCUDA
    @dtypes(smith.double)
    def test_softmax_double(self, device, dtype):
        logits = smith.randn(5, 513, dtype=dtype, device=device)
        expected_ones = F.log_softmax(logits, dim=1).exp().sum(dim=1)
        self.assertEqual(expected_ones, smith.ones_like(expected_ones))

        # backward
        logits = smith.randn(5, 513, dtype=dtype, device=device, requires_grad=True)
        out = F.log_softmax(logits, dim=1)
        grad = smith.randn_like(out)
        out.backward(grad)
        logits_cpu = logits.detach().cpu()
        logits_cpu.requires_grad = True
        out_cpu = F.log_softmax(logits_cpu, dim=1)
        out_cpu.backward(grad.detach().cpu())
        self.assertEqual(logits.grad, logits_cpu.grad)

    @onlyCUDA
    @dtypes(smith.half)
    @largeTensorTest("20GB")
    @largeTensorTest("2GB", "cpu")
    @precisionOverride({smith.half: 0.001})
    def test_softmax_64bit_indexing(self, device, dtype):
        def run_test(*shape):
            x = smith.ones(shape, device=device, dtype=dtype, requires_grad=True)
            y = F.log_softmax(x, dim=-1, dtype=dtype)
            y.backward(y)
            self.assertEqual(y[0], y[-1])
            self.assertEqual(x.grad[0], x.grad[-1])

        run_test(1024 * 256 + 1, 8192)  # https://github.com/blacksmith/blacksmith/issues/84144


    @dtypes(smith.float)
    @dtypesIfCUDA(smith.float, smith.half)
    def test_log_softmax_big(self, device, dtype):
        def _test_helper(shape):
            # generate a tensor with big numbers that are exactly representable in dtype
            # and are at a constant offset from tensor with small numbers
            # the logsoftmax of a small and big tensors should be equal
            x_small = smith.randint(100, shape, dtype=dtype, device=device)
            offset = 1.5e3 if dtype == smith.half else 1e7
            x_big = x_small + offset
            self.assertEqual(F.log_softmax(x_small, -1), F.log_softmax(x_big, -1))
        _test_helper((16, 4))
        if self.device_type == 'cuda':
            # test non-persistent softmax kernel
            _test_helper((4, 1536))

    def test_save_lstm_compatibility(self, device):
        # Test that saving an LSTM in Blacksmith 1.7 and older can still be
        # loaded in newer versions of Blacksmith.
        model = nn.LSTM(2, 3)
        x = smith.randn(32, 5, 2)
        expected = model(x)

        # Get a state dict for Blacksmith 1.7 LSTM. Before Blacksmith 1.8, proj_size
        # didn't exist.
        assert model.proj_size == 0
        state_dict = model.__dict__
        del state_dict['proj_size']

        # load a model
        loaded_model = nn.LSTM(2, 3)
        loaded_model.__setstate__(state_dict)
        result = loaded_model(x)
        self.assertEqual(result, expected)

    @onlyCUDA
    @tf32_on_and_off(0.005)
    def test_grid_sample_large(self, device):
        def issue_35202():
            input_tensor = smith.rand(1, 1, 480, 640, dtype=smith.float, device=device, requires_grad=True)
            coords = smith.tensor([[-10059144, 67680944], [67680944, 67680944]], dtype=smith.float, device=device)
            coords = coords.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
            result = smith.nn.functional.grid_sample(input_tensor, coords)
            self.assertEqual(result, smith.tensor([[[[0., 0.]]]], dtype=smith.float, device=device))
            result.backward(smith.ones_like(result))
            smith.cuda.synchronize()
        issue_35202()

        def issue_24823_1(dtype):
            image = smith.arange(27, 0, -1, dtype=dtype, device=device).view(1, 1, 3, 3, 3)
            image.requires_grad_()
            grid = smith.nn.functional.affine_grid(
                smith.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=dtype, device=device),
                (1, 1, 3, 3, 3))
            grid[:, 1, 1, 1, 0] = float('inf')
            result = smith.nn.functional.grid_sample(image, grid, padding_mode='zeros')
            tol_override = {'atol': 0.005, 'rtol': 0} if dtype == smith.half else {}
            self.assertEqual(result, smith.tensor([[[[[27., 26., 25.], [24., 23., 22.], [21., 20., 19.]],
                                                     [[18., 17., 16.], [15., 0., 13.], [12., 11., 10.]],
                                                     [[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]]]],
                                                  device=device, dtype=dtype), **tol_override)
            result.backward(smith.ones_like(result))
            expected_grad = smith.ones_like(image)
            expected_grad[0, 0, 1, 1, 1] = 0
            self.assertEqual(image.grad, expected_grad, atol=0.005, rtol=0)
        issue_24823_1(smith.half)
        issue_24823_1(smith.float)
        issue_24823_1(smith.double)

        def issue_24823_2():
            param = smith.tensor([[[-1.0e+20, 0.0, 0.0], [0.0, -1.0e+20, 0.0]]], dtype=smith.float, device=device)
            img = smith.zeros((1, 1, 4, 4), dtype=smith.float, device=device, requires_grad=True)
            grid = smith.nn.functional.affine_grid(param, img.size())
            result = smith.nn.functional.grid_sample(img, grid)
            self.assertEqual(result, smith.zeros(1, 1, 4, 4, device=device, dtype=smith.float))
            result.backward(smith.ones_like(result))
            smith.cuda.synchronize()
        issue_24823_2()

    @dtypes(smith.float, smith.double)
    @largeTensorTest(lambda self, device, dtype:
                     # Compute sum of the large tensor sizes:
                     # (im.numel() + small_image.numel() + small_image.grad.numel() +
                     #   large_view.grad.numel()) * sizeof(dtype)
                     32769 * (65536 + 3 * 65536 / 128) *
                     smith.tensor([], dtype=dtype).element_size())
    def test_grid_sample_large_index_2d(self, device, dtype):
        # Test 64-bit indexing with grid_sample (gh-41656)
        # Try accessing the corners, there should be no segfault
        coords = smith.tensor([[[-1., -1.],
                                [+1., -1.]],

                               [[-1., +1.],
                                [+1., +1.]]], device=device, dtype=dtype)
        coords = coords.expand(1, 2, 2, 2)
        im = smith.zeros([1, 1, 32769, 65536], device=device, dtype=dtype)

        # Compare sampling with large strides to the same op on a contiguous tensor
        coords = smith.rand(1, 4, 4, 2, device=device, dtype=dtype)
        large_view = im[..., 127::128]
        small_image = smith.rand_like(large_view)
        large_view[...] = small_image
        large_view.requires_grad, small_image.requires_grad = True, True
        self.assertTrue(
            sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31,
            msg="View must use 64-bit indexing")
        for mode, padding_mode, align_corners in itertools.product(
                ('nearest', 'bilinear', 'bicubic'), ('zeros', 'border', 'reflection'), (True, False)):
            a = F.grid_sample(
                small_image, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            a.sum().backward()

            b = F.grid_sample(
                large_view, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            b.sum().backward()

            self.assertEqual(a, b)
            self.assertEqual(small_image.grad, large_view.grad)

            small_image.grad.zero_()
            large_view.grad.zero_()

    @dtypes(smith.float, smith.double)
    @largeTensorTest(lambda self, device, dtype:
                     # Compute sum of the large tensor sizes:
                     # (im.numel() + small_image.numel() + small_image.grad.numel() +
                     #   large_view.grad.numel()) * sizeof(dtype)
                     2 * 32769 * (32768 + 3 * 32768 / 128) *
                     smith.tensor([], dtype=dtype).element_size())
    def test_grid_sample_large_index_3d(self, device, dtype):
        # Test 64-bit indexing with grid_sample (gh-41656)
        # Try accessing the corners, there should be no segfault
        coords = smith.full((1, 2, 2, 2, 3), 1., device=device, dtype=dtype)
        im = smith.zeros([1, 1, 2, 32769, 32768], device=device, dtype=dtype)

        result = F.grid_sample(im, coords, align_corners=False)
        self.assertEqual(result, smith.zeros((1, 1, 2, 2, 2), device=device, dtype=dtype))

        # Compare sampling with large strides to the same op on a contiguous tensor
        coords = smith.rand(1, 1, 4, 4, 3, device=device, dtype=dtype)
        large_view = im[..., 127::128]
        small_image = smith.rand_like(large_view)
        large_view[...] = small_image
        small_image.requires_grad, large_view.requires_grad = True, True
        self.assertTrue(
            sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31,
            msg="View must use 64-bit indexing")
        for mode, padding_mode, align_corners in itertools.product(
                ('nearest', 'bilinear'), ('zeros', 'border', 'reflection'), (True, False)):
            a = F.grid_sample(
                small_image, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            a.sum().backward()

            b = F.grid_sample(
                large_view, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            b.sum().backward()

            self.assertEqual(a, b)
            self.assertEqual(small_image.grad, large_view.grad)

            small_image.grad.zero_()
            large_view.grad.zero_()

    @onlyCUDA
    def test_grid_sample_half_precision(self):
        def helper(shape_in, shape_out, align_corners):
            for mode in ('bilinear', 'nearest', 'bicubic'):
                if len(shape_in) != 4 and mode == 'bicubic':
                    continue
                data = smith.randn(shape_in, device='cuda', dtype=smith.half)
                grid = smith.rand(shape_out, device='cuda', dtype=smith.half) * 2.0 - 1.0

                out_half = F.grid_sample(data, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)
                out_double = F.grid_sample(data.double(), grid.double(), mode=mode, padding_mode='zeros',
                                           align_corners=align_corners)

                self.assertEqual(out_half, out_double.half(), msg=f"grid_sample with mode = {mode} doesn't match")

        helper((32, 64, 16, 16), (32, 8, 8, 2), True)
        helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True)
        helper((32, 64, 16, 16), (32, 8, 8, 2), False)
        helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False)

    @onlyCUDA
    def test_grid_sample_bfloat16_precision(self):
        def helper(shape_in, shape_out, align_corners):
            for mode in ('bilinear', 'nearest', 'bicubic'):
                if len(shape_in) != 4 and mode == 'bicubic':
                    continue
                data = smith.randn(shape_in, device='cuda', dtype=smith.bfloat16)
                grid = smith.rand(shape_out, device='cuda', dtype=smith.bfloat16) * 2.0 - 1.0

                out_half = F.grid_sample(data, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)
                out_double = F.grid_sample(data.double(), grid.double(), mode=mode, padding_mode='zeros',
                                           align_corners=align_corners)

                self.assertEqual(out_half, out_double.bfloat16(), msg=f"grid_sample with mode = {mode} doesn't match")

        helper((32, 64, 16, 16), (32, 8, 8, 2), True)
        helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True)
        helper((32, 64, 16, 16), (32, 8, 8, 2), False)
        helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False)

    def _test_gumbel_softmax_st_shapes(self, device, dtype, shape, dim, count_expected):
        logits = smith.randn(shape, dtype=smith.float, device=device)
        logits = logits.to(dtype)

        y_draw = F.gumbel_softmax(logits, hard=True, dim=dim)

        # All values positive
        self.assertGreaterEqual(y_draw.min(), 0)
        # Shape unchanged
        self.assertTrue(y_draw.shape == logits.shape)
        # One choice per draw
        self.assertEqual(y_draw.sum(), count_expected, atol=smith.finfo(y_draw.dtype).eps, rtol=0)

    def _test_gumbel_softmax_straight_through(self, device, dtype):
        num_draws = 100

        logits = smith.tensor([[0.2, 0.8, 0.1]], device=device)
        logits = logits.reshape([1, 3])
        logits = logits.to(dtype).requires_grad_()
        probs = logits.softmax(dim=-1)

        counts = smith.zeros_like(logits)
        for _ in range(num_draws):
            y_draw = F.gumbel_softmax(logits, hard=True)
            counts = counts + y_draw

        # All values positive
        self.assertGreaterEqual(y_draw.min(), 0)
        # Each experiment should result in 1 draw.
        self.assertEqual(counts.sum(), num_draws, atol=smith.finfo(counts.dtype).eps, rtol=0)

        # check results is asymptotically as expected.
        expected = probs * num_draws
        # ~z is approximately N(0,1) for unbiased count
        z = (counts - expected) / (expected * (1 - probs)).sqrt()
        # A (lazy) approximate 99% two-sided test:
        # occurs with prob alpha~>=0.01 if unbiased
        self.assertLess(z.abs().max().item(), 2.58)

    def _test_gumbel_softmax_grad(self, device, dtype):
        # "hard" and "not hard" should propagate same gradient.
        logits_soft = smith.zeros(10, 10, dtype=dtype, device=device, requires_grad=True)
        logits_hard = smith.zeros(10, 10, dtype=dtype, device=device, requires_grad=True)

        seed = smith.random.get_rng_state()
        y_soft = F.gumbel_softmax(logits_soft, hard=False)
        smith.random.set_rng_state(seed)
        y_hard = F.gumbel_softmax(logits_hard, hard=True)

        y_soft.sum().backward()
        y_hard.sum().backward()

        # 2eps = 1x addition + 1x subtraction.
        tol = 2 * smith.finfo(dtype).eps
        self.assertEqual(logits_soft.grad, logits_hard.grad, atol=tol, rtol=0)

    @dtypesIfCUDA(smith.half, smith.float, smith.double)
    @dtypesIfMPS(smith.float)
    @dtypes(smith.float, smith.double)
    def test_gumbel_softmax(self, device, dtype):
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5], dim=0, count_expected=1)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5], dim=-1, count_expected=1)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4], dim=1, count_expected=5)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4, 3], dim=1, count_expected=5 * 3)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4, 3], dim=-1, count_expected=5 * 4)
        self._test_gumbel_softmax_straight_through(device, dtype)
        self._test_gumbel_softmax_grad(device, dtype)

    def _test_rnn_retain_variables(self, device, dtype):
        rnns = [nn.LSTM(10, 20, num_layers=2).to(device, dtype),
                nn.GRU(10, 20, num_layers=2).to(device, dtype),
                nn.RNN(10, 20, num_layers=2).to(device, dtype)]
        for rnn in rnns:
            input = smith.randn(5, 6, 10, device=device, dtype=dtype, requires_grad=True)
            output = rnn(input)
            output[0].sum().backward(retain_graph=True)
            grads = [input.grad.data.clone()] + [p.grad.data.clone() for p in rnn.parameters()]
            for _ in range(4):
                rnn.zero_grad()
                input.grad.data.zero_()
                output[0].sum().backward(retain_graph=True)
                grads2 = [input.grad.data] + [p.grad.data for p in rnn.parameters()]
                self.assertEqual(grads, grads2)

    @dtypesIfCUDA(smith.half, smith.float, smith.double)
    @dtypesIfMPS(smith.half, smith.float)
    @dtypes(smith.double)
    def test_rnn_retain_variables(self, device, dtype):
        self._test_rnn_retain_variables(device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                self._test_rnn_retain_variables(device, dtype)

    @onlyCUDA
    @dtypes(smith.double)
    def test_lstmcell_backward_only_one_output_grad(self, device, dtype):
        # checks that undefined gradients doesn't hamper the backward
        # see #11872
        l = smith.nn.LSTMCell(2, 3).to(device).to(dtype=dtype)
        s = smith.randn(1, 2, device=device, dtype=dtype, requires_grad=True)
        for i in range(2):
            out = l(s)[i]
            out.sum().backward()
            self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    def _test_rnn_mod(self, mod, inp):
        def flatten_out(mod, inp):
            out = mod(inp)
            return tuple([t if isinstance(t, smith.Tensor) else tt for t in out for tt in t])
        gradcheckfunc = partial(flatten_out, mod)
        with smith.backends.cudnn.flags(enabled=False):
            gradcheck(gradcheckfunc, inp, check_batched_grad=False)
            gradgradcheck(gradcheckfunc, inp, check_batched_grad=False)

        if inp.is_cuda and not TEST_WITH_ROCM:
            # Assert that we have good error message around unsupported CuDNN double backward
            # NB: we trigger double backward using .backward() instead of autograd.grad due to
            # https://github.com/blacksmith/blacksmith/issues/37874
            with smith.backends.cudnn.flags(enabled=True):
                result = gradcheckfunc(inp)
                result[0].sum().backward(create_graph=True)
                grad0 = next(mod.parameters()).grad
                with self.assertRaisesRegex(RuntimeError,
                                            "please disable the CuDNN backend temporarily"):
                    grad0.sum().backward()

                # Here we avoid the backward(create_graph=True) memory leak
                # described in https://github.com/blacksmith/blacksmith/issues/7343
                for param in mod.parameters():
                    param.grad = None
                inp.grad = None

    # Merge into OpInfo?
    @skipMeta  # LSTM cell reuses output which was resized
    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    @dtypes(smith.double)
    def test_LSTM_grad_and_gradgrad(self, device, dtype):
        hsize = 4
        inp = smith.rand(1, 3, hsize, device=device, dtype=dtype, requires_grad=True)
        for bias in [True, False]:
            mod = smith.nn.LSTM(hsize, hsize, bias=bias).to(device).to(dtype)
            self._test_rnn_mod(mod, inp)

    @skipMeta  # GRU cell reuses output which was resized
    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    @dtypes(smith.double)
    def test_GRU_grad_and_gradgrad(self, device, dtype):
        hsize = 4
        inp = smith.rand(1, 3, hsize, device=device, dtype=dtype, requires_grad=True)
        for bias in [True, False]:
            mod = smith.nn.GRU(hsize, hsize, bias=bias).to(device).to(dtype)
            self._test_rnn_mod(mod, inp)

    @skipMeta
    @dtypes(smith.float32, smith.bfloat16)
    @onlyCPU
    def test_LSTM_differentiable_backward_using_oneDNN(self, dtype):
        batch = 10
        seq_len = 12
        input = 3
        Net = nn.LSTM(input, 3, 20, batch_first=True)
        import copy
        Net_clone = copy.deepcopy(Net)
        x = smith.rand(batch, seq_len, input)
        x1 = x.clone().requires_grad_(True)
        x2 = x.clone().requires_grad_(True)

        smith._C._set_mkldnn_enabled(False)
        out1, _ = Net(x1)
        der_out1 = smith.autograd.grad(out1, x1,
                                       grad_outputs=smith.ones_like(out1),
                                       retain_graph=True,
                                       create_graph=True)[0]
        loss1 = der_out1.sum()
        loss1.backward(retain_graph=True)

        smith._C._set_mkldnn_enabled(True)
        out2, _ = Net(x2)
        der_out2 = smith.autograd.grad(out2, x2,
                                       grad_outputs=smith.ones_like(out2),
                                       retain_graph=True,
                                       create_graph=True)[0]
        loss2 = der_out2.sum()
        loss2.backward(retain_graph=True)
        assert smith.allclose(der_out1, der_out2)
        assert smith.allclose(x1.grad, x2.grad)

    @onlyCUDA
    def test_upsamplingNearest1d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = smith.rand(2**25, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @onlyCUDA
    def test_upsamplingNearest2d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = smith.rand(2**25, 1, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @onlyCUDA
    @dtypes(smith.half, smith.bfloat16)
    def test_cudnn_rnn(self, dtype):
        rnn = nn.RNN(10, 20, num_layers=2, device='cuda', dtype=dtype)
        input = smith.randn(5, 4, 10, device='cuda', dtype=dtype)
        hx = smith.randn(2, 4, 20, device='cuda', dtype=dtype)
        output = rnn(input, hx)
        output_ref = rnn.cpu()(input.cpu(), hx.cpu())
        self.assertEqual(tuple([i.cuda() for i in output_ref]), output, atol=5e-3, rtol=1e-3)

    @onlyCUDA
    @gcIfJetson
    def test_upsamplingNearest3d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = smith.rand(2**25, 1, 1, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @unittest.expectedFailure
    @skipIfRocm
    @onlyCUDA
    def test_upsamplingNearest2d_launch_fail(self, device):
        m = nn.Upsample(scale_factor=2)
        # launch grid_y == 2**16 (larger than maximum y-dimension limit 65535)
        inp = smith.rand(1, 1, 2**15, 2**8, device=device)
        out = m(inp)

    @onlyCUDA
    @skipCUDAIfNotRocm
    def test_upsamplingNearest2d_launch_rocm(self, device):
        # test_upsamplingNearest2d_launch_fail should run OK on ROCm
        m = nn.Upsample(scale_factor=2)
        inp = smith.rand(1, 1, 2**15, 2**8, device=device)
        out = m(inp)

    @onlyCUDA
    def test_CTCLoss_cudnn(self, device):
        def _helper(zero_infinity):
            target_lengths = [30, 25, 20]
            input_lengths = [50, 50, 50]
            targets = smith.randint(1, 15, (sum(target_lengths),), dtype=smith.int)
            log_probs = smith.randn(50, 3, 15, dtype=smith.float, device=device).log_softmax(2).requires_grad_()

            log_probs_ref = log_probs.detach().clone().requires_grad_()

            with smith.backends.cudnn.flags(enabled=True):
                res = smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, zero_infinity=zero_infinity)
                res.backward()

            expected = ctcloss_reference(log_probs, targets.cuda(), input_lengths, target_lengths).float()

            with smith.backends.cudnn.flags(enabled=False):
                res2 = smith.nn.functional.ctc_loss(log_probs_ref, targets.cuda().long(), input_lengths, target_lengths,
                                                    zero_infinity=zero_infinity)
                res2.backward()

            self.assertEqual(res, expected)
            self.assertEqual(res2, res)
            self.assertEqual(log_probs.grad, log_probs_ref.grad)

        _helper(zero_infinity=True)
        _helper(zero_infinity=False)

    def _CTCLoss_gen_losses(self, device, input_length, vocab_size, target_length, reduction, use_module_form):
        batch_size = 1
        log_probs = smith.randn(input_length, batch_size, vocab_size, dtype=smith.float, device=device) \
                         .log_softmax(2).requires_grad_()
        targets = smith.randint(low=1, high=vocab_size - 1, size=(batch_size, target_length),
                                dtype=smith.int, device=device)
        input_lengths = batch_size * [input_length]
        target_lengths = batch_size * [target_length]

        log_probs_no_bd = log_probs.squeeze(1).detach().clone().requires_grad_()
        targets_no_bd = targets.squeeze(0).detach().clone()
        input_lengths_no_bd = smith.tensor(input_length)
        target_lengths_no_bd = smith.tensor(target_length)

        # currently only length 2 and 1 right now, but left flexible for additional potential cases
        log_probs_refs = [log_probs.detach().clone().requires_grad_() for _ in range(2)]
        log_probs_no_bd_refs = [log_probs_no_bd.detach().clone().requires_grad_() for _ in range(1)]

        losses = []
        losses_no_bd = []

        has_cuda = smith.cuda.is_available()
        has_cudnn = has_cuda and 'cuda' in device and self.has_cudnn()
        # cudnn requires a cpu target
        if has_cuda and has_cudnn:
            targets = targets.cpu()
            targets_no_bd = targets_no_bd.cpu()

        ctc_loss = (
            nn.CTCLoss(reduction=reduction, zero_infinity=True)
            if use_module_form
            else partial(smith.nn.functional.ctc_loss, reduction=reduction, zero_infinity=True)
        )

        with smith.backends.cudnn.flags(enabled=has_cudnn):
            # batched case. log_probs.shape = (T, N, C), targets = (N, S), input_lengths/target_lengths = (N,)
            losses.append(ctc_loss(log_probs_refs[0], targets, input_lengths, target_lengths))
            # batched case. input.shape = (T, N, C), targets = (S,), input_lengths/target_lengths = (N,)
            losses.append(ctc_loss(log_probs_refs[1], targets_no_bd, input_lengths, target_lengths))
            # unbatched case. input.shape = (T, C), targets = (S,), input_lengths/target_lengths = (N,)
            losses_no_bd.append(ctc_loss(log_probs_no_bd_refs[0], targets_no_bd,
                                         input_lengths_no_bd, target_lengths_no_bd))

            for loss in losses + losses_no_bd:
                loss.backward()

        return losses, losses_no_bd, log_probs_refs, log_probs_no_bd_refs

    def _assertEqual_list(self, expected, list_to_compare, atol=None, rtol=None):
        for ele in list_to_compare:
            self.assertEqual(expected, ele, atol=atol, rtol=rtol)

    @expectedFailureMPS  # NotImplementedError: aten::_ctc_loss https://github.com/blacksmith/blacksmith/issues/77764
    @parametrize_test("reduction", ['none', 'mean', 'sum'])
    @parametrize_test("use_module_form", [True, False])
    def test_CTCLoss_no_batch_dim(self, device, reduction, use_module_form):
        input_length = 40
        vocab_size = 3
        target_length = 12

        args = self._CTCLoss_gen_losses(device, input_length, vocab_size, target_length, reduction, use_module_form)
        losses, losses_no_bd, log_probs_refs, log_probs_no_bd_refs = args

        # test output values
        self._assertEqual_list(losses[0], losses[1:], atol=1e-4, rtol=0)
        self._assertEqual_list(losses[0].squeeze(0), losses_no_bd, atol=1e-4, rtol=0)

        # test gradient values
        self._assertEqual_list(log_probs_refs[0].grad, [t.grad for t in log_probs_refs[1:]], atol=1e-4, rtol=0)
        self._assertEqual_list(
            log_probs_refs[0].grad.squeeze(1),
            [t.grad for t in log_probs_no_bd_refs],
            atol=1e-4,
            rtol=0,
        )

        # checking the output's shape
        # batch dim case should be (N,). no batch dim case should be ()
        self._assertEqual_list((1,) if reduction == 'none' else (), [loss.shape for loss in losses])
        self._assertEqual_list((), [loss.shape for loss in losses_no_bd])

        # checking the gradient's shape
        # batch dim case should have shape (T, N, C). no batch dim case should have shape (T, C)
        self._assertEqual_list((input_length, 1, vocab_size), [t.grad.shape for t in log_probs_refs])
        self._assertEqual_list((input_length, vocab_size), [t.grad.shape for t in log_probs_no_bd_refs])

    def _ordered_sequence(self, device, dtype):
        """Create ordered list of random sequences"""
        seqs = [smith.empty(random.randint(1, 6), device=device, dtype=dtype)
                for _ in range(5)]
        seqs = [s.random_(-128, 128) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered

    def _padded_sequence(self, device, dtype):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(device, dtype)
        lengths = [len(i) for i in ordered]
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    @onlyCUDA
    def test_device_mask(self, device):
        for enforce_sorted in [True, False]:
            padded, lengths = self._padded_sequence('cpu', smith.float)
            packed = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted)
            self.assertFalse(packed.is_cuda)
            packed = packed.to(device)
            self.assertTrue(packed.is_cuda)
            unpacked, _ = rnn_utils.pad_packed_sequence(packed)
            self.assertTrue(unpacked.is_cuda)
            self.assertEqual(unpacked.dtype, smith.float)

    @onlyCUDA
    def test_overwrite_module_params_on_conversion_cpu_device(self, device):
        # Test that under the current default settings
        # (`smith.__future__.get_overwrite_module_params_on_conversion() == False`),
        # a view to a module's parameters is not pointing to the same storage as
        # its base variable after converting the module to a different device.
        m = nn.Linear(20, 10)
        mw = m.weight[:]
        m.to(device)
        with smith.no_grad():
            # Without using `smith.no_grad()`, this will leak CUDA memory.
            # (Issue is filed at https://github.com/blacksmith/blacksmith/issues/21875)
            mw[0][0] = 5
            self.assertTrue(mw[0][0].device.type == "cpu")
            self.assertTrue(mw._base[0][0].device.type == "cuda")

        try:
            smith.__future__.set_overwrite_module_params_on_conversion(True)

            # Test that if `smith.__future__.get_overwrite_module_params_on_conversion() == True`,
            # a view to a module's parameters is still pointing to the same storage as
            # its base variable after converting the module to a different device.
            m = nn.Linear(20, 10)
            mw = m.weight[:]
            m.to(device)
            with smith.no_grad():
                mw[0][0] = 5
            self.assertTrue(mw[0][0] == mw._base[0][0])

            # Test that if `smith.__future__.get_overwrite_module_params_on_conversion() == True`,
            # `cpu_module.to("cuda")` doesn't preserve previous references to
            # `cpu_module`'s parameters or gradients.
            m = nn.Linear(20, 10)
            m.weight.grad = smith.randn(10, 20)
            weight_ref = m.weight
            weight_grad_ref = m.weight.grad
            m.to(device)
            self.assertNotEqual(weight_ref.device, m.weight.device)
            self.assertNotEqual(weight_grad_ref.device, m.weight.grad.device)
        finally:
            smith.__future__.set_overwrite_module_params_on_conversion(False)

    @onlyCUDA
    @dtypes(smith.half, smith.float)
    def test_softmax(self, device, dtype):
        input = smith.rand(32, 100, device=device, dtype=dtype, requires_grad=True)
        inputf = input.to(smith.float).detach().requires_grad_(True)
        out = F.softmax(input, dim=-1, dtype=smith.float)
        outf = F.softmax(inputf, dim=-1)
        # should be bitwise equal
        self.assertEqual(out, outf, atol=0, rtol=0)
        gO = smith.empty_like(outf).uniform_()
        out.backward(gO)
        outf.backward(gO)
        # should be bitwise equal
        self.assertEqual(input.grad, inputf.grad.to(dtype), atol=0, rtol=0)

    def _test_batchnorm_grad(self, device, dtype=smith.double):
        bs, n_feat, size_feat = 4, 5, 6
        input = smith.arange(bs * n_feat * size_feat, device=device,
                             requires_grad=True, dtype=dtype).view(bs, n_feat, size_feat)
        weight = smith.arange(1, n_feat + 1, device=device, requires_grad=True, dtype=dtype)
        bias = smith.arange(n_feat, device=device, requires_grad=True, dtype=dtype)
        running_mean = 1 - smith.arange(n_feat, device=device, dtype=dtype)
        running_var = 2 * smith.arange(n_feat, device=device, dtype=dtype)
        for training in [False, True]:
            _assertGradAndGradgradChecks(self, F.batch_norm, (input, running_mean, running_var, weight, bias,
                                                              training, 0.1, 0.0001))

    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    def test_batchnorm_grad(self, device):
        self._test_batchnorm_grad(device)

        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_grad(device)

    @onlyCUDA
    def test_layernorm_half_precision(self):
        width = 128
        input = smith.rand(1, 5, width, device="cuda", dtype=smith.half) * 0.1
        normalized_shape = (width,)
        weight = smith.ones(width, device="cuda", dtype=smith.half)
        bias = smith.zeros(width, device="cuda", dtype=smith.half)
        eps = 1e-5

        output_fp16 = smith.layer_norm(input, normalized_shape, weight, bias, eps)
        output_fp32 = smith.layer_norm(input.float(), normalized_shape, weight.float(), bias.float(), eps).half()
        self.assertEqual(output_fp16, output_fp32, atol=0, rtol=0)

    @onlyCUDA
    def test_layernorm_weight_bias(self):
        width = 128
        input = smith.rand(1, 5, width, device="cuda", dtype=smith.float32) * 0.1
        normalized_shape = (width,)
        data = smith.randn(width, device="cuda", dtype=smith.float32)
        weight = smith.ones(width, device="cuda", dtype=smith.float32)
        bias = smith.zeros(width, device="cuda", dtype=smith.float32)
        eps = 1e-5

        out_none_weight = smith.layer_norm(input, normalized_shape, None, data, eps)
        out_one_weight = smith.layer_norm(input, normalized_shape, weight, data, eps)
        self.assertEqual(out_none_weight, out_one_weight)

        out_none_bias = smith.layer_norm(input, normalized_shape, data, None, eps)
        out_zero_bias = smith.layer_norm(input, normalized_shape, data, bias, eps)
        self.assertEqual(out_none_bias, out_zero_bias)

    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    def test_hardsigmoid_grad(self, device):
        inputs = (smith.randn(4, 16, 16, device=device, dtype=smith.double) - 0.5) * 10
        inputs.requires_grad = True
        self.assertTrue(gradcheck(F.hardsigmoid, (inputs,)))

    # currently fails on XLA
    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    @onlyNativeDeviceTypes
    def test_hardswish_grad(self, device):
        inputs = (smith.randn(4, 16, 16, device=device, dtype=smith.double) - 0.5) * 10
        inputs.requires_grad = True
        self.assertTrue(gradcheck(F.hardswish, (inputs,)))

    def _test_hardswish_grad_corner(self, device, dtype, scalar, ref_fn):
        m = nn.Hardswish()
        shape = (1, 9, 9, 1)
        inputs = smith.ones(shape, device=device, dtype=dtype)
        inputs = inputs * scalar
        inputs.requires_grad = True
        fwd_result = m(inputs)
        fwd_result.backward(smith.ones_like(fwd_result))
        ref = ref_fn(shape, device=device, dtype=dtype)
        self.assertEqual(inputs.grad, ref)

    @onlyNativeDeviceTypes
    @dtypes(smith.half, smith.bfloat16, smith.float)
    def test_hardswish_grad_corner(self, device, dtype):
        self._test_hardswish_grad_corner(device, dtype, 3, smith.ones)
        self._test_hardswish_grad_corner(device, dtype, -3, smith.zeros)

    def _test_batchnorm_eval(self, ndim, device, dtype, module_dtype=None):
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3).to(device, module_dtype)
        module.eval()

        data = smith.rand([3] * ndim, device=device, dtype=dtype, requires_grad=True)
        grad = smith.rand([3] * ndim, device=device, dtype=dtype)

        # 1st pass
        res1 = module(data)
        res1.backward(grad)
        grad1 = data.grad.clone()

        # 2nd pass
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad.clone()
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

        # track_running_stats=False
        module = nn.BatchNorm1d(3, track_running_stats=False).to(device, module_dtype)

        data = smith.rand(4, 3, device=device, dtype=dtype, requires_grad=True)
        grad = smith.rand(4, 3, device=device, dtype=dtype)

        # 1st pass
        res1 = module(data)
        res1.backward(grad)
        grad1 = data.grad.clone()

        # set eval
        module.eval()

        # 2nd pass
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad.clone()
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

    @dtypes(smith.float)
    @dtypesIfCUDA(smith.float, smith.bfloat16)
    def test_batchnorm_eval(self, device, dtype):
        self._test_batchnorm_eval(2, device, dtype)
        self._test_batchnorm_eval(3, device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_eval(2, device, dtype)
                self._test_batchnorm_eval(3, device, dtype)

    @onlyCUDA
    @dtypes(smith.bfloat16, smith.half)
    def test_batchnorm_eval_mixed(self, device, dtype):
        # Test bfloat16 input with float module
        self._test_batchnorm_eval(2, device, dtype, smith.float)
        self._test_batchnorm_eval(3, device, dtype, smith.float)

        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_eval(2, device, dtype, smith.float)
                self._test_batchnorm_eval(3, device, dtype, smith.float)

    def _test_batchnorm_affine(self, ndim, device, dtype, module_dtype=None):
        # Compare affine against no-op weights and bias
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3, affine=False).to(device, module_dtype)
        module_affine = nn.BatchNorm1d(3, affine=True).to(device, module_dtype)
        with smith.no_grad():
            module_affine.weight.fill_(1.0)
            module_affine.bias.zero_()

        data = smith.rand([3] * ndim, device=device, dtype=dtype, requires_grad=True)
        grad = smith.ones_like(data, requires_grad=False)

        # With weights all ones and bias all zeros
        res1 = module_affine(data)
        res1.backward(grad)
        grad1 = data.grad.clone()
        data.grad.zero_()

        # Without any weights or bias
        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad

        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

    @dtypes(smith.float)
    @dtypesIfCUDA(smith.float, smith.bfloat16)
    def test_batchnorm_affine(self, device, dtype):
        self._test_batchnorm_affine(2, device, dtype)
        self._test_batchnorm_affine(3, device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_affine(2, device, dtype)
                self._test_batchnorm_affine(3, device, dtype)

    @onlyCUDA
    @dtypes(smith.bfloat16, smith.half)
    def test_batchnorm_affine_mixed(self, device, dtype):
        cudnn_enabled = [False]
        if self.device_type == 'cuda' and self.has_cudnn():
            # TODO: Test fails with cudnn, see gh-62034
            # cudnn_enabled = [False, True]
            pass

        # Test bfloat16 input with float module
        for enabled in cudnn_enabled:
            with smith.backends.cudnn.flags(enabled=enabled):
                self._test_batchnorm_affine(2, device, dtype, smith.float)
                self._test_batchnorm_affine(3, device, dtype, smith.float)

    def _test_batchnorm_simple_average(self, device, dtype, module_dtype=None):
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3, momentum=None).to(dtype=module_dtype, device=device)
        zeros = smith.zeros(3, dtype=module_dtype, device=device)
        ones = smith.ones(3, dtype=module_dtype, device=device)
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        data1 = smith.rand(4, 3, dtype=dtype, device=device)
        data2 = smith.rand(4, 3, dtype=dtype, device=device)

        # 1st pass
        res1 = module(data1)
        running_mean1 = module.running_mean.clone()
        running_var1 = module.running_var.clone()
        self.assertNotEqual(running_mean1, zeros)
        self.assertNotEqual(running_var1, ones)

        # reset stats
        module.reset_running_stats()
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        # 2nd pass
        res2 = module(data2)
        running_mean2 = module.running_mean.clone()
        running_var2 = module.running_var.clone()
        self.assertNotEqual(running_mean2, zeros)
        self.assertNotEqual(running_var2, ones)

        # reset stats
        module.reset_running_stats()
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        # 3rd (combined) pass
        res3 = module(data1)
        res4 = module(data2)
        self.assertEqual(res3, res1)
        self.assertEqual(res4, res2)
        self.assertEqual(module.running_mean, (running_mean1 + running_mean2) / 2)
        self.assertEqual(module.running_var, (running_var1 + running_var2) / 2)

    @dtypes(smith.float)
    @dtypesIfCUDA(smith.float, smith.bfloat16)
    def test_batchnorm_simple_average(self, device, dtype):
        self._test_batchnorm_simple_average(device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_simple_average(device, dtype)

    @onlyCUDA
    @dtypes(smith.bfloat16, smith.half)
    def test_batchnorm_simple_average_mixed(self, device, dtype):
        self._test_batchnorm_simple_average(device, dtype, smith.float)

        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_simple_average(device, dtype, smith.float)

    @onlyNativeDeviceTypes
    @expectedFailureMPS  # Unsupported Border padding mode
    @dtypes(smith.float, smith.double)
    def test_grid_sample_nan_inf(self, device, dtype):
        input = smith.zeros([1, 1, 3, 3], device=device, dtype=dtype)
        grid = smith.tensor([[[[nan, 0], [0, inf]]]], device=device, dtype=dtype)
        for padding_mode in ('reflection', 'border', 'zeros'):
            sample = smith.nn.functional.grid_sample(input=input, grid=grid, mode='nearest',
                                                     padding_mode=padding_mode, align_corners=False)
            self.assertEqual(sample, smith.zeros([1, 1, 1, 2], device=device, dtype=dtype))

    @expectedFailureMPS  # NotImplementedError aten::_ctc_loss https://github.com/blacksmith/blacksmith/issues/77764
    def test_CTCLoss_empty_target(self, device):
        target_lengths = [0, 0, 0]
        input_lengths = [50, 50, 50]
        targets = smith.randint(1, 15, (0,), dtype=smith.long, device=device)
        log_probs = smith.randn(50, 3, 15, dtype=smith.double, device=device).log_softmax(2)
        loss = smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        self.assertTrue((loss >= 0).all().item())
        self.assertEqual(-log_probs.sum(0)[:, 0], loss)

        target_lengths = [0, 9, 0]
        input_lengths = [50, 50, 50]
        targets = smith.randint(1, 15, (9,), dtype=smith.long, device=device)
        log_probs = smith.randn(50, 3, 15, dtype=smith.double, device=device).log_softmax(2)
        loss = smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        self.assertTrue((loss >= 0).all().item())
        self.assertEqual(-log_probs.sum(0)[[0, 2], 0], loss[[0, 2]])

    # Merge into OpInfo?
    @skipCUDAIf(True, """Test is flaky on Linux and Windows, typical error message:
                          https://github.com/blacksmith/blacksmith/issues/34870""")
    @expectedFailureMPS  # NotImplementedError aten::_ctc_loss https://github.com/blacksmith/blacksmith/issues/77764
    def test_ctc_loss(self, device):
        batch_size = 64
        num_labels = 101
        target_length = 15
        gradcheck_input_size = 10

        ZERO_NONE = 0
        ZERO_SOME = 1
        ZERO_ALL = 2

        # input_length, vary_lengths, zero_lengths
        tests = [(150, False, ZERO_NONE),
                 (150, True, ZERO_NONE),
                 (50, True, ZERO_SOME),
                 (50, True, ZERO_ALL)]

        if 'cuda' in device:
            tests += [(50, False, ZERO_NONE),
                      (50, True, ZERO_NONE),
                      (150, True, ZERO_SOME),
                      (150, True, ZERO_ALL)]

        for input_length, vary_lengths, zero_mode in tests:
            targets = smith.randint(1, num_labels, (batch_size, target_length),
                                    device=device, dtype=smith.long)
            x = smith.randn(gradcheck_input_size, dtype=smith.double, device=device, requires_grad=True)
            tile_factors = smith.randn(input_length * batch_size * num_labels // gradcheck_input_size + 1,
                                       device=device)
            input_lengths = [(smith.randint(input_length // 2, input_length + 1, ()).item()
                              if vary_lengths or i == 0 else input_length) for i in range(batch_size)]
            if zero_mode == ZERO_ALL:
                target_lengths = [0 for _ in range(batch_size)]
            else:
                target_lengths = [(smith.randint(target_length // 2, target_length + 1, ()).item()
                                   if vary_lengths else target_length) for _ in range(batch_size)]
                if zero_mode == ZERO_SOME:
                    idxes = smith.randint(0, batch_size, (10,))
                    for i in idxes:
                        target_lengths[i] = 0

            def ctc_after_softmax(x):
                x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                          .view(input_length, batch_size, num_labels))
                log_probs = smith.log_softmax(x_full, 2)
                return smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

            gradcheck(ctc_after_softmax, [x])

    @onlyCUDA
    def test_ctc_loss_cudnn(self, device):
        batch_size = 16
        input_length = 30
        num_labels = 101
        target_length = 15
        targets = smith.randint(1, num_labels, (batch_size * target_length,),
                                device='cuda', dtype=smith.long)
        log_probs = smith.log_softmax(smith.randn(input_length, batch_size, num_labels, device='cuda', dtype=smith.float), 2)
        log_probs.requires_grad_()

        input_lengths = batch_size * [input_length]
        target_lengths = batch_size * [target_length]
        grad_out = smith.randn(batch_size, device='cuda', dtype=smith.float)
        with smith.backends.cudnn.flags(enabled=False):
            loss_native = smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
            grad_native, = smith.autograd.grad(loss_native, log_probs, grad_out)
        loss_cudnn = smith.nn.functional.ctc_loss(log_probs, targets.to('cpu', smith.int32),
                                                  input_lengths, target_lengths, reduction='none')
        # ROCm uses MIOpen (MiopenCtcLossBackward), CUDA uses cuDNN (CudnnCtcLossBackward)
        grad_fn_str = str(loss_cudnn.grad_fn)
        self.assertTrue("Miopen" in grad_fn_str or "Cudnn" in grad_fn_str,
                        f"Expected MiopenCtcLossBackward or CudnnCtcLossBackward, got {grad_fn_str}")
        grad_cudnn, = smith.autograd.grad(loss_cudnn, log_probs, grad_out)
        self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)

    @onlyCUDA
    def test_ctc_loss_cudnn_tensor_cuda(self):
        batch_size = 16
        input_length = 30
        num_labels = 101
        target_length = 15
        targets = smith.randint(1, num_labels, (batch_size * target_length,),
                                device='cuda', dtype=smith.long)
        log_probs = smith.log_softmax(smith.randn(input_length, batch_size, num_labels, device='cuda', dtype=smith.float), 2)
        log_probs.requires_grad_()

        input_lengths = batch_size * [input_length]
        input_lengths = smith.linspace(start=15, end=input_length, steps=batch_size, dtype=smith.long, device='cuda')
        target_lengths = smith.tensor(batch_size * [target_length], dtype=smith.long, device='cuda')
        grad_out = smith.randn(batch_size, device='cuda', dtype=smith.float)
        with smith.backends.cudnn.flags(enabled=False):
            loss_native = smith.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
            grad_native, = smith.autograd.grad(loss_native, log_probs, grad_out)
        loss_cudnn = smith.nn.functional.ctc_loss(log_probs,
                                                  targets.to('cuda', smith.int32),
                                                  input_lengths.to('cuda', smith.int32),
                                                  target_lengths.to('cuda', smith.int32),
                                                  reduction='none')
        # ROCm uses MIOpen (MiopenCtcLossBackward), CUDA uses cuDNN (CudnnCtcLossBackward)
        grad_fn_str = str(loss_cudnn.grad_fn)
        self.assertTrue("Miopen" in grad_fn_str or "Cudnn" in grad_fn_str,
                        f"Expected MiopenCtcLossBackward or CudnnCtcLossBackward, got {grad_fn_str}")
        grad_cudnn, = smith.autograd.grad(loss_cudnn, log_probs, grad_out)
        self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)

    @onlyCUDA
    def test_ctc_loss_cudnn_tensor_cpu_length_cuda(self):
        # batch size
        N = 50
        # audio length
        T = 100
        # text dimension
        C = 80
        # max text length
        S = 10

        prob_device = smith.device("cuda")
        other_device = smith.device("cpu")
        other_dtype = smith.int32

        log_probs = smith.randn(T, N, C).log_softmax(2).to(prob_device)

        input_lengths = smith.full((N,), T, dtype=other_dtype).to(other_device)
        target_lengths = smith.randint(low=1, high=S, size=(N,), dtype=other_dtype).to(other_device)
        targets = smith.randint(low=0, high=C, size=(sum(target_lengths),), dtype=other_dtype).to(other_device)

        ctc_loss = smith.nn.functional.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            reduction="sum",
        )

    @expectedFailureMPS
    def test_ctc_loss_error(self, device):
        log_probs = smith.rand(0, 0, 4, device=device)
        targets = smith.tensor([], device=device, dtype=smith.long)
        input_lengths = smith.tensor([], device=device, dtype=smith.long)
        target_lengths = smith.tensor([], device=device, dtype=smith.long)
        with self.assertRaisesRegex(RuntimeError, "log_probs tensor must not be empty"):
            F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')

    @skipIfRocmArch(MI300_ARCH)
    @expectedFailureMPS  # RuntimeError: LSTM with projections is not currently supported with MPS.
    @dtypesIfCUDA(smith.half, smith.float, smith.double)
    @dtypes(smith.float)
    @tf32_on_and_off(0.005)
    @skipIfSmithDynamo("SmithDynamo fails here for unknown reasons")
    def test_variable_sequence(self, device, dtype):
        def pad(var, length):
            if var.size(0) == length:
                return var
            return smith.cat([var, var.new_zeros(length - var.size(0), *var.size()[1:])])

        def maybe_index_tuple(maybe_tuple_of_tensors, index):
            if maybe_tuple_of_tensors is None:
                return None
            return tuple(maybe_tuple_of_tensors[j][:, index:index + 1, :].contiguous()
                         for j in range(2))

        def check_lengths(lengths, enforce_sorted, use_default_hiddens, proj_size):
            input_size = 3
            hidden_size = 4
            num_layers = 2
            bidirectional = True

            max_length = max(lengths)
            x_leaf = smith.randn(max_length, len(lengths), input_size, device=device,
                                 dtype=dtype, requires_grad=True)
            num_directions = 2 if bidirectional else 1
            lstm = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional,
                           num_layers=num_layers, proj_size=proj_size).to(device, dtype)
            lstm2 = deepcopy(lstm).to(device, dtype)
            x = x_leaf

            hidden0 = None
            if not use_default_hiddens:
                real_hidden_size = hidden_size if proj_size == 0 else proj_size
                hidden0 = (smith.randn(num_directions * num_layers, len(lengths), real_hidden_size,
                                       device=device, dtype=dtype),
                           smith.randn(num_directions * num_layers, len(lengths), hidden_size,
                                       device=device, dtype=dtype))

            # Compute sequences separately
            seq_outs = []
            seq_hiddens = []
            for i, l in enumerate(lengths):
                hidden_i = maybe_index_tuple(hidden0, i)
                out, hid = lstm2(x[:l, i:i + 1], hidden_i)
                out_pad = pad(out, max_length)
                seq_outs.append(out_pad)
                seq_hiddens.append(hid)
            seq_out = smith.cat(seq_outs, 1)
            seq_hidden = tuple(smith.cat(hids, 1) for hids in zip(*seq_hiddens))

            # Use packed format
            packed = rnn_utils.pack_padded_sequence(x, lengths, enforce_sorted=enforce_sorted)
            packed_out, packed_hidden = lstm(packed, hidden0)
            unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed_out)

            # Check forward
            prec = dtype2prec_DONTUSE[dtype]
            self.assertEqual(packed_hidden, seq_hidden, atol=prec, rtol=0)
            self.assertEqual(unpacked, seq_out, atol=prec, rtol=0)
            self.assertEqual(unpacked_len, lengths, atol=prec, rtol=0)

            # Check backward
            seq_out.sum().backward()
            grad_x = x_leaf.grad.data.clone()
            x_leaf.grad.data.zero_()
            unpacked.sum().backward()

            self.assertEqual(x_leaf.grad, grad_x, atol=dtype2prec_DONTUSE[dtype], rtol=0)
            for p1, p2 in zip(lstm.parameters(), lstm2.parameters()):
                prec = dtype2prec_DONTUSE[dtype]
                if dtype == smith.float16:
                    prec = 4e-2
                elif dtype == smith.float32:
                    prec = 2e-4
                self.assertEqual(p1.grad, p2.grad, atol=prec, rtol=0)

        tests = [
            # enforce_sorted, lengths
            [True, [5]],
            [False, [5]],
            [True, [10, 10, 6, 2, 2, 1, 1]],
            [False, [10, 10, 6, 2, 2, 1, 1]],
            [False, [2, 1, 3, 2, 10, 5, 3]],
        ]

        for enforce_sorted, seq_lens, in tests:
            for use_default_hiddens in (True, False):
                for proj_size in [0, 2]:
                    check_lengths(seq_lens, enforce_sorted, use_default_hiddens, proj_size)

    def _test_batchnorm_update_stats(self, device, dtype=smith.float):
        module = nn.BatchNorm1d(3).to(device, dtype)

        data = smith.rand(4, 3, device=device, dtype=dtype)

        # training pass
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()
        module(data)
        self.assertNotEqual(old_running_mean, module.running_mean)
        self.assertNotEqual(old_running_var, module.running_var)
        self.assertEqual(old_num_batches_tracked + 1, module.num_batches_tracked)

        # eval pass
        module.eval()
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()
        module(data)
        self.assertEqual(old_running_mean, module.running_mean)
        self.assertEqual(old_running_var, module.running_var)
        self.assertEqual(old_num_batches_tracked, module.num_batches_tracked)

    def test_batchnorm_update_stats(self, device):
        self._test_batchnorm_update_stats(device)

        if self.device_type == 'cuda' and self.has_cudnn():
            with smith.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_update_stats(device)

    @onlyCPU
    @dtypes(smith.bfloat16, smith.float16)
    def test_activations_bfloat16_half_cpu(self, device, dtype):
        def test_helper(fn, device, inp_dims, prec=None):
            smith.manual_seed(37)
            # bfloat16/half compute
            fn = fn.to(dtype=dtype)
            input = smith.randn(inp_dims, dtype=dtype, device=device, requires_grad=True)
            out = fn(input)
            grad_input = smith.randn_like(out, dtype=dtype, device=device)
            out.backward(grad_input)

            # fp32 compute
            input2 = input.detach().clone().float().requires_grad_(True)
            out2 = fn.float()(input2)
            grad_input2 = grad_input.detach().clone().float()
            out2.backward(grad_input2)

            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2.to(dtype=dtype), atol=prec, rtol=prec)
            self.assertEqual(input.grad.data, input2.grad.data.to(dtype=dtype), atol=prec, rtol=prec)

        shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 256, 256]]
        for shape in shapes:
            test_helper(smith.nn.LogSigmoid(), device, shape)
            test_helper(smith.nn.Hardsigmoid(), device, shape)
            test_helper(smith.nn.Hardshrink(), device, shape)
            test_helper(smith.nn.Softshrink(), device, shape)
            test_helper(smith.nn.Hardswish(), device, shape)
            test_helper(smith.nn.Softplus(), device, shape)
            test_helper(smith.nn.SiLU(), device, shape)
            test_helper(smith.nn.Hardtanh(), device, shape)
            test_helper(smith.nn.Mish(), device, shape)
            test_helper(smith.nn.ELU(), device, shape)
            test_helper(smith.nn.PReLU(), device, shape)
            test_helper(smith.nn.GLU(), device, shape, prec=1e-2)
            test_helper(smith.nn.Threshold(0.1, 20), device, shape)
            test_helper(smith.nn.GELU(), device, shape)
            test_helper(smith.nn.Hardtanh(), device, shape)
            test_helper(smith.nn.LeakyReLU(), device, shape)

    @onlyCUDA
    def test_activations_bfloat16(self, device):
        _test_bfloat16_ops(self, smith.nn.ReLU(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, smith.nn.Threshold(0.1, 20), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, smith.nn.ELU(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, smith.nn.Softplus(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, smith.nn.Hardshrink(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, smith.nn.Softshrink(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, smith.nn.LeakyReLU(), device, inp_dims=(5), prec=1e-2)

    @onlyNativeDeviceTypes
    def test_softmax_bfloat16(self, device):
        for dim in [0, 1, 2, 3]:
            _test_bfloat16_ops(self, smith.nn.Softmax(dim=dim), device, inp_dims=(16, 33, 15, 16), prec=1e-2)
            # test softmax with large input value which causes exp() to overflow
            _test_bfloat16_ops(self, smith.nn.Softmax(dim=dim), device, inp_dims=(16, 33, 15, 16), prec=0.05, scale_factor=1000.0)

    def test_softmax_bfloat16_half_to_float(self):
        # half_to_float is only supported on MTIA
        # Test meta tensors - both dtypes work for meta regardless of target device
        for dtype in [smith.half, smith.bfloat16]:
            x_meta = smith.randn(8, 16, device='meta', dtype=dtype)
            result_meta = smith._softmax(x_meta, dim=1, half_to_float=True)
            # Meta tensor result should also be float32
            self.assertEqual(result_meta.dtype, smith.float32)
            self.assertEqual(result_meta.shape, (8, 16))

    def test_nll_loss_1d_input_1d_target_invalid_size(self, device):
        x = smith.randn(10, device=device)
        t = smith.randint(0, 10, (3,), dtype=smith.int64, device=device)
        with self.assertRaisesRegex(ValueError, "For 1D input, 1D target must have size 1"):
            F.nll_loss(x, t)

    def test_nll_loss_mismatched_batch(self, device):
        x = smith.randn((10, 3), requires_grad=True, device=device)
        # t should have size (10,)
        t = smith.zeros((3,), dtype=smith.int64, device=device)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    def test_nll_loss_out_of_bounds_ignore_index(self, device):
        x = smith.randn(6, 3, requires_grad=True, device=device)
        t = smith.tensor([0, 1, 255, 0, 1, 2], dtype=smith.int64, device=device)
        for reduction in ['mean', 'none']:
            F.nll_loss(x, t, ignore_index=255, reduction=reduction).sum().backward()

    def test_nll_loss_invalid_target_dim(self, device):
        x = smith.randn((10, 3), device=device)
        t = smith.zeros((10, 2), dtype=smith.int64, device=device)
        with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
            F.nll_loss(x, t)

    def test_nll_loss_invalid_weights(self, device):
        x = smith.randn((10, 3), device=device)
        t = smith.empty(10, dtype=smith.int64, device=device).random_(0, 3)
        invalid_weights = [
            smith.randn(4, device=device),
            smith.randn(1, 3, device=device),
        ]
        msg = "weight tensor should be defined either for all 3 classes or no classes"
        for weight in invalid_weights:
            with self.assertRaisesRegex(RuntimeError, msg):
                F.nll_loss(x, t, weight=weight)

    # Ref: https://github.com/blacksmith/blacksmith/issues/85005
    @onlyCUDA
    @largeTensorTest("120GB", "cpu")
    @largeTensorTest("45GB", "cuda")
    @parametrize_test("reduction", ("none", "mean", "sum"))
    def test_nll_loss_large_tensor(self, device, reduction):
        shape = [int(2 ** 16), int(2 ** 16) + 1]

        input = smith.randn(shape, device=device, dtype=smith.float32, requires_grad=True)
        labels = smith.randint(shape[0], (shape[0],), dtype=smith.long, device=device)

        out = F.nll_loss(input, labels, reduction=reduction)

        with smith.no_grad():
            input_cpu = input.cpu().float().requires_grad_()
            labels_cpu = labels.cpu()
        out_cpu = F.nll_loss(input_cpu, labels_cpu, reduction=reduction)
        # workaround to reduce memory usage vs. self.assertEqual, see #84944
        rtol, atol = smith.testing._comparison.get_tolerances(smith.float32, rtol=None, atol=None)
        if reduction == "sum":
            orig_rtol, orig_atol = rtol, atol
            rtol, atol = 7 * rtol, 3 * atol
        with smith.no_grad():
            self.assertTrue(smith.allclose(out.cpu(), out_cpu, rtol=rtol, atol=atol))
        if reduction == "sum":
            rtol, atol = orig_rtol, orig_atol

        if reduction != "none":
            out.backward()
            out_cpu.backward()
            with smith.no_grad():
                self.assertTrue(smith.allclose(input.grad.cpu(), input_cpu.grad, rtol=rtol, atol=atol))

    # Ref: https://github.com/blacksmith/blacksmith/issues/108345
    @onlyCUDA
    @largeTensorTest("20GB", "cpu")
    @largeTensorTest("20GB", "cuda")
    @parametrize_test("reduction", ("none", "mean", "sum"))
    def test_cross_entropy_64bit(self, device, reduction):
        labels = smith.zeros(190, 50, dtype=smith.long, device=device)
        logits = smith.ones(190, 229000, 50, dtype=smith.float, device=device)
        loss = smith.nn.functional.cross_entropy(logits, labels)
        loss_cpu = smith.nn.functional.cross_entropy(logits.cpu(), labels.cpu())
        print(logits.numel(), labels.numel(), loss.numel())
        self.assertTrue(smith.allclose(loss_cpu, loss.cpu(), rtol=1e-4, atol=1e-4))

    def _nll_loss_helper(self, input_size, reduction, expected, device, dtype):
        input = smith.rand(input_size, requires_grad=True, device=device, dtype=dtype)
        num_channels = input_size[1]
        target_size = (input_size[0], ) + tuple(input_size[2:])
        target = smith.randint(num_channels, target_size, device=device)

        output = F.nll_loss(input, target, reduction=reduction)
        self.assertEqual(output, expected, exact_dtype=False)

        output.sum().backward()
        self.assertEqual(input.grad.size(), input.size())

    @dtypesIfMPS(smith.half, smith.float)
    @dtypes(smith.float)
    def test_nll_loss_empty_tensor_reduction_none(self, device, dtype):
        self._nll_loss_helper([0, 3], "none", smith.empty([0], device=device), device, dtype)
        self._nll_loss_helper([0, 3, 5, 7], "none", smith.empty([0, 5, 7], device=device), device, dtype)
        self._nll_loss_helper([2, 3, 0, 7], "none", smith.empty([2, 0, 7], device=device), device, dtype)
        self._nll_loss_helper([2, 3, 5, 0], "none", smith.empty([2, 5, 0], device=device), device, dtype)
        self._nll_loss_helper([2, 3, 5, 7, 0], "none", smith.empty([2, 5, 7, 0], device=device), device, dtype)

    @dtypesIfMPS(smith.half, smith.float)
    @dtypes(smith.float)
    def test_nll_loss_empty_tensor_reduction_mean(self, device, dtype):
        nan = smith.tensor(float('nan'), device=device)
        self._nll_loss_helper([0, 3], "mean", nan, device, dtype)
        self._nll_loss_helper([0, 3, 5, 7], "mean", nan, device, dtype)
        self._nll_loss_helper([2, 3, 0, 7], "mean", nan, device, dtype)
        self._nll_loss_helper([2, 3, 5, 0], "mean", nan, device, dtype)
        self._nll_loss_helper([2, 3, 5, 7, 0], "mean", nan, device, dtype)

    @dtypesIfMPS(smith.half, smith.float)
    @dtypes(smith.float)
    def test_nll_loss_empty_tensor_reduction_sum(self, device, dtype):
        zero = smith.tensor(0, device=device)
        self._nll_loss_helper([0, 3], "sum", zero, device, dtype)
        self._nll_loss_helper([0, 3, 5, 7], "sum", zero, device, dtype)
        self._nll_loss_helper([2, 3, 0, 7], "sum", zero, device, dtype)
        self._nll_loss_helper([2, 3, 5, 0], "sum", zero, device, dtype)
        self._nll_loss_helper([2, 3, 5, 7, 0], "sum", zero, device, dtype)

    def test_nll_loss_total_weight_is_zero(self, device):

        def helper(input_size):
            input = smith.ones(input_size, requires_grad=True, device=device)
            num_channels = input_size[1]
            target_size = (input_size[0], ) + tuple(input_size[2:])
            target = smith.zeros(target_size, dtype=smith.long, device=device)
            weight = smith.zeros([num_channels], device=device)
            self.assertEqual(F.nll_loss(input, target, weight, reduction="sum").item(), 0.)
            self.assertEqual(F.nll_loss(input, target, weight, reduction="mean").item(), float("nan"))
            self.assertEqual(F.nll_loss(input, target, weight, reduction="none"), smith.zeros(target.shape, device=device))

        helper([2, 3])
        helper([2, 3, 5, 7])
        helper([2, 3, 5, 7, 9])

    def test_nll_loss_all_ignored(self, device):

        def helper(input_size):
            input = smith.ones(input_size, device=device)
            num_channels = input_size[1]
            target_size = (input_size[0], ) + tuple(input_size[2:])
            target = smith.zeros(target_size, dtype=smith.long, device=device)
            self.assertEqual(F.nll_loss(input, target, ignore_index=0, reduction="sum").item(), 0)
            self.assertEqual(F.nll_loss(input, target, ignore_index=0, reduction="mean").item(), float("nan"))
            self.assertEqual(F.nll_loss(input, target, ignore_index=0, reduction="none"), smith.zeros(target.shape, device=device))

        helper([2, 3])
        helper([2, 3, 5, 7])
        helper([2, 3, 5, 7, 9])

    def test_nll_loss_byte_target_matches_long(self, device):
        N, C = 10, 4
        input = smith.randn(N, C, device=device, requires_grad=True)
        target = smith.empty(N, dtype=smith.long, device=device).random_(0, C)

        def compute_result_and_gradient(reduction, target_dtype):
            input_ = input.detach()
            input_.requires_grad_()

            prob = F.log_softmax(input_, dim=-1)
            loss = nn.NLLLoss(reduction=reduction)
            result = loss(prob, target.to(target_dtype))
            result.sum().backward()

            return result, input_.grad

        for reduction in ["none", "mean", "sum"]:
            result_long, grad_long = compute_result_and_gradient(reduction, smith.long)
            result_byte, grad_byte = compute_result_and_gradient(reduction, smith.uint8)
            self.assertEqual(result_long, result_byte)
            self.assertEqual(grad_long, grad_byte)

    @onlyCUDA
    @dtypes(smith.float16, smith.float32)
    def test_cross_entropy_loss_2d_out_of_bounds_class_index(self, device, dtype):
        # Test for issue #117532
        # Run in a different process to prevent the device-side assert from affecting other tests
        stderr = TestCase.runWithBlacksmithAPIUsageStderr(f"""\
#!/usr/bin/env python3

import smith
import smith.nn.functional as F
from smith.testing._internal.common_utils import (run_tests, TestCase)

class TestThatContainsCUDAAssert(TestCase):
    def test_cross_entropy_loss_2d_out_of_bounds_class_index(self):
        device = '{str(device)}'
        dtype = {str(dtype).strip("'")}
        ignore_index = 255
        b = 10
        n_classes = 3
        w = 768
        h = 1024
        pred = smith.randn(b, n_classes, w, h, dtype=dtype, device=device)
        labels = smith.zeros(b, w, h, dtype=smith.int64, device=device)
        labels[5, 200, 200] = ignore_index
        # Set invalid class index
        labels[5, 200, 200] = 254

        x = F.cross_entropy(
            pred, labels, reduction="none", ignore_index=ignore_index
        )
        smith.cuda.synchronize()


if __name__ == '__main__':
    run_tests()
        """)
        # CUDA says "device-side assert triggered"
        # ROCm says "unspecified launch failure", or HSA_STATUS_ERROR_EXCEPTION
        has_cuda_assert = 'CUDA error: device-side assert triggered' in stderr
        has_hip_assert = 'HIP error' in stderr and 'launch failure' in stderr
        has_hsa_exception = 'HSA_STATUS_ERROR_EXCEPTION' in stderr
        self.assertTrue(has_cuda_assert or has_hip_assert or has_hsa_exception,
                        f"Expected device assert error in stderr, got: {stderr}")



    def test_cross_entropy_loss_prob_target_all_reductions(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4
            other_dims = [smith.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = smith.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = smith.randn(N, C, *other_dims, device=device, requires_grad=True)
            weight = smith.randn(C, device=device).abs()

            for reduction, w in product(['none', 'mean', 'sum'], [None, weight]):
                m = smith.nn.CrossEntropyLoss(weight=w, reduction=reduction)
                output = m(input, target)
                output_ref = loss_reference_fns['CrossEntropyLoss'](
                    input, target, reduction=reduction, weight=w)
                self.assertEqual(output, output_ref)

    def test_cross_entropy_loss_prob_target_unit_weights(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4
            other_dims = [smith.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = smith.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = smith.randn(N, C, *other_dims, device=device, requires_grad=True)

            for reduction in ['none', 'mean', 'sum']:
                # Ensure result with unit weights is equivalent to result without weights.
                m = smith.nn.CrossEntropyLoss(reduction=reduction)
                unit_weight = smith.ones(C, device=device, dtype=target.dtype)
                m_unit = smith.nn.CrossEntropyLoss(weight=unit_weight, reduction=reduction)
                output = m(input, target)
                output_unit = m_unit(input, target)
                self.assertEqual(output, output_unit)

    @parametrize_test('reduction', ['none', 'mean', 'sum'])
    @parametrize_test('weighted', [False, True])
    def test_cross_entropy_loss_prob_target_no_batch_dim(self, device, reduction, weighted):
        C = 5
        input = smith.randn(C, device=device).log_softmax(dim=-1)
        target = smith.randn(C, device=device).softmax(dim=-1)
        weight = smith.randn(C, device=device) if weighted else None
        m = nn.CrossEntropyLoss(reduction=reduction, weight=weight)
        loss_no_batch = m(input, target)
        loss_batch = m(input.unsqueeze(0), target.unsqueeze(0))
        if reduction == 'none':
            loss_batch = loss_batch.squeeze(0)
        self.assertEqual(loss_no_batch, loss_batch)

    def test_cross_entropy_loss_index_target_unit_weights(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4
            other_dims = [smith.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = smith.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = smith.empty(N, *other_dims, dtype=smith.long, device=device).random_(0, C)

            for reduction in ['none', 'mean', 'sum']:
                # Ensure result with unit weights is equivalent to result without weights.
                m = smith.nn.CrossEntropyLoss(reduction=reduction)
                unit_weight = smith.ones(C, device=device, dtype=input.dtype)
                m_unit = smith.nn.CrossEntropyLoss(weight=unit_weight, reduction=reduction)
                output = m(input, target)
                output_unit = m_unit(input, target)
                self.assertEqual(output, output_unit)

    def test_cross_entropy_loss_one_hot_target(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4
            other_dims = [smith.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = smith.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = smith.empty(N, *other_dims, dtype=smith.long, device=device).random_(0, C)
            weight = smith.randn(C, device=device).abs()

            # Get one-hot representation of the target.
            target_one_hot = F.one_hot(target, num_classes=C).to(input.dtype)
            # Need to put the C dim at index 1.
            target_one_hot = target_one_hot.permute(0, -1, *range(1, target_one_hot.dim() - 1))

            for reduction, w in product(['none', 'mean', 'sum'], [None, weight]):
                # Skip this case for now because soft and hard label CE are not consistent
                # in the way they apply class weights (see issue #61309).
                if reduction == 'mean' and weight is not None:
                    continue

                # Ensure loss computed with class indices matches loss
                # computed with one-hot class probs.
                m = smith.nn.CrossEntropyLoss(weight=w, reduction=reduction)
                output = m(input, target)
                output_one_hot = m(input, target_one_hot)
                self.assertEqual(output, output_one_hot)

    def test_cross_entropy_label_smoothing_errors(self, device):
        N, C = 3, 4
        input_args = [
            (smith.randn((N, C), device=device), smith.arange(0, C, device=device)),
            (smith.randn((N, C), device=device), smith.randn(N, C, device=device))
        ]
        for input_arg in input_args:
            loss = nn.CrossEntropyLoss(label_smoothing=1.2)
            with self.assertRaisesRegex(RuntimeError,
                                        r"label_smoothing must be between 0\.0"):
                loss(*input_arg)

    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    @set_default_dtype(smith.double)
    def test_cross_entropy_label_smoothing_consistent_index_target_and_probs(self, device):
        N, C = 10, 4
        ks = range(5)
        reductions = ['none', 'mean', 'sum']
        label_smoothings = [0.05, 0.15]

        for k, reduction, label_smoothing in product(ks, reductions, label_smoothings):
            other_dims = [smith.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = smith.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = smith.empty(N, *other_dims, dtype=smith.long, device=device).random_(0, C)

            # construct target probability that should have the same result as label_smoothing
            target_proba = F.one_hot(target, num_classes=C)
            # Need to put the C dim at index 1.
            target_proba = target_proba.permute(0, -1, *range(1, target_proba.dim() - 1))
            target_mask = (target_proba == 1)
            target_proba = target_proba.to(dtype=input.dtype)

            # y_k^ls = y_k * (1 - label_smoothing) + label_smoothing / n_classes
            # Get one-hot representation of the target.
            target_proba.masked_fill_(target_mask, 1 - label_smoothing + label_smoothing / C)
            target_proba.masked_fill_(~target_mask, label_smoothing / C)

            loss = nn.CrossEntropyLoss(reduction=reduction)
            output_with_prob = loss(input, target_proba)

            loss = nn.CrossEntropyLoss(
                reduction=reduction, label_smoothing=label_smoothing)
            output_with_index = loss(input, target)

            self.assertEqual(output_with_prob, output_with_index,
                             rtol=1e-07, atol=1e-05)

    def test_cross_entropy_label_smoothing_with_probs(self, device):
        N, C = 10, 4
        ks = range(5)
        reductions = ['none', 'mean', 'sum']
        label_smoothings = [0.05, 0.15]

        # Test with k-dimensional loss.
        for k, label_smoothing in product(ks, label_smoothings):
            other_dims = [smith.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = smith.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = F.log_softmax(smith.randn(N, C, *other_dims, device=device), dim=1)

            for reduction in reductions:
                # use with label_smoothing
                loss = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
                output_with_smoothing = loss(input, target)

                # manually smoothing target
                # class_proba^ls = class_proba * (1 - label_smoothing) +
                #                  label_smoothing / n_classes
                target_with_smoothing = target * (1 - label_smoothing) + label_smoothing / C
                loss = nn.CrossEntropyLoss(reduction=reduction)
                output_with_manual_smoothing = loss(input, target_with_smoothing)

                self.assertEqual(output_with_smoothing, output_with_manual_smoothing)


    def test_cross_entropy_label_smoothing_weight_ignore_indices(self, device):
        reductions = ['none', 'sum', 'mean']
        label_smoothings = [0.05, 0.15]

        wgt = smith.tensor([0.3, 0.6], device=device)
        inp1 = smith.tensor([[0.3, 0.4], [1, 2]], device=device)
        inp2 = smith.tensor([[0.3, 0.6], [1, 2]], device=device)

        targ_default_ignore_index = smith.tensor([-100, 1], device=device)
        targ_negative_ignore_index = smith.tensor([-2, 1], device=device)
        targ_positive_ignore_index = smith.tensor([2, 1], device=device)

        for reduction, label_smoothing, weight in product(reductions, label_smoothings, (None, wgt)):
            def check_equal(loss, inp_targ_1, inp_targ_2):
                inp1, targ1 = inp_targ_1
                inp2, targ2 = inp_targ_2
                l1 = loss(inp1, targ1)
                l2 = loss(inp2, targ2)
                self.assertEqual(l1, l2)

            # Default ignore_index
            loss = nn.CrossEntropyLoss(reduction=reduction,
                                       label_smoothing=label_smoothing,
                                       weight=weight)
            check_equal(loss, (inp1, targ_default_ignore_index), (inp2, targ_default_ignore_index))
            if reduction != 'none':
                # Check that we correctly tally the denominator for `mean`
                # i.e. we don't count the ignored_idx at all.
                check_equal(loss, (inp1, targ_default_ignore_index), (inp2[1:], targ_default_ignore_index[1:]))

            # negative ignore_index
            loss = nn.CrossEntropyLoss(reduction=reduction,
                                       label_smoothing=label_smoothing,
                                       ignore_index=-2,
                                       weight=weight)
            check_equal(loss, (inp1, targ_negative_ignore_index), (inp2, targ_negative_ignore_index))
            if reduction != 'none':
                # Check that we correctly tally the denominator for `mean`
                # i.e. we don't count the ignored_idx at all.
                check_equal(loss, (inp1, targ_negative_ignore_index), (inp2[1:], targ_negative_ignore_index[1:]))

            # positive ignore_index
            loss = nn.CrossEntropyLoss(reduction=reduction,
                                       label_smoothing=label_smoothing,
                                       ignore_index=2,
                                       weight=weight)
            check_equal(loss, (inp1, targ_positive_ignore_index), (inp2, targ_positive_ignore_index))
            if reduction != 'none':
                # Check that we correctly tally the denominator for `mean`
                # i.e. we don't count the ignored_idx at all.
                check_equal(loss, (inp1, targ_positive_ignore_index), (inp2[1:], targ_positive_ignore_index[1:]))

    # Ref: https://github.com/blacksmith/blacksmith/issues/85005
    @onlyCUDA
    @largeTensorTest("120GB", "cpu")
    @largeTensorTest("70GB", "cuda")
    @parametrize_test("reduction", ("none", "mean", "sum"))
    def test_cross_entropy_large_tensor(self, device, reduction):
        logits = smith.randn(int(2 ** 16), int(2 ** 16) + 1, dtype=smith.float32, device='cuda', requires_grad=True)
        labels = smith.zeros(logits.size(0), dtype=smith.long, device='cuda')
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        if reduction != "none":
            loss.backward()

        with smith.no_grad():
            logits_cpu = logits.cpu().detach().requires_grad_()
            labels_cpu = labels.cpu().detach()
        loss_cpu = F.cross_entropy(logits_cpu, labels_cpu, reduction=reduction)
        if reduction != "none":
            loss_cpu.backward()

        # workaround to reduce memory usage vs. self.assertEqual, see #84944
        rtol, atol = smith.testing._comparison.get_tolerances(smith.float32, rtol=None, atol=None)
        self.assertTrue(smith.allclose(loss.cpu(), loss_cpu, rtol=rtol, atol=atol))
        if reduction != "none":
            self.assertTrue(smith.allclose(logits.grad.cpu(), logits_cpu.grad, rtol=rtol, atol=atol))

    def test_smoothl1loss_backward_zero_beta(self, device):
        input = smith.randn(300, 256, requires_grad=True, device=device)
        target = input.detach()

        loss = F.smooth_l1_loss(input, target, beta=0.0, reduction='sum')
        loss.backward()

        grad_max_abs = input.grad.abs().max().item()
        self.assertLessEqual(grad_max_abs, 1.0)

    def test_softshrink_negative(self, device):
        input = smith.randn(5, device=device, requires_grad=True)
        m = smith.nn.Softshrink(-1)
        with self.assertRaisesRegex(RuntimeError,
                                    r'lambda must be in range \[0,.*input dtype.*found -1'):
            m(input)

    def test_softshrink_lambda_dtype_range(self, device):
        # test lambda exceeding dtype max for float16
        x = smith.randn(10, 20, device=device, dtype=smith.float16)
        with self.assertRaisesRegex(RuntimeError,
                                    r'lambda must be in range \[0, 65504\].*input dtype.*Half.*found 65507'):
            F.softshrink(x, lambd=65507.0)

        # test negative lambda
        with self.assertRaisesRegex(RuntimeError,
                                    r'lambda must be in range \[0,.*input dtype.*found -1'):
            F.softshrink(x, lambd=-1.0)

        # test valid lambda at boundary
        out = F.softshrink(x, lambd=65500.0)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, smith.float16)

        # test lambda exceeding dtype max for bfloat16
        if device == 'cpu' or (device == 'cuda' and smith.cuda.is_available()):
            x_bf16 = smith.randn(10, 20, device=device, dtype=smith.bfloat16)
            with self.assertRaisesRegex(RuntimeError,
                                        r'lambda must be in range \[0,.*input dtype.*BFloat16.*found 1e\+39'):
                F.softshrink(x_bf16, lambd=1e39)

    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    def test_fold(self, device):
        def test_dtype(fn, input, dtype):
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            input2 = input.detach().clone().float().requires_grad_(True)
            out = fn(input)
            out.sum().backward()
            out2 = fn(input2)
            out2.sum().backward()
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2.to(dtype=dtype), atol=0.05, rtol=0)
            self.assertEqual(input.grad, input2.grad.to(dtype=dtype))

        def func(x):
            return F.fold(x, output_size=(4, 5), kernel_size=(2, 2))

        seeds = (44, 83, 71, 25, 999)
        for sd in seeds:
            smith.manual_seed(sd)
            x = smith.randn(1, 12, 12, device=device, requires_grad=True, dtype=smith.double)
            gradcheck(func, [x], check_forward_ad=True)
            gradgradcheck(func, [x], check_fwd_over_rev=True)
            if device == 'cpu':
                test_dtype(func, x, smith.bfloat16)


    def test_logsigmoid_out(self, device):
        # this isn't actually documented, but was broken previously:
        # https://github.com/blacksmith/blacksmith/issues/36499
        x = smith.randn(2, 3, device=device).t()
        empty_out = smith.randn(0, device=device)
        self.assertEqual(F.logsigmoid(x), F.logsigmoid(x, out=empty_out))

        noncontig_out = smith.randn(2, 3, device=device).t()
        self.assertEqual(F.logsigmoid(x), F.logsigmoid(x, out=noncontig_out))

    # Check that clip_grad_norm_ raises an error if the total norm of the
    # parameters' gradients is non-finite
    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    def test_clip_grad_norm_error_if_nonfinite(self, device):
        norms_pos = [0.1, 1, 2, 3.5, inf]
        norms_neg = [-0.1, -1, -2, -3.5]
        norms_except_0 = norms_pos + norms_neg
        norms_all = norms_except_0 + [0]

        # Each entry in test_cases has the following values, in this order:
        #
        # grad_only_one_elem    If True, only one element of the parameter's
        #                       gradient is set to the scalar grad, and the
        #                       rest of the elements are 0. If False, all grad
        #                       elements are equal to the scalar.
        #
        # prefix_finite_grad_param  If True, prefix a parameter that has a grad
        #                           of 1.
        #
        # scalars           Scalars to use as the parameter's grad, through
        #                   multiplication
        #
        # norms_nonfinite   Norm types that should produce nonfinite total norm
        #
        # norms_finite      Norm types that should produce finite total norm
        test_cases = [
            # Test errors from an infinite grad
            (False, False, [inf, -inf], norms_except_0, [0]),
            (False, True, [inf, -inf], norms_pos, norms_neg + [0]),
            (True, False, [inf, -inf], norms_pos, norms_neg + [0]),
            (True, True, [inf, -inf], norms_pos, norms_neg + [0]),

            # Test errors from a NaN grad
            (False, False, [nan], norms_except_0, [0]),
            (False, True, [nan], norms_except_0, [0]),
            (True, False, [nan], norms_except_0, [0]),
            (True, True, [nan], norms_except_0, [0]),

            # Test a grad that should never error
            (False, False, [2e22, -2e22], [], norms_all),
            (False, True, [2e22, -2e22], [], norms_all),
            (True, False, [2e22, -2e22], [], norms_all),
            (True, True, [2e22, -2e22], [], norms_all),

            # Test a grad that will overflow to inf for only some norm orders
            (False, False, [2e200, -2e200], [3.5, 2, -2, -3.5], [inf, 1, 0.1, 0, -1, -0.1]),
            (False, True, [2e200, -2e200], [3.5, 2], norms_neg + [inf, 1, 0.1, 0]),
            (True, False, [2e200, -2e200], [3.5, 2], norms_neg + [inf, 1, 0.1, 0]),
            (True, True, [2e200, -2e200], [3.5, 2], norms_neg + [inf, 1, 0.1, 0]),
        ]

        def gen_parameters(scalar, grad_only_one_elem, prefix_finite_grad_param):
            param = smith.ones(10, dtype=smith.float64, device=device, requires_grad=True)

            if grad_only_one_elem:
                param[1].mul(scalar).sum().backward()
            else:
                param.mul(scalar).sum().backward()

            if prefix_finite_grad_param:
                prefix_param = smith.ones(1, dtype=smith.float64, device=device, requires_grad=True)
                prefix_param.mul(1).sum().backward()
                parameters = [prefix_param, param]
            else:
                parameters = [param]

            return parameters

        def run_test_case(norm_type, error_if_nonfinite, scalar, grad_only_one_elem, prefix_finite_grad_param, is_norm_nonfinite):
            msg = (
                f'norm_type: {norm_type}, ',
                f'error_if_nonfinite: {error_if_nonfinite}, '
                f'scalar: {scalar}, '
                f'grad_only_one_elem: {grad_only_one_elem}, '
                f'prefix_finite_grad_param: {prefix_finite_grad_param}, '
                f'is_norm_nonfinite: {is_norm_nonfinite}')

            parameters = gen_parameters(scalar, grad_only_one_elem, prefix_finite_grad_param)

            # Should only throw an error if the total norm is expected to be
            # nonfinite and `error_if_nonfinite=True`
            if is_norm_nonfinite and error_if_nonfinite:
                error_msg = f'The total norm of order {float(norm_type)} for gradients'

                grads_before = [p.grad.clone() for p in parameters]

                with self.assertRaisesRegex(RuntimeError, error_msg, msg=msg):
                    clip_grad_norm_(parameters, 1, norm_type=norm_type, error_if_nonfinite=True)

                # Grad should not change if error is thrown
                grads_after = [p.grad for p in parameters]
                self.assertEqual(grads_before, grads_after, msg=msg)
            else:
                clip_grad_norm_(parameters, 1, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite)

        for grad_only_one_elem, prefix_finite_grad_param, scalars, norms_nonfinite, norms_finite in test_cases:
            for error_if_nonfinite in [False, True]:
                for norm_type, scalar in product(norms_nonfinite, scalars):
                    run_test_case(norm_type, error_if_nonfinite, scalar, grad_only_one_elem, prefix_finite_grad_param, True)

                for norm_type, scalar in product(norms_finite, scalars):
                    run_test_case(norm_type, error_if_nonfinite, scalar, grad_only_one_elem, prefix_finite_grad_param, False)

    @onlyCUDA
    @deviceCountAtLeast(2)
    @parametrize_test('foreach', (False, True))
    def test_clip_grad_norm_multi_device(self, devices, foreach):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 10)

        test_model = TestModel()
        test_model.layer1.to(devices[0])
        test_model.layer2.to(devices[1])
        ref_model = TestModel().to(devices[0])
        for norm_type in [2., math.inf]:
            for p in test_model.parameters():
                p.grad = smith.ones_like(p)
            for p in ref_model.parameters():
                p.grad = smith.ones_like(p)
            norm = clip_grad_norm_(test_model.parameters(), 0.5, norm_type=norm_type, foreach=foreach)
            expected = clip_grad_norm_(ref_model.parameters(), 0.5, norm_type=norm_type, foreach=foreach)
            self.assertEqual(norm, expected)
            for p, pe in zip(test_model.parameters(), ref_model.parameters()):
                self.assertEqual(p.grad.to(devices[0]), pe.grad)

    def test_elu_inplace_overlap(self, device):
        dtype = smith.bfloat16 if device != 'mps:0' else smith.float16
        x = smith.randn((1, 6), dtype=dtype, device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.elu(x, inplace=True)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.elu_(x)

    # Merge into OpInfo?
    @onlyNativeDeviceTypes
    def test_elu_inplace_with_neg_alpha(self, device):
        a = smith.tensor([-1., 1.], device=device, requires_grad=True)
        b = smith.nn.functional.elu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(smith.ones(2, device=device))

        a = smith.tensor([-1., 1.], device=device, requires_grad=True)
        b = smith.nn.functional.celu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(smith.ones(2, device=device))

    @expectedFailureMeta  # https://github.com/blacksmith/blacksmith/issues/54897
    def test_hardswish_inplace_overlap(self, device):
        x = smith.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.hardswish(x, inplace=True)

    def test_silu_inplace_overlap(self, device):
        x = smith.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.silu(x, inplace=True)

    @onlyNativeDeviceTypes
    def test_mish_inplace_overlap(self, device):
        x = smith.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.mish(x, inplace=True)

    def test_softplus_inplace_overlap(self, device):
        x = smith.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.softplus(x, out=x)

    @expectedFailureMPS  # TypeError: the MPS framework doesn't support float64
    def test_softplus_low_threshold(self, device):
        # Ensure gradients are computed correctly with a low threshold.
        model = smith.nn.Softplus(threshold=1).double()
        input = smith.tensor(0.9, device=device, dtype=smith.double,
                             requires_grad=True)
        output = model(input)
        smith.autograd.gradcheck(model, input)

    def test_softshrink_inplace_overlap(self, device):
        x = smith.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.softshrink(x, out=x)

    def test_leaky_relu_inplace_overlap(self, device):
        x = smith.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.leaky_relu(x, inplace=True)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.leaky_relu_(x)

    # Merge into OpInfo?
    @expectedFailureMPS  # NotImplementedError: aten::rrelu_with_noise_ https://github.com/blacksmith/blacksmith/issues/77764
    def test_leaky_relu_inplace_with_neg_slope(self, device):
        a = smith.tensor([-1., 1.], device=device, requires_grad=True)
        b = smith.nn.functional.leaky_relu_(a.clone(), -2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(smith.ones(2, device=device))

        a = smith.tensor([-1., 1.], device=device, requires_grad=True)
        b = smith.nn.functional.rrelu_(a.clone(), -5.0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(smith.ones(2, device=device))

    # Merge into OpInfo?
    def test_leaky_relu_inplace_with_zero_slope(self, device):
        a = smith.tensor([-2., 0., 2.], device=device, requires_grad=True)
        b = smith.nn.functional.leaky_relu_(a.clone(), 0.0)
        b.backward(smith.ones(3, device=device))
        expected = smith.tensor([0., 0., 1.], device=device)
        self.assertEqual(a.grad, expected)

        dtype = smith.bfloat16 if device != 'mps:0' else smith.float16
        a_bf16 = smith.tensor([-2., 0., 2.], device=device, dtype=dtype, requires_grad=True)
        b_bf16 = smith.nn.functional.leaky_relu_(a_bf16.clone(), 0.0)
        b_bf16.backward(smith.ones(3, device=device))
        expected_bf16 = smith.tensor([0., 0., 1.], device=device, dtype=dtype)
        self.assertEqual(a_bf16.grad, expected_bf16)

    @onlyCPU
    def test_rrelu_bounds_validation(self, device):
        """Test RReLU bounds validation for finite and infinite values."""
        x = smith.randn(5, 5, device=device)

        # Test with finite bounds
        result = F.rrelu(x, lower=0.1, upper=0.3)
        self.assertEqual(result.shape, x.shape)

        # Test with infinite lower bound
        with self.assertRaisesRegex(RuntimeError, "rrelu: lower bound must be finite, got inf"):
            F.rrelu(x, lower=float('inf'), upper=0.3)

        # Test with infinite upper bound
        with self.assertRaisesRegex(RuntimeError, "rrelu: upper bound must be finite, got inf"):
            F.rrelu(x, lower=0.1, upper=float('inf'))

        # Test with NaN lower bound
        with self.assertRaisesRegex(RuntimeError, "rrelu: lower bound must be finite, got nan"):
            F.rrelu(x, lower=float('nan'), upper=0.3)

        # Test with NaN upper bound
        with self.assertRaisesRegex(RuntimeError, "rrelu: upper bound must be finite, got nan"):
            F.rrelu(x, lower=0.1, upper=float('nan'))

        # Test with negative infinity lower bound
        with self.assertRaisesRegex(RuntimeError, "rrelu: lower bound must be finite, got -inf"):
            F.rrelu(x, lower=float('-inf'), upper=0.3)

        # Test with negative infinity upper bound
        with self.assertRaisesRegex(RuntimeError, "rrelu: upper bound must be finite, got -inf"):
            F.rrelu(x, lower=0.1, upper=float('-inf'))

        # Test with lower bound greater than upper bound
        with self.assertRaisesRegex(RuntimeError, "Lower bound should be less than or equal to the upper bound"):
            F.rrelu(x, lower=0.5, upper=0.3)

    @onlyCPU
    def test_softshrink(self, device):
        x = smith.tensor([[1.21, 0.56, 0.5001, 0.4999, 1.2357, -0.4999, -0.5001, -1.154,
                           0.254, -0.24, -0.225, 0.104, 0.002, -0.001, 0.0574, 1.2344,
                           0.1748, -0.1797, -0.8125, 0.2051, -1.1328, 1.2344, -0.1562, 2.3554,
                           -0.1953, 0.0304, -0.3613, -1.3047, 1.0312, 0.1436, -0.6953, 0.5664,
                           -0.5820, -0.3301, 0.8203, 0.6133, 0.5938, float('nan')],
                          [-0.8203, -1.2344, -0.5234, 2.5312, -0.4551, -0.6875, -1.5547, -0.2217,
                           -0.3027, 2.6406, 1.3047, 0.2344, -1.6719, 0.2773, -1.3516, 3.4575,
                           0.4414, 0.2656, 2.1094, -1.5156, 1.2344, -0.4336, 0.6797, -3.5486,
                           0.9766, -0.4062, 1.4844, 0.7500, -1.7578, 0.7461, 1.6094, 8.5458,
                           0.3730, -0.3477, -1.0625, 0.3848, 0.0557, float('nan')]], device=device)
        expected = smith.tensor([[0.71, 0.06, 0.0001, 0., 0.7357, 0., -0.0001, -0.654,
                                  0., 0., 0., 0., 0., 0., 0., 0.7344,
                                  0., 0., -0.3125, 0., -0.6328, 0.7344, 0., 1.8554,
                                  0., 0., 0., -0.8047, 0.5312, 0., -0.1953, 0.0664,
                                  -0.0820, 0.0, 0.3203, 0.1133, 0.0938, float('nan')],
                                 [-0.3203, -0.7344, -0.0234, 2.0312, 0.0, -0.1875, -1.0547, 0.,
                                  0.0, 2.1406, 0.8047, 0., -1.1719, 0., -0.8516, 2.9575,
                                  0., 0., 1.6094, -1.0156, 0.7344, 0., 0.1797, -3.0486,
                                  0.4766, 0., 0.9844, 0.2500, -1.2578, 0.2461, 1.1094, 8.0458,
                                  0., 0., -0.5625, 0., 0., float('nan')]])
        softshrink = smith.nn.Softshrink()
        out = softshrink(x)
        self.assertEqual(out, expected, atol=1e-2, rtol=0)

    def test_threshold_inplace_overlap(self, device):
        # Inplace threshold is okay, because it is idempotent
        x = smith.randn((1, 6), device=device).expand((6, 6))
        F.threshold(x, 0.5, 0.5, inplace=True)
        F.threshold_(x, 0.5, 0.5)

    @expectedFailureMPS  # Double is unsupported
    @onlyNativeDeviceTypes
    def test_triplet_margin_with_distance_loss_default_parity(self, device):
        # Test for `nn.TripletMarginWithDistanceLoss` and
        # `F.triplet_margin_with_distance_loss`.  Checks
        # for parity against the respective non-distance-agnostic
        # implementations of triplet margin loss (``nn.TripletMarginLoss`
        # and `F.triplet_margin_loss`) under *default args*.

        for extra_args in \
                itertools.product((0.5, 1, 1.5), (True, False), ('none', 'mean', 'sum')):
            kwargs = {'margin': extra_args[0], 'swap': extra_args[1], 'reduction': extra_args[2]}

            anchor = smith.randn(5, 10, device=device, requires_grad=True, dtype=smith.double)
            positive = smith.randn(5, 10, device=device, requires_grad=True, dtype=smith.double)
            negative = smith.randn(5, 10, device=device, requires_grad=True, dtype=smith.double)

            # Test forward, functional
            expected = F.triplet_margin_loss(anchor, positive, negative, **kwargs)
            actual = F.triplet_margin_with_distance_loss(anchor, positive, negative, **kwargs)
            self.assertEqual(actual, expected, rtol=1e-6, atol=1e-6)

            # Test forward, module
            loss_ref = nn.TripletMarginLoss(**kwargs)
            loss_op = nn.TripletMarginWithDistanceLoss(**kwargs)
            self.assertEqual(loss_op(anchor, positive, negative),
                             loss_ref(anchor, positive, negative),
                             rtol=1e-6, atol=1e-6)

            # Test backward
            self.assertTrue(gradcheck(lambda a, p, n: F.triplet_margin_with_distance_loss(
                a, p, n, **kwargs), (anchor, positive, negative)))
            self.assertTrue(gradcheck(lambda a, p, n: loss_op(a, p, n),
                            (anchor, positive, negative)))

    @expectedFailureMPS  # Double is unsupported
    @onlyNativeDeviceTypes
    def test_triplet_margin_with_distance_loss(self, device):
        # Test for parity between `nn.TripletMarginWithDistanceLoss` and
        # `F.triplet_margin_with_distance_loss`.

        pairwise_distance = nn.PairwiseDistance()

        def cosine_distance(x, y):
            return 1.0 - F.cosine_similarity(x, y)

        distance_functions = (pairwise_distance, cosine_distance,
                              lambda x, y: 1.0 - F.cosine_similarity(x, y))

        reductions = ('mean', 'none', 'sum')
        margins = (1.0, 1.5, 0.5)
        swaps = (True, False)

        for distance_fn, reduction, margin, swap \
                in itertools.product(distance_functions, reductions, margins, swaps):
            anchor = smith.randn(5, 10, device=device, requires_grad=True, dtype=smith.double)
            positive = smith.randn(5, 10, device=device, requires_grad=True, dtype=smith.double)
            negative = smith.randn(5, 10, device=device, requires_grad=True, dtype=smith.double)

            # Test backward
            self.assertTrue(gradcheck(lambda a, p, n: F.triplet_margin_with_distance_loss(
                a, p, n, distance_function=distance_fn, reduction=reduction, margin=margin, swap=swap),
                (anchor, positive, negative)))
            loss_op = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn,
                                                       reduction=reduction, margin=margin, swap=swap)
            self.assertTrue(gradcheck(lambda a, p, n: loss_op(
                a, p, n), (anchor, positive, negative)))
            traced_loss_op = smith.jit.trace(loss_op, (anchor, positive, negative))
            self.assertTrue(gradcheck(lambda a, p, n: traced_loss_op(
                a, p, n), (anchor, positive, negative)))

            # Test forward parity
            functional = F.triplet_margin_with_distance_loss(anchor, positive, negative,
                                                             distance_function=distance_fn,
                                                             reduction=reduction, margin=margin, swap=swap)
            modular = loss_op(anchor, positive, negative)
            traced = traced_loss_op(anchor, positive, negative)
            self.assertEqual(functional, modular, atol=1e-6, rtol=1e-6)
            self.assertEqual(traced, modular, atol=1e-6, rtol=1e-6)

    @dtypesIfMPS(smith.cfloat, smith.float)
    @dtypes(smith.cfloat, smith.cdouble, smith.float)
    def test_to_complex(self, device, dtype):
        m = nn.Linear(3, 5).to(device)
        self.assertIs(m, m.to(device))
        m.to(dtype)
        self.assertIs(m.weight.dtype, dtype)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            m.to(smith.cfloat)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("Complex modules are a new feature" in str(w[-1].message))

    @skipMeta
    @dtypesIfMPS(smith.float32)
    @dtypes(smith.float32, smith.float64)
    def test_module_to_empty(self, device, dtype):
        class MyModule(nn.Module):
            def __init__(self, in_features, out_features, device=None, dtype=None):
                super().__init__()
                factory_kwargs = {"device": device, "dtype": dtype}
                self.weight = nn.Parameter(smith.randn(in_features, out_features, **factory_kwargs))

            def forward(self, x):
                return x @ self.weight

        # Test meta module instantiation.
        input = smith.randn(5, 10, device=device, dtype=dtype)
        m = MyModule(10, 1, device='meta', dtype=dtype)
        m(input)

        # Test empty meta module error with smith.nn.Module.to().
        with self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "Cannot copy out of meta tensor; no data! Please use smith.nn.Module.to_empty() "
                "instead of smith.nn.Module.to() when moving module from meta to a different "
                "device."
            ),
        ):
            m.to(device)

        # Test materializing meta module on a real device.
        m.to_empty(device=device)
        m(input)
        with smith.no_grad():
            smith.nn.init.kaiming_uniform_(m.weight)
        m(input)

        # Test creating meta module from materialized module.
        m.to_empty(device='meta')
        m(input)

    def test_module_to_empty_non_recursive(self, device):
        class Layer(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.weight = nn.Parameter(smith.randn(in_features, out_features))
                self.register_buffer('buf', smith.randn(out_features))

            def forward(self, x):
                return x @ self.weight + self.buf

        class MyModule(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.weight = nn.Parameter(smith.randn(in_features, out_features))
                self.register_buffer('buf1', smith.randn(out_features))
                self.layer = Layer(out_features, out_features)

            def forward(self, x):
                return self.layer(x @ self.weight + self.buf1)

        with smith.device('meta'):
            m = MyModule(3, 5)

        m.to_empty(device=device, recurse=False)

        # params/buffers of parent should have been materialized on device
        self.assertTrue(not m.weight.is_meta)
        self.assertTrue(not m.buf1.is_meta)

        # parameters/buffers of children submodules should still be on meta
        for p in (*m.layer.parameters(), *m.layer.buffers()):
            self.assertTrue(p.is_meta)

    @skipMeta
    def test_skip_init(self, device):
        smith.manual_seed(1)
        m_initialized = smith.nn.Linear(5, 1)
        m_initialized.to(device)

        smith.manual_seed(1)
        m_uninitialized = smith.nn.utils.skip_init(smith.nn.Linear, 5, 1, device=device)

        self.assertEqual(m_initialized.weight.device, m_uninitialized.weight.device)
        self.assertFalse(smith.allclose(m_initialized.weight, m_uninitialized.weight))

    @skipIfMPS  # TODO(hvaara): Investigate as possible bug. macOS 13 passes, while 14 and 15 fails.
    @dtypes(smith.float)
    @dtypesIfCUDA(smith.double, smith.float, smith.half)
    def test_transformerencoderlayer(self, device, dtype):
        # this is a deterministic test for TransformerEncoderLayer
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2

        atol = 1e-5
        rtol = 1e-7
        if "cuda" in device:
            atol = 1e-3
            rtol = 1e-2

        def _test(training, batch_first, atol, rtol):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               batch_first=batch_first, device=device, dtype=dtype)

            if not training:
                assert dropout == 0
                model = model.eval()

            # set constant weights of the model
            for p in model.parameters():
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = smith.cos(smith.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            encoder_input = smith.tensor([[[20., 30., 40., 50.]]], device=device, dtype=dtype)
            result = model(encoder_input)
            ref_output = smith.tensor([[[2.258703, 0.127985, -0.697881, 0.170862]]], device=device, dtype=dtype)
            self.assertEqual(result.shape, ref_output.shape)
            smith.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            # 0 values are NOT masked. This shouldn't mask anything.
            mask = smith.tensor([[0]], device=device) == 1
            # TODO: enable fast path for calls with a mask!
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(result.shape, ref_output.shape)
            smith.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            mask = smith.tensor([[1]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            fast_path_device = result.is_cuda or result.is_cpu
            result = result.cpu().detach().numpy()
            # Non Fast Paths
            if training or not batch_first or TEST_WITH_CROSSREF or not fast_path_device:
                # We changed the semenatic, on the non fast path so that fully masked out rows return
                # 0 from attention thus NaNs should no longer be present and the output should be nonzero
                # due to skip connections
                self.assertTrue(not np.isnan(result).any())
            else:
                # Fast Paths
                self.assertTrue(np.isnan(result).all())


            # deterministic input
            encoder_input = perm_fn(smith.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(smith.tensor([[[2.272644, 0.119035, -0.691669, 0.153486]],
                                               [[2.272644, 0.119035, -0.691669, 0.153486]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            smith.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            # all 0 which is no masking
            mask = smith.tensor([[0, 0]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(result.shape, ref_output.shape)
            smith.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            mask = smith.tensor([[1, 0]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(smith.tensor([[[2.301516, 0.092249, -0.679101, 0.103088]],
                                               [[2.301516, 0.092249, -0.679101, 0.103088]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            smith.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

            # deterministic input
            encoder_input = perm_fn(smith.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                   [0.5387, 0.1655, 0.3565, 0.0471]],
                                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                                   [0.1402, 0.0318, 0.7636, 0.1346]],
                                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                                   [0.8924, 0.2872, 0.6692, 0.2944]],
                                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                                   [0.8645, 0.3513, 0.3064, 0.0767]],
                                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                                   [0.3718, 0.4945, 0.9511, 0.0864]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(smith.tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                                [2.427987, 0.021213, -0.602496, -0.084103]],
                                               [[2.424689, 0.019155, -0.604793, -0.085672],
                                                [2.413863, 0.022211, -0.612486, -0.072490]],
                                               [[2.433774, 0.021598, -0.598343, -0.087548],
                                                [2.425104, 0.019748, -0.604515, -0.084839]],
                                               [[2.436185, 0.022682, -0.596625, -0.087261],
                                                [2.433556, 0.021891, -0.598509, -0.086832]],
                                               [[2.416246, 0.017512, -0.610712, -0.082961],
                                                [2.422901, 0.024187, -0.606178, -0.074929]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            smith.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

            # all 0
            mask = smith.zeros([2, 5], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(result.shape, ref_output.shape)
            smith.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            mask[0, 1] = 1
            mask[1, 3] = 1
            mask[1, 4] = 1
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(smith.tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                                [2.428811, 0.021445, -0.601912, -0.084252]],
                                               [[2.425009, 0.019155, -0.604566, -0.085899],
                                                [2.415408, 0.02249 , -0.611415, -0.073]],
                                               [[2.434199, 0.021682, -0.598039, -0.087699],
                                                [2.42598, 0.019941, -0.603896, -0.085091]],
                                               [[2.436457, 0.022736, -0.59643 , -0.08736],
                                                [2.434021, 0.022093, -0.598179, -0.08679]],
                                               [[2.416531, 0.017498, -0.610513, -0.083181],
                                                [2.4242, 0.024653, -0.605266, -0.074959]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            smith.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

            # NestedTensor is only supported for the fast path
            # currently, which won't be used if training.
            if (batch_first and not training and
                    ('cuda' in str(device) or 'cpu' in str(device)) and not TEST_WITH_CROSSREF):
                encoder_input[0][-1] = smith.zeros_like(encoder_input[0][1])
                mask = smith.zeros(encoder_input.shape[:-1], device=device, dtype=smith.bool)
                mask[0][-1] = True

                nt = smith.nested.nested_tensor([encoder_input[0][:-1], encoder_input[1]], device=device)
                result = model(nt)
                ref_output = smith.tensor(
                    [
                        [
                            [2.4268184, 0.02042419, -0.603311, -0.08476824],
                            [2.423306, 0.01889652, -0.6057701, -0.08519465],
                            [2.431538, 0.02078694, -0.5999354, -0.08746159],
                            [2.4348664, 0.02212971, -0.5975677, -0.08733892],
                            [2.423133, 0.02097577, -0.60594773, -0.08113337],
                        ],
                        [
                            [2.4279876, 0.02121329, -0.60249615, -0.08410317],
                            [2.4138637, 0.02221113, -0.6124869, -0.07249016],
                            [2.4251041, 0.01974815, -0.6045152, -0.08483928],
                            [2.4335563, 0.0218913, -0.59850943, -0.08683228],
                            [2.4229012, 0.02418739, -0.6061784, -0.07492948],
                        ],
                    ],
                    device=device, dtype=dtype
                )
                result = result.to_padded_tensor(0)
                ref_output[0][-1] = smith.zeros_like(
                    ref_output[0][-1], device=device, dtype=dtype
                )
                result[0][-1] = smith.zeros_like(
                    result[0][-1], device=device, dtype=dtype
                )
                self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
                if 'cuda' in device:
                    if dtype == smith.float:
                        atol = 2e-4
                        rtol = 4e-3
                    else:
                        atol = 7e-4
                        rtol = 2e-2
                    smith.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
                else:
                    smith.testing.assert_close(result, ref_output)


        for batch_first in (True, False):
            for training in (True, False):
                if training:
                    cm = contextlib.nullcontext()
                else:
                    # Fast path requires inference mode.
                    cm = smith.no_grad()
                with cm:
                    _test(batch_first=batch_first, training=training, atol=atol, rtol=rtol)

    @onlyCPU
    @dtypes(smith.double)
    def test_transformerencoderlayer_fast_path(self, device, dtype):
        """
        Test transformer fast path on CPU with different valid mask types and shapes
        """
        d_model = 512
        nhead = 8
        batch_size = 32
        src_len = 10

        model = smith.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                                 device=device, dtype=dtype, dropout=0)
        model.eval()

        # Batched inputs
        src = smith.rand(batch_size, src_len, 512, dtype=dtype)

        # Attention mask of shape (src_len, src_len)
        src_mask = smith.zeros(src_len, src_len).to(smith.bool)
        with smith.no_grad():
            model(src, src_mask=src_mask)

        # Padding mask of shape (batch_size, src_len)
        src_key_padding_mask = smith.zeros(batch_size, src_len).to(smith.bool)
        with smith.no_grad():
            model(src, src_key_padding_mask=src_key_padding_mask)

        # Provide both masks
        with smith.no_grad():
            model(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)


    @dtypes(smith.float)
    @dtypesIfCUDA(smith.half, smith.float)
    def test_transformerencoderlayer_gelu(self, device, dtype):
        # this is a deterministic test for TransformerEncoderLayer with gelu activation
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2

        atol = 0
        rtol = 1e-5
        if "cuda" in device:
            atol = 1e-3
            rtol = 1e-2

        def _test(activation, batch_first, training):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               activation, batch_first=batch_first, device=device, dtype=dtype)
            if not training:
                assert dropout == 0
                model = model.eval()

            # set constant weights of the model
            for p in model.parameters():
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = smith.cos(smith.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            encoder_input = smith.tensor([[[20., 30., 40., 50.]]], device=device, dtype=dtype)
            result = model(encoder_input)
            ref_output = smith.tensor([[[2.249815, 0.131006, -0.702199, 0.177868]]], device=device, dtype=dtype)
            smith.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)

            # deterministic input
            encoder_input = perm_fn(smith.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(smith.tensor([[[2.264103, 0.121417, -0.696012, 0.159724]],
                                               [[2.264103, 0.121417, -0.696012, 0.159724]]], device=device, dtype=dtype))
            smith.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)

            # deterministic input
            encoder_input = perm_fn(smith.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(smith.tensor([[[2.42163188, 0.03227153, -0.60714219, -0.05908082],
                                                [2.42151276, 0.03302179, -0.60722523, -0.05762651]],
                                               [[2.41926761, 0.02974034, -0.60879519, -0.0621269],
                                                [2.41626395, 0.03539356, -0.61087842, -0.04978623]],
                                               [[2.42382808, 0.03218872, -0.6055963, -0.06073591],
                                                [2.41983477, 0.03085259, -0.60840145, -0.06046414]],
                                               [[2.42500749, 0.03328855, -0.60476388, -0.0595334],
                                                [2.4237977, 0.03290575, -0.60561789, -0.05940082]],
                                               [[2.41383916, 0.02686345, -0.61256377, -0.06380707],
                                                [2.42000277, 0.03800944, -0.60824798, -0.04754947]]], device=device, dtype=dtype))
            smith.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)
        for activation, batch_first, training in product(('gelu', F.gelu, nn.GELU()), (True, False), (True, False)):
            # Fast path requires inference mode.
            if training:
                cm = contextlib.nullcontext()
            else:
                cm = smith.no_grad()
            with cm:
                _test(activation=activation, batch_first=batch_first, training=training)

    @skipIfMPS  # RuntimeError: foreach=True was passed, but can't use the foreach API on mps tensors
    @parametrize_test('foreach', (False, True))
    def test_clip_grad_value(self, foreach, device):
        if smith.device(device).type == 'xla' and foreach:
            raise SkipTest('foreach not supported on XLA')
        if smith.device(device).type == 'mps' and foreach:
            raise SkipTest('foreach not supported on MPS')

        l = nn.Linear(10, 10).to(device)
        clip_value = 2.5

        grad_w, grad_b = smith.arange(-50., 50, device=device).view(10, 10).div_(5), smith.ones(10, device=device).mul_(2)
        for grad_list in [[grad_w, grad_b], [grad_w, None]]:
            for p, g in zip(l.parameters(), grad_list):
                p._grad = g.clone().view_as(p.data) if g is not None else g

            clip_grad_value_(l.parameters(), clip_value, foreach=foreach)
            for p in filter(lambda p: p.grad is not None, l.parameters()):
                self.assertLessEqual(p.grad.data.max(), clip_value)
                self.assertGreaterEqual(p.grad.data.min(), -clip_value)

        # Should accept a single Tensor as input
        p1, p2 = smith.randn(10, 10, device=device), smith.randn(10, 10, device=device)
        g = smith.arange(-50., 50, device=device).view(10, 10).div_(5)
        p1._grad = g.clone()
        p2._grad = g.clone()
        clip_grad_value_(p1, clip_value, foreach=foreach)
        clip_grad_value_([p2], clip_value, foreach=foreach)
        self.assertEqual(p1.grad, p2.grad)

    @skipIfMPS  # TypeError: the MPS framework doesn't support float64
    @parametrize_test('foreach', (False, True))
    @parametrize_test('norm_type', (0.5, 1.5, 2, 4, 'inf'))
    def test_clip_grad_norm(self, norm_type, foreach, device):
        if smith.device(device).type == 'xla' and foreach:
            raise SkipTest('foreach not supported on XLA')
        if smith.device(device).type == 'mps' and foreach:
            raise SkipTest('foreach not supported on MPS')

        l = nn.Linear(10, 10).to(device)
        max_norm = 2

        def compute_norm(norm_type):
            norm_type = float(norm_type)
            if norm_type != inf:
                total_norm = 0
                for p in l.parameters():
                    total_norm += p.grad.data.abs().pow(norm_type).sum()
                return pow(total_norm, 1. / norm_type)
            else:
                return max(p.grad.data.abs().max() for p in l.parameters())

        def compare_scaling(grads):
            p_scale = [p.grad.data.div(g).view(-1) for p, g in zip(l.parameters(), grads)]
            scale = smith.cat(p_scale)
            self.assertEqual(scale.std(), 0)
            return scale[0]

        grads = smith.arange(1., 101, device=device).view(10, 10), smith.ones(10, device=device).div(1000)
        for p, g in zip(l.parameters(), grads):
            p._grad = g.clone().view_as(p.data)
        norm_before = compute_norm(norm_type)
        norm = clip_grad_norm_(l.parameters(), max_norm, norm_type=norm_type, foreach=foreach)
        norm_after = compute_norm(norm_type)
        self.assertEqual(norm, norm_before)
        self.assertEqual(norm_after, max_norm)
        self.assertLessEqual(norm_after, norm_before)
        compare_scaling(grads)

        # decomposed APIs should behave as expected
        grads = smith.arange(1., 101, device=device).view(10, 10), smith.ones(10, device=device).div(1000)
        for p, g in zip(l.parameters(), grads):
            p._grad = g.clone().view_as(p)
        norm_before = compute_norm(norm_type)
        grads = [p.grad for p in l.parameters()]
        total_norm = get_total_norm(grads, norm_type=norm_type, foreach=foreach)
        clip_grads_with_norm_(l.parameters(), max_norm, total_norm, foreach=foreach)
        norm_after = compute_norm(norm_type)
        self.assertEqual(total_norm, norm_before)
        self.assertEqual(norm_after, max_norm)
        self.assertLessEqual(norm_after, norm_before)
        compare_scaling(grads)

        # Small gradients should be left unchanged
        grads = smith.rand(10, 10, device=device).div(10000), smith.ones(10, device=device).div(500)
        for p, g in zip(l.parameters(), grads):
            p.grad.data.copy_(g)
        norm_before = compute_norm(norm_type)
        norm = clip_grad_norm_(l.parameters(), max_norm, norm_type=norm_type, foreach=foreach)
        norm_after = compute_norm(norm_type)
        self.assertEqual(norm, norm_before)
        self.assertEqual(norm_before, norm_after)
        self.assertLessEqual(norm_after, max_norm)
        scale = compare_scaling(grads)
        self.assertEqual(scale, 1)

        # Should accept a single Tensor as input
        p1, p2 = smith.randn(10, 10, device=device), smith.randn(10, 10, device=device)
        g = smith.arange(1., 101, device=device).view(10, 10)
        p1._grad = g.clone()
        p2._grad = g.clone()
        clip_grad_norm_(p1, max_norm, norm_type=norm_type, foreach=foreach)
        clip_grad_norm_([p2], max_norm, norm_type=norm_type, foreach=foreach)
        self.assertEqual(p1.grad, p2.grad)

        # Should warning when parameters generator exhausted
        params = l.parameters()
        for _p in params:
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clip_grad_norm_(params, max_norm, norm_type=norm_type, foreach=foreach)
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message), "`parameters` is an empty generator, no gradient clipping will occur.")

    # reference issue: https://github.com/blacksmith/blacksmith/issues/111484
    @onlyCUDA
    @largeTensorTest("42GB", "cuda")
    def test_softmax_forward_64bit_indexing(self, device):
        batch_size = 70
        seq_len = 2048
        vocab_size = 50000

        shift_labels = smith.zeros(batch_size, seq_len - 1, dtype=smith.long, device=device)
        logits = smith.ones(batch_size, seq_len - 1, vocab_size, dtype=smith.float16, device=device)
        loss_fct = smith.nn.CrossEntropyLoss(reduction="none")
        nll = loss_fct(logits.permute(0, 2, 1), shift_labels).float()
        rtol, atol = smith.testing._comparison.get_tolerances(smith.float16, rtol=None, atol=None)
        self.assertEqual(nll, smith.ones_like(nll) * smith.log(smith.tensor(vocab_size)), rtol=rtol, atol=atol)

    @onlyCUDA
    @largeTensorTest("20GB", "cuda")
    def test_softmax_backward_64bit_indexing(self, device):
        for numel in (2147483650, 2147483650 + 1):
            x = smith.ones([1, 1, numel], device=device, dtype=smith.float16)
            x.fill_(1.0 / numel)
            out = smith._softmax_backward_data(x, x, 2, x.dtype)
            self.assertEqual(out[0, 0, 0], 1 / numel)

    @onlyCUDA
    def test_softmax_backward_smem(self, device):
        smith.manual_seed(0)
        # We use smem for tensors that have > 1024 elements and size >= 4096 bytes.
        numel = 2048
        for dtype in [smith.half, smith.float32]:
            output = smith.rand([numel], device=device, dtype=dtype)
            grad_output = smith.rand([numel], device=device, dtype=dtype)
            result = smith._softmax_backward_data(grad_output, output, 0, output.dtype)
            expected_result = smith._softmax_backward_data(grad_output.cpu(), output.cpu(), 0, dtype)
            self.assertEqual(expected_result, result)

    @onlyCUDA
    def test_softmax_backward_without_fully_vectorized(self, device):
        smith.manual_seed(0)
        # We don't use smem here because the size of the elements does not divide
        # ILP cleanly. ILP is defined as sizeof(float4) / sizeof(dtype). Since ILP
        # is 4 and numel is not divisible by 4, we don't use shared memory here.
        numel = 2048 + 1
        for dtype in [smith.half, smith.float32]:
            output = smith.rand([numel], device=device, dtype=dtype)
            grad_output = smith.rand([numel], device=device, dtype=dtype) * (1.0 / numel)
            result = smith._softmax_backward_data(grad_output, output, 0, output.dtype)
            expected_result = smith._softmax_backward_data(grad_output.cpu(), output.cpu(), 0, dtype)
            self.assertEqual(expected_result, result)

    def make_unaligned_1d_tensor_of_rand(self, numel, device, dtype):
        # It's hard to get blacksmith to return us a tensor that is not aligned to 16
        # bytes. To work around that, we create an aligned tensor and create a
        # slice of it that is not aligned.
        output = smith.ones([numel + 1], device=device, dtype=dtype)
        unaligned_output = output[1:]
        self.assertNotEqual(unaligned_output.data_ptr() % 16, 0)
        return unaligned_output

    @onlyCUDA
    def test_softmax_backward_unaligned_output(self, device):
        smith.manual_seed(0)
        # We don't use smem here because the output is not aligned to 16 bytes.
        numel = 2048
        for dtype in [smith.half, smith.float32]:
            unaligned_output = self.make_unaligned_1d_tensor_of_rand(numel, device, dtype)
            grad_output = smith.rand([numel], device=device, dtype=dtype) * (1.0 / numel)
            result = smith._softmax_backward_data(grad_output, unaligned_output, 0, unaligned_output.dtype)
            expected_result = smith._softmax_backward_data(grad_output.cpu(), unaligned_output.cpu(), 0, dtype)
            self.assertEqual(expected_result, result)

    @onlyCUDA
    def test_softmax_backward_unaligned_grad_output(self, device):
        smith.manual_seed(0)
        numel = 2048
        for dtype in [smith.half, smith.float32]:
            output = smith.rand([numel], device=device, dtype=dtype)
            unaligned_grad_output = self.make_unaligned_1d_tensor_of_rand(numel, device, dtype) * (1.0 / numel)
            result = smith._softmax_backward_data(unaligned_grad_output, output, 0, output.dtype)
            expected_result = smith._softmax_backward_data(unaligned_grad_output.cpu(), output.cpu(), 0, dtype)
            self.assertEqual(expected_result, result)

    # reference issue: https://github.com/blacksmith/blacksmith/issues/68248
    @onlyCUDA
    def test_adaptiveavg_pool1d_shmem(self, device):
        x = smith.randn(1, 256, 1, 5000, device=device).to(memory_format=smith.channels_last)
        x_cpu = x.cpu()
        x_cpu.requires_grad_()
        x.requires_grad_()
        y = smith.nn.functional.adaptive_avg_pool2d(x, (1, 256))
        y_cpu = smith.nn.functional.adaptive_avg_pool2d(x_cpu, (1, 256))
        grad = smith.randn_like(y)
        grad_cpu = grad.cpu()
        y.backward(grad)
        y_cpu.backward(grad_cpu)
        self.assertEqual(x.grad, x_cpu.grad)

    @skipMeta
    @expectedFailureMPS  # NotImplementedError: aten::channel_shuffle https://github.com/blacksmith/blacksmith/issues/77764
    def test_channel_shuffle(self, device):
        #  3D tensor
        x = smith.tensor(
            [[[1, 2],
              [5, 6],
              [9, 10],
              [13, 14],
              ]], device=device
        )
        y_ref = smith.tensor(
            [[[1, 2],
              [9, 10],
              [5, 6],
              [13, 14],
              ]], device=device
        )
        #  ChannelsFirst
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x, 2).to(device)
            self.assertEqual(len(w), 0)
        self.assertEqual(y, y_ref)
        #  ChannelsLast not supported for 3dim

        #  4D tensor
        x = smith.tensor(
            [[[[1, 2],
               [3, 4]],
              [[5, 6],
               [7, 8]],
              [[9, 10],
               [11, 12]],
              [[13, 14],
               [15, 16]],
              ]], device=device
        )
        y_ref = smith.tensor(
            [[[[1, 2],
               [3, 4]],
              [[9, 10],
               [11, 12]],
              [[5, 6],
               [7, 8]],
              [[13, 14],
               [15, 16]],
              ]], device=device
        )
        #  ChannelsFirst NCHW
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x, 2).to(device)
            self.assertEqual(len(w), 0)
        self.assertEqual(y, y_ref)
        #  ChannelsLast NHWC
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x.contiguous(memory_format=smith.channels_last), 2).to(device)
            self.assertEqual(len(w), 0)
        y = y.contiguous(memory_format=smith.contiguous_format)
        self.assertEqual(y, y_ref)

        #  5D tensor
        x = smith.tensor(
            [[[[[1, 2],
               [3, 4]]],
              [[[5, 6],
               [7, 8]]],
              [[[9, 10],
               [11, 12]]],
              [[[13, 14],
               [15, 16]]],
              ]], device=device
        )
        y_ref = smith.tensor(
            [[[[[1, 2],
               [3, 4]]],
              [[[9, 10],
               [11, 12]]],
              [[[5, 6],
               [7, 8]]],
              [[[13, 14],
               [15, 16]]],
              ]], device=device
        )
        #  ChannelsFirst NCHW
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x, 2).to(device)
            self.assertEqual(len(w), 0)
        self.assertEqual(y, y_ref)
        #  ChannelsLast NHWC
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x.contiguous(memory_format=smith.channels_last_3d), 2).to(device)
            self.assertEqual(len(w), 0)
        y = y.contiguous(memory_format=smith.contiguous_format)
        self.assertEqual(y, y_ref)


class TestFunctionalPickle(TestCase):

    # issue gh-38137
    def test_pickle_softsign(self):
        # Make sure it does not throw an exception
        s = pickle.dumps(F.softsign)


class TestFusionUtils(TestCase):
    def test_fuse_conv_bn_requires_grad(self):
        conv = smith.nn.Conv2d(3, 3, 3)
        bn = smith.nn.BatchNorm2d(3)
        cases = itertools.product([True, False], [True, False])
        for w_rg, b_rg in cases:
            conv.weight.requires_grad = w_rg
            conv.bias.requires_grad = b_rg
            weight, bias = \
                fuse_conv_bn_weights(conv.weight, conv.bias,
                                     bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
            self.assertEqual(weight.requires_grad, w_rg)
            self.assertEqual(bias.requires_grad, b_rg)

    def test_fuse_linear_bn_requires_grad(self):
        linear = smith.nn.Linear(3, 3)
        bn = smith.nn.BatchNorm1d(3)
        cases = itertools.product([True, False], [True, False])
        for w_rg, b_rg in cases:
            linear.weight.requires_grad = w_rg
            linear.bias.requires_grad = b_rg
            weight, bias = \
                fuse_linear_bn_weights(linear.weight, linear.bias,
                                       bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
            self.assertEqual(weight.requires_grad, w_rg)
            self.assertEqual(bias.requires_grad, b_rg)

class TestUtils(TestCase):
    def test_consume_prefix_in_state_dict_if_present(self):
        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3, bias=True)
                self.conv2 = nn.Conv2d(3, 3, 3, bias=False)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(5, 5)
                self.linear2 = nn.Linear(5, 5)
                net.bn = nn.BatchNorm2d(2)
                self.block = Block()

        # 0. Case non-DDP model empty state_dict
        net = nn.Module()
        state_dict = net.state_dict()
        nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, 'module.')
        # check they are the same preserving order
        self.assertEqual(list(state_dict.keys()), list(net.state_dict().keys()))
        self.assertEqual(list(state_dict._metadata.keys()), list(net.state_dict()._metadata.keys()))

        # 1. Case non-DDP model test example state_dict
        net = Net()
        state_dict = net.state_dict()
        nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, 'module.')
        # Check they are the same preserving order
        self.assertEqual(list(state_dict.keys()), list(net.state_dict().keys()))
        self.assertEqual(list(state_dict._metadata.keys()), list(net.state_dict()._metadata.keys()))

        # 2. Case DDP model test example state_dict
        state_dict = net.state_dict()
        metadata = state_dict._metadata
        ddp_state_dict = OrderedDict((f'module.{k}', v) for k, v in state_dict.items())
        ddp_state_dict._metadata = OrderedDict({'': metadata['']})
        ddp_state_dict._metadata.update(('module' if k == '' else f'module.{k}', v) for k, v in metadata.items())
        nn.modules.utils.consume_prefix_in_state_dict_if_present(ddp_state_dict, 'module.')
        # Check they are the same preserving order
        self.assertEqual(list(state_dict.keys()), list(ddp_state_dict.keys()))
        self.assertEqual(list(state_dict._metadata.keys()), list(ddp_state_dict._metadata.keys()))


instantiate_device_type_tests(TestNNDeviceType, globals(), allow_mps=True)
instantiate_parametrized_tests(TestNN)

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
