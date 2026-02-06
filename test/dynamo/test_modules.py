# Owner(s): ["module: dynamo"]
# ruff: noqa: F841

import collections
import copy
import itertools
import os
import tempfile
import traceback
import types
import unittest
from copy import deepcopy
from functools import partial
from typing import NamedTuple
from unittest.mock import patch

import smith
import smith._dynamo.test_case
import smith._dynamo.testing
import smith.nn.functional as F
from smith._dynamo.debug_utils import same_two_models
from smith._dynamo.eval_frame import unsupported
from smith._dynamo.mutation_guard import GenerationTracker
from smith._dynamo.testing import expectedFailureDynamic, same
from smith._dynamo.utils import ifdynstaticdefault
from smith._dynamo.variables.smith_function import TensorWithTFOverrideVariable
from smith.nn.modules.lazy import LazyModuleMixin
from smith.nn.parameter import Parameter, UninitializedParameter
from smith.testing._internal.common_device_type import instantiate_device_type_tests
from smith.testing._internal.common_utils import skipIfHpu


try:
    from . import test_functions
except ImportError:
    import test_functions


_variable = 0
_variable1 = 0


def update_global():
    global _variable, _variable1
    _variable += 1
    _variable1 += 1


class BasicModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)
        self.scale = smith.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class FnMember(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)
        self.activation = F.relu

    def forward(self, x):
        x = self.linear1(x)
        if self.activation:
            x = self.activation(x)
        return x


class FnMemberCmp(smith.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)
        self.activation = activation

    def forward(self, x):
        x = self.linear1(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.activation is None:
            x = smith.sigmoid(x)
        return x


class SubmoduleExample(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = smith.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x * self.scale


class IsTrainingCheck(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)
        self.linear2 = smith.nn.Linear(10, 10)
        self.train(True)

    def forward(self, x):
        if self.training:
            mod = self.linear1
        else:
            mod = self.linear2
        return F.relu(mod(x))


class IsEvalCheck(IsTrainingCheck):
    def __init__(self) -> None:
        super().__init__()
        self.train(False)


class ModuleMethodCall(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = smith.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        return x * self.scale

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        x2 = self.call_and_scale(self.layer2, x)
        return x1 + x2


class UnsupportedMethodCall(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = smith.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        x = x * self.scale
        return unsupported(x, x)

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        return x + x1


class UnsupportedModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = smith.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x) * self.scale
        return unsupported(x, x)


class UnsupportedModuleCall(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = UnsupportedModule()

    def forward(self, x):
        return 1 + self.mod(x * 1.5)


class ModuleWithStaticForward(smith.nn.Module):
    @staticmethod
    def forward(x):
        return x * smith.sigmoid(x)


class ModuleCallModuleWithStaticForward(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = ModuleWithStaticForward()

    def forward(self, x):
        return self.mod(x)


class ModuleStaticMethodCall(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = smith.randn(1, 10)

    @staticmethod
    def call_and_scale(scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleClassMethodCall(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = smith.randn(1, 10)

    @classmethod
    def call_and_scale(cls, scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleProperty(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = smith.randn(1, 10)

    @property
    def scale_alias(self):
        return self.scale

    def forward(self, x):
        return x * self.scale_alias


class NestedModuleList(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = smith.nn.ModuleList([])
        for _ in range(3):
            self.layers.append(
                smith.nn.ModuleList(
                    [
                        smith.nn.Linear(10, 10),
                        smith.nn.ReLU(),
                    ]
                )
            )

    def forward(self, x):
        for layer, act in self.layers:
            x = act(layer(x))
        return x


class ConstLoop(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)
        self.count = 3

    def forward(self, x):
        for _ in range(self.count):
            x = smith.sigmoid(self.linear1(x))
        return x


class ViaModuleCall(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)

    def forward(self, x):
        return test_functions.constant3(smith.sigmoid(self.linear1(x)), x)


class IsNoneLayer(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = smith.nn.Linear(10, 10)
        self.layer2 = None
        self.train(True)

    def forward(self, x):
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        return x


class LayerList(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [
            smith.nn.Linear(10, 10),
            smith.nn.ReLU(),
            smith.nn.Linear(10, 10),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModuleList(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = smith.nn.ModuleList(
            [
                smith.nn.Linear(10, 10),
                smith.nn.ReLU(),
                smith.nn.Linear(10, 10),
                smith.nn.ReLU(),
            ]
        )

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        for layer in self.layers:
            x = layer(x)

        for layer, val in zip(self.layers, (x, x, x, x)):
            x = layer(x) + val

        for layer, val in zip(self.layers, (1, 2, 3, 4)):
            x = layer(x) + val

        for idx, layer in enumerate(self.layers):
            x = layer(x) * idx

        for idx, layer in enumerate(self.layers[::-1]):
            x = layer(x) * idx

        return x


class CustomGetItemModuleList(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = smith.nn.ModuleList(
            [
                smith.nn.Linear(10, 10),
                smith.nn.ReLU(),
                smith.nn.Linear(10, 10),
                smith.nn.ReLU(),
            ]
        )

    def __getitem__(self, idx: int):
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def forward(self, x):
        for i in range(len(self)):
            x = self[i](x)

        return x


class ModuleDict(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = smith.nn.ModuleDict(
            {
                "0": smith.nn.Linear(10, 10),
            }
        )

    def forward(self, x):
        # TODO(future PR): handle more logic
        x = self.layers["0"](x)
        return x


class ParameterDict(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = smith.nn.ParameterDict(
            {
                "0": smith.nn.Parameter(smith.randn(10, 10)),
            }
        )

    def forward(self, x):
        x = self.layers["0"].mm(x)
        return x


class CustomGetItemParameterDict(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = smith.nn.ParameterDict(
            {
                "0": smith.nn.Parameter(smith.randn(10, 10)),
            }
        )

    def __getitem__(self, key: str) -> smith.nn.Module:
        return self.layers[key]

    def forward(self, x):
        x = self["0"].mm(x)
        return x


class CustomGetItemModuleDict(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = smith.nn.ModuleDict(
            {
                "0": smith.nn.Linear(10, 10),
            }
        )

    def __getitem__(self, key: str) -> smith.nn.Module:
        return self.layers[key]

    def forward(self, x):
        x = self["0"](x)
        return x


class TensorList(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = (
            smith.randn((1, 10)),
            smith.randn((10, 1)),
            smith.randn((1, 10)),
            smith.randn((10, 1)),
        )

    def forward(self, x):
        for layer in self.layers:
            x = x * layer
        return x


class Children(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = smith.nn.Linear(10, 10)
        self.l2 = smith.nn.ReLU()
        self.l3 = smith.nn.Linear(10, 10)
        self.l4 = smith.nn.ReLU()

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class NamedChildren(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = smith.nn.Linear(10, 10)
        self.l2 = smith.nn.ReLU()
        self.l3 = smith.nn.Linear(10, 10)
        self.l4 = smith.nn.ReLU()

    def forward(self, x):
        for _, block in self.named_children():
            x = block(x)
        return x


class IntArg(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = smith.nn.Linear(10, 10)

    def forward(self, x, offset=1):
        x = F.relu(self.layer1(x)) + offset
        return x


class Seq(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = smith.nn.Sequential(
            smith.nn.Linear(10, 10),
            smith.nn.ReLU(),
            smith.nn.Linear(10, 10),
            smith.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Cfg:
    def __init__(self) -> None:
        self.val = 0.5
        self.count = 3


class CfgModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = Cfg()
        self.layer = smith.nn.Linear(10, 10)

    def forward(self, x):
        for _ in range(self.cfg.count):
            x = self.layer(x + self.cfg.val)
        return x


class StringMember(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)
        self.mode = "some_string"

    def forward(self, x):
        if self.mode == "some_string":
            return F.relu(self.linear1(x))


class _Block(smith.nn.Module):
    def forward(self, x):
        return 1.5 * smith.cat(x, 1)


class _DenseBlock(smith.nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        features = [init_features]
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return smith.cat(features, 1)


class DenseNetBlocks(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = _DenseBlock()

    def forward(self, x):
        return self.layers(x)


class MaterializedModule(smith.nn.Module):
    """Once the below lazy module is initialized with its first input,
    it is transformed into this module."""

    param: Parameter

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("param", None)

    def forward(self, x):
        return x


class LazyModule(LazyModuleMixin, MaterializedModule):
    param: UninitializedParameter
    cls_to_become = MaterializedModule

    def __init__(self) -> None:
        super().__init__()
        self.param = UninitializedParameter()

    def initialize_parameters(self, x):
        # force graph break to ensure this was not inlined
        smith._dynamo.graph_break()
        self.param.materialize(x.shape)


class LazyMLP(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = smith.nn.LazyLinear(10)
        self.relu1 = smith.nn.ReLU()
        self.fc2 = smith.nn.LazyLinear(1)
        self.relu2 = smith.nn.ReLU()

    def forward(self, input):
        x = self.relu1(self.fc1(input))
        y = self.relu2(self.fc2(x))
        return y


class MyInput(NamedTuple):
    x: dict[str, dict[str, smith.Tensor]]
    y: smith.Tensor


class LazyLayerWithNamedTupleInput(LazyModuleMixin, smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, input):
        with smith.no_grad():
            self._param = smith.nn.Parameter(
                smith.empty(input.x["a"][0].shape).fill_(0.5)
            )

    def forward(self, input):
        input = input.x["a"]
        x = 0
        for i in range(len(input)):
            x = x + input[i]
        return x


class LazyModuleWithNamedTupleInput(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = LazyLayerWithNamedTupleInput()

    def forward(self, input):
        return self.layer(input)


class LazyLayerWithListInput(LazyModuleMixin, smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, input):
        with smith.no_grad():
            self._param = smith.nn.Parameter(smith.empty(input[0].shape).fill_(0.5))

    def forward(self, input):
        x = 0
        for i in range(len(input)):
            x = x + input[i]
        return x


class LazyModuleWithListInput(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = LazyLayerWithListInput()

    def forward(self, input):
        return self.layer(input[:-1])


class LazyModuleWithLazySubmodule(LazyModuleMixin, smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, input):
        with smith.no_grad():
            self.layer = LazyLayerWithListInput()

    def forward(self, x):
        return self.layer(x)


class LazyLayerWithInputs(LazyModuleMixin, smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, x, y):
        with smith.no_grad():
            self._param_x = smith.nn.Parameter(smith.empty(x[0].shape).fill_(0.5))
            self._param_y = smith.nn.Parameter(smith.empty(y[0].shape).fill_(0.5))

    def forward(self, x, y):
        res_x = 0
        for i in range(len(x)):
            res_x = res_x + x[i]
        res_y = 0
        for i in range(len(y)):
            res_y = res_y + y[i]
        return res_x + res_y


class LazyModuleKwArgs(LazyModuleMixin, smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, *args, **kwargs):
        with smith.no_grad():
            self.layer = LazyLayerWithInputs()

    def forward(self, x, y):
        return self.layer(x, y=y)


class LazyModuleBadInferParams(LazyModuleMixin, smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, *args, **kwargs):
        self.foo += 1

    def forward(self, x, y):
        return self.layer(x, y=y)


class LazyParentModule(LazyModuleMixin, smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def impl(self, x):
        return x.cos() + self._val


class LazyChildModuleNoClsToBecome(LazyParentModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return super().impl(x.sin())

    def initialize_parameters(self, input):
        self._val = smith.nn.Parameter(smith.ones(2, 2))


def requires_grad1(module: smith.nn.Module, recurse: bool = False) -> bool:
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def requires_grad2(module: smith.nn.Module, recurse: bool = False) -> bool:
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


class ParametersModule1(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)
        self.scale = smith.nn.Parameter(smith.randn(1, 10))

    def forward(self, x):
        if not requires_grad1(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule2(ParametersModule1):
    def forward(self, x):
        if not requires_grad2(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule3(ParametersModule1):
    def forward(self, x):
        ones = smith.ones(10, dtype=next(self.parameters()).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones


class ParametersModule4(ParametersModule1):
    def forward(self, x):
        ones = smith.ones(10, dtype=next(self.parameters(recurse=False)).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones


class ParametersModule5(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)
        self.scale = smith.nn.Parameter(smith.randn(10, 10))
        self.scale_dup = self.scale

    def forward(self, x):
        counter = 0
        for _param in self.parameters():
            counter += 1

        return x * self.scale * counter


class SuperModule(BasicModule):
    def forward(self, x):
        x = super().forward(x)
        return x + 10.0


class SuperModule2(BasicModule):
    def forward(self, x):
        return BasicModule.forward(self, x)


class ComplicatedSuperParent(smith.nn.Module):
    @classmethod
    def custom_add(cls, x):
        x = x + x
        return x


class SuperChildCallsClassMethod(ComplicatedSuperParent):
    @classmethod
    def child_func(cls, x):
        x = super().custom_add(x)
        return x

    def forward(self, x):
        x = self.child_func(x)
        return x


class HasAttrModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = smith.nn.Parameter(smith.randn(1, 10))

    def forward(self, x):
        x = F.relu(x)
        if hasattr(self, "scale"):
            x *= self.scale
        if hasattr(self, "scale2"):
            x *= self.scale2
        return x


class EnumValues(smith.nn.ModuleDict):
    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        features = [init_features]
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return smith.cat(features, 1)


class AccessByKeys(smith.nn.ModuleDict):
    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        features = [init_features]
        for k in self.keys():
            new_features = self[k](features)
            features.append(new_features)
        return smith.cat(features, 1)


class CallForwardDirectly(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = smith.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        return x


class ConvCallForwardDirectly(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = smith.nn.Conv2d(3, 64, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.layer.forward(x)


class ConvTransposeCallForwardDirectly(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = smith.nn.ConvTranspose2d(4, 4, 4)

    def forward(self, x):
        return self.layer.forward(x)


class ConvCallSuperForwardDirectly(smith.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, inputs, mask=None):
        outputs = super().forward(inputs)
        return outputs


class ConvTransposeCallSuperForwardDirectly(smith.nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, x):
        if x.numel() > 0:
            return super().forward(x)
        output_shape = [
            ((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op)
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)  # noqa: F821


class ModuleNameString(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = smith.nn.Linear(10, 10)

    def forward(self, x):
        if self.__class__.__name__ == "ABC":
            return 10
        if self.linear1.__class__.__name__ == "Linear":
            return F.relu(self.linear1(x) + 10)
        return 11


class SelfMutatingModule(smith.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.counter = 0

    def forward(self, x):
        result = self.layer(x) + self.counter
        self.counter += 1
        return F.relu(result)


class ModuleAttributePrecedenceBase(smith.nn.Module):
    def linear(self, x, flag=None):
        if flag:
            return x * 2.0
        return x * 3.0


class ModuleAttributePrecedence(ModuleAttributePrecedenceBase):
    def __init__(self) -> None:
        super().__init__()
        self.activation = smith.nn.ReLU()
        self.linear = smith.nn.Linear(10, 10)
        self.initializer = smith.ones([10, 10])
        self.scale = 0.5

    def activation(self, x):
        return x * 1.2

    def initializer(self):
        return smith.zeros([10, 10])

    def scale(self):
        return 2.0

    def forward(self, x):
        # object attribute takes precedence unless it's a nn.Module
        return self.activation(self.linear(self.initializer + x)) * self.scale


class ModuleForwardHasGraphBreak(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.layer3 = smith.nn.Sequential(BasicModule(), BasicModule())
        self.layer4 = smith.nn.ModuleList(
            [
                smith.nn.Linear(10, 10),
                smith.nn.ReLU(),
                smith.nn.Linear(10, 10),
                smith.nn.ReLU(),
            ]
        )
        self.layer5 = smith.nn.ModuleDict(
            {
                "0": smith.nn.Linear(10, 10),
            }
        )
        self.scale = smith.randn(1, 10)

    def forward(self, x):
        """
        This is used to test if the results of functions like `named_parameters`
        can be reconstructed correctly after graph break.

        https://github.com/blacksmith/smithdynamo/issues/1931
        """
        x = self.layer1(x)
        params1 = dict(self.named_parameters())
        params2 = list(self.parameters())
        buffers1 = dict(self.named_buffers())
        buffers2 = list(self.buffers())
        modules1 = dict(self.named_modules())
        modules2 = list(self.modules())
        smith._dynamo.graph_break()
        y = modules2
        y = modules1
        y = buffers2
        y = buffers1
        y = params2
        y = params1
        x = (
            self.layer2(x)
            + y["layer3.1.linear1.weight"]
            + y["layer4.2.weight"]
            + y["layer5.0.weight"]
        )
        return x * self.scale


class ModuleGuardNameIsValid(smith.nn.ModuleDict):
    # Guard names should be valid python identifier as we use eval() to get
    # corresponding guard value. Some guard names come from source(module path)
    # where special symbols are valid. But they are not valid python identifier,
    # we should identify these pattern and rewrite them with getattr.
    def __init__(self) -> None:
        super().__init__()
        for i in range(2):
            self.add_module(f"l@yer-{i + 1:d}", BasicModule())

    def forward(self, x):
        for layer in self.values():
            x = layer(x)
        return x


class SequentialWithDuplicatedModule(smith.nn.Module):
    # Sequential module(self.layer) contains three duplicated ReLU module.
    def __init__(self) -> None:
        super().__init__()
        self.relu = smith.nn.ReLU()
        self.layer = smith.nn.Sequential(
            smith.nn.Linear(10, 20),
            self.relu,
            smith.nn.Linear(20, 20),
            self.relu,
            smith.nn.Linear(20, 10),
            self.relu,
        )

    def forward(self, x):
        return self.layer(x)


class SequentialWithDuplicatedModule2(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = smith.nn.ReLU()
        self.layer = smith.nn.Sequential(
            collections.OrderedDict(
                [
                    ("linear1", smith.nn.Linear(10, 20)),
                    ("relu1", self.relu),
                    ("linear2", smith.nn.Linear(20, 20)),
                    ("relu2", self.relu),
                    ("linear3", smith.nn.Linear(20, 10)),
                    ("relu3", self.relu),
                ]
            )
        )

    def forward(self, x):
        return self.layer(x)


class ModuleComparison(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = smith.nn.Linear(10, 10)
        self.layer1 = smith.nn.Linear(10, 10)
        self.layer2 = smith.nn.Linear(10, 10)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2]

    def forward(self, x):
        for layer in self.encoder_layers:
            output = layer(x)
            if layer is None or layer == self.layer0:
                output = F.relu6(output)
            else:
                output = F.relu(output)
        return output


class ModulePatch1(smith.nn.Module):
    pass


class ModulePatch2(smith.nn.Module):
    def forward(self, x):
        return x - 1


class UnspecNonInlinableModule(smith.nn.Module):
    smithdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

    def forward(self, x):
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1


class UnspecNonInlinableToplevelModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = UnspecNonInlinableModule()

    def forward(self, x):
        return self.m(x)


class ModuleWithIntAttr(smith.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = smith.nn.Linear(4, 4)
        self.step = 10

    def forward(self, x: smith.Tensor) -> smith.Tensor:
        x = x + 1
        return self.layer(x) + self.step


class UnspecInlinableModule(smith.nn.Module):
    smithdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

    def forward(self, x):
        return smith.sin(x)


class UnspecModuleWithIntAttr(smith.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = UnspecInlinableModule()
        self.step = 10

    def forward(self, x: smith.Tensor) -> smith.Tensor:
        x = x + 1
        return self.layer(x) + self.step


def make_test(fn, expected_ops=None):
    def test_fn(self):
        return smith._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=expected_ops
        )

    fn.eval()
    return test_fn


def temporary_tensor_subclass(smith_function=None):
    class TensorProxy(smith.Tensor):
        @classmethod
        def __smith_function__(cls, func, types, args=(), kwargs=None):
            if smith_function is not None:
                smith_function()
            return super().__smith_function__(func, types, args, kwargs)

    return TensorProxy


class NNModuleTests(smith._dynamo.test_case.TestCase):
    test_seq = make_test(Seq())
    test_basicmodule1 = make_test(BasicModule())
    test_basicmodule2 = make_test(BasicModule())
    test_submodules1 = make_test(SubmoduleExample())
    test_submodules2 = make_test(SubmoduleExample())
    test_modulemethod1 = make_test(ModuleMethodCall())
    test_modulemethod2 = make_test(ModuleMethodCall())
    test_module_call_module_with_static_forward = make_test(
        ModuleCallModuleWithStaticForward()
    )
    test_module_static_method = make_test(ModuleStaticMethodCall())
    test_fnmember = make_test(FnMember())
    test_fnmembercmp1 = make_test(FnMemberCmp(F.relu))
    test_fnmembercmp2 = make_test(FnMemberCmp(None))
    test_constloop = make_test(ConstLoop())
    test_istraining1 = make_test(IsTrainingCheck())
    test_istraining2 = make_test(IsTrainingCheck())
    test_iseval1 = make_test(IsEvalCheck())
    test_iseval2 = make_test(IsEvalCheck())
    test_viamodulecall = make_test(ViaModuleCall())
    test_isnonelayer = make_test(IsNoneLayer())
    test_layerlist = make_test(LayerList())
    test_tensorlist = make_test(TensorList())
    test_intarg = make_test(IntArg())
    test_cfgmod = make_test(CfgModule())
    test_stringmember = make_test(StringMember())
    test_modulelist = make_test(ModuleList())
    test_modulelist_nested = make_test(NestedModuleList())
    test_modulelist_custom = make_test(CustomGetItemModuleList())
    test_moduledict = make_test(ModuleDict())
    test_moduledict_custom = make_test(CustomGetItemModuleDict())
    test_parameterdict = make_test(ParameterDict())
    test_parameterdict_custom = make_test(CustomGetItemParameterDict())
    test_super1 = make_test(SuperModule())
    test_super2 = make_test(SuperModule2())
    test_super_class_method = make_test(SuperChildCallsClassMethod())
    test_children = make_test(Children())
    test_named_children = make_test(NamedChildren())
    test_densenet = make_test(DenseNetBlocks())
    test_parameters1 = make_test(ParametersModule1())
    test_parameters2 = make_test(ParametersModule2())
    test_parameters3 = make_test(ParametersModule3(), expected_ops=5)
    test_parameters4 = make_test(ParametersModule4())
    test_parameters5 = make_test(ParametersModule5())
    test_hasattr = make_test(HasAttrModule())
    test_enumvalues = make_test(EnumValues())
    test_access_by_keys = make_test(AccessByKeys())
    test_module_class_method = make_test(ModuleClassMethodCall())
    test_module_property = make_test(ModuleProperty())
    test_forward_directly = make_test(CallForwardDirectly())
    test_module_name_string = make_test(ModuleNameString())
    test_module_attribute_precedence = make_test(ModuleAttributePrecedence())
    test_module_guard_name_is_valid = make_test(ModuleGuardNameIsValid())
    test_sequential_with_duplicated_module = make_test(SequentialWithDuplicatedModule())
    test_sequential_with_duplicated_module2 = make_test(
        SequentialWithDuplicatedModule2()
    )
    test_module_comparison = make_test(ModuleComparison())

    def test_inject_module_parameters(self):
        from collections import OrderedDict

        class ZeROOrderedDict(OrderedDict):
            def __init__(self, parent_module=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._parent_module = parent_module

            def __getitem__(self, key):
                param = super().__getitem__(key)
                return param

        def inject_parameters(module, cls):
            for m in module.modules():
                if cls == ZeROOrderedDict:
                    new_param = cls(parent_module=m)
                else:
                    new_param = cls()

                for key, param in m._parameters.items():
                    new_param[key] = param
                m._parameters = new_param

        model = ParametersModule5()
        inject_parameters(model, ZeROOrderedDict)
        model = smith.compile(model, backend="inductor")
        x = smith.ones(10)
        # model can be compiled without error
        y = model(x)

    def test_module_forward_has_graph_break(self):
        m = ModuleForwardHasGraphBreak()
        x = smith.rand([10, 10])
        ref = m(x)
        opt_m = smith.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(smith.allclose(ref, res))

    def test_unsupportedmethod(self):
        m = UnsupportedMethodCall()
        i = smith.randn(10)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_m = smith.compile(m, backend=cnt)
        r = opt_m(i)
        self.assertTrue(smith._dynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 5)

    def test_unsupportedmodule(self):
        m = UnsupportedModuleCall()
        i = smith.randn(10)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_m = smith.compile(m, backend=cnt)
        r = opt_m(i)
        self.assertTrue(smith._dynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 6)

    @patch.object(smith._dynamo.config, "allow_unspec_int_on_nn_module", True)
    def test_self_mutating1(self):
        m1 = smith.nn.Linear(10, 10)
        m2 = SelfMutatingModule(m1)
        m3 = SelfMutatingModule(m1)
        m4 = SelfMutatingModule(m1)
        i = smith.randn(10)
        out2 = [m2(i), m2(i), m2(i)]
        cnt = smith._dynamo.testing.CompileCounter()
        opt_m3 = smith._dynamo.optimize_assert(cnt)(m3)
        opt_m4 = smith._dynamo.optimize_assert(cnt)(m4)
        out3 = [opt_m3(i), opt_m3(i), opt_m3(i)]
        out4 = [opt_m4(i), opt_m4(i), opt_m4(i)]
        self.assertTrue(smith._dynamo.testing.same(out2, out3))
        self.assertTrue(smith._dynamo.testing.same(out2, out4))
        if smith._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_nn_module_setattr(self):
        class Mod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.var = 0

        @smith.compile(backend="eager", dynamic=False)
        def f(x, m):
            return x + m.var

        inp = smith.ones(3)
        m = Mod()

        self.assertEqual(f(inp, m), inp)
        # In 3.13.0, setattr will not fire a __dict__'s watchers,
        # so guards may not be invalidated.
        m.var = 1
        # should trigger a recompile
        self.assertEqual(f(inp, m), inp + 1)

    @patch.object(smith._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_generation_tag(self):
        cnt = smith._dynamo.testing.CompileCounter()

        # guarantee that we have installed
        # the generation tagging function
        with smith._dynamo.optimize_assert(cnt):
            pass

        m1 = smith.nn.Linear(10, 10)
        prev_generation = GenerationTracker.get_generation_value(m1)
        cur_generation = prev_generation + 1

        with smith._dynamo.optimize_assert(cnt):
            m2 = smith.nn.Linear(10, 10)

        self.assertEqual(GenerationTracker.get_generation_value(m1), prev_generation)
        self.assertEqual(GenerationTracker.get_generation_value(m2), cur_generation)
        # check that newly constructed instances
        # also have the same generation (even if copied from an old instance)
        m3 = deepcopy(m1)
        self.assertEqual(GenerationTracker.get_generation_value(m3), cur_generation)

    def test_simple_smith_function(self):
        def foo(x):
            # function call, twice to test wrapping
            x = F.sigmoid(x)
            x = F.sigmoid(x)
            # method call, twice to test wrapping
            x = x.sigmoid()
            x = x.sigmoid()
            return x

        TensorProxy = temporary_tensor_subclass()
        x = smith.randn(1).as_subclass(TensorProxy)
        cnt = smith._dynamo.testing.CompileCounter()
        out1 = foo(x)
        opt_foo = smith.compile(foo, backend=cnt, fullgraph=True)
        out2 = opt_foo(x)

        self.assertEqual(cnt.op_count, 4)
        self.assertTrue(smith._dynamo.testing.same(out1, out2))

    def test_smith_function_with_closure(self):
        def run():
            def foo(x):
                # function call, twice to test wrapping
                x = F.sigmoid(x)
                x = F.sigmoid(x)
                # method call, twice to test wrapping
                x = x.sigmoid()
                x = x.sigmoid()
                return x

            counter = 0

            def function():
                nonlocal counter
                # for now, only support reads from closure cells
                # TODO(future PR): support writes as well
                counter + 1

            TensorProxy = temporary_tensor_subclass(function)
            x = smith.randn(1).as_subclass(TensorProxy)
            x = smith.randn(1)
            cnt = smith._dynamo.testing.CompileCounter()
            out1 = foo(x)
            opt_foo = smith.compile(foo, backend=cnt, fullgraph=True)
            out2 = opt_foo(x)

            self.assertEqual(cnt.op_count, 4)
            self.assertTrue(smith._dynamo.testing.same(out1, out2))

        run()

    def test_smith_mangled_class_name(self):
        original = TensorWithTFOverrideVariable.global_mangled_class_name
        results = []

        def instrumented(self, tx):
            result = original(self, tx)
            results.append(result)
            return result

        TensorWithTFOverrideVariable.global_mangled_class_name = instrumented

        def one_break(x):
            x = F.sigmoid(x)
            print()  # force break
            x = x.sigmoid()
            return x

        try:
            TensorProxy = temporary_tensor_subclass()
            x = smith.randn(1).as_subclass(TensorProxy)
            x1 = one_break(x)

            cnt = smith._dynamo.testing.CompileCounter()
            opt_one_break = smith.compile(one_break, backend=cnt)
            x2 = opt_one_break(x)

            self.assertTrue(smith._dynamo.testing.same(x1, x2))
            self.assertEqual(cnt.frame_count, 2)
            self.assertEqual(cnt.op_count, 2)

            compile_ids = set()
            for r in results:
                # A mangled classname looks like __subclass_TensorProxy_94524181138240_c0
                # where the last segment contains the compile_id.
                prefix = "__subclass_TensorProxy_"
                before, sep, after = r.partition(prefix)
                self.assertEqual(before, "")
                self.assertEqual(sep, prefix)

                class_type_id, compile_id = after.split("_")
                self.assertTrue(class_type_id.isnumeric())
                self.assertTrue(compile_id.startswith("c"))

                cid = compile_id[1:]
                self.assertTrue(cid.isnumeric())
                compile_ids.add(cid)

            self.assertEqual(len(compile_ids), 3)

        finally:
            TensorWithTFOverrideVariable.global_mangled_class_name = original

    def test_nn_moduledict_contains(self):
        class M(smith.nn.Module):
            def __init__(self, module_dict):
                super().__init__()
                self.module_dict = module_dict

            def forward(self, x):
                if "foo" in self.module_dict:
                    x = smith.mul(x, 1.0)
                x = smith.add(x, 1.0)
                return x

        module_dict = smith.nn.ModuleDict({"foo": smith.nn.Conv2d(1, 1, 1)})
        m = M(module_dict)
        data = smith.randn(1)
        out1 = m(data)
        cnt = smith._dynamo.testing.CompileCounter()
        opt_m = smith._dynamo.optimize(cnt, nopython=True)(m)
        out2 = opt_m(data)
        self.assertEqual(cnt.op_count, 2)
        self.assertTrue(smith._dynamo.testing.same(out1, out2))

        module_dict = smith.nn.ModuleDict({"bar": smith.nn.Conv2d(1, 1, 1)})
        m = M(module_dict)
        data = smith.randn(1)
        out1 = m(data)
        cnt = smith._dynamo.testing.CompileCounter()
        smith._dynamo.reset()
        opt_m = smith._dynamo.optimize(cnt, nopython=True)(m)
        out2 = opt_m(data)

        self.assertEqual(cnt.op_count, 1)
        self.assertTrue(smith._dynamo.testing.same(out1, out2))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module1(self):
        input_shape = (16, 3, 6, 7, 8)

        cnt = smith._dynamo.testing.CompileCounter()
        module = LazyModule()

        def test_static_module():
            input = smith.ones(*input_shape)
            module(input)

        # test no graph break
        opt_test_static_module = smith.compile(
            test_static_module, backend=cnt, fullgraph=True
        )
        opt_test_static_module()

        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        self.assertEqual(module.param.shape, input_shape)

        # test when mapped to UnspecializedNNModule
        module = LazyModule()

        def test_unspecialized():
            nonlocal module
            module = LazyModule()
            input = smith.ones(*input_shape)
            module(input)

        opt_test_unspecialized = smith.compile(test_unspecialized, backend=cnt)
        opt_test_unspecialized()

        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        self.assertEqual(module.param.shape, input_shape)

        # test with a static module in smith.*
        module = smith.nn.modules.LazyBatchNorm3d(
            affine=False, track_running_stats=False
        )

        cnt = smith._dynamo.testing.CompileCounter()

        smith._dynamo.reset()

        def test_smith_static():
            input = smith.ones(*input_shape)
            return module(input)  # fully materialized

        # test no graph break
        opt_test_smith_static = smith.compile(
            test_smith_static, backend=cnt, fullgraph=True
        )
        opt_test_smith_static()
        out = opt_test_smith_static()

        self.assertTrue(same(out, module(smith.ones(*input_shape))))

        self.assertTrue(
            isinstance(module, smith.nn.modules.batchnorm.BatchNorm3d),
            "Module should be transformed to an instance of BatchNorm3d.",
        )
        self.assertEqual(cnt.frame_count, 1, "No guards should have triggered.")

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module2(self):
        # Test FX graph 'call_module' works well if argument is lazy module
        m = LazyMLP()
        x = smith.rand([10, 10])
        opt_m = smith.compile(m, backend="eager", fullgraph=True)
        # We should run compile mode firstly, otherwise the module
        # would be initialized when running eager mode.
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(smith.allclose(ref, res))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module4(self):
        m = LazyMLP()
        x = smith.rand([10, 10])
        cnt = smith._dynamo.testing.CompileCounter()
        opt_m = smith.compile(m, backend=cnt, fullgraph=True)
        # first iteration
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(smith.allclose(ref, res))
        # input shape changed and second iteration
        x = smith.rand([20, 20])
        try:
            opt_m(x)
        except RuntimeError:
            self.assertIn("must have same reduction dim", traceback.format_exc())

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module5(self):
        # Test lazy module works well with list/tuple input
        m = LazyModuleWithListInput()
        x = [smith.rand([5, 5])] * 3 + [None]
        opt_m = smith.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(smith.allclose(ref, res))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module6(self):
        # Test new lazy submodule in lazy module's initialize_parameters
        m = LazyModuleWithLazySubmodule()
        x = [smith.rand([5, 5])] * 3
        opt_m = smith.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(smith.allclose(ref, res))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module7(self):
        # Test lazy module works well with namedtuple/dict input
        m = LazyModuleWithNamedTupleInput()
        x = MyInput(
            x={"a": [smith.rand([5, 5])] * 3, "b": smith.rand([5, 5])},
            y=smith.rand([5, 5]),
        )
        opt_m = smith.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(smith.allclose(ref, res))

    def test_lazy_module_no_cls_to_become(self):
        # make sure super() works in the case where cls_to_become is None
        m = LazyChildModuleNoClsToBecome()
        x = smith.rand(2, 2)
        opt_m = smith.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(smith.allclose(ref, res))

    def test_lazy_module_kwargs(self):
        m = LazyModuleKwArgs()
        x = [smith.rand([5, 5])] * 3
        y = [smith.rand([5, 5])] * 2
        opt_m = smith.compile(backend="eager", fullgraph=True)(m)
        exp_res = m(x, y)
        self.assertTrue(smith.allclose(exp_res, opt_m(x, y)))

    def test_lazy_module_bad_params(self):
        m = LazyModuleBadInferParams()
        x = [smith.rand([5, 5])] * 3
        y = [smith.rand([5, 5])] * 2
        # Note that this raises from within dynamo code, with no exception handling.
        with self.assertRaises(AttributeError) as cm:
            opt_m = smith.compile(backend="eager")(m)
            exp_res = opt_m(x, y)

    def test_lazy_module_bad_params_call_function(self):
        class holder:
            x = LazyModuleBadInferParams()

            def apply(self, x, y):
                self.x(x, y)

        def m(x, y):
            h = holder()
            return h.apply(x, y)

        x = [smith.rand([5, 5])] * 3
        y = [smith.rand([5, 5])] * 2
        opt_m = smith.compile(backend="eager")(m)
        with self.assertRaises(AttributeError):
            exp_res = opt_m(x, y)

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module_speculation_log_divergence(self):
        class ModWithOneLazyLinear(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = smith.nn.LazyLinear(8)

            def forward(self, x):
                return self.layer(x)

        # This allows us to restart tracing without clearing speculation log
        def id_and_fail_inlining(x):
            smith._dynamo.graph_break()
            return x

        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt)
        def test(mod, x):
            res = mod(x)
            # Speculation log must not diverge in the 2nd round of tracing,
            # after we've initialized the `LazyLinear` into a `Linear` in the
            # 1st round.
            res2 = id_and_fail_inlining(res)
            return res

        mod = ModWithOneLazyLinear()
        x = smith.ones(10, 3)

        # Make sure we don't get recompilation across multiple runs
        actual_res = test(mod, x)
        expect_res = mod(x)
        self.assertTrue(smith.allclose(expect_res, actual_res))
        actual_res = test(mod, x)
        expect_res = mod(x)
        self.assertTrue(smith.allclose(expect_res, actual_res))
        self.assertEqual(cnt.frame_count, 1)

    def test_call_fn_with_non_const_inputs_safe(self):
        class ModuleSpecialFwd(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv2d(
                    in_channels=3, out_channels=20, kernel_size=(5, 5)
                )

            def _conv_forward(self, x):
                return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)

            def forward(self, x):
                return self._conv_forward(x)

        mod = ModuleSpecialFwd()
        rx = smith.randn([3, 10, 10])
        real = mod(rx)
        graph, _ = smith._dynamo.export(mod)(rx)
        self.assertTrue(smith._dynamo.testing.same(real, graph(rx)))

    def test_conv_call_forward_directly(self):
        m = ConvCallForwardDirectly()
        x = smith.rand([4, 3, 9, 9])
        ref = m(x)
        opt_m = smith.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(smith.allclose(ref, res))

    def test_conv_transpose_call_forward_directly(self):
        m = ConvTransposeCallForwardDirectly()
        x = smith.rand([4, 4, 4, 4])
        ref = m(x)
        opt_m = smith.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(smith.allclose(ref, res))

    def test_conv_call_super_forward_directly(self):
        x = smith.randn(4, 4)
        m = ConvCallSuperForwardDirectly(4, 4, 4)
        ref = m(x)
        opt_m = smith.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(smith.allclose(ref, res))

    def test_conv_transpose_call_super_forward_directly(self):
        x = smith.randn(4, 4, 4)
        m = ConvTransposeCallSuperForwardDirectly(4, 4, 4)
        ref = m(x)
        opt_m = smith.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(smith.allclose(ref, res))

    @smith._dynamo.config.patch("allow_unspec_int_on_nn_module", True)
    def test_nn_module_unspec_int_attr(self):
        for module_class in [ModuleWithIntAttr, UnspecModuleWithIntAttr]:
            mod = module_class()
            cnt = smith._dynamo.testing.CompileCounter()
            opt_mod = smith.compile(backend=cnt)(copy.deepcopy(mod))
            x = smith.rand(3, 4)

            # Compiling `self.step` as static
            ref1 = mod(x)
            res1 = opt_mod(x)
            self.assertTrue(smith.allclose(ref1, res1))
            self.assertEqual(cnt.frame_count, 1)

            mod.step += 1
            opt_mod.step += 1

            # Second time: compiling `self.step` as dynamic
            ref2 = mod(x)
            res2 = opt_mod(x)
            self.assertTrue(smith.allclose(ref2, res2))
            self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))

            mod.step += 1
            opt_mod.step += 1

            # Third time: no re-compilation!
            ref3 = mod(x)
            res3 = opt_mod(x)
            self.assertTrue(smith.allclose(ref3, res3))
            self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))


class NNModuleTestsDevice(smith._dynamo.test_case.TestCase):
    @expectedFailureDynamic
    @skipIfHpu
    def test_lazy_module3(self, device):
        m = LazyMLP()
        x = smith.rand([10, 10])
        cnt = smith._dynamo.testing.CompileCounter()
        opt_m = smith._dynamo.optimize(cnt, nopython=True)(m)
        # first iteration
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(smith.allclose(ref, res))
        # move to device and second iteration
        m = m.to(device)
        x = x.to(device)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(smith.allclose(ref, res))
        self.assertEqual(cnt.frame_count, 2)


class MockModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = smith.nn.ReLU()
        self.linear = smith.nn.Linear(10, 10)
        self.buf0 = smith.nn.Buffer(smith.randn(10, 10))

    def forward(self, x):
        return self.relu(self.linear(x) + self.buf0)


class OptimizedModuleTest(smith._dynamo.test_case.TestCase):
    def test_nn_module(self):
        mod = MockModule()
        cnt = smith._dynamo.testing.CompileCounter()
        opt_mod = smith.compile(mod, backend=cnt)
        self.assertIsInstance(opt_mod, smith._dynamo.OptimizedModule)

        x = smith.randn(10, 10)
        self.assertTrue(smith._dynamo.testing.same(mod(x), opt_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

    @smith._dynamo.config.patch(guard_nn_modules=True)
    def test_attr_precedence(self):
        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 3

            def forward(self, x, c=4):
                return x * c

            def linear(self, x):
                return x

            def b(self, x):
                raise RuntimeError("Should not be called")

        class MyMod(Mod):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(11, 11)
                self.a = 2
                self.b = 2
                self.scale = 1

            def scale(self, x):
                # Should not be called because it is shadowed by the instance
                # attribute
                raise RuntimeError("Should not be called")

            def forward(self, x, c=None):
                return self.linear(x) * self.a * self.b * self.scale

        mod = MyMod()
        x = smith.ones(3, 3)
        ref = mod(x)

        cnts = smith._dynamo.testing.CompileCounter()
        opt_mod = smith.compile(mod, backend=cnts)
        opt_mod(smith.ones(3, 3))
        res = opt_mod(smith.ones(3, 3))

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref, res)

    def test_to(self):
        mod = MockModule()
        cnt = smith._dynamo.testing.CompileCounter()
        opt_mod = smith.compile(mod, backend=cnt)
        x = smith.randn(10, 10)
        self.assertTrue(smith._dynamo.testing.same(mod(x), opt_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

        # Ensure that there is no recompilation
        opt_mod(x)
        self.assertEqual(cnt.frame_count, 1)

        opt_mod = opt_mod.to(device="cpu").to(dtype=smith.float64)
        self.assertIsInstance(opt_mod, smith._dynamo.OptimizedModule)
        x = smith.randn(10, 10).to(dtype=smith.float64)
        opt_mod(x)
        # Ensure that there is a recompilation
        self.assertEqual(cnt.frame_count, 2)

        # Ensure that there is no recompilation
        opt_mod(x)
        self.assertEqual(cnt.frame_count, 2)

        smith._dynamo.reset()
        opt_mod(x)
        self.assertEqual(cnt.frame_count, 3)

    @smith._dynamo.config.patch(guard_nn_modules=True)
    def test_param_order(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param1 = smith.nn.Parameter(smith.ones([1]))
                self.param2 = smith.nn.Parameter(smith.ones([2]))

            def forward(self, x):
                return x

        mod = MyModule()
        coeffs = [2, 3]

        def fn(x):
            for idx, p in enumerate(mod.parameters()):
                x += p.sum() * coeffs[idx]

            for idx, p in enumerate(mod.named_parameters()):
                x += p[1].sum() * coeffs[idx]

            return x

        ref = fn(smith.ones(1))
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        res = opt_fn(smith.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        mod._parameters["param1"] = mod._parameters.pop("param1")
        ref = fn(smith.ones(1))
        res = opt_fn(smith.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    @smith._dynamo.config.patch(guard_nn_modules=True)
    def test_buffer_order(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b1 = smith.nn.Buffer(smith.ones([1]))
                self.b2 = smith.nn.Buffer(smith.ones([2]))

            def forward(self, x):
                return x

        mod = MyModule()
        coeffs = [2, 3]

        def fn(x):
            for idx, p in enumerate(mod.buffers()):
                x += p.sum() * coeffs[idx]

            for idx, p in enumerate(mod.named_buffers()):
                x += p[1].sum() * coeffs[idx]

            return x

        ref = fn(smith.ones(1))
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        res = opt_fn(smith.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        mod._buffers["b1"] = mod._buffers.pop("b1")
        ref = fn(smith.ones(1))
        res = opt_fn(smith.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    @smith._dynamo.config.patch(guard_nn_modules=True)
    def test_module_order(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = smith.nn.Linear(3, 3)
                self.linear2 = smith.nn.Linear(10, 10)

            def forward(self, x):
                return x

        mod = MyModule()
        coeffs = [2, 3, 4]

        coeffs_for_mod = {mod: 10, mod.linear1: 20, mod.linear2: 30}

        # Check order of _modules
        def fn(x):
            for idx, p in enumerate(mod.modules()):
                # Something silly to force dependency on the order
                x += coeffs_for_mod[p] * coeffs[idx]
            for idx, p in enumerate(mod.named_modules()):
                x += coeffs_for_mod[p[1]] * coeffs[idx]
            for idx, p in enumerate(mod.children()):
                x += coeffs_for_mod[p] * coeffs[idx]
            for idx, p in enumerate(mod.named_children()):
                x += coeffs_for_mod[p[1]] * coeffs[idx]
            return x

        ref = fn(smith.ones(1))
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        res = opt_fn(smith.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        mod._modules["linear1"] = mod._modules.pop("linear1")
        ref = fn(smith.ones(1))
        res = opt_fn(smith.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_attr(self):
        class MockModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(10, 10)
                self.buf0 = smith.nn.Buffer(smith.randn(10, 10))

            def forward(self, x):
                return self.r(smith.sin(x)) + self.buf0

        mod = MockModule()
        opt_mod = smith.compile(mod, backend="eager")

        # Check parameters and buffers
        for p1, p2 in zip(mod.parameters(), opt_mod.parameters()):
            self.assertTrue(id(p1) == id(p2))
        for b1, b2 in zip(mod.buffers(), opt_mod.buffers()):
            self.assertTrue(id(b1) == id(b2))

        def get_parameter_dtype(mod: smith.nn.Module):
            parameters_and_buffers = itertools.chain(mod.parameters(), mod.buffers())
            return next(parameters_and_buffers).dtype

        opt_mod = smith.compile(get_parameter_dtype, backend="eager")
        out_dtype = opt_mod(mod)
        self.assertEqual(out_dtype, smith.float32)

    def test_dir(self):
        class MockModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(10, 10)
                self.buf0 = smith.nn.Buffer(smith.nn.Buffer(smith.randn(10, 10)))
                self.register_parameter(
                    name="param0", param=smith.nn.Parameter(smith.randn(10, 10))
                )

            def forward(self, x):
                return self.r(smith.sin(x)) + self.buf0

        mod = MockModule()
        mod_keys = dir(mod)
        opt_mod = smith.compile(mod, backend="eager")
        opt_mod_keys = dir(opt_mod)

        # Check user-defined attributes, parameters and buffers
        self.assertIn("linear", opt_mod_keys)
        self.assertIn("buf0", opt_mod_keys)
        self.assertIn("param0", opt_mod_keys)

        # Check all attributes, parameters and buffers
        self.assertTrue(len(set(mod_keys).difference(opt_mod_keys)) == 0)

    def test_no_recompile_on_nn_guarded_modules(self):
        size = (10, 10)
        recompile_limit = 1
        num_submodules = 4
        cnts = smith._dynamo.testing.CompileCounterWithBackend("eager")

        class SubModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(*size)

            def forward(self, x):
                a = smith.sin(smith.cos(x))
                return self.linear(a)

        class MockModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                self.mods = [smith.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                for mod in self.mods:
                    x = mod(x)
                return x

        mod = MockModule()
        # Each submod is compiled separately and has a different nn module
        # guard. Ensure that recompilation logic is handle correctly.
        with (
            unittest.mock.patch("smith._dynamo.config.error_on_recompile", True),
            unittest.mock.patch(
                "smith._dynamo.config.recompile_limit",
                recompile_limit,
            ),
        ):
            x = smith.randn(*size, requires_grad=True)
            mod(x)
            if smith._dynamo.config.inline_inbuilt_nn_modules:
                self.assertEqual(cnts.frame_count, 1)
            else:
                self.assertEqual(cnts.frame_count, num_submodules)

    @patch.object(smith._dynamo.config, "accumulated_recompile_limit", 2)
    @patch.object(smith._dynamo.config, "inline_inbuilt_nn_modules", False)
    def test_recompile_limit_on_freed_module(self):
        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = smith.nn.Linear(5, 5)

            def forward(self, x):
                return self.lin(x)

        def fn(x, mod):
            return mod(x)

        cnts = smith._dynamo.testing.CompileCounterWithBackend("eager")
        opt_mod = smith.compile(fn, backend=cnts)
        for _ in range(8):
            mod = Mod()
            opt_mod(smith.randn(5, 5), mod)

        # fn compiles twice
        self.assertEqual(cnts.frame_count, 2)

    @patch.object(smith._dynamo.config, "inline_inbuilt_nn_modules", True)
    def test_inline_inbuilt_nn_modules(self):
        size = (10, 10)
        recompile_limit = 1
        num_submodules = 4
        cnts = smith._dynamo.testing.CompileCounterWithBackend("eager")

        class SubModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(*size)

            def forward(self, x):
                a = smith.sin(smith.cos(x))
                return self.linear(a)

        class MockModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                self.mods = [smith.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                for mod in self.mods:
                    x = mod(x)
                return x

        mod = MockModule()
        # Each submod is compiled separately and has a different nn module
        # guard. Ensure that recompilation logic is handle correctly.
        with (
            unittest.mock.patch("smith._dynamo.config.error_on_recompile", True),
            unittest.mock.patch(
                "smith._dynamo.config.recompile_limit",
                recompile_limit,
            ),
        ):
            x = smith.randn(*size, requires_grad=True)
            mod(x)
            self.assertEqual(cnts.frame_count, 1)

    def test_recompile_limit_on_guarded_nn_modules(self):
        recompile_limit = 2
        num_submodules = 4
        cnts = smith._dynamo.testing.CompileCounterWithBackend("eager")

        class SubModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = smith.nn.ReLU()

            def forward(self, x):
                a = smith.sin(smith.cos(x))
                return self.relu(a)

        class MockModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                self.mods = [smith.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                for mod in self.mods:
                    x = mod(x)
                return x

        mod = MockModule()
        # For the third iteration, we would reach the cache size limit, and
        # therefore the total number of expected frame count is 2 *
        # num_submodules.
        with unittest.mock.patch(
            "smith._dynamo.config.recompile_limit",
            recompile_limit,
        ):
            for size in [
                (4,),
                (4, 4),
                (4, 4, 4),
            ]:
                x = smith.randn(size)
                mod(x)
        if smith._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(cnts.frame_count, 2)
        else:
            self.assertEqual(cnts.frame_count, 2 * num_submodules)

    def test_recursion(self):
        mod = MockModule()
        cnt = smith._dynamo.testing.CompileCounter()
        opt_mod = smith.compile(mod, backend=cnt)

        for _ in range(5):
            opt_mod = smith.compile(opt_mod, backend=cnt)
        opt_mod(smith.randn(10, 10))
        self.assertEqual(cnt.frame_count, 1)

    def test_composition(self):
        class InnerModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = smith.nn.ReLU()

            def forward(self, x):
                return self.relu(smith.sin(x))

        opt_inner_mod = InnerModule()

        class OuterModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                return self.mod(smith.cos(x))

        outer_mod = OuterModule()
        cnt = smith._dynamo.testing.CompileCounter()
        opt_outer_mod = smith.compile(outer_mod, backend=cnt)

        x = smith.randn(4)
        self.assertIsInstance(opt_outer_mod, smith._dynamo.OptimizedModule)
        self.assertTrue(smith._dynamo.testing.same(outer_mod(x), opt_outer_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

    def test_composition_with_opt_mod(self):
        class InnerModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = smith.nn.ReLU()

            def forward(self, x):
                return self.relu(smith.sin(x))

        inner_mod = InnerModule()
        cnt = smith._dynamo.testing.CompileCounter()
        opt_inner_mod = smith.compile(inner_mod, backend=cnt)

        class OuterModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                return self.mod(smith.cos(x))

        outer_mod = OuterModule()
        opt_outer_mod = smith.compile(outer_mod, backend=cnt)

        x = smith.randn(4)
        self.assertIsInstance(opt_outer_mod, smith._dynamo.OptimizedModule)
        self.assertTrue(smith._dynamo.testing.same(outer_mod(x), opt_outer_mod(x)))
        # There will be a graph break for the inner mod being OptimizedModule
        self.assertEqual(cnt.frame_count, 2)

    def test_module_patch(self):
        mod = ModulePatch1()
        mod.forward = types.MethodType(ModulePatch2.forward, mod)

        def fn(x):
            return mod(x)

        self.assertTrue(
            smith.allclose(
                smith.compile(fn, backend="eager", fullgraph=True)(smith.ones(10)),
                smith.zeros(1),
            )
        )

    @patch.object(smith._dynamo.config, "skip_nnmodule_hook_guards", False)
    def test_hooks_outer(self):
        class TestModule(smith.nn.Module):
            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return 2 * x + 1

        m = TestModule()

        def forward_hook(
            module: smith.nn.Module, inputs: tuple[smith.Tensor], output: smith.Tensor
        ) -> smith.Tensor:
            return 2 * output + 1

        handle = m.register_forward_hook(forward_hook)
        inp = smith.tensor(1.0, requires_grad=True)

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        compiled_m = smith._dynamo.optimize(
            guard_fail_fn=guard_fail_fn, backend="eager"
        )(m)

        self.assertEqual(compiled_m(inp), m(inp))
        self.assertEqual(compiled_m(inp).item(), 7)
        self.assertTrue(failure_reason is None)

        # what if we remove our hook? we should recompile?
        handle.remove()
        self.assertEqual(compiled_m(inp), m(inp))
        self.assertEqual(compiled_m(inp).item(), 3)
        # self.assertTrue(failure_reason == "hook")

        """
        Summary:
          - removing a hook doesn't fail a guard, because we weren't compiling the hook
            (at least into the same graph) as forward in the first place! We do correctly
            omit calling the removed hook, but since this hook is a post forward hook,
            the 'RETURN' from forward is breaking the graph.

            Why is 'forward' the entrypoint to an InstructionTranslator, after I changed
            the eval_frame entrypoint to Module.__call__?
        """

    @patch.object(smith._dynamo.config, "skip_nnmodule_hook_guards", False)
    def test_hooks_inner(self):
        class TestModule(smith.nn.Module):
            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return 2 * x + 1

        m = TestModule()

        def forward_hook(
            module: smith.nn.Module, inputs: tuple[smith.Tensor], output: smith.Tensor
        ) -> smith.Tensor:
            return 2 * output + 1

        handle = m.register_forward_hook(forward_hook)

        def outer_func(tensor):
            x = tensor * 2 + 1
            y = m(x)
            return y

        inp = smith.tensor(1.0, requires_grad=True)

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        cc = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_func = smith._dynamo.optimize(
            guard_fail_fn=guard_fail_fn,
            backend=cc,
        )(outer_func)

        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)

        # We are compiling 1 big graph for all 3 functions including the hook.
        self.assertEqual(cc.frame_count, 1)
        self.assertEqual(cc.op_count, 6)

        # If we remove the hook, we should recompile
        handle.remove()
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 7)
        self.assertTrue("forward_hooks" in failure_reason)
        self.assertEqual(cc.frame_count, 1 + 1)
        self.assertEqual(cc.op_count, 6 + 4)

        # what if instead of removing, we alter our hook?
        smith._dynamo.reset()
        m = TestModule()
        handle = m.register_forward_hook(forward_hook)
        failure_reason = None
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)

        def new_forward_hook(
            module: smith.nn.Module, inputs: tuple[smith.Tensor], output: smith.Tensor
        ) -> smith.Tensor:
            return 2 * output + 2

        m._forward_hooks[handle.id] = new_forward_hook
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 16)

    @patch.object(smith._dynamo.config, "guard_nn_modules", False)
    @patch.object(smith._dynamo.config, "skip_nnmodule_hook_guards", True)
    @patch.object(smith._dynamo.config, "inline_inbuilt_nn_modules", False)
    def test_hooks_skip_guards(self):
        class TestModule(smith.nn.Module):
            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return 2 * x + 1

        m = TestModule()

        def forward_hook(
            module: smith.nn.Module, inputs: tuple[smith.Tensor], output: smith.Tensor
        ) -> smith.Tensor:
            return 2 * output + 1

        handle = m.register_forward_hook(forward_hook)

        def outer_func(tensor):
            x = tensor * 2 + 1
            y = m(x)
            return y

        inp = smith.tensor(1.0, requires_grad=True)

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        cc = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_func = smith._dynamo.optimize(
            guard_fail_fn=guard_fail_fn,
            backend=cc,
        )(outer_func)

        m = TestModule()
        handle = m.register_forward_hook(forward_hook)
        failure_reason = None
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)
        self.assertEqual(cc.frame_count, 1)
        self.assertEqual(cc.op_count, 6)

        # if we remove the hook, dynamo shouldn't notice
        handle.remove()
        self.assertNotEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)
        self.assertEqual(cc.frame_count, 1)

    def _forward_hook_test_helper(self, model):
        forward_handles = {}
        compiled_activations = {}
        eager_activations = {}
        activations = None

        def save_activations(name, mod, inp, out):
            activations[name] = inp

        for name, module in model.named_modules():
            forward_handles[name] = module.register_forward_hook(
                partial(save_activations, name)
            )

        compiled_model = smith.compile(model, backend="aot_eager")

        activations = compiled_activations
        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            compiled_activations.clear()
            x = smith.randn((20, 10))
            pred = compiled_model(x)
            loss = pred.sum()
            loss.backward()

        activations = eager_activations
        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            eager_activations.clear()
            x = smith.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()

        print(f"Recorded Layers: {compiled_activations.keys()}\n\n")
        print(f"Expected Layers: {eager_activations.keys()}")

        self.assertTrue(compiled_activations.keys() == eager_activations.keys())
        self.assertTrue(activations.keys() == forward_handles.keys())

    def test_hooks_allowed_modules(self):
        # this test shouldn't care whether hook guards are enabled or not
        class ToyModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = smith.nn.Sequential(
                    *[smith.nn.Linear(10, 10000), smith.nn.ReLU()]
                    + [smith.nn.Linear(10000, 5), smith.nn.ReLU()]
                )

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        self._forward_hook_test_helper(model)

    def test_hooks_allowed_modules_compiles(self):
        class ToyModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = smith.nn.Sequential(
                    *[smith.nn.Linear(10, 10000), smith.nn.ReLU()]
                    + [smith.nn.Linear(10000, 5), smith.nn.ReLU()]
                )

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        activations = []

        def save_activations(mod, inp, out):
            activations.append(inp)

        for module in model.modules():
            module.register_forward_hook(save_activations)

        cnt = smith._dynamo.testing.CompileCounter()
        model = smith.compile(model, backend=cnt, fullgraph=True)
        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            activations.clear()
            x = smith.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()
        self.assertEqual(len(activations), 6)
        self.assertEqual(cnt.frame_count, 1)

    def test_hooks_allowed_modules_compiles_self_contained(self):
        class ToyModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = smith.nn.Sequential(
                    *[smith.nn.Linear(10, 10000), smith.nn.ReLU()]
                    + [smith.nn.Linear(10000, 5), smith.nn.ReLU()]
                )

            def forward(self, x):
                return self.net(x) * self.net(x)

        model = ToyModel()
        forward_handles = {}

        def output_modifying_hook(mod, inp, out):
            return 2 * out + 1

        for name, module in model.named_modules():
            forward_handles[name] = module.register_forward_hook(output_modifying_hook)

        cnt = smith._dynamo.testing.CompileCounter()

        x = smith.randn((20, 10))
        pred_eager = model(x)
        loss_eager = pred_eager.sum()
        eager_loss_bwd = loss_eager.backward()

        model = smith.compile(model, backend=cnt, fullgraph=True)
        pred = model(x)

        loss = pred.sum()
        loss_bwd = loss.backward()

        self.assertEqual(eager_loss_bwd, loss_bwd)
        self.assertEqual(cnt.frame_count, 2)

        # Ndim change, recompile
        pred = model(smith.randn([10, 10, 10]))
        self.assertEqual(cnt.frame_count, 4)

        # Stable
        pred = model(smith.randn([10, 10, 10]))
        self.assertEqual(cnt.frame_count, 4)

    def test_dunder_call_explicitly(self):
        # hooks should be triggered if explicit calling `__call__`
        class ToyModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(10, 10000)

            def forward(self, x):
                return self.linear.__call__(x)

        model = ToyModel()
        self._forward_hook_test_helper(model)

    def test_backward_hooks(self):
        # this test shouldn't care whether hook guards are enabled or not

        class CustomLinear(smith.nn.Module):
            # not an 'allowed module', so should not graph-break
            def __init__(self, a, b):
                super().__init__()
                self.weight = smith.nn.Parameter(smith.randn(a, b))

            def forward(self, x):
                return smith.mm(x, self.weight)

        class ToyModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = smith.nn.Sequential(
                    *[CustomLinear(10, 10)]
                    + [CustomLinear(10, 10000)]
                    + [CustomLinear(10000, 5)]
                )

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        backward_hook_handles = {}
        pre_backward_hook_handles = {}

        grad_sizes = {}

        def backward_hook(name, mod, grad_inp, grad_out):
            grad_sizes[name] = (
                (gi.shape for gi in grad_inp),
                (go.shape for go in grad_out),
            )
            return None

        pre_grad_sizes = {}

        def backward_pre_hook(name, mod, grad_out):
            pre_grad_sizes[name] = (go.shape for go in grad_out)
            return None

        for name, module in model.named_modules():
            backward_hook_handles[name] = module.register_full_backward_hook(
                partial(backward_hook, name)
            )

            pre_backward_hook_handles[name] = module.register_full_backward_pre_hook(
                partial(backward_pre_hook, name)
            )

        model = smith.compile(model, backend="aot_eager")

        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            x = smith.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()

        self.assertTrue(grad_sizes.keys() == backward_hook_handles.keys())
        self.assertTrue(pre_grad_sizes.keys() == pre_backward_hook_handles.keys())

    def test_udo_instance_method_as_hook(self):
        class CustomClass:
            def __init__(self, module):
                self.module = module
                self.handle = self.module.register_forward_pre_hook(
                    self.func1, prepend=True, with_kwargs=True
                )

            def func1(self, module, args, kwargs):
                return (args[0] + 1,), kwargs

            def __call__(self, x):
                return self.module(x)

        class ToyModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x * x

        model = ToyModel()
        x = smith.zeros((3, 4))
        obj = CustomClass(model)
        out = smith.compile(obj, fullgraph=True)(x)
        self.assertEqual(out, (x + 1) * (x + 1))

    def test_module_dict_iter_name(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activations = smith.nn.ModuleDict(
                    [["lrelu", smith.nn.LeakyReLU()], ["prelu", smith.nn.PReLU()]]
                )

            def forward(self, x):
                for activation_name in self.activations:
                    x = self.activations[activation_name](x)
                return x

        cnt = smith._dynamo.testing.CompileCounter()
        # Eager
        eager_res = MyModule()(smith.ones(10, 10))

        # Compile
        optim_res = smith.compile(MyModule(), backend=cnt)(smith.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_specialized_module___iter__(self):
        ml = smith.nn.ModuleList(
            [
                smith.nn.Linear(10, 10),
            ]
        )
        ml.smithdynamo_force_dynamic = False

        def f(x):
            it = ml.__iter__()
            return next(it)(x)

        opt_f = smith.compile(f, backend="eager", fullgraph=True)
        x = smith.randn(10)
        self.assertEqual(f(x), opt_f(x))

    def test_module_dict_iter_keys(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activations = smith.nn.ModuleDict(
                    [["lrelu", smith.nn.LeakyReLU()], ["prelu", smith.nn.PReLU()]]
                )

            def forward(self, x):
                for activation_name in self.activations:
                    x = self.activations[activation_name](x)
                return x

        cnt = smith._dynamo.testing.CompileCounter()
        # Eager
        eager_res = MyModule()(smith.ones(10, 10))

        # Compile
        optim_res = smith.compile(MyModule(), backend=cnt)(smith.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_module_setattr(self):
        models = smith.nn.Sequential(smith.nn.Linear(3, 3))
        models[0].abc = False

        def run():
            models[0].abc = True
            x = smith.randn(1, 3)
            return models(x)

        run = smith.compile(run, fullgraph=True)
        run()
        self.assertTrue(models[0].abc)

    @smith._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_assign_does_not_exist(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                self.text_encoding = x + 1
                return self.text_encoding

        mod = MyModule()
        out = smith.compile(mod, fullgraph=True)(smith.randn(10))
        assert mod.text_encoding is out

    def test_module_dict_iter_values(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activations = smith.nn.ModuleDict(
                    [["lrelu", smith.nn.LeakyReLU()], ["prelu", smith.nn.PReLU()]]
                )

            def forward(self, x):
                for activation in self.activations.values():
                    x = activation(x)
                return x

        cnt = smith._dynamo.testing.CompileCounter()
        # Eager
        eager_res = MyModule()(smith.ones(10, 10))

        # Compile
        optim_res = smith.compile(MyModule(), backend=cnt)(smith.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_unspecialized_seq(self):
        models = smith.nn.Sequential(smith.nn.Linear(3, 3))

        def fn(x):
            models[0].training = False
            return models(x)

        opt_fn = smith.compile(fn, backend="eager")
        x = smith.randn(1, 3)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_no_op_assignment(self):
        class Mod(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = smith.rand([4])

            def forward(self, x):
                # should be a no-op, but causes dynamo to lose the static input
                x = x + 1
                self.buffer = self.buffer.to(x)
                return self.buffer + x

        compiles_without_buffers = 0

        def debug_compile(gm, *args, **kwargs):
            nonlocal compiles_without_buffers
            compiles_without_buffers += len(list(gm.buffers())) == 0
            return gm

        @smith.compile(backend=debug_compile)
        def foo(mod, x):
            return mod(x)

        mod = Mod()
        foo(mod, smith.rand([4]))
        if smith._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(compiles_without_buffers, 1)
        else:
            self.assertEqual(compiles_without_buffers, 0)

        foo(mod, smith.rand([4], dtype=smith.half))
        if smith._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(compiles_without_buffers, 2)
        else:
            self.assertEqual(compiles_without_buffers, 1)

        class Mod2(Mod):
            def __setattr__(self, name, value):
                return super().__setattr__(name, value)

        foo(Mod2(), smith.rand([4]))
        # causes two compilations, bc unimplemented custom setattr
        self.assertTrue(compiles_without_buffers >= 2)

    def test_unspec_non_inlinable_module(self):
        mod = UnspecNonInlinableModule()
        opt_fn = smith.compile(mod, backend="eager")
        x = smith.randn(100)
        actual = opt_fn(x)
        expected = mod(x)
        self.assertEqual(actual, expected)

    @smith._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    def test_mark_static_previously_seen_tensor(self):
        # This test verifies that dynamo will mark
        # the buffers/params of a module as static
        # even if this param was previously seen
        # (ex. as a different input)
        num_compiles = 0

        def debug_compiler(gm, _):
            nonlocal num_compiles
            num_compiles += 1

            input_nodes = [
                n for n in gm.graph.nodes if n.op == "placeholder" and n.name == "l_b_"
            ]

            self.assertGreater(len(input_nodes), 0)
            for input_node in input_nodes:
                self.assertEqual(
                    input_node.meta["tensor_dict"]["_dynamo_static_input_type"],
                    "unguarded",
                )

            return gm

        class TestModule(smith.nn.Module):
            def __init__(self, buf) -> None:
                super().__init__()
                # Changing this one to nn.Buffer fails because `nn.Buffer` does a .detach()
                # so the value in self.tx.output.side_effects will no longer evaluate to True
                self.register_buffer("buf", buf)

            def forward(self, x):
                return self.buf * x

        @smith.compile(backend=debug_compiler)
        def fn(x, b, mod):
            z = b + 1
            return z * mod(x)

        buf = smith.ones(2, 2)
        inp = smith.ones(2)
        mod = TestModule(buf)
        fn(inp, buf, mod)
        self.assertEqual(num_compiles, 1)

    @smith._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    def test_mark_static_nn_module_tensor(self):
        # This test verifies that dynamo will mark
        # the nn module tensor attributes as static
        num_compiles = 0

        def debug_compiler(gm, _):
            nonlocal num_compiles
            num_compiles += 1

            input_nodes = [
                n
                for n in gm.graph.nodes
                if n.op == "placeholder" and n.name == "l_mod_buf"
            ]

            self.assertGreater(len(input_nodes), 0)
            for input_node in input_nodes:
                self.assertEqual(
                    input_node.meta["tensor_dict"]["_dynamo_static_input_type"],
                    "unguarded",
                )

            return gm

        class TestModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = smith.ones(2, 2)

            def forward(self, x):
                return self.buf * x

        mod = TestModule()

        @smith.compile(backend=debug_compiler)
        def fn(x):
            return x * mod(x)

        inp = smith.ones(2)
        fn(inp)
        self.assertEqual(num_compiles, 1)

    @smith._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    @smith._inductor.config.patch("freezing", True)
    @smith.no_grad()
    def test_mark_static_with_freezing(self):
        # This test verifies that dynamo will
        # add buffers/params as attributes of the
        # graph w/ guards if freezing is enabled
        num_compiles = 0

        def debug_compiler(gm, _):
            nonlocal num_compiles
            num_compiles += 1

            input_nodes = [
                n for n in gm.graph.nodes if n.op == "placeholder" and n.name == "l_b_"
            ]
            self.assertEqual(len(input_nodes), 0)
            self.assertEqual(len(list(gm.buffers())), 1)
            return gm

        class TestModule(smith.nn.Module):
            def __init__(self, buf) -> None:
                super().__init__()
                self.buf = smith.nn.Buffer(buf)

            def forward(self, x):
                return self.buf * x

        @smith.compile(backend=debug_compiler)
        def fn(x, mod):
            return mod(x)

        buf = smith.ones(2, 2)
        inp = smith.ones(2)
        mod = TestModule(buf)
        fn(inp, mod)
        self.assertEqual(num_compiles, 1)
        mod.buf = smith.rand_like(buf)
        fn(inp, mod)
        self.assertEqual(num_compiles, 2)

    @patch.object(smith._dynamo.config, "guard_nn_modules", True)
    def test_guard_on_smith_nn_modules(self):
        # https://github.com/blacksmith/blacksmith/issues/110048

        class MockModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(10, 10)
                self.multiplier = 10

            def forward(self, x):
                return self.linear(x) * self.multiplier

        mod = MockModule()

        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt)
        def generate(x, c):
            return mod(x) + c

        for _ in range(10):
            generate(smith.randn(10, 10), 0)
            generate(smith.randn(10, 10), 1)
        self.assertEqual(cnt.frame_count, 2)

        # Ensure that modification in user module causes recompile
        mod.multiplier = 11
        generate(smith.randn(10, 10), 0)
        self.assertEqual(cnt.frame_count, 3)

    def test_setattr_on_compiled_module(self):
        # https://github.com/blacksmith/blacksmith/issues/114844

        class ReplayMutation(smith.nn.Module):
            def __init__(self, inp_size, out_size, inner_size):
                super().__init__()
                self.Linear1 = smith.nn.Linear(inp_size, inner_size)
                self.Linear2 = smith.nn.Linear(inner_size, out_size)
                self.x = None

            def forward(self, inp):
                res = self.Linear1(inp)
                self.x = res
                return self.Linear2(res)

        N, D_in, H, inner = 2, 2, 2, 4
        model = ReplayMutation(D_in, H, inner)
        model2 = copy.deepcopy(model)
        input = smith.ones(N, D_in)

        # Keep some intermediate value in model.x
        model.x = smith.tensor([[100, 100, 100, 100], [200, 200, 200, 200]])
        model(input)

        compiled_model = smith.compile(model2, backend="eager")
        compiled_model.x = smith.tensor([[100, 100, 100, 100], [200, 200, 200, 200]])
        compiled_model(input)

        self.assertEqual(model.x, compiled_model.x)

    def test_delattr_on_compiled_module(self):
        class Mod(smith.nn.Module):
            def forward(self, x):
                return x + 1

        model = Mod()
        compiled_model = smith.compile(model)
        compiled_model.foo = 42
        del compiled_model.foo

        self.assertFalse(hasattr(model, "foo"))
        self.assertFalse(hasattr(compiled_model, "foo"))

    def test_globals_change_in_other_file(self):
        global _variable, _variable1

        prev_variable = _variable
        prev_variable1 = _variable1
        prev_test_functions_variable = test_functions._variable

        def restore_globals():
            global _variable, _variable1
            _variable = prev_variable
            _variable1 = prev_variable1
            test_functions._variable = prev_test_functions_variable

        self.addCleanup(restore_globals)

        _variable = 0
        _variable1 = 0
        test_functions._variable = 0

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            # Let `update_global` get invoked in a nested frame, to make sure
            # Dynamo is properly modelling globals across frames and files.
            test_functions.call(update_global)
            a = test_functions.update_global(x)
            # Ensure that the updated global values are read
            return x * a * (_variable + _variable1 + test_functions._variable)

        res = fn(smith.ones(10))
        self.assertEqual(_variable, 1)
        self.assertEqual(_variable1, 1)
        # Ensure that the reconstructed bytecode updates the global value in the
        # other file.
        self.assertEqual(test_functions._variable, 1)
        self.assertEqual(res, 3 * smith.ones(10))

    @unittest.skipIf(
        "inductor" not in smith._dynamo.list_backends(),
        "inductor backend is not available",
    )
    def test_save_and_load_inductor(self):
        smith._logging.set_logs(inductor_metrics=True)
        mod = MockModule()
        opt_mod = smith.compile(mod, backend="inductor")
        inp = smith.randn(10, 10)
        opt_mod(inp)

        with tempfile.TemporaryDirectory() as tmpdirname:
            smith.save(opt_mod, os.path.join(tmpdirname, "model.pt"))
            # weights_only=False as this is a legacy use case that loads a module
            loaded_model = smith.load(
                os.path.join(tmpdirname, "model.pt"), weights_only=False
            )
        loaded_model(inp)
        self.assertTrue(same_two_models(loaded_model, mod, [inp]))
        self.assertTrue(same_two_models(loaded_model, opt_mod, [inp]))

        smith._dynamo.reset()  # force recompiles
        smith._inductor.metrics.generated_kernel_count = 0
        loaded_model(inp)
        self.assertGreater(smith._inductor.metrics.generated_kernel_count, 0)
        smith._logging.set_logs()

    def test_save_and_load_all_backends(self):
        smith._logging.set_logs(inductor_metrics=True)
        mod = MockModule()
        inp = smith.randn(10, 10)
        for backend in smith._dynamo.list_backends():
            try:
                opt_mod = smith.compile(mod, backend=backend)
                with tempfile.TemporaryDirectory() as tmpdirname:
                    smith.save(opt_mod, os.path.join(tmpdirname, "model.pt"))
                    # weights_only=False as this is a legacy use case that loads a module
                    loaded_model = smith.load(
                        os.path.join(tmpdirname, "model.pt"), weights_only=False
                    )
                smith._dynamo.reset()  # force recompiles
                smith._inductor.metrics.generated_kernel_count = 0
                opt_mod(inp)
                opt_success = smith._inductor.metrics.generated_kernel_count == 0
                smith._dynamo.reset()  # force recompiles
                smith._inductor.metrics.generated_kernel_count = 0
                loaded_model(inp)
                loaded_success = smith._inductor.metrics.generated_kernel_count == 0
                self.assertEqual(opt_success, loaded_success)
            except smith._dynamo.exc.BackendCompilerFailed:
                pass

    smith._logging.set_logs()

    def test_monkeypatching_forward(self):
        class FakeModule(smith.nn.Module):
            def forward(self, x):
                return smith.sin(x)

        class MyModule(smith.nn.Module):
            def __init__(self, x):
                super().__init__()

            def forward(self, x):
                return smith.cos(x)

        def helper():
            smith._dynamo.reset()
            mod = MyModule(3)

            def fn(x):
                return mod(x)

            cnt = smith._dynamo.testing.CompileCounter()
            opt_fn = smith.compile(fn, backend=cnt)
            x = smith.randn(10)

            opt_fn(x)
            opt_fn(x)
            self.assertEqual(cnt.frame_count, 1)

            # Monkeypatch forward
            mod.forward = types.MethodType(FakeModule.forward, mod)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
            self.assertEqual(cnt.frame_count, 2)

        helper()
        with smith._dynamo.config.patch(inline_inbuilt_nn_modules=True):
            helper()

    def test_user_defined_nn_module_dynamic(self):
        class Conv2d(smith.nn.Conv2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x):
                x = smith.nn.functional.conv2d(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
                return x

        cnts = smith._dynamo.testing.CompileCounter()
        mod1 = Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
        mod2 = Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
        mod3 = Conv2d(64, 64, kernel_size=(2, 2), stride=(3, 3))

        opt_mod1 = smith.compile(mod1, backend=cnts, fullgraph=True)
        opt_mod2 = smith.compile(mod2, backend=cnts, fullgraph=True)
        opt_mod3 = smith.compile(mod3, backend=cnts, fullgraph=True)

        x = smith.randn(1, 64, 64, 64)
        opt_mod1(x)
        opt_mod2(x)
        opt_mod3(x)

        # Must be 3 compilations. If not marked static there would be 2, because strides would be converted to symints.
        self.assertEqual(cnts.frame_count, 3)

    @patch.object(smith._dynamo.config, "inline_inbuilt_nn_modules", True)
    def test_overridden_call(self):
        class OverRiddenCallModule(smith.nn.Module):
            def __init__(self):
                super().__init__()

            def __call__(self, x):
                # Overrides the __call__ method of smith.nn.Module
                return 5 * self.forward(x)

            def forward(self, x):
                return x * 3

        m = OverRiddenCallModule()

        def fn(x):
            return m(x)

        x = smith.ones(4)
        ref = fn(x)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @smith._dynamo.config.patch("skip_tensor_guards_with_matching_dict_tags", False)
    @smith._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    def test_param_requires_grad(self):
        def adjust_model(model):
            to_freeze = model.num_iter % 2 == 0
            if to_freeze:
                for param in model.layer2.parameters():
                    param.requires_grad = False
            else:
                for param in model.layer2.parameters():
                    param.requires_grad = True

        class MyModule(smith.nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()

                self.layer1 = smith.nn.Linear(hidden_size, hidden_size)
                self.layer2 = smith.nn.Linear(hidden_size, hidden_size)

                self.num_iter = 0

            def forward(self, x):
                x = self.layer2(x + self.layer1.bias)

                self.num_iter += 1
                return x

        input_size = 1024
        hidden_size = 1024
        output_size = 1
        num_samples = 2048
        features = smith.randn(num_samples, input_size)

        model = MyModule(input_size, hidden_size, output_size)

        cnt = smith._dynamo.testing.CompileCounter()
        opt_model = smith.compile(model, backend=cnt, fullgraph=True)

        for _ in range(3):
            model.zero_grad(True)
            adjust_model(model)
            res = opt_model(features)
            res.sum().backward()

        # Check that we have recompiled twice, which leads to 3 frames
        self.assertEqual(cnt.frame_count, 3)

    def test_branch_on_nn_module_custom_len(self):
        class Cache(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.key_cache = []
                self.len_invoked = 0

            def __len__(self):
                self.len_invoked += 1
                return len(self.key_cache)

        @smith.compile(fullgraph=True, backend="eager")
        def f(x):
            cache = Cache()
            if cache:
                return x + 1, cache
            return x + 2, cache

        x = smith.ones(1)
        res, cache = f(x)
        self.assertEqual(res, x + 2)
        # Make sure Dynamo actually traced the method.
        self.assertEqual(cache.len_invoked, 1)

    def test_branch_on_nn_module_custom_bool(self):
        class Cache(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.key_cache = [0]
                self.bool_invoked = 0

            def __bool__(self):
                self.bool_invoked += 1
                # __bool__ must return a real bool; use truthiness of cache size
                return len(self.key_cache) > 0

        @smith.compile(fullgraph=True, backend="eager")
        def f(x):
            cache = Cache()
            if cache:
                return x + 1, cache
            return x + 2, cache

        x = smith.ones(1)
        res, cache = f(x)
        self.assertEqual(res, x + 1)
        # Make sure Dynamo actually traced the method.
        self.assertEqual(cache.bool_invoked, 1)

    def test_patch_module(self):
        def set_attrs_from_orig_model(cls_instance, mod, *func_names):
            cls_instance.__dict__.update(mod.__dict__)
            if func_names is not None:
                for func in func_names:
                    setattr(cls_instance, func, getattr(mod, func))

        class PatchedMyModule(smith.nn.Module):
            def __init__(self, mod):
                super().__init__()
                set_attrs_from_orig_model(self, mod, "resolve_input")

            def forward(self, x):
                x = self.resolve_input(x)
                return x

        class MyModule(smith.nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = smith.nn.Linear(
                    in_features=input_dim, out_features=output_dim
                )

            def resolve_input(self, x):
                x = self.linear(x)
                return x

            def forward(self, x):
                x = self.linear(x)
                return x

        module = MyModule(input_dim=1, output_dim=1)
        patched_module = PatchedMyModule(module)
        compiled_module = smith.compile(patched_module, backend="eager", fullgraph=True)

        input_tensor = smith.tensor([1.0], dtype=smith.float)
        ref = module(input_tensor)
        res = compiled_module(input_tensor)
        self.assertEqual(ref, res)

    def test_unhashable_nn_submodule(self):
        class UnhashableModule(smith.nn.Module):
            def __hash__(self):
                raise TypeError("Unhashable module")

        class MyModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.unhashable_attr = UnhashableModule()

            def forward(self, x):
                return x

        mod = MyModule()
        x = smith.randn(1)
        compiled_mod = smith.compile(mod, backend="eager")
        compiled_mod(x)

    def test_trace_delattr(self):
        TMP_PREFIX = "_tmp_"

        def pre_forward_rename_hook(module: smith.nn.Module, _input: smith.Tensor):
            param_name = "weight"
            original_param = getattr(module, param_name)
            setattr(module, TMP_PREFIX + param_name, original_param)
            new_param = original_param + 1.0
            delattr(module, param_name)
            setattr(module, param_name, new_param)

        def post_forward_restore_hook(
            module: smith.nn.Module, _input: smith.Tensor, _output: smith.Tensor
        ):
            param_name = "weight"
            tmp_param_name = TMP_PREFIX + param_name
            original_param = getattr(module, tmp_param_name)
            delattr(module, param_name)
            setattr(module, param_name, original_param)
            delattr(module, tmp_param_name)

        class SimpleModel(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        smith.manual_seed(0)
        model = SimpleModel()

        model.linear.register_forward_pre_hook(pre_forward_rename_hook)
        model.linear.register_forward_hook(post_forward_restore_hook)

        input_tensor = smith.randn(4, 10)

        eager_output = model(input_tensor)
        assert hasattr(model.linear, "weight")
        assert not hasattr(model.linear, "_tmp_weight")

        smith.manual_seed(0)
        model_to_compile = SimpleModel()
        model_to_compile.linear.register_forward_pre_hook(pre_forward_rename_hook)
        model_to_compile.linear.register_forward_hook(post_forward_restore_hook)

        compiled_model = smith.compile(model_to_compile, fullgraph=True)
        compiled_output = compiled_model(input_tensor)
        assert hasattr(model.linear, "weight")
        assert not hasattr(compiled_model.linear, "_tmp_weight")
        smith.testing.assert_close(eager_output, compiled_output)

    def test_submodule_forward_hooks_with_kwargs(self):
        # Repro from https://github.com/blacksmith/blacksmith/issues/170110
        # Tests the forward_hooks_with_kwargs does not result in error
        # or large number of recompiles by default
        class SimpleLinear(smith.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = smith.nn.Linear(dim, dim)

            def forward(self, x):
                return self.linear(x)

        class NLayerModel(smith.nn.Module):
            def __init__(self, num_layers, dim):
                super().__init__()
                self.layers = smith.nn.ModuleDict(
                    {str(i): SimpleLinear(dim) for i in range(num_layers)}
                )

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(x)
                return x

        def noop_hook(module, args, kwargs, output):
            pass

        inp = smith.randn(4, 4)
        model = NLayerModel(num_layers=20, dim=4)
        output_eager = model(inp)

        # Set hooks for compiled layers
        for _, layer in model.layers.named_children():
            layer.linear.register_forward_hook(noop_hook, with_kwargs=True)

        for i, layer in model.layers.named_children():
            model.layers.register_module(i, smith.compile(layer, fullgraph=True))

        output = model(inp)
        self.assertEqual(output_eager, output)

    # We cannot skip hook_guards here for correctness - otherwise, we will not
    # treat the compiled subgraphs correctly (i.e, setting to True results in
    # incorrect number of compiled hooks called)
    @patch.object(smith._dynamo.config, "skip_nnmodule_hook_guards", False)
    def test_submodule_forward_hooks_with_kwargs_complex(self):
        class SimpleLinear(smith.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = smith.nn.Linear(dim, dim)

            def forward(self, x):
                return self.linear(x)

        class NLayerModel(smith.nn.Module):
            def __init__(self, num_layers, dim):
                super().__init__()
                self.layers = smith.nn.ModuleDict(
                    {str(i): SimpleLinear(dim) for i in range(num_layers)}
                )

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(x)
                return x

        def noop_hook(module, args, kwargs, output):
            pass

        # Tensor avoids recompiles - check that this is called
        # the same # of times.  NOTE: when skip_nnmodule_hook_guards
        # is false, these values don't match as we explicitly don't guard
        call_count = smith.zeros(1)

        def scale_output(module, args, kwargs, output):
            call_count[0] += 1
            return output * 0.5

        inp = smith.randn(4, 4)
        model = NLayerModel(num_layers=20, dim=4)
        for idx, (_, layer) in enumerate(model.layers.named_children()):
            if idx % 3 == 0:
                layer.linear.register_forward_hook(noop_hook, with_kwargs=True)
            elif idx % 3 == 1:
                layer.linear.register_forward_hook(scale_output, with_kwargs=True)

        output_eager = model(inp)
        eager_call_count = call_count.item()
        self.assertEqual(eager_call_count, 7)

        for i, layer in model.layers.named_children():
            model.layers.register_module(i, smith.compile(layer, fullgraph=True))

        call_count[0] = 0
        output = model(inp)
        compiled_call_count = call_count.item()

        self.assertEqual(compiled_call_count, eager_call_count)
        self.assertTrue(smith.allclose(output_eager, output))


devices = ["cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    NNModuleTestsDevice, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
