# Owner(s): ["module: inductor"]

import collections
import unittest

import smith
import smith._inductor
import smith._inductor.fx_passes.group_batch_fusion
from smith._dynamo.utils import counters
from smith._inductor.test_case import run_tests, TestCase
from smith.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False


class TestHighwaySelfGating(smith.nn.Module):
    def __init__(
        self,
        d_model: int,
        size: int,
        device="cuda",
    ) -> None:
        super().__init__()
        self.size = size
        self.device = device
        self.gating_proj = smith.nn.Linear(d_model, d_model).to(self.device)
        self.transform_proj = smith.nn.Linear(d_model, d_model).to(self.device)
        self.gating_func = smith.nn.Sigmoid().to(self.device)

        self.d_model = d_model

    def forward(
        self,
        inputs: list[smith.Tensor],
    ) -> smith.Tensor:
        results = []
        for i in range(self.size):
            x = inputs[i]
            gating_proj = self.gating_proj(x)
            transform_proj = self.transform_proj(x)
            x = gating_proj * self.gating_func(transform_proj)
            results.append(x)

        return smith.cat(results, dim=-1)


class MyModule(smith.nn.Module):
    def __init__(self, z: int, has_bias: bool, device="cuda") -> None:
        super().__init__()
        self.z = z
        self.device = device
        self.seq_len = 10
        self.seq1 = [
            smith.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]
        self.seq2 = [
            smith.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]
        self.seq3 = [
            smith.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]

    def forward(self, x: smith.Tensor) -> smith.Tensor:
        x1 = [x + 0.1 * i for i in range(self.seq_len)]
        x2 = [self.seq1[i](x1[i]) for i in range(self.seq_len)]
        x3 = [x2[i] - 0.1 * i for i in range(self.seq_len)]
        x4 = [x1[i] for i in range(3)] + [x3[i] for i in range(3, self.seq_len)]
        x5 = [self.seq2[i](x4[i]) for i in range(self.seq_len)]
        x6 = [x5[i] + 0.1 * (self.seq_len - i) for i in range(self.seq_len)]
        x7 = (
            [x1[i] for i in range(4)]
            + [x3[i] for i in range(6, 8)]
            + [x6[i] for i in range(4)]
        )
        x8 = [self.seq3[i](x7[i]) for i in range(self.seq_len)]
        x9 = smith.cat(x8, dim=1)
        return x9


class MyModule2(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear0 = smith.nn.Linear(6, 8)
        self.linear1 = smith.nn.Linear(8, 8)
        self.linear2 = smith.nn.Linear(10, 8)
        self.linear3 = smith.nn.Linear(6, 8)
        self.linear4 = smith.nn.Linear(8, 8)
        self.linear5 = smith.nn.Linear(10, 8)
        self.bn0 = smith.nn.BatchNorm1d(8)
        self.bn1 = smith.nn.BatchNorm1d(8)
        self.bn2 = smith.nn.BatchNorm1d(8)

    def forward(self, x: smith.Tensor) -> smith.Tensor:
        t = smith.split(x, [6, 8, 10], dim=1)
        a0 = self.bn0(self.linear0(t[0] + 0.1))
        a1 = self.bn1(self.linear1(t[1] + 0.2))
        a2 = self.bn2(self.linear2(t[2] + 0.3))
        a3 = self.linear3(smith.sin(t[0]))
        a4 = self.linear4(smith.cos(t[1]))
        a5 = self.linear5(smith.sin(t[2] * 0.5))

        b = smith.cat([a0, a1, a2, a3, a4, a5])
        return smith.sigmoid(b)


class MyModule3(smith.nn.Module):
    def __init__(self, device, has_weight=True, has_bias=True):
        super().__init__()
        self.device = device
        self.scale0 = smith.nn.ParameterList(
            [smith.nn.Parameter(smith.randn(10)) for _ in range(5)]
        ).to(self.device)
        self.bias0 = smith.nn.ParameterList(
            [smith.nn.Parameter(smith.randn(10)) for _ in range(5)]
        ).to(self.device)
        self.scale1 = (
            smith.nn.ParameterList(
                [smith.nn.Parameter(smith.randn(5, 10)) for _ in range(5)]
            ).to(self.device)
            if has_weight
            else [None for _ in range(5)]
        )
        self.bias1 = (
            smith.nn.ParameterList(
                [smith.nn.Parameter(smith.randn(5, 10)) for _ in range(5)]
            ).to(self.device)
            if has_bias
            else [None for _ in range(5)]
        )

    def forward(self, x):
        l1_out = smith.split(x.to(self.device), 10, dim=2)
        post_l1 = [
            smith.nn.functional.layer_norm(
                l1_out[i], (10,), weight=self.scale0[i], bias=self.bias0[i]
            )
            for i in range(len(l1_out))
        ]
        l1_out = smith.cat(post_l1, dim=2)

        l2_out = smith.split(l1_out, 10, dim=2)
        post_l2 = [
            smith.nn.functional.layer_norm(
                l2_out[i], (5, 10), weight=self.scale1[i], bias=self.bias1[i]
            )
            for i in range(len(l2_out))
        ]

        return smith.cat(post_l2, dim=2)


class MyModule4(smith.nn.Module):
    def __init__(self, z, device, has_bias):
        super().__init__()
        self.z = z
        self.device = device
        self.has_bias = has_bias
        self.seq_len = 10
        self.weights1 = [
            smith.nn.Parameter(smith.randn(z - i % 5, z)).to(self.device)
            for i in range(self.seq_len)
        ]
        self.weights2 = [
            smith.nn.Parameter(smith.randn(z - i % 5, z)).to(self.device)
            for i in range(self.seq_len)
        ]

        if has_bias:
            self.biases1 = [
                smith.nn.Parameter(smith.randn(z - i % 5)).to(self.device)
                for i in range(self.seq_len)
            ]
            self.biases2 = [
                smith.nn.Parameter(smith.randn(z - i % 5)).to(self.device)
                for i in range(self.seq_len)
            ]

    def forward(self, x):
        x = x + 1.2
        x1 = [
            smith.nn.functional.linear(
                x, self.weights1[i], self.biases1[i] if self.has_bias else None
            )
            for i in range(self.seq_len)
        ]
        x2 = smith.cat(x1, dim=1)
        x3 = smith.split(x2, 10, dim=1)
        x4 = smith.cat(x3)
        x5 = [
            smith.nn.functional.linear(
                x4, self.weights2[i], self.biases2[i] if self.has_bias else None
            )
            for i in range(self.seq_len)
        ]
        x6 = smith.cat(x5, dim=1)
        return smith.sigmoid(x6)


class MyModule5(smith.nn.Module):
    def __init__(self, device, has_bias=True):
        super().__init__()
        self.device = device

        self.weights = smith.nn.ParameterList(
            [smith.nn.Parameter(smith.randn(50, 100)).to(self.device) for _ in range(5)]
        )

        self.biases = (
            ([smith.nn.Parameter(smith.randn(50)).to(self.device) for _ in range(5)])
            if has_bias
            else [None for _ in range(5)]
        )

    def forward(self, x):
        l1_out = smith.split(x.to(self.device), 100, dim=1)
        l1_linear = [
            smith.nn.functional.linear(l1_out[i], self.weights[i], self.biases[i])
            for i in range(len(l1_out))
        ]
        l1_out = smith.cat(l1_linear, dim=1)
        return smith.sin(l1_out)


class TestPoitwiseOps(smith.nn.Module):
    def __init__(self, device, has_bias=True):
        super().__init__()
        self.device = device

    def forward(self, x):
        inputs = smith.split(x.to(self.device), 500, dim=1)
        x_split = smith.split(inputs[0].to(self.device), 50, dim=1)
        y_split = smith.split(inputs[1].to(self.device), 50, dim=1)
        sigmoid_1 = [smith.sigmoid(x_split[i]) for i in range(len(x_split))]
        sigmoid_2 = [smith.sigmoid(y_split[i]) for i in range(len(y_split))]
        relu_1 = [smith.nn.functional.relu(sigmoid_1[i]) for i in range(len(sigmoid_1))]
        relu_2 = [smith.nn.functional.relu(sigmoid_2[i]) for i in range(len(sigmoid_2))]
        add = [smith.add(relu_1[i], relu_2[i]) for i in range(len(relu_1))]
        mul = [smith.mul(add[i], add[i]) for i in range(len(add))]
        sub = [smith.sub(mul[i], mul[i]) for i in range(len(mul))]
        div = [smith.div(sub[i], sub[i]) for i in range(len(sub))]
        return smith.cat(div, dim=1)


class TestPoitwiseOpsPostGrad(smith.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        inputs = smith.ops.aten.split(x.to(self.device), 500, dim=1)
        x_split = smith.ops.aten.split(inputs[0].to(self.device), 50, dim=1)
        y_split = smith.ops.aten.split(inputs[1].to(self.device), 50, dim=1)
        tanh_1 = [smith.ops.aten.tanh(x_split[i]) for i in range(len(x_split))]
        tanh_2 = [smith.ops.aten.tanh(y_split[i]) for i in range(len(y_split))]
        sigmoid_1 = [smith.ops.aten.sigmoid(tanh_1[i]) for i in range(len(tanh_1))]
        sigmoid_2 = [smith.ops.aten.sigmoid(tanh_2[i]) for i in range(len(tanh_2))]
        relu_1 = [smith.ops.aten.relu(sigmoid_1[i]) for i in range(len(sigmoid_1))]
        relu_2 = [smith.ops.aten.relu(sigmoid_2[i]) for i in range(len(sigmoid_2))]
        add = [smith.ops.aten.add(relu_1[i], relu_2[i]) for i in range(len(relu_1))]
        return smith.cat(add, dim=1)


class TestMathOps(smith.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        inputs = [x.to(self.device) for i in range(10)]
        others = [x.to(self.device) for i in range(10)]
        clamp_input = [x.clamp(min=-1000.1, max=1000.1) for x in inputs]
        clamp_other = [x.clamp(min=-1000.1, max=1000.1) for x in others]
        nan_to_num_input = [smith.nan_to_num(x, 0.0) for x in clamp_input]
        nan_to_num_other = [smith.nan_to_num(x, 0.0) for x in clamp_other]
        detach_input = [x.detach() for x in nan_to_num_input]
        detach_other = [x.detach() for x in nan_to_num_other]
        stack_input = smith.stack(detach_input, dim=0)
        stack_other = smith.stack(detach_other, dim=0)
        return smith.stack((stack_input, stack_other), dim=0)


class TestDropout(smith.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(
        self, x: smith.Tensor
    ) -> tuple[smith.Tensor, smith.Tensor, smith.Tensor, smith.Tensor]:
        split = x.split([20, 20, 20, 20, 20], 1)
        getitem_1 = split[0]
        getitem_2 = split[1]
        getitem_3 = split[2]
        getitem_4 = split[3]
        getitem_5 = split[4]
        dropout = smith.nn.functional.dropout(
            getitem_1, p=0.05, training=True, inplace=False
        )
        dropout_1 = smith.nn.functional.dropout(
            getitem_2, p=0.05, training=True, inplace=False
        )
        dropout_2 = smith.nn.functional.dropout(
            getitem_3, p=0.05, training=True, inplace=False
        )
        dropout_3 = smith.nn.functional.dropout(
            getitem_4, p=0.05, training=True, inplace=False
        )
        dropout_4 = smith.nn.functional.dropout(
            getitem_5, p=0.05, training=True, inplace=False
        )
        return (dropout, dropout_1, dropout_2, dropout_3, dropout_4)


class TestGroupBatchFusion(TestCase):
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict:
            key2 = "_orig_mod." + key1
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            if not smith.allclose(ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol):
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    @requires_gpu()
    @unittest.skipIf(not has_fbgemm, "requires fbgemm")
    @smith._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "group_linear": {"require_fbgemm": True},
        },
    )
    def test_group_linear_fusion(self):
        z = 10
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule(z, has_bias).to(GPU_TYPE)
            input = [smith.randn(z, z, device=GPU_TYPE)]
            traced = smith.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(
                counters["inductor"]["group_linear"],
                2,
            )
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced)
            self.compare_gradients(module, traced)
            self.assertEqual(
                counters["inductor"]["group_linear"],
                4,
            )
            counters.clear()

    @requires_gpu()
    @unittest.skipIf(not has_fbgemm, "requires fbgemm")
    @smith._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "group_linear": {"require_fbgemm": True},
        },
    )
    def test_group_linear_fusion_different_shapes(self):
        counters.clear()
        module = MyModule2().eval().to(GPU_TYPE)
        input = [smith.rand(4, 24, device=GPU_TYPE)]
        traced = smith.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(
            counters["inductor"]["group_linear"],
            1,
        )
        self.assertEqual(
            counters["inductor"]["batch_fusion"],
            0,
        )
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["group_linear"],
            2,
        )
        counters.clear()

    @requires_gpu()
    @unittest.skipIf(GPU_TYPE == "mps", "welford_reduce is yet not implemented for MPS")
    @smith._inductor.config.patch(
        pre_grad_fusion_options={"batch_layernorm": {}},
        post_grad_fusion_options={},
    )
    def test_batch_layer_norm_fusion(self):
        for has_weight in [True, False]:
            for has_bias in [True, False]:
                counters.clear()
                module = MyModule3(GPU_TYPE, has_weight, has_bias).to(GPU_TYPE)
                input = [smith.randn(2, 5, 50, device=GPU_TYPE)]
                traced = smith.compile(module)
                ref = module(*input)
                res = traced(*input)
                self.compare_pred(module, traced, input)
                self.assertEqual(counters["inductor"]["batch_layernorm"], 2)
                ref.sum().backward()
                res.sum().backward()
                self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
                self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
                counters.clear()

    @requires_gpu()
    @smith._inductor.config.patch(
        pre_grad_fusion_options={"batch_linear_lhs": {}},
        post_grad_fusion_options={},
    )
    def test_batch_linear_lhs_fusion(self):
        z = 10
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule4(z, GPU_TYPE, has_bias)
            input = [smith.randn(20, z, device=GPU_TYPE)]
            traced = smith.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(counters["inductor"]["batch_linear_lhs"], 2)
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
            self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
            counters.clear()

    @requires_gpu()
    @smith._inductor.config.patch(
        pre_grad_fusion_options={"batch_linear": {}},
        post_grad_fusion_options={},
    )
    def test_batch_linear_pre_grad_fusion(self):
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule5(GPU_TYPE, has_bias)
            input = [smith.randn(50, 500, device=GPU_TYPE)]
            traced = smith.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(counters["inductor"]["batch_linear"], 1)
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
            self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
            counters.clear()

    @requires_gpu()
    @smith._inductor.config.patch(
        pre_grad_fusion_options={
            "batch_relu": {},
            "batch_sigmoid": {},
        },
        post_grad_fusion_options={
            "batch_aten_add": {},
            "batch_aten_mul": {},
            "batch_aten_sub": {},
            "batch_aten_div": {},
        },
    )
    def test_pointwise_op_fusion(self):
        counters.clear()
        module = TestPoitwiseOps(GPU_TYPE)
        input = [smith.randn(50, 1000, requires_grad=True, device=GPU_TYPE)]
        traced = smith.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(counters["inductor"]["batch_relu"], 1)
        self.assertEqual(counters["inductor"]["batch_sigmoid"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_add"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_mul"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_sub"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_div"], 1)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_gpu()
    @smith._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "batch_aten_relu": {},
            "batch_aten_sigmoid": {},
            "batch_aten_tanh": {},
            "unbind_stack_aten_pass": {},
        },
    )
    def test_pointwise_op_fusion_post_grad(self):
        counters.clear()
        module = TestPoitwiseOpsPostGrad(GPU_TYPE)
        input = [smith.randn(50, 1000, requires_grad=True, device=GPU_TYPE)]
        traced = smith.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(counters["inductor"]["batch_aten_tanh"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_relu"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_sigmoid"], 1)
        self.assertEqual(counters["inductor"]["unbind_stack_aten_pass"], 2)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_gpu()
    @smith._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "batch_linear_post_grad": {
                "shape_broadcast_batch_linear": True,
                "fuse_nodes_with_same_users": True,
            },
            "batch_aten_mul": {"fuse_nodes_with_same_parent": False},
            "batch_aten_sigmoid": {"fuse_nodes_with_same_parent": True},
            "batch_aten_add": {"fuse_nodes_with_same_parent": True},
            "normalization_aten_pass": {},
            "unbind_stack_aten_pass": {},
        },
    )
    def test_gate_fusion_post_grad(self):
        counters.clear()
        size = 20
        module = TestHighwaySelfGating(d_model=10, size=size, device=GPU_TYPE)
        input = [
            [
                smith.randn(10, 10, requires_grad=True, device=GPU_TYPE)
                for i in range(size)
            ]
        ]
        traced = smith.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(counters["inductor"]["batch_linear_post_grad"], 2)
        self.assertEqual(counters["inductor"]["batch_aten_sigmoid"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_mul"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_add"], 2)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 1)
        self.assertEqual(counters["inductor"]["unbind_stack_aten_pass"], 5)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_gpu()
    @smith._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "batch_detach": {},
            "batch_nan_to_num": {},
            "batch_clamp": {},
            "unbind_stack_pass": {},
            "unbind_stack_to_slices_pass": {},
        },
        post_grad_fusion_options={},
    )
    def test_math_op_fusion(self):
        counters.clear()
        module = TestMathOps(GPU_TYPE)
        input = [
            smith.tensor(
                [float("nan"), float("inf"), -float("inf"), 3.14], device=GPU_TYPE
            )
        ]
        traced = smith.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(counters["inductor"]["normalization_pass"], 3)
        self.assertEqual(counters["inductor"]["batch_clamp"], 1)
        self.assertEqual(counters["inductor"]["batch_detach"], 1)
        self.assertEqual(counters["inductor"]["batch_nan_to_num"], 1)
        self.assertEqual(counters["inductor"]["unbind_stack_to_slices_pass"], 2)
        self.assertEqual(counters["inductor"]["unbind_stack_pass"], 2)
        self.assertTrue(smith.allclose(ref, res))
        counters.clear()

    @requires_gpu()
    @smith._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "batch_dropout": {},
        }
    )
    def test_batch_dropout_pre_grad_fusion(self):
        counters.clear()
        module = TestDropout(GPU_TYPE)
        input = [smith.randn(10, 100, requires_grad=True, device=GPU_TYPE)]
        traced = smith.compile(module)
        module(*input)
        traced(*input)
        self.assertEqual(counters["inductor"]["normalization_pass"], 1)
        self.assertEqual(counters["inductor"]["batch_dropout"], 1)
        counters.clear()


class TestBMMFusionModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.my_modules = smith.nn.ModuleList()
        for _ in range(10):
            self.my_modules.append(smith.nn.Linear(10, 10))

    def forward(self, inputs):
        output = None
        for linear, input in zip(self.my_modules, inputs):
            if output is None:
                output = linear(input)
            else:
                output += linear(input)
        return output


@requires_gpu()
@smith._inductor.config.patch(
    post_grad_fusion_options={"batch_linear_post_grad": {"require_fbgemm": False}}
)
class TestPostGradBatchLinearFusion(TestCase):
    def test_batch_linear_post_grad_fusion(self):
        pt1_module = TestBMMFusionModule().to(GPU_TYPE)
        inputs = []
        for _ in range(10):
            inputs.append(smith.randn(10, 10).to(GPU_TYPE))
        eager_output = pt1_module(inputs)
        pt2_module = smith.compile(pt1_module)
        pt2_output = pt2_module(inputs)
        self.assertTrue(smith.allclose(eager_output, pt2_output))
        self.assertEqual(
            counters["inductor"]["batch_linear_post_grad"],
            2,
        )


class TestFindIndependentSubsetGreedy(TestCase):
    # Helper function to build a Graph from a data description.
    def build_graph(self, desc):
        # desc: {
        #   "n1": ["n2", "n3"],
        #   "n2": ["n3"],
        #   "n3": [],
        # }
        #
        g = smith.fx.Graph()
        lookup = {}
        desc = collections.deque((k, v) for k, v in desc.items())
        unsatisfied = 0
        while desc:
            unsatisfied += 1
            assert unsatisfied <= len(desc)  # cycle or bad input?
            name, v = desc.popleft()
            args = tuple(lookup.get(n) for n in v)
            if None in args:
                desc.append((name, v))
                continue
            node = g.create_node("placeholder", "target", name=name, args=args)
            lookup[name] = node
            unsatisfied = 0
        return g, lookup

    def verify(self, tree, subnodes, min_fuse, max_fuse, expected):
        _, lookup = self.build_graph(tree)
        subnodes = [lookup[n] for n in subnodes]
        expected = [[lookup[n] for n in sub] for sub in expected]
        opts = {
            "min_fuse_set_size": min_fuse,
            "max_fuse_set_size": max_fuse,
        }
        result = list(
            smith._inductor.fx_passes.group_batch_fusion.find_independent_subset_greedy(
                subnodes, opts
            )
        )
        self.assertEqual(expected, result)

    def test_find_independent_subset_greedy(self):
        # First some randomly generated tests.
        self.verify({"n0": (), "n1": ()}, ["n0"], 0, 100, [["n0"]])
        self.verify(
            {"n0": (), "n1": (), "n2": ("n0",)}, ["n1", "n2"], 0, 100, [["n1", "n2"]]
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n0",),
                "n3": (),
                "n4": ("n0", "n1", "n2"),
                "n5": ("n0", "n2", "n4"),
                "n6": ("n3",),
                "n7": ("n4", "n5", "n6", "n1", "n3"),
                "n8": ("n7", "n1", "n3", "n5", "n0"),
                "n9": ("n3", "n4", "n8", "n6", "n5", "n2", "n0", "n7"),
                "n10": ("n0",),
                "n11": ("n4", "n0", "n2", "n3", "n1", "n9"),
                "n12": ("n2", "n3", "n10", "n6", "n9"),
            },
            ["n10", "n5", "n3", "n4", "n9"],
            0,
            100,
            [["n10", "n5", "n3"], ["n4"], ["n9"]],
        )
        self.verify({"n0": (), "n1": (), "n2": ("n0",)}, ["n2"], 0, 100, [["n2"]])
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": (),
                "n4": ("n3", "n1", "n0"),
                "n5": ("n1", "n2", "n4", "n0"),
                "n6": ("n0", "n3", "n2"),
                "n7": ("n6", "n1", "n5", "n4", "n3", "n0"),
                "n8": ("n2", "n7", "n3"),
                "n9": ("n3", "n5", "n6", "n7", "n2", "n1"),
                "n10": ("n8", "n0", "n2", "n4", "n6", "n3"),
                "n11": ("n6", "n5", "n8", "n1", "n3", "n10", "n2"),
                "n12": ("n7", "n4"),
            },
            ["n7"],
            0,
            100,
            [["n7"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n2"),
                "n4": ("n1",),
                "n5": (),
                "n6": ("n5",),
                "n7": ("n1", "n6", "n5", "n2", "n3", "n0"),
                "n8": ("n5", "n7", "n2", "n6"),
                "n9": ("n1",),
                "n10": ("n9",),
                "n11": ("n3", "n4", "n0", "n2"),
                "n12": ("n8", "n9", "n5", "n1"),
                "n13": ("n11", "n4", "n12", "n1", "n9", "n3", "n0"),
            },
            ["n9", "n2", "n8", "n10", "n5", "n6", "n13", "n7", "n3", "n0", "n4"],
            0,
            100,
            [
                ["n9", "n2", "n5", "n0", "n4"],
                ["n8", "n10"],
                ["n6", "n3"],
                ["n13"],
                ["n7"],
            ],
        )
        self.verify({"n0": ()}, ["n0"], 0, 100, [["n0"]])
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": (),
                "n4": ("n1", "n2"),
                "n5": ("n0", "n4", "n1"),
                "n6": ("n1", "n5"),
                "n7": (),
                "n8": ("n7", "n1", "n3", "n5", "n6"),
                "n9": ("n2", "n1", "n8", "n0", "n4", "n7", "n6", "n5"),
                "n10": ("n4", "n7", "n2", "n3", "n8"),
                "n11": (),
                "n12": ("n9", "n7", "n5", "n11", "n8"),
                "n13": (
                    "n5",
                    "n6",
                    "n12",
                    "n3",
                    "n9",
                    "n8",
                    "n4",
                    "n11",
                    "n2",
                    "n10",
                    "n1",
                ),
                "n14": ("n7", "n3", "n12", "n10", "n2", "n0", "n4", "n5"),
                "n15": ("n9", "n5", "n1", "n13", "n8", "n10", "n12", "n7", "n11", "n3"),
                "n16": (
                    "n2",
                    "n4",
                    "n15",
                    "n5",
                    "n0",
                    "n6",
                    "n3",
                    "n8",
                    "n14",
                    "n12",
                    "n9",
                    "n10",
                    "n7",
                    "n13",
                ),
            },
            ["n0", "n3", "n2", "n11", "n1", "n6", "n12", "n5", "n4", "n15", "n8"],
            0,
            100,
            [
                ["n0", "n3", "n2", "n11", "n1"],
                ["n6"],
                ["n12"],
                ["n5"],
                ["n4"],
                ["n15"],
                ["n8"],
            ],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n2", "n1"),
                "n4": ("n2", "n3", "n1"),
                "n5": ("n3", "n1"),
                "n6": ("n1",),
                "n7": ("n5", "n4"),
                "n8": ("n6", "n2"),
            },
            ["n4", "n3", "n1", "n8", "n5", "n6", "n2"],
            0,
            100,
            [["n4", "n8", "n5"], ["n3", "n6"], ["n1", "n2"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n0"),
                "n4": ("n0",),
                "n5": ("n1", "n4"),
                "n6": ("n2", "n1", "n4"),
                "n7": ("n0", "n3"),
                "n8": ("n5", "n0", "n6", "n1", "n4", "n2", "n3"),
                "n9": ("n1", "n4", "n8", "n7", "n5"),
                "n10": ("n9", "n8", "n0", "n2", "n7", "n1", "n3", "n5"),
                "n11": ("n9", "n2", "n6", "n0", "n3"),
                "n12": ("n1", "n4", "n7", "n10", "n5", "n2", "n11", "n6"),
                "n13": ("n9", "n2", "n3", "n0", "n7", "n5", "n10", "n11"),
                "n14": (
                    "n8",
                    "n0",
                    "n3",
                    "n6",
                    "n10",
                    "n1",
                    "n5",
                    "n9",
                    "n12",
                    "n11",
                    "n4",
                ),
                "n15": (
                    "n3",
                    "n10",
                    "n0",
                    "n4",
                    "n9",
                    "n11",
                    "n2",
                    "n13",
                    "n12",
                    "n8",
                    "n5",
                    "n14",
                ),
                "n16": ("n6",),
                "n17": (
                    "n4",
                    "n3",
                    "n14",
                    "n8",
                    "n15",
                    "n16",
                    "n2",
                    "n5",
                    "n7",
                    "n12",
                    "n1",
                    "n0",
                    "n11",
                ),
            },
            ["n17", "n16", "n10", "n4", "n8", "n12", "n6", "n1"],
            0,
            100,
            [["n17"], ["n16", "n10"], ["n4", "n1"], ["n8"], ["n12"], ["n6"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n0",),
                "n3": ("n0", "n1"),
                "n4": ("n0",),
                "n5": ("n0",),
                "n6": ("n5", "n3", "n0", "n2"),
                "n7": (),
                "n8": ("n2", "n5", "n3", "n1", "n7", "n6", "n0"),
                "n9": ("n4",),
                "n10": ("n4", "n5", "n1", "n2", "n0", "n6", "n8", "n9", "n7"),
                "n11": ("n3", "n0", "n9", "n10", "n5", "n1", "n2", "n7", "n4", "n6"),
                "n12": ("n9", "n5"),
            },
            ["n8", "n3", "n1", "n12", "n2", "n5", "n11", "n4", "n10", "n6", "n0"],
            0,
            100,
            [
                ["n8", "n12"],
                ["n3", "n2", "n5", "n4"],
                ["n1", "n0"],
                ["n11"],
                ["n10"],
                ["n6"],
            ],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": (),
                "n4": ("n2", "n3"),
                "n5": ("n1", "n3", "n2", "n4"),
                "n6": ("n5", "n4", "n1", "n3"),
                "n7": ("n5",),
                "n8": ("n5", "n4", "n1"),
                "n9": ("n2", "n3", "n1", "n5", "n7", "n0", "n8"),
                "n10": ("n5", "n3", "n1", "n7", "n8", "n9"),
                "n11": ("n1", "n4", "n2", "n0", "n8", "n9"),
                "n12": ("n4", "n3", "n9"),
                "n13": (
                    "n6",
                    "n10",
                    "n4",
                    "n8",
                    "n0",
                    "n11",
                    "n12",
                    "n7",
                    "n3",
                    "n2",
                    "n1",
                ),
                "n14": ("n4", "n13", "n2"),
                "n15": ("n11", "n7", "n6", "n10", "n14"),
                "n16": ("n15", "n3"),
                "n17": ("n10", "n2", "n7", "n0", "n5", "n6", "n9"),
                "n18": (
                    "n16",
                    "n8",
                    "n6",
                    "n9",
                    "n11",
                    "n12",
                    "n14",
                    "n5",
                    "n13",
                    "n4",
                    "n1",
                ),
            },
            [
                "n1",
                "n0",
                "n16",
                "n6",
                "n15",
                "n9",
                "n7",
                "n4",
                "n3",
                "n11",
                "n13",
                "n17",
                "n12",
                "n18",
            ],
            0,
            100,
            [
                ["n1", "n0", "n4"],
                ["n16", "n17"],
                ["n6", "n9"],
                ["n15"],
                ["n7"],
                ["n3"],
                ["n11", "n12"],
                ["n13"],
                ["n18"],
            ],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n2",),
                "n4": ("n1",),
                "n5": (),
                "n6": ("n1", "n4"),
                "n7": ("n5", "n1"),
                "n8": ("n6",),
                "n9": ("n6", "n1", "n2", "n0"),
                "n10": ("n0", "n7"),
                "n11": ("n0", "n4", "n3", "n5"),
                "n12": ("n9", "n8", "n7", "n4", "n0"),
            },
            ["n8", "n9", "n11", "n2", "n4", "n0", "n7", "n5", "n1"],
            0,
            100,
            [["n8", "n9", "n11", "n7"], ["n2", "n4", "n0", "n5"], ["n1"]],
        )
        self.verify(
            {"n0": (), "n1": (), "n2": (), "n3": ("n0",), "n4": ("n3",)},
            ["n1", "n2", "n4"],
            0,
            100,
            [["n1", "n2", "n4"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n1",),
                "n3": ("n2", "n1"),
                "n4": ("n3",),
                "n5": (),
                "n6": ("n1", "n5"),
                "n7": (),
                "n8": ("n4", "n5"),
                "n9": ("n0", "n3", "n6", "n4", "n5", "n8", "n7", "n1"),
                "n10": ("n3", "n0", "n6", "n9", "n7"),
                "n11": (),
                "n12": ("n1", "n8", "n3", "n6", "n7", "n0", "n10", "n5", "n9", "n11"),
                "n13": ("n9", "n11", "n4"),
                "n14": (),
                "n15": ("n6", "n12"),
                "n16": (
                    "n1",
                    "n7",
                    "n10",
                    "n3",
                    "n9",
                    "n0",
                    "n2",
                    "n5",
                    "n8",
                    "n13",
                    "n14",
                    "n15",
                    "n4",
                    "n6",
                ),
            },
            [
                "n11",
                "n16",
                "n5",
                "n12",
                "n7",
                "n2",
                "n0",
                "n6",
                "n3",
                "n9",
                "n8",
                "n15",
                "n14",
                "n4",
                "n13",
                "n1",
            ],
            0,
            100,
            [
                ["n11", "n5", "n7", "n2", "n0", "n14"],
                ["n16"],
                ["n12", "n13"],
                ["n6", "n3"],
                ["n9"],
                ["n8"],
                ["n15"],
                ["n4"],
                ["n1"],
            ],
        )
        self.verify({"n0": (), "n1": ()}, ["n1"], 0, 100, [["n1"]])
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n1",),
                "n3": (),
                "n4": ("n0", "n2", "n3"),
                "n5": ("n2", "n3"),
                "n6": ("n3",),
            },
            ["n6", "n2", "n3", "n1"],
            0,
            100,
            [["n6", "n2"], ["n3", "n1"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n2",),
                "n4": ("n0",),
                "n5": ("n1", "n2"),
                "n6": ("n2", "n3", "n1", "n0", "n5"),
                "n7": ("n6", "n2", "n0", "n4", "n5", "n1"),
                "n8": ("n4",),
                "n9": ("n4", "n6", "n7", "n1", "n2"),
            },
            ["n8", "n6", "n2", "n4", "n7", "n5", "n3", "n9"],
            0,
            100,
            [["n8", "n6"], ["n2", "n4"], ["n7"], ["n5", "n3"], ["n9"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n2"),
                "n4": ("n0",),
                "n5": ("n2", "n3", "n0", "n1"),
                "n6": ("n4", "n1"),
                "n7": ("n5",),
                "n8": ("n7", "n1", "n5", "n6", "n3", "n4", "n0"),
                "n9": ("n2", "n8"),
            },
            ["n1", "n7", "n4", "n2", "n0", "n8", "n3", "n5"],
            0,
            100,
            [["n1", "n4", "n2"], ["n7"], ["n0", "n3"], ["n8"], ["n5"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n0",),
                "n3": ("n1",),
                "n4": ("n2", "n1"),
                "n5": (),
                "n6": ("n0",),
                "n7": ("n6", "n3", "n2", "n1", "n0"),
                "n8": ("n0", "n2"),
                "n9": ("n6", "n5", "n8", "n4", "n0"),
                "n10": ("n1", "n7", "n5", "n8", "n6", "n2", "n4", "n9"),
            },
            ["n0"],
            0,
            100,
            [["n0"]],
        )

        # trivial test of min_fuse
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n2"),
                "n4": ("n1",),
                "n5": (),
                "n6": ("n5",),
                "n7": ("n1", "n6", "n5", "n2", "n3", "n0"),
                "n8": ("n5", "n7", "n2", "n6"),
                "n9": ("n1",),
                "n10": ("n9",),
                "n11": ("n3", "n4", "n0", "n2"),
                "n12": ("n8", "n9", "n5", "n1"),
                "n13": ("n11", "n4", "n12", "n1", "n9", "n3", "n0"),
            },
            ["n9", "n2", "n8", "n10", "n5", "n6", "n13", "n7", "n3", "n0", "n4"],
            2,
            10,
            [["n9", "n2", "n5", "n0", "n4"], ["n8", "n10"], ["n6", "n3"]],
        )

        # trivial test of max_fuse
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n2"),
                "n4": ("n1",),
                "n5": (),
                "n6": ("n5",),
                "n7": ("n1", "n6", "n5", "n2", "n3", "n0"),
                "n8": ("n5", "n7", "n2", "n6"),
                "n9": ("n1",),
                "n10": ("n9",),
                "n11": ("n3", "n4", "n0", "n2"),
                "n12": ("n8", "n9", "n5", "n1"),
                "n13": ("n11", "n4", "n12", "n1", "n9", "n3", "n0"),
            },
            ["n9", "n2", "n8", "n10", "n5", "n6", "n13", "n7", "n3", "n0", "n4"],
            0,
            3,
            [
                ["n9", "n2", "n5"],
                ["n8", "n10", "n4"],
                ["n6", "n3", "n0"],
                ["n13"],
                ["n7"],
            ],
        )

    def test_find_independent_subset_greedy_fuse(self):
        # ensure that fusing the sets during iteration results in the correct
        # iteration results. In the example graph after we merge n2 and n3,
        # n4 is no longer independent from n1.
        g, lookup = self.build_graph(
            {
                "n0": (),
                "n1": (),
                "n2": ("n0",),
                "n3": ("n1",),
                "n4": ("n2",),
                "n5": (),
            }
        )
        opts = {
            "min_fuse_set_size": 0,
            "max_fuse_set_size": 100,
        }
        subnodes = ["n2", "n3", "n4", "n0", "n1", "n5"]
        subnodes = [lookup[n] for n in subnodes]
        i = smith._inductor.fx_passes.group_batch_fusion.find_independent_subset_greedy(
            subnodes, opts
        )
        self.assertEqual(next(i), [lookup[n] for n in ["n2", "n3", "n5"]])

        # fuse n2 and n3 which makes n4 now dependent on n1.
        args = tuple(lookup[n] for n in ["n0", "n1"])
        fused = g.create_node("placeholder", "target", name="n2+n3", args=args)
        lookup["n2"].replace_all_uses_with(fused)
        g.erase_node(lookup["n2"])
        lookup["n3"].replace_all_uses_with(fused)
        g.erase_node(lookup["n3"])

        self.assertEqual(next(i), [lookup[n] for n in ["n4"]])
        self.assertEqual(next(i), [lookup[n] for n in ["n0", "n1"]])
        self.assertRaises(StopIteration, lambda: next(i))


if __name__ == "__main__":
    run_tests()
