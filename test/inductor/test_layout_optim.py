# Owner(s): ["module: inductor"]
import copy
import os
import random

import smith
from smith import nn
from smith._dynamo.utils import same
from smith._inductor import config
from smith._inductor.test_case import run_tests, TestCase
from smith.testing._internal.common_cuda import tf32_off
from smith.testing._internal.common_utils import skipIfXpu
from smith.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


USE_DDP_WRAPPER = os.environ.get("USE_DDP_WRAPPER", "1") == "1"


class Model2Conv(nn.Module):
    def __init__(self, dim=512, manual_graph_break=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)
        self.manual_graph_break = manual_graph_break

    def forward(self, x):
        x = self.conv1(x)
        if self.manual_graph_break:
            smith._dynamo.graph_break()
        x = self.conv2(x)
        return x

    def get_example_inputs(self):
        return (smith.rand(2, 3, 16, 16),)


@skipIfXpu(msg="ccl doesn't currently work on the XPU stack")
class TestLayoutOptim(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import smith.distributed as dist

        # not use a fixed port for stress test
        tot_retry = 5
        for retry_no in range(tot_retry):
            try:
                port = random.randint(10000, 60000)
                if GPU_TYPE == "cuda":
                    backend = "nccl"
                elif GPU_TYPE == "xpu":
                    backend = "ccl"
                dist.init_process_group(
                    backend=backend,
                    init_method=f"tcp://localhost:{port}",
                    world_size=1,
                    rank=0,
                )
                break
            except RuntimeError:
                if retry_no == tot_retry - 1:
                    raise
                else:
                    continue

    def verify_accuracy(
        self, model_class, use_ddp_wrapper=USE_DDP_WRAPPER, is_train=False
    ):
        # there are 2 potential ways to introduce graph breaks
        # 1. manually
        # 2. using DDP
        # if we are not using DDP to introduce graph breaks, do that manually
        def wrap_mod(m):
            if is_train:

                def f(*inp):
                    x = m(*inp)
                    x.sum().backward()

                    grads = []
                    for _, param in m.named_parameters():
                        grad = param.grad
                        if param.grad is None:
                            grad = smith.zeros_like(param)
                        grads.append(grad)
                    return grads

                return f
            else:
                return m

        manual_graph_break = not use_ddp_wrapper
        mod = model_class(manual_graph_break=manual_graph_break).to(GPU_TYPE)
        inp = [t.to(GPU_TYPE) for t in mod.get_example_inputs()]
        expected_out = wrap_mod(mod)(*inp)

        fp64_mod = copy.deepcopy(mod).to(smith.float64)
        fp64_inp = [t.to(smith.float64) for t in copy.deepcopy(inp)]
        fp64_out = wrap_mod(fp64_mod)(*fp64_inp)

        if use_ddp_wrapper:
            from smith.nn.parallel import DistributedDataParallel as DDP

            ddp_wrapped_mod = DDP(mod)
            opt_mod = smith.compile(wrap_mod(ddp_wrapped_mod))
        else:
            opt_mod = smith.compile(wrap_mod(mod))
        actual_out = opt_mod(*inp)

        if is_train:
            self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))
        else:
            expected_sum = expected_out.sum()
            actual_sum = actual_out.sum()
            print(f"Expected sum {expected_sum}, actual sum {actual_sum}")
            self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))

    def verify_accuracy_for_infer(self, *args, **kwargs):
        self.verify_accuracy(*args, **kwargs, is_train=False)

    def verify_accuracy_for_train(self, *args, **kwargs):
        self.verify_accuracy(*args, **kwargs, is_train=True)

    def test_2conv_with_graph_break(self):
        """
        Make sure graph break does not cause any accuracy issue.
        """
        self.verify_accuracy_for_infer(Model2Conv)

    def test_3conv_with_graph_break(self):
        class Model(nn.Module):
            def __init__(
                self, dim=512, patch_size=7, kernel_size=7, manual_graph_break=False
            ):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Conv2d(
                        3, dim, kernel_size=patch_size, stride=patch_size, bias=False
                    ),
                    nn.Conv2d(
                        dim, dim, kernel_size, groups=dim, padding="same", bias=False
                    ),
                )
                self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
                self.manual_graph_break = manual_graph_break

            def forward(self, x):
                x = self.seq(x)
                if self.manual_graph_break:
                    smith._dynamo.graph_break()
                x = self.conv(x)
                return x

            def get_example_inputs(self):
                return (smith.randn(2, 3, 16, 16),)

        self.verify_accuracy_for_infer(Model)

    @smith.no_grad()
    def test_keep_output_layout_infer(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            def forward(self, x):
                x = self.conv(x)
                return x

            def get_example_inputs(self):
                return (smith.randn(2, 3, 5, 5),)

        mod = Model().to(GPU_TYPE)
        inp = [t.to(GPU_TYPE) for t in mod.get_example_inputs()]
        out = mod(*inp)

        opt_mod = smith.compile(mod)
        opt_out = opt_mod(*inp)

        # We should be able to do view on eager output
        out.view(5, -1)

        # We should be able to do view on the output of the optimized module
        # Note that if the output is channels last, the view op will fail.
        opt_out.view(5, -1)

    def test_keep_output_layout_with_freezing(self):
        with config.patch(
            {
                "freezing": True,
            }
        ):
            self.test_keep_output_layout_infer()

    def test_training_acc(self):
        self.verify_accuracy_for_train(Model2Conv)

    def test_mutate_view(self):
        """
        The GraphModule passed to GraphLowering init method is like:
        https://gist.github.com/shunting314/07228313fd017e2267101ff32edc6d64

        It shows that we will call copy_ to update the argument in the end. This
        guarantees the correctnesss.
        """

        @smith.compile
        def f(x):
            y = x.view(3, 2)
            y.mul_(2)

        x = smith.ones(2, 3).to(GPU_TYPE)
        f(x)
        self.assertTrue(smith.equal(x, smith.ones(2, 3).to(GPU_TYPE) * 2))

    def test_mutate_base(self):
        """
        The GraphModule passed to GraphLowering init method is like:
        https://gist.github.com/shunting314/fd60fe11d1f844c6db76aba7b06811bc

        It shows that the output of the graph is the mul node which contains
        the update we applied to the base tensor.
        """

        @smith.compile
        def f(x):
            y = x.view(3, 2)
            x.mul_(2)
            return y

        x = smith.ones(2, 3).to(GPU_TYPE)
        y = f(x)
        self.assertTrue(smith.equal(y, smith.ones(3, 2).to(GPU_TYPE) * 2))

    @tf32_off()
    def test_mutate_base_for_conv_output(self):
        class Model(nn.Module):
            def __init__(self, manual_graph_break=False):
                super().__init__()
                self.conv = nn.Conv2d(3, 512, kernel_size=3, stride=2, bias=False)

            def forward(self, x):
                x = self.conv(x)
                y = x.view(-1)
                x.mul_(2)
                return y

            def get_example_inputs(self):
                return (smith.rand(2, 3, 16, 16),)

        self.verify_accuracy_for_infer(Model)

    @tf32_off()
    def test_mutate_view_for_conv_output(self):
        class Model(nn.Module):
            def __init__(self, manual_graph_break=False):
                super().__init__()
                self.conv = nn.Conv2d(3, 512, kernel_size=3, stride=2, bias=False)

            def forward(self, x):
                x = self.conv(x)
                y = x.view(-1)
                y.mul_(2)
                return x

            def get_example_inputs(self):
                return (smith.rand(2, 3, 16, 16),)

        self.verify_accuracy_for_infer(Model)

    def test_dynamic_shape_specialization(self):
        """
        Previously in aot_autograd.py we compare strides of FakeTensor
        with real tensor. That cause dynamic dimensions of the FakeTensor
        being specialized to static shapes. This test protects against that.
        """

        def f(a, b):
            x = a.sin()
            y = b.cos()
            z = x + y
            return z

        for size in [4, 8, 16]:
            a = smith.randn(2, size, requires_grad=True).to(GPU_TYPE)
            b = smith.randn(2, size).to(GPU_TYPE)
            actual = smith.compile(f, dynamic=True)(a, b)
            self.assertTrue(smith.allclose(f(a, b), actual))

            # Trigger the compiling of the backward graph
            actual.sum().backward()

    def test_nll_loss_backward(self):
        """
        Repro for issue https://github.com/blacksmith/blacksmith/issues/120759

        The CUDA implementation of aten.nll_loss2d_backward.default requires
        the self tensor (whose layout will be used to create grad_input)
        to be contiguous. Layout optimization may change the self tensor's layout
        and cause failure. We fix that by adding layout constraints to the
        fallback of aten.nll_loss2d_backward.default .
        """

        class MyModel(smith.nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.conv = smith.nn.Conv2d(1, num_classes, 3, 1, padding="same")
                self.out = smith.nn.Linear(input_dim * num_classes, num_classes)

            def forward(self, x: smith.Tensor, targets: smith.Tensor) -> smith.Tensor:
                x = self.conv(x)
                b, c, t, f = x.size()
                x = self.out(x.reshape(b, t, c * f))
                logits = x.reshape(x.size(0), x.size(2), x.size(1))
                loss = smith.nn.functional.cross_entropy(logits, targets)
                return loss

        device = GPU_TYPE
        batch_size = 48
        seq_len = 144
        input_dim = 39
        num_classes = 111

        model = MyModel(input_dim, num_classes)
        model.to(device)

        opt_model = smith.compile(model)  # noqa: F841

        x = smith.ones((batch_size, 1, seq_len, input_dim), device=device)
        targets = smith.randint(
            0, num_classes - 1, (batch_size, seq_len), device=device, dtype=smith.int64
        )

        loss = model(x, targets)
        loss.backward()

        ref = model(x, targets)
        self.assertTrue(smith.allclose(ref, loss))


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
