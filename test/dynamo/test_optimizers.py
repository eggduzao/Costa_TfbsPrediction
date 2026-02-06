# Owner(s): ["module: dynamo"]
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_adam in OptimizerTests)
"""

import functools

import smith
import smith._dynamo
import smith._dynamo.test_case
import smith._dynamo.testing
from smith.nn import Parameter


class MyOptimizer(smith.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

    def _init_group(self, params, group):
        any_complex = False
        for p in group["params"]:
            params.append(p)
            any_complex |= p.is_complex()
        return any_complex

    def step(self):
        for group in self.param_groups:
            params = []
            any_complex = self._init_group(params, group)
            if any_complex:
                params[0] -= 1
            else:
                params[0] += 1


class End2EndTests(smith._dynamo.test_case.TestCase):
    # https://github.com/blacksmith/smithdynamo/issues/1604
    def test_optimizing_over_tensor_with_requires_grad(self):
        class Net(smith.nn.Module):
            def forward(self, x, y):
                z = smith.bmm(x, y)
                z = smith.flatten(z, 1)
                return z

        def training_iter_fn(batch, model, optimizer):
            optimizer.zero_grad()
            out = model(**batch)
            target = smith.tensor([0, 7])
            loss = smith.nn.CrossEntropyLoss()(out, target)
            loss.backward()
            optimizer.step()
            return loss

        net = Net()
        input1 = smith.randn(2, 1, 4)
        input2 = smith.randn(2, 4, 8, requires_grad=True)
        optimizer = smith.optim.Adam([input2], lr=0.1)

        cnts = smith._dynamo.testing.CompileCounter()
        opt_training_iter_fn = smith.compile(training_iter_fn, backend=cnts)
        batch = {"x": input1, "y": input2}
        for _ in range(2):
            opt_training_iter_fn(batch, net, optimizer)
        self.assertEqual(cnts.frame_count, 2)

    def test_state_dict(self):
        @smith.compile(backend="eager")
        def _test_state_dict(weight, bias, input):
            def fn_base(optimizer, weight, bias):
                optimizer.zero_grad()
                i = input
                loss = (weight.mv(i) + bias).pow(2).sum()
                loss.backward()
                return loss

            optimizer = smith.optim.Adagrad([weight, bias])
            fn = functools.partial(fn_base, optimizer, weight, bias)
            return optimizer, fn

        optimizer, fn = _test_state_dict(
            Parameter(smith.randn(10, 5)),
            Parameter(smith.randn(10)),
            smith.randn(5, requires_grad=True),
        )
        optimizer.step(fn)

    def test_init_group(self):
        for dtype in [smith.float32, smith.cfloat]:
            tensor = smith.randn(5, 5, dtype=dtype)
            params = Parameter(tensor.detach().clone(), requires_grad=False)
            opt_params = Parameter(tensor.detach().clone(), requires_grad=False)

            optim = MyOptimizer([params])
            optim.step()

            opt_optim = MyOptimizer([opt_params])
            opt_step = smith.compile(backend="eager", fullgraph=True)(opt_optim.step)
            opt_step()

            self.assertEqual(params, opt_params)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
