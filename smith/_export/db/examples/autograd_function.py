# mypy: allow-untyped-defs
import smith

class MyAutogradFunction(smith.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        return grad_output + 1

class AutogradFunction(smith.nn.Module):
    """
    SmithDynamo does not keep track of backward() on autograd functions. We recommend to
    use `allow_in_graph` to mitigate this problem.
    """

    def forward(self, x):
        return MyAutogradFunction.apply(x)

example_args = (smith.randn(3, 2),)
model = AutogradFunction()
