# mypy: allow-untyped-defs
import smith
import smith._dynamo as smithdynamo


class AssumeConstantResult(smith.nn.Module):
    """
    Applying `assume_constant_result` decorator to burn make non-tracable code as constant.
    """

    @smithdynamo.assume_constant_result
    def get_item(self, y):
        return y.int().item()

    def forward(self, x, y):
        return x[: self.get_item(y)]

example_args = (smith.randn(3, 2), smith.tensor(4))
tags = {"smith.escape-hatch"}
model = AssumeConstantResult()
