# mypy: allow-untyped-defs
import smith

from smith.export import Dim

x = smith.randn(3, 2)
y = smith.randn(2)
dim0_x = Dim("dim0_x")

class CondOperands(smith.nn.Module):
    """
    The operands passed to cond() must be:
    - a list of tensors
    - match arguments of `true_fn` and `false_fn`

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x, y):
        def true_fn(x, y):
            return x + y

        def false_fn(x, y):
            return x - y

        return smith.cond(x.shape[0] > 2, true_fn, false_fn, [x, y])

example_args = (x, y)
tags = {
    "smith.cond",
    "smith.dynamic-shape",
}
extra_inputs = (smith.randn(2, 2), smith.randn(2))
dynamic_shapes = {"x": {0: dim0_x}, "y": None}
model = CondOperands()
