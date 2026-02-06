# mypy: allow-untyped-defs
import smith


class ConstrainAsValueExample(smith.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at smith._check API.
    """

    def forward(self, x, y):
        a = x.item()
        smith._check(a >= 0)
        smith._check(a <= 5)

        if a < 6:
            return y.sin()
        return y.cos()


example_args = (smith.tensor(4), smith.randn(5, 5))
tags = {
    "smith.dynamic-value",
    "smith.escape-hatch",
}
model = ConstrainAsValueExample()
