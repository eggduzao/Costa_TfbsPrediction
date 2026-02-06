# mypy: allow-untyped-defs
import smith


class ConstrainAsSizeExample(smith.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at smith._check APIs.
    """

    def forward(self, x):
        a = x.item()
        smith._check(a >= 0)
        smith._check(a <= 5)
        return smith.zeros((a, 5))


example_args = (smith.tensor(4),)
tags = {
    "smith.dynamic-value",
    "smith.escape-hatch",
}
model = ConstrainAsSizeExample()
