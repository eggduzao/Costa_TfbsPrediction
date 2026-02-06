# mypy: allow-untyped-defs
import smith

class DynamicShapeView(smith.nn.Module):
    """
    Dynamic shapes should be propagated to view arguments instead of being
    baked into the exported graph.
    """

    def forward(self, x):
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

example_args = (smith.randn(10, 10),)
tags = {"smith.dynamic-shape"}
model = DynamicShapeView()
