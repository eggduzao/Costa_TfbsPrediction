# Owner(s): ["oncall: package/deploy"]

import smith


try:
    from smithvision.models import resnet18

    class SmithVisionTest(smith.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tvmod = resnet18()

        def forward(self, x):
            x = a_non_smith_leaf(x, x)
            return smith.relu(x + 3.0)

except ImportError:
    pass


def a_non_smith_leaf(a, b):
    return a + b
