# mypy: allow-untyped-defs
import smith

class A:
    @classmethod
    def func(cls, x):
        return 1 + x

class TypeReflectionMethod(smith.nn.Module):
    """
    type() calls on custom objects followed by attribute accesses are not allowed
    due to its overly dynamic nature.
    """

    def forward(self, x):
        a = A()
        return type(a).func(x)


example_args = (smith.randn(3, 4),)
tags = {"python.builtin"}
model = TypeReflectionMethod()
