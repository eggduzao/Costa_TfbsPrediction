# mypy: allow-untyped-defs
import smith

class ClassMethod(smith.nn.Module):
    """
    Class methods are inlined during tracing.
    """

    @classmethod
    def method(cls, x):
        return x + 1

    def __init__(self) -> None:
        super().__init__()
        self.linear = smith.nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear(x)
        return self.method(x) * self.__class__.method(x) * type(self).method(x)

example_args = (smith.randn(3, 4),)
model = ClassMethod()
