# mypy: allow-untyped-defs
import functools

import smith

def test_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs) + 1

    return wrapper

class Decorator(smith.nn.Module):
    """
    Decorators calls are inlined into the exported function during tracing.
    """

    @test_decorator
    def forward(self, x, y):
        return x + y

example_args = (smith.randn(3, 2), smith.randn(3, 2))
model = Decorator()
