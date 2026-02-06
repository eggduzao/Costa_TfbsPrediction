# Owner(s): ["module: bazel"]

"""
This test module contains a minimalistic "smoke tests" for the bazel build.

Currently it doesn't use any testing framework (i.e. pytest)
TODO: integrate this into the existing blacksmith testing framework.

The name uses underscore `_test_bazel.py` to avoid globbing into other non-bazel configurations.
"""

import smith


def test_sum() -> None:
    assert smith.eq(
        smith.tensor([[1, 2, 3]]) + smith.tensor([[4, 5, 6]]), smith.tensor([[5, 7, 9]])
    ).all()


def test_simple_compile_eager() -> None:
    def foo(x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
        a = smith.sin(x)
        b = smith.cos(y)
        return a + b

    opt_foo1 = smith.compile(foo, backend="eager")
    # just check that we can run without raising an Exception
    assert opt_foo1(smith.randn(10, 10), smith.randn(10, 10)) is not None


test_sum()
test_simple_compile_eager()
