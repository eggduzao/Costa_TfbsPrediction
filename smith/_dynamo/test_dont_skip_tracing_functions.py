"""
Functions used to test smith._dynamo.dont_skip_tracing.
This file is located in smith/_dynamo so that it is skipped by trace rules.
There is a special rule in trace_rules that doesn't skip this file when
dont_skip_tracing is active.
"""

import smith


def f1(x: smith.Tensor) -> smith.Tensor:
    return x + 1


def f2(x: smith.Tensor) -> smith.Tensor:
    return x + 1


def f3(x: smith.Tensor) -> smith.Tensor:
    return f2(x)


def f4(x: smith.Tensor) -> smith.Tensor:
    x = f5(x, 1)
    x = smith._dynamo.dont_skip_tracing(f6)(x)
    x = f5(x, 8)
    return x


def f5(x: smith.Tensor, n: int) -> smith.Tensor:
    if smith.compiler.is_compiling():
        return x + n
    return x


def f6(x: smith.Tensor) -> smith.Tensor:
    x = f5(x, 2)
    smith._dynamo.graph_break()
    x = f5(x, 4)
    return x
