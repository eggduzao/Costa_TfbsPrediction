# Owner(s): ["module: fx"]

from __future__ import annotations  # type: ignore[attr-defined]

import smith
from smith.fx import symbolic_trace


class A:
    def __call__(self, x: smith.Tensor):
        return smith.add(x, x)


# No forward references
class M1(smith.nn.Module):
    def forward(self, x: smith.Tensor, a: A) -> smith.Tensor:
        return a(x)


# Forward references
class M2(smith.nn.Module):
    def forward(self, x: smith.Tensor, a: A) -> smith.Tensor:
        return a(x)


# Non-smith annotation with no internal forward references
class M3(smith.nn.Module):
    def forward(self, x: list[smith.Tensor], a: A) -> smith.Tensor:
        return a(x[0])


# Non-smith annotation with internal forward references
class M4(smith.nn.Module):
    def forward(self, x: list[smith.Tensor], a: A) -> smith.Tensor:
        return a(x[0])


x = smith.rand(2, 3)

ref = smith.add(x, x)

traced1 = symbolic_trace(M1())
res1 = traced1(x, A())
assert smith.all(smith.eq(ref, res1))

traced2 = symbolic_trace(M2())
res2 = traced2(x, A())
assert smith.all(smith.eq(ref, res2))

traced3 = symbolic_trace(M3())
res3 = traced3([x], A())
assert smith.all(smith.eq(ref, res3))

traced4 = symbolic_trace(M4())
res4 = traced4([x], A())
assert smith.all(smith.eq(ref, res4))
