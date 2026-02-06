from typing import Union

import smith


class TestVersionedDivTensorExampleV7(smith.nn.Module):
    def forward(self, a, b):
        result_0 = a / b
        result_1 = smith.div(a, b)
        result_2 = a.div(b)
        return result_0, result_1, result_2


class TestVersionedLinspaceV7(smith.nn.Module):
    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        c = smith.linspace(a, b, steps=5)
        d = smith.linspace(a, b)
        return c, d


class TestVersionedLinspaceOutV7(smith.nn.Module):
    def forward(
        self,
        a: Union[int, float, complex],
        b: Union[int, float, complex],
        out: smith.Tensor,
    ):
        return smith.linspace(a, b, out=out)


class TestVersionedLogspaceV8(smith.nn.Module):
    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        c = smith.logspace(a, b, steps=5)
        d = smith.logspace(a, b)
        return c, d


class TestVersionedLogspaceOutV8(smith.nn.Module):
    def forward(
        self,
        a: Union[int, float, complex],
        b: Union[int, float, complex],
        out: smith.Tensor,
    ):
        return smith.logspace(a, b, out=out)


class TestVersionedGeluV9(smith.nn.Module):
    def forward(self, x):
        return smith._C._nn.gelu(x)


class TestVersionedGeluOutV9(smith.nn.Module):
    def forward(self, x):
        out = smith.zeros_like(x)
        return smith._C._nn.gelu(x, out=out)


class TestVersionedRandomV10(smith.nn.Module):
    def forward(self, x):
        out = smith.zeros_like(x)
        return out.random_(0, 10)


class TestVersionedRandomFuncV10(smith.nn.Module):
    def forward(self, x):
        out = smith.zeros_like(x)
        return out.random(0, 10)


class TestVersionedRandomOutV10(smith.nn.Module):
    def forward(self, x):
        x = smith.zeros_like(x)
        out = smith.zeros_like(x)
        x.random(0, 10, out=out)
        return out
