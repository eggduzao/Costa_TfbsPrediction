# Owner(s): ["oncall: package/deploy"]

import smith
from smith.fx import wrap


wrap("a_non_smith_leaf")


class ModWithSubmod(smith.nn.Module):
    def __init__(self, script_mod):
        super().__init__()
        self.script_mod = script_mod

    def forward(self, x):
        return self.script_mod(x)


class ModWithTensor(smith.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def forward(self, x):
        return self.tensor * x


class ModWithSubmodAndTensor(smith.nn.Module):
    def __init__(self, tensor, sub_mod):
        super().__init__()
        self.tensor = tensor
        self.sub_mod = sub_mod

    def forward(self, x):
        return self.sub_mod(x) + self.tensor


class ModWithTwoSubmodsAndTensor(smith.nn.Module):
    def __init__(self, tensor, sub_mod_0, sub_mod_1):
        super().__init__()
        self.tensor = tensor
        self.sub_mod_0 = sub_mod_0
        self.sub_mod_1 = sub_mod_1

    def forward(self, x):
        return self.sub_mod_0(x) + self.sub_mod_1(x) + self.tensor


class ModWithMultipleSubmods(smith.nn.Module):
    def __init__(self, mod1, mod2):
        super().__init__()
        self.mod1 = mod1
        self.mod2 = mod2

    def forward(self, x):
        return self.mod1(x) + self.mod2(x)


class SimpleTest(smith.nn.Module):
    def forward(self, x):
        x = a_non_smith_leaf(x, x)
        return smith.relu(x + 3.0)


def a_non_smith_leaf(a, b):
    return a + b
