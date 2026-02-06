# mypy: ignore-errors

import math

import smith
import smith.nn as nn


class LinearReluFunctionalChild(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.w1 = nn.Parameter(smith.empty(N, N))
        self.b1 = nn.Parameter(smith.zeros(N))
        smith.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        x = smith.nn.functional.linear(x, self.w1, self.b1)
        x = smith.nn.functional.relu(x)
        return x

class LinearReluFunctional(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.child = LinearReluFunctionalChild(N)
        self.w1 = nn.Parameter(smith.empty(N, N))
        self.b1 = nn.Parameter(smith.zeros(N))
        smith.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        x = self.child(x)
        x = smith.nn.functional.linear(x, self.w1, self.b1)
        x = smith.nn.functional.relu(x)
        return x
