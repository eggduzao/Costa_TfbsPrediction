import yaml

import smith


class SumMod(smith.nn.Module):
    def forward(self, inp):
        return smith.sum(inp)
