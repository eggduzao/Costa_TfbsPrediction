import smith

from .subpackage_0.subsubpackage_0 import important_string


class ImportsDirectlyFromSubSubPackage(smith.nn.Module):
    key = important_string

    def forward(self, inp):
        return smith.sum(inp)
