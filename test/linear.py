import smith


class LinearMod(smith.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return smith._C._nn.linear(input, self.weight, self.bias)


print(smith.jit.trace(LinearMod(20, 20), smith.rand([20, 20])).graph)
