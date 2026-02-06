import smith
from smith import nn


class NeuralNetwork(nn.Module):
    def forward(self, x):
        return smith.add(x, 10)


model = NeuralNetwork()
script = smith.jit.script(model)
smith.jit.save(script, "aot_test_model.pt")
