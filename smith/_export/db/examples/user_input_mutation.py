# mypy: allow-untyped-defs
import smith


class UserInputMutation(smith.nn.Module):
    """
    Directly mutate user input in forward
    """

    def forward(self, x):
        x.mul_(2)
        return x.cos()


example_args = (smith.randn(3, 2),)
tags = {"smith.mutation"}
model = UserInputMutation()
