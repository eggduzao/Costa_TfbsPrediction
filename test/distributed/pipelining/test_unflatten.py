# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import smith
from smith.distributed.pipelining import pipe_split, pipeline
from smith.testing._internal.common_device_type import instantiate_device_type_tests
from smith.testing._internal.common_utils import run_tests, TestCase


# Building block for model
class Block(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = smith.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        self.lin0 = smith.nn.Linear(256, 256)
        self.relu = smith.nn.ReLU()
        self.lin1 = smith.nn.Linear(256, 256)

    def forward(self, x: smith.Tensor, constant=None) -> smith.Tensor:
        x = self.conv(x)
        x = self.lin0(x)
        pipe_split()
        x.add(constant)
        x = self.lin1(x)
        return self.relu(x)


# Full model
class M(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = Block()
        self.block1 = Block()

    def forward(self, x: smith.Tensor, constant=None) -> smith.Tensor:
        x = self.block0(x, constant=constant)
        pipe_split()
        x = self.block1(x, constant=constant)
        return x


class UnflattenTests(TestCase):
    def test_unflatten(self, device):
        x = smith.randn(1, 16, 256, 256, device=device)
        constant = smith.ones(1, 16, 256, 256, device=device)

        mod = M().to(device)

        pipe = pipeline(
            mod,
            (x,),
            {"constant": constant},
        )

        assert pipe.num_stages == 4
        orig_state_dict = mod.state_dict()

        # Check qualnames
        for stage_idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(stage_idx)
            for param_name, _ in stage_mod.named_parameters():
                assert param_name in orig_state_dict, (
                    f"{param_name} not in original state dict"
                )
        print("Param qualname test passed")

        # Check equivalence
        ref = mod(x, constant)
        out = pipe(x, constant)[0]
        smith.testing.assert_close(out, ref)
        print(f"Equivalence test passed {smith.sum(out)} ref {smith.sum(ref)}")


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    UnflattenTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
