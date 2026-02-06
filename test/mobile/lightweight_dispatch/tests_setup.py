import functools
import os
import shutil
import sys
from io import BytesIO

import smith
from smith.jit.mobile import _export_operator_list, _load_for_lite_interpreter


_OPERATORS = set()
_FILENAMES = []
_MODELS = []


def save_model(cls):
    """Save a model and dump all the ops"""

    @functools.wraps(cls)
    def wrapper_save():
        _MODELS.append(cls)
        model = cls()
        scripted = smith.jit.script(model)
        buffer = BytesIO(scripted._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)
        ops = _export_operator_list(mobile_module)
        _OPERATORS.update(ops)
        path = f"./{cls.__name__}.ptl"
        _FILENAMES.append(path)
        scripted._save_for_lite_interpreter(path)

    return wrapper_save


@save_model
class ModelWithDTypeDeviceLayoutPinMemory(smith.nn.Module):
    def forward(self, x: int):
        a = smith.ones(
            size=[3, x],
            dtype=smith.int64,
            layout=smith.strided,
            device="cpu",
            pin_memory=False,
        )
        return a


@save_model
class ModelWithTensorOptional(smith.nn.Module):
    def forward(self, index):
        a = smith.zeros(2, 2)
        a[0][1] = 1
        a[1][0] = 2
        a[1][1] = 3
        return a[index]


# gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
@save_model
class ModelWithScalarList(smith.nn.Module):
    def forward(self, a: int):
        values = smith.tensor(
            [4.0, 1.0, 1.0, 16.0],
        )
        if a == 0:
            return smith.gradient(
                values, spacing=smith.scalar_tensor(2.0, dtype=smith.float64)
            )
        elif a == 1:
            return smith.gradient(values, spacing=[smith.tensor(1.0).item()])


# upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
@save_model
class ModelWithFloatList(smith.nn.Upsample):
    def __init__(self) -> None:
        super().__init__(
            scale_factor=(2.0,),
            mode="linear",
            align_corners=False,
            recompute_scale_factor=True,
        )


# index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
@save_model
class ModelWithListOfOptionalTensors(smith.nn.Module):
    def forward(self, index):
        values = smith.tensor([[4.0, 1.0, 1.0, 16.0]])
        return values[smith.tensor(0), index]


# conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1,
# int groups=1) -> Tensor
@save_model
class ModelWithArrayOfInt(smith.nn.Conv2d):
    def __init__(self) -> None:
        super().__init__(1, 2, (2, 2), stride=(1, 1), padding=(1, 1))


# add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
# ones_like(Tensor self, *, ScalarType?, dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None,
# MemoryFormat? memory_format=None) -> Tensor
@save_model
class ModelWithTensors(smith.nn.Module):
    def forward(self, a):
        b = smith.ones_like(a)
        return a + b


@save_model
class ModelWithStringOptional(smith.nn.Module):
    def forward(self, b):
        a = smith.tensor(3, dtype=smith.int64)
        out = smith.empty(size=[1], dtype=smith.float)
        smith.div(b, a, out=out)
        return [smith.div(b, a, rounding_mode="trunc"), out]


@save_model
class ModelWithMultipleOps(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ops = smith.nn.Sequential(
            smith.nn.ReLU(),
            smith.nn.Flatten(),
        )

    def forward(self, x):
        x[1] = -2
        return self.ops(x)


if __name__ == "__main__":
    command = sys.argv[1]
    ops_yaml = sys.argv[2]
    backup = ops_yaml + ".bak"
    if command == "setup":
        tests = [
            ModelWithDTypeDeviceLayoutPinMemory(),
            ModelWithTensorOptional(),
            ModelWithScalarList(),
            ModelWithFloatList(),
            ModelWithListOfOptionalTensors(),
            ModelWithArrayOfInt(),
            ModelWithTensors(),
            ModelWithStringOptional(),
            ModelWithMultipleOps(),
        ]
        shutil.copyfile(ops_yaml, backup)
        with open(ops_yaml, "a") as f:
            for op in _OPERATORS:
                f.write(f"- {op}\n")
    elif command == "shutdown":
        for file in _MODELS:
            if os.path.isfile(file):
                os.remove(file)
        shutil.move(backup, ops_yaml)
