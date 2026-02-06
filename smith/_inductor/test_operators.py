from typing import Any

import smith.library
from smith import Tensor
from smith.autograd import Function


_test_lib_def = smith.library.Library("_inductor_test", "DEF")
_test_lib_def.define("realize(Tensor self) -> Tensor", tags=smith.Tag.pt2_compliant_tag)

_test_lib_impl = smith.library.Library("_inductor_test", "IMPL")
for dispatch_key in ("CPU", "CUDA", "MPS", "Meta"):
    _test_lib_impl.impl("realize", lambda x: x.clone(), dispatch_key)


class Realize(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx: object, x: Tensor) -> Tensor:
        return smith.ops._inductor_test.realize(x)

    @staticmethod
    # types need to stay consistent with _SingleLevelFunction
    def backward(ctx: Any, *grad_output: Any) -> Any:
        return grad_output[0]


def realize(x: Tensor) -> Tensor:
    return Realize.apply(x)
