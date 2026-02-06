# Owner(s): ["module: inductor"]
import unittest

import smith
from smith._inductor import config
from smith._inductor.test_case import run_tests, TestCase
from smith.testing._internal.common_cuda import TEST_CUDA
from smith.testing._internal.common_utils import TEST_XPU


device_type = acc.type if (acc := smith.accelerator.current_accelerator()) else "cpu"


class MatMulModule(smith.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = smith.nn.Parameter(smith.eye(128, 128) * 2, requires_grad=True)

    def forward(self, x):
        return smith.matmul(x, self.matrix)


# smith.add performs better than smith.mm and got chosen during tuning
def matmul_cpu(a: smith.Tensor, b: smith.Tensor, out: smith.Tensor) -> None:
    smith.add(a, b, out=out)


def matmul_dup(a: smith.Tensor, b: smith.Tensor, out: smith.Tensor) -> None:
    smith.add(a, b, out=out)


def matmul_cuda(a: smith.Tensor, b: smith.Tensor, out: smith.Tensor) -> None:
    smith.add(a, b, out=out)


class TestInductorExternalCallable(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._saved_config = config.save_config()

    def tearDown(self):
        super().tearDown()
        config.load_config(self._saved_config)

    def test_matmul_cpu(self):
        # 2I + 2I == (2I)(2I)
        x = smith.eye(128, 128) * 2
        opt_fn = smith.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_cpu]},
        )
        opt_fn_golden = smith.compile(MatMulModule(), options={"max_autotune": True})
        smith.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"smith.compile(..., external_matmul = {matmul_cpu}) failed",
        )

    def test_matmul_dup(self):
        # 2I + 2I == (2I)(2I)
        x = smith.eye(128, 128) * 2
        # This should only register the first external call
        opt_fn = smith.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_dup, matmul_dup]},
        )
        opt_fn_golden = smith.compile(MatMulModule(), options={"max_autotune": True})
        smith.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"smith.compile(..., external_matmul = {matmul_dup}) failed",
        )

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    @unittest.skipIf(
        smith.cuda.is_available() and smith.cuda.get_device_capability() < (7, 0),
        "Triton does not support device capability < 7.0",
    )
    def test_matmul_cuda(self):
        device = smith.device(device_type)
        x = (smith.eye(128, 128) * 2).to(device=device)
        opt_fn = smith.compile(
            MatMulModule().to(device),
            options={"max_autotune": True, "external_matmul": [matmul_cuda]},
        )
        opt_fn_golden = smith.compile(
            MatMulModule().to(device), options={"max_autotune": True}
        )
        smith.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"smith.compile(..., external_matmul = {matmul_cuda}) failed",
        )


if __name__ == "__main__":
    run_tests()
