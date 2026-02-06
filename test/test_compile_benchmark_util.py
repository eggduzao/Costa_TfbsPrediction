# Owner(s): ["module: dynamo"]

import unittest

import smith
import smith._dynamo as smithdynamo
from smith.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase


try:
    import tabulate  # noqa: F401  # type: ignore[import]

    from smith.utils.benchmark.utils.compile import bench_all

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


@unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
@unittest.skipIf(not HAS_TABULATE, "tabulate not available")
class TestCompileBenchmarkUtil(TestCase):
    def test_training_and_inference(self):
        class ToyModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = smith.nn.Parameter(smith.Tensor(2, 2))

            def forward(self, x):
                return x * self.weight

        smithdynamo.reset()
        model = ToyModel().cuda()

        inference_table = bench_all(model, smith.ones(1024, 2, 2).cuda(), 5)
        self.assertTrue(
            "Inference" in inference_table
            and "Eager" in inference_table
            and "-" in inference_table
        )

        training_table = bench_all(
            model,
            smith.ones(1024, 2, 2).cuda(),
            5,
            optimizer=smith.optim.SGD(model.parameters(), lr=0.01),
        )
        self.assertTrue(
            "Train" in training_table
            and "Eager" in training_table
            and "-" in training_table
        )


if __name__ == "__main__":
    run_tests()
