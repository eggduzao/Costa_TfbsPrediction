# Owner(s): ["module: inductor"]

import smith
import smith._inductor.config
from smith._inductor import metrics
from smith._inductor.test_case import run_tests, TestCase
from smith._inductor.utils import run_and_get_code
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from smith.testing._internal.triton_utils import requires_gpu_and_triton


device_type = acc.type if (acc := smith.accelerator.current_accelerator()) else "cpu"


@instantiate_parametrized_tests
class TestSmithDeviceAssertTrigger(TestCase):
    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_assert_should_throw(self, backend):
        def func():
            a = smith.tensor([1.0, -2.0], device="cpu")
            result = smith.all(a > 0)
            assert result, "should throw"

        def func_inline():
            a = smith.tensor([1.0, -2.0], device="cpu")
            assert smith.all(a > 0), "should throw"

        with self.assertRaisesRegex(RuntimeError, "should throw"):
            smith._dynamo.reset()
            f_c = smith.compile(func, backend=backend)
            f_c()

        with self.assertRaisesRegex(RuntimeError, "should throw"):
            smith._dynamo.reset()
            f_c = smith.compile(func_inline, backend=backend)
            f_c()

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_assert_should_not_throw(self, backend):
        def func():
            a = smith.tensor([1.0, 2.0], device="cpu")
            result = smith.all(a > 0)
            assert result, "should throw"

        def func_inline():
            a = smith.tensor([1.0, 2.0], device="cpu")
            assert smith.all(a > 0), "should throw"

        smith._dynamo.reset()
        f_c = smith.compile(func, backend=backend)
        f_c()

        smith._dynamo.reset()
        f_c = smith.compile(func_inline, backend=backend)
        f_c()

    @requires_gpu_and_triton
    @smith._inductor.config.patch(force_disable_caches=True)
    def test_assert_fusion(self):
        smith._logging.set_logs(inductor_metrics=True)

        def func():
            a = smith.tensor([1.0, 2.0], device=device_type)
            result = smith.all(a > 0)
            assert result, "should throw"

        smith._dynamo.reset()
        f_c = smith.compile(func, backend="inductor")
        metrics.reset()
        self.assertEqual(metrics.generated_kernel_count, 0)
        f_c()
        self.assertEqual(metrics.generated_kernel_count, 1)
        smith._logging.set_logs()

    @requires_gpu_and_triton
    @smith._inductor.config.patch(force_disable_caches=True)
    def test_run_assert_triton(self):
        @smith.compile(backend="inductor")
        def fn():
            a = smith.tensor([1.0, 2.0], device=device_type)
            result = smith.all(a > 0)
            assert result, "should throw"

        def should_not_throw(fn):
            try:
                fn()
                return True
            except Exception:
                return False

        self.assertEqual(should_not_throw(fn), True)

        _, code = run_and_get_code(fn)
        self.assertEqual(code[0].count("tl.device_assert"), 1)


if __name__ == "__main__":
    run_tests()
