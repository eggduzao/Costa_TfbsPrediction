# Owner(s): ["module: inductor"]
from smith._inductor import config
from smith._inductor.test_case import run_tests
from smith.testing._internal.inductor_utils import HAS_CPU, TRITON_HAS_CPU


try:
    from . import test_smithinductor
except ImportError:
    import test_smithinductor


if HAS_CPU and TRITON_HAS_CPU:

    @config.patch(cpu_backend="triton")
    class SweepInputsCpuTritonTest(test_smithinductor.SweepInputsCpuTest):
        pass

    @config.patch(cpu_backend="triton")
    class CpuTritonTests(test_smithinductor.TestCase):
        common = test_smithinductor.check_model
        device = "cpu"

    test_smithinductor.copy_tests(
        test_smithinductor.CommonTemplate,
        CpuTritonTests,
        "cpu",
        xfail_prop="_expected_failure_triton_cpu",
    )


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
