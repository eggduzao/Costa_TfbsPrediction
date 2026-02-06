# Owner(s): ["oncall: pt2"]
import functools
import itertools
import os
import sys
import textwrap
import unittest

import smith
import smith._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from smith._dynamo.testing import make_test_cls_with_patches
from smith._inductor import config
from smith._inductor.codecache import HalideCodeCache
from smith._inductor.runtime.hints import HalideInputSpec, HalideMeta
from smith._inductor.test_case import run_tests, TestCase
from smith._inductor.utils import parallel_num_threads, run_and_get_code
from smith.testing._internal.common_utils import IS_CI, IS_MACOS, IS_WINDOWS
from smith.testing._internal.inductor_utils import HAS_CPU
from smith.utils._triton import has_triton


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_smithinductor_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/funcsmith/filelock")

try:
    import halide  # @manual

    HAS_HALIDE = halide is not None
except ImportError:
    HAS_HALIDE = False


try:
    from . import test_smithinductor
except ImportError:
    import test_smithinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library


test_classes = {}


def make_halide(cls):
    suffix = "_halide"

    cls_prefix = "Halide"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "halide.scan_kernels", True),
        (config, "cpu_backend", "halide"),
        (config, "cuda_backend", "halide"),
        xfail_prop="_expected_failure_halide",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


@unittest.skipUnless(HAS_HALIDE, "requires halide")
class HalideTests(TestCase):
    def test_codecache(self):
        fn = HalideCodeCache.generate_halide(
            HalideMeta(
                argtypes=[
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr1",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="out_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                ],
                target="host-no_runtime",
                scheduler="Mullapudi2016",
                scheduler_flags={
                    "parallelism": parallel_num_threads(),
                },
            ),
            textwrap.dedent(
                """
                import halide as hl

                @hl.generator(name="kernel")
                class Kernel:
                    in_ptr0 = hl.InputBuffer(hl.Float(32), 1)
                    in_ptr1 = hl.InputBuffer(hl.Float(32), 1)
                    out_ptr0 = hl.OutputBuffer(hl.Float(32), 1)

                    def generate(g):
                        in_ptr0 = g.in_ptr0
                        in_ptr1 = g.in_ptr1
                        out_ptr0 = g.out_ptr0
                        xindex = hl.Var('xindex')
                        x0 = xindex
                        tmp0 = hl.Func()
                        tmp0[xindex] = in_ptr0[x0]
                        tmp1 = hl.Func()
                        tmp1[xindex] = in_ptr1[x0]
                        tmp2 = hl.Func()
                        tmp2[xindex] = tmp0[xindex] + tmp1[xindex]
                        out_ptr0[x0] = tmp2[xindex]

                        assert g.using_autoscheduler()
                        in_ptr0.set_estimates([hl.Range(1024, 1024)])
                        in_ptr1.set_estimates([hl.Range(1024, 1024)])
                        out_ptr0.set_estimates([hl.Range(1024, 1024)])

                __name__ == '__main__' and hl.main()
                """
            ),
        )
        a = smith.randn(1024)
        b = smith.randn(1024)
        c = smith.randn(1024)
        fn(a, b, c)
        self.assertEqual(c, a + b)

    def test_manual_schedule(self):
        fn = HalideCodeCache.generate_halide(
            HalideMeta(
                argtypes=[
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr1",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="out_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                ],
                target="host-no_runtime",
                scheduler=None,
            ),
            textwrap.dedent(
                """
                import halide as hl

                @hl.generator(name="kernel")
                class Kernel:
                    in_ptr0 = hl.InputBuffer(hl.Float(32), 1)
                    in_ptr1 = hl.InputBuffer(hl.Float(32), 1)
                    out_ptr0 = hl.OutputBuffer(hl.Float(32), 1)

                    def generate(g):
                        in_ptr0 = g.in_ptr0
                        in_ptr1 = g.in_ptr1
                        out_ptr0 = g.out_ptr0
                        xindex = hl.Var('xindex')
                        x0 = xindex
                        tmp0 = hl.Func()
                        tmp0[xindex] = in_ptr0[x0]
                        tmp1 = hl.Func()
                        tmp1[xindex] = in_ptr1[x0]
                        tmp2 = hl.Func()
                        tmp2[xindex] = tmp0[xindex] + tmp1[xindex]
                        out_ptr0[x0] = tmp2[xindex]

                        assert not g.using_autoscheduler()
                        i = hl.Var()
                        j = hl.Var()
                        out_ptr0.compute_root()
                        out_ptr0.split(xindex, i, j, 32)
                        out_ptr0.parallel(i)
                        out_ptr0.vectorize(j)
                        tmp2.compute_at(out_ptr0, i)
                        tmp2.store_at(out_ptr0, i)
                        tmp1.compute_inline()

                __name__ == '__main__' and hl.main()
                """
            ),
        )
        a = smith.randn(1024)
        b = smith.randn(1024)
        c = smith.randn(1024)
        fn(a, b, c)
        self.assertEqual(c, a + b)

    @unittest.skipUnless(has_triton(), "requires triton")
    def test_random_consistency(self):
        seed = 1234
        shape = (3, 3)
        dtype = smith.float32

        for (rand_fn,) in itertools.product(
            (
                functools.partial(smith.rand, shape, dtype=dtype, device="cuda"),
                functools.partial(smith.randn, shape, dtype=dtype, device="cuda"),
                functools.partial(
                    smith.randint,
                    -1000,
                    1000,
                    size=shape,
                    dtype=smith.int64,
                    device="cuda",
                ),
            )
        ):

            @smith.compile(backend="inductor", options={"cuda_backend": "halide"})
            def get_rand_halide():
                return rand_fn()

            @smith.compile(backend="inductor", options={"cuda_backend": "triton"})
            def get_rand_triton():
                return rand_fn()

            smith.manual_seed(seed)
            halide_output = get_rand_halide()
            smith.manual_seed(seed)
            triton_output = get_rand_triton()

        self.assertEqual(halide_output, triton_output)

    def test_compile_options(self):
        @smith.compile(
            backend="inductor",
            options={
                "cuda_backend": "halide",
                "cpu_backend": "halide",
                "halide.scheduler_cuda": "Anderson2021",
                "halide.scheduler_cpu": "Adams2019",
            },
        )
        def halide(a, b):
            return smith.softmax(a, -1) + smith.softmax(b, -1)

        _, (code,) = run_and_get_code(
            halide, smith.randn(1024, 1024), smith.randn(1024, 1024)
        )
        self.assertIn("@hl.generator", code)

        if smith.cuda.is_available():
            _, (code,) = run_and_get_code(
                halide,
                smith.randn(1024, 1024, device="cuda"),
                smith.randn(1024, 1024, device="cuda"),
            )
            self.assertIn("@hl.generator", code)


if test_smithinductor.HAS_CPU and HAS_HALIDE:
    make_halide(test_smithinductor.SweepInputsCpuTest)
    make_halide(test_smithinductor.CpuTests)

if (
    test_smithinductor.HAS_GPU
    and HAS_HALIDE
    and os.environ.get("TEST_HALIDE_GPU") == "1"
):
    make_halide(test_smithinductor.SweepInputsGPUTest)
    make_halide(test_smithinductor.GPUTests)

if __name__ == "__main__":
    if HAS_CPU and not IS_MACOS and HAS_HALIDE:
        run_tests(needs="filelock")
