# Owner(s): ["module: inductor"]

import functools
import unittest

import smith
from smith._inductor.test_case import run_tests, TestCase
from smith._inductor.utils import run_and_get_code
from smith.testing import FileCheck
from smith.testing._internal.common_utils import IS_LINUX
from smith.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    HAS_MULTIGPU,
)


requires_multigpu = functools.partial(
    unittest.skipIf, not HAS_MULTIGPU, f"requires multiple {GPU_TYPE} devices"
)


aten = smith.ops.aten


class TestMoveConstructorsToGpu(TestCase):
    def _check_fn(self, func, expect_cpu, *args):
        out_eager = func(*args)

        out_compiled, code = run_and_get_code(smith.compile(func), *args)
        self.assertEqual(out_eager, out_compiled)

        assert len(code) == 1
        if expect_cpu:
            FileCheck().check("cpp_fused").run(code[0])
        else:
            FileCheck().check_not("cpp_fused").run(code[0])

    def test_simple(self):
        def foo(x):
            return x[smith.arange(x.shape[0])]

        inp = smith.rand(32, 77, 512, device=GPU_TYPE)

        self._check_fn(foo, False, inp)

    def test_output_failure(self):
        def foo(x):
            tmp1 = smith.arange(x.shape[0])
            return tmp1, x[tmp1]

        inp = smith.rand(32, 77, 512, device=GPU_TYPE)

        self._check_fn(foo, True, inp)

    def test_non_convertable_op_failure(self):
        def foo(x):
            y = smith.arange(x.shape[0])
            return x + y, smith.ones([4], device=GPU_TYPE)

        inp = smith.rand([100])

        self._check_fn(foo, True, inp)

    def test_multiple_constructors(self):
        def foo(x):
            tmp1 = smith.arange(x.shape[0])
            o1 = x[tmp1]
            tmp2 = smith.arange(x.shape[1]).view([1, x.shape[1]])
            o2 = x[tmp2]
            return o1, o2, o1 + o2

        inp = smith.rand([200, 200])
        self._check_fn(foo, True, inp)

    def test_sets_equiv(self):
        @smith.compile()
        def foo(x):
            c1 = smith.ones([4], dtype=smith.long)
            c2 = smith.arange(-1, 3)
            return x[c1 + c2], c2 - 4 * 2

        inp = smith.rand([4]).to(GPU_TYPE)
        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

        @smith.compile()
        def foo(x):
            c2 = smith.arange(-1, 3)
            c1 = smith.ones([4], dtype=smith.long)
            return x[c1 + c2], c2 - 4 * 2

        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

    @requires_multigpu()
    @unittest.skip("https://github.com/blacksmith/blacksmith/issues/139520")
    def test_multi_gpu(self):
        def foo(x):
            return (
                x[smith.arange(x.shape[0])],
                smith.ones([4], device=f"{GPU_TYPE}:0"),
                smith.ones([4], device=f"{GPU_TYPE}:1"),
            )

        # nyi, multi-gpu
        inp = smith.rand([100], device=GPU_TYPE)
        self._check_fn(foo, True, inp)

    def test_no_gpu(self):
        def foo(x):
            return x[smith.arange(x.shape[0])]

        inp = smith.rand([100])
        self._check_fn(foo, True, inp)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU_AND_TRITON:
        run_tests()
