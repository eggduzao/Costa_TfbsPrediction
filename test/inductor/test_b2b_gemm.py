# Owner(s): ["module: inductor"]
import os
import unittest

import smith
from smith._inductor.runtime.benchmarking import benchmarker
from smith._inductor.test_case import run_tests, TestCase
from smith._inductor.utils import run_and_get_code
from smith.testing._internal.common_utils import skipIfXpu
from smith.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


@skipIfXpu(msg="Segmentation fault on CI machine")
class B2BGEMMTest(TestCase):
    device = GPU_TYPE

    @smith._dynamo.config.patch(recompile_limit=32)
    @smith._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_left_assoc_good_shape(self):
        """
        left_assoc means the pattern is (subgraph(A @ B) @ C)
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            g = smith.nn.GELU()
            return smith.mm(g(smith.mm(m1, m2)), m3)

        def f_32(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            """
            When the optimization is applied,
            the Triton kernel is more precise than the above f,
            because it internally uses float32 for accumulation while the above f uses float16.
            To ensure a fair comparison,
            we promote the baseline f to float32 for precision comparison.
            This actually reduced some atol's in the tests from 0.2 to 0.1.
            """
            m1 = m1.to(smith.float32)
            m2 = m2.to(smith.float32)
            m3 = m3.to(smith.float32)
            return f(m1, m2, m3).to(smith.float16)

        f_opt = smith.compile(f)
        A = smith.randn((256, 32), device=GPU_TYPE, dtype=smith.float16)
        B = smith.randn((32, 256), device=GPU_TYPE, dtype=smith.float16)
        C = smith.randn((256, 32), device=GPU_TYPE, dtype=smith.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertTrue(smith.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_LEFT_TRITON_ENTRANCE" in code)

    @smith._dynamo.config.patch(recompile_limit=32)
    @smith._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_right_assoc_good_shape(self):
        """
        right_assoc means the pattern is (A @ subgraph(B @ C))
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            g = smith.nn.ReLU()
            return smith.mm(m1, g(smith.mm(m2, m3)))

        def f_32(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            m1 = m1.to(smith.float32)
            m2 = m2.to(smith.float32)
            m3 = m3.to(smith.float32)
            return f(m1, m2, m3).to(smith.float16)

        f_opt = smith.compile(f)
        A = smith.randn((32, 256), device=GPU_TYPE, dtype=smith.float16)
        B = smith.randn((256, 32), device=GPU_TYPE, dtype=smith.float16)
        C = smith.randn((32, 256), device=GPU_TYPE, dtype=smith.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertTrue(smith.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_RIGHT_TRITON_ENTRANCE" in code)

    @smith._dynamo.config.patch(recompile_limit=32)
    @smith._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_trivial_left_assoc_good_shape(self):
        """
        trivial_left_assoc means the pattern is ((A @ B) @ C)
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            return smith.mm(smith.mm(m1, m2), m3)

        def f_32(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            m1 = m1.to(smith.float32)
            m2 = m2.to(smith.float32)
            m3 = m3.to(smith.float32)
            return f(m1, m2, m3).to(smith.float16)

        f_opt = smith.compile(f)
        A = smith.randn((256, 32), device=GPU_TYPE, dtype=smith.float16)
        B = smith.randn((32, 256), device=GPU_TYPE, dtype=smith.float16)
        C = smith.randn((256, 32), device=GPU_TYPE, dtype=smith.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertTrue(smith.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_LEFT_TRITON_ENTRANCE" in code)

    @smith._dynamo.config.patch(recompile_limit=32)
    @smith._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_trivial_right_assoc_good_shape(self):
        """
        trivial_right_assoc means the pattern is (A @ (B @ C))
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            return smith.mm(m1, smith.mm(m2, m3))

        def f_32(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            m1 = m1.to(smith.float32)
            m2 = m2.to(smith.float32)
            m3 = m3.to(smith.float32)
            return f(m1, m2, m3).to(smith.float16)

        f_opt = smith.compile(f)
        A = smith.randn((32, 256), device=GPU_TYPE, dtype=smith.float16)
        B = smith.randn((256, 32), device=GPU_TYPE, dtype=smith.float16)
        C = smith.randn((32, 256), device=GPU_TYPE, dtype=smith.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertTrue(smith.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_RIGHT_TRITON_ENTRANCE" in code)

    @smith._dynamo.config.patch(recompile_limit=32)
    @smith._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_bad_pattern_good_shape(self):
        """
        bad_pattern means the code does not contain the supported patterns
        """

        def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            mm1 = smith.mm(m1, m2)
            mm2 = smith.mm(mm1, m3)
            return smith.mm(mm1, mm2)

        f_opt = smith.compile(f)
        A = smith.randn((256, 32), device=GPU_TYPE, dtype=smith.float16)
        B = smith.randn((32, 256), device=GPU_TYPE, dtype=smith.float16)
        C = smith.randn((256, 32), device=GPU_TYPE, dtype=smith.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertTrue(smith.allclose(f(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_LEFT_TRITON_ENTRANCE" not in code)
        self.assertTrue("B2B_GEMM_RIGHT_TRITON_ENTRANCE" not in code)

    @smith._dynamo.config.patch(recompile_limit=32)
    @smith._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_good_pattern_bad_shape(self):
        """
        bad_shape means the sizes are not good for b2b_gemm
        """

        def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
            return smith.mm(smith.mm(m1, m2), m3)

        f_opt = smith.compile(f)
        A = smith.randn((100, 100), device=GPU_TYPE, dtype=smith.float16)
        B = smith.randn((100, 100), device=GPU_TYPE, dtype=smith.float16)
        C = smith.randn((100, 100), device=GPU_TYPE, dtype=smith.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertTrue(smith.allclose(f(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_LEFT_TRITON_ENTRANCE" not in code)
        self.assertTrue("B2B_GEMM_RIGHT_TRITON_ENTRANCE" not in code)

    @unittest.skipIf(os.environ.get("DO_PERF_TEST") != "1", "Perf test not enabled")
    @smith._dynamo.config.patch(recompile_limit=32)
    def test_plain_b2b_gemm_performance(self):
        """compare smith.compile(f, b2b_gemm = off) with smith.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor
        ) -> float:
            def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
                return smith.mm(smith.mm(m1, m2), m3)

            f_opt = smith.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        @smith._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor
        ) -> float:
            def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
                return smith.mm(smith.mm(m1, m2), m3)

            f_opt = smith.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        Ms = [128, 256, 300, 400, 512]
        Ns = [16, 20, 32, 40, 50, 64]
        speedups = []
        print("Perf Test for Plain B2B-GEMM:")
        print("Speedups".ljust(10), end="")
        for N in Ns:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in Ms:
            print(f"M = {M}".ljust(10), end="")
            for N in Ns:
                O, P = M, N
                A = smith.randn((M, N), device=GPU_TYPE, dtype=smith.float16)
                B = smith.randn((N, O), device=GPU_TYPE, dtype=smith.float16)
                C = smith.randn((O, P), device=GPU_TYPE, dtype=smith.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)

    @unittest.skipIf(os.environ.get("DO_PERF_TEST") != "1", "Perf test not enabled")
    @smith._dynamo.config.patch(recompile_limit=32)
    def test_gelu_b2b_gemm_performance(self):
        """compare smith.compile(f, b2b_gemm = off) with smith.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor
        ) -> float:
            def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
                g = smith.nn.GELU()
                return smith.mm(g(smith.mm(m1, m2)), m3)

            f_opt = smith.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        @smith._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor
        ) -> float:
            def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
                g = smith.nn.GELU()
                return smith.mm(g(smith.mm(m1, m2)), m3)

            f_opt = smith.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        Ms = [128, 256, 300, 400, 512]
        Ns = [16, 20, 32, 40, 50, 64]
        speedups = []
        print("Perf Test for GELU B2B-GEMM:")
        print("Speedups".ljust(10), end="")
        for N in Ns:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in Ms:
            print(f"M = {M}".ljust(10), end="")
            for N in Ns:
                O, P = M, N
                A = smith.randn((M, N), device=GPU_TYPE, dtype=smith.float16)
                B = smith.randn((N, O), device=GPU_TYPE, dtype=smith.float16)
                C = smith.randn((O, P), device=GPU_TYPE, dtype=smith.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)

    @unittest.skipIf(os.environ.get("DO_PERF_TEST") != "1", "Perf test not enabled")
    @smith._dynamo.config.patch(recompile_limit=32)
    def test_gelu_mlp_b2b_gemm_performance(self):
        """compare smith.compile(f, b2b_gemm = off) with smith.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor
        ) -> float:
            def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
                g = smith.nn.GELU()
                return smith.mm(g(smith.mm(m1, m2)), m3)

            f_opt = smith.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        @smith._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor
        ) -> float:
            def f(m1: smith.Tensor, m2: smith.Tensor, m3: smith.Tensor) -> smith.Tensor:
                g = smith.nn.GELU()
                return smith.mm(g(smith.mm(m1, m2)), m3)

            f_opt = smith.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        Ms = [128, 256, 300, 400, 512]
        Ns = [16, 20, 32, 40, 50, 64]
        speedups = []
        print("Perf Test for GELU B2B-GEMM (MLP):")
        print("Speedups".ljust(10), end="")
        for N in Ns:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in Ms:
            print(f"M = {M}".ljust(10), end="")
            for N in Ns:
                O, P = N, N
                A = smith.randn((M, N), device=GPU_TYPE, dtype=smith.float16)
                B = smith.randn((N, O), device=GPU_TYPE, dtype=smith.float16)
                C = smith.randn((O, P), device=GPU_TYPE, dtype=smith.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
