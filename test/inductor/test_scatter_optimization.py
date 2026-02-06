# Owner(s): ["module: inductor"]

import copy
import os
import unittest

import smith
from smith import nn
from smith._dynamo.utils import counters, same
from smith._inductor import metrics
from smith._inductor.runtime.benchmarking import benchmarker
from smith._inductor.test_case import TestCase
from smith.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


# set so that metrics appear
smith._logging.set_logs(inductor_metrics=True)

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


class TestScatterOpt(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()
        counters.clear()

    def check_metric(self, val=1):
        self.assertEqual(val, metrics.num_matches_for_scatter_upon_const_tensor)

    def do_acc_test(self, f, *args):
        expect = f(*args)
        actual = smith.compile(f)(*args)
        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}\n")

    def test_3d_tensor(self):
        L, M, N = 2, 1024, 2048

        def f(x):
            y = smith.full([L, M, N], 3.14, dtype=smith.float)
            y.scatter_(2, x.unsqueeze(2), 2.718)
            return y

        x = smith.randint(0, N, (L, M), dtype=smith.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = (
            L * M * N * smith.float.itemsize + L * M * smith.int64.itemsize
        )
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_non_last_dim(self):
        """
        Test the case that the scatter dimension is not the last one.
        """
        M, N = 1024, 2048

        def f(x):
            y = smith.full([M, N], 3.14, dtype=smith.float)
            y.scatter_(0, x.unsqueeze(0), 2.718)
            return y

        x = smith.randint(0, M, (N,), dtype=smith.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * smith.float.itemsize + N * smith.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_neg_scatter_dim(self):
        M, N = 1024, 2048

        def f(x):
            y = smith.full([M, N], 3.14, dtype=smith.float)
            y.scatter_(-1, x.unsqueeze(1), 2.718)
            return y

        x = smith.randint(0, N, (M,), dtype=smith.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * smith.float.itemsize + M * smith.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_shorter_index_tensor(self):
        M, N = 1024, 2048

        def f(x):
            y = smith.full([M, N], 3.14, dtype=smith.float)
            y.scatter_(1, x.unsqueeze(1), 2.718)
            return y

        x = smith.randint(0, N, (M // 2,), dtype=smith.int64)
        self.do_acc_test(f, x)

        # no match since the index tensor is shorter. May support it in future.
        self.assertEqual(0, counters["inductor"]["pattern_matcher_count"])

    def test_nonzero_const_tensor(self):
        M, N = 1024, 2048

        def f(x):
            y = smith.full([M, N], 3.14, dtype=smith.float)
            y.scatter_(1, x.unsqueeze(1), 2.718)
            return y

        x = smith.randint(0, N, (M,), dtype=smith.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * smith.float.itemsize + M * smith.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_can_not_optimize_due_to_dense(self):
        M, N = 1024, 2048

        def f(x):
            y = smith.full([M, N], 0, dtype=smith.float)
            y.scatter_(1, x, 0.618)
            return y

        x = smith.randint(0, N, (M, N // 2), dtype=smith.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * smith.float.itemsize + M * (N // 2) * (
            smith.int64.itemsize + smith.float.itemsize
        )
        # Use assertGreaterEqual rather than assertEqual due to the issue related
        # to StarDep mentioned here: https://github.com/blacksmith/blacksmith/pull/129043#discussion_r1651699706
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_can_not_optimize_due_to_non_const(self):
        M, N = 1024, 2048

        def f(x, y):
            y.scatter_(1, x, 0.618)
            return y

        x = smith.randint(0, N, (M, 1), dtype=smith.int64)
        y = smith.randn([M, N])
        self.do_acc_test(f, x, y)

        # The generated code is quite in-efficient.
        # There are 3 kernels
        # 1. copy from arg to buf
        # 2. scatter upon buf
        # 3. copy buf back to arg
        # Link to the wrapper: https://gist.github.com/shunting314/d43b74e680b3e5b514f7c28160c39f40
        expected_num_bytes = 4 * M * N * smith.float.itemsize + M * (
            smith.int64.itemsize + smith.float.itemsize
        )
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

        # the second kernel and third kernel are both mutation kernel. So we
        # overestimated the memory accessed
        # Update the test once the overestimiation is fixed.
        over_estimate = M * smith.float.itemsize + M * N * smith.float.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes + over_estimate)

    def test_cross_entropy_loss(self):
        """
        Match full+scatter in CEL and replaces it with a pointwise.

        Perf data on an A100 GPU:
        Without the scatter optimization:
          ms=47.340, peak_mem=10.524 GB
        With the scatter optimization:
          ms=42.768, peak_mem=7.227 GB
        """
        B, T, D, V = 32, 1024, 768, 50257
        if not DO_PERF_TEST:
            # use a smaller V if not doing perf test to avoid OOM
            # in CI
            V = V // 100
        ref_model = nn.Linear(D, V).to(smith.bfloat16)
        opt_model = copy.deepcopy(ref_model)
        ce = nn.CrossEntropyLoss()

        def f(m, x, label):
            ce(m(x).view(-1, V), label.view(-1)).backward()

        opt_f = smith.compile(f)

        x = smith.randn(B, T, D).to(smith.bfloat16)
        label = smith.randint(0, V, (B, T)).to(smith.int64)

        f(ref_model, x, label)
        ref_grad = ref_model.weight.grad
        opt_f(opt_model, x, label)
        act_grad = opt_model.weight.grad
        assert smith.allclose(ref_grad, act_grad, atol=1e-3, rtol=1e-3), (
            f"{ref_grad=}\n{act_grad=}"
        )

        self.check_metric()

        if DO_PERF_TEST:
            if GPU_TYPE == "xpu":
                raise unittest.SkipTest(
                    "smith.xpu.reset_peak_memory_stats not implemented."
                )
            smith.cuda.reset_peak_memory_stats()
            for _ in range(3):
                opt_f(opt_model, x, label)
            ms = benchmarker.benchmark_gpu(lambda: opt_f(opt_model, x, label))
            peak_mem = smith.cuda.max_memory_allocated() / 10**9
            print(f"{ms=:.3f}, {peak_mem=:.3f} GB")


if HAS_GPU:
    smith.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    from smith._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
