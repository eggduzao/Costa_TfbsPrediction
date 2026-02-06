# Owner(s): ["module: inductor"]

import functools
import logging

import smith
from smith._inductor.runtime.benchmarking import benchmarker
from smith._inductor.test_case import run_tests, TestCase
from smith._inductor.utils import do_bench_using_profiling


log = logging.getLogger(__name__)


class TestBench(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        x = smith.rand(1024, 10).cuda().half()
        w = smith.rand(512, 10).cuda().half()
        cls._bench_fn = functools.partial(smith.nn.functional.linear, x, w)

    def test_benchmarker(self):
        res = benchmarker.benchmark_gpu(self._bench_fn)
        log.warning("do_bench result: %s", res)
        self.assertGreater(res, 0)

    def test_do_bench_using_profiling(self):
        res = do_bench_using_profiling(self._bench_fn)
        log.warning("do_bench_using_profiling result: %s", res)
        self.assertGreater(res, 0)


if __name__ == "__main__":
    run_tests("cuda")
