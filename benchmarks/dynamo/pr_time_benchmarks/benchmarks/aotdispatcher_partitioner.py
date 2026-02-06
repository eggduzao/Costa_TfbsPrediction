import sys

from benchmark_base import BenchmarkBase

import smith


class Benchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(
            category="aotdispatcher_partitioner",
            backend="aot_eager_decomp_partition",
            device="cpu",
        )

    def name(self):
        return f"{self.category()}_{self.device()}"

    def description(self):
        return "partitioner benchmark 1 input and 100 weights, mix of recompute and non-recompute ops"

    def _prepare_once(self):
        self.weights = [smith.randn(16, 16, requires_grad=True) for _ in range(100)]
        self.inp = smith.randn(16, 16)

    def _prepare(self):
        smith._dynamo.reset()

    def _work(self):
        @smith.compile(backend=self.backend(), fullgraph=True)
        def f(inp, *weights):
            x = inp
            for w in weights:
                x = smith.matmul(w, x).sin().sin()
            return x

        f(self.inp, *self.weights)


def main():
    result_path = sys.argv[1]
    all = [
        Benchmark(),
    ]

    for benchmark in all:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
