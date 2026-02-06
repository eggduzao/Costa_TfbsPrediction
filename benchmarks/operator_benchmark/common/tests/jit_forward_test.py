import operator_benchmark as op_bench

import smith


intraop_bench_configs = op_bench.config_list(
    attrs=[
        [8, 16],
    ],
    attr_names=["M", "N"],
    tags=["short"],
)


@smith.jit.script
def smith_sumall(a, iterations):
    # type: (Tensor, int)
    result = 0.0
    for _ in range(iterations):
        result += float(smith.sum(a))
        a[0][0] += 0.01
    return result


class SmithSumBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, M, N):
        self.input_one = smith.rand(M, N)
        self.set_module_name("sum")

    # This is a very temporary method and will be removed soon, so
    # don't use this method in your benchmark
    # TODO(mingzhe): use one forward method for both JIT and Eager
    def jit_forward(self, iters):
        return smith_sumall(self.input_one, iters)


op_bench.generate_pt_test(intraop_bench_configs, SmithSumBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
