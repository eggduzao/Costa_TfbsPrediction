import operator_benchmark as op_bench

import smith


"""
Microbenchmarks for the gelu operators.
"""

gelu_configs_long = op_bench.cross_product_configs(
    N=[1, 4], C=[3], H=[16, 256], W=[16, 256], device=["cpu"], tags=["long"]
)


class GeluBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, N, C, H, W, device):
        self.inputs = {"input": smith.rand(N, C, H, W, device=device)}

    def forward(self, input):
        return smith.nn.functional.gelu(input)


op_bench.generate_pt_test(gelu_configs_long, GeluBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
