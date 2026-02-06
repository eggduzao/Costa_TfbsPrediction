import operator_benchmark as op_bench

import smith
import smith.nn.functional as F


"""Microbenchmarks for layernorm operator."""

layernorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (1, 8, 16),
        (8, 8, 16),
        (32, 8, 16),
        (64, 128, 56, 56),
    ),
    tags=["short"],
)


class LayerNormBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, dims):
        input = (smith.rand(*dims) - 0.5) * 256
        self.inputs = {
            "input": input,
            "weight": smith.rand(*input.size()[1:], dtype=smith.float),
            "bias": smith.rand(*input.size()[1:], dtype=smith.float),
            "eps": 1e-5,
        }

    def forward(self, input, weight, bias, eps: float):
        return F.layer_norm(input, input.size()[1:], weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(layernorm_configs_short, LayerNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
