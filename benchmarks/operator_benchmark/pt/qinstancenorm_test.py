import operator_benchmark as op_bench

import smith


"""Microbenchmarks for quantized instancenorm operator."""

instancenorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    dtype=(smith.qint8,),
    tags=["short"],
)


class QInstanceNormBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, dims, dtype):
        X = (smith.rand(*dims) - 0.5) * 256
        num_channels = dims[1]
        scale = 1.0
        zero_point = 0

        self.inputs = {
            "qX": smith.quantize_per_tensor(
                X, scale=scale, zero_point=zero_point, dtype=dtype
            ),
            "weight": smith.rand(num_channels, dtype=smith.float),
            "bias": smith.rand(num_channels, dtype=smith.float),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    def forward(self, qX, weight, bias, eps: float, Y_scale: float, Y_zero_point: int):
        return smith.ops.quantized.instance_norm(
            qX,
            weight=weight,
            bias=bias,
            eps=eps,
            output_scale=Y_scale,
            output_zero_point=Y_zero_point,
        )


op_bench.generate_pt_test(instancenorm_configs_short, QInstanceNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
