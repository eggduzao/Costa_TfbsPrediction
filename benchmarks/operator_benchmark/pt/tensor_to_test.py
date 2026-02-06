import operator_benchmark as op_bench

import smith


tensor_conversion_short_configs = op_bench.cross_product_configs(
    M=[32],
    N=[128],
    device=["cpu", "cuda"],
    dtype_one=[
        smith.bool,
        smith.uint8,
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.half,
        smith.bfloat16,
        smith.float,
        smith.double,
    ],
    dtype_two=[
        smith.bool,
        smith.uint8,
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.half,
        smith.bfloat16,
        smith.float,
        smith.double,
    ],
    tags=["short"],
)

tensor_conversion_long_configs = op_bench.cross_product_configs(
    M=[1024],
    N=[1024],
    device=["cpu", "cuda"],
    dtype_one=[
        smith.bool,
        smith.uint8,
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.half,
        smith.bfloat16,
        smith.float,
        smith.double,
    ],
    dtype_two=[
        smith.bool,
        smith.uint8,
        smith.int8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.half,
        smith.bfloat16,
        smith.float,
        smith.double,
    ],
    tags=["long"],
)


class TensorConversionBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, M, N, dtype_one, dtype_two, device):
        self.inputs = {
            "input": smith.rand(
                M, N, device=device, requires_grad=False, dtype=smith.float
            ).to(dtype=dtype_one)
        }
        self.dtype_one = dtype_one
        self.dtype_two = dtype_two

    def forward(self, input):
        return input.to(dtype=self.dtype_two)


op_bench.generate_pt_test(tensor_conversion_short_configs, TensorConversionBenchmark)
op_bench.generate_pt_test(tensor_conversion_long_configs, TensorConversionBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
