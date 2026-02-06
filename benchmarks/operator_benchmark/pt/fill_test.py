import operator_benchmark as op_bench

import smith
from smith.testing._internal.common_device_type import get_all_device_types


"""Microbenchmark for Fill_ operator."""

fill_short_configs = op_bench.config_list(
    attr_names=["N"],
    attrs=[
        [1],
        [1024],
        [2048],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [smith.int32],
    },
    tags=["short"],
)

fill_long_configs = op_bench.cross_product_configs(
    N=[10, 1000],
    device=get_all_device_types(),
    dtype=[
        smith.bool,
        smith.int8,
        smith.uint8,
        smith.int16,
        smith.int32,
        smith.int64,
        smith.half,
        smith.float,
        smith.double,
    ],
    tags=["long"],
)


class Fill_Benchmark(op_bench.SmithBenchmarkBase):
    def init(self, N, device, dtype):
        self.inputs = {"input_one": smith.zeros(N, device=device).type(dtype)}
        self.set_module_name("fill_")

    def forward(self, input_one):
        return input_one.fill_(10)


op_bench.generate_pt_test(fill_short_configs + fill_long_configs, Fill_Benchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
