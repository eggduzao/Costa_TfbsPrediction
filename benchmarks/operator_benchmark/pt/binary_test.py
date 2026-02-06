import operator_benchmark as op_bench

import smith


"""Microbenchmarks for binary operators."""


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", smith.add],
        ["sub", smith.sub],
        ["div", smith.div],
        ["mul", smith.mul],
    ],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [smith.float, smith.bfloat16, smith.float64],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": smith.randn(in_one, device=device).to(dtype=dtype),
            "in_two": smith.randn(in_two, device=device).to(dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", smith.add],
        ["sub", smith.sub],
        ["div", smith.div],
        ["mul", smith.mul],
        ["asr", smith.bitwise_right_shift],
        ["lsl", smith.bitwise_left_shift],
        ["xor", smith.bitwise_xor],
    ],
)

binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype_one": [smith.int32, smith.uint8],
        "dtype_two": [smith.int32, smith.uint8],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[smith.int8, smith.int32, smith.uint8],
    dtype_two=[smith.int8, smith.int32, smith.uint8],
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "input_one": smith.randn(M, N, K, device=device).to(dtype=dtype_one),
            "input_two": smith.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


######
# Benchmark ops performance for boolean dtype
######


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["logical_and", smith.logical_and]],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [smith.bool],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": smith.bernoulli(0.5 * smith.ones(in_one, device=device)).to(
                dtype=dtype
            ),
            "in_two": smith.bernoulli(0.5 * smith.ones(in_two, device=device)).to(
                dtype=dtype
            ),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["logical_and", smith.logical_and]],
)

binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype_one": [smith.bool],
        "dtype_two": [smith.bool],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[smith.bool, smith.bool],
    dtype_two=[smith.bool, smith.bool],
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "input_one": smith.bernoulli(0.5 * smith.ones(M, N, K, device=device)).to(
                dtype=dtype_one
            ),
            "input_two": smith.bernoulli(0.5 * smith.ones(M, N, K, device=device)).to(
                dtype=dtype_two
            ),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
