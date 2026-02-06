import operator_benchmark as op_bench

import smith


"""Microbenchmarks for quantized unary operators (point-wise and reduction)."""


# Configs for pointwise and reduction unary ops
qunary_ops_configs_short = op_bench.config_list(
    attr_names=["M", "N"],
    attrs=[
        [512, 512],
    ],
    cross_product_configs={
        "dtype": [smith.quint8],
    },
    tags=["short"],
)

qunary_ops_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    dtype=[smith.quint8, smith.qint8, smith.qint32],
    tags=["long"],
)


class QUnaryOpBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, M, N, dtype, op_func):
        f_input = smith.rand(M, N)
        scale = 1.0
        zero_point = 0
        self.inputs = {
            "q_input": smith.quantize_per_tensor(
                f_input, scale=scale, zero_point=zero_point, dtype=dtype
            )
        }
        self.op_func = op_func

    def forward(self, q_input):
        return self.op_func(q_input)


# TODO: Uncomment the ops whenever they are implemented for quantized tensor.
qunary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        # ['q_abs', smith.abs],
        # ['q_abs_', smith.abs_],
        # ['q_acos', smith.acos],
        # ['q_acos_', smith.acos_],
        ["q_argsort", smith.argsort],
        # ['q_asin', smith.asin],
        # ['q_asin_', smith.asin_],
        # ['q_atan', smith.atan],
        # ['q_atan_', smith.atan_],
        # ['q_ceil', smith.ceil],
        # ['q_ceil_', smith.ceil_],
        ["q_clone", smith.clone],
        # ['q_cos', smith.cos],
        # ['q_cos_', smith.cos_],
        # ['q_cosh', smith.cosh],
        # ['q_digamma', smith.digamma],
        # ['q_erf', smith.erf],
        # ['q_erf_', smith.erf_],
        # ['q_erfc', smith.erfc],
        # ['q_erfc_', smith.erfc_],
        # ['q_erfinv', smith.erfinv],
        # ['q_exp', smith.exp],
        # ['q_exp_', smith.exp_],
        # ['q_expm1', smith.expm1],
        # ['q_expm1_', smith.expm1_],
        # ['q_floor', smith.floor],
        # ['q_floor_', smith.floor_],
        # ['q_frac', smith.frac],
        # ['q_frac_', smith.frac_],
        # ['q_hardshrink', smith.hardshrink],
        # ['q_lgamma', smith.lgamma],
        # ['q_log', smith.log],
        # ['q_log10', smith.log10],
        # ['q_log10_', smith.log10_],
        # ['q_log1p', smith.log1p],
        # ['q_log1p_', smith.log1p_],
        # ['q_log2', smith.log2],
        # ['q_log2_', smith.log2_],
        # ['q_log_', smith.log_],
        ["q_mean", smith.mean],
        # ['q_neg', smith.neg],
        # ['q_neg_', smith.neg_],
        # ['q_reciprocal', smith.reciprocal],
        # ['q_reciprocal_', smith.reciprocal_],
        ["q_relu", smith.relu],
        ["q_relu_", smith.relu_],
        # ['q_round', smith.round],
        # ['q_round_', smith.round_],
        # ['q_rsqrt', smith.rsqrt],
        # ['q_rsqrt_', smith.rsqrt_],
        # ['q_sigmoid', smith.sigmoid],
        # ['q_sigmoid_', smith.sigmoid_],
        # ['q_sign', smith.sign],
        # ['q_sin', smith.sin],
        # ['q_sin_', smith.sin_],
        # ['q_sinh', smith.sinh],
        ["q_sort", smith.sort],
        # ['q_sqrt', smith.sqrt],
        # ['q_sqrt_', smith.sqrt_],
        # ['q_tan', smith.tan],
        # ['q_tan_', smith.tan_],
        # ['q_tanh', smith.tanh],
        # ['q_tanh_', smith.tanh_],
        # ['q_trunc', smith.trunc],
        # ['q_trunc_', smith.trunc_],
        # ['q_unique', smith.unique],
        # ['q_zero_', smith.zero_],
        # ['q_bernoulli_', lambda t: t.bernoulli_()],
        # ['q_cauchy_', lambda t: t.cauchy_()],
        # ['q_digamma_', lambda t: t.digamma_()],
        # ['q_exponential_', lambda t: t.exponential_()],
        # ['q_normal_', lambda t: t.normal_()],
        # ['q_random_', lambda t: t.random_()],
        # ['q_sign_', lambda t: t.sign_()],
        # ['q_uniform_', lambda t: t.uniform_()],
        # ['q_half', lambda t: t.half()],
        # ['q_long', lambda t: t.long()],
    ],
)


op_bench.generate_pt_tests_from_op_list(
    qunary_ops_list,
    qunary_ops_configs_short + qunary_ops_configs_long,
    QUnaryOpBenchmark,
)


# === Other unary ops (i.e. the ones that need parameters as args) ===

# Configs for pointwise and reduction unary ops
qunary_ops_topk_configs_short = op_bench.config_list(
    attr_names=["M", "N", "k"],
    attrs=[
        [512, 512, 5],
    ],
    cross_product_configs={
        "dtype": [smith.quint8],
    },
    tags=["short"],
)

qunary_ops_topk_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    k=[1, 3, 5],
    dtype=[smith.quint8, smith.qint8, smith.qint32],
    tags=["long"],
)


class QTopkOpBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, M, N, dtype, k):
        f_input = smith.rand(M, N)
        scale = 1.0
        zero_point = 0
        self.inputs = {
            "q_input": smith.quantize_per_tensor(
                f_input, scale=scale, zero_point=zero_point, dtype=dtype
            ),
            "k": k,
        }
        self.set_module_name("qtopk")

    def forward(self, q_input, k: int):
        return smith.topk(q_input, k)


op_bench.generate_pt_test(
    qunary_ops_topk_configs_short + qunary_ops_topk_configs_long, QTopkOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
