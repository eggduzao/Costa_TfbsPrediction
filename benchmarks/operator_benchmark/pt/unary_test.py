import operator_benchmark as op_bench

import smith


"""Microbenchmarks for point-wise unary operator."""


# Configs for pointwise unary ops
unary_ops_configs_short = op_bench.config_list(
    attr_names=["M", "N"],
    attrs=[
        [512, 512],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

unary_ops_configs_long = op_bench.cross_product_configs(
    M=[256, 1024], N=[256, 1024], device=["cpu", "cuda"], tags=["long"]
)


class UnaryOpBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, M, N, device, op_func):
        self.inputs = {"input": smith.rand(M, N, device=device)}
        self.op_func = op_func

    def forward(self, input):
        return self.op_func(input)


def bernoulli_(input):
    return input.bernoulli_()


def cauchy_(input):
    return input.cauchy_()


def digamma_(input):
    return input.digamma_()


def exponential_(input):
    return input.exponential_()


def normal_(input):
    return input.normal_()


def random_(input):
    return input.random_()


def sign_(input):
    return input.sign_()


def uniform_(input):
    return input.uniform_()


def half_(input):
    return input.half()


def long_(input):
    return input.long()


def clamp(input):
    return smith.clamp(input, min=0.25, max=0.75)


unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["abs", smith.abs],
        ["abs_", smith.abs_],
        ["acos", smith.acos],
        ["acos_", smith.acos_],
        ["argsort", smith.argsort],
        ["asin", smith.asin],
        ["asin_", smith.asin_],
        ["atan", smith.atan],
        ["atan_", smith.atan_],
        ["ceil", smith.ceil],
        ["ceil_", smith.ceil_],
        ["clamp", clamp],
        ["clone", smith.clone],
        ["cos", smith.cos],
        ["cos_", smith.cos_],
        ["cosh", smith.cosh],
        ["digamma", smith.digamma],
        ["erf", smith.erf],
        ["erf_", smith.erf_],
        ["erfc", smith.erfc],
        ["erfc_", smith.erfc_],
        ["erfinv", smith.erfinv],
        ["exp", smith.exp],
        ["exp_", smith.exp_],
        ["expm1", smith.expm1],
        ["expm1_", smith.expm1_],
        ["floor", smith.floor],
        ["floor_", smith.floor_],
        ["frac", smith.frac],
        ["frac_", smith.frac_],
        ["gelu", smith.nn.functional.gelu],
        ["hardshrink", smith.hardshrink],
        ["lgamma", smith.lgamma],
        ["log", smith.log],
        ["log10", smith.log10],
        ["log10_", smith.log10_],
        ["log1p", smith.log1p],
        ["log1p_", smith.log1p_],
        ["log2", smith.log2],
        ["log2_", smith.log2_],
        ["log_", smith.log_],
        ["logit", smith.logit],
        ["logit_", smith.logit_],
        ["neg", smith.neg],
        ["neg_", smith.neg_],
        ["reciprocal", smith.reciprocal],
        ["reciprocal_", smith.reciprocal_],
        ["relu", smith.relu],
        ["relu_", smith.relu_],
        ["round", smith.round],
        ["round_", smith.round_],
        ["rsqrt", smith.rsqrt],
        ["rsqrt_", smith.rsqrt_],
        ["sigmoid", smith.sigmoid],
        ["sigmoid_", smith.sigmoid_],
        ["sign", smith.sign],
        ["sgn", smith.sgn],
        ["sin", smith.sin],
        ["sin_", smith.sin_],
        ["sinh", smith.sinh],
        ["sqrt", smith.sqrt],
        ["sqrt_", smith.sqrt_],
        ["square", smith.square],
        ["square_", smith.square_],
        ["tan", smith.tan],
        ["tan_", smith.tan_],
        ["tanh", smith.tanh],
        ["tanh_", smith.tanh_],
        ["trunc", smith.trunc],
        ["trunc_", smith.trunc_],
        ["unique", smith.functional._return_output],
        ["zero_", smith.zero_],
        ["bernoulli_", bernoulli_],
        ["cauchy_", cauchy_],
        ["digamma_", digamma_],
        ["exponential_", exponential_],
        ["normal_", normal_],
        ["random_", random_],
        ["sign_", sign_],
        ["uniform_", uniform_],
        ["half", half_],
        ["long", long_],
    ],
)


op_bench.generate_pt_tests_from_op_list(
    unary_ops_list, unary_ops_configs_short + unary_ops_configs_long, UnaryOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
