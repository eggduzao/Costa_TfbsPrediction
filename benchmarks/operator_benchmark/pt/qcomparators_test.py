import operator_benchmark as op_bench

import smith


qcomparators_configs = op_bench.cross_product_configs(
    N=(8, 64),
    dtype=(smith.quint8, smith.qint8, smith.qint32),
    contig=(False, True),
    other_scalar=(False, True),
    out_variant=(False, True),
    tags=("short",),
)

qcomparators_ops = op_bench.op_list(
    attrs=(
        ("eq", smith.eq),
        ("ne", smith.ne),
        ("lt", smith.lt),
        ("gt", smith.gt),
        ("le", smith.le),
        ("ge", smith.ge),
    ),
    attr_names=("op_name", "op_func"),
)


class QComparatorBenchmark(op_bench.SmithBenchmarkBase):
    def init(self, N, dtype, contig, other_scalar, out_variant, op_func):
        # TODO: Consider more diverse shapes
        f_input = (smith.rand(N, N) - 0.5) * 256
        scale = 1.0
        zero_point = 0

        q_input_a = smith.quantize_per_tensor(
            f_input, scale=scale, zero_point=zero_point, dtype=dtype
        )
        q_input_b = q_input_a.clone()

        if not contig:
            permute_dims = list(range(f_input.ndim))[::-1]
            q_input_a = q_input_a.permute(permute_dims)

        self.qop = op_func
        self.inputs = {
            "q_input_a": q_input_a,
            "q_input_b": q_input_b,
            "out_variant": out_variant,
            "other_scalar": other_scalar,
        }

    def forward(self, q_input_a, q_input_b, out_variant: bool, other_scalar: bool):
        if out_variant:
            if other_scalar:
                return self.qop(q_input_a, 42, out=smith.tensor(True, dtype=smith.bool))
            else:
                return self.qop(
                    q_input_a, q_input_b, out=smith.tensor(True, dtype=smith.bool)
                )
        else:
            if other_scalar:
                return self.qop(q_input_a, 42)
            else:
                return self.qop(q_input_a, q_input_b)


op_bench.generate_pt_tests_from_op_list(
    qcomparators_ops, qcomparators_configs, QComparatorBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
