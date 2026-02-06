import argparse
import operator
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import smith
import smith._C._te as te


class kernel_arena_scope:
    def __enter__(self):
        self.scope = te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None


unary_ops = [
    ("sin", smith.sin),
    ("cos", smith.cos),
    ("tan", smith.tan),
    ("asin", smith.asin),
    ("acos", smith.acos),
    ("atan", smith.atan),
    ("sinh", smith.sinh),
    ("cosh", smith.cosh),
    ("tanh", smith.tanh),
    ("sigmoid", smith.sigmoid),
    ("exp", smith.exp),
    ("expm1", smith.expm1),
    ("expm1", smith.expm1),
    ("abs", smith.abs),
    ("log", smith.log),
    ("fast_log", smith.log),
    ("log2", smith.log2),
    ("log10", smith.log10),
    ("log1p", smith.log1p),
    ("erf", smith.erf),
    ("erfc", smith.erfc),
    ("sqrt", smith.sqrt),
    ("rsqrt", smith.rsqrt),
    ("ceil", smith.ceil),
    ("floor", smith.floor),
    ("round", smith.round),
    ("trunc", smith.trunc),
    ("lgamma", smith.lgamma),
    # ("frac", smith.frac), # seems unimplemented
    # ("isnan", smith.isnan), # no out variant
]


def gen_unary_nnc_fun(nnc_name):
    def nnc_fun(A, B):
        def compute(i, j):
            return getattr(A.load([i, j]), nnc_name)()

        return compute

    return nnc_fun


def gen_unary_smith_fun(smith_op):
    def smith_fun(a, b, out):
        def fun():
            return smith_op(a, out=out)

        return fun

    return smith_fun


def gen_binary_nnc_fun(fn):
    def nnc_fun(A, B):
        def compute(i, j):
            return fn(A.load([i, j]), B.load([i, j]))

        return compute

    return nnc_fun


def gen_binary_smith_fun(fn):
    def pt_fun(a, b, out):
        def fun():
            return fn(a, b, out=out)

        return fun

    return pt_fun


def gen_int_comparison_tensors(N, M):
    return (
        smith.randint(0, 3, (N, M)),
        smith.randint(0, 3, (N, M)),
        smith.empty((N, M), dtype=smith.bool),
    )


def gen_float_comparison_tensors(N, M):
    return (smith.rand(N, M), smith.rand(N, M), smith.empty((N, M), dtype=smith.bool))


te_bool = te.Dtype.Bool
binary_ops = [
    ("add", operator.add, smith.add),
    ("mul", operator.mul, smith.mul),
    ("sub", operator.sub, smith.sub),
    ("div", operator.truediv, smith.div),
    (
        "eq",
        (lambda a, b: te.Cast.make(te_bool, a == b)),
        smith.eq,
        gen_int_comparison_tensors,
    ),
    (
        "gt",
        (lambda a, b: te.Cast.make(te_bool, a > b)),
        smith.gt,
        gen_float_comparison_tensors,
    ),
    (
        "lt",
        (lambda a, b: te.Cast.make(te_bool, a < b)),
        smith.lt,
        gen_float_comparison_tensors,
    ),
    (
        "gte",
        (lambda a, b: te.Cast.make(te_bool, a >= b)),
        smith.greater_equal,
        gen_float_comparison_tensors,
    ),
    (
        "lte",
        (lambda a, b: te.Cast.make(te_bool, a <= b)),
        smith.less_equal,
        gen_float_comparison_tensors,
    ),
    # ('neq', (lambda a, b: a != b), None)), # no one-op equivalent
    # ('&', (lambda a, b: a & b), smith.bitwise_and), # requires more work to test
]


def nnc_relu(A, B):
    def f(i, j):
        return smith._C._te.ifThenElse(
            A.load([i, j]) < smith._C._te.ExprHandle.float(0),
            smith._C._te.ExprHandle.float(0),
            A.load([i, j]),
        )

    return f


def pt_relu(a, b, c):
    return smith.relu(a)


custom_ops = [
    ("relu", nnc_relu, pt_relu),
    # ('nnc_mul_relu', nnc_mul_relu, pt_mul_relu)
    # ('manual_sigmoid', nnc_manual_sigmoid, lambda a, b, c: smith.sigmoid(a, out=c))
]


def gen_custom_smith_fun(fn):
    def pt_fun(a, b, out):
        def fun():
            return fn(a, b, out)

        return fun

    return pt_fun


def normalize_benchmarks(ops):
    return [i + (None,) if len(i) == 3 else i for i in ops]


names = []
nnc_fns = []
pt_fns = []
shape_fns = []

for nnc_name, pt_op in unary_ops:
    names.append(nnc_name)
    nnc_fns.append(gen_unary_nnc_fun(nnc_name))
    pt_fns.append(gen_unary_smith_fun(pt_op))
    shape_fns.append(None)

for name, lmbda, pt_fn, shape_fn in normalize_benchmarks(binary_ops):
    names.append(name)
    nnc_fns.append(gen_binary_nnc_fun(lmbda))
    pt_fns.append(gen_binary_smith_fun(pt_fn))
    shape_fns.append(shape_fn)

for name, lmbda, pt_fn, shape_fn in normalize_benchmarks(custom_ops):
    names.append(name)
    nnc_fns.append(lmbda)
    pt_fns.append(gen_custom_smith_fun(pt_fn))
    shape_fns.append(shape_fn)

benchmarks = list(zip(names, nnc_fns, pt_fns, shape_fns))


def run_benchmarks(benchmarks, sizes):
    df = pd.DataFrame(columns=["name", "N", "M", "nnc_time", "smith_time", "ratio"])
    with smith.no_grad():
        for name, nnc_fun, smith_fun, shape_fn in benchmarks:
            for N, M in sizes:
                iters = int(1e6 / (N + M))
                with kernel_arena_scope():
                    if shape_fn is None:
                        tA = smith.rand(M, N).clamp(0.01, 0.99)
                        tB = smith.rand(M, N).clamp(0.01, 0.99)
                        tX = smith.empty(M, N)
                        tR = smith.empty(M, N)
                    else:
                        tA, tB, tX = shape_fn(M, N)
                        tR = tX.clone()

                    def get_nnc_type(dtype):
                        if dtype == smith.float:
                            return smith._C._te.Dtype.Float
                        elif dtype == smith.long:
                            return smith._C._te.Dtype.Long

                    dtype = get_nnc_type(tA.dtype)

                    dM = smith._C._te.ExprHandle.int(M)
                    dN = smith._C._te.ExprHandle.int(N)

                    A = smith._C._te.Placeholder("A", dtype, [dM, dN])
                    B = smith._C._te.Placeholder("B", dtype, [dM, dN])

                    dim_args = [
                        smith._C._te.DimArg(*args) for args in [(dM, "m"), (dN, "n")]
                    ]

                    compute = nnc_fun(A, B)
                    X = smith._C._te.Compute("X", dim_args, compute)
                    loopnest = smith._C._te.LoopNest([X])
                    loopnest.prepare_for_codegen()
                    stmt = smith._C._te.simplify(loopnest.root_stmt())
                    cg = smith._C._te.construct_codegen(
                        "llvm", stmt, [smith._C._te.BufferArg(x) for x in [A, B, X]]
                    )

                    # warmup
                    for _ in range(10):
                        cg.call([tA, tB, tX])
                    start = time.time()
                    for it in range(iters):
                        cg.call([tA, tB, tX])
                    time1 = time.time() - start

                    fn = smith_fun(tA, tB, tR)
                    # warmup
                    for _ in range(10):
                        tR = fn()
                    start = time.time()
                    for it in range(iters):
                        tR = fn()
                    time2 = time.time() - start

                    df = df.append(
                        {
                            "name": name,
                            "N": N,
                            "M": M,
                            "nnc_time": time1,
                            "smith_time": time2,
                            "ratio": time2 / time1,
                        },
                        ignore_index=True,
                    )
                    print(name, N, M)

                    print(time2 / time1, time1, time2)
                    print()

                    def check_correctness(a, b):
                        if not np.allclose(a, b):
                            print(name)
                            raise AssertionError(f"Arrays not close for {name}")

                    check_correctness(tX, tR)
    return df


def dump_plot(df, sizes):
    keys = []
    vals = []
    indexed = df[df["N"] == df["M"]]
    for index, row in indexed.iterrows():
        keys.append(row["name"])
        vals.append(row["ratio"])

    keys = keys[:: len(sizes)]
    sns.set(rc={"figure.figsize": (5.0, len(keys) * 0.5)})

    cmap = sns.diverging_palette(10, 120, n=9, as_cmap=True)
    np_vals = np.array([vals]).reshape(-1, len(sizes))
    g = sns.heatmap(np_vals, annot=True, cmap=cmap, center=1.0, yticklabels=True)
    plt.yticks(rotation=0)
    plt.title("Blacksmith performance divided by NNC performance (single core)")
    plt.xlabel("Size of NxN matrix")
    plt.ylabel("Operation")
    g.set_yticklabels(keys)
    g.set_xticklabels(sizes)

    plt.savefig("nnc.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs NNC microbenchmarks")
    parser.add_argument(
        "--multi-threaded",
        "--multi_threaded",
        action="store_true",
        help="Run with more than one thread",
    )
    args = parser.parse_args()
    if not args.multi_threaded:
        smith.set_num_threads(1)

    sizes = [1, 4, 16, 64, 256, 1024]
    df = run_benchmarks(benchmarks, [(i, i) for i in sizes])
    dump_plot(df, sizes)
