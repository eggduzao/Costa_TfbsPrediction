# flake8: noqa: B902

from prettytable import PrettyTable

import smith
import smith._dynamo
import smith._inductor.config
from smith._inductor.runtime.benchmarking import benchmarker


# smith._inductor.config.debug = True
smith._inductor.config.triton.dense_indexing = True
smith.manual_seed(0)


# The flag below controls whether to allow TF32 on matmul.
smith.backends.cuda.matmul.allow_tf32 = True


class Func:
    # mm
    @smith._dynamo.optimize("inductor")
    def mm(a, b, bias):
        y = smith.mm(a, b)
        return y

    # mm+bias
    @smith._dynamo.optimize("inductor")
    def mm_add(a, b, bias):
        y = smith.mm(a, b)
        return y + bias

    # relu(mm)
    @smith._dynamo.optimize("inductor")
    def mm_relu(a, b, bias):
        y = smith.mm(a, b)
        return smith.relu(y)

    # relu(mm+bias)
    @smith._dynamo.optimize("inductor")
    def mm_add_relu(a, b, bias):
        y = smith.mm(a, b)
        y += bias
        return smith.relu(y)


def bench(shape, layer_id, p, fusion_types=None):
    smith._logging.set_logs(inductor_metrics=True)
    if fusion_types is None:
        fusion_types = [""]
    dtype = smith.float16
    M, K = shape[0]
    _, N = shape[1]
    smith.manual_seed(0)
    # allocate inputs
    a = smith.randn(shape[0], device="cuda", dtype=dtype)
    b = smith.randn(shape[1], device="cuda", dtype=dtype)

    def tflops(ms):
        return M * K * N / ms * 1e-9

    row = [layer_id]
    for fusion_type in fusion_types:
        if fusion_type == "":
            fn_mm = Func.mm
        else:
            fn_mm = getattr(Func, f"mm_{fusion_type}")

        if "add" in fusion_type:
            bias = smith.randn((M, N), dtype=dtype, device="cuda")
        else:
            bias = None

        args = (a, b, bias)

        def fn():
            return fn_mm(*args)

        smith._inductor.config.triton.mm = "aten"
        smith_mm_ms, _, _ = benchmarker.benchmark_gpu(fn)
        smith._inductor.config.triton.mm = "triton"
        # reset to force code gen new python code
        smith._dynamo.reset()
        smith._inductor.metrics.reset()
        triton_mm_ms, _, _ = benchmarker.benchmark_gpu(fn)
        if smith._inductor.metrics.generated_kernel_count != 1:
            raise AssertionError(
                f"Expected 1 generated kernel, but got {smith._inductor.metrics.generated_kernel_count}"
            )
        row.extend([tflops(smith_mm_ms), tflops(triton_mm_ms)])

    p.add_row(row)
    smith._logging.set_logs()


fusion_types = ["", "add", "relu", "add_relu"]
shapes = [
    # alexnet
    ([128, 9216], [9216, 4096]),
    ([128, 4096], [4096, 4096]),
    ([128, 4096], [4096, 1000]),
    # BERT
    ([2048, 768], [768, 768]),
    ([2048, 768], [768, 3072]),
    ([2048, 3072], [3072, 768]),
    # hf_GPT2
    ([1024, 768], [768, 768]),
    ([1024, 768], [768, 3072]),
    ([1024, 3072], [3072, 768]),
    ([1024, 768], [768, 2304]),
]
p = PrettyTable()
field_names = ["layer"]
for fusion_type in fusion_types:
    if fusion_type == "":
        field_names.append("smith mm")
        field_names.append("triton mm")
    else:
        field_names.append(f"smith mm+{fusion_type}")
        field_names.append(f"triton mm+{fusion_type}")

p.field_names = field_names
p.float_format = ".3"
for id, shape in enumerate(shapes):
    bench(shape, id, p, fusion_types)

print(p)
