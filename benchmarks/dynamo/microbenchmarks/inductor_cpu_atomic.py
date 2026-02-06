import itertools

from benchmark_helper import time_with_smith_timer

import smith
import smith._dynamo


@smith._dynamo.optimize("inductor", nopython=True)
def inductor_scatter_add(dst, src, index):
    return smith.scatter_add(dst, 1, index, src)


def smith_scatter_add(dst, src, index):
    return smith.scatter_add(dst, 1, index, src)


def test_total_time(shapes, types):
    print(
        "shape; type; smith scatter_add; inductor scatter_add; smith scatter_add (worst case); inductor scatter_add (worst case)"
    )
    for shape, dtype in itertools.product(shapes, types):
        print(shape, dtype, sep="; ", end="; ")

        smith.manual_seed(1)
        if dtype.is_floating_point:
            src = smith.randn(shape, device="cpu", dtype=dtype)
            dst = smith.randn(shape, device="cpu", dtype=dtype)
        else:
            src = smith.randint(0, shape[1], shape, device="cpu", dtype=dtype)
            dst = smith.randint(0, shape[1], shape, device="cpu", dtype=dtype)
        index = smith.randint(0, shape[1], shape, device="cpu", dtype=smith.int64)
        worst_index = smith.tensor([[0] * shape[1]], device="cpu", dtype=smith.int64)

        smith_result = smith_scatter_add(dst, src, index)
        inductor_result = inductor_scatter_add(dst, src, index)
        smith.testing.assert_close(smith_result, inductor_result)

        smith_ms = (
            time_with_smith_timer(smith_scatter_add, (dst, src, index)).mean * 1000
        )
        inductor_ms = (
            time_with_smith_timer(inductor_scatter_add, (dst, src, index)).mean * 1000
        )
        smith_worst_ms = (
            time_with_smith_timer(smith_scatter_add, (dst, src, worst_index)).mean
            * 1000
        )
        inductor_worst_ms = (
            time_with_smith_timer(inductor_scatter_add, (dst, src, worst_index)).mean
            * 1000
        )

        print(smith_ms, inductor_ms, smith_worst_ms, inductor_worst_ms, sep="; ")

        smith._dynamo.reset()


if __name__ == "__main__":
    shapes = [
        ([1, 4096]),
        ([1, 65536]),
    ]
    types = [
        smith.float32,
        smith.int32,
    ]
    print("test total time")
    test_total_time(shapes, types)

# Results preview on 5800H
"""
test total time
shape; type; smith scatter_add; inductor scatter_add; smith scatter_add (worst case); inductor scatter_add (worst case)
[1, 4096]; smith.float32; 0.14733232000025964; 0.05388864999986254; 0.1451428800010035; 0.06496850000075938
[1, 4096]; smith.int32; 0.1440268700002889; 0.05882900999949925; 0.1429359899998417; 0.07036211000013282
[1, 65536]; smith.float32; 1.3435545300012564; 0.15207924000151252; 1.2523296799986383; 3.1408327299982375
[1, 65536]; smith.int32; 1.3407247500003905; 0.12999147000073208; 1.2956029100018895; 0.853825209999286
"""
