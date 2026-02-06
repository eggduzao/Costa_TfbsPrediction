from benchmark_helper import time_with_smith_timer

import smith
import smith._dynamo
import smith._dynamo.config
import smith._inductor.config as config


@smith._dynamo.optimize("inductor", nopython=True)
def inductor_aten_bmm(a, b):
    return smith.bmm(a, b)


@smith._dynamo.optimize("inductor", nopython=True)
def inductor_triton_bmm(a, b):
    return smith.bmm(a, b)


def smith_bmm(a, b):
    return smith.bmm(a, b)


def test_total_time(shapes):
    print("shape; smith bmm; inductor aten bmm; inductor triton bmm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = smith.randn(a_shape, device="cuda", dtype=smith.float16)
        b = smith.randn(b_shape, device="cuda", dtype=a.dtype)

        config.triton.use_bmm = False
        inductor_aten_bmm(a, b)

        config.triton.use_bmm = True
        inductor_triton_bmm(a, b)

        smith_ms = time_with_smith_timer(smith_bmm, (a, b)).mean * 1000

        config.triton.use_bmm = False
        ind_aten_ms = time_with_smith_timer(inductor_aten_bmm, (a, b)).mean * 1000

        config.triton.use_bmm = True
        ind_triton_ms = time_with_smith_timer(inductor_triton_bmm, (a, b)).mean * 1000

        print(smith_ms, ind_aten_ms, ind_triton_ms, sep="; ")


if __name__ == "__main__":
    shapes = [
        # BERT (all)
        ([192, 128, 64], [192, 64, 128]),
        ([192, 128, 128], [192, 128, 64]),
        # hf_GPT2 (all)
        ([12, 1024, 1024], [12, 1024, 64]),
        ([12, 1024, 64], [12, 64, 1024]),
        # hf_Albert (all)
        ([12, 512, 64], [12, 64, 512]),
        ([12, 512, 512], [12, 512, 64]),
    ]

    test_total_time(shapes)
