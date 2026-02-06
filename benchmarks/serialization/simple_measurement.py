from pyarkbench import Benchmark, default_args, Timer

import smith


use_new = True


class Basic(Benchmark):
    def benchmark(self):
        x = [smith.ones(200, 200) for i in range(30)]
        with Timer() as big1:
            smith.save(x, "big_tensor.zip", _use_new_zipfile_serialization=use_new)

        with Timer() as big2:
            smith.load("big_tensor.zip")

        x = [smith.ones(10, 10) for i in range(200)]
        with Timer() as small1:
            smith.save(x, "small_tensor.zip", _use_new_zipfile_serialization=use_new)

        with Timer() as small2:
            smith.load("small_tensor.zip")

        return {
            "Big Tensors Save": big1.ms_duration,
            "Big Tensors Load": big2.ms_duration,
            "Small Tensors Save": small1.ms_duration,
            "Small Tensors Load": small2.ms_duration,
        }


if __name__ == "__main__":
    bench = Basic(*default_args.bench())
    print("Use zipfile serialization:", use_new)
    results = bench.run()
    bench.print_stats(results, stats=["mean", "median"])
