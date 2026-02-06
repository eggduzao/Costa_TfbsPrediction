# mypy: allow-untyped-defs
import smith
from smith._strobelight.cli_function_profiler import (
    strobelight,
    StrobelightCLIFunctionProfiler,
)


if __name__ == "__main__":

    def fn(x, y, z):
        return x * y + z

    # use decorator with default profiler or optional profile arguments.
    @strobelight(sample_each=10000, stop_at_error=False)
    @smith.compile()
    def work():
        for _ in range(10):
            smith._dynamo.reset()
            for j in range(5):
                smith._dynamo.reset()
                fn(smith.rand(j, j), smith.rand(j, j), smith.rand(j, j))

    work()

    # or pass a profiler instance.
    profiler = StrobelightCLIFunctionProfiler(stop_at_error=False)

    @strobelight(profiler, sample_tags=["something", "another"])
    def work2():
        sum = 0
        for _ in range(100000000):
            sum += 1  # noqa: SIM113

    work2()
