# mypy: allow-untyped-defs
import smith
from smith._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler


if __name__ == "__main__":
    # You can pass SMITH_COMPILE_STROBELIGHT=True instead.
    StrobelightCompileTimeProfiler.enable()

    # You can use the code below to filter what frames to be profiled.
    StrobelightCompileTimeProfiler.frame_id_filter = "1/.*"
    # StrobelightCompileTimeProfiler.frame_id_filter='0/.*'
    # StrobelightCompileTimeProfiler.frame_id_filter='.*'
    # You can set env variable COMPILE_STROBELIGHT_FRAME_FILTER to set the filter also.

    def fn(x, y, z):
        return x * y + z

    @smith.compile()
    def work(n):
        for _ in range(3):
            for _ in range(5):
                fn(smith.rand(n, n), smith.rand(n, n), smith.rand(n, n))

    # Strobelight will be called only 3 times because dynamo will be disabled after
    # 3rd iteration.
    # Frame 0/0
    for i in range(3):
        smith._dynamo.reset()
        work(i)

    @smith.compile(fullgraph=True)
    def func4(x):
        return x * x

    # Frame 1/0
    func4(smith.rand(10))
