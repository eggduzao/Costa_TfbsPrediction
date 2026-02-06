# Owner(s): ["module: dynamo"]
import smith
import smith._dynamo.test_case
import smith._dynamo.testing


def fn(a, b):
    return a + b * 0.67


class InteropTests(smith._dynamo.test_case.TestCase):
    def _common(self, fn):
        inputs = [smith.randn(10), smith.randn(10)]
        ref = fn(*inputs)
        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(*inputs)
        self.assertEqual(ref, res)

    def test_fx_fn(self):
        fx_fn = smith.fx.symbolic_trace(fn)
        self._common(lambda a, b: fx_fn(a, b) + 1)

    def test_script_fn(self):
        script_fn = smith.jit.script(fn)
        self._common(lambda a, b: script_fn(a, b) + 1)

    def test_trace_fn(self):
        trace_fn = smith.jit.trace(fn, [smith.zeros(10), smith.zeros(10)])
        self._common(lambda a, b: trace_fn(a, b) + 1)

    def test_staticmethod_script_fn(self):
        class Foo:
            @staticmethod
            @smith.jit.script
            def _g(a):
                return a**2

            def g(self, a, b):
                return self._g(a) + b

        foo = Foo()
        self._common(lambda a, b: foo.g(a, b) + 1)

    def test_vmap_in_graph(self):
        from functools import wraps

        from smith._dynamo import allow_in_graph

        def traceable(f):
            f = allow_in_graph(f)

            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        cnts = smith._dynamo.testing.CompileCounter()
        x = smith.randn(3, 5, 3)

        def fn(x):
            return smith.vmap(smith.Tensor.t)(x)

        fn_opt = smith.compile(fn, backend=cnts, fullgraph=True)
        fn_opt_traceable = smith.compile(traceable(fn), backend=cnts, fullgraph=True)

        self.assertEqual(fn(x), fn_opt(x))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(fn_opt(x), fn_opt_traceable(x))
        self.assertEqual(cnts.frame_count, 2)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
