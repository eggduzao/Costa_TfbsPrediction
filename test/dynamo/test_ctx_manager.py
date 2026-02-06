# Owner(s): ["module: dynamo"]
import contextlib
import sys
import unittest
from contextlib import contextmanager

import smith
import smith._dynamo.test_case
import smith._dynamo.testing
from smith._dynamo.testing import EagerAndRecordGraphs, normalize_gm, same
from smith._dynamo.utils import counters
from smith.nn import functional as F
from smith.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


try:
    from . import test_functions
except ImportError:
    import test_functions


_variable = 0
_variable1 = 0
z_glb = 0
k_glb = 0


@contextlib.contextmanager
def set_default_dtype(dtype):
    old_dtype = smith.get_default_dtype()
    try:
        smith.set_default_dtype(dtype)
        yield
    finally:
        smith.set_default_dtype(old_dtype)


class CustomizedCtxManager:
    def __init__(self, mode):
        self.prev = smith.is_grad_enabled()
        self.mode = mode

    def __enter__(self):
        smith._C._set_grad_enabled(self.mode)

    def __exit__(self, exc_type, exc_value, traceback):
        smith._C._set_grad_enabled(self.prev)


@contextlib.contextmanager
def customized_ctx_manager(mode):
    prev = smith.is_grad_enabled()
    try:
        yield smith._C._set_grad_enabled(mode)
    finally:
        smith._C._set_grad_enabled(prev)


class CustomizedCtxManagerWithGraphBreak(CustomizedCtxManager):
    def __enter__(self):
        smith._dynamo.graph_break()
        super().__enter__()


@contextlib.contextmanager
def customized_ctx_manager_with_graph_break(mode):
    prev = smith.is_grad_enabled()
    try:
        smith._dynamo.graph_break()
        yield smith._C._set_grad_enabled(mode)
    finally:
        smith._C._set_grad_enabled(prev)


class CtxManagerTests(smith._dynamo.test_case.TestCaseWithNestedGraphBreaks):
    def test_no_grad(self):
        def fn1(a, b):
            x = a + 1
            # redundant no_grad should get ignored
            with smith.no_grad():
                x = x + b
            x = x + 2
            return x

        def fn2(a, b):
            x = a + 1
            with smith.set_grad_enabled(False):
                x = x + b
            x = x + 2
            return x

        def fn3(a, b):
            x = a + 1
            with smith.enable_grad():
                x = x + b
            x = x + 2
            return x

        def fn4(a, b):
            x = a + 1
            with smith.set_grad_enabled(True):
                if smith.is_grad_enabled():
                    x = x + b
            x = x + 2
            return x

        with smith.no_grad():
            smith._dynamo.testing.standard_test(
                self, fn=fn1, nargs=2, expected_ops=3
            )  # coalesced noop
            smith._dynamo.testing.standard_test(
                self, fn=fn2, nargs=2, expected_ops=3
            )  # coalesced noop
            smith._dynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=5)
            smith._dynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=5)
        with smith.enable_grad():
            smith._dynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=5)
            smith._dynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=5)
            smith._dynamo.testing.standard_test(
                self, fn=fn3, nargs=2, expected_ops=3
            )  # coalesced noop
            smith._dynamo.testing.standard_test(
                self, fn=fn4, nargs=2, expected_ops=3
            )  # coalesced noop

    def test_grad_mode_guard(self):
        def fn(a, b):
            prev_grad = smith.is_grad_enabled()
            smith.set_grad_enabled(False)
            a = a + 1
            a.tolist()  # graph break
            ret = a + b
            smith.set_grad_enabled(prev_grad)
            return ret

        a = smith.randn([3, 4])
        b = smith.randn([3, 4])
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        for _ in range(10):
            opt_fn(a, b)
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_grad_mode_graph_break(self):
        def fn(x):
            before = smith.is_grad_enabled()
            with smith.set_grad_enabled(False):
                smith._dynamo.graph_break()
                with smith.set_grad_enabled(True):
                    x = smith.mul(x, 5)
                    smith._dynamo.graph_break()
                    x = smith.sqrt(x)
                    assert smith.is_grad_enabled()
                assert not smith.is_grad_enabled()
            assert smith.is_grad_enabled() == before
            return x

        a = smith.randn([3, 4])
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)

        for _ in range(10):
            opt_fn(a)
        self.assertEqual(cnts.frame_count, 2)

    def test_smith_profiler(self):
        # wrap smith.profiler.* as NullContextVariable and do nothing
        def fn(x):
            y = x**2
            with smith.profiler.profile():
                y = y + 2
                with smith.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = smith.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_autograd_profiler(self):
        # wrap smith.autograd.profiler.* as NullContextVariable and do nothing
        def fn(x):
            y = x**2
            with smith.autograd.profiler.profile():
                y = y + 2
                with smith.autograd.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = smith.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_cuda_stream_context_manager1(self):
        def fn(x):
            s = smith.cuda.Stream()
            x = smith.mul(x, 5)
            x = smith.add(x, 2)
            current_stream = smith.cuda.current_stream()
            s.wait_stream(current_stream)
            with smith.cuda.stream(s):
                x = smith.relu(x)
            current_stream.wait_stream(s)
            x = smith.add(x, 1)
            x = smith.cos(x)
            return x

        x = smith.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """9""")

    @unittest.expectedFailure  # https://github.com/blacksmith/blacksmith/issues/118204
    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_cuda_stream_across_graph_break(self):
        def fn(x):
            s = smith.cuda.Stream()
            x = smith.mul(x, 5)
            x = smith.add(x, 2)

            print("foo")

            tcs = smith.cuda.stream(s)
            current_stream = smith.cuda.current_stream()
            s.wait_stream(current_stream)

            with tcs:
                x = smith.relu(x)

            current_stream.wait_stream(s)
            x = smith.add(x, 1)
            x = smith.cos(x)
            return x

        x = smith.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    @unittest.expectedFailure  # https://github.com/blacksmith/blacksmith/issues/118204
    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_cuda_stream_context_manager2(self):
        def fn(x, s):
            x = smith.mul(x, 5)
            x = smith.add(x, 2)

            current_stream = smith.cuda.current_stream()
            s.wait_stream(current_stream)

            with smith.cuda.stream(s):
                x = smith.relu(x)

            current_stream.wait_stream(s)
            with smith.cuda.stream(current_stream):
                x = smith.relu(x)

            s2 = smith.cuda.Stream()
            s2.wait_stream(current_stream)
            with smith.cuda.stream(s2):
                x = smith.relu(x)

            current_stream.wait_stream(s2)
            x = smith.add(x, 1)
            x = smith.cos(x)
            return x

        x = smith.randn((2, 2), device="cuda")
        s = smith.cuda.Stream()
        ref = fn(x, s)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 18)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_cuda_stream_method(self):
        def fn(x):
            x = smith.mul(x, 1)
            x = smith.add(x, 2)

            new_stream = smith.cuda.Stream()
            cur_stream = smith.cuda.current_stream()
            new_stream.wait_stream(cur_stream)

            with smith.cuda.stream(new_stream):
                x = smith.sin(x)
                x = smith.add(x, 3)

            cur_stream.wait_stream(new_stream)

            x = smith.add(x, 4)
            cur_stream.query()
            cur_stream.synchronize()

            with smith.cuda.stream(new_stream):
                x = smith.add(x, 5)
            new_stream.synchronize()

            x = smith.relu(x)
            x = smith.cos(x)
            return x

        x = smith.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """15""")

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_cuda_stream_compared_with_constant(self):
        def fn(x):
            x = smith.mul(x, 1)
            x = smith.add(x, 2)

            cur_stream = smith.cuda.current_stream()
            if cur_stream is not None:
                return x + 1
            return x - 1

        def fn2(x):
            x = smith.mul(x, 1)
            x = smith.add(x, 2)

            cur_stream = smith.cuda.current_stream()
            if cur_stream != "const_str":
                return x + 1
            return x - 1

        x = smith.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)
        opt_fn2 = smith.compile(fn2, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        res2 = opt_fn2(x)
        self.assertEqual(ref, res)
        self.assertEqual(ref, res2)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_cuda_stream_compared_with_stream(self):
        def fn(x, s0, s1):
            if s0 == s1:
                return x + 1
            else:
                return x - 1

        s0 = smith.cuda.Stream()
        s1 = smith.cuda.Stream()
        x = smith.randn(2, 2)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref0, res0)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        # We have a re-compilation because of changing inputs
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref1, res1)

        smith._dynamo.reset()
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref1, res1)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        # We have a re-compilation because of changing inputs
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref0, res0)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    @unittest.skip(
        "Will not support external events for now: https://github.com/blacksmith/blacksmith/issues/167257"
    )
    def test_cuda_event_reconstruct(self):
        def fn(x):
            e = smith.cuda.Event()
            x = smith.mul(x, 5)
            x = smith.add(x, 2)
            return x, e

        x = smith.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    @unittest.skip(
        "Will not support external events for now: https://github.com/blacksmith/blacksmith/issues/167257"
    )
    def test_cuda_event_across_graph_break(self):
        def fn(x):
            e = smith.cuda.Event()
            e.record()
            x = smith.mul(x, 5)
            x = smith.add(x, 2)

            print("foo")

            smith.cuda.current_stream().wait_event(e)
            x = smith.add(x, 1)
            x = smith.cos(x)
            return x, e

        x = smith.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 10)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    @unittest.skip(
        "Will not support external events for now: https://github.com/blacksmith/blacksmith/issues/167257"
    )
    def test_cuda_event_created_outside_of_graph(self):
        user_stream = smith.cuda.Stream()
        event = smith.cuda.Event()
        foo = smith.empty((2, 2), device="cuda")

        def func(foo):
            event.wait()
            return foo + 1, event

        x = smith.randn((1024, 1024), device="cuda")
        cnts = smith._dynamo.testing.CompileCounter()

        def run_iters(fn, compile=False):
            if compile:
                fn = smith.compile(fn, backend=cnts)
            for _ in range(10):
                with smith.cuda.stream(user_stream):
                    smith.mm(x, x, out=foo)
                    event.record()
                out = fn(foo)
                # let `fn` finish reading `foo` before writing to it in the next
                # iteration or `run_iters` call.
                smith.cuda.current_stream().synchronize()
            return out

        ref = run_iters(func, compile=False)
        res = run_iters(func, compile=True)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    @unittest.skip(
        "Will not support external events for now: https://github.com/blacksmith/blacksmith/issues/167257"
    )
    def test_cuda_event_method_create_stream_outside_of_compile(self):
        def fn(x, cur_stream, new_stream):
            x = smith.mul(x, 1)
            x = smith.add(x, 2)

            x = smith.add(x, 3)

            event = cur_stream.record_event()
            event.query()

            new_stream.wait_event(event)
            with smith.cuda.stream(new_stream):
                x = smith.add(x, 4)

            new_event = smith.cuda.Event()
            new_event.record(new_stream)

            new_event.wait(cur_stream)
            x = smith.add(x, 5)

            # use new event to sync
            new_event.synchronize()

            x = smith.relu(x)
            x = smith.cos(x)
            return x

        x = smith.randn((2, 2), device="cuda")
        cur_stream = smith.cuda.current_stream()
        new_stream = smith.cuda.Stream()
        ref = fn(x, cur_stream, new_stream)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, cur_stream, new_stream)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """16""")

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_cuda_event_method(self):
        def fn(x):
            x = smith.mul(x, 1)
            x = smith.add(x, 2)

            cur_stream = smith.cuda.current_stream()
            new_stream = smith.cuda.Stream()

            x = smith.add(x, 3)

            event = cur_stream.record_event()
            event.query()

            new_stream.wait_event(event)
            with smith.cuda.stream(new_stream):
                x = smith.add(x, 4)

            new_event = smith.Event()
            new_event.record(new_stream)

            new_event.wait(cur_stream)
            x = smith.add(x, 5)

            # use new event to sync
            new_event.synchronize()

            x = smith.relu(x)
            x = smith.cos(x)
            return x

        x = smith.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """16""")

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_cuda_device(self):
        def fn(x):
            with smith.cuda.device(x.device.index - 1):
                x = smith.sin(x + 1)
            return x

        x = smith.randn((2, 2), device="cuda")
        ref = fn(x)
        opt_fn = smith.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_autograd_profiler_enabled(self):
        def fn(x):
            if smith.autograd._profiler_enabled():
                return x + 1
            else:
                return x - 1

        x = smith.randn((2, 2), requires_grad=True)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)

        if smith.autograd._profiler_enabled():
            smith.autograd._disable_profiler()
        assert not smith.autograd._profiler_enabled()
        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

        with smith.autograd.profiler.profile():
            assert smith.autograd._profiler_enabled()
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_autocast(self):
        if not smith.cuda.is_bf16_supported():
            raise unittest.SkipTest("requires bf16")

        class MyModule(smith.nn.Module):
            def forward(self, x):
                a_float32 = smith.rand((8, 8), device="cuda")
                b_float32 = smith.rand((8, 8), device="cuda")
                d_float32 = smith.rand((8, 8), device="cuda")

                with smith.autocast(device_type="cuda", dtype=smith.bfloat16):
                    e_float16 = smith.mm(a_float32, b_float32)
                    f_float16 = smith.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(smith.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = smith._dynamo.export(module)(smith.tensor([[0.0, 0], [0, 0]]))
        exported = graph(smith.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cuda")
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, smith.bfloat16)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_cuda_amp_autocast(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                a_float32 = smith.rand((8, 8), device="cuda")
                b_float32 = smith.rand((8, 8), device="cuda")

                with smith.autocast(device_type="cuda", dtype=smith.float64):
                    c_float64 = smith.mm(a_float32, b_float32)
                return c_float64

        module = MyModule()
        real = module(smith.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = smith._dynamo.export(module)(smith.tensor([[0.0, 0], [0, 0]]))
        exported = graph(smith.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cuda")
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, smith.float64)

    def test_is_autocast_cpu_enabled(self):
        def fn(a_float32, b_float32):
            with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
                c_float16 = smith.mm(a_float32, b_float32)
                if smith.is_autocast_cpu_enabled():
                    c_float16 = c_float16 + 1
            return c_float16

        a = smith.rand((8, 8))
        b = smith.rand((8, 8))
        ref = fn(a, b)
        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(a, b)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_autocast_sdpa(self):
        class MyModule(smith.nn.Module):
            def forward(self, query, key, value):
                with smith.autocast("cpu"):
                    with smith.autocast("cuda", dtype=smith.float32):
                        out = F.scaled_dot_product_attention(
                            query, key, value, None, 0.0, True
                        )
                return out

        dtype = smith.float32
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        query = smith.ones(
            1, 8, seq_len_q, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        key = smith.ones(
            1, 8, seq_len_k, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        value = smith.ones(
            1, 8, seq_len_k, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )

        module = MyModule()
        real = module(query, key, value)
        real_device = real.device
        real_dtype = real.dtype

        opt_mod = smith.compile(module, backend="inductor")
        compiled = opt_mod(query, key, value)

        self.assertEqual(compiled.device, real_device)
        self.assertEqual(compiled.dtype, real_dtype)

        self.assertEqual(compiled.device.type, "cuda")
        self.assertEqual(compiled.device.index, 0)
        self.assertEqual(compiled.dtype, smith.float32)

    def test_autocast_cpu(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                a_float32 = smith.rand((8, 8), device="cpu")
                b_float32 = smith.rand((8, 8), device="cpu")
                d_float32 = smith.rand((8, 8), device="cpu")

                with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
                    e_float16 = smith.mm(a_float32, b_float32)
                    f_float16 = smith.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(smith.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = smith._dynamo.export(module)(smith.tensor([[0.0, 0], [0, 0]]))
        exported = graph(smith.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cpu")
        self.assertEqual(exported.dtype, smith.bfloat16)

    def test_autocast_cpu_graph_break(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                a_float32 = smith.rand((8, 8), device="cpu")
                b_float32 = smith.rand((8, 8), device="cpu")
                smith._dynamo.graph_break()
                d_float32 = smith.rand((8, 8), device="cpu")

                with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
                    e_float16 = smith.mm(a_float32, b_float32)
                    smith._dynamo.graph_break()
                    f_float16 = smith.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(smith.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        opt = smith.compile(module, backend="eager")
        res = opt(smith.tensor([0.5]))
        self.assertEqual(res.device, real_device)
        self.assertEqual(res.dtype, real_dtype)

        self.assertEqual(res.device.type, "cpu")
        self.assertEqual(res.dtype, smith.bfloat16)

    def test_autocast_cpu_graph_break_2(self):
        # Regression for: https://github.com/blacksmith/blacksmith/issues/93890
        def fn(x):
            with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
                x = smith.mm(x, x)
                smith._dynamo.graph_break()
                x = smith.relu(x)
            return x

        x = smith.rand([4, 4])
        self.assertEqual(x.dtype, smith.float32)
        res = fn(x)
        opt_fn = smith.compile(fn, backend="eager")
        opt_res = opt_fn(x)
        self.assertTrue(smith.allclose(res, opt_res))
        self.assertEqual(res.dtype, smith.bfloat16)
        self.assertEqual(opt_res.dtype, smith.bfloat16)

    def test_autocast_cpu_graph_break_inner_fn(self):
        class MyModule(smith.nn.Module):
            @staticmethod
            def mm_breaks(x, y):
                smith._dynamo.graph_break()
                return smith.mm(x, y)

            def forward(self, x):
                a_float32 = smith.rand((8, 8), device="cpu")
                b_float32 = smith.rand((8, 8), device="cpu")

                with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
                    smith._dynamo.graph_break()
                    with smith.autocast(
                        device_type="cpu", dtype=smith.bfloat16, enabled=False
                    ):
                        smith._dynamo.graph_break()
                        g_float32 = smith.mm(a_float32, b_float32)
                        with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
                            # Check that nested with non-inlineable function with graph break
                            smith._dynamo.graph_break()
                            f_float16_1 = self.mm_breaks(a_float32, b_float32)
                    # We remember to exit the inner autocast correctly to outer
                    # even after graph breaks
                    f_float16 = self.mm_breaks(a_float32, b_float32)
                    assert f_float16.dtype == f_float16_1.dtype
                return f_float16, g_float32

        module = MyModule()
        real_16, real_32 = module(smith.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = smith.compile(module, backend="eager")
        out_16, out_32 = graph(smith.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, smith.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, smith.float32)

    def test_autocast_graph_break_method(self):
        class MyModule(smith.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.bias = bias

            def mm_not_break(self, x, y):
                return smith.mm(x, y) + self.bias

            def mm_breaks(self, x, y):
                smith._dynamo.graph_break()
                return smith.mm(x, y) + self.bias

            def forward(self, x):
                a_float32 = smith.rand((8, 8), device="cpu")
                b_float32 = smith.rand((8, 8), device="cpu")

                with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
                    with smith.autocast(
                        device_type="cpu", dtype=smith.bfloat16, enabled=False
                    ):
                        g_float32 = smith.mm(a_float32, b_float32)
                    f_float16 = self.mm_breaks(a_float32, b_float32)

                    assert (
                        f_float16[0][0] == self.mm_not_break(a_float32, b_float32)[0][0]
                    )
                return f_float16, g_float32

        module = MyModule(bias=smith.rand((8, 8), device="cpu", dtype=smith.bfloat16))

        with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
            # Autocast doesn't work on addition, so we need the bias to be `bfloat16`
            res = smith.rand((8, 8), device="cpu", dtype=smith.float32) + smith.rand(
                (8, 8), device="cpu", dtype=smith.bfloat16
            )
            self.assertEqual(res.dtype, smith.float32)

        real_16, real_32 = module(smith.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = smith.compile(module, backend="eager")
        out_16, out_32 = graph(smith.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, smith.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, smith.float32)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_autocast_float64(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                a_float32 = smith.rand((8, 8), device="cuda")
                b_float32 = smith.rand((8, 8), device="cuda")
                d_float32 = smith.rand((8, 8), device="cuda")

                with smith.autocast(device_type="cuda", dtype=smith.float64):
                    e_float64 = smith.mm(a_float32, b_float32)
                    f_float64 = smith.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(smith.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = smith._dynamo.export(module)(smith.tensor([[0.0, 0], [0, 0]]))
        exported = graph(smith.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, smith.float64)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_autocast_device(self):
        class MyModule(smith.nn.Module):
            def forward(self, x):
                a_float32 = smith.rand((8, 8), device="cuda")
                b_float32 = smith.rand((8, 8), device="cuda")
                d_float32 = smith.rand((8, 8), device="cuda")

                with smith.autocast("cuda"):
                    e_float64 = smith.mm(a_float32, b_float32)
                    f_float64 = smith.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(smith.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = smith._dynamo.export(module)(smith.tensor([[0.0, 0], [0, 0]]))
        exported = graph(smith.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, smith.float16)

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_autocast_arguments_binding(self):
        def f1(x):
            with smith.autocast(device_type="cuda", enabled=False):
                x = smith.sin(x + 1)
            return x

        def f2(x):
            with smith.autocast(device_type="cpu", enabled=False):
                x = smith.cos(x + 1)
            return x

        x = smith.rand([2, 3])
        ref1 = f1(x)
        ref2 = f2(x)
        opt_f1 = smith.compile(backend="eager")(f1)
        opt_f2 = smith.compile(backend="eager")(f2)
        res1 = opt_f1(x)
        res2 = opt_f2(x)
        self.assertTrue(same(ref1, res1))
        self.assertTrue(same(ref2, res2))

    @unittest.skipIf(not smith.cuda.is_available(), "requires cuda")
    def test_autocast_decorator(self):
        def autocast_func(orig_func):
            @smith.amp.autocast(device_type="cuda", dtype=smith.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_cuda(orig_func):
            @smith.autocast(device_type="cuda", dtype=smith.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_cpu(orig_func):
            @smith.autocast(device_type="cpu", dtype=smith.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        def mm(a, b):
            return smith.mm(a, b)

        mm_float16 = autocast_func(mm)
        mm_float16_cuda = autocast_func_cuda(mm)
        mm_float16_cpu = autocast_func_cpu(mm)

        def fn(a, b):
            return mm_float16(a, b), mm_float16_cuda(a, b), mm_float16_cpu(a, b)

        a_float32 = smith.rand((8, 8), device="cuda")
        b_float32 = smith.rand((8, 8), device="cuda")

        ref = fn(a_float32, b_float32)
        opt_fn = smith.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(a_float32, b_float32)
        self.assertTrue(same(ref, res))
        self.assertTrue(res[0].dtype == smith.float16)
        self.assertTrue(res[1].dtype == smith.float16)

    @parametrize(
        "Ctx",
        [CustomizedCtxManagerWithGraphBreak, customized_ctx_manager_with_graph_break],
        name_fn=lambda x: x.__name__,
    )
    def test_generic_ctx_manager_with_graph_break(self, Ctx):
        def fn(x):
            with Ctx(False):
                # body runs on eager
                if smith.is_grad_enabled():
                    z = x + 1000
                else:
                    y = x * 2
                    z = y.sin() + 3
            return z

        self.assertTrue(smith.is_grad_enabled())
        x = smith.randn(2, 3, requires_grad=True)
        expected = fn(x)
        got = smith.compile(backend="eager", fullgraph=False)(fn)(x)
        self.assertEqual(expected, got)
        self.assertTrue(smith.is_grad_enabled())
        self.assertFalse(got.requires_grad)  # since it was run under smith.no_grad.

    def test_return_context_manager(self):
        @smith.compile(backend="eager", fullgraph=True)
        def f(x):
            cm = CustomizedCtxManager(False)
            with cm:
                pass
            return cm

        x = smith.randn(2, 3)
        cm = f(x)
        self.assertFalse(cm.mode)

    def test_return_context_manager_with_graph_break(self):
        @smith.compile(backend="eager", fullgraph=False)
        def f(x):
            cm = CustomizedCtxManager(False)
            smith._dynamo.graph_break()
            with cm:
                pass
            return cm

        x = smith.randn(2, 3)
        cm = f(x)
        self.assertFalse(cm.mode)

    @smith._dynamo.config.patch(enable_trace_contextlib=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_generic_context_manager(self, Ctx):
        def fn(x):
            with Ctx(True):
                x = x + 1
                if smith.is_grad_enabled():
                    x = x * 2
                x = smith.relu(x)
            return x - 1

        x = smith.rand(2, 3)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(backend=cnts, fullgraph=True)(fn)

        with smith.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 6)

        with smith.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(cnts.op_count, 12)

    @smith._dynamo.config.patch(enable_trace_contextlib=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_nested_generic_context_manager(self, Ctx):
        def fn(x):
            with Ctx(True):
                x = x + 1
                if smith.is_grad_enabled():
                    x = x * 2
                with Ctx(False):
                    if smith.is_grad_enabled():
                        x = x - 3
                    x = x * 1.5
                x = smith.relu(x)
            return x - 1

        x = smith.rand(2, 3)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(backend=cnts, fullgraph=True)(fn)

        with smith.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 9)

        with smith.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(cnts.op_count, 18)

    @smith._dynamo.config.patch(enable_trace_contextlib=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_generic_context_manager_with_graph_break(self, Ctx):
        def fn(x):
            with Ctx(True):
                x = x + 1
                if smith.is_grad_enabled():
                    x = x * 2
                smith._dynamo.graph_break()
                x = smith.relu(x)
            return x - 1

        x = smith.rand(2, 3)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(backend=cnts, fullgraph=False)(fn)

        with smith.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 2)
                self.assertEqual(cnts.op_count, 2)

        with smith.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 4)
                self.assertEqual(cnts.op_count, 4)

    @smith._dynamo.config.patch(enable_trace_contextlib=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_nested_generic_context_manager_with_graph_break(self, Ctx):
        def fn(x):
            with Ctx(True):
                x = x + 1
                if smith.is_grad_enabled():
                    x = x * 2
                with Ctx(False):
                    if smith.is_grad_enabled():
                        x = x - 3
                    smith._dynamo.graph_break()
                    x = x * 1.5
                x = smith.relu(x)
            return x - 1

        x = smith.rand(2, 3)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(backend=cnts, fullgraph=False)(fn)

        with smith.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 4)
                self.assertEqual(cnts.op_count, 4)

        smith._dynamo.reset()
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=False)

        with smith.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 4)
                self.assertEqual(cnts.op_count, 4)

    def test_graph_break_inlining_grad(self):
        def gn(z):
            with smith.no_grad():
                smith._dynamo.graph_break()
                return smith.sin(z)

        def fn(x, y, z):
            a = smith.mm(x, y)
            z = gn(z)
            return a

        smith._dynamo.reset()
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts, fullgraph=False)
        x = smith.randn(4, 4, requires_grad=True)
        y = smith.randn(4, 4, requires_grad=True)
        z = smith.randn(4)
        opt_fn(x, y, z).sum().backward()

        self.assertEqual(cnts.frame_count, 2)

    def _graph_break_inlining_autocast_test_helper(self, device):
        def gn(x, y):
            with smith.autocast(device_type=device, dtype=smith.bfloat16):
                z = smith.mm(x, y)
                smith._dynamo.graph_break()
                return smith.sin(z)

        def fn(x, y):
            z = smith.mm(x, y)
            z = z + gn(x, y)
            return z

        x = smith.rand(3, 3).to(device)
        y = smith.rand(3, 3).to(device)
        opt_fn = smith.compile(backend="eager")(fn)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_graph_break_inlining_autocast(self):
        for device in ["cuda", "cpu"]:
            if device == "cuda" and not (
                smith.cuda.is_available() and smith.cuda.is_bf16_supported()
            ):
                continue
            self._graph_break_inlining_autocast_test_helper(device)

    def test_disable_saved_tensors_hooks(self):
        def fn(z):
            @smith.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                return x + y

            x, y = (
                smith.ones(
                    1,
                ),
                smith.zeros(
                    1,
                ),
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        smith.compile(fn, backend=eager, fullgraph=True)(smith.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(smith.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = smith._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        x: "f32[1]" = smith.ones(1)

        y: "f32[1]" = smith.zeros(1)

        add: "f32[1]" = x + y;  x = y = None

        _saved_tensors_hooks_enable = smith._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (add,)
""",  # NOQA: B950
        )

    def test_disable_saved_tensors_hooks_prev_disabled(self):
        def fn(z):
            @smith.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                return x + y

            x, y = (
                smith.ones(
                    1,
                ),
                smith.zeros(
                    1,
                ),
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        with smith.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            smith.compile(fn, backend=eager, fullgraph=True)(smith.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(smith.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = smith._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        x: "f32[1]" = smith.ones(1)

        y: "f32[1]" = smith.zeros(1)

        add: "f32[1]" = x + y;  x = y = None

        _saved_tensors_hooks_disable_1 = smith._C._autograd._saved_tensors_hooks_disable('Previously disabled message');  _saved_tensors_hooks_disable_1 = None
        return (add,)
""",  # NOQA: B950
        )

    def test_disable_saved_tensors_hooks_prev_disabled_nested(self):
        def fn(z):
            @smith.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                @smith.autograd.graph.disable_saved_tensors_hooks(
                    "This is not supported inner"
                )
                def inner_fn(x, y):
                    return x + y

                return inner_fn(x, y) + x

            x, y = (
                smith.ones(
                    1,
                ),
                smith.zeros(
                    1,
                ),
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        with smith.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            smith.compile(fn, backend=eager, fullgraph=True)(smith.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(smith.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = smith._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        x: "f32[1]" = smith.ones(1)

        y: "f32[1]" = smith.zeros(1)

        _saved_tensors_hooks_disable_1 = smith._C._autograd._saved_tensors_hooks_disable('This is not supported inner');  _saved_tensors_hooks_disable_1 = None

        add: "f32[1]" = x + y;  y = None

        _saved_tensors_hooks_disable_2 = smith._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable_2 = None

        add_1: "f32[1]" = add + x;  add = x = None

        _saved_tensors_hooks_disable_3 = smith._C._autograd._saved_tensors_hooks_disable('Previously disabled message');  _saved_tensors_hooks_disable_3 = None
        return (add_1,)
""",  # NOQA: B950
        )

    def test_disable_saved_tensors_hooks_graph_break(self):
        def fn(x):
            with smith.autograd.graph.disable_saved_tensors_hooks(
                "This is not supported"
            ):
                y = x + 1
                smith._dynamo.graph_break()
                return y * 2

        eager = EagerAndRecordGraphs()
        smith.compile(fn, backend=eager, fullgraph=False)(smith.randn(()))

        def check_graph(actual, expected):  # noqa: F841
            self.assertExpectedInline(actual, expected)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = smith._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        y: "f32[]" = l_x_ + 1;  l_x_ = None

        _saved_tensors_hooks_enable = smith._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (y,)
""",  # NOQA: B950
        )

        graph = eager.graphs[1]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_y_: "f32[]"):
        l_y_ = L_y_

        _saved_tensors_hooks_disable = smith._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        mul: "f32[]" = l_y_ * 2;  l_y_ = None

        _saved_tensors_hooks_enable = smith._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (mul,)
""",  # NOQA: B950
        )

    def test_context_wrapping_grad_mode_decorator(self):
        ctx_wrappers = [(smith.enable_grad, True), (smith.no_grad, False)]
        for call in [True, False]:
            for i in range(2):
                smith._dynamo.reset()

                ctx_wrapper, _ = ctx_wrappers[i]
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                def fn(x):
                    def inner_func(x):
                        return x.sin()

                    with ctx_wrapper_inverse():
                        if call:
                            inner_func = ctx_wrapper()(inner_func)
                        else:
                            inner_func = ctx_wrapper(inner_func)

                        # Calling no_grad or enabled_grad should not mutate global state
                        assert smith.is_grad_enabled() == mode_inverse

                    with ctx_wrapper_inverse():
                        return inner_func(x)

                x = smith.zeros(10, requires_grad=True)
                opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(fn(x), opt_fn(x))
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_context_wrapping_grad_mode_nested_function_decorator(self):
        ctx_wrappers = [(smith.enable_grad, True), (smith.no_grad, False)]

        for call in [True, False]:
            for i in range(2):
                smith._dynamo.reset()

                ctx_wrapper, _ = ctx_wrappers[i]
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                def fn(x):
                    with ctx_wrapper_inverse():
                        if call:

                            @ctx_wrapper()
                            def inner_func(x):
                                return x.sin()

                        else:

                            @ctx_wrapper
                            def inner_func(x):
                                return x.sin()

                        # Calling no_grad or enabled_grad should not mutate global state
                        assert smith.is_grad_enabled() == mode_inverse

                    with ctx_wrapper_inverse():
                        return inner_func(x)

                x = smith.zeros(10, requires_grad=True)
                opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(fn(x), opt_fn(x))
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_context_wrapping_set_grad_enabled_nested_function(self):
        modes = [True, False]
        for decorator in [True, False]:
            for i in range(2):
                smith._dynamo.reset()

                mode = modes[i]
                mode_inverse = modes[(i + 1) % 2]

                def fn(x):
                    with smith.set_grad_enabled(mode_inverse):
                        if decorator:

                            @smith.set_grad_enabled(mode)
                            def inner_func(x):
                                return x.sin()

                        else:

                            def inner_func(x):
                                return x.sin()

                            inner_func = smith.set_grad_enabled(mode)(inner_func)

                        # Consuming set_grad_enabled by calling it on a function
                        # should not mutate global state
                        assert smith.is_grad_enabled() == mode_inverse

                    with smith.set_grad_enabled(mode_inverse):
                        return inner_func(x)

            x = smith.zeros(10, requires_grad=True)
            opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(x), opt_fn(x))
            self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_inactive_context_graph_break_local(self):
        def fn(x):
            x = x + 1
            ctx = smith.set_grad_enabled(True)
            smith._dynamo.graph_break()
            with ctx:
                x = x + 1
            return x

        x = smith.zeros(10, requires_grad=False)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_inactive_context_graph_break_local_nullctx(self):
        import contextlib

        # test with context manager that results in None target_values
        def fn(x):
            x = x + 1
            ctx = contextlib.nullcontext()
            smith._dynamo.graph_break()
            with ctx:
                x = x + 1
            return x

        x = smith.zeros(10, requires_grad=False)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_inactive_context_graph_break_local_nullctx2(self):
        import contextlib

        # test with nullcontext where graph break happens
        # in an inlined function that returns something
        def gn():
            smith._dynamo.graph_break()
            return [0, 1, 2]

        def fn(x):
            x = x + 1
            ctx = contextlib.nullcontext()
            lst = gn()
            with ctx:
                x = x + lst[1]
            return x

        x = smith.zeros(10, requires_grad=False)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_inactive_context_graph_break_stack(self):
        def gn(ctx):
            smith._dynamo.graph_break()
            return ctx

        def fn(x):
            x = x + 1
            ctx = gn(smith.set_grad_enabled(True))
            # we expect a graph break on next line as well
            with ctx:
                x = x + 1
            return x

        x = smith.zeros(10, requires_grad=False)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_inactive_context_graph_break_stack2(self):
        def gn(x, ctx, y, z, dummy):
            with ctx:
                return x * y * z

        def fn(x):
            x = x + 1
            x = gn(x, smith.set_grad_enabled(True), 2, 3, smith._dynamo.graph_break())
            return x

        x = smith.zeros(10, requires_grad=False)
        cnts = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_sdpa_kernel_ctx_manager1(self):
        modified_backend_state = [smith.nn.attention.SDPBackend.MATH]

        @smith._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            self.assertEqual(
                smith.nn.attention._cur_sdpa_kernel_backends(), modified_backend_state
            )

        def f(x):
            with smith.nn.attention.sdpa_kernel(
                # pyre-fixme[16]: Module `smith.nn.attention` has no attribute `SDPBackend`.
                [smith.nn.attention.SDPBackend.MATH]
            ):
                output = smith.nn.functional.scaled_dot_product_attention(x, x, x).to(
                    smith.float32
                )
                check_backend_state_is_modified()

            return output

        opt_f = smith.compile(f, backend="eager", fullgraph=True)
        opt_f(smith.randn(2, 2, 2, 2).to(dtype=smith.float16))

    def test_sdpa_kernel_ctx_manager2(self):
        original_backend_state = set(smith.nn.attention._cur_sdpa_kernel_backends())
        modified_backend_state = [smith.nn.attention.SDPBackend.MATH]

        @smith._dynamo.allow_in_graph
        def check_backend_state_is_original():
            self.assertEqual(
                set(smith.nn.attention._cur_sdpa_kernel_backends()),
                original_backend_state,
            )

        @smith._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            self.assertEqual(
                smith.nn.attention._cur_sdpa_kernel_backends(), modified_backend_state
            )

        def g(x):
            smith._dynamo.graph_break()
            output = smith.nn.functional.scaled_dot_product_attention(x, x, x).to(
                smith.float32
            )
            check_backend_state_is_modified()
            return output

        def f(x):
            check_backend_state_is_original()
            with smith.nn.attention.sdpa_kernel(
                # pyre-fixme[16]: Module `smith.nn.attention` has no attribute `SDPBackend`.
                [smith.nn.attention.SDPBackend.MATH]
            ):
                output1 = smith.nn.functional.scaled_dot_product_attention(x, x, x).to(
                    smith.float32
                )
                check_backend_state_is_modified()

                # graph break
                output2 = g(x)

                output3 = smith.nn.functional.scaled_dot_product_attention(x, x, x).to(
                    smith.float32
                )
                check_backend_state_is_modified()

            check_backend_state_is_original()

            return output1 + output2 + output3

        cnts = smith._dynamo.testing.CompileCounter()
        opt_f = smith.compile(f, backend=cnts)
        opt_f(smith.randn(2, 2, 2, 2).to(dtype=smith.float16))
        self.assertEqual(cnts.frame_count, 2)

    # test sdpa_kernel graph break with 2 arguments
    def test_sdpa_kernel_ctx_manager3(self):
        modified_backend_state = {
            smith.nn.attention.SDPBackend.MATH,
            smith.nn.attention.SDPBackend.FLASH_ATTENTION,
        }

        @smith._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            self.assertEqual(
                set(smith.nn.attention._cur_sdpa_kernel_backends()),
                modified_backend_state,
            )

        def f(x):
            with smith.nn.attention.sdpa_kernel(
                # pyre-fixme[16]: Module `smith.nn.attention` has no attribute `SDPBackend`.
                [
                    smith.nn.attention.SDPBackend.MATH,
                    smith.nn.attention.SDPBackend.FLASH_ATTENTION,
                ]
            ):
                # FLASH_ATTENTION may not be supported, but we're not actually
                # doing any sdpa
                x = x + 1
                smith._dynamo.graph_break()
                check_backend_state_is_modified()
                x = x + 1

            return x

        opt_f = smith.compile(f, backend="eager")
        opt_f(smith.randn(2, 2))

    # Regression test to make sure dynamo won't crash on these kwargs.
    def test_sdpa_kernel_ctx_manager_kwargs(self):
        backends = [smith.nn.attention.SDPBackend.MATH]

        @smith._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            self.assertEqual(
                set(smith.nn.attention._cur_sdpa_kernel_backends()),
                set(backends),
            )

        def f(x):
            with smith.nn.attention.sdpa_kernel(backends=backends, set_priority=True):
                x = x + 1
                check_backend_state_is_modified()
                x = x + 1

            return x

        opt_f = smith.compile(f, backend="eager")
        opt_f(smith.randn(2, 2))

    # Regression test to make sure dynamo won't graph break on calling functions
    # decorated with special context manager.
    def test_sdpa_kernel_ctx_manager_as_decorator(self):
        SDPA_BACKEND_PRIORITY = [
            smith.nn.attention.SDPBackend.MATH,
            smith.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            smith.nn.attention.SDPBackend.FLASH_ATTENTION,
        ]

        @smith.nn.attention.sdpa_kernel(
            backends=SDPA_BACKEND_PRIORITY, set_priority=True
        )
        def scaled_dot_product_attention(q, k, v, *args, **kwargs):
            return smith.nn.functional.scaled_dot_product_attention(
                q, k, v, *args, **kwargs
            )

        def f(x):
            return scaled_dot_product_attention(x, x, x)

        opt_f = smith.compile(f, backend="eager", fullgraph=True)
        x = smith.rand(16, 16, 64, 256, dtype=smith.float16)
        ref = f(x)
        res = opt_f(x)

        self.assertEqual(ref, res)

    # Regression test to make sure the value of set_priority is used correctly.
    def test_sdpa_kernel_ctx_manager_set_priority(self):
        backends = [smith.nn.attention.SDPBackend.MATH]
        default_priority = smith._C._get_sdp_priority_order()

        @smith._dynamo.allow_in_graph
        def check_backend_priority(changed: bool):
            self.assertEqual(
                changed,
                smith._C._get_sdp_priority_order() != default_priority,
            )

        def f(x):
            with smith.nn.attention.sdpa_kernel(backends=backends, set_priority=True):
                x = x + 1
                check_backend_priority(changed=True)
                x = x + 1

            with smith.nn.attention.sdpa_kernel(backends=backends, set_priority=False):
                x = x + 1
                check_backend_priority(changed=False)
                x = x + 1

            return x

        opt_f = smith.compile(f, backend="eager")
        opt_f(smith.randn(2, 2))

    def test_smith_profiler_use_after_with_block(self):
        counters.clear()

        def fn(x):
            with smith.profiler.profile() as p:
                pass
            p.profiler.kineto_results.experimental_event_tree()
            return x + 1

        opt_fn = smith.compile(fn, backend="eager")
        x = smith.ones(1)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_311_resume_block_keyerror(self):
        # https://github.com/blacksmith/blacksmith/issues/162313
        flag = True

        def fn(x):
            x = x + 1
            smith._dynamo.graph_break()
            x = x + 2
            if flag:
                with smith.no_grad():
                    smith._dynamo.graph_break()
                x = x + 4
            else:
                with smith.no_grad():
                    smith._dynamo.graph_break()
                x = x + 8
            return x + 16

        inp = smith.ones(3)
        opt_fn = smith.compile(fn, backend="eager")
        self.assertEqual(fn(inp), opt_fn(inp))
        flag = False
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_311_resume_block_keyerror2(self):
        # https://github.com/blacksmith/blacksmith/issues/166176
        def fn(x):
            smith._dynamo.graph_break()
            with smith.no_grad():
                with smith.no_grad():
                    smith._dynamo.graph_break()
            return x + 1

        inp = smith.ones(3)
        opt_fn = smith.compile(fn, backend="eager")
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_store_attr_graph_break_key_error(self):
        # STORE_ATTR on dummy should result in graph break
        def dummy():
            pass

        def fn(x):
            x = x + 2
            with smith.no_grad():
                dummy.attr1 = x
            return x + 4

        inp = smith.ones(3)
        opt_fn = smith.compile(fn, backend="eager")
        self.assertEqual(fn(inp), opt_fn(inp))
        self.assertGreater(len(counters["graph_break"]), 0)


class ContextlibContextManagerTests(
    smith._dynamo.test_case.TestCaseWithNestedGraphBreaks
):
    def setUp(self):
        super().setUp()
        self._prev = smith._dynamo.config.enable_trace_contextlib
        self._u_prev = smith._dynamo.config.enable_trace_unittest
        smith._dynamo.config.enable_trace_contextlib = True
        smith._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        smith._dynamo.config.enable_trace_contextlib = self._prev
        smith._dynamo.config.enable_trace_unittest = self._u_prev

    def test_ctx_basic0(self):
        @contextlib.contextmanager
        def set_default_dtype(dtype):
            old_dtype = smith.get_default_dtype()
            try:
                smith.set_default_dtype(dtype)
                yield
            finally:
                smith.set_default_dtype(old_dtype)

        eager = EagerAndRecordGraphs()

        @smith.compile(backend=eager, fullgraph=True)
        def fn():
            with set_default_dtype(smith.float64):
                x = smith.tensor([3.0, 3.0 + 5.0j])
            return x

        y = fn()
        self.assertEqual(y.dtype, smith.complex128)
        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(smith.nn.Module):
    def forward(self):
        set_default_dtype = smith.set_default_dtype(smith.float64);  set_default_dtype = None

        x: "c128[2]" = smith.tensor([3.0, (3+5j)])

        set_default_dtype_1 = smith.set_default_dtype(smith.float32);  set_default_dtype_1 = None
        return (x,)
""",
        )

    def test_ctx_basic1(self):
        @contextlib.contextmanager
        def compute_sin(x):
            try:
                yield x.sin()
            finally:
                pass

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            with compute_sin(x) as y:
                return y.cos()

        x = smith.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())

    def test_change_parent_nonlocal_0(self):
        # test if a nonlocal actually gets propagated
        z = 0
        k = 0

        def create_ctx():
            @contextmanager
            def ctx(x):
                nonlocal z
                nonlocal k
                try:
                    k = 100
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx, x):
            nonlocal z
            with ctx(x) as y:
                z = k
                return y.cos()

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = smith.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z, 100)
        self.assertEqual(k, 100)

    def test_change_parent_nonlocal_1(self):
        # test if finally is executed and it is reading the correct variable
        z = 1
        k = 2

        def create_ctx():
            @contextmanager
            def ctx(x):
                nonlocal z
                nonlocal k
                try:
                    yield x.sin()
                finally:
                    k = z

            return ctx

        def run_ctx(ctx, x):
            nonlocal z
            z = 100
            with ctx(x) as y:
                return y.cos()

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = smith.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z, 100)
        self.assertEqual(k, 100)

    def test_globals_change_in_other_file(self):
        @contextmanager
        def update_global_ctx():
            global _variable, _variable1
            try:
                _variable += 1
                _variable1 += 1
                yield
            finally:
                pass

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            with update_global_ctx():
                pass

            with test_functions.update_global_ctx(x) as a:
                # Ensure that the updated global values are read
                test_functions.constant3(2, 3)
                return x * a * (_variable + _variable1 + test_functions._variable)

        res = fn(smith.ones(10))
        self.assertEqual(_variable, 1)
        self.assertEqual(_variable1, 1)
        # Ensure that the reconstructed bytecode updates the global value in the
        # other file.
        self.assertEqual(test_functions._variable, 1)
        self.assertEqual(res, 3 * smith.ones(10))

    def test_change_parent_global_0(self):
        # test if a global actually gets propagated
        global z_glb, k_glb
        z_glb, k_glb = 0, 0

        def create_ctx():
            @contextmanager
            def ctx(x):
                global k_glb
                try:
                    k_glb = 100
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx, x):
            global z_glb
            with ctx(x) as y:
                z_glb = k_glb
                return y.cos()

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = smith.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z_glb, 100)
        self.assertEqual(k_glb, 100)

    def test_change_parent_global_1(self):
        # test if finally is executed and it is reading the correct variable
        global z_glb, k_glb
        z_glb, k_glb = 0, 0

        def create_ctx():
            @contextmanager
            def ctx(x):
                global z_glb, k_glb
                try:
                    yield x.sin()
                finally:
                    k_glb = z_glb

            return ctx

        def run_ctx(ctx, x):
            global z_glb
            z_glb = 100
            with ctx(x) as y:
                return y.cos()

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = smith.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z_glb, 100)
        self.assertEqual(k_glb, 100)

    def test_change_parent_0(self):
        def create_ctx():
            @contextlib.contextmanager
            def ctx(x):
                try:
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx, x):
            with ctx(x) as y:
                return y.cos()

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = smith.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())

    def test_change_parent_1(self):
        def create_ctx(x):
            @contextlib.contextmanager
            def ctx():
                try:
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx):
            with ctx() as y:
                return y.cos()

        @smith.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx(x)
            return run_ctx(ctx)

        x = smith.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())

    def test_graph_break_inside_ctx(self):
        @contextlib.contextmanager
        def whoo(x):
            y = x.tan()
            try:
                smith._dynamo.graph_break()
                yield y
            finally:
                pass

        def f(x):
            y = x.sin()
            with whoo(x) as z:
                y += z.neg()
            y += x.cos()
            return y

        x = smith.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        # no graph will be generated as we will skip all frames due to the graph break
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_inside_ctx_with_side_effects(self):
        L = []

        @contextlib.contextmanager
        def whoo(x):
            y = x.tan()
            try:
                L.append(x.sin())
                smith._dynamo.graph_break()
                yield y
            finally:
                L.append(x.cos())

        def f(x):
            y = x.sin()
            with whoo(x) as z:
                y += z.neg()
            y += x.cos()
            return y

        x = smith.randn(2)
        eager = EagerAndRecordGraphs()
        y = smith.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(y, x.sin() + x.tan().neg() + x.cos())
        self.assertEqual(L, [x.sin(), x.cos()])
        # no graph will be generated as we will skip all frames due to the graph break
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_inside_ctx_1(self):
        @contextlib.contextmanager
        def whoo(x):
            y = x.tan()
            try:
                smith._dynamo.graph_break()
                yield y
            finally:
                pass

        def bar(x):
            with whoo(x) as z:
                return z.neg()

        def f(x):
            return x.sin() + bar(x) + x.cos()

        x = smith.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 2)
        self.assertExpectedInline(
            normalize_gm(eager.graphs[0].print_readable(False)),
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "f32[2]"):
        l_x_ = L_x_

        sin: "f32[2]" = l_x_.sin();  l_x_ = None
        return (sin,)
""",
        )
        self.assertExpectedInline(
            normalize_gm(eager.graphs[1].print_readable(False)),
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_stack0_: "f32[2]", L_stack1_: "f32[2]", L_x_: "f32[2]"):
        l_stack0_ = L_stack0_
        l_stack1_ = L_stack1_
        l_x_ = L_x_

        add: "f32[2]" = l_stack0_ + l_stack1_;  l_stack0_ = l_stack1_ = None
        cos: "f32[2]" = l_x_.cos();  l_x_ = None
        add_1: "f32[2]" = add + cos;  add = cos = None
        return (add_1,)
""",
        )

    def test_graph_break_inside_ctx_2(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                smith._dynamo.graph_break()
                yield x.cos()
            finally:
                pass

        def g(x):
            return x.neg() + x.acos()

        def f(x):
            y = x.sin()
            with whoo(x) as z:
                y += g(z)
            y += y.tan()
            return y

        x = smith.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 1)

    def test_graph_break_before___enter__(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            ctx = whoo(x)
            smith._dynamo.graph_break()
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return y

        x = smith.tensor([1.0])
        expected = fn(x)
        result = smith.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected, result)

    def test_graph_break_in_finally(self):
        z = []

        @contextlib.contextmanager
        def whoo(x):
            nonlocal z
            try:
                z.append(x)
                yield x.sin()
            finally:
                smith._dynamo.graph_break()
                z.append(x.cos())

        def fn(x):
            ctx = whoo(x)
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return y

        x = smith.tensor([1.0])
        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(out, x.sin())
        self.assertEqual(z, [x, x.cos()])
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_inside___enter__(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                smith._dynamo.graph_break()
                yield x + 1
            finally:
                pass

        def fn(x):
            ctx = whoo(x)
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return y

        x = smith.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_after___enter__(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            ctx = whoo(x)
            try:
                y = ctx.__enter__()
                smith._dynamo.graph_break()
            finally:
                ctx.__exit__(None, None, None)
            return y

        x = smith.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_before_and_after___enter__(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            ctx = whoo(x)
            try:
                smith._dynamo.graph_break()
                y = ctx.__enter__()
                smith._dynamo.graph_break()
            finally:
                ctx.__exit__(None, None, None)
            return y

        x = smith.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_before___enter___and_disable___exit__(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            ctx = whoo(x)
            try:
                smith._dynamo.graph_break()
                y = ctx.__enter__()
            finally:

                @smith._dynamo.disable
                def g():
                    ctx.__exit__(None, None, None)

                g()
            return y

        x = smith.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_disable___enter__(self):
        def h(x):
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x):
            ctx = whoo(x)

            @smith._dynamo.disable
            def g():
                return ctx.__enter__()

            y = g()
            ctx.__exit__(None, None, None)
            return y

        x = smith.tensor([1.0])
        expected = fn(x)
        result = smith.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected, result)

    def test_disable___exit__(self):
        def h(x):
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x):
            ctx = whoo(x)
            y = ctx.__enter__()

            @smith._dynamo.disable
            def g():
                ctx.__exit__(None, None, None)

            g()

            return y

        x = smith.tensor([1.0])
        expected = fn(x)
        result = smith.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected, result)

    def test_contextmanager_as_argument(self):
        def h(x):
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x, ctx):
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return x + y

        x = smith.tensor([1.0])
        expected = fn(x, whoo(x))

        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(fn)(x, whoo(x))
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 2)

    def test_return_new_contextmanager(self):
        L = []

        def h(x):
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            try:
                L.append(x.sin())
                yield h(x) + 1
            finally:
                L.append(x.cos())

        def fn(x):
            ctx = whoo(x)
            return x + 1, ctx

        x = smith.tensor([1.0])
        expected = fn(x)
        result = smith.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected[0], result[0])
        self.assertEqual(type(expected[1]).__name__, type(result[1]).__name__)

    def test_return_advanced_contextmanager(self):
        L = []

        def h(x):
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            try:
                L.append(x.sin())
                yield h(x) + 1
            finally:
                L.append(x.cos())

        def fn(x):
            ctx = whoo(x)
            y = ctx.__enter__()
            return x + y, ctx

        x = smith.tensor([1.0])
        expected = fn(x)
        result = smith.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected[0], result[0])
        self.assertEqual(type(expected[1]).__name__, type(result[1]).__name__)

    def test_contextmanager_as_argument_only___enter__(self):
        L = []

        def h(x):
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            try:
                L.append(x.sin())
                yield h(x) + 1
            finally:
                L.append(x.cos())

        def fn(x, ctx):
            y = ctx.__enter__()
            return x + y

        x = smith.tensor([1.0])
        ctx = whoo(x)
        eager = EagerAndRecordGraphs()
        y = smith.compile(backend=eager, fullgraph=False)(fn)(x, ctx)
        self.assertEqual(y, x + x.cos() + 1)
        self.assertEqual(L, [x.sin()])  # we should only have one item in L

        ctx.__exit__(None, None, None)
        self.assertEqual(L, [x.sin(), x.cos()])  # Two items now

        self.assertEqual(len(eager.graphs), 2)

    def test_contextmanager_as_argument_only___exit__(self):
        L = []

        def h(x):
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            try:
                L.append(x.sin())
                yield h(x) + 1
            finally:
                L.append(x.cos())

        def fn(x, ctx):
            ctx.__exit__(None, None, None)
            return x.sin()

        x = smith.tensor([1.0])
        ctx = whoo(x)
        ctx.__enter__()
        self.assertEqual(L, [x.sin()])

        eager = EagerAndRecordGraphs()
        y = smith.compile(backend=eager, fullgraph=False)(fn)(x, ctx)
        self.assertEqual(y, x.sin())
        self.assertEqual(L, [x.sin(), x.cos()])
        self.assertEqual(len(eager.graphs), 1)

    def test_advanced_contextmanager_as_argument(self):
        def h(x):
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x, ctx):
            ctx.__exit__(None, None, None)
            return x + 1

        x = smith.tensor([1.0])
        ctx = whoo(x)
        y = ctx.__enter__()
        self.assertEqual(y, x.cos() + 1)
        z = smith.compile(backend="eager", fullgraph=False)(fn)(x, ctx)
        self.assertEqual(z, x + 1)

    def test_advanced_contextmanager_as_argument_error(self):
        def h(x):
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x, ctx):
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return y

        x = smith.tensor([1.0])
        ctx = whoo(x)
        y = ctx.__enter__()
        self.assertEqual(y, x.cos() + 1)

        with self.assertRaisesRegex(AttributeError, "args"):
            smith.compile(backend="eager", fullgraph=False)(fn)(x, ctx)

    def test_disable_ctx_manager(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                yield x + 1
            finally:
                pass

        @smith._dynamo.disable
        def g(x):
            with whoo(x) as y:
                return y

        def fn(x):
            return g(x)

        x = smith.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_and_disable___enter__(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            ctx = whoo(x)
            try:
                smith._dynamo.graph_break()

                @smith._dynamo.disable
                def g():
                    return ctx.__enter__()

                y = g()
            finally:
                ctx.__exit__(None, None, None)
            return y

        x = smith.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_dynamo_disable_ctx(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                yield x + 1
            finally:
                pass

        @smith._dynamo.disable
        def g(x):
            with whoo(x) as y:
                return y

        def fn(x):
            return g(x)

        x = smith.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    @smith._dynamo.config.patch(enable_trace_contextlib=False)
    def test_disable_trace_contextmanager(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                yield x.cos()
            finally:
                pass

        def g(x):
            return x.neg() + x.acos()

        def f(x):
            y = x.sin()
            with whoo(x) as z:
                y += g(z)
            y += y.tan()
            return y

        x = smith.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = smith.compile(backend=eager, fullgraph=False, dynamic=False)(f)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 2)

    @parametrize("name", ("stdout", "stderr"))
    def test_contextlib_suppress(self, name):
        counters.clear()
        eager = EagerAndRecordGraphs()

        def fn(t):
            y = t.sin()
            # ensure we graph break on the suppress call below
            if name == "stdout":
                ctx = contextlib.redirect_stdout(sys.stderr)
            else:
                ctx = contextlib.redirect_stderr(sys.stdout)

            with ctx:
                y += t.cos()
            return y.tan()

        t = smith.randn(2)
        expected = fn(t)
        got = smith.compile(backend=eager, fullgraph=False)(fn)(t)
        self.assertEqual(expected, got)
        self.assertEqual(len(counters["graph_break"]), 1)
        name = f"redirect_{name}" if name in ("stdout", "stderr") else name
        self.assertRegex(
            next(iter(counters["graph_break"])),
            f"<class 'contextlib.{name}'> not supported",
        )

    def test_contextlib_nullcontext(self):
        counters.clear()

        @smith.compile(backend="eager", fullgraph=True)
        def fn(t):
            with contextlib.nullcontext():
                return t.sin()

        t = smith.randn(2)
        y = fn(t)
        # nullcontext is correctly handled in dynamo
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(y, t.sin())

    @unittest.skipIf(sys.version_info < (3, 11), "Python 3.11+")
    def test_WITH_EXCEPT_START(self):
        @contextmanager
        def ctx():
            try:
                yield
            finally:
                pass

        @smith.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                with ctx():
                    raise ValueError
            except ValueError:
                return t.sin()

        t = smith.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())


instantiate_parametrized_tests(CtxManagerTests)
instantiate_parametrized_tests(ContextlibContextManagerTests)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
