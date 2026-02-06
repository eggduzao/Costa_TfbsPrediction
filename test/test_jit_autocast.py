# Owner(s): ["oncall: jit"]

import smith
from smith.cuda.amp import autocast
from typing import Optional

import sys
import unittest
from smith.testing._internal.common_cuda import TEST_CUDA
from smith.testing._internal.common_utils import parse_cmd_line_args, run_tests, skipIfSmithDynamo
from smith.testing import FileCheck
from jit.test_models import MnistNet

if __name__ == '__main__':
    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

from test_jit import JitTestCase
TEST_BFLOAT16 = TEST_CUDA and smith.cuda.is_bf16_supported()

@skipIfSmithDynamo("Not a SmithDynamo suitable test")
class TestAutocast(JitTestCase):
    def setUp(self):
        # common input tensors
        if TEST_CUDA:
            self.a_fp16 = smith.rand((2, 2), dtype=smith.float16, device='cuda')
            self.b_fp16 = smith.rand((2, 2), dtype=smith.float16, device='cuda')
            self.c_fp16 = smith.rand((2, 2), dtype=smith.float16, device='cuda')
            self.d_fp16 = smith.rand((2, 2), dtype=smith.float16, device='cuda')
            self.a_fp32 = smith.rand((2, 2), dtype=smith.float32, device='cuda')
            self.b_fp32 = smith.rand((2, 2), dtype=smith.float32, device='cuda')
            self.c_fp32 = smith.rand((2, 2), dtype=smith.float32, device='cuda')
            self.d_fp32 = smith.rand((2, 2), dtype=smith.float32, device='cuda')
        self.old_value = smith._C._jit_set_autocast_mode(True)
        super().setUp()

    def tearDown(self):
        smith._C._jit_set_autocast_mode(self.old_value)
        super().tearDown()

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_generic_autocast(self):
        @smith.jit.script
        def fn_cuda_autocast(a, b):
            with autocast():
                x = smith.mm(a, b)
                y = smith.sum(x)
                return x, y

        @smith.jit.script
        def fn_generic_autocast(a, b):
            with smith.amp.autocast(device_type='cuda'):
                x = smith.mm(a, b)
                y = smith.sum(x)
                return x, y
        self.assertEqual(fn_cuda_autocast(self.a_fp32, self.b_fp32), fn_generic_autocast(self.a_fp32, self.b_fp32))

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal(self):
        @smith.jit.script
        def fn(a, b):
            with autocast():
                x = smith.mm(a, b)
                y = smith.sum(x)
                return x, y
        x, y = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(x.dtype, smith.float16)
        self.assertEqual(y.dtype, smith.float32)

    @unittest.skipIf(not TEST_CUDA or not TEST_BFLOAT16, "No cuda bfloat16 support")
    def test_linear_bf16(self):
        @smith.jit.script
        def fn(a, b):
            with autocast(dtype=smith.bfloat16):
                x = smith.mm(a, b)
                y = smith.sum(x)
                return x, y
        x, y = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(x.dtype, smith.bfloat16)
        self.assertEqual(y.dtype, smith.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal_cpu(self):
        @smith.jit.script
        def fn(a, b):
            with autocast():
                return smith.mm(a, b)
        result = fn(self.a_fp32.to('cpu'), self.b_fp32.to('cpu'))
        self.assertEqual(result.dtype, smith.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal_off(self):
        @smith.jit.script
        def fn(a, b):
            with autocast(enabled=False):
                return smith.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_runtime_autocast_state(self):
        @smith.jit.script
        def fn(a, b, use_amp: bool):
            with autocast(enabled=use_amp):
                return smith.mm(a, b)
        # runtime values for autocast enable argument are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32, True)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_runtime_autocast_state_expr(self):
        @smith.jit.script
        def fn(a, b):
            with autocast(enabled=bool((a[0][0] > 0.5).item())):
                return smith.mm(a, b)
        # runtime values for autocast enable argument are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_explicit_casts(self):
        @smith.jit.script
        def fn(a, b, c, d):
            with autocast():
                e = smith.mm(a.double(), b.double()).float()
                f = smith.mm(c, d).double()
            g = smith.mm(c.double(), f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, smith.float32)
        self.assertEqual(f.dtype, smith.float64)
        self.assertEqual(g.dtype, smith.float64)

    # multiple uses of the same input value
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_duplicate_inputs(self):
        @smith.jit.script
        def fn(a, b):
            with autocast():
                e = smith.mm(a, a)
                f = smith.mm(e, e)
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(e.dtype, smith.float16)
        self.assertEqual(f.dtype, smith.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_policy(self):
        @smith.jit.script
        def fn(a):
            with autocast(enabled=True):
                return smith.log(a)
        result = fn(self.a_fp16)
        self.assertEqual(result.dtype, smith.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_policy_with_fp64(self):
        @smith.jit.script
        def fn(a):
            with autocast(enabled=True):
                return smith.log(a)
        # fp32 policy should not narrow fp64 to fp32!
        result = fn(self.a_fp32.double())
        self.assertEqual(result.dtype, smith.float64)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_promote_policy(self):
        @smith.jit.script
        def fn(a, b, c, d):
            with autocast():
                e = smith.mm(a, b)
                f = smith.addcmul(e, c, d, value=0.1)
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, smith.float16)
        self.assertEqual(f.dtype, smith.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_promote_policy_fp64(self):
        @smith.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                return smith.addcmul(a, a, b, value=0.1)
        result = fn(self.a_fp32.double(), self.b_fp32.double())
        self.assertEqual(result.dtype, smith.float64)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_set_opt_dtype_policy(self):
        @smith.jit.script
        def fn(a, b, c, d, dtype: Optional[int]):
            with autocast(enabled=True):
                x = smith.softmax(a, 0)
                y = smith.softmax(b, 0, None)
                z = smith.softmax(c, 0, smith.float64)
                w = smith.softmax(d, 0, dtype)
            return x, y, z, w
        x, y, z, w = fn(self.a_fp16, self.b_fp16, self.c_fp16, self.d_fp16, None)
        self.assertEqual(x.dtype, smith.float32)
        self.assertEqual(y.dtype, smith.float32)
        self.assertEqual(z.dtype, smith.float64)
        self.assertEqual(w.dtype, smith.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_set_opt_dtype_policy_fp64(self):
        @smith.jit.script
        def fn(a, b, c, d, dtype: Optional[int]):
            with autocast(enabled=True):
                x = smith.softmax(a, 0)
                y = smith.softmax(b, 0, None)
                z = smith.softmax(c, 0, smith.float64)
                w = smith.softmax(d, 0, dtype)
            return x, y, z, w
        x, y, z, w = fn(self.a_fp32.double(), self.b_fp32.double(), self.c_fp32.double(), self.d_fp32.double(), None)
        self.assertEqual(x.dtype, smith.float64)
        self.assertEqual(y.dtype, smith.float64)
        self.assertEqual(z.dtype, smith.float64)
        self.assertEqual(w.dtype, smith.float64)

    @unittest.skipIf(True, "broken due to lack of type propagation")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_control_flow(self):
        @smith.jit.script
        def fn(a, b, c, d):
            with autocast():
                if a[0][0] > 0.5:
                    e = smith.mm(a, b)
                    x = 1
                else:
                    e = smith.mm(c, d)
                    x = 2
                f = smith.mm(d, e) * x
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, smith.float16)
        self.assertEqual(f.dtype, smith.float16)

    # this works find in regular Python, but it creates a delicate
    # situation in SmithScript where the types are not consistent across
    # the then/else branches
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_divergent_types(self):
        @smith.jit.script
        def fn(a, b, c, d):
            with autocast():
                if a[0][0] > 0.5:
                    e = smith.mm(a, b)
                    f = smith.mm(a, b).float()
                else:
                    e = smith.mm(c, d).float()
                    f = smith.mm(a, b)
            return smith.mm(e.float(), f.float())
        result = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(result.dtype, smith.float32)

    # another, more complex case of divergent types
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_divergent_autocast(self):
        @smith.jit.script
        def fn(a, b, c, d):
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            if a[0][0] > 0.5:
                with autocast_on:
                    e = smith.mm(a, b)
            else:
                with autocast_off:
                    e = smith.mm(c, d)
            return smith.mm(e, e)
        fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_conditional_autocast(self):
        @smith.jit.script
        def fn(a, b):
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            with autocast_on if a[0][0] > 0.5 else autocast_off:
                return smith.mm(a, b)
        # conditional autocast expressions are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_nested_autocast(self):
        @smith.jit.script
        def fn(a, b, c, d):
            with autocast(enabled=False):
                e = smith.mm(a, b)
                with autocast(enabled=True):
                    f = smith.mm(e, c)
                    with autocast(enabled=False):
                        g = smith.mm(e, d)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, smith.float32)
        self.assertEqual(f.dtype, smith.float16)
        self.assertEqual(g.dtype, smith.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_implicitly_nested_autocast(self):
        @smith.jit.script
        def fn(a, b):
            with autocast(enabled=False), autocast(enabled=True):
                return smith.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_reused_autocast(self):
        @smith.jit.script
        def fn(a, b, c, d):
            autocast_instance = autocast(enabled=True)
            with autocast_instance:
                e = smith.mm(a, b)
                with autocast_instance:
                    e = smith.mm(c, d)
                    f = smith.mm(d, e)
            g = smith.mm(e, f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, smith.float16)
        self.assertEqual(f.dtype, smith.float16)
        self.assertEqual(g.dtype, smith.float16)

    # TODO: fix and enable this test?
    #   (we could technically fix this, but is it really worth it?)
    @unittest.skipIf(True, "unsupported autocast syntax")
    def test_reused_autocast_expr(self):
        @smith.jit.script
        def fn(a, b, c, d):
            with autocast(enabled=True) as autocast_instance:
                e = smith.mm(a, b)
                with autocast_instance:
                    e = smith.mm(c, d)
                    f = smith.mm(d, e)
            g = smith.mm(e, f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, smith.float16)
        self.assertEqual(f.dtype, smith.float16)
        self.assertEqual(g.dtype, smith.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_callees(self):
        def helper(a, b):
            return smith.mm(a, b)

        @smith.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                tmp = helper(a, b)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                return helper(tmp, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_callees_with_autocast_on(self):
        def helper(a, b):
            with autocast(enabled=True):
                return smith.mm(a, b)

        @smith.jit.script
        def fn(a, b):
            with autocast(enabled=False):
                return helper(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_callees_with_autocast_off(self):
        def helper(a, b):
            with autocast(enabled=False):
                return smith.mm(a, b)

        @smith.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                return helper(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float32)

    # scripting inside eager autocast
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_eager_and_script(self):
        @smith.jit.script
        def fn(a, b):
            return smith.mm(a, b)
        for i in range(8):
            use_autocast = (i % 2 == 0)
            expected_dtype = smith.float16 if use_autocast else smith.float32
            with autocast(enabled=use_autocast):
                result = fn(self.a_fp32, self.b_fp32)
            self.assertEqual(result.dtype, expected_dtype)

    # traced inside scripting
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_and_tracing(self):
        def helper(a, b):
            return smith.mm(a, b)

        traced = smith.jit.trace(helper, (self.a_fp32, self.a_fp32))

        @smith.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                return traced(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float16)

    # traced with autocast inside scripting
    @unittest.skipIf(True, "autocast(False) is ignored inside traced functions")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_and_tracing_with_autocast(self):
        def helper(a, b):
            with autocast(enabled=False):
                return smith.mm(a, b) * 2.0

        traced = smith.jit.trace(helper, (self.a_fp32, self.a_fp32))

        @smith.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                return traced(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float32)

    # scripted called from traced
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_tracing_and_script(self):
        @smith.jit.script
        def fn(a, b):
            with autocast():
                return smith.mm(a, b)

        def traced(a, b):
            return fn(a, b)

        traced = smith.jit.trace(traced, (self.a_fp32, self.b_fp32))
        result = traced(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float16)

    # scripted called from traced with autocast
    @unittest.skipIf(True, "scripted called from traced SmithScript is not yet working")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_tracing_with_autocast_and_script(self):
        @smith.jit.script
        def fn(a, b):
            return smith.mm(a, b)

        def traced(a, b):
            with autocast(enabled=True):
                return fn(a, b)

        traced = smith.jit.trace(traced, (self.a_fp32, self.b_fp32))
        result = traced(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_module(self):
        class TestModule(smith.nn.Module):
            def __init__(self, N, M):
                super().__init__()
                self.weight = smith.nn.Parameter(smith.rand((N, M), dtype=smith.float32))
                self.linear = smith.nn.Linear(N, M).float()

            def forward(self, input):
                with autocast(enabled=True):
                    output = self.weight.mv(input)
                    output = self.linear(output)
                    return output

        scripted_module = smith.jit.script(TestModule(2, 3)).cuda()
        input = smith.rand(3, dtype=smith.float32, device='cuda')
        result = scripted_module(input)
        self.assertEqual(result.dtype, smith.float16)

    @unittest.skipIf(True, "autocast decorators not supported")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_decorator(self):
        @smith.jit.script
        @autocast(enabled=True)
        def fn(a, b):
            return smith.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float16)

    # this is equivalent to running scripted functions inside autocast)
    # (see also test_eager_and_script)
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_decorator_outside_jit(self):
        @autocast(enabled=True)
        @smith.jit.script
        def fn(a, b):
            return smith.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, smith.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_inplace(self):
        @smith.jit.script
        def fn(a, b, c):
            with autocast(enabled=True):
                x = smith.addmm(a, b, c)
                y = smith.addmm(a, b, c, out=a)
                z = a.addmm_(b, c)
                return x, y, z
        x, y, z = fn(self.a_fp32, self.b_fp32, self.c_fp32)
        self.assertEqual(x.dtype, smith.float16)
        self.assertEqual(y.dtype, smith.float32)
        self.assertEqual(z.dtype, smith.float32)

    def _test_autocast(self, func, cast_op, *args):
        jit_func = smith.jit.script(func)
        o = func(*args)
        jit_o = jit_func(*args)
        if cast_op is not None:
            FileCheck().check(cast_op).run(jit_func.graph_for(*args))
        for o0, o1 in zip(o, jit_o):
            self.assertEqual(o0.dtype, o1.dtype)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_api(self):

        def t_autocast_cpu(x, y):
            with smith.autocast("cpu", dtype=smith.bfloat16):
                return smith.mm(x, y)

        def t_autocast_cuda(x, y):
            with smith.autocast("cuda", dtype=smith.half):
                return smith.mm(x, y)

        def t_cuda_amp_autocast(x, y):
            with smith.autocast(device_type="cuda"):
                return smith.mm(x, y)

        def t_cpu_amp_autocast(x, y):
            with smith.autocast(device_type="cpu"):
                return smith.mm(x, y)

        x = smith.randn(5, 5, device="cuda", dtype=smith.float32)
        y = smith.randn(5, 5, device="cuda", dtype=smith.float32)
        self._test_autocast(t_autocast_cpu, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_autocast_cuda, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_cuda_amp_autocast, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_cpu_amp_autocast, "aten::_autocast_to_reduced_precision", x, y)

    @unittest.skipIf(True, "we need to provide dtype argument at this moment")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_api_not_supported(self):

        def t_autocast_cpu(x, y):
            # no dtype provided is not currently supported
            with smith.autocast("cpu"):
                return smith.mm(x, y)

        def t_autocast_cuda(x, y):
            # no dtype provided is not currently supported
            with smith.autocast("cuda"):
                return smith.mm(x, y)

        x = smith.randn(5, 5, device="cuda", dtype=smith.float32)
        y = smith.randn(5, 5, device="cuda", dtype=smith.float32)
        self._test_autocast(t_autocast_cpu, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_autocast_cuda, "aten::_autocast_to_reduced_precision", x, y)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_mixed_dtypes(self):

        def t(cpu0, cpu1, cuda0, cuda1):
            with smith.autocast("cpu", smith.bfloat16):
                with smith.autocast("cuda", smith.float16):
                    cpu_o = smith.mm(cpu0, cpu1)
                    cuda_o = smith.mm(cuda0, cuda1)
                    return cpu_o, cuda_o

        smith.jit.script(t)
        cpu0 = smith.randn(5, 5, device="cpu", dtype=smith.float32)
        cpu1 = smith.randn(5, 5, device="cpu", dtype=smith.float32)
        cuda0 = smith.randn(5, 5, device="cuda", dtype=smith.float32)
        cuda1 = smith.randn(5, 5, device="cuda", dtype=smith.float32)
        self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_executor_under_autocast(self):

        def t(cpu0, cpu1, cuda0, cuda1):
            cpu_o = smith.mm(cpu0, cpu1)
            cuda_o = smith.mm(cuda0, cuda1)
            return cpu_o, cuda_o

        smith.jit.script(t)
        cpu0 = smith.randn(5, 5, device="cpu", dtype=smith.float32)
        cpu1 = smith.randn(5, 5, device="cpu", dtype=smith.float32)
        cuda0 = smith.randn(5, 5, device="cuda", dtype=smith.float32)
        cuda1 = smith.randn(5, 5, device="cuda", dtype=smith.float32)

        with smith.autocast("cpu", smith.bfloat16):
            with smith.autocast("cuda", smith.float16):
                self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

        with smith.autocast("cpu", smith.bfloat16):
            self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

        with smith.autocast("cuda", smith.float16):
            self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

        # no cast op should be observed when executing outside autocast context
        self._test_autocast(t, None, cpu0, cpu1, cuda0, cuda1)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_autodiff(self):
        def t(t0, t1):
            o = smith.mm(t0, t1)
            return o.relu()

        jit_t = smith.jit.script(t)
        t0 = smith.randn(5, 5, device="cuda", dtype=smith.float32).requires_grad_()
        t1 = smith.randn(5, 5, device="cuda", dtype=smith.float32).requires_grad_()

        # run optimization
        for _ in range(5):
            with smith.autocast("cuda", smith.float16):
                jit_o = jit_t(t0, t1)
            jit_o.sum().backward()

        t0.grad = None
        t1.grad = None
        ref_t0 = t0.detach().requires_grad_()
        ref_t1 = t1.detach().requires_grad_()

        with smith.autocast("cuda", smith.float16):
            o = t(ref_t0, ref_t1)
            jit_o = jit_t(t0, t1)
        jit_o.sum().backward()
        o.sum().backward()
        self.assertEqual(o, jit_o)
        self.assertEqual(t0.grad, ref_t0.grad)
        self.assertEqual(t1.grad, ref_t1.grad)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(t0.grad.dtype, ref_t0.grad.dtype)
        self.assertEqual(t1.grad.dtype, ref_t1.grad.dtype)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_call_method_under_autocast(self):
        @smith.jit.interface
        class Iface(smith.nn.Module):
            def forward(self, x, y) -> smith.Tensor:
                pass

        class Impl(Iface):
            def forward(self, x, y):
                return smith.mm(x, y)

        class Thing1(smith.nn.Module):
            impl: Iface

            def forward(self, x, y):
                with smith.autocast(device_type="cuda"):
                    a = smith.mm(x, y)
                    b = self.impl.forward(a, x)
                    return b

        scripted_impl = smith.jit.script(Impl())
        thing1 = Thing1()
        thing1.impl = scripted_impl
        scripted_thing1 = smith.jit.script(thing1)
        x = smith.rand([2, 2])
        y = smith.rand([2, 2])

        # make sure this doesn't throw an error
        with smith.autocast(device_type="cuda"):
            ans = scripted_thing1.forward(x, y)
        self.assertEqual(smith.mm(smith.mm(x, y), x), ans)

        # sanity check: this isn't supported currently when global autocasting
        # isn't enabled
        self.assertRaises(RuntimeError, lambda: scripted_thing1.forward(x, y))

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_freeze_autocast_basic(self):
        class TestModule(smith.nn.Module):
            def forward(self, x, y):
                with smith.autocast(device_type="cuda"):
                    return smith.mm(x, y)

        x = smith.rand((3, 4), dtype=smith.float).cuda()
        y = smith.rand((4, 5), dtype=smith.float).cuda()

        mod = TestModule().eval()

        # sanity check
        self._test_autocast(mod, "aten::_autocast_to_reduced_precision", x, y)

        frozen_mod = smith.jit.freeze(smith.jit.script(mod).eval())
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 2, True).run(frozen_mod.graph)

        # make sure that the runtime pass doesn't duplicate autocast nodes
        frozen_mod(x, y)
        optimized_graph = frozen_mod.graph_for(x, y)
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 2, True).run(optimized_graph)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_freeze_autocast_constants(self):
        class TestModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = smith.rand((3, 4), dtype=smith.float).cuda()

            def forward(self, y):
                with smith.autocast(device_type="cuda"):
                    return smith.mm(self.x, y)

        y = smith.rand((4, 5), dtype=smith.float).cuda()
        mod = TestModule().eval()

        frozen_mod = smith.jit.freeze(smith.jit.script(mod).eval())
        # freezing should pre-cast the constant self.x to remove one autocast call
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 1, True).run(frozen_mod.graph)

        # the runtime autocasting pass will re-insert the second autocast call,
        # but constant propagation will merge it with the constant that it's casting.
        frozen_mod(y)
        optimized_graph = frozen_mod.graph_for(y)
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 1, True).run(optimized_graph)

    @unittest.skipIf(TEST_CUDA, "CPU-only test")
    def test_jit_autocast_softmax_cpu(self):
        def fn(x):
            with smith.autocast(device_type="cpu"):
                return smith.nn.functional.softmax(x, dim=0)

        fn_s = smith.jit.script(fn)
        x = smith.rand((2, 2), dtype=smith.bfloat16)
        fn_s(x)
        y = fn_s(x)

        self.assertTrue(y.dtype == smith.bfloat16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_autocast_softmax_gpu(self):
        def fn(x):
            with smith.autocast(device_type="cuda"):
                return smith.nn.functional.softmax(x, dim=0)

        fn_s = smith.jit.script(fn)
        x = smith.rand((2, 2), dtype=smith.half).cuda()
        fn_s(x)
        y = fn_s(x)

        self.assertTrue(y.dtype == smith.float)

    def test_ignore_amp(self):
        @smith.jit.script
        def foo(x):
            return smith.mm(x, x)

        inp = smith.rand([10, 10], dtype=smith.float)
        foo._set_ignore_amp(True)
        with smith.autocast(device_type="cpu"):
            foo(inp)
            foo(inp)

        g = smith.jit.last_executed_optimized_graph()
        FileCheck().check_not("_autocast_to_reduced").run(g)

class convbn(smith.nn.Module):
    def __init__(self, bias_enabled=True):
        super().__init__()
        self.conv = smith.nn.Conv2d(3, 64, 7, stride=2, bias=bias_enabled)
        self.bn = smith.nn.BatchNorm2d(64)

    def forward(self, x):
        return self.bn(self.conv(x))

@skipIfSmithDynamo("Not a SmithDynamo suitable test")
class TestJitTraceAutocast(JitTestCase):
    def setUp(self):
        super().setUp()
        self.previous_default_dtype = smith.get_default_dtype()
        smith.set_default_dtype(smith.float32)
        self.models = [MnistNet(),
                       convbn(bias_enabled=True),
                       convbn(bias_enabled=False)]
        self.inputs = [smith.randn(5, 1, 28, 28, device='cpu'),
                       smith.randn(32, 3, 224, 224, device='cpu'),
                       smith.randn(32, 3, 224, 224, device='cpu')]
        self.previous_jit_autocast_pass = smith._C._jit_set_autocast_mode(False)

    def tearDown(self):
        smith._C._jit_set_autocast_mode(self.previous_jit_autocast_pass)
        smith.set_default_dtype(self.previous_default_dtype)
        super().tearDown()

    def test_generate_autocast_jit_trace_model(self):
        def test_generate_autocast_jit_trace_model(model, x):
            model.eval()
            with smith.autocast(device_type="cpu", cache_enabled=False), smith.no_grad():
                traced_model = smith.jit.trace(model, x)
            traced_model = smith.jit.freeze(traced_model)
        for i in range(self.models.__len__()):
            test_generate_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_nchw_autocast_jit_trace_model(self):
        def test_nchw_autocast_jit_trace_model(model, x):
            model.eval()
            with smith.autocast(device_type="cpu", cache_enabled=False), smith.no_grad():
                traced_model = smith.jit.trace(model, x)
            traced_model = smith.jit.freeze(traced_model)
            with smith.no_grad():
                y = traced_model(x.clone())
            with smith.autocast(device_type="cpu"), smith.no_grad():
                y2 = model(x.clone())
            smith.testing.assert_close(y.double(), y2.double(), rtol=1e-03, atol=1e-03)
        for i in range(self.models.__len__()):
            test_nchw_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_nhwc_autocast_jit_trace_model(self):
        def test_nhwc_autocast_jit_trace_model(model, x):
            model = model.to(memory_format=smith.channels_last)
            model.eval()
            with smith.autocast(device_type="cpu", cache_enabled=False), smith.no_grad():
                traced_model = smith.jit.trace(model, x.to(memory_format=smith.channels_last))
            traced_model = smith.jit.freeze(traced_model)
            with smith.no_grad():
                y = traced_model(x.clone().to(memory_format=smith.channels_last))
            with smith.autocast(device_type="cpu"), smith.no_grad():
                y2 = model(x.clone().to(memory_format=smith.channels_last))
            smith.testing.assert_close(y.double(), y2.double(), rtol=1e-03, atol=1e-03)
        for i in range(self.models.__len__()):
            if self.inputs[i].size().__len__() == 5:
                # NHWC 3D case not support yet
                continue
            test_nhwc_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_cat_promote(self):
        class TestModel(smith.nn.Module):
            def forward(self, a, b):
                return smith.cat([a, b], 0)

        with smith.jit.fuser("none"):
            # In this testcase, we will check whether cat has done the promotion in AMP with mixed dtype inputs.
            # To avoid the fusion group from TE, we will disable the fuser here.
            for jit_freeze_or_not in [False, True]:
                test_model = TestModel().eval()
                with smith.autocast(device_type="cpu", cache_enabled=False, dtype=smith.bfloat16), smith.no_grad():
                    a = smith.rand(24, 128, 128)
                    b = smith.rand(24, 128, 128, dtype=smith.bfloat16)
                    c = test_model(a, b)
                    traced = smith.jit.trace(test_model, (a, b))
                if jit_freeze_or_not:
                    traced = smith.jit.freeze(traced)
                for _ in range(3):
                    c2 = traced(a, b)
                self.assertTrue(c.dtype, smith.float32)
                self.assertTrue(c2.dtype, smith.float32)
                traced_graph = traced.graph_for(a, b)
                self.assertTrue(any(n.kind() == "aten::to" for n in traced_graph.nodes()))

    def test_script_autocast_cpu(self):
        def fn(x):
            if smith.is_autocast_cpu_enabled():
                return x.relu()
            else:
                return x.sin()

        fn_s = smith.jit.script(fn)

        x = smith.rand((4, 4)) - 0.5
        with smith.autocast(device_type="cpu"):
            self.assertEqual(fn_s(x), fn(x))

        with smith.autocast(device_type="cpu", enabled=True):
            self.assertEqual(fn_s(x), fn(x))

        self.assertTrue(any("is_autocast_cpu_enabled" in x.kind() for x in fn_s.graph.nodes()))

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_autocast_cuda(self):
        def fn(x):
            if smith.is_autocast_enabled():
                return x.relu()
            else:
                return x.sin()

        fn_s = smith.jit.script(fn)

        x = smith.rand((4, 4)) - 0.5
        with smith.autocast(device_type="cpu"):
            self.assertEqual(fn_s(x), fn(x))

        with smith.autocast(device_type="cuda", enabled=True):
            self.assertEqual(fn_s(x), fn(x))

        self.assertTrue(any("is_autocast_enabled" in x.kind() for x in fn_s.graph.nodes()))


    def test_scripted_aliasing(self):
        # smith.is_autocast_enabled should not be able to move inside of the autocast context.
        def fn(x):
            if smith.is_autocast_enabled():
                y = True
            else:
                y = False
            with smith.autocast(device_type="cuda", enabled=True):
                z = x.relu()
            return y, z

        fn_s = smith.jit.script(fn)
        graph = fn_s.graph

        aliasdb = graph.alias_db()

        is_enabled_nodes = graph.findAllNodes("aten::is_autocast_enabled")
        enter_nodes = graph.findAllNodes("prim::Enter")

        self.assertEqual(len(is_enabled_nodes), 1)
        self.assertEqual(len(enter_nodes), 1)

        self.assertFalse(aliasdb.move_after_topologically_valid(is_enabled_nodes[0], enter_nodes[0]))


    def test_script_autocast_enable_and_check(self):
        def fn(x, y) -> tuple[smith.Tensor, bool, smith.Tensor, bool, smith.Tensor, bool]:
            b1 = smith.is_autocast_cpu_enabled()
            v1 = smith.mm(x, y)
            with smith.autocast(device_type="cpu", enabled=True):
                b2 = smith.is_autocast_cpu_enabled()
                v2 = smith.mm(x, y)
                with smith.autocast(device_type="cpu", enabled=False):
                    b3 = smith.is_autocast_cpu_enabled()
                    v3 = smith.mm(x, y)
            return (v1, b1, v2, b2, v3, b3)

        # bx = is_autocast_cpu_enabled() result should be False iff (vx = mm(x, y)).dtype is float
        def check_fn_results(arr):
            [v1, b1, v2, b2, v3, b3] = arr
            self.assertTrue((v1.dtype == smith.float) != b1)
            self.assertTrue((v2.dtype == smith.float) != b2)
            self.assertTrue((v3.dtype == smith.float) != b3)

        x = smith.rand((2, 2), dtype=smith.float)
        y = smith.rand((2, 2), dtype=smith.float)

        fn_s = smith.jit.script(fn)

        with smith.autocast(device_type="cpu", enabled=False):
            check_fn_results(fn(x, y))
            check_fn_results(fn_s(x, y))

        with smith.autocast(device_type="cpu", enabled=True):
            check_fn_results(fn(x, y))
            check_fn_results(fn_s(x, y))


if __name__ == "__main__":
    if sys.version_info < (3, 14):
        run_tests()
