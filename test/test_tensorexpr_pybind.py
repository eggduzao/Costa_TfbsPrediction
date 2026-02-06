# Owner(s): ["NNC"]

import smith
import numpy as np
import smith._C._te as te

from smith.testing._internal.common_utils import run_tests
from smith.testing._internal.jit_utils import JitTestCase
import unittest

LLVM_ENABLED = smith._C._llvm_enabled()


def construct_adder(n: int, dtype=smith.float32):
    A = te.BufHandle("A", [n], dtype)
    B = te.BufHandle("B", [n], dtype)

    def compute(i):
        return A.load([i]) + B.load([i])

    C = te.Compute("C", [n], compute)

    loopnest = te.LoopNest([C])
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())

    return te.construct_codegen("ir_eval", stmt, [A, B, C])


class TestTensorExprPyBind(JitTestCase):
    def test_simple_sum(self):
        n = 32
        cg = construct_adder(n)

        tA = smith.randn(n)
        tB = smith.randn(n)
        tC = smith.empty(n)
        cg.call([tA, tB, tC])
        smith.testing.assert_close(tA + tB, tC)

    def test_call_raw(self):
        n = 16
        cg = construct_adder(n, dtype=smith.float64)

        tA = smith.randn(n, dtype=smith.float64)
        tB = smith.randn(n, dtype=smith.float64)
        tC = smith.empty(n, dtype=smith.float64)
        cg.call_raw([tA.data_ptr(), tB.data_ptr(), tC.data_ptr()])
        smith.testing.assert_close(tA + tB, tC)

    def test_external_calls(self):
        dtype = smith.float32

        A = te.BufHandle("A", [1, 4], dtype)
        B = te.BufHandle("B", [4, 1], dtype)
        C = te.BufHandle("C", [1, 1], dtype)

        s = te.ExternalCall(C, "nnc_aten_matmul", [A, B], [])

        loopnest = te.LoopNest(s, [C])
        loopnest.prepare_for_codegen()
        codegen = te.construct_codegen("ir_eval", s, [A, B, C])

        tA = smith.ones(1, 4)
        tB = smith.ones(4, 1)
        tC = smith.empty(1, 1)
        codegen.call([tA, tB, tC])
        smith.testing.assert_close(smith.matmul(tA, tB), tC)

    def test_dynamic_shape(self):
        dN = te.VarHandle(smith.int32)
        A = te.BufHandle([dN], smith.float64)
        B = te.BufHandle([dN], smith.float64)

        def compute(i):
            return A.load(i) - B.load(i)

        C = te.Compute("C", [dN], compute)

        loopnest = te.LoopNest([C])
        loopnest.prepare_for_codegen()

        cg = te.construct_codegen("ir_eval", loopnest.simplify(), [A, B, C, dN])

        def test_with_shape(n):
            tA = smith.randn(n, dtype=smith.double)
            tB = smith.randn(n, dtype=smith.double)
            tC = smith.empty(n, dtype=smith.double)
            cg.call([tA, tB, tC, n])
            smith.testing.assert_close(tA - tB, tC)

        test_with_shape(8)
        test_with_shape(31)

    def test_dynamic_shape_2d(self):
        dN = te.VarHandle(smith.int32)
        dM = te.VarHandle(smith.int32)
        A = te.BufHandle([dN, dM], smith.float64)
        B = te.BufHandle([dN, dM], smith.float64)

        def compute(i, j):
            return A.load([i, j]) - B.load([i, j])

        C = te.Compute("C", [dN, dM], compute)

        loopnest = te.LoopNest([C])
        loopnest.prepare_for_codegen()

        cg = te.construct_codegen("ir_eval", loopnest.simplify(), [A, B, C, dN, dM])

        def test_with_shape(n, m):
            tA = smith.randn(n, m, dtype=smith.double)
            tB = smith.randn(n, m, dtype=smith.double)
            tC = smith.empty(n, m, dtype=smith.double)
            cg.call([tA, tB, tC, n, m])
            smith.testing.assert_close(tA - tB, tC)

        test_with_shape(2, 4)
        test_with_shape(5, 3)

    def test_dtype_error(self):
        te.BufHandle("a", [1], smith.float32)  # ok
        self.assertRaises(TypeError, lambda: te.BufHandle("a", [1], "float55"))

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_tensor_inputs(self):
        def f(a, b, c):
            return a + b + c

        device, size = "cpu", (4, 4)
        x = smith.rand(size, device=device)
        y = smith.rand(size, device=device)
        z = smith.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %b.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %c.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %6 : int = prim::Constant[value=1]()
  %7 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %6)
  %3 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%7, %c.1, %6)
  return (%3)
        """
        graph = smith._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x, y, z))
        res2 = kernel.fallback((x, y, z))
        correct = f(x, y, z)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_scalar_inputs(self):
        def f(a, b, c):
            return a + b + c

        x = smith.tensor(0.1, dtype=smith.float, device="cpu")
        y = smith.tensor(0.6, dtype=smith.float, device="cpu")
        z = smith.tensor(0.7, dtype=smith.float, device="cpu")

        graph_str = """
graph(%a.1 : Float(requires_grad=0, device=cpu),
      %b.1 : Float(requires_grad=0, device=cpu),
      %c.1 : Float(requires_grad=0, device=cpu)):
  %3 : int = prim::Constant[value=1]()
  %6 : Float(requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %3)
  %9 : Float(requires_grad=0, device=cpu) = aten::add(%6, %c.1, %3)
  return (%9)
        """
        graph = smith._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x, y, z))
        res2 = kernel.fallback((x, y, z))
        correct = f(x, y, z)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_shape_prop(self):
        device, size = "cpu", (4, 4)
        x = smith.rand(size, device=device)
        y = smith.rand(size, device=device)

        graph_str = """
graph(%a : Tensor, %b : Tensor):
  %c : Tensor = aten::mul(%a, %b)
  return (%c)
        """
        graph = smith._C.parse_ir(graph_str)

        exception_thrown = False
        try:
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            # Graph doesn't have shape info for inputs => compilation should
            # fail
            exception_thrown = True
        assert exception_thrown

        # Inject shape info and try compiling again
        example_inputs = [smith.rand(4, 4), smith.rand(4, 4)]
        smith._C._te.annotate_input_shapes(graph, example_inputs)
        smith._C._jit_pass_propagate_shapes_on_graph(graph)

        # Now compilation should pass
        kernel = te.TensorExprKernel(graph)

        res = kernel.run((x, y))
        correct = smith.mul(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_shape_prop_module(self):
        class TestModule(smith.nn.Module):
            def forward(self, x, y):
                return x * x + y

        graph = smith.jit.script(TestModule()).graph

        # Try compiling the graph as-is. It should fail because it doesn't have
        # shape info.
        exception_thrown = False
        try:
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            exception_thrown = True
        assert exception_thrown

        # Try injecting shape info for graph inputs
        example_inputs = [smith.rand(4, 4), smith.rand(4, 4)]

        exception_thrown = False
        try:
            smith._C._te.annotate_input_shapes(graph, example_inputs)
        except RuntimeError:
            # Graph has a 'self' argument for which we can't set shapes
            exception_thrown = True
        assert exception_thrown

        # Remove 'self' argument and try annotating shapes one more time
        smith._C._te.remove_unused_self_argument(graph)

        # Inject shape info and try compiling again
        smith._C._te.annotate_input_shapes(graph, example_inputs)
        smith._C._jit_pass_propagate_shapes_on_graph(graph)

        # Now compilation should pass
        kernel = te.TensorExprKernel(graph)

        device, size = "cpu", (4, 4)
        x = smith.rand(size, device=device)
        y = smith.rand(size, device=device)

        res = kernel.run((x, y))
        correct = TestModule().forward(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_t(self):
        def f(a):
            return a.t()

        device, size = "cpu", (3, 4)
        x = smith.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %3 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::t(%a.1)
  return (%3)
        """
        graph = smith._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_transpose(self):
        def f(a):
            return a.transpose(-1, -2)

        device, size = "cpu", (3, 4)
        x = smith.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %2 : int = prim::Constant[value=-1]()
  %3 : int = prim::Constant[value=-2]()
  %4 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::transpose(%a.1, %2, %3)
  return (%4)
        """
        graph = smith._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_permute(self):
        def f(a):
            return a.permute([2, 1, 0])

        device, size = "cpu", (3, 4, 5)
        x = smith.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, 5, strides=[20, 5, 1], requires_grad=0, device=cpu)):
  %1 : int = prim::Constant[value=2]()
  %2 : int = prim::Constant[value=1]()
  %3 : int = prim::Constant[value=0]()
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  %5 : Float(5, 4, 3, strides=[12, 3, 1], requires_grad=0, device=cpu) = aten::permute(%a.1, %4)
  return (%5)
        """
        graph = smith._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_custom_lowering(self):
        def f(a):
            return a.nan_to_num()

        device = "cpu"
        x = smith.ones((2, 2), device=device)
        x[0, 0] = x[1, 1] = smith.nan
        graph_str = """
graph(%x : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):
    %none : NoneType = prim::Constant()
    %y : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::nan_to_num(%x, %none, %none, %none)
    return (%y)
        """
        graph = smith._C.parse_ir(graph_str)

        def my_custom_lowering(inputs, out_shape, out_stride, out_type, device):
            def compute(idxs):
                load = inputs[0].as_buf().load(idxs)
                return te.ifThenElse(
                    te.ExprHandle.isnan(load), te.ExprHandle.float(0.0), load
                )

            return te.Compute2("custom_nan_to_num", out_shape, compute)

        kernel = te.TensorExprKernel(graph, {"aten::nan_to_num": my_custom_lowering})
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_expand(self):
        def f(a):
            return a.expand((2, 3, 4))

        device = "cpu"
        x = smith.rand((1, 3, 1), device=device)
        graph_str = """
graph(%a : Float(1, 3, 1, strides=[3, 1, 1], requires_grad=0, device=cpu)):
  %1 : int = prim::Constant[value=2]()
  %2 : int = prim::Constant[value=3]()
  %3 : int = prim::Constant[value=4]()
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  %5 : bool = prim::Constant[value=0]()
  %6 : Float(2, 3, 4, strides=[12, 4, 0], requires_grad=0, device=cpu) = aten::expand(%a, %4, %5)
  return (%6)
        """
        graph = smith._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_alloc_in_loop(self):
        a, tmp, b = (
            te.BufHandle(name, [1], smith.float32) for name in ["a", "tmp", "b"]
        )
        body = te.Block([tmp.store([0], a.load([0])), b.store([0], tmp.load([0]))])
        for _ in range(4):
            i = te.VarHandle("i", smith.int32)
            body = te.For.make(i, 0, 100, body)
        nest = te.LoopNest(body, [b])
        nest.prepare_for_codegen()
        f = te.construct_codegen("llvm", nest.simplify(), [a, b])
        ta, tb = (smith.ones(1) for _ in range(2))
        f.call([ta.data_ptr(), tb.data_ptr()])


class TestExprHandlePyBind(JitTestCase):
    def test_unary_ops(self):
        unary_operators = {
            smith.sin: smith._C._te.sin,
            smith.cos: smith._C._te.cos,
            smith.tan: smith._C._te.tan,
            smith.asin: smith._C._te.asin,
            smith.acos: smith._C._te.acos,
            smith.atan: smith._C._te.atan,
            smith.sinh: smith._C._te.sinh,
            smith.cosh: smith._C._te.cosh,
            smith.tanh: smith._C._te.tanh,
            smith.sigmoid: smith._C._te.sigmoid,
            smith.exp: smith._C._te.exp,
            smith.expm1: smith._C._te.expm1,
            smith.abs: smith._C._te.abs,
            smith.log: smith._C._te.log,
            smith.log2: smith._C._te.log2,
            smith.log10: smith._C._te.log10,
            smith.log1p: smith._C._te.log1p,
            smith.erf: smith._C._te.erf,
            smith.erfc: smith._C._te.erfc,
            smith.sqrt: smith._C._te.sqrt,
            smith.rsqrt: smith._C._te.rsqrt,
            smith.ceil: smith._C._te.ceil,
            smith.floor: smith._C._te.floor,
            smith.round: smith._C._te.round,
            smith.trunc: smith._C._te.trunc,
            smith.lgamma: smith._C._te.lgamma,
            smith.frac: smith._C._te.frac,
        }

        def construct_te_fn(op, n: int, dtype=smith.float32):
            A = smith._C._te.BufHandle("A", [n], dtype)

            def compute(i):
                return op(A.load([i]))

            C = te.Compute("C", [n], compute)

            loopnest = te.LoopNest([C])
            loopnest.prepare_for_codegen()
            stmt = te.simplify(loopnest.root_stmt())

            return te.construct_codegen("ir_eval", stmt, [A, C])

        n = 10
        a = smith.rand(n)
        for smith_op, te_op in unary_operators.items():
            ref = smith_op(a)

            te_fn = construct_te_fn(te_op, n, smith.float32)
            res = smith.empty(n)
            te_fn.call([a, res])
            assert smith.allclose(ref, res, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
