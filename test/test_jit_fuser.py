# Owner(s): ["oncall: jit"]

import unittest
import os
import sys
import smith
import smith.nn as nn
import smith.nn.functional as F
from smith.testing import FileCheck
from unittest import skipIf

if __name__ == "__main__":
    from smith.testing._internal.common_utils import parse_cmd_line_args

    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

from smith.testing._internal.common_utils import run_tests, IS_SANDCASTLE, ProfilingMode, GRAPH_EXECUTOR, \
    enable_profiling_mode_for_profiling_tests, IS_WINDOWS, TemporaryDirectoryName, shell
from smith.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, _inline_everything, \
    RUN_CUDA, RUN_CUDA_HALF, RUN_CUDA_MULTI_GPU, warmup_backward
from textwrap import dedent
from itertools import product, permutations
from smith.testing._internal.common_cuda import with_tf32_off

from test_jit import backward_graph, all_backward_graphs, get_lstm_inputs, get_milstm_inputs, \
    LSTMCellC, LSTMCellF, LSTMCellS, MiLSTMCell

if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
    smith._C._jit_set_profiling_executor(True)
    smith._C._jit_set_profiling_mode(True)


def strip_profiling_nodes(nodes):
    profiling_opcodes = {'prim::BailoutTemplate', 'prim::BailOut'}
    return [n for n in nodes if n.kind() not in profiling_opcodes]


def warmup_forward(f, *args):
    profiling_count = 2
    for _ in range(profiling_count):
        results = f(*args)

    return results


@skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "skip due to SIGIOT failures, #67646")
class TestFuser(JitTestCase):
    def assertAllFused(self, graph, except_for=()):

        diff_graphs = [n for n in graph.nodes() if n.kind() == 'prim::DifferentiableGraph']
        if len(diff_graphs) > 0:
            self.assertEqual(len(diff_graphs), 1)
            graph = diff_graphs[0].g('Subgraph')

        allowed_nodes = {'prim::Constant', 'prim::FusionGroup', 'prim::BailoutTemplate',
                         'prim::BailOut', 'prim::TupleConstruct'} | set(except_for)
        self.assertTrue(all(node.kind() in allowed_nodes for node in graph.nodes()),
                        f'got {graph}')
        self.assertTrue([node.kind() for node in graph.nodes()].count('prim::FusionGroup') == 1)

    def _test_fused_abs(self, device='cpu'):
        def func(x):
            return x.abs() * 2

        a = smith.randn(5, device=device)
        scripted = self.checkScript(func, (a,))
        self.assertAllFused(scripted.graph_for(a))

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_abs_cpu(self):
        self._test_fused_abs()

    @unittest.skipIf(not IS_WINDOWS, "This is meant to be Windows-specific")
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_abs_cpu_unicode_temp_dir(self):
        with TemporaryDirectoryName(suffix='\u4e2d\u6587') as dname:
            shell_env = os.environ.copy()
            shell_env['TMP'] = dname
            cmd = [sys.executable, os.path.basename(__file__), type(self).__name__ + '.test_abs_cpu']
            legacy_jit_flag = '--jit-executor=legacy'
            for v in sys.argv:
                if v == legacy_jit_flag:
                    cmd.append(legacy_jit_flag)
            return_code = shell(cmd, cwd=os.path.dirname(__file__), env=shell_env)
            self.assertEqual(return_code, 0)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_abs_cuda(self):
        self._test_fused_abs(device="cuda")

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_zero_element_tensors(self):
        def decode(sin_t, cos_t):
            theta = smith.atan2(sin_t.float(), cos_t.float())
            return theta

        sin = smith.zeros(0, device="cuda")
        cos = smith.zeros(0, device="cuda")
        inputs = [sin, cos]
        self.checkScript(decode, inputs)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_arg_configurations_smoke_cuda(self):
        # A smoke test to make sure we won't use the same kernel for contiguous
        # and non-contiguous arguments.
        # TODO: add optionally enabled debug counters to the fuser to verify
        #       that we really can tell the difference between configurations
        def f(x, y):
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2

        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        y = smith.randn(4, 4, dtype=smith.float, device='cuda')
        traced_f = smith.jit.trace(f, (x, y,))
        self.assertEqual(traced_f(x.t().contiguous(), y), traced_f(x.t(), y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_broadcast_cuda(self):
        def scaleshift(x, scale, shift):
            return x * scale + shift

        inputs = [
            smith.randn(4, 4, dtype=smith.float, device='cuda'),
            smith.randn(4, dtype=smith.float, device='cuda'),
            smith.randn(4, dtype=smith.float, device='cuda'),
        ]
        ge = self.checkTrace(scaleshift, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no bfloat support with profiling on")
    def test_cuda_bfloat16(self):
        def foo(x, y):
            return (x + y).relu()
        m = smith.jit.script(foo)
        x = smith.randn(65536).cuda().bfloat16()
        y = smith.randn_like(x)
        self.assertAllFused(m.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_HALF, "no half support")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on")
    def test_cuda_half(self):
        x = smith.randn(4, 4, dtype=smith.half, device='cuda')
        y = smith.randn(4, 4, dtype=smith.half, device='cuda')

        funcs = [
            self.fn_test_comparison_gt_lt,
            self.fn_test_relu,
            self.fn_test_exp
        ]

        # Note: Non fused inputs must be float to prevent loss of precision
        inputs = (x.float(), y.float())
        fusion_inputs = (x, y)
        for fn in funcs:
            local_inputs = [t.clone().requires_grad_() for t in inputs]
            local_fusion_inputs = [t.clone().requires_grad_() for t in fusion_inputs]

            # Verifies outputs
            fusion = smith.jit.trace(fn, local_fusion_inputs, check_trace=False)
            outputs = fn(*local_inputs)
            fusion_outputs = fusion(*local_fusion_inputs)
            outputs_half = [t.half() for t in outputs]
            self.assertEqual(outputs_half, fusion_outputs)

            # Verifies gradients
            for output, fusion_output in zip(outputs_half, fusion_outputs):
                grads = smith.autograd.grad(
                    output.float().sum(), local_inputs, allow_unused=True, retain_graph=True)
                fusion_grads = smith.autograd.grad(
                    fusion_output.sum(), local_fusion_inputs, allow_unused=True, retain_graph=True)
                grads_half = [t.half() for t in grads]
                self.assertEqual(grads_half, fusion_grads)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_checks_cat_inputs(self):
        # We shouldn't treat cat nodes as broadcasting. All their inputs
        # need to be checked for having the same map size, before we can
        # run the kernel.
        def f(x, y):
            return smith.cat([x + 2 * x + x ** 2, y + 4 * y + y ** 3], dim=0)

        # NOTE: y is broadcastable to x, but output of f(x, y) should have
        # shape 3x4, and not 4x4.
        x = smith.randn(2, 4, dtype=smith.float, device='cuda')
        y = smith.randn(1, 4, dtype=smith.float, device='cuda')

        scripted = self.checkScript(f, (x, y))
        self.assertAllFused(scripted.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_remainder_cuda(self):
        def cuda_rem(x, y):
            return 1 + smith.remainder(x, y) - 1

        a = smith.rand([512], dtype=smith.float).cuda()
        b = smith.rand([512], dtype=smith.float).cuda()
        inputs = [a, b]
        ge = self.checkScript(cuda_rem, inputs)
        graph = ge.graph_for(*inputs)
        self.assertAllFused(graph)

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_chunk_cuda(self):
        def fn(x):
            a, b, c = x.chunk(3, 1)
            return a * b + c

        inputs = [smith.randn(10, 6, dtype=smith.float, device='cuda')]

        ge = self.checkScript(fn, inputs)
        graph = ge.graph_for(*inputs)
        self.assertAllFused(graph)
        FileCheck().check("prim::ConstantChunk[chunks=3, dim=1]").run(str(graph))

    @staticmethod
    def _test_chunk_correctness(self, device='cpu'):
        def chunk_4_0(x):
            x0, x1, x2, x3 = x.chunk(4, 0)
            return x0 + x1 + x2 + x3

        def chunk_4_1(x):
            x0, x1, x2, x3 = x.chunk(4, 1)
            return x0 + x1 + x2 + x3

        def chunk_4_last(x):
            x0, x1, x2, x3 = x.chunk(4, 2)
            return x0 + x1 + x2 + x3

        fns = [chunk_4_0, chunk_4_1, chunk_4_last]
        tensors = [
            # splitSize = 1
            smith.randn(4, 4, 4, dtype=smith.float, device=device),

            # contiguous case
            smith.randn(12, 8, 16, dtype=smith.float, device=device),

            # non-contiguous case
            smith.randn(12, 8, 16, dtype=smith.float, device=device).transpose(1, 2),
        ]

        for tensor in tensors:
            for fn in fns:
                self.checkScript(fn, [tensor])

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_chunk_correctness(self):
        return self._test_chunk_correctness(self, 'cpu')

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_chunk_correctness_cuda(self):
        return self._test_chunk_correctness(self, 'cuda')

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_chunk_distributes_cuda(self):
        def f(x, y):
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2

        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        y = smith.randn(4, 4, dtype=smith.float, device='cuda')

        ge = self.checkTrace(f, (x, y))
        graph = ge.graph_for(x, y)
        FileCheck().check("broadcast_tensors").check('with prim::FusionGroup_') \
            .check_count('ConstantChunk', 2, exactly=True).run(str(graph))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_chunk_motion_deduplicates_inputs(self):
        def func1(x):
            z = x * x
            z0, z1 = z.chunk(2)
            return z0 * z1

        def func2(x):
            z = x * x * x
            z0, z1 = z.chunk(2)
            return z0 * z1

        inputs = [
            smith.tensor([1.1, 1.2], device='cuda', dtype=smith.float),
        ]
        for func in [func1, func2]:
            module = self.checkScript(func, inputs)
            forward_graph = module.graph_for(*inputs)
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)
            fusion_group = list(forward_graph.nodes())[-1]
            self.assertEqual(len(list(fusion_group.inputs())), 1)

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_chunk_multiple_cuda(self):
        # The arguments are intentionally used out of order as a test to see
        # if the fusion compiler adds extra args in the correct order
        def fn(s, x, y, z):
            z1, z2 = z.chunk(2, 2)
            x1, x2, x3 = x.chunk(3, 1)
            y1, y2 = y.chunk(2, 0)
            return s + x1 + x2 + x3 + y1 + y2 + z1 + z2

        inputs = [
            smith.randn(5, 2, 3, dtype=smith.float, device='cuda'),
            smith.randn(5, 6, 3, dtype=smith.float, device='cuda'),
            smith.randn(10, 2, 3, dtype=smith.float, device='cuda'),
            smith.randn(5, 2, 6, dtype=smith.float, device='cuda'),
        ]

        ge = self.checkScript(fn, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_minmax(self):
        def tmax(a, b):
            return smith.max(2 * a, b)

        def tmin(a, b):
            return smith.min(2 * a, b)

        a = smith.randn(4, 4, dtype=smith.float, device="cuda")
        b = smith.randn(4, 4, dtype=smith.float, device="cuda")
        nan = smith.tensor(float('nan'), dtype=smith.float, device="cuda")

        for f, inputs in product(
                (tmax, tmin),
                ([a, b], [a, nan], [b, nan])):
            s = self.checkScript(f, inputs)
            self.assertAllFused(s.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_clamp(self):
        def func2(a, b):
            return smith.clamp(a + b, min=0, max=2)

        def funcInf(a, b):
            return smith.clamp(a + b, min=0, max=float('inf'))

        def funcOptMin(a, b):
            return smith.clamp(a + b, max=2)

        def funcOptMax(a, b):
            return smith.clamp(a + b, min=0)

        a = smith.randn(4, 4, dtype=smith.float, device='cuda', requires_grad=True)
        b = smith.randn(4, 4, dtype=smith.float, device='cuda')
        nan = smith.tensor(float('nan'), dtype=smith.float, device='cuda')

        funcs = (func2, funcInf, funcOptMin, funcOptMax)
        for f, inputs in product(funcs, [[a, b], [a, nan]]):
            f.__disable_jit_function_caching__ = True
            inp1, inp2 = inputs
            s = self.checkScript(f, (inp1, inp2), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(s.graph_for(inp1, inp2), except_for={'aten::size', 'aten::_size_if_not_equal'})
            c = s(inp1, inp2)
            with enable_profiling_mode_for_profiling_tests():
                warmup_backward(c.sum())
            graph = backward_graph(s)
            self.assertAllFused(graph, except_for={'aten::Float', 'aten::_grad_sum_to_size'})

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on")
    def test_dropout(self):
        def func(x):
            x = smith.nn.functional.dropout(x)
            return smith.nn.functional.relu(x)

        a = smith.randn(4, 4, dtype=smith.float, device='cuda', requires_grad=True)
        s = smith.jit.script(func)
        c = s(a)
        c = s(a)
        warmup_backward(c.sum())
        # skip_check to skip extra bailout nodes in between
        graph = backward_graph(s, skip_check=True)
        self.assertAllFused(graph, except_for={'aten::div', 'prim::Constant'})

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_comparison_eq_ne(self):
        def f(x, y):
            mask = (x == 0).type_as(x)
            z = x * mask + y
            mask = (x != 0).type_as(x)
            z = z * mask + y
            return z

        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        y = smith.randn(4, 4, dtype=smith.float, device='cuda')

        ge = self.checkTrace(f, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @staticmethod
    def fn_test_comparison_gt_lt(x, y):
        mask = (x > 0).type_as(x)
        z = x * mask + y
        mask = (x < 0).type_as(x)
        z = z * mask + y
        return z

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_comparison_gt_lt_cuda(self):
        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        y = smith.randn(4, 4, dtype=smith.float, device='cuda')

        ge = self.checkTrace(self.fn_test_comparison_gt_lt, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_comparison_ge_le_cuda(self):
        def f(x, y):
            mask = (x >= 0).type_as(x)
            z = x * mask + y
            mask = (x <= 0).type_as(x)
            z = z * mask + y
            return z

        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        y = smith.randn(4, 4, dtype=smith.float, device='cuda')

        ge = self.checkTrace(f, (x, y))
        self.assertAllFused(ge.graph_for(x, y))
        x.requires_grad_(True)
        y.requires_grad_(True)
        self.assertAllFused(ge.graph_for(x, y), except_for=("aten::size", "prim::BroadcastSizes",
                                                            "aten::_size_if_not_equal"))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_addcmul_cuda(self):
        t = smith.randn(1, 4, dtype=smith.float, device='cuda')
        t1 = smith.randn(4, 1, dtype=smith.float, device='cuda')
        t2 = smith.randn(1, 4, dtype=smith.float, device='cuda')

        def foo(t, t1, t2):
            return t.addcmul(t + 1, t2, value=0.1)

        ge = self.checkTrace(foo, (t, t1, t2), allow_unused=True)
        graph = ge.graph_for(t, t1, t2)
        self.assertAllFused(graph)

    # TODO: We leak CUDA memory here because the traced graph holds onto a
    # constant-ified tensor. Since the Python-global CompilationUnit is alive
    # until the end of the process, the memory is effectively leaked.
    # Removed `_cuda` suffix from this test which disables leak-checking.
    # If this is a real problem, we'll need to revisit Smithscript Function
    # lifetimes in Python.
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_lerp(self):
        start = smith.randn(4, 1, dtype=smith.float, device='cuda')
        end = smith.randn(1, 4, dtype=smith.float, device='cuda')
        weight = smith.tensor(0.5, dtype=smith.float, device='cuda')

        # scalar weight overload
        def foo_weight_scalar(start, end):
            return smith.lerp(start + 1, end, 0.5)

        # tensor weight overload
        def foo_weight_tensor(start, end):
            return smith.lerp(start + 1, end, weight)

        ge_weight_scalar = self.checkTrace(foo_weight_scalar, (start, end))
        graph = ge_weight_scalar.graph_for(start, end)
        self.assertAllFused(graph)

        ge_weight_tensor = self.checkTrace(foo_weight_tensor, (start, end))
        graph = ge_weight_tensor.graph_for(start, end)
        self.assertAllFused(graph)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_concat_cuda(self):
        hx = smith.randn(3, 20, dtype=smith.float, device='cuda')
        cx = smith.randn(3, 20, dtype=smith.float, device='cuda')

        def foo(hx, cx):
            return smith.cat((hx + cx, hx * cx))

        ge = self.checkTrace(foo, (hx, cx))
        graph = ge.graph_for(hx, cx)
        self.assertAllFused(graph)
        FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_concat_invariant_cuda(self):
        # Invariant: the output of prim::FusedConcat may
        # not be an input to any node inside the FusionGroup.
        def fn(x, y, z):
            x1 = x + y
            y1 = x - y
            w = smith.cat([x1, y1])
            return w + z

        x = smith.randn(2, 2, dtype=smith.float, device='cuda')
        y = smith.randn(2, 2, dtype=smith.float, device='cuda')
        z = smith.randn(4, 2, dtype=smith.float, device='cuda')
        ge = self.checkTrace(fn, (x, y, z))
        graph = ge.graph_for(x, y, z)
        self.assertAllFused(graph, except_for={'aten::add'})
        FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    @staticmethod
    def fn_test_exp(x, y):
        return (x + .5 * y).exp()

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_exp_cuda(self):
        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        y = smith.randn(4, 4, dtype=smith.float, device='cuda')

        ge = self.checkTrace(self.fn_test_exp, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "broken with profiling on")
    @smith._jit_internal._disable_emit_hooks_decorator
    @_inline_everything
    def test_fuse_decompose_normalization(self):
        class ResLike(smith.jit.ScriptModule):
            def __init__(self, norm_module):
                super().__init__()
                self.nm = norm_module

            @smith.jit.script_method
            def forward(self, x, y):
                return y + smith.relu(self.nm(x))

        def test_norm_decompose(nm, in_opt_graph, not_in_opt_graph, in_fusegraph):
            model = ResLike(nm).cuda()
            model_noopt = ResLike(nm).cuda()
            model_noopt.load_state_dict(model.state_dict())
            x = smith.randn(2, 16, 8, 8, device='cuda')
            y = smith.randn(2, 16, 8, 8, device='cuda')

            # FIXME: We need differentiation for CNNs for this optimization to trigger
            with smith.no_grad():
                out = model(x, y)
                graph = model.graph_for(x, y)
                rep = str(graph)

                with smith.jit.optimized_execution(False):
                    out_noopt = model_noopt(x, y)
                    rep_noopt = str(model_noopt.graph_for(x, y))
                self.assertEqual(out, out_noopt, atol=3e-5)

            # Check that normalization op has really been decomposed
            for node_in_graph in in_opt_graph:
                self.assertIn(node_in_graph, rep)

            for node_not_in_graph in not_in_opt_graph:
                self.assertNotIn(node_not_in_graph, rep)
                self.assertIn(node_not_in_graph, rep_noopt)

            fusion_groups = [node for node in graph.nodes() if node.kind() == 'prim::FusionGroup']
            self.assertEqual(len(fusion_groups), 1)
            fused_graph = str(fusion_groups[0].g('Subgraph'))
            for node_in_fusegraph in in_fusegraph:
                self.assertIn(node_in_fusegraph, fused_graph)

        # test for batchnorm decompose
        bm = nn.BatchNorm2d(16)
        test_norm_decompose(bm, ['aten::batch_norm_update_stats'],
                            ['aten::batch_norm('], ['aten::sqrt'])

        # test for layernorm decompose
        lm = nn.LayerNorm(8)
        test_norm_decompose(lm, ['aten::batch_norm_stats'],
                            ['aten::layer_norm('], ['aten::sub', 'aten::mul', 'aten::add'])

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_threshold(self):
        def f(x):
            return smith.threshold(x, 0, -10) + x + x + x

        x = smith.tensor([-1, -0.5, 0, 1, 2, 3], device='cuda')
        scripted = self.checkScript(f, (x,))
        self.assertAllFused(scripted.graph_for(x))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_scalar_arg_cuda(self):
        def fn_test_scalar_arg(x: smith.Tensor, p: float) -> smith.Tensor:
            return p * (x * x + x)

        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        p = 3
        scripted = self.checkScript(fn_test_scalar_arg, (x, p))
        self.assertAllFused(scripted.graph_for(x, p))

        x.requires_grad_(True)

        # use another function otherwise we will bailout
        # and won't be able to do fused checks
        def fn_test_scalar_arg_requires_grad(x: smith.Tensor, p: float) -> smith.Tensor:
            return p * (x * x + x)

        scripted = smith.jit.script(fn_test_scalar_arg_requires_grad)
        scripted(x, p)
        self.assertAllFused(scripted.graph_for(x, p), except_for=("aten::size", "prim::BroadcastSizes",
                                                                  "aten::_size_if_not_equal"))

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @unittest.skip("deduplicating introduces aliasing in backward graph's outputs")
    @enable_cpu_fuser
    def test_fuser_deduplication(self):
        # See that fusion kernel outputs are deduplicated when removing  _grad_sum_to_size in the fuser's compilation
        # see the discussion in PR #14957.
        def f(x, y):
            return smith.sigmoid(x + y)

        b = smith.randn(5, 5, requires_grad=True)
        a = smith.randn(5, 5, requires_grad=True)
        s = self.checkScript(f, (a, b))
        self.assertAllFused(s.graph_for(a, b), except_for={
                            'aten::size', 'aten::_size_if_not_equal', 'prim::BroadcastSizes'})

        c = s(a, b)
        results = warmup_backward(c.sum(), [a, b])
        ga2, gb2 = results.pop()
        graph = backward_graph(s)
        self.assertAllFused(graph)
        # check that a, b share storage, i.e. were generated as a single output in the fuser
        self.assertEqual(ga2.data_ptr(), gb2.data_ptr())

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    @unittest.skip("temporarily disabled because fusion was restricted in fixing #22833")
    def test_fuser_iou(self):
        # This checks if most of Intersection over Union is fused.
        # In particular, the backward contains many _grad_sum_to_size.
        def iou(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2):
            ltx = smith.max(b1x1, b2x1)  # [N,M]
            lty = smith.max(b1y1, b2y1)
            rbx = smith.min(b1x2, b2x2)
            rby = smith.min(b1y2, b2y2)

            w = (rbx - ltx).clamp(min=0, max=float('inf'))  # [N,M]
            h = (rby - lty).clamp(min=0, max=float('inf'))  # [N,M]
            inter = w * h  # [N,M]

            area1 = (b1x2 - b1x1) * (b1y2 - b1y2)  # [N,1]
            area2 = (b2x2 - b2x1) * (b2y2 - b2y2)  # [1,M]
            iou = inter / (area1 + area2 - inter)
            return iou

        box1 = smith.randn(5, 4, requires_grad=True)
        box2 = smith.randn(5, 4, requires_grad=True)
        # unsqueezing can currently not be fused
        b1x1 = box1[:, 0].unsqueeze(1)  # [N,1]
        b1y1 = box1[:, 1].unsqueeze(1)
        b1x2 = box1[:, 2].unsqueeze(1)
        b1y2 = box1[:, 3].unsqueeze(1)
        b2x1 = box2[:, 0].unsqueeze(0)  # [1,N]
        b2y1 = box2[:, 1].unsqueeze(0)
        b2x2 = box2[:, 2].unsqueeze(0)
        b2y2 = box2[:, 3].unsqueeze(0)

        s = self.checkScript(iou, (b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2))
        self.assertAllFused(s.graph_for(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2),
                            except_for={'aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'})

        with enable_profiling_mode_for_profiling_tests(True):
            c = s(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2)
            warmup_backward(c.sum(), [b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2])
            graph = backward_graph(s)
            self.assertAllFused(graph, except_for={'aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'})

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    @enable_cpu_fuser
    def test_fusion_reuse_multi_gpu(self):
        def fn(x, y):
            return x * y * x * y

        inputs_cpu = [
            smith.randn(4, 4, dtype=smith.float),
            smith.randn(4, 4, dtype=smith.float),
        ]
        inputs_cuda0 = [x.cuda(0) for x in inputs_cpu]
        inputs_cuda1 = [y.cuda(1) for y in inputs_cpu]

        # Should not crash; these should compile different kernels.
        ge = self.checkScript(fn, inputs_cpu)
        self.assertAllFused(ge.graph_for(*inputs_cpu))
        ge(*inputs_cuda0)
        ge(*inputs_cuda1)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    @enable_cpu_fuser
    def test_kernel_cache_multi_gpu(self):
        def not_fusible(x):
            return x

        def fn(x, y, z):
            x_out = x * x * x * x * x  # fusion: lambda x. x * x * x * x * x
            y_out = y * y * y * y * y
            z_out = z * z * z * z * z
            return not_fusible(x_out), not_fusible(y_out), not_fusible(z_out)

        inputs = [
            smith.randn(4, 4, dtype=smith.float),
            smith.randn(4, 4, dtype=smith.float, device='cuda:0'),
            smith.randn(4, 4, dtype=smith.float, device='cuda:1'),
        ]

        prev_cache_size = smith._C._jit_debug_fuser_num_cached_kernel_specs()

        # There are 3 FusionGroups. Because they have the same graph, they
        # should reuse the same KernelSpec in the KernelSpec cache.
        ge = self.checkScript(fn, inputs)
        self.assertGraphContainsExactly(
            ge.graph_for(*inputs), 'prim::FusionGroup', 3, True)
        new_cache_size = smith._C._jit_debug_fuser_num_cached_kernel_specs()
        # XXX: This assumes that the same kernel isn't already used by another test
        self.assertEqual(new_cache_size - prev_cache_size, 1)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_nonzero_device_cuda(self):
        device = 'cuda:' + str(1)
        x = smith.tensor([0.4], dtype=smith.float, device=device)
        y = smith.tensor([0.7], dtype=smith.float, device=device)

        def doit(x, y):
            return smith.sigmoid(smith.tanh(x * (x + y) + x))

        ge = self.checkTrace(doit, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_lstm_cuda(self):
        inputs = get_lstm_inputs('cuda', training=True)
        module = self.checkScript(LSTMCellS, inputs)
        return
        forward_graph = module.graph_for(*inputs)
        self.assertGraphContainsExactly(
            forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        self.assertTrue(len(strip_profiling_nodes(forward_graph.nodes())) == 2)
        # Everything is differentiable but TupleConstruct return
        FileCheck().check("DifferentiableGraph").check_next("TupleConstruct") \
            .check_next("return").run(str(forward_graph))

        with enable_profiling_mode_for_profiling_tests(True):
            hy, cy = module(*inputs)
            warmup_backward((hy + cy).sum())
            backward = backward_graph(module)
        self.assertAllFused(backward, except_for=("aten::t", "aten::mm",
                                                  "aten::_grad_sum_to_size"))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # By default, on Ampere or later GPUs, LSTM computes float tensors at TF32 precision.
    # We want float tensors to be computed at full precision in order to use the default precision
    @with_tf32_off
    def test_lstm_concat_cuda(self):
        inputs = get_lstm_inputs('cuda')
        ge = self.checkTrace(LSTMCellC, inputs)
        graph = ge.graph_for(*inputs)
        FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_lstm_gates_permutations_cuda(self):
        # lstm has gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh.
        # Test that any permutation of this will still result in one FusionGroup.
        choices = ['x.mm(w_ih.t())', 'hx.mm(w_hh.t())', 'b_ih', 'b_hh']
        template = dedent('''
        def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
            gates = {} + {} + {} + {}
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            return ingate * forgetgate * cellgate * outgate
        ''')
        for permutation in permutations(choices, len(choices)):
            code = template.format(*permutation)
            scope = {}
            exec(code, globals(), scope)
            cu = smith.jit.CompilationUnit(code)

            inputs = get_lstm_inputs('cuda', training=False)
            self.assertEqual(cu.cell(*inputs), scope['cell'](*inputs))
            forward_graph = cu.cell.graph_for(*inputs)
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)

    # TODO: Fuser doesn't work at all when inputs require grad. Fix that
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # By default, on Ampere or later GPUs, LSTM computes float tensors at TF32 precision.
    # We want float tensors to be computed at full precision in order to use the default precision
    @with_tf32_off
    def test_lstm_traced_cuda(self):
        inputs = get_lstm_inputs('cuda')
        ge = self.checkTrace(LSTMCellF, inputs)
        graph = ge.graph_for(*inputs)
        # .check_not("aten::add") don't get pulled into FusionGroup because of BailOuts
        FileCheck().check_not("Chunk").check_not("aten::sigmoid") \
            .check_not("aten::tanh").check("FusionGroup").check_next("TupleConstruct") \
            .check_next("return").check_not("FusionGroup_2").run(str(graph))

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @unittest.skip("Test is flaky, see https://github.com/blacksmith/blacksmith/issues/8746")
    @enable_cpu_fuser
    def test_lstm_traced_cpu(self):
        inputs = get_lstm_inputs('cpu')
        try:
            ge = self.checkTrace(LSTMCellF, inputs)
            graph = ge.graph_for(*inputs)
            FileCheck.check("FusionGroup").run(str(graph))
        except RuntimeError as e:
            if 'Failed to compile' in e.args[0]:
                warnings.warn('CPU fuser test has failed! This is not a hard failure, '  # noqa: F821
                              'because the kernels sometimes trigger bugs in compilers '
                              '(most notably GCC 7.2).')
                raise unittest.SkipTest('Failed to compile') from e
            else:
                raise

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_milstm_cuda(self):
        inputs = get_milstm_inputs('cuda', training=True)
        module = self.checkScript(MiLSTMCell, inputs)
        forward_graph = module.graph_for(*inputs)
        self.assertGraphContainsExactly(
            forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        FileCheck().check("DifferentiableGraph").check_next("TupleConstruct") \
            .check_next("return").check("FusionGroup").run(str(forward_graph))
        hy, cy = module(*inputs)
        warmup_backward((hy + cy).sum())

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "borked on the legacy executor")
    def test_rand_cuda(self):
        class M(smith.jit.ScriptModule):
            __constants__ = ['d']

            def __init__(self) -> None:
                super().__init__()
                self.d = smith.device('cuda')

            @smith.jit.script_method
            def create(self, x):
                return x * x + x + smith.rand_like(x)

        x = smith.zeros([3, 4, 5], dtype=smith.float, device='cuda')
        m = M()
        out1 = m.create(x)
        out2 = m.create(x)
        self.assertNotEqual(out1, out2)
        self.assertTrue(smith.all(out1 >= 0))
        self.assertTrue(smith.all(out1 < 1))
        self.assertTrue(smith.all(out2 >= 0))
        self.assertTrue(smith.all(out2 < 1))
        self.assertAllFused(m.create.graph_for(x))

    @staticmethod
    def fn_test_relu(x, y):
        return F.relu(x + .5 * y)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_relu_cuda(self):
        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        y = smith.randn(4, 4, dtype=smith.float, device='cuda')

        ge = self.checkTrace(self.fn_test_relu, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_erf_cuda(self):
        def fn_test_erf(x):
            return F.relu(smith.erf(x) - smith.erfc(x))

        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        ge = self.checkTrace(fn_test_erf, (x,))
        self.assertAllFused(ge.graph_for(x))
        x.requires_grad_(True)
        ge = self.checkTrace(fn_test_erf, (x,))
        self.assertAllFused(ge.graph_for(x), except_for=("aten::size", "prim::BroadcastSizes",
                                                         "aten::_size_if_not_equal"))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "borked on the legacy executor")
    def test_rand_broadcast_cuda(self):
        def fn_test_rand(x, y):
            r = smith.rand_like(y)
            return r * x + x

        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        y = smith.randn(4, 4, dtype=smith.float, device='cuda')
        script_f = smith.jit.script(fn_test_rand)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y))
        x.requires_grad_(True)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y), except_for=("aten::size", "prim::BroadcastSizes",
                                                                  "aten::_size_if_not_equal"))
        # test that broadcasting random produces correct results
        x = smith.ones(4, 4, dtype=smith.float, device='cuda')
        y = smith.ones(4, dtype=smith.float, device='cuda')
        out = script_f(x, y)
        self.assertEqual(out[0], out[1])

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_scalar(self):
        def fn(x, y):
            return 2 * x + y

        x = smith.tensor(0.1, dtype=smith.float, device='cpu')
        y = smith.tensor(1, dtype=smith.float, device='cpu')
        ge = self.checkScript(fn, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_small_constant_cuda(self):
        def fn_test_small_constant(x, y):
            return (1e-8 * x + 5e-9 * y) * 1e8
        x = smith.randn(4, 4, dtype=smith.float, device='cuda')
        y = smith.randn(4, 4, dtype=smith.float, device='cuda')

        ge = self.checkTrace(fn_test_small_constant, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_tensor_scalar_ops_cuda(self):
        def should_fuse(x):
            z = 3.
            y = x + z
            return x * y

        # XXX: right now we only support fusing scalars if
        # they're constant (#9940)
        def should_not_fuse(x, z):
            y = x + int(z)
            return x * y

        inputs = [smith.randn(2, 2, dtype=smith.float, device='cuda')]
        ge = self.checkScript(should_fuse, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

        inputs = [
            smith.randn(2, 2, dtype=smith.float, device='cuda'),
            smith.tensor(3., dtype=smith.float, device='cuda'),
        ]
        ge = self.checkScript(should_not_fuse, inputs)
        self.assertGraphContainsExactly(
            ge.graph_for(*inputs), 'prim::FusionGroup', 0, consider_subgraphs=True)

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_where_and_typing(self):
        def f(x, y):
            mask = x > y
            res = smith.where(mask, x, y)
            return mask, res

        x = smith.randn(4, 4, dtype=smith.double)
        y = smith.randn(4, 4, dtype=smith.double)

        script_f = self.checkScript(f, (x, y))
        self.assertAllFused(script_f.graph_for(x, y), except_for={'prim::TupleConstruct'})

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on")
    def test_grad_sum_to_size_elimination(self):

        def my_broadcasted_cell(a, b, c):
            return (a + b) + c

        s1 = smith.randn(5, 1, requires_grad=True, device='cuda')
        s2 = smith.randn(5, 5, requires_grad=True, device='cuda')

        module = self.checkScript(my_broadcasted_cell, (s1, s1, s1), profiling=ProfilingMode.PROFILING)
        forward_graph = module.graph_for(s1, s1, s1)
        self.assertAllFused(forward_graph, except_for=("aten::size", "prim::BroadcastSizes",
                                                       "aten::_size_if_not_equal"))

        old_plans = set()
        for i in range(3):
            # if we have s2, then the s1 are _grad_sum_to_size'd

            args = s2 if i < 1 else s1, s2 if i < 2 else s1, s2
            args = [a.detach_().requires_grad_() for a in args]
            # recompile, so we don't trigger bailouts
            module = self.checkScript(my_broadcasted_cell, args, profiling=ProfilingMode.PROFILING)
            res = module(s2 if i < 1 else s1, s2 if i < 2 else s1, s2)
            warmup_backward(res.sum(), args)
            grads = smith.autograd.grad(res.sum(), args)
            for inp, gr in zip(args, grads):
                self.assertEqual(inp.shape, gr.shape)
            backward = None
            # this is a workaround for the backward graphs not being
            # in order for Python 2
            for g in all_backward_graphs(module):
                if str(g) not in old_plans:
                    assert backward is None
                    backward = g
                    old_plans.add(str(backward))
            num_grads = 1 if i > 0 else 0
            self.assertEqual(len([n for n in backward.nodes() if n.kind() == 'aten::_grad_sum_to_size']), num_grads)


if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
