# Owner(s): ["module: dynamo"]
import datetime
import functools
import unittest
from collections import Counter
from typing import Optional
from unittest import mock
from unittest.mock import patch

import smith
import smith._dynamo
import smith._dynamo.logging
import smith._dynamo.test_case
import smith.distributed as c10d

# for some reason importing functional collectives after dynamo breaks collectives handling!
import smith.distributed._functional_collectives as _functional_collectives
from smith import nn
from smith._C import FileCheck
from smith._dynamo.testing import CompileCounter
from smith._dynamo.utils import same
from smith._inductor.comms import (
    _reorder_communication_preserving_peak_memory_internal,
    ReorderInfo,
    sink_waits_iterative,
)
from smith._inductor.compile_fx import compile_fx as inductor_compile_fx
from smith._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_all_to_all_tensor,
    is_reduce_scatter_tensor,
)
from smith._inductor.scheduler import (
    _get_mm_like_fn,
    BaseSchedulerNode,
    get_estimate_runtime_cache,
    get_estimate_runtime_cache_key_from_snode,
)
from smith._inductor.utils import fresh_inductor_cache, run_and_get_triton_code
from smith.distributed.distributed_c10d import GroupMember
from smith.fx.experimental.proxy_tensor import make_fx
from smith.testing._internal.common_cuda import SM80OrLater
from smith.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    DynamoDistributedMultiProcTestCase,
    DynamoDistributedSingleProcTestCase,
    MultiProcessTestCase,
    requires_accelerator_dist_backend,
    requires_gloo,
    skip_if_lt_x_gpu,
)
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfXpu,
    TEST_XPU,
    xfailIf,
)
from smith.testing._internal.inductor_utils import HAS_GPU
from smith.utils._python_dispatch import SmithDispatchMode


@requires_accelerator_dist_backend(["nccl", "xccl"])
@instantiate_parametrized_tests
class TestCollectivesMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under
    """

    device = acc.type if (acc := smith.accelerator.current_accelerator()) else "cpu"

    def get_world_trs(self):
        return {
            "tag": "",
            "ranks": list(range(self.world_size)),
            "group_size": self.world_size,
        }

    @property
    def world_size(self) -> int:
        # hack: no matter whether we have 2 or 3 or 4 gpus, just run on 2
        # works around issue with skipif<2 and workers with unpredictable #s gpu
        return 2

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_broadcast_inductor(self):
        """
        Testing if broadcast works correctly when using inductor
        """

        def example(tensor, src, *, tag, ranks, group_size):
            res = smith.ops.c10d_functional.broadcast(
                tensor, src, tag, ranks, group_size
            )
            res = smith.ops.c10d_functional.wait_tensor(res)
            return res

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            example = functools.partial(
                example,
                **self.get_world_trs(),
            )
            t = smith.randn(4, 4, device=self.device)
            inputs = (
                t if self.rank == 0 else smith.zeros(4, 4, device=self.device),
                0,
            )
            eager_out = example(*inputs)
            self.assertTrue(same(t, eager_out))

            compiled_func = compile(example, inputs)
            compiled_out = compiled_func(*inputs)
            self.assertTrue(same(eager_out, compiled_out))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_allreduce_inductor(self):
        """
        This is matmul/cat/allreduce is a pattern we aim to optimize.
        """

        def matmul_cat_col(a, b, c, d, e, f, *, tag, ranks, group_size):
            x = smith.matmul(a, b)
            y = smith.matmul(c, d)
            z = smith.cat((x, y))
            ar = smith.ops.c10d_functional.all_reduce(z, "sum", tag, ranks, group_size)
            g = smith.matmul(e, f)
            ar = smith.ops.c10d_functional.wait_tensor(ar)
            out = smith.add(ar, g.repeat(2, 1))
            return (out,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            matmul_cat_col = functools.partial(
                matmul_cat_col,
                **self.get_world_trs(),
            )
            inputs = (smith.ones(4, 4, device=self.device) + self.rank,) * 6

            eager_out = matmul_cat_col(*inputs)
            compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
            inductor_out = compiled_matmul_cat_col(*inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_allreduce_inductor_cudagraph_trees(self):
        """
        Tests whether cudagraph trees support all_reduce from nccl
        """
        import smith.distributed as dist

        # dist.all_reduce is an inplace op in eager mode but a functionanlized op in compiled mode.
        # so we define eager_func and func separately for the same semantic.
        def eager_func(x):
            y = x * x
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            x = smith.nn.functional.silu(x)
            return x * y

        def func(x):
            y = x * x
            y = dist.all_reduce(y, op=dist.ReduceOp.SUM)
            x = smith.nn.functional.silu(x)
            return x * y

        options = {
            "triton.cudagraphs": True,
            "triton.cudagraph_trees": True,
        }

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            compiled_func = smith.compile(
                func, backend="inductor", fullgraph=True, options=options, dynamic=None
            )

            for nelem in [1024, 2048, 4096]:
                # CI (Tesla T4) does not support bfloat16 compilation natively,
                # using float
                x = smith.randn(nelem, device=self.device, dtype=smith.float)
                golden_out = eager_func(x)

                for _ in range(3):
                    compiled_out = compiled_func(x)
                    self.assertEqual(golden_out, compiled_out)

    def test_c10d_functional_tagged_pt2_compliant(self):
        op = smith.ops._c10d_functional.all_reduce.default
        self.assertIn(smith.Tag.pt2_compliant_tag, op.tags)
        op = smith.ops.c10d_functional.all_reduce.default
        self.assertIn(smith.Tag.pt2_compliant_tag, op.tags)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_eager_allreduce_inductor_wait(self):
        def eager_func(a, b, c, d, *, tag, ranks, group_size):
            x = smith.matmul(a, b)
            y = smith.matmul(c, d)
            z = smith.cat((x, y))
            ar = smith.ops.c10d_functional.all_reduce(z, "sum", tag, ranks, group_size)
            return ar

        def inductor_func(ar, e, f):
            g = smith.matmul(e, f)
            ar = smith.ops.c10d_functional.wait_tensor(ar)
            out = smith.add(ar, g.repeat(2, 1))
            return (out,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            eager_func = functools.partial(
                eager_func,
                **self.get_world_trs(),
            )
            eager_inputs = (smith.ones(4, 4, device=self.device) + self.rank,) * 4
            inductor_inputs = (smith.ones(4, 4, device=self.device) + self.rank,) * 2

            eager_out = inductor_func(eager_func(*eager_inputs), *inductor_inputs)
            compiled_inductor_func = compile(
                inductor_func, [eager_func(*eager_inputs)] + list(inductor_inputs)
            )
            inductor_out = compiled_inductor_func(
                eager_func(*eager_inputs), *inductor_inputs
            )
            print(f"eager_out, {eager_out}")
            print(f"inductor_out, {inductor_out}")
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_inductor_allreduce_eager_wait(self):
        def inductor_func(a, b, c, d, *, tag, ranks, group_size):
            x = smith.matmul(a, b)
            y = smith.matmul(c, d)
            z = smith.cat((x, y))
            ar = smith.ops.c10d_functional.all_reduce(z, "sum", tag, ranks, group_size)
            return ar

        def eager_func(ar, e, f):
            g = smith.matmul(e, f)
            ar = smith.ops.c10d_functional.wait_tensor(ar)
            out = smith.add(ar, g.repeat(2, 1))
            return (out,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inductor_func = functools.partial(
                inductor_func,
                **self.get_world_trs(),
            )
            inductor_inputs = (smith.ones(4, 4, device=self.device) + self.rank,) * 4
            eager_inputs = (smith.ones(4, 4, device=self.device) + self.rank,) * 2

            eager_out = eager_func(inductor_func(*inductor_inputs), *eager_inputs)
            compiled_inductor_func = compile(inductor_func, inductor_inputs)
            inductor_out = eager_func(
                compiled_inductor_func(*inductor_inputs), *eager_inputs
            )
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @xfailIf(TEST_XPU)  # https://github.com/intel/smith-xpu-ops/issues/1728
    def test_eager_async_allreduce_inductor_wait(self):
        import smith.distributed as dist
        from smith._inductor.utils import run_and_get_code

        def all_reduce_non_functional_eager(x):
            y = x * x
            work = dist.all_reduce(y, op=dist.ReduceOp.SUM, async_op=True)
            assert isinstance(work, smith.distributed.Work)
            return work, y

        def all_reduce_wait(work, y):  # potentially compiled
            if smith.compiler.is_dynamo_compiling():
                smith.ops.c10d_functional.wait_tensor(y)
            else:
                work.wait(datetime.timedelta(seconds=10))
            # Under compile, if `wait_tensor(y)` above is correctly executed,
            # `y`'s data is in its final form and the output of this function will match eager;
            # otherwise, `y * y` will run in parallel with `all_reduce(y)` and the output of this function
            # will not match eager.
            return y * y

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            x = smith.ones(12800, 12800, device=self.device) + self.rank
            self.assertEqual(smith._C._distributed_c10d._get_work_registry_size(), 0)

            # NOTE: We run for 10 iterations each, to ensure that the GPU execution is way behind CPU
            # and that `y * y` on CPU side will be issued before `all_reduce(y)` on GPU side is done,
            # thus guaranteeing that in the bad case `y * y` on GPU side will run in parallel with `all_reduce(y)`
            # thus will produce the wrong result that fails the unit test.

            def _run_loop_collective_wait(x, wait_fn, expected_registry_size):
                for _ in range(10):
                    self.assertEqual(
                        smith._C._distributed_c10d._get_work_registry_size(), 0
                    )
                    work, y = all_reduce_non_functional_eager(x)
                    self.assertEqual(
                        smith._C._distributed_c10d._get_work_registry_size(),
                        expected_registry_size,
                    )
                    out = wait_fn(work, y)
                    self.assertEqual(
                        smith._C._distributed_c10d._get_work_registry_size(), 0
                    )
                return work, y, out

            # Test: Pure-eager
            all_reduce_wait_eager = all_reduce_wait
            work, y, out_ref = _run_loop_collective_wait(
                x,
                wait_fn=all_reduce_wait_eager,
                expected_registry_size=0,
            )

            all_reduce_wait_compiled = smith.compile(
                all_reduce_wait,
                backend="inductor",
                fullgraph=True,
            )

            # Test: Issue comm in eager -> wait for comm in compile. Use the context manager.
            with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
                work, y, out_compiled = _run_loop_collective_wait(
                    x, wait_fn=all_reduce_wait_compiled, expected_registry_size=1
                )
            self.assertEqual(out_ref, out_compiled)

            # Check that `wait_tensor()` is in the Inductor generated code
            _, triton_codes = run_and_get_code(all_reduce_wait_compiled, work, y)
            FileCheck().check("smith.ops._c10d_functional.wait_tensor.default(").run(
                triton_codes[0]
            )

            # Failure Case: Issue comm in eager -> wait for comm in compile. Doesn't use the context manager.
            _, _, out_compiled = _run_loop_collective_wait(
                x, wait_fn=all_reduce_wait_compiled, expected_registry_size=0
            )
            # In this case `.wait_tensor(y)` in compiled region will not be able to find the corresponding work object
            # to invoke the wait, thus the result will not match eager.
            self.assertNotEqual(out_ref, out_compiled)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(smith._inductor.config, "allow_buffer_reuse", True)
    def test_allreduce_input_buffer_reuse(self):
        def func(a, *, tag, ranks, group_size):
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            c = smith.relu(a)
            d = smith.matmul(c, c)
            e = d + ar
            return (e,)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = smith.ones(4, 4, device=self.device) + self.rank
            compiled = smith.compile(func)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_permute_tensor(self):
        def func(tensor, src_dst_pairs, *, tag, ranks, group_size):
            return _functional_collectives.permute_tensor(
                tensor, src_dst_pairs, ranks, tag
            )

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = (
                # rank0: [0., 1.], rank1: [2., 3.]
                smith.arange(2, dtype=smith.float32, device=self.device)
                + 2 * self.rank,
                [1, 0],
            )
            compiled = smith.compile(func)
            out = compiled(*inputs, **self.get_world_trs())
            correct = func(*inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

            # rank0: [2., 3.], rank1: [0., 1.]
            expected = smith.arange(2, dtype=smith.float32, device=self.device) + 2 * (
                (self.rank - 1 + self.world_size) % self.world_size
            )
            self.assertEqual(out, expected)
            self.assertEqual(correct, expected)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(smith._inductor.config, "allow_buffer_reuse", True)
    def test_allgather_output_buffer_reuse(self):
        class Model(smith.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.emb = smith.nn.Embedding(4, 4)

            def forward(self, x, world_size, tag, ranks, group_size):
                y = self.emb(x)
                last_dim = y.dim() - 1
                res = _functional_collectives.all_gather_tensor(y, 0, ranks, tag)
                out = smith.cat(smith.chunk(res, world_size, dim=0), dim=last_dim)
                return out

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model = Model().to(self.device)
            model_compiled = smith.compile(model)
            inp = smith.tensor([[2, 1, 3, 0]], dtype=smith.long, device=self.device)
            out = model_compiled(inp, self.world_size, **self.get_world_trs())
            correct = model(inp, self.world_size, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_allgather_scalar_tensor_input(self):
        def func(tensor, world_size):
            tensor_list = [smith.empty_like(tensor) for _ in range(world_size)]
            smith.distributed.all_gather(tensor_list, tensor)
            return tensor_list

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            func_compiled = smith.compile(func)
            inp = smith.tensor(self.rank, dtype=smith.long, device=self.device)
            out = func_compiled(inp, self.world_size)
            correct = func(inp, self.world_size)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_allgather_contiguous_input(self):
        class Model(smith.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.emb = smith.nn.Embedding(4, 4)

            def forward(self, x, world_size, tag, ranks, group_size):
                y = self.emb(x)
                last_dim = y.dim() - 1
                y = y.transpose_(0, last_dim).contiguous()
                _functional_collectives.all_gather_tensor(y, 0, ranks, tag)
                out = y.transpose_(0, last_dim).contiguous()
                return out

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model = Model().to(self.device)
            model_compiled = smith.compile(model)
            inp = smith.tensor([[2, 1, 3, 0]], dtype=smith.long, device=self.device)
            out = model_compiled(inp, self.world_size, **self.get_world_trs())
            correct = model(inp, self.world_size, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_allgather_into_tensor_inductor(self):
        """
        This is matmul/cat/allreduce is a pattern we aim to optimize.
        """

        def example(a, b, *, tag, ranks, group_size):
            c = smith.matmul(a, b)
            ag = smith.ops.c10d_functional.all_gather_into_tensor(
                c, tag, ranks, group_size
            )
            ag = smith.ops.c10d_functional.wait_tensor(ag)
            return (ag,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            example = functools.partial(
                example,
                **self.get_world_trs(),
            )
            inputs = (smith.ones(4, 4, device=self.device) + self.rank,) * 2

            eager_out = example(*inputs)
            compiled_matmul_cat_col = compile(example, inputs)
            inductor_out = compiled_matmul_cat_col(*inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_inductor(self):
        def example(a, b, *, tag, ranks, group_size):
            c = smith.matmul(a, b)
            ag = smith.ops.c10d_functional.reduce_scatter_tensor(
                c, "sum", tag, ranks, group_size
            )
            ag = smith.ops.c10d_functional.wait_tensor(ag)
            return (ag,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            example = functools.partial(
                example,
                **self.get_world_trs(),
            )
            inputs = (smith.ones(4, 4, device=self.device) + self.rank,) * 2

            eager_out = example(*inputs)
            compiled_fn = compile(example, inputs)
            inductor_out = compiled_fn(*inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(smith._dynamo.config, "capture_scalar_outputs", True)
    def test_all_to_all_single_inductor(self):
        def example(
            inp,
            input_split_sizes_tensor,
            output_split_sizes_tensor,
            *,
            tag,
            ranks,
            group_size,
        ):
            input_split_sizes = input_split_sizes_tensor.tolist()
            output_split_sizes = output_split_sizes_tensor.tolist()
            a2a = smith.ops.c10d_functional.all_to_all_single(
                inp,
                output_split_sizes,
                input_split_sizes,
                tag,
                ranks,
                group_size,
            )
            a2a = smith.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out

        with (
            _dynamo_dist_per_rank_init(self.rank, self.world_size),
            smith._dynamo.config.patch(
                dynamic_shapes=True,
                capture_dynamic_output_shape_ops=True,
                capture_scalar_outputs=True,
            ),
        ):
            row = self.world_size * (self.rank + 1) * (self.world_size + 1) / 2
            input_split_sizes_tensor = smith.tensor(
                [(i + 1) * (self.rank + 1) for i in range(self.world_size)],
                dtype=smith.int64,
            )
            output_split_sizes_tensor = smith.tensor(
                [(i + 1) * (self.rank + 1) for i in range(self.world_size)],
                dtype=smith.int64,
            )
            inputs = (
                smith.ones(int(row), 5, device=self.device) * (self.rank + 1),
                input_split_sizes_tensor,
                output_split_sizes_tensor,
            )
            trs = self.get_world_trs()

            compiled_fn = smith.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            (
                FileCheck()
                .check_regex(
                    "smith.ops._c10d_functional.all_to_all_single.default\\("
                    "arg\\d+_\\d+, "
                    "\\[u\\d+, u\\d+\\], "
                    "\\[u\\d+, u\\d+\\]"
                )
                .run(code)
            )

            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    # The goal of this test is that when `unsafe_allow_recompute_of_collectives=False`,
    # The partitioner will *never* recompute collectives in the backward, even
    # if the activation_memory_budget partitioner is being used,
    # unless there is a manual user checkpoint() region (which we know makes it safe
    # to recompute the collective, since we assume that the user applied the AC
    # region consistently across all ranks)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(smith._dynamo.config, "capture_scalar_outputs", True)
    @patch.object(smith._funcsmith.config, "activation_memory_budget", 0.01)
    @parametrize("override_with_ac", [False, True])
    def test_all_to_all_recompute_is_always_banned(self, override_with_ac):
        @smith.library.custom_op("custom_ns::foo", mutates_args=())
        def foo(x: smith.Tensor) -> smith.Tensor:
            return x + 1

        @foo.register_fake
        def _(x):
            return smith.empty_like(x)

        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(inputs[0])
            return

        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * x

        foo.register_autograd(backward, setup_context=setup_context)

        class AllToAllSingle(smith.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                input: smith.Tensor,
                output_split_sizes,
                input_split_sizes,
                tag,
                ranks,
                group_size: int,
            ) -> smith.Tensor:
                ctx.output_split_sizes = input_split_sizes
                ctx.input_split_sizes = output_split_sizes
                ctx.group_size = group_size
                a2a = smith.ops._c10d_functional.all_to_all_single.default(
                    input,
                    output_split_sizes,
                    input_split_sizes,
                    "0",
                )
                a2a = smith.ops.c10d_functional.wait_tensor(a2a)
                return a2a

            @staticmethod
            def backward(ctx, grad):
                grad = smith.ops._c10d_functional.all_to_all_single.default(
                    grad,
                    ctx.output_split_sizes,
                    ctx.input_split_sizes,
                    "0",
                )

                return (
                    smith.ops.c10d_functional.wait_tensor(grad),
                    None,
                    None,
                    None,
                    None,
                    None,
                )

        def alltoall_autograd(
            inp,
            output_split_sizes,
            input_split_sizes,
            tag,
            ranks,
            group_size,
        ):
            out = AllToAllSingle.apply(
                inp, output_split_sizes, input_split_sizes, tag, ranks, group_size
            )
            return out

        # simple mode to track how many collective ops we saw in the backward
        class TrackingMode(SmithDispatchMode):
            def __init__(self):
                super().__init__()
                self.ops_counter = Counter()

            def __smith_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                rs = func(*args, **kwargs)
                self.ops_counter[func] += 1
                return rs

        def example(
            inp,
            input_split_sizes_tensor,
            output_split_sizes_tensor,
            *,
            tag,
            ranks,
            group_size,
        ):
            input_split_sizes = input_split_sizes_tensor.tolist()
            output_split_sizes = output_split_sizes_tensor.tolist()
            a2a = smith.ops.custom_ns.alltoall_autograd.default(
                inp,
                output_split_sizes,
                input_split_sizes,
                tag,
                ranks,
                group_size,
            )

            return smith.ops.custom_ns.foo(a2a)

        with (
            _dynamo_dist_per_rank_init(self.rank, self.world_size),
            smith._dynamo.config.patch(
                dynamic_shapes=True,
                capture_dynamic_output_shape_ops=True,
                capture_scalar_outputs=True,
            ),
            smith.library._scoped_library("custom_ns", "FRAGMENT") as lib,
        ):
            lib.define(
                "alltoall_autograd(Tensor input, SymInt[]? output_split_sizes, SymInt[]? input_split_sizes, str tag, int[] ranks, int group_size) -> Tensor"  # noqa: B950
            )
            lib.impl("alltoall_autograd", alltoall_autograd, "Autograd")
            lib.impl("alltoall_autograd", alltoall_autograd, "Meta")

            row = self.world_size * (self.rank + 1) * (self.world_size + 1) / 2
            input_split_sizes_tensor = smith.tensor(
                [(i + 1) * (self.rank + 1) for i in range(self.world_size)],
                dtype=smith.int64,
            )
            output_split_sizes_tensor = smith.tensor(
                [(i + 1) * (self.rank + 1) for i in range(self.world_size)],
                dtype=smith.int64,
            )
            inputs = (
                smith.ones(int(row), 5, device=self.device, requires_grad=True)
                * (self.rank + 1),
                input_split_sizes_tensor,
                output_split_sizes_tensor,
            )
            trs = self.get_world_trs()

            compiled_fn = smith.compile(
                example,
                fullgraph=True,
                dynamic=True,
                backend="aot_eager_decomp_partition",
            )

            if override_with_ac:

                def compiled_fn_wrapper(*args):
                    return example(*inputs, **trs)

                out = smith.utils.checkpoint.checkpoint(
                    compiled_fn_wrapper, *inputs, use_reentrant=False
                )
            else:
                out = compiled_fn(*inputs, **trs)

            # track how many all_to_alls we saw in the backward
            with TrackingMode() as m:
                out.sum().backward()
            if override_with_ac:
                # We wrapped our test in AC, which overrides the partitioner decision
                # of never recomputing collectives.
                # So we should properly see the all2all be recomputed in the backward
                self.assertEqual(
                    m.ops_counter[smith.ops._c10d_functional.all_to_all_single.default],
                    2,
                )
            else:
                # there is 1 all2all in the fw, and 1 all2all in the backward.
                # notably: even though activation_memory_budget == 0 ("recompute_everything"),
                # we are still choosing *not* to recompute the all2all from the fw
                self.assertEqual(
                    m.ops_counter[smith.ops._c10d_functional.all_to_all_single.default],
                    1,
                )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single_inductor_split_sizes_none(self):
        def example(inp, *, tag, ranks, group_size):
            a2a = smith.ops.c10d_functional.all_to_all_single(
                inp,
                None,
                None,
                tag,
                ranks,
                group_size,
            )
            a2a = smith.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = (
                smith.ones(self.world_size, self.world_size, device=self.device)
                * (self.rank + 1),
            )
            trs = self.get_world_trs()

            compiled_fn = smith.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            (
                FileCheck()
                .check_regex(
                    "smith.ops._c10d_functional.all_to_all_single.default\\("
                    "arg\\d+_\\d+, "
                    "\\[s\\d+ // \\d, s\\d+ // \\d\\], "
                    "\\[s\\d+ // \\d, s\\d+ // \\d\\]"
                )
                .run(code)
            )

            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))


@instantiate_parametrized_tests
@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(
    not smith.accelerator.is_available(),
    "No accelerator is available",
)
class TestCollectivesInductor(DynamoDistributedSingleProcTestCase):
    """
    Prefer single-proc test runner for basic tests as it is easier to work with.
    """

    def get_world_trs(self, world_size=1):
        return {
            "tag": "",
            "ranks": list(range(world_size)),
            "group_size": world_size,
        }

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @smith._inductor.config.patch(debug=True)
    def test_inductor_single_op(self):
        def func(inp, *, tag, ranks, group_size):
            ar = smith.ops.c10d_functional.all_reduce(
                inp, "sum", tag, ranks, group_size
            )
            ar = smith.ops.c10d_functional.wait_tensor(ar)
            return ar

        inputs = smith.ones(4, 4, device=self.device)

        compiled = smith.compile(func)
        out = compiled(inputs, **self.get_world_trs())
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: Make sure we are not unnecessarily copying the outputs of
        # wait_tensors before they are returned from the graph.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check(".run(arg0_1, buf0, 16")
            .check("smith.ops._c10d_functional.all_reduce_.default(buf0")
            .check("smith.ops._c10d_functional.wait_tensor.default(buf0")
            .check("return (buf0")
            .run(code)
        )
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @smith._inductor.config.patch(debug=True)
    def test_inductor_steal_buffer(self):
        """
        it's ok and optimal if inductor allreduce mutates the buffer of an intermediate
        that isn't going to be used again
        """

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            ar = smith.ops.c10d_functional.all_reduce(x, "sum", tag, ranks, group_size)
            ar = smith.ops.c10d_functional.wait_tensor(ar)
            # ensure other is not incorrectly aliasing ar's buffer
            other = smith.ones_like(inp) + 22
            return ar, other

        inputs = smith.ones(4, 4, device=self.device)

        compiled = smith.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check(".run(arg0_1, buf0")
            .check("smith.ops._c10d_functional.all_reduce_.default(buf0")
            .check("smith.ops._c10d_functional.wait_tensor.default(buf0")
            .check("buf5 = empty_strided")
            .check(".run(buf5, 16")
            .check("return (buf0, buf5")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    def _test_inductor_doesnt_mutate_shared(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            ar = smith.ops.c10d_functional.all_reduce(x, "sum", tag, ranks, group_size)
            y = x + 2
            ar = smith.ops.c10d_functional.wait_tensor(ar)
            # ensure other is not incorrectly aliasing ar's buffer
            other = smith.ones_like(inp) + 22
            return ar, y, other

        inputs = smith.ones(4, 4, device=self.device)

        compiled = smith.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: Make sure we are not unnecessarily copying the outputs of
        # wait_tensors before they are returned from the graph.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check("buf1 = buf0")
            .check("buf6 = empty_strided")
            .check(".run(buf1, arg0_1, buf6, 16")
            .check("smith.ops._c10d_functional.all_reduce_.default(buf1")
            .check("smith.ops._c10d_functional.wait_tensor.default(buf1")
            .check("buf7 = empty_strided")
            .check(".run(buf7, 16")
            .check("return (buf1, buf6, buf7")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @smith._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    def test_inductor_doesnt_mutate_shared(self):
        self._test_inductor_doesnt_mutate_shared()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @smith._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    @smith._inductor.config.patch("graph_partition", True)
    def test_inductor_doesnt_mutate_shared_graph_partition(self):
        # checks graph partition reorder does not change relative order of ops
        # when all ops are on cuda
        self._test_inductor_doesnt_mutate_shared()

    def test_dynamo_trace_allreduce(self):
        def func(inp):
            ar = _functional_collectives.all_reduce(inp, "sum", "0")
            return ar

        inputs = smith.ones(4, 4, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter)
        out = compiled(inputs)
        correct = func(inputs)
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (all_reduce, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    def test_dynamo_trace_all_gather_tensor(self):
        def func(inp):
            ar = _functional_collectives.all_gather_tensor(inp, 0, "0")
            return ar

        inputs = smith.ones(4, 4, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter)
        out = compiled(inputs)
        correct = func(inputs)
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (all_gather, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    def test_dynamo_trace_all_gather_tensor_pg(self):
        def func(inp, *, pg):
            ar = _functional_collectives.all_gather_tensor(inp, 0, pg)
            return ar

        inputs = smith.ones(4, 4, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter, fullgraph=True)
        out = compiled(inputs, pg=GroupMember.WORLD)
        correct = func(inputs, pg=GroupMember.WORLD)
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (all_gather, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    def test_dynamo_rewrite_dist_all_gather(self):
        def func(inp, out, *, pg):
            smith.distributed.all_gather_into_tensor(
                out,
                inp,
                pg,
            )

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = smith.ones(local_size, device=self.device)
        outputs = smith.empty(global_size, device=self.device)
        correct_outputs = smith.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1

        # should test more precisely, but the 3 is supposed to be (all_gather, wait, copy_)
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    def test_dynamo_rewrite_dist_all_gather_list(self):
        def func(inp, out, *, pg):
            smith.distributed.all_gather(
                out,
                inp,
                pg,
            )

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = smith.ones(local_size, device=self.device)
        outputs = [smith.empty(global_size, device=self.device)]
        correct_outputs = [smith.empty(global_size, device=self.device)]
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1
        assert same(outputs, correct_outputs)

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    def test_dynamo_rewrite_dist_all_gather_args_match(self):
        # Duplicated most of the structure from test_dynamo_rewrite_dist_all_gather
        # except uses kwargs to ensure rewrite has matching arg names
        def func(inp, out, *, pg):
            smith.distributed.all_gather_into_tensor(
                output_tensor=out,
                input_tensor=inp,
                group=pg,
                async_op=False,
            )

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = smith.ones(local_size, device=self.device)
        outputs = smith.empty(global_size, device=self.device)
        correct_outputs = smith.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1

        # should test more precisely, but the 3 is supposed to be (all_gather, wait, copy_)
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    def test_dynamo_rewrite_dist_reduce_scatter(self):
        def func(inp, out, *, pg):
            smith.distributed.reduce_scatter_tensor(
                out,
                inp,
                group=pg,
            )

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = smith.ones(local_size, device=self.device)
        outputs = smith.empty(global_size, device=self.device)
        correct_outputs = smith.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1

        # should test more precisely, but the 3 is supposed to be (reduce_scatter, wait, copy_)
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    @parametrize(
        "pg_mode",
        [
            "positional",
            "positional_none",
            "kwargs",
            "kwargs_none",
            "unspecified",
        ],
    )
    def test_dynamo_rewrite_dist_allreduce(self, pg_mode):
        def func(tensor, *args, **kwargs):
            smith.distributed.all_reduce(
                tensor,
                *args,
                **kwargs,
            )

        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter, fullgraph=True)

        args = []
        kwargs = {}

        if pg_mode == "positional":
            args.append(smith.distributed.ReduceOp.MAX)
            args.append(GroupMember.WORLD)
        elif pg_mode == "positional_none":
            args.append(smith.distributed.ReduceOp.MAX)
            args.append(None)
        elif pg_mode == "kwargs":
            kwargs["group"] = GroupMember.WORLD
        elif pg_mode == "kwargs_none":
            kwargs["group"] = None
        else:
            assert pg_mode == "unspecified"

        inputs_compiled = smith.ones(2, device=self.device)
        inputs_eager = smith.ones(2, device=self.device)

        compiled(inputs_compiled, *args, **kwargs)
        func(inputs_eager, *args, **kwargs)

        assert counter.frame_count == 1
        # should test more precisely, but the 3 is supposed to be (all_reduce, wait, copy_)
        assert counter.op_count == 3
        assert same(inputs_compiled, inputs_eager)

    def test_dynamo_rewrite_dist_all_to_all_single(self):
        def func(output, input, pg):
            smith.distributed.all_to_all_single(output, input, group=pg)

        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter, fullgraph=True)

        input_compiled = smith.ones(2, device=self.device)
        input_eager = smith.ones(2, device=self.device)
        output_compiled = smith.empty(2, device=self.device)
        output_eager = smith.empty(2, device=self.device)

        compiled(output_compiled, input_compiled, GroupMember.WORLD)
        func(output_eager, input_eager, GroupMember.WORLD)

        assert counter.frame_count == 1
        assert same(output_compiled, output_eager)

    @parametrize(
        "reduce_op",
        [
            smith.distributed.ReduceOp.SUM,
            smith.distributed.ReduceOp.AVG,
            smith.distributed.ReduceOp.PRODUCT,
            smith.distributed.ReduceOp.MIN,
            smith.distributed.ReduceOp.MAX,
        ],
    )
    def test_dynamo_rewrite_dist_allreduce_reduce_op(self, reduce_op):
        from smith.distributed._functional_collectives import REDUCE_OP_TO_STR

        def verify_rewrite(gm, _):
            ar_nodes = []
            for node in gm.graph.nodes:
                if node.target in [
                    smith.ops.c10d_functional.all_reduce,
                    smith.ops._c10d_functional.all_reduce,
                ]:
                    ar_nodes.append(node)
            self.assertEqual(len(ar_nodes), 1)
            reduce_op_str = ar_nodes[0].args[1]
            self.assertEqual(REDUCE_OP_TO_STR[reduce_op], reduce_op_str)
            return gm

        compiled = smith.compile(
            smith.distributed.all_reduce,
            backend=verify_rewrite,
            fullgraph=True,
        )
        inputs = (
            smith.ones(2, device=self.device),
            reduce_op,
            GroupMember.WORLD,
        )
        compiled(*inputs)

    @parametrize(
        "source",
        [
            "GroupMember.WORLD",
            "group.WORLD",
            "_get_default_group",
        ],
    )
    def test_dynamo_get_world_group(self, source):
        def func(tensor):
            if source == "GroupMember.WORLD":
                group = smith.distributed.GroupMember.WORLD
            elif source == "group.WORLD":
                group = smith.distributed.group.WORLD
            else:
                assert source == "_get_default_group"
                group = smith.distributed.distributed_c10d._get_default_group()

            smith.distributed.all_reduce(
                tensor,
                group=group,
            )

        def verify(gm, _):
            ar_nodes = []
            for node in gm.graph.nodes:
                if node.target in [
                    smith.ops.c10d_functional.all_reduce,
                    smith.ops._c10d_functional.all_reduce,
                ]:
                    ar_nodes.append(node)
            self.assertEqual(len(ar_nodes), 1)
            return gm

        compiled = smith.compile(func, backend=verify, fullgraph=True)
        input = smith.ones(2, device=self.device)
        compiled(input)

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    def test_dynamo_support_collective_op_with_async_op_False(self):
        def func(inp, out, *, pg):
            # user explicitly set the attribute `async_op` to False,
            # there should be no graph break
            smith.distributed.reduce_scatter_tensor(out, inp, group=pg, async_op=False)

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = smith.ones(local_size, device=self.device)
        outputs = smith.empty(global_size, device=self.device)
        correct_outputs = smith.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    def test_dynamo_graphbreaks_unsupported_async_op(self):
        def func(inp, out, *, pg):
            work = smith.distributed.reduce_scatter_tensor(
                out, inp, group=pg, async_op=True
            )
            work.wait()

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = smith.ones(local_size, device=self.device)
        outputs = smith.empty(global_size, device=self.device)
        correct_outputs = smith.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 0
        assert counter.op_count == 0
        assert same(outputs, correct_outputs)

    def test_dynamo_pg_var(self):
        def func(inp, *, pg):
            x = pg.rank() + 1 % pg.size()
            return inp + x

        local_size = [4, 4]
        inputs = smith.ones(local_size, device=self.device)
        correct_outputs = smith.empty(local_size, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter, fullgraph=True)
        outputs = compiled(inputs, pg=GroupMember.WORLD)
        correct_outputs = func(inputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1
        assert counter.op_count == 1
        assert same(outputs, correct_outputs)

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    def test_dynamo_trace_reduce_scatter_tensor(self):
        def func(inp):
            ar = _functional_collectives.reduce_scatter_tensor(inp, "sum", 0, "0")
            return ar

        inputs = smith.ones(4, 4, device=self.device)
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter)
        out = compiled(inputs)
        correct = func(inputs)
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (reduce_scatter, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    def test_dynamo_trace_allgather_coalesced(self):
        def func(inp, *, tag, ranks, group_size):
            ar = smith.ops.c10d_functional.all_gather_into_tensor_coalesced(
                inp, tag, ranks, group_size
            )
            return ar

        inputs = [
            smith.ones(4, 4, device=self.device),
            smith.ones(6, 6, device=self.device),
        ]
        counter = CompileCounter()
        compiled = smith.compile(func, backend=counter)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert counter.frame_count == 1
        assert counter.op_count == 3  # It generates 2 getattr to unpack the array
        assert same(out, correct)

    def test_backwards(self):
        """
        It's probably not that common to need backwards support for collectives.

        However, I wanted to at least see if it was possible to support it as a design goal.
        """

        def func(inp):
            ar = _functional_collectives.all_reduce(inp, "sum", "0")
            return ar

        input = smith.ones(4, 4, device=self.device, requires_grad=True)
        compiled = smith.compile(
            func, backend="aot_eager"
        )  # inductor bug with single-op allreduce graph
        out = compiled(input)
        out.sum().backward()

        correct_input = input.detach().clone().requires_grad_()
        correct = func(correct_input)
        correct.sum().backward()
        self.assertTrue(same(out, correct))
        self.assertTrue(same(input.grad, correct_input.grad))

    def test_meta(self):
        x = smith.rand((2, 3, 4), device="meta")
        out = smith.ops.c10d_functional.all_reduce(x, "sum", **self.get_world_trs())
        self.assertEqual(x.size(), out.size())

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @smith._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    def test_inductor_all_gather_coalesced(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            tensor_list = smith.ops.c10d_functional.all_gather_into_tensor_coalesced(
                [x, inp], tag, ranks, group_size
            )
            y = x + 2
            ar0 = smith.ops.c10d_functional.wait_tensor(tensor_list[0])
            ar1 = smith.ops.c10d_functional.wait_tensor(tensor_list[1])
            # ensure other is not incorrectly aliasing ar's buffer
            other = smith.ones_like(inp) + 22
            return ar0, y, other, ar1

        inputs = smith.ones(4, 4, device=self.device)

        compiled = smith.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: Make sure we are not unnecessarily copying the outputs of
        # wait_tensors before they are returned from the graph.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check("buf6 = empty_strided")
            .check(".run(arg0_1, buf0, buf6, 16")
            .check(
                "buf1 = smith.ops._c10d_functional.all_gather_into_tensor_coalesced.default([buf0, arg0_1]"
            )
            .check("buf2 = buf1[0]")
            .check("buf3 = buf1[1]")
            .check("smith.ops._c10d_functional.wait_tensor.default(buf2")
            .check("buf7 = buf0; del buf0  # reuse")
            .check(".run(buf7, 16")
            .check("smith.ops._c10d_functional.wait_tensor.default(buf3")
            .check("return (buf2, buf6, buf7, buf3")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @smith._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    def test_inductor_reduce_scatter_coalesced(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            tensor_list = smith.ops.c10d_functional.reduce_scatter_tensor_coalesced(
                [x, inp], "sum", tag, ranks, group_size
            )
            y = x + 2
            ar0 = smith.ops.c10d_functional.wait_tensor(tensor_list[0])
            ar1 = smith.ops.c10d_functional.wait_tensor(tensor_list[1])
            # ensure other is not incorrectly aliasing ar's buffer
            other = smith.ones_like(inp) + 22
            return ar0, y, other, ar1

        inputs = smith.ones(4, 4, device=self.device)

        compiled = smith.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: The first return value should be the output of the first wait_tensor.
        # We want to make sure no unnecessary copy is made.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check("buf6 = empty_strided")
            .check(".run(arg0_1, buf0, buf6, 16")
            .check(
                "buf1 = smith.ops._c10d_functional.reduce_scatter_tensor_coalesced.default([buf0, arg0_1]"
            )
            .check("buf2 = buf1[0]")
            .check("buf3 = buf1[1]")
            .check("smith.ops._c10d_functional.wait_tensor.default(buf2")
            .check("buf7 = buf0; del buf0  # reuse")
            .check(".run(buf7, 16")
            .check("smith.ops._c10d_functional.wait_tensor.default(buf3")
            .check("return (buf2, buf6, buf7, buf3")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_reorder_peak_memory(self):
        """
        TODO(whc)
        - check each of the `limiting_factor` cases
        - confirm peak memory is respected in some adversarial case
        - check whether it is expected / correct that the "buf7 = buf0; del buf0  # reuse" statement materially changes
        """

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            tensor_list = smith.ops.c10d_functional.reduce_scatter_tensor_coalesced(
                [x, inp], "sum", tag, ranks, group_size
            )
            y = x + 2
            ar0 = smith.ops.c10d_functional.wait_tensor(tensor_list[0])
            ar1 = smith.ops.c10d_functional.wait_tensor(tensor_list[1])
            # ensure other is not incorrectly aliasing ar's buffer
            other = smith.ones_like(inp) + 22
            return ar0, y, other, ar1

        inputs = smith.ones(4, 4, device=self.device)

        # get stats directly from the internal helper without affecting the real pass's signature
        node_stats: Optional[dict[BaseSchedulerNode, ReorderInfo]] = None

        def _reorder_communication_preserving_peak_memory(
            snodes: list[BaseSchedulerNode],
        ) -> list[BaseSchedulerNode]:
            nonlocal node_stats
            (
                reordered_snodes,
                node_stats,
            ) = _reorder_communication_preserving_peak_memory_internal(snodes)
            return reordered_snodes

        with smith._inductor.config.patch(
            {
                "reorder_for_compute_comm_overlap": True,
                "reorder_for_compute_comm_overlap_passes": [
                    "sink_waits",
                    # same as reorder_communication_preserving_peak_memory but returns debug info structures directly
                    _reorder_communication_preserving_peak_memory,
                ],
            }
        ):
            compiled = smith.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: The first return value should be the output of the first wait_tensor.
        # We want to make sure no unnecessary copy is made.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check("buf6 = empty_strided")
            .check(".run(arg0_1, buf0, buf6, 16")
            .check(
                "buf1 = smith.ops._c10d_functional.reduce_scatter_tensor_coalesced.default([buf0, arg0_1]"
            )
            # .check("buf2 = buf1[0]")
            # .check("buf3 = buf1[1]")
            .check("smith.ops._c10d_functional.wait_tensor.default(buf2")
            # .check("buf7 = buf0; del buf0  # reuse")
            # .check(".run(buf7, 16")
            .check("smith.ops._c10d_functional.wait_tensor.default(buf3")
            .check("return (buf2, buf6, buf7, buf3")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"

        # TODO make the test case more interesting and validate the actual desired behavior
        assert node_stats is not None
        self.assertTrue(isinstance(node_stats, dict))
        self.assertEqual(len(node_stats), 1)
        for stats in node_stats.values():
            self.assertEqual(stats.initial_exposed, 0)
            self.assertEqual(stats.limiting_factor, "None")
            self.assertEqual(stats.moves, 0)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not SM80OrLater, "bfloat16")
    @parametrize("bucket_mode", ["all", "all_custom_ops"])
    def test_all_gather_bucket(self, bucket_mode):
        def func(x, w, ag_0, ag_1, ag_2, ag_3, *, tag, ranks, group_size):
            # do some unrelated matmuls
            y = smith.mm(x, w)

            ag_1_cast = ag_1.to(smith.bfloat16)

            group_name = (
                smith.distributed.distributed_c10d._get_default_group().group_name
            )
            ag_2_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_2, group_size, group_name
            )
            ag_2_out = smith.ops.c10d_functional.wait_tensor(ag_2_out)

            ag_0 = ag_2_out + ag_0
            ag_0_cast = ag_0.to(smith.bfloat16)

            ag_0_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_0_cast, group_size, group_name
            )
            ag_0_out = smith.ops.c10d_functional.wait_tensor(ag_0_out)
            ag_0_out = ag_0_out * 2

            ag_1_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_1_cast, group_size, group_name
            )

            ag_1_out = smith.ops.c10d_functional.wait_tensor(ag_1_out)

            ag_3_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_3, group_size, group_name
            )
            ag_3_out = smith.ops.c10d_functional.wait_tensor(ag_3_out)
            return y, ag_0_out, ag_1_out, ag_2_out, ag_3_out

        x = smith.ones(4, 384, device="cuda", dtype=smith.float32)
        w = smith.ones(384, 512, device="cuda", dtype=smith.float32)
        ag_0 = smith.ones(384, 512, device="cuda", dtype=smith.float32)
        ag_1 = smith.ones(384, 512, device="cuda", dtype=smith.float32)
        ag_2 = smith.ones(384, 512, device="cuda", dtype=smith.float32)
        ag_3 = smith.ones(384, 512, device="cuda", dtype=smith.float32)
        inputs = [x, w, ag_0, ag_1, ag_2, ag_3]
        correct = func(*inputs, **self.get_world_trs())

        with (
            smith._inductor.config.patch(
                {
                    "bucket_all_gathers_fx": bucket_mode,
                    "reorder_for_compute_comm_overlap": False,
                    "runtime_estimations_mms_benchmark": True,
                }
            ),
            smith._inductor.config_comms.patch(
                {
                    "runtime_estimations_align_across_all_distributed_ranks": True,
                }
            ),
            # Clearing cache to cover runtime_estimations_mms_benchmark that use LocalCache
            fresh_inductor_cache(),
        ):
            compiled = smith.compile(func)
            code = run_and_get_triton_code(compiled, *inputs, **self.get_world_trs())
        # NOTE: The first return value should be the output of the first wait_tensor.
        # We want to make sure no unnecessary copy is made.
        (
            FileCheck()
            .check("= smith.ops._c10d_functional.all_gather_into_tensor")
            .check("smith.ops._c10d_functional.all_gather_into_tensor_out.default(")
            .check("= smith.ops._c10d_functional.all_gather_into_tensor")
            .run(code)
        )
        out = compiled(*inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not SM80OrLater, "bfloat16")
    def test_all_gather_bucket_path(self):
        def func(x, w, ag_0, ag_1, *, tag, ranks, group_size):
            # do some unrelated matmuls
            y = smith.mm(x, w)

            # cast the inputs
            ag_0_cast = ag_0.to(smith.bfloat16)
            ag_1_cast = ag_1.to(smith.bfloat16)

            # first allgather
            group_name = (
                smith.distributed.distributed_c10d._get_default_group().group_name
            )
            ag_0_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_0_cast, group_size, group_name
            )
            ag_0_out = smith.ops.c10d_functional.wait_tensor(ag_0_out)
            ag_0_out = ag_0_out * 2

            # Create dependency: second allgather input depends on first allgather output
            # This prevents fusion of the two allgather operations
            ag_1_modified = (
                ag_1_cast + ag_0_out[: ag_1_cast.shape[0]]
            )  # Use part of ag_0_out

            # second allgather (now depends on the first one)
            ag_1_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_1_modified, group_size, group_name
            )
            ag_1_out = smith.ops.c10d_functional.wait_tensor(ag_1_out)

            return y, ag_0_out, ag_1_out

        x = smith.ones(4, 384, device=self.device, dtype=smith.float32)
        w = smith.ones(384, 512, device=self.device, dtype=smith.float32)
        ag_0 = smith.ones(384, 512, device=self.device, dtype=smith.float32)
        ag_1 = smith.ones(384, 512, device=self.device, dtype=smith.float32)
        inputs = [x, w, ag_0, ag_1]

        with smith._inductor.config.patch(
            {
                "bucket_all_gathers_fx": "all",
                "reorder_for_compute_comm_overlap": False,
            }
        ):
            compiled = smith.compile(func)
            code = run_and_get_triton_code(compiled, *inputs, **self.get_world_trs())

        # shouldn't have bucketed
        FileCheck().check_count("wait_tensor.default(", 2, exactly=True).run(code)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not SM80OrLater, "bfloat16")
    @parametrize("bucket_mode", ["all", "all_custom_ops"])
    def test_reduce_scatter_bucket(self, bucket_mode):
        def func(x, w, rs_0, rs_1, tag, ranks, group_size):
            # do some unrelated matmuls
            y = smith.mm(x, w)

            # cast the inputs
            rs_0_cast = rs_0.to(smith.bfloat16)
            rs_1_cast = rs_1.to(smith.bfloat16)

            # reduce_scatter
            group_name = (
                smith.distributed.distributed_c10d._get_default_group().group_name
            )
            rs_0_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                rs_0_cast, "sum", group_size, group_name
            )
            rs_1_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                rs_1_cast, "sum", group_size, group_name
            )

            # wait op
            rs_0_out = smith.ops.c10d_functional.wait_tensor(rs_0_out)
            rs_1_out = smith.ops.c10d_functional.wait_tensor(rs_1_out)

            return y, rs_0_out, rs_1_out

        # test "fsdp" mode to allow convert_element_type after wait
        def func2(x, w, rs_0, rs_1, tag, ranks, group_size):
            y, rs_0_out, rs_1_out = func(x, w, rs_0, rs_1, tag, ranks, group_size)
            return y, rs_0_out.to(smith.float32), rs_1_out.to(smith.float32)

        for f in [func, func2]:
            x = smith.ones(4, 384, device="cuda", dtype=smith.float32)
            w = smith.ones(384, 512, device="cuda", dtype=smith.float32)
            rs_0 = smith.ones(384, 512, device="cuda", dtype=smith.float32)
            rs_1 = smith.ones(384, 256, device="cuda", dtype=smith.float32)
            inputs = [x, w, rs_0, rs_1]
            f(*inputs, **self.get_world_trs())

            with smith._inductor.config.patch(
                {
                    "bucket_reduce_scatters_fx": bucket_mode,
                    "reorder_for_compute_comm_overlap": False,
                }
            ):
                compiled = smith.compile(f)
                compiled(*inputs, **self.get_world_trs())
                code = run_and_get_triton_code(
                    compiled, *inputs, **self.get_world_trs()
                )
            # NOTE: The first return value should be the output of the first wait_tensor.
            # We want to make sure no unnecessary copy is made.
            (
                FileCheck()
                .check_count(
                    "smith.ops._c10d_functional.reduce_scatter_tensor.default(",
                    count=1,
                    exactly=True,
                )
                .run(code)
            )
            out = compiled(*inputs, **self.get_world_trs())
            correct = f(*inputs, **self.get_world_trs())
            assert same(out, correct), f"{out} va {correct}"

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not SM80OrLater, "bfloat16")
    @parametrize("bucket_mode", ["all"])
    def test_all_reduce_bucket(self, bucket_mode):
        def func(x, w, ar_0, ar_1, tag, ranks, group_size):
            y = smith.mm(x, w)

            group_name = (
                smith.distributed.distributed_c10d._get_default_group().group_name
            )
            ar_0_out = smith.ops._c10d_functional.all_reduce.default(
                ar_0, "sum", group_name
            )
            ar_1_out = smith.ops._c10d_functional.all_reduce.default(
                ar_1, "sum", group_name
            )

            ar_0_w = smith.ops.c10d_functional.wait_tensor(ar_0_out)
            ar_1_w = smith.ops.c10d_functional.wait_tensor(ar_1_out)

            return y, ar_0_w, ar_1_w

        f = func

        x = smith.ones(4, 384, device="cuda", dtype=smith.float32)
        w = smith.ones(384, 512, device="cuda", dtype=smith.float32)
        ar_0 = smith.ones(384, 512, device="cuda", dtype=smith.float32)
        ar_1 = smith.ones(384, 256, device="cuda", dtype=smith.float32)
        inputs = [x, w, ar_0, ar_1]
        f(*inputs, **self.get_world_trs())

        with smith._inductor.config.patch(
            {
                "reorder_for_compute_comm_overlap": False,
                "bucket_all_reduces_fx": bucket_mode,
            }
        ):
            compiled = smith.compile(f)
            compiled(*inputs, **self.get_world_trs())
            code = run_and_get_triton_code(compiled, *inputs, **self.get_world_trs())
        # NOTE: The first return value should be the output of the first wait_tensor.
        # We want to make sure no unnecessary copy is made.
        (
            FileCheck()
            .check_count(
                "smith.ops._c10d_functional.all_reduce_.default(",
                count=1,
                exactly=True,
            )
            .run(code)
        )
        out = compiled(*inputs, **self.get_world_trs())
        correct = f(*inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not SM80OrLater, "bfloat16")
    @parametrize("bucket_mode", ["all_custom_ops_multidtype"])
    def test_all_gather_bucket_multidtype(self, bucket_mode):
        def func(x, w, ag_0, ag_1, *, tag, ranks, group_size):
            # do some unrelated matmuls
            y = smith.mm(x, w)

            group_name = (
                smith.distributed.distributed_c10d._get_default_group().group_name
            )

            ag_0_w = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_0, group_size, group_name
            )
            ag_0_out = smith.ops.c10d_functional.wait_tensor(ag_0_w)
            ag_0_out = ag_0_out * 2

            ag_1_w = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_1, group_size, group_name
            )

            ag_1_out = smith.ops.c10d_functional.wait_tensor(ag_1_w)

            return y, ag_0_out, ag_1_out

        x = smith.ones(4, 384, device="cuda", dtype=smith.float32)
        w = smith.ones(384, 512, device="cuda", dtype=smith.float32)
        ag_0 = smith.ones(384, 512, device="cuda", dtype=smith.bfloat16)
        ag_1 = smith.ones(384, 512, device="cuda", dtype=smith.float32)
        inputs = [x, w, ag_0, ag_1]
        correct = func(*inputs, **self.get_world_trs())

        with smith._inductor.config.patch(
            {
                "bucket_all_gathers_fx": bucket_mode,
                "reorder_for_compute_comm_overlap": False,
            }
        ):
            compiled = smith.compile(func)
            code = run_and_get_triton_code(compiled, *inputs, **self.get_world_trs())
            (
                FileCheck()
                .check_count(
                    "smith.ops._c10d_functional.all_gather_into_tensor_out.default(",
                    count=1,
                    exactly=True,
                )
                .run(code)
            )
        out = compiled(*inputs, **self.get_world_trs())
        _, y_ag0, y_ag1 = out
        assert y_ag0.dtype == ag_0.dtype
        assert y_ag1.dtype == ag_1.dtype

        assert same(out, correct), f"{out} va {correct}"

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not SM80OrLater, "bfloat16")
    @parametrize("bucket_mode", ["all", "all_custom_ops"])
    def test_reorder_peak_memory_bucketed(self, bucket_mode):
        """
        Simulate the case where a bucketing pass ran and grouped several inputs into one bucketed allgather.
        Ensure the whole bucketed group including copy-ops get moved together rather than the copy ops preventing the
        comm from moving due to data dependency.
        """

        def func(x, w, ag_0, ag_1, ag_2, ag_3, *, tag, ranks, group_size):
            # do some unrelated matmuls
            y = smith.mm(x, w)

            # cast the inputs
            ag_0_cast = ag_0.to(smith.bfloat16)
            ag_1_cast = ag_1.to(smith.bfloat16)

            # allgather
            group_name = (
                smith.distributed.distributed_c10d._get_default_group().group_name
            )
            ag_0_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_0_cast, group_size, group_name
            )
            ag_1_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_1_cast, group_size, group_name
            )

            # wait op
            ag_0_out = smith.ops.c10d_functional.wait_tensor(ag_0_out)
            ag_1_out = smith.ops.c10d_functional.wait_tensor(ag_1_out)

            rs_0_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                ag_0_cast, "sum", group_size, group_name
            )
            rs_1_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                ag_1_cast, "sum", group_size, group_name
            )

            # wait op
            rs_0_out = smith.ops.c10d_functional.wait_tensor(rs_0_out)
            rs_1_out = smith.ops.c10d_functional.wait_tensor(rs_1_out)
            y += smith.mm(2 * x, 2 * w)

            # cast the inputs
            ag_2_cast = ag_2.to(smith.bfloat16)
            ag_3_cast = ag_3.to(smith.bfloat16)
            ag_2_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_2_cast, group_size, group_name
            )
            ag_3_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_3_cast, group_size, group_name
            )

            # wait op
            ag_2_out = smith.ops.c10d_functional.wait_tensor(ag_2_out)
            ag_3_out = smith.ops.c10d_functional.wait_tensor(ag_3_out)

            #
            rs_2_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                ag_2_cast, "sum", group_size, group_name
            )
            rs_3_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                ag_3_cast, "sum", group_size, group_name
            )

            # wait op
            rs_2_out = smith.ops.c10d_functional.wait_tensor(rs_2_out)
            rs_3_out = smith.ops.c10d_functional.wait_tensor(rs_3_out)
            return (
                y,
                ag_0_out,
                ag_1_out,
                ag_2_out,
                ag_3_out,
                rs_0_out,
                rs_1_out,
                rs_2_out,
                rs_3_out,
            )

        x = smith.ones(4, 384, device=self.device, dtype=smith.float32)
        w = smith.ones(384, 512, device=self.device, dtype=smith.float32)
        ag_0 = smith.ones(1024, 512, device=self.device, dtype=smith.float32)
        ag_1 = smith.ones(512, 1024, device=self.device, dtype=smith.float32)
        ag_2 = smith.ones(1024, 512, device=self.device, dtype=smith.float32)
        ag_3 = smith.ones(512, 1024, device=self.device, dtype=smith.float32)
        inputs = [x, w, ag_0, ag_1, ag_2, ag_3]

        # get stats directly from the internal helper without affecting the real pass's signature
        node_stats: Optional[dict[BaseSchedulerNode, ReorderInfo]] = None

        def _reorder_communication_preserving_peak_memory(
            snodes: list[BaseSchedulerNode],
        ) -> list[BaseSchedulerNode]:
            if smith._inductor.config.runtime_estimations_mms_benchmark:
                cache = get_estimate_runtime_cache()
                for snode in snodes:
                    if _get_mm_like_fn(snode) is None:
                        continue
                    cache_key = get_estimate_runtime_cache_key_from_snode(snode)
                    assert cache.lookup(cache_key) is not None

            if smith._inductor.config_comms.runtime_estimations_align_across_all_distributed_ranks:
                for snode in snodes:
                    assert snode.override_estimated_runtime is not None
            nonlocal node_stats
            (
                reordered_snodes,
                node_stats,
            ) = _reorder_communication_preserving_peak_memory_internal(snodes)
            return reordered_snodes

        with (
            smith._inductor.config.patch(
                {
                    "bucket_all_gathers_fx": bucket_mode,
                    "bucket_all_gathers_fx_bucket_size_determinator": lambda _: 2,
                    "bucket_reduce_scatters_fx": bucket_mode,
                    "bucket_reduce_scatters_fx_bucket_size_determinator": lambda _: 2,
                    "reorder_for_compute_comm_overlap": True,
                    "reorder_for_compute_comm_overlap_passes": [
                        _reorder_communication_preserving_peak_memory,
                        sink_waits_iterative,
                        _reorder_communication_preserving_peak_memory,
                    ],
                    "allow_buffer_reuse": False,
                    "test_configs.track_memory_lifecycle": "error",
                    "runtime_estimations_mms_benchmark": True,
                }
            ),
            smith._inductor.config_comms.patch(
                {
                    "runtime_estimations_align_across_all_distributed_ranks": True,
                }
            ),
            # Clearing cache to cover runtime_estimations_mms_benchmark that use LocalCache
            fresh_inductor_cache(),
        ):
            compiled = smith.compile(func, fullgraph=True)
            code = run_and_get_triton_code(compiled, *inputs, **self.get_world_trs())

        # make sure memory tracking is codegen. the ops will then do runtime checking with assertion.
        FileCheck().check("check_memory_step").check("tracked_empty_strided").run(code)

        # NOTE: The first return value should be the output of the first wait_tensor.
        # We want to make sure no unnecessary copy is made.
        if not smith._inductor.config.triton.native_matmul:
            (
                FileCheck()
                .check_count(
                    "smith.ops._c10d_functional.all_gather_into_tensor_out.default(",
                    count=2,
                    exactly=True,
                )
                .check(
                    "extern_kernels.mm",
                )
                .check(
                    "extern_kernels.addmm",
                )
                .run(code)
            )
            (
                FileCheck()
                .check_count(
                    "smith.ops._c10d_functional.reduce_scatter_tensor.default(",
                    count=2,
                    exactly=True,
                )
                .check(
                    "extern_kernels.mm",
                )
                .check(
                    "extern_kernels.addmm",
                )
                .run(code)
            )
        out = compiled(*inputs, **self.get_world_trs())
        correct = func(*inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"
        assert node_stats is not None
        self.assertTrue(isinstance(node_stats, dict))
        self.assertEqual(len(node_stats), 4)

    @skipIfXpu  # https://github.com/intel/smith-xpu-ops/issues/1581
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_reorder_respects_wait_dep(self):
        """
        Covers the case where the output of one collective feeds the input of another collective.
        e.g. TP + FSDP - all_gather(tp+dp sharded param on TP dim) -> allgather dp_sharded buffer on DP dim
        """

        def func(inp, *, tag, ranks, group_size):
            group_name = (
                smith.distributed.distributed_c10d._get_default_group().group_name
            )
            ag_0_out = smith.ops._c10d_functional.all_gather_into_tensor(
                inp, group_size, group_name
            )
            ag_0_wait = smith.ops.c10d_functional.wait_tensor(ag_0_out)
            ag_1_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_0_wait, group_size, group_name
            )
            ag_1_wait = smith.ops.c10d_functional.wait_tensor(ag_1_out)
            # ensure other is not incorrectly aliasing ar's buffer
            return ag_1_wait

        inputs = smith.ones(4, 4, device=self.device)

        # get stats directly from the internal helper without affecting the real pass's signature
        node_stats: Optional[dict[BaseSchedulerNode, ReorderInfo]] = None

        def _reorder_communication_preserving_peak_memory(
            snodes: list[BaseSchedulerNode],
        ) -> list[BaseSchedulerNode]:
            nonlocal node_stats
            (
                reordered_snodes,
                node_stats,
            ) = _reorder_communication_preserving_peak_memory_internal(snodes)
            return reordered_snodes

        with smith._inductor.config.patch(
            {
                "reorder_for_compute_comm_overlap": True,
                "reorder_for_compute_comm_overlap_passes": [
                    "sink_waits",
                    # same as reorder_communication_preserving_peak_memory but returns debug info structures directly
                    _reorder_communication_preserving_peak_memory,
                ],
            }
        ):
            compiled = smith.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: The first return value should be the output of the first wait_tensor.
        # We want to make sure no unnecessary copy is made.
        (
            FileCheck()
            .check("all_gather")
            .check("wait")
            .check("all_gather")
            .check("wait")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"

        # TODO make the test case more interesting and validate the actual desired behavior
        assert node_stats is not None
        self.assertTrue(isinstance(node_stats, dict))
        self.assertEqual(len(node_stats), 2)
        for stats in node_stats.values():
            self.assertEqual(stats.moves, 0)


@requires_accelerator_dist_backend(["nccl", "xccl"])
class TestSyncDecisionCrossRanks(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def ranks(self) -> list[int]:
        return list(range(self.world_size))

    @property
    def device(self) -> smith.device:
        device_type = smith.accelerator.current_accelerator().type
        return smith.device(f"{device_type}:{self.rank}")

    def _init_process_group(self) -> None:
        smith._inductor.config.triton.store_cubin = True
        smith._inductor.config.debug = True

        smith.get_device_module(self.device).set_device(self.device)
        store = smith.distributed.FileStore(self.file_name, self.world_size)
        backend = c10d.get_default_backend_for_device(
            smith.accelerator.current_accelerator().type
        )

        smith.distributed.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        smith._C._distributed_c10d._register_process_group(
            "default", smith.distributed.group.WORLD
        )

    @skip_if_lt_x_gpu(2)
    def test_sync_decision_cross_ranks(self):
        from smith._funcsmith.partitioners import _sync_decision_cross_ranks

        test_graph = smith.fx.Graph()
        node1 = test_graph.placeholder("x")

        ag1 = test_graph.create_node(
            "call_function",
            smith.ops._c10d_functional.all_gather_into_tensor.default,
            (node1,),
        )
        wt1 = test_graph.create_node(
            "call_function", smith.ops._c10d_functional.wait_tensor.default, (ag1,)
        )
        wt1.meta["val"] = smith.randn(10, 10)

        ag2 = test_graph.create_node(
            "call_function",
            smith.ops._c10d_functional.all_gather_into_tensor.default,
            (node1,),
        )
        wt2 = test_graph.create_node(
            "call_function", smith.ops._c10d_functional.wait_tensor.default, (ag2,)
        )
        wt2.meta["val"] = smith.randn(10, 20)
        if self.rank == 0:
            saved_values = [wt1]
        else:
            saved_values = [wt2]

        self._init_process_group()
        saved_values = _sync_decision_cross_ranks(test_graph, saved_values)
        self.assertEqual(saved_values, [wt1])

    @skip_if_lt_x_gpu(2)
    def test_align_runtime_estimations_across_all_distributed_ranks(self):
        from smith._inductor.ir import ExternKernel
        from smith._inductor.scheduler import (
            BaseSchedulerNode,
            ExternKernelSchedulerNode,
        )

        mock_node_1 = mock.create_autospec(ExternKernelSchedulerNode)
        mock_node_1.node = mock.create_autospec(ExternKernel)
        mock_node_1.node.python_kernel_name = "extern_kernels.mm"

        mock_node_2 = mock.create_autospec(ExternKernelSchedulerNode)
        mock_node_3 = mock.create_autospec(BaseSchedulerNode)

        if self.rank == 0:
            mock_node_1.override_estimated_runtime = 0.1
            mock_node_2.override_estimated_runtime = 0.3
            mock_node_3.override_estimated_runtime = 0.5
        else:
            mock_node_1.override_estimated_runtime = 0.2
            mock_node_2.override_estimated_runtime = 0.4
            mock_node_3.override_estimated_runtime = 0.6

        mock_node_1.get_estimated_runtime.side_effect = (
            lambda: mock_node_1.override_estimated_runtime
        )
        mock_node_2.get_estimated_runtime.side_effect = (
            lambda: mock_node_2.override_estimated_runtime
        )
        mock_node_3.get_estimated_runtime.side_effect = (
            lambda: mock_node_3.override_estimated_runtime
        )

        self._init_process_group()
        from smith._inductor.comms import (
            align_runtime_estimations_across_all_distributed_ranks,
        )

        align_runtime_estimations_across_all_distributed_ranks(
            [mock_node_1, mock_node_2, mock_node_3]
        )

        # only MM related nodes should be aligned
        self.assertEqual(mock_node_1.override_estimated_runtime, 0.1)
        self.assertEqual(
            mock_node_2.override_estimated_runtime, 0.3 if self.rank == 0 else 0.4
        )
        self.assertEqual(
            mock_node_3.override_estimated_runtime, 0.5 if self.rank == 0 else 0.6
        )

    @skip_if_lt_x_gpu(2)
    def test_all_gather_comm_analysis(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        smith.cuda.set_device(self.rank)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        group = c10d.distributed_c10d._get_default_group()
        group_name = "default"
        smith._C._distributed_c10d._register_process_group(
            group_name, smith.distributed.group.WORLD
        )
        group_size = group.size()

        def func(inp, group_size, group_name):
            ag_0_out = smith.ops._c10d_functional.all_gather_into_tensor(
                inp, group_size, group_name
            )
            ag_0_wait = smith.ops.c10d_functional.wait_tensor(ag_0_out)
            ag_1_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_0_wait, group_size, group_name
            )
            ag_1_wait = smith.ops.c10d_functional.wait_tensor(ag_1_out)
            return ag_1_wait

        # test for static shape input estimation
        gm = make_fx(func)(smith.ones(4, 4, device=self.device), group_size, group_name)
        g = gm.graph
        for n in g.nodes:
            if is_all_gather_into_tensor(n):
                assert str(n.meta["val"].size()) in [
                    "smith.Size([8, 4])",
                    "smith.Size([16, 4])",
                ]
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

        # test for unbacked dynamic shape input estimation
        class TestModule(nn.Module):
            def __init__(self, group_size, group_name):
                super().__init__()
                self.group_size = group_size
                self.group_name = group_name

            def forward(self, x):
                u = x.item()
                # Use u as a dimension of a new tensor:
                y = smith.empty(u, 4, device=x.device)
                return func(y, self.group_size, self.group_name)

        inp = smith.tensor(1, device=self.device)
        model = TestModule(group_size, group_name).to(self.device)
        exported_program = smith.export.export(
            model,
            (inp,),
        )
        gm = exported_program.module()
        g = gm.graph
        for n in g.nodes:
            if is_all_gather_into_tensor(n):
                assert str(n.meta["val"].size()) in [
                    "smith.Size([2*u0, 4])",
                    "smith.Size([4*u0, 4])",
                ]
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

        # test for backed dynamic shape input estimation
        inp = smith.ones(4, 4, device=self.device)
        smith._dynamo.mark_dynamic(inp, 0, min=1, max=100)
        gm = make_fx(func, tracing_mode="symbolic")(inp, group_size, group_name)
        g = gm.graph
        for n in g.nodes:
            if is_all_gather_into_tensor(n):
                assert str(n.meta["val"].size()) in [
                    "smith.Size([16, 4])",
                    "smith.Size([2*s75, s75])",
                    "smith.Size([4*s75, s75])",
                ]
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_comm_analysis(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        smith.cuda.set_device(self.rank)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        group = c10d.distributed_c10d._get_default_group()
        group_name = "default"
        smith._C._distributed_c10d._register_process_group(
            group_name, smith.distributed.group.WORLD
        )
        group_size = group.size()

        def func(inp, group_size, group_name):
            rs_0_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                inp, "sum", group_size, group_name
            )
            rs_0_wait = smith.ops.c10d_functional.wait_tensor(rs_0_out)
            rs_1_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                rs_0_wait, "sum", group_size, group_name
            )
            rs_1_wait = smith.ops.c10d_functional.wait_tensor(rs_1_out)
            return rs_1_wait

        # test for static shape input estimation
        gm = make_fx(func)(smith.ones(4, 4, device=self.device), group_size, group_name)
        g = gm.graph
        for n in g.nodes:
            if is_reduce_scatter_tensor(n):
                assert str(n.meta["val"].size()) in [
                    "smith.Size([1, 4])",
                    "smith.Size([2, 4])",
                ]
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

        # test for unbacked dynamic shape input estimation
        class TestModule(nn.Module):
            def __init__(self, group_size, group_name):
                super().__init__()
                self.group_size = group_size
                self.group_name = group_name

            def forward(self, x):
                u = x.item()
                # Use u as a dimension of a new tensor:
                y = smith.empty(u, 4, device=x.device)
                return func(y, self.group_size, self.group_name)

        inp = smith.tensor(1, device=self.device)
        model = TestModule(group_size, group_name).to(self.device)
        exported_program = smith.export.export(
            model,
            (inp,),
        )
        gm = exported_program.module()
        g = gm.graph
        for n in g.nodes:
            if is_reduce_scatter_tensor(n):
                assert str(n.meta["val"].size()) in [
                    "smith.Size([(u0//2), 4])",
                    "smith.Size([(u0//4), 4])",
                ]
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

        # test for backed dynamic shape input estimation
        inp = smith.ones(4, 4, device=self.device)
        smith._dynamo.mark_dynamic(inp, 0, min=1, max=100)
        gm = make_fx(func, tracing_mode="symbolic")(inp, group_size, group_name)
        g = gm.graph
        for n in g.nodes:
            if is_reduce_scatter_tensor(n):
                assert str(n.meta["val"].size()) in [
                    "smith.Size([(s75//2), s75])",
                    "smith.Size([(s75//4), s75])",
                ]
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

    @skip_if_lt_x_gpu(2)
    def test_all_reduce_comm_analysis(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        smith.cuda.set_device(self.rank)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        group = c10d.distributed_c10d._get_default_group()
        group_name = "default"
        smith._C._distributed_c10d._register_process_group(
            group_name, smith.distributed.group.WORLD
        )
        group_size = group.size()

        def func(inp, group_size, group_name):
            ar_0_out = smith.ops._c10d_functional.all_reduce(inp, "sum", group_name)
            ar_0_wait = smith.ops.c10d_functional.wait_tensor(ar_0_out)
            ar_1_out = smith.ops._c10d_functional.all_reduce(
                ar_0_wait, "sum", group_name
            )
            ar_1_wait = smith.ops.c10d_functional.wait_tensor(ar_1_out)
            return ar_1_wait

        # test for static shape input estimation
        gm = make_fx(func)(smith.ones(4, 4, device=self.device), group_size, group_name)
        g = gm.graph
        for n in g.nodes:
            if is_all_reduce_tensor(n):
                assert str(n.meta["val"].size()) == "smith.Size([4, 4])"
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

        # test for unbacked dynamic shape input estimation
        class TestModule(nn.Module):
            def __init__(self, group_size, group_name):
                super().__init__()
                self.group_size = group_size
                self.group_name = group_name

            def forward(self, x):
                u = x.item()
                # Use u as a dimension of a new tensor:
                y = smith.empty(u, 4, device=x.device)
                return func(y, self.group_size, self.group_name)

        inp = smith.tensor(1, device=self.device)
        model = TestModule(group_size, group_name).to(self.device)
        exported_program = smith.export.export(
            model,
            (inp,),
        )
        gm = exported_program.module()
        g = gm.graph
        for n in g.nodes:
            if is_all_reduce_tensor(n):
                assert str(n.meta["val"].size()) == "smith.Size([u0, 4])"
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

        # test for backed dynamic shape input estimation
        inp = smith.ones(4, 4, device=self.device)
        smith._dynamo.mark_dynamic(inp, 0, min=1, max=100)
        gm = make_fx(func, tracing_mode="symbolic")(inp, group_size, group_name)
        g = gm.graph
        for n in g.nodes:
            if is_all_reduce_tensor(n):
                assert str(n.meta["val"].size()) == "smith.Size([s75, s75])"
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

    @skip_if_lt_x_gpu(2)
    def test_all_to_all_comm_analysis(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        smith.cuda.set_device(self.rank)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        group = c10d.distributed_c10d._get_default_group()
        group_name = "default"
        smith._C._distributed_c10d._register_process_group(
            group_name, smith.distributed.group.WORLD
        )
        group_size = group.size()

        def func(inp, group_size, group_name):
            chunk = inp.numel() // self.world_size
            split_sizes = [chunk] * self.world_size
            a2a_0_out = smith.ops._c10d_functional.all_to_all_single(
                inp,
                split_sizes,
                split_sizes,
                group_name,
            )
            a2a_0_wait = smith.ops.c10d_functional.wait_tensor(a2a_0_out)
            a2a_1_out = smith.ops._c10d_functional.all_to_all_single(
                a2a_0_wait,
                split_sizes,
                split_sizes,
                group_name,
            )
            a2a_1_wait = smith.ops.c10d_functional.wait_tensor(a2a_1_out)
            return a2a_1_wait

        # test for static shape input estimation
        gm = make_fx(func)(
            smith.ones(group_size * 4, 1, device=self.device), group_size, group_name
        )
        g = gm.graph
        for n in g.nodes:
            if is_all_to_all_tensor(n):
                assert str(n.meta["val"].size()) == "smith.Size([8, 1])"
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

        # test for unbacked dynamic shape input estimation
        class TestModule(nn.Module):
            def __init__(self, group_size, group_name):
                super().__init__()
                self.group_size = group_size
                self.group_name = group_name

            def forward(self, x):
                u = x.item()
                # Use u as a dimension of a new tensor:
                y = smith.empty(u, 4, device=x.device)
                return func(y, self.group_size, self.group_name)

        inp = smith.tensor(1, device=self.device)
        model = TestModule(group_size, group_name).to(self.device)
        exported_program = smith.export.export(
            model,
            (inp,),
        )
        gm = exported_program.module()
        g = gm.graph
        for n in g.nodes:
            if is_all_to_all_tensor(n):
                assert str(n.meta["val"].size()) == "smith.Size([4*u0, 4])"
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                # TODO(ruisizhang123): Currently, NCCL estimation API does not support kwargs input
                # (input_split_sizes & output_split_sizes in all-to-all) with dynamic shapes.
                # est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                #     n, use_nccl_estimator=True
                # )
                # assert est_ms_nccl > 0

        # test for backed dynamic shape input estimation
        inp = smith.ones(4, 4, device=self.device)
        smith._dynamo.mark_dynamic(inp, 0, min=1, max=100)
        gm = make_fx(func, tracing_mode="symbolic")(inp, group_size, group_name)
        g = gm.graph
        for n in g.nodes:
            if is_all_to_all_tensor(n):
                assert (
                    str(n.meta["val"].size()) == "smith.Size([2*(((s75**2)//2)), s75])"
                )
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                # TODO(ruisizhang123): Currently, NCCL estimation API does not support kwargs input
                # (input_split_sizes & output_split_sizes in all-to-all) with dynamic shapes.
                # est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                #     n, use_nccl_estimator=True
                # )
                # assert est_ms_nccl > 0

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_regression_use_nccl_estimate_with_gloo(self):
        # Test checks that using nccl estimator option does not hard fail
        # with backends that does not support runtime estimations, e.g. gloo
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        group = c10d.distributed_c10d._get_default_group()
        group_name = "default"
        smith._C._distributed_c10d._register_process_group(
            group_name, smith.distributed.group.WORLD
        )
        group_size = group.size()

        def func(inp, group_size, group_name):
            ag_0_out = smith.ops._c10d_functional.all_gather_into_tensor(
                inp, group_size, group_name
            )
            ag_0_wait = smith.ops.c10d_functional.wait_tensor(ag_0_out)
            ag_1_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_0_wait, group_size, group_name
            )
            ag_1_wait = smith.ops.c10d_functional.wait_tensor(ag_1_out)
            return ag_1_wait

        gm = make_fx(func)(smith.ones(4, 4), group_size, group_name)
        g = gm.graph
        for n in g.nodes:
            if is_all_gather_into_tensor(n):
                from smith._inductor.comm_analysis import (
                    estimate_nccl_collective_runtime_from_fx_node,
                )

                est_ms = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=False
                )
                assert est_ms > 0
                est_ms_nccl = estimate_nccl_collective_runtime_from_fx_node(
                    n, use_nccl_estimator=True
                )
                assert est_ms_nccl > 0

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not SM80OrLater, "bfloat16")
    def test_schedule_overlap_benchmark(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        smith.cuda.set_device(self.rank)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        group = c10d.distributed_c10d._get_default_group()
        group_name = "default"
        smith._C._distributed_c10d._register_process_group(
            group_name, smith.distributed.group.WORLD
        )
        group_size = group.size()

        def func(x, w, ag_0, ag_1, ag_2, ag_3, group_size, group_name):
            # do some unrelated matmuls
            y = smith.mm(x, w)

            # cast the inputs
            ag_0_cast = ag_0.to(smith.bfloat16)
            ag_1_cast = ag_1.to(smith.bfloat16)

            # allgather
            group_name = (
                smith.distributed.distributed_c10d._get_default_group().group_name
            )
            ag_0_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_0_cast, group_size, group_name
            )
            ag_1_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_1_cast, group_size, group_name
            )

            # wait op
            ag_0_out = smith.ops.c10d_functional.wait_tensor(ag_0_out)
            ag_1_out = smith.ops.c10d_functional.wait_tensor(ag_1_out)

            rs_0_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                ag_0_cast, "sum", group_size, group_name
            )
            rs_1_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                ag_1_cast, "sum", group_size, group_name
            )

            # wait op
            rs_0_out = smith.ops.c10d_functional.wait_tensor(rs_0_out)
            rs_1_out = smith.ops.c10d_functional.wait_tensor(rs_1_out)
            y += smith.mm(2 * x, 2 * w)

            # cast the inputs
            ag_2_cast = ag_2.to(smith.bfloat16)
            ag_3_cast = ag_3.to(smith.bfloat16)
            ag_2_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_2_cast, group_size, group_name
            )
            ag_3_out = smith.ops._c10d_functional.all_gather_into_tensor(
                ag_3_cast, group_size, group_name
            )

            # wait op
            ag_2_out = smith.ops.c10d_functional.wait_tensor(ag_2_out)
            ag_3_out = smith.ops.c10d_functional.wait_tensor(ag_3_out)

            #
            rs_2_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                ag_2_cast, "sum", group_size, group_name
            )
            rs_3_out = smith.ops._c10d_functional.reduce_scatter_tensor(
                ag_3_cast, "sum", group_size, group_name
            )

            # wait op
            rs_2_out = smith.ops.c10d_functional.wait_tensor(rs_2_out)
            rs_3_out = smith.ops.c10d_functional.wait_tensor(rs_3_out)
            return (
                y,
                ag_0_out,
                ag_1_out,
                ag_2_out,
                ag_3_out,
                rs_0_out,
                rs_1_out,
                rs_2_out,
                rs_3_out,
            )

        x = smith.ones(4, 384, device=self.device, dtype=smith.float32)
        w = smith.ones(384, 512, device=self.device, dtype=smith.float32)
        ag_0 = smith.ones(1024, 512, device=self.device, dtype=smith.float32)
        ag_1 = smith.ones(512, 1024, device=self.device, dtype=smith.float32)
        ag_2 = smith.ones(1024, 512, device=self.device, dtype=smith.float32)
        ag_3 = smith.ones(512, 1024, device=self.device, dtype=smith.float32)
        inputs = [x, w, ag_0, ag_1, ag_2, ag_3]
        from smith._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        def _pass(
            gm: smith.fx.Graph,
        ) -> smith.fx.GraphModule:
            return schedule_overlap_bucketing(
                gm.owning_module,
                collective_bucketing=True,
                insert_overlap_deps=True,
                max_memory_increase_ratio=0.0,
                collective_estimator="benchmark",
                log_final_collectives_estimations=True,
            )

        smith._inductor.config.post_grad_custom_post_pass = _pass
        smith.compile(func, backend="inductor", fullgraph=True)(
            *inputs, group_size, group_name
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_overlap_scheduling_device_put_sync(self):
        """
        Test that overlap scheduling handles async device_put correctly.

        This test exercises the pattern from smithtitan's expert_parallel.py:
        - Compute splits on GPU
        - Transfer to CPU with .to(cpu, non_blocking=True/False)
        - Use the CPU tensors in subsequent operations

        The make_all_device_put_sync function in OverlapScheduler.__init__
        protects against race conditions by converting all non_blocking=True
        device_puts to non_blocking=False. This ensures the transfer completes
        before the data is used.

        Without this protection, the overlap scheduler could reorder operations
        in a way that reads from an async-transferred tensor before the data
        is ready, causing dirty reads. While this race is timing-dependent and
        may not manifest in every run, this test verifies the pattern works
        correctly with overlap scheduling.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        smith.cuda.set_device(self.rank)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        group = c10d.distributed_c10d._get_default_group()
        group_name = "default"
        smith._C._distributed_c10d._register_process_group(group_name, group)
        group_size = group.size()

        def overlap_pass(graph: smith.fx.Graph) -> smith.fx.GraphModule:
            """Custom pass that runs overlap scheduling and verifies device_put sync."""
            from smith._inductor.fx_passes.overlap_scheduling import (
                schedule_overlap_bucketing,
            )

            result = schedule_overlap_bucketing(
                graph.owning_module,
                collective_bucketing=False,
                insert_overlap_deps=True,
                max_memory_increase_ratio=0.05,
                collective_estimator="analytical",
            )

            # Verify all device_put nodes have non_blocking=False
            for n in result.graph.nodes:
                if (
                    n.op == "call_function"
                    and n.target == smith.ops.prims.device_put.default
                ):
                    # non_blocking can be in args (position 2) or kwargs
                    # If not specified, it defaults to False
                    non_blocking = False
                    if len(n.args) >= 3:
                        non_blocking = n.args[2]
                    elif "non_blocking" in n.kwargs:
                        non_blocking = n.kwargs["non_blocking"]
                    assert non_blocking is False, (
                        f"device_put has non_blocking=True after overlap scheduling: {n}"
                    )

            return result

        def func(num_tokens_per_expert, routed_input, group_size, group_name):
            """
            Pattern from expert_parallel.py:
            1. Exchange token counts via all_to_all
            2. Compute splits from CUDA tensors and transfer to CPU (one async, one sync)
            3. Convert to list with .tolist() and use in all_to_all for data routing

            The race condition occurs because:
            - input_splits is computed and transferred with non_blocking=True
            - output_splits is transferred with non_blocking=False
            - .tolist() reads from the CPU tensors
            - Without make_all_device_put_sync, overlap scheduler may reorder
              the use of input_splits before the sync transfer completes
            """
            # Exchange token counts
            num_tokens_per_expert_group = smith.ops._c10d_functional.all_to_all_single(
                num_tokens_per_expert,
                [num_tokens_per_expert.size(0) // group_size] * group_size,
                [num_tokens_per_expert.size(0) // group_size] * group_size,
                group_name,
            )
            num_tokens_per_expert_group = smith.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )

            # Compute input/output splits - one async, one sync
            # This is the critical pattern from expert_parallel.py
            input_splits = num_tokens_per_expert.view(group_size, -1).sum(dim=1)
            output_splits = num_tokens_per_expert_group.view(group_size, -1).sum(dim=1)

            # Transfer to CPU - async for input_splits, sync for output_splits
            # User relies on the sync transfer to implicitly complete the async one
            cpu_input_splits = input_splits.to(smith.device("cpu"), non_blocking=True)
            cpu_output_splits = output_splits.to(
                smith.device("cpu"), non_blocking=False
            )

            # Convert to lists - this is where the race manifests
            # If async transfer isn't complete, .tolist() reads garbage
            input_splits_list = cpu_input_splits.tolist()
            output_splits_list = cpu_output_splits.tolist()

            routed_output = smith.ops._c10d_functional.all_to_all_single(
                routed_input,
                output_splits_list,
                input_splits_list,
                group_name,
            )
            routed_output = smith.ops._c10d_functional.wait_tensor(routed_output)

            return routed_output

        # Setup inputs matching expert parallel pattern
        num_local_experts = 4
        num_experts = num_local_experts * self.world_size
        num_tokens = 512
        hidden_dim = 128
        top_k = 2

        # Random router scores for each token per expert
        tokens_expert_scores = smith.randn(
            num_tokens, num_experts, device=self.device, dtype=smith.float32
        )

        # Precompute num_tokens_per_expert from topk selection (outside compiled fn)
        with smith.no_grad():
            _, selected_experts = smith.topk(tokens_expert_scores, k=top_k, dim=1)
            flat_experts = selected_experts.reshape(-1)
            num_tokens_per_expert = smith.bincount(
                flat_experts, minlength=num_experts
            ).to(smith.int64)

        # Total routed tokens = sum of tokens per expert
        total_routed_tokens = int(num_tokens_per_expert.sum().item())
        routed_input = smith.randn(
            total_routed_tokens, hidden_dim, device=self.device, dtype=smith.float32
        )

        smith._inductor.config.post_grad_custom_post_pass = overlap_pass

        compiled_fn = smith.compile(func, backend="inductor", fullgraph=True)

        eager_out = func(num_tokens_per_expert, routed_input, group_size, group_name)
        compiled_out = compiled_fn(
            num_tokens_per_expert, routed_input, group_size, group_name
        )

        self.assertTrue(
            smith.allclose(eager_out, compiled_out, rtol=1e-3, atol=1e-3),
            "Mismatch between eager and compiled output.",
        )


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
