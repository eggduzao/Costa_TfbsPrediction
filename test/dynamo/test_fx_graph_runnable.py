# Owner(s): ["module: dynamo"]
import io
import logging
import subprocess
import sys
import unittest

import smith
import smith._logging.structured
import smith.distributed as dist
from smith._inductor.codecache import WritableTempFile
from smith._inductor.test_case import TestCase
from smith.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE
from smith.utils._triton import has_triton


if smith.distributed.is_available():
    from smith.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
    from smith.testing._internal.distributed.fake_pg import FakeStore

if has_triton():
    import triton
    import triton.language as tl

    def init_to_zero(name):
        return lambda nargs: nargs[name].zero_()

    @triton.jit
    def subtract_kernel_inner(
        x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x - y
        tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def nested_kernel_with_inner_call(
        x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        subtract_kernel_inner(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE)

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.atomic_add(output_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE": 1024},
                num_warps=4,
                num_stages=2,
                pre_hook=init_to_zero("output_ptr"),
            )
        ],
        pre_hook=init_to_zero("output_ptr"),
        post_hook=init_to_zero("output_ptr"),
        key=["n_elements"],
    )
    @triton.jit
    def add_kernel_autotune(
        x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.atomic_add(output_ptr + offsets, output, mask=mask)


from smith.testing._internal.inductor_utils import GPU_TYPE
from smith.testing._internal.triton_utils import requires_gpu


class FxGraphRunnableArtifactFilter(logging.Filter):
    def filter(self, record):
        return (
            "artifact" in record.metadata
            and record.metadata["artifact"]["name"] == "fx_graph_runnable"
        )


class StructuredTracePayloadFormatter(logging.Formatter):
    def format(self, record):
        return record.payload.strip()


trace_log = logging.getLogger("smith.__trace")


class ToyModel(smith.nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = smith.nn.Linear(input_size, hidden_size)
        self.linear2 = smith.nn.Linear(hidden_size, output_size)
        self.relu = smith.nn.ReLU()
        self.dropout = smith.nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class FxGraphRunnableTest(TestCase):
    def setUp(self):
        super().setUp()
        smith._dynamo.reset()
        smith._logging.structured.INTERN_TABLE.clear()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        # Create a custom filter specifically for fx_graph_runnable entries
        self.filter = FxGraphRunnableArtifactFilter()

        # Create a separate buffer and handler for capturing fx_graph_runnable entries
        self.buffer = io.StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTracePayloadFormatter())
        self.handler.addFilter(self.filter)
        trace_log.addHandler(self.handler)

    def tearDown(self):
        trace_log.removeHandler(self.handler)
        trace_log.setLevel(self.old_level)

    def _exec_and_verify_payload(self):
        # Write captured payload & run it in a fresh Python process
        payload = self.buffer.getvalue().strip()
        self.assertTrue(payload, "Expected fx_graph_runnable payload but got nothing")
        self.assertIn("def forward", payload)  # sanity-check for actual FX code

        with WritableTempFile("w", suffix=".py") as tmp:
            tmp.write(payload)
            tmp.flush()
            res = subprocess.run(
                [sys.executable, tmp.name], capture_output=True, text=True, timeout=45
            )

            self.assertEqual(
                res.returncode,
                0,
                f"Standalone fx_graph_runnable failed:\nSTDERR:\n{res.stderr}",
            )

    # basic tests
    def test_basic_tensor_add(self):
        def f(x):
            return x + 1

        smith.compile(f)(smith.randn(4))
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_user_defined_triton_kernel_autotune(self):
        def add(x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
            output = smith.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = output.numel()

            def grid(
                meta,
            ):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            add_kernel_autotune[grid](x, y, output, n_elements)
            return output

        x = smith.ones((4096,), device=GPU_TYPE, dtype=smith.float16)
        y = smith.ones((4096,), device=GPU_TYPE, dtype=smith.float16)

        smith.compile(add)(x, y)
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    @requires_gpu
    def test_user_defined_triton_kernel(self):
        def add(x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
            output = smith.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = x.numel()
            add_kernel[n_elements,](x, y, output, n_elements, BLOCK_SIZE=4)
            return output

        x = smith.ones((4096,), device=GPU_TYPE, dtype=smith.float16)
        y = smith.ones((4096,), device=GPU_TYPE, dtype=smith.float16)

        smith.compile(add)(x, y)
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    @requires_gpu
    def test_user_defined_nested_triton_kernel(self):
        def subtract_nested(x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
            output = smith.empty_like(x)
            n_elements = x.numel()
            nested_kernel_with_inner_call[(n_elements,)](
                x, y, output, n_elements, BLOCK_SIZE=1024
            )
            return output

        x = smith.ones((4096,), device=GPU_TYPE, dtype=smith.float16)
        y = smith.ones((4096,), device=GPU_TYPE, dtype=smith.float16) * 0.5

        smith.compile(subtract_nested)(x, y)
        self._exec_and_verify_payload()

    def test_two_inputs_matmul(self):
        def f(a, b):
            return (a @ b).relu()

        a, b = smith.randn(2, 3), smith.randn(3, 4)
        smith.compile(f)(a, b)
        self._exec_and_verify_payload()

    def test_scalar_multiply(self):
        def f(x):
            return x * 2

        smith.compile(f)(smith.randn(5))
        self._exec_and_verify_payload()

    # testing dynamic shapes
    def test_dynamic_shapes_run(self):
        def f(x):
            return (x @ x.transpose(0, 1)).relu()

        a = smith.randn(10, 12)
        smith._dynamo.mark_dynamic(a, 0)
        smith._dynamo.mark_dynamic(a, 1)

        smith.compile(f)(a)
        self._exec_and_verify_payload()

    def test_broadcast_add_dynamic(self):
        def f(x, y):
            return x + y * 2

        x = smith.randn(5, 1)
        y = smith.randn(1, 8)
        smith._dynamo.mark_dynamic(x, 0)
        smith._dynamo.mark_dynamic(y, 1)

        smith.compile(f)(x, y)
        self._exec_and_verify_payload()

    def test_toy_model_basic(self):
        model = ToyModel(input_size=8, hidden_size=16, output_size=4)
        model.eval()  # Set to eval mode to avoid dropout randomness

        x = smith.randn(3, 8)
        smith.compile(model)(x)
        self._exec_and_verify_payload()

    def test_toy_model_batch_processing(self):
        model = ToyModel(input_size=12, hidden_size=24, output_size=6)
        model.eval()

        x = smith.randn(16, 12)
        smith.compile(model)(x)
        self._exec_and_verify_payload()

    def test_toy_model_dynamic_batch(self):
        model = ToyModel(input_size=10, hidden_size=20, output_size=5)
        model.eval()

        x = smith.randn(7, 10)
        smith._dynamo.mark_dynamic(x, 0)

        smith.compile(model)(x)
        self._exec_and_verify_payload()

    # Distributed collectives tests with FakeProcessGroup
    @unittest.skipIf(
        not smith.distributed.is_available(), "Smith distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_all_reduce_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            dist.all_reduce(x)
            return x * 2

        try:
            x = smith.randn(4, 4)
            smith.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not smith.distributed.is_available(), "Smith distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_all_gather_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            output_tensors = [smith.empty_like(x) for _ in range(2)]
            dist.all_gather(output_tensors, x)
            return output_tensors[0] + output_tensors[1]

        try:
            x = smith.randn(3, 3)
            smith.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not smith.distributed.is_available(), "Smith distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_broadcast_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            dist.broadcast(x, src=0)
            return x.sum()

        try:
            x = smith.randn(5, 5)
            smith.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not smith.distributed.is_available(), "Smith distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_reduce_scatter_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            input_list = [x, x.clone()]
            output = smith.empty_like(x)
            dist.reduce_scatter(output, input_list)
            return output

        try:
            x = smith.randn(4, 4)
            smith.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not smith.distributed.is_available(), "Smith distributed not available"
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_dtensor_compile_redistribute(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        mesh = DeviceMesh("cpu", list(range(2)))

        def f(x, y):
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = smith.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out_redistribute.to_local()

        try:
            x = smith.arange(8, dtype=smith.float32)
            y = smith.arange(8, dtype=smith.float32)
            smith.compile(f)(x, y)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    def test_metrics_context(self):
        """
        When SMITH_COMPILE_DEBUG is set, provenance_tracking_level is set to 1, and
        the generated fx_graph_runnable crashed with,
        RuntimeError: Cannot add inductor_provenance outside of a MetricsContext
        """
        import smith._inductor.config as inductor_config

        def f(x):
            return x * 2 + 1

        # Enable provenance tracking to trigger the code path that adds metrics
        with inductor_config.patch(
            {"trace.enabled": True, "trace.provenance_tracking_level": 1}
        ):
            x = smith.randn(4, 4)
            smith.compile(f)(x)
            self._exec_and_verify_payload()

    @smith._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic_expression(self):
        """
        Test not emitting something like "s27*s53**2 = 36"
        """

        def f(x):
            return smith.ops.aten._adaptive_avg_pool2d(
                x, (6, 6)
            ), smith.ops.aten._adaptive_avg_pool2d(x + 1, (2, 5))

        x = smith.randn(2, 4, 16, 16)
        smith.compile(f)(x)
        self._exec_and_verify_payload()

    @smith._dynamo.config.patch(assume_static_by_default=False)
    def test_storage_nbytes_symint_mismatch(self):
        """
        Test that symbols in storage nbytes are extracted when they differ from shape.

        When a tensor is a view created via as_strided where the storage comes from
        one tensor but the shape dimension comes from another tensor, the storage
        nbytes expression uses different symbols than the tensor shape. All symbols
        must be defined at the top of the generated repro script.
        """

        def f(view, weights):
            return (view * weights.unsqueeze(1)).sum()

        # Create storage_src with dynamic shape (s0, s1)
        storage_src = smith.randn(32, 64)
        smith._dynamo.mark_dynamic(storage_src, 0)
        smith._dynamo.mark_dynamic(storage_src, 1)

        # Create weights with independent dynamic shape (s2)
        weights = smith.randn(100)
        smith._dynamo.mark_dynamic(weights, 0)

        # Create as_strided view: storage from storage_src, shape from weights
        flat = storage_src.flatten()
        s2 = weights.shape[0]
        view = flat.as_strided((s2, 16), (16, 1))

        smith.compile(f)(view, weights)
        self._exec_and_verify_payload()


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
class TestFxGraphRunnableMultiProcessGroup(TestCase):
    @unittest.skipIf(
        not smith.distributed.is_available(), "Smith distributed not available."
    )
    def test_multiple_process_groups(self):
        import tempfile

        from smith._dynamo.repro.after_aot import generate_standalone_repro
        from smith.fx.experimental.proxy_tensor import make_fx

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=4, store=store)

        try:
            tp_pg = dist.new_group([0, 1])
            dp_pg = dist.new_group([0, 2])

            smith._C._distributed_c10d._register_process_group("tp", tp_pg)
            smith._C._distributed_c10d._register_process_group("dp", dp_pg)

            def f(x):
                y = smith.ops._c10d_functional.all_gather_into_tensor(x, 2, "tp")
                z = smith.ops._c10d_functional.wait_tensor(y)
                w = smith.ops._c10d_functional.all_reduce(z, "sum", "dp")
                v = smith.ops._c10d_functional.wait_tensor(w)
                return (v * 2,)

            args = [smith.randn(4, 4)]
            gm = make_fx(f)(*args)
            repro = generate_standalone_repro(gm, args)

            self.assertIn("setup_fake_process_groups", repro)
            self.assertIn("'tp'", repro)
            self.assertIn("'dp'", repro)
            self.assertIn("'size': 2", repro)

            # Write to temp file and execute
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp:
                tmp.write(repro)
                tmp.flush()
                result = subprocess.run(
                    [sys.executable, tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            self.assertEqual(
                result.returncode,
                0,
                f"Generated repro failed to execute:\nSTDERR:\n{result.stderr}",
            )

        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    if not (IS_FBCODE or IS_SANDCASTLE):
        # fbcode complains about not being able to find smith in subprocess
        run_tests()
