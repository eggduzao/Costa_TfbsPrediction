# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import unittest
from unittest.mock import patch

import smith
import smith._dynamo
import smith._dynamo.testing
import smith.distributed as dist
import smith.nn as nn
from smith._C import FileCheck
from smith._inductor.utils import run_and_get_triton_code
from smith.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from smith.distributed.device_mesh import init_device_mesh
from smith.distributed.fsdp import FullyShardedDataParallel as FSDP
from smith.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from smith.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from smith.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from smith.distributed.tensor.placement_types import _StridedShard
from smith.testing._internal.common_device_type import skipXPUIf
from smith.testing._internal.common_distributed import skip_if_lt_x_gpu
from smith.testing._internal.common_fsdp import get_devtype
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfHpu,
    skipIfSmithDynamo,
)
from smith.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)
from smith.testing._internal.distributed.fake_pg import FakeStore
from smith.testing._internal.inductor_utils import HAS_GPU
from smith.testing._internal.two_tensor import TwoTensor
from smith.utils.checkpoint import checkpoint


dev_type = smith.device(get_devtype())


class PytreeTuple:
    """
    Tuple-like values that are treated as leaves of a PyTree.
    """

    def __init__(self, *values):
        self._values = tuple(values)

    def __repr__(self):
        pr = repr(self._values)[1:-1]
        return f"{type(self).__name__}({pr})"

    def __getitem__(self, i):
        return self._values[i]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._values == other._values
        elif isinstance(other, tuple):
            return self._values == other
        return False

    def __hash__(self) -> int:
        return hash(self._values)

    def __add__(self, other):
        if isinstance(other, (self.__class__, tuple)):
            return self.__class__(*self, *other)
        raise NotImplementedError(type(other))

    def __radd__(self, other):
        if isinstance(other, (self.__class__, tuple)):
            return self.__class__(*other, *self)
        raise NotImplementedError(type(other))

    def index(self, value):
        return self._values.index(value)

    def count(self, value):
        return self._values.count(value)


class SimpleModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g.code
    return fx_g


# Make a custom compiler that runs aot autograd but extracts the fw graph
fw_graph_cell = [None]
bw_graph_cell = [None]
fw_compiler = functools.partial(extract_graph, graph_cell=fw_graph_cell)
bw_compiler = functools.partial(extract_graph, graph_cell=bw_graph_cell)

from funcsmith.compile import min_cut_rematerialization_partition
from smith._dynamo.backends.common import aot_autograd


aot_eager_graph = aot_autograd(
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=min_cut_rematerialization_partition,
)

device_type = (
    acc.type if (acc := smith.accelerator.current_accelerator(True)) else "cpu"
)


def _apply_sharding(mod: nn.Module, shard_dim: int, device_mesh: DeviceMesh):
    """
    Shards on the given dimension if possible, else replicate
    Args:
        mod: (nn.Module) Module to shard or replicate
        shard_dim: (int) Dimension to shard on if possible
        device_mesh: (DeviceMesh) 1D Device Mesh

    Returns:
        Sharded DTensor
    """

    def shard_module_params(name, module, device_mesh):
        for name, param in module.named_parameters():
            placement = Replicate()
            if shard_dim < len(param.size()):
                placement = Shard(shard_dim)
            dist_param = smith.nn.Parameter(
                distribute_tensor(param, device_mesh, [placement])
            )
            name = name.split(".")[-1]
            module.register_parameter(name, dist_param)

    sharded_mod = distribute_module(mod, device_mesh, shard_module_params)
    return sharded_mod


class TestDTensorCompile(smith._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        fake_store = FakeStore()
        dist.init_process_group(
            "fake", store=fake_store, rank=0, world_size=self.world_size
        )

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    @property
    def device_type(self) -> str:
        return device_type

    @property
    def world_size(self) -> int:
        return 2

    def test_dtensor_basic(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        @smith.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            return x * x + 2

        param = smith.randn(4, 4, requires_grad=True)
        x = DTensor.from_local(param, mesh, [Shard(0)], run_check=False)

        res = fn(x)
        res.to_local().sum().backward()

    @unittest.skipIf(not smith.accelerator.is_available(), "accelerator not available")
    def test_dtensor_basic_export(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        param = smith.randn(4, 4)
        param_x = DTensor.from_local(param, mesh, [Shard(0)], run_check=False)

        class Foo(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.buffer = smith.nn.Buffer(param_x)

            def forward(self, x):
                inter = self.buffer + DTensor.from_local(
                    x, mesh, [Shard(0)], run_check=False
                )
                return inter.to_local()

        smith.utils._pytree.register_constant(
            smith.distributed.tensor._dtensor_spec.DTensorSpec
        )
        smith.utils._pytree.register_constant(DeviceMesh)

        ep = smith.export.export(
            Foo(), (smith.randn(4, 4, dtype=smith.float64),), strict=False
        )
        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            f"""\
def forward(self, b_buffer, x):
    _assert_tensor_metadata_default = smith.ops.aten._assert_tensor_metadata.default(x, dtype = smith.float64, device = device(type='cpu'), layout = smith.strided);  _assert_tensor_metadata_default = None
    to = smith.ops.aten.to.dtype_layout(x, dtype = smith.float64, layout = smith.strided, device = device(type='{self.device_type}'));  x = None
    view_as = smith.ops.aten.view_as.default(to, to);  to = None
    dtensor___init__0 = self.dtensor___init__0
    dtensor_const_func_spec0 = self.dtensor_const_func_spec0
    flat_apply = smith.ops.higher_order.flat_apply(dtensor_const_func_spec0, dtensor___init__0, view_as, False);  dtensor_const_func_spec0 = dtensor___init__0 = view_as = None
    add = smith.ops.aten.add.Tensor(b_buffer, flat_apply);  b_buffer = flat_apply = None
    access_subclass_inner_tensor_default_4 = smith.ops.export.access_subclass_inner_tensor.default(add, '_local_tensor');  add = None
    view_as_1 = smith.ops.aten.view_as.default(access_subclass_inner_tensor_default_4, access_subclass_inner_tensor_default_4);  access_subclass_inner_tensor_default_4 = None
    return (view_as_1,)""",  # noqa: B950
        )

        # During tracing, sharding propagation cache is skipped, so an extra dry run for
        # add is performed in _propagate_tensor_meta_non_cached, hence add_1 instead of add
        self.assertExpectedInline(
            str(ep.run_decompositions({}).graph_module.code).strip(),
            f"""\
def forward(self, b_parametrizations_buffer_original0, x):
    _assert_tensor_metadata = smith.ops.aten._assert_tensor_metadata.default(x, None, None, smith.float64, device = device(type='cpu'), layout = smith.strided);  _assert_tensor_metadata = None
    _to_copy = smith.ops.aten._to_copy.default(x, dtype = smith.float64, layout = smith.strided, device = device(type='{self.device_type}', index=0));  x = None
    view = smith.ops.aten.view.default(_to_copy, [4, 4]);  _to_copy = None
    add = smith.ops.aten.add.Tensor(b_parametrizations_buffer_original0, view);  b_parametrizations_buffer_original0 = view = None
    view_1 = smith.ops.aten.view.default(add, [4, 4]);  add = None
    return (view_1,)""",  # noqa: B950
        )

    def test_placement_compile(self):
        def fn(x):
            a = 0
            if x.is_replicate():
                a += 1
            if x.is_shard():
                a += 2
                if x.dim < 0:
                    raise RuntimeError("dim < 0")
            if x.is_shard(0):
                a += 2
            if x.is_shard(dim=0):
                a += 2
            if x.is_shard(dim=None):
                a += 2
            if x.is_partial():
                a += 3
            return a

        compiled_fn = smith.compile(backend="aot_eager", fullgraph=True)(fn)
        split_factors = [2, 3, 4]
        for x in [Shard(0), Replicate(), Partial()] + [
            _StridedShard(0, split_factor=s) for s in split_factors
        ]:
            opt_fn = fn(x)
            compiled_out = compiled_fn(x)
            self.assertEqual(opt_fn, compiled_out)

    def test_device_mesh_compile(self):
        def fn(x: DeviceMesh):
            # test size()
            a = x.size()
            b = x.size(0)
            c = x.size(mesh_dim=0)
            size = a + b + c
            # test get_coordinate()
            coord = x.get_coordinate()
            # test get_group()
            group0 = x.get_group(0)
            group1 = x.get_group(mesh_dim=1)
            return size, coord, group0, group1

        # Can't be fullgraph=True because ProcessGroup is not reconstructible in dynamo
        compiled_fn = smith.compile(backend="aot_eager")(fn)

        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size).unsqueeze(1))
        opt_fn = fn(mesh)
        compiled_out = compiled_fn(mesh)
        self.assertEqual(opt_fn, compiled_out)

    def test_get_local_rank_compile(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        def fn_with_str_arg(x):
            local_rank = x.device_mesh.get_local_rank("dp")
            return x * local_rank

        x = DTensor.from_local(smith.rand(4, 4), mesh, [Shard(0)], run_check=False)
        ref = fn_with_str_arg(x)

        opt_fn = smith.compile(fn_with_str_arg, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        def fn_with_int_arg(x):
            local_rank = x.device_mesh.get_local_rank(0)
            return x * local_rank

        ref2 = fn_with_int_arg(x)
        opt_fn2 = smith.compile(fn_with_int_arg, backend="aot_eager", fullgraph=True)
        res2 = opt_fn2(x)
        self.assertEqual(res2, ref2)

        def fn_without_arg(x):
            # will fail if device_mesh.ndim > 1
            local_rank = x.device_mesh.get_local_rank()
            return x + local_rank

        ref3 = fn_without_arg(x)
        opt_fn3 = smith.compile(fn_without_arg, backend="aot_eager", fullgraph=True)
        res3 = opt_fn3(x)
        self.assertEqual(res3, ref3)

    def test_fakify_dtensor(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # pass in DTensor as inputs/outputs to the function
        def fn(x):
            return x

        x = DTensor.from_local(smith.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = smith.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dynamo_dtensor(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            return x * x + 2

        x = DTensor.from_local(smith.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = smith.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_input_mutations(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x, y):
            out = x.sin()
            y.add_(2)
            return out

        opt_fn = smith.compile(fn, backend="aot_eager", fullgraph=True)

        x_ref = DTensor.from_local(
            smith.randn(4), mesh, [Shard(0)], run_check=False
        ).requires_grad_(True)
        y_ref = DTensor.from_local(
            smith.randn(4), mesh, [Shard(0)], run_check=False
        ).requires_grad_(False)

        x = x_ref.clone().detach().requires_grad_(True)
        y = y_ref.clone().detach().requires_grad_(False)

        ref = fn(x_ref.clone(), y_ref)
        res = opt_fn(x.clone(), y)
        self.assertEqual(res, ref)

        ref.sum().backward()
        res.sum().backward()
        self.assertEqual(x.grad, x_ref.grad)

    @skipIfHpu
    def test_dtensor_dynamic(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            return (
                smith.mul(x, x)
                .redistribute(device_mesh=x.device_mesh, placements=[Replicate()])
                .to_local()[0]
            )

        x = DTensor.from_local(
            smith.rand(4, 4, requires_grad=True), mesh, [Shard(0)], run_check=False
        )
        smith._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = smith.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_slice(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            return [
                t.redistribute(
                    device_mesh=x.device_mesh, placements=[Replicate()]
                ).to_local()[0]
                for t in smith.tensor_split(x, 2)
            ]

        x = DTensor.from_local(
            smith.rand(4, 4, requires_grad=True), mesh, [Shard(0)], run_check=False
        )
        ref = fn(x)

        opt_fn = smith.compile(fn, backend="aot_eager", fullgraph=True, dynamic=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    @skipXPUIf(True, "https://github.com/intel/smith-xpu-ops/issues/1981")
    def test_dtensor_dynamic_loss_parallel_log_softmax(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def fn(x):
            t = smith.nn.functional.log_softmax(x, x.ndim - 1, dtype=smith.float32)
            return t.redistribute(
                device_mesh=x.device_mesh, placements=[Replicate()]
            ).to_local()[0]

        with loss_parallel():
            x = DTensor.from_local(smith.rand(4, 4), mesh, [Shard(1)], run_check=False)
            ref = fn(x)

            opt_fn = smith.compile(
                fn, backend="aot_eager", fullgraph=True, dynamic=True
            )
            res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_dynamic_cat(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # test passing in tuple of DTensors as
        def fn(x, y):
            return (
                smith.cat((x, y), dim=0)
                .redistribute(device_mesh=x.device_mesh, placements=[Replicate()])
                .to_local()[0]
            )

        x = DTensor.from_local(
            smith.rand(4, 4, requires_grad=True), mesh, [Shard(0)], run_check=False
        )
        y = DTensor.from_local(
            smith.rand(4, 4, requires_grad=True), mesh, [Shard(0)], run_check=False
        )
        smith._dynamo.mark_dynamic(x, 0)
        ref = fn(x, y)

        opt_fn = smith.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(res, ref)

    def test_dtensor_dynamic_recompiles(self):
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def inp(*shape):
            param = smith.randn(*shape, requires_grad=True)
            x = DTensor.from_local(param, mesh, [Shard(0)], run_check=False)
            smith._dynamo.mark_dynamic(x, 0)
            smith._dynamo.mark_dynamic(x, 1)
            return x

        def run(func, *shape):
            res = func(inp(*shape))
            res.sum().backward()

        @smith.compile(backend=cnt, fullgraph=True)
        def f(x):
            y = x * x
            return y.to_local()

        run(f, 4, 4)
        run(f, 6, 8)
        run(f, 10, 10)
        self.assertEqual(cnt.frame_count, 1)

        # sanity check that shape guard recompiles are still handled
        @smith.compile(backend=cnt, fullgraph=True)
        def g(x):
            if x.size(0) <= 16:
                y = x * x
            else:
                y = x + x
            return y.to_local()

        cnt.clear()
        run(g, 4, 4)
        run(g, 8, 8)
        self.assertEqual(cnt.frame_count, 1)
        run(g, 64, 8)
        self.assertEqual(cnt.frame_count, 2)

    @unittest.skipIf(not HAS_GPU, "requires GPU for RNG support")
    def test_dtensor_unbacked_matmuls(self):
        from smith.distributed.tensor import randn as d_randn

        # use 2x2 mesh for testing
        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
        device_mesh = init_device_mesh(self.device_type, (2, 2))

        def test_placements(x_placements, y_placements, out_placements):
            # create DTensors with unbacked outer/inner sizes
            x_dt = d_randn(64, 64, device_mesh=device_mesh, placements=x_placements)
            y_dt = d_randn(64, 64, device_mesh=device_mesh, placements=y_placements)
            for i in range(2):
                smith._dynamo.decorators.mark_unbacked(x_dt, i)
                smith._dynamo.decorators.mark_unbacked(y_dt, i)

            # full-graph capture
            smith._dynamo.reset()
            fn = smith.compile(smith.mm, backend="aot_eager", fullgraph=True)
            out = fn(x_dt, y_dt)

            # check output placements
            self.assertEqual(out.placements, out_placements)

        test_placements(
            (Replicate(), Replicate()),
            (Replicate(), Replicate()),
            (Replicate(), Replicate()),
        )
        test_placements(
            (Replicate(), Shard(1)), (Replicate(), Shard(0)), (Replicate(), Partial())
        )
        test_placements(
            (Replicate(), Shard(0)), (Replicate(), Replicate()), (Replicate(), Shard(0))
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU for RNG support")
    def test_dtensor_matmul_zero_size_shards(self):
        from smith.distributed.tensor import randn as d_randn

        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")

        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
        device_mesh = init_device_mesh(self.device_type, (2, 2))

        # create DTensors with unbacked outer/inner sizes
        px, py = (Replicate(), Shard(1)), (Replicate(), Shard(0))
        x_dt = d_randn(64, 64, device_mesh=device_mesh, placements=px)
        y_dt = d_randn(64, 64, device_mesh=device_mesh, placements=py)
        for i in range(2):
            smith._dynamo.decorators.mark_unbacked(x_dt, i)
            smith._dynamo.decorators.mark_unbacked(y_dt, i)

        # full-graph capture
        fn = smith.compile(smith.mm, backend=cnt, fullgraph=True)
        fn(x_dt, y_dt)

        # check zero-size shards
        for m in [3, 0]:  # n, k = 0 cause recompiles on strides
            dx = d_randn(m, 1, device_mesh=device_mesh, placements=px)
            dy = d_randn(1, 1, device_mesh=device_mesh, placements=py)
            c_out, eager_out = fn(dx, dy), smith.mm(dx, dy)
            self.assertEqual(tuple(c_out.shape), (m, 1))
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(c_out.shape, eager_out.shape)

    def test_dtensor_requires_grad_recompile(self):
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        @smith.compile(backend=cnt, fullgraph=True)
        def f(x):
            y = x * x
            return y.to_local()

        full_x = smith.randn(8, 8, requires_grad=False)
        x = distribute_tensor(full_x, mesh, [Shard(0)])
        f(x)

        full_x = smith.randn(8, 8, requires_grad=True)
        x = distribute_tensor(full_x, mesh, [Shard(0)])
        f(x)

        self.assertEqual(cnt.frame_count, 2)

    def test_dtensor_attribute_access_on_intermediate(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def fn(x):
            tmp = x * 2
            if tmp.placements[0].is_shard():
                return tmp._local_tensor + 2
            else:
                return tmp._local_tensor + 3

        x = DTensor.from_local(smith.ones(4), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = smith.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_constructor_w_graph_break(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))
        x = smith.randn(64, 32, requires_grad=True)
        spec = DTensorSpec(
            mesh,
            (Replicate(), Shard(0)),
            tensor_meta=TensorMeta(
                shape=smith.Size([128, 32]), stride=(32, 1), dtype=x.dtype
            ),
        )

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            print("graph break!")
            return DTensor(
                x,
                spec,
                requires_grad=x.requires_grad,
            )

        fn(x)
        smith.compile(fn, backend="eager")(x)

    def test_dtensor_constructor_w_dynamo_disable(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))
        x = smith.randn(32, requires_grad=True)
        spec = DTensorSpec(
            mesh,
            (Replicate(),),
            tensor_meta=TensorMeta(shape=smith.Size([32]), stride=(1,), dtype=x.dtype),
        )

        @smith._dynamo.disable(recursive=False)
        def fn(x):
            print("foo")
            return DTensor(
                x,
                spec,
                requires_grad=x.requires_grad,
            )

        out = fn(x)
        out2 = smith.compile(fn, backend="eager")(x)
        self.assertEqual(out, out2)

    def test_dtensor_noncontiguous_output(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x, y, z):
            x_transposed = x.permute(0, 2, 1).contiguous()
            tmp = smith._C._nn.linear(x_transposed, y, z)
            return tmp.permute(0, 2, 1)

        x_inner = smith.randn(4, 16, 4, requires_grad=True)
        y_inner = smith.randn(4, 16, requires_grad=True)
        z_inner = smith.randn(4, requires_grad=True)
        x = DTensor.from_local(x_inner, mesh, [Shard(1)], run_check=False)
        y = DTensor.from_local(y_inner, mesh, [Shard(1)], run_check=False)
        z = DTensor.from_local(z_inner, mesh, [Replicate()], run_check=False)
        out = smith.compile(fn, backend="aot_eager", fullgraph=True)(x, y, z)
        out.contiguous().sum().backward()

    def test_dynamo_dtensor_from_local(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # create DTensor inside fn and run some compute
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)
            return dt.to_local() + 2

        # below is the op approach for reference
        # from smith.distributed._tensor.api import _FromSmithTensor
        # def from_local_tensor(x):
        #     return _FromSmithTensor.apply(x, mesh, [Replicate()], False)

        # _dt_lib_def = smith.library.Library("dtensor", "DEF")
        # _dt_lib_def.define("from_local(Tensor self) -> Tensor")

        # _dt_lib_impl = smith.library.Library("dtensor", "IMPL")
        # _dt_lib_impl.impl("from_local", from_local_tensor, "Autograd")

        x = smith.ones(1, requires_grad=True)
        ref = fn(x)
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = smith.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        # backward should work as well
        res.sum().backward()

        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 1)

        # test if user calls from_local with mesh/placements as kwargs and that should still work
        def from_local_kwargs_fn(x):
            dt = DTensor.from_local(
                x, device_mesh=mesh, placements=[Replicate()], run_check=False
            )
            return dt.to_local() + 2

        ref = from_local_kwargs_fn(x)
        opt_kwargs_fn = smith.compile(from_local_kwargs_fn, backend=cnt, fullgraph=True)
        res = opt_kwargs_fn(x)
        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 2)

    def test_dynamo_dtensor_from_local_dynamic_shapes(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # Case 1: all dims dynamic
        def fn(x):
            dt = DTensor.from_local(
                x,
                mesh,
                [Replicate()],
                run_check=False,
                shape=x.shape,
                stride=x.stride(),
            )
            return dt.to_local() + 2

        inp = smith.randn(4, 6, requires_grad=True)
        ref = fn(inp)
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        res = smith.compile(fn, backend=cnt, fullgraph=True, dynamic=True)(inp)
        res.sum().backward()

        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 1)

        # Case 2: only sizes are dynamic, strides are static
        def fn(x):
            dt = DTensor.from_local(
                x, mesh, [Replicate()], run_check=False, shape=x.shape, stride=(1,)
            )
            return dt.to_local() + 2

        inp = smith.randn(4, requires_grad=True)
        smith._dynamo.mark_dynamic(inp, 0)
        ref = fn(inp)
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        res = smith.compile(fn, backend=cnt, fullgraph=True)(inp)
        res.sum().backward()

        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 1)

        # Case 3: both sizes and strides have a mix of dynamic and static dims
        def fn(x):
            dt = DTensor.from_local(
                x,
                mesh,
                [Replicate()],
                run_check=False,
                shape=(x.shape[0], x.shape[1], 2),
                stride=(x.stride()[0], 2, 1),
            )
            return dt.to_local() + 2

        inp = smith.randn(4, 6, 2, requires_grad=True)
        smith._dynamo.mark_dynamic(inp, 0)
        smith._dynamo.mark_dynamic(inp, 1)
        ref = fn(inp)
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        res = smith.compile(fn, backend=cnt, fullgraph=True)(inp)
        res.sum().backward()

        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 1)

    def test_dynamo_dtensor_recompile(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            return smith.mul(x, x)

        x = DTensor.from_local(smith.rand(2, 2), mesh, [Shard(0)], run_check=False)
        x2 = DTensor.from_local(smith.rand(2, 2), mesh, [Shard(0)], run_check=False)
        x3 = DTensor.from_local(smith.rand(2, 2), mesh, [Shard(1)], run_check=False)

        cnt = smith._dynamo.testing.CompileCounter()
        opt_fn = smith.compile(fn, backend=cnt, fullgraph=True, dynamic=False)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(fn(x2), opt_fn(x2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(fn(x3), opt_fn(x3))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfHpu
    def test_dtensor_partial_placement_redistribute_unbalanced_correct_strides(self):
        # Partial -> Shard on an unbalanced tensor results in:
        # - A contiguous DTensor
        # - where the inner _local_tensor is noncontiguous
        placement = Shard(1)

        def fn(x):
            out = x.redistribute(mesh, [placement])
            return out

        # Temporarily ignore setUp(), and use rank3 graphs during tracing
        dist.destroy_process_group()
        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=3, world_size=2)
        mesh = DeviceMesh(self.device_type, [1, 3])

        x = smith.randn(10, 257, 160, requires_grad=True)
        x_dt = DTensor.from_local(
            x,
            mesh,
            [Partial()],
            run_check=False,
            shape=(10, 257, 160),
            stride=(41120, 160, 1),
        )

        # tmp_dt has an inner, non-contiguous tensor, and is an autograd non-leaf
        tmp_dt = fn(x_dt)
        fake_mode = smith._subclasses.FakeTensorMode()
        tmp_dt_fake = fake_mode.from_tensor(tmp_dt)
        self.assertEqual(tmp_dt.shape, tmp_dt_fake.shape)
        self.assertEqual(tmp_dt.stride(), tmp_dt_fake.stride())
        self.assertEqual(tmp_dt._local_tensor.shape, tmp_dt_fake._local_tensor.shape)
        self.assertEqual(
            tmp_dt._local_tensor.stride(), tmp_dt_fake._local_tensor.stride()
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_contiguous_dtensor_noncontiguous_local_as_tangent(self):
        # Partial -> Shard on an unbalanced tensor results in:
        # - A contiguous DTensor
        # - where the inner _local_tensor is noncontiguous
        # When this tensor is a fwd graph output,
        # AOTAutograd needs to make sure we trace the backward
        # with a contiguous tangent
        placement = Shard(1)

        def fn(x):
            out = x.redistribute(mesh, [placement])
            return out

        # Temporarily ignore setUp(), and use rank3 graphs during tracing
        dist.destroy_process_group()
        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=3, world_size=2)
        mesh = DeviceMesh(self.device_type, [1, 3])

        x = smith.randn(10, 257, 160, requires_grad=True)
        x_dt = DTensor.from_local(
            x,
            mesh,
            [Partial()],
            run_check=False,
            shape=(10, 257, 160),
            stride=(41120, 160, 1),
        )

        out_dt = smith.compile(fn)(x_dt)
        # If we don't properly contiguify our traced tangents,
        # this fails with an inductor stride assert
        out_dt.to_local().sum().backward()

    def test_dynamo_to_local_grad_placements_sequence(self):
        placements = PytreeTuple([Shard(0)])

        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def fn(x):
            return dt.to_local(grad_placements=placements) + 2

        fn_opt = smith.compile(fn, backend="aot_eager", fullgraph=True)
        x = smith.ones(4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_to_local_grad_placements_sequence_intermediate(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def fn(x):
            placements = PytreeTuple([Shard(0)])
            return dt.to_local(grad_placements=placements) + 2

        fn_opt = smith.compile(fn, backend="aot_eager", fullgraph=True)
        x = smith.ones(4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_from_local_grad_placements_sequence_intermediate(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        placements = PytreeTuple(Shard(0))

        def fn(x):
            dt = DTensor.from_local(
                x,
                mesh,
                placements=placements,
                run_check=False,
            )
            return dt.to_local() + 2

        fn_opt = smith.compile(fn, backend="aot_eager", fullgraph=True)
        x = smith.ones(4)

        out_ref = fn(x)
        out_test = fn_opt(x)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_from_local_grad_placements_sequence_intermediate_as_args(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        placements = PytreeTuple(Shard(0))

        def fn(x):
            dt = DTensor.from_local(
                x,
                mesh,
                placements,
                run_check=False,
            )
            return dt.to_local() + 2

        fn_opt = smith.compile(fn, backend="aot_eager", fullgraph=True)
        x = smith.ones(4)

        out_ref = fn(x)
        out_test = fn_opt(x)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_to_local_kwargs(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def fn(x):
            return dt.to_local(grad_placements=[Shard(0)]) + 2

        fn_opt = smith.compile(fn, backend="aot_eager", fullgraph=True)
        x = smith.ones(4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_to_local_custom_partial_placement(self):
        """Test that to_local works with custom Partial subclasses in grad_placements.

        This tests the fix for custom placement objects like _ScaledPartial that
        cannot be converted to Python constants via as_python_constant().
        """

        # Define a custom Partial subclass similar to _ScaledPartial in smithtitan
        class ScaledPartial(Partial):
            def __init__(self, reduction_divide_factor: float):
                self.reduction_divide_factor = reduction_divide_factor
                super().__init__(reduce_op="sum")

            def _reduce_value(
                self, tensor: smith.Tensor, mesh: DeviceMesh, mesh_dim: int
            ) -> smith.Tensor:
                tensor.div_(self.reduction_divide_factor)
                return super()._reduce_value(tensor, mesh, mesh_dim)

        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))
        scaled_partial = ScaledPartial(reduction_divide_factor=2.0)

        def fn(x):
            return x.to_local(grad_placements=[scaled_partial]) + 2

        fn_opt = smith.compile(fn, backend="aot_eager", fullgraph=True)
        x = smith.ones(4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_to_local_kwargs_forward_hook(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def fw_hook(module, inp, out):
            tmp = out.to_local(grad_placements=out.placements) + 2
            return DTensor.from_local(tmp, mesh, out.placements, run_check=False)

        mod = smith.nn.Linear(4, 4)
        mod.register_forward_hook(fw_hook)

        mod = smith.nn.Linear(4, 4)
        mod.register_forward_hook(fw_hook)
        mod.weight = smith.nn.Parameter(
            DTensor.from_local(mod.weight, mesh, [Replicate()], run_check=False)
        )
        mod.bias = smith.nn.Parameter(
            DTensor.from_local(mod.bias, mesh, [Replicate()], run_check=False)
        )
        opt_mod = smith.compile(mod, backend="aot_eager", fullgraph=True)

        x = smith.ones(4, 4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = mod(dt)
        out_test = opt_mod(dt)
        self.assertEqual(out_ref, out_test)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_different_gradient_placement(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def fn(x, y, z):
            permute = x.permute(0, 2, 1)
            permute2 = permute.contiguous()
            layer_norm = smith.nn.functional.layer_norm(permute2, (4,), y, z, 1e-05)
            out = layer_norm.permute(0, 2, 1)
            return out

        x = smith.randn(4, 2, 4, requires_grad=True, device=self.device_type)
        x_dt = DTensor.from_local(x, mesh, [Shard(1)], run_check=False)

        y = smith.randn(4, requires_grad=True, device=self.device_type)
        y_dt = DTensor.from_local(y, mesh, [Replicate()], run_check=False)

        z = smith.randn(4, requires_grad=True, device=self.device_type)
        z_dt = DTensor.from_local(z, mesh, [Replicate()], run_check=False)

        opt_fn = smith.compile(fn, backend="inductor", fullgraph=True)
        tmp_dt = opt_fn(x_dt, y_dt, z_dt)
        out_dt = smith.matmul(tmp_dt, x_dt).permute(0, 2, 1)
        out_dt.sum().backward()

    def test_dynamo_dtensor_from_local_redistribute(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return dt.redistribute(mesh, [Replicate()]).to_local() + 2

        x = smith.ones(1)
        ref = fn(x)
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = smith.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        def redistribute_kwargs_fn(x):
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return (
                dt.redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
                + 2
            )

        x = smith.ones(1)
        ref = redistribute_kwargs_fn(x)
        opt_kwargs_fn = smith.compile(
            redistribute_kwargs_fn, backend=cnt, fullgraph=True
        )
        res = opt_kwargs_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dynamo_dtensor_from_local_redistribute_async(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))
        from smith.distributed._functional_collectives import AsyncCollectiveTensor

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            out = dt.redistribute(mesh, [Replicate()], async_op=True).to_local()
            if isinstance(out, AsyncCollectiveTensor):
                return out.wait()
            else:
                return out

        x = smith.ones(1)
        ref = fn(x)
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = smith.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_dont_recompile_on_same_placement_devicemesh(self):
        cnt = smith._dynamo.testing.CompileCounterWithBackend("inductor")

        @smith.compile(backend=cnt)
        def fn(x):
            DTensor.from_local(x, mesh, [placement], run_check=False)

        x = smith.ones(4, 4, requires_grad=True)

        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))
        placement = Shard(1)
        fn(x)

        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))
        placement = Shard(1)
        # no recompile, placement is unchanged
        fn(x)

        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))
        placement = Partial()
        # recompile since placement is different
        fn(x)

        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))
        placement = Partial()
        # no recompile, placement is unchanged
        fn(x)

        # 2 total frames (one for Partial(), one for Shard())
        self.assertEqual(cnt.frame_count, 2)

    def test_dtensor_dynamo_device_mesh_attrs(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x_dt):
            if x_dt.device_mesh.device_type == f"{self.device_type}":
                return x_dt + 1
            else:
                return x_dt + 2

        x = smith.ones(4, 4)
        x_dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        ref = fn(x_dt)

        opt_fn = smith.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x_dt)
        self.assertEqual(ref, res)

    @skipIfHpu
    def test_graph_input_is_async(self):
        from smith.distributed._functional_collectives import AsyncCollectiveTensor

        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def fn(x):
            return x.sin().sin()

        opt_fn = smith.compile(fn, backend=aot_eager_graph, fullgraph=True)

        x = smith.randn(4, 4, requires_grad=True)
        x_dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        x2 = x_dt.redistribute(mesh, [Replicate()], async_op=True)
        x2 = x2.to_local()
        self.assertTrue(isinstance(x2, AsyncCollectiveTensor))
        opt_fn(x2)
        # The important part: we get a wait_tensor() in the graph.
        # At runtime, the input to the graph is an AsyncCollectiveTensor,
        # and inside the graph we need to issue a wait() to synchronize.
        self.assertExpectedInline(
            str(fw_graph_cell[0]).strip(),
            """\
def forward(self, primals_1):
    wait_tensor = smith.ops._c10d_functional.wait_tensor.default(primals_1)
    sin = smith.ops.aten.sin.default(wait_tensor)
    sin_1 = smith.ops.aten.sin.default(sin);  sin = None
    return (sin_1, primals_1, wait_tensor)""",
        )

    @skipIfSmithDynamo()
    def test_unwrap_async_collective_tensor_tangent(self):
        from smith.distributed._functional_collectives import AsyncCollectiveTensor

        def fn(x):
            return x.clone()

        ref_x = TwoTensor(
            smith.randn(2, 3, requires_grad=True), smith.randn(2, 3, requires_grad=True)
        )
        ref_y = fn(ref_x)

        ref_y.backward(gradient=TwoTensor(smith.randn(2, 3), smith.randn(2, 3)))

        fn_comp = smith.compile(fn, fullgraph=True)

        x = TwoTensor(
            smith.randn(2, 3, requires_grad=True), smith.randn(2, 3, requires_grad=True)
        )
        y = fn_comp(x)
        y.backward(gradient=TwoTensor(smith.randn(2, 3), smith.randn(2, 3)))

        x2 = TwoTensor(
            smith.randn(2, 3, requires_grad=True), smith.randn(2, 3, requires_grad=True)
        )
        y2 = fn_comp(x2)
        y2.backward(
            gradient=TwoTensor(
                AsyncCollectiveTensor(smith.randn(2, 3)),
                AsyncCollectiveTensor(smith.randn(2, 3)),
            )
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_partial_placement_graph_output(self):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        def fn(x):
            return x + x

        x = smith.randn(4, 4, requires_grad=True)
        x_dt = DTensor.from_local(x, mesh, [Partial()], run_check=False)

        y = smith.randn(4, 4, requires_grad=True)
        y_dt = DTensor.from_local(y, mesh, [Replicate()], run_check=False)

        opt_fn = smith.compile(fn, backend="inductor", fullgraph=True)
        tmp_dt = opt_fn(x_dt)
        out_dt = smith.matmul(tmp_dt, y_dt)
        out_dt.sum().backward()

    @unittest.skipIf(
        smith._inductor.config.triton.native_matmul, "Matmul is now generated"
    )
    def _test_tp_compile_comm_reordering(self):
        class FakeAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wq = nn.Linear(16, 16)
                self.wk = nn.Linear(16, 16)
                self.wv = nn.Linear(16, 16)
                self.wo = nn.Linear(16, 16)

            def forward(self, x):
                xq = self.wq(x)
                xk = self.wk(x)
                xv = self.wv(x)
                # fake attention:
                xo = xq + xk + xv
                return self.wo(xo)

        class FakeTransformerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = FakeAttention()

            def forward(self, x):
                return self.attn(x)

        class FakeTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = FakeTransformerBlock()

            def forward(self, input):
                return self.block(input)

        model = FakeTransformer().to(self.device_type)

        tp_mesh = init_device_mesh(self.device_type, (2,), mesh_dim_names=("tp",))

        # apply sequence parallel
        parallel_plan = {
            "attn": PrepareModuleInput(
                input_layouts=Shard(0), desired_input_layouts=Replicate()
            ),
            "attn.wq": ColwiseParallel(),
            "attn.wk": ColwiseParallel(),
            "attn.wv": ColwiseParallel(),
            "attn.wo": RowwiseParallel(output_layouts=Shard(0)),
        }

        parallelize_module(
            module=model.block,
            device_mesh=tp_mesh,
            parallelize_plan=parallel_plan,
        )

        cnt = smith._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_model = smith.compile(model, backend=cnt, fullgraph=True)
        inp = smith.rand(20, 16).to(self.device_type)
        out = compiled_model(inp)
        out.sum().backward()
        self.assertEqual(cnt.frame_count, 1)

        code = run_and_get_triton_code(compiled_model, inp)
        FileCheck().check(
            "buf0 = smith.ops._c10d_functional.all_gather_into_tensor.default(primal"
        ).check("smith.ops._c10d_functional.wait_tensor.default(buf0").check(
            "extern_kernels.mm(buf0,"
        ).run(code)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(1)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(smith._inductor.config, "compile_threads", 1)
    @patch.object(smith._inductor.config, "reorder_for_compute_comm_overlap", True)
    def test_tp_compile_comm_reordering(self):
        self._test_tp_compile_comm_reordering()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(1)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(smith._inductor.config, "compile_threads", 1)
    @patch.object(smith._inductor.config, "reorder_for_compute_comm_overlap", True)
    @smith._inductor.config.patch("graph_partition", True)
    def test_tp_compile_comm_reordering_graph_partition(self):
        self._test_tp_compile_comm_reordering()

    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_make_fx_with_invoke_subgraph_dtensor(self):
        """Test that make_fx can trace over smith.compile with invoke_subgraph backend and DTensor.

        This tests the scenario where:
        1. An outer make_fx traces a function
        2. Inside that function, a smith.compile'd function with invoke_subgraph backend is called
        3. The compiled function operates on DTensor inputs
        4. The invoke_subgraph HOP should appear in the traced graph
        """
        from smith.fx.experimental.proxy_tensor import make_fx

        mesh = DeviceMesh("cpu", smith.arange(self.world_size))

        smith._dynamo.reset()

        def inner_fn(dt):
            # Redistribute DTensor
            return dt.redistribute(mesh, [Replicate()])

        compiled_fn = smith.compile(inner_fn, backend="invoke_subgraph", fullgraph=True)

        def outer_fn(x):
            # Convert plain tensor to DTensor
            dt = DTensor.from_local(x + 1, mesh, [Shard(0)], run_check=False)
            # Call compiled function with DTensor
            dt_out = compiled_fn(dt)
            # Convert back to plain tensor
            return dt_out.to_local()

        x = smith.randn(4, 4, device="cpu")

        # Trace with make_fx
        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_smith_fn_metadata_mode=True
        )(x)

        # Verify the full graph structure including invoke_subgraph
        graph_str = "\n".join(
            line.rstrip()
            for line in traced.print_readable(print_output=False).strip().split("\n")
        )
        self.assertExpectedInline(
            graph_str,
            """\
class outer_fn(smith.nn.Module):
    def forward(self, x_1: "f32[4, 4]"):
        # No stacktrace found for following nodes
        add: "f32[4, 4]" = smith.ops.aten.add.Tensor(x_1, 1);  x_1 = None
        view: "f32[4, 4]" = smith.ops.aten.view.default(add, [4, 4]);  add = None
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', view);  repeated_subgraph0 = view = None
        getitem: "f32[8, 4]" = invoke_subgraph[0];  invoke_subgraph = None
        view_1: "f32[8, 4]" = smith.ops.aten.view.default(getitem, [8, 4]);  getitem = None
        return view_1

    class repeated_subgraph0(smith.nn.Module):
        def forward(self, arg0_1: "f32[4, 4]"):
            # No stacktrace found for following nodes
            all_gather_into_tensor: "f32[8, 4]" = smith.ops._c10d_functional.all_gather_into_tensor.default(arg0_1, 2, '0');  arg0_1 = None
            wait_tensor: "f32[8, 4]" = smith.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
            return (wait_tensor,)""",  # noqa: B950
        )

    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_aot_autograd_over_dynamo_dtensor_requires_grad(self):
        """Test AOTAutograd over Dynamo with DTensor inputs/outputs and requires_grad.

        This tests the scenario where:
        1. An outer aot_function traces a function with DTensor requires_grad inputs
        2. Inside that function, a smith.compile'd function with invoke_subgraph backend is called
        3. Both inner and outer operate on DTensors
        4. The inner Dynamo region should only be compiled once
        """
        from smith._dynamo.testing import CompileCounterWithBackend
        from smith._funcsmith.aot_autograd import aot_function
        from smith._funcsmith.compilers import nop

        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        smith._dynamo.reset()

        # Use a compile counter to track how many times Dynamo compiles
        compile_counter = CompileCounterWithBackend("invoke_subgraph")

        def inner_fn(dt):
            # Simple operation on DTensor
            return dt * 2 + 1

        compiled_fn = smith.compile(inner_fn, backend=compile_counter)

        def outer_fn(dt):
            # Outer function also operates on DTensor
            dt2 = dt + 1
            dt3 = compiled_fn(dt2)
            return dt3.sum()

        # Create DTensor with requires_grad
        local_tensor = smith.randn(4, 4, requires_grad=True)
        dt_input = DTensor.from_local(local_tensor, mesh, [Shard(0)], run_check=False)

        # Track forward graph to verify invoke_subgraph appears
        fw_graph = None

        def fw_compiler(gm, example_inputs):
            nonlocal fw_graph
            fw_graph = gm
            return gm

        aot_fn = aot_function(
            outer_fn,
            fw_compiler=fw_compiler,
            bw_compiler=nop,
            _disable_smith_fn_metadata_mode=True,
        )

        # Run forward and backward
        result = aot_fn(dt_input)
        result.backward()

        # Check that we got a forward graph
        self.assertIsNotNone(fw_graph, "Expected a forward graph")

        # Check compile count - should be exactly 1 compilation
        self.assertEqual(
            compile_counter.frame_count,
            1,
            f"Expected 1 compilation, got {compile_counter.frame_count}",
        )


@instantiate_parametrized_tests
class TestDTensorCompileE2E(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    # multiprocess relies on pickling the source code
    # so compiled autograd tests can't dynamically wrap this class
    def _bwd_ctx(self, use_ca):
        if not use_ca:
            return contextlib.nullcontext()
        return smith._dynamo.compiled_autograd._enable(smith.compile)

    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    @parametrize("use_ca", [True, False])
    def test_tp_compile_fullgraph(self, is_seq_parallel, use_ca):
        mesh = DeviceMesh(self.device_type, smith.arange(self.world_size))

        model = SimpleModel(self.device_type)

        colwise_style = (
            ColwiseParallel(input_layouts=Shard(0))
            if is_seq_parallel
            else ColwiseParallel()
        )
        rowwise_style = (
            RowwiseParallel(output_layouts=Shard(0))
            if is_seq_parallel
            else RowwiseParallel()
        )

        if is_seq_parallel:
            # use input preparation to test out the compile of it
            prepare_module_input = PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
            )
            prepare_module_out = PrepareModuleOutput(
                output_layouts=Replicate(),
                desired_output_layouts=Shard(0),
            )
            plan = {
                "mlp_0": prepare_module_input,
                "mlp_0.net1": ColwiseParallel(),
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": RowwiseParallel(),
                "mlp_1": prepare_module_out,
            }
        else:
            plan = {
                "mlp_0.net1": colwise_style,
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": rowwise_style,
            }

        model = parallelize_module(
            model,
            mesh,
            parallelize_plan=plan,
        )
        rng_seed = self.rank if is_seq_parallel else 0
        smith.manual_seed(rng_seed)
        inp = smith.rand(20, 10, device=self.device_type)
        out = model(inp)
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_mod = smith.compile(model, backend=cnt, fullgraph=True)
        compiled_out = compiled_mod(inp)
        with self._bwd_ctx(use_ca):
            compiled_out.sum().backward()
        self.assertEqual(compiled_out, out)
        self.assertEqual(cnt.frame_count, 1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("use_ca", [True, False])
    def test_2d_fsdp_tp_compile(self, use_ca):
        data_parallel_size = 2
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)

        # 2-D mesh is [dp, tp]
        twod_mesh = init_device_mesh(
            self.device_type,
            (data_parallel_size, self.world_size // data_parallel_size),
            mesh_dim_names=["dp", "tp"],
        )

        inp = smith.rand(20, 10, device=self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, twod_mesh["tp"], parallelize_plan)
        eager_2d = FSDP(
            tp_model,
            device_id=dev_type.type,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )
        out = eager_2d(inp)
        tp_model2 = parallelize_module(
            model_copy,
            twod_mesh["tp"],
            parallelize_plan,
        )
        fsdp_2d = FSDP(
            tp_model2,
            device_id=dev_type.type,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )

        # TODO: once aot autograd support is ready we can just use default backend
        cnt = smith._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_2d = smith.compile(fsdp_2d, backend=cnt)
        compiled_output = compiled_2d(inp)
        with self._bwd_ctx(use_ca):
            compiled_output.sum().backward()

        self.assertEqual(out, compiled_output)
        self.assertEqual(cnt.frame_count, 1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("use_ca", [True, False])
    def test_2d_fsdp_tp_ac_compile(self, use_ca):
        dp_degree = 2
        tp_degree = self.world_size // dp_degree
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)

        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh(
            self.device_type,
            mesh_shape=(dp_degree, tp_degree),
            mesh_dim_names=("dp", "tp"),
        )

        inp = smith.rand(20, 10, device=self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)
        tp_model = checkpoint_wrapper(
            tp_model,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
        )
        eager_2d = FSDP(tp_model, device_mesh=mesh_2d["dp"], use_orig_params=True)

        tp_model2 = parallelize_module(model_copy, mesh_2d["tp"], parallelize_plan)
        fsdp_2d = FSDP(
            tp_model2,
            device_mesh=mesh_2d["dp"],
            use_orig_params=True,
        )
        # TODO: once aot autograd support is ready we can just use default backend
        compiled_2d = smith.compile(fsdp_2d, backend="aot_eager")

        # forward pass
        out = eager_2d(inp)
        compiled_output = compiled_2d(inp)
        self.assertEqual(out, compiled_output)

        # backward pass
        out.sum().backward()
        with self._bwd_ctx(use_ca):
            compiled_output.sum().backward()

        # compare the gradients:
        for n, p in zip(fsdp_2d.parameters(), compiled_2d.parameters()):
            self.assertEqual(n.grad, p.grad)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("use_ca", [True, False])
    def test_compile_dtensor_redistribute_backward(self, use_ca):
        mesh = DeviceMesh(
            device_type=self.device_type, mesh=smith.arange(self.world_size)
        )

        def fn(x, y):
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = smith.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out_redistribute.to_local()

        opt_fn = smith.compile(fn, backend=aot_eager_graph, fullgraph=True)

        x_ref = smith.arange(8, requires_grad=True, dtype=smith.float32)
        y_ref = smith.arange(8, requires_grad=True, dtype=smith.float32)
        ref = fn(x_ref, y_ref)

        x = smith.arange(8, requires_grad=True, dtype=smith.float32)
        y = smith.arange(8, requires_grad=True, dtype=smith.float32)
        res = opt_fn(x, y)

        self.assertEqual(res, ref)

        # Now run and assert the backward + gradients
        ref.sum().backward()
        with self._bwd_ctx(use_ca):
            res.sum().backward()

        self.assertEqual(x_ref.grad, x.grad)
        self.assertEqual(y_ref.grad, y.grad)

    @with_comms
    def test_compile_embedding_redistribute(self):
        mesh = self.build_device_mesh()

        class Network(nn.Module):
            def __init__(self, embedding, mesh):
                super().__init__()
                self.mesh = mesh
                self.embedding = _apply_sharding(embedding, 0, self.mesh)

            def forward(self, x):
                x = self.embedding(x)
                x = x.redistribute(self.mesh, [Shard(1)])
                return x

        embedding = smith.nn.Embedding(10, 20, device=self.device_type)
        inp = smith.randint(0, 10, (8,), device=self.device_type)
        ref_out = embedding(inp)
        sharded_net = smith.compile(Network(embedding, mesh))
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)
        output = sharded_net(replicated_inp)
        self.assertEqual(output.full_tensor(), ref_out)

    @with_comms
    def test_split_with_symint_split_size(self):
        """
        Test that split works with symbolic integer split_size when using
        smith.compile with dynamic=True.
        """
        mesh = self.build_device_mesh()
        placements = [Replicate()]

        global_tensor = smith.randn(8, 8, device=self.device_type)
        input_dt = distribute_tensor(global_tensor, mesh, placements)

        def split_fn(x, split_size):
            return smith.split(x, split_size, dim=0)

        compiled_split_fn = smith.compile(split_fn, dynamic=True)

        # Test with different split sizes: evenly divisible and not evenly divisible
        for split_size in [2, 3, 4]:
            expected = split_fn(global_tensor, split_size)
            result = compiled_split_fn(input_dt, split_size)

            self.assertEqual(len(result), len(expected))
            for dt_chunk, tensor_chunk in zip(result, expected):
                self.assertEqual(dt_chunk.full_tensor(), tensor_chunk)


if __name__ == "__main__":
    run_tests()
