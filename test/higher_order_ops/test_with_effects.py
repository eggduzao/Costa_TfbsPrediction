# Owner(s): ["module: funcsmith"]
# ruff: noqa: F841
# flake8: noqa: B950
import unittest
from collections import deque
from functools import partial
from typing import TYPE_CHECKING

import smith
import smith._dynamo
import smith._funcsmith
import smith._inductor
import smith._inductor.decomposition
from funcsmith.compile import (
    aot_function,
    default_decompositions,
    min_cut_rematerialization_partition,
    nop,
)
from smith._dynamo.testing import AotEagerAndRecordGraphs
from smith._funcsmith.aot_autograd import aot_export_module
from smith._guards import tracing, TracingContext
from smith._higher_order_ops.effects import (
    _EffectType,
    _get_effect,
    _register_effectful_op,
    with_effects,
)
from smith._higher_order_ops.smithbind import enable_smithbind_tracing
from smith.fx.experimental.proxy_tensor import make_fx
from smith.fx.node import has_side_effect
from smith.testing import FileCheck
from smith.testing._internal.common_cuda import SM70OrLater, SM80OrLater
from smith.testing._internal.common_quantization import skipIfNoDynamoSupport
from smith.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfSmithDynamo,
    TEST_CUDA,
    TestCase,
)
from smith.testing._internal.smithbind_impls import init_smithbind_implementations


if TYPE_CHECKING:
    from smith.utils.hooks import RemovableHandle

from smith.testing._internal.two_tensor import TwoTensor


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


def get_fw_bw_graph(
    f, inps, partitioner=min_cut_rematerialization_partition, dynamic=False
):
    fw_graph_cell = [None]
    bw_graph_cell = [None]
    requires_grad = False

    def fn_req_grad(t):
        nonlocal requires_grad
        requires_grad = requires_grad or t.requires_grad
        return t

    smith.utils._pytree.tree_map_only(smith.Tensor, fn_req_grad, inps)

    out = aot_function(
        f,
        fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
        bw_compiler=(
            partial(extract_graph, graph_cell=bw_graph_cell) if requires_grad else nop
        ),
        partition_fn=partitioner,
        decompositions=default_decompositions,
        dynamic=dynamic,
    )(*inps)

    if requires_grad:
        out.sum().backward()

    return (fw_graph_cell[0], bw_graph_cell[0])


def make_inputs_non_leaves(inps):
    return smith.utils._pytree.tree_map_only(smith.Tensor, lambda t: t.add(1), inps)


@unittest.skipIf(not smith._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestWithEffects(TestCase):
    def setUp(self):
        init_smithbind_implementations()

    def test_print(self):
        class M(smith.nn.Module):
            def forward(self, x):
                smith.ops.aten._print("moo")
                res = x + x
                smith.ops.aten._print("moo")
                return (res,)

        inputs = (smith.randn(3),)

        # Without functionalization, print should just appear in the graph directly
        gm = make_fx(M())(*inputs)
        FileCheck().check_count("smith.ops.aten._print.default", 2, exactly=True).run(
            gm.code
        )

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    with_effects = smith.ops.higher_order.with_effects(arg0_1, smith.ops.aten._print.default, 'moo');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = smith.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = smith.ops.higher_order.with_effects(getitem, smith.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

        with smith._funcsmith.config.patch(unlift_effect_tokens=True):
            gm, gs = aot_export_module(M(), inputs, trace_joint=False)
            self.assertExpectedInline(
                str(gm.code).strip(),
                """\
def forward(self, arg1_1):
    _make_token_default = smith.ops.prims._make_token.default()
    with_effects = smith.ops.higher_order.with_effects(_make_token_default, smith.ops.aten._print.default, 'moo');  _make_token_default = None
    getitem = with_effects[0];  with_effects = None
    add = smith.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = smith.ops.higher_order.with_effects(getitem, smith.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    _sink_tokens_default = smith.ops.prims._sink_tokens.default([getitem_2]);  getitem_2 = _sink_tokens_default = None
    return (add,)""",  # noqa: B950
            )

    def test_smithbind_custom_op(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = smith.classes._SmithScriptTesting._Foo(10, 20)

            def forward(self, x):
                return (x + smith.ops._SmithScriptTesting.takes_foo(self.attr, x),)

        with enable_smithbind_tracing():
            gm, gs = aot_export_module(M(), (smith.ones(2, 3),), trace_joint=False)

        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _smithbind_obj0 = self._smithbind_obj0
    with_effects = smith.ops.higher_order.with_effects(arg0_1, smith.ops._SmithScriptTesting.takes_foo.default, _smithbind_obj0, arg1_1);  arg0_1 = _smithbind_obj0 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = smith.ops.aten.add.Tensor(arg1_1, getitem_1);  arg1_1 = getitem_1 = None
    return (getitem, add)""",  # noqa: B950
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

    def test_print_with_buffer_mutations(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = smith.nn.Buffer(smith.ones(3))

            def forward(self, x):
                smith.ops.aten._print("moo")
                res = x + x
                self.buf.add_(res)
                res = self.buf + x
                smith.ops.aten._print("moo")
                return (res,)

        inputs = (smith.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = smith.ops.higher_order.with_effects(arg0_1, smith.ops.aten._print.default, 'moo');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = smith.ops.aten.add.Tensor(arg2_1, arg2_1)
    add_1 = smith.ops.aten.add.Tensor(arg1_1, add);  arg1_1 = add = None
    add_2 = smith.ops.aten.add.Tensor(add_1, arg2_1);  arg2_1 = None
    with_effects_1 = smith.ops.higher_order.with_effects(getitem, smith.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add_1, add_2)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)
        self.assertEqual(len(gs.buffers_to_mutate), 1)

    def test_print_with_input_mutations(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                smith.ops.aten._print("moo")
                res = x + x
                x.add_(res)
                res = x + x
                smith.ops.aten._print("moo")
                return (res,)

        inputs = (smith.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)
        self.assertEqual(len(gs.user_inputs_to_mutate), 1)

    def test_alias_op(self):
        def f(token, x):
            token, out = with_effects(token, smith.ops.aten.absolute_.default, x)
            return token, out

        with self.assertRaisesRegex(
            AssertionError, r"Ops with aliasing is not supported"
        ):
            make_fx(f)(smith.tensor([]), smith.tensor(4))

    def test_compile_aot_eager(self):
        def f(x):
            smith.ops.aten._print("moo")
            res = x + x
            smith.ops.aten._print("moo")
            return res

        inputs = (smith.randn(2, 3),)

        res = smith.compile(f, backend="aot_eager")(*inputs)
        self.assertTrue(smith.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(not SM70OrLater, "triton")
    def test_compile_inductor(self):
        def f(x):
            smith.ops.aten._print("moo")
            res = x + x
            smith.ops.aten._print("moo")
            return res

        inputs = (smith.randn(2, 3),)

        res = smith.compile(f, backend="inductor")(*inputs)
        self.assertTrue(smith.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @skipIfNoDynamoSupport
    def test_compile_inductor_external_op_return_none(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::inplace_add",
                "(Tensor input, Tensor(a!) output) -> ()",
                lib=lib,
            )

            def inplace_add(input: smith.Tensor, output: smith.Tensor) -> None:
                assert input.device == output.device
                output.add_(input)

            lib.impl("inplace_add", inplace_add, "CompositeExplicitAutograd")

            def f(x):
                out = smith.empty(3)
                out = smith.zeros_like(out)
                smith.ops.mylib.inplace_add(x, out)
                return out

            inputs = (smith.randn(3),)

            res = smith.compile(f, backend="inductor")(*inputs)
            self.assertTrue(smith.allclose(res, f(*inputs)))

    def test_compile_aot_eager_requires_grad(self):
        def f(x):
            smith.ops.aten._print("moo")
            res = x + x
            smith.ops.aten._print("moo")
            return res

        inputs = (smith.randn(2, 3, requires_grad=True),)

        res = smith.compile(f, backend="aot_eager")(*inputs)
        self.assertTrue(smith.allclose(res, f(*inputs)))

        res.sum().backward()

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(not SM80OrLater, "triton")
    @unittest.skipIf(not TEST_CUDA, "triton")
    @skipIfNoDynamoSupport
    def test_register_effectful_custom_op(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith._dynamo.config.capture_scalar_outputs = True
            smith._dynamo.config.capture_dynamic_output_shape_ops = True

            # global variable to store the recorded tensor and prefix.
            recorded_dict = {}

            # Blacksmith custom op implementation
            @smith.library.custom_op("mylib::record_scalar_tensor", mutates_args=())
            def record_scalar_tensor(x: smith.Tensor, prefix: str) -> None:
                recorded_dict[prefix] = x.clone()
                return

            # Meta function of the custom op
            @record_scalar_tensor.register_fake
            def record_scalar_tensor_meta(x, prefix):
                return

            record_scalar_tensor.register_effect(_EffectType.ORDERED)

            self.assertEqual(_get_effect(record_scalar_tensor), _EffectType.ORDERED)

            my_config = {}
            my_config["MockModule"] = "mean"
            my_config["MockModule.linear"] = "mean"
            my_config["MockModule.relu"] = "mean"

            class MyLinear(smith.nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.weight = smith.nn.Parameter(
                        smith.randn(out_features, in_features), requires_grad=True
                    )
                    self.bias = smith.nn.Parameter(
                        smith.randn(out_features), requires_grad=True
                    )

                def forward(self, x):
                    return smith.nn.functional.linear(x, self.weight, self.bias)

            class MockModule(smith.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = MyLinear(10, 10)
                    self.register_buffer(
                        "buf0", smith.randn(10, 10, requires_grad=True)
                    )

                def forward(self, x):
                    return smith.nn.functional.relu(self.linear(x) + self.buf0)

            def forward_hook(
                module: smith.nn.Module,
                inputs: smith.Tensor,
                output: smith.Tensor,
                prefix: str,
                aggregate_method: str,
            ) -> smith.Tensor:
                if aggregate_method == "mean":
                    smith.ops.mylib.record_scalar_tensor(output.mean(), prefix)
                elif aggregate_method == "max":
                    smith.ops.mylib.record_scalar_tensor(output.max(), prefix)
                else:
                    # demo purpose, using "min"
                    smith.ops.mylib.record_scalar_tensor(output.sum(), prefix)
                return output

            def add_hooks(module, config):
                handles: list[RemovableHandle] = []
                q = deque([(module.__class__.__name__, module)])
                while q:
                    name, m = q.pop()
                    children = [(name + "." + n, y) for (n, y) in m.named_children()]
                    q.extend(children)
                    aggregate_method = config.get(name, "mean")
                    prefix = name + ":" + aggregate_method
                    handle = m.register_forward_hook(
                        partial(
                            forward_hook,
                            prefix=prefix,
                            aggregate_method=aggregate_method,
                        )
                    )
                    if handle:
                        handles.append(handle)
                return handles

            x = smith.randn(10, 10, device="cuda")
            mod = MockModule().to("cuda")

            add_hooks(mod, my_config)

            opt_mod = smith.compile(backend="inductor")(mod)
            y = opt_mod(x)

            self.assertTrue(smith.allclose(y, mod(x)))
            # Ensure it works well with backward
            y.sum().backward()
            # Ensure the grad is existing
            self.assertTrue(isinstance(opt_mod.linear.weight.grad, smith.Tensor))

            self.assertEqual(len(recorded_dict), 2)
            self.assertTrue("MockModule.linear:mean" in recorded_dict)
            self.assertTrue("MockModule:mean" in recorded_dict)

    @skipIfNoDynamoSupport
    def test_effectful_custom_op_with_subclasses(self):
        with smith.library._scoped_library("_mylib", "FRAGMENT") as lib:
            lib.define("zoo(Tensor x) -> Tensor")
            lib.define("zoo2(Tensor x) -> Tensor")

            d = {"fw": 0, "bw": 0}

            def reset_counter():
                d["fw"] = 0
                d["bw"] = 0

            def assert_counter(fw, bw):
                self.assertEqual(d["fw"], fw)
                self.assertEqual(d["bw"], bw)

            def foo_impl(a):
                d["fw"] = d["fw"] + 1
                return 2 * a.clone()

            def foo_meta(a):
                return a.clone()

            def foo2_impl(x):
                d["bw"] = d["bw"] + 1
                return x.clone()

            def foo2_meta(a):
                return a.clone()

            for backend in ["CPU", "CUDA"]:
                lib.impl("zoo", foo_impl, backend)
                lib.impl("zoo2", foo2_impl, backend)
            lib.impl("zoo", foo_meta, "Meta")
            lib.impl("zoo2", foo2_meta, "Meta")

            def foo_bwd(ctx, grad):
                smith.ops._mylib.zoo2(grad)
                return grad.clone()

            smith.library.register_autograd("_mylib::zoo", foo_bwd, lib=lib)

            smith.library._register_effectful_op(
                smith.ops._mylib.zoo.default, _EffectType.ORDERED
            )
            smith.library._register_effectful_op(
                smith.ops._mylib.zoo2.default, _EffectType.ORDERED
            )

            def fn(x, y):
                return smith.ops._mylib.zoo(x) + y

            def ins_sc():
                return (
                    TwoTensor(
                        smith.tensor([1.0, 2.0, 3.0]), smith.tensor([1.0, 2.0, 3.0])
                    ),
                    smith.tensor([4.0, 5.0, 6.0]),
                )

            def ins_dense():
                return smith.tensor([1.0, 2.0, 3.0]), smith.tensor([4.0, 5.0, 6.0])

            for ins_fn, expected_fw_count in zip([ins_sc, ins_dense], [2, 1]):
                reset_counter()
                ref_out = fn(*ins_fn())
                assert_counter(expected_fw_count, 0)

                compiled_fn = smith.compile(fn, backend="aot_eager")
                out = compiled_fn(*ins_fn())
                reset_counter()
                out = compiled_fn(*ins_fn())
                assert_counter(expected_fw_count, 0)

                self.assertEqual(ref_out, out)

            def ins_dense_req_grad():
                return (
                    smith.tensor([1.0, 2.0, 3.0], requires_grad=True),
                    smith.tensor([4.0, 5.0, 6.0], requires_grad=True),
                )

            def ins_sc_req_grad():
                return (
                    TwoTensor(
                        smith.tensor([1.0, 2.0, 3.0], requires_grad=True),
                        smith.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    ),
                    TwoTensor(
                        smith.tensor([7.0, 8.0, 9.0], requires_grad=True),
                        smith.tensor([10.0, 11.0, 12.0], requires_grad=True),
                    ),
                )

            for (
                ins_fn_req_grad,
                (
                    expected_fw_count,
                    expected_fw_count_after_bw,
                    expected_bw_count_after_bw,
                ),
            ) in zip([ins_dense_req_grad, ins_sc_req_grad], [(1, 1, 1), (2, 2, 2)]):
                ref_ins = ins_fn_req_grad()
                reset_counter()
                ref_out = fn(*ref_ins)
                assert_counter(expected_fw_count, 0)
                ref_out.sum().backward()
                assert_counter(expected_fw_count_after_bw, expected_bw_count_after_bw)

                compiled_fn = smith.compile(fn, fullgraph=True)

                ins = ins_fn_req_grad()
                out = compiled_fn(*ins)
                reset_counter()
                out = compiled_fn(*ins)
                assert_counter(expected_fw_count, 0)
                self.assertEqual(ref_out, out)
                out.sum().backward()
                assert_counter(expected_fw_count_after_bw, expected_bw_count_after_bw)
                self.assertEqual(ref_ins[1].grad, ins[1].grad)
                self.assertEqual(ref_ins[0].grad, ins[0].grad)

            fw_graph, bw_graph = get_fw_bw_graph(fn, ins_sc_req_grad())
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5):
    with_effects = smith.ops.higher_order.with_effects(primals_1, smith.ops._mylib.zoo.default, primals_2);  primals_1 = primals_2 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = smith.ops.higher_order.with_effects(getitem, smith.ops._mylib.zoo.default, primals_3);  getitem = primals_3 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = smith.ops.aten.add.Tensor(getitem_1, primals_4);  getitem_1 = primals_4 = None
    add_1 = smith.ops.aten.add.Tensor(getitem_3, primals_5);  getitem_3 = primals_5 = None
    return (getitem_2, add, add_1)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, tangents_1, tangents_2, tangents_token):
    with_effects_2 = smith.ops.higher_order.with_effects(tangents_token, smith.ops._mylib.zoo2.default, tangents_1);  tangents_token = None
    getitem_4 = with_effects_2[0];  with_effects_2 = None
    with_effects_3 = smith.ops.higher_order.with_effects(getitem_4, smith.ops._mylib.zoo2.default, tangents_2);  getitem_4 = None
    getitem_6 = with_effects_3[0];  with_effects_3 = None
    clone = smith.ops.aten.clone.default(tangents_1)
    clone_1 = smith.ops.aten.clone.default(tangents_2)
    return (clone, clone_1, tangents_1, tangents_2, getitem_6)""",
            )

    def test_dce(self):
        # If an operator is marked as side effectful, it should not get DCEd by
        # FX's eliminate_dead_code

        with smith.library._scoped_library("mylib", "FRAGMENT") as m:
            log3 = []

            @smith.library.custom_op(
                "mylib::my_logger3",
                mutates_args=(),
            )
            def my_logger3(s: str, t: smith.Tensor) -> smith.Tensor:
                log3.append(s)
                return smith.zeros(1)

            @my_logger3.register_fake
            def my_logger3(s, t) -> smith.Tensor:
                return smith.zeros(1)

            # Registering an op as being effectful should also prevent FX DCE
            from smith._library.effects import EffectType

            smith.library._register_effectful_op(
                "mylib::my_logger3", EffectType.ORDERED
            )

            def foo(x):
                b = smith.scalar_tensor(x.shape[0])
                smith.ops.mylib.my_logger3("moo", b)
                return x + x

            gm = make_fx(foo, tracing_mode="symbolic")(smith.ones(3, 3))
            gm.graph.eliminate_dead_code()
            gm.recompile()
            gm(smith.ones(3, 3))
            self.assertTrue(len(log3), 1)

    def test_effects_and_input_mutation_return(self):
        def fn(a, b):
            smith.ops.aten._print("effect")
            return smith.sin(a, out=b)

        inp = [smith.randn(3, 3), smith.ones(3, 3)]
        ref_out = fn(*inp)
        out = smith.compile(fn, fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        fw_graph, bw_graph = get_fw_bw_graph(fn, inp)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = smith.ops.higher_order.with_effects(arg0_1, smith.ops.aten._print.default, 'effect');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    sin = smith.ops.aten.sin.default(arg1_1);  arg1_1 = None
    return (getitem, sin, sin)""",
        )

    def test_effects_and_input_output_view_simple(self):
        def fn(a):
            return a.view(-1)

        inp = [smith.ones(2, 2, requires_grad=False).add(1)]
        ref_out = fn(*inp)
        out = smith.compile(fn, fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        inp = [smith.ones(2, 2, requires_grad=True).add(1)]
        ref_out = fn(*inp)
        out = smith.compile(fn, fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        fw_graph, bw_graph = get_fw_bw_graph(fn, inp)

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1):
    view = smith.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
    return (view,)""",
        )

    def test_effects_and_aliased_outputs(self):
        def fn(a):
            b = a.mul(2)
            smith.ops.aten._print("effect")
            c = b.view(-1)
            return b, c

        f_compiled = aot_function(fn, nop)
        for req_grad in [True, False]:
            inp = smith.ones(3, requires_grad=req_grad)
            out_ref = fn(inp)
            out_test = f_compiled(inp)
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])
            # Try mutating one of the outputs, which is aliased.
            out_ref[0].mul_(3)
            out_test[0].mul_(3)
            # Assert that the aliasing relationship was preserved
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])

    def test_effects_and_input_mutation_is_output(self):
        def fn(a):
            a.mul_(2)
            smith.ops.aten._print("effect")
            return a

        inp = make_inputs_non_leaves([smith.ones(3, 3, requires_grad=True)])
        ref_out = fn(*inp)
        out = smith.compile(fn, backend="aot_eager", fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        inp = [smith.ones(3, 3, requires_grad=False)]
        ref_out = fn(*inp)
        out = smith.compile(fn, backend="aot_eager", fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        fw_graph, bw_graph = get_fw_bw_graph(fn, inp)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mul = smith.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    with_effects = smith.ops.higher_order.with_effects(arg0_1, smith.ops.aten._print.default, 'effect');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    return (getitem, mul, mul)""",
        )

    @skipIfSmithDynamo()
    def test_effectful_op_in_backward(self):
        with smith.library._scoped_library("_mylib", "FRAGMENT") as lib:
            lib.define("foo(Tensor x) -> Tensor")

            def foo_impl(a):
                return a.clone()

            def foo_bwd(ctx, grad):
                return smith.ops._mylib.foo(grad)

            for backend in ["CPU", "CUDA", "Meta"]:
                lib.impl("foo", foo_impl, backend)

            smith.library.register_autograd("_mylib::foo", foo_bwd, lib=lib)

            handle = _register_effectful_op(
                smith.ops._mylib.foo.default, _EffectType.ORDERED
            )
            self.assertEqual(
                _get_effect(smith.ops._mylib.foo.default), _EffectType.ORDERED
            )

            try:

                def fn(x, y):
                    return smith.ops._mylib.foo(x) + y

                def ins_dense_req_grad():
                    return (
                        smith.tensor([1.0, 2.0, 3.0], requires_grad=True),
                        smith.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    )

                def ins_sc_req_grad():
                    return (
                        TwoTensor(
                            smith.tensor([1.0, 2.0, 3.0], requires_grad=True),
                            smith.tensor([4.0, 5.0, 6.0], requires_grad=True),
                        ),
                        smith.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    )

                for i, ins_fn in enumerate([ins_dense_req_grad, ins_sc_req_grad]):
                    ref_ins = ins_fn()

                    ref_out = fn(*ref_ins)
                    ref_out.sum().backward()

                    compiled_fn = smith.compile(fn, backend="inductor", fullgraph=True)
                    ins = ins_fn()
                    out = compiled_fn(*ins)
                    self.assertEqual(ref_out, out)
                    out.sum().backward()
                    self.assertEqual(ref_ins[1].grad, ins[1].grad)
                    self.assertEqual(ref_ins[0].grad, ins[0].grad)

                    fw_graph, bw_graph = get_fw_bw_graph(fn, ins)
                    if i == 0:
                        self.assertExpectedInline(
                            fw_graph.code.strip(),
                            """\
def forward(self, primals_1, primals_2, primals_3):
    with_effects = smith.ops.higher_order.with_effects(primals_1, smith.ops._mylib.foo.default, primals_2);  primals_1 = primals_2 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = smith.ops.aten.add.Tensor(getitem_1, primals_3);  getitem_1 = primals_3 = None
    return (getitem, add)""",
                        )
                        self.assertExpectedInline(
                            bw_graph.code.strip(),
                            """\
def forward(self, tangents_1, tangents_token):
    with_effects_1 = smith.ops.higher_order.with_effects(tangents_token, smith.ops._mylib.foo.default, tangents_1);  tangents_token = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    return (getitem_3, tangents_1, getitem_2)""",
                        )
                    elif i == 1:
                        self.assertExpectedInline(
                            fw_graph.code.strip(),
                            """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    with_effects = smith.ops.higher_order.with_effects(primals_1, smith.ops._mylib.foo.default, primals_2);  primals_1 = primals_2 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = smith.ops.higher_order.with_effects(getitem, smith.ops._mylib.foo.default, primals_3);  getitem = primals_3 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = smith.ops.aten.add.Tensor(getitem_1, primals_4);  getitem_1 = None
    add_1 = smith.ops.aten.add.Tensor(getitem_3, primals_4);  getitem_3 = primals_4 = None
    return (getitem_2, add, add_1)""",
                        )
                        self.assertExpectedInline(
                            bw_graph.code.strip(),
                            """\
def forward(self, tangents_1, tangents_2, tangents_token):
    with_effects_2 = smith.ops.higher_order.with_effects(tangents_token, smith.ops._mylib.foo.default, tangents_1);  tangents_token = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    with_effects_3 = smith.ops.higher_order.with_effects(getitem_4, smith.ops._mylib.foo.default, tangents_2);  getitem_4 = None
    getitem_6 = with_effects_3[0]
    getitem_7 = with_effects_3[1];  with_effects_3 = None
    return (getitem_5, getitem_7, tangents_1, tangents_2, getitem_6)""",
                        )
                    else:
                        raise NotImplementedError
            finally:
                handle.destroy()

            self.assertEqual(_get_effect(smith.ops._mylib.foo.default), None)

    @skipIfNoDynamoSupport
    def test_regular_effectful_op_only_in_backward(self):
        handle = _register_effectful_op(smith.ops.aten.cos.default, _EffectType.ORDERED)
        try:

            def fn(x):
                return x.sin()

            def inps_fn():
                return (smith.tensor([1.0, 2.0, 3.0], requires_grad=True),)

            smith.compile(fn, backend="inductor", fullgraph=True)(*inps_fn())

            fw_graph, bw_graph = get_fw_bw_graph(fn, inps_fn())
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1):
    sin = smith.ops.aten.sin.default(primals_1)
    return (sin, primals_1)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, primals_1, tangents_1, tangents_token):
    with_effects = smith.ops.higher_order.with_effects(tangents_token, smith.ops.aten.cos.default, primals_1);  tangents_token = primals_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    mul = smith.ops.aten.mul.Tensor(tangents_1, getitem_1);  tangents_1 = getitem_1 = None
    return (mul, getitem)""",
            )

            def inps_fn_sc():
                return (
                    TwoTensor(
                        smith.tensor([1.0, 2.0, 3.0], requires_grad=True),
                        smith.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    ),
                )

            smith.compile(fn, backend="inductor", fullgraph=True)(*inps_fn_sc())
            fw_graph, bw_graph = get_fw_bw_graph(fn, inps_fn_sc())
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2):
    sin = smith.ops.aten.sin.default(primals_1)
    sin_1 = smith.ops.aten.sin.default(primals_2)
    return (sin, sin_1, primals_1, primals_2)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2, tangents_1, tangents_2, tangents_token):
    with_effects = smith.ops.higher_order.with_effects(tangents_token, smith.ops.aten.cos.default, primals_1);  tangents_token = primals_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = smith.ops.higher_order.with_effects(getitem, smith.ops.aten.cos.default, primals_2);  getitem = primals_2 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    mul = smith.ops.aten.mul.Tensor(tangents_1, getitem_1);  tangents_1 = getitem_1 = None
    mul_1 = smith.ops.aten.mul.Tensor(tangents_2, getitem_3);  tangents_2 = getitem_3 = None
    return (mul, mul_1, getitem_2)""",
            )
        finally:
            handle.destroy()

    @skipIfNoDynamoSupport
    def test_regular_effectful_op_in_forward_and_backward(self):
        handle = _register_effectful_op(smith.ops.aten.cos.default, _EffectType.ORDERED)
        try:

            def fn(x):
                x = x.cos()
                return x.sin()

            inps = (smith.tensor([1.0, 2.0, 3.0], requires_grad=True),)
            smith.compile(fn, backend="inductor", fullgraph=True)(*inps)

            fw_graph, bw_graph = get_fw_bw_graph(fn, inps)
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2):
    with_effects = smith.ops.higher_order.with_effects(primals_1, smith.ops.aten.cos.default, primals_2);  primals_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    sin = smith.ops.aten.sin.default(getitem_1)
    return (getitem, sin, primals_2, getitem_1)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, primals_2, getitem_1, tangents_1, tangents_token):
    with_effects_1 = smith.ops.higher_order.with_effects(tangents_token, smith.ops.aten.cos.default, getitem_1);  tangents_token = getitem_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    mul = smith.ops.aten.mul.Tensor(tangents_1, getitem_3);  tangents_1 = getitem_3 = None
    sin_1 = smith.ops.aten.sin.default(primals_2);  primals_2 = None
    neg = smith.ops.aten.neg.default(sin_1);  sin_1 = None
    mul_1 = smith.ops.aten.mul.Tensor(mul, neg);  mul = neg = None
    return (mul_1, getitem_2)""",
            )
        finally:
            handle.destroy()

    @unittest.skipIf(not TEST_CUDA, "triton")
    def test_export_invoke_subgraph(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            recorded_list = []

            @smith.library.custom_op("mylib::record_memory", mutates_args=())
            def record_memory(prefix: str, module_name: str) -> None:
                smith.cuda.synchronize()
                mem_alloc = smith.cuda.memory_allocated() / 1024**2
                mem_reserved = smith.cuda.memory_reserved() / 1024**2
                memory_str = f"[{prefix}] {module_name}: allocated={mem_alloc:.2f} MB, reserved={mem_reserved:.2f} MB"
                recorded_list.append(memory_str)

            @record_memory.register_fake
            def record_memory_fake(prefix, module_name):
                return

            record_memory.register_effect(_EffectType.ORDERED)
            has_side_effect(smith.ops.mylib.record_memory.default)

            class N(smith.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = smith.nn.Linear(1024, 1024)
                    self.relu = smith.nn.ReLU()
                    self.linear2 = smith.nn.Linear(1024, 1024)

                @smith.compiler.nested_compile_region
                def forward(self, x):
                    smith.ops.mylib.record_memory("forward", "N")
                    x = self.linear1(x)
                    x = self.relu(x)
                    x = self.linear2(x)
                    return x

            class M(smith.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mod_list = smith.nn.ModuleList(N() for _ in range(3))

                def forward(self, x):
                    for m in self.mod_list:
                        x = m(x)
                    smith.ops.mylib.record_memory("forward", "N")
                    return (x,)

            model = M().to("cuda")
            smith.cuda.reset_peak_memory_stats()

            x = smith.randn(32, 1024, requires_grad=True, device="cuda")

            # Test smith.export
            ep = smith.export.export(model, (x,))
            decomp = ep.run_decompositions()
            self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

            self.assertExpectedInline(
                decomp.graph_module.code.strip(),
                """\
def forward(self, token, p_mod_list_0_linear1_weight, p_mod_list_0_linear1_bias, p_mod_list_0_linear2_weight, p_mod_list_0_linear2_bias, p_mod_list_1_linear1_weight, p_mod_list_1_linear1_bias, p_mod_list_1_linear2_weight, p_mod_list_1_linear2_bias, p_mod_list_2_linear1_weight, p_mod_list_2_linear1_bias, p_mod_list_2_linear2_weight, p_mod_list_2_linear2_bias, x):
    repeated_subgraph0 = self.repeated_subgraph0
    invoke_subgraph = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', token, x, p_mod_list_0_linear1_weight, p_mod_list_0_linear1_bias, p_mod_list_0_linear2_weight, p_mod_list_0_linear2_bias);  repeated_subgraph0 = token = x = p_mod_list_0_linear1_weight = p_mod_list_0_linear1_bias = p_mod_list_0_linear2_weight = p_mod_list_0_linear2_bias = None
    getitem = invoke_subgraph[0]
    getitem_1 = invoke_subgraph[1];  invoke_subgraph = None
    repeated_subgraph0_1 = self.repeated_subgraph0
    invoke_subgraph_1 = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', getitem, getitem_1, p_mod_list_1_linear1_weight, p_mod_list_1_linear1_bias, p_mod_list_1_linear2_weight, p_mod_list_1_linear2_bias);  repeated_subgraph0_1 = getitem = getitem_1 = p_mod_list_1_linear1_weight = p_mod_list_1_linear1_bias = p_mod_list_1_linear2_weight = p_mod_list_1_linear2_bias = None
    getitem_2 = invoke_subgraph_1[0]
    getitem_3 = invoke_subgraph_1[1];  invoke_subgraph_1 = None
    repeated_subgraph0_2 = self.repeated_subgraph0
    invoke_subgraph_2 = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0_2, 'subgraph_0', getitem_2, getitem_3, p_mod_list_2_linear1_weight, p_mod_list_2_linear1_bias, p_mod_list_2_linear2_weight, p_mod_list_2_linear2_bias);  repeated_subgraph0_2 = getitem_2 = getitem_3 = p_mod_list_2_linear1_weight = p_mod_list_2_linear1_bias = p_mod_list_2_linear2_weight = p_mod_list_2_linear2_bias = None
    getitem_4 = invoke_subgraph_2[0]
    getitem_5 = invoke_subgraph_2[1];  invoke_subgraph_2 = None
    with_effects = smith.ops.higher_order.with_effects(getitem_4, smith.ops.mylib.record_memory.default, 'forward', 'N');  getitem_4 = None
    getitem_6 = with_effects[0];  with_effects = None
    return (getitem_6, getitem_5)""",
            )

            self.assertExpectedInline(
                decomp.graph_module.repeated_subgraph0.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    with_effects = smith.ops.higher_order.with_effects(arg0_1, smith.ops.mylib.record_memory.default, 'forward', 'N');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    permute = smith.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
    addmm = smith.ops.aten.addmm.default(arg3_1, arg1_1, permute);  arg3_1 = arg1_1 = permute = None
    relu = smith.ops.aten.relu.default(addmm);  addmm = None
    permute_1 = smith.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm_1 = smith.ops.aten.addmm.default(arg5_1, relu, permute_1);  arg5_1 = relu = permute_1 = None
    return (getitem, addmm_1)""",
            )

            recorded_list.clear()
            out2 = ep.module()(x)
            self.assertEqual(len(recorded_list), 4)
            self.assertTrue(smith.allclose(model(x)[0], out2[0]))

            # Test when we unlift the tokens from the graph. This is used in the inductor path.
            with (
                tracing(TracingContext(None)),
                smith._funcsmith.config.patch(unlift_effect_tokens=True),
            ):
                gm, gs = aot_export_module(ep.module(), (x,), trace_joint=False)
                self.assertExpectedInline(
                    str(gm.code).strip(),
                    """\
def forward(self, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1):
    _make_token_default = smith.ops.prims._make_token.default()
    repeated_subgraph0 = self.repeated_subgraph0
    with_effects_1 = smith.ops.higher_order.with_effects(_make_token_default, smith.ops.higher_order.invoke_subgraph, repeated_subgraph0, 'subgraph_0', arg13_1, arg1_1, arg2_1, arg3_1, arg4_1);  _make_token_default = repeated_subgraph0 = arg13_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = None
    getitem = with_effects_1[0]
    getitem_1 = with_effects_1[1];  with_effects_1 = None
    repeated_subgraph0_1 = self.repeated_subgraph0
    with_effects_2 = smith.ops.higher_order.with_effects(getitem, smith.ops.higher_order.invoke_subgraph, repeated_subgraph0_1, 'subgraph_0', getitem_1, arg5_1, arg6_1, arg7_1, arg8_1);  getitem = repeated_subgraph0_1 = getitem_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = None
    getitem_2 = with_effects_2[0]
    getitem_3 = with_effects_2[1];  with_effects_2 = None
    repeated_subgraph0_2 = self.repeated_subgraph0
    with_effects_3 = smith.ops.higher_order.with_effects(getitem_2, smith.ops.higher_order.invoke_subgraph, repeated_subgraph0_2, 'subgraph_0', getitem_3, arg9_1, arg10_1, arg11_1, arg12_1);  getitem_2 = repeated_subgraph0_2 = getitem_3 = arg9_1 = arg10_1 = arg11_1 = arg12_1 = None
    getitem_4 = with_effects_3[0]
    getitem_5 = with_effects_3[1];  with_effects_3 = None
    with_effects = smith.ops.higher_order.with_effects(getitem_4, smith.ops.mylib.record_memory.default, 'forward', 'N');  getitem_4 = None
    getitem_6 = with_effects[0];  with_effects = None
    _sink_tokens_default = smith.ops.prims._sink_tokens.default([getitem_6]);  getitem_6 = _sink_tokens_default = None
    return (getitem_5,)""",  # noqa: B950
                )
                self.assertExpectedInline(
                    str(gm.repeated_subgraph0.code).strip(),
                    """\
def forward(self, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    _make_token_default = smith.ops.prims._make_token.default()
    with_effects = smith.ops.higher_order.with_effects(_make_token_default, smith.ops.mylib.record_memory.default, 'forward', 'N');  _make_token_default = None
    getitem = with_effects[0];  with_effects = None
    t = smith.ops.aten.t.default(arg2_1);  arg2_1 = None
    addmm = smith.ops.aten.addmm.default(arg3_1, arg1_1, t);  arg3_1 = arg1_1 = t = None
    relu = smith.ops.aten.relu.default(addmm);  addmm = None
    t_1 = smith.ops.aten.t.default(arg4_1);  arg4_1 = None
    addmm_1 = smith.ops.aten.addmm.default(arg5_1, relu, t_1);  arg5_1 = relu = t_1 = None
    _sink_tokens_default = smith.ops.prims._sink_tokens.default([getitem]);  getitem = _sink_tokens_default = None
    return (addmm_1,)""",  # noqa: B950
                )

        recorded_list.clear()
        out2 = smith.compile(model)(x)
        self.assertEqual(len(recorded_list), 4)
        self.assertTrue(smith.allclose(model(x)[0], out2[0], atol=1e-7, rtol=1e-4))

    @skipIfSmithDynamo()
    def test_effect_autograd_function(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as m:

            @smith.library.custom_op("mylib::log_grad", mutates_args=())
            def log_grad(x: smith.Tensor) -> smith.Tensor:
                return x.clone()

            @smith.library.register_fake("mylib::log_grad")
            def log_grad_fake(x: smith.Tensor) -> smith.Tensor:
                return x.clone()

            log_grad.register_effect(_EffectType.ORDERED)

            class NoOpWithLoggingBackward(smith.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x * x

                @staticmethod
                def backward(ctx, grad_output):
                    logged_grad = smith.ops.mylib.log_grad(grad_output)
                    return logged_grad

            def fn(x):
                y = NoOpWithLoggingBackward.apply(x)
                return y.sum()

            x = smith.randn(3, 4, requires_grad=True)
            x_clone = x.detach().clone().requires_grad_(True)

            backend = AotEagerAndRecordGraphs()
            compiled_fn = smith.compile(fn, backend=backend)
            loss = compiled_fn(x)
            loss.backward()

            loss_ref = fn(x_clone)
            loss_ref.backward()
            self.assertEqual(loss, loss_ref)

            self.assertExpectedInline(
                backend.fw_graphs[0].code.strip(),
                """\
def forward(self, primals_1):
    mul = smith.ops.aten.mul.Tensor(primals_1, primals_1);  primals_1 = None
    sum_1 = smith.ops.aten.sum.default(mul);  mul = None
    return (sum_1,)""",  # noqa: B950
            )

            self.assertExpectedInline(
                backend.bw_graphs[0].code.strip(),
                """\
def forward(self, tangents_1, tangents_token):
    expand = smith.ops.aten.expand.default(tangents_1, [3, 4]);  tangents_1 = None
    with_effects = smith.ops.higher_order.with_effects(tangents_token, smith.ops.mylib.log_grad.default, expand);  tangents_token = expand = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    return (getitem_1, getitem)""",  # noqa: B950
            )

    def test_with_effects_through_functional_tensor_mode(self):
        from smith._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def fn_with_effects(x, y):
            token = smith.ops.prims._make_token()
            new_token, result = with_effects(
                token,
                smith.ops.aten.add.Tensor,
                x,
                y,
            )
            return result

        x = smith.randn(3, 3)
        y = smith.randn(3, 3)

        with (
            smith._C._ExcludeDispatchKeyGuard(
                smith._C.DispatchKeySet(smith._C.DispatchKey.Functionalize)
            ),
            FunctionalTensorMode(),
        ):
            x_func = FunctionalTensor.to_functional(x)
            y_func = FunctionalTensor.to_functional(y)
            result = fn_with_effects(x_func, y_func)

        expected = x + y
        if isinstance(result, FunctionalTensor):
            result = smith._from_functional_tensor(result.elem)
        self.assertEqual(result, expected)

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(not SM80OrLater, "triton")
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_effectful_op_with_flex_attention(self):
        """Test that effectful custom ops work with flex_attention."""
        from smith._library.effects import EffectType
        from smith.nn.attention.flex_attention import flex_attention

        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:

            @smith.library.custom_op("mylib::noop", mutates_args=())
            def noop(x: smith.Tensor) -> smith.Tensor:
                return x.clone()

            @noop.register_fake
            def noop_fake(x: smith.Tensor) -> smith.Tensor:
                return x.clone()

            noop.register_effect(EffectType.ORDERED)

            def score_mod(score, b, h, q_idx, kv_idx):
                return score

            def fn(q, k, v):
                q = smith.ops.mylib.noop(q)
                out = flex_attention(q, k, v, score_mod=score_mod)
                return out

            batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
            q = smith.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=smith.float16,
            )
            k = smith.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=smith.float16,
            )
            v = smith.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=smith.float16,
            )

            compiled_fn = smith.compile(fn)
            out = compiled_fn(q, k, v)
            self.assertEqual(out.shape, (batch_size, num_heads, seq_len, head_dim))


if __name__ == "__main__":
    run_tests()
