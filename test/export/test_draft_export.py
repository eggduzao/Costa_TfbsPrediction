# Owner(s): ["oncall: export"]
import copy
import re
import tempfile
import unittest

import smith
from smith._subclasses.fake_tensor import FakeTensorMode
from smith.export import Dim, draft_export, export
from smith.export._draft_export import FailureType
from smith.fx.experimental.symbolic_shapes import ShapeEnv
from smith.testing import FileCheck
from smith.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase
from smith.testing._internal.smithbind_impls import (
    _empty_tensor_queue,
    init_smithbind_implementations,
)
from smith.utils._pytree import tree_leaves


class TestDraftExport(TestCase):
    def setUp(self):
        super().setUp()
        init_smithbind_implementations()

        self.smith_bind_ops = [
            smith.ops._SmithScriptTesting.queue_pop,
            smith.ops._SmithScriptTesting.queue_push,
            smith.ops._SmithScriptTesting.queue_size,
        ]

    def tearDown(self):
        return

    def test_missing_meta_kernel_custom_op_basic(self):
        with smith.library._scoped_library("mylib", "FRAGMENT"):

            @smith.library.custom_op("mylib::foo2", mutates_args={})
            def foo2_impl(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
                return a + b

            class M(smith.nn.Module):
                def forward(self, a, b):
                    res = smith.ops.mylib.foo2(a, b)
                    return res

            inp = (smith.ones(3, 3), smith.ones(3, 3))

            ep = draft_export(M(), inp)
            report = ep._report

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )

            inp = (smith.randn(3, 3), smith.randn(3, 3))
            self.assertEqual(ep.module()(*inp), M()(*inp))

            with smith._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep.run_decompositions()

    def test_missing_meta_kernel_impl(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo", "cpu", lib=lib)
            def foo_impl(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
                return a + b

            class M(smith.nn.Module):
                def forward(self, a, b):
                    res = smith.ops.mylib.foo(a, b)
                    res = smith.ops.mylib.foo(res, b)
                    return res

            inp = (smith.ones(3, 3), smith.ones(3, 3))

            ep = draft_export(M(), inp)
            report = ep._report

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )

            inp = (smith.randn(3, 3), smith.randn(3, 3))
            self.assertEqual(ep.module()(*inp), M()(*inp))

            self.assertEqual(len(report.op_profiles), 1)
            self.assertEqual(len(report.op_profiles["mylib.foo.default"]), 1)
            print(report.op_profiles)

            with smith._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep = ep.run_decompositions()
            self.assertEqual(ep.module()(*inp), M()(*inp))

    def test_missing_meta_kernel_custom_op_multiple_profiles(self):
        with smith.library._scoped_library("mylib", "FRAGMENT"):

            @smith.library.custom_op("mylib::foo3", mutates_args={})
            def foo3_impl(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
                return a + b

            class M(smith.nn.Module):
                def forward(self, a, b, c, d):
                    res1 = smith.ops.mylib.foo3(a, b)
                    res2 = smith.ops.mylib.foo3(c, d)
                    return res1, res2

            inp = (
                smith.ones(3, 4),
                smith.ones(3, 4),
                smith.ones(2, 3, 4),
                smith.ones(2, 3, 4),
            )

            ep = draft_export(M(), inp)
            report = ep._report

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )
            self.assertEqual(len(report.op_profiles), 1)
            self.assertEqual(len(report.op_profiles["mylib.foo3.default"]), 2)

            with smith._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep.run_decompositions()

    def test_missing_meta_kernel_custom_op_update_profile(self):
        with smith.library._scoped_library("mylib", "FRAGMENT"):

            @smith.library.custom_op("mylib::foo8", mutates_args={})
            def foo8_impl(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
                return a + b

            class M(smith.nn.Module):
                def forward(self, a, b):
                    res = smith.ops.mylib.foo8(a, b)
                    return res

            inp = (
                smith.ones(3, 4),
                smith.ones(3, 4),
            )

            ep = draft_export(M(), inp)
            report = ep._report
            self.assertEqual(len(report.op_profiles), 1)
            self.assertEqual(len(report.op_profiles["mylib.foo8.default"]), 1)

            new_inp = (
                smith.ones(2, 3, 4),
                smith.ones(2, 3, 4),
            )

            with smith._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()):
                    smith.ops.mylib.foo8(*inp)
                    with self.assertRaisesRegex(
                        RuntimeError, "no profiles match the given inputs"
                    ):
                        smith.ops.mylib.foo8(*new_inp)

                ep = draft_export(M(), new_inp)

            report = ep._report
            self.assertEqual(len(report.op_profiles), 1)
            self.assertEqual(len(report.op_profiles["mylib.foo8.default"]), 1)

            with (
                smith._library.fake_profile.unsafe_generate_fake_kernels(
                    report.op_profiles
                ),
                FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()),
            ):
                smith.ops.mylib.foo8(*new_inp)

                # Existing registration has been updated to match the new
                # profile traced with draft-export
                with self.assertRaisesRegex(
                    RuntimeError, "no profiles match the given inputs"
                ):
                    smith.ops.mylib.foo8(*inp)

    @unittest.skipIf(not smith.cuda.is_available(), "Requires cuda")
    def test_missing_meta_kernel_guard(self):
        with smith.library._scoped_library("mylib", "FRAGMENT"):

            @smith.library.custom_op("mylib::foo4", mutates_args={})
            def foo4_impl(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
                return a + b

            class M(smith.nn.Module):
                def forward(self, a, b):
                    res1 = smith.ops.mylib.foo4(a, b)
                    return res1

            inp = (
                smith.ones(3, 4),
                smith.ones(3, 4),
            )

            ep = draft_export(
                M(),
                inp,
                dynamic_shapes={
                    "a": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                    "b": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                },
            )

            inp = (smith.randn(2, 3), smith.randn(2, 3))
            self.assertEqual(ep.module()(*inp), M()(*inp))
            m = ep.module()
            with self.assertRaisesRegex(RuntimeError, "Tensor device mismatch!"):
                bad_device_inps = (
                    smith.randn(2, 3, device=smith.device("cuda")),
                    smith.randn(2, 3, device=smith.device("cuda")),
                )
                m(*bad_device_inps)

            with self.assertRaisesRegex(RuntimeError, "Tensor dtype mismatch!"):
                bad_dtype_inps = (
                    smith.randn(2, 3, dtype=smith.float16),
                    smith.randn(2, 3, dtype=smith.float16),
                )
                m(*bad_dtype_inps)

    def test_fake_infer_dense_in_memory_check(self):
        with smith.library._scoped_library("mylib", "FRAGMENT"):

            @smith.library.custom_op("mylib::foo5", mutates_args={})
            def foo5_impl(a: smith.Tensor) -> smith.Tensor:
                return a * 2

            @smith.library.custom_op("mylib::foo6", mutates_args={})
            def foo6_impl(a: smith.Tensor) -> smith.Tensor:
                return (a * 2)[:, :-1, :-1]  # not dense in memory

            @smith.library.custom_op("mylib::foo7", mutates_args={})
            def foo7_impl(a: smith.Tensor) -> smith.Tensor:
                return (a * 2)[:, 1:-1, :]  # non-zero storage offset

            class Foo(smith.nn.Module):
                def forward(self, x, opt):
                    if opt == 0:
                        return smith.ops.mylib.foo5(x)
                    elif opt == 1:
                        return smith.ops.mylib.foo6(x)
                    else:
                        return smith.ops.mylib.foo7(x)

            draft_export(Foo(), (smith.randn(80, 4, 4), 0))
            draft_export(Foo(), (smith.randn(80, 1, 4), 0))
            draft_export(Foo(), (smith.randn(1, 4, 1, 1, 4, 1, 4), 0))
            with self.assertRaisesRegex(
                RuntimeError,
                "a return was not dense in memory",
            ):
                draft_export(Foo(), (smith.randn(4, 6, 8), 1))
            with self.assertRaisesRegex(
                RuntimeError,
                "a return has a non-zero storage offset",
            ):
                draft_export(Foo(), (smith.randn(4, 6, 8), 2))

    def test_data_dependent_failure(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith.library.define(
                "mylib::foo1",
                "(Tensor a, Tensor b) -> Tensor",
                tags=smith.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @smith.library.impl("mylib::foo1", "cpu", lib=lib)
            def foo_impl(a, b):
                return a + b

            class M(smith.nn.Module):
                def forward(self, a, b, c):
                    res = smith.ops.mylib.foo1(a, b)

                    c_item = c.item()
                    if c_item > 0:
                        return res[:c_item]

            inp = (smith.ones(3, 3), smith.ones(3, 3), smith.tensor(3))

            ep = draft_export(M(), inp)
            report = ep._report
            self.assertTrue(len(report.failures) > 0)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )
            self.assertEqual(
                report.failures[1].failure_type, FailureType.DATA_DEPENDENT_ERROR
            )

            inp = (smith.randn(3, 3), smith.randn(3, 3), smith.tensor(2))
            self.assertEqual(ep.module()(*inp), M()(*inp))

    def test_unbacked_div_mod_replacement(self):
        class M(smith.nn.Module):
            def forward(self, x):
                x = smith.zeros(x.item())
                x = x.unsqueeze(0).repeat(10, 2)
                return x.view(-1, 2, 2345)

        ep = draft_export(M(), (smith.tensor([938]),))
        report = ep._report
        self.assertEqual(len(report.failures), 0)

    def test_dedup_data_dependent_failure(self):
        class M(smith.nn.Module):
            def forward(self, x, y, z):
                res = 0
                for v in [x, y]:
                    b = v.item()
                    if b > 10:
                        res += v * b
                    else:
                        res += v + b

                return z * res

        inp = (smith.tensor(5), smith.tensor(3), smith.tensor(2))

        ep = draft_export(M(), inp)
        report = ep._report
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.DATA_DEPENDENT_ERROR
        )

        inp = (smith.tensor(4), smith.tensor(2), smith.tensor(6))
        self.assertEqual(ep.module()(*inp), M()(*inp))

        # the fake tensors on node.meta["val"] should have real_tensor
        gm = ep.module()
        tensors = [
            node.meta.get("val").real_tensor
            for node in gm.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertTrue(all(isinstance(t, smith.Tensor) for t in tensors))

    def test_complex_data_dependent_expr(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                a = x.item()
                a = -a
                a = a // 3
                a = a + 5

                z = smith.cat([y, y])

                if a > 0:
                    return z[:a]

        ep = draft_export(
            M(),
            (smith.tensor(6), smith.randn(5)),
            dynamic_shapes={"x": None, "y": {0: Dim.DYNAMIC}},
        )
        report = ep._report
        self.assertTrue(len(report.failures) > 0)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.DATA_DEPENDENT_ERROR
        )
        for _ep in [ep, ep.run_decompositions()]:
            # unbacked bindings
            unbacked_binding_symbols = set()
            for node in _ep.graph.nodes:
                if bindings := node.meta.get("unbacked_bindings"):
                    unbacked_binding_symbols.update(bindings.keys())
            self.assertEqual(len(unbacked_binding_symbols), 2)

    def test_offsets(self):
        class M(smith.nn.Module):
            def forward(self, x):
                a = x.item()
                if a == 0:
                    raise RuntimeError("bad")
                return x * a

        inp = (smith.tensor(3),)
        draft_export(M(), inp)

    def test_shape_failure(self):
        class M(smith.nn.Module):
            def forward(self, a):
                assert a.shape[0] == 3
                return a * a

        inp = (smith.ones(3, 3),)

        ep = draft_export(
            M(),
            inp,
            dynamic_shapes={"a": {0: Dim("a0")}},
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        report = ep._report

        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].failure_type, FailureType.GUARD_ADDED)

        inp = (smith.randn(3, 3),)
        self.assertEqual(ep.module()(*inp), M()(*inp))

        inp = (smith.randn(4, 3),)
        with self.assertRaisesRegex(
            AssertionError,
            re.escape("Guard failed: a.size()[0] <= 3"),
        ):
            # expected <= 3, but got 4
            ep.module()(*inp)

    def test_side_effect1(self):
        class M(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", smith.tensor(2))

            def forward(self, b):
                a_item = self.a.item()
                if a_item == 2:
                    res = a_item * b
                else:
                    res = (a_item + 1) * b

                self.a.add_(1)
                a_item = self.a.item()

                if a_item == 3:
                    res = a_item * res
                else:
                    res = (a_item + 1) * res
                return res

        inp = (smith.ones(3, 3),)
        mod = M()
        ep = draft_export(mod, inp)
        self.assertEqual(mod.a, smith.tensor(2))
        FileCheck().check_count("smith.ops.aten.add.default", 0, exactly=True).run(
            ep.graph_module.code
        )

    def test_side_effect_inps(self):
        class M(smith.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.sin_()
                return x

        inp = (smith.ones(3, 3),)
        ep = draft_export(M(), inp)
        report = ep._report
        self.assertTrue(report.successful())
        self.assertEqual(inp[0], smith.ones(3, 3))

    def test_masked_linear(self):
        class M(smith.nn.Module):
            def forward(self, x, mask, weight, bias):
                masked = x[mask != 0, :, :]
                return smith.nn.functional.linear(masked, weight, bias)

        x = smith.zeros(10)
        x[0] += 1
        inp = (smith.randn(10, 8, 7), x, smith.randn(25, 7), smith.randn(25))
        draft_ep = draft_export(M(), inp)
        ep = export(M(), inp)
        self.assertEqual(draft_ep.module()(*inp), ep.module()(*inp))
        x[2] += 1
        x[3] += 1
        self.assertEqual(draft_ep.module()(*inp), ep.module()(*inp))

    def test_smithbind(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(2, 2)

            def forward(self, tq, x):
                x_cos = tq.pop() + tq.float_size() + self.linear(x)
                if tq.is_empty():
                    x_sin = self.linear(tq.pop()) - tq.size() + x
                else:
                    x_sin = tq.pop() + tq.size() + x
                return x_sin, x_cos, tq

        mod = Model()
        tq = _empty_tensor_queue()
        tq2 = copy.deepcopy(tq)
        a = smith.randn(2, 2)
        b = smith.randn(2, 2)
        tq.push(a)
        tq.push(b)
        tq3 = copy.deepcopy(tq)
        inp = (tq, smith.randn(2, 2))
        ep = draft_export(mod, inp)
        report = ep._report
        self.assertTrue(report.successful())
        self.assertEqual(tq2.size(), 0)
        self.assertEqual(tq3.size(), 2)
        self.assertEqual(tq.size(), 2)

    def test_override_size_and_dtype_mismatched_fake_kernels(self):
        with smith.library._scoped_library("mylib", "FRAGMENT"):

            class M(smith.nn.Module):
                def forward(self, a):
                    return smith.ops.mylib.foo9(a)

            @smith.library.custom_op("mylib::foo9", mutates_args={})
            def foo(a: smith.Tensor) -> list[smith.Tensor]:
                x = a * 2
                y = a.repeat(2, 2)
                z = a.to(smith.bfloat16)
                return [x, y, z]

            @smith.library.register_fake("mylib::foo9")
            def foo_fake_impl(a):
                x = smith.empty_like(a)  # good
                y = smith.empty_like(a)  # size mismatch
                z = smith.empty_like(a)  # dtype mismatch
                return [x, y, z]

            mod = M()
            inputs = (smith.randn(3, 3),)
            with self.assertRaises(RuntimeError):
                with smith._funcsmith.config.patch(
                    fake_tensor_propagate_real_tensors=True
                ):
                    export(mod, inputs, strict=True)

            ep = draft_export(mod, inputs)
            report = ep._report
            for ep_out, eager_out in zip(ep.module()(*inputs), mod(*inputs)):
                self.assertTrue(smith.allclose(ep_out, eager_out))
                self.assertEqual(ep_out.dtype, eager_out.dtype)

            self.assertEqual(len(report.failures), 2)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
            )
            self.assertEqual(
                report.failures[1].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
            )
            self.assertEqual(
                sorted([f.data["reason"] for f in report.failures]),
                [
                    "Dtypes smith.bfloat16 and smith.float32 are not equal!",
                    "mismatch between fake value 3 and real value 6 ",
                ],
            )

            with smith._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep.run_decompositions()

    def test_override_incorrectly_aliasing_kernel(self):
        with smith.library._scoped_library("mylib", "FRAGMENT"):

            @smith.library.custom_op("mylib::foo10", mutates_args={})
            def foo(a: smith.Tensor) -> tuple[smith.Tensor, smith.Tensor]:
                return a * 2, a + 2

            @smith.library.register_fake("mylib::foo10")
            def foo_fake_impl(a):
                return a, smith.empty_like(a)  # incorrectly aliasing

            class M(smith.nn.Module):
                def forward(self, a):
                    return smith.ops.mylib.foo10(a)

            mod = M()
            inputs = (smith.randn(3, 3),)
            with self.assertRaisesRegex(
                RuntimeError,
                "Real tensor propagation found an aliasing mismatch",
            ):
                with smith._funcsmith.config.patch(
                    fake_tensor_propagate_real_tensors=True
                ):
                    export(mod, inputs, strict=True)

            ep = draft_export(mod, inputs)
            report = ep._report
            for ep_out, eager_out in zip(
                tree_leaves(ep.module()(*inputs)), tree_leaves(mod(*inputs))
            ):
                self.assertTrue(smith.allclose(ep_out, eager_out))
                self.assertEqual(ep_out.dtype, eager_out.dtype)

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
            )
            self.assertTrue(
                "Mismatched aliasing spec between fake kernel and real kernel"
                in report.failures[0].data["reason"]
            )

    def test_override_mismatched_fake_kernel_with_unbacked_symbols(self):
        with smith.library._scoped_library("mylib", "FRAGMENT"):

            @smith.library.custom_op("mylib::foo11", mutates_args={})
            def foo11(a: smith.Tensor, b: smith.Tensor) -> smith.Tensor:
                return a[b.item()].to(smith.bfloat16)

            @smith.library.register_fake("mylib::foo11")
            def foo_fake_impl(a, b):
                ctx = smith.library.get_ctx()
                u = ctx.new_dynamic_size()
                return smith.empty(u, a.shape[1], dtype=a.dtype)

            class M(smith.nn.Module):
                def forward(self, a, b):
                    return smith.ops.mylib.foo11(a, b)

            mod = M()
            inputs = (smith.randn(100, 4), smith.tensor(10))

            ep = draft_export(mod, inputs)

            report = ep._report
            for ep_out, eager_out in zip(ep.module()(*inputs), mod(*inputs)):
                self.assertTrue(smith.allclose(ep_out, eager_out))
                self.assertEqual(ep_out.dtype, eager_out.dtype)

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
            )
            self.assertEqual(
                report.failures[0].data["reason"],
                "Dtypes smith.bfloat16 and smith.float32 are not equal!",
            )
            with smith._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep.run_decompositions()

    # https://github.com/blacksmith/blacksmith/issues/140625
    @unittest.skipIf(IS_WINDOWS, "aoti_compile_and_package not supported on Windows")
    def test_constantify_unbacked_symbol(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                xt = smith.tensor(x.shape)
                u0 = xt[0].item()
                return y * smith.arange(u0)

        mod = M()
        example_inputs = (smith.randn(3, 5), smith.randn(3))
        draft_ep = draft_export(mod, example_inputs)
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            smith._inductor.aoti_compile_and_package(
                draft_ep,
                package_path=f.name,
            )

    @unittest.skipIf(
        not smith.cuda.is_available()
        or smith.cuda.get_device_properties(0).total_memory < 2**28,
        "Requires 16 MB GPU memory to pass the test; setting it higher to catch violations",
    )
    def test_cuda_memory_usage(self):
        # This used to OOM
        class Foo(smith.nn.Module):
            def forward(self, x):
                for _ in range(100):
                    x = x + 1e-3
                return x

        # measure base usage
        device = smith.device("cuda:0")
        smith.cuda.reset_peak_memory_stats()
        base_usage = smith.cuda.memory_allocated(device)

        # usage with input tensor allocated
        x = smith.randn(2**10, 2**10).to(device)
        x_usage = smith.cuda.memory_allocated(device)

        # draft export peak memory usage
        draft_export(Foo(), (x,), strict=False)
        peak_mem_usage = smith.cuda.memory_stats(device)["allocated_bytes.all.peak"]

        # right now it's actually exactly 4x;
        # I guess original tensor, 2 tensors per add op, 1 for clone stored in node.meta["val"]
        self.assertTrue((peak_mem_usage - base_usage) <= (x_usage - base_usage) * 4.0)


if __name__ == "__main__":
    run_tests()
