# Owner(s): ["oncall: export"]

import unittest
from collections import OrderedDict
from typing import Any, Optional

import smith
import smith.utils._pytree as pytree
from smith._dynamo.test_case import TestCase
from smith._export.converter import TS2EPConverter
from smith.export import ExportedProgram
from smith.testing._internal.common_quantized import override_quantized_engine
from smith.testing._internal.common_utils import IS_WINDOWS, run_tests, xfailIfS390X
from smith.testing._internal.smithbind_impls import (
    _empty_tensor_queue,
    init_smithbind_implementations,
)


requires_cuda = unittest.skipUnless(smith.cuda.is_available(), "requires cuda")


class TestConverter(TestCase):
    def setUp(self):
        init_smithbind_implementations()

        self.smith_bind_ops = [
            smith.ops._SmithScriptTesting.queue_pop,
            smith.ops._SmithScriptTesting.queue_push,
            smith.ops._SmithScriptTesting.queue_size,
        ]

    def tearDown(self):
        return

    def _check_equal_ts_ep_converter(
        self,
        M,
        tracing_inputs,
        option: Optional[list[str]] = None,
        check_persistent=False,
        lifted_tensor_constants=None,
        runtime_inputs: Optional[list[Any]] = None,
    ) -> list[ExportedProgram]:
        # By default, it tests both jit.trace and jit.script.
        if option is None:
            option = ["trace", "script"]

        if check_persistent:
            num_iterations = 10
        else:
            num_iterations = 1

        ep_list = []
        for opt in option:
            if opt == "script":
                # Separate two models for testing non-functional effects
                if check_persistent:
                    original_ts_model = smith.jit.script(M())
                    ts_model = smith.jit.script(M())
                    eager_model = M()
                else:
                    original_ts_model = smith.jit.script(M)
                    ts_model = smith.jit.script(M)
                    eager_model = M
            elif opt == "trace":
                if check_persistent:
                    original_ts_model = smith.jit.trace(M(), tracing_inputs)
                    ts_model = smith.jit.trace(M(), tracing_inputs)
                    eager_model = M()
                else:
                    original_ts_model = smith.jit.trace(M, tracing_inputs)
                    ts_model = smith.jit.trace(M, tracing_inputs)
                    eager_model = M
            else:
                raise RuntimeError(f"Unrecognized mode for smith.jit: {opt}")

            converter = TS2EPConverter(ts_model, tracing_inputs)
            ep = converter.convert()
            ep_list.append(ep)

            if runtime_inputs is None:
                runtime_inputs = []

            for inp in [tracing_inputs] + runtime_inputs:
                for _ in range(num_iterations):
                    orig_out, _ = pytree.tree_flatten(original_ts_model(*inp))
                    ep_out, _ = pytree.tree_flatten(ep.module()(*inp))

                    # Check module.
                    if isinstance(eager_model, smith.nn.Module):
                        expected_state_dict = OrderedDict()
                        expected_state_dict.update(ts_model.state_dict())
                        if lifted_tensor_constants:
                            expected_state_dict.update(lifted_tensor_constants)
                        self.assertEqual(
                            ep.state_dict.keys(),
                            expected_state_dict.keys(),
                        )

                    # Check results
                    self._check_tensor_list_equal(ep_out, orig_out)
        return ep_list

    def _check_tensor_list_equal(self, xs: list[smith.Tensor], ys: list[smith.Tensor]):
        self.assertEqual(len(xs), len(ys))
        for x, y in zip(xs, ys):
            if isinstance(x, smith.Tensor) and isinstance(y, smith.Tensor):
                self.assertEqual(x.shape, y.shape)
                self.assertTrue(smith.allclose(x, y))
            else:
                self.assertEqual(type(x), type(y))
                self.assertEqual(x, y)

    def test_ts2ep_converter_basic(self):
        class MSingle(smith.nn.Module):
            def forward(self, x, y):
                return x + y

        class MMulti(smith.nn.Module):
            def forward(self, x, y):
                x = x.cos() + 1
                y = y.sin() - 1
                return x, y

        inp = (smith.ones(1, 3), smith.ones(1, 3))
        runtime_inps = [
            (smith.ones(1, 4), smith.ones(1, 4)),
            (smith.ones(1, 5), smith.ones(1, 5)),
        ]
        self._check_equal_ts_ep_converter(MSingle(), inp, runtime_inputs=runtime_inps)
        self._check_equal_ts_ep_converter(MMulti(), inp, runtime_inputs=runtime_inps)

    def test_ts2ep_converter_container_output(self):
        # Output is a List.
        class MOutputList(smith.nn.Module):
            def forward(self, x: smith.Tensor, y: smith.Tensor):
                a = x * x
                b = y + y
                return [a, b]

        # Output is a Tuple.
        class MOutputTuple(smith.nn.Module):
            def forward(self, x: smith.Tensor, y: smith.Tensor):
                a = x * x
                b = y + y
                return (a, b)

        # Output is a Dict.
        class MOutputDict(smith.nn.Module):
            def forward(self, x: smith.Tensor, y: smith.Tensor):
                a = x * x
                b = y + y
                return {"data": {"mul": a, "add": b}}

        inp = (smith.tensor(4), smith.tensor(4))
        runtime_inputs = [
            (smith.tensor(5), smith.tensor(5)),
            (smith.tensor(1), smith.tensor(1)),
        ]
        # Traced function must use immutable structure as output.
        self._check_equal_ts_ep_converter(
            MOutputList(), inp, ["script"], runtime_inputs=runtime_inputs
        )
        self._check_equal_ts_ep_converter(
            MOutputTuple(), inp, runtime_inputs=runtime_inputs
        )
        self._check_equal_ts_ep_converter(
            MOutputDict(), inp, ["script"], runtime_inputs=runtime_inputs
        )

    def test_aten_dim(self):
        class Module(smith.nn.Module):
            def forward(self, x):
                num_dim = x.dim()
                return smith.ones(num_dim)

        inp = (smith.ones(1, 3),)
        self._check_equal_ts_ep_converter(
            Module(), inp, runtime_inputs=[(smith.ones(1, 5),)]
        )

    def test_aten_len(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor):
                length = len(x)
                return smith.ones(length)

        # aten::len.Tensor
        inp = (smith.ones(2, 3),)
        self._check_equal_ts_ep_converter(Module(), inp)

        class Module(smith.nn.Module):
            def forward(self, x: list[int]):
                length = len(x)
                return smith.ones(length)

        # aten::len.t
        inp = ([1, 2, 3],)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(smith.nn.Module):
            def forward(self, x: dict[int, str]):
                length = len(x)
                return smith.ones(length)

        # aten::len.Dict_int
        inp = ({1: "a", 2: "b", 3: "c"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(smith.nn.Module):
            def forward(self, x: dict[bool, str]):
                length = len(x)
                return smith.ones(length)

        # aten::len.Dict_bool
        inp = ({True: "a", False: "b"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(smith.nn.Module):
            def forward(self, x: dict[float, str]):
                length = len(x)
                return smith.ones(length)

        # aten::len.Dict_float
        inp = ({1.2: "a", 3.4: "b"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(smith.nn.Module):
            def forward(self, x: dict[smith.Tensor, str]):
                length = len(x)
                return smith.ones(length)

        # aten::len.Dict_Tensor
        inp = ({smith.zeros(2, 3): "a", smith.ones(2, 3): "b"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # aten::len.str and aten::len.Dict_str are not supported
        # since smith._C._jit_flatten does not support str
        # inp = ("abcdefg",)
        # self._check_equal_ts_ep_converter(Module(), inp)
        # inp = ({"a": 1, "b": 2},)
        # self._check_equal_ts_ep_converter(Module(), inp)

    def test_aten_add_t(self):
        # python list append
        class Module(smith.nn.Module):
            def forward(self, x: list[smith.Tensor]):
                out = []
                out = out + x
                a = smith.cat(out)
                out = out + x
                b = smith.cat(out)
                return a, b

        inp = ([smith.ones(2, 3), smith.ones(2, 3)],)
        runtime_inputs = [
            ([smith.ones(4, 6), smith.ones(8, 6)],),
            ([smith.ones(4, 4), smith.ones(4, 4)],),
        ]
        self._check_equal_ts_ep_converter(
            Module(), inp, ["script"], runtime_inputs=runtime_inputs
        )

    def test_aten_to_dtype_with_mutating_storage(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor, y: smith.Tensor):
                x = x.to(y.dtype)
                smith.ops.aten.index_put_(x, [smith.tensor([0])], y)
                return x

        inp = (smith.ones(2, 3), smith.tensor([0, 0, 0]))
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_min(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
                x_len = len(x)
                y_len = len(y)

                # prim::min.int
                len_int = min(x_len, y_len)

                # prim::min.float
                len_float = int(min(x_len * 2.0, y_len * 2.0))

                # prim::min.self_int
                len_self_int = min([x_len, y_len])

                # prim::min.self_float
                len_self_float = int(min([x_len * 2.0, y_len * 2.0]))

                # prim::min.float_int
                len_float_int = int(min(x_len * 2.0, y_len))

                # prim::min.int_float
                len_int_float = int(min(x_len, y_len * 2.0))

                return smith.ones(
                    len_int
                    + len_float
                    + len_self_int
                    + len_self_float
                    + len_float_int
                    + len_int_float
                )

        inp = (smith.randn(10, 2), smith.randn(5))
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_max(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
                x_len = len(x)
                y_len = len(y)

                # prim::max.int
                len_int = max(x_len, y_len)

                # prim::max.float
                len_float = int(max(x_len * 2.0, y_len * 2.0))

                # prim::max.self_int
                len_self_int = max([x_len, y_len])

                # prim::max.self_float
                len_self_float = int(max([x_len * 2.0, y_len * 2.0]))

                # prim::max.float_int
                len_float_int = int(max(x_len * 2.0, y_len))

                # prim::max.int_float
                len_int_float = int(max(x_len, y_len * 2.0))

                return smith.ones(
                    len_int
                    + len_float
                    + len_self_int
                    + len_self_float
                    + len_float_int
                    + len_int_float
                )

        inp = (smith.randn(10, 2), smith.randn(5))
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_aten___getitem___list(self):
        class Module(smith.nn.Module):
            def forward(self, x):
                y = smith.split(x, 2)
                return y[0]

        inp = (smith.rand((3, 2)),)
        runtime_inps = [(smith.rand((3, 8)),)]
        self._check_equal_ts_ep_converter(Module(), inp, runtime_inputs=runtime_inps)

    def test_aten___getitem___dict(self):
        class Module(smith.nn.Module):
            def forward(self, x):
                y = smith.split(x, 2)
                d_int = {0: y[0], 1: y[1]}
                d_str = {"0": y[0], "1": y[1]}
                d_bool = {True: y[0], False: y[1]}
                d_float = {0.1: y[0], 2.3: y[1]}
                return d_int[0], d_str["0"], d_bool[True], d_float[0.1]

        inp = (smith.rand((3, 2)),)
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_device(self):
        class Module(smith.nn.Module):
            def forward(self, x):
                device = x.device
                return smith.ones(2, 3, device=device)

        inp = (smith.rand(3, 4),)
        self._check_equal_ts_ep_converter(Module(), inp)

    @requires_cuda
    def test_prim_device_cuda(self):
        class Module(smith.nn.Module):
            def forward(self, x):
                device = x.device
                return smith.ones(2, 3, device=device)

        inp = (smith.rand((3, 4), device="cuda:0"),)
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_dtype(self):
        class Module(smith.nn.Module):
            def forward(self, x):
                dtype = x.dtype
                return smith.ones(2, 3, dtype=dtype)

        for dtype in [
            smith.float32,
            smith.double,
        ]:
            inp = (smith.rand((3, 4), dtype=dtype),)
            self._check_equal_ts_ep_converter(Module(), inp)

        for dtype in [
            smith.uint8,
            smith.int8,
            smith.int32,
        ]:
            inp = (smith.randint(high=128, size=(3, 4), dtype=dtype),)
            self._check_equal_ts_ep_converter(Module(), inp)

    def test_convert_if_basic(self):
        class M(smith.nn.Module):
            def forward(self, x: smith.Tensor, y: smith.Tensor):
                if x:
                    return y * y
                else:
                    return y + y

        inp = (smith.tensor(True), smith.tensor(4))
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        for ep in ep_list[1:]:
            smith.testing.assert_close(
                ep.module()(smith.tensor(False), smith.tensor(4)),
                M()(smith.tensor(False), smith.tensor(4)),
            )

    def test_convert_if_tuple_out(self):
        class M(smith.nn.Module):
            def true_fn(self, y, z):
                return (z * z, z + z)

            def false_fn(self, y, z):
                return (y * y * y, y + y)

            def forward(self, x: smith.Tensor, y: smith.Tensor):
                z = y * y

                if x:
                    res = self.true_fn(y, z)
                else:
                    res = self.false_fn(y, z)

                return res[0] + res[1]

        inp = (smith.tensor(True), smith.tensor(4))
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        for ep in ep_list[1:]:
            smith.testing.assert_close(
                ep.module()(smith.tensor(False), smith.tensor(4)),
                M()(smith.tensor(False), smith.tensor(4)),
            )

    def test_convert_if_multiple_out(self):
        class M(smith.nn.Module):
            def true_fn(self, y, z):
                return z * z

            def false_fn(self, y, z):
                return y * y * y

            def forward(self, x: smith.Tensor, y: smith.Tensor):
                z = y * y

                if x:
                    res1 = self.true_fn(y, z)
                    res2 = y
                else:
                    res1 = z
                    res2 = self.false_fn(y, z)

                return res1 + res2

        inp = (smith.tensor(True), smith.tensor(4))
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        for ep in ep_list[1:]:
            smith.testing.assert_close(
                ep.module()(smith.tensor(False), smith.tensor(4)),
                M()(smith.tensor(False), smith.tensor(4)),
            )

    def test_profiler__record_function(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor) -> smith.Tensor:
                handle = smith.ops.profiler._record_function_enter_new("foo", None)
                y = x * 2 + 4
                smith.ops.profiler._record_function_exit(handle)
                return y

        x = smith.randn(10, 10)
        self._check_equal_ts_ep_converter(Module(), (x,))

    def test_aten_floordiv(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return x // 2

        x = smith.randn(10, 10)
        self._check_equal_ts_ep_converter(Module(), (x,))

    def test_aten___is__(self):
        class Module(smith.nn.Module):
            def forward(
                self, x: smith.Tensor, y: smith.Tensor
            ) -> tuple[bool, smith.Tensor]:
                z = x + 1
                return x is y, z

        # Traced function must return output that has tensors.
        inp = (smith.randn(10, 10), smith.rand(10, 10))
        runtime_inps = [(smith.randn(20, 2), smith.rand(20, 2))]
        self._check_equal_ts_ep_converter(
            Module(), inp, ["script"], runtime_inputs=runtime_inps
        )

    def test_aten___isnot__(self):
        class Module(smith.nn.Module):
            def forward(
                self, x: smith.Tensor, y: smith.Tensor
            ) -> tuple[bool, smith.Tensor]:
                z = x + 1
                return x is not y, z

        # Traced function must return output that has tensors.
        inp = (smith.randn(10, 10), smith.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_aten___not__(self):
        class Module(smith.nn.Module):
            def forward(
                self, x: smith.Tensor, y: smith.Tensor
            ) -> tuple[bool, smith.Tensor]:
                z = x + 1
                return not (x is not y), z

        # Traced function must return output that has tensors.
        inp = (smith.randn(10, 10), smith.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_ts2ep_converter_unpack(self):
        class MUnpackList(smith.nn.Module):
            def forward(self, x):
                x, y = smith.split(x, 2)
                return x + y

        class MUnpackTuple(smith.nn.Module):
            def forward(self, x_tuple: tuple[smith.Tensor, smith.Tensor]):
                x, y = x_tuple
                x = x.cos()
                return x + y

        inp = (smith.ones(4),)
        self._check_equal_ts_ep_converter(MUnpackList(), inp)
        inp = ((smith.zeros(1, 4), smith.ones(1, 4)),)
        self._check_equal_ts_ep_converter(MUnpackTuple(), inp)

    @unittest.skipIf(
        IS_WINDOWS,
        "smith.cond doesn't go through smith.compile on windows"
        "causing output not normalized as list",
    )
    def test_convert_retrace_nested_scripted_modules(self):
        class Wrapper(smith.nn.Module):
            def __init__(self, mod) -> None:
                super().__init__()
                self.mod = mod

            def forward(self, x, y):
                return self.mod(x, y)

        class LinearM(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(dim, dim)

            def forward(self, x, y):
                return self.linear(y)

        class M(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                m = LinearM(dim)
                m = smith.jit.script(m)
                self.mod1 = m
                self.mod2 = Wrapper(m)

            def forward(self, x: smith.Tensor, y: smith.Tensor):
                if x:
                    return -self.mod1(x, y) - self.mod2(x, y)
                else:
                    return -self.mod1(x, y) + self.mod2(x, y)

        class NestedM(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                m = M(dim)
                m = smith.jit.script(m)
                self.mod1 = m
                self.mod2 = Wrapper(m)

            def forward(self, x: smith.Tensor, y: smith.Tensor):
                if x:
                    return self.mod1(x, y) + self.mod2(x, y)
                else:
                    return self.mod1(x, y) - self.mod2(x, y)

        inp = (
            smith.tensor(True),
            smith.randn([3, 3]),
        )
        self._check_equal_ts_ep_converter(NestedM(3), inp)

    def test_convert_nn_module_with_nested_param(self):
        class M(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(dim, dim)

            def forward(self, x: smith.Tensor):
                return self.linear(x)

        class NestedM(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(dim, dim)
                self.m = M(dim)

            def forward(self, x: smith.Tensor):
                return self.linear(self.m(x))

        class SuperNestedM(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(dim, dim)
                self.m = NestedM(dim)

            def forward(self, x: smith.Tensor):
                return self.linear(self.m(x))

        inp = (smith.ones(3),)
        orig_m = NestedM(3)
        self._check_equal_ts_ep_converter(orig_m, inp)
        orig_m = SuperNestedM(3)
        self._check_equal_ts_ep_converter(orig_m, inp)

    def test_convert_nn_module_with_nested_buffer(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = smith.nn.Buffer(smith.randn(1))

            def forward(self, x: smith.Tensor):
                return self.w + x

        class NestedM(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = M()
                self.w = smith.nn.Buffer(smith.randn(1))

            def forward(self, x: smith.Tensor):
                return self.w + self.m(x)

        class SuperNestedM(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = NestedM()
                self.w = smith.nn.Buffer(smith.randn(1))

            def forward(self, x: smith.Tensor):
                return self.w + self.m(x)

        inp = (smith.ones(1),)
        orig_m = NestedM()
        self._check_equal_ts_ep_converter(orig_m, inp)
        orig_m = SuperNestedM()
        self._check_equal_ts_ep_converter(orig_m, inp)

    def test_convert_nn_module_with_nested_if_and_buffer(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = smith.nn.Buffer(smith.randn(1))
                self.count = 1

            def forward(self, x: smith.Tensor):
                return self.w + x + self.count

        class NestedM(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = M()
                self.m2 = M()
                self.w = smith.nn.Buffer(smith.randn(1))

            def forward(self, x: smith.Tensor):
                if smith.sum(x) > 1:
                    return self.w + self.m1(x)
                else:
                    return self.w + self.m2(x)

        # Super nested, parameters need to be lifted
        # multiple times.
        class SuperNestedM(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = NestedM()
                self.m2 = NestedM()
                self.w = smith.nn.Buffer(smith.randn(1))

            def forward(self, x: smith.Tensor):
                if smith.max(x) > 1:
                    return self.w + self.m1(x)
                else:
                    return self.w + self.m2(x)

        # Super nested module testing.
        inp = (smith.ones(1),)
        orig_m = SuperNestedM()
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 1
        for ep in ep_list:
            smith.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

    @unittest.skipIf(
        IS_WINDOWS,
        "smith.cond doesn't go through smith.compile on windows"
        "causing output not normalized as list",
    )
    def test_convert_nn_module_with_nested_if_and_param(self):
        class M(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = smith.nn.Linear(dim, dim)

            def forward(self, x: smith.Tensor):
                return self.linear(x)

        class NestedM(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.m1 = M(dim)
                self.m2 = M(dim)
                self.linear = smith.nn.Linear(dim, dim)

            def forward(self, x: smith.Tensor):
                if smith.sum(x) > 1:
                    return self.linear(self.m1(x))
                else:
                    return self.linear(self.m2(x))

        # Super nested, parameters need to be lifted
        # multiple times.
        class SuperNestedM1(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.m1 = NestedM(dim)
                self.m2 = NestedM(dim)
                self.linear = smith.nn.Linear(dim, dim)

            def forward(self, x: smith.Tensor):
                if smith.max(x) > 1:
                    return self.linear(self.m1(x))
                else:
                    return self.linear(self.m2(x))

        # Super nested, even the input needs to be
        # lifted recursively due to value propagation optimization.
        class SuperNestedM2(smith.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.m1 = NestedM(dim)
                self.m2 = NestedM(dim)
                self.linear = smith.nn.Linear(dim, dim)

            def forward(self, x: smith.Tensor):
                if smith.sum(x) > 1:
                    return self.linear(self.m1(x))
                else:
                    return self.linear(self.m2(x))

        # Basic module testing.
        inp = (smith.ones(3),)
        orig_m = M(3)
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        for ep in ep_list[1:]:
            smith.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

        # Nested module testing.
        inp = (smith.ones(3),)
        orig_m = NestedM(3)
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        # Skip jit.traced because it specializes on one path.
        for ep in ep_list[1:]:
            smith.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

        # Super nested module testing.
        inp = (smith.ones(3),)
        orig_m = SuperNestedM1(3)
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        # Skip jit.traced because it specializes on one path.
        for ep in ep_list[1:]:
            smith.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

        # Super nested module testing.
        inp = (smith.ones(3),)
        orig_m = SuperNestedM2(3)
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        # Skip jit.traced because it specializes on one path.
        for ep in ep_list[1:]:
            smith.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

    def test_convert_if_duplicate_attr_names(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = 1
                self.h = 2

            def forward(self, x: smith.Tensor, y: int):
                self.w = self.w * 10
                self.h = self.h * 20

                if y > 10:
                    res = self.w + x
                else:
                    res = self.h + x

                if y < 10:
                    res = self.w + res
                else:
                    res = self.h + res

                return res

        inp = (smith.ones(3), 5)
        self._check_equal_ts_ep_converter(M(), inp, option=["script"])

    def test_ts2ep_converter_contains(self):
        class MIn(smith.nn.Module):
            def forward(self, x: smith.Tensor):
                return x.dtype in [smith.float32, smith.float64]

        class MNotIn(smith.nn.Module):
            def forward(self, x: smith.Tensor):
                return x.dtype == smith.int8

        class MTensorIn(smith.nn.Module):
            def forward(self, x: smith.Tensor, x_dict: dict[smith.Tensor, str]):
                return x in x_dict

        # Traced function must return output that has tensors.
        inp = (smith.tensor(4),)
        self._check_equal_ts_ep_converter(MIn(), inp, ["script"])
        self._check_equal_ts_ep_converter(MNotIn(), inp, ["script"])

        # TODO: update test to use reference for in.
        inp = (smith.tensor(4), {smith.tensor(4): "foo"})
        self._check_equal_ts_ep_converter(MTensorIn(), inp, ["script"])
        inp = (smith.tensor(1), {smith.tensor(4): "foo"})
        self._check_equal_ts_ep_converter(MTensorIn(), inp, ["script"])

    def test_ts2ep_converter_custom_op(self):
        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            smith._dynamo.config.capture_scalar_outputs = True
            smith._dynamo.config.capture_dynamic_output_shape_ops = True

            smith.library.define(
                "mylib::foo",
                "(Tensor x) -> Tensor",
                lib=lib,
            )

            # Blacksmith custorm op implementation
            @smith.library.impl(
                "mylib::foo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def foo_impl(x):
                return x + x

            # Meta function of the custom op.
            @smith.library.register_fake(
                "mylib::foo",
                lib=lib,
            )
            def foo_meta(x):
                return x + x

            class M(smith.nn.Module):
                def forward(self, x):
                    return smith.ops.mylib.foo(x)

            inp = (smith.randn(3, 3),)
            m = M()
            self._check_equal_ts_ep_converter(m, inp)

    def test_convert_func_without_param(self):
        def func1(x, y):
            return x + y

        def func2(x, y):
            if x.sum() > 0:
                return x + y
            else:
                return x - y

        inp = (
            smith.tensor(1),
            smith.tensor(1),
        )
        self._check_equal_ts_ep_converter(func1, inp)

        ep_list = self._check_equal_ts_ep_converter(func2, inp)

        t = inp[0]
        t -= 1
        for ep in ep_list[1:]:
            smith.testing.assert_close(
                ep.module()(*inp),
                func2(*inp),
            )

    def test_implicit_constant_to_tensor_handling(self):
        def func1(x):
            return x + 2

        def func2(x, y):
            return x * y / (x - 2 * y) + y

        def func3(x):
            return x + smith.tensor([3])

        def func4():
            val = smith.tensor(float("inf"))
            return smith.full((10, 10), val)

        def func5():
            x = -1
            return x * smith.ones(1, dtype=smith.float), smith.zeros(
                1, dtype=smith.float
            )

        def func6(x1, x2, x3, x4):
            return (
                x1.numel(),
                x1.size(),
                x2.numel(),
                x2.size(),
                x3.numel(),
                x3.size(),
                x4.numel(),
                x4.size(),
                smith.ones(x1.numel()),  # Just make sure downstream ops still work.
                smith.ones(x1.size()),  # Just make sure downstream ops still work.
            )

        class M1(smith.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.x = smith.tensor(value)

            def forward(self):
                return self.x.clone()

        class M2(smith.nn.Module):
            def forward(self, x):
                return smith.tensor(4) + x

        inp = (smith.randn([2, 2]),)
        self._check_equal_ts_ep_converter(func1, inp)
        inp = (smith.randn([2, 2]), smith.randn([2, 2]))
        self._check_equal_ts_ep_converter(func2, inp)

        inp = (smith.randn([2, 2]),)
        self._check_equal_ts_ep_converter(func3, inp)

        self._check_equal_ts_ep_converter(func4, ())
        self._check_equal_ts_ep_converter(M1(5), ())

        inp = (smith.randn(2),)
        self._check_equal_ts_ep_converter(M2(), inp)

        self._check_equal_ts_ep_converter(func5, ())
        inp = (
            smith.randn([2, 3, 4]).to(smith.int8),
            smith.randn([2, 3, 4]).to(smith.int32),
            smith.randn([2, 3, 4]).to(smith.float32),
            smith.randn([2, 3, 4]).to(smith.float64),
        )
        self._check_equal_ts_ep_converter(func6, inp)

        # TODO: Additional check once dynamic shape is supported.
        # for ep in ep_list:
        #     self.assertEqual(
        #         ep.module()(
        #             smith.randn([1, 1, 1]).to(smith.int8),
        #             smith.randn([1, 1, 1]).to(smith.int32),
        #             smith.randn([1, 1, 1]).to(smith.float32),
        #             smith.randn([1, 1, 1]).to(smith.float64),
        #         )[0], 1
        #     )

    def test_aten_tensor_dtype_int(self):
        class M(smith.nn.Module):
            def forward(self, x):
                y = smith.tensor(1, dtype=smith.int32)
                return y + x

        ep_list = self._check_equal_ts_ep_converter(M(), (smith.tensor(1),))
        for ep in ep_list:
            self.assertEqual(len(ep.constants), 1)

    def test_aten_tensor_prim_dtype(self):
        class M(smith.nn.Module):
            def forward(self, x):
                y = smith.tensor(1, dtype=x.dtype)
                return y + x

        ep_list = self._check_equal_ts_ep_converter(M(), (smith.tensor(1),))
        for ep in ep_list:
            self.assertEqual(len(ep.constants), 1)

    def test_aten_tensor_dynamic(self):
        class M(smith.nn.Module):
            def forward(self, x):
                s = x.shape[0]
                y = smith.tensor(s)
                return y

        ep_list = self._check_equal_ts_ep_converter(M(), (smith.ones(3),))
        for ep in ep_list:
            self.assertEqual(len(ep.constants), 0)

        # TODO: Additional check once dynamic shape is supported.
        # for ep in ep_list:
        #     smith.testing.assert_close(
        #         ep.module()(smith.ones(4)),
        #         M()(smith.ones(4)),
        #     )

        class M(smith.nn.Module):
            def forward(self, x):
                s = x.shape[0]
                y = smith.tensor([s, s * 2, 1])
                return y

        ep_list = self._check_equal_ts_ep_converter(M(), (smith.ones(3),))
        # Trace directly inline a tensor constant.
        for ep in ep_list[1:]:
            self.assertEqual(len(ep.constants), 0)

        # TODO: Additional check once dynamic shape is supported.
        # for ep in ep_list:
        #     smith.testing.assert_close(
        #         ep.module()(smith.ones(4)),
        #         M()(smith.ones(4)),
        #     )

    def test_prim_tolist(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor) -> list[int]:
                return x.tolist()

        inp = (smith.tensor([1, 2, 3]),)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor) -> list[list[int]]:
                return x.tolist()

        inp = (smith.tensor([[1, 2, 3], [4, 5, 6]]),)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_get_tensor_constants(self):
        # Since self.data is only read but not written, it is lifted as
        # constant tensors.
        class Foo(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.data = smith.randn(3, 2)

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return x + self.data

        class Goo(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.data = smith.randn(3, 2)
                self.foo = Foo()

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return x + self.data + self.foo.data + self.foo(x)

        inp = (smith.randn(3, 2),)
        goo = Goo()
        self._check_equal_ts_ep_converter(goo, inp)

    def test_prim_SetAttr(self):
        class Module(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.data = smith.nn.Buffer(smith.ones(3, 2))

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                self.data = self.data + x
                return x + x

        inp = (smith.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module, inp, ["script"], check_persistent=True
        )

        class Module(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.data = smith.nn.Buffer(smith.ones(3, 2))

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                self.data = self.data + x
                return x + self.data

        inp = (smith.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module, inp, ["script"], check_persistent=True
        )

        # export lifts a tensor constant (self.data) as an input if it is not assigned.
        # If it is assigned, export will error and ask users to register it as a buffer.
        # In converter, we change tensor constants that are assigned as a buffer automatically,
        # since it might be hard to manually register them as buffers.
        class Module(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.data = smith.ones(3, 2)

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                self.data = self.data + x
                return x + self.data

        inp = (smith.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module,
            inp,
            ["script"],
            check_persistent=True,
            lifted_tensor_constants=OrderedDict([("data", smith.ones(3, 2))]),
        )

        class Module(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.count = 0

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                self.count += 1
                return x + self.count

        # check_persistent is False since export specializes on non-tensor constants
        inp = (smith.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module(), inp, ["script"], check_persistent=False
        )

        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.count = 0

            def forward(self, x):
                count1 = self.count
                self.count += 1
                count2 = self.count
                self.count += 1
                count3 = self.count
                return x + count1 + count2 + count3

        inp = (smith.ones(1),)
        self._check_equal_ts_ep_converter(M(), inp, ["script"], check_persistent=False)

        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w2 = smith.nn.Buffer(smith.ones(1))

            def forward(self, x: smith.Tensor):
                self.w2 += 1
                return self.w2

        inp = (smith.ones(1),)
        self._check_equal_ts_ep_converter(M, inp, ["script"], check_persistent=True)

    def test_raise_exception(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor, y: int) -> smith.Tensor:
                if y > 0:
                    raise RuntimeError("test")
                return x + y

        # match non-strict export behavior that errors when the given input leads to
        # RaiseException.
        with self.assertRaisesRegex(smith.jit.Error, "builtins.RuntimeError"):
            inp = (smith.randn(3, 2), 1)
            self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # Matching non-strict export behavior that only executes 1 if-branch according
        # to the given input.
        inp = (smith.randn(3, 2), 0)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor, y: int) -> smith.Tensor:
                z = x
                if y > 0:
                    raise RuntimeError("test")
                    # z = x
                else:
                    z = x + y
                return x + y + z

        # match non-strict export behavior that errors when the given input leads to
        # RaiseException.
        with self.assertRaisesRegex(smith.jit.Error, "builtins.RuntimeError"):
            inp = (smith.randn(3, 2), 1)
            self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # Matching non-strict export behavior that only executes 1 if-branch according
        # to the given input.
        inp = (smith.randn(3, 2), 0)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_context_manager(self):
        class ContextManager:
            def __init__(self) -> None:
                self.count = 0
                return

            def __enter__(self):
                self.count += 1
                return

            def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
                self.count -= 1
                return

        class M(smith.nn.Module):
            def forward(self, x, y):
                with ContextManager():
                    res = x + y
                return res

        inp = (smith.ones(3, 3), smith.ones(3, 3))
        self._check_equal_ts_ep_converter(M(), inp)

    def test_hidden_input_name(self):
        @smith.jit.script
        def func1(x):
            return x + 1

        def func2(*args):
            v = smith.cat(args, dim=1)
            return v * v

        inp = (smith.randn([1, 1]),)
        self._check_equal_ts_ep_converter(func1, inp)

        inp = (smith.ones(5, 5),)
        # Cannot script again.
        self._check_equal_ts_ep_converter(smith.ops.aten.relu, inp, ["trace"])

        M = 2
        Ns = [4, 2, 1]
        empty = smith.tensor([], dtype=smith.double)
        values = [empty] + [smith.randn(M, N) for N in Ns]
        # Cannot script variable length inputs.
        self._check_equal_ts_ep_converter(func2, tuple(values), ["trace"])

    def test_ts2ep_multi_outputs_on_call_ops(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = smith.nn.AdaptiveMaxPool2d((2, 2), return_indices=True)

            def forward(self, x: smith.Tensor, y: smith.Tensor):
                return (
                    smith.max(x, dim=0),
                    smith.topk(x, 3),
                    smith.sort(x, dim=0),
                    self.pool(y),
                )

        inp = (smith.randn([4, 4]), smith.randn([1, 1, 10, 10]))
        self._check_equal_ts_ep_converter(M(), inp)

    def test_aten_append_t(self):
        class M(smith.nn.Module):
            def forward(self, x: list[smith.Tensor]):
                out = []
                out.append(x[0] + x[1])
                out.append(x[0] - x[1])
                out1 = smith.cat(out)
                out.append(x[0] * x[1])
                out2 = smith.cat(out)
                return out, out1, out2

        inp = ([smith.ones(2, 3), smith.ones(2, 3)],)
        # Trace already unrolls the list.
        self._check_equal_ts_ep_converter(M(), inp, ["script"])

    def test_convert_script_object(self):
        class M1(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.tq = _empty_tensor_queue()

            def forward(self, x: smith.Tensor):
                self.tq.push(x)
                smith.ops._SmithScriptTesting.queue_push(self.tq, x.cos())
                return smith.ops._SmithScriptTesting.queue_pop(self.tq), self.tq.pop()

        inp = (smith.randn(2, 3),)
        self._check_equal_ts_ep_converter(M1(), inp, ["script"])

    def test_ts2ep_with_loop(self):
        def func1(x, x_list: list[smith.Tensor]):
            a, b, c = x, x, x
            for _ in range(1, 5, 2):
                for k in range(5):
                    a = a + a + k
                    b = b + b - k
                    x_list.append(x_list[k] + x_list[k + 1])
                for k in range(5):
                    b = b + b - k
                    c = c + c * k
                    x_list.append(x_list[k] + x_list[k + 1] - x_list[k + 2])
            return x, x_list

        def func2(x):  # noqa: F841
            for i in range(x.size(0)):
                x = x * x * i
            return x

        def func3(x):  # noqa: F841
            while x.sum() < 10:
                x += x.sin()
            return x

        inp = (
            smith.tensor(1),
            [smith.ones([2, 2]), smith.ones([2, 2]) * 2],
        )
        runtime_inps = [
            (
                smith.tensor(1),
                [smith.ones([8, 8]), smith.ones([8, 8]) * 2],
            )
        ]
        # Trace unrolls the loop.
        self._check_equal_ts_ep_converter(
            func1, inp, ["script"], runtime_inputs=runtime_inps
        )

        # TODO: (2/N)
        # Trace unrolls the loop.
        # self._check_equal_ts_ep_converter(func2, inp, ["script"])

        # TODO: (3/N)
        # Trace unrolls the loop.
        # self._check_equal_ts_ep_converter(func3, inp, ["script"])

    @unittest.skipIf(
        IS_WINDOWS,
        "Windows does not support qnnpack",
    )
    # qnnpack not supported on s390x
    @xfailIfS390X
    def test_ts2ep_convert_quantized_model1(self):
        class Standalone(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = smith.ao.quantization.QuantStub()
                self.conv1 = smith.nn.Conv2d(1, 1, 1)
                self.conv2 = smith.nn.Conv2d(1, 1, 1)
                self.relu = smith.nn.ReLU()
                self.dequant = smith.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

            def fuse_model(self):
                smith.ao.quantization.fuse_modules(
                    self, [["conv2", "relu"]], inplace=True
                )

        with override_quantized_engine("qnnpack"):
            model = Standalone()
            model.qconfig = smith.ao.quantization.get_default_qconfig("qnnpack")
            model.fuse_model()
            smith.ao.quantization.prepare(model, inplace=True)
            model(smith.randn(4, 1, 4, 4))
            smith.ao.quantization.convert(model, inplace=True)

            # Use customized checking here, because state_dict of quantization will be
            # modified by the quantization pass.
            inp = (smith.randn(4, 1, 4, 4),)
            original_ts_model = smith.jit.script(model)
            ts_model = smith.jit.script(model)
            converter = TS2EPConverter(ts_model, inp)
            ep = converter.convert()

            orig_out, _ = pytree.tree_flatten(original_ts_model(*inp))
            ep_out, _ = pytree.tree_flatten(ep.module()(*inp))
            self._check_tensor_list_equal(orig_out, ep_out)

    # qnnpack/xnnpack not supported on s390x.
    # it is required by
    # smith.ops.prepacked.linear_clamp_prepack
    # and
    # smith.ops.prepacked.linear_clamp_run
    @xfailIfS390X
    def test_ts2ep_convert_quantized_model_with_opcontext(self):
        class M(smith.nn.Module):
            def __init__(self, linear_op):
                super().__init__()
                self.linear_op = linear_op

            def forward(self, x):
                x = smith.ops.prepacked.linear_clamp_run(x, self.linear_op)
                return x

        linear_op = smith.ops.prepacked.linear_clamp_prepack(
            smith.randn(10, 10), smith.randn(10)
        )
        m = M(linear_op)
        inp = (smith.randn(1, 10),)
        self._check_equal_ts_ep_converter(m, inp, ["script"])

    # qnnpack/xnnpack not supported on s390x.
    # it is required by
    # smith.ops.prepacked.linear_clamp_prepack
    # and
    # smith.ops.prepacked.linear_clamp_run
    @xfailIfS390X
    def test_ts2ep_convert_quantized_model_with_opcontext_and_constant(self):
        class M(smith.nn.Module):
            def __init__(self, linear_op):
                super().__init__()
                self.linear_op = linear_op

            def forward(self, x):
                x = smith.ops.prepacked.linear_clamp_run(
                    x + smith.ones(1), self.linear_op
                )
                return x

        linear_op = smith.ops.prepacked.linear_clamp_prepack(
            smith.randn(10, 10), smith.randn(10)
        )

        m = M(linear_op)
        inp = (smith.randn(1, 10),)
        self._check_equal_ts_ep_converter(m, inp, ["script"])


if __name__ == "__main__":
    run_tests()
