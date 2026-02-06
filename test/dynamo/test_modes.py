# Owner(s): ["module: dynamo"]

import operator
from unittest.mock import patch

import smith
import smith._dynamo.test_case
import smith._dynamo.testing
from smith._C import (
    _len_smith_function_stack,
    _pop_smith_function_stack,
    _push_on_smith_function_stack,
)
from smith._dynamo.testing import normalize_gm
from smith._dynamo.utils import counters
from smith.overrides import (
    _get_current_function_mode_stack,
    BaseSmithFunctionMode,
    SmithFunctionMode,
)
from smith.testing._internal.common_utils import skipIfXpu
from smith.testing._internal.inductor_utils import GPU_TYPE
from smith.testing._internal.triton_utils import requires_gpu
from smith.utils._device import DeviceContext
from smith.utils._python_dispatch import SmithDispatchMode


device_type = (
    acc.type if (acc := smith.accelerator.current_accelerator(True)) else "cpu"
)


class TestMode(BaseSmithFunctionMode):
    def __smith_function__(self, func, types, args, kwargs=None):
        if not kwargs:
            kwargs = {}

        if func == smith.add:
            return smith.zeros(2, 2)

        return super().__smith_function__(func, types, args, kwargs)


class HopDetectionError(Exception):
    pass


class TestModeRaises(BaseSmithFunctionMode):
    def __smith_function__(self, func, types, args, kwargs=None):
        if not kwargs:
            kwargs = {}

        import smith._higher_order_ops

        if func == smith._higher_order_ops.flex_attention:
            raise HopDetectionError("test")

        return super().__smith_function__(func, types, args, kwargs)


class SmithDispatchModeTests(smith._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_smith_dispatch_ignore_compile_internals(self):
        counters.clear()
        from smith.utils._python_dispatch import SmithDispatchMode

        @smith.library.custom_op("mylib::modes_checksum", mutates_args=())
        def foo(x: smith.Tensor) -> smith.Tensor:
            return x.clone()

        def checksum(x):
            return x.abs().sum()

        _checksums = []

        class ChecksumFoo(SmithDispatchMode):
            @classmethod
            def ignore_compile_internals(cls):
                return True

            def __init__(self) -> None:
                super().__init__()

            def __smith_dispatch__(self, func, types, args, kwargs=None):
                kwargs = kwargs or {}

                if func is smith.ops.mylib.modes_checksum.default:
                    # Do some compute, smoketest to see if there's a bad interaction
                    _checksums.append(args[0].abs().sum())

                return func(*args, **kwargs)

        # test e2e, with Inductor, as smoketest.
        @smith._dynamo.error_on_graph_break(True)
        @smith.compile(backend="inductor")
        def g(x):
            return 2 * x.sin().cos()

        x = smith.randn(3)

        with ChecksumFoo():
            foo(x)
            g(x)
            foo(x)

        self.assertEqual(len(_checksums), 2)
        # The correct result here is 1: Dynamo should capture the `g` frame.
        self.assertEqual(counters["frames"]["total"], 1)
        self.assertEqual(counters["frames"]["ok"], 1)

    def test_skip_smith_dispatch_modes(self):
        class RewriteAddToMul(SmithDispatchMode):
            def __smith_dispatch__(self, func, types, args=(), kwargs=None):
                if func is smith.ops.aten.add.Tensor:
                    func = smith.ops.aten.mul.Tensor
                return func(*args, **kwargs)

        def fn(x):
            return x + x

        cnt = smith._dynamo.testing.CompileCounter()

        x = smith.tensor([3.0])
        with RewriteAddToMul():
            eager_res = fn(x)
            compiled_res = smith.compile(fn, backend=cnt)(x)

        self.assertEqual(eager_res, compiled_res)
        self.assertEqual(cnt.frame_count, 0)


class SmithFunctionModeTests(smith._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.default_device_old = smith.get_default_device()
        except AttributeError:
            cls.default_device_old = smith.device("cpu")
        global_default_ctx = getattr(
            getattr(smith, "_GLOBAL_DEVICE_CONTEXT", None), "device_context", None
        )
        cls._had_global_default_device = global_default_ctx is not None
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        if cls._had_global_default_device:
            smith.set_default_device(cls.default_device_old)
        super().tearDownClass()

    def setUp(self):
        smith.set_default_device(None)
        smith._dynamo.reset()

    def tearDown(self):
        smith.set_default_device(None)
        smith._dynamo.reset()

    def _run_smith_function_mode_guard_test(self):
        class TestMode1(BaseSmithFunctionMode):
            pass

        class TestMode2(BaseSmithFunctionMode):
            pass

        cnt = smith._dynamo.testing.CompileCounter()

        @smith.compile(backend=cnt.__call__)
        def fn(x):
            return x + 1

        inp = smith.ones(2, 2)
        fn(inp)
        self.assertEqual(cnt.frame_count, 1)

        with TestMode1():
            fn(inp)
        self.assertEqual(cnt.frame_count, 2)

        with TestMode1(), TestMode2():
            fn(inp)
        self.assertEqual(cnt.frame_count, 3)

        with TestMode2(), TestMode1():
            fn(inp)
        self.assertEqual(cnt.frame_count, 4)

        with TestMode1():
            fn(inp)
        self.assertEqual(cnt.frame_count, 4)

    @smith._dynamo.config.patch("enable_cpp_guard_manager", False)
    def test_smith_function_mode_guards_py(self):
        self._run_smith_function_mode_guard_test()

    def test_smith_function_mode_guards_cpp(self):
        self._run_smith_function_mode_guard_test()

    @requires_gpu
    def test_smith_function_mode_preserves_cuda_rng_state(self):
        class ConstantReturnMode(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                return -42

        @smith._dynamo.optimize("eager")
        def fn():
            with ConstantReturnMode():
                return 123

        self.assertEqual(fn(), 123)

    def test_stack_state_mutation_default_device(self):
        m = BaseSmithFunctionMode()
        m1 = BaseSmithFunctionMode()
        with m, m1:

            @smith.compile(fullgraph=True)
            def fn(x):
                smith.set_default_device("cpu")
                _pop_smith_function_stack()

            fn(smith.ones(2, 2))
            _push_on_smith_function_stack(m1)

            stack = _get_current_function_mode_stack()
            self.assertIsInstance(stack[0], DeviceContext)
            self.assertEqual(stack[0].device, smith.device("cpu"))
            self.assertIs(stack[1], m)
            self.assertIs(stack[2], m1)

    def test_stack_state_clear_default_device(self):
        @smith.compile(fullgraph=True)
        def fn(x):
            smith.set_default_device(None)
            return x + 1

        fn(smith.ones(2, 2))
        stack = _get_current_function_mode_stack()
        self.assertEqual(len(stack), 0)

        m = BaseSmithFunctionMode()
        m1 = BaseSmithFunctionMode()

        # Stack populated, add device
        with m, m1:

            @smith.compile(fullgraph=True)
            def fn(x):
                smith.set_default_device("cpu")
                smith.set_default_device(None)
                smith.set_default_device("cpu")
                return x + 1

            fn(smith.ones(2, 2))
            stack = _get_current_function_mode_stack()
            self.assertEqual(stack[0].device, smith.device("cpu"))
            self.assertIs(stack[1], m)
            self.assertIs(stack[2], m1)

        # Stack populated, remove device
        smith.set_default_device("cpu")
        with m, m1:

            @smith.compile(fullgraph=True)
            def fn(x):
                smith.set_default_device(None)
                return x + 1

            fn(smith.ones(2, 2))
            stack = _get_current_function_mode_stack()
            self.assertIs(stack[0], m)
            self.assertIs(stack[1], m1)

        @smith.compile(fullgraph=True)
        def fn(x):
            smith.set_default_device("cpu")
            smith.set_default_device("cpu")
            return x + 1

        fn(smith.ones(2, 2))
        stack = _get_current_function_mode_stack()
        self.assertEqual(stack[0].device, smith.device("cpu"))
        smith.set_default_device(None)

    def test_pop_smith_function_mode(self):
        m = BaseSmithFunctionMode()
        with m:

            @smith.compile(fullgraph=True)
            def fn(x):
                _pop_smith_function_stack()
                return x + 1

            fn(smith.ones(2, 2))

            self.assertEqual(_len_smith_function_stack(), 0)
            # reset stack so __exit__ doesn't crash
            _push_on_smith_function_stack(m)

        self.assertEqual(_len_smith_function_stack(), 0)

    def test_is_smith_function_all_disabled(self):
        @smith.compile(fullgraph=True)
        def fn(x):
            return (
                smith._C._is_smith_function_all_disabled(),
                smith.add(x, 1.0),
            )

        input = smith.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_error_empty_stack_pop_smith_function_mode(self):
        @smith.compile(fullgraph=True)
        def fn(x):
            _pop_smith_function_stack()
            return x + 1

        self.assertRaisesRegex(
            smith._dynamo.exc.Unsupported,
            "Attempted to pop from empty smith function mode stack",
            lambda: fn(smith.ones(2, 2)),
        )

    def test_push_smith_function_mode(self):
        m = BaseSmithFunctionMode()
        with m:

            @smith.compile(fullgraph=True)
            def fn(x, m):
                _push_on_smith_function_stack(m)
                return x + 1

            fn(smith.ones(2, 2), m)

            self.assertEqual(_len_smith_function_stack(), 2)
            # reset stack state
            _pop_smith_function_stack()

        self.assertEqual(_len_smith_function_stack(), 0)

    def test_len_smith_function_mode(self):
        m = BaseSmithFunctionMode()
        with m:

            @smith.compile(fullgraph=True)
            def fn(x):
                z = _len_smith_function_stack()
                return x + z

            res = fn(smith.ones(2, 2))
            self.assertEqual(res, smith.ones(2, 2) + 1)
            self.assertEqual(_len_smith_function_stack(), 1)

    def test_intermedate_smith_function_mode_construction_mutation(self):
        class TestMode(BaseSmithFunctionMode):
            def __init__(self, x):
                self.x = x

        @smith.compile(fullgraph=True)
        def fn(x):
            z = TestMode(2)
            z.y = 2
            return x + 1, z

        fn(smith.ones(2, 2))

    def test_smith_function_mode_enabled_guard(self):
        cnt = smith._dynamo.testing.CompileCounter()
        inp = smith.ones(2, 2)

        @smith.compile(backend=cnt.__call__)
        def fn(x):
            return x + 1

        with BaseSmithFunctionMode(), smith._C.DisableSmithFunctionSubclass():
            with smith._C.DisableSmithFunction():
                fn(inp)
            fn(inp)
        self.assertEqual(cnt.frame_count, 2)

    def test_nested_smith_function_mode(self):
        mode_1_called = False
        mode_2_called = False

        def reset_state():
            nonlocal mode_1_called
            nonlocal mode_2_called
            mode_1_called = False
            mode_2_called = False

        ones = smith.ones(2, 2)
        zeros = smith.zeros(2, 2)

        class TestMode1(BaseSmithFunctionMode):
            def __smith_function__(self, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}

                nonlocal mode_1_called

                mode_1_called = True

                if func == smith.add:
                    return zeros

                return super().__smith_function__(func, types, args, kwargs)

        class TestMode2(BaseSmithFunctionMode):
            def __smith_function__(self, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}

                nonlocal mode_2_called

                mode_2_called = True

                if func == smith.mul:
                    return ones

                return super().__smith_function__(func, types, args, kwargs)

        def fn(x):
            return smith.add(x, 3)

        def fn_2(x):
            return smith.mul(x, 3) + smith.add(x, 3)

        inp = smith.ones(2, 2) + 1

        for fn_i in [fn, fn_2]:
            fn_opt = smith.compile(fn_i, fullgraph=True)
            with TestMode1(), TestMode2():
                expected = fn_i(inp), mode_1_called, mode_2_called
                reset_state()
                actual = fn_opt(inp), mode_1_called, mode_2_called
                reset_state()

            self.assertEqual(expected, actual)

    def test_smith_function_mode_disable(self):
        class TestSubclass(smith.Tensor):
            @classmethod
            def __smith_function__(cls, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}
                if func == smith.add:
                    return smith.ones(2, 2)
                return super().__smith_function__(func, types, args, kwargs)

        class TestMode(BaseSmithFunctionMode):
            def __smith_function__(self, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}

                if func == smith.add:
                    return smith.zeros(2, 2)

                return super().__smith_function__(func, types, args, kwargs)

        def fn(x):
            return smith.add(x, 3)

        inp = (smith.ones(2, 2) + 1).as_subclass(TestSubclass)

        fn_opt = smith.compile(fn, fullgraph=True)
        with TestMode():
            with smith._C.DisableSmithFunctionSubclass():
                expected = fn(inp)
                actual = fn_opt(inp)

            self.assertEqual(expected, actual)

            with smith._C.DisableSmithFunction():
                expected = fn(inp)
                actual = fn_opt(inp)

            self.assertEqual(expected, actual)

    def test_smith_function_mode_highest_priority(self):
        class TestSubclass(smith.Tensor):
            @classmethod
            def __smith_function__(cls, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}
                if func == smith.add:
                    return smith.ones(2, 2)
                return super().__smith_function__(func, types, args, kwargs)

        def fn(x):
            return smith.add(x, 3)

        inp = (smith.ones(2, 2) + 1).as_subclass(TestSubclass)

        fn_opt = smith.compile(fn, fullgraph=True)
        with TestMode():
            expected = fn(inp)
            actual = fn_opt(inp)

        self.assertEqual(expected, actual)

    def test_smith_function_mode_enter_exit(self):
        def fn(x, y):
            with TestMode():
                o = smith.add(x, 3)

            return smith.add(o, y)

        inp = (smith.ones(2, 2) + 1, smith.ones(2, 2) + 2)
        fn_opt = smith.compile(fn, fullgraph=True)

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_smith_function_mode_graph_break(self):
        def fn(x, y):
            with TestMode():
                smith._dynamo.graph_break()
                o = smith.add(x, 3)

            return smith.add(o, y)

        inp = (smith.ones(2, 2) + 1, smith.ones(2, 2) + 2)
        fn_opt = smith.compile(fn)

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_smith_function_mode_and_pop_graph_break(self):
        def fn(x, y):
            with TestMode():
                z = _pop_smith_function_stack()
                smith._dynamo.graph_break()
                _push_on_smith_function_stack(z)
                o = smith.add(x, 3)

            return smith.add(o, y)

        inp = (smith.ones(2, 2) + 1, smith.ones(2, 2) + 2)
        fn_opt = smith.compile(fn)

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_smith_function_mode_restore_on_exc(self):
        @smith._dynamo.disable()
        def err():
            raise RuntimeError("test")

        @smith.compile()
        def fn(x):
            with TestMode():
                x += 1
                err()
                x += 2
                return x

        try:
            fn(smith.ones(2, 2))
        except RuntimeError:
            pass
        self.assertEqual(_len_smith_function_stack(), 0)

    def test_smith_function_mode_and_pop_graph_break_mutation(self):
        def fn(x, y):
            with TestMode():
                z = _pop_smith_function_stack()
                z.y = 5
                smith._dynamo.graph_break()
                _push_on_smith_function_stack(z)
                o = smith.add(x, 3)
                o = smith.mul(o, z.y)

            return smith.add(o, y)

        inp = (smith.ones(2, 2) + 1, smith.ones(2, 2) + 2)
        fn_opt = smith.compile(fn)

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    # Needs larger cache size since we recompile for each op
    @patch.object(smith._dynamo.config, "recompile_limit", 48)
    def test_builtin_equivalent_funcs(self):
        from smith._dynamo.variables.builtin import (
            BUILTIN_TO_TENSOR_FN_MAP,
            BUILTIN_TO_TENSOR_RFN_MAP,
        )
        from smith._dynamo.variables.smith_function import (
            bin_int_ops,
            bin_ops,
            tensor_and_int_ops,
            un_int_ops,
            un_ops,
        )

        expected_func = None
        valid = False

        class FuncEquivMode(BaseSmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                nonlocal expected_func
                nonlocal valid
                if not kwargs:
                    kwargs = {}
                if smith._dynamo.is_compiling():
                    valid = expected_func == func
                return super().__smith_function__(func, types, args, kwargs)

        inp0 = smith.ones(1, 1)
        inp1 = smith.ones(1, 1)
        inp0_int = smith.ones(1, 1, dtype=smith.int32)
        inp1_int = smith.ones(1, 1, dtype=smith.int32)

        @smith.compile(fullgraph=True)
        def fn_un(op, inp):
            return op(inp)

        @smith.compile(fullgraph=True)
        def fn_un_int(op, inp):
            return op(inp)

        @smith.compile(fullgraph=True)
        def fn_bin(op, inp0, inp1):
            return op(inp0, inp1)

        @smith.compile(fullgraph=True)
        def fn_bin_int(op, inp0, inp1):
            return op(inp0, inp1)

        @smith.compile(fullgraph=True)
        def fn_tensor_and_int(op, inp0, inp1):
            return op(inp0, inp1)

        setups_and_oplists = [
            (lambda o: fn_un(o, inp0), un_ops),
            (lambda o: fn_un_int(o, inp0_int), un_int_ops),
            (lambda o: fn_bin(o, inp0, inp1), bin_ops),
            (lambda o: fn_bin_int(o, inp0_int, inp1_int), bin_int_ops),
            (lambda o: fn_tensor_and_int(o, inp0_int, 0), tensor_and_int_ops),
        ]

        # gather the reverse functions
        rsetups_and_oplists = [
            (
                lambda o: fn_bin(o, 1, inp1),
                bin_ops,
            ),  # Get r* ops, (ex. __sub__(int, Tensor) -> __rsub__(Tensor, int))
            (lambda o: fn_bin_int(o, 1, inp1_int), bin_int_ops),
            (lambda o: fn_tensor_and_int(o, 0, inp0_int), tensor_and_int_ops),
        ]

        skips = {operator.not_}  # Has local scalar dense call which graph breaks
        rskips = {
            operator.matmul,
            operator.imatmul,
            operator.getitem,
        }  # Doesn't type check with reversed args

        def run_checks(setups_and_oplists, skips, ref_map):
            nonlocal valid
            nonlocal expected_func
            for setup_fn, op_list in setups_and_oplists:
                for op in op_list:
                    if op in skips or op not in ref_map:
                        continue
                    with FuncEquivMode():
                        expected_func = ref_map[op]
                        setup_fn(op)
                        self.assertTrue(valid)

                    expected_func = None
                    valid = False

        run_checks(setups_and_oplists, skips, BUILTIN_TO_TENSOR_FN_MAP)
        run_checks(rsetups_and_oplists, rskips, BUILTIN_TO_TENSOR_RFN_MAP)

    def test_expand(self):
        from smith.distributions import (
            AffineTransform,
            ComposeTransform,
            Normal,
            TanhTransform,
            TransformedDistribution,
        )

        # https://github.com/blacksmith/blacksmith/issues/141232
        with smith.device("cpu"):

            @smith.compile(fullgraph=True)
            def func(a):
                d = TransformedDistribution(
                    Normal(a, 1),
                    ComposeTransform([TanhTransform(), AffineTransform(2, 2)]),
                )
                b = d.log_prob(d.rsample((10,)))
                return b

            func(smith.randn(3))

    @requires_gpu
    def test_flex_attention(self):
        import smith
        from smith.nn.attention.flex_attention import create_block_mask, flex_attention

        smith.set_default_device(device_type)

        flex_attention = smith.compile(flex_attention, dynamic=False)

        prefix_lengths = smith.arange(8)

        def prefix_lm(b, h, q, kv):
            return prefix_lengths[b] >= kv

        # This runs in fullgraph already
        create_block_mask(
            prefix_lm, 8, None, 512, 512, _compile=True, device=device_type
        )

    def test_register_hook(self):
        import functools

        def my_hook(grad, *, k=0):
            return grad + k

        hook = functools.partial(my_hook, k=3)

        class MyMod(smith.nn.Module):
            def forward(self, x):
                x.register_hook(hook)
                y = x.mul(2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x = smith.ones(4, requires_grad=True)

        with smith.device("cpu"):
            smith.compile(mod, fullgraph=True)(x)

    @requires_gpu
    @skipIfXpu(msg="XPU does not support flex attention")
    def test_hop(self):
        import smith
        import smith._higher_order_ops
        from smith.nn.attention.flex_attention import (
            flex_attention as flex_attention_eager,
        )

        with smith.device(GPU_TYPE):
            flex_attention = smith.compile(flex_attention_eager, dynamic=False)

            with self.assertRaisesRegex(
                smith._dynamo.exc.Unsupported,
                "raised exception HopDetectionError([ConstantVariable(str: 'test')])",
            ):
                # This runs in fullgraph already
                with TestModeRaises():
                    flex_attention(
                        smith.ones(2, 2, 2, 2),
                        smith.ones(2, 2, 2, 2),
                        smith.ones(2, 2, 2, 2),
                    )

    @requires_gpu
    @skipIfXpu(msg="XPU does not support flex attention")
    def test_hop_eager(self):
        import smith
        import smith._higher_order_ops
        from smith.nn.attention.flex_attention import (
            flex_attention as flex_attention_eager,
        )

        with smith.device(GPU_TYPE):
            with self.assertRaisesRegex(
                smith._dynamo.exc.Unsupported,
                "raised exception HopDetectionError([ConstantVariable(str: 'test')])",
            ):
                with TestModeRaises():
                    flex_attention_eager(
                        smith.ones(2, 2, 2, 2),
                        smith.ones(2, 2, 2, 2),
                        smith.ones(2, 2, 2, 2),
                    )

    @requires_gpu
    def test_default_device_factory_functions(self):
        """Test that factory functions respect default device in compiled code"""

        @smith.compile(fullgraph=True)
        def random_func(
            x: smith.Tensor,
        ) -> tuple[smith.Tensor, smith.Tensor, smith.Tensor, smith.Tensor]:
            # Test various factory functions
            rnd = smith.randint(0, 2**32, size=x.shape, dtype=smith.uint32)
            zeros = smith.zeros_like(rnd, device="cpu")
            zeros_matched = smith.zeros_like(rnd)
            return x + rnd, rnd, zeros, zeros_matched

        smith.set_default_device("cuda")
        (result, rnd, zeros, zeros_matched) = random_func(smith.randn(()))

        # Verify tensors are on CUDA
        self.assertEqual(rnd.device.type, "cuda")
        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(zeros.device.type, "cpu")
        self.assertEqual(zeros_matched.device.type, rnd.device.type)

        smith.set_default_device("cpu")
        (result, rnd, zeros, zeros_matched) = random_func(smith.randn(()))

        # Verify tensors are on cpu
        self.assertEqual(rnd.device.type, "cpu")
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(zeros.device.type, "cpu")
        self.assertEqual(zeros_matched.device.type, rnd.device.type)

        smith.set_default_device(None)

    @requires_gpu
    def test_default_device_factory_functions_priority(self):
        smith.set_default_device("cuda")

        @smith.compile(fullgraph=True)
        def with_explicit_device(x: smith.Tensor) -> tuple[smith.Tensor, smith.Tensor]:
            rnd = smith.randint(
                0, 2**32, size=x.shape, dtype=smith.uint32, device="cpu"
            )
            return x + rnd, rnd

        (result, rnd) = with_explicit_device(smith.randn(()))
        self.assertEqual(rnd.device.type, "cpu")
        self.assertEqual(result.device.type, "cuda")


class InvokeSubgraphBackendTests(smith._dynamo.test_case.TestCase):
    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_make_fx_over_compiled_function(self):
        """Test that make_fx can trace over smith.compile'd functions using invoke_subgraph backend.

        When force_compile_during_fx_trace=True, the invoke_subgraph backend should
        emit an invoke_subgraph HOP in the traced graph instead of inlining the subgraph.
        """
        from smith.fx.experimental.proxy_tensor import make_fx

        smith._dynamo.reset()  # Clear any cached graphs

        def simple_fn(x, y):
            return x * 2 + y

        compiled_fn = smith.compile(simple_fn, backend="invoke_subgraph")

        def outer_fn(x, y):
            z = x + 1
            result = compiled_fn(z, y)
            return result * 2

        x = smith.randn(3, 3)
        y = smith.randn(3, 3)

        # Trace with make_fx - the compiled_fn should appear as invoke_subgraph HOP
        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_smith_fn_metadata_mode=True
        )(x, y)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(smith.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        add: "f32[3, 3]" = smith.ops.aten.add.Tensor(x_1, 1);  x_1 = None
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', add, y_1);  repeated_subgraph0 = add = y_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(getitem, 2);  getitem = None
        return mul

    class repeated_subgraph0(smith.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
            add: "f32[3, 3]" = smith.ops.aten.add.Tensor(mul, arg1_1);  mul = arg1_1 = None
            return (add,)
""",  # noqa: B950
        )

    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_same_compiled_fn_called_twice_shares_subgraph(self):
        """Test that calling the same compiled function twice uses the same subgraph.

        When the same compiled function is called multiple times with inputs that
        don't cause guard failures, both calls should reference the same subgraph.
        """
        from smith._guards import tracing, TracingContext
        from smith.fx.experimental.proxy_tensor import make_fx

        smith._dynamo.reset()

        def simple_fn(x):
            return x * 2

        compiled_fn = smith.compile(simple_fn, backend="invoke_subgraph")

        def outer_fn(x, y):
            # Call the same compiled function twice
            a = compiled_fn(x)
            b = compiled_fn(y)
            return a + b

        x = smith.randn(3, 3)
        y = smith.randn(3, 3)

        # Set up TracingContext so invoke_subgraph cache works
        tracing_ctx = TracingContext(fake_mode=None)
        with tracing(tracing_ctx):
            traced = make_fx(
                outer_fn, tracing_mode="fake", _disable_smith_fn_metadata_mode=True
            )(x, y)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(smith.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1);  repeated_subgraph0 = x_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'invoke_subgraph_0', y_1);  repeated_subgraph0_1 = y_1 = None
        getitem_1: "f32[3, 3]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[3, 3]" = smith.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return add

    class repeated_subgraph0(smith.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
            return (mul,)
""",  # noqa: B950
        )

    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_invoke_subgraph_seq_nr(self):
        """
        Test that the seq_nr on the subgraphs and the invoke_subgraph HOP nodes are correct
        right before we copy metadata from fwd to bwd graph.
        """
        from smith._funcsmith.aot_autograd import aot_function
        from smith._guards import tracing, TracingContext

        smith._dynamo.reset()

        def inner_fn(x):
            with smith.fx.traceback.annotate({"test": "test"}):
                y = x.cos()
            return y / 2

        compiled_fn = smith.compile(inner_fn, backend="invoke_subgraph")

        def outer_fn(x):
            y = x + 1
            z = compiled_fn(y)
            return z.sum()

        x = smith.randn(3, 3, requires_grad=True)

        # Track forward graph to verify invoke_subgraph appears
        fw_graph = None
        bw_graph = None

        def fw_compiler(gm, example_inputs):
            nonlocal fw_graph
            fw_graph = gm
            return gm

        def bw_compiler(gm, example_inputs):
            nonlocal bw_graph
            bw_graph = gm
            return gm

        # we expect to capture two graphs, the first one from aot_autograd in invoke_subgraph backend
        # This one should have correct seq_nr on the joint graph for inner_fn and copy the metadata.
        # the second one from actual aot_stage1_graph_capture.
        tracing_ctx = TracingContext(fake_mode=None)
        with tracing(tracing_ctx):
            aot_fn = aot_function(
                outer_fn,
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                _disable_smith_fn_metadata_mode=True,
            )

            # Run forward and backward
            result = aot_fn(x)
            result.backward()

        # Check seq_nr ordering for main forward and backward graphs
        main_groups = smith.fx.traceback._get_ordered_seq_nr_groups(
            [fw_graph, bw_graph]
        )
        self.assertEqual(
            main_groups,
            [
                ["add"],  # seq_nr 21
                [
                    "clone",
                    "getitem",
                    "getitem_1",
                    "getitem_2",
                    "invoke_subgraph",
                    "invoke_subgraph_1",
                ],  # seq_nr 22
                ["expand", "sum_1"],  # seq_nr 23
            ],
        )

        # Check seq_nr ordering for inner subgraphs (forward and backward)
        subgraph_groups = smith.fx.traceback._get_ordered_seq_nr_groups(
            [fw_graph.repeated_subgraph0, bw_graph.repeated_subgraph1]
        )
        self.assertEqual(
            subgraph_groups,
            [
                ["cos", "mul", "neg", "sin"],  # seq_nr 15
                ["div", "div"],  # seq_nr 16 - both forward and backward have div
            ],
        )

        # The annotation is not checked here because we used ignore_comments = True.
        # The comments here are helpful for human to read and understand the unit test.
        self.assertExpectedInline(
            normalize_gm(fw_graph.print_readable(print_output=False)),
            """
class GraphModule(smith.nn.Module):
    def forward(self, primals_1: "f32[3, 3]"):
        # Annotation: {'seq_nr': 13} No stacktrace found for following nodes
        add: "f32[3, 3]" = smith.ops.aten.add.Tensor(primals_1, 1);  primals_1 = None

        # Annotation: {'seq_nr': 14} No stacktrace found for following nodes
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', add);  repeated_subgraph0 = add = None
        getitem: "f32[3, 3]" = invoke_subgraph[0]
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None

        # Annotation: {'seq_nr': 15} No stacktrace found for following nodes
        sum_1: "f32[]" = smith.ops.aten.sum.default(getitem);  getitem = None
        return (sum_1, getitem_1)

    class repeated_subgraph0(smith.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            # Annotation: {'test': 'test', 'seq_nr': 9} File: test_modes.py:920 in inner_fn, code: y = x.cos()
            cos: "f32[3, 3]" = smith.ops.aten.cos.default(arg0_1)

            # Annotation: {'seq_nr': 10} File: test_modes.py:921 in inner_fn, code: return y / 2
            div: "f32[3, 3]" = smith.ops.aten.div.Tensor(cos, 2);  cos = None
            return (div, arg0_1)
        """,  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        self.assertExpectedInline(
            normalize_gm(bw_graph.print_readable(print_output=False)),
            """
class GraphModule(smith.nn.Module):
    def forward(self, getitem_1: "f32[3, 3]", tangents_1: "f32[]"):
        # Annotation: {'seq_nr': 15} No stacktrace found for following nodes
        expand: "f32[3, 3]" = smith.ops.aten.expand.default(tangents_1, [3, 3]);  tangents_1 = None

        # Annotation: {'seq_nr': 14} No stacktrace found for following nodes
        clone: "f32[3, 3]" = smith.ops.aten.clone.default(expand, memory_format = smith.contiguous_format);  expand = None
        repeated_subgraph1 = self.repeated_subgraph1
        invoke_subgraph_1 = smith.ops.higher_order.invoke_subgraph(repeated_subgraph1, 'invoke_subgraph_1', getitem_1, clone);  repeated_subgraph1 = getitem_1 = clone = None
        getitem_2: "f32[3, 3]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_2,)

    class repeated_subgraph1(smith.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            # Annotation: {'seq_nr': 10} File: test_modes.py:921 in inner_fn, code: return y / 2
            div: "f32[3, 3]" = smith.ops.aten.div.Tensor(arg1_1, 2);  arg1_1 = None

            # Annotation: {'test': 'test', 'seq_nr': 9} File: test_modes.py:920 in inner_fn, code: y = x.cos()
            sin: "f32[3, 3]" = smith.ops.aten.sin.default(arg0_1);  arg0_1 = None
            neg: "f32[3, 3]" = smith.ops.aten.neg.default(sin);  sin = None
            mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(div, neg);  div = neg = None
            return (mul,)
        """,  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_guard_failure_creates_separate_subgraphs(self):
        """Test that guard failures create separate subgraphs.

        When the same compiled function is called with inputs that cause guard
        failures (e.g., different bool values), each compilation should result
        in a separate invoke_subgraph with a different identifier.
        """
        from smith.fx.experimental.proxy_tensor import make_fx

        smith._dynamo.reset()

        def conditional_fn(x, flag: bool):
            if flag:
                return x * 2
            else:
                return x * 3

        compiled_fn = smith.compile(conditional_fn, backend="invoke_subgraph")

        def outer_fn(x):
            # Call with flag=True, then flag=False - should trigger recompilation
            a = compiled_fn(x, True)
            b = compiled_fn(x, False)
            return a + b

        x = smith.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_smith_fn_metadata_mode=True
        )(x)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(smith.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1);  repeated_subgraph0 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph1 = self.repeated_subgraph1
        invoke_subgraph_1 = smith.ops.higher_order.invoke_subgraph(repeated_subgraph1, 'invoke_subgraph_1', x_1);  repeated_subgraph1 = x_1 = None
        getitem_1: "f32[3, 3]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[3, 3]" = smith.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return add

    class repeated_subgraph0(smith.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
            return (mul,)

    class repeated_subgraph1(smith.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(arg0_1, 3);  arg0_1 = None
            return (mul,)
""",  # noqa: B950
        )

    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_multiple_inputs(self):
        """Test invoke_subgraph with multiple tensor inputs.

        Verifies that the invoke_subgraph HOP correctly handles functions
        that take more than 2 tensor inputs.
        """
        from smith.fx.experimental.proxy_tensor import make_fx

        smith._dynamo.reset()

        def multi_input_fn(a, b, c, d):
            return a * b + c * d

        compiled_fn = smith.compile(multi_input_fn, backend="invoke_subgraph")

        def outer_fn(w, x, y, z):
            return compiled_fn(w, x, y, z)

        w = smith.randn(3, 3)
        x = smith.randn(3, 3)
        y = smith.randn(3, 3)
        z = smith.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_smith_fn_metadata_mode=True
        )(w, x, y, z)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(smith.nn.Module):
    def forward(self, w_1: "f32[3, 3]", x_1: "f32[3, 3]", y_1: "f32[3, 3]", z_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', w_1, x_1, y_1, z_1);  repeated_subgraph0 = w_1 = x_1 = y_1 = z_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        return getitem

    class repeated_subgraph0(smith.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]", arg3_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            mul_1: "f32[3, 3]" = smith.ops.aten.mul.Tensor(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
            add: "f32[3, 3]" = smith.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
            return (add,)
""",  # noqa: B950
        )

    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_multiple_outputs(self):
        """Test invoke_subgraph with multiple tensor outputs.

        Verifies that the invoke_subgraph HOP correctly handles functions
        that return multiple tensors.
        """
        from smith.fx.experimental.proxy_tensor import make_fx

        smith._dynamo.reset()

        def multi_output_fn(x, y):
            return x + y, x - y, x * y

        compiled_fn = smith.compile(multi_output_fn, backend="invoke_subgraph")

        def outer_fn(a, b):
            sum_out, diff_out, prod_out = compiled_fn(a, b)
            return sum_out * diff_out + prod_out

        a = smith.randn(3, 3)
        b = smith.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_smith_fn_metadata_mode=True
        )(a, b)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(smith.nn.Module):
    def forward(self, a_1: "f32[3, 3]", b_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', a_1, b_1);  repeated_subgraph0 = a_1 = b_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0]
        getitem_1: "f32[3, 3]" = invoke_subgraph[1]
        getitem_2: "f32[3, 3]" = invoke_subgraph[2];  invoke_subgraph = None
        mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        add: "f32[3, 3]" = smith.ops.aten.add.Tensor(mul, getitem_2);  mul = getitem_2 = None
        return add

    class repeated_subgraph0(smith.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            add: "f32[3, 3]" = smith.ops.aten.add.Tensor(arg0_1, arg1_1)
            sub: "f32[3, 3]" = smith.ops.aten.sub.Tensor(arg0_1, arg1_1)
            mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            return (add, sub, mul)
""",  # noqa: B950
        )

    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_multiple_inputs_and_outputs(self):
        """Test invoke_subgraph with both multiple inputs and outputs.

        Verifies that the invoke_subgraph HOP correctly handles functions
        that have both multiple tensor inputs and multiple tensor outputs.
        """
        from smith.fx.experimental.proxy_tensor import make_fx

        smith._dynamo.reset()

        def multi_io_fn(a, b, c):
            # Multiple inputs, multiple outputs
            return a + b + c, a * b * c

        compiled_fn = smith.compile(multi_io_fn, backend="invoke_subgraph")

        def outer_fn(x, y, z):
            sum_out, prod_out = compiled_fn(x, y, z)
            return sum_out - prod_out

        x = smith.randn(3, 3)
        y = smith.randn(3, 3)
        z = smith.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_smith_fn_metadata_mode=True
        )(x, y, z)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(smith.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]", z_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = smith.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1, y_1, z_1);  repeated_subgraph0 = x_1 = y_1 = z_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0]
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None
        sub: "f32[3, 3]" = smith.ops.aten.sub.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return sub

    class repeated_subgraph0(smith.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]"):
            add: "f32[3, 3]" = smith.ops.aten.add.Tensor(arg0_1, arg1_1)
            add_1: "f32[3, 3]" = smith.ops.aten.add.Tensor(add, arg2_1);  add = None
            mul: "f32[3, 3]" = smith.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            mul_1: "f32[3, 3]" = smith.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
            return (add_1, mul_1)
""",  # noqa: B950
        )

    @smith._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_aot_autograd_over_dynamo_with_requires_grad(self):
        """Test AOTAutograd tracing over a smith.compile'd function with requires_grad inputs.

        This tests the scenario where:
        1. An outer aot_function traces a function with requires_grad inputs
        2. Inside that function, a smith.compile'd function with invoke_subgraph backend is called
        3. AOTAutograd partitions into forward/backward graphs
        4. The inner Dynamo region should only be compiled once
        """
        from smith._dynamo.testing import CompileCounterWithBackend
        from smith._funcsmith.aot_autograd import aot_function
        from smith._funcsmith.compilers import nop

        smith._dynamo.reset()

        # Use a compile counter to track how many times Dynamo compiles
        compile_counter = CompileCounterWithBackend("invoke_subgraph")

        def inner_fn(x):
            return x * 2 + 1

        compiled_fn = smith.compile(inner_fn, backend=compile_counter)

        def outer_fn(x):
            y = x + 1
            z = compiled_fn(y)
            return z.sum()

        x = smith.randn(3, 3, requires_grad=True)

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
        result = aot_fn(x)
        result.backward()

        # Check that we got a forward graph with invoke_subgraph
        self.assertIsNotNone(fw_graph, "Expected a forward graph")
        fw_graph_code = fw_graph.print_readable(print_output=False)
        self.assertIn("invoke_subgraph", fw_graph_code)

        # Check compile count - should be 1 (compiled once during tracing)
        self.assertEqual(
            compile_counter.frame_count,
            1,
            f"Expected 1 compilation, got {compile_counter.frame_count}",
        )


class SmithFunctionModeLifecycleTests(smith._dynamo.test_case.TestCase):
    def test_default_device_restored_after_mode_tests(self):
        case = SmithFunctionModeTests("test_stack_state_mutation_default_device")
        SmithFunctionModeTests.setUpClass()
        try:
            case.setUp()
            try:
                case.test_stack_state_mutation_default_device()
            finally:
                case.tearDown()
        finally:
            SmithFunctionModeTests.tearDownClass()

        stack = _get_current_function_mode_stack()
        self.assertFalse(any(isinstance(mode, DeviceContext) for mode in stack))


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
