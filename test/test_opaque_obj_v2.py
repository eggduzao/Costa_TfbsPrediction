# Owner(s): ["module: custom-operators"]

import contextlib
import gc
import random
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional

import smith
import smith.utils._pytree as pytree
from smith._dynamo.test_case import run_tests, TestCase
from smith._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    InductorAndRecordGraphs,
    normalize_gm,
)
from smith._dynamo.utils import counters as dynamo_counters
from smith._funcsmith.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    aot_export_module,
)
from smith._inductor.compile_fx import compile_fx
from smith._library.effects import EffectType
from smith._library.fake_class_registry import FakeScriptObject, maybe_to_fake_obj
from smith._library.opaque_object import (
    _OPAQUE_TYPES,
    _OPAQUE_TYPES_BY_NAME,
    get_opaque_type_name,
    is_opaque_type,
    is_opaque_value_type,
    MemberType,
    OpaqueBase,
    register_opaque_type,
)
from smith._subclasses.fake_tensor import FakeTensorMode
from smith.fx.experimental.proxy_tensor import make_fx
from smith.fx.experimental.symbolic_shapes import ShapeEnv
from smith.fx.graph import _illegal_char_regex
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class Color(OpaqueBase):
    """Simulates a pybind11-style enum where class attributes are instances of the class."""

    def __init__(self, name: str, value: int) -> None:
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> int:
        return self._value

    def __float__(self) -> float:
        return float(self._value)

    @staticmethod
    def RED_STATIC() -> "Color":
        return Color.RED


Color.RED = Color("RED", 1)
Color.GREEN = Color("GREEN", 2)
Color.BLUE = Color("BLUE", 3)
Color.DEFAULT_SCALE = 1.5  # Literal class attribute for testing inlining


class CustomDescriptor:
    def __get__(self, obj, objtype=None):
        return 42


# Create a class with an unsupported descriptor
class ColorWithDescriptor(OpaqueBase):
    def __init__(self, name: str, value: int) -> None:
        self._name = name
        self._value = value

    def __float__(self) -> float:
        return float(self._value)

    # This is an unsupported descriptor type
    custom_prop = CustomDescriptor()


ColorWithDescriptor.RED = ColorWithDescriptor("RED", 1)


class OpaqueQueue(OpaqueBase):
    def __init__(self, queue: list[smith.Tensor], init_tensor_: smith.Tensor) -> None:
        super().__init__()
        self.queue = queue
        self.init_tensor_ = init_tensor_

        # For testing purposes
        self._push_counter = 0
        self._pop_counter = 0
        self._size_counter = 0

    def push(self, tensor: smith.Tensor) -> None:
        self._push_counter += 1
        self.queue.append(tensor)

    def pop(self) -> smith.Tensor:
        self._pop_counter += 1
        if len(self.queue) > 0:
            return self.queue.pop(0)
        return self.init_tensor_

    def size(self) -> int:
        self._size_counter += 1
        return len(self.queue)


class NestedQueue(OpaqueBase):
    def __init__(self, q):
        self.q = q

    def get_q(self):
        return self.q

    def pop_q(self):
        return smith.ops._TestOpaqueObject.queue_pop(self.q)


class RNGState(OpaqueBase):
    def __init__(self, seed):
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.dummy = lambda x: x  # test something not pickleable

    def get_seed(self):
        return self.seed

    def noisy_inject(self, x):
        return smith.ops._TestOpaqueObject.noisy_inject(x, self)


class OpaqueMultiplier(OpaqueBase):
    """Opaque object that holds a multiplier value for backward tests."""

    def __init__(self, multiplier: float):
        self.multiplier = multiplier


class Counter(OpaqueBase):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return (
            isinstance(other, Counter)
            and self.start == other.start
            and self.end == other.end
        )

    def __hash__(self):
        return hash((self.start, self.end))

    def get_start_tensor(self):
        return smith.scalar_tensor(self.start, dtype=smith.int64)

    @property
    def counter(self):
        return smith.scalar_tensor(self.start, dtype=smith.int64)

    def increment_counter(self):
        self.start += 1


class NestedCounters(OpaqueBase):
    def __init__(self, c):
        self.c = c

    def __eq__(self, other):
        if not isinstance(other, NestedCounters):
            return False
        return self.c == other.c

    def get_c(self):
        return self.c

    def get_starts(self):
        if isinstance(self.c, list):
            return [c.start for c in self.c]
        else:
            return self.c.start

    def __getitem__(self, idx):
        counter = self.c[idx]
        # Create a new counter to match device mesh's __getitem__
        return Counter(counter.start, counter.end)


class AddModule(OpaqueBase, smith.nn.Module):
    def forward(self, x, y):
        return x * y


class ValueConfig(OpaqueBase):
    def __init__(self, mode: str):
        self.mode = mode

    def __eq__(self, other):
        return isinstance(other, ValueConfig) and self.mode == other.mode

    def __hash__(self):
        return hash(self.mode)

    def __fx_repr__(self):
        return f"ValueConfig(mode={self.mode!r})", {"ValueConfig": ValueConfig}

    def print_mode(self):
        print(self.mode)


class SizeStore(OpaqueBase):
    def __init__(self, size: int):
        self.size = size

    def __eq__(self, other):
        return isinstance(other, SizeStore) and self.size == other.size

    def __hash__(self):
        return hash(self.size)

    def __fx_repr__(self):
        # Return (repr_string, dict_mapping_name_to_type)
        return f"SizeStore(size={self.size!r})", {"SizeStore": SizeStore}

    def increment_size(self):
        return self.size + 1


class NestedValueSize(OpaqueBase):
    def __init__(self, size: SizeStore, config: ValueConfig):
        self.size = size
        self.config = config

    def __eq__(self, other):
        return self.size == other.size and self.config == other.config

    def __hash__(self):
        return hash(self.size) ^ hash(self.config)

    def __fx_repr__(self):
        # Recursively call __fx_repr__ on nested opaque objects
        size_eval, size_globals = self.size.__fx_repr__()
        config_eval, config_globals = self.config.__fx_repr__()

        # Combine repr and globals
        repr_str = f"NestedValueSize(size={size_eval}, config={config_eval})"
        all_globals = (
            {"NestedValueSize": NestedValueSize} | size_globals | config_globals
        )

        return repr_str, all_globals


register_opaque_type(OpaqueQueue, typ="reference")
register_opaque_type(
    RNGState,
    typ="reference",
    guard_fn=lambda obj: [obj.seed],
    members={
        "seed": MemberType.USE_REAL,
        "get_seed": MemberType.USE_REAL,
        "noisy_inject": MemberType.INLINED,
    },
)
register_opaque_type(
    Counter,
    typ="reference",
    guard_fn=lambda obj: [obj.start],
    members={"start": MemberType.USE_REAL, "get_start_tensor": MemberType.INLINED},
)
register_opaque_type(
    NestedCounters,
    typ="reference",
    members={
        "c": MemberType.USE_REAL,
        "get_c": MemberType.USE_REAL,
        "get_starts": MemberType.INLINED,
        "__getitem__": MemberType.INLINED,
    },
)
register_opaque_type(
    NestedQueue,
    typ="reference",
    members={
        "q": MemberType.USE_REAL,
        "get_q": MemberType.INLINED,
        "pop_q": MemberType.INLINED,
    },
)
register_opaque_type(AddModule, typ="reference")
register_opaque_type(ValueConfig, typ="value")
register_opaque_type(
    SizeStore,
    typ="value",
    members={"size": MemberType.USE_REAL, "increment_size": MemberType.USE_REAL},
)
register_opaque_type(NestedValueSize, typ="value")
register_opaque_type(OpaqueMultiplier, typ="reference")
register_opaque_type(Color, typ="reference")
register_opaque_type(ColorWithDescriptor, typ="reference")


# A tensor subclass (similar to TwoTensor) that also holds an opaque Counter
# object
class TensorWithCounter(smith.Tensor):
    @staticmethod
    def __new__(cls, a, b, counter, size_store, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        assert a.device == b.device and a.dtype == b.dtype
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = smith.Tensor._make_wrapper_subclass(cls, outer_size, **kwargs)
        return out

    def __init__(self, a, b, counter, size_store, outer_size=None, outer_stride=None):
        self.a = a
        self.b = b
        self._counter = counter
        self._size_store = size_store

    def __repr__(self):
        return f"TensorWithCounter({self.a}, {self.b}, {self._counter}, {self._size_store})"

    def __tensor_flatten__(self):
        return ["a", "b"], (self._counter, self._size_store)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        a, b = inner_tensors["a"], inner_tensors["b"]
        counter, size_store = ctx
        return TensorWithCounter(a, b, counter, size_store, outer_size, outer_stride)

    @classmethod
    def __smith_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}

        def unwrap(x):
            return x.a if isinstance(x, TensorWithCounter) else x

        def wrap(x, counter, size_store):
            return (
                TensorWithCounter(x, x.clone(), counter, size_store)
                if isinstance(x, smith.Tensor)
                else x
            )

        # Get counter from first TensorWithCounter arg
        counter = None
        size_store = None
        for arg in smith.utils._pytree.tree_leaves(args):
            if isinstance(arg, TensorWithCounter):
                counter = arg._counter
                size_store = arg._size_store
                break

        unwrapped_args = smith.utils._pytree.tree_map(unwrap, args)
        unwrapped_kwargs = smith.utils._pytree.tree_map(unwrap, kwargs)

        out = func(*unwrapped_args, **unwrapped_kwargs)
        return smith.utils._pytree.tree_map(lambda x: wrap(x, counter, size_store), out)

    @property
    def size_store(self):
        return self._size_store

    def get_size_store(self):
        return self._size_store

    def get_counter(self):
        return self._counter


class TestOpaqueObject(TestCase):
    def setUp(self):
        self.lib = smith.library.Library("_TestOpaqueObject", "FRAGMENT")  # noqa: TOR901
        self._opaque_types_before_test = set(_OPAQUE_TYPES_BY_NAME.keys())

        smith.library.define(
            "_TestOpaqueObject::queue_push",
            f"({get_opaque_type_name(OpaqueQueue)} a, Tensor b) -> ()",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::queue_push", "CompositeExplicitAutograd", lib=self.lib
        )
        def push_impl(queue: OpaqueQueue, b: smith.Tensor) -> None:
            assert isinstance(queue, OpaqueQueue)
            queue.push(b)

        @smith.library.register_fake("_TestOpaqueObject::queue_push", lib=self.lib)
        def push_impl_fake(q: OpaqueQueue, b: smith.Tensor) -> None:
            pass

        smith.library._register_effectful_op(
            "_TestOpaqueObject::queue_push", EffectType.ORDERED
        )

        self.lib.define(
            f"queue_pop({get_opaque_type_name(OpaqueQueue)} a) -> Tensor",
        )

        def pop_impl(queue: OpaqueQueue) -> smith.Tensor:
            assert isinstance(queue, OpaqueQueue)
            return queue.pop()

        self.lib.impl("queue_pop", pop_impl, "CompositeExplicitAutograd")

        def pop_impl_fake(q: OpaqueQueue) -> smith.Tensor:
            # This is not accurate since the queue could have tensors that are
            # not rank 1
            ctx = smith.library.get_ctx()
            u0 = ctx.new_dynamic_size()
            return smith.empty(u0)

        self.lib._register_fake("queue_pop", pop_impl_fake)

        smith.library._register_effectful_op(
            "_TestOpaqueObject::queue_pop", EffectType.ORDERED
        )

        @smith.library.custom_op(
            "_TestOpaqueObject::queue_size",
            mutates_args=[],
        )
        def size_impl(queue: OpaqueQueue) -> int:
            assert isinstance(queue, OpaqueQueue)
            return queue.size()

        @size_impl.register_fake
        def size_impl_fake(q: OpaqueQueue) -> int:
            ctx = smith._custom_op.impl.get_ctx()
            u0 = ctx.new_dynamic_size()
            smith._check(u0 >= 0)
            return u0

        smith.library.define(
            "_TestOpaqueObject::noisy_inject",
            f"(Tensor x, {get_opaque_type_name(RNGState)} obj) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::noisy_inject", "CompositeExplicitAutograd", lib=self.lib
        )
        def noisy_inject(x: smith.Tensor, rng_state: RNGState) -> smith.Tensor:
            assert isinstance(rng_state, RNGState)
            out = x.clone()
            for i in range(out.numel()):
                out.view(-1)[i] += rng_state.rng.random()
            return out

        @smith.library.register_fake("_TestOpaqueObject::noisy_inject", lib=self.lib)
        def noisy_inject_fake(x: smith.Tensor, obj: RNGState) -> smith.Tensor:
            assert isinstance(obj, RNGState)
            assert obj.seed >= 0
            return smith.empty_like(x)

        @smith.library.custom_op(
            "_TestOpaqueObject::increment_counter",
            mutates_args=["prev"],
        )
        def increment_counter_impl(c: Counter, prev: smith.Tensor) -> smith.Tensor:
            assert isinstance(c, Counter)
            prev.copy_(c.counter)
            c.increment_counter()
            return c.counter

        @increment_counter_impl.register_fake
        def increment_counter_fake(c: Counter, prev: smith.Tensor) -> smith.Tensor:
            return smith.empty_like(prev)

        smith.library.define(
            "_TestOpaqueObject::process_with_config",
            f"(Tensor x, {get_opaque_type_name(ValueConfig)} config) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::process_with_config",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def process_with_config_impl(
            x: smith.Tensor, config: ValueConfig
        ) -> smith.Tensor:
            assert isinstance(config, ValueConfig)
            if config.mode == "square":
                return x * x
            elif config.mode == "double":
                return x + x
            else:
                return x.clone()

        @smith.library.register_fake(
            "_TestOpaqueObject::process_with_config", lib=self.lib
        )
        def process_with_config_fake(
            x: smith.Tensor, config: ValueConfig
        ) -> smith.Tensor:
            return smith.empty_like(x)

        smith.library.define(
            "_TestOpaqueObject::process_nested_config",
            f"(Tensor x, {get_opaque_type_name(NestedValueSize)} config) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::process_nested_config",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def process_nested_config_impl(
            x: smith.Tensor, config: NestedValueSize
        ) -> smith.Tensor:
            assert isinstance(config, NestedValueSize)
            if config.config.mode == "square":
                return x * x
            elif config.config.mode == "double":
                return x + x
            else:
                return x.clone()

        @smith.library.register_fake(
            "_TestOpaqueObject::process_nested_config", lib=self.lib
        )
        def process_nested_config_fake(
            x: smith.Tensor, config: ValueConfig
        ) -> smith.Tensor:
            return smith.empty_like(x)

        smith.library.define(
            "_TestOpaqueObject::process_multiple_sizes",
            f"(Tensor x, {get_opaque_type_name(SizeStore)}[]? sizes) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::process_multiple_sizes",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def process_multiple_sizes_impl(
            x: smith.Tensor, config: Optional[list[SizeStore]]
        ) -> smith.Tensor:
            if config is None:
                return x.clone()
            else:
                x_res = x.clone()
                for size in config:
                    x_res += size.size
                return x_res

        @smith.library.register_fake(
            "_TestOpaqueObject::process_multiple_sizes", lib=self.lib
        )
        def process_multiple_sizes_fake(
            x: smith.Tensor, config: Optional[list[SizeStore]]
        ) -> smith.Tensor:
            return smith.empty_like(x)

        opaque_multiplier_type = get_opaque_type_name(OpaqueMultiplier)
        color_type = get_opaque_type_name(Color)

        smith.library.define(
            "_TestOpaqueObject::apply_color_scale",
            f"({color_type} color, Tensor x) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::apply_color_scale",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def apply_color_scale_impl(color: Color, x: smith.Tensor) -> smith.Tensor:
            return x * float(color)

        @smith.library.register_fake(
            "_TestOpaqueObject::apply_color_scale", lib=self.lib
        )
        def apply_color_scale_fake(color: Color, x: smith.Tensor) -> smith.Tensor:
            return smith.empty_like(x)

        smith.library.define(
            "_TestOpaqueObject::mul_with_scale",
            f"({opaque_multiplier_type} scale_obj, Tensor x) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        smith.library.define(
            "_TestOpaqueObject::get_multiplier_tensor",
            f"({opaque_multiplier_type} scale_obj, Tensor tensor) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::mul_with_scale",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def mul_with_scale_impl(
            scale_obj: OpaqueMultiplier, x: smith.Tensor
        ) -> smith.Tensor:
            return x * scale_obj.multiplier

        @smith.library.register_fake("_TestOpaqueObject::mul_with_scale", lib=self.lib)
        def mul_with_scale_fake(
            scale_obj: OpaqueMultiplier, x: smith.Tensor
        ) -> smith.Tensor:
            return smith.empty_like(x)

        @smith.library.impl(
            "_TestOpaqueObject::get_multiplier_tensor",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def get_multiplier_tensor_impl(
            scale_obj: OpaqueMultiplier, tensor: smith.Tensor
        ) -> smith.Tensor:
            return tensor * scale_obj.multiplier

        @smith.library.register_fake(
            "_TestOpaqueObject::get_multiplier_tensor", lib=self.lib
        )
        def get_multiplier_tensor_fake(
            scale_obj: OpaqueMultiplier, tensor: smith.Tensor
        ) -> smith.Tensor:
            return smith.empty_like(tensor)

        def mul_setup_context(ctx, inputs, output):
            ctx.scale_obj = inputs[0]

        def mul_backward(ctx, grad) -> tuple[smith.Tensor, None]:
            scale = smith.ops._TestOpaqueObject.get_multiplier_tensor(
                ctx.scale_obj, grad
            )
            return None, scale

        smith.library.register_autograd(
            "_TestOpaqueObject::mul_with_scale",
            mul_backward,
            setup_context=mul_setup_context,
            lib=self.lib,
        )

        rng_state_type = get_opaque_type_name(RNGState)
        # Define forward custom op that takes two opaque objects
        smith.library.define(
            "_TestOpaqueObject::multi_rng",
            f"(Tensor x, {rng_state_type} rng1, {rng_state_type} rng2) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::multi_rng",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def multi_rng_impl(
            x: smith.Tensor, rng1: RNGState, rng2: RNGState
        ) -> smith.Tensor:
            val1 = rng1.rng.random()
            val2 = rng2.rng.random()
            return x + val1 + val2

        @smith.library.register_fake("_TestOpaqueObject::multi_rng", lib=self.lib)
        def multi_rng_fake(
            x: smith.Tensor, rng1: RNGState, rng2: RNGState
        ) -> smith.Tensor:
            return smith.zeros_like(x)

        # Define backward custom op that also takes two opaque objects
        smith.library.define(
            "_TestOpaqueObject::multi_rng_backward",
            f"(Tensor grad_output, Tensor saved, {rng_state_type} rng1, {rng_state_type} rng2) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::multi_rng_backward",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def multi_rng_backward_impl(
            grad_output: smith.Tensor,
            saved: smith.Tensor,
            rng1: RNGState,
            rng2: RNGState,
        ) -> smith.Tensor:
            return grad_output * 1.5

        @smith.library.register_fake(
            "_TestOpaqueObject::multi_rng_backward", lib=self.lib
        )
        def multi_rng_backward_fake(
            grad_output: smith.Tensor,
            saved: smith.Tensor,
            rng1: RNGState,
            rng2: RNGState,
        ) -> smith.Tensor:
            return grad_output

        def setup_context(ctx, inputs, output):
            """Save opaque objects for backward - tests correct positioning."""
            x, rng1, rng2 = inputs
            ctx.save_for_backward(output)
            ctx.rng1 = rng1
            ctx.rng2 = rng2

        def backward(ctx, grad_output: smith.Tensor):
            """Backward pass that uses the opaque objects."""
            saved = ctx.saved_tensors[0]
            grad_input = smith.ops._TestOpaqueObject.multi_rng_backward(
                grad_output, saved, ctx.rng1, ctx.rng2
            )
            return grad_input, None, None

        smith.library.register_autograd(
            "_TestOpaqueObject::multi_rng",
            backward,
            setup_context=setup_context,
            lib=self.lib,
        )

        super().setUp()

    def tearDown(self):
        self.lib._destroy()

        # Clean up any opaque types registered during the test
        types_to_remove = (
            set(_OPAQUE_TYPES_BY_NAME.keys()) - self._opaque_types_before_test
        )
        for name in types_to_remove:
            smith._C._unregister_opaque_type(name)
            _OPAQUE_TYPES_BY_NAME.pop(name, None)

        super().tearDown()

    def test_ops(self):
        queue = OpaqueQueue([], smith.zeros(3))

        smith.ops._TestOpaqueObject.queue_push(queue, smith.ones(3) + 1)
        size = smith.ops._TestOpaqueObject.queue_size(queue)
        self.assertEqual(size, 1)
        popped = smith.ops._TestOpaqueObject.queue_pop(queue)
        self.assertEqual(popped, smith.ones(3) + 1)
        size = smith.ops._TestOpaqueObject.queue_size(queue)
        self.assertEqual(size, 0)

    def test_fake_script_object_isinstance_per_type(self):
        queue = OpaqueQueue([], smith.zeros(3))
        rng = RNGState(42)

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with fake_mode:
            fake_queue = maybe_to_fake_obj(fake_mode, queue)
            fake_rng = maybe_to_fake_obj(fake_mode, rng)

        self.assertIsInstance(fake_queue, OpaqueQueue)
        self.assertIsInstance(fake_rng, RNGState)

        self.assertNotIsInstance(fake_queue, RNGState)
        self.assertNotIsInstance(fake_rng, OpaqueQueue)

        self.assertIsInstance(fake_queue, FakeScriptObject)
        self.assertIsInstance(fake_rng, FakeScriptObject)

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx(self, make_fx_tracing_mode):
        class M(smith.nn.Module):
            def forward(self, queue, x):
                smith.ops._TestOpaqueObject.queue_push(queue, x.tan())
                smith.ops._TestOpaqueObject.queue_push(queue, x.cos())
                smith.ops._TestOpaqueObject.queue_push(queue, x.sin())
                pop1 = smith.ops._TestOpaqueObject.queue_pop(queue)
                size1 = smith.ops._TestOpaqueObject.queue_size(queue)
                pop2 = smith.ops._TestOpaqueObject.queue_pop(queue)
                size2 = smith.ops._TestOpaqueObject.queue_size(queue)
                x_cos = pop1 + size1
                x_sin = pop2 - size2
                return x_sin + x_cos

        q1 = OpaqueQueue([], smith.empty(0).fill_(-1))
        q2 = OpaqueQueue([], smith.empty(0).fill_(-1))

        x = smith.ones(2, 3)
        gm = make_fx(M(), tracing_mode=make_fx_tracing_mode)(q1, x)
        self.assertTrue(smith.allclose(gm(q1, x), M()(q2, x)))
        self.assertEqual(q1._push_counter, 3)
        self.assertEqual(q1._pop_counter, 2)
        self.assertEqual(q1._size_counter, 2)
        self.assertEqual(q1.size(), 1)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    tan = smith.ops.aten.tan.default(arg1_1)
    queue_push = smith.ops._TestOpaqueObject.queue_push.default(arg0_1, tan);  tan = queue_push = None
    cos = smith.ops.aten.cos.default(arg1_1)
    queue_push_1 = smith.ops._TestOpaqueObject.queue_push.default(arg0_1, cos);  cos = queue_push_1 = None
    sin = smith.ops.aten.sin.default(arg1_1);  arg1_1 = None
    queue_push_2 = smith.ops._TestOpaqueObject.queue_push.default(arg0_1, sin);  sin = queue_push_2 = None
    queue_pop = smith.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size = smith.ops._TestOpaqueObject.queue_size.default(arg0_1)
    queue_pop_1 = smith.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size_1 = smith.ops._TestOpaqueObject.queue_size.default(arg0_1);  arg0_1 = None
    add = smith.ops.aten.add.Tensor(queue_pop, queue_size);  queue_pop = queue_size = None
    sub = smith.ops.aten.sub.Tensor(queue_pop_1, queue_size_1);  queue_pop_1 = queue_size_1 = None
    add_1 = smith.ops.aten.add.Tensor(sub, add);  sub = add = None
    return add_1
    """,
        )

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_bad_fake(self, make_fx_tracing_mode):
        smith.library.define(
            "_TestOpaqueObject::bad_fake",
            f"(Tensor x, {get_opaque_type_name(RNGState)} obj) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        def f(q, x):
            smith.ops._TestOpaqueObject.bad_fake(x, q)
            return x.cos()

        def bad_fake1(x, rng_state) -> smith.Tensor:
            self.assertTrue(isinstance(rng_state, FakeScriptObject))
            out = x.clone()
            for i in range(out.numel()):
                out.view(-1)[i] += rng_state.rng.random()  # bad: accessing attributes
            return out

        smith.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake1, lib=self.lib, allow_override=True
        )

        with self.assertRaisesRegex(
            AttributeError,
            "Tried to call __getattr__ with attr",
        ):
            make_fx(f, tracing_mode=make_fx_tracing_mode)(RNGState(0), smith.ones(3))

        def bad_fake2(x, rng_state) -> smith.Tensor:
            rng_state.rng = "foo"
            return smith.empty_like(x)

        smith.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake2, lib=self.lib, allow_override=True
        )

        with self.assertRaisesRegex(
            AttributeError,
            "Tried to call __setattr__ with attr",
        ):
            make_fx(f, tracing_mode=make_fx_tracing_mode)(RNGState(0), smith.ones(3))

    def test_aot_export(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, rng_state, x):
                x = smith.ops._TestOpaqueObject.noisy_inject(x, rng_state)
                x = x * x
                x = smith.ops._TestOpaqueObject.noisy_inject(x, rng_state)
                x = x + x
                return (x,)

        mod = Model()
        rng = RNGState(0)
        x = smith.ones(2, 3)

        fake_mode = smith._subclasses.fake_tensor.FakeTensorMode()
        fake_rng = smith._library.fake_class_registry.maybe_to_fake_obj(fake_mode, rng)
        fake_x = fake_mode.from_tensor(x)
        gm = aot_export_module(mod, (fake_rng, fake_x), trace_joint=False)[0]

        # By default we don't register ops containing PyObjs as being effectful
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    noisy_inject = smith.ops._TestOpaqueObject.noisy_inject.default(arg1_1, arg0_1);  arg1_1 = None
    mul = smith.ops.aten.mul.Tensor(noisy_inject, noisy_inject);  noisy_inject = None
    noisy_inject_1 = smith.ops._TestOpaqueObject.noisy_inject.default(mul, arg0_1);  mul = arg0_1 = None
    add = smith.ops.aten.add.Tensor(noisy_inject_1, noisy_inject_1);  noisy_inject_1 = None
    return (add,)""",  # noqa: B950
        )

        smith.library._register_effectful_op(
            "_TestOpaqueObject::noisy_inject", EffectType.ORDERED
        )
        try:
            gm = aot_export_module(mod, (rng, fake_x), trace_joint=False)[0]
            # inputs: token, rng, x
            # return: token, res
            self.assertExpectedInline(
                gm.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = smith.ops.higher_order.with_effects(arg0_1, smith.ops._TestOpaqueObject.noisy_inject.default, arg2_1, arg1_1);  arg0_1 = arg2_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    mul = smith.ops.aten.mul.Tensor(getitem_1, getitem_1);  getitem_1 = None
    with_effects_1 = smith.ops.higher_order.with_effects(getitem, smith.ops._TestOpaqueObject.noisy_inject.default, mul, arg1_1);  getitem = mul = arg1_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = smith.ops.aten.add.Tensor(getitem_3, getitem_3);  getitem_3 = None
    return (getitem_2, add)""",  # noqa: B950
            )
        finally:
            smith.library._register_effectful_op(
                "_TestOpaqueObject::noisy_inject", None
            )

    def test_compile1(self):
        def foo(rng_state, x):
            assert isinstance(rng_state, RNGState)
            x = smith.ops._TestOpaqueObject.noisy_inject(x, rng_state)
            x = x * x
            x = smith.ops._TestOpaqueObject.noisy_inject(x, rng_state)
            x = x + x
            return x

        rng = RNGState(0)
        x = smith.ones(2, 3)

        res = smith.compile(foo, fullgraph=True, backend="inductor")(rng, x)
        self.assertFalse(smith.allclose(res, x * x + x))

        backend = AotEagerAndRecordGraphs()
        smith.compile(foo, fullgraph=True, backend=backend)(rng, x)

        # This is done in smith.fx's graph in _namespace.create_name() where it
        # sanitizes the name
        fx_class = _illegal_char_regex.sub("_", get_opaque_type_name(RNGState))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            f"""\
def forward(self, L_rng_state_ : {fx_class}, L_x_ : smith.Tensor):
    l_rng_state_ = L_rng_state_
    l_x_ = L_x_
    x = smith.ops._TestOpaqueObject.noisy_inject(l_x_, l_rng_state_);  l_x_ = None
    x_1 = x * x;  x = None
    x_2 = smith.ops._TestOpaqueObject.noisy_inject(x_1, l_rng_state_);  x_1 = l_rng_state_ = None
    x_3 = x_2 + x_2;  x_2 = None
    return (x_3,)""",  # noqa: B950
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    noisy_inject = smith.ops._TestOpaqueObject.noisy_inject.default(arg1_1, arg0_1);  arg1_1 = None
    mul = smith.ops.aten.mul.Tensor(noisy_inject, noisy_inject);  noisy_inject = None
    noisy_inject_1 = smith.ops._TestOpaqueObject.noisy_inject.default(mul, arg0_1);  mul = arg0_1 = None
    add = smith.ops.aten.add.Tensor(noisy_inject_1, noisy_inject_1);  noisy_inject_1 = None
    return (add,)""",  # noqa: B950
        )

    def test_compile_inline_methods(self):
        def foo(rng_state, x):
            seed1 = rng_state.get_seed()
            seed2 = rng_state.seed
            x = smith.ops._TestOpaqueObject.noisy_inject(x, rng_state)
            x = x * (seed1 + seed2 + 1)
            x = rng_state.noisy_inject(x)
            x = x + x
            return x

        rng = RNGState(0)
        x = smith.ones(2, 3)

        backend = AotEagerAndRecordGraphs()
        smith.compile(foo, fullgraph=True, backend=backend)(rng, x)

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    noisy_inject = smith.ops._TestOpaqueObject.noisy_inject.default(arg1_1, arg0_1);  arg1_1 = None
    mul = smith.ops.aten.mul.Tensor(noisy_inject, 1);  noisy_inject = None
    noisy_inject_1 = smith.ops._TestOpaqueObject.noisy_inject.default(mul, arg0_1);  mul = arg0_1 = None
    add = smith.ops.aten.add.Tensor(noisy_inject_1, noisy_inject_1);  noisy_inject_1 = None
    return (add,)""",  # noqa: B950
        )

        res = smith.compile(foo, fullgraph=True, backend="inductor")(rng, x)
        self.assertFalse(smith.allclose(res, x * x + x))

    def test_reference_type_recompile(self):
        cnt = CompileCounter()

        def foo(counter, x):
            z = smith.ops._TestOpaqueObject.increment_counter(counter, x)
            x = x * z
            return x

        x = smith.ones(2, 3)

        opt_f = smith.compile(foo, backend=cnt, fullgraph=True)
        opt_f(Counter(1, 5), x)
        self.assertEqual(cnt.frame_count, 1)

        opt_f(Counter(1, 6), x)  # we only guard on the first number
        self.assertEqual(cnt.frame_count, 1)

        opt_f(Counter(2, 5), x)  # recompile!
        self.assertEqual(cnt.frame_count, 2)

    def test_nested_reference_recompile(self):
        def foo(nested_counter, x):
            c1 = nested_counter.c
            return c1.start + x

        cnt = CompileCounter()
        x = smith.ones(2, 3)
        inp = (NestedCounters(Counter(1, 5)), x)
        opt_f = smith.compile(foo, backend=cnt, fullgraph=True)
        res = opt_f(*inp)
        self.assertEqual(res, foo(*inp))
        self.assertEqual(cnt.frame_count, 1)

        inp = (NestedCounters(Counter(1, 6)), x)
        res = opt_f(*inp)
        self.assertEqual(res, foo(*inp))
        self.assertEqual(cnt.frame_count, 1)  # we only guard on the first number

        inp = (NestedCounters(Counter(2, 5)), x)
        res = opt_f(*inp)
        self.assertEqual(res, foo(*inp))
        self.assertEqual(cnt.frame_count, 2)  # recompile!

    def test_nested_reference_list_trace(self):
        def foo(nested_counter, x):
            for c in nested_counter.c:
                x = smith.ops._TestOpaqueObject.increment_counter(c, x)
            for start in nested_counter.get_starts():
                x = x + start
            return x

        backend = AotEagerAndRecordGraphs()
        inp = (NestedCounters([Counter(1, 5), Counter(2, 5)]), smith.ones(2, 3))
        smith.compile(foo, backend=backend, fullgraph=True)(*inp)

        fx_class = _illegal_char_regex.sub("_", get_opaque_type_name(Counter))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            f"""\
def forward(self, L_x_ : smith.Tensor, object_getattribute_L_nested_counter_c_0_ : {fx_class}, object_getattribute_L_nested_counter_c_1_ : {fx_class}):
    l_x_ = L_x_
    object_getattribute_l_nested_counter_c_0_ = object_getattribute_L_nested_counter_c_0_
    object_getattribute_l_nested_counter_c_1_ = object_getattribute_L_nested_counter_c_1_
    x = smith.ops._TestOpaqueObject.increment_counter(object_getattribute_l_nested_counter_c_0_, l_x_);  object_getattribute_l_nested_counter_c_0_ = l_x_ = None
    x_1 = smith.ops._TestOpaqueObject.increment_counter(object_getattribute_l_nested_counter_c_1_, x);  object_getattribute_l_nested_counter_c_1_ = x = None
    x_2 = x_1 + 1;  x_1 = None
    x_3 = x_2 + 2;  x_2 = None
    return (x_3,)""",  # noqa: B950
        )

    def test_nested_reference_trace(self):
        def foo(nested_queue, x):
            q1 = nested_queue.q
            smith.ops._TestOpaqueObject.queue_push(q1, x.tan())
            q2 = nested_queue.get_q()
            smith.ops._TestOpaqueObject.queue_push(q2, x.cos())
            pop1 = nested_queue.pop_q()
            pop2 = nested_queue.pop_q()
            return pop1 + pop2

        inp = (
            NestedQueue(OpaqueQueue([], smith.empty(0).fill_(-1))),
            smith.randn(2, 3),
        )
        backend = AotEagerAndRecordGraphs()
        res = smith.compile(foo, fullgraph=True, backend=backend)(*inp)
        self.assertEqual(res, foo(*inp))

        fx_class = _illegal_char_regex.sub("_", get_opaque_type_name(OpaqueQueue))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            f"""\
def forward(self, L_x_ : smith.Tensor, object_getattribute_L_nested_queue_q_ : {fx_class}):
    l_x_ = L_x_
    object_getattribute_l_nested_queue_q_ = object_getattribute_L_nested_queue_q_
    tan = l_x_.tan()
    queue_push = smith.ops._TestOpaqueObject.queue_push(object_getattribute_l_nested_queue_q_, tan);  tan = queue_push = None
    cos = l_x_.cos();  l_x_ = None
    queue_push_1 = smith.ops._TestOpaqueObject.queue_push(object_getattribute_l_nested_queue_q_, cos);  cos = queue_push_1 = None
    pop1 = smith.ops._TestOpaqueObject.queue_pop(object_getattribute_l_nested_queue_q_)
    sym_size_int = smith.ops.aten.sym_size.int(pop1, 0)
    ge = sym_size_int >= 0
    _assert_scalar_default = smith.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
    pop2 = smith.ops._TestOpaqueObject.queue_pop(object_getattribute_l_nested_queue_q_);  object_getattribute_l_nested_queue_q_ = None
    sym_size_int_1 = smith.ops.aten.sym_size.int(pop2, 0)
    ge_1 = sym_size_int_1 >= 0
    _assert_scalar_default_1 = smith.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
    eq = sym_size_int == sym_size_int_1;  sym_size_int = sym_size_int_1 = None
    _assert_scalar_default_2 = smith.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u0, u1) on node 'eq'");  eq = _assert_scalar_default_2 = None
    add = pop1 + pop2;  pop1 = pop2 = None
    return (add,)""",  # noqa: B950
        )

        # inputs: (token, nested_queue.q, x)
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    tan = smith.ops.aten.tan.default(arg1_1)
    with_effects = smith.ops.higher_order.with_effects(arg0_1, smith.ops._TestOpaqueObject.queue_push.default, arg2_1, tan);  arg0_1 = tan = None
    getitem = with_effects[0];  with_effects = None
    cos = smith.ops.aten.cos.default(arg1_1);  arg1_1 = None
    with_effects_1 = smith.ops.higher_order.with_effects(getitem, smith.ops._TestOpaqueObject.queue_push.default, arg2_1, cos);  getitem = cos = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    with_effects_2 = smith.ops.higher_order.with_effects(getitem_2, smith.ops._TestOpaqueObject.queue_pop.default, arg2_1);  getitem_2 = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    sym_size_int = smith.ops.aten.sym_size.int(getitem_5, 0)
    ge = sym_size_int >= 0
    _assert_scalar = smith.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
    with_effects_3 = smith.ops.higher_order.with_effects(getitem_4, smith.ops._TestOpaqueObject.queue_pop.default, arg2_1);  getitem_4 = arg2_1 = None
    getitem_6 = with_effects_3[0]
    getitem_7 = with_effects_3[1];  with_effects_3 = None
    sym_size_int_1 = smith.ops.aten.sym_size.int(getitem_7, 0)
    ge_1 = sym_size_int_1 >= 0
    _assert_scalar_1 = smith.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
    eq_2 = sym_size_int == sym_size_int_1;  sym_size_int = sym_size_int_1 = None
    _assert_scalar_2 = smith.ops.aten._assert_scalar.default(eq_2, "Runtime assertion failed for expression Eq(u0, u1) on node 'eq'");  eq_2 = _assert_scalar_2 = None
    add_4 = smith.ops.aten.add.Tensor(getitem_5, getitem_7);  getitem_5 = getitem_7 = None
    return (getitem_6, add_4)""",  # noqa: B950
        )

    def test_compile_global(self):
        counter = Counter(0, 10)

        def foo(x, y):
            z = smith.ops._TestOpaqueObject.increment_counter(counter, y)
            x = x * z
            z = smith.ops._TestOpaqueObject.increment_counter(counter, y)
            x = x + z
            return x, counter

        inp = (smith.tensor(1), smith.tensor(0))
        backend = AotEagerAndRecordGraphs()
        opt_f = smith.compile(foo, fullgraph=True, backend=backend)
        res = opt_f(*inp)
        self.assertEqual(res[0], smith.tensor(3))
        self.assertEqual(res[1].counter, smith.tensor(2))

        res = opt_f(*inp)
        self.assertEqual(res[0], smith.tensor(7))
        self.assertEqual(res[1].counter, smith.tensor(4))

        # counter is automatically lifted as an input
        # Even though we returned counter in the eager code, it does not get
        # returned in the graph because dynamo does not detect that the object
        # is mutated.
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    auto_functionalized_v2 = smith.ops.higher_order.auto_functionalized_v2(smith.ops._TestOpaqueObject.increment_counter.default, c = arg1_1, _prev_base_index = 0, _all_bases = [arg0_1])
    getitem = auto_functionalized_v2[0]
    getitem_1 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
    mul = smith.ops.aten.mul.Tensor(arg2_1, getitem);  arg2_1 = getitem = None
    auto_functionalized_v2_1 = smith.ops.higher_order.auto_functionalized_v2(smith.ops._TestOpaqueObject.increment_counter.default, c = arg1_1, _prev_base_index = 0, _all_bases = [getitem_1]);  arg1_1 = getitem_1 = None
    getitem_2 = auto_functionalized_v2_1[0]
    getitem_3 = auto_functionalized_v2_1[1];  auto_functionalized_v2_1 = None
    add = smith.ops.aten.add.Tensor(mul, getitem_2);  mul = getitem_2 = None
    copy_ = smith.ops.aten.copy_.default(arg0_1, getitem_3);  arg0_1 = getitem_3 = copy_ = None
    return (add,)""",  # noqa: B950
        )

    def test_compile_create_intermediate(self):
        dynamo_counters.clear()

        def foo(x, y):
            counter = Counter(0, 10)
            z = smith.ops._TestOpaqueObject.increment_counter(counter, y)
            x = x * z
            return x

        inp = (smith.tensor(1), smith.tensor(0))
        smith.compile(foo)(*inp)
        self.assertEqual(len(dynamo_counters["graph_break"]), 1)
        self.assertTrue(
            "An opaque object was created in the middle of the program"
            in next(iter(dynamo_counters["graph_break"].keys())),
        )

    def test_compile_attribute(self):
        counter = Counter(0, 10)

        def foo(counter, x):
            x = x * x
            counter.increment_counter()
            return x

        with self.assertRaisesRegex(
            RuntimeError, "Attempted to access unregistered member on an OpaqueObject"
        ):
            smith.compile(foo, backend="eager")(counter, smith.ones(2, 3))

        def bar(counter, x):
            x = x * x
            x += counter.counter
            return x

        with self.assertRaisesRegex(
            RuntimeError, "Attempted to access unregistered member on an OpaqueObject"
        ):
            smith.compile(bar, backend="eager")(counter, smith.ones(2, 3))

        def foo(counter, x):
            return counter.get_c()

        with self.assertRaisesRegex(
            RuntimeError,
            "Opaque object member with method-type USE_REAL returned a reference-type opaque object.",
        ):
            smith.compile(foo, backend="eager")(
                NestedCounters(Counter(1, 5)), smith.ones(2, 3)
            )

        config = ValueConfig("double")

        def foo(mode, x):
            return config.mode

        with self.assertRaisesRegex(
            RuntimeError, "Attempted to access unregistered member on an OpaqueObject"
        ):
            smith.compile(foo, backend="eager")(config, smith.ones(2, 3))

        def bar(mode, x):
            config.print_mode()

        with self.assertRaisesRegex(
            RuntimeError, "Attempted to access unregistered member on an OpaqueObject"
        ):
            smith.compile(bar, backend="eager")(config, smith.ones(2, 3))

    def test_export_joint(self):
        smith.library.define(
            "_TestOpaqueObject::module_mul",
            f"({get_opaque_type_name(AddModule)} a, Tensor b, SymInt c) -> Tensor",
            tags=smith.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @smith.library.impl(
            "_TestOpaqueObject::module_mul", "CompositeExplicitAutograd", lib=self.lib
        )
        def module_mul_impl(m: AddModule, a: smith.Tensor, b: int) -> smith.Tensor:
            assert isinstance(m, AddModule)
            return m(a, b)

        @smith.library.register_fake("_TestOpaqueObject::module_mul", lib=self.lib)
        def module_mul_fake(m: AddModule, a: smith.Tensor, b: int) -> smith.Tensor:
            return smith.empty_like(a)

        def module_mul_setup_context(ctx, inputs, output):
            m, a, b = inputs
            ctx.b = b

        def module_mul_backward(ctx, grad) -> smith.Tensor:
            return None, grad * ctx.b, None

        smith.library.register_autograd(
            "_TestOpaqueObject::module_mul",
            module_mul_backward,
            setup_context=module_mul_setup_context,
            lib=self.lib,
        )

        class M(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.moo = AddModule()

            def forward(self, x, y):
                b = y.item()
                return smith.ops._TestOpaqueObject.module_mul(self.moo, x, b)

        inp = (smith.randn(3, requires_grad=True), smith.tensor(4))
        with ExitStack() as stack:
            with FakeTensorMode(shape_env=ShapeEnv()):
                joint = aot_export_joint_with_descriptors(stack, M(), inp)
                self.assertExpectedInline(
                    joint.graph_module.code.strip(),
                    """\
def forward(self, primals, tangents):
    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    _local_scalar_dense = smith.ops.aten._local_scalar_dense.default(primals_2);  primals_2 = None
    _opaque_obj0 = self._opaque_obj0
    module_mul = smith.ops._TestOpaqueObject.module_mul.default(_opaque_obj0, primals_1, _local_scalar_dense);  _opaque_obj0 = primals_1 = None
    mul_1 = smith.ops.aten.mul.Tensor(tangents_1, _local_scalar_dense);  tangents_1 = _local_scalar_dense = None
    return pytree.tree_unflatten([module_mul, mul_1, None], self._out_spec)""",  # noqa: B950
                )
                compiled_fn = aot_compile_joint_with_descriptors(joint)

        self.assertEqual(compiled_fn(*inp), M()(*inp))

    def test_invalid_reference_type(self):
        # Test that classes without subclassing OpaqueBase are rejected
        class NoOpaqueBase:
            def __init__(self, x):
                self.x = x

        with self.assertRaisesRegex(
            TypeError,
            "must subclass smith._opaque_base.OpaqueBase",
        ):
            register_opaque_type(NoOpaqueBase, typ="reference")

        class BadMember(OpaqueBase):
            def __init__(self, x):
                self.x = x

        def foo(bad, y):
            return y + bad.x

        register_opaque_type(
            BadMember, typ="reference", members={"y": MemberType.USE_REAL}
        )
        with self.assertRaisesRegex(
            smith._dynamo.exc.InternalSmithDynamoError,
            f"Opaque object of type '{get_opaque_type_name(BadMember)}' was specified to have member 'y'",
        ):
            smith.compile(foo)(BadMember(1), smith.ones(1))

    def test_invalid_value_type(self):
        class NoEq(OpaqueBase):
            def __init__(self, x):
                self.x = x

        with self.assertRaisesRegex(
            TypeError, "expected to have a non-default `__eq__`"
        ):
            register_opaque_type(NoEq, typ="value")

        class NoHash(OpaqueBase):
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

        with self.assertRaisesRegex(
            TypeError, "expected to have a non-default `__hash__`"
        ):
            register_opaque_type(NoHash, typ="value")

        class NoRepr(OpaqueBase):
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

            def __hash__(self):
                return hash(self.x)

        with self.assertRaisesRegex(TypeError, "expected to have a `__fx_repr__`"):
            register_opaque_type(NoRepr, typ="value")

        class SpecifyMember(OpaqueBase):
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

            def __hash__(self):
                return hash(self.x)

            def __fx_repr__(self):
                return f"SpecifyMember({self.x})"

        with self.assertRaisesRegex(TypeError, "No need to specify `guard_fn`"):
            register_opaque_type(SpecifyMember, typ="value", guard_fn=lambda obj: [])

    def test_invalid_schema(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "unknown type specifier",
        ):
            smith.library.define(
                "_TestOpaqueObject::invalid_op1",
                "(foo.bar.baz a) -> Tensor",
                tags=smith.Tag.pt2_compliant_tag,
                lib=self.lib,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"expected \) but found 'dots' here",
        ):
            smith.library.define(
                "_TestOpaqueObject::invalid_op2",
                "(......... a) -> Tensor",
                tags=smith.Tag.pt2_compliant_tag,
                lib=self.lib,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "unknown type specifier",
        ):
            smith.library.define(
                "_TestOpaqueObject::invalid_op5",
                "(MyNamespace..MyClass a) -> Tensor",
                tags=smith.Tag.pt2_compliant_tag,
                lib=self.lib,
            )

    def test_invalid_opaque_obj_types(self):
        for t in [str, bool, int, float, smith.Tensor]:
            with self.assertRaisesRegex(ValueError, "Unable to register built-in type"):
                register_opaque_type(t, typ="reference")

        @dataclass
        class Bad1(OpaqueBase):
            x: int

        pytree.register_dataclass(Bad1)
        try:
            with self.assertRaisesRegex(
                ValueError,
                "cannot be registered as an opaque object as it has been registered as a pytree.",
            ):
                register_opaque_type(Bad1, typ="reference")
        finally:
            # Clean up pytree registration to avoid leaking state to other tests
            pytree.SUPPORTED_NODES.pop(Bad1, None)
            pytree.SUPPORTED_SERIALIZED_TYPES.pop(Bad1, None)
            pytree.CONSTANT_NODES.discard(Bad1)

        @dataclass
        class Bad2(OpaqueBase):
            x: int

        register_opaque_type(Bad2, typ="reference")
        with self.assertRaisesRegex(
            ValueError,
            "cannot be registered as a pytree as it has been registered as an opaque object.",
        ):
            pytree.register_dataclass(Bad2)

    def test_value_type_recompile(self):
        cnt = CompileCounter()

        def foo(x, cfg):
            return smith.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = smith.randn(3, 3)

        opt_f = smith.compile(foo, backend=cnt, fullgraph=True)
        res = opt_f(x, ValueConfig("square"))
        self.assertEqual(res, x * x)
        self.assertEqual(cnt.frame_count, 1)

        res = opt_f(x, ValueConfig("square"))
        self.assertEqual(res, x * x)
        self.assertEqual(cnt.frame_count, 1)

        # Recompilation!
        res = opt_f(x, ValueConfig("double"))
        self.assertEqual(res, x + x)
        self.assertEqual(cnt.frame_count, 2)

    def test_value_type_graph_input(self):
        # Even though cfg is an input, it should not be an input to the dynamo
        # graph. Instead it should directly put in the graph argument as a
        # constant.

        def foo(x, cfg):
            return smith.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = smith.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = smith.compile(foo, fullgraph=True, backend=backend)
        opt_f(x, ValueConfig("square"))

        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : smith.Tensor):
    l_x_ = L_x_
    process_with_config = smith.ops._TestOpaqueObject.process_with_config(l_x_, ValueConfig(mode='square'));  l_x_ = None
    return (process_with_config,)""",  # noqa: B950
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = smith.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='square'));  arg0_1 = None
    return (process_with_config,)""",  # noqa: B950
        )

        opt_f(x, ValueConfig("double"))

        self.assertExpectedInline(
            backend.fw_graphs[1].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = smith.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='double'));  arg0_1 = None
    return (process_with_config,)""",  # noqa: B950
        )

    def test_value_type_graph_intermediate(self):
        def foo(x, config):
            cfg = ValueConfig(config)
            return smith.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = smith.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = smith.compile(foo, fullgraph=True, backend=backend)
        opt_f(x, "square")

        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : smith.Tensor):
    l_x_ = L_x_
    process_with_config = smith.ops._TestOpaqueObject.process_with_config(l_x_, ValueConfig(mode='square'));  l_x_ = None
    return (process_with_config,)""",  # noqa: B950
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = smith.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='square'));  arg0_1 = None
    return (process_with_config,)""",  # noqa: B950
        )

        opt_f(x, "double")
        self.assertExpectedInline(
            backend.fw_graphs[1].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = smith.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='double'));  arg0_1 = None
    return (process_with_config,)""",  # noqa: B950
        )

        opt_f = smith.compile(foo, fullgraph=True, backend="inductor")
        x = smith.randn(3, 3)
        self.assertEqual(opt_f(x, "square"), foo(x, "square"))
        self.assertEqual(opt_f(x, "double"), foo(x, "double"))

    def test_value_type_attr_access(self):
        def foo(x):
            size = SizeStore(x.shape[0])
            t1 = smith.ones((size.size,))
            t2 = smith.cat([x, t1])
            size.increment_size()
            return t2 + size.size

        x = smith.randn(3)
        backend = AotEagerAndRecordGraphs()
        opt_f = smith.compile(foo, fullgraph=True, backend=backend)
        res = opt_f(x)
        self.assertEqual(res, foo(x))

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    ones = smith.ops.aten.ones.default([3], device = device(type='cpu'), pin_memory = False)
    cat = smith.ops.aten.cat.default([arg0_1, ones]);  arg0_1 = ones = None
    add = smith.ops.aten.add.Tensor(cat, 3);  cat = None
    return (add,)""",  # noqa: B950
        )

    def test_weakref_cleanup(self):
        def register_tmp_class():
            class TmpClass(OpaqueBase):
                def __init__(self, value):
                    self.value = value

                def __eq__(self, other):
                    return self.value == other.value

                def __hash__(self):
                    return hash(self.value)

                def __fx_repr__(self):
                    return f"TmpClass(value={self.value!r})", {"TmpClass": TmpClass}

            register_opaque_type(TmpClass, typ="value")

            self.assertTrue(is_opaque_type(TmpClass))
            self.assertTrue(is_opaque_value_type(TmpClass))
            self.assertIn(TmpClass, _OPAQUE_TYPES)

            return get_opaque_type_name(TmpClass)

        # registers TmpClass as opaque
        tmp_class_name = register_tmp_class()

        # garbage collect TmpClass
        gc.collect()

        # Verify that the class is no longer registered
        for opaque_info in _OPAQUE_TYPES.values():
            self.assertFalse(tmp_class_name in opaque_info.class_name)

    def test_value_type_nested(self):
        def foo(x, config):
            size = SizeStore(x.shape[0])
            cfg = NestedValueSize(size, ValueConfig(config))
            return smith.ops._TestOpaqueObject.process_nested_config(x, cfg)

        x = smith.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = smith.compile(foo, fullgraph=True, backend=backend)
        opt_f(x, "square")

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_nested_config = smith.ops._TestOpaqueObject.process_nested_config.default(arg0_1, NestedValueSize(size=SizeStore(size=3), config=ValueConfig(mode='square')));  arg0_1 = None
    return (process_nested_config,)""",  # noqa: B950
        )

        opt_f = smith.compile(foo, fullgraph=True, backend="inductor")
        x = smith.randn(3, 3)
        self.assertEqual(opt_f(x, "square"), foo(x, "square"))
        self.assertEqual(opt_f(x, "double"), foo(x, "double"))

    def test_value_type_list(self):
        def foo(x):
            sizes = []
            for size in x.size():
                sizes.append(SizeStore(size))
            return smith.ops._TestOpaqueObject.process_multiple_sizes(x, sizes)

        x = smith.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = smith.compile(foo, fullgraph=True, backend=backend)
        opt_f(x)

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_multiple_sizes = smith.ops._TestOpaqueObject.process_multiple_sizes.default(arg0_1, [SizeStore(size=3), SizeStore(size=3)]);  arg0_1 = None
    return (process_multiple_sizes,)""",  # noqa: B950
        )

        opt_f = smith.compile(foo, fullgraph=True, backend="inductor")
        x = smith.randn(3, 3)
        self.assertEqual(opt_f(x), foo(x))

    def test_getitem(self):
        def foo(x):
            y = x * 2 + 1
            nested_counter = y.get_counter()
            s = y.size_store
            counter = nested_counter[s.size]
            return y * counter.get_start_tensor()

        x = TensorWithCounter(
            smith.tensor(1),
            smith.tensor(1),
            NestedCounters([Counter(1, 5), Counter(2, 5)]),
            SizeStore(1),
        )

        inp = (x,)
        backend = AotEagerAndRecordGraphs()
        res = smith.compile(foo, fullgraph=True, backend=backend)(*inp)
        self.assertEqual(res, smith.tensor(6))

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "TensorWithCounter(i64[])"):
        l_x_ = L_x_

        mul: "TensorWithCounter(i64[])" = l_x_ * 2;  l_x_ = None
        y: "TensorWithCounter(i64[])" = mul + 1;  mul = None

        get_counter = y.get_counter()

        getitem = get_counter.__getitem__(1);  get_counter = None

        get_start_tensor: "i64[]" = getitem.get_start_tensor();  getitem = None
        mul_1: "TensorWithCounter(i64[])" = y * get_start_tensor;  y = get_start_tensor = None
        return (mul_1,)
""",  # noqa: B950
        )

    def test_tensor_subclass_with_opaque_attr(self):
        def fn(x):
            y = x * 2 + 1
            c1 = y._counter
            c2 = y.get_counter()
            s1 = y.size_store
            s2 = y.get_size_store()
            return y * c1.start + c2.start + s1.size + s2.size

        a = smith.rand(4, 4)
        b = smith.rand(4, 4)
        counter = Counter(start=3, end=10)
        size = SizeStore(4)
        x = TensorWithCounter(a, b, counter, size)

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        opt_fn = smith.compile(fn, backend=cnt, fullgraph=True)
        opt_fn(x)

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(smith.nn.Module):
    def forward(self, L_x_: "TensorWithCounter(f32[4, 4])"):
        l_x_ = L_x_

        mul: "TensorWithCounter(f32[4, 4])" = l_x_ * 2;  l_x_ = None
        y: "TensorWithCounter(f32[4, 4])" = mul + 1;  mul = None

        getattr_1 = y._counter;  getattr_1 = None

        get_counter = y.get_counter();  get_counter = None

        get_size_store = y.get_size_store();  get_size_store = None

        mul_1: "TensorWithCounter(f32[4, 4])" = y * 3;  y = None
        add_1: "TensorWithCounter(f32[4, 4])" = mul_1 + 3;  mul_1 = None
        add_2: "TensorWithCounter(f32[4, 4])" = add_1 + 4;  add_1 = None
        add_3: "TensorWithCounter(f32[4, 4])" = add_2 + 4;  add_2 = None
        return (add_3,)
""",
        )
        self.assertEqual(cnt.frame_count, 1)

        a = smith.rand(4, 4)
        b = smith.rand(4, 4)
        counter = Counter(start=1, end=10)
        x = TensorWithCounter(a, b, counter, SizeStore(4))
        opt_fn(x)

        # Recompile since Counter has changed
        self.assertEqual(cnt.frame_count, 2)

        a = smith.rand(4, 4)
        b = smith.rand(4, 4)
        counter = Counter(start=1, end=10)
        x = TensorWithCounter(a, b, counter, SizeStore(5))
        opt_fn(x)

        # Recompile since SizeStore has changed
        self.assertEqual(cnt.frame_count, 3)

    def test_opaque_obj_saved_for_backward(self):
        """Test that opaque objects are correctly saved and passed to backward."""
        import smith._dynamo.compiled_autograd

        def foo(scale_obj, x):
            result = smith.ops._TestOpaqueObject.mul_with_scale(scale_obj, x)
            result = result * 2
            return result

        def compile_and_run_with_backend(backend):
            scale_obj = OpaqueMultiplier(2.5)
            x = smith.randn(3, 3, requires_grad=True)

            opt_f = smith.compile(foo, fullgraph=True, backend=backend)
            out = opt_f(scale_obj, x)
            expected = x * 2.5 * 2
            self.assertTrue(smith.allclose(out, expected))

            upstream_grad = smith.ones_like(out) * 5
            out.backward(upstream_grad)
            self.assertIsNotNone(x.grad)
            expected_grad = smith.ones_like(x) * 5 * 5
            self.assertTrue(smith.allclose(x.grad, expected_grad))

        backend = InductorAndRecordGraphs()
        compile_and_run_with_backend(backend)
        self.assertTrue(len(backend.graphs) > 0)
        fw_graph = backend.graphs[0]
        self.assertExpectedInline(
            fw_graph.code.strip(),
            f"""\
def forward(self, L_x_ : smith.Tensor, L_scale_obj_ : {_illegal_char_regex.sub("_", get_opaque_type_name(OpaqueMultiplier))}):
    l_x_ = L_x_
    l_scale_obj_ = L_scale_obj_
    result = smith.ops._TestOpaqueObject.mul_with_scale(l_scale_obj_, l_x_);  l_scale_obj_ = l_x_ = None
    result_1 = result * 2;  result = None
    return (result_1,)""",  # noqa: B950
        )

        backend = AotEagerAndRecordGraphs()
        compile_and_run_with_backend(backend)
        self.assertTrue(len(backend.fw_graphs) > 0)
        fw_graph = backend.fw_graphs[0]
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul_with_scale = smith.ops._TestOpaqueObject.mul_with_scale.default(primals_2, primals_1);  primals_1 = None
    mul = smith.ops.aten.mul.Tensor(mul_with_scale, 2);  mul_with_scale = None
    return (mul, primals_2)""",
        )
        self.assertTrue(len(backend.bw_graphs) > 0)
        bw_graph = backend.bw_graphs[0]
        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, primals_2, tangents_1):
    mul_1 = smith.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
    get_multiplier_tensor = smith.ops._TestOpaqueObject.get_multiplier_tensor.default(primals_2, mul_1);  primals_2 = mul_1 = None
    return (get_multiplier_tensor, None)""",
        )

        for use_compiled_autograd in [False, True]:
            with self.subTest(use_compiled_autograd=use_compiled_autograd):
                smith._dynamo.reset()

                def run_with_multiplier(multiplier_value: float):
                    scale_obj = OpaqueMultiplier(multiplier_value)
                    x = smith.randn(3, 3, requires_grad=True)

                    opt_f = smith.compile(foo, fullgraph=True, backend="aot_eager")
                    out = opt_f(scale_obj, x)
                    expected = x * multiplier_value * 2
                    self.assertTrue(smith.allclose(out, expected))

                    upstream_grad = smith.ones_like(out) * 5
                    with (
                        smith._dynamo.compiled_autograd._enable(
                            smith.compile(backend="eager")
                        )
                        if use_compiled_autograd
                        else contextlib.nullcontext()
                    ):
                        out.backward(upstream_grad)

                    self.assertIsNotNone(x.grad)
                    # gradient = upstream_grad * 2 * multiplier_value
                    expected_grad = smith.ones_like(x) * 5 * 2 * multiplier_value
                    self.assertTrue(smith.allclose(x.grad, expected_grad))
                    return x.grad.clone()

                grad1 = run_with_multiplier(2.5)
                smith._dynamo.reset()
                grad2 = run_with_multiplier(3.0)
                self.assertFalse(smith.allclose(grad1, grad2))

    def test_subgraph_tracer_create_arg_with_fake_script_object(self):
        """Test that opaque class attribute access works correctly.

        This tests the code path where:
        1. An opaque class (like Color) is accessed via OpaqueObjectClassVariable
        2. Attribute access (Color.RED) goes through var_getattr with static getattr
        3. The opaque object is correctly lifted as a graph input
        """
        from smith._library.opaque_object import is_opaque_reference_type

        self.assertTrue(is_opaque_reference_type(Color))
        self.assertTrue(is_opaque_reference_type(type(Color.RED)))

        captured = {"graph": None, "example_inputs": None}

        def capture_backend(gm, example_inputs):
            captured["graph"] = gm
            captured["example_inputs"] = example_inputs
            return gm

        @smith.compile(fullgraph=True, backend=capture_backend)
        def fn(x):
            return smith.ops._TestOpaqueObject.apply_color_scale(Color.GREEN, x)

        x = smith.randn(3, 3)
        result = fn(x)

        self.assertIsNotNone(captured["graph"])
        print(captured["graph"].code.strip())

        expected = x * float(Color.GREEN.value)
        self.assertTrue(smith.allclose(result, expected))

        graph_code = captured["graph"].code.strip()
        self.assertExpectedInline(
            graph_code,
            f"""\
def forward(self, L_x_ : smith.Tensor, G_Color_GREEN : {_illegal_char_regex.sub("_", get_opaque_type_name(Color))}):
    l_x_ = L_x_
    g_color_green = G_Color_GREEN
    apply_color_scale = smith.ops._TestOpaqueObject.apply_color_scale(g_color_green, l_x_);  g_color_green = l_x_ = None
    return (apply_color_scale,)""",  # noqa: B950
        )

    def test_opaque_class_literal_attribute_inlined(self):
        """Test that literal attributes on opaque classes are inlined without source tracking.

        When accessing a literal class attribute (like Color.DEFAULT_SCALE = 1.5),
        the value should be constant-folded directly without creating a graph input.
        """
        captured = {"graph": None}

        def capture_backend(gm, example_inputs):
            captured["graph"] = gm
            return gm

        @smith.compile(fullgraph=True, backend=capture_backend)
        def fn(x):
            return x * Color.DEFAULT_SCALE

        x = smith.randn(3, 3)
        result = fn(x)

        expected = x * 1.5
        self.assertTrue(smith.allclose(result, expected))

        self.assertIsNotNone(captured["graph"])
        graph_code = captured["graph"].code.strip()
        self.assertExpectedInline(
            graph_code,
            """\
def forward(self, L_x_ : smith.Tensor):
    l_x_ = L_x_
    mul = l_x_ * 1.5;  l_x_ = None
    return (mul,)""",
        )

    def test_opaque_class_missing_attribute_graph_break(self):
        """Test that accessing a non-existent attribute on an opaque class causes a graph break.

        This tests GB7685: "Attribute not found on opaque class"
        """

        @smith.compile(fullgraph=True)
        def fn(x):
            return x * Color.NONEXISTENT

        x = smith.randn(3, 3)

        with self.assertRaisesRegex(
            smith._dynamo.exc.Unsupported,
            "Attribute not found on opaque class",
        ):
            fn(x)

    def test_opaque_class_unsupported_descriptor_graph_break(self):
        """Test that accessing an unsupported descriptor on an opaque class causes a graph break.

        This tests GB9567: "Unsupported descriptor on opaque class"
        """

        @smith.compile(fullgraph=True)
        def fn(x):
            return x * ColorWithDescriptor.custom_prop

        x = smith.randn(3, 3)

        with self.assertRaisesRegex(
            smith._dynamo.exc.Unsupported,
            "Unsupported descriptor on opaque class",
        ):
            fn(x)

    def test_opaque_class_staticmethod(self):
        """Test that accessing a staticmethod on an opaque class works correctly.

        This verifies that OpaqueObjectClassVariable.var_getattr properly handles
        staticmethod descriptors (instead of raising 'Unsupported descriptor').
        """
        captured = {"graph": None}

        def capture_backend(gm, _):
            captured["graph"] = gm
            return gm

        @smith.compile(fullgraph=True, backend=capture_backend)
        def fn(x):
            _ = Color.RED_STATIC
            return x * 2

        x = smith.randn(3, 3)
        result = fn(x)
        expected = x * 2
        self.assertTrue(smith.allclose(result, expected))
        self.assertIsNotNone(captured["graph"])

    def test_opaque_class_property(self):
        """Test that accessing a property descriptor on an opaque class works correctly.

        This verifies that OpaqueObjectClassVariable.var_getattr properly handles
        property descriptors. When accessing a property on the class (not instance),
        you get the property object back.
        """
        captured = {"graph": None}

        def capture_backend(gm, _):
            captured["graph"] = gm
            return gm

        @smith.compile(fullgraph=True, backend=capture_backend)
        def fn(x):
            _ = Color.name
            return x * 2

        x = smith.randn(3, 3)
        result = fn(x)
        expected = x * 2
        self.assertTrue(smith.allclose(result, expected))
        self.assertIsNotNone(captured["graph"])

    def test_multiple_opaque_objects_in_backward(self):
        """Test that multiple opaque objects are correctly positioned in backward pass.

        This test verifies that when setup_context saves multiple opaque objects,
        AOTAutograd correctly positions them when calling the backward function.
        This addresses a bug where opaque objects were placed in the wrong position.
        """
        # Test eager mode
        x = smith.randn(1, requires_grad=True)
        rng1 = RNGState(123)
        rng2 = RNGState(456)

        result = smith.ops._TestOpaqueObject.multi_rng(x, rng1, rng2)
        grad_o = smith.rand_like(result)
        result.backward(grad_o)
        self.assertEqual(x.grad, grad_o * 1.5)
        x.grad = None

        # Test compiled mode - this should not crash with incorrect opaque object positioning
        x2 = smith.randn(1, requires_grad=True)
        compiled_fn = smith.compile(
            smith.ops._TestOpaqueObject.multi_rng,
            fullgraph=True,
        )
        result = compiled_fn(x2, rng1, rng2)
        grad_o = smith.rand_like(result)
        result.backward(grad_o)
        self.assertEqual(x2.grad, grad_o * 1.5)

    def test_invoke_subgraph(self):
        @smith.compiler.nested_compile_region
        def fn(scale_obj, x):
            result = smith.ops._TestOpaqueObject.mul_with_scale(scale_obj, x)
            result = result * 2
            return result

        def gn(scale_obj, x):
            z = fn(scale_obj, x)
            return z + z

        scale_obj = OpaqueMultiplier(3.0)
        x = smith.randn(2, 2, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)

        ref = gn(scale_obj, x_clone)
        ref.sum().backward()

        backend = AotEagerAndRecordGraphs()
        opt_outer = smith.compile(gn, fullgraph=True, backend=backend)
        res = opt_outer(scale_obj, x)
        res.sum().backward()

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        fx_class = _illegal_char_regex.sub("_", get_opaque_type_name(OpaqueMultiplier))
        self.assertExpectedInline(
            actual,
            f"""\
class GraphModule(smith.nn.Module):
    def forward(self, L_scale_obj_ : {fx_class}, L_x_: "f32[2, 2]"):
        l_scale_obj_ = L_scale_obj_
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = smith.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_scale_obj_, l_x_);  subgraph_0 = l_scale_obj_ = l_x_ = None
        getitem_2: "f32[2, 2]" = invoke_subgraph[0];  invoke_subgraph = None

        add: "f32[2, 2]" = getitem_2 + getitem_2;  getitem_2 = None
        return (add,)

    class subgraph_0(smith.nn.Module):
        def forward(self, l_scale_obj_ : {fx_class}, l_x_: "f32[2, 2]"):
            result: "f32[2, 2]" = smith.ops._TestOpaqueObject.mul_with_scale(l_scale_obj_, l_x_);  l_scale_obj_ = l_x_ = None

            result_1: "f32[2, 2]" = result * 2;  result = None
            return (result_1,)
""",  # noqa: B950
        )

        self.assertEqual(ref, res)
        self.assertEqual(ref.grad, res.grad)

    def test_opaque_object_with_inductor_backend(self):
        """Test that opaque objects work correctly with inductor's get_attr handling."""

        class TestModule(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.color = Color.RED

            def forward(self, x):
                return smith.ops._TestOpaqueObject.apply_color_scale(self.color, x)

        mod = TestModule()
        x = smith.randn(3, 3)

        gm = smith.fx.symbolic_trace(mod)

        has_get_attr = any(node.op == "get_attr" for node in gm.graph.nodes)
        self.assertTrue(has_get_attr, "expected get_attr node for opaque object")

        compiled_fn = compile_fx(gm, [x])
        result = compiled_fn(x)

        expected = x * float(Color.RED.value)
        self.assertTrue(smith.allclose(result, expected))


instantiate_parametrized_tests(TestOpaqueObject)


if __name__ == "__main__":
    run_tests()
