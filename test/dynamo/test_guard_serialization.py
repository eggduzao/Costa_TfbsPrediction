# Owner(s): ["module: dynamo"]

import dataclasses
import pickle
import sys
import tempfile
import types
import unittest
import weakref
from collections.abc import Iterator
from typing import NamedTuple
from unittest.mock import patch

import smith
import smith._dynamo.testing
import smith._inductor.config
import smith._inductor.test_case
import smith.fx.graph as fx_graph
import smith.onnx.operators
import smith.utils.cpp_extension
from smith._dynamo.bytecode_transformation import transform_code_object
from smith._dynamo.exc import PackageError
from smith._dynamo.guards import CheckFunctionManager, CompileId
from smith._dynamo.package import CompilePackage
from smith._dynamo.source import LocalSource
from smith._dynamo.symbolic_convert import (
    ExceptionStack,
    InstructionTranslator,
    SpeculationLog,
)
from smith._dynamo.utils import dynamo_timed, get_metrics_context
from smith._guards import compile_context, CompileContext, tracing
from smith.overrides import SmithFunctionMode
from smith.testing._internal.common_utils import IS_MACOS
from smith.testing._internal.inductor_utils import HAS_GPU
from smith.utils import _pytree as pytree


@dataclasses.dataclass
class _FrameState:
    f_locals: dict
    f_globals: dict
    f_code: types.CodeType
    f_builtins: dict


class GlobalModule(smith.nn.Module):
    def forward(self, x):
        return x + 1


class GlobalNestedModule(smith.nn.Module):
    def __init__(self, submodule=None):
        super().__init__()
        self.linear = smith.nn.Linear(10, 10)
        self.param = smith.nn.Parameter(smith.randn(3, 2))
        self.nested = submodule or GlobalModule()

    def forward(self, x):
        return self.linear(x) + 1


def global_func(x):
    return x + 1


class ModuleNotSerializable(smith.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = smith.nn.Parameter(smith.randn(3, 2))

    def __getstate__(self):
        raise NotImplementedError("not serialzable")

    def forward(self, x):
        return x + self.param


class GlobalSmithFunctionMode(SmithFunctionMode):
    def __smith_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class MyClass:
    def __getstate__(self):
        raise RuntimeError("Cannot pickle")

    def add(self, x):
        return x + 1


class MyClassNotSerializable:
    def __getstate__(self):
        raise NotImplementedError

    def add(self, x):
        return x + 1


class Inputs:
    def __init__(self, x, unused):
        self.x = x
        self.unused = unused


def _global_func_wrong_fqn(x):
    return x + 1


global_func_wrong_fqn = _global_func_wrong_fqn
del _global_func_wrong_fqn


class FlatModule(smith.nn.Module):
    def forward(self, x):
        return x + 2


class ModWithDict(smith.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d


class SubclassWithMeta(smith.Tensor):
    @staticmethod
    def __new__(cls, a, extra, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return smith.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, a, extra, outer_size=None, outer_stride=None):
        self.a = a
        self.extra = extra

    @classmethod
    def __smith_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(SubclassWithMeta, lambda x: x.a, args)
        kwargs_a = pytree.tree_map_only(SubclassWithMeta, lambda x: x.a, kwargs)
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, smith.Tensor):
            assert isinstance(args[0], SubclassWithMeta)
            return SubclassWithMeta(out_a, extra=args[0].extra)
        return out_a

    def __tensor_flatten__(self):
        # store extra in meta
        return ["a"], {"extra": self.extra}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert isinstance(meta, dict)
        a = inner_tensors["a"]
        # pull out extra from meta
        extra = meta["extra"]
        if type(a) is smith.Tensor:
            assert outer_size is not None
            assert outer_stride is not None
        return SubclassWithMeta(a, extra, outer_size, outer_stride)


class SubclassWithCustomMetadataGuard(smith.Tensor):
    @staticmethod
    def __new__(cls, a, extra, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return smith.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, a, extra, outer_size=None, outer_stride=None):
        self.a = a
        self.extra = extra

    @classmethod
    def __smith_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(
            SubclassWithCustomMetadataGuard, lambda x: x.a, args
        )
        kwargs_a = pytree.tree_map_only(
            SubclassWithCustomMetadataGuard, lambda x: x.a, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, smith.Tensor):
            assert isinstance(args[0], SubclassWithCustomMetadataGuard)
            return SubclassWithCustomMetadataGuard(out_a, extra=args[0].extra)
        return out_a

    @classmethod
    def __metadata_guard__(cls, meta1, meta2):
        # Define custom metadata guard logic that only looks at "bar" to determine
        # metadata equivalence. This is more purposefully more lax than the default
        # guard behavior.
        return meta1["extra"]["bar"] == meta2["extra"]["bar"]

    def __tensor_flatten__(self):
        # store extra in meta
        return ["a"], {"extra": self.extra}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert isinstance(meta, dict)
        a = inner_tensors["a"]
        # pull out extra from meta
        extra = meta["extra"]
        if type(a) is smith.Tensor:
            assert outer_size is not None
            assert outer_stride is not None
        return SubclassWithCustomMetadataGuard(a, extra, outer_size, outer_stride)


class SubclassWithSubclassInnerTensor(smith.Tensor):
    @staticmethod
    def __new__(cls, a, extra, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return smith.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, a, extra, outer_size=None, outer_stride=None):
        self.a = a
        self.inner_sub = SubclassWithMeta(a + 1, extra=extra)

    @classmethod
    def __smith_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(
            SubclassWithSubclassInnerTensor, lambda x: x.a, args
        )
        kwargs_a = pytree.tree_map_only(
            SubclassWithSubclassInnerTensor, lambda x: x.a, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, smith.Tensor):
            assert isinstance(args[0], SubclassWithSubclassInnerTensor)
            return SubclassWithSubclassInnerTensor(out_a, extra=args[0].inner_sub.extra)
        return out_a

    def __tensor_flatten__(self):
        return ["a", "inner_sub"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        a = inner_tensors["a"]
        extra = inner_tensors["inner_sub"].extra
        if type(a) is smith.Tensor:
            assert outer_size is not None
            assert outer_stride is not None
        return SubclassWithSubclassInnerTensor(a, extra, outer_size, outer_stride)


# defines a custom __eq__() / __hash__() to be registered as a pytree constant type
class CustomConstantType(smith._opaque_base.OpaqueBase):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        # custom eq ignores b
        return self.a == other.a

    def __hash__(self):
        # custom hash ignores b
        return hash(self.a)

    def __repr__(self):
        return f"CustomConstantType(a={self.a!r}, b={self.b!r})"

    def __fx_repr__(self):
        return f"CustomConstantType(a={self.a!r}, b={self.b!r})", {
            "CustomConstantType": CustomConstantType
        }


smith._library.opaque_object.register_opaque_type(CustomConstantType, typ="value")


class TestGuardSerializationBase(smith._inductor.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._fx_magic_methods_snapshot = fx_graph.magic_methods.copy()
        self._saved_default_device_context = getattr(
            smith._GLOBAL_DEVICE_CONTEXT, "device_context", None
        )

    def tearDown(self):
        fx_graph.magic_methods.clear()
        fx_graph.magic_methods.update(self._fx_magic_methods_snapshot)

        current_ctx = getattr(smith._GLOBAL_DEVICE_CONTEXT, "device_context", None)
        if current_ctx is not self._saved_default_device_context:
            if self._saved_default_device_context is None:
                smith.set_default_device(None)
            else:
                smith.set_default_device(self._saved_default_device_context.device)

        super().tearDown()

    def _tracefunc(self, frame, event, arg):
        if event != "call":
            return

        if self._frame_state is not None:
            return

        self._frame_state = _FrameState(
            f_locals=dict(frame.f_locals),
            f_globals=frame.f_globals,
            f_code=frame.f_code,
            f_builtins=frame.f_builtins,
        )

    def _test_serialization(self, guard_type, fn, *args, **kwargs):
        # kwargs might contain a callable that generates kwargs
        smith._dynamo.reset()
        kwarg_gen_fn = kwargs.get("_gen_fn")
        if kwarg_gen_fn is not None:
            kwargs = kwarg_gen_fn()

        self._frame_state = None
        sys.settrace(self._tracefunc)
        if isinstance(fn, smith.nn.Module):
            fn = fn.forward
        try:
            fn(*args, **kwargs)
        finally:
            sys.settrace(None)

        assert self._frame_state is not None

        # Set f_locals from regenerated kwargs to handle exhausted input iterators
        # NB: This is super janky and might cause unforeseen problems
        if kwarg_gen_fn is not None:
            kwargs = kwarg_gen_fn()
            for key in self._frame_state.f_locals:
                if key in kwargs and isinstance(kwargs[key], Iterator):
                    self._frame_state.f_locals[key] = kwargs[key]

        def guard_filter_fn(guards):
            ret = [
                g.guard_type == guard_type or guard_type in g.derived_guard_types
                for g in guards
            ]
            self.assertTrue(any(ret))
            return ret

        ref_gm = None
        loaded_gm = None

        def transform(instructions: list, code_options: dict[str, object]):
            """
            The goal is here is not to reimplement dynamo, but just to have a
            simplified version to extract the state from symbolic convert.
            Should not work on all cases, but should work on simple functions
            in this test file.
            """
            nonlocal ref_gm
            nonlocal loaded_gm

            smith._dynamo.convert_frame.initial_global_state = (
                smith._C._dynamo.guards.GlobalStateGuard()
            )
            tracer = InstructionTranslator(
                instructions,
                self._frame_state.f_code,
                self._frame_state.f_locals,
                self._frame_state.f_globals,
                self._frame_state.f_builtins,
                fn.__closure__ or (),
                smith.overrides._get_current_function_mode_stack(),
                code_options,
                smith._dynamo.lookup_backend("eager"),
                one_graph=False,
                export=False,
                export_constraints=None,
                frame_state=None,
                speculation_log=SpeculationLog(),
                exn_vt_stack=ExceptionStack(),
                distributed_state=None,
                package=None,
            )
            with (
                compile_context(
                    CompileContext(CompileId(frame_id=0, frame_compile_id=0))
                ),
                tracing(tracer.output.tracing_context),
                tracer.set_current_tx(),
                get_metrics_context(),
                dynamo_timed(""),
            ):
                tracer.run()

                ref_gm = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                ).guard_manager

                check_fn_manager = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                    save_guards=True,
                )
                guards_state = check_fn_manager.guards_state
                self._cached_guards_state = guards_state
                self._cached_f_code = self._frame_state.f_code
                self.assertIsNotNone(guards_state)
                guards_state = smith._dynamo.package.load_guards_state(guards_state)

                loaded_gm = smith._dynamo.package.load_guard_manager(
                    guards_state,
                    self._frame_state.f_code,
                    self._frame_state.f_globals,
                )

        try:
            transform_code_object(self._frame_state.f_code, transform)
        finally:
            smith._dynamo.convert_frame.initial_global_state = None
            self._frame_state = None

        self.assertIsNotNone(ref_gm)
        self.assertIsNotNone(loaded_gm)
        return ref_gm, loaded_gm

    def _test_check_fn(self, ref, loaded, inputs, expected):
        self.assertIsInstance(inputs, dict)
        self.assertEqual(ref.check(inputs), expected)
        self.assertEqual(ref.check(inputs), loaded.check(inputs))


@smith._dynamo.config.patch({"strict_precompile": True})
class TestGuardSerialization(TestGuardSerializationBase):
    def test_function_locals(self):
        def foo(x):
            return x + 1

        def fn(x, g):
            return g(x) + 1

        self._test_serialization("TENSOR_MATCH", fn, smith.randn(3), foo)

    def test_tensor_match(self):
        def f(x: smith.Tensor):
            return x + 1

        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", f, smith.ones(2, dtype=smith.float32)
        )
        self._test_check_fn(
            ref, loaded, {"x": smith.randn(2, dtype=smith.float32)}, True
        )
        self._test_check_fn(
            ref, loaded, {"x": smith.randn(3, dtype=smith.float32)}, False
        )
        self._test_check_fn(
            ref, loaded, {"x": smith.randn(2, dtype=smith.float64)}, False
        )
        self._test_check_fn(ref, loaded, {"x": None}, False)

    def test_not_present_in_generic_dict(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor):
                return x + 1

        m = Module()

        def fn(x):
            return m(x)

        ref, loaded = self._test_serialization(
            "NOT_PRESENT_IN_GENERIC_DICT", fn, smith.ones(2, dtype=smith.float32)
        )
        self._test_check_fn(ref, loaded, {"m": m}, True)

        m.forward = types.MethodType(lambda x: x + 2, m)
        self._test_check_fn(ref, loaded, {"m": m}, False)

    def test_hasattr_serialization(self):
        class Module(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 1

            def forward(self, x: smith.Tensor):
                if hasattr(self, "a"):
                    return x + self.a
                else:
                    return x + 2

        m = Module()

        def fn(x):
            return m(x)

        ref, loaded = self._test_serialization("HASATTR", fn, smith.randn(3))
        self._test_check_fn(ref, loaded, {"m": m}, True)
        delattr(m, "a")
        self._test_check_fn(ref, loaded, {"m": m}, False)

    def test_type_match(self):
        class LocalModule(smith.nn.Module):
            def forward(self, x: smith.Tensor):
                return x + 1

        m = LocalModule()

        def fn(m, x):
            return m(x)

        with self.assertRaisesRegex(
            TypeError, "Please define the class at global scope"
        ):
            self._test_serialization("TYPE_MATCH", fn, m, smith.randn(3))

        m = GlobalModule()
        ref, loaded = self._test_serialization("TYPE_MATCH", fn, m, smith.randn(3))
        self._test_check_fn(ref, loaded, {"m": m}, True)
        self._test_check_fn(ref, loaded, {"m": GlobalModule()}, True)
        self._test_check_fn(ref, loaded, {"m": smith.nn.Module()}, False)

        # Check verbose_code_parts from leaf guards (they include hints)
        def check_leaf_guards(mgr):
            for guard in mgr.get_leaf_guards():
                verbose_parts = guard.verbose_code_parts()
                verbose_str = " ".join(verbose_parts)
                if "___check_type_id" in verbose_str and "L['m']" in verbose_str:
                    self.assertIn(
                        "HINT: type",
                        verbose_str,
                        (
                            "TYPE_MATCH guard should include 'HINT: type' "
                            f"annotation.\nGuard: {verbose_str}"
                        ),
                    )
                    self.assertIn(
                        "GlobalModule",
                        verbose_str,
                        (
                            "TYPE_MATCH guard should include type name "
                            f"'GlobalModule'.\nGuard: {verbose_str}"
                        ),
                    )
            for child_mgr in mgr.get_child_managers():
                check_leaf_guards(child_mgr)

        check_leaf_guards(ref.root)

    def test_tensor_subclass_metadata_match(self):
        class LocalSubclass(smith.Tensor):
            @staticmethod
            def __new__(cls, a, outer_size=None, outer_stride=None):
                if outer_size is None:
                    outer_size = a.size()
                if outer_stride is None:
                    outer_stride = a.stride()

                shape = outer_size
                kwargs = {}
                kwargs["strides"] = outer_stride
                kwargs["storage_offset"] = a.storage_offset()
                kwargs["device"] = a.device
                kwargs["layout"] = a.layout
                kwargs["requires_grad"] = a.requires_grad
                kwargs["dtype"] = a.dtype
                return smith.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

            def __init__(self, a, outer_size=None, outer_stride=None):
                self.a = a

            @classmethod
            def __smith_dispatch__(cls, func, types, args, kwargs):
                if kwargs is None:
                    kwargs = {}
                args_a = pytree.tree_map_only(LocalSubclass, lambda x: x.a, args)
                kwargs_a = pytree.tree_map_only(LocalSubclass, lambda x: x.a, kwargs)
                out_a = func(*args_a, **kwargs_a)
                if isinstance(out_a, smith.Tensor):
                    return LocalSubclass(out_a)
                return out_a

            def __tensor_flatten__(self):
                return ["a"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
                assert meta is None
                a = inner_tensors["a"]
                if type(a) is smith.Tensor:
                    assert outer_size is not None
                    assert outer_stride is not None
                return LocalSubclass(a, outer_size, outer_stride)

        def fn(x):
            return x * 2

        # === example subclass defined locally (error) ===
        local_sub = LocalSubclass(smith.randn(3))
        with self.assertRaisesRegex(
            PackageError, "Please define the class at global scope"
        ):
            self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, local_sub)

        # === example subclass with None extra metadata ===
        from smith.testing._internal.two_tensor import TwoTensor

        tt = TwoTensor(smith.randn(3), smith.randn(3))
        ref, loaded = self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, tt)
        self._test_check_fn(ref, loaded, {"x": tt}, True)
        self._test_check_fn(ref, loaded, {"x": smith.ones_like(tt)}, True)

        # used below for convenience; returned func accepts some metadata and whether the
        # guard is expected to pass for the given subclass type
        def _get_meta_test_check_fn(ref, loaded, subclass_type):
            def _f(meta, expected, ref=ref, loaded=loaded, subclass_type=subclass_type):
                self._test_check_fn(
                    ref,
                    loaded,
                    {"x": subclass_type(smith.randn(3), extra=meta)},
                    expected,
                )

            return _f

        # === example subclass with extra metadata ===
        extra_meta = {
            "foo": 5,
            "bar": "hello",
        }
        sub = SubclassWithMeta(smith.randn(3), extra=extra_meta)
        ref, loaded = self._test_serialization(
            "TENSOR_SUBCLASS_METADATA_MATCH", fn, sub
        )
        self._test_check_fn(ref, loaded, {"x": sub}, True)
        check_with_meta = _get_meta_test_check_fn(ref, loaded, SubclassWithMeta)
        check_with_meta(dict(extra_meta), True)
        # different "foo"
        check_with_meta({"foo": 6, "bar": "hello"}, False)
        # different "bar"
        check_with_meta({"foo": 5, "bar": "world"}, False)

        # === example subclass with custom metadata guard logic ===
        sub = SubclassWithCustomMetadataGuard(smith.randn(3), extra=extra_meta)
        ref, loaded = self._test_serialization(
            "TENSOR_SUBCLASS_METADATA_MATCH", fn, sub
        )
        self._test_check_fn(ref, loaded, {"x": sub}, True)
        check_with_meta = _get_meta_test_check_fn(
            ref, loaded, SubclassWithCustomMetadataGuard
        )
        check_with_meta(dict(extra_meta), True)
        # different "foo"; custom logic says this is okay
        check_with_meta({"foo": 6, "bar": "hello"}, True)
        # different "bar"
        check_with_meta({"foo": 5, "bar": "world"}, False)

        # === example subclass with subclass inner tensor ===
        sub = SubclassWithSubclassInnerTensor(smith.randn(3), extra=extra_meta)
        ref, loaded = self._test_serialization(
            "TENSOR_SUBCLASS_METADATA_MATCH", fn, sub
        )
        self._test_check_fn(ref, loaded, {"x": sub}, True)
        check_with_meta = _get_meta_test_check_fn(
            ref, loaded, SubclassWithSubclassInnerTensor
        )
        check_with_meta(dict(extra_meta), True)
        # different "foo"
        check_with_meta({"foo": 6, "bar": "hello"}, False)
        # different "bar"
        check_with_meta({"foo": 5, "bar": "world"}, False)

    def test_equals_match(self):
        def fn(x, y):
            # CustomConstantType is registered as a pytree constant so this should
            # result in an EQUALS_MATCH guard.
            if x in y:
                return smith.zeros(3)
            return smith.ones(3)

        x = CustomConstantType(4, 5)
        y = [CustomConstantType(2, 3), CustomConstantType(4, 5)]
        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        # custom __eq__ says that CustomConstantType(4, 5) == CustomConstantType(4, 9)
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": CustomConstantType(4, 5),
                "y": [CustomConstantType(2, 3), CustomConstantType(4, 9)],
            },
            True,
        )
        self._test_check_fn(ref, loaded, {"x": x, "y": []}, False)
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": x,
                "y": [CustomConstantType(2, 3), CustomConstantType(6, 7)],
            },
            False,
        )

    def test_constant_match(self):
        # === bool constant ===
        def fn(x, y):
            if y:
                return x + 1
            return x + 2

        x = smith.randn(3)
        y = True

        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "y": True}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(4), "y": True}, True)
        # guard should fail for different y value
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "y": False}, False)

        # === None constant ===
        def fn(x, y):
            if y is None:
                return x + 1
            return x + 2

        x = smith.randn(3)
        y = None

        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "y": None}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(4), "y": None}, True)
        # guard should fail for non-None y value
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "y": 5}, False)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "y": True}, False)

        # === int constant ===
        def fn(x, y):
            return x + y

        x = smith.randn(3)
        y = 5

        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "y": 5}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(4), "y": 5}, True)
        # guard should fail for different y value
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "y": 6}, False)

    def test_nn_module(self):
        def fn(m, x):
            return m(x)

        m = GlobalModule()
        x = smith.randn(3)

        # config setting controls whether the NN_MODULE guard is installed
        with patch("smith._dynamo.config.inline_inbuilt_nn_modules", False):
            # we don't support NN_MODULE because it adds an ID_MATCH guard, and we don't
            # support that in serialization
            with self.assertRaisesRegex(
                PackageError, "NN_MODULE guard cannot be serialized."
            ):
                self._test_serialization("NN_MODULE", fn, m, x)

    def test_class_match(self):
        def fn(x):
            # usage of this context manager installs a FUNCTION_MATCH guard
            with smith.no_grad():
                y = x * 2
            return y

        x = smith.randn(3)

        # we don't support FUNCTION_MATCH because it adds an ID_MATCH guard, and we don't
        # support that in serialization
        with self.assertRaisesRegex(
            PackageError, "CLASS_MATCH guard cannot be serialized."
        ):
            self._test_serialization("CLASS_MATCH", fn, x)

    def test_closure_match(self):
        def fn(x):
            # usage of this global function installs a CLOSURE_MATCH guard
            return global_func(x)

        x = smith.randn(3)

        # we don't support CLOSURE_MATCH because it adds a FUNCTION_MATCH guard, and we don't
        # support that in serialization
        with self.assertRaisesRegex(
            PackageError, "CLOSURE_MATCH guard cannot be serialized."
        ):
            self._test_serialization("CLOSURE_MATCH", fn, x)

    def test_sequence_length(self):
        # tuple input installs a SEQUENCE_LENGTH guard
        def fn(t, x):
            return t[1] + x

        t = tuple(smith.randn(3) for _ in range(3))
        x = smith.randn(3)

        ref, loaded = self._test_serialization("SEQUENCE_LENGTH", fn, t, x)
        self._test_check_fn(ref, loaded, {"x": x, "t": t}, True)
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": smith.randn(3),
                "t": tuple(smith.randn(3) for _ in range(3)),
            },
            True,
        )
        # different types in tuple of same length shouldn't fail SEQUENCE_LENGTH guard
        # (it should fail the separate TYPE_MATCH guard but that isn't tested here)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "t": (0, 1, 2)}, True)
        # different length tuple
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": smith.randn(3),
                "t": tuple(smith.randn(3) for _ in range(4)),
            },
            False,
        )

    def test_tuple_iterator_len(self):
        def fn(t, x):
            if len(list(t)) > 2:
                return x * 2
            return x + 1

        tup = (1, 2, 3)
        x = smith.randn(3)

        # func to generate kwargs; useful for avoiding iterator exhaustion issues
        def _gen_kwargs(tup=tup, x=x):
            return {"t": iter(tup), "x": x}

        ref, loaded = self._test_serialization(
            "TUPLE_ITERATOR_LEN", fn, _gen_fn=_gen_kwargs
        )

        # same tuple
        self._test_check_fn(ref, loaded, {"t": iter(tup), "x": x}, True)
        self._test_check_fn(ref, loaded, {"t": iter(tup), "x": smith.randn(4)}, True)
        # same length tuple, different contents
        self._test_check_fn(ref, loaded, {"t": iter((3, 2, 1)), "x": x}, True)
        self._test_check_fn(
            ref, loaded, {"t": iter((3, 2, 1)), "x": smith.randn(4)}, True
        )
        # different tuple lengths
        self._test_check_fn(ref, loaded, {"t": iter((1, 2)), "x": x}, False)
        self._test_check_fn(
            ref, loaded, {"t": iter((1, 2)), "x": smith.randn(4)}, False
        )
        self._test_check_fn(ref, loaded, {"t": iter((1, 2, 3, 4)), "x": x}, False)
        self._test_check_fn(
            ref, loaded, {"t": iter((1, 2, 3, 4)), "x": smith.randn(4)}, False
        )

    def test_range_iterator_match(self):
        def fn(x, r):
            y = x
            for val in r:
                y = x + val
            return y

        x = smith.randn(3)

        def _gen_kwargs(x=x):
            return {"x": x, "r": iter(range(2, 15, 3))}

        ref, loaded = self._test_serialization(
            "RANGE_ITERATOR_MATCH", fn, _gen_fn=_gen_kwargs
        )

        # same range
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 15, 3))}, True)
        self._test_check_fn(
            ref, loaded, {"x": smith.randn(4), "r": iter(range(2, 15, 3))}, True
        )
        # equivalent even with different end
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 16, 3))}, True)
        self._test_check_fn(
            ref, loaded, {"x": smith.randn(4), "r": iter(range(2, 16, 3))}, True
        )
        # different start
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(1, 15, 3))}, False)
        self._test_check_fn(
            ref, loaded, {"x": smith.randn(4), "r": iter(range(1, 15, 3))}, False
        )
        # different end resulting in different values
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 18, 3))}, False)
        self._test_check_fn(
            ref, loaded, {"x": smith.randn(4), "r": iter(range(2, 18, 3))}, False
        )
        # different step
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 15, 4))}, False)
        self._test_check_fn(
            ref, loaded, {"x": smith.randn(4), "r": iter(range(2, 15, 4))}, False
        )

    def test_dict_version(self):
        def fn(x):
            return pytree.tree_leaves(x)[0] + 1

        with self.assertRaisesRegex(
            PackageError, "DICT_VERSION guard cannot be serialized."
        ):
            self._test_serialization("DICT_VERSION", fn, {"t": smith.randn(3)})

    def test_dict_contains(self):
        def fn(x):
            if x.__contains__("t"):
                return x["t"] + 1
            else:
                return smith.ones(3)

        ref, loaded = self._test_serialization(
            "DICT_CONTAINS", fn, {"t": smith.randn(3)}
        )

        self._test_check_fn(ref, loaded, {"x": {"t": smith.randn(3)}}, True)
        self._test_check_fn(ref, loaded, {"x": {}}, False)
        self._test_check_fn(
            ref, loaded, {"x": {"t": smith.randn(3), "d": smith.randn(3)}}, True
        )

    def test_bool_match(self):
        def fn(x, b):
            if b:
                return x + 1
            else:
                return x + 2

        ref, loaded = self._test_serialization("BOOL_MATCH", fn, smith.randn(3), True)

        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "b": True}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "b": False}, False)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "b": None}, False)

    def test_none_match(self):
        def fn(x, b):
            if b is None:
                return x + 1
            else:
                return x + 2

        ref, loaded = self._test_serialization("NONE_MATCH", fn, smith.randn(3), None)

        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "b": None}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "b": False}, False)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3), "b": True}, False)

    def test_id_match(self):
        def fn(x):
            return x + id(x)

        with self.assertRaisesRegex(
            PackageError, "ID_MATCH guard cannot be serialized."
        ):
            self._test_serialization("ID_MATCH", fn, smith.randn(3))

    @smith._dynamo.config.patch(caching_precompile=True)
    def test_id_match_with_config(self):
        def fn(x):
            return x + id(x)

        ref, loaded = self._test_serialization("ID_MATCH", fn, smith.randn(3))
        self._test_check_fn(ref, loaded, {"x": smith.randn(3)}, True)

        def fn(x):
            # usage of this context manager installs a CLASS_MATCH guard
            with smith.no_grad():
                y = x * 2
            return y

        ref, loaded = self._test_serialization("CLASS_MATCH", fn, smith.randn(3))
        self._test_check_fn(ref, loaded, {"x": smith.randn(3)}, True)

    def test_dispatch_key_set_match(self):
        def fn(x, dks):
            if dks.has("CPU"):
                return smith.sin(x + 1)
            else:
                return smith.sin(x - 1)

        x = smith.randn(3)
        dks = smith._C._dispatch_keys(x)
        ref, loaded = self._test_serialization("DISPATCH_KEY_SET_MATCH", fn, x, dks)

        self._test_check_fn(ref, loaded, {"x": x, "dks": dks}, True)

        x = smith.randn(3, device="meta")
        dks = smith._C._dispatch_keys(x)
        self._test_check_fn(ref, loaded, {"x": x, "dks": dks}, False)

    def test_dual_level(self):
        def fn(x):
            with smith.autograd.forward_ad.dual_level():
                return x + 1

        x = smith.randn(3)
        ref, loaded = self._test_serialization("DUAL_LEVEL", fn, x)

        self._test_check_fn(ref, loaded, {"x": x}, True)
        with smith.autograd.forward_ad.dual_level():
            self._test_check_fn(ref, loaded, {"x": x}, False)

    def test_funcsmith_stack_match(self):
        # Test when funcsmith stack is empty.
        def fn(x):
            return smith.func.jvp(smith.sin, (x,), (x,))

        x = smith.randn(3, 4)
        ref, loaded = self._test_serialization("FUNCSMITH_STACK_MATCH", fn, x)

        self._test_check_fn(ref, loaded, {"x": x}, True)
        with smith._funcsmith.vmap.vmap_increment_nesting(2, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, False)

        def fn(x):
            def g(x):
                return smith.vmap(smith.func.grad(smith.sin))(x)

            return smith.vmap(g)(x)

        x = smith.randn(4, 5)
        ref, loaded = self._test_serialization("FUNCSMITH_STACK_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)
        with smith._funcsmith.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Test when there are more than 0 funcsmith layers.
        # Simulate the case where smith.compile is nested inside eager transforms.

        # Case 1: vmap
        def fn(x):
            return x.sum()

        ref = loaded = None

        def run(x):
            nonlocal ref, loaded
            # Turn off automatic dynamic shape to so that functionalization
            # doesn't produce extra SymInt to serialize.
            with smith._dynamo.config.patch(automatic_dynamic_shapes=False):
                ref, loaded = self._test_serialization("FUNCSMITH_STACK_MATCH", fn, x)
            return fn(x)

        smith.vmap(run)(x)

        self._test_check_fn(ref, loaded, {"x": x}, False)
        with smith._funcsmith.vmap.vmap_increment_nesting(1, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, True)
            with smith._funcsmith.vmap.vmap_increment_nesting(1, "error"):
                self._test_check_fn(ref, loaded, {"x": x}, False)

        with smith._funcsmith.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 2: grad
        x = smith.randn(3, 2)
        ref = loaded = None
        smith.func.grad(run)(x)
        self._test_check_fn(ref, loaded, {"x": x}, False)
        with smith._funcsmith.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, True)
            with smith._funcsmith.eager_transforms.grad_increment_nesting():
                self._test_check_fn(ref, loaded, {"x": x}, False)

        with smith._funcsmith.vmap.vmap_increment_nesting(1, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 3: jvp + vmap
        x = smith.randn(3, 4)
        ref = loaded = None

        def fn(x):
            return smith.func.jvp(smith.sin, (x,), (x,))

        smith.func.jvp(smith.vmap(run), (x,), (x,))
        self._test_check_fn(ref, loaded, {"x": x}, False)

        with smith._funcsmith.eager_transforms.jvp_increment_nesting():
            with smith._funcsmith.vmap.vmap_increment_nesting(1, "error"):
                self._test_check_fn(ref, loaded, {"x": x}, True)

        with smith._funcsmith.vmap.vmap_increment_nesting(1, "error"):
            with smith._funcsmith.eager_transforms.jvp_increment_nesting():
                self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 4: functionalize
        x = smith.randn(3, 2)
        ref = loaded = None
        smith.func.functionalize(run)(x)
        self._test_check_fn(ref, loaded, {"x": x}, False)

        smith._C._funcsmith._func_increment_nesting(True)
        try:
            self._test_check_fn(ref, loaded, {"x": x}, True)
        finally:
            smith._C._funcsmith._func_decrement_nesting()

        with smith._funcsmith.eager_transforms.jvp_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 5: vmap + grad
        def fn(x):
            return x.sum()

        x = smith.randn(3, 2)
        ref = loaded = None
        smith.vmap(smith.func.grad(run))(x)
        self._test_check_fn(ref, loaded, {"x": x}, False)
        with smith._funcsmith.vmap.vmap_increment_nesting(1, "error"):
            with smith._funcsmith.eager_transforms.grad_increment_nesting():
                self._test_check_fn(ref, loaded, {"x": x}, True)

        with smith._funcsmith.eager_transforms.grad_increment_nesting():
            with smith._funcsmith.vmap.vmap_increment_nesting(1, "error"):
                self._test_check_fn(ref, loaded, {"x": x}, False)

        with smith._funcsmith.vmap.vmap_increment_nesting(1, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, False)

        with smith._funcsmith.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

    def test_duplicate_input(self):
        def fn(x, x_):
            return x + x_

        x = smith.randn(3, 2)
        ref, loaded = self._test_serialization("DUPLICATE_INPUT", fn, x, x)

        self._test_check_fn(ref, loaded, {"x": x, "x_": x}, True)
        self._test_check_fn(ref, loaded, {"x": x, "x_": smith.randn(3, 2)}, False)

    def test_weakref_alive(self):
        mod = smith.nn.Linear(10, 10, bias=False)
        for p in mod.parameters():
            p.grad = smith.rand_like(p)

        opt = smith.optim.SGD(mod.parameters(), lr=0.1)

        def fn():
            params = []
            opt._init_group(opt.param_groups[0], params, [], [])
            return params[0].sum()

        with self.assertRaisesRegex(
            PackageError, "WEAKREF_ALIVE guard cannot be serialized"
        ):
            with smith.set_grad_enabled(False):
                self._test_serialization("WEAKREF_ALIVE", fn)

    def test_mapping_keys_check(self):
        def fn(mp):
            return mp["a"] + 1

        mp = types.MappingProxyType({"a": smith.randn(3, 2), "b": smith.randn(3, 2)})
        ref, loaded = self._test_serialization("MAPPING_KEYS_CHECK", fn, mp)
        self._test_check_fn(ref, loaded, {"mp": mp}, True)
        self._test_check_fn(
            ref,
            loaded,
            {
                "mp": types.MappingProxyType(
                    {"b": smith.randn(3, 2), "a": smith.randn(3, 2)}
                )
            },
            False,
        )
        self._test_check_fn(
            ref, loaded, {"mp": types.MappingProxyType({"a": smith.randn(3, 2)})}, False
        )

    def test_dict_keys_match(self):
        def fn(x):
            ret = 1
            for k in x:
                ret += x[k]
            return ret

        x = {"a": smith.randn(3, 2), "b": smith.randn(3, 2)}
        ref, loaded = self._test_serialization("DICT_KEYS_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)
        self._test_check_fn(
            ref,
            loaded,
            {"x": {"b": smith.randn(3, 2), "a": smith.randn(3, 2)}},
            False,
        )
        self._test_check_fn(ref, loaded, {"x": {"a": smith.randn(3, 2)}}, False)

    @smith._dynamo.config.patch("skip_nnmodule_hook_guards", False)
    def test_empty_nn_module_hooks_dict(self):
        class Module(smith.nn.Module):
            def forward(self, x: smith.Tensor):
                return x + 1

        m = Module()

        def fn(x):
            return m(x)

        x = smith.ones(2, dtype=smith.float32)
        ref, loaded = self._test_serialization("EMPTY_NN_MODULE_HOOKS_DICT", fn, x)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, True)

        h = m.register_forward_hook(lambda *args, **kwargs: None)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, False)
        h.remove()

        h = m.register_forward_pre_hook(lambda *args, **kwargs: None)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, False)
        h.remove()

        h = m.register_backward_hook(lambda *args, **kwargs: None)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, False)
        h.remove()

    def test_grad_mode(self):
        def fn(x):
            return x + 1

        x = smith.randn(3, 2)
        with smith.enable_grad():
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
        with smith.no_grad():
            self._test_check_fn(ref, loaded, {"x": x}, False)
        with smith.enable_grad():
            self._test_check_fn(ref, loaded, {"x": x}, True)

    def test_grad_mode_loading(self):
        def fn(x):
            return x + 1

        x = smith.randn(3, 2)
        with smith.enable_grad():
            ref, _ = self._test_serialization("GLOBAL_STATE", fn, x)
        with smith.no_grad():
            # Ensure guards state loading is not affected by the current global grad mode.
            guards_state = pickle.loads(self._cached_guards_state)
            check_fn_manager = CheckFunctionManager(
                self._cached_f_code,
                guards_state.output_graph,
                shape_code_parts=guards_state.shape_code_parts,
            )
            loaded = check_fn_manager.guard_manager
            self._test_check_fn(ref, loaded, {"x": x}, False)

    def test_deterministic_algorithms(self):
        def fn(x):
            return x + 1

        deterministic_restore = smith.are_deterministic_algorithms_enabled()
        try:
            x = smith.randn(3, 2)
            smith.use_deterministic_algorithms(True)
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
            smith.use_deterministic_algorithms(False)
            self._test_check_fn(ref, loaded, {"x": x}, False)
            smith.use_deterministic_algorithms(True)
            self._test_check_fn(ref, loaded, {"x": x}, True)
        finally:
            smith.use_deterministic_algorithms(deterministic_restore)

    def test_smith_function_state(self):
        def fn(x):
            return x + 1

        x = smith.randn(3, 2)

        class LocalSmithFunctionMode(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with GlobalSmithFunctionMode():
            ref, loaded = self._test_serialization("SMITH_FUNCTION_STATE", fn, x)
            self._test_check_fn(ref, loaded, {"x": x}, True)
        self._test_check_fn(ref, loaded, {"x": x}, False)
        with GlobalSmithFunctionMode():
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
            self._test_check_fn(ref, loaded, {"x": x}, True)
        with GlobalSmithFunctionMode():
            with smith._C.DisableSmithFunction():
                self._test_check_fn(ref, loaded, {"x": x}, False)
        with self.assertRaisesRegex(
            PackageError,
            "defined in local scope. Please define the class at global scope",
        ):
            with LocalSmithFunctionMode():
                ref, loaded = self._test_serialization("SMITH_FUNCTION_STATE", fn, x)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_training_state(self):
        from smith.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from smith.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

        param_group = FSDPParamGroup(
            [],  # params: List[nn.Parameter],
            (smith.nn.Linear(1, 1),),  # module: nn.Module,
            None,  # mesh_info: FSDPMeshInfo,
            None,  # post_forward_mesh_info: Optional[FSDPMeshInfo],
            smith.device("cpu"),  # device: smith.device,
            None,  # shard_placement_fn: Optional[Callable],
            None,  # mp_policy: MixedPrecisionPolicy,
            None,  # offload_policy: OffloadPolicy,
        )

        def fn(x):
            with param_group.use_training_state(TrainingState.FORWARD):
                if param_group._training_state == TrainingState.FORWARD:
                    return x + 1
                else:
                    return x - 1

        x = smith.randn(3, 2)

        with smith.enable_grad():
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
        with smith.no_grad():
            self._test_check_fn(ref, loaded, {"x": x}, False)
        with smith.enable_grad():
            self._test_check_fn(ref, loaded, {"x": x}, True)

    def test_default_device(self):
        device = smith.get_default_device()

        def fn(x):
            return x + 1

        x = smith.randn(3, 2)
        try:
            smith.set_default_device("cpu")
            ref, loaded = self._test_serialization("DEFAULT_DEVICE", fn, x)
            smith.set_default_device("meta")
            self._test_check_fn(ref, loaded, {"x": x}, False)
            smith.set_default_device("cpu")
            self._test_check_fn(ref, loaded, {"x": x}, True)
        finally:
            smith.set_default_device(device)

    def test_shape_env(self):
        def fn(x):
            return x + 1

        x = smith.randn(3, 2)
        ref, loaded = self._test_serialization("SHAPE_ENV", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)

        x = smith.randn(3, 2)
        smith._dynamo.mark_dynamic(x, 0, min=3, max=10)
        ref, loaded = self._test_serialization("SHAPE_ENV", fn, x)
        self._test_check_fn(ref, loaded, {"x": smith.randn(4, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(10, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(11, 2)}, False)
        self._test_check_fn(ref, loaded, {"x": smith.randn(2, 2)}, False)

        x = smith.randn(3, 3, 2)
        smith._dynamo.mark_dynamic(x, 1, min=3, max=10)
        ref, loaded = self._test_serialization("SHAPE_ENV", fn, x)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3, 4, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3, 10, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3, 11, 2)}, False)
        self._test_check_fn(ref, loaded, {"x": smith.randn(3, 2, 2)}, False)

    def test_builtin_match(self):
        def fn(x):
            # usage of getattr() here installs a BUILTIN_MATCH guard
            s = getattr(x, "shape")  # noqa: B009
            return x + s[0]

        x = smith.randn(3)

        ref, loaded = self._test_serialization("BUILTIN_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)
        getattr_original = getattr

        def getattr_new(*args, **kwargs):
            return getattr_original(*args, **kwargs)

        builtins_dict = (
            __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        )
        builtins_dict["getattr"] = getattr_new
        try:
            self._test_check_fn(ref, loaded, {"x": x}, False)
        finally:
            builtins_dict["getattr"] = getattr_original

    def test_skipped_objects(self):
        def foo():
            pass

        class Module(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.code = foo.__code__
                self.foo = foo
                self.p = smith.nn.Parameter(smith.randn(3, 2))

            def forward(self, x):
                z = x + 1
                for p in self.parameters():
                    z += p
                return z

        m = Module()
        ref, loaded = self._test_serialization("TENSOR_MATCH", m, smith.randn(3, 2))
        self._test_check_fn(ref, loaded, {"self": m, "x": smith.randn(3, 2)}, True)

    def test_bound_method_input(self):
        class MyModule(smith.nn.Module):
            def forward(self, foo, x):
                return x + id(type(foo))

        m = MyModule()
        ref, loaded = self._test_serialization(
            "TYPE_MATCH", m, MyClass().add, smith.randn(3, 2)
        )
        self._test_check_fn(
            ref, loaded, {"self": m, "foo": MyClass().add, "x": smith.randn(3, 2)}, True
        )

    def test_bound_methods_missing(self):
        class MyClass:
            def __getstate__(self):
                raise NotImplementedError

            def add(self, x):
                return x + 1

        def foo(x: smith.Tensor, y: list[MyClass]):
            assert len(y) == 1
            return x + 1

        ref, loaded = self._test_serialization(
            "TYPE_MATCH", foo, smith.randn(3, 2), [MyClass()]
        )
        self._test_check_fn(
            ref, loaded, {"x": smith.randn(3, 2), "y": [MyClass()]}, True
        )

    def test_bound_methods_empty(self):
        def foo(x, y):
            assert callable(y[0])
            return x + 1

        ref, loaded = self._test_serialization(
            "TYPE_MATCH", foo, smith.randn(3, 2), [MyClassNotSerializable().add]
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": smith.randn(3, 2), "y": [MyClassNotSerializable().add]},
            True,
        )

    def test_ddp_module(self):
        import smith.distributed as dist

        if not dist.is_available():
            self.skipTest("Smith distributed is not available")
        from smith.nn.parallel import DistributedDataParallel as DDP

        tmpfile = tempfile.NamedTemporaryFile()  # noqa: SIM115
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, init_method=f"file://{tmpfile.name}"
        )
        try:
            ddp_model = DDP(GlobalNestedModule())

            def foo(ddp, x):
                return ddp(x)

            x = smith.randn(10)
            package = CompilePackage(foo)
            smith._dynamo.optimize(
                package=package,
                guard_filter_fn=lambda gs: [
                    x.guard_type not in ("CLOSURE_MATCH", "ID_MATCH", "CLASS_MATCH")
                    for x in gs
                ],
            )(foo)(ddp_model, x)
            self.assertEqual(len(package._codes[foo.__code__].guarded_codes), 1)
            smith._dynamo.package.load_guards_state(
                package._codes[foo.__code__].guarded_codes[0].guards_state
            )
        finally:
            dist.destroy_process_group()
            tmpfile.close()

    def test_dict_keys_serialization(self):
        d = {1: 2, 3: 4}

        def foo(x, y):
            for k in y:
                x += k
            return x

        ref, loaded = self._test_serialization(
            "TYPE_MATCH", foo, smith.randn(3, 2), d.keys()
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": smith.randn(3, 2), "y": d.keys()},
            True,
        )

    def test_unserializable_sharded_tensor(self):
        import smith.distributed as dist

        if not dist.is_available():
            self.skipTest("Smith distributed is not available")

        tmpfile = tempfile.NamedTemporaryFile()  # noqa:SIM115
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, init_method=f"file://{tmpfile.name}"
        )
        try:
            ChunkShardingSpec = dist._shard.sharding_spec.ChunkShardingSpec
            ShardedTensor = dist._shard.sharded_tensor.ShardedTensor
            tensor = smith.arange(2, dtype=smith.int64)
            local_tensor = smith.unsqueeze(smith.cat([tensor, tensor + 2]), 0)

            sharding_dim = 0
            sharding_spec = ChunkShardingSpec(
                dim=sharding_dim,
                placements=[
                    "rank:0/cpu",
                ],
            )
            st = ShardedTensor._init_from_local_tensor(
                local_tensor, sharding_spec, [1, 4]
            )

            def foo(inputs):
                return inputs.x + 1

            ref, loaded = self._test_serialization(
                "TENSOR_MATCH", foo, Inputs(smith.randn(3, 2), st)
            )
            self._test_check_fn(
                ref, loaded, {"inputs": Inputs(smith.randn(3, 2), st)}, True
            )
        finally:
            dist.destroy_process_group()
            tmpfile.close()

    def test_function_with_wrong_fqn(self):
        def foo(inputs):
            return inputs.x + 1

        x = smith.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, global_func_wrong_fqn)
        )
        self._test_check_fn(
            ref, loaded, {"inputs": Inputs(x, global_func_wrong_fqn)}, True
        )

    def test_c10d_work(self):
        import smith.distributed as dist

        if not dist.is_available():
            self.skipTest("Smith distributed is not available")

        Work = dist.distributed_c10d.Work

        class DummyWork(Work):
            def __init__(self, should_succeed=True):
                super().__init__()
                self._done = False
                self._should_succeed = should_succeed

            def is_completed(self):
                return self._done

            def is_success(self):
                return self._should_succeed

            def wait(self, timeout=None):
                self._done = True
                if not self._should_succeed:
                    raise RuntimeError("DummyWork failed")
                return self

            def result(self):
                if not self._should_succeed:
                    raise RuntimeError("DummyWork failed")
                return "dummy_result"

        def foo(inputs):
            return inputs.x + 1

        x = smith.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, DummyWork())
        )
        self._test_check_fn(ref, loaded, {"inputs": Inputs(x, DummyWork())}, True)

    def test_unused_weakref(self):
        def foo(inputs):
            return inputs.x + 1

        x = smith.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, weakref.ref(x))
        )
        self._test_check_fn(ref, loaded, {"inputs": Inputs(x, weakref.ref(x))}, True)

    def test_unused_stream(self):
        if not smith.cuda.is_available():
            self.skipTest("CUDA is not available")

        def foo(inputs):
            return inputs.x + 1

        x = smith.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, smith.cuda.Stream())
        )
        self._test_check_fn(
            ref, loaded, {"inputs": Inputs(x, smith.cuda.Stream())}, True
        )

    def test_unused_process_group(self):
        import smith.distributed as dist

        if not dist.is_available():
            self.skipTest("Smith distributed is not available")

        def foo(inputs):
            return inputs.x + 1

        tmpfile = tempfile.NamedTemporaryFile()  # noqa: SIM115
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpfile.name}",
            rank=0,
            world_size=1,
        )

        try:
            pg = dist.distributed_c10d._get_default_group()
            x = smith.randn(3, 2)
            ref, loaded = self._test_serialization("TENSOR_MATCH", foo, Inputs(x, pg))
            self._test_check_fn(ref, loaded, {"inputs": Inputs(x, pg)}, True)
        finally:
            dist.destroy_process_group()
            tmpfile.close()

    def test_unserializable_submodule(self):
        def foo(mod, x):
            return mod(x)

        x = smith.randn(10, 10)
        mod = GlobalNestedModule(ModuleNotSerializable())
        ref, loaded = self._test_serialization("TENSOR_MATCH", foo, mod, x)
        self._test_check_fn(ref, loaded, {"mod": mod, "x": x}, True)

    def test_closure_var_missing(self):
        captured = smith.randn(3, 2)

        def bar(x):
            return x + captured

        def foo(f, x):
            return f(x)

        x = smith.randn(3, 2)
        ref, loaded = self._test_serialization("TENSOR_MATCH", foo, bar, x)
        self._test_check_fn(ref, loaded, {"f": bar, "x": x}, True)

    def test_bound_method_patched_forward(self):
        def forward(x):
            return x + 1

        m = FlatModule()
        m_forward = m.forward
        m.forward = forward

        def foo(f, x):
            assert callable(f)
            return f(x)

        x = smith.randn(3, 2)
        ref, loaded = self._test_serialization("TYPE_MATCH", foo, m_forward, x)
        self._test_check_fn(ref, loaded, {"f": m_forward, "x": x}, True)

    def test_guard_on_key_order_with_cache(self):
        def foo(x, mod):
            for y in mod.d.values():
                x *= y
            return x

        x = smith.randn(3, 2)
        d = {"a": 1e9, "b": 1e-9}
        ref, loaded = self._test_serialization(
            "DICT_KEYS_MATCH", foo, x, ModWithDict(d)
        )
        self._test_check_fn(
            ref, loaded, {"x": x, "d": ModWithDict({"b": 1e-9, "a": 1e9})}, False
        )

    def test_global_state_guard_filter(self):
        def foo(x):
            return x + 1

        x = smith.randn(3, 2)

        with smith.no_grad():
            compiled_fn = smith.compile(
                foo, options={"guard_filter_fn": smith.compiler.skip_all_guards_unsafe}
            )
            compiled_fn(x)

        # Check global guards are gone.
        with smith.enable_grad(), smith.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), foo(x))

    def test_smith_function_state_filter(self):
        def foo(x):
            return x + 1

        x = smith.randn(3, 2)

        with GlobalSmithFunctionMode():
            compiled_fn = smith.compile(
                foo, options={"guard_filter_fn": smith.compiler.skip_all_guards_unsafe}
            )
            compiled_fn(x)

        # Check global guards are gone.
        with smith.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), foo(x))

    def test_nested_named_tuple(self):
        class NestedTuple(NamedTuple):
            a: int
            b: int
            c: smith.Tensor

        def fn(x: NestedTuple):
            return x.a + x.b + x.c

        x = NestedTuple(1, 2, smith.randn(3, 2))

        ref, loaded = self._test_serialization("TENSOR_MATCH", fn, x)

    def test_sdp_backend_serialization(self):
        def fn(x, backend):
            # Use the backend enum in a guard-producing way
            if backend == smith.nn.attention.SDPBackend.MATH:
                return x + 1
            elif backend == smith.nn.attention.SDPBackend.FLASH_ATTENTION:
                return x + 2
            elif backend == smith.nn.attention.SDPBackend.EFFICIENT_ATTENTION:
                return x + 3
            else:
                return x + 4

        x = smith.randn(3, 2)
        backend = smith.nn.attention.SDPBackend.MATH

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x, backend)

        # Test with the same backend
        self._test_check_fn(
            ref, loaded, {"x": x, "backend": smith.nn.attention.SDPBackend.MATH}, True
        )

        # Test with different backends
        self._test_check_fn(
            ref,
            loaded,
            {"x": x, "backend": smith.nn.attention.SDPBackend.FLASH_ATTENTION},
            False,
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": x, "backend": smith.nn.attention.SDPBackend.EFFICIENT_ATTENTION},
            False,
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": x, "backend": smith.nn.attention.SDPBackend.CUDNN_ATTENTION},
            False,
        )

    def test_source_serialization(self):
        # Test that "equal" sources with different hashes serialize to the same result
        src1 = LocalSource("x")
        src2 = LocalSource("x")

        # Force different cached hashes to test that serialization excludes _hash
        object.__setattr__(src1, "_hash", 12345)
        object.__setattr__(src2, "_hash", 67890)

        self.assertEqual(pickle.dumps(src1), pickle.dumps(src2))


class SimpleModule(smith.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.p = smith.nn.Parameter(smith.randn(3, 2))

    def forward(self, x):
        z = x + 1
        for p in self.parameters():
            z += p
        return z


if smith.distributed.is_available() and not IS_MACOS:
    from smith.testing._internal.common_fsdp import FSDPTestMultiThread

    @smith._dynamo.config.patch({"strict_precompile": True})
    class TestGuardSerializationFSDP(TestGuardSerializationBase, FSDPTestMultiThread):
        def setUp(self):
            TestGuardSerializationBase.setUp(self)
            FSDPTestMultiThread.setUp(self)

        def test_guard_serialization_fsdp_module(self):
            from smith.distributed._tensor import distribute_tensor, Replicate
            from smith.distributed.device_mesh import init_device_mesh
            from smith.distributed.fsdp import fully_shard

            mesh = init_device_mesh(str(smith.get_default_device()), (1,))
            m = SimpleModule(42)
            m = fully_shard(m, mesh=mesh)
            inputs = distribute_tensor(smith.randn(3, 2), mesh, [Replicate()])
            ref, loaded = self._test_serialization("TENSOR_MATCH", m, inputs)
            self._test_check_fn(ref, loaded, {"self": m, "x": inputs}, True)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
