# mypy: allow-untyped-defs
import operator
import traceback
import typing
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional, Union

import smith
from smith import fx
from smith._dispatch.python import enable_python_dispatcher
from smith._export.pass_infra.node_metadata import NodeMetadata
from smith._export.pass_infra.proxy_value import ProxyValue
from smith._higher_order_ops.map import _unstack_pytree
from smith._subclasses import FakeTensor, UnsupportedFakeTensorException
from smith._subclasses.fake_tensor import FakeTensorMode
from smith.fx import traceback as fx_traceback
from smith.fx.experimental.proxy_tensor import PythonKeyTracer
from smith.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    PropagateUnbackedSymInts,
)
from smith.fx.graph import CodeGen
from smith.fx.passes.infra.pass_base import PassBase, PassResult
from smith.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from smith.utils import _pytree as pytree


__all__ = ["_ExportPassBaseDeprecatedDoNotUse"]


Argument = Any
Value = Any
Fn = Callable[..., Any]
PassType = Callable[[smith.fx.GraphModule], Optional[PassResult]]


_SMITH_SYM_OPS: set[Callable] = {
    smith.sym_int,
    smith.sym_float,
    smith.sym_ite,
    smith.sym_max,
    smith.sym_min,
    smith.sym_not,
    smith.sym_sqrt,
}


class ExportPassBaseError(RuntimeError):
    pass


class _ExportPassBaseDeprecatedDoNotUse(PassBase):
    """
    Interpreter-based pass class to help users maintain the IR spec while writing
    transformations.
    """

    @staticmethod
    def _create_dummy_node_metadata():
        return NodeMetadata({"stack_trace": "".join(traceback.format_stack(limit=1))})

    class ExportTracer(PythonKeyTracer):
        def __init__(
            self, callback: "_ExportPassBaseDeprecatedDoNotUse", codegen: CodeGen
        ) -> None:
            super().__init__()
            self.callback = callback
            self.root = smith.nn.Module()
            self.graph = smith.fx.Graph()
            self.graph.set_codegen(codegen)
            self.tensor_attrs: dict[str, smith.Tensor] = {}  # type: ignore[assignment]
            self.fake_tensor_mode: Optional[FakeTensorMode] = None
            self.submodules: dict[smith.nn.Module, str] = {}

        def trace(self) -> None:  # type: ignore[override]
            raise ExportPassBaseError("ExportTracer doesn't support trace().")

        def create_arg(self, a: Argument) -> smith.fx.Node:
            if isinstance(a, smith.nn.Module):
                if a not in self.submodules:
                    name_submodule = f"submodule_{len(self.submodules)}"
                    self.root.add_module(name_submodule, a)
                    self.submodules[a] = name_submodule
            elif isinstance(a, FakeTensor):
                if not hasattr(a, "constant") or a.constant is None:
                    raise ExportPassBaseError(f"Cannot add {a} to graph.")
                a = a.constant
            node = super().create_arg(a)
            if (
                isinstance(a, smith.Tensor)
                and isinstance(node, smith.fx.Node)
                and node.op == "get_attr"
            ):
                self.set_metadata(node, a)
                self.callback.on_attr(ProxyValue(a, node))
            return node

        def set_metadata(
            self,
            node: smith.fx.Node,
            value: Argument,
        ) -> None:
            # propagate the fake tensor or sym nodes
            def make_val(
                x: Argument,
            ) -> Union[
                FakeTensor,
                smith.SymInt,
                smith.SymFloat,
                smith.SymBool,
                int,
                float,
                bool,
                str,
                None,
            ]:
                if isinstance(x, FakeTensor):
                    return x
                elif isinstance(x, smith.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = smith.dequantize(x)

                    try:
                        if self.fake_tensor_mode is None:
                            raise AssertionError("fake_tensor_mode must not be None")
                        # TODO we should allocate static shapes
                        # for param/buffer values
                        if isinstance(x, smith.nn.Parameter):
                            fake_tensor = self.fake_tensor_mode.from_tensor(
                                x, static_shapes=True
                            )
                        else:
                            fake_tensor = self.fake_tensor_mode.from_tensor(x)
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        print(
                            "Fakeifying a Tensor subclass is not supported \
                            right now. Instead a TensorMetadata is used."
                        )
                        fake_tensor = None
                    return fake_tensor
                elif isinstance(
                    x,
                    (
                        smith.SymInt,
                        smith.SymFloat,
                        smith.SymBool,
                        int,
                        float,
                        bool,
                        str,
                    ),
                ):
                    return x
                else:
                    return None

            node.meta["val"] = pytree.tree_map(make_val, value)

            # Set the tensor_metadata for values that do not have a corresponding FakeTensor
            def make_tensor_meta(x: Argument) -> Optional[TensorMetadata]:
                if not isinstance(x, FakeTensor) and isinstance(x, smith.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = smith.dequantize(x)

                    try:
                        if self.fake_tensor_mode is None:
                            raise AssertionError("fake_tensor_mode must not be None")
                        _ = self.fake_tensor_mode.from_tensor(x)
                        tensor_meta = None
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        tensor_meta = _extract_tensor_metadata(x)
                    return tensor_meta
                else:
                    return None

            node.meta["tensor_meta"] = pytree.tree_map(make_tensor_meta, value)

    class ExportInterpreter(fx.Interpreter):
        def __init__(
            self, callback: "_ExportPassBaseDeprecatedDoNotUse", gm: fx.GraphModule
        ) -> None:
            super().__init__(gm)
            self.callback = callback
            self.node: smith.fx.Node = next(iter(gm.graph.nodes))

        # pyrefly: ignore [bad-override]
        def placeholder(
            self,
            target: str,  # type: ignore[override]
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            arg = super().placeholder(target, args, kwargs)
            return self.callback.placeholder(target, arg, NodeMetadata(self.node.meta))

        def output(
            self,
            target: smith.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            return self.callback.output(args[0], NodeMetadata(self.node.meta)).data  # type: ignore[return-value]

        def call_function(
            self,
            target: smith.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            meta = NodeMetadata(self.node.meta)

            if target is operator.getitem:
                value, key = args
                return self.callback.call_getitem(value, key, meta)
            elif getattr(target, "__module__", None) in {
                "_operator",
                "builtins",
                "math",
            }:
                if not callable(target):
                    raise AssertionError(f"expected callable target, got {target}")
                return self.callback.call_sym(target, args, meta)
            elif target in _SMITH_SYM_OPS:
                if not callable(target):
                    raise AssertionError(f"expected callable target, got {target}")
                return self.callback.call_sym(target, args, meta)
            elif isinstance(
                target, (smith._ops.OpOverload, smith._ops.OpOverloadPacket)
            ):
                return self.callback.call_operator(
                    target,
                    args,
                    kwargs,
                    meta,
                )
            elif target is smith.ops.higher_order.cond:
                pred, true_fn, false_fn, inputs = args
                return self.callback.call_cond(pred, true_fn, false_fn, inputs, meta)
            elif target is smith.ops.higher_order.map_impl:
                f, mapped_args, operands = args  # type: ignore[assignment]
                return self.callback.call_map(f, mapped_args, operands, meta)
            # For other unregistered HigherOrderOps, just interpret them blindly
            elif isinstance(target, smith._ops.HigherOrderOperator):
                return self.callback._fx(
                    "call_function",
                    target,
                    args,
                    kwargs,
                    meta,
                )
            else:
                raise ExportPassBaseError(f"Unsupported target type: {target}")

        def get_attr(  # type: ignore[override]
            self,
            target: str,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> Argument:
            return super().get_attr(target, args, kwargs)

        def call_module(
            self,
            target: smith.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> None:
            raise ExportPassBaseError("call_module is not supported.")

        def call_method(  # type: ignore[override]
            self,
            target: str,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> None:
            raise ExportPassBaseError("call_method is not supported.")

        def run_node(self, n: smith.fx.Node) -> Argument:
            self.node = n
            self.callback.node_debug_str = n.format_node()
            return super().run_node(n)

    def __init__(self) -> None:
        self.interpreter = PropagateUnbackedSymInts(
            smith.fx.GraphModule(smith.nn.Module(), smith.fx.Graph())
        )
        self.tracer = self.ExportTracer(self, CodeGen())
        self.fake_tensor_mode: Optional[FakeTensorMode] = None
        self._initialized = True
        self.node_debug_str: typing.Optional[str] = None

    def _fx(
        self,
        kind: str,
        target: smith.fx.node.Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        res_data = getattr(self.interpreter, kind)(target, args_data, kwargs_data)
        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue, lambda x: x.proxy, (args, kwargs)
        )

        name = None
        if isinstance(target, smith._ops.OpOverload):
            name = self.tracer.graph._target_to_str(target.overloadpacket.__name__)

        res_proxy = self.tracer.create_proxy(
            kind, target, args_proxy, kwargs_proxy, name=name
        )
        res_proxy.node.meta.update(meta.data)
        if self.fake_tensor_mode and (shape_env := self.fake_tensor_mode.shape_env):
            if symbol_to_path := compute_unbacked_bindings(shape_env, res_data):
                res_proxy.node.meta["unbacked_bindings"] = symbol_to_path
        self.tracer.set_metadata(res_proxy.node, res_data)
        return ProxyValue(res_data, res_proxy)

    def inputs(self, graph_module: smith.fx.GraphModule) -> list[Argument]:
        # TODO(angelayi): Update this with what we decide to do for metadata in
        # the exported graph module
        if (args := graph_module.meta.get("args", None)) is not None:
            return list(args)

        def extract_input(node: smith.fx.Node) -> Optional[FakeTensor]:
            if "val" in node.meta:
                fake = node.meta["val"]
                if hasattr(fake, "constant") and fake.constant is not None:
                    return fake.constant
                return fake
            elif tensor_meta := node.meta.get("tensor_meta"):
                if self.fake_tensor_mode is None:
                    raise AssertionError("fake_tensor_mode must not be None")
                return FakeTensor(
                    self.fake_tensor_mode,
                    smith.empty(
                        tensor_meta.shape,
                        dtype=tensor_meta.dtype,
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                        memory_format=tensor_meta.memory_format,
                    ),
                    smith.device("cpu"),
                )
            elif len(node.users) == 0:
                return None
            raise ExportPassBaseError(
                f"Cannot construct an input for graph module: {graph_module}.",
            )

        return [
            extract_input(node)
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
        ]

    def on_attr(self, attr: ProxyValue) -> None:
        pass

    def placeholder(self, name: str, arg: Argument, meta: NodeMetadata) -> ProxyValue:
        arg_proxy = self.tracer.create_proxy("placeholder", name, (), {})
        arg_proxy.node.meta = meta.data
        self.tracer.set_metadata(arg_proxy.node, arg)
        return ProxyValue(arg, arg_proxy)

    def call_operator(
        self,
        op,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", op, args, kwargs, meta)

    def call_sym(
        self,
        target: Fn,
        args: tuple[Argument, ...],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", target, args, {}, meta)

    def call_cond(
        self,
        pred: ProxyValue,
        true_fn: smith.fx.GraphModule,
        false_fn: smith.fx.GraphModule,
        inputs: list[Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        true_branch = self.call_submodule(true_fn, tuple(inputs))
        false_branch = self.call_submodule(false_fn, tuple(inputs))
        if true_branch is None:
            raise AssertionError("true_branch must not be None")
        if false_branch is None:
            raise AssertionError("false_branch must not be None")
        return self._fx(
            "call_function",
            smith.ops.higher_order.cond,
            (pred, true_branch.graph_module, false_branch.graph_module, list(inputs)),
            {},
            meta,
        )

    def call_map(
        self,
        f: smith.fx.GraphModule,
        mapped_args: list[ProxyValue],
        operands: list[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        xs = _unstack_pytree([arg.data for arg in mapped_args])[0]
        f_branch = self.call_submodule(f, tuple(xs + [arg.data for arg in operands]))
        if f_branch is None:
            raise AssertionError("f_branch must not be None")
        return self._fx(
            "call_function",
            smith.ops.higher_order.map_impl,
            (f_branch.graph_module, mapped_args, operands),
            {},
            meta,
        )

    def call_getitem(
        self, value: ProxyValue, key: int, meta: NodeMetadata
    ) -> ProxyValue:
        return self._fx("call_function", operator.getitem, (value, key), {}, meta)

    def output(self, results: list[Argument], meta: NodeMetadata) -> ProxyValue:
        return self._fx("output", "output", (results,), {}, meta)

    def call_submodule(
        self, graph_module: fx.GraphModule, inputs: tuple[Argument, ...]
    ) -> PassResult:
        prev_tracer, self.tracer = (
            self.tracer,
            self.ExportTracer(self, graph_module.graph._codegen),
        )
        self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode
        interpreter = self.ExportInterpreter(self, graph_module)
        # pyrefly: ignore [bad-assignment]
        prev_interpreter, self.interpreter = (
            self.interpreter,
            smith.fx.Interpreter(  # type: ignore[assignment]
                smith.fx.GraphModule(smith.nn.Module(), smith.fx.Graph())
            ),
        )
        inputs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, inputs)
        with fx_traceback.preserve_node_meta():
            interpreter.run(*inputs_data)

        new_graph_module = smith.fx.GraphModule(self.tracer.root, self.tracer.graph)

        self.tracer = prev_tracer
        self.interpreter = prev_interpreter
        return PassResult(
            new_graph_module,
            True,
        )

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        if not getattr(self, "_initialized", False):
            raise ExportPassBaseError(
                "ExportPass is not initialized with __init__().",
            )

        inputs = self.inputs(graph_module)

        fake_tensor_mode = None
        for i in inputs:
            if isinstance(i, FakeTensor):
                if fake_tensor_mode is not None and fake_tensor_mode is not i.fake_mode:
                    raise AssertionError("Multiple fake tensor mode detected.")
                fake_tensor_mode = i.fake_mode
        if fake_tensor_mode is None:
            self.tracer.fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_tensor_mode = nullcontext()  # type: ignore[assignment]
            dispatcher_mode = nullcontext()  # type: ignore[assignment]
        else:
            fake_tensor_mode.allow_non_fake_inputs = True
            self.tracer.fake_tensor_mode = fake_tensor_mode
            dispatcher_mode = enable_python_dispatcher()  # type: ignore[assignment]
        self.fake_tensor_mode = self.tracer.fake_tensor_mode

        with fake_tensor_mode, dispatcher_mode:  # type: ignore[assignment, union-attr]
            result = self.call_submodule(graph_module, tuple(inputs))

        return result
