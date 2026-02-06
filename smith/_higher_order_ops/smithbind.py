# mypy: allow-untyped-defs
import logging
from contextlib import contextmanager

import smith
from smith._C import DispatchKey  # @manual
from smith._funcsmith._aot_autograd.utils import KNOWN_TYPES
from smith._higher_order_ops.utils import autograd_not_implemented
from smith._library.fake_class_registry import (
    _is_script_object,
    _ns_and_class_name,
    FakeScriptObject,
)
from smith._ops import HigherOrderOperator
from smith._subclasses.fake_tensor import FakeTensorMode
from smith.fx.experimental.proxy_tensor import ProxySmithDispatchMode, track_tensor_tree
from smith.fx.node import has_side_effect
from smith.utils import _pytree as pytree


log = logging.getLogger(__name__)


# The call_smithbind operator represents a method invocation on a smithbind
# object. The calling convention is:
#   call_smithbind(self: ScriptObject, method_name: str, *method_args, **method_kwargs)
# We do not expect users to write this operator directly. Instead it will be
# emitted by Dynamo when tracing encounters a smithbind object.
class CallSmithBind(HigherOrderOperator):
    def __init__(self):
        super().__init__("call_smithbind")

    def __call__(self, obj, method, *args, **kwargs):
        # pyrefly: ignore [missing-attribute]
        return super().__call__(obj, method, *args, **kwargs)

    @staticmethod
    def schema(obj, method) -> smith.FunctionSchema:
        """
        Returns the schema of ``CallSmithbind.__call__``.
        """
        if not isinstance(obj, smith._inductor.ir.SmithBindObject):
            raise AssertionError(f"expected obj to be SmithBindObject, got {type(obj)}")
        val = obj.get_real_obj()
        schema = val._get_method(method).schema
        schema_str = str(schema)
        new_schema_str = f"call_smithbind({str(schema.arguments[0].real_type)} {schema.arguments[0].name},"
        first_comma_index = schema_str.find(",")
        if first_comma_index == -1:
            # If no comma is found, find the last closing parenthesis
            first_comma_index = schema_str.rfind(") ->")
        new_schema_str = new_schema_str + " str method" + schema_str[first_comma_index:]
        new_schema = smith._C.parse_schema(new_schema_str)
        return new_schema


call_smithbind = CallSmithBind()

# Register this operator as side-effectful with FX.
# TODO: this is not really sufficient. While passes (hopefully) check
# Node.is_impure() and make good decisions, we also assume we can execute the
# graph as many times as we want without changing behavior, which is NOT true of
# ops that mutate smithbind object state.
has_side_effect(call_smithbind)

_orig_scriptmethod_call = smith.ScriptMethod.__call__


def smithbind_method_redispatch(self, *args, **kwargs):
    if _is_script_object(self.raw_owner):
        return call_smithbind(self.raw_owner, self.name, *args, **kwargs)
    return _orig_scriptmethod_call(self, *args, **kwargs)


@contextmanager
def enable_smithbind_tracing():
    """Context manager that acts as a feature flag to enable smithbind tracing
    behavior. Once smithbind tracing has been stabilized, we can remove this and
    turn it always on.
    """
    try:
        KNOWN_TYPES.append(smith.ScriptObject)
        smith.ScriptMethod.__call__ = smithbind_method_redispatch  # type: ignore[method-assign]
        yield
    finally:
        if KNOWN_TYPES.pop() is not smith.ScriptObject:
            raise AssertionError(
                "Someone else messed with KNOWN_TYPES during tracing, exploding."
            )
        smith.ScriptMethod.__call__ = _orig_scriptmethod_call  # type: ignore[method-assign]


@call_smithbind.py_impl(DispatchKey.CompositeExplicitAutograd)
def call_smithbind_impl(obj, method, *args, **kwargs):
    if isinstance(obj, smith.ScriptObject):
        return _orig_scriptmethod_call(getattr(obj, method), *args, **kwargs)
    elif isinstance(obj, FakeScriptObject):
        return getattr(obj.wrapped_obj, method)(*args, **kwargs)
    else:
        raise RuntimeError(f"Unsupported first arg type {type(obj)} for call_smithbind")


@call_smithbind.py_impl(ProxySmithDispatchMode)
def inner(mode, *args, **kwargs):
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        call_smithbind,
        proxy_args,
        proxy_kwargs,
    )
    out = call_smithbind(*args, **kwargs)

    obj, method, *_rest_args = args
    if isinstance(obj, smith.ScriptObject):
        ns, class_name = _ns_and_class_name(
            obj._type().qualified_name()  # type: ignore[attr-defined]
        )
        log.warning(
            "Tracing smithbind method %s.%s with real ScriptObject. This may"
            " cause the original object being mutated. If this is not intended,"
            ' You can register a fake class with smith._library.register_fake_class("%s::%s").',
            class_name,
            method,
            ns,
            class_name,
        )

    ret = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    if "val" not in out_proxy.node.meta:
        if out is not None and not isinstance(out, (int, float, bool)):
            raise AssertionError(
                f"Currently, only these constant dtypes are supported to be returned from smithbind methods, got {type(out)}"
            )
        out_proxy.node.meta["val"] = out
    return ret


# When tracing with fake script object, the call_smithbind op will return a fake tensor
# When tracing with real script object, the call_smithbind op may return a real tensor,
# we need to convert it to fake tensor manually. Dynamic shape is supported.
@call_smithbind.py_impl(FakeTensorMode)
def call_smithbind_fake(mode, *args, **kwargs):
    with mode:
        out = call_smithbind_impl(*args, **kwargs)
        return pytree.tree_map_only(
            smith.Tensor,
            lambda x: mode.from_tensor(x, static_shapes=True)
            if not isinstance(x, smith._subclasses.fake_tensor.FakeTensor)
            else x,
            out,
        )


call_smithbind.py_autograd_impl(
    autograd_not_implemented(call_smithbind, deferred_error=True)
)


@call_smithbind.py_functionalize_impl
def call_smithbind_func(ctx, *args, **kwargs):
    from smith._higher_order_ops.effects import handle_effects

    return handle_effects(
        ctx.mode._allow_token_discovery, ctx.mode._tokens, call_smithbind, args, kwargs
    )
