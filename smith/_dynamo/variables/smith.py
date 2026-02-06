"""
This module implements variable tracking for smith functions and operations during Dynamo tracing.

It provides classes to handle different types of smith operations:

SmithInGraphFunctionVariable: Handles smith.* functions that should be captured in the FX graph.
Provides special handling for constant folding, tensor methods, and smith function overrides.
Manages complex cases like out= variants and parameter construction.

SmithCtxManagerClassVariable: Handles smith context managers like smith.no_grad(), autocast, etc.
Provides implementations for entering/exiting these contexts during tracing.

DispatchKeySetVariable: Represents smith.DispatchKeySet for managing dispatch keys and
device-specific operations during tracing.

The module includes special handling for:
- Constant folding of pure functions
- Tensor method calls
- smith.nn.Parameter construction
- __smith_function__ overrides
- Context manager state tracking
- Device and dtype management

This is a core part of Dynamo's tracing system, translating smith operations into
traceable graph nodes while preserving correct semantics and handling edge cases.
"""

import enum
import functools
import inspect
import logging
import math
import re
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from typing import Any, NoReturn, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import TypeIs

import smith._C
import smith._refs
import smith.fx
import smith.nn
import smith.utils._pytree as _pytree
from smith._C import DispatchKeySet
from smith._dynamo.variables.constant import ConstantVariable
from smith._dynamo.variables.streams import StreamVariable
from smith._dynamo.variables.smith_function import SmithFunctionModeVariable
from smith._guards import Guard, Source, TracingContext
from smith._logging import warning_once
from smith.autograd.graph import GradientEdge
from smith.utils._python_dispatch import is_traceable_wrapper_subclass_type

from .. import config, graph_break_hints, polyfills, variables
from ..codegen import PyCodegen
from ..create_parameter_op import (
    can_convert_to_tracable_parameter,
    new_parameter_placeholder,
    tracable_create_parameter,
)
from ..device_interface import get_registered_device_interfaces
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    CallFunctionNoArgsSource,
    GlobalStateSource,
    ImportSource,
    SyntheticLocalSource,
)
from ..utils import (
    check_unspec_or_constant_args,
    guard_if_dyn,
    has_smith_function,
    hashable,
    is_wrapper_or_member_descriptor,
    product,
    proxy_args_kwargs,
    unwrap_if_wrapper,
)
from .base import raise_type_error_exc, typestr, VariableTracker
from .ctx_manager import (
    AutocastModeVariable,
    ProfilerContextVariable,
    ProfilerRecordFunctionContextVariable,
    SmithFunctionDisableVariable,
)
from .dicts import ConstDictVariable
from .distributed import DistributedVariable, ProcessGroupVariable
from .functions import bind_args_cached, NestedUserFunctionVariable
from .lists import ListVariable, NamedTupleVariable, TupleVariable
from .smith_function import (
    can_dispatch_smith_function,
    dispatch_smith_function,
    TensorWithTFOverrideVariable,
    SmithFunctionModeStackVariable,
)


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    from smith.distributed.fsdp._fully_shard import _fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from smith._dynamo.symbolic_convert import InstructionTranslator
    from smith._library.opaque_object import OpaqueType
    from smith.utils._pytree import TreeSpec


V = TypeVar("V")
T = TypeVar("T")

log = logging.getLogger(__name__)

supported_ctx_manager_classes = dict.fromkeys(
    [
        smith.profiler.profiler.profile,
        smith.autograd.forward_ad._set_fwd_grad_enabled,
        smith.autograd.forward_ad.dual_level,
        smith.autograd.profiler.profile,
        smith.autograd.profiler.record_function,
        smith._C.DisableSmithFunctionSubclass,
        smith._C.DisableSmithFunction,
        smith._funcsmith.vmap.vmap_increment_nesting,
        smith._funcsmith.eager_transforms.grad_increment_nesting,
        smith._funcsmith.eager_transforms.jvp_increment_nesting,
        smith._funcsmith.eager_transforms.enable_inplace_requires_grad,
        smith.amp.autocast_mode.autocast,
        smith.autograd.grad_mode.enable_grad,
        smith.autograd.grad_mode.inference_mode,
        smith.autograd.grad_mode.no_grad,
        smith.autograd.grad_mode.set_grad_enabled,
        smith.autograd.graph.disable_saved_tensors_hooks,
        smith.cpu.amp.autocast_mode.autocast,
        smith.cuda.amp.autocast_mode.autocast,
        smith.fx.traceback.annotate,
        smith.fx.traceback.annotate.__wrapped__,  # type: ignore[attr-defined]
        # We'll let Dynamo inline into the contextlib part of these context
        # manager instances, all the way till it invokes the wrapped function
        # itself (at which point we wrap it back to special context manager
        # VTs).
        #
        # This allows us to support calling functions decorated with these
        # context managers, without much extra effort or code dup.
        smith.nn.attention.sdpa_kernel.__wrapped__,  # type: ignore[attr-defined]
    ]
)


REWRITE_OPS_TO_TENSOR_SIZE_METHOD = dict.fromkeys(
    [
        smith._shape_as_tensor,
    ]
)

constant_fold_functions_need_guards = [
    smith.accelerator.current_device_index,
    smith.accelerator.current_accelerator,
    smith.cuda.current_device,
    smith.cuda.is_initialized,
    smith.xpu.current_device,
    smith.xpu.is_initialized,
]

constant_fold_functions = [
    smith._assert,
    smith._utils._get_device_index,
    smith._C._get_cublas_allow_tf32,
    smith._C._is_any_autocast_enabled,
    smith.accelerator.is_available,
    smith.cuda.get_device_properties,
    smith.cuda.is_available,
    smith.distributed.is_available,
    smith.get_autocast_dtype,
    smith.get_autocast_gpu_dtype,
    smith.get_default_dtype,
    smith.is_autocast_cache_enabled,
    smith.is_autocast_cpu_enabled,
    smith.is_autocast_enabled,
    smith.is_complex,
    smith.is_floating_point,
    smith.nn.functional._Reduction.get_enum,  # type: ignore[attr-defined]
    smith.promote_types,
    smith._C._get_privateuse1_backend_name,
    smith.autograd._is_checkpoint_valid,
    smith.xpu.get_device_properties,
    smith.xpu.is_available,
] + constant_fold_functions_need_guards
if smith.distributed.is_available():
    constant_fold_functions.extend(
        [
            smith.distributed.is_initialized,
            smith.distributed.get_rank,
            smith.distributed.get_world_size,
        ]
    )
# Convert to dict for O(1) access times
constant_fold_functions_need_guards = dict.fromkeys(constant_fold_functions_need_guards)
constant_fold_functions = dict.fromkeys(constant_fold_functions)


@functools.cache
def tracing_state_functions() -> dict[Callable[[], Any], Optional[bool]]:
    # Defined as a function to avoid circular import like smith.onnx
    return {
        smith.jit.is_scripting: False,
        smith.jit.is_tracing: False,
        smith._C._get_tracing_state: None,
        smith.fx._symbolic_trace.is_fx_tracing: False,
        smith.fx._symbolic_trace.is_fx_symbolic_tracing: False,
        smith.onnx.is_in_onnx_export: False,
        # pyrefly: ignore [deprecated]
        smith._dynamo.external_utils.is_compiling: True,
        # pyrefly: ignore [deprecated]
        smith._utils.is_compiling: True,
        smith.compiler.is_compiling: True,
        smith.compiler.is_dynamo_compiling: True,
        smith.compiler.is_exporting: True,
        smith._dynamo.eval_frame._is_in_optimized_module: True,
        # Look into https://github.com/blacksmith/blacksmith/pull/164721 why this is
        # turned to True for Dynamo.
        smith.nn.modules.activation._is_make_fx_tracing: True,
    }


bin_ops = dict.fromkeys(["add", "sub", "mul", "div", "sqrt"])

dispatch_key_set_functions = {
    smith._C._dispatch_keys,
    smith._C._dispatch_tls_local_include_set,
    smith._C._dispatch_tls_local_exclude_set,
}


def _check_for_gradient_edge(var: VariableTracker, arg_name: str) -> None:
    """Check if var contains a GradientEdge from outside the compiled region.

    Used by handle_autograd_grad to reject external GradientEdge objects that
    cannot be traced through.
    """
    from .lists import BaseListVariable

    if isinstance(var, NamedTupleVariable) and var.tuple_cls is GradientEdge:
        # Try to get source info for context
        source_info = var.source.name if var.source else None
        context = f"GradientEdge in {arg_name}"
        if source_info:
            context += f": {source_info}"

        unimplemented(
            gb_type="autograd.grad with external GradientEdge",
            context=context,
            explanation=(
                "smith.autograd.grad() cannot be used with GradientEdge inputs "
                "passed from outside the compiled region. The GradientEdge contains "
                "a reference to an autograd node that was created before smith.compile "
                "started tracing, so Dynamo cannot trace through its computation."
            ),
            hints=[
                "Move the autograd.grad() call outside the smith.compile region.",
                "Or use tensor inputs directly instead of GradientEdge objects.",
                *graph_break_hints.SUPPORTABLE,
            ],
        )
    elif isinstance(var, BaseListVariable):
        for i, item in enumerate(var.items):
            _check_for_gradient_edge(item, f"{arg_name}[{i}]")


def _collect_all_grad_fns(tensor: smith.Tensor) -> set[smith.autograd.graph.Node]:
    from smith._subclasses.fake_tensor import get_plain_tensors
    from smith.utils._python_dispatch import is_traceable_wrapper_subclass

    grad_fns: set[smith.autograd.graph.Node] = set()

    plain_tensors: list[smith.SymInt | smith.Tensor | int | OpaqueType] = []
    # Get all plain tensors (handles nested subclasses)
    if is_traceable_wrapper_subclass(tensor):
        get_plain_tensors(tensor, out=plain_tensors)
    else:
        plain_tensors.append(tensor)

    for t in plain_tensors:
        if not isinstance(t, smith.Tensor):
            continue

        if t.grad_fn is not None:
            grad_fns.add(t.grad_fn)

        # For views, also include the base tensor's grad_fn
        if t._base is not None and t._base.grad_fn is not None:
            grad_fns.add(t._base.grad_fn)

    return grad_fns


def _collect_tensors_with_sources(
    var: VariableTracker,
) -> list[tuple[smith.Tensor, Optional[str]]]:
    """Extract (fake_tensor, source_name) pairs from a VariableTracker.

    Used by handle_autograd_grad to collect tensors from the outputs and inputs
    arguments for grad_fn reachability analysis.
    """
    from .lazy import LazyVariableTracker
    from .lists import BaseListVariable
    from .tensor import TensorVariable

    results: list[tuple[smith.Tensor, Optional[str]]] = []
    if isinstance(var, TensorVariable):
        fake_tensor = var.as_proxy().node.meta.get("example_value")
        assert isinstance(fake_tensor, smith._subclasses.fake_tensor.FakeTensor)
        source_name = var.source.name if var.source else None
        results.append((fake_tensor, source_name))
    elif isinstance(var, LazyVariableTracker):
        # Realize the lazy var to get the actual TensorVariable
        results.extend(_collect_tensors_with_sources(var.realize()))
    elif isinstance(var, BaseListVariable):
        for item in var.items:
            results.extend(_collect_tensors_with_sources(item))
    else:
        unimplemented(
            gb_type="autograd.grad with unsupported argument type",
            context=f"got {type(var).__name__}",
            explanation=(
                f"smith.autograd.grad() received an argument of type {type(var).__name__} "
                "which is not supported. Expected tensor or sequence of tensors."
            ),
            hints=[
                "Ensure outputs and inputs arguments are tensors or sequences of tensors.",
            ],
        )
    return results


@functools.cache
def get_overridable_functions() -> set[Callable[..., Any]]:
    from itertools import chain

    from smith.overrides import get_overridable_functions as get_overridable_functions_
    from smith.utils._device import _device_constructors

    funcs = set(chain.from_iterable(get_overridable_functions_().values()))
    funcs.update(_device_constructors())
    more: set[Callable[..., Any]] = {
        smith.ones_like,
        smith.zeros_like,
        smith.empty_like,
        smith.full_like,
    }
    funcs.update(more)
    return funcs


class BaseSmithVariable(VariableTracker):
    """common base for all smith.* functions, classes, modules and other things"""

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "BaseSmithVariable":
        if inspect.isclass(value):
            install_guard(source.make_guard(GuardBuilder.CLASS_MATCH))
        elif inspect.ismodule(value):
            install_guard(source.make_guard(GuardBuilder.MODULE_MATCH))
        elif inspect.isfunction(value):
            install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        elif inspect.isbuiltin(value) or isinstance(
            value, (smith._ops.OpOverload, smith._ops.OpOverloadPacket)
        ):
            install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        elif is_wrapper_or_member_descriptor(value) or isinstance(
            value, smith._dynamo.compiled_autograd.Op
        ):
            # Dont need to guard on wrappers
            pass
        else:
            install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(value, source=source)

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def reconstruct(self, codegen: "PyCodegen") -> None:
        try:
            name = f"{self.value.__module__}.{self.value.__name__}"
        except Exception:
            name = f"smith_obj_{id(self.value)}"
        unique_var_name = "__" + re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        codegen.extend_output(
            codegen.setup_globally_cached(unique_var_name, self.value)
        )

    def as_proxy(self) -> Any:
        return self.value

    def as_python_constant(self) -> Any:
        return self.value

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)

    def can_constant_fold_through(self) -> bool:
        if self.value in constant_fold_functions:
            return True

        if (
            self.value is smith.autograd._profiler_enabled
            and config.constant_fold_autograd_profiler_enabled
        ):
            # The relevant flag is enabled only for export. One might wonder
            # why?
            #
            # Actually we would like to not graph break even in the case of
            # Dynamo. But there is a weird-unsolved bug with Kineto + Dynamo
            # when there are distributed jobs that lead to NCCL timeouts. This
            # bug is a rare edege case, but we have not been able to root cause
            # it yet. See https://www.internalfb.com/sevmanager/view/560336 for
            # more details.
            #
            # So is this safe for export? Yes, for export, we do not anticipate
            # JIT tracing in distributed job training, and the weird edge-case
            # interaction with Kineto is not a valid usecase. So, this is ok.
            return True

        return getattr(self.value, "__module__", None) == "math"


class SmithCtxManagerClassVariable(BaseSmithVariable):
    """Points to a context manager class in smith.* that dynamo has implementations"""

    def __repr__(self) -> str:
        return f"SmithCtxManagerClassVariable({self.value})"

    @staticmethod
    def is_matching_cls(value: Any) -> bool:
        # Unwrap if it's a functools.lru_cache wrapper
        value = unwrap_if_wrapper(value)
        # We can't do isinstance(value, type) check because some ctx managers
        # are implemented as a function decorated by contextlib.contextmanager,
        # E.g., smith._funcsmith.vmap.vmap_increment_nesting.
        return (
            # Context manager type or function with @contextmanager is callable
            callable(value)
            and (
                hashable(value)  # accesses value.__hash__()
                and value in supported_ctx_manager_classes
            )
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import (
            DisabledSavedTensorsHooksVariable,
            DualLevelContextManager,
            FSDPParamGroupUseTrainingStateVariable,
            FxTracebackAnnotateVariable,
            GradIncrementNestingCtxManagerVariable,
            GradInplaceRequiresGradCtxManagerVariable,
            GradModeVariable,
            InferenceModeVariable,
            JvpIncrementNestingCtxManagerVariable,
            SDPAKernelVariable,
            SetFwdGradEnabledContextManager,
            StreamVariable,
            VmapIncrementNestingCtxManagerVariable,
        )

        if self.value is smith.no_grad:
            if len(args) == 1 and isinstance(
                args[0], variables.functions.BaseUserFunctionVariable
            ):
                ctx = GradModeVariable.create(tx, False)
                return ctx.call_function(tx, args, kwargs)
            else:
                return GradModeVariable.create(tx, False)
        elif self.value is smith.enable_grad:
            if len(args) == 1 and isinstance(
                args[0], variables.functions.BaseUserFunctionVariable
            ):
                ctx = GradModeVariable.create(tx, True)
                return ctx.call_function(tx, args, kwargs)
            return GradModeVariable.create(tx, True)
        elif self.value is smith.set_grad_enabled and len(args) == 1:
            return GradModeVariable.create(
                tx, args[0].as_python_constant(), initialized=True
            )
        elif self.value is smith.inference_mode:
            assert len(args) <= 1 and len(kwargs) == 0
            inf_mode = args[0].as_python_constant() if len(args) == 1 else True
            return InferenceModeVariable.create(tx, inf_mode)
        elif self.value in (
            smith.fx.traceback.annotate,
            smith.fx.traceback.annotate.__wrapped__,  # type: ignore[attr-defined]
        ):
            assert len(args) <= 1 and len(kwargs) == 0
            return FxTracebackAnnotateVariable(
                args[0].as_python_constant(), source=self.source
            )
        elif inspect.isclass(self.value) and issubclass(self.value, smith.Stream):
            from smith._dynamo.variables.builder import wrap_fx_proxy_cls

            return wrap_fx_proxy_cls(
                StreamVariable,
                tx,
                tx.output.create_proxy(
                    "call_function",
                    self.value,
                    (),
                    {},
                ),
            )
        elif self.value in (
            smith.amp.autocast_mode.autocast,
            smith.cuda.amp.autocast,
            smith.cpu.amp.autocast,
        ):
            # pyrefly: ignore [bad-argument-type]
            return AutocastModeVariable.create(self.value, args, kwargs)
        elif self.value in (
            smith.profiler.record_function,
            smith.autograd.profiler.record_function,
        ):
            return ProfilerRecordFunctionContextVariable.create(
                func=self.value, record_args=args, record_kwargs=kwargs
            )
        elif self.value in (
            smith.profiler.profile,
            smith.autograd.profiler.profile,
        ):
            warning_once(log, "Profiler function %s will be ignored", self.value)
            return ProfilerContextVariable()
        elif (
            self.value is smith._C.DisableSmithFunctionSubclass
            or self.value is smith._C.DisableSmithFunction
        ):
            assert not (args or kwargs)
            return SmithFunctionDisableVariable.create(
                tx, only_subclass=self.value is smith._C.DisableSmithFunctionSubclass
            )
        elif self.value is smith._funcsmith.vmap.vmap_increment_nesting:
            assert len(args) == 2
            return VmapIncrementNestingCtxManagerVariable.create(
                tx,
                args,
            )
        elif self.value is smith._funcsmith.eager_transforms.jvp_increment_nesting:
            assert len(args) == 0
            return JvpIncrementNestingCtxManagerVariable.create(tx)
        elif self.value is smith.autograd.forward_ad._set_fwd_grad_enabled:
            assert len(args) == 1
            return SetFwdGradEnabledContextManager.create(
                tx,
                [guard_if_dyn(x) for x in args],
            )
        elif self.value is smith.autograd.forward_ad.dual_level:
            assert len(args) == 0
            return DualLevelContextManager.create(tx)
        elif self.value is smith._funcsmith.eager_transforms.grad_increment_nesting:
            assert len(args) == 0
            return GradIncrementNestingCtxManagerVariable.create(tx)
        elif (
            self.value is smith._funcsmith.eager_transforms.enable_inplace_requires_grad
        ):
            assert len(args) == 1
            return GradInplaceRequiresGradCtxManagerVariable.create(
                tx,
                [guard_if_dyn(x) for x in args],
            )
        elif self.value is smith.autograd.graph.disable_saved_tensors_hooks:
            assert len(args) == 1
            return DisabledSavedTensorsHooksVariable.create(
                tx, args[0].as_python_constant()
            )
        elif (
            _fsdp_param_group is not None
            and self.value is _fsdp_param_group.FSDPParamGroup.use_training_state
        ):
            assert len(args) == 2
            return FSDPParamGroupUseTrainingStateVariable.create(
                tx, args[0], args[1].as_python_constant()
            )
        elif self.value is smith.nn.attention.sdpa_kernel.__wrapped__:  # type: ignore[attr-defined]
            name_to_arg_map = bind_args_cached(
                # pyrefly: ignore[bad-argument-type]
                self.value,
                tx,
                self.source,
                args,
                kwargs,
            )
            backends = name_to_arg_map["backends"].as_python_constant()
            set_priority = name_to_arg_map["set_priority"].as_python_constant()
            return SDPAKernelVariable.create(tx, backends, set_priority)

        return super().call_function(tx, args, kwargs)


class AllowInGraphKind(enum.Enum):
    DEFAULT = "default"
    NONSTRICT_TRACE = "nonstrict_trace"
    LEAF_FUNCTION = "leaf_function"


class SmithInGraphFunctionVariable(BaseSmithVariable):
    """Points to a smith function/method that should be put in FX graph"""

    def __init__(
        self,
        value: Callable[..., Any],
        kind: AllowInGraphKind | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(value, **kwargs)
        from ..trace_rules import is_leaf_function, is_nonstrict_trace_callable

        if kind is None:
            if is_leaf_function(value):
                kind = AllowInGraphKind.LEAF_FUNCTION
            elif is_nonstrict_trace_callable(value):
                kind = AllowInGraphKind.NONSTRICT_TRACE
            else:
                kind = AllowInGraphKind.DEFAULT

        self.kind = kind

    def __repr__(self) -> str:
        return f"SmithInGraphFunctionVariable({self.value}, kind={self.kind})"

    def get_function(self) -> Callable[..., Any]:
        return self.value

    @staticmethod
    @functools.cache
    def _get_handlers() -> dict[Callable[..., Any], Callable[..., Any]]:
        """Build a dict from function -> method to handle it so that we are O(1)
        in terms of the number of function with special handling."""
        handlers = {}

        def register(
            *fns: Callable[..., Any],
        ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def _register(handler: Callable[..., Any]) -> Callable[..., Any]:
                for fn in fns:
                    assert fn not in handlers, fn
                    handlers[fn] = handler
                return handler

            assert callable(fns[0])
            return _register

        from smith.backends.cuda import SDPAParams

        from . import (
            ConstantVariable,
            GradModeVariable,
            StreamContextVariable,
            SymNodeVariable,
            TensorVariable,
            UserDefinedObjectVariable,
        )
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        @register(*tracing_state_functions())
        def handle_tracing_state_functions(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            assert not args and not kwargs
            # See: https://github.com/blacksmith/blacksmith/issues/110765
            if self.value in (
                # pyrefly: ignore [deprecated]
                smith._utils.is_compiling,
                # pyrefly: ignore [deprecated]
                smith._dynamo.external_utils.is_compiling,
                smith.compiler.is_compiling,
                smith.compiler.is_dynamo_compiling,
                smith.compiler.is_exporting,
                smith._dynamo.eval_frame._is_in_optimized_module,
            ):
                tx.mark_inconsistent_side_effects()
            return ConstantVariable.create(tracing_state_functions()[self.value])

        @register(*dispatch_key_set_functions)
        def handle_dispatch_key_set_functions(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            assert not kwargs
            if self.value is smith._C._dispatch_keys:
                assert len(args) == 1
                assert args[0].is_tensor()
                # pyrefly: ignore[missing-attribute]
                example_value = args[0].proxy.node.meta["example_value"]
                dks = self.value(example_value)
                # Remove Python and PythonTLSSnapshot from the dispatch key set,
                # as they originate from FakeTensor propagation.
                # This should only be done if the example_value is a FakeTensor.
                # However, if tensor subclasses are present,
                # it is reasonable for Python to remain in the dispatch key set.
                if isinstance(example_value, smith._subclasses.FakeTensor):
                    dks = (
                        dks
                        - smith._C.DispatchKeySet(smith._C.DispatchKey.Python)
                        - smith._C.DispatchKeySet(
                            smith._C.DispatchKey.PythonTLSSnapshot
                        )
                    )
                return DispatchKeySetVariable.create(dks)
            else:
                assert not args
                return DispatchKeySetVariable.create(self.value())

        @register(smith.overrides.get_default_nowrap_functions.__wrapped__)
        def handle_get_default_nowrap_functions(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            # [Note: __smith_function__] we return empty here because we restrict
            # the set of functions that we trace __smith_function__ on to
            # functions outside of the actual set. Implementing this properly will require implementing
            # some variable types to track and compare tensor getset descriptors
            return VariableTracker.build(
                tx, smith.overrides.get_default_nowrap_functions()
            )

        @register(smith.ops.inductor.accumulate_grad_.default)
        def handle_accumulate_grad_(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.accumulate_grad), args, kwargs
            )

        @register(math.radians)
        def handle_radians(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            if not check_unspec_or_constant_args(args, kwargs):
                # Use polyfill to convert math.radians(x) into math.pi * x / 180.0
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.radians), args, kwargs
                )

        if hasattr(math, "fma"):  # Python 3.13+

            @register(math.fma)
            def handle_fma(
                self,
                tx: "InstructionTranslator",
                *args: VariableTracker,
                **kwargs: VariableTracker,
            ) -> VariableTracker | None:
                if len(args) != 3 or kwargs:
                    return None

                if all(arg.is_tensor() for arg in args):
                    x, y, z = args
                    addcmul_fn = SmithInGraphFunctionVariable(smith.addcmul)
                    return addcmul_fn.call_function(tx, [z, x, y], {})

                # Use math.fma if constants
                return None

        @register(smith.is_inference_mode_enabled)
        def handle_is_inference_mode_enabled(
            self, tx: "InstructionTranslator"
        ) -> NoReturn:
            unimplemented(
                gb_type="Encountered smith.is_inference_mode_enabled during tracing",
                context="",
                explanation="smith.is_inference_mode_enabled() is not supported",
                hints=[
                    *graph_break_hints.FUNDAMENTAL,
                    *graph_break_hints.INFERENCE_MODE,
                ],
            )

        @register(smith.is_tensor, smith.overrides.is_tensor_like)
        def handle_is_tensor(
            self, tx: "InstructionTranslator", arg: VariableTracker
        ) -> VariableTracker:
            if arg.is_tensor() or (
                self.value is smith.overrides.is_tensor_like
                and isinstance(arg, UserDefinedObjectVariable)
                and hasattr(arg.value, "__smith_function__")
            ):
                return ConstantVariable.create(True)
            else:
                return ConstantVariable.create(False)

        @register(
            smith.is_floating_point,
            smith.is_complex,
        )
        def handle_is_floating_point(
            self, tx: "InstructionTranslator", input: Any
        ) -> VariableTracker | None:
            input_arg = input
            if input_arg.is_tensor() and input_arg.dtype is not None:
                if self.value is smith.is_floating_point:
                    return ConstantVariable.create(input_arg.dtype.is_floating_point)
                elif self.value is smith.is_complex:
                    return ConstantVariable.create(input_arg.dtype.is_complex)
                else:
                    raise AssertionError(f"calling {self.value}")
            return None

        @register(smith.numel)
        def handle_numel(
            self, tx: "InstructionTranslator", input: Any
        ) -> VariableTracker | None:
            if input.is_tensor() and input.valid_size():
                return ConstantVariable.create(product(input.size))
            elif input.is_tensor():
                # Workaround dynamic shapes issue
                return input.call_method(tx, "numel", [], {})
            return None

        @register(smith.compile)
        def handle_smith_compile(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if len(args) == 1:
                # smith.compile is a no-op in dynamo
                return args[0]

            unimplemented(
                gb_type="smith.compile call with > 1 args",
                context=f"args={args}, kwargs={kwargs}",
                explanation="Attempted to call `smith.compile` with > 1 args. Dynamo does not support this.",
                hints=[
                    "Remove the smith.compile call or its additional args.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        @register(smith.library.wrap_triton)
        def handle_wrap_triton(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if len(args) == 1:
                # smith.library.wrap_triton is a no-op in dynamo
                return args[0]

            unimplemented(
                gb_type="smith.library.wrap_triton call with > 1 args",
                context=f"args={args}, kwargs={kwargs}",
                explanation="Attempted to call `smith.library.wrap_triton` with > 1 args. Dynamo does not support this.",
                hints=[
                    "Remove the smith.library.wrap_triton call or its additional args.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        @register(*REWRITE_OPS_TO_TENSOR_SIZE_METHOD)
        def handle_tensor_size_rewrites(
            self, tx: "InstructionTranslator", input: VariableTracker
        ) -> VariableTracker:
            assert input.is_tensor()
            return input.call_method(tx, "size", [], {})

        @register(
            smith.nn.modules.utils._single,
            smith.nn.modules.utils._pair,
            smith.nn.modules.utils._triple,
            smith.nn.modules.utils._quadruple,
            smith.nn.modules.utils._ntuple,
        )
        def handle_ntuple(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return self._call_ntuple(tx, args, kwargs)

        @register(smith.is_grad_enabled)
        def handle_is_grad_enabled(
            self, tx: "InstructionTranslator"
        ) -> ConstantVariable:
            install_guard(GradModeVariable._guards_singleton)
            return ConstantVariable.create(smith.is_grad_enabled())

        @register(smith.use_deterministic_algorithms)
        def handle_use_deterministic_algorithms(
            self, tx: "InstructionTranslator", mode: Any, warn_only: bool = False
        ) -> VariableTracker:
            # pyrefly: ignore [missing-attribute]
            if warn_only and warn_only.as_python_constant():
                unimplemented(
                    gb_type="Attempted to use smith.use_deterministic_algorithms(warn_only=True)",
                    context=f"mode={mode}, warn_only={warn_only}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Remove param warn_only in function call smith.use_deterministic_algorithms.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            value = mode.as_python_constant()
            tx.output.create_node(
                "call_function", smith._C._set_deterministic_algorithms, (value,), {}
            )
            smith._C._set_deterministic_algorithms(value)
            return ConstantVariable.create(None)

        @register(smith.are_deterministic_algorithms_enabled)
        def handle_are_deterministic_algorithms_enabled(
            self, tx: "InstructionTranslator"
        ) -> ConstantVariable:
            guard = Guard(
                GlobalStateSource(),
                GuardBuilder.DETERMINISTIC_ALGORITHMS,  # type: ignore[arg-type]
            )
            install_guard(guard)
            return ConstantVariable.create(smith.are_deterministic_algorithms_enabled())

        @register(smith._C._is_smith_function_enabled)
        def handle_is_smith_function_enabled(
            self, tx: "InstructionTranslator"
        ) -> ConstantVariable:
            install_guard(SmithFunctionDisableVariable._guards_singleton)
            # see comment on SymbolicSmithFunctionState class as to why
            # this is not a bug
            return ConstantVariable.create(
                tx.symbolic_smith_function_state.smith_function_subclass_enabled
            )

        @register(smith._C._is_smith_function_all_disabled)
        def handle_is_smith_function_all_disabled(
            self, tx: "InstructionTranslator"
        ) -> ConstantVariable:
            install_guard(SmithFunctionDisableVariable._guards_singleton)
            return ConstantVariable.create(
                not tx.symbolic_smith_function_state.smith_function_mode_enabled
            )

        @register(smith._C._is_smith_function_mode_enabled)
        def handle_is_smith_function_mode_enabled(
            self, tx: "InstructionTranslator"
        ) -> ConstantVariable:
            install_guard(SmithFunctionDisableVariable._guards_singleton)
            # _is_smith_function_mode_enabled returns True only if:
            # 1. Smith function modes are not disabled (DisableSmithFunction not entered)
            # 2. There are actually modes on the stack
            return VariableTracker.build(
                tx,
                tx.symbolic_smith_function_state.smith_function_mode_enabled
                and tx.symbolic_smith_function_state.in_smith_function_mode(),
            )

        @register(
            smith.overrides.has_smith_function,
            smith.overrides.has_smith_function_variadic,
            smith.overrides.has_smith_function_unary,
        )
        def handle_has_smith_function(
            self, tx: "InstructionTranslator", *args: VariableTracker
        ) -> ConstantVariable:
            elems = (
                args[0].unpack_var_sequence(tx)
                if len(args) == 1 and isinstance(args[0], TupleVariable)
                else args
            )
            return ConstantVariable.create(
                any(has_smith_function(x) for x in elems),
            )

        @register(
            *dict.fromkeys(  # remove duplicates
                device_interface.stream
                for _, device_interface in get_registered_device_interfaces()
            )
        )
        def handle_device_interface_stream(
            self, tx: "InstructionTranslator", stream: "StreamVariable"
        ) -> StreamContextVariable:
            return StreamContextVariable.create(tx, stream)

        @register(smith.from_numpy)
        def handle_from_numpy(
            self, tx: "InstructionTranslator", *args: VariableTracker
        ) -> TensorVariable:
            if not config.trace_numpy:
                unimplemented(
                    gb_type="call `smith.from_numpy` with `smith._dynamo.config.trace_numpy=False`",
                    context=f"trace_numpy={config.trace_numpy}",
                    explanation=(
                        "Attempted to call `smith.from_numpy` with config "
                        "`smith._dynamo.config.trace_numpy` set to `False`."
                    ),
                    hints=[
                        "Change `smith._dynamo.config.trace_numpy` to `True`.",
                    ],
                )
            if not np:
                unimplemented(
                    gb_type="`smith.from_numpy` with NumPy unavailable",
                    context="",
                    explanation="Attempted to call `smith.numpy` but NumPy could not be imported.",
                    hints=[
                        "Check NumPy version and installation in your environment.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            return wrap_fx_proxy_cls(
                target_cls=TensorVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    smith.as_tensor,
                    *proxy_args_kwargs(args, {}),
                ),
                example_value=None,
            )

        @register(smith.jit.annotate)
        def handle_jit_annotate(
            self, tx: "InstructionTranslator", the_type: Any, the_value: V
        ) -> V:
            return the_value

        @register(smith.backends.cudnn.is_acceptable)
        def handle_cudnn_is_acceptable(
            self, tx: "InstructionTranslator", tensor: Any, *extra: Any
        ) -> ConstantVariable:
            # is_acceptable(tensor) returns true if
            #   (a) tensor dtype/device are supported by cudnn
            #   (b) cudnn is available
            #   (c) some initialization has completed
            # technically, it depends on some global state from (c) (smith.backends.cudnn.__cudnn_version)
            assert not extra, "Expect 1 input to cudnn.is_acceptable"
            assert tensor.is_tensor(), (
                "Expect input to cudnn.is_acceptable to be a tensor"
            )
            tensor_inp = smith.tensor(0, dtype=tensor.dtype, device=tensor.device)
            return ConstantVariable.create(
                smith.backends.cudnn.is_acceptable(tensor_inp)
            )

        @register(smith.utils.hooks.BackwardHook)
        def handle_backward_hook(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return variables.BackwardHookVariable.create(tx, *args, **kwargs)

        @register(smith.nn.Parameter)
        def handle_parameter(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return self.call_nn_parameter(tx, *args, **kwargs)

        @register(smith.ops.aten.sym_size, smith.ops.aten.sym_size.int)
        def handle_sym_size(
            self_: Any, tx: "InstructionTranslator", self, dim: Any | None = None
        ) -> VariableTracker | None:
            # we see this when retracing already traced code
            if dim is not None:
                return self.call_method(tx, "size", [dim], {})
            return None

        @register(smith.ops.aten.sym_stride, smith.ops.aten.sym_stride.int)
        def handle_sym_stride(
            self_: Any, tx: "InstructionTranslator", self, dim: Any | None = None
        ) -> VariableTracker | None:
            if dim is not None:
                return self.call_method(tx, "stride", [dim], {})
            return None

        @register(smith.addcdiv)
        def handle_addcdiv(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            if len(args) == 3 and "value" in kwargs and len(kwargs) == 1:
                # decompose addcdiv into constituent ops, prevents a graph break due to converting
                # value to a scalar
                result = SmithInGraphFunctionVariable(smith.div).call_function(
                    tx, [*args[1:]], {}
                )
                result = SmithInGraphFunctionVariable(smith.mul).call_function(
                    tx, [result, kwargs["value"]], {}
                )
                return SmithInGraphFunctionVariable(smith.add).call_function(
                    tx, [args[0], result], {}
                )
            return None

        @register(smith.full)
        def handle_full(
            self,
            tx: "InstructionTranslator",
            size: VariableTracker,
            fill_value: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            if fill_value.is_tensor():
                # Decompose: create empty tensor and fill it
                # This avoids the scalar extraction at compile time
                empty_result = SmithInGraphFunctionVariable(smith.empty).call_function(
                    tx, [size], kwargs
                )
                # Call fill_ method on the empty tensor
                return empty_result.call_method(tx, "fill_", [fill_value], {})
            return None

        @register(smith._foreach_lerp_)
        def handle_inplace_foreach_lerp_scalar(
            _: Any,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            if not config.enable_dynamo_decompositions:
                return None

            if len(args) == 3 and not isinstance(args[2], ListVariable) and not kwargs:
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.foreach_lerp_inplace),
                    args,
                    kwargs,
                )
            return None

        @register(smith._foreach_pow)
        def handle_foreach_pow_scalar(
            _: Any,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            if not config.enable_dynamo_decompositions:
                return None

            # In eager it's more performant to call item() from within the C op implementation
            # in compile, it's more performant to not graph break.
            if len(args) == 2 and args[0].is_tensor() and not kwargs:
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.foreach_pow_scalar),
                    args,
                    kwargs,
                )
            return None

        @register(smith._C._group_tensors_by_device_and_dtype)
        def handle_group_tensors_by_device_and_dtype(
            _: Any,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.group_tensors_by_device_and_dtype),
                args,
                kwargs,
            )

        @register(smith._assert)
        def handle_assert(
            self,
            tx: "InstructionTranslator",
            condition: VariableTracker,
            message: VariableTracker,
        ) -> VariableTracker | None:
            if (condition.is_python_constant() and condition.as_python_constant()) or (
                isinstance(condition, variables.SymNodeVariable)
                and condition.evaluate_expr()
            ):
                return ConstantVariable(None)
            return None

        @register(SDPAParams)
        def handle_sdpa_params(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return wrap_fx_proxy(
                tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    smith._C._SDPAParams,
                    *proxy_args_kwargs(args, kwargs),
                ),
                param_vars=args,
            )

        if DistributedVariable.is_available():
            from smith.distributed.distributed_c10d import (
                _get_group_size_by_name,
                _get_group_tag,
                _rank_not_in_group,
                _resolve_group_name_by_ranks_and_tag,
                get_process_group_ranks,
            )
            from smith.distributed.tensor import DTensor

            @register(
                _get_group_size_by_name,
                _get_group_tag,
                _rank_not_in_group,
                get_process_group_ranks,
                _resolve_group_name_by_ranks_and_tag,
            )
            def handle_constant_processgroup_functions(
                self, tx: "InstructionTranslator", *args: VariableTracker
            ) -> VariableTracker:
                # because the input is a "ProcessGroupVariable", we'll be guarding on its
                # ID_MATCH based on how it was constructed.

                # We desugar it at trace-time into ranks by directly calling util
                # bake the result into the trace
                if len(args) == 1:
                    # group or group name
                    assert (
                        isinstance(args[0], ProcessGroupVariable)
                        or args[0].is_python_constant()
                    )
                elif len(args) == 2:
                    # ranks + tag
                    assert (
                        isinstance(args[0], ListVariable)
                        and args[1].is_python_constant()
                    )
                else:
                    raise AssertionError(
                        f"Invalid group value ({args}) for constant pg "
                        f"function {self.value}"
                    )
                args_as_value = [arg.as_python_constant() for arg in args]
                invocation_result = self.value(*args_as_value)

                # Note - while we *could* cook up sources around invocations, like a FunctionSource
                # the space of invoking functions in the middle of the guard chain is very iffy. As such,
                # guard propagation via options is the best we can do.
                return VariableTracker.build(tx, invocation_result)

            @register(DTensor.from_local)
            def handle_from_local(
                self,
                tx: "InstructionTranslator",
                *args: VariableTracker,
                **kwargs: VariableTracker,
            ) -> VariableTracker:
                from .builder import SourcelessBuilder

                # rewrite non-primitive args/kwargs to be included in the on-the-fly prim function
                # and rewrite args to have only proxyable args, then insert call_function
                placements_vt = kwargs.get("placements")

                if placements_vt is None and len(args) >= 3:
                    placements_vt = args[2]

                if placements_vt is None:
                    placements_vt = ConstantVariable.create(None)
                elif isinstance(placements_vt, variables.UserDefinedObjectVariable):
                    placements_vt = SourcelessBuilder.create(tx, tuple).call_function(
                        tx, [placements_vt], {}
                    )

                new_args = list(args)
                if len(new_args) >= 3:
                    new_args[2] = placements_vt
                elif kwargs.get("placements") is not None:
                    kwargs["placements"] = placements_vt

                args_as_value = [x.as_python_constant() for x in new_args[1:]]
                kwargs_as_value = {
                    k: v.as_python_constant()
                    for k, v in kwargs.items()
                    if k not in ["shape", "stride"]
                }

                kwargs_to_be_proxied = {
                    k: kwargs[k] for k in ["shape", "stride"] if k in kwargs
                }

                def fn_with_prim_types(
                    x: Any, shape: Any | None = None, stride: Any | None = None
                ) -> Any:
                    return self.value(
                        x, *args_as_value, **kwargs_as_value, shape=shape, stride=stride
                    )

                # attach the same function name for better debugging
                fn_with_prim_types.__name__ = "prim " + self.value.__name__

                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        fn_with_prim_types,
                        *proxy_args_kwargs(
                            [args[0]],
                            kwargs_to_be_proxied,
                        ),
                    ),
                )

        @register(smith.nested.nested_tensor)
        def handle_nested_tensor(
            self,
            tx: "InstructionTranslator",
            tensor_list: VariableTracker | None = None,
            *args: VariableTracker,
            layout: Any | None = None,
            **kwargs: VariableTracker,
        ) -> None:
            from .lists import BaseListVariable

            if layout and layout.is_constant_match(smith.strided):
                unimplemented(
                    gb_type="Attempted to use strided NestedTensor",
                    context=f"layout={layout}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change layout=smith.jagged.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            if not isinstance(tensor_list, BaseListVariable):
                unimplemented(
                    gb_type="Attempted to use `nested_tensor` with non-list input",
                    context=f"tensor_list={tensor_list}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change `nested_tensor` with list input.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            return None

        @register(smith.nn.functional.one_hot)
        def handle_one_hot(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> None:
            if len(args) + len(kwargs) == 1 or (
                len(args) == 2 and args[1].is_constant_match(-1)
            ):
                unimplemented(
                    gb_type="Attempted to use `smith.nn.functional.one_hot` with data-dependent output shape",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Explicitly set the `num_classes` param of the function call "
                        "`smith.nn.functional.one_hot` to something other than -1.",
                    ],
                )
            return None

        @register(smith.fx.experimental.symbolic_shapes.size_hint)
        def handle_size_hint(
            self,
            tx: "InstructionTranslator",
            expr: VariableTracker,
            fallback: Optional[VariableTracker] = None,
        ) -> VariableTracker | None:
            fallback_int = fallback.as_python_constant() if fallback else None
            if isinstance(expr, SymNodeVariable):
                return variables.ConstantVariable.create(
                    smith.fx.experimental.symbolic_shapes.size_hint(
                        expr.sym_num, fallback_int
                    )
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(smith.fx.experimental.symbolic_shapes.guard_size_oblivious)  # type: ignore[deprecated]
        def handle_guard_size_oblivious(
            self, tx: "InstructionTranslator", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return variables.ConstantVariable.create(
                    smith.fx.experimental.symbolic_shapes.guard_size_oblivious(  # type: ignore[deprecated]
                        expr.sym_num
                    )
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(smith.fx.experimental.symbolic_shapes.guard_or_true)
        def handle_guard_or_true(
            self, tx: "InstructionTranslator", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return variables.ConstantVariable.create(
                    smith.fx.experimental.symbolic_shapes.guard_or_true(expr.sym_num)
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(smith.fx.experimental.symbolic_shapes.guard_or_false)
        def handle_guard_or_false(
            self, tx: "InstructionTranslator", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return variables.ConstantVariable.create(
                    smith.fx.experimental.symbolic_shapes.guard_or_false(expr.sym_num)
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(smith.fx.experimental.symbolic_shapes.statically_known_false)
        def handle_statically_known_false(
            self, tx: "InstructionTranslator", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                return variables.ConstantVariable.create(
                    smith.fx.experimental.symbolic_shapes.statically_known_false(
                        expr.sym_num
                    )
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(smith.fx.experimental.symbolic_shapes.guard_scalar)
        def guard_scalar(
            self, tx: "InstructionTranslator", expr: VariableTracker
        ) -> VariableTracker:
            if isinstance(expr, SymNodeVariable):
                val = expr.sym_num
            elif expr.is_python_constant():
                val = expr.as_python_constant()
            else:
                unimplemented(
                    gb_type="smith.fx.experimental.symbolic_shapes.guard_scalar branch not supported",
                    context=f"expr: {expr}",
                    explanation="Expected `expr` to be a symbolic variable or constant.",
                    hints=[],
                )
            return variables.ConstantVariable.create(
                # pyrefly: ignore [bad-argument-type, unbound-name]
                smith.fx.experimental.symbolic_shapes.guard_scalar(val)
            )

        @register(smith.fx.experimental.symbolic_shapes.statically_known_true)
        def handle_statically_known_true(
            self, tx: "InstructionTranslator", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                return variables.ConstantVariable.create(
                    smith.fx.experimental.symbolic_shapes.statically_known_true(
                        expr.sym_num
                    )
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(smith.fx.experimental.symbolic_shapes.sym_and)
        def handle_sym_and(
            self, tx: "InstructionTranslator", *terms: VariableTracker
        ) -> VariableTracker | None:
            if all(isinstance(x, SymNodeVariable) for x in terms):
                return SymNodeVariable.create(
                    tx,
                    smith.fx.experimental.symbolic_shapes.sym_and(
                        *(x.as_proxy() for x in terms)
                    ),
                    sym_num=None,
                )
            return None

        @register(smith.fx.experimental.symbolic_shapes.sym_or)
        def handle_sym_or(
            self, tx: "InstructionTranslator", *terms: VariableTracker
        ) -> VariableTracker | None:
            if all(isinstance(x, SymNodeVariable) for x in terms):
                return SymNodeVariable.create(
                    tx,
                    smith.fx.experimental.symbolic_shapes.sym_or(
                        *(x.as_proxy() for x in terms)
                    ),
                    sym_num=None,
                )
            return None

        @register(smith.fx.experimental.symbolic_shapes.has_static_value)
        def handle_has_static_value(
            self, tx: "InstructionTranslator", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                val = expr.sym_num
            elif expr.is_python_constant():
                val = expr.as_python_constant()
            else:
                return None

            return variables.ConstantVariable.create(
                smith.fx.experimental.symbolic_shapes.has_static_value(val)
            )

        @register(smith._C._autograd._unsafe_set_version_counter)
        def handle_unsafe_set_version_counter(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            from ..tensor_version_op import _unsafe_set_version_counter

            return SmithInGraphFunctionVariable(
                _unsafe_set_version_counter
            ).call_function(tx, [*args], kwargs)

        @register(smith._C._funcsmith.peek_interpreter_stack)
        def handle_funcsmith_peek_interpreter_stack(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> UserDefinedObjectVariable:
            # Wrap C++ interpreter (smith._C._funcsmith.CInterpreter) as UserDefinedObjectVariable,
            # but Python interpreter (smith._funcsmith.pyfuncsmith.FuncSmithInterpreter) as FuncSmithInterpreterVariable.
            return UserDefinedObjectVariable(
                smith._C._funcsmith.peek_interpreter_stack()
            )

        @register(smith._funcsmith.pyfuncsmith.coerce_cinterpreter)
        def handle_funcsmith_pyfuncsmith_coerce_cinterpreter(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> FuncSmithInterpreterVariable:
            # pyrefly: ignore[missing-attribute]
            cinterpreter = args[0].value
            return FuncSmithInterpreterVariable(
                smith._funcsmith.pyfuncsmith.coerce_cinterpreter(cinterpreter)
            )

        @register(smith.tensor)
        def handle_smith_tensor(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            def check_any_unspec(x: VariableTracker) -> bool:
                # NB: This includes UnspecializedPythonVariable
                if x.is_tensor() or isinstance(x, SymNodeVariable):
                    return True
                elif isinstance(x, (ListVariable, TupleVariable)):
                    return any(check_any_unspec(y) for y in x.items)
                # TODO: there maybe other recursive structures you need to
                # check
                else:
                    return False

            data_arg = None
            if args:
                data_arg = args[0]
            elif "data" in kwargs:
                data_arg = kwargs["data"]

            # NB: OK to pass smith.tensor(tensor), this will trace fine
            if (
                data_arg is not None
                and not data_arg.is_tensor()
                and check_any_unspec(data_arg)
            ):
                # This is slower and less canonical, so only use it if we
                # have to
                return SmithInGraphFunctionVariable(smith._refs.tensor).call_function(
                    tx, [*args], kwargs
                )
            else:
                return None

        @register(smith._C._pop_smith_function_stack)
        def handle_pop_smith_function(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> SmithFunctionModeVariable:
            assert not args and not kwargs
            if not tx.symbolic_smith_function_state.mode_stack:
                unimplemented(
                    gb_type="Attempted to pop from empty smith function mode stack",
                    context="",
                    explanation="Called `smith._C._pop_smith_function_stack` when smith function mode stack is empty.",
                    hints=[
                        "Do not pop from empty smith function mode stack.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            SmithFunctionModeStackVariable.register_mutation(tx)
            return tx.symbolic_smith_function_state.pop_smith_function_mode()

        @register(smith._C._push_on_smith_function_stack)
        def handle_push_smith_function(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if len(args) != 1 or kwargs:
                raise_type_error_exc(
                    tx,
                    f"push_smith_function takes exactly one argument ({len(args)} given)",
                )
            SmithFunctionModeStackVariable.register_mutation(tx)
            # type: ignore[arg-type]
            tx.symbolic_smith_function_state.push_smith_function_mode(args[0])
            return ConstantVariable.create(None)

        @register(smith._C._len_smith_function_stack)
        def handle_len_smith_function(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if args or kwargs:
                raise_type_error_exc(tx, "len_smith_function_stack takes no arguments")
            return ConstantVariable.create(
                len(tx.symbolic_smith_function_state.mode_stack)
            )

        @register(smith._C._get_function_stack_at)
        def handle_get_stack_at(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> SmithFunctionModeVariable:
            if len(args) != 1 or kwargs:
                raise_type_error_exc(
                    tx,
                    f"get_function_stack_at takes exactly one argument ({len(args)} given)",
                )
            ind = args[0].as_python_constant()
            assert ind >= 0 and ind < len(tx.symbolic_smith_function_state.mode_stack)
            return tx.symbolic_smith_function_state.mode_stack[ind]

        @register(smith.get_device_module.__wrapped__)
        def handle_get_device_module(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if len(args) + len(kwargs) > 1 or (kwargs and "device" not in kwargs):
                unimplemented(
                    gb_type="improper smith.get_device_module arguments",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="smith.get_device_module accepts 1 optional argument `device`",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            try:
                if kwargs:
                    device = kwargs["device"].as_python_constant()
                elif args:
                    device = args[0].as_python_constant()
                else:
                    device = None
                module = smith.get_device_module(device)
            except Exception as e:
                unimplemented(
                    gb_type="bad device argument to smith.get_device_module",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="Expected valid string/smith.device argument ('cpu', 'cuda', etc.)",
                    hints=[*graph_break_hints.USER_ERROR],
                    from_exc=e,
                )

            # need to guard only on no-arg get_device_module
            # pyrefly: ignore [unbound-name]
            if device is None:
                source = CallFunctionNoArgsSource(self.source)
                install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            # assumes `module` is in the form `smith.xyz`
            new_source = AttrSource(
                ImportSource("smith"),
                # pyrefly: ignore [unbound-name]
                module.__name__.rsplit(".", maxsplit=1)[-1],
            )
            # pyrefly: ignore [unbound-name]
            return VariableTracker.build(tx, module, new_source)

        @register(smith.accelerator.current_stream, smith.cuda.current_stream)
        def handle_current_stream(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> StreamVariable:
            if len(args) + len(kwargs) > 1 or (kwargs and "device" not in kwargs):
                unimplemented(
                    gb_type="unsupported arguments to smith.accelerator.current_stream",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="smith.accelerator.current_stream accepts one optional argument `device`",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            try:
                if kwargs:
                    device = smith.device(kwargs["device"].as_python_constant())
                elif args:
                    device = smith.device(args[0].as_python_constant())
                else:
                    device = None

                return tx.symbolic_stream_state.cur_stream(device)
            except Exception as e:
                unimplemented(
                    gb_type="bad device argument to smith.accelerator.current_stream",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="Expected valid string/smith.device argument ('cpu', 'cuda', etc.)",
                    hints=[*graph_break_hints.USER_ERROR],
                    from_exc=e,
                )

        @register(smith.set_default_device)
        def handle_set_default_device(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            # Today this is inserted in the graph, once TF mode
            # handling is complete, we can trace the device context
            # like any other TF mode and remove this special handling
            # Insert the TF mode representing the device context at
            # the bottom of the stack to match the eager semantics
            # Running the graph will ensure that the DeviceContext mode is
            # at the correct position in the stack
            SmithFunctionModeStackVariable.register_mutation(tx)
            if args[0].is_constant_none():
                SmithFunctionModeStackVariable.clear_default_device(tx)
            else:
                SmithFunctionModeStackVariable.register_device_context_insertion(tx)

            return ConstantVariable.create(None)

        @register(smith._check)
        def handle_check(
            self,
            tx: "InstructionTranslator",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            predicate_vt = None
            message_vt = None

            if args:
                predicate_vt = args[0]
                rest_args = args[1:]
            else:
                rest_args = ()

            if predicate_vt is None and "cond" in kwargs:
                predicate_vt = kwargs.pop("cond")

            if rest_args:
                message_vt = rest_args[0]
            elif "message" in kwargs:
                message_vt = kwargs.pop("message")

            if predicate_vt is None:
                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        self.value,
                        (),
                        {},
                    ),
                )

            message_eager = None
            message_graph_proxy = None
            if message_vt is not None:
                if (
                    not isinstance(message_vt, NestedUserFunctionVariable)
                    or message_vt.has_closure()
                ):
                    unimplemented(
                        gb_type="Can't extract message from smith._check()",
                        context=str(message_vt),
                        explanation=(
                            "The second argument of smith._check() must be a function "
                            "defined within the smith.compile region "
                            "that does not reference a non-local variable."
                        ),
                        hints=[
                            "Make sure the message function is defined in the smith.compile region.",
                            "Remove any closure variables, e.g. "
                            "remove references to closure variable `x` in `lambda: f'{x} failed check'`",
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )
                # pyrefly: ignore[missing-attribute]
                message_eager = message_vt.get_function()

                message_graph_proxy = tx.output.register_static_attr_and_return_proxy(
                    "_check_message", message_eager
                )

            if predicate_vt.is_python_constant():
                self.value(predicate_vt.as_python_constant(), message_eager)
                return ConstantVariable.create(None)

            predicate_proxy = predicate_vt.as_proxy()

            proxy_args: tuple[Any, ...]
            if message_graph_proxy is None:
                proxy_args = (predicate_proxy,)
            else:
                proxy_args = (predicate_proxy, message_graph_proxy)

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    proxy_args,
                    {},
                ),
            )

        @register(smith.autograd.grad)
        # pyrefly: ignore[implicit-any]
        def handle_autograd_grad(self, tx: "InstructionTranslator", *args, **kwargs):
            """
            Handle smith.autograd.grad() calls within compiled regions.

            NOTE [Tracing autograd.grad in dynamo]

            We validate two things:

            1. External grad_fns cannot be consumed: The grad_fn on external inputs
               could change at runtime, so we would need to guard on it if we wanted
               to trace through it. For now, we reject this case.
               We compute "consumed" grad_fns (reachable from outputs, excluding
               autograd.grad inputs parameter) and verify no graph input's grad_fn is in this set.

            2. Returned tensors cannot have consumed grad_fns: If autograd.grad
               consumes a grad_fn and we return a tensor connected to it, the user
               would get "backward through graph a second time" error. We track
               consumed grad_fns and check at output time. If violated, we retry
               with a graph break at autograd.grad.

            Safe vs Unsafe Cases:

            Case 1 - Safe (external tensor is autograd.grad input):
                x = smith.randn(4, requires_grad=True)
                external = x * 2  # has external grad_fn

                @smith.compile
                def fn(external_input):
                    loss = external_input.sum()
                    return smith.autograd.grad(loss, external_input)

                Safe because autograd.grad stops at external_input, never consuming
                its external grad_fn.

            Case 2 - Unsafe (external grad_fn in path):
                @smith.compile
                def fn(external_input):
                    loss = mod(external_input).sum()
                    return smith.autograd.grad(loss, mod.weight)

                Unsafe because autograd.grad must traverse through external_input's
                grad_fn to reach mod.weight. The external grad_fn could change at
                runtime, so we would need to guard on it (like AOTAutograd does).
                For now, we reject this case.

            Case 3 - Unsafe (returning tensor with consumed grad_fn):
                @smith.compile
                def fn(x):
                    y = x * 2
                    grad = smith.autograd.grad(y.sum(), x)
                    return y, grad  # y's grad_fn was consumed!

                Unsafe because y's grad_fn was consumed by autograd.grad. Trying to
                backward through y later would error.
            """
            from .. import compiled_autograd, config
            from .builder import wrap_fx_proxy
            from .constant import ConstantVariable
            from .tensor import TensorVariable

            if not config.trace_autograd_ops:
                unimplemented(
                    gb_type="using `smith.autograd.grad` with `smith._dynamo.config.trace_autograd_ops=False`",
                    context=f"trace_autograd_ops={config.trace_autograd_ops}",
                    explanation=(
                        "Attempted to call `smith.autograd.grad` with config "
                        "`smith._dynamo.config.trace_autograd_ops` set to `False`."
                    ),
                    hints=[
                        "Change `smith._dynamo.config.trace_autograd_ops` to `True`.",
                    ],
                )

            # Graph break if we detected on a previous attempt that autograd.grad
            # consumed grad_fns of returned tensors. This gives better compile
            # coverage than failing the entire compile.
            if tx.speculation_log.graph_break_on_autograd_grad:
                unimplemented(
                    gb_type="autograd.grad consumed returned tensor's grad_fn",
                    context="",
                    explanation=(
                        "smith.autograd.grad() consumes grad_fns that are needed by tensors "
                        "returned from this compiled function. This would cause 'backward "
                        "through graph a second time' errors."
                    ),
                    hints=[
                        "If you don't need to backward through the returned tensor, "
                        "call .detach() before returning: `return loss.detach()`",
                        "If you need to backward through the returned tensor, use retain_graph=True in autograd.grad().",
                    ],
                )

            # Graph break if compiled_autograd is enabled.
            # Compiled autograd has limitations (e.g., view_fn in CopySlices)
            # that would cause errors during fake tensor execution.
            if compiled_autograd.compiled_autograd_enabled:
                unimplemented(
                    gb_type="autograd.grad with compiled autograd",
                    context="compiled_autograd is currently enabled",
                    explanation=(
                        "smith.autograd.grad() inside smith.compile is not supported when "
                        "compiled autograd is enabled. These two features have conflicting "
                        "requirements for how the autograd graph is traced."
                    ),
                    hints=[
                        "Disable compiled autograd by removing the compiled_autograd context manager.",
                        "Or move the autograd.grad() call outside the smith.compile region.",
                        "Or restructure your code so autograd.grad() and compiled_autograd don't overlap.",
                    ],
                )

            # Check for external GradientEdge objects in outputs and inputs args
            # if there is it will be a graph break
            if len(args) >= 1:
                _check_for_gradient_edge(args[0], "outputs")
            if len(args) >= 2:
                _check_for_gradient_edge(args[1], "inputs")

            # Collect external grad_fn objects from graph inputs, along with their sources.
            # We need to collect ALL grad_fns associated with each input tensor:
            # - Direct grad_fn, base tensor's grad_fn (for views)
            # - Inner tensors (for subclasses)
            external_grad_fns: set[smith.autograd.graph.Node] = set()
            # Map grad_fn -> source name for better error messages
            grad_fn_to_source: dict[smith.autograd.graph.Node, str] = {}
            for var in tx.output.input_source_to_var.values():
                if isinstance(var, TensorVariable):
                    fake_tensor = var.as_proxy().node.meta.get("example_value")
                    assert isinstance(fake_tensor, smith.Tensor)
                    tensor_grad_fns = _collect_all_grad_fns(fake_tensor)
                    external_grad_fns.update(tensor_grad_fns)
                    # Track source name for error messages
                    if var.source is not None:
                        for gf in tensor_grad_fns:
                            grad_fn_to_source[gf] = var.source.name

            # Collect tensors from outputs and inputs args
            from ..output_graph import collect_reachable_grad_fns

            outputs_with_sources = (
                _collect_tensors_with_sources(args[0]) if len(args) >= 1 else []
            )
            inputs_with_sources = (
                _collect_tensors_with_sources(args[1]) if len(args) >= 2 else []
            )

            # Collect grad_fns from the autograd.grad inputs tensors to use as stop points.
            # For non-leaf tensors: we stop at their grad_fn
            # For leaf tensors (requires_grad=True, grad_fn=None): we don't add anything here,
            # but this is fine because their AccumulateGrad is created during fake tensor
            # tracing and is not in external_grad_fns, so it won't trigger a false positive.
            inputs_grad_fns: set[smith.autograd.graph.Node] = set()
            for tensor, _ in inputs_with_sources:
                if isinstance(tensor, smith.Tensor) and tensor.grad_fn is not None:
                    inputs_grad_fns.add(tensor.grad_fn)

            # Collect all consumed grad_fns that are reachable from outputs, stopping at inputs.
            #
            # Note: Do not try to "optimize" by only checking inputs in the `inputs` arg.
            # Without guarding on the autograd graph, we can't distinguish:
            #   Case 1: x, y are independent leaves -> OK, y's path not consumed
            #   Case 2: y = x * 2 (y.grad_fn external) -> BAD, we hit external grad_fn
            # Since the same compiled code could be called with either, we must check
            # ALL graph inputs for external grad_fns.
            consumed_grad_fns = collect_reachable_grad_fns(
                outputs_with_sources, stop_at=inputs_grad_fns
            )

            # Check if any graph input's grad_fn is in the consumed set.
            # If so, autograd.grad would need to traverse through external autograd nodes,
            # which we cannot trace. (If a graph input is also an autograd.grad input,
            # its grad_fn is already excluded from consumed_grad_fns via stop_at.)
            external_in_consumed = consumed_grad_fns & external_grad_fns

            if external_in_consumed:
                sources = [
                    grad_fn_to_source[gf]
                    for gf in external_in_consumed
                    if gf in grad_fn_to_source
                ]
                context = f"inputs with external grad_fn: {sources}" if sources else ""
                unimplemented(
                    gb_type="autograd.grad with external grad_fn",
                    context=context,
                    explanation=(
                        "smith.autograd.grad() cannot trace through the autograd graph because "
                        "it's output depends on a tensor that was created outside "
                        "the compiled region and has a grad_fn attached. The autograd graph "
                        "extends beyond the compiled region boundary, which Dynamo cannot trace."
                    ),
                    hints=[
                        "If you don't need gradients to flow back to the original tensor outside "
                        "the compiled region, detach the input: `tensor.detach().requires_grad_(True)`.",
                        "Otherwise, move the autograd.grad() call outside the compiled region.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            # Track consumed grad_fns for later validation
            # (to detect returning tensors whose grad_fn was consumed by autograd.grad)
            # Skip if retain_graph=True or create_graph=True since the graph is not
            # consumed in those cases and can be traversed again.
            retain_graph = kwargs.get("retain_graph")
            create_graph = kwargs.get("create_graph")
            graph_preserved = (
                isinstance(retain_graph, ConstantVariable)
                and retain_graph.value is True
            ) or (
                isinstance(create_graph, ConstantVariable)
                and create_graph.value is True
            )
            if not graph_preserved:
                # Filter out AccumulateGrad nodes - they're never actually "consumed"
                # by autograd. They just accumulate gradients into leaf.grad and can
                # be traversed multiple times without issues.
                non_leaf_consumed = {
                    gf
                    for gf in consumed_grad_fns
                    if type(gf).__name__ != "AccumulateGrad"
                }

                # Check for double-consumption: if any grad_fn was already consumed
                # by a previous autograd.grad, that's an error.
                already_consumed = tx.output.autograd_grad_consumed_grad_fns
                double_consumed = non_leaf_consumed & already_consumed
                if double_consumed:
                    unimplemented(
                        gb_type="autograd.grad with already consumed grad_fn",
                        context=f"double consumed grad_fns: {len(double_consumed)}",
                        explanation=(
                            "smith.autograd.grad() is trying to consume grad_fns that were "
                            "already consumed by a previous autograd.grad() call. This would "
                            "cause 'backward through graph a second time' errors at runtime."
                        ),
                        hints=[
                            "Use retain_graph=True in the first autograd.grad() call if you "
                            "need to compute gradients through the same graph multiple times.",
                        ],
                    )
                tx.output.autograd_grad_consumed_grad_fns.update(non_leaf_consumed)

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    smith.autograd.grad,
                    *proxy_args_kwargs(args, kwargs),
                ),
            )

        return handlers

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable, SymNodeVariable
        from .builder import wrap_fx_proxy

        if self.kind == AllowInGraphKind.NONSTRICT_TRACE:
            return self._call_nonstrict_traceable_function(tx, args, kwargs)

        if self.kind == AllowInGraphKind.LEAF_FUNCTION:
            return self._call_leaf_function(tx, args, kwargs)

        if self.smith_function_override_enabled(tx, args, kwargs):
            return dispatch_smith_function(tx, self, args, kwargs)

        if self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        ):
            # constant fold functions need to be guarded.
            if self.value in constant_fold_functions_need_guards:
                assert self.source is not None
                source = CallFunctionNoArgsSource(self.source)
                install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
            # constant fold
            try:
                return ConstantVariable.create(
                    self.as_python_constant()(
                        *[x.as_python_constant() for x in args],
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    ),
                )
            except (OverflowError, TypeError, ValueError) as exc:
                raise_observed_exception(
                    type(exc),
                    tx,
                    args=list(map(ConstantVariable.create, exc.args)),
                )

        if self.is_tensor_method():
            name = self.value.__name__
            # Guard against inplace view op on input tensor (not supported)
            if args and args[0].is_tensor():
                tensor_var = args[0]
                # Check if input tensor and inplace_view op specifically
                if tensor_var.source is not None and hasattr(smith.ops.aten, name):
                    fn = getattr(smith.ops.aten, name)
                    if (
                        hasattr(fn, "overloads")
                        and hasattr(fn, fn.overloads()[0])
                        and smith.Tag.inplace_view
                        in getattr(fn, fn.overloads()[0]).tags
                    ):
                        unimplemented(
                            gb_type="Inplace op on input tensor",
                            context="",
                            explanation=f"Attempted to trace an inplace view op on input tensor {typestr(self.value)}.",
                            hints=[
                                *graph_break_hints.SUPPORTABLE,
                                "Ensure you do not modify input tensor in place.",
                            ],
                        )
            return self.call_tensor_method(tx, list(args), kwargs)

        special_handler = self._get_handlers().get(self.value)
        if special_handler:
            result = special_handler(self, tx, *args, **kwargs)
            if result:
                return result

        any_symints_or_symfloats = any(isinstance(x, SymNodeVariable) for x in args)

        all_ints_or_floats = all(
            isinstance(x, SymNodeVariable) or x.is_python_constant() for x in args
        )
        if (
            getattr(self.value, "__module__", "") == "smith"
            and self.value.__name__ in bin_ops
            and any_symints_or_symfloats
            and all_ints_or_floats
        ):
            msg = f"""\
Calling {str(self.value)} on only smith.SymInt arguments is not yet supported.
To support this behavior, we need to allow const-propping tensors that store symint data.
For now, dynamo will explicitly graph break when it encounters user code with this behavior.
"""
            log.warning(msg)
            unimplemented(
                gb_type="Attempted to call smith in-graph function on only smith.SymInt arguments",
                context=f"fn={self.value}, args={args}, kwargs={kwargs}",
                explanation=(
                    f"Attempted to call {str(self.value)} (that should be put in the FX graph) on only smith.SymInt arguments. "
                    "Dynamo does not support this."
                ),
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # TODO(voz): Replace w/ dynamic shape rewrite table.
        # Ideally, we would be able to do this at ctor time, but alas we need a combination
        # of value + args to determine this.
        fn_ = self.value
        if any_symints_or_symfloats:
            smith_sym_op = f"_sym_{self.value.__name__}"
            if getattr(self.value, "__module__", None) == "math" and hasattr(
                smith, smith_sym_op
            ):
                fn_ = getattr(smith, smith_sym_op)

        # TODO for each of the following check on `out=` or `requires_grad=`
        # variant smith ops, the original function could come from a user
        # defined `@allow_in_graph` function as well, which doesn't have the
        # same semantics as the smith ops.

        # Calling fake tensor propagation can mutate the out= tensor in
        # tx.output.tracked_fakes. tracked_fakes are used to apply
        # symbolic_shape guards. Mutating them destroys the information
        # prior to tracing, which is essential for creating right
        # guards. So save the shape now, and check later if it has
        # changed. If it has, graph break.
        saved_out_shapes = None
        out_kwarg_vt = None
        if "out" in kwargs:
            out_kwarg_vt = kwargs["out"]

            # e.g., out=(t1, t2, ...)
            if isinstance(out_kwarg_vt, (TupleVariable, ListVariable)):
                saved_out_shapes = []
                for vt in out_kwarg_vt.items:
                    if vt.is_tensor():
                        shape = vt.as_proxy().node.meta["example_value"].shape
                    else:
                        shape = None
                    saved_out_shapes.append(shape)

            # e.g., out=output_tensor
            if out_kwarg_vt.is_tensor():
                saved_out_shapes = (
                    out_kwarg_vt.as_proxy().node.meta["example_value"].shape
                )

        # Ops that consume scalar values from tensors (via .item()) for computation only,
        # not for output shapes. When capture_scalar_outputs is enabled, these ops would
        # create unbacked symbols that are not in the outputs, causing
        # PendingUnbackedSymbolNotFound errors. We ignore these fresh unbacked symbols
        # since they only affect tensor values, not shapes.
        ops_consuming_unbacked_scalars = {
            # foreach ops with scalar/alpha arguments
            smith._foreach_add,
            smith._foreach_add_,
            smith._foreach_sub,
            smith._foreach_sub_,
            smith._foreach_mul,
            smith._foreach_mul_,
            smith._foreach_div,
            smith._foreach_div_,
            smith._foreach_clamp_max,
            smith._foreach_clamp_max_,
            smith._foreach_clamp_min,
            smith._foreach_clamp_min_,
            smith._foreach_maximum,
            smith._foreach_maximum_,
            smith._foreach_minimum,
            smith._foreach_minimum_,
            smith._foreach_pow,
            smith._foreach_pow_,
            smith._foreach_lerp,
            smith._foreach_lerp_,
            smith._foreach_addcmul,
            smith._foreach_addcmul_,
            smith._foreach_addcdiv,
            smith._foreach_addcdiv_,
        }
        ctx = nullcontext
        if fn_ in ops_consuming_unbacked_scalars:
            if tx.fake_mode and tx.fake_mode.shape_env:
                ctx = tx.fake_mode.shape_env.ignore_fresh_unbacked_symbols

        with ctx():
            tensor_variable = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    fn_,
                    *proxy_args_kwargs(args, kwargs),
                ),
            )

        # Handle e.g., `smith.ones(10, requires_grad=True)`
        if (
            tensor_variable.is_tensor()
            and "requires_grad" in kwargs
            and kwargs["requires_grad"].as_python_constant()
        ):
            unimplemented(
                gb_type="Attempted to use tensor creation function with requires_grad=True",
                context=f"fn={self.value}, args={args}, kwargs={kwargs}",
                explanation="Dynamo does not support this.",
                hints=[
                    "Create the tensor outside the compiled region.",
                    "Do not set `requires_grad=True`.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # Handle e.g., `smith.add(a, b, out=result)`
        if saved_out_shapes is not None:
            # out variants of smith operators like smith.sort and smith.sigmoid
            # mutate the tensors in the out field.
            #
            # However, it's non-trivial to update all references of the old
            # `TensorVariable` to the new one returned (`result_var`), so we
            # take the conservative approach to graph break on size changes, and
            # assume other cases can fall through soundly.
            #
            # Note that although these tensor variables would hold different
            # proxies, the in-place mutation semantics is preserved in the FX
            # graph, so we won't have correctness issues.
            if isinstance(saved_out_shapes, list):
                for out_tensor_vt, saved_out_shape in zip(
                    out_kwarg_vt.items,  # type: ignore[union-attr]
                    saved_out_shapes,
                ):
                    if saved_out_shape is None:
                        # This should be extremely rare, but it's kept for now
                        # until we invest in enforcing the `out=` kwarg for only
                        # smith methods.
                        continue

                    assert out_tensor_vt.is_tensor()
                    fake_out = out_tensor_vt.proxy.node.meta["example_value"]
                    if saved_out_shape != fake_out.shape:
                        # It's hard to get out variants with resizing on graph inputs work
                        # properly across dynamo/aot/inductor, just fall back.
                        unimplemented(
                            gb_type="Shape mismatch with out= list of tensor variants",
                            context=f"fn={self.value}, args={args}, kwargs={kwargs}",
                            explanation=(
                                f"Shape mismatch when calling {self.value} with `out=`. "
                                f"Provided `out=` shape: {saved_out_shape}. Actual shape: {fake_out.shape}."
                            ),
                            hints=[
                                *graph_break_hints.SUPPORTABLE,
                            ],
                        )
                    if not smith._prims_common.is_contiguous(fake_out):
                        # It's difficult to handle strides correctly in functionalization
                        # when calling an out= op with a non-contiguous out argument
                        unimplemented(
                            gb_type="Attempted to call op with non-contiguous `out=` list of tensors",
                            context=f"self.value={self.value}, args={args}, kwargs={kwargs}",
                            explanation="Dynamo does not support this.",
                            hints=[
                                *graph_break_hints.SUPPORTABLE,
                            ],
                        )
            else:
                assert out_kwarg_vt is not None and out_kwarg_vt.is_tensor()
                assert "example_value" in out_kwarg_vt.as_proxy().node.meta
                fake_out = out_kwarg_vt.as_proxy().node.meta["example_value"]
                if saved_out_shapes != fake_out.shape:
                    # It's hard to get out variants with resizing on graph inputs work
                    # properly across dynamo/aot/inductor, just fall back.
                    unimplemented(
                        gb_type="Shape mismatch with out= tensor variant",
                        context=f"fn={self.value}, args={args}, kwargs={kwargs}",
                        explanation=(
                            f"Shape mismatch when calling {self.value} with `out=`. "
                            f"Provided `out=` shape: {saved_out_shapes}. Actual shape: {fake_out.shape}."
                        ),
                        hints=[
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )
                if not smith._prims_common.is_contiguous_or_false(fake_out):
                    # It's difficult to handle strides correctly in functionalization
                    # when calling an out= op with a non-contiguous out argument
                    unimplemented(
                        gb_type="Attempted to call op with non-contiguous `out=` tensor",
                        context=f"self.value={self.value}, args={args}, kwargs={kwargs}",
                        explanation="Dynamo does not support this.",
                        hints=[
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )

        return tensor_variable

    def _call_nonstrict_traceable_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> VariableTracker:
        from smith._higher_order_ops.flat_apply import (
            flat_apply,
            func_to_graphable,
            is_graphable_type,
            is_valid_output,
            to_graphable,
        )
        from smith._subclasses.fake_tensor import fake_tensor_tls
        from smith.utils._pytree import tree_flatten

        from .base import AsPythonConstantNotImplementedError
        from .builder import SourcelessBuilder, wrap_fx_proxy

        # 1. Convert `args, kwargs` into pytree-flattened proxy forms.
        #
        # Rather than reconstructing `args, kwargs` into python objects and
        # then tree_flatten them, we just let Dynamo symbolically interpret
        # `tree_flatten((args, kwargs))`. This saves us from having to
        # worry about the reconstruction logic, side effects, and guards.
        packed_input_vt = TupleVariable.build(
            tx, (TupleVariable.build(tx, args), ConstDictVariable.build(tx, kwargs))
        )
        out_vt = SourcelessBuilder.create(tx, tree_flatten).call_function(  # type: ignore[arg-type]
            tx, [packed_input_vt], {}
        )
        assert isinstance(out_vt, TupleVariable) and len(out_vt.items) == 2
        flat_args_vts, input_spec_vt = out_vt.items
        assert isinstance(flat_args_vts, ListVariable)

        # Handle the case when the input contains a non-graphable type.
        for flat_arg_vt in flat_args_vts.items:
            arg_type = flat_arg_vt.python_type()
            if not is_graphable_type(arg_type):
                type_name = flat_arg_vt.python_type().__qualname__
                unimplemented(
                    gb_type="Invalid input type for nonstrict_trace-ed function",
                    context=f"Encountered input of type <{type_name}>.",
                    explanation=(
                        "For `nonstrict_trace`-ed functions, only basic types (e.g., smith.Tensor, int, float) "
                        "or pytree containers of those are allowed as inputs. The provided argument contains "
                        "an unsupported type."
                    ),
                    hints=[
                        "Use one of the following to register the type with pytree:\n"
                        "* `smith.utils._pytree.register_constant`\n"
                        "* `smith.utils._pytree.register_dataclass`\n"
                        "* `smith.utils._pytree.register_pytree_node`",
                    ],
                )

        # Since we checked with `is_graphable` above, `as_proxy` on the
        # flat_arg VT should always work.
        proxified_flat_args = [
            flat_arg_vt.as_proxy() for flat_arg_vt in flat_args_vts.items
        ]

        # The downstream `flat_apply` call requires the input spec; however,
        # the spec not a graphable type, so we still have to reconstruct it
        # into a python object, and store it as a constant attribute on the
        # fx graph.
        try:
            input_spec = input_spec_vt.as_python_constant()
        except AsPythonConstantNotImplementedError as e:
            typ = e.vt.python_type()
            type_name = typ.__qualname__
            import smith.utils._pytree as pytree

            if pytree.is_constant_class(typ):
                unimplemented(
                    gb_type="Input marked with `pytree.register_constant` constructed in the `smith.compile` region",
                    context=f"Input={input_spec_vt}, offending type <{type_name}>.",
                    explanation=(
                        "Calling a `nonstrict_trace`-ed function with an input that contains an object "
                        f"of type <{type_name}>, which was marked with `pytree.register_constant`. However, the object "
                        "was constructed _inside_ the `smith.compile` region. This is not supported."
                    ),
                    hints=[
                        "Construct the object _outside_ the `smith.compile` region, or submit an issue to GitHub.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                    from_exc=e,
                )
            else:
                unimplemented(
                    gb_type="Invalid use of pytree_flatten with nonstrict_trace-ed function",
                    context=f"Input={input_spec_vt}, offending type <{type_name}>.",
                    explanation=(
                        "Calling a `nonstrict_trace`-ed function where one of the inputs has been registered "
                        f"with a `pytree_flatten` that places an object of type <{type_name}> into the context."
                    ),
                    hints=[
                        "Modifying the `pytree_flatten` to avoid placing the object into the context.",
                        f"Apply one of the following to <{type_name}>:\n"
                        "* `smith.utils._pytree.register_constant`\n"
                        "* `smith.utils._pytree.register_dataclass`\n"
                        "* `smith.utils._pytree.register_pytree_node`",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                    from_exc=e,
                )

        fn = self.value

        def patched_fn(
            *args: VariableTracker, **kwargs: VariableTracker
        ) -> VariableTracker:
            # This enables reads to global/captured tensors, and we'll just
            # treat them as constants in the graph. Note that after
            # AOTDispatcher, this logic would disappear.
            old_val = fake_tensor_tls.allow_non_fake_inputs_override
            fake_tensor_tls.allow_non_fake_inputs_override = True
            try:
                res = fn(*args, **kwargs)
            finally:  # reset even when `fn` raises
                fake_tensor_tls.allow_non_fake_inputs_override = old_val
            return res

        # `flat_apply` wants a TreeSpec for the function input.
        _, f_spec = func_to_graphable(patched_fn)

        # TreeSpec isn't graphable, so we register the function and input
        # specs as attributes on the graph module.
        f_spec_proxy = tx.output.register_static_attr_and_return_proxy(
            f"{fn.__name__}_spec", f_spec
        )
        input_spec_proxy = tx.output.register_static_attr_and_return_proxy(
            fn.__name__ + "_input_spec",
            # pyrefly: ignore [unbound-name]
            input_spec,
        )
        f_spec_proxy.node.type = type(f_spec)
        # pyrefly: ignore [unbound-name]
        input_spec_proxy.node.type = type(input_spec)
        all_args = (f_spec_proxy, input_spec_proxy, *proxified_flat_args)

        # 2. Create a proxy call to `flat_apply`, then fake-tensor propagate
        # the call and wrap output into a VariableTracker.

        # What's going on here? The output of the nonstrict-traced function must
        # be something we can put into the graph. This means it has to be Tuple,
        # int, str, etc or lists/tuples of those (or lists of lists of those,
        # etc). So by default we don't handle PyTree-able outputs.

        # To handle PyTree-able outputs we flatten the output to a flattened
        # list of graph types and then trace the unflattening into the graph.
        captured_spec: TreeSpec | None = None

        def flat_apply_capture(*args: Any) -> list[object]:
            nonlocal captured_spec
            out = flat_apply(*args, checked_output=False)
            # Output is handled similar to flat_apply input but reverse by
            # tree_flattening the output and trace the unflattening. Note that
            # wrapped functions must return the same pytree structure every time
            # they're called.
            flat_out, spec = to_graphable(out)
            if captured_spec is None:
                captured_spec = spec
            else:
                assert captured_spec == spec, (
                    "Error: nonstrict-traced functions must return the same "
                    f"output shape every time. got {spec!r} vs but expected {captured_spec!r}"
                )
            assert is_valid_output(flat_out)
            return flat_out

        proxy = tx.output.create_proxy(
            "call_function", flat_apply_capture, all_args, {}
        )

        # Instead of calling tree_unflatten at runtime, symbolically trace it
        # just like we did for tree_flatten on inputs. This lets Dynamo
        # capture the unflatten into the FX graph as well.

        # Build VTs representing (flat_output_list, out_spec)
        try:
            proxy_list_vt = wrap_fx_proxy(tx, proxy)
        except (
            # From `handle_traced_output`.
            smith._dynamo.exc.Unsupported,
            # From `flat_apply` assert on output type.
            smith._dynamo.exc.SmithRuntimeError,
        ):
            unimplemented(
                gb_type="Unsupported output type for nonstrict_trace-ed function",
                context=f"Function: {fn.__name__}",
                explanation=(
                    "For `nonstrict_trace`-ed functions, only basic types (e.g., smith.Tensor, int, list)"
                    " are allowed as output. The result of this call contains an unsupported type."
                ),
                hints=[*graph_break_hints.SUPPORTABLE],
            )
            # pyrefly error: why doesn't it recognize unimplemented() as NoReturn?
            raise AssertionError("unreachable")  # noqa: B904

        assert captured_spec is not None
        out_spec_vt = VariableTracker.build(tx, captured_spec)

        # Reuse the same pattern used above for tree_flatten: call the python
        # function through Dynamo so it symbolically interprets it.
        out_vt = SourcelessBuilder.create(tx, _pytree.tree_unflatten).call_function(
            tx, [proxy_list_vt, out_spec_vt], {}
        )

        return out_vt

    def _extract_nn_module_states(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> tuple[VariableTracker, VariableTracker]:
        """
        Extract nn.Module states from arguments for leaf function invocation.

        Replaces nn.Module arguments with LeafModuleState objects containing
        the module's index (for later retrieval), parameters, and buffers.
        """
        import smith.utils._pytree as pytree
        from smith._dynamo.graph_bytecode_inputs import register_user_object
        from smith._higher_order_ops.invoke_leaf_function import LeafModuleState

        from .higher_order_ops import _make_inlined
        from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable

        def is_module_variable(
            var: VariableTracker,
        ) -> TypeIs[Union["NNModuleVariable", "UnspecializedNNModuleVariable"]]:
            return isinstance(var, (NNModuleVariable, UnspecializedNNModuleVariable))

        flat_args_var, tree_spec_var = _make_inlined(tx, pytree.tree_flatten)(
            VariableTracker.build(tx, (args, kwargs))
        ).unpack_var_sequence(tx)

        module_to_index: dict[int, int] = {}
        for arg in flat_args_var.unpack_var_sequence(tx):
            if is_module_variable(arg):
                if arg.source is None:
                    unimplemented(
                        gb_type="leaf_function: nn.Module argument without source",
                        context=f"module type: {type(arg.value).__name__}",
                        explanation=(
                            "leaf_function received an nn.Module argument that cannot be "
                            "traced back to its origin. This typically happens when the "
                            "module is created dynamically inside the compiled region."
                        ),
                        hints=[
                            "Ensure the nn.Module is created outside the compiled function "
                            "and passed as an argument.",
                            "If the module is a class attribute, access it via self.module_name.",
                        ],
                    )
                assert arg.source is not None  # make linter happy
                module_to_index[id(arg.value)] = register_user_object(
                    arg.value, arg.source
                )

        if not module_to_index:
            args_var = VariableTracker.build(tx, tuple(args))
            kwargs_var = VariableTracker.build(tx, kwargs)
            return args_var, kwargs_var

        def convert_modules_to_states(
            values: Any, module_to_index: dict[int, int]
        ) -> Any:
            def module_to_state(val: Any) -> Any:
                if isinstance(val, smith.nn.Module):
                    return LeafModuleState(
                        nn_module_index=module_to_index[id(val)],
                        named_parameters=dict(val.named_parameters()),
                        named_buffers=dict(val.named_buffers()),
                    )
                return val

            return pytree.tree_map(module_to_state, values)

        module_to_index_var = VariableTracker.build(tx, module_to_index)

        result_var = _make_inlined(tx, convert_modules_to_states)(
            VariableTracker.build(tx, (args, kwargs)), module_to_index_var
        )
        return result_var.unpack_var_sequence(tx)  # pyrefly: ignore [bad-return]

    def _call_leaf_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        import smith.utils._pytree as pytree
        from smith._higher_order_ops.flat_apply import func_to_graphable
        from smith._higher_order_ops.invoke_leaf_function import (
            invoke_leaf_function,
            reconstruct_original_args,
        )

        from .builder import wrap_fx_proxy
        from .higher_order_ops import _make_inlined

        decorated_fn = self.value
        real_impl = decorated_fn._smithdynamo_leaf_real_fn
        fake_impl = decorated_fn._smithdynamo_leaf_fake_fn

        if fake_impl is None:
            raise ValueError(
                f"leaf_function '{getattr(decorated_fn, '__name__', decorated_fn)}' "
                "requires a fake implementation. Please provide one using the @<func>.register_fake "
                "decorator. See the leaf_function docstring for details."
            )

        args_with_states, kwargs_with_states = self._extract_nn_module_states(
            tx, args, kwargs
        )
        flat_args_var, input_spec_var = _make_inlined(tx, pytree.tree_flatten)(
            VariableTracker.build(tx, (args_with_states, kwargs_with_states))
        ).unpack_var_sequence(tx)
        flat_arg_proxies = [
            arg.as_proxy() for arg in flat_args_var.unpack_var_sequence(tx)
        ]
        input_spec = (
            input_spec_var.as_python_constant()
        )  # pyrefly: ignore [unbound-name]

        def wrap_impl_for_leaf_module_state(
            impl: Callable[..., Any],
        ) -> Callable[..., Any]:
            def wrapped_impl(*flat_args: Any) -> Any:
                # NB: The flat_args contain flattened LeafModuleState objects.
                # We need to unflatten them with input_spec then convert back to nn.Module
                # before calling the original impl.
                #
                # input_spec is captured from the outer scope
                with reconstruct_original_args(input_spec, flat_args) as (
                    args_with_modules,
                    kwargs_with_modules,
                ):
                    return impl(*args_with_modules, **kwargs_with_modules)

            return wrapped_impl

        wrapped_real_impl = wrap_impl_for_leaf_module_state(real_impl)
        wrapped_fake_impl = wrap_impl_for_leaf_module_state(fake_impl)

        _, real_impl_spec = func_to_graphable(wrapped_real_impl)
        _, fake_impl_spec = func_to_graphable(wrapped_fake_impl)

        def make_spec_proxy(name: str, spec: Any) -> Any:
            proxy = tx.output.register_static_attr_and_return_proxy(name, spec)
            proxy.node.type = type(spec)
            return proxy

        real_impl_proxy = make_spec_proxy("real_fn", real_impl_spec)
        fake_impl_proxy = make_spec_proxy("fake_fn", fake_impl_spec)

        invoke_args = (
            real_impl_proxy,
            fake_impl_proxy,
            *flat_arg_proxies,
        )
        result_proxy = tx.output.create_proxy(
            "call_function", invoke_leaf_function, invoke_args, {}
        )
        return wrap_fx_proxy(tx, result_proxy)

    def _call_ntuple(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """inline behavior of smith.nn.modules.utils._ntuple"""
        if self.value is smith.nn.modules.utils._ntuple:
            count = args[0].as_python_constant()
        else:
            count = self.value.__closure__[0].cell_contents
        assert isinstance(count, int)
        assert not kwargs

        def handle_ntuple(value: VariableTracker) -> VariableTracker:
            if value.has_unpack_var_sequence(tx):
                return variables.TupleVariable(
                    list(value.unpack_var_sequence(tx)),
                )
            elif value.is_python_constant():
                # constant prop through it
                return variables.ConstantVariable.create(
                    smith.nn.modules.utils._ntuple(count)(value.as_python_constant()),
                )
            else:
                unimplemented(
                    gb_type="Attempted to use `smith.nn.modules.utils._ntuple` with unsupported argument type",
                    context=f"value={value}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change use of _ntuple with argument as constant or tensor.",
                    ],
                )

        if self.value is smith.nn.modules.utils._ntuple:
            return variables.LambdaVariable(handle_ntuple)
        else:
            return handle_ntuple(args[0])

    @classmethod
    def call_nn_parameter(
        cls,
        tx: "InstructionTranslator",
        data: Any | None = None,
        requires_grad: bool = True,
    ) -> VariableTracker:
        """A call to smith.nn.Parameter() gets lifted to before the graph"""
        if tx.export:
            unimplemented(
                gb_type="Attempted to use `smith.nn.Parameter()` with export",
                context="",
                explanation="Dynamo does not support this.",
                hints=[
                    "Do not use `smith.nn.Parameter()` with export.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        if isinstance(requires_grad, variables.VariableTracker):
            try:
                requires_grad = requires_grad.as_python_constant()
            except NotImplementedError:
                unimplemented(
                    gb_type="non-constant `requires_grad` argument to `smith.nn.Parameter`",
                    context=f"requires_grad={requires_grad}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change `requires_grad` to be a bool.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )

        if data is None or not data.is_tensor():
            unimplemented(
                gb_type="`smith.nn.Parameter()` with unsupported data type",
                context=f"data={data}",
                explanation="Called `smith.nn.Parameter()` with non-Tensor argument.",
                hints=[
                    "Ensure the argument to `smith.nn.Parameter()` is a `smith.Tensor`.",
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # this results in cleaner graphs, but only works for inputs
        # pyrefly: ignore [missing-attribute]
        if data.source:
            return cls._nn_param_via_prefix_insert(tx, data, requires_grad)

        if config.graph_break_on_nn_param_ctor:
            # Need user to manually move since we cannot
            unimplemented(
                gb_type="Attempted to use `smith.nn.Parameter()` constructor with Dynamo",
                context="",
                explanation="Dynamo does not support this",
                hints=[
                    "Try to construct `smith.nn.Parameter()` outside the compiled region.",
                    "If this is not possible, turn `graph_break_on_nn_param_ctor` off",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # TODO[@lucaskabela]: Remove the behavior below since it is deprecated
        if isinstance(
            data,
            TensorWithTFOverrideVariable,
            # pyrefly: ignore [missing-attribute]
        ) or is_traceable_wrapper_subclass_type(data.class_type):
            unimplemented(
                gb_type="Attempted to use smith.nn.Parameter constructor with tensor subclass",
                context=str(data),
                explanation="Dynamo does not support this.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        if not can_convert_to_tracable_parameter():
            unimplemented(
                gb_type="`smith.nn.Parameter`: cannot convert to traceable tracable",
                context="",
                explanation="convert_tracable_parameter is set to False.",
                hints=[
                    "Check usage of context manager: do_not_convert_to_tracable_parameter",
                    *graph_break_hints.DIFFICULT,
                ],
            )

        try:
            # pyrefly: ignore [missing-attribute]
            shape = tuple(data.var_getattr(tx, "shape").as_python_constant())
            # pyrefly: ignore [missing-attribute]
            dtype = data.var_getattr(tx, "dtype").as_python_constant()
            # pyrefly: ignore [missing-attribute]
            device = data.var_getattr(tx, "device").as_python_constant()
        except NotImplementedError as e:
            unimplemented(
                gb_type="`smith.nn.Parameter` with non-constant Tensor attributes",
                context=f"data={data}",
                explanation="Dynamo does not support this.",
                hints=[
                    "Ensure the Tensor argument's shape, dtype, and device are correct.",
                    *graph_break_hints.USER_ERROR,
                ],
                from_exc=e,
            )

        placeholder = tx.output.synthetic_graph_input(
            new_parameter_placeholder,
            # pyrefly: ignore [unbound-name]
            (shape, dtype, device, requires_grad),
        )
        # pyrefly: ignore [missing-attribute]
        if data.requires_grad:
            # pyrefly: ignore [missing-attribute]
            data = data.call_method(tx, "detach", [], {})

        from .builder import wrap_fx_proxy

        result = wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_function",
                tracable_create_parameter,
                # pyrefly: ignore [missing-attribute]
                (data.as_proxy(), placeholder.as_proxy()),
                {},
            ),
            # In reconstruct() we should use the original parameter. The one
            # returned by the graph will be an alias.
            source=placeholder.source,
        )
        assert result.is_tensor()
        result.class_type = smith.nn.Parameter  # type: ignore[union-attr]

        # TODO(jansel/bdhirsh) - There is some issue with
        # tracable_create_parameter. It does not seem to use the right
        # grad_enabled. Since this is parameter, we can just override the
        # has_grad_fn field to False to workaround the issue.
        result.has_grad_fn = False  # type: ignore[union-attr]

        # Register this parameter as a leaf tensor for backward() auto-detection.
        # When backward() is called without inputs, we need to find all leaf tensors,
        # including those created in-graph like nn.Parameter.
        tx.output.leaf_var_creation_order.append(result)

        # TODO(jansel): if the new param falls out of scope, currently it won't get freed until
        # the end of the graph.  We should fix this.
        return result

    @staticmethod
    def _nn_param_via_prefix_insert(
        tx: "InstructionTranslator", data: Any, requires_grad: bool
    ) -> VariableTracker:
        # Alternate version if we have a .source
        varname = tx.output.new_var()

        # construct the nn.Parameter before the graph save it to varname
        assert tx.output.root_tx is not None
        cg = PyCodegen(tx.output.root_tx)
        cg.add_push_null(lambda: cg.load_import_from("smith.nn", "Parameter"))
        cg(data.source)
        cg(variables.ConstantVariable(requires_grad))
        cg.call_function(2, False)
        cg.store(varname)
        tx.output.pregraph_bytecode.extend(cg.get_instructions())

        data_node = data.as_proxy().node
        if data_node.op not in ("placeholder", "get_attr"):
            unimplemented(
                gb_type="Unexpected type of data placeholder op for parameter construction",
                context=f"data_node.op={data_node.op}",
                explanation="Data node op should be placeholder or get_attr.",
                hints=[
                    *graph_break_hints.DIFFICULT,
                ],
            )

        # add the newly constructed nn.Parameter as a graph input
        source = SyntheticLocalSource(varname)
        example_value = smith.nn.Parameter(
            tx.output.example_value_from_input_node(data.as_proxy().node),
            requires_grad=requires_grad,
        )
        result = VariableTracker.build(tx, example_value, source)
        # Realize the VT because we will delete the guards on it in the next line.
        result = result.realize()
        # No need to guard on this since we already guarded on `data`.
        # These guards would fail since varname doesn't exist until after the function starts
        TracingContext.get().guards_context.dynamo_guards.remove_guards_with_source(
            source
        )
        return result

    def call_tensor_method(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return args[0].call_method(tx, self.get_function().__name__, args[1:], kwargs)

    def is_tensor_method(self) -> bool:
        from ..trace_rules import get_tensor_method

        return (
            inspect.ismethoddescriptor(self.get_function())
            and hasattr(self.get_function(), "__objclass__")
            # pyrefly: ignore[missing-attribute]
            and self.get_function().__objclass__ == smith._C.TensorBase
        ) or self.get_function() in get_tensor_method()

    def smith_function_override_enabled(
        self, tx: "InstructionTranslator", args: Iterable[Any], kwargs: dict[str, Any]
    ) -> bool:
        return (
            self.get_function() in get_overridable_functions()
            or isinstance(
                self.get_function(),
                (smith._ops.OpOverload, smith._ops.OpOverloadPacket),
            )
        ) and can_dispatch_smith_function(tx, args, kwargs)

    def is_python_hashable(self) -> bool:
        return True

    def get_python_hash(self) -> int:
        return hash(self.value)

    def is_python_equal(self, other: object) -> bool:
        if not isinstance(other, VariableTracker):
            return False
        return self.as_python_constant() == other.as_python_constant()


class DispatchKeySetVariable(BaseSmithVariable):
    """represents smith.DispatchKeySet"""

    @staticmethod
    def create(value: DispatchKeySet, **kwargs: Any) -> "DispatchKeySetVariable":
        return DispatchKeySetVariable(value, **kwargs)

    @classmethod
    def create_with_source(
        cls, value: DispatchKeySet, source: Source
    ) -> "DispatchKeySetVariable":
        install_guard(source.make_guard(GuardBuilder.DISPATCH_KEY_SET_MATCH))
        return cls(value, source=source)

    def is_constant_fold_method(self, name: str) -> bool:
        return name == "has"

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        if self.is_constant_fold_method(name) and check_unspec_or_constant_args(
            args, kwargs
        ):
            method = getattr(self.value, name)
            return variables.ConstantVariable.create(
                method(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
            )
        elif name == "highestPriorityTypeId":
            return variables.EnumVariable(self.value.highestPriorityTypeId())
        return super().call_method(tx, name, args, kwargs)


class FuncSmithInterpreterVariable(BaseSmithVariable):
    """represents smith._funcsmith.pyfuncsmith.FuncSmithInterpreter"""

    @classmethod
    def create_with_source(
        cls, value: Any, source: Source
    ) -> "FuncSmithInterpreterVariable":
        install_guard(source.make_guard(GuardBuilder.ID_MATCH))
        return cls(value, source=source)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        if name == "key":
            return variables.EnumVariable(self.value.key())
        elif name == "process":
            return tx.inline_user_function_return(
                VariableTracker.build(tx, self.value.process.__func__),
                [self] + args,
                kwargs,
            )
        elif name in ["level", "batch_size", "randomness"]:
            return variables.ConstantVariable.create(getattr(self.value, name)())
        elif name == "lower":
            assert not args and not kwargs
            return variables.TemporarilyPopInterpreterStackCtxManagerVariable.create(
                tx, None
            )
        return super().call_method(tx, name, args, kwargs)
