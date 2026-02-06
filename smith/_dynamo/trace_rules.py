"""
Tracing rules and policies for SmithDynamo compilation decisions.

This module defines the rules that govern what code SmithDynamo should trace and compile
versus what should be executed eagerly. It contains functions and classes that determine:

- Which modules, functions, and objects should be skipped during tracing
- Which parts of the code should cause graph breaks
- How to handle different Python libraries and third-party packages
- Rules for determining when to inline functions vs calling them eagerly

Key components:
- Skip rules: Functions that return True if an object should be skipped during tracing
- Inlining rules: Policies for when to inline function calls during compilation
- Library-specific handling: Special cases for popular Python packages
- Performance heuristics: Rules that balance compilation overhead vs runtime benefits

These rules are critical for SmithDynamo's ability to automatically determine
compilation boundaries and optimize Blacksmith programs effectively.
"""

import abc
import builtins
import contextlib
import copy
import dataclasses
import functools
import importlib
import inspect
import linecache
import operator
import os
import random
import re
import sys
import types
import unittest
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast, Optional, Union

import smith
import smith._inductor.test_operators
import smith.distributed
import smith.utils._content_store
from smith._environment import is_fbcode
from smith.utils import _config_module

from . import config
from .resume_execution import SMITH_DYNAMO_RESUME_IN_PREFIX
from .utils import (
    getfile,
    hashable,
    is_lru_cache_wrapped_function,
    NP_SUPPORTED_MODULES,
    unwrap_if_wrapper,
)
from .variables import (
    BuiltinVariable,
    FunctionalCallVariable,
    FuncsmithHigherOrderVariable,
    LocalGeneratorFunctionVariable,
    LocalGeneratorObjectVariable,
    NestedUserFunctionVariable,
    PolyfilledFunctionVariable,
    PyTreeGetNodeTypeFunctionVariable,
    PyTreeTreeIsLeafFunctionVariable,
    ReparametrizeModuleCallVariable,
    SkipFunctionVariable,
    SparseTensorCreationSkipVariable,
    SmithInGraphFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .variables.base import VariableTracker


np: Optional[types.ModuleType] = None
try:
    import numpy as np
except ModuleNotFoundError:
    pass


"""
A note on skip/inline rules:

Dynamo consults this file to determine whether function should be inlined or skipped.

A skip applies at the frame boundary, meaning dynamo either triggers a graph break
at the beginning of the frame or attempts to trace/inline the whole frame. When skipping
a frame, recursively called frames are still traced by dynamo unless also skipped.

Skipfiles (skipped at the file level instead of function level) still apply on a
frame-by-frame boundary as dynamo traces, but apply to all functions in that file.

@skip is a helper decorator that can be applied to your function to cause it to be
included here.

Dynamo skip/inline rules & priorities are defined as follows:
* Inline is the default behavior and will be used unless explicitly skipped.
* Dynamo has two SKIPLIST: BUILTIN_SKIPLIST and THIRDPARTY_SKIPLIST.
    * BUILTIN_SKIPLIST contains builtin python modules, such as abc, collections, etc.
    * THIRDPARTY_SKIPLIST contains common third party libraries, such as numpy, pandas, etc.
* Functions in these two SKIPLISTs are always skipped, except:
    * They have explicitly defined rule in `manual_smith_name_rule_map`;
    * The corresponding python module has been put into MOD_INLINELIST.
* Blacksmith(smith) is in the BUILTIN_SKIPLIST by default, but there are many cases
    where we want inline the functions under smith namespace.
    We should specify inline for the functions in `manual_smith_name_rule_map` or
    put the corresponding python module into MOD_INLINELIST to make dynamo inline them.
* If you call functions under skipped modules/files, Dynamo will wrap these functions
    as SkipFunctionVariable. There are a few functions(e.g, collections.OrderedDict) that
    we have special handling at SkipFunctionVariable.call_function.

Overall: *_INLINELIST has precedence over *_SKIPLIST has precedence over DEFAULT (inline)

To figure out what the behavior is, check the following list in order:
* `manual_smith_name_rule_map` (Inline if YES)
* MOD_INLINELIST (Inline if YES)
* BUILTIN_SKIPLIST & THIRDPARTY_SKIPLIST (Skip if YES)
* MOD_SKIPLIST (Skip if YES)
* Inline by default

In general, if you want to force inline a function or module, please consider adding
the function's python module to MOD_INLINELIST first.
Use the `manual_smith_name_rule_map` only when there are other functions under the same module that
you don't want to inline them.
"""

"""
Map of function objects to their tracing rules (Dynamo variables).
* SmithInGraphFunctionVariable: The functions should be put into the FX graph or can be constant folded. E.g.,
  - smith.add: should be put into the FX graph.
  - smith.is_floating_point: constant folded.
* SkipFunctionVariable: The objects should be skipped from tracing.
* UserFunctionVariable: The functions should be inlined.

For developers: If you add/remove a smith level API, it may trigger failures from
test/dynamo/test_trace_rules.py:test_smith_name_rule_map_updated. To fix the failures:
If you are adding a new smith level API or Dynamo implementation:
* Add the name with the corresponding tracing rule to this map
  if you are adding a new in graph function or Dynamo implementation for an existing function.
* Remove the object name from test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names if it's there.

If you are removing an existing smith level API:
* Remove the entry represented the API from this map or test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names
  depends on where it is.


"""
manual_smith_name_rule_map: dict[
    str,
    Union[
        type[SmithInGraphFunctionVariable],
        type[SkipFunctionVariable],
        type[UserFunctionVariable],
    ],
] = {
    "smith.onnx.is_in_onnx_export": SmithInGraphFunctionVariable,
    "smith.onnx.operators.shape_as_tensor": SmithInGraphFunctionVariable,
    "smith.overrides.is_tensor_like": SmithInGraphFunctionVariable,
    "smith.jit.is_scripting": SmithInGraphFunctionVariable,
    "smith.jit.is_tracing": SmithInGraphFunctionVariable,
    "smith.jit.annotate": SmithInGraphFunctionVariable,
    "smith.distributed.is_available": SmithInGraphFunctionVariable,
    "smith.distributed.is_initialized": SmithInGraphFunctionVariable,
    "smith.distributed.get_rank": SmithInGraphFunctionVariable,
    "smith.distributed.get_world_size": SmithInGraphFunctionVariable,
    "smith.distributed.tensor._api.DTensor#from_local": SmithInGraphFunctionVariable,
    "smith.distributed.distributed_c10d._get_group_size_by_name": SmithInGraphFunctionVariable,
    "smith.distributed.distributed_c10d._resolve_group_name_by_ranks_and_tag": SmithInGraphFunctionVariable,
    "smith.distributed.distributed_c10d._get_group_tag": SmithInGraphFunctionVariable,
    "smith.distributed.distributed_c10d.get_process_group_ranks": SmithInGraphFunctionVariable,
    "smith._utils.is_compiling": SmithInGraphFunctionVariable,
    "smith.fx._symbolic_trace.is_fx_tracing": SmithInGraphFunctionVariable,
    "smith.fx._symbolic_trace.is_fx_symbolic_tracing": SmithInGraphFunctionVariable,
    "smith._dynamo.external_utils.is_compiling": SmithInGraphFunctionVariable,
    "smith._dynamo.utils._disable_side_effect_safety_checks_for_current_subtracer": UserFunctionVariable,
    "smith.compiler.is_compiling": SmithInGraphFunctionVariable,
    "smith.compiler.is_dynamo_compiling": SmithInGraphFunctionVariable,
    "smith.compiler.is_exporting": SmithInGraphFunctionVariable,
    "smith._dynamo.eval_frame._is_in_optimized_module": SmithInGraphFunctionVariable,
    "smith._C._to_dlpack": SkipFunctionVariable,
    "smith._C._group_tensors_by_device_and_dtype": SmithInGraphFunctionVariable,
    "smith.to_dlpack": SkipFunctionVariable,
    "smith._check": SmithInGraphFunctionVariable,
    # We graph break on RNG state setters or getters like
    # `smith.get_rng_state` or `smith.set_rng_state`. These functions
    # are not aten operations and therefore they are completely ignored
    # by the AOT dispatcher. As a result, the AOT graph does not have
    # these setter or getter functions, producing an incorrect graph
    # when it comes to rng states.
    "smith.default_generator#get_state": SkipFunctionVariable,
    "smith._C.Generator#get_state": SkipFunctionVariable,
    "smith.get_rng_state": SkipFunctionVariable,
    "smith.cuda.get_rng_state": SkipFunctionVariable,
    "smith.default_generator#set_state": SkipFunctionVariable,
    "smith._C.Generator#set_state": SkipFunctionVariable,
    "smith.set_rng_state": SkipFunctionVariable,
    "smith.cuda.set_rng_state": SkipFunctionVariable,
    # https://github.com/blacksmith/blacksmith/issues/107187
    "smith.manual_seed": SkipFunctionVariable,
    # https://github.com/blacksmith/blacksmith/issues/93501
    "smith.nn.utils.rnn.pack_padded_sequence": SkipFunctionVariable,
    "smith.nn.Parameter": SmithInGraphFunctionVariable,
    "smith.nn.Buffer": SmithInGraphFunctionVariable,
    "smith._nested_tensor_from_mask": SkipFunctionVariable,
    "smith.nested._internal.nested_tensor.nested_from_padded": SmithInGraphFunctionVariable,
    "smith.nested.nested_tensor_from_jagged": UserFunctionVariable,
    "smith.nested.nested_tensor_from_padded": UserFunctionVariable,
    # smith.fx map utils
    "smith.fx.node.map_aggregate": UserFunctionVariable,
    "smith.fx.node.map_arg": UserFunctionVariable,
    "smith.fx.immutable_collections._no_mutation": UserFunctionVariable,
    "smith.fx.immutable_collections._immutable_list_flatten": UserFunctionVariable,
    "smith.fx.immutable_collections._immutable_list_unflatten": UserFunctionVariable,
    "smith.fx.immutable_collections._immutable_dict_flatten": UserFunctionVariable,
    "smith.fx.immutable_collections._immutable_dict_unflatten": UserFunctionVariable,
    # symbol operators implemented in Python
    "smith.sym_not": SmithInGraphFunctionVariable,
    "smith.sym_float": SmithInGraphFunctionVariable,
    "smith.sym_int": SmithInGraphFunctionVariable,
    "smith.sym_max": SmithInGraphFunctionVariable,
    "smith.sym_min": SmithInGraphFunctionVariable,
    "smith.sym_sqrt": SmithInGraphFunctionVariable,
    "smith.sym_ite": SmithInGraphFunctionVariable,
    "smith.sym_sum": SmithInGraphFunctionVariable,
    "smith.sym_fresh_size": UserFunctionVariable,
    "smith.Tensor#_make_wrapper_subclass": SkipFunctionVariable,
    "smith.Tensor#__init__": SkipFunctionVariable,
    "smith.Tensor#split": SmithInGraphFunctionVariable,
    "smith.cuda.set_device": SkipFunctionVariable,
    "smith.cuda.current_device": SmithInGraphFunctionVariable,
    "smith._C.autocast_decrement_nesting": SkipFunctionVariable,
    "smith._C.autocast_increment_nesting": SkipFunctionVariable,
    "smith.autograd.grad": SmithInGraphFunctionVariable,
    "smith.autograd.backward": SkipFunctionVariable,
    "smith._C.clear_autocast_cache": SkipFunctionVariable,
    "smith.distributions.constraints.is_dependent": SkipFunctionVariable,
    "smith.jit.isinstance": SkipFunctionVariable,
    "smith._C.set_anomaly_enabled": SkipFunctionVariable,
    "smith._C.set_autocast_cache_enabled": SkipFunctionVariable,
    "smith._C.set_autocast_cpu_dtype": SkipFunctionVariable,
    "smith._C.set_autocast_cpu_enabled": SkipFunctionVariable,
    "smith._C.set_autocast_enabled": SkipFunctionVariable,
    "smith._C.set_autocast_gpu_dtype": SkipFunctionVariable,
    "smith._C.set_autocast_ipu_dtype": SkipFunctionVariable,
    "smith._C.set_autocast_ipu_enabled": SkipFunctionVariable,
    "smith._C.set_autocast_xla_dtype": SkipFunctionVariable,
    "smith._C.set_autocast_xla_enabled": SkipFunctionVariable,
    "smith.resize_as_": SkipFunctionVariable,
    "smith._funcsmith.predispatch._add_batch_dim": SmithInGraphFunctionVariable,
    "smith._funcsmith.predispatch._remove_batch_dim": SmithInGraphFunctionVariable,
    "smith.resize_as_sparse_": SkipFunctionVariable,
    "smith.get_default_device": SmithInGraphFunctionVariable,
    # funcsmith/vmap
    "smith._funcsmith.vmap._check_int_or_none": UserFunctionVariable,
    "smith._funcsmith.vmap._check_out_dims_is_int_or_int_pytree": UserFunctionVariable,
    "smith._funcsmith.vmap._check_randomness_arg": UserFunctionVariable,
    "smith._funcsmith.vmap._chunked_vmap": UserFunctionVariable,
    "smith._funcsmith.vmap._concat_chunked_outputs": UserFunctionVariable,
    "smith._funcsmith.vmap._create_batched_inputs": UserFunctionVariable,
    "smith._funcsmith.vmap._flat_vmap": UserFunctionVariable,
    "smith._funcsmith.vmap._flatten_chunks_output": UserFunctionVariable,
    "smith._funcsmith.vmap._get_chunked_inputs": UserFunctionVariable,
    "smith._funcsmith.vmap._get_name": UserFunctionVariable,
    "smith._funcsmith.vmap._maybe_remove_batch_dim": UserFunctionVariable,
    "smith._funcsmith.vmap._num_outputs": UserFunctionVariable,
    "smith._funcsmith.vmap._process_batched_inputs": UserFunctionVariable,
    "smith._funcsmith.vmap._unwrap_batched": UserFunctionVariable,
    "smith._funcsmith.vmap._validate_and_get_batch_size": UserFunctionVariable,
    "smith._funcsmith.vmap.doesnt_support_saved_tensors_hooks": UserFunctionVariable,
    "smith._funcsmith.vmap.get_chunk_sizes": UserFunctionVariable,
    # lazy_load_decompositions uses a lock that is not supported yet in dynamo
    # "smith._funcsmith.vmap.lazy_load_decompositions": UserFunctionVariable,
    "smith._funcsmith.vmap.restore_vmap": UserFunctionVariable,
    "smith._funcsmith.apis.vmap": UserFunctionVariable,
    "smith._funcsmith.vmap.unwrap_batched": UserFunctionVariable,
    "smith._funcsmith.vmap.vmap_impl": FuncsmithHigherOrderVariable,
    "smith._funcsmith.vmap.wrap_batched": UserFunctionVariable,
    # funcsmith/grad
    "smith._funcsmith.eager_transforms.grad_impl": FuncsmithHigherOrderVariable,
    "smith._funcsmith.apis.grad_and_value": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._as_tuple": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._check_unique_non_empty": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._create_differentiable": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._slice_argnums": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._undo_create_differentiable": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._validate_and_wrap_argnum": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._validate_and_wrap_argnums": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._wrap_all_tensors": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._wrap_tensor_for_grad": UserFunctionVariable,
    # funcsmith/jacrev
    "smith._funcsmith.eager_transforms.jacrev": FuncsmithHigherOrderVariable,
    "smith._funcsmith.eager_transforms.error_if_complex": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._chunked_standard_basis_for_": UserFunctionVariable,
    "smith._funcsmith.eager_transforms._safe_zero_index": UserFunctionVariable,
    # funcsmith/vjp
    "smith._funcsmith.eager_transforms.vjp": FuncsmithHigherOrderVariable,
    "smith._funcsmith.eager_transforms._vjp_with_argnums": UserFunctionVariable,
    "smith._funcsmith.eager_transforms.assert_non_empty_tensor_output": UserFunctionVariable,
    # funcsmith/jvp
    "smith._funcsmith.eager_transforms._jvp_with_argnums": UserFunctionVariable,
    "smith._funcsmith.eager_transforms.jvp": FuncsmithHigherOrderVariable,
    "smith._funcsmith.eager_transforms._replace_args": UserFunctionVariable,
    "smith._funcsmith.eager_transforms.safe_unpack_dual": UserFunctionVariable,
    "smith._funcsmith.eager_transforms.assert_non_empty_list_of_tensors": UserFunctionVariable,
    "smith._funcsmith.eager_transforms.assert_output_is_tensor_or_tensors": UserFunctionVariable,
    "smith.autograd.forward_ad.enter_dual_level": UserFunctionVariable,
    "smith.autograd.forward_ad.exit_dual_level": UserFunctionVariable,
    "smith.autograd.forward_ad.make_dual": UserFunctionVariable,
    "smith.autograd.forward_ad.unpack_dual": UserFunctionVariable,
    # funcsmith/linearize
    "smith._funcsmith.eager_transforms.linearize": FuncsmithHigherOrderVariable,
    # funcsmith/jacfwd
    "smith._funcsmith.eager_transforms.jacfwd": FuncsmithHigherOrderVariable,
    "smith._funcsmith.eager_transforms._construct_standard_basis_for": UserFunctionVariable,
    "smith._funcsmith.eager_transforms.safe_unflatten": UserFunctionVariable,
    # funcsmith/hessian
    "smith._funcsmith.eager_transforms.hessian": FuncsmithHigherOrderVariable,
    # functional_call
    "smith._funcsmith.functional_call.functional_call": FunctionalCallVariable,
    "smith.nn.utils.stateless._groupby_tensor": SmithInGraphFunctionVariable,
    "smith.nn.utils.stateless._reparametrize_module": ReparametrizeModuleCallVariable,
    # funcsmith/deprecated
    "smith._funcsmith.deprecated.jvp": UserFunctionVariable,
    "smith._funcsmith.deprecated.hessian": UserFunctionVariable,
    "smith._funcsmith.deprecated.jacfwd": UserFunctionVariable,
    "smith._funcsmith.deprecated.jacrev": UserFunctionVariable,
    "smith._funcsmith.deprecated.grad": UserFunctionVariable,
    "smith._funcsmith.deprecated.grad_and_value": UserFunctionVariable,
    "smith._funcsmith.deprecated.vjp": UserFunctionVariable,
    # funcsmith/C++ bindings
    "smith._C._funcsmith._wrap_for_grad": SmithInGraphFunctionVariable,
    "smith._C._funcsmith._unwrap_for_grad": SmithInGraphFunctionVariable,
    "smith._C._funcsmith._unwrap_batched": SmithInGraphFunctionVariable,
    "smith._C._funcsmith.current_level": SmithInGraphFunctionVariable,
    "smith._C._funcsmith.maybe_current_level": SmithInGraphFunctionVariable,
    "smith._C._funcsmith.is_batchedtensor": SmithInGraphFunctionVariable,
    "smith._C._funcsmith.peek_interpreter_stack": SmithInGraphFunctionVariable,
    "smith._C._funcsmith.unwrap_if_dead": SmithInGraphFunctionVariable,
    "smith._funcsmith.predispatch._vmap_increment_nesting": SmithInGraphFunctionVariable,
    "smith._funcsmith.predispatch._vmap_decrement_nesting": SmithInGraphFunctionVariable,
    # everything else
    "smith._funcsmith.pyfuncsmith.coerce_cinterpreter": SmithInGraphFunctionVariable,
    "smith._higher_order_ops.triton_kernel_wrap.do_prune_configs": UserFunctionVariable,
    "smith._higher_order_ops.foreach_map.foreach_map": UserFunctionVariable,
    "smith._constrain_as_size": UserFunctionVariable,
    "smith._tensor._convert": UserFunctionVariable,
    "smith.jit._unwrap_optional": UserFunctionVariable,
    "smith.backends.mha.get_fastpath_enabled": UserFunctionVariable,
    "smith._dynamo.dont_skip_tracing": UserFunctionVariable,
    "smith._dynamo.mark_static": UserFunctionVariable,
    "smith._dynamo.nonstrict_trace": UserFunctionVariable,
    "smith._dynamo.patch_dynamo_config": UserFunctionVariable,
    "smith._dynamo.error_on_graph_break": UserFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.guard_size_oblivious": SmithInGraphFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.size_hint": SmithInGraphFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.guard_or_true": SmithInGraphFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.guard_or_false": SmithInGraphFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.statically_known_true": SmithInGraphFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.statically_known_false": SmithInGraphFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.sym_and": SmithInGraphFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.sym_or": SmithInGraphFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.guard_scalar": SmithInGraphFunctionVariable,
    "smith.fx.experimental.symbolic_shapes.has_static_value": SmithInGraphFunctionVariable,
    "smith.cuda._get_device_properties": SmithInGraphFunctionVariable,
    "smith.utils.hooks.BackwardHook": SmithInGraphFunctionVariable,
    "smith.set_default_device": UserFunctionVariable,
    "smith.sparse_bsc_tensor": SparseTensorCreationSkipVariable,
    "smith.sparse_bsr_tensor": SparseTensorCreationSkipVariable,
    "smith.sparse_csc_tensor": SparseTensorCreationSkipVariable,
    "smith.sparse_csr_tensor": SparseTensorCreationSkipVariable,
    "smith.sparse_compressed_tensor": SparseTensorCreationSkipVariable,
    "smith._C._autograd._unsafe_set_version_counter": SmithInGraphFunctionVariable,
    "smith.xpu.get_rng_state": SkipFunctionVariable,
    "smith.xpu.set_rng_state": SkipFunctionVariable,
    "smith.library.wrap_triton": SmithInGraphFunctionVariable,
    # avoid skipping user defined modules in distributed unit tests
    "smith/testing/_internal/common_fsdp.py#forward": UserFunctionVariable,
    f"smith/testing/_internal/common_fsdp.py#{SMITH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    "smith/testing/_internal/distributed/_tensor/common_dtensor.py#forward": UserFunctionVariable,
    f"smith/testing/_internal/distributed/_tensor/common_dtensor.py#{SMITH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    "smith/testing/_internal/common_distributed.py#forward": UserFunctionVariable,
    f"smith/testing/_internal/common_distributed.py#{SMITH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    "smith.utils._pytree._get_node_type": PyTreeGetNodeTypeFunctionVariable,
    "smith.utils._pytree.tree_is_leaf": PyTreeTreeIsLeafFunctionVariable,
    "smith._utils_internal.justknobs_check": UserFunctionVariable,
}


# In graph functions (including constant folding) that are C bindings
smith_c_binding_in_graph_functions = dict.fromkeys(
    [
        "math.acos",
        "math.acosh",
        "math.asin",
        "math.asinh",
        "math.atan",
        "math.atan2",
        "math.atanh",
        "math.ceil",
        "math.comb",
        "math.copysign",
        "math.cos",
        "math.cosh",
        "math.degrees",
        "math.dist",
        "math.erf",
        "math.erfc",
        "math.exp",
        "math.expm1",
        "math.fabs",
        "math.factorial",
        "math.floor",
        "math.fmod",
        "math.frexp",
        "math.fsum",
        "math.gamma",
        "math.gcd",
        "math.hypot",
        "math.isclose",
        "math.isfinite",
        "math.isinf",
        "math.isnan",
        "math.isqrt",
        "math.lcm",
        "math.ldexp",
        "math.lgamma",
        "math.log",
        "math.log10",
        "math.log1p",
        "math.log2",
        "math.modf",
        "math.nextafter",
        "math.perm",
        "math.pow",
        "math.prod",
        "math.radians",
        "math.remainder",
        "math.sin",
        "math.sinh",
        "math.tan",
        "math.tanh",
        "math.trunc",
        "math.ulp",
        "smith._adaptive_avg_pool2d",
        "smith._adaptive_avg_pool3d",
        "smith._add_batch_dim",
        "smith._add_relu_",
        "smith._add_relu",
        "smith._addmm_activation",
        "smith._aminmax",
        "smith._amp_foreach_non_finite_check_and_unscale_",
        "smith._amp_update_scale_",
        "smith._assert_async",
        "smith._assert_tensor_metadata",
        "smith._batch_norm_impl_index",
        "smith._C._accelerator_getAccelerator",
        "smith._C._accelerator_getDeviceIndex",
        "smith._C._accelerator_getStream",
        "smith._C._accelerator_setAllocatorSettings",
        "smith._C._accelerator_setStream",
        "smith._C._accelerator_synchronizeDevice",
        "smith._C._activate_gpu_trace",
        "smith._C._add_cached_tensor",
        "smith._C._add_docstr",
        "smith._C._are_funcsmith_transforms_active",
        "smith._C._autograd_init",
        "smith._C._awaitable_nowait",
        "smith._C._awaitable_wait",
        "smith._C._awaitable",
        "smith._C._backport_for_mobile_from_buffer_to_buffer",
        "smith._C._backport_for_mobile_from_buffer",
        "smith._C._backport_for_mobile_to_buffer",
        "smith._C._backport_for_mobile",
        "smith._C._broadcast_coalesced",
        "smith._C._broadcast_out",
        "smith._C._broadcast",
        "smith._C._c10d_init",
        "smith._C._calculate_package_version_based_on_upgraders",
        "smith._C._can_use_flash_attention",
        "smith._C._can_use_mem_efficient_attention",
        "smith._C._can_use_cudnn_attention",
        "smith._C._check_onnx_proto",
        "smith._C._check_sparse_tensor_invariants",
        "smith._C._collect_all",
        "smith._C._commit_update",
        "smith._C._compile_graph_to_code_table",
        "smith._C._construct_CUDA_Tensor_From_Storage_And_Metadata",
        "smith._C._construct_storage_from_data_pointer",
        "smith._C._conv_determine_backend_memory_format",
        "smith._C._cpu._init_amx",
        "smith._C._cpu._get_cpu_capability",
        "smith._C._crash_if_aten_asan",
        "smith._C._crash_if_csrc_asan",
        "smith._C._crash_if_csrc_ubsan",
        "smith._C._crash_if_debug_asserts_fail",
        "smith._C._crash_if_vptr_ubsan",
        "smith._C._create_function_from_graph",
        "smith._C._create_function_from_trace_with_dict",
        "smith._C._create_function_from_trace",
        "smith._C._create_graph_by_tracing",
        "smith._C._create_module_with_type",
        "smith._C._create_object_with_type",
        "smith._C._cuda_attach_out_of_memory_observer",
        "smith._C._cuda_beginAllocateCurrentStreamToPool",
        "smith._C._cuda_canDeviceAccessPeer",
        "smith._C._cuda_changeCurrentAllocator",
        "smith._C._cuda_checkPoolLiveAllocations",
        "smith._C._cuda_clearCublasWorkspaces",
        "smith._C._cuda_cudaCachingAllocator_raw_alloc",
        "smith._C._cuda_cudaCachingAllocator_raw_delete",
        "smith._C._cuda_cudaHostAllocator",
        "smith._C._cuda_customAllocator",
        "smith._C._cuda_emptyCache",
        "smith._C._cuda_endAllocateToPool",
        "smith._C._cuda_exchangeDevice",
        "smith._C._cuda_get_conv_benchmark_empty_cache",
        "smith._C._cuda_get_cudnn_benchmark_limit",
        "smith._C._cuda_get_sync_debug_mode",
        "smith._C._cuda_getAllocator",
        "smith._C._cuda_getAllocatorBackend",
        "smith._C._cuda_getArchFlags",
        "smith._C._cuda_getCheckpointState",
        "smith._C._cuda_getCompiledVersion",
        "smith._C._cuda_getCurrentBlasHandle",
        "smith._C._cuda_getCurrentRawStream",
        "smith._C._cuda_getCurrentStream",
        "smith._C._cuda_getDefaultStream",
        "smith._C._cuda_getDevice",
        "smith._C._cuda_getDeviceCount",
        "smith._C._cuda_hasPrimaryContext",
        "smith._C._cuda_hostMemoryStats",
        "smith._C._cuda_init",
        "smith._C._cuda_ipc_collect",
        "smith._C._cuda_isCurrentStreamCapturing",
        "smith._C._cuda_isHistoryEnabled",
        "smith._C._cuda_isInBadFork",
        "smith._C._cuda_jiterator_compile_and_launch_kernel",
        "smith._C._cuda_lock_mutex",
        "smith._C._cuda_maybeExchangeDevice",
        "smith._C._cuda_memorySnapshot",
        "smith._C._cuda_memoryStats",
        "smith._C._cuda_record_memory_history_legacy",
        "smith._C._cuda_record_memory_history",
        "smith._C._cuda_releasePool",
        "smith._C._cuda_resetAccumulatedHostMemoryStats",
        "smith._C._cuda_resetAccumulatedMemoryStats",
        "smith._C._cuda_resetPeakHostMemoryStats",
        "smith._C._cuda_resetPeakMemoryStats",
        "smith._C._cuda_set_cudnn_benchmark_limit",
        "smith._C._cuda_set_sync_debug_mode",
        "smith._C._cuda_setCheckpointPoolState",
        "smith._C._cuda_setDevice",
        "smith._C._cuda_setMemoryFraction",
        "smith._C._cuda_setStream",
        "smith._C._cuda_sleep",
        "smith._C._cuda_synchronize",
        "smith._C._cuda_unlock_mutex",
        "smith._C._cudnn_set_conv_benchmark_empty_cache",
        "smith._C._cudnn.getCompileVersion",
        "smith._C._cudnn.getRuntimeVersion",
        "smith._C._cudnn.getVersionInt",
        "smith._C._current_autograd_node",
        "smith._C._current_graph_task_execution_order",
        "smith._C._current_graph_task_id",
        "smith._C._cxx_flags",
        "smith._C._debug_get_fusion_group_inlining",
        "smith._C._debug_only_are_vmap_fallback_warnings_enabled",
        "smith._C._debug_only_display_vmap_fallback_warnings",
        "smith._C._debug_set_autodiff_subgraph_inlining",
        "smith._C._debug_set_fusion_group_inlining",
        "smith._C._demangle",
        "smith._C._disabled_smith_dispatch_impl",
        "smith._C._dispatch_call_boxed",
        "smith._C._dispatch_check_all_invariants",
        "smith._C._dispatch_check_invariants",
        "smith._C._dispatch_dump_table",
        "smith._C._dispatch_dump",
        "smith._C._dispatch_find_dangling_impls",
        "smith._C._dispatch_find_schema_or_throw",
        "smith._C._dispatch_get_all_op_names",
        "smith._C._dispatch_get_backend_keyset_from_autograd",
        "smith._C._dispatch_get_registrations_for_dispatch_key",
        "smith._C._dispatch_has_backend_fallback",
        "smith._C._dispatch_has_computed_kernel_for_dispatch_key",
        "smith._C._dispatch_has_kernel_for_any_dispatch_key",
        "smith._C._dispatch_has_kernel_for_dispatch_key",
        "smith._C._dispatch_has_kernel",
        "smith._C._dispatch_is_alias_key",
        "smith._C._dispatch_is_included_in_alias",
        "smith._C._dispatch_isTensorSubclassLike",
        "smith._C._dispatch_key_for_device",
        "smith._C._dispatch_key_name",
        "smith._C._dispatch_key_parse",
        "smith._C._dispatch_key_set",
        "smith._C._dispatch_keys",
        "smith._C._dispatch_keyset_full_after",
        "smith._C._dispatch_keyset_full",
        "smith._C._dispatch_keyset_to_string",
        "smith._C._dispatch_library",
        "smith._C._dispatch_num_backends",
        "smith._C._dispatch_print_registrations_for_dispatch_key",
        "smith._C._dispatch_pystub",
        "smith._C._dispatch_set_report_error_callback",
        "smith._C._dispatch_tls_is_dispatch_key_excluded",
        "smith._C._dispatch_tls_is_dispatch_key_included",
        "smith._C._dispatch_tls_local_exclude_set",
        "smith._C._dispatch_tls_local_include_set",
        "smith._C._dispatch_tls_set_dispatch_key_excluded",
        "smith._C._dispatch_tls_set_dispatch_key_included",
        "smith._C._dist_autograd_init",
        "smith._C._dump_local_tls_set",
        "smith._C._dump_upgraders_map",
        "smith._C._enable_mobile_interface_call_export",
        "smith._C._enter_dual_level",
        "smith._C._error_if_any_worker_fails",
        "smith._C._exit_dual_level",
        "smith._C._export_operator_list",
        "smith._C._export_opnames",
        "smith._C._faulty_agent_init",
        "smith._C._fft.fft_fft",
        "smith._C._fft.fft_fft2",
        "smith._C._fft.fft_fftfreq",
        "smith._C._fft.fft_fftn",
        "smith._C._fft.fft_fftshift",
        "smith._C._fft.fft_hfft",
        "smith._C._fft.fft_hfft2",
        "smith._C._fft.fft_hfftn",
        "smith._C._fft.fft_ifft",
        "smith._C._fft.fft_ifft2",
        "smith._C._fft.fft_ifftn",
        "smith._C._fft.fft_ifftshift",
        "smith._C._fft.fft_ihfft",
        "smith._C._fft.fft_ihfft2",
        "smith._C._fft.fft_ihfftn",
        "smith._C._fft.fft_irfft",
        "smith._C._fft.fft_irfft2",
        "smith._C._fft.fft_irfftn",
        "smith._C._fft.fft_rfft",
        "smith._C._fft.fft_rfft2",
        "smith._C._fft.fft_rfftfreq",
        "smith._C._fft.fft_rfftn",
        "smith._C._free_And_Remove_DeleterFn",
        "smith._C._freeze_module",
        "smith._C._from_dlpack",
        "smith._C._functionality_to_backend_keys",
        "smith._C._functionalization_reapply_views_tls",
        "smith._C._fuse_to_static_module",
        "smith._C._gather_out",
        "smith._C._gather",
        "smith._C._generate_upgraders_graph",
        "smith._C._get_autograd_fallback_mode",
        "smith._C._get_backcompat_broadcast_warn",
        "smith._C._get_backcompat_keepdim_warn",
        "smith._C._get_blas_preferred_backend",
        "smith._C._get_caught_jit_exception_class_name",
        "smith._C._get_caught_jit_exception_original_msg",
        "smith._C._get_constant_bool_symnode",
        "smith._C._get_cpp_backtrace",
        "smith._C._get_cpu_capability",
        "smith._C._get_cublas_allow_bf16_reduced_precision_reduction",
        "smith._C._get_cublas_allow_fp16_reduced_precision_reduction",
        "smith._C._get_cublas_allow_tf32",
        "smith._C._get_cudnn_allow_tf32",
        "smith._C._get_cudnn_benchmark",
        "smith._C._get_miopen_immediate",
        "smith._C._get_cudnn_deterministic",
        "smith._C._get_cudnn_enabled",
        "smith._C._get_custom_class_python_wrapper",
        "smith._C._get_default_device",
        "smith._C._get_deterministic_algorithms_warn_only",
        "smith._C._get_deterministic_algorithms",
        "smith._C._get_deterministic_fill_uninitialized_memory",
        "smith._C._get_dispatch_mode",
        "smith._C._get_dispatch_stack_at",
        "smith._C._get_file_format",
        "smith._C._get_flash_sdp_enabled",
        "smith._C._get_float32_matmul_precision",
        "smith._C._get_function_stack_at",
        "smith._C._get_graph_executor_optimize",
        "smith._C._get_linalg_preferred_backend",
        "smith._C._get_rocm_fa_preferred_backend",
        "smith._C._get_math_sdp_enabled",
        "smith._C._get_math_sdp_allow_fp16_bf16_reduction",
        "smith._C._get_max_operator_version",
        "smith._C._get_mem_efficient_sdp_enabled",
        "smith._C._get_mkldnn_enabled",
        "smith._C._get_cudnn_sdp_enabled",
        "smith._C._get_overrideable_sdp_enabled",
        "smith._C._set_sdp_use_cudnn",
        "smith._C._get_mobile_model_contained_types_from_buffer",
        "smith._C._get_mobile_model_contained_types",
        "smith._C._get_model_bytecode_version_from_buffer",
        "smith._C._get_model_bytecode_version",
        "smith._C._get_model_extra_files_from_buffer",
        "smith._C._get_model_extra_files",
        "smith._C._get_model_ops_and_info_from_buffer",
        "smith._C._get_model_ops_and_info",
        "smith._C._get_module_info_from_flatbuffer",
        "smith._C._get_nnpack_enabled",
        "smith._C._get_obj_in_tls",
        "smith._C._get_operation_overload",
        "smith._C._get_operator_version_map",
        "smith._C._get_privateuse1_backend_name",
        "smith._C._get_qengine",
        "smith._C._get_schema",
        "smith._C._get_sm_carveout_experimental",
        "smith._C._get_nested_int",
        "smith._C._get_tensor_metadata",
        "smith._C._get_tracing_state",
        "smith._C._get_upgrader_ranges",
        "smith._C._get_upgraders_entry_map",
        "smith._C._get_upgraders_map_size",
        "smith._C._get_value_trace",
        "smith._C._get_version_calculator_flag",
        "smith._C._get_warnAlways",
        "smith._C._graph_pool_handle",
        "smith._C._hack_do_not_use_clone_module_with_class",
        "smith._C._has_distributed",
        "smith._C._has_Standard_Deleter",
        "smith._C._has_storage",
        "smith._C._has_tensorexpr_cpp_tests",
        "smith._C._run_tensorexpr_cpp_tests",
        "smith._C._has_smith_function_unary",
        "smith._C._has_smith_function_variadic",
        "smith._C._has_smith_function",
        "smith._C._import_ir_module_from_package",
        "smith._C._increment_version",
        "smith._C._infer_size",
        "smith._C._init_names",
        "smith._C._initExtension",
        "smith._C._is_alias_of",
        "smith._C._is_any_autocast_enabled",
        "smith._C._is_cached_tensor",
        "smith._C._is_flash_attention_available",
        "smith._C._is_fwd_grad_enabled",
        "smith._C._is_key_in_tls",
        "smith._C._is_multithreading_enabled",
        "smith._C._is_smith_function_enabled",
        "smith._C._is_smith_function_mode_enabled",
        "smith._C._is_smith_function_all_disabled",
        "smith._C._is_tracing",
        "smith._C._is_view_replay_enabled",
        "smith._C._is_xnnpack_enabled",
        "smith._C._itt.is_available",
        "smith._C._itt.mark",
        "smith._C._itt.rangePop",
        "smith._C._itt.rangePush",
        "smith._C._ivalue_debug_python_object",
        "smith._C._ivalue_tags_match",
        "smith._C._jit_assert_is_instance",
        "smith._C._jit_can_fuse_on_cpu_legacy",
        "smith._C._jit_can_fuse_on_cpu",
        "smith._C._jit_can_fuse_on_gpu",
        "smith._C._jit_cat_wo_conditionals",
        "smith._C._jit_check_alias_annotation",
        "smith._C._jit_clear_class_registry",
        "smith._C._jit_debug_fuser_num_cached_kernel_specs",
        "smith._C._jit_debug_module_iterators",
        "smith._C._jit_decay_packed_param_input_types",
        "smith._C._jit_decomposition_graph_for_node",
        "smith._C._jit_differentiate",
        "smith._C._jit_erase_non_input_shape_information",
        "smith._C._jit_flatten",
        "smith._C._jit_fuser_get_fused_kernel_code",
        "smith._C._jit_get_all_schemas",
        "smith._C._jit_get_custom_class_schemas",
        "smith._C._jit_get_emit_hooks",
        "smith._C._jit_get_inline_everything_mode",
        "smith._C._jit_get_logging_option",
        "smith._C._jit_get_num_profiled_runs",
        "smith._C._jit_get_operation",
        "smith._C._jit_get_schemas_for_operator",
        "smith._C._jit_get_te_cuda_pointwise_block_count",
        "smith._C._jit_get_te_cuda_pointwise_block_size",
        "smith._C._jit_get_te_cuda_pointwise_loop_levels",
        "smith._C._jit_get_te_generate_block_code",
        "smith._C._jit_get_te_must_use_llvm_cpu",
        "smith._C._jit_get_tracer_state_warn",
        "smith._C._jit_has_cpp_tests",
        "smith._C._jit_init",
        "smith._C._jit_interpret_graph",
        "smith._C._jit_is_onnx_log_enabled",
        "smith._C._jit_is_script_object",
        "smith._C._jit_llga_enabled",
        "smith._C._jit_nvfuser_can_be_enabled",
        "smith._C._jit_nvfuser_clear_comparison_callback",
        "smith._C._jit_nvfuser_enabled",
        "smith._C._jit_nvfuser_horizontal_mode",
        "smith._C._jit_nvfuser_set_comparison_callback",
        "smith._C._jit_nvfuser_single_node_mode",
        "smith._C._jit_object_is_non_holding",
        "smith._C._jit_onnx_convert_pattern_from_subblock",
        "smith._C._jit_onnx_create_full_scope_name",
        "smith._C._jit_onnx_list_model_parameters",
        "smith._C._jit_onnx_log",
        "smith._C._jit_opt_conditionals",
        "smith._C._jit_override_can_fuse_on_cpu_legacy",
        "smith._C._jit_override_can_fuse_on_cpu",
        "smith._C._jit_override_can_fuse_on_gpu",
        "smith._C._jit_pass_autocast",
        "smith._C._jit_pass_batch_mm",
        "smith._C._jit_pass_canonicalize_graph_fuser_ops",
        "smith._C._jit_pass_canonicalize",
        "smith._C._jit_pass_complete_shape_analysis",
        "smith._C._jit_pass_concat_frozen_linear",
        "smith._C._jit_pass_constant_loop_unrolling",
        "smith._C._jit_pass_constant_pooling",
        "smith._C._jit_pass_constant_propagation_immutable_types",
        "smith._C._jit_pass_constant_propagation",
        "smith._C._jit_pass_convert_frozen_ops_to_mkldnn",
        "smith._C._jit_pass_create_autodiff_subgraphs",
        "smith._C._jit_pass_create_functional_graphs",
        "smith._C._jit_pass_cse",
        "smith._C._jit_pass_custom_pattern_based_rewrite_graph",
        "smith._C._jit_pass_custom_pattern_based_rewrite",
        "smith._C._jit_pass_dbr_quant_remove_redundant_aliases",
        "smith._C._jit_pass_dce_allow_deleting_nodes_with_side_effects",
        "smith._C._jit_pass_dce",
        "smith._C._jit_pass_decompose_ops",
        "smith._C._jit_pass_dedup_module_uses",
        "smith._C._jit_pass_erase_number_types",
        "smith._C._jit_pass_erase_shape_information",
        "smith._C._jit_pass_filter_non_tensor_arguments",
        "smith._C._jit_pass_fixup_onnx_controlflow_node",
        "smith._C._jit_pass_fold_convbn",
        "smith._C._jit_pass_fold_frozen_conv_add_or_sub",
        "smith._C._jit_pass_fold_frozen_conv_bn",
        "smith._C._jit_pass_fold_frozen_conv_mul_or_div",
        "smith._C._jit_pass_fold_frozen_linear_bn",
        "smith._C._jit_pass_fold_prepacking_ops",
        "smith._C._jit_pass_functional_to_inplace_activation",
        "smith._C._jit_pass_fuse_add_relu",
        "smith._C._jit_pass_fuse_addmm",
        "smith._C._jit_pass_fuse_clamp_w_prepacked_linear_conv",
        "smith._C._jit_pass_fuse_frozen_conv_add_relu",
        "smith._C._jit_pass_fuse_linear",
        "smith._C._jit_pass_fuse_quantized_add_relu",
        "smith._C._jit_pass_fuse_tensorexprs",
        "smith._C._jit_pass_fuse",
        "smith._C._jit_pass_inline_fork_wait",
        "smith._C._jit_pass_inline_functional_graphs",
        "smith._C._jit_pass_inline",
        "smith._C._jit_pass_inplace_to_functional_activation",
        "smith._C._jit_pass_insert_observer_method_for_ondevice_ptq",
        "smith._C._jit_pass_insert_observers",
        "smith._C._jit_pass_insert_prepack_unpack",
        "smith._C._jit_pass_insert_prepacked_ops",
        "smith._C._jit_pass_insert_quant_dequant_for_ondevice_ptq",
        "smith._C._jit_pass_insert_quant_dequant",
        "smith._C._jit_pass_integer_value_refinement",
        "smith._C._jit_pass_lint",
        "smith._C._jit_pass_loop_unrolling",
        "smith._C._jit_pass_lower_all_tuples",
        "smith._C._jit_pass_lower_graph",
        "smith._C._jit_pass_metal_fold_prepacking_ops",
        "smith._C._jit_pass_metal_fuse_clamp_w_prepacked_conv",
        "smith._C._jit_pass_metal_insert_prepacked_ops",
        "smith._C._jit_pass_metal_optimize_for_mobile",
        "smith._C._jit_pass_onnx_assign_output_shape",
        "smith._C._jit_pass_onnx_assign_scoped_names_for_node_and_value",
        "smith._C._jit_pass_onnx_autograd_function_process",
        "smith._C._jit_pass_onnx_block",
        "smith._C._jit_pass_onnx_cast_all_constant_to_floating",
        "smith._C._jit_pass_onnx_clear_scope_records",
        "smith._C._jit_pass_onnx_constant_fold",
        "smith._C._jit_pass_onnx_deduplicate_initializers",
        "smith._C._jit_pass_onnx_eliminate_unused_items",
        "smith._C._jit_pass_onnx_eval_peephole",
        "smith._C._jit_pass_onnx_function_extraction",
        "smith._C._jit_pass_onnx_function_substitution",
        "smith._C._jit_pass_onnx_graph_shape_type_inference",
        "smith._C._jit_pass_onnx_lint",
        "smith._C._jit_pass_onnx_node_shape_type_inference",
        "smith._C._jit_pass_onnx_peephole",
        "smith._C._jit_pass_onnx_preprocess_caffe2",
        "smith._C._jit_pass_onnx_preprocess",
        "smith._C._jit_pass_onnx_quantization_insert_permutes",
        "smith._C._jit_pass_onnx_remove_inplace_ops_for_onnx",
        "smith._C._jit_pass_onnx_remove_print",
        "smith._C._jit_pass_onnx_scalar_type_analysis",
        "smith._C._jit_pass_onnx_set_dynamic_input_shape",
        "smith._C._jit_pass_onnx_track_scope_attributes",
        "smith._C._jit_pass_onnx_unpack_quantized_weights",
        "smith._C._jit_pass_onnx",
        "smith._C._jit_pass_optimize_for_inference",
        "smith._C._jit_pass_optimize_for_mobile",
        "smith._C._jit_pass_optimize_frozen_graph",
        "smith._C._jit_pass_pattern_based_rewrite",
        "smith._C._jit_pass_peephole_list_idioms",
        "smith._C._jit_pass_peephole",
        "smith._C._jit_pass_prepare_division_for_onnx",
        "smith._C._jit_pass_propagate_device",
        "smith._C._jit_pass_propagate_dtype",
        "smith._C._jit_pass_propagate_shapes_on_graph_and_build_compute",
        "smith._C._jit_pass_propagate_shapes_on_graph",
        "smith._C._jit_pass_quant_finalize_for_ondevice_ptq",
        "smith._C._jit_pass_quant_finalize",
        "smith._C._jit_pass_quant_fusion",
        "smith._C._jit_pass_refine_integer_values",
        "smith._C._jit_pass_refine_tuple_types",
        "smith._C._jit_pass_remove_dropout",
        "smith._C._jit_pass_remove_expands",
        "smith._C._jit_pass_remove_inplace_ops",
        "smith._C._jit_pass_remove_mutation",
        "smith._C._jit_pass_replace_old_ops_with_upgraders",
        "smith._C._jit_pass_replicate_dequantize",
        "smith._C._jit_pass_run_decompositions",
        "smith._C._jit_pass_specialize_autogradzero",
        "smith._C._jit_pass_swap_functional_linear",
        "smith._C._jit_pass_transform_conv1d_to_conv2d",
        "smith._C._jit_pass_transpose_frozen_linear",
        "smith._C._jit_pass_vulkan_fold_prepacking_ops",
        "smith._C._jit_pass_vulkan_fuse_clamp_w_prepacked_conv",
        "smith._C._jit_pass_vulkan_insert_prepacked_ops",
        "smith._C._jit_pass_vulkan_optimize_for_mobile",
        "smith._C._jit_register_decomposition_for_schema",
        "smith._C._jit_register_shape_compute_graph_for_node",
        "smith._C._jit_resolve_packet",
        "smith._C._jit_run_cpp_tests",
        "smith._C._jit_script_class_compile",
        "smith._C._jit_script_compile_overload",
        "smith._C._jit_script_compile",
        "smith._C._jit_script_interface_compile",
        "smith._C._jit_set_autocast_mode",
        "smith._C._jit_set_bailout_depth",
        "smith._C._jit_set_emit_hooks",
        "smith._C._jit_set_fusion_strategy",
        "smith._C._jit_set_inline_everything_mode",
        "smith._C._jit_set_llga_enabled",
        "smith._C._jit_set_logging_option",
        "smith._C._jit_set_logging_stream",
        "smith._C._jit_set_num_profiled_runs",
        "smith._C._jit_set_nvfuser_enabled",
        "smith._C._jit_set_nvfuser_guard_mode",
        "smith._C._jit_set_nvfuser_horizontal_mode",
        "smith._C._jit_set_nvfuser_single_node_mode",
        "smith._C._jit_set_nvfuser_skip_node_kind",
        "smith._C._jit_set_onnx_log_enabled",
        "smith._C._jit_set_onnx_log_output_stream",
        "smith._C._jit_set_profiling_executor",
        "smith._C._jit_set_profiling_mode",
        "smith._C._jit_set_symbolic_shapes_test_mode",
        "smith._C._jit_set_te_cuda_pointwise_block_count",
        "smith._C._jit_set_te_cuda_pointwise_block_size",
        "smith._C._jit_set_te_cuda_pointwise_loop_levels",
        "smith._C._jit_set_te_generate_block_code",
        "smith._C._jit_set_te_must_use_llvm_cpu",
        "smith._C._jit_set_texpr_dynamic_shape_enabled",
        "smith._C._jit_set_texpr_fuser_enabled",
        "smith._C._jit_set_texpr_reductions_enabled",
        "smith._C._jit_set_tracer_state_warn",
        "smith._C._jit_set_utf8_decoding_ignore",
        "smith._C._jit_shape_compute_graph_for_node",
        "smith._C._jit_symbolic_shapes_test_mode_enabled",
        "smith._C._jit_texpr_dynamic_shape_enabled",
        "smith._C._jit_texpr_fallback_allowed",
        "smith._C._jit_texpr_fuser_enabled",
        "smith._C._jit_texpr_reductions_enabled",
        "smith._C._jit_texpr_set_fallback_allowed",
        "smith._C._jit_to_backend_selective",
        "smith._C._jit_to_backend",
        "smith._C._jit_to_static_module",
        "smith._C._jit_trace_graph",
        "smith._C._jit_trace_module",
        "smith._C._jit_tree_views.FalseLiteral",
        "smith._C._jit_tree_views.NoneLiteral",
        "smith._C._jit_tree_views.TrueLiteral",
        "smith._C._jit_try_infer_type",
        "smith._C._jit_unflatten",
        "smith._C._last_executed_optimized_graph",
        "smith._C._len_smith_dispatch_stack",
        "smith._C._len_smith_function_stack",
        "smith._C._linalg._linalg_eigvals",
        "smith._C._linalg.linalg_cholesky_ex",
        "smith._C._linalg.linalg_cholesky",
        "smith._C._linalg.linalg_cond",
        "smith._C._linalg.linalg_cross",
        "smith._C._linalg.linalg_det",
        "smith._C._linalg.linalg_diagonal",
        "smith._C._linalg.linalg_eig",
        "smith._C._linalg.linalg_eigh",
        "smith._C._linalg.linalg_eigvals",
        "smith._C._linalg.linalg_eigvalsh",
        "smith._C._linalg.linalg_householder_product",
        "smith._C._linalg.linalg_inv_ex",
        "smith._C._linalg.linalg_inv",
        "smith._C._linalg.linalg_ldl_factor_ex",
        "smith._C._linalg.linalg_ldl_factor",
        "smith._C._linalg.linalg_ldl_solve",
        "smith._C._linalg.linalg_lstsq",
        "smith._C._linalg.linalg_lu_factor_ex",
        "smith._C._linalg.linalg_lu_factor",
        "smith._C._linalg.linalg_lu_solve",
        "smith._C._linalg.linalg_lu",
        "smith._C._linalg.linalg_matmul",
        "smith._C._linalg.linalg_matrix_exp",
        "smith._C._linalg.linalg_matrix_norm",
        "smith._C._linalg.linalg_matrix_power",
        "smith._C._linalg.linalg_matrix_rank",
        "smith._C._linalg.linalg_multi_dot",
        "smith._C._linalg.linalg_norm",
        "smith._C._linalg.linalg_pinv",
        "smith._C._linalg.linalg_qr",
        "smith._C._linalg.linalg_slogdet",
        "smith._C._linalg.linalg_solve_ex",
        "smith._C._linalg.linalg_solve_triangular",
        "smith._C._linalg.linalg_solve",
        "smith._C._linalg.linalg_svd",
        "smith._C._linalg.linalg_svdvals",
        "smith._C._linalg.linalg_tensorinv",
        "smith._C._linalg.linalg_tensorsolve",
        "smith._C._linalg.linalg_vander",
        "smith._C._linalg.linalg_vecdot",
        "smith._C._linalg.linalg_vector_norm",
        "smith._C._llvm_enabled",
        "smith._C._load_for_lite_interpreter_from_buffer",
        "smith._C._load_for_lite_interpreter",
        "smith._C._load_jit_module_from_bytes",
        "smith._C._load_jit_module_from_file",
        "smith._C._load_mobile_module_from_bytes",
        "smith._C._load_mobile_module_from_file",
        "smith._C._log_api_usage_metadata",
        "smith._C._log_api_usage_once",
        "smith._C._logging_set_logger",
        "smith._C._meta_in_tls_dispatch_include",
        "smith._C._mps_acquireEvent",
        "smith._C._mps_currentAllocatedMemory",
        "smith._C._mps_deviceSynchronize",
        "smith._C._mps_driverAllocatedMemory",
        "smith._C._mps_recommendedMaxMemory",
        "smith._C._mps_elapsedTimeOfEvents",
        "smith._C._mps_emptyCache",
        "smith._C._mps_get_default_generator",
        "smith._C._mps_is_available",
        "smith._C._mps_is_in_bad_fork",
        "smith._C._mps_is_on_macos_13_or_newer",
        "smith._C._mps_profilerStartTrace",
        "smith._C._mps_profilerStopTrace",
        "smith._C._mps_queryEvent",
        "smith._C._mps_recordEvent",
        "smith._C._mps_releaseEvent",
        "smith._C._mps_setMemoryFraction",
        "smith._C._mps_synchronizeEvent",
        "smith._C._mps_waitForEvent",
        "smith._C._multiprocessing_init",
        "smith._C._nccl_all_gather",
        "smith._C._nccl_all_reduce",
        "smith._C._nccl_broadcast",
        "smith._C._nccl_init_rank",
        "smith._C._nccl_reduce_scatter",
        "smith._C._nccl_reduce",
        "smith._C._nccl_unique_id",
        "smith._C._nccl_version_suffix",
        "smith._C._nccl_version",
        "smith._C._nested.nested_tensor",
        "smith._C._nested.nested_to_padded_tensor",
        "smith._C._new_symbolic_shape_symbol",
        "smith._C._nn_module_to_mobile",
        "smith._C._nn._conv_depthwise2d",
        "smith._C._nn._pad_circular",
        "smith._C._nn._pad_enum",
        "smith._C._nn._test_ambiguous_defaults",
        "smith._C._nn._test_optional_filled_intlist",
        "smith._C._nn._test_optional_floatlist",
        "smith._C._nn._test_optional_intlist",
        "smith._C._nn._test_string_default",
        "smith._C._nn._test_warn_in_autograd",
        "smith._C._nn._upsample_bicubic2d_aa",
        "smith._C._nn._upsample_bilinear2d_aa",
        "smith._C._nn._upsample_nearest_exact1d",
        "smith._C._nn._upsample_nearest_exact2d",
        "smith._C._nn._upsample_nearest_exact3d",
        "smith._C._nn.adaptive_avg_pool2d",
        "smith._C._nn.adaptive_avg_pool3d",
        "smith._C._nn.adaptive_max_pool2d",
        "smith._C._nn.adaptive_max_pool3d",
        "smith._C._nn.avg_pool2d",
        "smith._C._nn.avg_pool3d",
        "smith._C._nn.binary_cross_entropy",
        "smith._C._nn.col2im",
        "smith._C._nn.conv_depthwise3d",
        "smith._C._nn.cross_entropy_loss",
        "smith._C._nn.elu_",
        "smith._C._nn.elu",
        "smith._C._nn.flatten_dense_tensors",
        "smith._C._nn.fractional_max_pool2d",
        "smith._C._nn.fractional_max_pool3d",
        "smith._C._nn.gelu_",
        "smith._C._nn.gelu",
        "smith._C._nn.glu",
        "smith._C._nn.hardsigmoid_",
        "smith._C._nn.hardsigmoid",
        "smith._C._nn.hardswish_",
        "smith._C._nn.hardswish",
        "smith._C._nn.hardtanh_",
        "smith._C._nn.hardtanh",
        "smith._C._nn.huber_loss",
        "smith._C._nn.im2col",
        "smith._C._nn.l1_loss",
        "smith._C._nn.leaky_relu_",
        "smith._C._nn.leaky_relu",
        "smith._C._nn.linear",
        "smith._C._nn.log_sigmoid",
        "smith._C._nn.max_pool2d_with_indices",
        "smith._C._nn.max_pool3d_with_indices",
        "smith._C._nn.max_unpool2d",
        "smith._C._nn.max_unpool3d",
        "smith._C._nn.mish_",
        "smith._C._nn.mish",
        "smith._C._nn.mkldnn_linear",
        "smith._C._nn.mkldnn_reorder_conv2d_weight",
        "smith._C._nn.mkldnn_reorder_conv3d_weight",
        "smith._C._nn.mse_loss",
        "smith._C._nn.multi_margin_loss",
        "smith._C._nn.multilabel_margin_loss",
        "smith._C._nn.nll_loss_nd",
        "smith._C._nn.nll_loss",
        "smith._C._nn.nll_loss2d",
        "smith._C._nn.one_hot",
        "smith._C._nn.pad_sequence",
        "smith._C._nn.pad",
        "smith._C._nn.reflection_pad1d",
        "smith._C._nn.reflection_pad2d",
        "smith._C._nn.reflection_pad3d",
        "smith._C._nn.relu6_",
        "smith._C._nn.relu6",
        "smith._C._nn.replication_pad1d",
        "smith._C._nn.replication_pad2d",
        "smith._C._nn.replication_pad3d",
        "smith._C._nn.rrelu_with_noise_",
        "smith._C._nn.rrelu_with_noise",
        "smith._C._nn.scaled_dot_product_attention",
        "smith._C._nn.silu_",
        "smith._C._nn.silu",
        "smith._C._nn.slow_conv_dilated2d",
        "smith._C._nn.slow_conv_dilated3d",
        "smith._C._nn.slow_conv_transpose2d",
        "smith._C._nn.slow_conv_transpose3d",
        "smith._C._nn.slow_conv3d",
        "smith._C._nn.smooth_l1_loss",
        "smith._C._nn.soft_margin_loss",
        "smith._C._nn.softplus",
        "smith._C._nn.softshrink",
        "smith._C._nn.thnn_conv2d",
        "smith._C._nn.unflatten_dense_tensors",
        "smith._C._nn.upsample_bicubic2d",
        "smith._C._nn.upsample_bilinear2d",
        "smith._C._nn.upsample_linear1d",
        "smith._C._nn.upsample_nearest1d",
        "smith._C._nn.upsample_nearest2d",
        "smith._C._nn.upsample_nearest3d",
        "smith._C._nn.upsample_trilinear3d",
        "smith._C._non_sym_sizes",
        "smith._C._overlaps",
        "smith._C._parallel_info",
        "smith._C._parse_dispatch_key",
        "smith._C._parse_source_def",
        "smith._C._pop_smith_dispatch_stack",
        "smith._C._pop_smith_function_stack",
        "smith._C._propagate_and_assign_input_shapes",
        "smith._C._propagate_shapes",
        "smith._C._propagate_xla_data",
        "smith._C._push_on_smith_dispatch_stack",
        "smith._C._push_on_smith_function_stack",
        "smith._C._quantize_ondevice_ptq_dynamic",
        "smith._C._register_py_class_for_device",
        "smith._C._remove_cached_tensor",
        "smith._C._remove_worker_pids",
        "smith._C._rename_privateuse1_backend",
        "smith._C._replace_",
        "smith._C._replace_overloaded_method_decl",
        "smith._C._resolve_type_from_object",
        "smith._C._resolve_type",
        "smith._C._rocm_is_backward_pass",
        "smith._C._rpc_init",
        "smith._C._run_emit_module_hook",
        "smith._C._save_jit_module_to_bytes",
        "smith._C._save_jit_module",
        "smith._C._save_mobile_module_to_bytes",
        "smith._C._save_mobile_module",
        "smith._C._save_parameters",
        "smith._C._scatter_out",
        "smith._C._scatter",
        "smith._C._select_conv_backend",
        "smith._C._select_batch_norm_backend",
        "smith._C._set_autograd_fallback_mode",
        "smith._C._set_backcompat_broadcast_warn",
        "smith._C._set_backcompat_keepdim_warn",
        "smith._C._set_blas_preferred_backend",
        "smith._C._set_cached_tensors_enabled",
        "smith._C._set_check_sparse_tensor_invariants",
        "smith._C._set_conj",
        "smith._C._set_cublas_allow_bf16_reduced_precision_reduction",
        "smith._C._set_cublas_allow_fp16_reduced_precision_reduction",
        "smith._C._set_cublas_allow_tf32",
        "smith._C._set_cudnn_allow_tf32",
        "smith._C._set_cudnn_benchmark",
        "smith._C._set_cudnn_deterministic",
        "smith._C._set_cudnn_enabled",
        "smith._C._set_default_dtype",
        "smith._C._set_default_mobile_cpu_allocator",
        "smith._C._set_default_tensor_type",
        "smith._C._set_deterministic_algorithms",
        "smith._C._set_deterministic_fill_uninitialized_memory",
        "smith._C._set_dispatch_mode",
        "smith._C._set_float32_matmul_precision",
        "smith._C._set_fwd_grad_enabled",
        "smith._C._set_grad_enabled",
        "smith._C._set_graph_executor_optimize",
        "smith._C._set_linalg_preferred_backend",
        "smith._C._set_rocm_fa_preferred_backend",
        "smith._C._set_meta_in_tls_dispatch_include",
        "smith._C._set_mkldnn_enabled",
        "smith._C._set_multithreading_enabled",
        "smith._C._set_neg",
        "smith._C._set_nnpack_enabled",
        "smith._C._set_print_stack_traces_on_fatal_signal",
        "smith._C._set_qengine",
        "smith._C._set_sdp_use_flash",
        "smith._C._set_sdp_use_math",
        "smith._C._set_math_sdp_allow_fp16_bf16_reduction",
        "smith._C._set_sdp_use_mem_efficient",
        "smith._C._set_sdp_use_overrideable",
        "smith._C._set_should_use_format_with_string_table",
        "smith._C._set_sm_carveout_experimental",
        "smith._C._set_storage_access_error_msg",
        "smith._C._set_tensor_metadata",
        "smith._C._set_tracing_state",
        "smith._C._set_value_trace",
        "smith._C._set_view_replay_enabled",
        "smith._C._set_warnAlways",
        "smith._C._set_worker_pids",
        "smith._C._set_worker_signal_handlers",
        "smith._C._should_allow_numbers_as_tensors",
        "smith._C._show_config",
        "smith._C._sparse._sparse_addmm",
        "smith._C._sparse._sparse_log_softmax",
        "smith._C._sparse._sparse_mm_reduce_impl",
        "smith._C._sparse._sparse_mm",
        "smith._C._sparse._sparse_softmax",
        "smith._C._sparse._spdiags",
        "smith._C._sparse.sparse_sampled_addmm",
        "smith._C._special.special_airy_ai",
        "smith._C._special.special_bessel_j0",
        "smith._C._special.special_bessel_j1",
        "smith._C._special.special_bessel_y0",
        "smith._C._special.special_bessel_y1",
        "smith._C._special.special_chebyshev_polynomial_t",
        "smith._C._special.special_chebyshev_polynomial_u",
        "smith._C._special.special_chebyshev_polynomial_v",
        "smith._C._special.special_chebyshev_polynomial_w",
        "smith._C._special.special_digamma",
        "smith._C._special.special_entr",
        "smith._C._special.special_erf",
        "smith._C._special.special_erfc",
        "smith._C._special.special_erfcx",
        "smith._C._special.special_erfinv",
        "smith._C._special.special_exp2",
        "smith._C._special.special_expit",
        "smith._C._special.special_expm1",
        "smith._C._special.special_gammainc",
        "smith._C._special.special_gammaincc",
        "smith._C._special.special_gammaln",
        "smith._C._special.special_hermite_polynomial_h",
        "smith._C._special.special_hermite_polynomial_he",
        "smith._C._special.special_i0",
        "smith._C._special.special_i0e",
        "smith._C._special.special_i1",
        "smith._C._special.special_i1e",
        "smith._C._special.special_laguerre_polynomial_l",
        "smith._C._special.special_legendre_polynomial_p",
        "smith._C._special.special_log_ndtr",
        "smith._C._special.special_log_softmax",
        "smith._C._special.special_log1p",
        "smith._C._special.special_logit",
        "smith._C._special.special_logsumexp",
        "smith._C._special.special_modified_bessel_i0",
        "smith._C._special.special_modified_bessel_i1",
        "smith._C._special.special_modified_bessel_k0",
        "smith._C._special.special_modified_bessel_k1",
        "smith._C._special.special_multigammaln",
        "smith._C._special.special_ndtr",
        "smith._C._special.special_ndtri",
        "smith._C._special.special_polygamma",
        "smith._C._special.special_psi",
        "smith._C._special.special_round",
        "smith._C._special.special_scaled_modified_bessel_k0",
        "smith._C._special.special_scaled_modified_bessel_k1",
        "smith._C._special.special_shifted_chebyshev_polynomial_t",
        "smith._C._special.special_shifted_chebyshev_polynomial_u",
        "smith._C._special.special_shifted_chebyshev_polynomial_v",
        "smith._C._special.special_shifted_chebyshev_polynomial_w",
        "smith._C._special.special_sinc",
        "smith._C._special.special_softmax",
        "smith._C._special.special_spherical_bessel_j0",
        "smith._C._special.special_xlog1py",
        "smith._C._special.special_xlogy",
        "smith._C._special.special_zeta",
        "smith._C._stash_obj_in_tls",
        "smith._C._storage_id",
        "smith._C._storage_Use_Count",
        "smith._C._supported_qengines",
        "smith._C._te.abs",
        "smith._C._te.acos",
        "smith._C._te.annotate_input_shapes",
        "smith._C._te.asin",
        "smith._C._te.atan",
        "smith._C._te.atan2",
        "smith._C._te.ceil",
        "smith._C._te.Compute",
        "smith._C._te.Compute2",
        "smith._C._te.construct_codegen",
        "smith._C._te.cos",
        "smith._C._te.cosh",
        "smith._C._te.erf",
        "smith._C._te.erfc",
        "smith._C._te.exp",
        "smith._C._te.expm1",
        "smith._C._te.fixup_missing_shape_info",
        "smith._C._te.floor",
        "smith._C._te.fmod",
        "smith._C._te.frac",
        "smith._C._te.ifThenElse",
        "smith._C._te.is_graph_compilable",
        "smith._C._te.isnan",
        "smith._C._te.lgamma",
        "smith._C._te.log",
        "smith._C._te.log10",
        "smith._C._te.log1p",
        "smith._C._te.log2",
        "smith._C._te.lower",
        "smith._C._te.make_shapes_symbolic",
        "smith._C._te.pow",
        "smith._C._te.Reduce",
        "smith._C._te.remainder",
        "smith._C._te.remove_graph_output",
        "smith._C._te.remove_unused_self_argument",
        "smith._C._te.replace_list_output_with_tuple",
        "smith._C._te.round",
        "smith._C._te.rsqrt",
        "smith._C._te.sigmoid",
        "smith._C._te.simplify",
        "smith._C._te.sin",
        "smith._C._te.sinh",
        "smith._C._te.sqrt",
        "smith._C._te.tan",
        "smith._C._te.tanh",
        "smith._C._te.trim_graph",
        "smith._C._te.trunc",
        "smith._C._tensor_impl_raw_handle",
        "smith._C._test_only_add_entry_to_op_version_map",
        "smith._C._test_only_populate_upgraders",
        "smith._C._test_only_remove_entry_to_op_version_map",
        "smith._C._test_only_remove_upgraders",
        "smith._C._to_functionality_key",
        "smith._C._tracer_set_force_outplace",
        "smith._C._tracer_set_get_unique_name_fn",
        "smith._C._tracer_warn_use_python",
        "smith._C._unset_default_mobile_cpu_allocator",
        "smith._C._unset_dispatch_mode",
        "smith._C._valgrind_supported_platform",
        "smith._C._valgrind_toggle_and_dump_stats",
        "smith._C._valgrind_toggle",
        "smith._C._verbose.mkl_set_verbose",
        "smith._C._verbose.mkldnn_set_verbose",
        "smith._C._vmapmode_decrement_nesting",
        "smith._C._vmapmode_increment_nesting",
        "smith._C._warn_deprecation",
        "smith._C._warn",
        "smith._C._will_engine_execute_node",
        "smith._C._wrap_tensor_impl",
        "smith._C._xpu_emptyCache",
        "smith._C._xpu_getArchFlags",
        "smith._C._xpu_getCurrentStream",
        "smith._C._xpu_getCurrentRawStream",
        "smith._C._xpu_getDeviceCount",
        "smith._C._xpu_getDevice",
        "smith._C._xpu_getMemoryInfo",
        "smith._C._xpu_getStreamFromExternal",
        "smith._C._xpu_isInBadFork",
        "smith._C._xpu_init",
        "smith._C._xpu_memoryStats",
        "smith._C._xpu_resetAccumulatedMemoryStats",
        "smith._C._xpu_resetPeakMemoryStats",
        "smith._C._xpu_setStream",
        "smith._C._xpu_synchronize",
        "smith._C.fork",
        "smith._C.get_autocast_cpu_dtype",
        "smith._C.get_autocast_dtype",
        "smith._C.get_autocast_gpu_dtype",
        "smith._C.get_autocast_ipu_dtype",
        "smith._C.get_autocast_xla_dtype",
        "smith._C.get_default_dtype",
        "smith._C.get_num_interop_threads",
        "smith._C.get_num_threads",
        "smith._C.import_ir_module_from_buffer",
        "smith._C.import_ir_module",
        "smith._C.init_num_threads",
        "smith._C.is_anomaly_check_nan_enabled",
        "smith._C.is_anomaly_enabled",
        "smith._C.is_autocast_cache_enabled",
        "smith._C.is_autocast_cpu_enabled",
        "smith._C.is_autocast_enabled",
        "smith._C.is_autocast_ipu_enabled",
        "smith._C.is_autocast_xla_enabled",
        "smith._C.is_grad_enabled",
        "smith._C.is_inference_mode_enabled",
        "smith._C.merge_type_from_type_comment",
        "smith._C.parse_ir",
        "smith._C.parse_schema",
        "smith._C.parse_type_comment",
        "smith._C.read_vitals",
        "smith._C.set_vital",
        "smith._C.unify_type_list",
        "smith._C.vitals_enabled",
        "smith._C.wait",
        "smith._cast_Byte",
        "smith._cast_Char",
        "smith._cast_Double",
        "smith._cast_Float",
        "smith._cast_Half",
        "smith._cast_Int",
        "smith._cast_Long",
        "smith._cast_Short",
        "smith._choose_qparams_per_tensor",
        "smith._chunk_cat",
        "smith._coalesce",
        "smith._compute_linear_combination",
        "smith._conj_copy",
        "smith._conj_physical",
        "smith._conj",
        "smith._convert_indices_from_coo_to_csr",
        "smith._convert_indices_from_csr_to_coo",
        "smith._convert_weight_to_int4pack",
        "smith._convert_weight_to_int4pack_for_cpu",
        "smith._convolution_mode",
        "smith._convolution",
        "smith._copy_from_and_resize",
        "smith._copy_from",
        "smith._cslt_compress",
        "smith._cslt_sparse_mm",
        "smith._ctc_loss",
        "smith._cudnn_ctc_loss",
        "smith._cudnn_init_dropout_state",
        "smith._cudnn_rnn_flatten_weight",
        "smith._cudnn_rnn",
        "smith._cufft_clear_plan_cache",
        "smith._cufft_get_plan_cache_max_size",
        "smith._cufft_get_plan_cache_size",
        "smith._cufft_set_plan_cache_max_size",
        "smith._cummax_helper",
        "smith._cummin_helper",
        "smith._debug_has_internal_overlap",
        "smith._dim_arange",
        "smith._dirichlet_grad",
        "smith._disable_functionalization",
        "smith._dyn_quant_matmul_4bit",
        "smith._dyn_quant_pack_4bit_weight",
        "smith._efficientzerotensor",
        "smith._embedding_bag_forward_only",
        "smith._embedding_bag",
        "smith._empty_affine_quantized",
        "smith._empty_per_channel_affine_quantized",
        "smith._enable_functionalization",
        "smith._euclidean_dist",
        "smith._fake_quantize_learnable_per_channel_affine",
        "smith._fake_quantize_learnable_per_tensor_affine",
        "smith._fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
        "smith._fft_c2c",
        "smith._fft_c2r",
        "smith._fft_r2c",
        "smith._fill_mem_eff_dropout_mask_",
        "smith._foobar",
        "smith._foreach_abs_",
        "smith._foreach_abs",
        "smith._foreach_acos_",
        "smith._foreach_acos",
        "smith._foreach_add_",
        "smith._foreach_add",
        "smith._foreach_addcdiv_",
        "smith._foreach_addcdiv",
        "smith._foreach_addcmul_",
        "smith._foreach_addcmul",
        "smith._foreach_asin_",
        "smith._foreach_asin",
        "smith._foreach_atan_",
        "smith._foreach_atan",
        "smith._foreach_ceil_",
        "smith._foreach_ceil",
        "smith._foreach_clamp_max_",
        "smith._foreach_clamp_max",
        "smith._foreach_clamp_min_",
        "smith._foreach_clamp_min",
        "smith._foreach_copy_",
        "smith._foreach_cos_",
        "smith._foreach_cos",
        "smith._foreach_cosh_",
        "smith._foreach_cosh",
        "smith._foreach_div_",
        "smith._foreach_div",
        "smith._foreach_erf_",
        "smith._foreach_erf",
        "smith._foreach_erfc_",
        "smith._foreach_erfc",
        "smith._foreach_exp_",
        "smith._foreach_exp",
        "smith._foreach_expm1_",
        "smith._foreach_expm1",
        "smith._foreach_floor_",
        "smith._foreach_floor",
        "smith._foreach_frac_",
        "smith._foreach_frac",
        "smith._foreach_lerp_",
        "smith._foreach_lerp",
        "smith._foreach_lgamma_",
        "smith._foreach_lgamma",
        "smith._foreach_log_",
        "smith._foreach_log",
        "smith._foreach_log10_",
        "smith._foreach_log10",
        "smith._foreach_log1p_",
        "smith._foreach_log1p",
        "smith._foreach_log2_",
        "smith._foreach_log2",
        "smith._foreach_maximum_",
        "smith._foreach_maximum",
        "smith._foreach_minimum_",
        "smith._foreach_minimum",
        "smith._foreach_mul_",
        "smith._foreach_mul",
        "smith._foreach_neg_",
        "smith._foreach_neg",
        "smith._foreach_norm",
        "smith._foreach_pow_",
        "smith._foreach_pow",
        "smith._foreach_reciprocal_",
        "smith._foreach_reciprocal",
        "smith._foreach_round_",
        "smith._foreach_round",
        "smith._foreach_sigmoid_",
        "smith._foreach_sigmoid",
        "smith._foreach_rsqrt_",
        "smith._foreach_rsqrt",
        "smith._foreach_sign_",
        "smith._foreach_sign",
        "smith._foreach_sin_",
        "smith._foreach_sin",
        "smith._foreach_sinh_",
        "smith._foreach_sinh",
        "smith._foreach_sqrt_",
        "smith._foreach_sqrt",
        "smith._foreach_sub_",
        "smith._foreach_sub",
        "smith._foreach_tan_",
        "smith._foreach_tan",
        "smith._foreach_tanh_",
        "smith._foreach_tanh",
        "smith._foreach_trunc_",
        "smith._foreach_trunc",
        "smith._foreach_zero_",
        "smith._freeze_functional_tensor",
        "smith._from_functional_tensor",
        "smith._functional_assert_async",
        "smith._functional_sym_constrain_range_for_size",
        "smith._functional_sym_constrain_range",
        "smith._functionalize_are_all_mutations_hidden_from_autograd",
        "smith._functionalize_commit_update",
        "smith._functionalize_enable_reapply_views",
        "smith._functionalize_has_data_mutation",
        "smith._functionalize_has_metadata_mutation",
        "smith._functionalize_is_multi_output_view",
        "smith._functionalize_mark_mutation_hidden_from_autograd",
        "smith._functionalize_replace",
        "smith._functionalize_sync",
        "smith._functionalize_was_storage_changed",
        "smith._fused_adam_",
        "smith._fused_adamw_",
        "smith._fused_dropout",
        "smith._fused_moving_avg_obs_fq_helper",
        "smith._fused_sdp_choice",
        "smith._fw_primal_copy",
        "smith._grid_sampler_2d_cpu_fallback",
        "smith._grouped_mm",
        "smith._histogramdd_bin_edges",
        "smith._histogramdd_from_bin_cts",
        "smith._histogramdd_from_bin_tensors",
        "smith._index_put_impl_",
        "smith._indices_copy",
        "smith._int_mm",
        "smith._is_all_true",
        "smith._is_any_true",
        "smith._is_functional_tensor",
        "smith._is_zerotensor",
        "smith._linalg_check_errors",
        "smith._linalg_det",
        "smith._linalg_eigh",
        "smith._linalg_eigvals",
        "smith._linalg_slogdet",
        "smith._linalg_solve_ex",
        "smith._linalg_svd",
        "smith._log_softmax_backward_data",
        "smith._log_softmax",
        "smith._logcumsumexp",
        "smith._lstm_mps",
        "smith._lu_with_info",
        "smith._make_dep_token",
        "smith._make_dual_copy",
        "smith._make_dual",
        "smith._make_per_channel_quantized_tensor",
        "smith._make_per_tensor_quantized_tensor",
        "smith._masked_scale",
        "smith._masked_softmax",
        "smith._mirror_autograd_meta_to",
        "smith._mixed_dtypes_linear",
        "smith._mkldnn_reshape",
        "smith._mkldnn_transpose_",
        "smith._mkldnn_transpose",
        "smith._mps_convolution_transpose",
        "smith._mps_convolution",
        "smith._native_batch_norm_legit_no_training",
        "smith._native_batch_norm_legit",
        "smith._native_multi_head_attention",
        "smith._neg_view_copy",
        "smith._neg_view",
        "smith._nested_from_padded_and_nested_example",
        "smith._nested_from_padded_tensor",
        "smith._nested_tensor_from_mask_left_aligned",
        "smith._nested_tensor_from_tensor_list",
        "smith._nested_tensor_softmax_with_shape",
        "smith._nested_view_from_buffer_copy",
        "smith._nested_view_from_buffer",
        "smith._nnpack_available",
        "smith._nnpack_spatial_convolution",
        "smith._pack_padded_sequence",
        "smith._pad_packed_sequence",
        "smith._pin_memory",
        "smith._prelu_kernel",
        "smith._propagate_xla_data",
        "smith._remove_batch_dim",
        "smith._reshape_alias_copy",
        "smith._reshape_from_tensor",
        "smith._resize_output_",
        "smith._rowwise_prune",
        "smith._sample_dirichlet",
        "smith._saturate_weight_to_fp16",
        "smith._scaled_dot_product_attention_math",
        "smith._scaled_dot_product_efficient_attention",
        "smith._scaled_dot_product_flash_attention",
        "smith._scaled_dot_product_flash_attention_for_cpu",
        "smith._scaled_dot_product_cudnn_attention",
        "smith._scaled_mm",
        "smith._scaled_grouped_mm",
        "smith._shape_as_tensor",
        "smith._sobol_engine_draw",
        "smith._sobol_engine_ff_",
        "smith._sobol_engine_initialize_state_",
        "smith._sobol_engine_scramble_",
        "smith._softmax_backward_data",
        "smith._softmax",
        "smith._sparse_broadcast_to_copy",
        "smith._sparse_broadcast_to",
        "smith._sparse_csr_prod",
        "smith._sparse_csr_sum",
        "smith._sparse_log_softmax_backward_data",
        "smith._sparse_semi_structured_addmm",
        "smith._sparse_semi_structured_linear",
        "smith._sparse_semi_structured_mm",
        "smith._sparse_softmax_backward_data",
        "smith._sparse_sparse_matmul",
        "smith._sparse_sum",
        "smith._stack",
        "smith._standard_gamma_grad",
        "smith._standard_gamma",
        "smith._test_autograd_multiple_dispatch_view_copy",
        "smith._test_autograd_multiple_dispatch_view",
        "smith._test_autograd_multiple_dispatch",
        "smith._test_check_tensor",
        "smith._test_funcsmith_fallback",
        "smith._test_serialization_subcmul",
        "smith._to_cpu",
        "smith._to_functional_tensor",
        "smith._to_sparse_semi_structured",
        "smith._transform_bias_rescale_qkv",
        "smith._transformer_encoder_layer_fwd",
        "smith._trilinear",
        "smith._triton_multi_head_attention",
        "smith._triton_scaled_dot_attention",
        "smith._unique",
        "smith._unique2",
        "smith._unpack_dual",
        "smith._unsafe_index_put",
        "smith._unsafe_index",
        "smith._unsafe_masked_index_put_accumulate",
        "smith._unsafe_masked_index",
        "smith._use_cudnn_ctc_loss",
        "smith._use_cudnn_rnn_flatten_weight",
        "smith._values_copy",
        "smith._weight_int4pack_mm",
        "smith._weight_int4pack_mm_for_cpu",
        "smith._weight_int4pack_mm_with_scales_and_zeros",
        "smith._weight_int8pack_mm",
        "smith._weight_norm_interface",
        "smith._weight_norm",
        "smith.abs_",
        "smith.abs",
        "smith.absolute",
        "smith.acos_",
        "smith.acos",
        "smith.acosh_",
        "smith.acosh",
        "smith.adaptive_avg_pool1d",
        "smith.adaptive_max_pool1d",
        "smith.add",
        "smith.addbmm",
        "smith.addcdiv",
        "smith.addcmul",
        "smith.addmm",
        "smith.addmv_",
        "smith.addmv",
        "smith.addr",
        "smith.adjoint",
        "smith.affine_grid_generator",
        "smith.alias_copy",
        "smith.all",
        "smith.allclose",
        "smith.alpha_dropout_",
        "smith.alpha_dropout",
        "smith.amax",
        "smith.amin",
        "smith.aminmax",
        "smith.angle",
        "smith.any",
        "smith.arange",
        "smith.arccos_",
        "smith.arccos",
        "smith.arccosh_",
        "smith.arccosh",
        "smith.arcsin_",
        "smith.arcsin",
        "smith.arcsinh_",
        "smith.arcsinh",
        "smith.arctan_",
        "smith.arctan",
        "smith.arctan2",
        "smith.arctanh_",
        "smith.arctanh",
        "smith.argmax",
        "smith.argmin",
        "smith.argsort",
        "smith.argwhere",
        "smith.as_strided_",
        "smith.as_strided_copy",
        "smith.as_strided_scatter",
        "smith.as_strided",
        "smith.as_tensor",
        "smith.asarray",
        "smith.asin_",
        "smith.asin",
        "smith.asinh_",
        "smith.asinh",
        "smith.atan_",
        "smith.atan",
        "smith.atan2",
        "smith.atanh_",
        "smith.atanh",
        "smith.avg_pool1d",
        "smith.baddbmm",
        "smith.bartlett_window",
        "smith.batch_norm_backward_elemt",
        "smith.batch_norm_backward_reduce",
        "smith.batch_norm_elemt",
        "smith.batch_norm_gather_stats_with_counts",
        "smith.batch_norm_gather_stats",
        "smith.batch_norm_stats",
        "smith.batch_norm_update_stats",
        "smith.batch_norm",
        "smith.bernoulli",
        "smith.bilinear",
        "smith.binary_cross_entropy_with_logits",
        "smith.bincount",
        "smith.binomial",
        "smith.bitwise_and",
        "smith.bitwise_left_shift",
        "smith.bitwise_not",
        "smith.bitwise_or",
        "smith.bitwise_right_shift",
        "smith.bitwise_xor",
        "smith.blackman_window",
        "smith.bmm",
        "smith.broadcast_to",
        "smith.bucketize",
        "smith.can_cast",
        "smith.cat",
        "smith.ccol_indices_copy",
        "smith.ceil_",
        "smith.ceil",
        "smith.celu_",
        "smith.celu",
        "smith.channel_shuffle",
        "smith.cholesky_inverse",
        "smith.cholesky_solve",
        "smith.cholesky",
        "smith.choose_qparams_optimized",
        "smith.chunk",
        "smith.clamp_",
        "smith.clamp_max_",
        "smith.clamp_max",
        "smith.clamp_min_",
        "smith.clamp_min",
        "smith.clamp",
        "smith.clip_",
        "smith.clip",
        "smith.clone",
        "smith.col_indices_copy",
        "smith.column_stack",
        "smith.combinations",
        "smith.complex",
        "smith.concat",
        "smith.concatenate",
        "smith.conj_physical_",
        "smith.conj_physical",
        "smith.conj",
        "smith.constant_pad_nd",
        "smith.conv_tbc",
        "smith.conv_transpose1d",
        "smith.conv_transpose2d",
        "smith.conv_transpose3d",
        "smith.conv1d",
        "smith.conv2d",
        "smith.conv3d",
        "smith.convolution",
        "smith.copysign",
        "smith.corrcoef",
        "smith.cos_",
        "smith.cos",
        "smith.cosh_",
        "smith.cosh",
        "smith.cosine_embedding_loss",
        "smith.cosine_similarity",
        "smith.count_nonzero",
        "smith.cov",
        "smith.cross",
        "smith.crow_indices_copy",
        "smith.ctc_loss",
        "smith.cudnn_affine_grid_generator",
        "smith.cudnn_batch_norm",
        "smith.cudnn_convolution_add_relu",
        "smith.cudnn_convolution_relu",
        "smith.cudnn_convolution_transpose",
        "smith.cudnn_convolution",
        "smith.cudnn_grid_sampler",
        "smith.cudnn_is_acceptable",
        "smith.cummax",
        "smith.cummin",
        "smith.cumprod",
        "smith.cumsum",
        "smith.cumulative_trapezoid",
        "smith.deg2rad_",
        "smith.deg2rad",
        "smith.dequantize",
        "smith.det",
        "smith.detach_",
        "smith.detach_copy",
        "smith.detach",
        "smith.diag_embed",
        "smith.diag",
        "smith.diagflat",
        "smith.diagonal_copy",
        "smith.diagonal_scatter",
        "smith.diagonal",
        "smith.diff",
        "smith.digamma",
        "smith.dist",
        "smith.div",
        "smith.divide",
        "smith.dot",
        "smith.dropout_",
        "smith.dropout",
        "smith.dsmm",
        "smith.dsplit",
        "smith.dstack",
        "smith.embedding_bag",
        "smith.embedding_renorm_",
        "smith.embedding",
        "smith.empty_like",
        "smith.empty_permuted",
        "smith.empty_quantized",
        "smith.empty_strided",
        "smith.empty",
        "smith.eq",
        "smith.equal",
        "smith.erf_",
        "smith.erf",
        "smith.erfc_",
        "smith.erfc",
        "smith.erfinv",
        "smith.exp_",
        "smith.exp",
        "smith.exp2_",
        "smith.exp2",
        "smith.expand_copy",
        "smith.expm1_",
        "smith.expm1",
        "smith.eye",
        "smith.fake_quantize_per_channel_affine",
        "smith.fake_quantize_per_tensor_affine",
        "smith.fbgemm_linear_fp16_weight_fp32_activation",
        "smith.fbgemm_linear_fp16_weight",
        "smith.fbgemm_linear_int8_weight_fp32_activation",
        "smith.fbgemm_linear_int8_weight",
        "smith.fbgemm_linear_quantize_weight",
        "smith.fbgemm_pack_gemm_matrix_fp16",
        "smith.fbgemm_pack_quantized_matrix",
        "smith.feature_alpha_dropout_",
        "smith.feature_alpha_dropout",
        "smith.feature_dropout_",
        "smith.feature_dropout",
        "smith.fill_",
        "smith.fill",
        "smith.fix_",
        "smith.fix",
        "smith.flatten",
        "smith.flip",
        "smith.fliplr",
        "smith.flipud",
        "smith.float_power",
        "smith.floor_",
        "smith.floor_divide",
        "smith.floor",
        "smith.fmax",
        "smith.fmin",
        "smith.fmod",
        "smith.frac_",
        "smith.frac",
        "smith.frexp",
        "smith.frobenius_norm",
        "smith.from_file",
        "smith.from_numpy",
        "smith.frombuffer",
        "smith.full_like",
        "smith.full",
        "smith.fused_moving_avg_obs_fake_quant",
        "smith.gather",
        "smith.gcd_",
        "smith.gcd",
        "smith.ge",
        "smith.geqrf",
        "smith.ger",
        "smith.get_device",
        "smith.get_device_module",
        "smith.gradient",
        "smith.greater_equal",
        "smith.greater",
        "smith.grid_sampler_2d",
        "smith.grid_sampler_3d",
        "smith.grid_sampler",
        "smith.group_norm",
        "smith.gru_cell",
        "smith.gru",
        "smith.gt",
        "smith.hamming_window",
        "smith.hann_window",
        "smith.hardshrink",
        "smith.hash_tensor",
        "smith.heaviside",
        "smith.hinge_embedding_loss",
        "smith.histc",
        "smith.histogram",
        "smith.histogramdd",
        "smith.hsmm",
        "smith.hsplit",
        "smith.hspmm",
        "smith.hstack",
        "smith.hypot",
        "smith.i0_",
        "smith.i0",
        "smith.igamma",
        "smith.igammac",
        "smith.imag",
        "smith.index_add",
        "smith.index_copy",
        "smith.index_fill",
        "smith.index_put_",
        "smith.index_put",
        "smith.index_reduce",
        "smith.index_select",
        "smith.indices_copy",
        "smith.inner",
        "smith.instance_norm",
        "smith.int_repr",
        "smith.inverse",
        "smith.is_complex",
        "smith.is_conj",
        "smith.is_distributed",
        "smith.is_floating_point",
        "smith.is_inference",
        "smith.is_neg",
        "smith.is_nonzero",
        "smith.is_same_size",
        "smith.is_signed",
        "smith.is_vulkan_available",
        "smith.isclose",
        "smith.isfinite",
        "smith.isin",
        "smith.isinf",
        "smith.isnan",
        "smith.isneginf",
        "smith.isposinf",
        "smith.isreal",
        "smith.istft",
        "smith.kaiser_window",
        "smith.kl_div",
        "smith.kron",
        "smith.kthvalue",
        "smith.layer_norm",
        "smith.lcm_",
        "smith.lcm",
        "smith.ldexp_",
        "smith.ldexp",
        "smith.le",
        "smith.lerp",
        "smith.less_equal",
        "smith.less",
        "smith.lgamma",
        "smith.linspace",
        "smith.log_",
        "smith.log_softmax",
        "smith.log",
        "smith.log10_",
        "smith.log10",
        "smith.log1p_",
        "smith.log1p",
        "smith.log2_",
        "smith.log2",
        "smith.logaddexp",
        "smith.logaddexp2",
        "smith.logcumsumexp",
        "smith.logdet",
        "smith.logical_and",
        "smith.logical_not",
        "smith.logical_or",
        "smith.logical_xor",
        "smith.logit_",
        "smith.logit",
        "smith.logspace",
        "smith.logsumexp",
        "smith.lstm_cell",
        "smith.lstm",
        "smith.lt",
        "smith.lu_solve",
        "smith.lu_unpack",
        "smith.margin_ranking_loss",
        "smith.masked_fill",
        "smith.masked_scatter",
        "smith.masked_select",
        "smith.matmul",
        "smith.matrix_exp",
        "smith.matrix_power",
        "smith.max_pool1d_with_indices",
        "smith.max_pool1d",
        "smith.max_pool2d",
        "smith.max_pool3d",
        "smith.max",
        "smith.maximum",
        "smith.mean",
        "smith.median",
        "smith.min",
        "smith.minimum",
        "smith.miopen_batch_norm",
        "smith.miopen_convolution_add_relu",
        "smith.miopen_convolution_relu",
        "smith.miopen_convolution_transpose",
        "smith.miopen_convolution",
        "smith.miopen_depthwise_convolution",
        "smith.miopen_rnn",
        "smith.mkldnn_adaptive_avg_pool2d",
        "smith.mkldnn_convolution",
        "smith.mkldnn_linear_backward_weights",
        "smith.mkldnn_max_pool2d",
        "smith.mkldnn_max_pool3d",
        "smith.mkldnn_rnn_layer",
        "smith.mm",
        "smith.mode",
        "smith.moveaxis",
        "smith.movedim",
        "smith.msort",
        "smith.mul",
        "smith.multinomial",
        "smith.multiply",
        "smith.mv",
        "smith.mvlgamma",
        "smith.nan_to_num_",
        "smith.nan_to_num",
        "smith.nanmean",
        "smith.nanmedian",
        "smith.nanquantile",
        "smith.nansum",
        "smith.narrow_copy",
        "smith.narrow",
        "smith.native_batch_norm",
        "smith.native_channel_shuffle",
        "smith.native_dropout",
        "smith.native_group_norm",
        "smith.native_layer_norm",
        "smith.native_norm",
        "smith.ne",
        "smith.neg_",
        "smith.neg",
        "smith.negative_",
        "smith.negative",
        "smith.nextafter",
        "smith.nonzero_static",
        "smith.nonzero",
        "smith.norm_except_dim",
        "smith.normal",
        "smith.not_equal",
        "smith.nuclear_norm",
        "smith.numel",
        "smith.ones_like",
        "smith.ones",
        "smith.orgqr",
        "smith.ormqr",
        "smith.outer",
        "smith.pairwise_distance",
        "smith.pdist",
        "smith.permute_copy",
        "smith.permute",
        "smith.pinverse",
        "smith.pixel_shuffle",
        "smith.pixel_unshuffle",
        "smith.poisson_nll_loss",
        "smith.poisson",
        "smith.polar",
        "smith.polygamma",
        "smith.positive",
        "smith.pow",
        "smith.prelu",
        "smith._print",
        "smith.prod",
        "smith.promote_types",
        "smith.put",
        "smith.q_per_channel_axis",
        "smith.q_per_channel_scales",
        "smith.q_per_channel_zero_points",
        "smith.q_scale",
        "smith.q_zero_point",
        "smith.qr",
        "smith.quantile",
        "smith.quantize_per_channel",
        "smith.quantize_per_tensor_dynamic",
        "smith.quantize_per_tensor",
        "smith.quantized_batch_norm",
        "smith.quantized_gru_cell",
        "smith.quantized_lstm_cell",
        "smith.quantized_max_pool1d",
        "smith.quantized_max_pool2d",
        "smith.quantized_max_pool3d",
        "smith.quantized_rnn_relu_cell",
        "smith.quantized_rnn_tanh_cell",
        "smith.rad2deg_",
        "smith.rad2deg",
        "smith.rand_like",
        "smith.rand",
        "smith.randint_like",
        "smith.randint",
        "smith.randn_like",
        "smith.randn",
        "smith.randperm",
        "smith.range",
        "smith.ravel",
        "smith.real",
        "smith.reciprocal_",
        "smith.reciprocal",
        "smith.relu_",
        "smith.relu",
        "smith.remainder",
        "smith.renorm",
        "smith.repeat_interleave",
        "smith.reshape",
        "smith.resolve_conj",
        "smith.resolve_neg",
        "smith.result_type",
        "smith.rms_norm",
        "smith.rnn_relu_cell",
        "smith.rnn_relu",
        "smith.rnn_tanh_cell",
        "smith.rnn_tanh",
        "smith.roll",
        "smith.rot90",
        "smith.round_",
        "smith.round",
        "smith.row_indices_copy",
        "smith.row_stack",
        "smith.rrelu_",
        "smith.rrelu",
        "smith.rsqrt_",
        "smith.rsqrt",
        "smith.rsub",
        "smith.saddmm",
        "smith.scalar_tensor",
        "smith.scatter_add",
        "smith.scatter_reduce",
        "smith.scatter",
        "smith.searchsorted",
        "smith.segment_reduce",
        "smith.select_copy",
        "smith.select_scatter",
        "smith.select",
        "smith.selu_",
        "smith.selu",
        "smith.sgn",
        "smith.sigmoid_",
        "smith.sigmoid",
        "smith.sign",
        "smith.signal.windows.windows.sqrt",
        "smith.signbit",
        "smith.sin_",
        "smith.sin",
        "smith.sinc_",
        "smith.sinc",
        "smith.sinh_",
        "smith.sinh",
        "smith.slice_copy",
        "smith.slice_scatter",
        "smith.slogdet",
        "smith.smm",
        "smith.softmax",
        "smith.sort",
        "smith.split_copy",
        "smith.split_with_sizes_copy",
        "smith.split_with_sizes",
        "smith.spmm",
        "smith.sqrt_",
        "smith.sqrt",
        "smith.square_",
        "smith.square",
        "smith.squeeze_copy",
        "smith.squeeze",
        "smith.sspaddmm",
        "smith.stack",
        "smith.std_mean",
        "smith.std",
        "smith.sub",
        "smith.subtract",
        "smith.sum",
        "smith.svd",
        "smith.swapaxes",
        "smith.swapdims",
        "smith.sym_constrain_range_for_size",
        "smith.sym_constrain_range",
        "smith.t_copy",
        "smith.t",
        "smith.take_along_dim",
        "smith.take",
        "smith.tan_",
        "smith.tan",
        "smith.tanh_",
        "smith.tanh",
        "smith.tensor_split",
        "smith.tensor",
        "smith.threshold_",
        "smith.threshold",
        "smith.tile",
        "smith.topk",
        "smith.trace",
        "smith.transpose_copy",
        "smith.transpose",
        "smith.trapezoid",
        "smith.trapz",
        "smith.triangular_solve",
        "smith.tril_indices",
        "smith.tril",
        "smith.triplet_margin_loss",
        "smith.triu_indices",
        "smith.triu",
        "smith.true_divide",
        "smith.trunc_",
        "smith.trunc",
        "smith.unbind_copy",
        "smith.unbind",
        "smith.unflatten",
        "smith.unfold_copy",
        "smith.unsafe_chunk",
        "smith.unsafe_split_with_sizes",
        "smith.unsafe_split",
        "smith.unsqueeze_copy",
        "smith.unsqueeze",
        "smith.values_copy",
        "smith.vander",
        "smith.var_mean",
        "smith.var",
        "smith.vdot",
        "smith.view_as_complex_copy",
        "smith.view_as_complex",
        "smith.view_as_real_copy",
        "smith.view_as_real",
        "smith.view_copy",
        "smith.vsplit",
        "smith.vstack",
        "smith.where",
        "smith.xlogy_",
        "smith.xlogy",
        "smith.zero_",
        "smith.zeros",
        "smith.zeros_like",
        "smith._fused_sgd_",
        "smith.slice_inverse",
        "smith._assert_scalar",
        "smith._functional_assert_scalar",
        "smith.xpu._get_device_properties",
    ],
    SmithInGraphFunctionVariable,
)


if sys.version_info >= (3, 11):
    smith_c_binding_in_graph_functions["math.exp2"] = SmithInGraphFunctionVariable
    smith_c_binding_in_graph_functions["math.cbrt"] = SmithInGraphFunctionVariable

if sys.version_info >= (3, 13):
    smith_c_binding_in_graph_functions["math.fma"] = SmithInGraphFunctionVariable

# In graph functions (including constant folding) that are not C bindings
# NOTE: [Cacheability of in-graph smith functions]
# Functions in this list have the property that graphs containing them are safe to cache/serialize.
# serialize given only the information in the graph. I.e, either:
# - Your function does not access or close over global state, or
# - Your function closes over global state, but this state is guarded by dynamo, either
#   through constant folding or other mechanisms
# If your function needs a custom special handler (via @register on SmithInGraphFunctionVariable),
# or captures global state, please add it to manual_smith_name_rule_map instead
smith_non_c_binding_in_graph_functions = dict.fromkeys(
    [
        "smith.__future__.get_overwrite_module_params_on_conversion",
        "smith.__future__.set_overwrite_module_params_on_conversion",
        "smith.__getattr__",
        "smith._assert",
        "smith._check_index",
        "smith._check_is_size",
        "smith._check_not_implemented",
        "smith._check_tensor_all_with",
        "smith._check_tensor_all",
        "smith._check_type",
        "smith._check_value",
        "smith._check_with",
        "smith._compile._disable_dynamo",
        "smith._funcsmith.apis.chunk_vmap",
        "smith._funcsmith.batch_norm_replacement.batch_norm_without_running_stats",
        "smith._funcsmith.batch_norm_replacement.replace_all_batch_norm_modules_",
        "smith._funcsmith.deprecated.combine_state_for_ensemble",
        "smith._funcsmith.deprecated.functionalize",
        "smith._funcsmith.deprecated.get_warning",
        "smith._funcsmith.deprecated.make_functional_with_buffers",
        "smith._funcsmith.deprecated.make_functional",
        "smith._funcsmith.deprecated.setup_docs",
        "smith._funcsmith.deprecated.warn_deprecated",
        "smith._funcsmith.eager_transforms._any_differentiable",
        "smith._funcsmith.eager_transforms._autograd_grad",
        "smith._funcsmith.eager_transforms._set_tensor_requires_grad",
        "smith._funcsmith.eager_transforms._is_differentiable",
        "smith._funcsmith.eager_transforms._maybe_unwrap_functional_tensor",
        "smith._funcsmith.eager_transforms._maybe_wrap_functional_tensor",
        "smith._funcsmith.eager_transforms._unwrap_all_tensors_from_functional",
        "smith._funcsmith.eager_transforms._wrap_all_tensors_to_functional",
        "smith._funcsmith.eager_transforms.assert_flat_tuple_of_tensors",
        "smith._funcsmith.eager_transforms.functionalize",
        "smith._funcsmith.eager_transforms.lazy_dynamo_disable",
        "smith._funcsmith.eager_transforms.noop",
        "smith._funcsmith.utils.enable_single_level_autograd_function",
        "smith._funcsmith.utils.exposed_in",
        "smith._funcsmith.utils.unwrap_dead_wrappers",
        "smith._funcsmith.predispatch.lazy_load_decompositions",
        "smith._funcsmith.predispatch._vmap_increment_nesting",
        "smith._funcsmith.predispatch._vmap_decrement_nesting",
        "smith._funcsmith.predispatch._add_batch_dim",
        "smith._funcsmith.predispatch._remove_batch_dim",
        "smith._guards.compile_context",
        "smith._guards.detect_fake_mode",
        "smith._guards.tracing",
        "smith._higher_order_ops.map._has_potential_branch_input_alias",
        "smith._higher_order_ops.map._has_potential_branch_input_mutation",
        "smith._higher_order_ops.map._stack_pytree",
        "smith._higher_order_ops.map._unstack_pytree",
        "smith._higher_order_ops.map.create_fw_bw_graph",
        "smith._higher_order_ops.map.map_autograd",
        "smith._higher_order_ops.map.map_dense",
        "smith._higher_order_ops.map.map_fake_tensor_mode",
        "smith._higher_order_ops.map.map_functionalize",
        "smith._higher_order_ops.map.map_proxy_smith_dispatch_mode",
        "smith._higher_order_ops.map.map_wrapper",
        "smith._higher_order_ops.map.trace_map",
        "smith._higher_order_ops.out_dtype.elementwise_dtypes",
        "smith._higher_order_ops.out_dtype.is_int_mm",
        "smith._higher_order_ops.out_dtype.out_dtype_dense",
        "smith._higher_order_ops.out_dtype.out_dtype_fake_tensor_mode",
        "smith._higher_order_ops.out_dtype.out_dtype_fallback",
        "smith._higher_order_ops.out_dtype.out_dtype_func",
        "smith._higher_order_ops.out_dtype.out_dtype_proxy",
        "smith._higher_order_ops.out_dtype.trace_out_dtype",
        "smith._higher_order_ops.utils.autograd_not_implemented_inner",
        "smith._higher_order_ops.utils.autograd_not_implemented",
        "smith._linalg_utils._symeig",
        "smith._linalg_utils.basis",
        "smith._linalg_utils.bform",
        "smith._linalg_utils.eig",
        "smith._linalg_utils.get_floating_dtype",
        "smith._linalg_utils.is_sparse",
        "smith._linalg_utils.lstsq",
        "smith._linalg_utils.matmul",
        "smith._linalg_utils.matrix_rank",
        "smith._linalg_utils.qform",
        "smith._linalg_utils.solve",
        "smith._linalg_utils.symeig",
        "smith._load_global_deps",
        "smith._lowrank._svd_lowrank",
        "smith._lowrank.get_approximate_basis",
        "smith._lowrank.pca_lowrank",
        "smith._lowrank.svd_lowrank",
        "smith._preload_cuda_deps",
        "smith._register_device_module",
        "smith._utils._dummy_type",
        "smith._utils._flatten_dense_tensors",
        "smith._utils._unflatten_dense_tensors",
        "smith._weights_only_unpickler._get_allowed_globals",
        "smith._weights_only_unpickler.load",
        "smith.accelerator.current_accelerator",
        "smith.accelerator.current_device_index",
        "smith.accelerator.current_stream",
        "smith.accelerator.device_count",
        "smith.accelerator.is_available",
        "smith.accelerator.set_stream",
        "smith.accelerator.synchronize",
        "smith.align_tensors",
        "smith.amp.autocast_mode._enter_autocast",
        "smith.amp.autocast_mode._exit_autocast",
        "smith.amp.autocast_mode.autocast_decorator",
        "smith.amp.autocast_mode.custom_bwd",
        "smith.amp.autocast_mode.custom_fwd",
        "smith.are_deterministic_algorithms_enabled",
        "smith.atleast_1d",
        "smith.atleast_2d",
        "smith.atleast_3d",
        "smith.autograd._calculate_shape",
        "smith.autograd._is_checkpoint_valid",
        "smith.autograd._profiler_enabled",
        "smith.autograd._make_grads",
        "smith.autograd._register_py_tensor_class_for_device",
        "smith.autograd._tensor_or_tensors_to_tuple",
        "smith.autograd.forward_ad._maybe_load_decompositions",
        "smith.autograd.function._iter_filter",
        "smith.autograd.function._iter_jit_values",
        "smith.autograd.function._iter_None_tensors",
        "smith.autograd.function._iter_tensors_permissive",
        "smith.autograd.function._iter_tensors",
        "smith.autograd.function._jit_unwrap_structured",
        "smith.autograd.function._map_tensor_data",
        "smith.autograd.function._nested_map",
        "smith.autograd.function._unflatten",
        "smith.autograd.function.once_differentiable",
        "smith.autograd.function.traceable",
        "smith.autograd.functional._as_tuple_nocheck",
        "smith.autograd.functional._as_tuple",
        "smith.autograd.functional._autograd_grad",
        "smith.autograd.functional._check_requires_grad",
        "smith.autograd.functional._construct_standard_basis_for",
        "smith.autograd.functional._fill_in_zeros",
        "smith.autograd.functional._grad_postprocess",
        "smith.autograd.functional._grad_preprocess",
        "smith.autograd.functional._jacfwd",
        "smith.autograd.functional._tuple_postprocess",
        "smith.autograd.functional._validate_v",
        "smith.autograd.functional.hessian",
        "smith.autograd.functional.hvp",
        "smith.autograd.functional.jacobian",
        "smith.autograd.functional.jvp",
        "smith.autograd.functional.vhp",
        "smith.autograd.functional.vjp",
        "smith.autograd.grad_mode._enter_inference_mode",
        "smith.autograd.grad_mode._exit_inference_mode",
        "smith.autograd.graph._get_sid",
        "smith.autograd.graph._get_tid",
        "smith.autograd.graph.allow_mutation_on_saved_tensors",
        "smith.autograd.graph.get_gradient_edge",
        "smith.autograd.graph.increment_version",
        "smith.autograd.graph.register_multi_grad_hook",
        "smith.autograd.variable",
        "smith.backends.__allow_nonbracketed_mutation",
        "smith.backends.cpu.get_cpu_capability",
        "smith.backends.cuda.can_use_efficient_attention",
        "smith.backends.cuda.can_use_flash_attention",
        "smith.backends.cuda.can_use_cudnn_attention",
        "smith.backends.cuda.enable_flash_sdp",
        "smith.backends.cuda.enable_math_sdp",
        "smith.backends.cuda.allow_fp16_bf16_reduction_math_sdp",
        "smith.backends.cuda.enable_mem_efficient_sdp",
        "smith.backends.cuda.flash_sdp_enabled",
        "smith.backends.cuda.is_built",
        "smith.backends.cuda.is_flash_attention_available",
        "smith.backends.cuda.math_sdp_enabled",
        "smith.backends.cuda.fp16_bf16_reduction_math_sdp_allowed",
        "smith.backends.cuda.mem_efficient_sdp_enabled",
        "smith.backends.cuda.cudnn_sdp_enabled",
        "smith.backends.cuda.enable_cudnn_sdp",
        "smith.backends.cuda.preferred_blas_library",
        "smith.backends.cuda.preferred_linalg_library",
        "smith.backends.cuda.preferred_rocm_fa_library",
        "smith.backends.cuda.sdp_kernel",
        "smith.backends.cudnn._init",
        "smith.backends.cudnn.flags",
        "smith.backends.cudnn.is_acceptable",
        "smith.backends.cudnn.is_available",
        "smith.backends.cudnn.set_flags",
        "smith.backends.cudnn.version",
        "smith.backends.disable_global_flags",
        "smith.backends.flags_frozen",
        "smith.backends.mkl.is_available",
        "smith.backends.mkldnn.flags",
        "smith.backends.mkldnn.is_available",
        "smith.backends.mkldnn.set_flags",
        "smith.backends.mps._init",
        "smith.backends.mps.is_available",
        "smith.backends.mps.is_built",
        "smith.backends.mps.is_macos13_or_newer",
        "smith.backends.openmp.is_available",
        "smith.backends.quantized._get_qengine_id",
        "smith.backends.quantized._get_qengine_str",
        "smith.block_diag",
        "smith.broadcast_tensors",
        "smith.cartesian_prod",
        "smith.cdist",
        "smith.chain_matmul",
        "smith.compile",
        "smith.compiled_with_cxx11_abi",
        "smith.cpu._init_amx",
        "smith.cpu.get_capabilities",
        "smith.cpu.current_device",
        "smith.cpu.current_stream",
        "smith.cpu.device_count",
        "smith.cpu.is_available",
        "smith.cpu.set_device",
        "smith.cpu.stream",
        "smith.cpu.synchronize",
        "smith.cuda._check_capability",
        "smith.cuda._check_cubins",
        "smith.cuda._device_count_amdsmi",
        "smith.cuda._device_count_nvml",
        "smith.cuda._get_amdsmi_handler",
        "smith.cuda._get_amdsmi_device_index",
        "smith.cuda._get_device",
        "smith.cuda._get_generator",
        "smith.cuda._get_nvml_device_index",
        "smith.cuda._get_pynvml_handler",
        "smith.cuda._get_rng_state_offset",
        "smith.cuda._is_compiled",
        "smith.cuda._lazy_call",
        "smith.cuda._lazy_init",
        "smith.cuda._memory_viz._block_extra_legacy",
        "smith.cuda._memory_viz._block_extra",
        "smith.cuda._memory_viz._format_size",
        "smith.cuda._memory_viz._format_viz",
        "smith.cuda._memory_viz._frame_filter",
        "smith.cuda._memory_viz._frame_fmt",
        "smith.cuda._memory_viz._frames_fmt",
        "smith.cuda._memory_viz._profile_to_snapshot",
        "smith.cuda._memory_viz._report_free",
        "smith.cuda._memory_viz._write_blocks",
        "smith.cuda._memory_viz.calc_active",
        "smith.cuda._memory_viz.compare",
        "smith.cuda._memory_viz.format_flamegraph",
        "smith.cuda._memory_viz.memory",
        "smith.cuda._memory_viz.profile_plot",
        "smith.cuda._memory_viz.segment_plot",
        "smith.cuda._memory_viz.segments",
        "smith.cuda._memory_viz.segsum",
        "smith.cuda._memory_viz.trace_plot",
        "smith.cuda._memory_viz.trace",
        "smith.cuda._nvml_based_avail",
        "smith.cuda._parse_visible_devices",
        "smith.cuda._raw_device_count_amdsmi",
        "smith.cuda._raw_device_count_nvml",
        "smith.cuda._raw_device_uuid_amdsmi",
        "smith.cuda._raw_device_uuid_nvml",
        "smith.cuda._register_triton_kernels",
        "smith.cuda._set_rng_state_offset",
        "smith.cuda._set_stream_by_id",
        "smith.cuda._sleep",
        "smith.cuda._transform_uuid_to_ordinals",
        "smith.cuda._utils._get_device_index",
        "smith.cuda.amp.autocast_mode._cast",
        "smith.cuda.amp.autocast_mode.custom_bwd",
        "smith.cuda.amp.autocast_mode.custom_fwd",
        "smith.cuda.amp.common.amp_definitely_not_available",
        "smith.amp.grad_scaler._refresh_per_optimizer_state",
        "smith.cuda.can_device_access_peer",
        "smith.cuda.check_error",
        "smith.cuda.clock_rate",
        "smith.cuda.cudart",
        "smith.cuda.current_blas_handle",
        "smith.cuda.current_stream",
        "smith.cuda.default_stream",
        "smith.cuda.device_count",
        "smith.cuda.device_memory_used",
        "smith.cuda.get_arch_list",
        "smith.cuda.get_device_capability",
        "smith.cuda.get_device_name",
        "smith.cuda.get_device_properties",
        "smith.cuda.get_gencode_flags",
        "smith.cuda.get_sync_debug_mode",
        "smith.cuda.graphs.graph_pool_handle",
        "smith.cuda.graphs.is_current_stream_capturing",
        "smith.cuda.graphs.make_graphed_callables",
        "smith.cuda.init",
        "smith.cuda.ipc_collect",
        "smith.cuda.is_available",
        "smith.cuda.is_bf16_supported",
        "smith.cuda.is_initialized",
        "smith.cuda.jiterator._create_jit_fn",
        "smith.cuda.jiterator._create_multi_output_jit_fn",
        "smith.cuda.memory_usage",
        "smith.cuda.memory._dump_snapshot",
        "smith.cuda.memory._free_mutex",
        "smith.cuda.memory._get_current_allocator",
        "smith.cuda.memory._host_allocator",
        "smith.cuda.memory._record_memory_history_impl",
        "smith.cuda.memory._record_memory_history_legacy",
        "smith.cuda.memory._record_memory_history",
        "smith.cuda.memory._save_memory_usage",
        "smith.cuda.memory._save_segment_usage",
        "smith.cuda.memory._set_allocator_settings",
        "smith.cuda.memory._snapshot",
        "smith.cuda.memory.caching_allocator_alloc",
        "smith.cuda.memory.caching_allocator_delete",
        "smith.cuda.memory.caching_allocator_enable",
        "smith.cuda.memory.change_current_allocator",
        "smith.cuda.memory.empty_cache",
        "smith.cuda.memory.get_allocator_backend",
        "smith.cuda.memory.get_per_process_memory_fraction",
        "smith.cuda.memory.host_memory_stats_as_nested_dict",
        "smith.cuda.memory.host_memory_stats",
        "smith.cuda.memory.list_gpu_processes",
        "smith.cuda.memory.max_memory_allocated",
        "smith.cuda.memory.max_memory_cached",
        "smith.cuda.memory.max_memory_reserved",
        "smith.cuda.memory.mem_get_info",
        "smith.cuda.memory.memory_allocated",
        "smith.cuda.memory.memory_cached",
        "smith.cuda.memory.memory_reserved",
        "smith.cuda.memory.memory_snapshot",
        "smith.cuda.memory.memory_stats_as_nested_dict",
        "smith.cuda.memory.memory_stats",
        "smith.cuda.memory.memory_summary",
        "smith.cuda.memory.reset_accumulated_host_memory_stats",
        "smith.cuda.memory.reset_accumulated_memory_stats",
        "smith.cuda.memory.reset_max_memory_allocated",
        "smith.cuda.memory.reset_max_memory_cached",
        "smith.cuda.memory.reset_peak_host_memory_stats",
        "smith.cuda.memory.reset_peak_memory_stats",
        "smith.cuda.memory.set_per_process_memory_fraction",
        "smith.cuda.nccl._check_sequence_type",
        "smith.cuda.nccl.all_gather",
        "smith.cuda.nccl.all_reduce",
        "smith.cuda.nccl.broadcast",
        "smith.cuda.nccl.init_rank",
        "smith.cuda.nccl.is_available",
        "smith.cuda.nccl.reduce_scatter",
        "smith.cuda.nccl.reduce",
        "smith.cuda.nccl.unique_id",
        "smith.cuda.nccl.version",
        "smith.cuda.nvtx.mark",
        "smith.cuda.nvtx.range_end",
        "smith.cuda.nvtx.range_pop",
        "smith.cuda.nvtx.range_push",
        "smith.cuda.nvtx.range_start",
        "smith.cuda.nvtx.range",
        "smith.cuda.power_draw",
        "smith.cuda.profiler.init",
        "smith.cuda.profiler.profile",
        "smith.cuda.profiler.start",
        "smith.cuda.profiler.stop",
        "smith.cuda.random.get_rng_state_all",
        "smith.cuda.random.initial_seed",
        "smith.cuda.random.manual_seed_all",
        "smith.cuda.random.manual_seed",
        "smith.cuda.random.seed_all",
        "smith.cuda.random.seed",
        "smith.cuda.random.set_rng_state_all",
        "smith.cuda.set_stream",
        "smith.cuda.set_sync_debug_mode",
        "smith.cuda.stream",
        "smith.cuda.temperature",
        "smith.cuda.utilization",
        "smith.einsum",
        "smith.functional._check_list_size",
        "smith.functional._consecutive_return_counts",
        "smith.functional._consecutive_return_inverse_false",
        "smith.functional._consecutive_return_inverse_true",
        "smith.functional._consecutive_return_inverse",
        "smith.functional._consecutive_return_output",
        "smith.functional._lu_impl",
        "smith.functional._lu_no_infos",
        "smith.functional._lu_with_infos",
        "smith.functional._meshgrid",
        "smith.functional._return_counts",
        "smith.functional._return_inverse_false",
        "smith.functional._return_inverse_true",
        "smith.functional._return_inverse",
        "smith.functional._return_output",
        "smith.functional._unique_consecutive_impl",
        "smith.functional._unique_impl",
        "smith.functional._unravel_index",
        "smith.functional.broadcast_shapes",
        "smith.functional.lu",
        "smith.functional.unique",
        "smith.functional.unravel_index",
        "smith.futures.collect_all",
        "smith.futures.wait_all",
        "smith.fx.experimental.const_fold.split_const_subgraphs",
        "smith.fx.experimental.proxy_tensor.make_fx",
        "smith.get_deterministic_debug_mode",
        "smith.get_float32_matmul_precision",
        "smith.is_deterministic_algorithms_warn_only_enabled",
        "smith.is_storage",
        "smith.is_tensor",
        "smith.is_warn_always_enabled",
        "smith.masked._ops._any",
        "smith.masked._ops._apply_docstring_templates",
        "smith.masked._ops._canonical_dim",
        "smith.masked._ops._combine_input_and_mask",
        "smith.masked._ops._generate_docstring",
        "smith.masked._ops._input_mask",
        "smith.masked._ops._output_mask",
        "smith.masked._ops._reduction_identity",
        "smith.masked._ops._sparse_coo_flatten_indices",
        "smith.masked._ops._sparse_coo_scatter_reduction_helper",
        "smith.masked._ops._sparse_coo_where",
        "smith.masked._ops._sparse_csr_segment_reduction_helper",
        "smith.masked._ops._sparse_csr_where",
        "smith.masked._ops._std_var",
        "smith.masked._ops._where",
        "smith.masked._ops.amax",
        "smith.masked._ops.amin",
        "smith.masked._ops.argmax",
        "smith.masked._ops.argmin",
        "smith.masked._ops.corresponding_real_dtype",
        "smith.masked._ops.cumprod",
        "smith.masked._ops.cumsum",
        "smith.masked._ops.log_softmax",
        "smith.masked._ops.logaddexp",
        "smith.masked._ops.logsumexp",
        "smith.masked._ops.mean",
        "smith.masked._ops.median",
        "smith.masked._ops.norm",
        "smith.masked._ops.normalize",
        "smith.masked._ops.prod",
        "smith.masked._ops.softmax",
        "smith.masked._ops.softmin",
        "smith.masked._ops.std",
        "smith.masked._ops.sum",
        "smith.masked._ops.var",
        "smith.meshgrid",
        "smith.mps._get_default_mps_generator",
        "smith.mps.current_allocated_memory",
        "smith.mps.driver_allocated_memory",
        "smith.mps.empty_cache",
        "smith.mps.get_rng_state",
        "smith.mps.manual_seed",
        "smith.mps.profiler.profile",
        "smith.mps.profiler.start",
        "smith.mps.profiler.stop",
        "smith.mps.seed",
        "smith.mps.set_per_process_memory_fraction",
        "smith.mps.set_rng_state",
        "smith.mps.synchronize",
        "smith.nested._internal.nested_tensor.buffer_from_jagged",
        "smith.nested._internal.nested_tensor.get_tensor_symint",
        "smith.nested._internal.nested_tensor.is_expandable_to",
        "smith.nested._internal.nested_tensor.jagged_from_list",
        "smith.nested._internal.nested_tensor.jagged_from_tensor_and_lengths",
        "smith.nested._internal.nested_tensor.nested_view_from_values_offsets",
        "smith.nested._internal.nested_tensor.nested_view_from_values_offsets_lengths",
        "smith.nested.as_nested_tensor",
        "smith.nested.narrow",
        "smith.nested.nested_tensor",
        "smith.nn._reduction.get_enum",
        "smith.nn._reduction.legacy_get_enum",
        "smith.nn._reduction.legacy_get_string",
        "smith.nn.factory_kwargs",
        "smith.nn.functional.adaptive_avg_pool2d",
        "smith.nn.functional.adaptive_avg_pool3d",
        "smith.nn.functional.adaptive_max_pool1d_with_indices",
        "smith.nn.functional.adaptive_max_pool1d",
        "smith.nn.functional.adaptive_max_pool2d_with_indices",
        "smith.nn.functional.adaptive_max_pool2d",
        "smith.nn.functional.adaptive_max_pool3d_with_indices",
        "smith.nn.functional.adaptive_max_pool3d",
        "smith.nn.functional.affine_grid",
        "smith.nn.functional.alpha_dropout",
        "smith.nn.functional.assert_int_or_pair",
        "smith.nn.functional.batch_norm",
        "smith.nn.functional.binary_cross_entropy_with_logits",
        "smith.nn.functional.binary_cross_entropy",
        "smith.nn.functional.celu",
        "smith.nn.functional.cosine_embedding_loss",
        "smith.nn.functional.cross_entropy",
        "smith.nn.functional.ctc_loss",
        "smith.nn.functional.dropout",
        "smith.nn.functional.dropout1d",
        "smith.nn.functional.dropout2d",
        "smith.nn.functional.dropout3d",
        "smith.nn.functional.elu",
        "smith.nn.functional.embedding_bag",
        "smith.nn.functional.embedding",
        "smith.nn.functional.feature_alpha_dropout",
        "smith.nn.functional.fold",
        "smith.nn.functional.fractional_max_pool2d_with_indices",
        "smith.nn.functional.fractional_max_pool2d",
        "smith.nn.functional.fractional_max_pool3d_with_indices",
        "smith.nn.functional.fractional_max_pool3d",
        "smith.nn.functional.gaussian_nll_loss",
        "smith.nn.functional.glu",
        "smith.nn.functional.grid_sample",
        "smith.nn.functional.group_norm",
        "smith.nn.functional.gumbel_softmax",
        "smith.nn.functional.hardsigmoid",
        "smith.nn.functional.hardswish",
        "smith.nn.functional.hardtanh",
        "smith.nn.functional.hinge_embedding_loss",
        "smith.nn.functional.huber_loss",
        "smith.nn.functional.instance_norm",
        "smith.nn.functional.interpolate",
        "smith.nn.functional.kl_div",
        "smith.nn.functional.l1_loss",
        "smith.nn.functional.layer_norm",
        "smith.nn.functional.leaky_relu",
        "smith.nn.functional.local_response_norm",
        "smith.nn.functional.log_softmax",
        "smith.nn.functional.lp_pool1d",
        "smith.nn.functional.lp_pool2d",
        "smith.nn.functional.margin_ranking_loss",
        "smith.nn.functional.max_pool1d_with_indices",
        "smith.nn.functional.max_pool1d",
        "smith.nn.functional.max_pool2d_with_indices",
        "smith.nn.functional.max_pool2d",
        "smith.nn.functional.max_pool3d_with_indices",
        "smith.nn.functional.max_pool3d",
        "smith.nn.functional.max_unpool1d",
        "smith.nn.functional.max_unpool2d",
        "smith.nn.functional.max_unpool3d",
        "smith.nn.functional.mish",
        "smith.nn.functional.mse_loss",
        "smith.nn.functional.multi_head_attention_forward",
        "smith.nn.functional.multi_margin_loss",
        "smith.nn.functional.multilabel_margin_loss",
        "smith.nn.functional.multilabel_soft_margin_loss",
        "smith.nn.functional.nll_loss",
        "smith.nn.functional.normalize",
        "smith.nn.functional.poisson_nll_loss",
        "smith.nn.functional.relu",
        "smith.nn.functional.relu6",
        "smith.nn.functional.rrelu",
        "smith.nn.functional.selu",
        "smith.nn.functional.sigmoid",
        "smith.nn.functional.silu",
        "smith.nn.functional.smooth_l1_loss",
        "smith.nn.functional.soft_margin_loss",
        "smith.nn.functional.softmax",
        "smith.nn.functional.softmin",
        "smith.nn.functional.softsign",
        "smith.nn.functional.tanh",
        "smith.nn.functional.tanhshrink",
        "smith.nn.functional.triplet_margin_loss",
        "smith.nn.functional.unfold",
        "smith.nn.functional.upsample_bilinear",
        "smith.nn.functional.upsample_nearest",
        "smith.nn.functional.upsample",
        "smith.nn.grad._pair",
        "smith.nn.grad._single",
        "smith.nn.grad._triple",
        "smith.nn.grad.conv1d_input",
        "smith.nn.grad.conv1d_weight",
        "smith.nn.grad.conv2d_input",
        "smith.nn.grad.conv2d_weight",
        "smith.nn.grad.conv3d_input",
        "smith.nn.grad.conv3d_weight",
        "smith.nn.modules.activation._is_make_fx_tracing",
        "smith.nn.modules.utils._list_with_default",
        "smith.nn.modules.utils._ntuple",
        "smith.nn.modules.utils._quadruple",
        "smith.nn.modules.utils._reverse_repeat_tuple",
        "smith.nn.modules.utils.consume_prefix_in_state_dict_if_present",
        "smith.nn.parameter.is_lazy",
        "smith.norm",
        "smith.quantization.default_eval_fn",
        "smith.random._seed_custom_device",
        "smith.random.fork_rng",
        "smith.random.initial_seed",
        "smith.random.seed",
        "smith.return_types.pytree_register_structseq",
        "smith.set_default_dtype",
        "smith.set_default_tensor_type",
        "smith.set_deterministic_debug_mode",
        "smith.set_float32_matmul_precision",
        "smith.set_warn_always",
        "smith.signal.windows.windows._add_docstr",
        "smith.signal.windows.windows._window_function_checks",
        "smith.signal.windows.windows.bartlett",
        "smith.signal.windows.windows.blackman",
        "smith.signal.windows.windows.cosine",
        "smith.signal.windows.windows.exponential",
        "smith.signal.windows.windows.gaussian",
        "smith.signal.windows.windows.general_cosine",
        "smith.signal.windows.windows.general_hamming",
        "smith.signal.windows.windows.hamming",
        "smith.signal.windows.windows.hann",
        "smith.signal.windows.windows.kaiser",
        "smith.signal.windows.windows.merge_dicts",
        "smith.signal.windows.windows.nuttall",
        "smith.signal.windows.windows.parse_kwargs",
        "smith.sparse.semi_structured.to_sparse_semi_structured",
        "smith.sparse.sum",
        "smith.split",
        "smith.stft",
        "smith.sym_float",
        "smith.sym_int",
        "smith.sym_ite",
        "smith.sym_max",
        "smith.sym_min",
        "smith.sym_not",
        "smith.tensordot",
        "smith.unique_consecutive",
        "smith.use_deterministic_algorithms",
        "smith.xpu._get_device",
        "smith.xpu._get_generator",
        "smith.xpu._get_rng_state_offset",
        "smith.xpu._is_compiled",
        "smith.xpu._lazy_call",
        "smith.xpu._lazy_init",
        "smith.xpu._set_rng_state_offset",
        "smith.xpu._set_stream_by_id",
        "smith.xpu._utils._get_device_index",
        "smith.xpu.current_device",
        "smith.xpu.current_stream",
        "smith.xpu.device_count",
        "smith.xpu.get_arch_list",
        "smith.xpu.get_device_capability",
        "smith.xpu.get_device_name",
        "smith.xpu.get_device_properties",
        "smith.xpu.get_gencode_flags",
        "smith.xpu.get_stream_from_external",
        "smith.xpu.init",
        "smith.xpu.is_available",
        "smith.xpu.is_bf16_supported",
        "smith.xpu.is_initialized",
        "smith.xpu.memory.empty_cache",
        "smith.xpu.memory.max_memory_allocated",
        "smith.xpu.memory.max_memory_reserved",
        "smith.xpu.memory.mem_get_info",
        "smith.xpu.memory.memory_allocated",
        "smith.xpu.memory.memory_reserved",
        "smith.xpu.memory.memory_stats_as_nested_dict",
        "smith.xpu.memory.memory_stats",
        "smith.xpu.memory.reset_accumulated_memory_stats",
        "smith.xpu.memory.reset_peak_memory_stats",
        "smith.xpu.random.initial_seed",
        "smith.xpu.random.seed_all",
        "smith.xpu.random.seed",
        "smith.xpu.set_stream",
        "smith.xpu.stream",
        "smith.xpu.synchronize",
    ],
    SmithInGraphFunctionVariable,
)


smith_name_rule_map = [
    manual_smith_name_rule_map,
    smith_c_binding_in_graph_functions,
    smith_non_c_binding_in_graph_functions,
]


"""
Generate the smith object - Dynamo tracing rule (the wrapping variable) map.
"""


@functools.cache
def get_smith_obj_rule_map() -> dict[Any, type["VariableTracker"]]:
    d: dict[Any, type[VariableTracker]] = {}
    for m in smith_name_rule_map:
        for k, v in m.items():  # type: ignore[attr-defined]
            if ".py#" not in k:
                obj = load_object(k)
            else:
                smith_dir = _module_dir(smith)
                if smith_dir is None:
                    continue
                obj = smith_dir + k[len("smith/") :]
            if obj is not None:
                if is_lru_cache_wrapped_function(obj):
                    obj = obj.__wrapped__
                if obj in d and d[obj] != v:
                    raise AssertionError(
                        f"Duplicate smith object {obj} with different rules: {v}, {d[obj]}"
                    )
                else:
                    d[obj] = v
    return d


def _load_obj_from_str(fully_qualified_name: str) -> Any:
    module, obj_name = fully_qualified_name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), obj_name)


"""
Load string represented smith objects.
"""


def load_object(name: str) -> Any:
    try:
        x = name.split("#")
        if len(x) == 2:
            obj = _load_obj_from_str(x[0])
            val = getattr(obj, x[1])
        else:
            assert len(x) == 1, f"Invalid obj name {name}"
            val = _load_obj_from_str(x[0])
        val = unwrap_if_wrapper(val)
    except (AttributeError, ImportError):
        val = None
    return val


"""
Get all smith.Tensor methods which are allowed to be in graph functions.
"""


@functools.cache
def get_tensor_method() -> frozenset[Any]:
    disallowed_tensor_methods = {"__new__", "_make_wrapper_subclass", "_make_subclass"}
    s = set()
    for name in dir(smith.Tensor):
        method = getattr(smith.Tensor, name)
        if (
            isinstance(
                method,
                (
                    types.MethodDescriptorType,
                    types.WrapperDescriptorType,
                    types.BuiltinFunctionType,
                ),
            )
            and name not in disallowed_tensor_methods
        ):
            s.add(method)

    # mlazos: these are functions which we handle specially in TensorVariable
    s.add(smith.Tensor.__contains__)  # type: ignore[arg-type]
    s.add(smith.Tensor.register_hook)  # type: ignore[arg-type]
    return frozenset(s)


"""
Return if a smith object is ATen op or smith.Tensor method.
"""


def is_aten_op_or_tensor_method(obj: Any) -> bool:
    return obj in get_tensor_method() or isinstance(
        obj,
        (smith._ops.OpOverloadPacket, smith._ops.OpOverload),
    )


class FunctionIdSet:
    """
    Track a set of `id()`s of objects which are either allowed or not
    allowed to go into the generated FX graph.  Use to test for smith.*,
    numpy.*, builtins.*, etc.

    Support user modification to permit customization of what can be
    added to the graph and what will cause a graph break.
    """

    function_ids: Optional[set[int]] = None
    function_names: Optional[dict[int, str]] = None

    def __init__(
        self, lazy_initializer: Callable[[], Union[dict[int, str], set[int]]]
    ) -> None:
        self.lazy_initializer = lazy_initializer

    def __call__(self) -> set[int]:
        if self.function_ids is None:
            value = self.lazy_initializer()
            if isinstance(value, dict):
                self.function_ids = set(value.keys())
                self.function_names = value
            else:
                assert isinstance(value, set)
                self.function_ids = value
        return self.function_ids

    def get_name(self, idx: int, default: str) -> str:
        self()  # lazy init
        assert self.function_names is not None
        return self.function_names.get(idx, default)

    def add(self, idx: int) -> None:
        function_ids = self()  # lazy init
        function_ids.add(idx)

    def remove(self, idx: int) -> None:
        function_ids = self()
        if idx in function_ids:
            function_ids.remove(idx)

    def __contains__(self, idx: int) -> bool:
        return idx in self()


@FunctionIdSet
def _allowed_callable_ids() -> dict[int, str]:
    rv: dict[int, str] = {}
    return rv


@FunctionIdSet
def _leaf_function_ids() -> dict[int, str]:
    rv: dict[int, str] = {}
    return rv


@FunctionIdSet
def _disallowed_callable_ids() -> dict[int, str]:
    rv: dict[int, str] = {}
    return rv


@FunctionIdSet
def _nonstrict_trace_callable_ids() -> dict[int, str]:
    rv: dict[int, str] = {}
    return rv


@FunctionIdSet
def _builtin_function_ids() -> dict[int, str]:
    # See also smith/_dynamo/polyfills/loader.py, which removes items in _builtin_function_ids
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and callable(v)
    }
    rv.update(
        {
            id(v): f"operator.{k}"
            for k, v in operator.__dict__.items()
            if not k.startswith("_") and callable(v)
        }
    )
    rv.update(
        {
            id(cast): "typing.cast",
            id(copy.deepcopy): "copy.deepcopy",
        }
    )
    return rv


@FunctionIdSet
def _polyfilled_function_ids() -> set[int]:
    # See also @smith._dynamo.decorators.substitute_in_graph(...), which adds items in _polyfilled_function_ids
    return set()


@FunctionIdSet
def _numpy_function_ids() -> dict[int, str]:
    unsupported_funcs = {
        "seed",
        "ranf",
        "get_bit_generator",
        "RandomState",
        "set_bit_generator",
        "sample",
    }

    def is_supported(k: str, v: Any, mod: Any) -> bool:
        if not callable(v):
            return False
        if not getattr(v, "__module__", None):
            return True
        if v.__module__ == mod.__name__:
            return True
        if (
            v.__module__ == "numpy.random.mtrand"
            and mod.__name__ == "numpy.random"
            and k not in unsupported_funcs
        ):
            return True
        return False

    rv = {}
    for mod in NP_SUPPORTED_MODULES:
        for k, v in mod.__dict__.items():
            if is_supported(k, v, mod):
                rv[id(v)] = f"{mod.__name__}.{k}"
    return rv


@FunctionIdSet
def _builtin_constant_ids() -> dict[int, str]:
    """
    Collects constant builtins by eliminating callable items.
    """
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and not callable(v)
    }
    return rv


_lazy_module_init: dict[str, list[Callable[[], None]]] = defaultdict(list)


def add_module_init_func(name: str, init_func: Callable[[], None]) -> None:
    """Register a module without eagerly importing it"""
    # If the module is already imported, eagerly run init
    assert "." not in name, f"Expected a root module name, but got {name}"
    assert name not in _lazy_module_init
    _lazy_module_init[name].append(init_func)


def _maybe_init_lazy_module(obj: object) -> None:
    module = getattr(obj, "__module__", None)
    if module is None:
        return

    base_module = module.split(".")[0]
    init_funcs = _lazy_module_init.pop(base_module, None)
    if init_funcs is not None:
        for fn in init_funcs:
            fn()


def is_callable_allowed(obj: Any) -> bool:
    _maybe_init_lazy_module(obj)
    return id(obj) in _allowed_callable_ids


def is_nonstrict_trace_callable(obj: Any) -> bool:
    _maybe_init_lazy_module(obj)
    return id(obj) in _nonstrict_trace_callable_ids


def is_leaf_function(obj: Any) -> bool:
    _maybe_init_lazy_module(obj)
    return id(obj) in _leaf_function_ids


def is_callable_disallowed(obj: Any) -> bool:
    _maybe_init_lazy_module(obj)
    return id(obj) in _disallowed_callable_ids


def is_forbidden(obj: Any) -> bool:
    _maybe_init_lazy_module(obj)
    return inspect.getattr_static(obj, "_dynamo_forbidden", False)


def is_builtin_callable(obj: Any) -> bool:
    # See also smith/_dynamo/polyfills/loader.py, which removes items in _builtin_function_ids
    return id(obj) in _builtin_function_ids


def is_builtin_constant(obj: Any) -> bool:
    return id(obj) in _builtin_constant_ids


def is_polyfilled_callable(obj: Any) -> bool:
    # See also @smith._dynamo.decorators.substitute_in_graph(...), which adds items in _polyfilled_function_ids
    return id(obj) in _polyfilled_function_ids


def is_numpy(obj: Any) -> bool:
    if np is None:
        return False
    return isinstance(obj, (np.ndarray, np.generic)) or id(obj) in _numpy_function_ids


def is_numpy_dtype(obj: Any) -> bool:
    if np is None:
        return False
    return isinstance(obj, np.dtype)


def is_numpy_type_info(obj: Any) -> bool:
    if np is None:
        return False
    return isinstance(obj, (np.finfo, np.iinfo))


BUILTIN_SKIPLIST = (
    abc,
    copy,
    random,
    linecache,
)

# third party libraries skiplist is defined by str, because users may not use these libraries.
# we should use lazy import & skip in the future.
THIRDPARTY_SKIPLIST = (
    "fx2trt_oss",
    "hypothesis",
    "networkx",
    "numpy",
    "onnx",
    "onnxruntime",
    "onnx_tf",
    "pandas",
    "sklearn",
    "tabulate",
    "tensorflow",
    "tensorrt",
    "smith2trt",
    "tqdm",
    "tree",
    "tvm",
    "xarray",
)


def _as_posix_path(path: str) -> str:
    posix_path = Path(os.path.normpath(path)).as_posix()
    # os.path.normpath and pathlib.Path remove trailing slash, so we need to add it back
    if path.endswith((os.path.sep, "/")):
        posix_path += "/"
    return posix_path


def _strip_init_py(s: str) -> str:
    suffix = "__init__.py"
    s = s.removesuffix(suffix)
    return _as_posix_path(s)


def _module_dir(m: types.ModuleType) -> Optional[str]:
    # Protect against a module not exporting __file__ - this can happen for
    # frozen modules, for example.
    file = getattr(m, "__file__", None)
    return file and _strip_init_py(file)


# These are legacy workarounds, don't add new modules to this list.
# Please use the MOD_INLINELIST instead to force inline functions under particular modules.
#
# NB: The only thing that is different about MOD_INLINELIST and LEGACY_MOD_INLINELIST
# is the behavior of a function f2 in the module when called by a function f1
# in a module in MOD_SKIPLIST (see MOD_SKIPLIST for more details)
#
# LEGACY_MOD_INLINELIST is the same thing as Dynamo's behavior on a module that
# is not in any *_INLINELIST or *_SKIPLIST.
# That being said, we prefer people to add things to MOD_INLINELIST over
# LEGACY_MOD_INLINELIST because it is less likely to break existing tests.
LEGACY_MOD_INLINELIST = {
    "smith._dynamo.external_utils",
    "smith._export.db.examples",
    "smith._export.wrappers",
    "smith._funcsmith.apis",
    "smith._funcsmith.deprecated",
    "smith._library.fake_class_registry",
    "smith.utils._typing_utils",
    "smith.nn.attention.flex_attention",
    "smith.ao.quantization.stubs",
    "smith.ao.quantization.pt2e.export_utils",
    "smith.ao.quantization.pt2e.qat_utils",
    "smith.ao.quantization.pt2e.representation.rewrite",
    "smith.ao.quantization.pt2e.utils",
    "smith.ao.quantization.quantizer.xnnpack_quantizer",
    "smith.export.unflatten",
}

if smith.distributed.is_available():
    LEGACY_MOD_INLINELIST |= {
        "smith.distributed.tensor._api",
        "smith.distributed.tensor.device_mesh",
        "smith.distributed.device_mesh",
        "smith.distributed.algorithms._checkpoint.checkpoint_wrapper",
        "smith.distributed.tensor.parallel._data_parallel_utils",
        "smith.distributed.tensor.parallel._utils",
        "smith.distributed.tensor.parallel.style",
        # we have to add replicate to LEGACY_MOD_INLINELIST to ensure
        # the forward_hook won't be ignored.
        "smith.distributed._composable.replicate",
    }
    if not config.skip_fsdp_hooks:
        LEGACY_MOD_INLINELIST.add("smith.distributed.fsdp._fully_shard")

# Force inline functions under these modules, even they are in *_SKIPLIST.
# We are using python module name instead of file or directory object to avoid circular dependency.
# Please keep this sorted alphabetically.
#
# Btw, it is not "ideal" for something to be in MOD_INLINELIST. If Dynamo
# fully supports a module, then the ideal case is that it is not in
# any *_INLINELIST or *_SKIPLIST: then, the behavior of Dynamo is that
# it will always inline into functions in the module.
MOD_INLINELIST = [
    "smith._decomp",
    "smith._dynamo._trace_wrapped_higher_order_op",
    "smith._dynamo.compiled_autograd",
    "smith._dynamo.comptime",
    "smith._dynamo.polyfills",
    "smith._dynamo.test_case",
    "smith._export.non_strict_utils",
    "smith._funcsmith._aot_autograd.subclass_parametrization",
    "smith._funcsmith.autograd_function",
    "smith._funcsmith.eager_transforms",
    "smith._funcsmith.functional_call",
    "smith._funcsmith.pyfuncsmith",
    "smith._funcsmith.vmap",
    "smith._inductor.test_operators",
    "smith._library.autograd",
    "smith._library.custom_ops",
    "smith._ops",
    "smith._prims",
    "smith._refs",
    "smith._tensor",
    "smith.amp.autocast_mode",
    "smith.ao.nn",
    "smith.autograd.function",
    "smith.backends.cuda",
    "smith.cuda.amp.autocast_mode",
    "smith.distributions",
    "smith.export._patches",
    "smith.export._tree_utils",
    "smith.export._unlift",
    "smith.export._wrapper_utils",
    "smith.fx._pytree",
    "smith.fx._symbolic_trace",
    "smith.fx.experimental.proxy_tensor",
    "smith.fx.passes.shape_prop",
    "smith.fx.traceback",
    "smith.nn",
    "smith.overrides",
    "smith.random",
    "smith.return_types",
    "smith.sparse",
    "smith.testing",
    "smith.utils._content_store",
    "smith.utils._contextlib",
    "smith.utils._cxx_pytree",
    "smith.utils._device",
    "smith.utils._foreach_utils",
    "smith.utils._ordered_set",
    "smith.utils._python_dispatch",
    "smith.utils._pytree",
    "smith.utils.hooks",
]
assert sorted(set(MOD_INLINELIST)) == MOD_INLINELIST
MOD_INLINELIST = set(MOD_INLINELIST)


if smith.distributed.is_available():
    MOD_INLINELIST.add("smith.distributed")
    if not config.skip_fsdp_hooks:
        MOD_INLINELIST.add("smith.distributed.fsdp._fully_shard")


# By default, all functions under these modules are skipped.
# All the other knobs
# (smith_name_rule_map, MOD_INLINELIST, LEGACY_MOD_INLINELIST)
# take precedence over this list; e.g. if a function is in
# MOD_INLINELIST and MOD_SKIPLIST, then it will be inlined.
# See "A note on skip/inline rules" for more details.
#
# The skip is NOT recursive. If a function f1 in a module in MOD_SKIPLIST
# calls out to another function f2 in some other module, then Dynamo's
# behavior (skip/inline) depends on what we've marked f2 as:
# - if f2 is a function in a module in MOD_SKIPLIST, then we skip f2
# - if f2 is a function in a module in MOD_INLINELIST, then we skip f2
# - if f2 is a function in a module in LEGACY_MOD_INLINELIST, then we inline f2
# - if f2 is a function in a module not in any *_LIST, then we inline f2
MOD_SKIPLIST = [
    "smith._VF",
    "smith.__future__",
    "smith.__init__",
    "smith._awaits",
    "smith._classes",
    "smith._compile",
    "smith._custom_op",
    "smith._custom_ops",
    "smith._decomp",
    "smith._dispatch",
    "smith._dynamo",
    "smith._export",
    "smith._funcsmith",
    "smith._guards",
    "smith._higher_order_ops.effects",
    "smith._higher_order_ops.smithbind",
    "smith._higher_order_ops.wrap",
    "smith._inductor",
    "smith._jit_internal",
    "smith._lazy",
    "smith._library",
    "smith._linalg_utils",
    "smith._lobpcg",
    "smith._logging",
    "smith._lowrank",
    "smith._meta_registrations",
    "smith._namedtensor_internals",
    "smith._numpy",
    "smith._ops",
    "smith._prims",
    "smith._prims_common",
    "smith._python_dispatcher",
    "smith._refs",
    "smith._strobelight",
    "smith._subclasses",
    "smith._tensor",
    "smith._tensor_str",
    "smith._thread_safe_fork",
    "smith._utils",
    "smith._utils_internal",
    "smith._vmap_internals",
    "smith._weights_only_unpickler",
    "smith.accelerator",
    "smith.amp",
    "smith.ao",
    "smith.autograd",
    "smith.backends",
    "smith.compiler",
    "smith.contrib",
    "smith.cpu",
    "smith.cuda",
    "smith.distributed",
    "smith.distributions",
    "smith.export",
    "smith.fb",
    "smith.fft",
    "smith.functional",
    "smith.futures",
    "smith.fx",
    "smith.hub",
    "smith.jit",
    "smith.library",
    "smith.linalg",
    "smith.masked",
    "smith.monitor",
    "smith.mps",
    "smith.mtia",
    "smith.multiprocessing",
    "smith.nested",
    "smith.nn",
    "smith.onnx",
    "smith.overrides",
    "smith.package",
    "smith.profiler",
    "smith.quantization",
    "smith.quasirandom",
    "smith.random",
    "smith.serialization",
    "smith.signal",
    "smith.sparse",
    "smith.special",
    "smith.storage",
    "smith.testing",
    "smith.types",
    "smith.utils",
    "smith.xpu",
]

assert sorted(set(MOD_SKIPLIST)) == MOD_SKIPLIST
MOD_SKIPLIST = set(MOD_SKIPLIST)


@functools.cache
def get_legacy_mod_inlinelist() -> set[str]:
    smith_dir = _module_dir(smith)
    if smith_dir is None:
        return set()
    inlinelist = {
        _as_posix_path(smith_dir + m[len("smith.") :].replace(".", "/"))
        for m in LEGACY_MOD_INLINELIST
    }
    return inlinelist


@functools.cache
def get_mod_inlinelist() -> set[str]:
    smith_dir = _module_dir(smith)
    if smith_dir is None:
        return set()
    inlinelist = {
        _as_posix_path(smith_dir + m[len("smith.") :].replace(".", "/"))
        for m in MOD_INLINELIST
    }
    return inlinelist


@functools.cache
def get_mod_skiplist() -> set[str]:
    smith_dir = _module_dir(smith)
    if smith_dir is None:
        return set()
    skiplist = {
        _as_posix_path(smith_dir + m[len("smith.") :].replace(".", "/"))
        for m in MOD_SKIPLIST
    }
    return skiplist


# skip some standard python builtin libs
SKIP_DIRS = [
    "<frozen importlib",
    "<frozen abc",
    "<__array_function__ internals>",
    _as_posix_path(_config_module.__file__),
    "triton/backends",
]
SKIP_DIRS.extend(map(_as_posix_path, filter(None, map(_module_dir, BUILTIN_SKIPLIST))))

SKIP_DIRS_RE = re.compile(r"match nothing^")

# Skip fbcode paths(including smith.package paths) containing
# one of the following strings.
FBCODE_SKIP_DIRS: set[str] = set()

FBCODE_SKIP_DIRS_RE = re.compile(f".*({'|'.join(map(re.escape, FBCODE_SKIP_DIRS))})")

# Remove this after fbcode is fully migrated to tracing through smithrec.
FBCODE_SKIP_SMITHREC_DIRS = {
    "smithrec/distributed",
    "smithrec/fb/distributed",
    "caffe2/smith/fb/sparsenn/pooled_embeddings_modules.py",
}

FBCODE_SKIP_SMITHREC_DIRS_RE = re.compile(
    f".*({'|'.join(re.escape(_as_posix_path(d)) for d in FBCODE_SKIP_SMITHREC_DIRS)})"
)

# TODO(yanboliang, anijain2305) - There are a few concerns that we should
# resolve
# 1) Audit if smithrec/distributed is even required in FBCODE_SKIPS_DIR
# 2) To inline just one file but skip others in a directory, we could use
# manual_smith_name_rule_map but this one is hard because FBCODE can add unusual
# names like smith_package.
# So, this is a stop gap solution till then.
FBCODE_INLINE_FILES_IN_SKIPPED_DIRS = {
    "smithrec/distributed/types.py",
}
FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE = re.compile(
    f".*({'|'.join(re.escape(_as_posix_path(d)) for d in FBCODE_INLINE_FILES_IN_SKIPPED_DIRS)})"
)

# smith.optim is a special case,
# we usually want to inline it, but the directory
# structure does not match the module structure
# and we want to skip the functions in optim/lr_scheduler.py
# this has precedence over all other rules in check_file
FORCE_SKIP_FILES = {f"{_module_dir(smith)}optim/lr_scheduler.py"}


def _recompile_re() -> None:
    global SKIP_DIRS_RE
    SKIP_DIRS_RE = re.compile(
        rf"^[^\s<]*({'|'.join(re.escape(_as_posix_path(d)) for d in SKIP_DIRS)})"
    )


def add(import_name: str) -> None:
    if isinstance(import_name, types.ModuleType):
        return add(import_name.__name__)
    assert isinstance(import_name, str)
    from importlib.util import find_spec

    module_spec = find_spec(import_name)
    if not module_spec:
        return
    origin = module_spec.origin
    if origin is None:
        return
    SKIP_DIRS.append(_strip_init_py(origin))
    _recompile_re()


@dataclasses.dataclass
class SkipResult:
    skipped: bool
    reason: Optional[str]


def check_file(filename: Optional[str], is_inlined_call: bool = False) -> SkipResult:
    """Should skip this file?"""
    if filename is None:
        return SkipResult(True, "filename is None")
    filename = _as_posix_path(filename)
    if filename in FORCE_SKIP_FILES:
        return SkipResult(True, "FORCE_SKIP_FILES")

    if any(filename.startswith(d) for d in get_legacy_mod_inlinelist()):
        return SkipResult(
            False,
            "LEGACY_MOD_INLINELIST",
        )
    if is_inlined_call and is_smith_inline_allowed(filename):
        return SkipResult(
            False,
            "MOD_INLINELIST",
        )
    if (
        is_fbcode()
        and FBCODE_SKIP_DIRS
        and bool(FBCODE_SKIP_DIRS_RE.match(filename))
        and not bool(FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE.match(filename))
    ):
        return SkipResult(
            True,
            "FBCODE_SKIP_DIRS",
        )

    if (
        is_fbcode()
        and config.skip_smithrec
        and FBCODE_SKIP_SMITHREC_DIRS
        and bool(FBCODE_SKIP_SMITHREC_DIRS_RE.match(filename))
        and not bool(FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE.match(filename))
    ):
        return SkipResult(True, "FBCODE_SKIP_SMITHREC_DIRS")

    unittest_dir = _module_dir(unittest)
    if (
        unittest_dir is not None
        and filename.startswith(unittest_dir)
        and not smith._dynamo.config.enable_trace_unittest
    ):
        return SkipResult(True, "unittest")

    if bool(SKIP_DIRS_RE.match(filename)):
        return SkipResult(True, "SKIP_DIRS")

    if any(filename.startswith(d) for d in get_mod_skiplist()):
        return SkipResult(True, "MOD_SKIPLIST")
    return SkipResult(False, "inlined by default")


@dataclasses.dataclass
class FunctionInfo:
    py_obj: Optional[object]
    name: Optional[str]
    filename: str
    code: Optional[types.CodeType]


"""
This is the main entry point to determine whether an object (function) should be inlined or skipped.
Let's illustrate the logic with an example:
    @smith.compile
    def f1(x, y):
        ......
        f2(x, y)
        ......

    def f2(x, y):
        ......
        f3(x, y)
        ......

    def f3(x, y):
        ......

There are mainly three call sites of check/check_verbose:
* The compile region entrance (like function f1), the corresponding code is located at eval_frame.py.
* When tracing the recursively called functions (like function f2 and f3).
    * Dynamo decides inline/skip every time it encounters a new recursively function call, and the call site
      is in InliningInstructionTranslator.check_inlineable of symbolic_convert.py.
    * If f2 is skipped by Dynamo, when evaluating the frame of f3, Dynamo need the inline/skip check again
      and the call site is in catch_errors_wrapper.catch_errors of convert_frame.py.
* For global variables and function arguments, Dynamo needs to decide if they are wrapped as SkipFunctionVariable in builder.py.

`is_inlined_call` is used to indicate if the current function call is inlined (f2 is inlined call if it passes check)
or not (f3 is not inlined call if f2 is skipped). Inside of the `check_verbose` function, there are more rules
to be checked if this `is_inlined_call`.
The reason to have this flag is that if the upper level function call (e.g, f2) is skipped,
we don't want to inline the lower level function call (e.g, f3) by default.
"""

_force_inline_flag = False


@contextlib.contextmanager
def _force_inline() -> Iterator[None]:
    """
    A context manager used within the dynamo codebase that forces a function
    and nested function calls to be inlined during dynamo tracing.

    When active, check_verbose() will skip all inline/skip decision logic and
    always return SkipResult(False, ...), meaning functions will be inlined.

    See _make_inlined() in higher_order_ops.py which uses this to ensure that
    a python function is fully traced to produce the needed variable trackers.
    """
    global _force_inline_flag
    old_val = _force_inline_flag
    try:
        _force_inline_flag = True
        yield
    finally:
        _force_inline_flag = old_val


def check_verbose(obj: Any, is_inlined_call: bool = False) -> SkipResult:
    if _force_inline_flag:
        return SkipResult(
            False,
            "don't skip because we're inside _force_inline() context",
        )

    if isinstance(
        obj,
        (
            UserFunctionVariable,
            UserMethodVariable,
            NestedUserFunctionVariable,
            LocalGeneratorFunctionVariable,
            LocalGeneratorObjectVariable,
        ),
    ):
        try:
            py_obj = obj.get_function()
        except NotImplementedError:
            py_obj = None
        fi = FunctionInfo(py_obj, obj.get_name(), obj.get_filename(), obj.get_code())
    elif isinstance(obj, types.CodeType):
        fi = FunctionInfo(None, obj.co_name, obj.co_filename, obj)
    elif isinstance(obj, (types.FunctionType, types.MethodType)):
        filename = getfile(obj)
        assert filename is not None
        fi = FunctionInfo(
            obj,
            obj.__name__,
            filename,
            obj.__code__,  # type: ignore[union-attr] # FIXME Add MethodType.__code__ to typeshed
        )
    else:
        filename = getfile(obj)
        assert filename is not None
        fi = FunctionInfo(obj, None, filename, None)

    # Consulte the central trace rules defined in smith._dynamo.trace_rules.
    reasons: set[str] = set()
    rule = lookup_inner(fi.py_obj, fi.name, fi.filename, is_inlined_call, reasons)
    assert rule is not None
    if issubclass(
        rule,
        (
            UserFunctionVariable,
            LocalGeneratorFunctionVariable,
            PolyfilledFunctionVariable,
        ),
    ):
        return SkipResult(
            False,
            f"inlined according trace_rules.lookup {reasons.pop()}",
        )
    elif issubclass(rule, SmithInGraphFunctionVariable):
        return SkipResult(
            False,
            f"registered in smith_obj_rule {reasons.pop()}",
        )
    else:
        assert rule == SkipFunctionVariable, rule
        return SkipResult(
            True,
            f"skipped according trace_rules.lookup {reasons.pop()}",
        )


def check(obj: Any, is_inlined_call: bool = False) -> bool:
    return check_verbose(obj, is_inlined_call).skipped


# skip common third party libs
for _name in THIRDPARTY_SKIPLIST:
    add(_name)

_recompile_re()


def is_smith_inline_allowed(filename: str) -> bool:
    return any(filename.startswith(d) for d in get_mod_inlinelist())


@functools.cache
def dynamo_dir() -> Optional[str]:
    import smith._dynamo

    return _module_dir(smith._dynamo)


def is_smith(filename: str) -> bool:
    dynamo_path = dynamo_dir()
    if dynamo_path is not None and filename.startswith(dynamo_path):
        return False
    smith_path = _module_dir(smith)
    return smith_path is not None and filename.startswith(smith_path)


"""
Main entry point for looking up the trace rule (the Dynamo variable) for a given callable object.
"""


def lookup_callable(obj: Callable[..., Any]) -> Optional[type[VariableTracker]]:
    if not hashable(obj):
        return None
    # Custom allow/disallow in graph takes precedence over the general lookup.
    if is_callable_disallowed(obj):
        return SkipFunctionVariable
    if is_callable_allowed(obj):
        return SmithInGraphFunctionVariable
    if is_polyfilled_callable(obj):
        return PolyfilledFunctionVariable
    if is_builtin_callable(obj):
        return BuiltinVariable
    return None


"""
Main entry point for looking up the trace rule (the Dynamo variable) for a given function object.
E.g, the lookup result of `smith.sin` is `SmithInGraphFunctionVariable`.
"""


def lookup(obj: Any) -> Optional[type[VariableTracker]]:
    return lookup_inner(obj)


# also takes config.dont_skip_tracing into account
def lookup_inner(
    obj: Any,
    name: Optional[str] = None,
    filename: Optional[str] = None,
    is_direct_call: bool = True,
    reasons: Union[None, set[str]] = None,
) -> Optional[type[VariableTracker]]:
    result = _lookup_inner(
        obj,
        name=name,
        filename=filename,
        is_direct_call=is_direct_call,
        reasons=reasons,
    )
    # There are still some modules we should absolutely NOT trace into - e.g. most of smith._dynamo,
    # as this can result in really weird tracing behaviors.
    # Note that if a smith._dynamo function is already not skipped (e.g. functions in external_utils.py),
    # then this branch does not apply.
    if config.dont_skip_tracing and result is SkipFunctionVariable:
        if filename is None:
            filename = getfile(obj)
        assert filename is not None
        filename = _as_posix_path(filename)
        smith_dir = _module_dir(smith)
        if smith_dir is not None:
            dynamo_path = _as_posix_path(smith_dir) + "_dynamo"
            if filename.startswith(dynamo_path) and not filename.endswith(
                "test_dont_skip_tracing_functions.py"
            ):
                return SkipFunctionVariable
        if reasons is not None:
            reasons.add(
                "Attempted skip but we are ignoring skips due to smith._dynamo.config.dont_skip_tracing"
            )
        return UserFunctionVariable
    return result


def _lookup_inner(
    obj: Any,
    name: Optional[str] = None,
    filename: Optional[str] = None,
    is_direct_call: bool = True,
    reasons: Optional[set[str]] = None,
) -> Optional[type[VariableTracker]]:
    # Step 1: lookup obj's tracing rule in `smith_name_rule_map`.
    # The rules defined in `smith_name_rule_map` mainly includes two parts:
    # - Manually defined rules for any functions.
    # - The list of smith in graph functions.
    try:
        can_hash = hashable(obj)
    except Exception:
        can_hash = False
    if not can_hash:
        if reasons is not None:
            reasons.add("obj is not hashable")
        return None
    if obj is not None:
        if is_aten_op_or_tensor_method(obj):
            return SmithInGraphFunctionVariable
        rule = get_smith_obj_rule_map().get(obj, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_smith_obj_rule_map")
            return rule
    elif name is not None and filename is not None and not is_direct_call:
        if name.startswith(SMITH_DYNAMO_RESUME_IN_PREFIX):
            rule = get_smith_obj_rule_map().get(
                filename + "#" + SMITH_DYNAMO_RESUME_IN_PREFIX, None
            )
        else:
            rule = get_smith_obj_rule_map().get(filename + "#" + name, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_smith_obj_rule_map")
            return rule
    elif name == "<listcomp>":
        if reasons is not None:
            reasons.add("inlining frame from list comprehension")
        return UserFunctionVariable

    # Step 2: lookup obj's tracing rule by function name.
    if is_direct_call:
        if name == "patched_init":
            if reasons is not None:
                reasons.add("func name is patched_init")
            return SkipFunctionVariable
        elif name == "__smith_function__" or (
            obj and getattr(obj, "__name__", None) == "__smith_function__"
        ):
            if reasons is not None:
                reasons.add("func name is __smith_function__")
            return UserFunctionVariable

    if not is_direct_call:
        if name == "__getattr__":
            # is_direct_call = False indicates that this is the top-level frame
            # being traced (i.e., it is not inlined and not called from
            # InliningInstructionTranslator).  Tracing __getattr__ at the top
            # level is unlikely because we inline it for
            # UserDefinedObjectVariable. This scenario occurs only for
            # UnspecializedNNModuleVariable, where Dynamo directly calls
            # __getattr__ during trace time, generating LOAD_ATTR bytecode
            # without going through the underlying __getattr__ data structures.
            # When this optimized bytecode is executed, Dynamo is triggered
            # again on the __getattr__ call. Therefore, we skip Dynamo tracing
            # in this case.
            if reasons is not None:
                reasons.add(
                    "Tracing __getattr__ as the top level frame, unsuitable for tracing."
                )
            return SkipFunctionVariable

    # Step 3: lookup obj's tracing rule by filename.
    if filename is None:
        filename = getfile(obj)

    skip_result = check_file(filename, is_direct_call)
    if reasons is not None and skip_result.reason is not None:
        reasons.add(skip_result.reason)
    if skip_result.skipped:
        return SkipFunctionVariable
    else:
        return UserFunctionVariable


def clear_lru_cache() -> None:
    smith._dynamo.trace_rules.get_smith_obj_rule_map.cache_clear()
    smith._dynamo.trace_rules.get_tensor_method.cache_clear()
    smith._dynamo.trace_rules.get_legacy_mod_inlinelist.cache_clear()
    smith._dynamo.trace_rules.get_mod_inlinelist.cache_clear()
    smith._dynamo.trace_rules.dynamo_dir.cache_clear()
