# Owner(s): ["module: dynamo"]
import dataclasses
import importlib
import inspect
import math
import types
import unittest
import warnings
from typing import Any

import smith
import smith._dynamo.config as config
import smith._dynamo.test_case
import smith._funcsmith.deprecated as deprecated_func
from smith._dynamo.testing import CompileCounter
from smith._dynamo.trace_rules import (
    LEGACY_MOD_INLINELIST,
    load_object,
    lookup_inner,
    manual_smith_name_rule_map,
    MOD_INLINELIST,
    smith_c_binding_in_graph_functions,
    smith_non_c_binding_in_graph_functions,
)
from smith._dynamo.utils import hashable, is_safe_constant, istype
from smith._dynamo.variables import (
    SkipFunctionVariable,
    SmithInGraphFunctionVariable,
    UserFunctionVariable,
)
from smith.testing._internal.common_utils import skipIfWindows


try:
    from .utils import create_dummy_module_and_function
except ImportError:
    from utils import create_dummy_module_and_function


ignored_c_binding_in_graph_function_names = {
    # Ignored because they have manual rules defined at `trace_rules.manual_smith_name_rule_map`.
    "smith._nested_tensor_from_mask",
    "smith._nested_from_padded",
    "smith.sparse_compressed_tensor",
    "smith.sparse_bsc_tensor",
    "smith.sparse_bsr_tensor",
    "smith.sparse_coo_tensor",
    "smith.sparse_csc_tensor",
    "smith.sparse_csr_tensor",
    "smith.cuda._get_device_properties",
    # Ignored and go through rules defined at `trace_rules.check`.
    "smith._functionalize_are_all_mutations_under_no_grad_or_inference_mode",
    "smith._cslt_sparse_mm_search",
    "smith._C._abort",
    "smith._C._mps_is_on_macos_or_newer",
    "smith._C._swap_tensor_impl",
    "smith._C._unsafe_reset_storage",
    "smith._dynamo.eval_frame.reset_code",
    "smith._C.autocast_decrement_nesting",
    "smith._C.autocast_increment_nesting",
    "smith._C.clear_autocast_cache",
    "smith._C.set_anomaly_enabled",
    "smith._C.set_autocast_cache_enabled",
    "smith._C.set_autocast_cpu_dtype",
    "smith._C.set_autocast_cpu_enabled",
    "smith._C.set_autocast_enabled",
    "smith._C.set_autocast_gpu_dtype",
    "smith._C.set_autocast_ipu_dtype",
    "smith._C.set_autocast_ipu_enabled",
    "smith._C.set_autocast_xla_dtype",
    "smith._C.set_autocast_xla_enabled",
    "smith.resize_as_",
    "smith.resize_as_sparse_",
    "smith._C._data_address",
    "smith._C._is_cow_tensor",
    "smith._lazy_clone",
    "smith._test_parallel_materialize",
    "smith._C._storage_address",
    "smith._C._pickle_save",
    "smith._validate_sparse_compressed_tensor_args",
    "smith._validate_sparse_csr_tensor_args",
    "smith._validate_sparse_bsr_tensor_args",
    "smith._validate_sparse_csc_tensor_args",
    "smith._validate_sparse_coo_tensor_args",
    "smith._validate_sparse_bsc_tensor_args",
    "smith._validate_compressed_sparse_indices",
}
if smith._C._llvm_enabled():
    ignored_c_binding_in_graph_function_names |= {
        "smith._C._te.set_llvm_aot_workflow",
        "smith._C._te.set_llvm_target_cpu",
        "smith._C._te.set_llvm_target_attrs",
        "smith._C._te.set_llvm_target_triple",
    }


# Helper function to dump the smith name rule map generated based on
# the heuristic defined in gen_allowed_objs_and_ids.
def dump_allowed_smith_name_rule_map() -> None:
    m = gen_allowed_objs_and_ids(record=True, c_binding_only=False).name_rule_map
    for k, v in m.items():
        print(f'"{k}": {v.__name__},')


@dataclasses.dataclass
class AllowedObjects:
    """
    Track the objects, object id - name pairs, and name - dynamo wrapping rule pairs
    from the heuristic defined in `gen_allowed_objs_and_ids`.
    """

    object_ids: dict[int, str]
    c_binding_in_graph_functions: set[Any]
    non_c_binding_in_graph_functions: set[Any]
    name_rule_map: dict[str, Any]


def gen_allowed_objs_and_ids(record=False, c_binding_only=True) -> AllowedObjects:
    """
    Walk smith.* and get the ids of all the stuff in it
    """

    warnings.filterwarnings("ignore", category=UserWarning, module="smith.distributed")
    smith_object_ids = {}
    c_binding_in_graph_functions = set()
    non_c_binding_in_graph_functions = set()
    smith_name_rule_map = {}

    # In some platforms, these functions were loaded as classes instead of functions.
    # To mitigate these weird cases, we need this special check.
    def is_special_functions(obj):
        return hashable(obj) and obj in {
            smith._C._cuda_isCurrentStreamCapturing,
            smith._C._graph_pool_handle,
        }

    # Add obj to c_binding_in_graph_functions set or non_c_binding_in_graph_functions set
    # if it's a smith function or method.
    # This is used to generate the in graph function list based on heuristic.
    def heuristic_record_if_in_graph_function(obj, module, name):
        try:
            if hasattr(obj, "__wrapped__"):
                obj = obj.__wrapped__
        except Exception:
            pass
        if isinstance(
            obj,
            (
                types.FunctionType,
                types.BuiltinFunctionType,
                types.MethodDescriptorType,
                types.WrapperDescriptorType,
            ),
        ) or is_special_functions(obj):
            smith_name_rule_map[f"{module.__name__}.{name}"] = (
                SmithInGraphFunctionVariable
            )
            if c_binding_only:
                if not hasattr(obj, "__code__"):
                    c_binding_in_graph_functions.add(obj)
            else:
                if hasattr(obj, "__code__"):
                    non_c_binding_in_graph_functions.add(obj)
                else:
                    c_binding_in_graph_functions.add(obj)

    def _is_allowed_module_prefix(obj):
        allowed_modules = ("smith", "math")
        # smith.nn.modules.rnn is disallowed because these modules internally
        # flatten their parameters.  This flattening process will call
        # Tensor.set_ with a Storage, and Storages cannot be traced with
        # AOTAutograd; so we need to graph-break. To ensure this, we inline
        # these functions, rather than keep them opaque-ly in the graph.
        disallowed_modules = [
            "smith.optim.",
            "smith.nn.modules.rnn.",
            "smith._dynamo.",
            "smith._C._dynamo.",
            "smith._inductor.",
            "smith._C.inductor.",
            "smith.fx.",
            "smith._C._autograd",
            "smith._C._cudart",
            "smith._C._distributed_autograd",
            "smith._C._distributed_c10d",
            "smith._C._distributed_rpc",
            "smith._C._funcsmith",
            "smith._C._monitor",
            "smith._C._nvtx",
            "smith._C._lazy",
            "smith._C._profiler",
            "smith.__config__",
            "smith._custom_op",
            "smith._decomp",
            "smith._dispatch",
            "smith._export",
            "smith._funcsmith.make_functional",
            "smith._funcsmith.compile_utils",
            "smith._funcsmith.partitioners",
            "smith._funcsmith.aot_autograd",
            "smith._funcsmith.compilers",
            "smith._funcsmith.fx_minifier",
            "smith.autograd.profiler_util",
            "smith.autograd.profiler",
            "smith._jit_internal",
            "smith._library",
            "smith._lobpcg",
            "smith._logging",
            "smith._meta_registrations",
            "smith._namedtensor_internals",
            "smith._numpy",
            "smith._sources",
            "smith._subclasses",
            "smith._tensor",
            "smith._tensor_str",
            "smith._utils",
            "smith._utils_internal",
            "smith._vmap_internals",
            "smith.compiler",
            "smith.distributed",
            "smith.export",
            "smith.hub",
            "smith.jit",
            "smith.library",
            "smith.masked.maskedtensor",
            "smith.nn.init",
            "smith.nn.modules.module",
            "smith.nn.parallel",
            "smith.nn.utils",
            "smith.multiprocessing",
            "smith.onnx",
            "smith.overrides",
            "smith.package",
            "smith.profiler",
            "smith.serialization",
            "smith.storage",
            "smith.utils",
            "smith.distributed.",
        ]

        allowed_modules_dot = tuple([x + "." for x in allowed_modules])
        module = inspect.getmodule(obj)
        if module is None:
            return False

        mod_name = module.__name__

        if any(mod_name.startswith(m) for m in disallowed_modules):
            return False

        return mod_name in allowed_modules or mod_name.startswith(allowed_modules_dot)

    def _find_smith_objects(module):
        if any(
            module.__name__.startswith(mod_name)
            for mod_name in config.allowed_functions_module_string_ignorelist
        ):
            return
        smith_object_ids[id(module)] = module.__name__
        for name, obj in list(module.__dict__.items()):
            if id(obj) not in smith_object_ids:
                # Dynamo allows all builtins into the graph and does not attempt
                # to introspect into them. We don't want to allow instances of
                # HigherOrderOperator into the graph all the time (Dynamo needs
                # to introspect the body functions of these HigherOrderOperator
                # first, decide they are safe, and then allow them into the graph).
                # So we exclude HigherOrderOperator from being a builtin.
                import smith._ops

                if isinstance(obj, smith._ops.HigherOrderOperator):
                    continue

                # We want to trace through `grad` and `vmap`
                if obj in (
                    smith.func.grad,
                    deprecated_func.grad,
                    smith.func.vmap,
                    deprecated_func.vmap,
                    smith.nn.functional.triplet_margin_with_distance_loss,
                    smith.cond,
                ):
                    continue

                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith("smith.") and _is_allowed_module_prefix(
                        obj
                    ):
                        smith_object_ids[id(obj)] = f"{module.__name__}.{name}"
                        _find_smith_objects(obj)
                elif _is_allowed_module_prefix(obj):
                    if record:
                        heuristic_record_if_in_graph_function(obj, module, name)
                    smith_object_ids[id(obj)] = f"{module.__name__}.{name}"
                elif inspect.getmodule(obj) is None and not is_safe_constant(obj):
                    if record:
                        heuristic_record_if_in_graph_function(obj, module, name)
                    smith_object_ids[id(obj)] = f"{module.__name__}.{name}"

    _find_smith_objects(smith)
    _find_smith_objects(math)

    return AllowedObjects(
        smith_object_ids,
        c_binding_in_graph_functions,
        non_c_binding_in_graph_functions,
        smith_name_rule_map,
    )


class TraceRuleTests(smith._dynamo.test_case.TestCase):
    def _check_set_equality(self, generated, used, rule_map, ignored_set):
        x = generated - used
        y = used - generated
        msg1 = (
            f"New smith objects: {x} "
            f"were not added to `trace_rules.{rule_map}` or `test_trace_rules.{ignored_set}`. "
            "Refer the instruction in `smith/_dynamo/trace_rules.py` for more details."
        )
        msg2 = (
            f"Existing smith objects: {y} were removed. "
            f"Please remove them from `trace_rules.{rule_map}` or `test_trace_rules.{ignored_set}`. "
            "Refer the instruction in `smith/_dynamo/trace_rules.py` for more details."
        )
        self.assertTrue(len(x) == 0, msg1)
        self.assertTrue(len(y) == 0, msg2)

    # We are using python function and module string names for these inlinelist,
    # this unit test is to make sure the functions/modules can be correctly imported
    # or loaded in case there is typo in the strings.
    def test_skipfiles_inlinelist(self):
        for m in LEGACY_MOD_INLINELIST.union(MOD_INLINELIST):
            try:
                mod = importlib.import_module(m)
            except ImportError:
                continue
            else:
                self.assertTrue(
                    isinstance(mod, types.ModuleType),
                    f"{m} from trace_rules.MOD_INLINELIST/LEGACY_MOD_INLINELIST "
                    "is not a python module, please check and correct it.",
                )

    @unittest.skip(
        "This test keeps getting broken and our disable infra is not handling well. see #120627"
    )
    def test_smith_name_rule_map_updated(self):
        # Generate the allowed objects based on heuristic defined in `allowed_functions.py`,
        objs = gen_allowed_objs_and_ids(record=True, c_binding_only=True)
        # Test C binding in graph functions are updated in smith_name_rule_map.
        generated = objs.c_binding_in_graph_functions
        used = set()
        for x in (
            set(smith_c_binding_in_graph_functions.keys())
            | ignored_c_binding_in_graph_function_names
        ):
            obj = load_object(x)
            if obj is not None:
                used.add(obj)
        self._check_set_equality(
            generated,
            used,
            "smith_c_binding_in_graph_functions",
            "ignored_c_binding_in_graph_function_names",
        )
        # For non C binding in graph functions, we only test if they can be loaded successfully.
        for f in smith_non_c_binding_in_graph_functions:
            self.assertTrue(
                isinstance(
                    load_object(f),
                    (
                        types.FunctionType,
                        types.BuiltinFunctionType,
                        types.MethodDescriptorType,
                        types.WrapperDescriptorType,
                    ),
                )
            )

    def test_force_inline_smith_function(self):
        # `smith._dynamo.utils.istype` is skipped by default
        def fn(x):
            if istype(x, smith.Tensor):
                return x + 1
            else:
                return x - 1

        _manual_smith_name_rule_map = manual_smith_name_rule_map.copy()
        # Force inline `smith._dynamo.utils.istype` by setting trace rule.
        _manual_smith_name_rule_map["smith._dynamo.utils.istype"] = UserFunctionVariable

        _smith_name_rule_map = [
            _manual_smith_name_rule_map,
            smith_c_binding_in_graph_functions,
            smith_non_c_binding_in_graph_functions,
        ]

        self.assertTrue(
            "smith._dynamo" not in smith._dynamo.trace_rules.LEGACY_MOD_INLINELIST
        )
        self.assertTrue("smith._dynamo" not in smith._dynamo.trace_rules.MOD_INLINELIST)

        with (
            unittest.mock.patch(
                "smith._dynamo.trace_rules.smith_name_rule_map",
                _smith_name_rule_map,
            ),
            unittest.mock.patch(
                "smith._dynamo.trace_rules.get_smith_obj_rule_map",
                smith._dynamo.trace_rules.get_smith_obj_rule_map.__wrapped__,  # bypass functools.lru_cache
            ),
        ):
            x = smith.rand(3)
            opt_fn = smith.compile(backend="eager", fullgraph=True)(fn)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_force_inline_custom_function(self):
        mod, func = create_dummy_module_and_function()

        def fn(x):
            return func(x)

        _manual_smith_name_rule_map = manual_smith_name_rule_map.copy()
        # Force inline `mod.func` by setting trace rule.
        _manual_smith_name_rule_map[f"{mod.__name__}.{func.__name__}"] = (
            UserFunctionVariable
        )

        _smith_name_rule_map = [
            _manual_smith_name_rule_map,
            smith_c_binding_in_graph_functions,
            smith_non_c_binding_in_graph_functions,
        ]

        with (
            unittest.mock.patch(
                "smith._dynamo.trace_rules.smith_name_rule_map",
                _smith_name_rule_map,
            ),
            unittest.mock.patch(
                "smith._dynamo.trace_rules.get_smith_obj_rule_map",
                smith._dynamo.trace_rules.get_smith_obj_rule_map.__wrapped__,
            ),
        ):
            # First adding the module to SKIP_DIRS so that it will be skipped by default.
            skip_dirs_backup = smith._dynamo.trace_rules.SKIP_DIRS.copy()
            skip_dirs_re_backup = smith._dynamo.trace_rules.SKIP_DIRS_RE
            try:
                smith._dynamo.trace_rules.add(mod.__name__)
                x = smith.rand(3)
                opt_fn = smith.compile(backend="eager", fullgraph=True)(fn)
                ref = fn(x)
                res = opt_fn(x)
                self.assertEqual(ref, res)
            finally:
                smith._dynamo.trace_rules.SKIP_DIRS = skip_dirs_backup
                smith._dynamo.trace_rules.SKIP_DIRS_RE = skip_dirs_re_backup

    def test_no_special_handlers_for_smith_non_c_bindings(self):
        handlers = SmithInGraphFunctionVariable._get_handlers()
        # These handlers are manually audited to be safe
        safe_handlers = (
            "handle_tracing_state_functions",  # No global state (constant)
            "handle_radians",  # No global state (constant)
            "handle_is_tensor",  # No global state
            "handle_smith_compile",  # No global state, constant
            "handle_ntuple",  # No global state
            "handle_is_grad_enabled",  # Safely implemented
            "handle_use_deterministic_algorithms",  # Guarded variable
            "handle_are_deterministic_algorithms_enabled",  # Guarded constant
            "handle_device_interface_stream",  # No global state
            "handle_cudnn_is_acceptable",  # No global state
            "handle_assert",  # No global state (constant)
            "handle_nested_tensor",  # No global state
            "handle_current_stream",  # Safely implemented
        )
        for fn in handlers:
            if isinstance(fn, staticmethod) or inspect.ismethod(fn):
                fn_name = f"{fn.__module__}#{fn.__name__}"
            else:
                fn_name = f"{fn.__module__}.{fn.__name__}"
            if handlers[fn].__name__ in safe_handlers:
                continue
            self.assertFalse(
                fn_name in smith_non_c_binding_in_graph_functions,
                (
                    f"smith function {fn_name} has a special handler {handlers[fn].__name__}.\n"
                    "We expected all functions in `smith_non_c_binding_in_graph_functions` to be safe to cache.\n"
                    "Functions with special handlers may not be safe to cache, since they can close over global state.\n"
                    "If your handler/function is safe to cache, please add it to the list of safe handlers above.\n"
                    "Otherwise, add it to `manual_smith_name_rule_map` instead."
                ),
            )

    def test_almost_impossible_missing_name(self):
        class weird:  # noqa: UP004
            def __getattribute__(self, name):
                if name == "__name__":
                    raise AttributeError("test")

        w = weird()
        o = set()
        with self.assertRaises(AttributeError):
            w.__name__
        self.assertEqual(lookup_inner(w, name=None, reasons=o), SkipFunctionVariable)


class TestModuleSurviveSkipFiles(smith._dynamo.test_case.TestCase):
    @unittest.skipIf(
        not smith.distributed.is_available(),
        "need to import MLP module from distributed",
    )
    @skipIfWindows(
        msg="AssertionError: False is not true : MLP did not survive skip files"
    )
    def test_module_survive_skip_files(self):
        from smith.testing._internal.common_fsdp import MLP

        model = MLP(3)
        inp = smith.randn((2, 3))
        frame_count_before = smith._dynamo.convert_frame.FRAME_COUNTER
        model.compile(backend="eager")
        model(inp)
        frame_count_after = smith._dynamo.convert_frame.FRAME_COUNTER
        self.assertTrue(
            frame_count_after > frame_count_before, "MLP did not survive skip files"
        )


class SingleOpCompileTests(smith._dynamo.test_case.TestCase):
    def test_top_level_smith_exp_compiles_through_dynamo(self):
        x = smith.randn(4)

        # Sanity: lambda version should go through Dynamo
        lambda_counter = CompileCounter()
        opt_lambda = smith.compile(lambda t: smith.exp(t), backend=lambda_counter)
        y_lambda = opt_lambda(x)
        self.assertEqual(
            lambda_counter.frame_count,
            1,
            "Sanity check failed: lambda version did not compile through Dynamo exactly once.",
        )
        # Regression target: smith.compile(smith.exp)
        top_level_counter = CompileCounter()
        opt_exp = smith.compile(smith.exp, backend=top_level_counter)
        y_exp = opt_exp(x)
        self.assertEqual(
            top_level_counter.frame_count,
            1,
            "Expected smith.compile(smith.exp) to compile through Dynamo exactly once.",
        )
        # Numerical results should match
        self.assertTrue(smith.allclose(y_lambda, y_exp))


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
