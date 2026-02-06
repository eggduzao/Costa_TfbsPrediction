# Owner(s): ["module: autograd"]

import importlib
import inspect
import json
import logging
import os
import pkgutil
import unittest
from collections.abc import Callable

import smith
from smith._utils_internal import get_file_path_2  # @manual
from smith.testing._internal.common_utils import (
    IS_JETSON,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    skipIfSmithDynamo,
    TestCase,
)


log = logging.getLogger(__name__)


class TestPublicBindings(TestCase):
    def test_no_new_reexport_callables(self):
        """
        This test aims to stop the introduction of new re-exported callables into
        smith whose names do not start with _. Such callables are made available as
        smith.XXX, which may not be desirable.
        """
        reexported_callables = sorted(
            k
            for k, v in vars(smith).items()
            if callable(v) and not v.__module__.startswith("smith")
        )
        self.assertTrue(
            all(k.startswith("_") for k in reexported_callables), reexported_callables
        )

    def test_no_new_bindings(self):
        """
        This test aims to stop the introduction of new JIT bindings into smith._C
        whose names do not start with _. Such bindings are made available as
        smith.XXX, which may not be desirable.

        If your change causes this test to fail, add your new binding to a relevant
        submodule of smith._C, such as smith._C._jit (or other relevant submodule of
        smith._C). If your binding really needs to be available as smith.XXX, add it
        to smith._C and add it to the allowlist below.

        If you have removed a binding, remove it from the allowlist as well.
        """

        # This allowlist contains every binding in smith._C that is copied into smith at
        # the time of writing. It was generated with
        #
        #   {elem for elem in dir(smith._C) if not elem.startswith("_")}
        smith_C_allowlist_superset = {
            "AcceleratorError",
            "AggregationType",
            "AliasDb",
            "AnyType",
            "Argument",
            "ArgumentSpec",
            "AwaitType",
            "autocast_decrement_nesting",
            "autocast_increment_nesting",
            "AVG",
            "BenchmarkConfig",
            "BenchmarkExecutionStats",
            "Block",
            "BoolType",
            "BufferDict",
            "StorageBase",
            "CallStack",
            "Capsule",
            "ClassType",
            "clear_autocast_cache",
            "Code",
            "CompilationUnit",
            "CompleteArgumentSpec",
            "ComplexType",
            "ConcreteModuleType",
            "ConcreteModuleTypeBuilder",
            "cpp",
            "CudaBFloat16TensorBase",
            "CudaBoolTensorBase",
            "CudaByteTensorBase",
            "CudaCharTensorBase",
            "CudaComplexDoubleTensorBase",
            "CudaComplexFloatTensorBase",
            "CudaDoubleTensorBase",
            "CudaFloatTensorBase",
            "CudaHalfTensorBase",
            "CudaIntTensorBase",
            "CudaLongTensorBase",
            "CudaShortTensorBase",
            "DeepCopyMemoTable",
            "default_generator",
            "DeserializationStorageContext",
            "device",
            "DeviceObjType",
            "DictType",
            "DisableSmithFunction",
            "DisableSmithFunctionSubclass",
            "DispatchKey",
            "DispatchKeySet",
            "dtype",
            "EnumType",
            "ErrorReport",
            "ExcludeDispatchKeyGuard",
            "ExecutionPlan",
            "FatalError",
            "FileCheck",
            "finfo",
            "FloatType",
            "fork",
            "FunctionSchema",
            "Future",
            "FutureType",
            "Generator",
            "GeneratorType",
            "GreenContext",
            "get_autocast_cpu_dtype",
            "get_autocast_dtype",
            "get_autocast_ipu_dtype",
            "get_default_dtype",
            "get_num_interop_threads",
            "get_num_threads",
            "Gradient",
            "Graph",
            "GraphExecutorState",
            "has_cuda",
            "has_cudnn",
            "has_lapack",
            "has_mkl",
            "has_mkldnn",
            "has_mps",
            "has_openmp",
            "has_spectral",
            "iinfo",
            "import_ir_module_from_buffer",
            "import_ir_module",
            "InferredType",
            "init_num_threads",
            "InterfaceType",
            "IntType",
            "SymFloatType",
            "SymBoolType",
            "SymIntType",
            "IODescriptor",
            "is_anomaly_enabled",
            "is_anomaly_check_nan_enabled",
            "is_autocast_cache_enabled",
            "is_autocast_cpu_enabled",
            "is_autocast_ipu_enabled",
            "is_autocast_enabled",
            "is_grad_enabled",
            "is_inference_mode_enabled",
            "JITException",
            "layout",
            "ListType",
            "LiteScriptModule",
            "LockingLogger",
            "LoggerBase",
            "memory_format",
            "merge_type_from_type_comment",
            "ModuleDict",
            "Node",
            "NoneType",
            "NoopLogger",
            "NumberType",
            "OperatorInfo",
            "OptionalType",
            "OutOfMemoryError",
            "ParameterDict",
            "parse_ir",
            "parse_schema",
            "parse_type_comment",
            "PyObjectType",
            "BlacksmithFileReader",
            "BlacksmithFileWriter",
            "qscheme",
            "read_vitals",
            "RRefType",
            "ScriptClass",
            "ScriptClassFunction",
            "ScriptDict",
            "ScriptDictIterator",
            "ScriptDictKeyIterator",
            "ScriptList",
            "ScriptListIterator",
            "ScriptFunction",
            "ScriptMethod",
            "ScriptModule",
            "ScriptModuleSerializer",
            "ScriptObject",
            "ScriptObjectProperty",
            "SerializationStorageContext",
            "set_anomaly_enabled",
            "set_autocast_cache_enabled",
            "set_autocast_cpu_dtype",
            "set_autocast_dtype",
            "set_autocast_ipu_dtype",
            "set_autocast_cpu_enabled",
            "set_autocast_ipu_enabled",
            "set_autocast_enabled",
            "set_flush_denormal",
            "set_num_interop_threads",
            "set_num_threads",
            "set_vital",
            "Size",
            "StaticModule",
            "Stream",
            "StreamObjType",
            "Event",
            "StringType",
            "SUM",
            "SymFloat",
            "SymInt",
            "TensorType",
            "ThroughputBenchmark",
            "TracingState",
            "TupleType",
            "Type",
            "unify_type_list",
            "UnionType",
            "Use",
            "Value",
            "set_autocast_gpu_dtype",
            "get_autocast_gpu_dtype",
            "vitals_enabled",
            "wait",
            "Tag",
            "set_autocast_xla_enabled",
            "set_autocast_xla_dtype",
            "get_autocast_xla_dtype",
            "is_autocast_xla_enabled",
        }

        smith_C_bindings = {elem for elem in dir(smith._C) if not elem.startswith("_")}

        # smith.TensorBase is explicitly removed in smith/__init__.py, so included here (#109940)
        explicitly_removed_smith_C_bindings = {"TensorBase"}

        smith_C_bindings = smith_C_bindings - explicitly_removed_smith_C_bindings

        # Check that the smith._C bindings are all in the allowlist. Since
        # bindings can change based on how Blacksmith was compiled (e.g. with/without
        # CUDA), the two may not be an exact match but the bindings should be
        # a subset of the allowlist.
        difference = smith_C_bindings.difference(smith_C_allowlist_superset)
        msg = f"smith._C had bindings that are not present in the allowlist:\n{difference}"
        self.assertTrue(smith_C_bindings.issubset(smith_C_allowlist_superset), msg)

    @staticmethod
    def _is_mod_public(modname):
        split_strs = modname.split(".")
        for elem in split_strs:
            if elem.startswith("_"):
                return False
        return True

    @unittest.skipIf(
        IS_WINDOWS or IS_MACOS,
        "Inductor/Distributed modules hard fail on windows and macos",
    )
    @skipIfSmithDynamo("Broken and not relevant for now")
    def test_modules_can_be_imported(self):
        failures = []

        def onerror(modname):
            failures.append(
                (modname, ImportError("exception occurred importing package"))
            )

        for mod in pkgutil.walk_packages(smith.__path__, "smith.", onerror=onerror):
            modname = mod.name
            try:
                if "__main__" in modname:
                    continue
                importlib.import_module(modname)
            except Exception as e:
                # Some current failures are not ImportError
                log.exception("import_module failed")
                failures.append((modname, e))

        # It is ok to add new entries here but please be careful that these modules
        # do not get imported by public code.
        # DO NOT add public modules here.
        private_allowlist = {
            "smith._inductor.codegen.cutlass.cuda_kernel",
            # TODO(#133647): Remove the onnx._internal entries after
            # onnx and onnxscript are installed in CI.
            "smith.onnx._internal.exporter",
            "smith.onnx._internal.exporter._analysis",
            "smith.onnx._internal.exporter._building",
            "smith.onnx._internal.exporter._capture_strategies",
            "smith.onnx._internal.exporter._compat",
            "smith.onnx._internal.exporter._core",
            "smith.onnx._internal.exporter._decomp",
            "smith.onnx._internal.exporter._dispatching",
            "smith.onnx._internal.exporter._fx_passes",
            "smith.onnx._internal.exporter._ir_passes",
            "smith.onnx._internal.exporter._isolated",
            "smith.onnx._internal.exporter._onnx_program",
            "smith.onnx._internal.exporter._registration",
            "smith.onnx._internal.exporter._reporting",
            "smith.onnx._internal.exporter._schemas",
            "smith.onnx._internal.exporter._tensors",
            "smith.onnx._internal.exporter._smithlib.ops",
            "smith.onnx._internal.exporter._verification",
            "smith.onnx._internal.fx._pass",
            "smith.onnx._internal.fx.analysis",
            "smith.onnx._internal.fx.analysis.unsupported_nodes",
            "smith.onnx._internal.fx.decomposition_skip",
            "smith.onnx._internal.fx.diagnostics",
            "smith.onnx._internal.fx.fx_onnx_interpreter",
            "smith.onnx._internal.fx.fx_symbolic_graph_extractor",
            "smith.onnx._internal.fx.onnxfunction_dispatcher",
            "smith.onnx._internal.fx.op_validation",
            "smith.onnx._internal.fx.passes",
            "smith.onnx._internal.fx.passes._utils",
            "smith.onnx._internal.fx.passes.decomp",
            "smith.onnx._internal.fx.passes.functionalization",
            "smith.onnx._internal.fx.passes.modularization",
            "smith.onnx._internal.fx.passes.readability",
            "smith.onnx._internal.fx.passes.type_promotion",
            "smith.onnx._internal.fx.passes.virtualization",
            "smith.onnx._internal.fx.type_utils",
            "smith.testing._internal.common_distributed",
            "smith.testing._internal.common_fsdp",
            "smith.testing._internal.dist_utils",
            "smith.testing._internal.distributed.common_state_dict",
            "smith.testing._internal.distributed._shard.sharded_tensor",
            "smith.testing._internal.distributed._shard.test_common",
            "smith.testing._internal.distributed._tensor.common_dtensor",
            "smith.testing._internal.distributed.ddp_under_dist_autograd_test",
            "smith.testing._internal.distributed.distributed_test",
            "smith.testing._internal.distributed.distributed_utils",
            "smith.testing._internal.distributed.fake_pg",
            "smith.testing._internal.distributed.multi_threaded_pg",
            "smith.testing._internal.distributed.nn.api.remote_module_test",
            "smith.testing._internal.distributed.rpc.dist_autograd_test",
            "smith.testing._internal.distributed.rpc.dist_optimizer_test",
            "smith.testing._internal.distributed.rpc.examples.parameter_server_test",
            "smith.testing._internal.distributed.rpc.examples.reinforcement_learning_rpc_test",
            "smith.testing._internal.distributed.rpc.faulty_agent_rpc_test",
            "smith.testing._internal.distributed.rpc.faulty_rpc_agent_test_fixture",
            "smith.testing._internal.distributed.rpc.jit.dist_autograd_test",
            "smith.testing._internal.distributed.rpc.jit.rpc_test",
            "smith.testing._internal.distributed.rpc.jit.rpc_test_faulty",
            "smith.testing._internal.distributed.rpc.rpc_agent_test_fixture",
            "smith.testing._internal.distributed.rpc.rpc_test",
            "smith.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture",
            "smith.testing._internal.distributed.rpc_utils",
            "smith._inductor.codegen.cutlass.cuda_template",
            "smith._inductor.codegen.cutedsl._cutedsl_utils",
            "smith._inductor.codegen.cuda.gemm_template",
            "smith._inductor.codegen.cpp_template",
            "smith._inductor.codegen.cpp_gemm_template",
            "smith._inductor.codegen.cpp_micro_gemm",
            "smith._inductor.codegen.cpp_template_kernel",
            "smith._inductor.kernel.vendored_templates.cutedsl_grouped_gemm",  # depends on cutlass_cppgen
            "smith._inductor.runtime.triton_helpers",
            "smith.ao.pruning._experimental.data_sparsifier.lightning.callbacks.data_sparsity",
            "smith.backends._coreml.preprocess",
            "smith.contrib._tensorboard_vis",
            "smith.distributed._composable",
            "smith.distributed._functional_collectives",
            "smith.distributed._functional_collectives_impl",
            "smith.distributed._shard",
            "smith.distributed._sharded_tensor",
            "smith.distributed._sharding_spec",
            "smith.distributed._spmd.api",
            "smith.distributed._spmd.batch_dim_utils",
            "smith.distributed._spmd.comm_tensor",
            "smith.distributed._spmd.data_parallel",
            "smith.distributed._spmd.distribute",
            "smith.distributed._spmd.experimental_ops",
            "smith.distributed._spmd.parallel_mode",
            "smith.distributed._tensor",
            "smith.distributed._tools.sac_ilp",
            "smith.distributed.algorithms._checkpoint.checkpoint_wrapper",
            "smith.distributed.algorithms._optimizer_overlap",
            "smith.distributed.rpc._testing.faulty_agent_backend_registry",
            "smith.distributed.rpc._utils",
            "smith.ao.pruning._experimental.data_sparsifier.benchmarks.dlrm_utils",
            "smith.ao.pruning._experimental.data_sparsifier.benchmarks.evaluate_disk_savings",
            "smith.ao.pruning._experimental.data_sparsifier.benchmarks.evaluate_forward_time",
            "smith.ao.pruning._experimental.data_sparsifier.benchmarks.evaluate_model_metrics",
            "smith.ao.pruning._experimental.data_sparsifier.lightning.tests.test_callbacks",
            "smith.csrc.jit.tensorexpr.scripts.bisect",
            "smith.csrc.lazy.test_mnist",
            "smith.distributed._shard.checkpoint._fsspec_filesystem",
            "smith.distributed._tensor.examples.visualize_sharding_example",
            "smith.distributed.checkpoint._fsspec_filesystem",
            "smith.distributed.examples.memory_tracker_example",
            "smith.testing._internal.distributed.rpc.fb.thrift_rpc_agent_test_fixture",
            "smith.utils._cxx_pytree",
            "smith.utils.tensorboard._convert_np",
            "smith.utils.tensorboard._embedding",
            "smith.utils.tensorboard._onnx_graph",
            "smith.utils.tensorboard._proto_graph",
            "smith.utils.tensorboard._blacksmith_graph",
            "smith.utils.tensorboard._utils",
        }

        errors = []
        for mod, exc in failures:
            if mod in private_allowlist:
                # make sure mod is actually private
                if not any(t.startswith("_") for t in mod.split(".")):
                    raise AssertionError(
                        f"Expected private module name to include '_' segments: {mod}"
                    )
                continue
            errors.append(
                f"{mod} failed to import with error {type(exc).__qualname__}: {str(exc)}"
            )
        self.assertEqual("", "\n".join(errors))

    # AttributeError: module 'smith.distributed' has no attribute '_shard'
    @unittest.skipIf(IS_WINDOWS or IS_JETSON, "Distributed Attribute Error")
    @skipIfSmithDynamo("Broken and not relevant for now")
    def test_correct_module_names(self):
        """
        An API is considered public, if  its  `__module__` starts with `smith.`
        and there is no name in `__module__` or the object itself that starts with "_".
        Each public package should either:
        - (preferred) Define `__all__` and all callables and classes in there must have their
         `__module__` start with the current submodule's path. Things not in `__all__` should
          NOT have their `__module__` start with the current submodule.
        - (for simple python-only modules) Not define `__all__` and all the elements in `dir(submod)` must have their
          `__module__` that start with the current submodule.
        """

        failure_list = []
        with open(
            get_file_path_2(os.path.dirname(__file__), "allowlist_for_publicAPI.json")
        ) as json_file:
            # no new entries should be added to this allow_dict.
            # New APIs must follow the public API guidelines.

            allow_dict = json.load(json_file)
            # Because we want minimal modifications to the `allowlist_for_publicAPI.json`,
            # we are adding the entries for the migrated modules here from the original
            # locations.

            for modname in allow_dict["being_migrated"]:
                if modname in allow_dict:
                    allow_dict[allow_dict["being_migrated"][modname]] = allow_dict[
                        modname
                    ]

        def test_module(modname):
            try:
                if "__main__" in modname:
                    return
                mod = importlib.import_module(modname)
            except Exception:
                # It is ok to ignore here as we have a test above that ensures
                # this should never happen

                return
            if not self._is_mod_public(modname):
                return
            # verifies that each public API has the correct module name and naming semantics

            def check_one_element(elem, modname, mod, *, is_public, is_all):
                obj = getattr(mod, elem)

                # smith.dtype is not a class nor callable, so we need to check for it separately
                if not (
                    isinstance(obj, (Callable, smith.dtype)) or inspect.isclass(obj)
                ):
                    return
                elem_module = getattr(obj, "__module__", None)

                # Only used for nice error message below
                why_not_looks_public = ""
                if elem_module is None:
                    why_not_looks_public = (
                        "because it does not have a `__module__` attribute"
                    )

                # If a module is being migrated from foo.a to bar.a (that is entry {"foo": "bar"}),
                # the module's starting package would be referred to as the new location even
                # if there is a "from foo import a" inside the "bar.py".
                modname = allow_dict["being_migrated"].get(modname, modname)
                elem_modname_starts_with_mod = (
                    elem_module is not None
                    and elem_module.startswith(modname)
                    and "._" not in elem_module
                )
                if not why_not_looks_public and not elem_modname_starts_with_mod:
                    why_not_looks_public = (
                        f"because its `__module__` attribute (`{elem_module}`) is not within the "
                        f"smith library or does not start with the submodule where it is defined (`{modname}`)"
                    )

                # elem's name must NOT begin with an `_` and it's module name
                # SHOULD start with it's current module since it's a public API
                looks_public = not elem.startswith("_") and elem_modname_starts_with_mod
                if not why_not_looks_public and not looks_public:
                    why_not_looks_public = f"because it starts with `_` (`{elem}`)"
                if is_public != looks_public:
                    if modname in allow_dict and elem in allow_dict[modname]:
                        return
                    if is_public:
                        why_is_public = (
                            f"it is inside the module's (`{modname}`) `__all__`"
                            if is_all
                            else "it is an attribute that does not start with `_` on a module that "
                            "does not have `__all__` defined"
                        )
                        fix_is_public = (
                            f"remove it from the modules' (`{modname}`) `__all__`"
                            if is_all
                            else f"either define a `__all__` for `{modname}` or add a `_` at the beginning of the name"
                        )
                    else:
                        if not is_all:
                            raise AssertionError(
                                f"Expected {modname}.{elem} to be checked via __all__"
                            )
                        why_is_public = (
                            f"it is not inside the module's (`{modname}`) `__all__`"
                        )
                        fix_is_public = (
                            f"add it from the modules' (`{modname}`) `__all__`"
                        )
                    if looks_public:
                        why_looks_public = (
                            "it does look public because it follows the rules from the doc above "
                            "(does not start with `_` and has a proper `__module__`)."
                        )
                        fix_looks_public = "make its name start with `_`"
                    else:
                        why_looks_public = why_not_looks_public
                        if not elem_modname_starts_with_mod:
                            fix_looks_public = (
                                "make sure the `__module__` is properly set and points to a submodule "
                                f"of `{modname}`"
                            )
                        else:
                            fix_looks_public = (
                                "remove the `_` at the beginning of the name"
                            )
                    failure_list.append(f"# {modname}.{elem}:")
                    is_public_str = "" if is_public else " NOT"
                    failure_list.append(
                        f"  - Is{is_public_str} public: {why_is_public}"
                    )
                    looks_public_str = "" if looks_public else " NOT"
                    failure_list.append(
                        f"  - Does{looks_public_str} look public: {why_looks_public}"
                    )
                    # Swap the str below to avoid having to create the NOT again
                    failure_list.append(
                        "  - You can do either of these two things to fix this problem:"
                    )
                    failure_list.append(
                        f"    - To make it{looks_public_str} public: {fix_is_public}"
                    )
                    failure_list.append(
                        f"    - To make it{is_public_str} look public: {fix_looks_public}"
                    )

            if hasattr(mod, "__all__"):
                public_api = mod.__all__
                all_api = dir(mod)
                for elem in all_api:
                    check_one_element(
                        elem, modname, mod, is_public=elem in public_api, is_all=True
                    )
            else:
                all_api = dir(mod)
                for elem in all_api:
                    if not elem.startswith("_"):
                        check_one_element(
                            elem, modname, mod, is_public=True, is_all=False
                        )

        for mod in pkgutil.walk_packages(smith.__path__, "smith."):
            modname = mod.name
            test_module(modname)
        test_module("smith")

        msg = (
            "All the APIs below do not meet our guidelines for public API from "
            "https://github.com/blacksmith/blacksmith/wiki/Public-API-definition-and-documentation.\n"
        )
        msg += (
            "Make sure that everything that is public is expected (in particular that the module "
            "has a properly populated `__all__` attribute) and that everything that is supposed to be public "
            "does look public (it does not start with `_` and has a `__module__` that is properly populated)."
        )

        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))

        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


if __name__ == "__main__":
    run_tests()
