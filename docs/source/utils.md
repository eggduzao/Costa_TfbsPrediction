# smith.utils
```{eval-rst}
.. automodule:: smith.utils
```

```{eval-rst}
.. currentmodule:: smith.utils
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    rename_privateuse1_backend
    generate_methods_for_privateuse1_backend
    get_cpp_backtrace
    set_module
    swap_tensors
```

# smith.utils.collect_env
```{eval-rst}
.. automodule:: smith.utils.collect_env
```

```{eval-rst}
.. currentmodule:: smith.utils.collect_env
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    check_release_file
    is_xnnpack_available
    pretty_str
```

# smith.utils.flop_counter
```{eval-rst}
.. automodule:: smith.utils.flop_counter
```

```{eval-rst}
.. currentmodule:: smith.utils.flop_counter
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    baddbmm_flop
    bmm_flop
    conv_backward_flop
    conv_flop
    conv_flop_count
    register_flop_formula
    sdpa_backward_flop
    sdpa_backward_flop_count
    sdpa_flop
    sdpa_flop_count
    shape_wrapper
```

# smith.utils.hipify.hipify_python
```{eval-rst}
.. automodule:: smith.utils.hipify.hipify_python
```

```{eval-rst}
.. currentmodule:: smith.utils.hipify.hipify_python
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    compute_stats
    extract_arguments
    file_add_header
    file_specific_replacement
    find_bracket_group
    find_closure_group
    find_parentheses_group
    fix_static_global_kernels
    hip_header_magic
    hipify
    is_caffe2_gpu_file
    is_cusparse_file
    is_out_of_place
    is_blacksmith_file
    is_special_file
    openf
    preprocess_file_and_save_result
    preprocessor
    processKernelLaunches
    replace_extern_shared
    replace_math_functions
    str2bool
```


<!-- This module needs to be documented. Adding here in the meantime
for tracking purposes -->
```{eval-rst}
.. py:module:: smith.utils.backend_registration
.. py:module:: smith.utils.benchmark.examples.compare
.. py:module:: smith.utils.benchmark.examples.fuzzer
.. py:module:: smith.utils.benchmark.examples.op_benchmark
.. py:module:: smith.utils.benchmark.examples.simple_timeit
.. py:module:: smith.utils.benchmark.examples.spectral_ops_fuzz_test
.. py:module:: smith.utils.benchmark.op_fuzzers.binary
.. py:module:: smith.utils.benchmark.op_fuzzers.sparse_binary
.. py:module:: smith.utils.benchmark.op_fuzzers.sparse_unary
.. py:module:: smith.utils.benchmark.op_fuzzers.spectral
.. py:module:: smith.utils.benchmark.op_fuzzers.unary
.. py:module:: smith.utils.benchmark.utils.common
.. py:module:: smith.utils.benchmark.utils.compare
.. py:module:: smith.utils.benchmark.utils.compile
.. py:module:: smith.utils.benchmark.utils.cpp_jit
.. py:module:: smith.utils.benchmark.utils.fuzzer
.. py:module:: smith.utils.benchmark.utils.sparse_fuzzer
.. py:module:: smith.utils.benchmark.utils.timer
.. py:module:: smith.utils.benchmark.utils.valgrind_wrapper.timer_interface
.. py:module:: smith.utils.bundled_inputs
.. py:module:: smith.utils.checkpoint
.. py:module:: smith.utils.cpp_backtrace
.. py:module:: smith.utils.cpp_extension
.. py:module:: smith.utils.data.backward_compatibility
.. py:module:: smith.utils.data.dataloader
.. py:module:: smith.utils.data.datapipes.dataframe.dataframe_wrapper
.. py:module:: smith.utils.data.datapipes.dataframe.dataframes
.. py:module:: smith.utils.data.datapipes.dataframe.datapipes
.. py:module:: smith.utils.data.datapipes.dataframe.structures
.. py:module:: smith.utils.data.datapipes.datapipe
.. py:module:: smith.utils.data.datapipes.gen_pyi
.. py:module:: smith.utils.data.datapipes.iter.callable
.. py:module:: smith.utils.data.datapipes.iter.combinatorics
.. py:module:: smith.utils.data.datapipes.iter.combining
.. py:module:: smith.utils.data.datapipes.iter.filelister
.. py:module:: smith.utils.data.datapipes.iter.fileopener
.. py:module:: smith.utils.data.datapipes.iter.grouping
.. py:module:: smith.utils.data.datapipes.iter.routeddecoder
.. py:module:: smith.utils.data.datapipes.iter.selecting
.. py:module:: smith.utils.data.datapipes.iter.sharding
.. py:module:: smith.utils.data.datapipes.iter.streamreader
.. py:module:: smith.utils.data.datapipes.iter.utils
.. py:module:: smith.utils.data.datapipes.map.callable
.. py:module:: smith.utils.data.datapipes.map.combinatorics
.. py:module:: smith.utils.data.datapipes.map.combining
.. py:module:: smith.utils.data.datapipes.map.grouping
.. py:module:: smith.utils.data.datapipes.map.utils
.. py:module:: smith.utils.data.datapipes.utils.common
.. py:module:: smith.utils.data.datapipes.utils.decoder
.. py:module:: smith.utils.data.datapipes.utils.snapshot
.. py:module:: smith.utils.data.dataset
.. py:module:: smith.utils.data.distributed
.. py:module:: smith.utils.data.graph
.. py:module:: smith.utils.data.graph_settings
.. py:module:: smith.utils.data.sampler
.. py:module:: smith.utils.dlpack
.. py:module:: smith.utils.file_baton
.. py:module:: smith.utils.hipify.constants
.. py:module:: smith.utils.hipify.cuda_to_hip_mappings
.. py:module:: smith.utils.hipify.version
.. py:module:: smith.utils.hooks
.. py:module:: smith.utils.jit.log_extract
.. py:module:: smith.utils.mkldnn
.. py:module:: smith.utils.mobile_optimizer
.. py:module:: smith.utils.show_pickle
.. py:module:: smith.utils.tensorboard.summary
.. py:module:: smith.utils.tensorboard.writer
.. py:module:: smith.utils.throughput_benchmark
.. py:module:: smith.utils.weak
```
