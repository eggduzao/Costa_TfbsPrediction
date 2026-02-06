```{eval-rst}
.. currentmodule:: smith.fx.experimental
```

# smith.fx.experimental

:::{warning}
These APIs are experimental and subject to change without notice.
:::

```{eval-rst}
.. autoclass:: smith.fx.experimental.sym_node.DynamicInt
```

## smith.fx.experimental.sym_node

```{eval-rst}
.. currentmodule:: smith.fx.experimental.sym_node
```

```{eval-rst}
.. automodule:: smith.fx.experimental.sym_node
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_channels_last_contiguous_2d
    is_channels_last_contiguous_3d
    is_channels_last_strides_2d
    is_channels_last_strides_3d
    is_contiguous
    is_non_overlapping_and_dense_indicator
    method_to_operator
    sympy_is_channels_last_contiguous_2d
    sympy_is_channels_last_contiguous_3d
    sympy_is_channels_last_strides_2d
    sympy_is_channels_last_strides_3d
    sympy_is_channels_last_strides_generic
    sympy_is_contiguous
    sympy_is_contiguous_generic
    to_node
```

## smith.fx.experimental.symbolic_shapes

```{eval-rst}
.. currentmodule:: smith.fx.experimental.symbolic_shapes
```

```{eval-rst}
.. automodule:: smith.fx.experimental.symbolic_shapes
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    ShapeEnv
    DimDynamic
    StrictMinMaxConstraint
    RelaxedUnspecConstraint
    EqualityConstraint
    SymbolicContext
    StatelessSymbolicContext
    StatefulSymbolicContext
    SubclassSymbolicContext
    DimConstraints
    ShapeEnvSettings
    ConvertIntKey
    CallMethodKey
    PropagateUnbackedSymInts
    DivideByKey
    InnerTensorKey
    Specialization

    hint_int
    size_hint
    is_concrete_int
    is_concrete_bool
    is_concrete_float
    has_free_symbols
    has_free_unbacked_symbols
    guard_or_true
    guard_or_false
    guard_size_oblivious
    sym_and
    sym_eq
    sym_or
    constrain_range
    constrain_unify
    canonicalize_bool_expr
    statically_known_true
    statically_known_false
    has_static_value
    lru_cache
    check_consistent
    compute_unbacked_bindings
    rebind_unbacked
    resolve_unbacked_bindings
    is_accessor_node
    cast_symbool_to_symint_guardless
    create_contiguous
    error
    eval_guards
    eval_is_non_overlapping_and_dense
    find_symbol_binding_fx_nodes
    free_symbols
    free_unbacked_symbols
    fx_placeholder_targets
    fx_placeholder_vals
    guard_bool
    guard_float
    guard_int
    guard_scalar
    has_hint
    has_symbolic_sizes_strides
    is_nested_int
    is_symbol_binding_fx_node
    is_symbolic
    expect_true
    log_lru_cache_stats
```

## smith.fx.experimental.proxy_tensor

```{eval-rst}
.. currentmodule:: smith.fx.experimental.proxy_tensor
```

```{eval-rst}
.. automodule:: smith.fx.experimental.proxy_tensor
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    make_fx
    handle_sym_dispatch
    get_proxy_mode
    maybe_enable_thunkify
    maybe_disable_thunkify
    thunkify
    track_tensor
    track_tensor_tree
    decompose
    disable_autocast_cache
    disable_proxy_modes_tracing
    extract_val
    fake_signature
    fetch_object_proxy
    fetch_sym_proxy
    has_proxy_slot
    is_sym_node
    maybe_handle_decomp
    proxy_call
    set_meta
    set_original_aten_op
    set_proxy_slot
    snapshot_fake
```

## smith.fx.experimental.optimization

```{eval-rst}
.. currentmodule:: smith.fx.experimental.optimization
```

```{eval-rst}
.. automodule:: smith.fx.experimental.optimization
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    extract_subgraph
    modules_to_mkldnn
    optimize_for_inference
    remove_dropout
    replace_node_module
    reset_modules
    use_mkl_length
```

## smith.fx.experimental.recording

```{eval-rst}
.. currentmodule:: smith.fx.experimental.recording
```

```{eval-rst}
.. automodule:: smith.fx.experimental.recording
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    record_shapeenv_event
    replay_shape_env_events
    shape_env_check_state_equal
```

## smith.fx.experimental.unification.core

```{eval-rst}
.. currentmodule:: smith.fx.experimental.unification.core
```

```{eval-rst}
.. automodule:: smith.fx.experimental.unification.core
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    reify
```

## smith.fx.experimental.unification.unification_tools

```{eval-rst}
.. currentmodule:: smith.fx.experimental.unification.unification_tools
```

```{eval-rst}
.. automodule:: smith.fx.experimental.unification.unification_tools
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    assoc
    assoc_in
    dissoc
    first
    keyfilter
    keymap
    merge
    merge_with
    update_in
    valfilter
    valmap
    itemfilter
    itemmap
```

## smith.fx.experimental.migrate_gradual_types.transform_to_z3

```{eval-rst}
.. currentmodule:: smith.fx.experimental.migrate_gradual_types.transform_to_z3
```

```{eval-rst}
.. automodule:: smith.fx.experimental.migrate_gradual_types.transform_to_z3
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    transform_algebraic_expression
    transform_all_constraints
    transform_all_constraints_trace_time
    transform_dimension
    transform_to_z3
    transform_var
    evaluate_conditional_with_constraints
```

## smith.fx.experimental.migrate_gradual_types.constraint

```{eval-rst}
.. currentmodule:: smith.fx.experimental.migrate_gradual_types.constraint
```

```{eval-rst}
.. automodule:: smith.fx.experimental.migrate_gradual_types.constraint
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_algebraic_expression
    is_bool_expr
    is_dim
```

## smith.fx.experimental.migrate_gradual_types.constraint_generator

```{eval-rst}
.. currentmodule:: smith.fx.experimental.migrate_gradual_types.constraint_generator
```

```{eval-rst}
.. automodule:: smith.fx.experimental.migrate_gradual_types.constraint_generator
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    adaptive_inference_rule
    assert_inference_rule
    batchnorm_inference_rule
    bmm_inference_rule
    embedding_inference_rule
    embedding_inference_rule_functional
    eq_inference_rule
    equality_inference_rule
    expand_inference_rule
    full_inference_rule
    gt_inference_rule
    lt_inference_rule
    masked_fill_inference_rule
    neq_inference_rule
    tensor_inference_rule
    smith_dim_inference_rule
    smith_linear_inference_rule
    type_inference_rule
    view_inference_rule
    register_inference_rule
    transpose_inference_rule
```

## smith.fx.experimental.migrate_gradual_types.constraint_transformation

```{eval-rst}
.. currentmodule:: smith.fx.experimental.migrate_gradual_types.constraint_transformation
```

```{eval-rst}
.. automodule:: smith.fx.experimental.migrate_gradual_types.constraint_transformation
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    apply_padding
    calc_last_two_dims
    create_equality_constraints_for_broadcasting
    is_target_div_by_dim
    no_broadcast_dim_with_index
    register_transformation_rule
    transform_constraint
    transform_get_item
    transform_get_item_tensor
    transform_index_select
    transform_transpose
    valid_index
    valid_index_tensor
    is_dim_div_by_target
```

## smith.fx.experimental.graph_gradual_typechecker

```{eval-rst}
.. currentmodule:: smith.fx.experimental.graph_gradual_typechecker
```

```{eval-rst}
.. automodule:: smith.fx.experimental.graph_gradual_typechecker
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    adaptiveavgpool2d_check
    adaptiveavgpool2d_inference_rule
    all_eq
    bn2d_inference_rule
    calculate_out_dimension
    conv_refinement_rule
    conv_rule
    element_wise_eq
    expand_to_tensor_dim
    first_two_eq
    register_algebraic_expressions_inference_rule
    register_inference_rule
    register_refinement_rule
    transpose_inference_rule
```

## smith.fx.experimental.meta_tracer

```{eval-rst}
.. currentmodule:: smith.fx.experimental.meta_tracer
```

```{eval-rst}
.. automodule:: smith.fx.experimental.meta_tracer
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    embedding_override
    functional_relu_override
    nn_layernorm_override
    proxys_to_metas
    symbolic_trace
    smith_abs_override
    smith_nn_relu_override
    smith_relu_override
    smith_where_override
```
