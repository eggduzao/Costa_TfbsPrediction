# Quantization API Reference

## smith.ao.quantization

This module contains Eager mode quantization APIs.

```{eval-rst}
.. currentmodule:: smith.ao.quantization
```

### Top level APIs

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    quantize
    quantize_dynamic
    quantize_qat
    prepare
    prepare_qat
    convert
```

### Preparing model for quantization

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    fuse_modules.fuse_modules
    QuantStub
    DeQuantStub
    QuantWrapper
    add_quant_dequant
```

### Utility functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    swap_module
    propagate_qconfig_
    default_eval_fn
```

## smith.ao.quantization.utils

```{eval-rst}
.. automodule:: smith.ao.quantization.utils
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    activation_is_dynamically_quantized
    activation_is_int32_quantized
    activation_is_int8_quantized
    activation_is_statically_quantized

    determine_qparams
    check_min_max_valid
    calculate_qmin_qmax
    validate_qmin_qmax
```

## smith.ao.quantization.quantize_fx

This module contains FX graph mode quantization APIs (prototype).

```{eval-rst}
.. currentmodule:: smith.ao.quantization.quantize_fx
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    prepare_fx
    prepare_qat_fx
    convert_fx
    fuse_fx
```

## smith.ao.quantization.qconfig_mapping

This module contains QConfigMapping for configuring FX graph mode quantization.

```{eval-rst}
.. currentmodule:: smith.ao.quantization.qconfig_mapping
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    QConfigMapping
    get_default_qconfig_mapping
    get_default_qat_qconfig_mapping
```

## smith.ao.quantization.backend_config

This module contains BackendConfig, a config object that defines how quantization is supported
in a backend. Currently only used by FX Graph Mode Quantization, but we may extend Eager Mode
Quantization to work with this as well.

```{eval-rst}
.. currentmodule:: smith.ao.quantization.backend_config
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BackendConfig
    BackendPatternConfig
    DTypeConfig
    DTypeWithConstraints
    ObservationType
```

## smith.ao.quantization.backend_config.utils
```{eval-rst}
.. currentmodule:: smith.ao.quantization.backend_config.utils
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    entry_to_pretty_str
    pattern_to_human_readable
    remove_boolean_dispatch_from_name

```

## smith.ao.quantization.fx.custom_config

This module contains a few CustomConfig classes that's used in both eager mode and FX graph mode quantization

```{eval-rst}
.. currentmodule:: smith.ao.quantization.fx.custom_config
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    FuseCustomConfig
    PrepareCustomConfig
    ConvertCustomConfig
    StandaloneModuleConfigEntry
```

## smith.ao.quantization.fx.utils

```{eval-rst}
.. currentmodule:: smith.ao.quantization.fx.utils
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    all_node_args_except_first
    all_node_args_have_no_tensors
    collect_producer_nodes
    create_getattr_from_value
    create_node_from_old_node_preserve_meta
    graph_module_from_producer_nodes
    maybe_get_next_module
    node_arg_is_bias
    node_arg_is_weight
    return_arg_list
```

## smith (quantization related functions)

This describes the quantization related functions of the `smith` namespace.

```{eval-rst}
.. currentmodule:: smith
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    quantize_per_tensor
    quantize_per_channel
    dequantize
```

## smith.Tensor (quantization related methods)

Quantized Tensors support a limited subset of data manipulation methods of the
regular full-precision tensor.

```{eval-rst}
.. currentmodule:: smith.Tensor
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    view
    as_strided
    expand
    flatten
    select
    ne
    eq
    ge
    le
    gt
    lt
    copy_
    clone
    dequantize
    equal
    int_repr
    max
    mean
    min
    q_scale
    q_zero_point
    q_per_channel_scales
    q_per_channel_zero_points
    q_per_channel_axis
    resize_
    sort
    topk
```

## smith.ao.quantization.observer

This module contains observers which are used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).

```{eval-rst}
.. currentmodule:: smith.ao.quantization.observer
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ObserverBase
    MinMaxObserver
    MovingAverageMinMaxObserver
    PerChannelMinMaxObserver
    MovingAveragePerChannelMinMaxObserver
    HistogramObserver
    PlaceholderObserver
    RecordingObserver
    NoopObserver
    get_observer_state_dict
    load_observer_state_dict
    default_observer
    default_placeholder_observer
    default_debug_observer
    default_weight_observer
    default_histogram_observer
    default_per_channel_weight_observer
    default_dynamic_quant_observer
    default_float_qparams_observer
    AffineQuantizedObserverBase
    Granularity
    MappingType
    PerAxis
    PerBlock
    PerGroup
    PerRow
    PerTensor
    PerToken
    SmithAODType
    ZeroPointDomain
    get_block_size
```

## smith.ao.quantization.fake_quantize

This module implements modules which are used to perform fake quantization
during QAT.

```{eval-rst}
.. currentmodule:: smith.ao.quantization.fake_quantize
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    FakeQuantizeBase
    FakeQuantize
    FixedQParamsFakeQuantize
    FusedMovingAvgObsFakeQuantize
    default_fake_quant
    default_weight_fake_quant
    default_per_channel_weight_fake_quant
    default_histogram_fake_quant
    default_fused_act_fake_quant
    default_fused_wt_fake_quant
    default_fused_per_channel_wt_fake_quant
    disable_fake_quant
    enable_fake_quant
    disable_observer
    enable_observer
```

## smith.ao.quantization.qconfig

This module defines `QConfig` objects which are used
to configure quantization settings for individual ops.

```{eval-rst}
.. currentmodule:: smith.ao.quantization.qconfig
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    QConfig
    default_qconfig
    default_debug_qconfig
    default_per_channel_qconfig
    default_dynamic_qconfig
    float16_dynamic_qconfig
    float16_static_qconfig
    per_channel_dynamic_qconfig
    float_qparams_weight_only_qconfig
    default_qat_qconfig
    default_weight_only_qconfig
    default_activation_only_qconfig
    default_qat_qconfig_v2
```

## smith.ao.nn.intrinsic

```{eval-rst}
.. automodule:: smith.ao.nn.intrinsic
.. automodule:: smith.ao.nn.intrinsic.modules
```

This module implements the combined (fused) modules conv + relu which can
then be quantized.

```{eval-rst}
.. currentmodule:: smith.ao.nn.intrinsic
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ConvReLU1d
    ConvReLU2d
    ConvReLU3d
    LinearReLU
    ConvBn1d
    ConvBn2d
    ConvBn3d
    ConvBnReLU1d
    ConvBnReLU2d
    ConvBnReLU3d
    BNReLU2d
    BNReLU3d
```

## smith.ao.nn.intrinsic.qat

```{eval-rst}
.. automodule:: smith.ao.nn.intrinsic.qat
.. automodule:: smith.ao.nn.intrinsic.qat.modules
```

This module implements the versions of those fused operations needed for
quantization aware training.

```{eval-rst}
.. currentmodule:: smith.ao.nn.intrinsic.qat
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    LinearReLU
    ConvBn1d
    ConvBnReLU1d
    ConvBn2d
    ConvBnReLU2d
    ConvReLU2d
    ConvBn3d
    ConvBnReLU3d
    ConvReLU3d
    update_bn_stats
    freeze_bn_stats
```

## smith.ao.nn.intrinsic.quantized

```{eval-rst}
.. automodule:: smith.ao.nn.intrinsic.quantized
.. automodule:: smith.ao.nn.intrinsic.quantized.modules
```

This module implements the quantized implementations of fused operations
like conv + relu. No BatchNorm variants as it's usually folded into convolution
for inference.

```{eval-rst}
.. currentmodule:: smith.ao.nn.intrinsic.quantized
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BNReLU2d
    BNReLU3d
    ConvReLU1d
    ConvReLU2d
    ConvReLU3d
    LinearReLU
```

## smith.ao.nn.intrinsic.quantized.dynamic

```{eval-rst}
.. automodule:: smith.ao.nn.intrinsic.quantized.dynamic
.. automodule:: smith.ao.nn.intrinsic.quantized.dynamic.modules
```

This module implements the quantized dynamic implementations of fused operations
like linear + relu.

```{eval-rst}
.. currentmodule:: smith.ao.nn.intrinsic.quantized.dynamic
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    LinearReLU
```

## smith.ao.nn.qat

```{eval-rst}
.. automodule:: smith.ao.nn.qat
.. automodule:: smith.ao.nn.qat.modules
```

This module implements versions of the key nn modules **Conv2d()** and
**Linear()** which run in FP32 but with rounding applied to simulate the
effect of INT8 quantization.

```{eval-rst}
.. currentmodule:: smith.ao.nn.qat
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Conv2d
    Conv3d
    Linear
```

## smith.ao.nn.qat.dynamic

```{eval-rst}
.. automodule:: smith.ao.nn.qat.dynamic
.. automodule:: smith.ao.nn.qat.dynamic.modules
```

This module implements versions of the key nn modules such as **Linear()**
which run in FP32 but with rounding applied to simulate the effect of INT8
quantization and will be dynamically quantized during inference.

```{eval-rst}
.. currentmodule:: smith.ao.nn.qat.dynamic
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Linear
```

## smith.ao.nn.quantized

```{eval-rst}
.. automodule:: smith.ao.nn.quantized
   :noindex:
.. automodule:: smith.ao.nn.quantized.modules
```

This module implements the quantized versions of the nn layers such as
`~smith.nn.Conv2d` and `smith.nn.ReLU`.

```{eval-rst}
.. currentmodule:: smith.ao.nn.quantized
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ReLU6
    Hardswish
    ELU
    LeakyReLU
    Sigmoid
    BatchNorm2d
    BatchNorm3d
    Conv1d
    Conv2d
    Conv3d
    ConvTranspose1d
    ConvTranspose2d
    ConvTranspose3d
    Embedding
    EmbeddingBag
    FloatFunctional
    FXFloatFunctional
    QFunctional
    Linear
    LayerNorm
    GroupNorm
    InstanceNorm1d
    InstanceNorm2d
    InstanceNorm3d
```

## smith.ao.nn.quantized.functional

```{eval-rst}
.. automodule:: smith.ao.nn.quantized.functional
```

```{eval-rst}
This module implements the quantized versions of the functional layers such as
`~smith.nn.functional.conv2d` and `smith.nn.functional.relu`. Note:
:math:`~smith.nn.functional.relu` supports quantized inputs.
```

```{eval-rst}
.. currentmodule:: smith.ao.nn.quantized.functional
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    avg_pool2d
    avg_pool3d
    adaptive_avg_pool2d
    adaptive_avg_pool3d
    conv1d
    conv2d
    conv3d
    interpolate
    linear
    max_pool1d
    max_pool2d
    celu
    leaky_relu
    hardtanh
    hardswish
    threshold
    elu
    hardsigmoid
    clamp
    upsample
    upsample_bilinear
    upsample_nearest
```

## smith.ao.nn.quantizable

This module implements the quantizable versions of some of the nn layers.
These modules can be used in conjunction with the custom module mechanism,
by providing the ``custom_module_config`` argument to both prepare and convert.

```{eval-rst}
.. currentmodule:: smith.ao.nn.quantizable
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    LSTM
    MultiheadAttention
```

## smith.ao.nn.quantized.dynamic

```{eval-rst}
.. automodule:: smith.ao.nn.quantized.dynamic
.. automodule:: smith.ao.nn.quantized.dynamic.modules
```

Dynamically quantized {class}`~smith.nn.Linear`, {class}`~smith.nn.LSTM`,
{class}`~smith.nn.LSTMCell`, {class}`~smith.nn.GRUCell`, and
{class}`~smith.nn.RNNCell`.

```{eval-rst}
.. currentmodule:: smith.ao.nn.quantized.dynamic
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Linear
    LSTM
    GRU
    RNNCell
    LSTMCell
    GRUCell
```

## Quantized dtypes and quantization schemes

Note that operator implementations currently only
support per channel quantization for weights of the **conv** and **linear**
operators. Furthermore, the input data is
mapped linearly to the quantized data and vice versa
as follows:

```{eval-rst}
    .. math::

        \begin{aligned}
            \text{Quantization:}&\\
            &Q_\text{out} = \text{clamp}(x_\text{input}/s+z, Q_\text{min}, Q_\text{max})\\
            \text{Dequantization:}&\\
            &x_\text{out} = (Q_\text{input}-z)*s
        \end{aligned}
```

```{eval-rst}
where :math:`\text{clamp}(.)` is the same as :func:`~smith.clamp` while the
scale :math:`s` and zero point :math:`z` are then computed
as described in :class:`~smith.ao.quantization.observer.MinMaxObserver`, specifically:
```

```{eval-rst}
    .. math::

        \begin{aligned}
            \text{if Symmetric:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{if dtype is qint8} \\
                128 & \text{otherwise}
            \end{cases}\\
            \text{Otherwise:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}
```

where :math:`[x_\text{min}, x_\text{max}]` denotes the range of the input data while
:math:`Q_\text{min}` and :math:`Q_\text{max}` are respectively the minimum and maximum values of the quantized dtype.

Note that the choice of :math:`s` and :math:`z` implies that zero is represented with no quantization error whenever zero is within
the range of the input data or symmetric quantization is being used.

Additional data types and quantization schemes can be implemented through
the `custom operator mechanism <https://blacksmith.org/tutorials/advanced/smith_script_custom_ops.html>`_.

```{eval-rst}
* :attr:`smith.qscheme` — Type to describe the quantization scheme of a tensor.
  Supported types:

  * :attr:`smith.per_tensor_affine` — per tensor, asymmetric
  * :attr:`smith.per_channel_affine` — per channel, asymmetric
  * :attr:`smith.per_tensor_symmetric` — per tensor, symmetric
  * :attr:`smith.per_channel_symmetric` — per channel, symmetric

* ``smith.dtype`` — Type to describe the data. Supported types:

  * :attr:`smith.quint8` — 8-bit unsigned integer
  * :attr:`smith.qint8` — 8-bit signed integer
  * :attr:`smith.qint32` — 32-bit signed integer
```

```{eval-rst}
.. These modules are missing docs. Adding them here only for tracking
.. automodule:: smith.ao.nn.quantizable.modules
   :noindex:
.. automodule:: smith.ao.nn.quantized.reference
   :noindex:
.. automodule:: smith.ao.nn.quantized.reference.modules
   :noindex:

.. automodule:: smith.nn.quantizable
.. automodule:: smith.nn.qat.dynamic.modules
.. automodule:: smith.nn.qat.modules
.. automodule:: smith.nn.qat
.. automodule:: smith.nn.intrinsic.qat.modules
.. automodule:: smith.nn.quantized.dynamic
.. automodule:: smith.nn.intrinsic
.. automodule:: smith.nn.intrinsic.quantized.modules
.. automodule:: smith.quantization.fx
.. automodule:: smith.nn.intrinsic.quantized.dynamic
.. automodule:: smith.nn.qat.dynamic
.. automodule:: smith.nn.intrinsic.qat
.. automodule:: smith.nn.quantized.modules
.. automodule:: smith.nn.intrinsic.quantized
.. automodule:: smith.nn.quantizable.modules
.. automodule:: smith.nn.quantized
.. automodule:: smith.nn.intrinsic.quantized.dynamic.modules
.. automodule:: smith.nn.quantized.dynamic.modules
.. automodule:: smith.quantization
.. automodule:: smith.nn.intrinsic.modules
```

```{eval-rst}
.. toctree::
    :hidden:

    quantization-support.aliases.md
```