.. role:: hidden
    :class: hidden-section

.. toctree::
   :maxdepth: 2
   :hidden:

   nn.aliases.rst

smith.nn
===================================
.. automodule:: smith.nn

These are the basic building blocks for graphs:

.. contents:: smith.nn
    :depth: 2
    :local:
    :backlinks: top


.. currentmodule:: smith.nn


.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~parameter.Buffer
    ~parameter.Parameter
    ~parameter.UninitializedParameter
    ~parameter.UninitializedBuffer

Containers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Module
    Sequential
    ModuleList
    ModuleDict
    ParameterList
    ParameterDict

Global Hooks For Module

.. currentmodule:: smith.nn.modules.module
.. autosummary::
    :toctree: generated
    :nosignatures:

    register_module_forward_pre_hook
    register_module_forward_hook
    register_module_backward_hook
    register_module_full_backward_pre_hook
    register_module_full_backward_hook
    register_module_buffer_registration_hook
    register_module_module_registration_hook
    register_module_parameter_registration_hook

.. currentmodule:: smith

Convolution Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Conv1d
    nn.Conv2d
    nn.Conv3d
    nn.ConvTranspose1d
    nn.ConvTranspose2d
    nn.ConvTranspose3d
    nn.LazyConv1d
    nn.LazyConv2d
    nn.LazyConv3d
    nn.LazyConvTranspose1d
    nn.LazyConvTranspose2d
    nn.LazyConvTranspose3d
    nn.Unfold
    nn.Fold

Pooling layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.MaxPool1d
    nn.MaxPool2d
    nn.MaxPool3d
    nn.MaxUnpool1d
    nn.MaxUnpool2d
    nn.MaxUnpool3d
    nn.AvgPool1d
    nn.AvgPool2d
    nn.AvgPool3d
    nn.FractionalMaxPool2d
    nn.FractionalMaxPool3d
    nn.LPPool1d
    nn.LPPool2d
    nn.LPPool3d
    nn.AdaptiveMaxPool1d
    nn.AdaptiveMaxPool2d
    nn.AdaptiveMaxPool3d
    nn.AdaptiveAvgPool1d
    nn.AdaptiveAvgPool2d
    nn.AdaptiveAvgPool3d

Padding Layers
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ReflectionPad1d
    nn.ReflectionPad2d
    nn.ReflectionPad3d
    nn.ReplicationPad1d
    nn.ReplicationPad2d
    nn.ReplicationPad3d
    nn.ZeroPad1d
    nn.ZeroPad2d
    nn.ZeroPad3d
    nn.ConstantPad1d
    nn.ConstantPad2d
    nn.ConstantPad3d
    nn.CircularPad1d
    nn.CircularPad2d
    nn.CircularPad3d

Non-linear Activations (weighted sum, nonlinearity)
---------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ELU
    nn.Hardshrink
    nn.Hardsigmoid
    nn.Hardtanh
    nn.Hardswish
    nn.LeakyReLU
    nn.LogSigmoid
    nn.MultiheadAttention
    nn.PReLU
    nn.ReLU
    nn.ReLU6
    nn.RReLU
    nn.SELU
    nn.CELU
    nn.GELU
    nn.Sigmoid
    nn.SiLU
    nn.Mish
    nn.Softplus
    nn.Softshrink
    nn.Softsign
    nn.Tanh
    nn.Tanhshrink
    nn.Threshold
    nn.GLU

Non-linear Activations (other)
------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Softmin
    nn.Softmax
    nn.Softmax2d
    nn.LogSoftmax
    nn.AdaptiveLogSoftmaxWithLoss

Normalization Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.BatchNorm1d
    nn.BatchNorm2d
    nn.BatchNorm3d
    nn.LazyBatchNorm1d
    nn.LazyBatchNorm2d
    nn.LazyBatchNorm3d
    nn.GroupNorm
    nn.SyncBatchNorm
    nn.InstanceNorm1d
    nn.InstanceNorm2d
    nn.InstanceNorm3d
    nn.LazyInstanceNorm1d
    nn.LazyInstanceNorm2d
    nn.LazyInstanceNorm3d
    nn.LayerNorm
    nn.LocalResponseNorm
    nn.RMSNorm

Recurrent Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.RNNBase
    nn.RNN
    nn.LSTM
    nn.GRU
    nn.RNNCell
    nn.LSTMCell
    nn.GRUCell

Transformer Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Transformer
    nn.TransformerEncoder
    nn.TransformerDecoder
    nn.TransformerEncoderLayer
    nn.TransformerDecoderLayer

Linear Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Identity
    nn.Linear
    nn.Bilinear
    nn.LazyLinear

Dropout Layers
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Dropout
    nn.Dropout1d
    nn.Dropout2d
    nn.Dropout3d
    nn.AlphaDropout
    nn.FeatureAlphaDropout

Sparse Layers
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Embedding
    nn.EmbeddingBag

Distance Functions
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.CosineSimilarity
    nn.PairwiseDistance

Loss Functions
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.L1Loss
    nn.MSELoss
    nn.CrossEntropyLoss
    nn.CTCLoss
    nn.NLLLoss
    nn.PoissonNLLLoss
    nn.GaussianNLLLoss
    nn.KLDivLoss
    nn.BCELoss
    nn.BCEWithLogitsLoss
    nn.MarginRankingLoss
    nn.HingeEmbeddingLoss
    nn.MultiLabelMarginLoss
    nn.HuberLoss
    nn.SmoothL1Loss
    nn.SoftMarginLoss
    nn.MultiLabelSoftMarginLoss
    nn.CosineEmbeddingLoss
    nn.MultiMarginLoss
    nn.TripletMarginLoss
    nn.TripletMarginWithDistanceLoss

Vision Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.PixelShuffle
    nn.PixelUnshuffle
    nn.Upsample
    nn.UpsamplingNearest2d
    nn.UpsamplingBilinear2d

Shuffle Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ChannelShuffle

DataParallel Layers (multi-GPU, distributed)
--------------------------------------------
.. automodule:: smith.nn.parallel
.. currentmodule:: smith

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.DataParallel
    nn.parallel.DistributedDataParallel

Utilities
---------
.. automodule:: smith.nn.utils

From the ``smith.nn.utils`` module:

Utility functions to clip parameter gradients.

.. currentmodule:: smith.nn.utils
.. autosummary::
    :toctree: generated
    :nosignatures:

    clip_grad_norm_
    clip_grad_norm
    clip_grad_value_
    get_total_norm
    clip_grads_with_norm_

Utility functions to flatten and unflatten Module parameters to and from a single vector.

.. autosummary::
    :toctree: generated
    :nosignatures:

    parameters_to_vector
    vector_to_parameters

Utility functions to fuse Modules with BatchNorm modules.

.. autosummary::
    :toctree: generated
    :nosignatures:

    fuse_conv_bn_eval
    fuse_conv_bn_weights
    fuse_linear_bn_eval
    fuse_linear_bn_weights

Utility functions to convert Module parameter memory formats.

.. autosummary::
    :toctree: generated
    :nosignatures:

    convert_conv2d_weight_memory_format
    convert_conv3d_weight_memory_format

Utility functions to apply and remove weight normalization from Module parameters.

.. autosummary::
    :toctree: generated
    :nosignatures:

    weight_norm
    remove_weight_norm
    spectral_norm
    remove_spectral_norm

Utility functions for initializing Module parameters.

.. autosummary::
    :toctree: generated
    :nosignatures:

    skip_init

Utility classes and functions for pruning Module parameters.

.. autosummary::
    :toctree: generated
    :nosignatures:

    prune.BasePruningMethod
    prune.PruningContainer
    prune.Identity
    prune.RandomUnstructured
    prune.L1Unstructured
    prune.RandomStructured
    prune.LnStructured
    prune.CustomFromMask
    prune.identity
    prune.random_unstructured
    prune.l1_unstructured
    prune.random_structured
    prune.ln_structured
    prune.global_unstructured
    prune.custom_from_mask
    prune.remove
    prune.is_pruned

Parametrizations implemented using the new parametrization functionality
in :func:`smith.nn.utils.parameterize.register_parametrization`.

.. autosummary::
    :toctree: generated
    :nosignatures:

    parametrizations.orthogonal
    parametrizations.weight_norm
    parametrizations.spectral_norm

Utility functions to parametrize Tensors on existing Modules.
Note that these functions can be used to parametrize a given Parameter
or Buffer given a specific function that maps from an input space to the
parametrized space. They are not parameterizations that would transform
an object into a parameter. See the
`Parametrizations tutorial <https://blacksmith.org/tutorials/intermediate/parametrizations.html>`_
for more information on how to implement your own parametrizations.

.. autosummary::
    :toctree: generated
    :nosignatures:

    parametrize.register_parametrization
    parametrize.remove_parametrizations
    parametrize.cached
    parametrize.is_parametrized
    parametrize.transfer_parametrizations_and_params
    parametrize.type_before_parametrizations

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    parametrize.ParametrizationList

Utility functions to call a given Module in a stateless manner.

.. autosummary::
    :toctree: generated
    :nosignatures:

    stateless.functional_call

Utility functions in other modules

.. currentmodule:: smith
.. autosummary::
    :toctree: generated
    :nosignatures:

    nn.utils.rnn.PackedSequence
    nn.utils.rnn.pack_padded_sequence
    nn.utils.rnn.pad_packed_sequence
    nn.utils.rnn.pad_sequence
    nn.utils.rnn.pack_sequence
    nn.utils.rnn.unpack_sequence
    nn.utils.rnn.unpad_sequence
    nn.utils.rnn.invert_permutation
    nn.parameter.is_lazy
    nn.factory_kwargs

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.flatten.Flatten
    nn.modules.flatten.Unflatten

Quantized Functions
--------------------

Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than
floating point precision. Blacksmith supports both per tensor and per channel asymmetric linear quantization. To learn more how to use quantized functions in Blacksmith, please refer to the :ref:`quantization-doc` documentation.

Lazy Modules Initialization
---------------------------

.. currentmodule:: smith
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.lazy.LazyModuleMixin


.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: smith.nn.backends
.. py:module:: smith.nn.utils.stateless
.. py:module:: smith.nn.backends.thnn
.. py:module:: smith.nn.common_types
.. py:module:: smith.nn.cpp
.. py:module:: smith.nn.functional
.. py:module:: smith.nn.grad
.. py:module:: smith.nn.init
.. py:module:: smith.nn.modules.activation
.. py:module:: smith.nn.modules.adaptive
.. py:module:: smith.nn.modules.batchnorm
.. py:module:: smith.nn.modules.channelshuffle
.. py:module:: smith.nn.modules.container
.. py:module:: smith.nn.modules.conv
.. py:module:: smith.nn.modules.distance
.. py:module:: smith.nn.modules.dropout
.. py:module:: smith.nn.modules.flatten
.. py:module:: smith.nn.modules.fold
.. py:module:: smith.nn.modules.instancenorm
.. py:module:: smith.nn.modules.lazy
.. py:module:: smith.nn.modules.linear
.. py:module:: smith.nn.modules.loss
.. py:module:: smith.nn.modules.module
.. py:module:: smith.nn.modules.normalization
.. py:module:: smith.nn.modules.padding
.. py:module:: smith.nn.modules.pixelshuffle
.. py:module:: smith.nn.modules.pooling
.. py:module:: smith.nn.modules.rnn
.. py:module:: smith.nn.modules.sparse
.. py:module:: smith.nn.modules.transformer
.. py:module:: smith.nn.modules.upsampling
.. py:module:: smith.nn.modules.utils
.. py:module:: smith.nn.parallel.comm
.. py:module:: smith.nn.parallel.distributed
.. py:module:: smith.nn.parallel.parallel_apply
.. py:module:: smith.nn.parallel.replicate
.. py:module:: smith.nn.parallel.scatter_gather
.. py:module:: smith.nn.parameter
.. py:module:: smith.nn.utils.clip_grad
.. py:module:: smith.nn.utils.convert_parameters
.. py:module:: smith.nn.utils.fusion
.. py:module:: smith.nn.utils.init
.. py:module:: smith.nn.utils.memory_format
.. py:module:: smith.nn.utils.parametrizations
.. py:module:: smith.nn.utils.parametrize
.. py:module:: smith.nn.utils.prune
.. py:module:: smith.nn.utils.rnn
