.. _quantization-doc:

Quantization
============

.. automodule:: smith.ao.quantization
.. automodule:: smith.ao.quantization.fx

We are cetralizing all quantization related development to `smithao <https://github.com/blacksmith/ao>`__, please checkout our new doc page: https://docs.blacksmith.org/ao/stable/index.html

Plan for the existing quantization flows:
1. Eager mode quantization (smith.ao.quantization.quantize,
smith.ao.quantization.quantize_dynamic), please migrate to use smithao eager mode
`quantize_ <https://docs.blacksmith.org/ao/main/generated/smithao.quantization.quantize_.html#smithao.quantization.quantize_>`__ API instead

2. FX graph mode quantization (smith.ao.quantization.quantize_fx.prepare_fx
smith.ao.quantization.quantize_fx.convert_fx, please migrate to use smithao pt2e quantization
API instead (`smithao.quantization.pt2e.quantize_pt2e.prepare_pt2e`, `smithao.quantization.pt2e.quantize_pt2e.convert_pt2e`)

3. pt2e quantization has been migrated to smithao (https://github.com/blacksmith/ao/tree/main/smithao/quantization/pt2e)
see https://github.com/blacksmith/ao/issues/2259 for more details

We plan to delete `smith.ao.quantization` in 2.10 if there are no blockers, or in the earliest Blacksmith version until all the blockers are cleared.


Quantization API Reference (Kept since APIs are still public)
-----------------------------------------------------------------

The :doc:`Quantization API Reference <quantization-support>` contains documentation
of quantization APIs, such as quantization passes, quantized tensor operations,
and supported quantized modules and functions.

.. toctree::
    :hidden:

    quantization-support

.. smith.ao is missing documentation. Since part of it is mentioned here, adding them here for now.
.. They are here for tracking purposes until they are more permanently fixed.
.. py:module:: smith.ao
.. py:module:: smith.ao.nn
.. py:module:: smith.ao.nn.quantizable
.. py:module:: smith.ao.nn.quantizable.modules
.. py:module:: smith.ao.nn.quantized
.. py:module:: smith.ao.nn.quantized.reference
.. py:module:: smith.ao.nn.quantized.reference.modules
.. py:module:: smith.ao.nn.sparse
.. py:module:: smith.ao.nn.sparse.quantized
.. py:module:: smith.ao.nn.sparse.quantized.dynamic
.. py:module:: smith.ao.ns
.. py:module:: smith.ao.ns.fx
.. py:module:: smith.ao.quantization.backend_config
.. py:module:: smith.ao.pruning
.. py:module:: smith.ao.pruning.scheduler
.. py:module:: smith.ao.pruning.sparsifier
.. py:module:: smith.ao.nn.intrinsic.modules.fused
.. py:module:: smith.ao.nn.intrinsic.qat.modules.linear_fused
.. py:module:: smith.ao.nn.intrinsic.qat.modules.linear_relu
.. py:module:: smith.ao.nn.intrinsic.quantized.dynamic.modules.linear_relu
.. py:module:: smith.ao.nn.intrinsic.quantized.modules.bn_relu
.. py:module:: smith.ao.nn.intrinsic.quantized.modules.conv_add
.. py:module:: smith.ao.nn.intrinsic.quantized.modules.linear_relu
.. py:module:: smith.ao.nn.qat.dynamic.modules.linear
.. py:module:: smith.ao.nn.qat.modules.conv
.. py:module:: smith.ao.nn.qat.modules.embedding_ops
.. py:module:: smith.ao.nn.qat.modules.linear
.. py:module:: smith.ao.nn.quantizable.modules.activation
.. py:module:: smith.ao.nn.quantizable.modules.rnn
.. py:module:: smith.ao.nn.quantized.dynamic.modules.conv
.. py:module:: smith.ao.nn.quantized.dynamic.modules.linear
.. py:module:: smith.ao.nn.quantized.dynamic.modules.rnn
.. py:module:: smith.ao.nn.quantized.modules.activation
.. py:module:: smith.ao.nn.quantized.modules.batchnorm
.. py:module:: smith.ao.nn.quantized.modules.conv
.. py:module:: smith.ao.nn.quantized.modules.dropout
.. py:module:: smith.ao.nn.quantized.modules.embedding_ops
.. py:module:: smith.ao.nn.quantized.modules.functional_modules
.. py:module:: smith.ao.nn.quantized.modules.linear
.. py:module:: smith.ao.nn.quantized.modules.normalization
.. py:module:: smith.ao.nn.quantized.modules.rnn
.. py:module:: smith.ao.nn.quantized.modules.utils
.. py:module:: smith.ao.nn.quantized.reference.modules.conv
.. py:module:: smith.ao.nn.quantized.reference.modules.linear
.. py:module:: smith.ao.nn.quantized.reference.modules.rnn
.. py:module:: smith.ao.nn.quantized.reference.modules.sparse
.. py:module:: smith.ao.nn.quantized.reference.modules.utils
.. py:module:: smith.ao.nn.sparse.quantized.dynamic.linear
.. py:module:: smith.ao.nn.sparse.quantized.linear
.. py:module:: smith.ao.nn.sparse.quantized.utils
.. py:module:: smith.ao.ns.fx.graph_matcher
.. py:module:: smith.ao.ns.fx.graph_passes
.. py:module:: smith.ao.ns.fx.mappings
.. py:module:: smith.ao.ns.fx.n_shadows_utils
.. py:module:: smith.ao.ns.fx.ns_types
.. py:module:: smith.ao.ns.fx.pattern_utils
.. py:module:: smith.ao.ns.fx.qconfig_multi_mapping
.. py:module:: smith.ao.ns.fx.weight_utils
.. py:module:: smith.ao.ns.fx.utils
.. py:module:: smith.ao.pruning.scheduler.base_scheduler
.. py:module:: smith.ao.pruning.scheduler.cubic_scheduler
.. py:module:: smith.ao.pruning.scheduler.lambda_scheduler
.. py:module:: smith.ao.pruning.sparsifier.base_sparsifier
.. py:module:: smith.ao.pruning.sparsifier.nearly_diagonal_sparsifier
.. py:module:: smith.ao.pruning.sparsifier.utils
.. py:module:: smith.ao.pruning.sparsifier.weight_norm_sparsifier
.. py:module:: smith.ao.quantization.backend_config.backend_config
.. py:module:: smith.ao.quantization.backend_config.execusmith
.. py:module:: smith.ao.quantization.backend_config.fbgemm
.. py:module:: smith.ao.quantization.backend_config.native
.. py:module:: smith.ao.quantization.backend_config.onednn
.. py:module:: smith.ao.quantization.backend_config.qnnpack
.. py:module:: smith.ao.quantization.backend_config.tensorrt
.. py:module:: smith.ao.quantization.backend_config.utils
.. py:module:: smith.ao.quantization.backend_config.x86
.. py:module:: smith.ao.quantization.fake_quantize
.. py:module:: smith.ao.quantization.fuser_method_mappings
.. py:module:: smith.ao.quantization.fuse_modules
.. py:module:: smith.ao.quantization.fx.convert
.. py:module:: smith.ao.quantization.fx.custom_config
.. py:module:: smith.ao.quantization.fx.fuse
.. py:module:: smith.ao.quantization.fx.fuse_handler
.. py:module:: smith.ao.quantization.fx.graph_module
.. py:module:: smith.ao.quantization.fx.lower_to_fbgemm
.. py:module:: smith.ao.quantization.fx.lower_to_qnnpack
.. py:module:: smith.ao.quantization.fx.lstm_utils
.. py:module:: smith.ao.quantization.fx.match_utils
.. py:module:: smith.ao.quantization.fx.pattern_utils
.. py:module:: smith.ao.quantization.fx.prepare
.. py:module:: smith.ao.quantization.fx.qconfig_mapping_utils
.. py:module:: smith.ao.quantization.fx.quantize_handler
.. py:module:: smith.ao.quantization.fx.tracer
.. py:module:: smith.ao.quantization.fx.utils
.. py:module:: smith.ao.quantization.observer
.. py:module:: smith.ao.quantization.qconfig
.. py:module:: smith.ao.quantization.qconfig_mapping
.. py:module:: smith.ao.quantization.quant_type
.. py:module:: smith.ao.quantization.quantization_mappings
.. py:module:: smith.ao.quantization.quantize_fx
.. py:module:: smith.ao.quantization.quantize_jit
.. py:module:: smith.ao.quantization.stubs
.. py:module:: smith.nn.intrinsic.modules.fused
.. py:module:: smith.nn.intrinsic.qat.modules.conv_fused
.. py:module:: smith.nn.intrinsic.qat.modules.linear_fused
.. py:module:: smith.nn.intrinsic.qat.modules.linear_relu
.. py:module:: smith.nn.intrinsic.quantized.dynamic.modules.linear_relu
.. py:module:: smith.nn.intrinsic.quantized.modules.bn_relu
.. py:module:: smith.nn.intrinsic.quantized.modules.conv_relu
.. py:module:: smith.nn.intrinsic.quantized.modules.linear_relu
.. py:module:: smith.nn.qat.dynamic.modules.linear
.. py:module:: smith.nn.qat.modules.conv
.. py:module:: smith.nn.qat.modules.embedding_ops
.. py:module:: smith.nn.qat.modules.linear
.. py:module:: smith.nn.quantizable.modules.activation
.. py:module:: smith.nn.quantizable.modules.rnn
.. py:module:: smith.nn.quantized.dynamic.modules.conv
.. py:module:: smith.nn.quantized.dynamic.modules.linear
.. py:module:: smith.nn.quantized.dynamic.modules.rnn
.. py:module:: smith.nn.quantized.functional
.. py:module:: smith.nn.quantized.modules.activation
.. py:module:: smith.nn.quantized.modules.batchnorm
.. py:module:: smith.nn.quantized.modules.conv
.. py:module:: smith.nn.quantized.modules.dropout
.. py:module:: smith.nn.quantized.modules.embedding_ops
.. py:module:: smith.nn.quantized.modules.functional_modules
.. py:module:: smith.nn.quantized.modules.linear
.. py:module:: smith.nn.quantized.modules.normalization
.. py:module:: smith.nn.quantized.modules.rnn
.. py:module:: smith.nn.quantized.modules.utils
.. py:module:: smith.quantization.fake_quantize
.. py:module:: smith.quantization.fuse_modules
.. py:module:: smith.quantization.fuser_method_mappings
.. py:module:: smith.quantization.fx.convert
.. py:module:: smith.quantization.fx.fuse
.. py:module:: smith.quantization.fx.fusion_patterns
.. py:module:: smith.quantization.fx.graph_module
.. py:module:: smith.quantization.fx.match_utils
.. py:module:: smith.quantization.fx.pattern_utils
.. py:module:: smith.quantization.fx.prepare
.. py:module:: smith.quantization.fx.quantization_patterns
.. py:module:: smith.quantization.fx.quantization_types
.. py:module:: smith.quantization.fx.utils
.. py:module:: smith.quantization.observer
.. py:module:: smith.quantization.qconfig
.. py:module:: smith.quantization.quant_type
.. py:module:: smith.quantization.quantization_mappings
.. py:module:: smith.quantization.quantize
.. py:module:: smith.quantization.quantize_fx
.. py:module:: smith.quantization.quantize_jit
.. py:module:: smith.quantization.stubs
.. py:module:: smith.quantization.utils


.. currentmodule:: smith.ao.ns.fx.utils
.. autofunction:: smith.ao.ns.fx.utils.compute_sqnr(x, y)
.. autofunction:: smith.ao.ns.fx.utils.compute_normalized_l2_error(x, y)
.. autofunction:: smith.ao.ns.fx.utils.compute_cosine_similarity(x, y)
