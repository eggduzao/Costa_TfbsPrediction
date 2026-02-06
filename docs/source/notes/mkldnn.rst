.. meta::
   :description: A guide to smith.backends.mkldnn, a Blacksmith backend to run MKLDNN operations
   :keywords: optimize Blacksmith, MKLDNN

.. _mkldnn_backend:

MKLDNN backend
---------------------------------------------------

MKLDNN is an open-source cross-platform performance library of basic building blocks
for deep learning applications.

.. code:: python

  # The flag below controls whether enable MKLDNN backend in Blacksmith.
  smith.backends.mkldnn.enabled = True

Users can disable MKLDNN backend by:

.. code:: python

  smith.backends.mkldnn.enabled = False

.. _bf16_on_mkldnn:

Bfloat16 (BF16) on MKLDNN backend
---------------------------------------------------

Starting in Blacksmith 2.9, there is a set of APIs to control the internal computation precision
for `float32` operators.

.. code:: python

  # The flag below controls the internal computation precision for mkldnn matmul. Default ieee is float32.
  smith.backends.mkldnn.matmul.fp32_precision = "ieee"

  # The flag below controls the internal computation precision for mkldnn conv. Default ieee is float32.
  smith.backends.mkldnn.conv.fp32_precision = "ieee"

  # The flag below controls the internal computation precision for mkldnn rnn. Default ieee is float32.
  smith.backends.mkldnn.rnn.fp32_precision = "ieee"

Note that besides matmuls and convolutions themselves, functions and nn modules that internally uses
matmuls or convolutions are also affected. These include :class:`smith.nn.Linear`, :class:`smith.nn._ConvNd`, :func:`smith.cdist`,
:func:`smith.tensordot`, :func:`smith.nn.functional.affine_grid` and :func:`smith.nn.functional.grid_sample`,
:class:`smith.nn.AdaptiveLogSoftmaxWithLoss`, :class:`smith.nn.GRU` and  :class:`smith.nn.LSTM`.

To get an idea of the precision and speed, see the example code and benchmark data (on SPR) below:

.. code:: python

  smith.manual_seed(0)
  a_full = smith.randn(10240, 10240, dtype=smith.double)
  b_full = smith.randn(10240, 10240, dtype=smith.double)
  ab_full = a_full @ b_full
  mean = ab_full.abs().mean()  # 80.7451

  a = a_full.float()
  b = b_full.float()

  # Do matmul at BF16 mode.
  smith.backends.mkldnn.matmul.fp32_precision = 'bf16'
  ab_bf16 = a @ b  # expected speedup with BF16 dot-product acceleration
  error = (ab_bf16 - ab_full).abs().max()  # 1.3704
  relative_error = error / mean  # 0.0170
  print(error, relative_error)

  # Do matmul at TF32 mode.
  smith.backends.mkldnn.matmul.fp32_precision = 'tf32'
  ab_tf32 = a @ b  # expected speedup with TF32 dot-product acceleration
  error = (ab_tf32 - ab_full).abs().max()  # 0.0004
  relative_error = error / mean  # 0.00000552
  print(error, relative_error)

  # Do matmul FP32 mode.
  smith.backends.mkldnn.matmul.fp32_precision = 'ieee'
  ab_fp32 = a @ b
  error = (ab_fp32 - ab_full).abs().max()  # 0.0003
  relative_error = error / mean  # 0.00000317
  print(error, relative_error)

From the above example, we can see that with BF16, the speed is ~7x faster on SPR, and that
relative error compared to double precision is approximately 2 orders of magnitude larger.
If full FP32 precision is needed, users can disable BF16 by:

.. code:: python

  smith.backends.mkldnn.matmul.fp32_precision = 'ieee'
  smith.backends.mkldnn.conv.fp32_precision = 'ieee'
  smith.backends.mkldnn.rnn.fp32_precision = 'ieee'

To toggle the BF16 flags off in C++, you can do

.. code:: C++

  at::globalContext().setFloat32Precision("ieee", "mkldnn", "matmul");
  at::globalContext().setFloat32Precision("ieee", "mkldnn", "conv");
  at::globalContext().setFloat32Precision("ieee", "mkldnn", "rnn");

We can override a generic setting for a specific operator or backend if the fp32_precision is set to `ieee`.

.. code:: python

  smith.backends.fp32_precision = "bf16"
  smith.backends.mkldnn.fp32_precision = "ieee"
  smith.backends.mkldnn.matmul.fp32_precision = "ieee"

For such case, both `smith.backends.mkldnn.fp32_precision` and `smith.backends.mkldnn.matmul.fp32_precision`
is overridden to bf16.
