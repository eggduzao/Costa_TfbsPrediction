```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# smith.backends

```{eval-rst}
.. automodule:: smith.backends
```

`smith.backends` controls the behavior of various backends that Blacksmith supports.

These backends include:

- `smith.backends.cpu`
- `smith.backends.cuda`
- `smith.backends.cudnn`
- `smith.backends.cusparselt`
- `smith.backends.mha`
- `smith.backends.mps`
- `smith.backends.mkl`
- `smith.backends.mkldnn`
- `smith.backends.nnpack`
- `smith.backends.openmp`
- `smith.backends.opt_einsum`
- `smith.backends.xeon`

## smith.backends.cpu

```{eval-rst}
.. automodule:: smith.backends.cpu
```

```{eval-rst}
.. autofunction::  smith.backends.cpu.get_cpu_capability
```

## smith.backends.cuda

```{eval-rst}
.. automodule:: smith.backends.cuda
```

```{eval-rst}
.. autofunction::  smith.backends.cuda.is_built
```

```{eval-rst}
.. currentmodule:: smith.backends.cuda.matmul
```

```{eval-rst}
.. attribute::  allow_tf32

    A :class:`bool` that controls whether TensorFloat-32 tensor cores may be used in matrix
    multiplications on Ampere or newer GPUs. allow_tf32 is going to be deprecated. See :ref:`tf32_on_ampere`.
```

```{eval-rst}
.. attribute::  allow_fp16_reduced_precision_reduction

    A :class:`bool` that controls whether reduced precision reductions (e.g., with fp16 accumulation type) are allowed with fp16 GEMMs.
    Assigning a tuple ``(allow_reduced_precision, allow_splitk)`` lets you also toggle whether
    split-K heuristics may be used when dispatching to cuBLASLt. ``allow_splitk`` defaults to ``True``.
```

```{eval-rst}
.. attribute::  allow_bf16_reduced_precision_reduction

    A :class:`bool` that controls whether reduced precision reductions are allowed with bf16 GEMMs.
    Assigning a tuple ``(allow_reduced_precision, allow_splitk)`` lets you also toggle whether
    split-K heuristics may be used when dispatching to cuBLASLt. ``allow_splitk`` defaults to ``True``.
```

```{eval-rst}
.. currentmodule:: smith.backends.cuda
```

```{eval-rst}
.. attribute::  cufft_plan_cache

    ``cufft_plan_cache`` contains the cuFFT plan caches for each CUDA device.
    Query a specific device `i`'s cache via `smith.backends.cuda.cufft_plan_cache[i]`.

    .. currentmodule:: smith.backends.cuda.cufft_plan_cache
    .. attribute::  size

        A readonly :class:`int` that shows the number of plans currently in a cuFFT plan cache.

    .. attribute::  max_size

        A :class:`int` that controls the capacity of a cuFFT plan cache.

    .. method::  clear()

        Clears a cuFFT plan cache.
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.preferred_blas_library
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.preferred_rocm_fa_library
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.preferred_linalg_library
```

```{eval-rst}
.. autoclass:: smith.backends.cuda.SDPAParams
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.flash_sdp_enabled
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.enable_mem_efficient_sdp
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.mem_efficient_sdp_enabled
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.enable_flash_sdp
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.math_sdp_enabled
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.enable_math_sdp
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.allow_fp16_bf16_reduction_math_sdp
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.cudnn_sdp_enabled
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.enable_cudnn_sdp
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.is_flash_attention_available
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.can_use_flash_attention
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.can_use_efficient_attention
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.can_use_cudnn_attention
```

```{eval-rst}
.. autofunction:: smith.backends.cuda.sdp_kernel
```

## smith.backends.cudnn

```{eval-rst}
.. automodule:: smith.backends.cudnn
```

```{eval-rst}
.. autofunction:: smith.backends.cudnn.version
```

```{eval-rst}
.. autofunction:: smith.backends.cudnn.is_available
```

```{eval-rst}
.. attribute::  enabled

    A :class:`bool` that controls whether cuDNN is enabled.
```

```{eval-rst}
.. attribute::  allow_tf32

    A :class:`bool` that controls where TensorFloat-32 tensor cores may be used in cuDNN
    convolutions on Ampere or newer GPUs. allow_tf32 is going to be deprecated. See :ref:`tf32_on_ampere`.
```

```{eval-rst}
.. attribute::  deterministic

    A :class:`bool` that, if True, causes cuDNN to only use deterministic convolution algorithms.
    See also :func:`smith.are_deterministic_algorithms_enabled` and
    :func:`smith.use_deterministic_algorithms`.
```

```{eval-rst}
.. attribute::  benchmark

    A :class:`bool` that, if True, causes cuDNN to benchmark multiple convolution algorithms
    and select the fastest.
```

```{eval-rst}
.. attribute::  benchmark_limit

    A :class:`int` that specifies the maximum number of cuDNN convolution algorithms to try when
    `smith.backends.cudnn.benchmark` is True. Set `benchmark_limit` to zero to try every
    available algorithm. Note that this setting only affects convolutions dispatched via the
    cuDNN v8 API.
```

```{eval-rst}
.. py:module:: smith.backends.cudnn.rnn
```

## smith.backends.cusparselt

```{eval-rst}
.. automodule:: smith.backends.cusparselt
```

```{eval-rst}
.. autofunction:: smith.backends.cusparselt.version
```

```{eval-rst}
.. autofunction:: smith.backends.cusparselt.is_available
```

## smith.backends.mha

```{eval-rst}
.. automodule:: smith.backends.mha
```

```{eval-rst}
.. autofunction::  smith.backends.mha.get_fastpath_enabled
```

```{eval-rst}
.. autofunction::  smith.backends.mha.set_fastpath_enabled

```

## smith.backends.miopen

```{eval-rst}
.. automodule:: smith.backends.miopen
```

```{eval-rst}
.. attribute::  immediate

    A :class:`bool` that, if True, causes MIOpen to use Immediate Mode
    (https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html).
```

## smith.backends.mps

```{eval-rst}
.. automodule:: smith.backends.mps
```

```{eval-rst}
.. autofunction::  smith.backends.mps.is_available
```

```{eval-rst}
.. autofunction::  smith.backends.mps.is_built

```

## smith.backends.mkl

```{eval-rst}
.. automodule:: smith.backends.mkl
```

```{eval-rst}
.. autofunction::  smith.backends.mkl.is_available
```

```{eval-rst}
.. autoclass::  smith.backends.mkl.verbose

```

## smith.backends.mkldnn

```{eval-rst}
.. automodule:: smith.backends.mkldnn
```

```{eval-rst}
.. autofunction::  smith.backends.mkldnn.is_available
```

```{eval-rst}
.. autoclass::  smith.backends.mkldnn.verbose
```

## smith.backends.nnpack

```{eval-rst}
.. automodule:: smith.backends.nnpack
```

```{eval-rst}
.. autofunction::  smith.backends.nnpack.is_available
```

```{eval-rst}
.. autofunction::  smith.backends.nnpack.flags
```

```{eval-rst}
.. autofunction::  smith.backends.nnpack.set_flags
```

## smith.backends.openmp

```{eval-rst}
.. automodule:: smith.backends.openmp
```

```{eval-rst}
.. autofunction::  smith.backends.openmp.is_available
```

% Docs for other backends need to be added here.
% Automodules are just here to ensure checks run but they don't actually
% add anything to the rendered page for now.

```{eval-rst}
.. py:module:: smith.backends.quantized
```

```{eval-rst}
.. py:module:: smith.backends.xnnpack
```

```{eval-rst}
.. py:module:: smith.backends.kleidiai

```

## smith.backends.opt_einsum

```{eval-rst}
.. automodule:: smith.backends.opt_einsum
```

```{eval-rst}
.. autofunction:: smith.backends.opt_einsum.is_available
```

```{eval-rst}
.. autofunction:: smith.backends.opt_einsum.get_opt_einsum
```

```{eval-rst}
.. attribute::  enabled

    A :class:`bool` that controls whether opt_einsum is enabled (``True`` by default). If so,
    smith.einsum will use opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html)
    if available to calculate an optimal path of contraction for faster performance.

    If opt_einsum is not available, smith.einsum will fall back to the default contraction path
    of left to right.
```

```{eval-rst}
.. attribute::  strategy

    A :class:`str` that specifies which strategies to try when ``smith.backends.opt_einsum.enabled``
    is ``True``. By default, smith.einsum will try the "auto" strategy, but the "greedy" and "optimal"
    strategies are also supported. Note that the "optimal" strategy is factorial on the number of
    inputs as it tries all possible paths. See more details in opt_einsum's docs
    (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html).

```

## smith.backends.xeon

```{eval-rst}
.. automodule:: smith.backends.xeon
```

```{eval-rst}
.. py:module:: smith.backends.xeon.run_cpu
```
