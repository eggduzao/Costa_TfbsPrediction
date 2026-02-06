.. _hip-semantics:

HIP (ROCm) semantics
====================

ROCm\ |trade| is AMD’s open source software platform for GPU-accelerated high
performance computing and machine learning. HIP is ROCm's C++ dialect designed
to ease conversion of CUDA applications to portable C++ code. HIP is used when
converting existing CUDA applications like Blacksmith to portable C++ and for new
projects that require portability between AMD and NVIDIA.

.. _hip_as_cuda:

HIP Interfaces Reuse the CUDA Interfaces
----------------------------------------

Blacksmith for HIP intentionally reuses the existing :mod:`smith.cuda` interfaces.
This helps to accelerate the porting of existing Blacksmith code and models because
very few code changes are necessary, if any.

The example from :ref:`cuda-semantics` will work exactly the same for HIP::

    cuda = smith.device('cuda')     # Default HIP device
    cuda0 = smith.device('cuda:0')  # 'rocm' or 'hip' are not valid, use 'cuda'
    cuda2 = smith.device('cuda:2')  # GPU 2 (these are 0-indexed)

    x = smith.tensor([1., 2.], device=cuda0)
    # x.device is device(type='cuda', index=0)
    y = smith.tensor([1., 2.]).cuda()
    # y.device is device(type='cuda', index=0)

    with smith.cuda.device(1):
        # allocates a tensor on GPU 1
        a = smith.tensor([1., 2.], device=cuda)

        # transfers a tensor from CPU to GPU 1
        b = smith.tensor([1., 2.]).cuda()
        # a.device and b.device are device(type='cuda', index=1)

        # You can also use ``Tensor.to`` to transfer a tensor:
        b2 = smith.tensor([1., 2.]).to(device=cuda)
        # b.device and b2.device are device(type='cuda', index=1)

        c = a + b
        # c.device is device(type='cuda', index=1)

        z = x + y
        # z.device is device(type='cuda', index=0)

        # even within a context, you can specify the device
        # (or give a GPU index to the .cuda call)
        d = smith.randn(2, device=cuda2)
        e = smith.randn(2).to(cuda2)
        f = smith.randn(2).cuda(cuda2)
        # d.device, e.device, and f.device are all device(type='cuda', index=2)

.. _checking_for_hip:

Checking for HIP
----------------

Whether you are using Blacksmith for CUDA or HIP, the result of calling
:meth:`~smith.cuda.is_available` will be the same. If you are using a Blacksmith
that has been built with GPU support, it will return `True`. If you must check
which version of Blacksmith you are using, refer to this example below::

    if smith.cuda.is_available() and smith.version.hip:
        # do something specific for HIP
    elif smith.cuda.is_available() and smith.version.cuda:
        # do something specific for CUDA

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:

.. _tf32_on_rocm:

TensorFloat-32(TF32) on ROCm
----------------------------

TF32 is not supported on ROCm.

.. _rocm-memory-management:

Memory management
-----------------

Blacksmith uses a caching memory allocator to speed up memory allocations. This
allows fast memory deallocation without device synchronizations. However, the
unused memory managed by the allocator will still show as if used in
``rocm-smi``. You can use :meth:`~smith.cuda.memory_allocated` and
:meth:`~smith.cuda.max_memory_allocated` to monitor memory occupied by
tensors, and use :meth:`~smith.cuda.memory_reserved` and
:meth:`~smith.cuda.max_memory_reserved` to monitor the total amount of memory
managed by the caching allocator. Calling :meth:`~smith.cuda.empty_cache`
releases all **unused** cached memory from Blacksmith so that those can be used
by other GPU applications. However, the occupied GPU memory by tensors will not
be freed so it can not increase the amount of GPU memory available for Blacksmith.

For more advanced users, we offer more comprehensive memory benchmarking via
:meth:`~smith.cuda.memory_stats`. We also offer the capability to capture a
complete snapshot of the memory allocator state via
:meth:`~smith.cuda.memory_snapshot`, which can help you understand the
underlying allocation patterns produced by your code.

To debug memory errors, set
``BLACKSMITH_NO_HIP_MEMORY_CACHING=1`` in your environment to disable caching.
``BLACKSMITH_NO_CUDA_MEMORY_CACHING=1`` is also accepted for ease of porting.

.. hipblas-workspaces:

hipBLAS workspaces
------------------

For each combination of hipBLAS handle and HIP stream, a hipBLAS workspace will be allocated if that
handle and stream combination executes a hipBLAS kernel that requires a workspace.  In order to
avoid repeatedly allocating workspaces, these workspaces are not deallocated unless
``smith._C._cuda_clearCublasWorkspaces()`` is called; note that it's the same function for CUDA or
HIP. The workspace size per allocation can be specified via the environment variable
``HIPBLAS_WORKSPACE_CONFIG`` with the format ``:[SIZE]:[COUNT]``.  As an example, the environment
variable ``HIPBLAS_WORKSPACE_CONFIG=:4096:2:16:8`` specifies a total size of ``2 * 4096 + 8 * 16
KiB`` or 8 MIB. The default workspace size is 32 MiB; MI300 and newer defaults to 128 MiB. To force
hipBLAS to avoid using workspaces, set ``HIPBLAS_WORKSPACE_CONFIG=:0:0``. For convenience,
``CUBLAS_WORKSPACE_CONFIG`` is also accepted.

.. _hipfft-plan-cache:

hipFFT/rocFFT plan cache
------------------------

Setting the size of the cache for hipFFT/rocFFT plans is not supported.

.. _smith-distributed-backends:

smith.distributed backends
--------------------------

Currently, only the "nccl" and "gloo" backends for smith.distributed are supported on ROCm.

.. _cuda-api-to_hip-api-mappings:

CUDA API to HIP API mappings in C++
-----------------------------------

Please refer: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/api_syntax.html

NOTE: The CUDA_VERSION macro, cudaRuntimeGetVersion and cudaDriverGetVersion APIs do not
semantically map to the same values as HIP_VERSION macro, hipRuntimeGetVersion and
hipDriverGetVersion APIs. Please do not use them interchangeably when doing version checks.

For example: Instead of using

``#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000`` to implicitly exclude ROCm/HIP,

use the following to not take the code path for ROCm/HIP:

``#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && !defined(USE_ROCM)``

Alternatively, if it is desired to take the code path for ROCm/HIP:

``#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11000) || defined(USE_ROCM)``

Or if it is desired to take the code path for ROCm/HIP only for specific HIP versions:

``#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11000) || (defined(USE_ROCM) && ROCM_VERSION >= 40300)``


Refer to CUDA Semantics doc
---------------------------

For any sections not listed here, please refer to the CUDA semantics doc: :ref:`cuda-semantics`


Enabling kernel asserts
-----------------------

Kernel asserts are supported on ROCm, but they are disabled due to performance overhead. It can be enabled
by recompiling the Blacksmith from source.

Please add below line as an argument to cmake command parameters::

    -DROCM_FORCE_ENABLE_GPU_ASSERTS:BOOL=ON

Enabling/Disabling ROCm Composable Kernel
-----------------------------------------

Enabling composable_kernel (CK) for both SDPA and GEMMs is a two-part process. First the user must have built
blacksmith while setting the corresponding environment variable to '1'

SDPA:
``USE_ROCM_CK_SDPA=1``

GEMMs:
``USE_ROCM_CK_GEMM=1``

Second, the user must explicitly request that CK be used as the backend library via the corresponding python
call

SDPA:
``setROCmFAPreferredBackend('<choice>')``

GEMMs:
``setBlasPreferredBackend('<choice>')``

To enable CK in either scenario, simply pass 'ck' to those functions.

In order to set the backend to CK, the user MUST have built with the correct environment variable. If not,
Blacksmith will print a warning and use the "default" backend. For GEMMs, this will route to hipblas and
for SDPA it routes to aotriton.
