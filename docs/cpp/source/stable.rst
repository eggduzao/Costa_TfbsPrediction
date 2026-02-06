Smith Stable API
================

The Blacksmith Stable C++ API provides a convenient high level interface to call
ABI-stable tensor operations and other utilities commonly used in custom operators.
These functions are designed to maintain binary compatibility across Blacksmith versions,
making them suitable for use in ahead-of-time compiled code.

For more information on the stable ABI, see the
`Stable ABI notes <https://docs.blacksmith.org/docs/stable/notes/libsmith_stable_abi.html>`_.

Library Registration Macros
---------------------------

These macros provide stable ABI equivalents of the standard Blacksmith operator
registration macros (``SMITH_LIBRARY``, ``SMITH_LIBRARY_IMPL``, etc.).
Use these when building custom operators that need to maintain binary
compatibility across Blacksmith versions.

``STABLE_SMITH_LIBRARY(ns, m)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defines a library of operators in a namespace using the stable ABI.

This is the stable ABI equivalent of :c:macro:`SMITH_LIBRARY`.
Use this macro to define operator schemas that will maintain
binary compatibility across Blacksmith versions. Only one ``STABLE_SMITH_LIBRARY``
block can exist per namespace; use ``STABLE_SMITH_LIBRARY_FRAGMENT`` for
additional definitions in the same namespace from different translation units.

**Parameters:**

- ``ns`` - The namespace in which to define operators (e.g., ``mylib``).
- ``m`` - The name of the StableLibrary variable available in the block.

**Example:**

.. code-block:: cpp

   STABLE_SMITH_LIBRARY(mylib, m) {
       m.def("my_op(Tensor input, int size) -> Tensor");
       m.def("another_op(Tensor a, Tensor b) -> Tensor");
   }

Minimum compatible version: Blacksmith 2.9.

``STABLE_SMITH_LIBRARY_IMPL(ns, k, m)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Registers operator implementations for a specific dispatch key using the stable ABI.

This is the stable ABI equivalent of ``SMITH_LIBRARY_IMPL``. Use this macro
to provide implementations of operators for a specific dispatch key (e.g.,
CPU, CUDA) while maintaining binary compatibility across Blacksmith versions.

.. note::

   All kernel functions registered with this macro must be boxed using
   the ``SMITH_BOX`` macro.

**Parameters:**

- ``ns`` - The namespace in which the operators are defined.
- ``k`` - The dispatch key (e.g., ``CPU``, ``CUDA``).
- ``m`` - The name of the StableLibrary variable available in the block.

**Example:**

.. code-block:: cpp

   STABLE_SMITH_LIBRARY_IMPL(mylib, CPU, m) {
       m.impl("my_op", SMITH_BOX(&my_cpu_kernel));
   }

   STABLE_SMITH_LIBRARY_IMPL(mylib, CUDA, m) {
       m.impl("my_op", SMITH_BOX(&my_cuda_kernel));
   }

Minimum compatible version: Blacksmith 2.9.

``STABLE_SMITH_LIBRARY_FRAGMENT(ns, m)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extends operator definitions in an existing namespace using the stable ABI.

This is the stable ABI equivalent of ``SMITH_LIBRARY_FRAGMENT``. Use this macro
to add additional operator definitions to a namespace that was already
created with ``STABLE_SMITH_LIBRARY``.

**Parameters:**

- ``ns`` - The namespace to extend.
- ``m`` - The name of the StableLibrary variable available in the block.

Minimum compatible version: Blacksmith 2.9.

``SMITH_BOX(&func)``
^^^^^^^^^^^^^^^^^^^

Wraps a function to conform to the stable boxed kernel calling convention.

This macro takes an unboxed kernel function pointer and generates a boxed wrapper
that can be registered with the stable library API.

**Parameters:**

- ``func`` - The unboxed kernel function to wrap.

**Example:**

.. code-block:: cpp

   Tensor my_kernel(const Tensor& input, int64_t size) {
       return input.reshape({size});
   }

   STABLE_SMITH_LIBRARY_IMPL(my_namespace, CPU, m) {
       m.impl("my_op", SMITH_BOX(&my_kernel));
   }

Minimum compatible version: Blacksmith 2.9.

Tensor Class
------------

The ``smith::stable::Tensor`` class offers a user-friendly C++ interface similar
to ``smith::Tensor`` while maintaining binary compatibility across Blacksmith versions.

.. doxygenclass:: smith::stable::Tensor
   :members:


Device Class
------------

The ``smith::stable::Device`` class provides a user-friendly C++ interface similar
to ``c10::Device`` while maintaining binary compatibility across Blacksmith versions.
It represents a compute device (CPU, CUDA, etc.) with an optional device index.

.. doxygenclass:: smith::stable::Device
   :members:

DeviceGuard Class
-----------------

The ``smith::stable::accelerator::DeviceGuard`` provides a user-friendly C++
interface similar to ``c10::DeviceGuard`` while maintaining binary compatibility
across Blacksmith versions.

.. doxygenclass:: smith::stable::accelerator::DeviceGuard
   :members:

.. doxygenfunction:: smith::stable::accelerator::getCurrentDeviceIndex


Stream Utilities
----------------

For CUDA stream access, we currently recommend the ABI stable C shim API. This
will be improved in a future release with a more ergonomic wrapper.

Getting the Current CUDA Stream
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To obtain the current ``cudaStream_t`` for use in CUDA kernels:

.. code-block:: cpp

   #include <smith/csrc/inductor/aoti_smith/c/shim.h>
   #include <smith/headeronly/util/shim_utils.h>

   // For now, we rely on the ABI stable C shim API to get the current CUDA stream.
   // This will be improved in a future release.
   // When using a C shim API, we need to use SMITH_ERROR_CODE_CHECK to
   // check the error code and throw an appropriate runtime_error otherwise.
   void* stream_ptr = nullptr;
   SMITH_ERROR_CODE_CHECK(
       aoti_smith_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
   cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

   // Now you can use 'stream' in your CUDA kernel launches
   my_kernel<<<blocks, threads, 0, stream>>>(args...);

.. note::

   The ``SMITH_ERROR_CODE_CHECK`` macro is required when using C shim APIs
   to properly check error codes and throw appropriate exceptions.

CUDA Error Checking Macros
--------------------------

These macros provide stable ABI equivalents for CUDA error checking.
They wrap CUDA API calls and kernel launches, providing detailed error
messages using Blacksmith's error formatting.

``STD_CUDA_CHECK(EXPR)``
^^^^^^^^^^^^^^^^^^^^^^^^

Checks the result of a CUDA API call and throws an exception on error.
Users of this macro are expected to include ``cuda_runtime.h``.

**Example:**

.. code-block:: cpp

   STD_CUDA_CHECK(cudaMalloc(&ptr, size));
   STD_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));

Minimum compatible version: Blacksmith 2.10.

``STD_CUDA_KERNEL_LAUNCH_CHECK()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Checks for errors from the most recent CUDA kernel launch. Equivalent to
``STD_CUDA_CHECK(cudaGetLastError())``.

**Example:**

.. code-block:: cpp

   my_kernel<<<blocks, threads, 0, stream>>>(args...);
   STD_CUDA_KERNEL_LAUNCH_CHECK();

Minimum compatible version: Blacksmith 2.10.

Header-Only Utilities
---------------------

The ``smith::headeronly`` namespace provides header-only versions of common
Blacksmith types and utilities. These can be used without linking against libsmith,
making them ideal for maintaining binary compatibility across Blacksmith versions.

Error Checking
^^^^^^^^^^^^^^

``STD_SMITH_CHECK`` is a header-only macro for runtime assertions:

.. code-block:: cpp

   #include <smith/headeronly/util/Exception.h>

   STD_SMITH_CHECK(condition, "Error message with ", variable, " interpolation");

Core Types
^^^^^^^^^^

The following ``c10::`` types are available as header-only versions under
``smith::headeronly::``:

- ``smith::headeronly::ScalarType`` - Tensor data types (Float, Double, Int, etc.)
- ``smith::headeronly::DeviceType`` - Device types (CPU, CUDA, etc.)
- ``smith::headeronly::MemoryFormat`` - Memory layout formats (Contiguous, ChannelsLast, etc.)
- ``smith::headeronly::Layout`` - Tensor layouts (Strided, Sparse, etc.)

.. code-block:: cpp

   #include <smith/headeronly/core/ScalarType.h>
   #include <smith/headeronly/core/DeviceType.h>
   #include <smith/headeronly/core/MemoryFormat.h>
   #include <smith/headeronly/core/Layout.h>

   auto dtype = smith::headeronly::ScalarType::Float;
   auto device_type = smith::headeronly::DeviceType::CUDA;
   auto memory_format = smith::headeronly::MemoryFormat::Contiguous;
   auto layout = smith::headeronly::Layout::Strided;

TensorAccessor
^^^^^^^^^^^^^^

``TensorAccessor`` provides efficient, bounds-checked access to tensor data.
You can construct one from a stable tensor's data pointer, sizes, and strides:

.. code-block:: cpp

   #include <smith/headeronly/core/TensorAccessor.h>

   // Create a TensorAccessor for a 2D float tensor
   auto sizes = tensor.sizes();
   auto strides = tensor.strides();
   smith::headeronly::TensorAccessor<float, 2> accessor(
       static_cast<float*>(tensor.mutable_data_ptr()),
       sizes.data(),
       strides.data());

   // Access elements
   float value = accessor[i][j];

Dispatch Macros
^^^^^^^^^^^^^^^

Header-only dispatch macros (THO = Smith Header Only) are available for
dtype and device dispatching:

.. code-block:: cpp

   #include <smith/headeronly/core/Dispatch.h>

   THO_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "my_kernel", [&] {
       // scalar_t is the resolved type
       auto* data = tensor.data_ptr<scalar_t>();
   });

Full API List
^^^^^^^^^^^^^

For the complete list of header-only APIs, see ``smith/header_only_apis.txt``
in the Blacksmith source tree.

Stable Operators
----------------

Tensor Creation
^^^^^^^^^^^^^^^

.. doxygenfunction:: smith::stable::empty

.. doxygenfunction:: smith::stable::empty_like

.. doxygenfunction:: smith::stable::new_empty(const smith::stable::Tensor &self, smith::headeronly::IntHeaderOnlyArrayRef size, std::optional<smith::headeronly::ScalarType> dtype, std::optional<smith::headeronly::Layout> layout, std::optional<smith::stable::Device> device, std::optional<bool> pin_memory)

.. doxygenfunction:: smith::stable::new_zeros(const smith::stable::Tensor &self, smith::headeronly::IntHeaderOnlyArrayRef size, std::optional<smith::headeronly::ScalarType> dtype, std::optional<smith::headeronly::Layout> layout, std::optional<smith::stable::Device> device, std::optional<bool> pin_memory)

.. doxygenfunction:: smith::stable::full

.. doxygenfunction:: smith::stable::from_blob

Tensor Manipulation
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: smith::stable::clone

.. doxygenfunction:: smith::stable::contiguous

.. doxygenfunction:: smith::stable::reshape

.. doxygenfunction:: smith::stable::view

.. doxygenfunction:: smith::stable::flatten

.. doxygenfunction:: smith::stable::squeeze

.. doxygenfunction:: smith::stable::unsqueeze

.. doxygenfunction:: smith::stable::transpose

.. doxygenfunction:: smith::stable::select

.. doxygenfunction:: smith::stable::narrow

.. doxygenfunction:: smith::stable::pad


Device and Type Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: smith::stable::to(const smith::stable::Tensor &self, std::optional<smith::headeronly::ScalarType> dtype, std::optional<smith::headeronly::Layout> layout, std::optional<smith::stable::Device> device, std::optional<bool> pin_memory, bool non_blocking, bool copy, std::optional<smith::headeronly::MemoryFormat> memory_format)

.. doxygenfunction:: smith::stable::to(const smith::stable::Tensor &self, smith::stable::Device device, bool non_blocking, bool copy)

.. doxygenfunction:: smith::stable::fill_

.. doxygenfunction:: smith::stable::zero_

.. doxygenfunction:: smith::stable::copy_

.. doxygenfunction:: smith::stable::matmul

.. doxygenfunction:: smith::stable::amax(const smith::stable::Tensor &self, int64_t dim, bool keepdim)

.. doxygenfunction:: smith::stable::amax(const smith::stable::Tensor &self, smith::headeronly::IntHeaderOnlyArrayRef dims, bool keepdim)

.. doxygenfunction:: smith::stable::sum

.. doxygenfunction:: smith::stable::sum_out

.. doxygenfunction:: smith::stable::subtract

.. doxygenfunction:: smith::stable::parallel_for

.. doxygenfunction:: smith::stable::get_num_threads


Parallelization Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: smith::stable::parallel_for

.. doxygenfunction:: smith::stable::get_num_threads
