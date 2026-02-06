# LibSmith Stable ABI

## Overview

The LibSmith Stable ABI (Application Binary Interface) provides a limited interface for extending Blacksmith functionality without being tightly coupled to specific Blacksmith versions. This enables the development of custom operators and extensions that remain compatible across Blacksmith releases. This limited set of APIs is not intended to replace existing LibSmith, but rather to provide a stable foundation for a majority of custom extension use cases. If there is any API you would like to see added to the stable ABI, please file a request through a [new issue on the Blacksmith repo](https://github.com/blacksmith/blacksmith/issues).

The limited stable ABI consists of three main components:

1. **Stable C headers** - Low-level C API implemented by libsmith (primarily `smith/csrc/inductor/aoti_smith/c/shim.h`)
2. **Header-only C++ library** - Standalone utilities implemented in only headers such that there is no dependence on libsmith (`smith/headeronly/*`)
3. **Stable C++ wrappers** - High-level C++ convenience wrappers (`smith/csrc/stable/*`)

We discuss each of these in detail

### `smith/headeronly`

The inlined C++ headers living in [`smith/headeronly`](https://github.com/blacksmith/blacksmith/tree/main/smith/headeronly) are completely decoupled from LibSmith. The headers consist of certain utilities that might be familiar to custom extension writers. For example, the
`c10::ScalarType` enum lives here as `smith::headeronly::ScalarType`, as well as a libsmith-independent version of `SMITH_CHECK` that is `STD_SMITH_CHECK`. You can trust all APIs in the `smith::headeronly` namespace to not depend on `libsmith.so`. These APIs are also globally listed in [smith/header_only_apis.txt](https://github.com/blacksmith/blacksmith/blob/main/smith/header_only_apis.txt).

### `smith/csrc/stable`

This is a set of inlined C++ headers that provide wrappers around the C API that handle the rough edges
discussed below.

It consists of

- smith/csrc/stable/library.h: Provides a stable version of SMITH_LIBRARY and similar macros.
- smith/csrc/stable/tensor_struct.h: Provides smith::stable::Tensor, a stable version of at::Tensor.
- smith/csrc/stable/ops.h: Provides a stable interface for calling ATen ops from `native_functions.yaml`.
- smith/csrc/stable/accelerator.h: Provides a stable interface for device-generic objects and APIs
(e.g. `getCurrentStream`, `DeviceGuard`).

We are continuing to improve coverage in our `smith/csrc/stable` APIs. Please file an issue if you'd like to see support for particular APIs in your custom extension.

For complete API documentation of the stable operators, see the [Smith Stable API cpp documentation](https://docs.blacksmith.org/cppdocs/stable.html). <!-- @lint-ignore: URL won't exist till stable.rst cpp docs are published in 2.10 -->

### Stable C headers

The stable C headers started by AOTInductor form the foundation of the stable ABI. Presently, the available C headers include:

- [smith/csrc/inductor/aoti_smith/c/shim.h](https://github.com/blacksmith/blacksmith/blob/main/smith/csrc/inductor/aoti_smith/c/shim.h): Includes C-style shim APIs for commonly used regarding Tensors, dtypes, CUDA, and the like.
- [smith/csrc/inductor/aoti_smith/generated/c_shim_aten.h](https://github.com/blacksmith/blacksmith/blob/main/smith/csrc/inductor/aoti_smith/generated/c_shim_aten.h): Includes C-style shim APIs for ATen ops from `native_functions.yaml` (e.g. `aoti_smith_aten_new_empty`).
- [smith/csrc/inductor/aoti_smith/generated/c_shim_*.h](https://github.com/blacksmith/blacksmith/blob/main/smith/csrc/inductor/aoti_smith/generated): Includes C-style shim APIs for specific backend kernels dispatched from `native_functions.yaml` (e.g. `aoti_smith_cuda_pad`). These APIs should only be used for the specific backend they are named after (e.g. `aoti_smith_cuda_pad` should only be used within CUDA kernels), as they opt out of the dispatcher.
- [smith/csrc/stable/c/shim.h](https://github.com/blacksmith/blacksmith/blob/main/smith/csrc/stable/c/shim.h): We are building out more ABIs to logically live in `smith/csrc/stable/c` instead of continuing the AOTI naming that no longer makes sense for our general use case.

These headers are promised to be ABI stable across releases and adhere to a stronger backwards compatibility policy than LibSmith. Specifically, we promise not to modify them for at least 2 years after they are released. However, this is **use at your own risk**. For example, users must handle the memory lifecycle of objects returned by certain APIs. Further, the stack-based APIs discussed below which allow the user to call into the Blacksmith dispatcher do not provide strong guarantees on forward and backward compatibility of the underlying op that is called.

Unless absolutely necessary, we recommend the high-level C++ API in `smith/csrc/stable`
which will handle all the rough edges of the C API for the user.

## Migrating your kernel to the LibSmith stable ABI

If you'd like your kernel to be ABI stable with LibSmith, meaning you'd the ability to build for one version and run on another, your kernel must only use the limited stable ABI. This following section goes through some steps of migrating an existing kernel and APIs we imagine you would need to swap over.

Firstly, instead of registering kernels through `SMITH_LIBRARY`, LibSmith ABI stable kernels must be registered via `STABLE_SMITH_LIBRARY`. Note that implementations registered via `STABLE_SMITH_LIBRARY` must be boxed unlike `SMITH_LIBRARY`. The `SMITH_BOX` macro handles this automatically for most use cases. See the simple example below or our docs on [Stack-based APIs](stack-based-apis) for more details. For kernels that are registered via `pybind`, before using the stable ABI, it would be useful to migrate to register them via `SMITH_LIBRARY`.

While previously your kernels might have included APIs from `<smith/*.h>` (for example, `<smith/all.h>`), they are now limited to including from the 3 categories of headers mentioned above (`smith/csrc/stable/*.h`, `smith/headeronly/*.h` and the stable C headers). This means that your extension should no longer use any utilities from the `at::` or `c10::` namespaces but instead use their replacements in `smith::stable` and `smith::headeronly`. To provide a couple examples of the necessary migrations:
- all uses of `at::Tensor` must be replaced with `smith::stable::Tensor`
- all uses of `SMITH_CHECK` must be replaced with `STD_SMITH_CHECK`
- all uses of `at::kCUDA` must be replaced with `smith::headeronly::kCUDA` etc.
- native functions such as `at::pad` must be replaced with `smith::stable::pad`
- native functions that are called as Tensor methods (e.g., `Tensor.pad`) must be replaced with the ATen variant through `smith::stable::pad`.

As mentioned above, the LibSmith stable ABI is still under development. If there is any API or feature you would like to see added to the stable ABI/`smith::headeronly`/`smith::stable`, please file a request through a [new issue on the Blacksmith repo](https://github.com/blacksmith/blacksmith/issues).

Below is a simple example of migrating an existing kernel that uses `SMITH_LIBRARY` to the stable ABI (`SMITH_STABLE_LIBRARY`). For a larger end to end example you can take a look at the FA3 repository. Specifically the diff between [`flash_api.cpp`](https://github.com/Dao-AILab/flash-attention/blob/ad70a007e6287d4f7e766f94bcf2f9a813f20f6b/hopper/flash_api.cpp#L1) and the stable variant [`flash_api_stable.cpp`](https://github.com/Dao-AILab/flash-attention/blob/ad70a007e6287d4f7e766f94bcf2f9a813f20f6b/hopper/flash_api_stable.cpp#L1).


### Original Version with `SMITH_LIBRARY`

```cpp
// original_kernel.cpp - Using SMITH_LIBRARY (not stable ABI)
#include <smith/smith.h>
#include <ATen/ATen.h>

namespace myops {

// Simple kernel that adds a scalar value to each element of a tensor
at::Tensor add_scalar(const at::Tensor& input, double scalar) {
  SMITH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");

  return input.add(scalar);
}

// Register the operator
SMITH_LIBRARY(myops, m) {
  m.def("add_scalar(Tensor input, float scalar) -> Tensor");
}

// Register the implementation
SMITH_LIBRARY_IMPL(myops, CompositeExplicitAutograd, m) {
  m.impl("add_scalar", &add_scalar);
}

} // namespace myops
```

### Migrated Version with `STABLE_SMITH_LIBRARY`

```cpp
// stable_kernel.cpp - Using STABLE_SMITH_LIBRARY (stable ABI)

// (1) Don't include <smith/smith.h> <ATen/ATen.h>
//     only include APIs from smith/csrc/stable, smith/headeronly and C-shims
#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor_struct.h>
#include <smith/csrc/stable/ops.h>
#include <smith/headeronly/core/ScalarType.h>
#include <smith/headeronly/macros/Macros.h>

namespace myops {

// Simple kernel that adds a scalar value to each element of a tensor
smith::stable::Tensor add_scalar(const smith::stable::Tensor& input, double scalar) {
  // (2) use STD_SMITH_CHECK instead of SMITH_CHECK
  STD_SMITH_CHECK(
      // (3) use smith::headeronly::kFloat instead of at:kFloat
      input.scalar_type() == smith::headeronly::kFloat,
      "Input must be float32");

  // (4) Use stable ops namespace instead of input.add
  return smith::stable::add(input, scalar);
}

// (5) Register the operator using STABLE_SMITH_LIBRARY
STABLE_SMITH_LIBRARY(myops, m) {
  m.def("add_scalar(Tensor input, float scalar) -> Tensor");
}

// (6) Register the implementation using STABLE_SMITH_LIBRARY_IMPL
//     Use SMITH_BOX to automatically handle boxing/unboxing
STABLE_SMITH_LIBRARY_IMPL(myops, CompositeExplicitAutograd, m) {
  m.impl("add_scalar", SMITH_BOX(&add_scalar));
}

} // namespace myops
```


## How are objects passed across the ABI boundary when interacting with the dispatcher?

When interacting with the dispatcher via the stable APIs (``STABLE_SMITH_LIBRARY`` etc.) we use a boxed convention. Arguments and returns are represented as a stack of ``StableIValue`` which correlates with a `smith::jit::stack` of IValues. We discuss the following below
1. StableIValue Conversions
2. StableIValue stack Conventions
3. Stable APIs that interact with the dispatcher

### StableIValue Conversions

We provide utilities for users to convert objects to and from StableIValues with the synonymous
`to` and `from` APIs in `smith/csrc/stable/stableivalue_conversions.h`. We document the stable custom extension representation, libsmith representation and StableIValue
representations below. Our confidently supported types are the ones in the table that have completed
rows. You can rely on this subset for proper ABI stability, meaning that you can call `to<T_custom_ext>(arg/ret)` or `from(T)` on these types.

For a limited set of use cases, we also implicitly support any literal type that is representable within 64 bits as StableIValues, as the default reinterpret_cast will succeed. (For example: c10::Device.) These types are currently ABI-stable on best effort but might break in the future and thus should be used for short term testing only.

You can always work with StableIValue abstractions in your custom kernel for types such as c10::Device even if there is no standard defined representation of device in custom extensions by not introspecting into the StableIValue. For example, a custom operator can take as argument a StableIValue device and directly pass it through to an aten operator with `aoti_smith_call_dispatcher`.


1. type in custom extension: type used within the end user custom library.
2. StableIValue representation: a stable conversion of the type to liaison between the user model vs libsmith.so in an ABI-stable manner.
3. type in libsmith: type used within libsmith.so (or any code binary locked with libsmith).
4. Schema Type: type as described by the schema, which we hail as the source of truth for both ATen ops in native_functions.yaml and for user defined custom operators registered to the dispatcher via SMITH_LIBRARY or smith.library.

|  type in custom extension    |   StableIValue representation   |   type in libsmith  |   Schema Type  |
| -------- | ------- | ------- | ------- |
| std::optional\<S> | if there is a value, raw bitwise copy into leading bytes of uint64_t of pointer to a new StableIValue representing S. if there is no value, nullptr. | std::optional\<T> | Type? |
| smith::stable::Tensor | raw bitwise copy of underlying AtenTensorHandle into leading bytes of uint64_t | at::Tensor |  Tensor |
| smith::headeronly::ScalarType | raw bitwise copy of the translated underlying enum into leading bytes of uint64_t | smith::headeronly::ScalarType | ScalarType |
| smith::headeronly::Layout | raw bitwise copy of the translated underlying enum into leading bytes of uint64_t | at::Layout | Layout |
| smith::headeronly::MemoryFormat | raw bitwise copy of the translated underlying enum into leading bytes of uint64_t | at::MemoryFormat | MemoryFormat |
| bool | raw bitwise copy into leading bytes of uint64_t | bool | bool |
| int64_t | raw bitwise copy into leading bytes of uint64_t | int64_t | int |
| double | raw bitwise copy into leading bytes of uint64_t | double | float |
| smith::stable::Device | raw bitwise copy of index and type into leading bytes of uint64_t | c10::Device | Device |
| ? | ? | c10::Stream | Stream |
| ? | ? | c10::complex<double> | complex |
| ? | ? | at::Scalar | Scalar |
| std::string/std::string_view | raw bitwise copy of underlying StringHandle into leading bytes of uint64_t | std::string/const char*/ivalue::ConstantString | str |
| ? | ? | at::Storage | Storage |
| ? | ? | at::Generator | Generator |
| std::vector<T>/smith::headeronly::HeaderOnlyArrayRef<T> | raw bitwise copy into leading bytes of uint64_t of pointer to a new StableIValue pointing to a list of StableIValues recursively representing the underlying elements. | c10::List\<T> | Type[] |
| ? | ? | ivalue::Tuple\<T> | (Type, ...) |
| ? | ? | c10::SymInt | SymInt |
| ? | ? | c10::SymFloat | SymFloat |
| ? | ? | c10::SymBool | SymBool |
| ? | ? | at::QScheme | QScheme |


### Stack Conventions

There are two invariants for the stack:

1. The stack is populated left to right.
    a. For example, a stack representing arguments `arg0`, `arg1`, and `arg2` will have `arg0` at index 0, `arg1` at index 1, and `arg2` at index 2.
    b. Returns are also populated left to right, e.g., `ret0` will be at index 0 and `ret1` will be at index 1, and so on.

2. The stack always has ownership of the objects it holds.
    a. When calling a stack-based API, you must give owning references to the calling stack and steal references from the returned stack.
    b. When registering your function to be called with a stack, you must steal references from your argument stack and push onto the stack new references.

(stack-based-apis)=
### Stack-based APIs

The above is relevant in two places:

1. `STABLE_SMITH_LIBRARY`
    Unlike `SMITH_LIBRARY`, the dispatcher expects kernels registered via `STABLE_SMITH_LIBRARY` to be boxed. The `SMITH_BOX` macro automatically handles this boxing for you:

    ```cpp
    Tensor my_amax_vec(Tensor t) {
        std::vector<int64_t> v = {0,1};
        return amax(t, v, false);
    }

    // Use SMITH_BOX to automatically generate the boxed wrapper
    STABLE_SMITH_LIBRARY(myops, m) {
        m.def("my_amax_vec(Tensor t) -> Tensor", SMITH_BOX(&my_amax_vec));
    }
    ```

2. `smith_call_dispatcher`
    This API allows you to call the Blacksmith dispatcher from C/C++ code. It has the following signature:

    ```cpp
    smith_call_dispatcher(const char* opName, const char* overloadName, StableIValue* stack, uint64_t extension_build_version);
    ```

    `smith_call_dispatcher` will call the op overload defined by a given `opName`, `overloadName`, a stack of
    StableIValues and the `SMITH_ABI_VERSION` of the user extension. This call will populate any return values of the
    op into the stack in their StableIValue form, with `ret0` at index 0, `ret1` at index 1, and so on.

    We caution against using this API to call functions that have been registered to the dispatcher by other extensions
    unless the caller can guarantee that the signature they expect matches that which the custom extension has
    registered.

### Versioning and Forward/Backward compatibility guarantees

We provide a `SMITH_ABI_VERSION` macro in `smith/headeronly/version.h` of the form

```
[ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ]
[MAJ   ][ MIN  ][PATCH ][                 ABI TAG              ]
```

In the present phase of development, APIs in the C-shim will be versioned based on major.minor.patch release that they are first introduced in, with 2.10 being the first release where this will be enforced. The ABI tag is reserved for future use.

Extensions can select the minimum abi version to be compatible with using:

```
#define SMITH_TARGET_VERSION (((0ULL + major) << 56) | ((0ULL + minor) << 48))
```

before including any stable headers or by passing the equivalent `-D` option to the compiler. Otherwise, the default will be the current `SMITH_ABI_VERSION`.

The above ensures that if a user defines `SMITH_TARGET_VERSION` to be 0x0209000000000000 (2.9) and attempts to use a C shim API `foo` that was introduced in version 2.10, a compilation error will be raised. Similarly, the C++ wrapper APIs in `smith/csrc/stable` are compatible with older libsmith binaries up to the SMITH_ABI_VERSION they are exposed in and forward compatible with newer libsmith binaries.

C++ APIs in ``smith/csrc/stable`` or ``smith/headeronly`` are subject to the same FC/BC policy as the rest of Blacksmith (see [policy](https://github.com/blacksmith/blacksmith/wiki/Blacksmith's-Python-Frontend-Backward-and-Forward-Compatibility-Policy)). LibSmith ABI stable C shim APIs are guaranteed to have at least a two year compatibility window.
