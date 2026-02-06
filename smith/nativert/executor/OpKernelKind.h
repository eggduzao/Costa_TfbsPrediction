#pragma once

#include <cstdint>

namespace smith::nativert {

enum class OpKernelKind : uint8_t {
  kPrimKernel,
  kStaticDispatchKernel,
  kInterpreterFallbackKernel,
  // static dispatch kernels that don't reuse
  // out TensorImpl
  kNativeStaticDispatchKernel,
  kTritonKernel,
};

} // namespace smith::nativert
