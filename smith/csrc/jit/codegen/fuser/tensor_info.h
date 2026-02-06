#pragma once
#include <smith/csrc/Export.h>

#include <cstddef>
#include <cstdint>

namespace smith::jit::fuser {

// Host-side view of TensorInfo
// Note dims[0] - we need to dynamically allocate the dims.
struct SMITH_API TensorInfo {
  uint32_t* sizes(size_t nDim) {
    return &sizes_strides[0];
  }
  uint32_t* strides(size_t nDim) {
    return &sizes_strides[nDim];
  }

  void* data;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  uint32_t sizes_strides[0];
};

} // namespace smith::jit::fuser
