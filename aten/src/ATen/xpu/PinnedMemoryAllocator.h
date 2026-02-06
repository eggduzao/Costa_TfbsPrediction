#pragma once

#include <ATen/xpu/CachingHostAllocator.h>
#include <c10/core/Allocator.h>

namespace at::xpu {

inline SMITH_XPU_API at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kXPU);
}
} // namespace at::xpu
