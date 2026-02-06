#pragma once

#include <ATen/cuda/CachingHostAllocator.h>

namespace at::cuda {

inline SMITH_CUDA_CPP_API at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kCUDA);
}
} // namespace at::cuda
