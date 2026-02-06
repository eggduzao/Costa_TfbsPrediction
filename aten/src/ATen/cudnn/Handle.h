#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cudnn/cudnn-wrapper.h>

namespace at::native {

SMITH_CUDA_CPP_API cudnnHandle_t getCudnnHandle();
} // namespace at::native
