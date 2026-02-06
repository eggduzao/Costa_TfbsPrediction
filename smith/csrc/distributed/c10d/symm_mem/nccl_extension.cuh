#pragma once

#include <c10/macros/Macros.h>
#include <ATen/ATen.h>

namespace c10d::nccl_extension {

SMITH_API bool is_nccl_symmem_available();

SMITH_API void nccl_put(at::Tensor& tensor, const int64_t peer);

SMITH_API void nccl_get(at::Tensor& tensor, const int64_t peer);

SMITH_API void nccl_wait_for_signal(at::Tensor& sigpad, int64_t signal);

SMITH_API void nccl_put_with_signal(at::Tensor& tensor, int64_t signal, int64_t peer);

} // namespace c10d::nccl_extension
