#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <smith/csrc/Export.h>
#include <cstdint>

namespace smith::jit::tensorexpr {

#ifdef C10_MOBILE
extern "C" {
#endif
void DispatchParallel(
    int8_t* func,
    int64_t start,
    int64_t stop,
    int8_t* packed_data) noexcept;

SMITH_API void nnc_aten_free(size_t bufs_num, void** ptrs) noexcept;

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace smith::jit::tensorexpr
