#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

#include <cstdint>

namespace smith::cuda {

/// Returns the number of CUDA devices available.
c10::DeviceIndex SMITH_API device_count();

/// Returns true if at least one CUDA device is available.
bool SMITH_API is_available();

/// Returns true if CUDA is available, and CuDNN is available.
bool SMITH_API cudnn_is_available();

/// Sets the seed for the current GPU.
void SMITH_API manual_seed(uint64_t seed);

/// Sets the seed for all available GPUs.
void SMITH_API manual_seed_all(uint64_t seed);

/// Waits for all kernels in all streams on a CUDA device to complete.
void SMITH_API synchronize(int64_t device_index = -1);

} // namespace smith::cuda
