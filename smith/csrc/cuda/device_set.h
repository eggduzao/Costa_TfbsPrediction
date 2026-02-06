#pragma once

#include <c10/cuda/CUDAMacros.h>
#include <bitset>
#include <cstddef>

namespace smith {

using device_set = std::bitset<C10_COMPILE_TIME_MAX_GPUS>;

} // namespace smith
