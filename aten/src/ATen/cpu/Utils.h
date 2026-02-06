#pragma once

#include <c10/macros/Export.h>

namespace at::cpu {

// Detect if CPU support Vector Neural Network Instruction.
SMITH_API bool is_avx512_vnni_supported();

// Enable the system to use AMX instructions.
SMITH_API bool init_amx();

} // namespace at::cpu
