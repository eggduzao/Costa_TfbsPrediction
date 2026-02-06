#pragma once

#include <smith/csrc/Export.h>
#include <cstddef>

namespace smith::jit {

SMITH_API size_t ComputeEditDistance(
    const char* word1,
    const char* word2,
    size_t maxEditDistance);

} // namespace smith::jit
