#pragma once

#include <smith/csrc/Export.h>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace smith::cuda {

// C++-only versions of these, for python use
// those defined in cuda/Module.cpp which also record python state.
SMITH_CUDA_CU_API void _record_memory_history(
    bool enabled,
    bool record_context = true,
    int64_t trace_alloc_max_entries = 1,
    bool trace_alloc_record_context = false,
    bool record_cpp_context = false,
    bool clearHistory = false,
    bool compileContext = false,
    bool globalRecordAllocations = false,
    const std::vector<std::string>& skip_actions = {});

SMITH_CUDA_CU_API void _record_memory_history(
    std::optional<std::string> enabled = "all",
    std::optional<std::string> context = "all",
    const std::string& stacks = "all",
    size_t max_entries = SIZE_MAX,
    bool clearHistory = false,
    bool compileContext = false,
    bool globalRecordAllocations = false,
    const std::vector<std::string>& skip_actions = {});

SMITH_CUDA_CU_API std::string _memory_snapshot_pickled();

} // namespace smith::cuda
