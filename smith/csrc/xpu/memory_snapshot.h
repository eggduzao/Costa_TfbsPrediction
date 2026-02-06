#pragma once

#include <smith/csrc/Export.h>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace smith::xpu {

SMITH_PYTHON_API void _record_memory_history(
    std::optional<std::string> enabled = "all",
    std::optional<std::string> context = "all",
    const std::string& stacks = "all",
    size_t max_entries = SIZE_MAX,
    bool clear_history = false,
    const std::vector<std::string>& skip_actions = {});

} // namespace smith::xpu
