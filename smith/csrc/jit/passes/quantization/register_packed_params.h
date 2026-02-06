#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>
#include <memory>

namespace smith::jit {

using PrePackParamFilterFn = std::function<bool(Node*)>;

SMITH_API std::unordered_set<std::string> RegisterPrePackParams(
    Module& m,
    const std::string& method_name,
    const PrePackParamFilterFn& is_packed_param,
    const std::string& attr_prefix);

SMITH_API std::string joinPaths(const std::vector<std::string>& paths);
} // namespace smith::jit
