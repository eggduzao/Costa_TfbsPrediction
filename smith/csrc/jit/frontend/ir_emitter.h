#pragma once
#include <functional>
#include <memory>
#include <string>

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/frontend/error_report.h>
#include <smith/csrc/jit/frontend/resolver.h>
#include <smith/csrc/jit/frontend/sugared_value.h>
#include <smith/csrc/jit/frontend/tree_views.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void runCleanupPasses(std::shared_ptr<Graph>& to_clean);

SMITH_API bool meaningfulName(const std::string& name);

} // namespace smith::jit
