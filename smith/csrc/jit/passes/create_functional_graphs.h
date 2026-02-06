#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void CreateFunctionalGraphs(const std::shared_ptr<Graph>& graph);

SMITH_API void InlineFunctionalGraphs(const std::shared_ptr<Graph>& graph);

} // namespace smith::jit
