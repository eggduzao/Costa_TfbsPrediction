#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// return true if graph is modified
SMITH_API bool RefineIntegerValues(const std::shared_ptr<Graph>& graph);

} // namespace smith::jit
