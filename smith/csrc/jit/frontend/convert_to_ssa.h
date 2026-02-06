#pragma once
#include <functional>
#include <memory>
#include <string>

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Convert a graph with Loads & Stores into SSA form
SMITH_API void ConvertToSSA(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
