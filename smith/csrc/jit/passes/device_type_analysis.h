#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {
struct Graph;

// Propagates Device type info throughout the given graph.
SMITH_API bool DeviceTypePropagation(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
