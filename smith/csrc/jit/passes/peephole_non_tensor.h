#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// return true if graph is modified
// Optimizing General Graph Patterns that
// are not covered in peephole.cpp and peephole_list_idioms
SMITH_API bool PeepholeOptimizeNonTensor(const std::shared_ptr<Graph>& graph);

} // namespace smith::jit
