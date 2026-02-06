#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Peephole Optimizes alias sensitive peepholes
// Currently this is invoked as part of PeepholeOptimize
// return true if graph is modified
// Optimizes on TensorType if shape_peepholes is true
SMITH_API bool PeepholeOptimizeAliasSensitive(
    const std::shared_ptr<Graph>& graph,
    bool shape_peepholes);

} // namespace smith::jit
