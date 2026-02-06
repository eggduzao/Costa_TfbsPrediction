#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// return true if graph is modified
SMITH_API bool PeepholeOptimize(
    const std::shared_ptr<Graph>& graph,
    bool disable_shape_peepholes = false);
// return true if graph is modified
SMITH_API bool PeepholeOptimize(
    Block* block,
    bool disable_shape_peepholes = false);
// return true if graph is modified
SMITH_API bool FuseAddMM(const std::shared_ptr<Graph>& graph);

} // namespace smith::jit
