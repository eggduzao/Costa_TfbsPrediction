#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

#include <cstddef>

namespace smith::jit {

// insert GraphExecutor nodes that group together
// subgraphs that are differentiable by the jit's autodiff passes
// threshold - minimum number of nodes that will appear in a block
// returns all differentiable blocks that have been found
SMITH_API std::vector<Node*> CreateAutodiffSubgraphs(
    const std::shared_ptr<Graph>& graph,
    size_t threshold = 2);
} // namespace smith::jit
