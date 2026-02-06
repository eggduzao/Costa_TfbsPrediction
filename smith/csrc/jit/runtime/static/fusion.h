#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void fuseStaticSubgraphs(
    std::shared_ptr<Graph> graph,
    size_t min_size);

SMITH_API void performTensorExprFusion(
    std::shared_ptr<Graph> graph,
    std::vector<IValue> sample_inputs);

} // namespace smith::jit
