#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

void PeepholeOptimizeONNX(
    std::shared_ptr<Graph>& graph,
    int opset_version,
    bool fixed_batch_size);

} // namespace smith::jit
