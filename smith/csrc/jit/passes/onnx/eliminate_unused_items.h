#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// EliminateUnusedItemsONNX pass is removing unused
// initializers and inputs, this is needed because
// dce pass is only removing unused fork inputs
void EliminateUnusedItemsONNX(
    Block* b,
    std::map<std::string, IValue>& paramDict);

} // namespace smith::jit
