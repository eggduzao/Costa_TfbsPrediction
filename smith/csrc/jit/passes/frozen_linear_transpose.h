#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Transposes the weight matrix for frozen linear modules.
// and converts it into a matmul
SMITH_API bool FrozenLinearTranspose(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
