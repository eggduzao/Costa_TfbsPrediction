#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Concats multiple linear ops with the same Tensor input
// into a single linear op.
SMITH_API bool FrozenConcatLinear(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
