#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API bool AddIfThenElseOp(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
