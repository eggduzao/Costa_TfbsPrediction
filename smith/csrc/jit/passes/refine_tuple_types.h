#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// updates the types of tuples according to the type of their current inputs.
SMITH_API void RefineTupleTypes(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
