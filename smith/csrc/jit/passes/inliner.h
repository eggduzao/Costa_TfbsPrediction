#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Inline function and method calls.
SMITH_API void Inline(Graph& graph);

SMITH_API GraphFunction* tryToGraphFunction(Node* n);

} // namespace smith::jit
