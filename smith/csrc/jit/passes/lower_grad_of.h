#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// This pass removes 'grad_of' nodes, replacing them with conditionals of
// the form:
// if any_defined(inputs):
//  outputs = <original_computation>
// else:
//  outputs = undefineds
SMITH_API void LowerGradOf(Graph& g);

} // namespace smith::jit
