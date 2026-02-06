
#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Checks if the parameters, not including the
// first param are all constants.
bool nonConstantParameters(Node* n);

} // namespace smith::jit
