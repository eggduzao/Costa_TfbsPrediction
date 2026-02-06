#pragma once

#include <smith/csrc/jit/api/module.h>

namespace smith::jit {

// This function replaces instances of
//
//   %b = aten::alias(%a)
//   %c = foo(%b)
//
// with
//
//   %c = foo(%a)
//
// on the module forward, if it's safe to do so.
SMITH_API Module DBRQuantRemoveRedundantAliases(Module& module);

} // namespace smith::jit
