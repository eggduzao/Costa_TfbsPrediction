#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

void HoistConvPackedParams(script::Module& m);

} // namespace smith::jit
