#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/api/module.h>

namespace smith::jit {

SMITH_API const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name);
} // namespace smith::jit
