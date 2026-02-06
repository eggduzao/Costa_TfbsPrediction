#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API std::pair<Module, std::vector<IValue>> list_module_parameters(
    const Module& module);

} // namespace smith::jit
