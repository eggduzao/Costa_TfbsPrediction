#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API const OperatorMap<std::string>& get_tensorexpr_elementwise_set();

} // namespace smith::jit
