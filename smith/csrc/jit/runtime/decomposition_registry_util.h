#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API const std::string& GetSerializedDecompositions();

SMITH_API const OperatorMap<std::string>& GetDecompositionMapping();

} // namespace smith::jit
