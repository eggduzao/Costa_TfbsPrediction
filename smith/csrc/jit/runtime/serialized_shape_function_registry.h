#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API const std::string& GetSerializedShapeFunctions();

SMITH_API const OperatorMap<std::string>& GetShapeFunctionMappings();

SMITH_API const OperatorMap<std::pair<std::string, std::string>>&
GetBoundedShapeMappings();

} // namespace smith::jit
