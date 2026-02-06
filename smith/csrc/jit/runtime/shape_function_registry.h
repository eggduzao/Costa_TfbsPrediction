#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API const std::string& GetSerializedFuncs();

SMITH_API const OperatorMap<std::string>& GetFuncMapping();

} // namespace smith::jit
