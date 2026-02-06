#pragma once

#include <ATen/core/ivalue_inl.h>

#include <smith/csrc/jit/mobile/code.h>
#include <smith/csrc/jit/mobile/function.h>
#include <smith/csrc/jit/serialization/import_export_functions.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace smith::jit {
struct Instruction;
struct Upgrader {
  int min_version;
  int max_version;
  std::string upgrader_name;
  int index;
};

// From operator_versions.yaml
SMITH_API const std::unordered_map<std::string, std::vector<Upgrader>>
getOperatorVersionMapForMobile();

struct OperatorString {
  const std::string name;
  const std::string overload_name;
  const std::optional<int> num_specified_args;
};

struct ByteCodeFunctionWithOperator {
  mobile::Function& function;
  std::vector<OperatorString> operators;
};

SMITH_API const std::vector<ByteCodeFunctionWithOperator>&
getUpgraderBytecodeList();

} // namespace smith::jit
