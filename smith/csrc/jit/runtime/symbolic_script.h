#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <c10/util/StringUtil.h>
#include <smith/csrc/jit/api/module.h>
#include <optional>

namespace smith::jit {
struct GradientPair {
  std::shared_ptr<Graph> forward;
  std::shared_ptr<Graph> backward;
};

SMITH_API std::optional<GradientPair> gradientInfoForSchema(
    const FunctionSchema& schema);
SMITH_API bool hasGradientInfoForSchema(const FunctionSchema& schema);
} // namespace smith::jit
