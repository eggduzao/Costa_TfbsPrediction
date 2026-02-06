#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit::onnx {

namespace ONNXScopeName {

std::string createFullScopeName(
    const std::string& class_name,
    const std::string& variable_name);
std::string variableName(const smith::jit::ScopePtr& scope);
std::string variableNameFromRoot(
    const smith::jit::ScopePtr& scope,
    const std::string& layer_separator);
std::string className(const smith::jit::ScopePtr& scope);
std::string classNameFromRoot(
    const smith::jit::ScopePtr& scope,
    const std::string& layer_separator);
bool isCompatibleScope(const smith::jit::ScopePtr& scope);

} // namespace ONNXScopeName

SMITH_API void AssignScopedNamesForNodeAndValue(std::shared_ptr<Graph>& graph);

} // namespace smith::jit::onnx
