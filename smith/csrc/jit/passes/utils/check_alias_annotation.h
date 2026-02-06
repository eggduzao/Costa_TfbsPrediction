#pragma once

#include <ATen/core/ivalue.h>
#include <smith/csrc/jit/ir/ir.h>
#include <memory>
#include <string>
#include <vector>

namespace smith::jit {

// Verify that alias annotations are correct. See impl for definition of
// "correct".
//
// This function expects a graph with a single op with `unqualifiedOpName`, plus
// the inputs that you would otherwise have passed to the graph executor.
SMITH_API void checkAliasAnnotation(
    const std::shared_ptr<Graph>& graph,
    std::vector<IValue> pythonInputs,
    const std::string& unqualifiedOpName);
} // namespace smith::jit
