#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API std::optional<std::shared_ptr<Graph>> GetDecomposition(
    const FunctionSchema& schema);

SMITH_API void RegisterDecomposition(
    const FunctionSchema& schema,
    std::shared_ptr<Graph> g);

SMITH_API void RunDecompositions(std::shared_ptr<Graph> g);

SMITH_API std::optional<GraphFunction*> GetDecompositionFunction(
    const FunctionSchema& schema);

// For invocation in C++, recommended is to assign to static local variable
SMITH_API Function* GetDecompositionExecutor(const char* schema_literal);

SMITH_API Function* GetDecompositionExecutor(const FunctionSchema& schema);

SMITH_API void run_jit_decomposition(
    const c10::OperatorHandle& op,
    smith::jit::Stack* stack);

SMITH_API bool has_jit_decomposition(const FunctionSchema& schema);

} // namespace smith::jit
