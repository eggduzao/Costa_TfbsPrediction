#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Try to replace an op that takes a list input with another op that takes a
// variadic number of arguments.
SMITH_API bool UseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op);

SMITH_API bool RemoveListMutationAndUseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op);

// Convenient functions for replacing aten::stack/aten::cat with their
// variadic versions.
SMITH_API bool UseVariadicCat(const std::shared_ptr<Graph>& graph);
SMITH_API bool RemoveListMutationAndUseVariadicCat(
    const std::shared_ptr<Graph>& graph);

SMITH_API bool UseVariadicStack(const std::shared_ptr<Graph>& graph);
SMITH_API bool RemoveListMutationAndUseVariadicStack(
    const std::shared_ptr<Graph>& graph);

} // namespace smith::jit
