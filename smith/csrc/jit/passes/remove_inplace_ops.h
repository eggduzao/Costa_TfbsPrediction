#pragma once

#include <smith/csrc/jit/ir/ir.h>

#include <memory>

namespace smith::jit {
// see .cpp for docs
SMITH_API void RemoveInplaceOps(const std::shared_ptr<Graph>& graph);

SMITH_API void ImplicitCastForBinaryInplaceOps(Block* block);
} // namespace smith::jit
