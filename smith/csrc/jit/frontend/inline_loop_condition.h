#pragma once
#include <functional>
#include <memory>
#include <string>

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void InlineLoopCondition(std::shared_ptr<Graph>& graph);
SMITH_API void InlineBlockBeforeNode(Node* before_node, Block* block);

} // namespace smith::jit
