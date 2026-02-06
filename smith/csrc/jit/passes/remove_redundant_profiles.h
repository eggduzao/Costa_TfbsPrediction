#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void RemoveRedundantProfiles(std::shared_ptr<Graph>& graph);
SMITH_API void RemoveRedundantProfiles(Block* block, AliasDb& db);
} // namespace smith::jit
