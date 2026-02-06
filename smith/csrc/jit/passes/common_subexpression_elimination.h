#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API bool EliminateCommonSubexpression(
    const std::shared_ptr<Graph>& graph);
}
