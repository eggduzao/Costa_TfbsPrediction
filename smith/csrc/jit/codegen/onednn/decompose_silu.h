#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit::fuser::onednn {

void DecomposeSiluForLLGA(std::shared_ptr<Graph>& graph);

} // namespace smith::jit::fuser::onednn
