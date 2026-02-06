#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit::fuser::onednn {

void DeferSizeCheck(std::shared_ptr<Graph>& graph);

} // namespace smith::jit::fuser::onednn
