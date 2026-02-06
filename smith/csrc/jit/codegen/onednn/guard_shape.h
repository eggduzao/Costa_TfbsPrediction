#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit::fuser::onednn {

void prepareFusionGroupAndGuardOutputs(Block* block);

} // namespace smith::jit::fuser::onednn
