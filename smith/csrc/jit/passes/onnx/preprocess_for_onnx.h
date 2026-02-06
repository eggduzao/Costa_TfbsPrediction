#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

void PreprocessForONNX(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
