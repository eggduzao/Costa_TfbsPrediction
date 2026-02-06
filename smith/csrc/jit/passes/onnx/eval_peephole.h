#pragma once

#include <memory>

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

void EvalPeepholeONNX(
    std::shared_ptr<Graph>& g,
    std::map<std::string, IValue>& paramDict);

} // namespace smith::jit
