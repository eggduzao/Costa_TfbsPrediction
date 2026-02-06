#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void DecomposeOps(std::shared_ptr<Graph>& graph);

}
