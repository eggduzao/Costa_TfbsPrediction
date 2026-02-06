#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void CanonicalizeOps(const std::shared_ptr<Graph>& graph);

}
