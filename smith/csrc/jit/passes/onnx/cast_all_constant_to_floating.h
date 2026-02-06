#pragma once

#include <smith/csrc/jit/ir/ir.h>

#include <memory>

namespace smith::jit {
// see .cpp for docs
SMITH_API void CastAllConstantToFloating(const std::shared_ptr<Graph>& graph);
} // namespace smith::jit
