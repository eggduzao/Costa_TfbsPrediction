#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void AnnotateWarns(const std::shared_ptr<Graph>& graph);

} // namespace smith::jit
