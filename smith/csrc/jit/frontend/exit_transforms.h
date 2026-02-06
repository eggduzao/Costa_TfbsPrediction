#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void TransformExits(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
