#pragma once

#include <smith/csrc/Export.h>

#include <memory>

namespace smith::jit {

struct Graph;
struct ArgumentSpec;

SMITH_API void PropagateRequiresGrad(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
