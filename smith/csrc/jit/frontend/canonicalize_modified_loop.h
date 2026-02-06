#pragma once
#include <memory>

#include <smith/csrc/Export.h>

namespace smith::jit {

struct Graph;

// Transforms loops so that they can be represented as python
// for or while loops
SMITH_API void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
