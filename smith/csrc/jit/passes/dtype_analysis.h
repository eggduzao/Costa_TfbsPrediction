#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>
#include <memory>

namespace smith::jit {
struct Graph;

// Propagate tensor properties (e.g., dtype, device, is_contiguous, layout)
// propagation on all tensor objects. Currently, we only support dtype
// propagation
SMITH_API bool DtypePropagation(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
