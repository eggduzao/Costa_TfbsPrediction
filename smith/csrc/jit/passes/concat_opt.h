#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Eliminates common inputs among `aten::cat` ops.
SMITH_API bool EliminateConcatCommonInputs(const std::shared_ptr<Graph>& graph);

// Expands `aten::cat` ops into `aten::copy` ops and eliminates redudancies
// in the buffers used for concatenation if possible.
SMITH_API void ExpandConcatAndEliminateRedundancy(
    const std::shared_ptr<Graph>& graph);

SMITH_API bool CombineConcats(const std::shared_ptr<Graph>& graph);

} // namespace smith::jit
