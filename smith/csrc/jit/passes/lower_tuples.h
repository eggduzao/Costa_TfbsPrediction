#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// removes tuples where TupleConstruct and TupleUnpack are matched
// but leaves tuples in place across if statements, loops, and as inputs/outputs
SMITH_API void LowerSimpleTuples(const std::shared_ptr<Graph>& graph);

// removes _all_ tuples and raises an error if some cannot be removed
// this is used by ONNX to ensure there are not tuples before conversion,
// but will not work on graphs whose inputs contain tuples.
SMITH_API void LowerAllTuples(const std::shared_ptr<Graph>& graph);

SMITH_API void LowerSimpleTuples(Block* block);

} // namespace smith::jit
