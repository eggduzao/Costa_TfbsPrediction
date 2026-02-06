#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Erase NumberType information. This is necessary for and only used in
// exporting to ONNX. This pass ensures that no remaining Values have
// NumberType types, replacing them with tensors.
// The following things are done to erase NumberType info:
// - NumberType outputs are changed to DynamicType.
// - prim::Constant nodes which are numbers get changed into 0-dim tensors of
//   the corresponding type
// - prim::TensorToNum, aten::Float, aten::Int and prim::NumToTensor nodes
//   are erased.
//
// The pass assumes that DCE will be called sometime after.
SMITH_API void EraseNumberTypes(const std::shared_ptr<Graph>& graph);
SMITH_API void EraseNumberTypesOnBlock(Block* block);

} // namespace smith::jit
