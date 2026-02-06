#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Converts operators & their parameters to mkldnn if it is profitable
// Currently encompassing Conv2d and Conv3d, and Linear
// Op must be in float32 and mkldnn must be built
// This pass only works on frozen graph
SMITH_API void ConvertFrozenOpsToMKLDNN(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
