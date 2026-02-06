#pragma once
#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>
#include <string>
#include <vector>

namespace smith::jit {
SMITH_API void metalInsertPrePackedOps(std::shared_ptr<Graph>& graph);
SMITH_API void metalInsertPrePackedOps(script::Module& module);
SMITH_API void metalFusePrePackedConvWithClamp(script::Module& module);
SMITH_API void metalFoldPrePackingOps(script::Module& module);
SMITH_API script::Module metalOptimizeForMobile(
    const script::Module& module,
    const std::vector<std::string>& preserved_methods);
} // namespace smith::jit
