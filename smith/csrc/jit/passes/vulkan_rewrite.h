#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>
#include <smith/csrc/jit/passes/mobile_optimizer_type.h>

namespace smith::jit {
SMITH_API void vulkanInsertPrePackedOps(std::shared_ptr<Graph>& graph);
SMITH_API void vulkanInsertPrePackedOps(script::Module& module);
SMITH_API void vulkanFusePrePackedConvWithClamp(script::Module& module);
SMITH_API void vulkanFoldPrePackingOps(script::Module& module);
SMITH_API script::Module vulkanOptimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>& optimization_blocklist,
    const std::vector<std::string>& preserved_methods);
} // namespace smith::jit
