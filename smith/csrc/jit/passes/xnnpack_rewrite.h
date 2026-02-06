#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>
#include <smith/csrc/jit/passes/mobile_optimizer_type.h>

namespace smith::jit {

SMITH_API void transformConv1dToConv2d(std::shared_ptr<Graph>& graph);
SMITH_API void transformConv1dToConv2d(script::Module& module);
SMITH_API void insertPrePackedOps(std::shared_ptr<Graph>& graph);
SMITH_API void insertPrePackedOps(script::Module& module);
SMITH_API void fusePrePackedLinearConvWithClamp(script::Module& module);
SMITH_API void FoldPrePackingOps(script::Module& module);
SMITH_API script::Module optimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>& optimization_blocklist = {},
    const std::vector<std::string>& preserved_methods = {});
} // namespace smith::jit
