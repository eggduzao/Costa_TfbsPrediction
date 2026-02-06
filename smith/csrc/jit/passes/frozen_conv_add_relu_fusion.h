#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API extern std::function<void(std::shared_ptr<Graph>&)>&
getFuseFrozenConvAddReluImpl();

SMITH_API void FuseFrozenConvAddRelu(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
