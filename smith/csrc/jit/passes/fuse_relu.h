#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {
SMITH_API void FuseAddRelu(script::Module& module);
SMITH_API void FuseAddRelu(std::shared_ptr<Graph>& graph);
} // namespace smith::jit
