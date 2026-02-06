#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void removeDropout(std::shared_ptr<Graph>& graph);

SMITH_API void removeDropout(script::Module& module);

} // namespace smith::jit
