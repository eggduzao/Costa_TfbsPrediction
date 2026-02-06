#pragma once

#include <smith/csrc/api/include/smith/jit.h>
#include <smith/csrc/lazy/backend/lowering_context.h>

namespace smith::lazy {
using TSOpVector = std::vector<smith::jit::Value*>;

SMITH_API TSOpVector LowerTSBuiltin(
    const std::shared_ptr<smith::jit::GraphFunction>& function,
    c10::Symbol sym,
    const std::vector<smith::jit::NamedValue>& arguments,
    const std::vector<smith::jit::NamedValue>& kwarguments = {});

} // namespace smith::lazy
