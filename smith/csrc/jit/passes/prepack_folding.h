#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

using PrePackingOpsFilterFn = std::function<bool(Node*)>;

void PrePackingOpsFolder(
    script::Module& m,
    const PrePackingOpsFilterFn& is_foldable_op,
    const std::string& attr_prefix);

} // namespace smith::jit
