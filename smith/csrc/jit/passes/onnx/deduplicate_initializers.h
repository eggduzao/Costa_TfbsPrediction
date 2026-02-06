#pragma once

#include <memory>

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,
    std::map<std::string, IValue>& paramsDict,
    bool is_train);

} // namespace smith::jit
