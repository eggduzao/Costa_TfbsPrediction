
#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void Autocast(const std::shared_ptr<Graph>& graph);

SMITH_API bool setAutocastMode(bool value);
SMITH_API bool autocastEnabled();

} // namespace smith::jit
