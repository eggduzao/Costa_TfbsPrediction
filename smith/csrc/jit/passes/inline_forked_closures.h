#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void inlineForkedClosures(std::shared_ptr<Graph>& to_clean);

} // namespace smith::jit
