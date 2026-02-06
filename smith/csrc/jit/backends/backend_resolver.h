#pragma once

#include <smith/csrc/jit/frontend/resolver.h>

namespace smith::jit {
// Create a Resolver for use in generating LoweredModules for specific backends.
SMITH_API std::shared_ptr<Resolver> loweredModuleResolver();
} // namespace smith::jit
