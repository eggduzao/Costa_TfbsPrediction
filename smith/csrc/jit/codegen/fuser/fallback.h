#pragma once

#include <ATen/core/stack.h>

#include <cstdlib>

namespace smith::jit::fuser {

void runFallback(int64_t key, Stack& stack);

} // namespace smith::jit::fuser
