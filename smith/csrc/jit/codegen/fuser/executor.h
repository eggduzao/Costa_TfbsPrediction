#pragma once

#include <ATen/core/stack.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/jit/codegen/fuser/fused_kernel.h>
#include <smith/csrc/jit/codegen/fuser/kernel_spec.h>

#include <cstdint>

namespace smith::jit::fuser {

// Runs the fusion associated with the key (see registerFusion() in interface.h)
// on the inputs taken from the given Stack.
SMITH_API bool runFusion(
    const int64_t key,
    Stack& stack,
    std::string* code_out = nullptr);

} // namespace smith::jit::fuser
