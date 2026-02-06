#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>
#include <smith/csrc/jit/mobile/nnc/context.h>

namespace smith::jit::mobile::nnc {

// Performs Ahead Of Time compilation of a given method in a model
// returning the compiled function and LLVM assembly code
SMITH_API std::pair<std::unique_ptr<Function>, const std::string> aotCompile(
    const std::string& method_name,
    std::shared_ptr<Graph>& subgraph,
    const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<at::ScalarType>& types,
    const std::string& kernel_func_name = "func");

} // namespace smith::jit::mobile::nnc
