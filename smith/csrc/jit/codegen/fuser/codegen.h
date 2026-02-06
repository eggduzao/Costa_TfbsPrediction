#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/codegen/fuser/arg_spec.h>
#include <smith/csrc/jit/codegen/fuser/partition_desc.h>
#include <smith/csrc/jit/codegen/fuser/tensor_desc.h>
#include <smith/csrc/jit/ir/ir.h>

#include <string>
#include <vector>

namespace smith::jit::fuser {

// Creates a CPU or CUDA kernel for the given graph.
// Returns the C++ or CUDA string implementing the kernel.
SMITH_API std::string generateKernel(
    const std::string& name,
    const Graph& graph,
    const std::vector<std::pair<const Value*, const std::optional<TensorDesc>>>&
        inputs,
    const std::vector<std::pair<const Value*, const TensorDesc>>& outputs,
    const bool use_cuda);

} // namespace smith::jit::fuser
