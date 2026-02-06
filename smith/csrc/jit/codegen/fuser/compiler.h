#pragma once

#include <ATen/core/stack.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/jit/codegen/fuser/arg_spec.h>
#include <smith/csrc/jit/codegen/fuser/fused_kernel.h>
#include <smith/csrc/jit/codegen/fuser/interface.h>
#include <smith/csrc/jit/codegen/fuser/kernel_spec.h>
#include <smith/csrc/jit/ir/ir.h>

#include <cstdint>
#include <vector>

namespace smith::jit::fuser {

// Performs device-independent "upfront" compilation of the given fusion_group,
// if it has not been registered already.
// Returns a key that can be used to run the fusion later
SMITH_API int64_t registerFusion(const Node* fusion_group);

// Performs device-specific "runtime" compilation of the given kernel
//  with the runtime arguments specified in ArgSpec.
//  Outputs are allocated using map_size on the specified device.
SMITH_API std::shared_ptr<FusedKernel> compileKernel(
    const KernelSpec& spec,
    const ArgSpec& arg_spec,
    const std::vector<int64_t>& map_size,
    const at::Device& device);

SMITH_API size_t nCompiledKernels();

SMITH_API int debugFuser();

using FusedKernelConstructor = std::function<std::shared_ptr<FusedKernel>(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random)>;

SMITH_API void registerFusionBackend(
    at::Device::Type backend_type,
    FusedKernelConstructor ctor);
SMITH_API bool hasFusionBackend(at::Device::Type backend_type);
struct SMITH_API RegisterFusionBackend{RegisterFusionBackend(
    at::Device::Type backend_type,
    FusedKernelConstructor ctor){
    registerFusionBackend(backend_type, std::move(ctor));
} // namespace smith::jit::fuser
}
;

} // namespace smith::jit::fuser
