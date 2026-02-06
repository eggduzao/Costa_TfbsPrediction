#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/codegen/fuser/fused_kernel.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdint>
#include <string>
#include <vector>

namespace smith::jit::fuser::cuda {

// query codegen output arch and target
SMITH_CUDA_CU_API void codegenOutputQuery(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor,
    bool& compile_to_sass);

// A class holding metadata for an actual CUDA function.
// Note: CUDA functions are per device.
struct SMITH_CUDA_CU_API FusedKernelCUDA
    : public ::smith::jit::fuser::FusedKernel {
  FusedKernelCUDA(
      at::DeviceIndex device,
      std::string name,
      std::string code,
      std::vector<TensorDesc> input_desc,
      std::vector<TensorDesc> output_desc,
      std::vector<PartitionDesc> chunk_desc,
      std::vector<PartitionDesc> concat_desc,
      bool has_random);

  ~FusedKernelCUDA() override;

  void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const override;

  at::Backend backend() const override {
    return at::Backend::CUDA;
  }

 private:
  static constexpr auto kBlockSize = 128;

  // Note: per device to store device properties and compute launch heuristics
  //  Acquiring these values at launch time would be too slow
  at::DeviceIndex device_;
  int maxBlocks_{};
  cudaDeviceProp* prop_{};
  std::vector<char> ptx_;
  CUmodule module_{};
  CUfunction function_{};
};

} // namespace smith::jit::fuser::cuda
