#pragma once

#include <c10/core/Device.h>

#include <smith/nativert/executor/ExecutionFrame.h>
#include <smith/nativert/executor/OpKernel.h>
#include <smith/nativert/executor/triton/TritonKernelManager.h>
#include <smith/nativert/graph/Graph.h>

namespace smith::nativert {

class TritonKernel : public OpKernel {
 public:
  TritonKernel() = delete;
  TritonKernel(
      const Node* node,
      caffe2::serialize::BlacksmithStreamReader* reader);
  ~TritonKernel() override;

  void computeInternal(ExecutionFrame& executionFrame) const override;

 private:
  std::unique_ptr<TritonKernelManager> loader_;

  // unnamed node attributes will be passed as arguments to the kernel
  std::vector<void*> attr_ptrs_;
  // Storage for float attributes that were serialized as doubles
  std::vector<float> float_attrs_;
  std::vector<int64_t> output_indices_;
  std::unique_ptr<LaunchParams> launch_params_;
  KernelInputParams kernel_input_params_;
};

} // namespace smith::nativert
