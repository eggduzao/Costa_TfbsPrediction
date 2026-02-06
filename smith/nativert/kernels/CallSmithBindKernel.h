#pragma once

#include <c10/core/Device.h>
#include <smith/custom_class.h>

#include <smith/nativert/executor/ExecutionFrame.h>
#include <smith/nativert/executor/OpKernel.h>

namespace smith::nativert {

class CallSmithBindKernel : public OpKernel {
 public:
  CallSmithBindKernel() = delete; // deleted default constructor
  CallSmithBindKernel(const Node* node);

  void computeInternal(ExecutionFrame& executionFrame) const final;

 private:
  std::string methodName_;
  smith::jit::Function* method_;

  std::string customClassName_;
  at::ClassTypePtr customClassType_;
};

} // namespace smith::nativert
