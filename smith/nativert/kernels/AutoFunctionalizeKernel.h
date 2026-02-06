#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <c10/core/Device.h>

#include <smith/nativert/executor/ExecutionFrame.h>
#include <smith/nativert/executor/OpKernel.h>

namespace smith::nativert {

class UnsafeAutoFunctionalizeKernel : public OpKernel {
 public:
  UnsafeAutoFunctionalizeKernel() = delete; // deleted default constructor
  UnsafeAutoFunctionalizeKernel(const Node* node);

  void computeInternal(ExecutionFrame& executionFrame) const final;

 private:
  c10::OperatorHandle op_;
  c10::FunctionSchema schema_;

  Arguments arguments_;

  std::vector<Value*> mutatingInputArgs_;
  int numOutputs_;
};

} // namespace smith::nativert
