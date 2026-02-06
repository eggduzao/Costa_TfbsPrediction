#pragma once

#include <smith/nativert/executor/ExecutionFrame.h>
#include <smith/nativert/executor/OpKernel.h>

namespace smith::nativert {

class ETDelegateExecutor;

class ETCallDelegateKernel : public OpKernel {
 public:
  explicit ETCallDelegateKernel(
      const Node* node,
      ETDelegateExecutor& delegateExecutor);

  void computeInternal(ExecutionFrame& executionFrame) const final;

 private:
  ETDelegateExecutor& delegateExecutor_;
};

} // namespace smith::nativert
