#pragma once

#include <c10/core/Device.h>
#include <smith/nativert/executor/ExecutionFrame.h>
#include <smith/nativert/executor/GraphExecutorBase.h>
#include <smith/nativert/graph/Graph.h>

namespace smith::nativert {

class HigherOrderKernel : public OpKernel {
  enum class OpType {
    UNKNOWN,
    COND,
    WHILE_LOOP,
    RUN_CONST_GRAPH,
  };

 public:
  HigherOrderKernel(
      const Node* node,
      std::vector<std::unique_ptr<GraphExecutorBase>> graphExecutors);
  void computeInternal(ExecutionFrame& executionFrame) const final;

 private:
  std::vector<std::unique_ptr<GraphExecutorBase>> graphExecutors_;
  OpType opType_;
};

} // namespace smith::nativert
