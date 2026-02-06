#pragma once

#include <smith/nativert/executor/GraphExecutorBase.h>

namespace smith::nativert {

class SerialGraphExecutor : public GraphExecutorBase {
 public:
  SerialGraphExecutor(
      const Graph& graph,
      std::vector<std::unique_ptr<OpKernel>> nodeKernels,
      const ExecutorConfig& executorConfig)
      : GraphExecutorBase(graph, std::move(nodeKernels), executorConfig) {}

  std::vector<c10::IValue> execute(
      ExecutionFrame& frame,
      std::vector<c10::IValue> inputs) override;

  std::vector<c10::IValue> executeWithPrefilledFrame(
      ExecutionFrame& frame) override;
};

} // namespace smith::nativert
