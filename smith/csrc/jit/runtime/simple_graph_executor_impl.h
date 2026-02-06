#pragma once
#include <c10/util/Flags.h>
#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/runtime/graph_executor_impl.h>

namespace smith::jit {

struct SMITH_API SimpleGraphExecutorImpl : public GraphExecutorImplBase {
  SimpleGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);

  const ExecutionPlan& getPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth) override;
  GraphExecutorState getDebugState() override;
  ~SimpleGraphExecutorImpl() override = default;

 private:
  std::optional<ExecutionPlan> execution_plan_;
};

} // namespace smith::jit
