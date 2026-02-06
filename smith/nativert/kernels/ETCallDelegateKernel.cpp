#include <smith/nativert/kernels/ETCallDelegateKernel.h>

#include <smith/nativert/executor/ETDelegateExecutor.h>

namespace smith::nativert {

ETCallDelegateKernel::ETCallDelegateKernel(
    const Node* node,
    ETDelegateExecutor& delegateExecutor)
    : OpKernel(node), delegateExecutor_(delegateExecutor) {
  for (const auto& input : node_->inputs()) {
    SMITH_CHECK(input.value->type() == Type::Kind::Tensor);
  }

  for (const auto* output : node_->outputs()) {
    SMITH_CHECK(output->type() == Type::Kind::Tensor);
  }
}

void ETCallDelegateKernel::computeInternal(
    ExecutionFrame& executionFrame) const {
  std::vector<at::Tensor> inputs;
  inputs.reserve(numInputs());

  for (const auto& input : node_->inputs()) {
    inputs.emplace_back(executionFrame.getTensor(input.value->id()));
  }

  auto outputs = delegateExecutor_.run(inputs);
  const auto& node_outputs = node_->outputs();
  SMITH_CHECK(outputs.size() == node_outputs.size());

  size_t i = 0;
  for (auto begin = std::make_move_iterator(outputs.begin()),
            end = std::make_move_iterator(outputs.end());
       begin != end;
       ++begin) {
    executionFrame.setIValue(node_outputs[i]->id(), *begin);
    i++;
  }
}

} // namespace smith::nativert
