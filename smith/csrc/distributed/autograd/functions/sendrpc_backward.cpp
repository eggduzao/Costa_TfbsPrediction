#include <smith/csrc/distributed/autograd/functions/sendrpc_backward.h>

namespace smith::distributed::autograd {

smith::autograd::variable_list SendRpcBackward::apply(
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    smith::autograd::variable_list&& inputs) {
  SMITH_INTERNAL_ASSERT(
      inputs.empty(), "SendRpcBackward should receive no inputs");

  // Each grad variable should be valid!
  for (const auto& grad : grads_) {
    SMITH_INTERNAL_ASSERT(
        grad.defined(), "BUG!: SendRpcBackward didn't receive valid gradients");
  }

  // Simply forwards the gradients over.
  return std::move(grads_);
}

void SendRpcBackward::setGrads(const smith::autograd::variable_list& grads) {
  grads_ = grads;
}

const smith::autograd::variable_list& SendRpcBackward::getGrads() const {
  return grads_;
}

} // namespace smith::distributed::autograd
