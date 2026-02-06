#include <smith/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>

namespace smith::distributed::autograd {

c10::intrusive_ptr<rpc::Message> CleanupAutogradContextResp::
    toMessageImpl() && {
  std::vector<smith::Tensor> tensors;
  std::vector<char> payload;
  return c10::make_intrusive<rpc::Message>(
      std::move(payload),
      std::move(tensors),
      rpc::MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP);
}

std::unique_ptr<CleanupAutogradContextResp> CleanupAutogradContextResp::
    fromMessage(const rpc::Message& message /* unused */) {
  return std::unique_ptr<CleanupAutogradContextResp>();
}

} // namespace smith::distributed::autograd
