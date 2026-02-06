#pragma once

#include <smith/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <smith/csrc/distributed/rpc/message.h>
#include <smith/csrc/distributed/rpc/rpc_command_base.h>

namespace smith::distributed::autograd {

// Used to request other workers to clean up their autograd context.
class SMITH_API CleanupAutogradContextReq : public rpc::RpcCommandBase {
 public:
  explicit CleanupAutogradContextReq(int64_t context_id);
  // Serialization and deserialization methods.
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<CleanupAutogradContextReq> fromMessage(
      const rpc::Message& message);

  // Retrieve the context id we are cleaning up with this message.
  int64_t getContextId();

 private:
  int64_t context_id_;
};

} // namespace smith::distributed::autograd
