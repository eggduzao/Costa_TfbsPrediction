#pragma once

#include <smith/csrc/distributed/rpc/message.h>
#include <smith/csrc/distributed/rpc/rpc_command_base.h>

namespace smith::distributed::autograd {

// Empty response for CleanupAutogradContextReq. Send to acknowledge receipt of
// a CleanupAutogradContextReq.
class SMITH_API CleanupAutogradContextResp : public rpc::RpcCommandBase {
 public:
  CleanupAutogradContextResp() = default;
  // Serialization and deserialization methods.
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<CleanupAutogradContextResp> fromMessage(
      const rpc::Message& message);
};

} // namespace smith::distributed::autograd
