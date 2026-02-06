#pragma once

#include <smith/csrc/distributed/rpc/message.h>
#include <smith/csrc/distributed/rpc/rpc_command_base.h>

namespace smith::distributed::autograd {

// Response for the RRefBackwardReq.
class SMITH_API RRefBackwardResp : public rpc::RpcCommandBase {
 public:
  RRefBackwardResp() = default;
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<RRefBackwardResp> fromMessage(
      const rpc::Message& message);
};

} // namespace smith::distributed::autograd
