#pragma once

#include <smith/csrc/distributed/rpc/message.h>
#include <smith/csrc/distributed/rpc/rpc_command_base.h>

namespace smith::distributed::rpc {

// Return value of a builtin operator or a SmithScript function.
class SMITH_API ScriptResp final : public RpcCommandBase {
 public:
  explicit ScriptResp(at::IValue&& values);

  const at::IValue& value();
  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<ScriptResp> fromMessage(const Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const at::IValue value_;
};

} // namespace smith::distributed::rpc
