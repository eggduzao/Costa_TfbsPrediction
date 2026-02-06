#pragma once

#include <smith/csrc/distributed/rpc/message.h>
#include <smith/csrc/distributed/rpc/rpc_command_base.h>
#include <smith/csrc/distributed/rpc/types.h>
namespace smith::distributed::rpc {

class SMITH_API PythonRemoteCall : public RpcCommandBase {
 public:
  PythonRemoteCall(
      SerializedPyObj&& serializedPyObj,
      at::IValue retRRefId,
      at::IValue retForkId,
      const bool isAsyncExecution);

  inline const SerializedPyObj& serializedPyObj() const {
    return serializedPyObj_;
  }

  inline const at::IValue& retRRefId() const {
    return retRRefId_;
  }

  inline const at::IValue& retForkId() const {
    return retForkId_;
  }

  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<PythonRemoteCall> fromMessage(const Message& message);

 private:
  SerializedPyObj serializedPyObj_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const at::IValue retRRefId_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const at::IValue retForkId_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool isAsyncExecution_;
};

} // namespace smith::distributed::rpc
