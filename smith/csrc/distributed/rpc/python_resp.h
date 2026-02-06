#pragma once

#include <smith/csrc/distributed/rpc/rpc_command_base.h>
#include <smith/csrc/distributed/rpc/types.h>

namespace smith::distributed::rpc {

// RPC call representing the response of a Python UDF over RPC.
class SMITH_API PythonResp final : public RpcCommandBase {
 public:
  explicit PythonResp(SerializedPyObj&& serializedPyObj);

  c10::intrusive_ptr<Message> toMessageImpl() && override;

  static std::unique_ptr<PythonResp> fromMessage(const Message& message);

  const SerializedPyObj& serializedPyObj() const;

 private:
  SerializedPyObj serializedPyObj_;
};

} // namespace smith::distributed::rpc
