#include <smith/csrc/distributed/rpc/script_resp.h>

#include <smith/csrc/distributed/rpc/rpc_agent.h>
#include <smith/csrc/jit/serialization/pickle.h>
#include <smith/csrc/jit/serialization/unpickler.h>

namespace smith::distributed::rpc {

ScriptResp::ScriptResp(at::IValue&& value) : value_(std::move(value)) {}

const at::IValue& ScriptResp::value() {
  return value_;
}

c10::intrusive_ptr<Message> ScriptResp::toMessageImpl() && {
  std::vector<smith::Tensor> tensor_table;
  auto payload = jit::pickle(value_, &tensor_table);
  return c10::make_intrusive<Message>(
      std::move(payload), std::move(tensor_table), MessageType::SCRIPT_RET);
}

std::unique_ptr<ScriptResp> ScriptResp::fromMessage(const Message& message) {
  auto payload = message.payload().data();
  auto payload_size = message.payload().size();
  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  return std::make_unique<ScriptResp>(std::move(value));
}

} // namespace smith::distributed::rpc
