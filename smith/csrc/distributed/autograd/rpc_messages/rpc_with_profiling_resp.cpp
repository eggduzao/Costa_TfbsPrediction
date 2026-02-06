#include <c10/util/irange.h>
#include <smith/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>
#include <smith/csrc/distributed/rpc/utils.h>
#include <smith/csrc/jit/serialization/pickle.h>

namespace smith::distributed::autograd {
using rpc::RpcCommandBase;

constexpr auto kProfileEventsStartIdx = 3;
// This constructor is called when creating the RpcProfilingResp before sending
// it as a message over the wire.
RpcWithProfilingResp::RpcWithProfilingResp(
    rpc::MessageType messageType,
    c10::intrusive_ptr<rpc::Message> wrappedMessage,
    std::vector<smith::autograd::profiler::LegacyEvent> profiledEvents,
    rpc::ProfilingId profilingId)
    : messageType_(messageType),
      wrappedMessage_(std::move(wrappedMessage)),
      tensors_(wrappedMessage_->tensors()),
      profiledEvents_(std::move(profiledEvents)),
      profilingId_(profilingId) {
  SMITH_INTERNAL_ASSERT(
      messageType_ == rpc::MessageType::RUN_WITH_PROFILING_RESP,
      "Incorrect Message type");
  wrappedMessageType_ = wrappedMessage_->type();
}
// this constructor is called in fromMessage() which is called when
// reconstructing this RPC command when processing a message of this type
RpcWithProfilingResp::RpcWithProfilingResp(
    rpc::MessageType messageType,
    std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
    rpc::MessageType wrappedMessageType,
    std::vector<smith::Tensor> tensors,
    std::vector<smith::autograd::profiler::LegacyEvent> profiledEvents,
    rpc::ProfilingId profilingId)
    : messageType_(messageType),
      wrappedRpc_(std::move(wrappedRpc)),
      wrappedMessageType_(wrappedMessageType),
      tensors_(std::move(tensors)),
      profiledEvents_(std::move(profiledEvents)),
      profilingId_(profilingId) {
  SMITH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrapped RPC cannot be null");
}

std::unique_ptr<RpcCommandBase> RpcWithProfilingResp::moveWrappedRpc() && {
  SMITH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return std::move(wrappedRpc_);
}

rpc::MessageType RpcWithProfilingResp::wrappedMessageType() const {
  return wrappedMessageType_;
}

std::vector<smith::autograd::profiler::LegacyEvent> RpcWithProfilingResp::
    getProfiledEvents() const {
  return profiledEvents_;
}

const rpc::ProfilingId& RpcWithProfilingResp::getProfilingId() const {
  return profilingId_;
}

void RpcWithProfilingResp::setWrappedRpc(
    std::unique_ptr<RpcCommandBase> wrappedRpc) {
  wrappedRpc_ = std::move(wrappedRpc);
}

c10::intrusive_ptr<rpc::Message> RpcWithProfilingResp::toMessageImpl() && {
  auto wrappedMsgId = wrappedMessage_->id();
  auto wrappedMsgType = wrappedMessage_->type();
  auto wrappedPayload = std::move(*wrappedMessage_).movePayload();
  // Wrapped payload should not be empty
  SMITH_INTERNAL_ASSERT(
      !wrappedPayload.empty(), "Wrapped payload cannot be empty");
  // Create ivalues to send over
  std::vector<at::IValue> ivalues{wrappedMsgType, profilingId_.toIValue()};
  // Attach the serialized events.
  ivalues.emplace_back(static_cast<int32_t>(profiledEvents_.size()));
  for (const auto& e : profiledEvents_) {
    ivalues.emplace_back(e.toIValue());
  }
  std::vector<smith::Tensor> tensorTable;
  std::vector<char> profilingPayload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);
  rpc::writeWrappedPayload(wrappedPayload, profilingPayload);

  auto returnMsg = c10::make_intrusive<rpc::Message>(
      std::move(wrappedPayload),
      std::move(tensors_),
      messageType_,
      wrappedMsgId);
  return returnMsg;
}

RpcCommandBase& RpcWithProfilingResp::wrappedRpc() {
  SMITH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return *wrappedRpc_;
}

// Runs on client when deserializing this message.
std::unique_ptr<RpcWithProfilingResp> RpcWithProfilingResp::fromMessage(
    const rpc::Message& message) {
  rpc::MessageType origMsgType = message.type();
  std::vector<smith::Tensor> tensors = message.tensors();
  int64_t msgId = message.id();
  auto payload = message.payload();
  auto tupleElements = rpc::readWrappedPayload(payload, message);
  // Ensure that we have the expected number of elements
  SMITH_INTERNAL_ASSERT(
      tupleElements.size() >= kProfileEventsStartIdx,
      c10::str(
          "Expected payload size of at least ",
          kProfileEventsStartIdx,
          " but got size ",
          tupleElements.size()));
  rpc::MessageType wrappedMsgType =
      static_cast<rpc::MessageType>(tupleElements[0].toInt());
  rpc::ProfilingId profilingId = rpc::ProfilingId::fromIValue(tupleElements[1]);
  auto profiledEventsSize = tupleElements[2].toInt();
  std::vector<smith::autograd::profiler::LegacyEvent> remoteEvents;
  remoteEvents.reserve(profiledEventsSize);
  for (const auto i : c10::irange(
           kProfileEventsStartIdx,
           kProfileEventsStartIdx + profiledEventsSize)) {
    SMITH_CHECK(static_cast<size_t>(i) < tupleElements.size());
    // Reconstruct remote event from the ivalues.
    smith::autograd::profiler::LegacyEvent fromIvalueEvent =
        smith::autograd::profiler::LegacyEvent::fromIValue(tupleElements[i]);
    remoteEvents.push_back(std::move(fromIvalueEvent));
  }

  auto wrappedMessage = c10::make_intrusive<rpc::Message>(
      std::move(payload), std::move(tensors), wrappedMsgType, msgId);
  SMITH_INTERNAL_ASSERT(
      wrappedMessage->isResponse(),
      "Messages wrapped with profiling response must be responses.");
  std::unique_ptr<RpcCommandBase> wrappedRpc =
      deserializeResponse(*wrappedMessage, wrappedMsgType);
  return std::make_unique<RpcWithProfilingResp>(
      origMsgType,
      std::move(wrappedRpc),
      wrappedMsgType,
      std::move(wrappedMessage->tensors()),
      std::move(remoteEvents),
      profilingId);
}
} // namespace smith::distributed::autograd
