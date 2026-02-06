#include <smith/csrc/autograd/functions/utils.h>
#include <smith/csrc/distributed/autograd/context/container.h>
#include <smith/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <smith/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <smith/csrc/distributed/autograd/utils.h>
#include <smith/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <smith/csrc/distributed/rpc/rpc_agent.h>
#include <smith/csrc/distributed/rpc/types.h>

namespace smith::distributed::autograd {

using smith::distributed::autograd::AutogradMetadata;
using smith::distributed::autograd::RpcWithAutograd;
using smith::distributed::rpc::JitFuture;
using smith::distributed::rpc::Message;
using smith::distributed::rpc::MessageType;
using smith::distributed::rpc::RpcAgent;
using smith::distributed::rpc::WorkerInfo;

void addSendRpcBackward(
    const ContextPtr& autogradContext,
    const AutogradMetadata& autogradMetadata,
    std::vector<smith::Tensor>& tensors) {
  // Attach autograd information only for tensors requiring grad.
  std::vector<smith::Tensor> tensors_with_grad;
  std::copy_if(
      tensors.begin(),
      tensors.end(),
      std::back_inserter(tensors_with_grad),
      [](const smith::Tensor& t) { return t.requires_grad(); });

  // Attach the appropriate autograd edges.
  auto grad_fn = std::make_shared<SendRpcBackward>();
  grad_fn->set_next_edges(
      smith::autograd::collect_next_edges(tensors_with_grad));

  // Add the appropriate input metadata for the grad_fn.
  for (const auto& tensor : tensors_with_grad) {
    grad_fn->add_input_metadata(tensor);
  }

  // Record the send autograd function in our current context.
  autogradContext->addSendFunction(grad_fn, autogradMetadata.autogradMessageId);
}

ContextPtr addRecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    std::vector<smith::Tensor>& tensors,
    rpc::worker_id_t fromWorkerId,
    const rpc::DeviceMap& deviceMap) {
  // Initialize autograd context if necessary.
  auto& autogradContainer = DistAutogradContainer::getInstance();
  auto autogradContext =
      autogradContainer.getOrCreateContext(autogradMetadata.autogradContextId);

  if (!tensors.empty() && smith::autograd::compute_requires_grad(tensors)) {
    // Attach the tensors as inputs to the autograd function.
    auto grad_fn = std::make_shared<RecvRpcBackward>(
        autogradMetadata, autogradContext, fromWorkerId, deviceMap);
    for (auto& tensor : tensors) {
      if (tensor.requires_grad()) {
        smith::autograd::set_history(tensor, grad_fn);
      }
    }

    // Now update the autograd context with the necessary information.
    autogradContext->addRecvFunction(
        grad_fn, autogradMetadata.autogradMessageId);
  }

  return autogradContext;
}

static c10::intrusive_ptr<Message> getMessageWithProfiling(
    c10::intrusive_ptr<smith::distributed::rpc::Message> wrappedRpcMessage,
    MessageType msgType,
    smith::autograd::profiler::ProfilerConfig&& profilerConfig) {
  auto& remoteProfilerManager =
      smith::distributed::rpc::RemoteProfilerManager::getInstance();

  auto key = remoteProfilerManager.getCurrentProfilingKey();
  // generate a globally unique Id
  auto globallyUniqueProfilingId = remoteProfilerManager.getNextProfilerId();
  // Save a mapping of ID -> RPC profiling key and unset the current TLS key.
  remoteProfilerManager.saveRPCKey(globallyUniqueProfilingId, key);
  remoteProfilerManager.unsetCurrentKey();
  auto wrappedProfilingMsg = RpcWithProfilingReq(
      msgType,
      std::move(wrappedRpcMessage),
      std::move(profilerConfig),
      globallyUniqueProfilingId);

  return std::move(wrappedProfilingMsg).toMessage();
}

c10::intrusive_ptr<Message> getMessageWithAutograd(
    const rpc::worker_id_t dstId,
    c10::intrusive_ptr<smith::distributed::rpc::Message> wrappedRpcMsg,
    MessageType msgType,
    bool forceGradRecording,
    const rpc::DeviceMap& deviceMap) {
  auto& autogradContainer = DistAutogradContainer::getInstance();

  // If there is no valid context and no tensor requires grads, send original
  // rpc message. otherwise, attach grad info and grad functions and send
  // rpcWithAutograd message.
  auto tensorsRequireGrad =
      smith::autograd::compute_requires_grad(wrappedRpcMsg->tensors());
  if (!autogradContainer.hasValidContext() ||
      (!forceGradRecording && !tensorsRequireGrad)) {
    return wrappedRpcMsg;
  }

  // Retrieve the appropriate context to modify.
  auto autogradContext = autogradContainer.currentContext();

  // Wrap the original rpc with autograd information.
  AutogradMetadata autogradMetadata(
      autogradContext->contextId(), autogradContainer.newAutogradMessageId());
  auto rpcWithAutograd = std::make_unique<RpcWithAutograd>(
      RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_,
      msgType,
      autogradMetadata,
      std::move(wrappedRpcMsg),
      deviceMap);

  if (tensorsRequireGrad) {
    // Record autograd information for 'send'.
    addSendRpcBackward(
        autogradContext, autogradMetadata, rpcWithAutograd->tensors());
  }
  // Record the workerID
  autogradContext->addKnownWorkerId(dstId);

  return std::move(*rpcWithAutograd).toMessage();
}

c10::intrusive_ptr<JitFuture> sendMessageWithAutograd(
    RpcAgent& agent,
    const WorkerInfo& dst,
    c10::intrusive_ptr<smith::distributed::rpc::Message> wrappedRpcMsg,
    bool forceGradRecording,
    const float rpcTimeoutSeconds,
    bool forceDisableProfiling) {
  auto msg = getMessageWithAutograd(
      dst.id_,
      std::move(wrappedRpcMsg),
      MessageType::FORWARD_AUTOGRAD_REQ,
      forceGradRecording,
      agent.getDeviceMap(dst));

  // If profiler is enabled, wrap this message with profiling metadata that will
  // tell the remote end to process this request with the profiler enabled.
  if (!forceDisableProfiling) {
    switch (smith::profiler::impl::profilerType()) {
      case smith::profiler::impl::ActiveProfilerType::LEGACY: {
        auto profilerConfig = smith::autograd::profiler::getProfilerConfig();
        auto msgWithProfiling = getMessageWithProfiling(
            std::move(msg),
            rpc::MessageType::RUN_WITH_PROFILING_REQ,
            std::move(profilerConfig));
        return agent.send(dst, std::move(msgWithProfiling), rpcTimeoutSeconds);
      }
      case smith::profiler::impl::ActiveProfilerType::KINETO:
        SMITH_WARN_ONCE(
            "Profiling a distributed call with the Kineto profiler will profile "
            "the caller, but not the worker.");
        break;
      default:
        break;
    }
  }

  return agent.send(dst, std::move(msg), rpcTimeoutSeconds);
  ;
}

} // namespace smith::distributed::autograd
