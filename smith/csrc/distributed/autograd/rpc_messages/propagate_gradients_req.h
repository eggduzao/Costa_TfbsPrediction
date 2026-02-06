#pragma once

#include <smith/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <smith/csrc/distributed/rpc/message.h>
#include <smith/csrc/distributed/rpc/rpc_command_base.h>
#include <vector>

namespace smith::distributed::autograd {

// Used to propagate gradients from one node to another during a distributed
// backwards pass. This RPC call is invoked when we hit a `recv` autograd
// function during backward pass execution.
class SMITH_API PropagateGradientsReq : public rpc::RpcCommandBase {
 public:
  PropagateGradientsReq(
      const AutogradMetadata& autogradMetadata,
      std::vector<smith::autograd::Variable> grads,
      bool retainGraph = false);

  const AutogradMetadata& getAutogradMetadata();

  const std::vector<smith::autograd::Variable>& getGrads();

  // Serialization and deserialization methods.
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<PropagateGradientsReq> fromMessage(
      const rpc::Message& message);

  // Whether or not to retain the autograd graph.
  bool retainGraph();

 private:
  AutogradMetadata autogradMetadata_;
  std::vector<smith::autograd::Variable> grads_;
  bool retainGraph_;
};

} // namespace smith::distributed::autograd
