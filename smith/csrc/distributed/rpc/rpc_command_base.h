#pragma once

#include <smith/csrc/distributed/rpc/message.h>
#include <smith/csrc/distributed/rpc/types.h>

namespace smith::distributed::rpc {

// Base class for all RPC request and responses.
class RpcCommandBase {
 public:
  // Need to override this to serialize the RPC. This should destructively
  // create a message for the RPC (Hence the &&).
  c10::intrusive_ptr<Message> toMessage() && {
    JitRRefPickleGuard jitPickleGuard;
    return std::move(*this).toMessageImpl();
  }
  virtual c10::intrusive_ptr<Message> toMessageImpl() && = 0;
  virtual ~RpcCommandBase() = 0;
};

inline RpcCommandBase::~RpcCommandBase() = default;

} // namespace smith::distributed::rpc
