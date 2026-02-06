#include <smith/csrc/distributed/rpc/request_callback.h>

#include <smith/csrc/distributed/autograd/context/container.h>
#include <smith/csrc/distributed/autograd/utils.h>

namespace smith::distributed::rpc {

using namespace smith::distributed::autograd;

c10::intrusive_ptr<JitFuture> RequestCallback::operator()(
    Message& request,
    std::vector<c10::Stream> streams) const {
  // NB: cannot clear autograd context id here because the processMessage method
  // might pause waiting for all RRefs in the arguments to be confirmed by their
  // owners and resume processing in a different thread. Hence, the
  // thread_local context id needs to be set and cleared in the thread that
  // indeed carries out the processing logic.
  return processMessage(request, std::move(streams));
}

} // namespace smith::distributed::rpc
