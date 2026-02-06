#include <smith/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>

namespace smith::distributed::autograd {

AutogradMetadata::AutogradMetadata(
    int64_t autogradContextId_,
    int64_t autogradMessageId_)
    : autogradContextId(autogradContextId_),
      autogradMessageId(autogradMessageId_) {}

} // namespace smith::distributed::autograd
