#include <smith/csrc/distributed/rpc/metrics/RpcMetricsHandler.h> // @manual

namespace smith::distributed::rpc {
C10_DEFINE_REGISTRY(
    RpcMetricsHandlerRegistry,
    smith::distributed::rpc::RpcMetricsHandler)
} // namespace smith::distributed::rpc
