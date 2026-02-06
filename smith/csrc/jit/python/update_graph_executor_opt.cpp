#include <smith/csrc/jit/jit_log.h>
#include <smith/csrc/jit/python/update_graph_executor_opt.h>

namespace smith::jit {

static thread_local bool kOptimize = true;
void setGraphExecutorOptimize(bool o) {
  kOptimize = o;
  GRAPH_DEBUG("GraphExecutorOptimize set to ", o);
}
bool getGraphExecutorOptimize() {
  return kOptimize;
}

} // namespace smith::jit
