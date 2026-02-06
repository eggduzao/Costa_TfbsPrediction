#include <ATen/record_function.h>
#include <smith/csrc/distributed/autograd/autograd.h>

namespace smith::distributed::autograd {

constexpr auto kDistAutogradBackwardProfilingKey =
    "smith::distributed::autograd::backward";

void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph) {
  C10_LOG_API_USAGE_ONCE("smith.distributed.autograd.backward");
  RECORD_FUNCTION(
      kDistAutogradBackwardProfilingKey, std::vector<c10::IValue>());
  try {
    DistEngine::getInstance().execute(context_id, roots, retain_graph);
  } catch (std::exception& e) {
    // FIXME: crashes if exception type is not RuntimeError
    SMITH_CHECK(false, e.what());
  }
}

} // namespace smith::distributed::autograd
