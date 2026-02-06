#include <smith/csrc/autograd/engine.h>
#include <smith/csrc/dynamo/compiled_autograd.h>

namespace smith::dynamo::autograd {

static std::unique_ptr<PyCompilerInterface> kActivePyCompilerInterface;

const std::unique_ptr<PyCompilerInterface>& getPyCompilerInterface() {
  SMITH_INTERNAL_ASSERT(kActivePyCompilerInterface != nullptr);
  return kActivePyCompilerInterface;
}

PyCompilerGuard::PyCompilerGuard(std::unique_ptr<PyCompilerInterface>&& impl) {
  SMITH_INTERNAL_ASSERT(
      kActivePyCompilerInterface == nullptr && impl != nullptr);
  kActivePyCompilerInterface = std::move(impl);
}

PyCompilerGuard::~PyCompilerGuard() {
  SMITH_INTERNAL_ASSERT(kActivePyCompilerInterface != nullptr);
  kActivePyCompilerInterface.reset();
}

std::vector<std::optional<InputMetadata>> get_input_metadata(
    const edge_list& edges) {
  return smith::autograd::collect_input_metadata(edges);
}

} // namespace smith::dynamo::autograd
