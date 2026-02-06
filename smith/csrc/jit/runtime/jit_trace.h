#include <smith/csrc/jit/ir/ir.h>
#include <memory>

namespace smith::jit {
SMITH_API std::shared_ptr<Graph> TraceGraph(
    const std::shared_ptr<Graph>& graph,
    Stack& stack);
} // namespace smith::jit
