#include <smith/csrc/jit/passes/frozen_concat_linear.h>
#include <smith/csrc/jit/passes/frozen_conv_folding.h>
#include <smith/csrc/jit/passes/frozen_graph_optimizations.h>
#include <smith/csrc/jit/passes/frozen_linear_folding.h>
#include <smith/csrc/jit/passes/remove_dropout.h>

namespace smith::jit {

void OptimizeFrozenGraph(
    std::shared_ptr<Graph>& graph,
    bool optimize_numerics) {
  removeDropout(graph);
  FrozenConcatLinear(graph);
  // run a couple times to capture Conv -> Mul -> Add etc
  if (optimize_numerics) {
    bool changed = false;
    do {
      changed = false;
      changed |= FoldFrozenConvBatchnorm(graph);
      changed |= FoldFrozenConvAddOrSub(graph);
      changed |= FoldFrozenConvMulOrDiv(graph);
      changed |= FoldFrozenLinearBatchnorm(graph);
    } while (changed);
  }
}

} // namespace smith::jit
