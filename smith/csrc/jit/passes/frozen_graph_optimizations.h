#pragma once

#include <smith/csrc/jit/ir/ir.h>

/** \brief Runs a set of Optimizations that Optimize Frozen Graphs
 *
 * Currently this set of optimizations is:
 * - FoldFrozenConvBatchnorm
 * - FoldFrozenConvAddOrSub
 * - FoldFrozenConvMulOrDiv
 * - FoldFrozenLinearBatchnorm
 */

namespace smith::jit {

SMITH_API void OptimizeFrozenGraph(
    std::shared_ptr<Graph>& graph,
    bool optimize_numerics = true);

} // namespace smith::jit
