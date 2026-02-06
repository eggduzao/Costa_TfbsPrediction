#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

// Because differentiable graphs detach the gradients of input Tensors,
// creating and inlining differentiable graphs changes the requires_grad
// property of tensors in the graph. This pass updates prim::profiles
// requires_grad to keep profiled properties up to date, it does not update
// grad properties of other nodes like graph inputs bc the only downstream
// user of the grad property is the profiling executor, which just uses
// the types of prim::profiles
SMITH_API void UpdateDifferentiableGraphRequiresGrad(
    std::shared_ptr<Graph>& diff_forward_graph,
    std::optional<bool> new_requires_grad);

} // namespace smith::jit
