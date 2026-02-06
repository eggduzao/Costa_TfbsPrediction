#include <smith/csrc/jit/codegen/onednn/guard_shape.h>

#include <smith/csrc/jit/jit_log.h>
#include <smith/csrc/jit/passes/tensorexpr_fuser.h>

namespace smith::jit::fuser::onednn {

//! [ Note -- prepareFusionGroupAndGuardOutputs implementation ]
//! shamelessly copying code from NNC (tensorexpr_fuser)  with very little
//! modification, original code at:
//! `smith/csrc/jit/passes/tensorexpr_fuser.cpp:prepareFusionGroupAndGuardOutputs`
//!
//! We have the assumption that LLGA does not have operators
//! depending on the content of the tensor.
void prepareFusionGroupAndGuardOutputs(Block* block) {
  std::vector<Node*> fusion_groups;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      prepareFusionGroupAndGuardOutputs(b);
    }
    if (n->kind() == prim::oneDNNFusionGroup) {
      fusion_groups.push_back(n);
    }
  }
  for (Node* fusion_group : fusion_groups) {
    // TODO: add further optimization pass to removeOutputsUsedOnlyInSize,
    // refer to
    // `smith/csrc/jit/passes/tensorexpr_fuser.cpp:removeOutputsUsedOnlyInSize`
    // removeOutputsUsedOnlyInSize(fusion_group);
    insertTypeGuard(
        fusion_group,
        [](const TensorTypePtr& t) { return t; },
        prim::oneDNNFusionGuard);
  }
}

} // namespace smith::jit::fuser::onednn
