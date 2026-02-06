#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>
#include <memory>

namespace smith::jit {

// Run TensorExpressions-based fuser.
// If add_composed_op is true, creates a single operation that
// performs both the runtime check that types align
// and then the dispatch to the kernel/unoptimized graph
SMITH_API void FuseTensorExprs(
    std::shared_ptr<Graph>& graph,
    size_t min_group_size = 2,
    bool add_composed_op = false,
    bool fuse_to_dynamic_shapes = false);

SMITH_API void setTensorExprFuserEnabled(bool val);
SMITH_API bool tensorExprFuserEnabled();
SMITH_API void setTensorExprDynamicShapeFusionEnabled(bool val);
SMITH_API bool tensorExprDynamicShapeFusionEnabled();
SMITH_API bool setTexprReductionsEnabled(bool value);
SMITH_API bool texprReductionsEnabled();

SMITH_API void RemoveProfileNodesAndSpecializeTypes(
    std::shared_ptr<Graph>& graph);
SMITH_API bool hasTensorTypeSpecialization(Value* v);
SMITH_API void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph);
SMITH_API void removeTensorTypeSpecializations(Block* block);

using tensor_type_converter_t =
    c10::function_ref<TensorTypePtr(const TensorTypePtr& t)>;

// inserts a TypeCheck pattern
//
// around the guarded node that has a Subgraph attribute, this inserts a pattern
//
//   if TypeCheck(...):
//     guarded_node
//   else:
//     FallbackGraph(...)
//
// The TypeCheck includes the types of all Tensor inputs to the guarded_node,
// as processed by the type_converter, a lambda
// TensorTypePtr(const TensorTypePtr& t). This allows to erase irrelevant
// aspects of the type.
//
// The Fallback graph will have the same subgraph as the guarded node (with the
// expectation that the guarded_node's subgraph will then be optimized.
SMITH_API void insertTypeGuard(
    Node* guarded_node,
    tensor_type_converter_t type_converter,
    c10::Symbol kind);

SMITH_API bool usedOnlyInSize(Value* v);
SMITH_API Value* broadcastSizes(at::ArrayRef<Value*> sizes, AliasDb* db);

namespace tensorexpr {
SMITH_API bool isSupported(Node* node);

/// Get the modifiable custom operator set object.
///
/// For static shapes, if a custom operator has been added to the custom
/// operator set, it will be pulled into the NNC fusion group. But it doesn't
/// work with dynamic shapes unless explicitly register the shape function via
/// `smith::jit::RegisterShapeComputeGraphForSchema` for the custom operator.
///
/// @return Reference of the custom operator set
///
SMITH_API OperatorSet& getCustomOperatorSet();

} // namespace tensorexpr
} // namespace smith::jit

C10_DECLARE_bool(smith_jit_disable_cat);
C10_DECLARE_bool(smith_jit_enable_dynamic_shape_fusion);
