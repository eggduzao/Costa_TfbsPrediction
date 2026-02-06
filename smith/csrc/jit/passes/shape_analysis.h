#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>
#include <memory>

namespace smith::jit {

struct Graph;

struct propagation_error : std::exception {};

class PropertyPropBase {
  // Used for both Shape Propagation and Dtype/Device Propagation
 public:
  explicit PropertyPropBase(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}
  virtual ~PropertyPropBase() = default;

  void propagateBlock(Block* block, bool insert_expands = true);
  // insert_expands is used for shape inference

  void processIf(Node* node);
  void processLoop(Node* node);

 protected:
  virtual void propagateNode(Node* node, bool insert_expands = true) = 0;
  void setUnshapedType(Value* o);
  void setUnshapedType(Node* node);
  std::shared_ptr<Graph> graph_;
};

SMITH_API void EraseShapeInformation(const std::shared_ptr<Graph>& graph);
SMITH_API void PropagateInputShapes(const std::shared_ptr<Graph>& graph);

SMITH_API bool mergeTypes(
    ArrayRef<Value*> lhs,
    ArrayRef<Value*> rhs,
    ArrayRef<Value*> outputs);

} // namespace smith::jit
