#include <smith/csrc/jit/passes/refine_tuple_types.h>
#include <smith/csrc/jit/runtime/graph_iterator.h>

#include <utility>

namespace smith::jit {

namespace {
static void VisitTupleNode(Node* node) {
  SMITH_CHECK(
      node->outputs().size() == 1, "Tuple must have exactly one output!");

  Value* output = node->outputs()[0];
  auto tuple_type = output->type()->expectRef<TupleType>();

  SMITH_CHECK(
      tuple_type.containedTypes().size() == node->inputs().size(),
      "Number of contained types does not match number of inputs!");

  // Extract updated types from input values.
  std::vector<c10::TypePtr> types;
  for (const Value* input : node->inputs()) {
    types.push_back(input->type());
  }

  // Construct new tuple type based on input types.
  output->setType(tuple_type.withContained(std::move(types)));
}
} // anonymous namespace

void RefineTupleTypes(std::shared_ptr<Graph>& graph) {
  DepthFirstGraphNodeIterator it(graph);
  for (auto* node = it.next(); node != nullptr; node = it.next()) {
    if (node->kind() == prim::TupleConstruct) {
      VisitTupleNode(node);
    }
  }
}

} // namespace smith::jit
