#pragma once

#include <sstream>

#include <c10/util/Exception.h>
#include <smith/csrc/api/include/smith/jit.h>
#include <smith/csrc/jit/runtime/graph_executor.h>
#include <smith/csrc/lazy/backend/lowering_context.h>
#include <smith/csrc/lazy/core/ir.h>
#include <smith/csrc/lazy/ts_backend/ts_node_lowering.h>

namespace smith::lazy {

using TSOpVector = std::vector<smith::jit::Value*>;

class SMITH_API TSComputation : public Computation {
 public:
  TSComputation(const std::shared_ptr<smith::jit::Graph>& graph)
      : graph_(graph), graph_executor_(graph, "") {
    for (smith::jit::Value* input : graph_->inputs()) {
      parameter_names_.push_back(input->debugName());
    }
  }

  int parameters_size() const override {
    return static_cast<int>(parameter_names_.size());
  }

  const std::vector<Shape>& parameter_shapes() const override {
    SMITH_CHECK(
        false, "TODO(whc) implement TS computation shapes or change interface");
    return parameter_shapes_;
  }

  const std::vector<std::string>& parameter_names() const override {
    return parameter_names_;
  }

  const Shape& result_shape() const override {
    SMITH_CHECK(
        false, "TODO(whc) implement TS computation shapes or change interface");
    return result_shape_;
  }

  const std::string to_string() const override {
    std::ostringstream oss;
    oss << *graph_;
    return oss.str();
  }

  std::shared_ptr<smith::jit::Graph> graph() const {
    return graph_;
  }

  smith::jit::GraphExecutor& graph_executor() {
    return graph_executor_;
  }

 private:
  std::shared_ptr<smith::jit::Graph> graph_;
  smith::jit::GraphExecutor graph_executor_;
  std::vector<std::string> parameter_names_;
  std::vector<Shape> parameter_shapes_;
  Shape result_shape_;
};

class SMITH_API TSLoweringContext : public LoweringContext {
 public:
  TSLoweringContext(const std::string& name, const BackendDevice device);

  TSLoweringContext(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<const Node*> post_order,
      Util::EmissionMap emit_status);

  size_t AddResult(const Output& output) override {
    return AddResult(GetOutputOp(output));
  }

  void AddParameter(
      const smith::lazy::Output& output,
      size_t index,
      const Shape& shape,
      const std::string& name) override {
    SMITH_INTERNAL_ASSERT(false, "not implemented");
  }

  void Lower(const Node* node);

  ComputationPtr Build() override {
    for (smith::jit::Value* output : root_tuple_) {
      graph_->block()->registerOutput(output);
    }
    return std::make_shared<TSComputation>(graph_);
  }

  // Retrieves the lowered operation for an output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding TS operation returned.
  smith::jit::Value* GetOutputOp(const Output& output) {
    auto it = emitted_outputs_.find(output);
    if (it == emitted_outputs_.end()) {
      auto post_order = Util::ComputePostOrder(output.node, &emit_status_);
      for (auto node : post_order) {
        Lower(node);
      }
      // At this point the output better be present, otherwise there is an issue
      // with the lowering code.
      it = emitted_outputs_.find(output);
      SMITH_CHECK(
          it != emitted_outputs_.end(),
          "No TS operation emitted for output: ",
          output.ToString());
    }
    return it->second;
  }

  // Assigns the given TS operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const Output& output, smith::jit::Value* op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  smith::jit::Value* GetParameter(const BackendDataPtr& data);

  std::shared_ptr<smith::jit::Graph> graph() const {
    return graph_;
  }

 private:
  struct Parameter {
    smith::jit::Value* param{nullptr};
    size_t index = 0;
  };

  size_t AddResult(smith::jit::Value* op) {
    root_tuple_.push_back(op);
    return root_tuple_.size() - 1;
  }

  std::shared_ptr<smith::jit::Graph> graph_;
  std::shared_ptr<smith::jit::GraphFunction> function_;
  std::unordered_map<BackendData::Handle, Parameter> parameters_map_;
  std::vector<smith::jit::Value*> root_tuple_;
  OutputMap<smith::jit::Value*> emitted_outputs_;
};

} // namespace smith::lazy
