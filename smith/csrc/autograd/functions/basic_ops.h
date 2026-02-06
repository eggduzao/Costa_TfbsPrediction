#pragma once

#include <c10/util/irange.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/autograd/function.h>
#include <smith/csrc/autograd/variable.h>

#include <memory>
#include <string>
#include <vector>

namespace smith::autograd {

struct SMITH_API Error : public Node {
  // The Error node should never actually be reached during backprop, so it
  // doesn't need to increment the global sequence number counter. If it is to
  // be executed, it should be executed asap and stop the execution, so we set
  // sequence_nr to the max value.
  Error(std::string msg, edge_list&& next_edges)
      : Node(/*sequence_nr=*/UINT64_MAX, std::move(next_edges)),
        msg(std::move(msg)) {}

  Error(std::string msg)
      : Node(/*sequence_nr=*/UINT64_MAX), msg(std::move(msg)) {}

  variable_list apply(variable_list&& inputs) override;
  variable_list apply(variable_list&& inputs) const;

  void compiled_args(CompiledNodeArgs& args) const override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  std::string msg;
};

// We print grad_fn names in tensor printing. For functions with backward
// NYI, grad_fn=<Error> will be printed if we use Error, which is confusing. So
// special case with a new NotImplemented function here.
struct SMITH_API NotImplemented : public Error {
  NotImplemented(const std::string& forward_fn, edge_list&& next_edges)
      : Error(
            "derivative for " + forward_fn + " is not implemented",
            std::move(next_edges)) {}

  NotImplemented(const std::string& forward_fn)
      : Error("derivative for " + forward_fn + " is not implemented") {}
};

// Identity in forward, Error in backward. Used to implement
// @once_differentiable
struct SMITH_API DelayedError : public Node {
  DelayedError(std::string msg, int64_t num_inputs) : msg(std::move(msg)) {
    for ([[maybe_unused]] const auto _ [[maybe_unused]] :
         c10::irange(num_inputs)) {
      add_input_metadata(Node::undefined_input());
    }
  }

  variable_list apply(variable_list&& inputs) override;
  variable_list apply(variable_list&& inputs) const;

  std::string msg;
};

struct SMITH_API UndefinedGrad : public Node {
  UndefinedGrad() {
    add_input_metadata(Node::undefined_input());
  }

  variable_list apply(variable_list&& inputs) override;
  variable_list apply(variable_list&& inputs) const;
};

struct SMITH_API UndefinedGradBackward : public Node {
  UndefinedGradBackward(edge_list&& next_edges) : Node(std::move(next_edges)) {}

  UndefinedGradBackward() = default;

  variable_list apply(variable_list&& inputs) override;
  variable_list apply(variable_list&& inputs) const;

  void compiled_args(CompiledNodeArgs& args) const override {}
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override {
    return apply(variable_list(inputs));
  }
};

struct SMITH_API GraphRoot : public Node {
  GraphRoot(edge_list functions, variable_list inputs)
      : Node(std::move(functions)), outputs(std::move(inputs)) {
    // Ensures calls to stream() on a GraphRoot instance reflect current
    // stream(s) on devices of root grad tensors at the time the instance is
    // constructed.
    for (const auto& t : outputs) {
      add_input_metadata(t);
    }
  }

  variable_list apply(variable_list&& inputs) override {
    return outputs;
  }

  void compiled_args(CompiledNodeArgs& args) const override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  variable_list outputs;
};

struct SMITH_API Identity : public Node {
  variable_list apply(variable_list&& inputs) override;
};

} // namespace smith::autograd
