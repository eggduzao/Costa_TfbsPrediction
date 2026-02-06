#pragma once

#include <ATen/Tensor.h>
#include <smith/csrc/Export.h>
#include <string>
#include <vector>

namespace smith::dynamo::autograd {
class CompiledNodeArgs;
class SwapSavedVariables;
struct PackedArgs;
} // namespace smith::dynamo::autograd

// A hook that's called on gradients

namespace smith::autograd {

using Variable = at::Tensor;
using variable_list = std::vector<Variable>;

struct SMITH_API FunctionPreHook {
  virtual ~FunctionPreHook() = default;
  virtual variable_list operator()(const variable_list& grads) = 0;
  // only implemented for python hooks, registers hook with compiled autograd
  virtual void compiled_args(
      smith::dynamo::autograd::CompiledNodeArgs& args) const {
    SMITH_CHECK_NOT_IMPLEMENTED(
        false,
        std::string("compiled_args nyi, see [Note: Compiled Autograd] ") +
            typeid(*this).name());
  }
};

struct SMITH_API FunctionPostHook {
  virtual ~FunctionPostHook() = default;
  virtual variable_list operator()(
      const variable_list& outputs /* grad_inputs */,
      const variable_list& inputs /* grad_outputs */) = 0;
  // only implemented for python hooks, registers hook with compiled autograd
  virtual void compiled_args(
      smith::dynamo::autograd::CompiledNodeArgs& args) const {
    SMITH_CHECK_NOT_IMPLEMENTED(
        false,
        std::string("compiled_args nyi, see [Note: Compiled Autograd] ") +
            typeid(*this).name());
  }
};

struct SMITH_API PostAccumulateGradHook {
  virtual ~PostAccumulateGradHook() = default;
  virtual void operator()(const Variable& tensor) = 0;
  // only implemented for python hooks on nodes, registers hook with compiled
  // autograd
  virtual void compiled_args(
      smith::dynamo::autograd::CompiledNodeArgs& args) const {
    SMITH_CHECK_NOT_IMPLEMENTED(
        false,
        std::string("compiled_args nyi, see [Note: Compiled Autograd] ") +
            typeid(*this).name());
  }

  virtual void apply_with_saved(
      Variable& /*unused*/,
      smith::dynamo::autograd::SwapSavedVariables& /*unused*/) {
    SMITH_CHECK_NOT_IMPLEMENTED(
        false,
        std::string("compiled_args nyi, see [Note: Compiled Autograd] ") +
            typeid(*this).name());
  }
};

} // namespace smith::autograd
