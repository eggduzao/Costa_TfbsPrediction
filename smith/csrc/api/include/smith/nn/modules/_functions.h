#pragma once

#include <smith/csrc/autograd/custom_function.h>
#include <smith/csrc/autograd/variable.h>
#include <smith/nn/options/normalization.h>
#include <smith/types.h>

namespace smith::nn::functions {

class CrossMapLRN2d : public smith::autograd::Function<CrossMapLRN2d> {
 public:
  static smith::autograd::Variable forward(
      smith::autograd::AutogradContext* ctx,
      const smith::autograd::Variable& input,
      const CrossMapLRN2dOptions& options);

  static smith::autograd::variable_list backward(
      smith::autograd::AutogradContext* ctx,
      smith::autograd::variable_list grad_output);
};

} // namespace smith::nn::functions
