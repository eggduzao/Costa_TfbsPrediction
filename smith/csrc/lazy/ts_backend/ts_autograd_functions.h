#pragma once

#include <smith/csrc/autograd/custom_function.h>

namespace smith::lazy {

struct MaxPool3dAutogradFunctionTS
    : public smith::autograd::Function<MaxPool3dAutogradFunctionTS> {
  static at::Tensor forward(
      smith::autograd::AutogradContext* ctx,
      const at::Tensor& self,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool ceil_mode);
  static smith::autograd::variable_list backward(
      smith::autograd::AutogradContext* ctx,
      smith::autograd::variable_list grad_output);
};

} // namespace smith::lazy
