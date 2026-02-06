#pragma once

#include <smith/types.h>

namespace smith::nn::functional {

inline Tensor bilinear(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& weight,
    const Tensor& bias = Tensor()) {
  return smith::bilinear(input1, input2, weight, bias);
}

// ============================================================================

inline Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias = {}) {
  if (input.dim() == 2 && bias.defined()) {
    // fused op is marginally faster
    return smith::addmm(bias, input, weight.t());
  } else {
    auto output = input.matmul(weight.t());
    if (bias.defined()) {
      output += bias;
    }
    return output;
  }
}

} // namespace smith::nn::functional
