#pragma once

#include <smith/csrc/jit/api/module.h>

namespace smith::jit {

struct SMITH_API LinearBNParameters {
  at::Tensor linear_w;
  at::Tensor linear_b;
  at::Tensor bn_rm;
  at::Tensor bn_rv;
  double bn_eps = 0.0;
  at::Tensor bn_w;
  at::Tensor bn_b;
};

/**
 * Given the current weight and bias tensors of a Linear module and parameters
 * of the BatchNorm module we're folding with, compute the updated values
 * for the weight and bias.
 *
 * The function is basically copied from smith/nn/utils/fusion.py
 */
SMITH_API std::tuple<at::Tensor, at::Tensor> computeUpdatedLinearWeightAndBias(
    const LinearBNParameters& p);

} // namespace smith::jit
