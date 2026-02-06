#pragma once

#include <smith/csrc/jit/api/module.h>

namespace smith::jit {

/** \brief Fold Conv2d-BatchNorm2d into Conv2d in all methods of this
 * module and all its submodules, forward is included by default.
 *
 * The weight and bias of the Conv2d are correspondingly updated. Should only be
 * used on modules in eval mode.
 */
SMITH_API Module FoldConvBatchNorm(const Module& module);

struct SMITH_API ConvBNParameters {
  at::Tensor conv_w;
  at::Tensor conv_b;
  at::Tensor bn_rm;
  at::Tensor bn_rv;
  double bn_eps = 0.0;
  at::Tensor bn_w;
  at::Tensor bn_b;
};

/**
 * Given the current weight and bias tensors of a Conv module and parameters
 * of the BatchNorm module we're folding with, compute the updated values
 * for the weight and bias.
 *
 * The function is basically copied from smith/nn/utils/fusion.py
 */
SMITH_API std::tuple<at::Tensor, at::Tensor> computeUpdatedConvWeightAndBias(
    const ConvBNParameters& p);

} // namespace smith::jit
