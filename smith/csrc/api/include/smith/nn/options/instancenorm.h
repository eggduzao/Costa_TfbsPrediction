#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/nn/options/batchnorm.h>
#include <smith/types.h>

namespace smith::nn {

/// Options for the `InstanceNorm` module.
struct SMITH_API InstanceNormOptions {
  /* implicit */ InstanceNormOptions(int64_t num_features);

  /// The number of features of the input tensor.
  SMITH_ARG(int64_t, num_features);

  /// The epsilon value added for numerical stability.
  SMITH_ARG(double, eps) = 1e-5;

  /// A momentum multiplier for the mean and variance.
  SMITH_ARG(double, momentum) = 0.1;

  /// Whether to learn a scale and bias that are applied in an affine
  /// transformation on the input.
  SMITH_ARG(bool, affine) = false;

  /// Whether to store and update batch statistics (mean and variance) in the
  /// module.
  SMITH_ARG(bool, track_running_stats) = false;
};

/// Options for the `InstanceNorm1d` module.
///
/// Example:
/// ```
/// InstanceNorm1d
/// model(InstanceNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using InstanceNorm1dOptions = InstanceNormOptions;

/// Options for the `InstanceNorm2d` module.
///
/// Example:
/// ```
/// InstanceNorm2d
/// model(InstanceNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using InstanceNorm2dOptions = InstanceNormOptions;

/// Options for the `InstanceNorm3d` module.
///
/// Example:
/// ```
/// InstanceNorm3d
/// model(InstanceNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using InstanceNorm3dOptions = InstanceNormOptions;

namespace functional {

/// Options for `smith::nn::functional::instance_norm`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::instance_norm(input,
/// F::InstanceNormFuncOptions().running_mean(mean).running_var(variance).weight(weight).bias(bias).momentum(0.1).eps(1e-5));
/// ```
struct SMITH_API InstanceNormFuncOptions {
  SMITH_ARG(Tensor, running_mean);

  SMITH_ARG(Tensor, running_var);

  SMITH_ARG(Tensor, weight);

  SMITH_ARG(Tensor, bias);

  SMITH_ARG(bool, use_input_stats) = true;

  SMITH_ARG(double, momentum) = 0.1;

  SMITH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace smith::nn
