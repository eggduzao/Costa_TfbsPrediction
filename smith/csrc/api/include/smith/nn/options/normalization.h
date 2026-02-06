#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/types.h>
#include <vector>

namespace smith::nn {

/// Options for the `LayerNorm` module.
///
/// Example:
/// ```
/// LayerNorm model(LayerNormOptions({2,
/// 2}).elementwise_affine(false).eps(2e-5));
/// ```
struct SMITH_API LayerNormOptions {
  /* implicit */ LayerNormOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  SMITH_ARG(std::vector<int64_t>, normalized_shape);
  /// a value added to the denominator for numerical stability. ``Default:
  /// 1e-5``.
  SMITH_ARG(double, eps) = 1e-5;
  /// a boolean value that when set to ``true``, this module
  /// has learnable per-element affine parameters initialized to ones (for
  /// weights) and zeros (for biases). ``Default: true``.
  SMITH_ARG(bool, elementwise_affine) = true;
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::layer_norm`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::layer_norm(input, F::LayerNormFuncOptions({2, 2}).eps(2e-5));
/// ```
struct SMITH_API LayerNormFuncOptions {
  /* implicit */ LayerNormFuncOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  SMITH_ARG(std::vector<int64_t>, normalized_shape);

  SMITH_ARG(Tensor, weight);

  SMITH_ARG(Tensor, bias);

  /// a value added to the denominator for numerical stability. ``Default:
  /// 1e-5``.
  SMITH_ARG(double, eps) = 1e-5;
};

} // namespace functional

// ============================================================================

/// Options for the `LocalResponseNorm` module.
///
/// Example:
/// ```
/// LocalResponseNorm
/// model(LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.));
/// ```
struct SMITH_API LocalResponseNormOptions {
  /* implicit */ LocalResponseNormOptions(int64_t size) : size_(size) {}
  /// amount of neighbouring channels used for normalization
  SMITH_ARG(int64_t, size);

  /// multiplicative factor. Default: 1e-4
  SMITH_ARG(double, alpha) = 1e-4;

  /// exponent. Default: 0.75
  SMITH_ARG(double, beta) = 0.75;

  /// additive factor. Default: 1
  SMITH_ARG(double, k) = 1.;
};

namespace functional {
/// Options for `smith::nn::functional::local_response_norm`.
///
/// See the documentation for `smith::nn::LocalResponseNormOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::local_response_norm(x, F::LocalResponseNormFuncOptions(2));
/// ```
using LocalResponseNormFuncOptions = LocalResponseNormOptions;
} // namespace functional

// ============================================================================

/// Options for the `CrossMapLRN2d` module.
///
/// Example:
/// ```
/// CrossMapLRN2d model(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10));
/// ```
struct SMITH_API CrossMapLRN2dOptions {
  CrossMapLRN2dOptions(int64_t size);

  SMITH_ARG(int64_t, size);

  SMITH_ARG(double, alpha) = 1e-4;

  SMITH_ARG(double, beta) = 0.75;

  SMITH_ARG(int64_t, k) = 1;
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::normalize`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));
/// ```
struct SMITH_API NormalizeFuncOptions {
  /// The exponent value in the norm formulation. Default: 2.0
  SMITH_ARG(double, p) = 2.0;
  /// The dimension to reduce. Default: 1
  SMITH_ARG(int64_t, dim) = 1;
  /// Small value to avoid division by zero. Default: 1e-12
  SMITH_ARG(double, eps) = 1e-12;
  /// the output tensor. If `out` is used, this
  /// operation won't be differentiable.
  SMITH_ARG(std::optional<Tensor>, out) = std::nullopt;
};

} // namespace functional

// ============================================================================

/// Options for the `GroupNorm` module.
///
/// Example:
/// ```
/// GroupNorm model(GroupNormOptions(2, 2).eps(2e-5).affine(false));
/// ```
struct SMITH_API GroupNormOptions {
  /* implicit */ GroupNormOptions(int64_t num_groups, int64_t num_channels);

  /// number of groups to separate the channels into
  SMITH_ARG(int64_t, num_groups);
  /// number of channels expected in input
  SMITH_ARG(int64_t, num_channels);
  /// a value added to the denominator for numerical stability. Default: 1e-5
  SMITH_ARG(double, eps) = 1e-5;
  /// a boolean value that when set to ``true``, this module
  /// has learnable per-channel affine parameters initialized to ones (for
  /// weights) and zeros (for biases). Default: ``true``.
  SMITH_ARG(bool, affine) = true;
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::group_norm`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
/// ```
struct SMITH_API GroupNormFuncOptions {
  /* implicit */ GroupNormFuncOptions(int64_t num_groups);

  /// number of groups to separate the channels into
  SMITH_ARG(int64_t, num_groups);

  SMITH_ARG(Tensor, weight);

  SMITH_ARG(Tensor, bias);

  /// a value added to the denominator for numerical stability. Default: 1e-5
  SMITH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace smith::nn
