#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/types.h>

namespace smith::nn {

/// Options for the `CosineSimilarity` module.
///
/// Example:
/// ```
/// CosineSimilarity model(CosineSimilarityOptions().dim(0).eps(0.5));
/// ```
struct SMITH_API CosineSimilarityOptions {
  /// Dimension where cosine similarity is computed. Default: 1
  SMITH_ARG(int64_t, dim) = 1;
  /// Small value to avoid division by zero. Default: 1e-8
  SMITH_ARG(double, eps) = 1e-8;
};

namespace functional {
/// Options for `smith::nn::functional::cosine_similarity`.
///
/// See the documentation for `smith::nn::CosineSimilarityOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::cosine_similarity(input1, input2,
/// F::CosineSimilarityFuncOptions().dim(1));
/// ```
using CosineSimilarityFuncOptions = CosineSimilarityOptions;
} // namespace functional

// ============================================================================

/// Options for the `PairwiseDistance` module.
///
/// Example:
/// ```
/// PairwiseDistance
/// model(PairwiseDistanceOptions().p(3).eps(0.5).keepdim(true));
/// ```
struct SMITH_API PairwiseDistanceOptions {
  /// The norm degree. Default: 2
  SMITH_ARG(double, p) = 2.0;
  /// Small value to avoid division by zero. Default: 1e-6
  SMITH_ARG(double, eps) = 1e-6;
  /// Determines whether or not to keep the vector dimension. Default: false
  SMITH_ARG(bool, keepdim) = false;
};

namespace functional {
/// Options for `smith::nn::functional::pairwise_distance`.
///
/// See the documentation for `smith::nn::PairwiseDistanceOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::pairwise_distance(input1, input2, F::PairwiseDistanceFuncOptions().p(1));
/// ```
using PairwiseDistanceFuncOptions = PairwiseDistanceOptions;
} // namespace functional

} // namespace smith::nn
