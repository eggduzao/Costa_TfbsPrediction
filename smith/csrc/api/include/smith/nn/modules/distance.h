#pragma once

#include <smith/nn/cloneable.h>
#include <smith/nn/functional/distance.h>
#include <smith/nn/options/distance.h>
#include <smith/nn/pimpl.h>
#include <smith/types.h>

#include <smith/csrc/Export.h>

namespace smith::nn {

/// Returns the cosine similarity between :math:`x_1` and :math:`x_2`, computed
/// along `dim`.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.CosineSimilarity to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::CosineSimilarityOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CosineSimilarity model(CosineSimilarityOptions().dim(0).eps(0.5));
/// ```
class SMITH_API CosineSimilarityImpl : public Cloneable<CosineSimilarityImpl> {
 public:
  explicit CosineSimilarityImpl(const CosineSimilarityOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `CosineSimilarity` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input1, const Tensor& input2);

  /// The options with which this `Module` was constructed.
  CosineSimilarityOptions options;
};

/// A `ModuleHolder` subclass for `CosineSimilarityImpl`.
/// See the documentation for `CosineSimilarityImpl` class to learn what methods
/// it provides, and examples of how to use `CosineSimilarity` with
/// `smith::nn::CosineSimilarityOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(CosineSimilarity);

// ============================================================================

/// Returns the batchwise pairwise distance between vectors :math:`v_1`,
/// :math:`v_2` using the p-norm.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.PairwiseDistance to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::PairwiseDistanceOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PairwiseDistance
/// model(PairwiseDistanceOptions().p(3).eps(0.5).keepdim(true));
/// ```
class SMITH_API PairwiseDistanceImpl : public Cloneable<PairwiseDistanceImpl> {
 public:
  explicit PairwiseDistanceImpl(const PairwiseDistanceOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `PairwiseDistance` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input1, const Tensor& input2);

  /// The options with which this `Module` was constructed.
  PairwiseDistanceOptions options;
};

/// A `ModuleHolder` subclass for `PairwiseDistanceImpl`.
/// See the documentation for `PairwiseDistanceImpl` class to learn what methods
/// it provides, and examples of how to use `PairwiseDistance` with
/// `smith::nn::PairwiseDistanceOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(PairwiseDistance);

} // namespace smith::nn
