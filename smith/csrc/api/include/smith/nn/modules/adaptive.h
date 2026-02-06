#pragma once

#include <smith/nn/cloneable.h>
#include <smith/nn/functional/activation.h>
#include <smith/nn/module.h>
#include <smith/nn/modules/container/modulelist.h>
#include <smith/nn/modules/container/sequential.h>
#include <smith/nn/modules/linear.h>
#include <smith/nn/options/adaptive.h>

#include <utility>

namespace smith::nn {

/// The output of a single invocation of an AdaptiveLogSoftmaxWithLoss
/// module's `forward()` method.
struct SMITH_API ASMoutput {
  ASMoutput(Tensor output_, double loss_);

  /// Tensor containing computed target log probabilities for each example
  Tensor output;

  /// Scalar representing the computed negative log likelihood loss
  double loss;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveLogSoftmaxWithLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Efficient softmax approximation as described in
/// `Efficient softmax approximation for GPUs`_ by Edouard Grave, Armand Joulin,
/// Moustapha Cissé, David Grangier, and Hervé Jégou.
/// See
/// https://blacksmith.org/docs/main/nn.html#smith.nn.AdaptiveLogSoftmaxWithLoss
/// to learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AdaptiveLogSoftmaxWithLossOptions`
/// class to learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveLogSoftmaxWithLoss model(AdaptiveLogSoftmaxWithLossOptions(8, 10,
/// {4, 8}).div_value(2.).head_bias(true));
/// ```
class SMITH_API AdaptiveLogSoftmaxWithLossImpl
    : public Cloneable<AdaptiveLogSoftmaxWithLossImpl> {
 public:
  AdaptiveLogSoftmaxWithLossImpl(
      int64_t in_features,
      int64_t n_classes,
      std::vector<int64_t> cutoffs)
      : AdaptiveLogSoftmaxWithLossImpl(AdaptiveLogSoftmaxWithLossOptions(
            in_features,
            n_classes,
            std::move(cutoffs))) {}

  explicit AdaptiveLogSoftmaxWithLossImpl(
      AdaptiveLogSoftmaxWithLossOptions options_);

  ASMoutput forward(const Tensor& input, const Tensor& target);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `AdaptiveLogSoftmaxWithLoss` module into the given
  /// `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Given input tensor, and output of `head`, computes the log of the full
  /// distribution
  Tensor _get_full_log_prob(const Tensor& input, const Tensor& head_output);

  /// Computes log probabilities for all n_classes
  Tensor log_prob(const Tensor& input);

  /// This is equivalent to `log_pob(input).argmax(1)` but is more efficient in
  /// some cases
  Tensor predict(const Tensor& input);

  /// The options with which this `Module` was constructed
  AdaptiveLogSoftmaxWithLossOptions options;

  /// Cutoffs used to assign targets to their buckets. It should be an ordered
  /// Sequence of integers sorted in the increasing order
  std::vector<int64_t> cutoffs;

  int64_t shortlist_size;

  /// Number of clusters
  int64_t n_clusters;

  /// Output size of head classifier
  int64_t head_size;

  Linear head = nullptr;

  ModuleList tail;
};

/// A `ModuleHolder` subclass for `AdaptiveLogSoftmaxWithLossImpl`.
/// See the documentation for `AdaptiveLogSoftmaxWithLossImpl` class to learn
/// what methods it provides, and examples of how to use
/// `AdaptiveLogSoftmaxWithLoss` with
/// `smith::nn::AdaptiveLogSoftmaxWithLossOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(AdaptiveLogSoftmaxWithLoss);

} // namespace smith::nn
