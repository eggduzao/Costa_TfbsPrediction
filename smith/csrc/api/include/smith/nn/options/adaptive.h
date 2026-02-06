#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/types.h>

namespace smith::nn {

/// Options for the `AdaptiveLogSoftmaxWithLoss` module.
///
/// Example:
/// ```
/// AdaptiveLogSoftmaxWithLoss model(AdaptiveLogSoftmaxWithLossOptions(8, 10,
/// {4, 8}).div_value(2.).head_bias(true));
/// ```
struct SMITH_API AdaptiveLogSoftmaxWithLossOptions {
  /* implicit */ AdaptiveLogSoftmaxWithLossOptions(
      int64_t in_features,
      int64_t n_classes,
      std::vector<int64_t> cutoffs);

  /// Number of features in the input tensor
  SMITH_ARG(int64_t, in_features);

  /// Number of classes in the dataset
  SMITH_ARG(int64_t, n_classes);

  /// Cutoffs used to assign targets to their buckets
  SMITH_ARG(std::vector<int64_t>, cutoffs);

  /// value used as an exponent to compute sizes of the clusters. Default: 4.0
  SMITH_ARG(double, div_value) = 4.;

  /// If ``true``, adds a bias term to the 'head' of
  /// the adaptive softmax. Default: false
  SMITH_ARG(bool, head_bias) = false;
};

} // namespace smith::nn
