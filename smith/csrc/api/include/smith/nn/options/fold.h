#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/expanding_array.h>
#include <smith/types.h>

namespace smith::nn {

/// Options for the `Fold` module.
///
/// Example:
/// ```
/// Fold model(FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2,
/// 1}).stride(2));
/// ```
struct SMITH_API FoldOptions {
  FoldOptions(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      : output_size_(output_size), kernel_size_(kernel_size) {}

  /// describes the spatial shape of the large containing tensor of the sliding
  /// local blocks. It is useful to resolve the ambiguity when multiple input
  /// shapes map to same number of sliding blocks, e.g., with stride > 0.
  SMITH_ARG(ExpandingArray<2>, output_size);

  /// the size of the sliding blocks
  SMITH_ARG(ExpandingArray<2>, kernel_size);

  /// controls the spacing between the kernel points; also known as the à trous
  /// algorithm.
  SMITH_ARG(ExpandingArray<2>, dilation) = 1;

  /// controls the amount of implicit zero-paddings on both sides for padding
  /// number of points for each dimension before reshaping.
  SMITH_ARG(ExpandingArray<2>, padding) = 0;

  /// controls the stride for the sliding blocks.
  SMITH_ARG(ExpandingArray<2>, stride) = 1;
};

namespace functional {
/// Options for `smith::nn::functional::fold`.
///
/// See the documentation for `smith::nn::FoldOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
/// ```
using FoldFuncOptions = FoldOptions;
} // namespace functional

// ============================================================================

/// Options for the `Unfold` module.
///
/// Example:
/// ```
/// Unfold model(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2));
/// ```
struct SMITH_API UnfoldOptions {
  UnfoldOptions(ExpandingArray<2> kernel_size) : kernel_size_(kernel_size) {}

  /// the size of the sliding blocks
  SMITH_ARG(ExpandingArray<2>, kernel_size);

  /// controls the spacing between the kernel points; also known as the à trous
  /// algorithm.
  SMITH_ARG(ExpandingArray<2>, dilation) = 1;

  /// controls the amount of implicit zero-paddings on both sides for padding
  /// number of points for each dimension before reshaping.
  SMITH_ARG(ExpandingArray<2>, padding) = 0;

  /// controls the stride for the sliding blocks.
  SMITH_ARG(ExpandingArray<2>, stride) = 1;
};

namespace functional {
/// Options for `smith::nn::functional::unfold`.
///
/// See the documentation for `smith::nn::UnfoldOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
/// ```
using UnfoldFuncOptions = UnfoldOptions;
} // namespace functional

} // namespace smith::nn
