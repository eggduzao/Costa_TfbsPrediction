#pragma once

#include <smith/nn/options/fold.h>

namespace smith::nn::functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor fold(
    const Tensor& input,
    ExpandingArray<2> output_size,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> dilation,
    ExpandingArray<2> padding,
    ExpandingArray<2> stride) {
  if (input.dim() == 3 || input.dim() == 2) {
    return smith::col2im(
        input, output_size, kernel_size, dilation, padding, stride);
  } else {
    SMITH_CHECK(
        false,
        "Input Error: Only unbatched (2D) or batched (3D) input Tensors are supported "
        "(got ",
        input.dim(),
        "D)");
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://blacksmith.org/docs/main/nn.functional.html#smith.nn.functional.fold
/// about the exact behavior of this functional.
///
/// See the documentation for `smith::nn::functional::FoldFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
/// ```
inline Tensor fold(const Tensor& input, const FoldFuncOptions& options) {
  return detail::fold(
      input,
      options.output_size(),
      options.kernel_size(),
      options.dilation(),
      options.padding(),
      options.stride());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor unfold(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> dilation,
    ExpandingArray<2> padding,
    ExpandingArray<2> stride) {
  if (input.dim() == 4) {
    return smith::im2col(input, kernel_size, dilation, padding, stride);
  } else {
    SMITH_CHECK(
        false,
        "Input Error: Only 4D input Tensors are supported "
        "(got ",
        input.dim(),
        "D)");
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://blacksmith.org/docs/main/nn.functional.html#smith.nn.functional.unfold
/// about the exact behavior of this functional.
///
/// See the documentation for `smith::nn::functional::UnfoldFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
/// ```
inline Tensor unfold(const Tensor& input, const UnfoldFuncOptions& options) {
  return detail::unfold(
      input,
      options.kernel_size(),
      options.dilation(),
      options.padding(),
      options.stride());
}

} // namespace smith::nn::functional
