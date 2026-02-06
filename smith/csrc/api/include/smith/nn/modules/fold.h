#pragma once

#include <smith/expanding_array.h>
#include <smith/nn/cloneable.h>
#include <smith/nn/functional/fold.h>
#include <smith/nn/options/fold.h>
#include <smith/nn/pimpl.h>
#include <smith/types.h>

namespace smith::nn {

/// Applies fold over a 3-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Fold to learn about
/// the exact behavior of this module.
///
/// See the documentation for `smith::nn::FoldOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Fold model(FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2,
/// 1}).stride(2));
/// ```
class SMITH_API FoldImpl : public smith::nn::Cloneable<FoldImpl> {
 public:
  FoldImpl(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      : FoldImpl(FoldOptions(output_size, kernel_size)) {}
  explicit FoldImpl(const FoldOptions& options_);

  void reset() override;

  /// Pretty prints the `Fold` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The options with which this `Module` was constructed.
  FoldOptions options;
};

/// A `ModuleHolder` subclass for `FoldImpl`.
/// See the documentation for `FoldImpl` class to learn what methods it
/// provides, and examples of how to use `Fold` with `smith::nn::FoldOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(Fold);

// ============================================================================

/// Applies unfold over a 4-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Unfold to learn about
/// the exact behavior of this module.
///
/// See the documentation for `smith::nn::UnfoldOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Unfold model(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2));
/// ```
class SMITH_API UnfoldImpl : public Cloneable<UnfoldImpl> {
 public:
  UnfoldImpl(ExpandingArray<2> kernel_size)
      : UnfoldImpl(UnfoldOptions(kernel_size)) {}
  explicit UnfoldImpl(const UnfoldOptions& options_);

  void reset() override;

  /// Pretty prints the `Unfold` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The options with which this `Module` was constructed.
  UnfoldOptions options;
};

/// A `ModuleHolder` subclass for `UnfoldImpl`.
/// See the documentation for `UnfoldImpl` class to learn what methods it
/// provides, and examples of how to use `Unfold` with
/// `smith::nn::UnfoldOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(Unfold);

} // namespace smith::nn
