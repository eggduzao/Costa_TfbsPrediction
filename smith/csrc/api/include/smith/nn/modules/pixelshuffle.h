#pragma once

#include <smith/nn/cloneable.h>
#include <smith/nn/functional/pixelshuffle.h>
#include <smith/nn/options/pixelshuffle.h>

#include <smith/csrc/Export.h>

namespace smith::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PixelShuffle
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
/// to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an
/// upscale factor. See
/// https://blacksmith.org/docs/main/nn.html#smith.nn.PixelShuffle to learn about
/// the exact behavior of this module.
///
/// See the documentation for `smith::nn::PixelShuffleOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PixelShuffle model(PixelShuffleOptions(5));
/// ```
struct SMITH_API PixelShuffleImpl
    : public smith::nn::Cloneable<PixelShuffleImpl> {
  explicit PixelShuffleImpl(const PixelShuffleOptions& options_);

  /// Pretty prints the `PixelShuffle` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  void reset() override;

  /// The options with which this `Module` was constructed.
  PixelShuffleOptions options;
};

/// A `ModuleHolder` subclass for `PixelShuffleImpl`.
/// See the documentation for `PixelShuffleImpl` class to learn what methods it
/// provides, and examples of how to use `PixelShuffle` with
/// `smith::nn::PixelShuffleOptions`. See the documentation for `ModuleHolder`
/// to learn about Blacksmith's module storage semantics.
SMITH_MODULE(PixelShuffle);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PixelUnshuffle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Reverses the PixelShuffle operation by rearranging elements in a tensor of
/// shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape :math:`(*,
/// C \times r^2, H, W)`, where r is a downscale factor. See
/// https://blacksmith.org/docs/main/nn.html#smith.nn.PixelUnshuffle to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::PixelUnshuffleOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PixelUnshuffle model(PixelUnshuffleOptions(5));
/// ```
struct SMITH_API PixelUnshuffleImpl
    : public smith::nn::Cloneable<PixelUnshuffleImpl> {
  explicit PixelUnshuffleImpl(const PixelUnshuffleOptions& options_);

  /// Pretty prints the `PixelUnshuffle` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  void reset() override;

  /// The options with which this `Module` was constructed.
  PixelUnshuffleOptions options;
};

/// A `ModuleHolder` subclass for `PixelUnshuffleImpl`.
/// See the documentation for `PixelUnshuffleImpl` class to learn what methods
/// it provides, and examples of how to use `PixelUnshuffle` with
/// `smith::nn::PixelUnshuffleOptions`. See the documentation for `ModuleHolder`
/// to learn about Blacksmith's module storage semantics.
SMITH_MODULE(PixelUnshuffle);

} // namespace smith::nn
