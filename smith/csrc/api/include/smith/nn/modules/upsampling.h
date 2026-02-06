#pragma once

#include <smith/nn/cloneable.h>
#include <smith/nn/functional/upsampling.h>
#include <smith/nn/options/upsampling.h>
#include <smith/nn/pimpl.h>
#include <smith/types.h>

#include <smith/csrc/Export.h>

#include <cstddef>
#include <ostream>

namespace smith::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Upsample ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D
/// (volumetric) data.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Upsample to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::UpsampleOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Upsample
/// model(UpsampleOptions().scale_factor({3}).mode(smith::kLinear).align_corners(false));
/// ```
class SMITH_API UpsampleImpl : public Cloneable<UpsampleImpl> {
 public:
  explicit UpsampleImpl(UpsampleOptions options_ = {});

  void reset() override;

  /// Pretty prints the `Upsample` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The options with which this `Module` was constructed.
  UpsampleOptions options;
};

/// A `ModuleHolder` subclass for `UpsampleImpl`.
/// See the documentation for `UpsampleImpl` class to learn what methods it
/// provides, and examples of how to use `Upsample` with
/// `smith::nn::UpsampleOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(Upsample);

} // namespace smith::nn
