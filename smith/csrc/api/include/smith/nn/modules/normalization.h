#pragma once

#include <smith/nn/cloneable.h>
#include <smith/nn/functional/normalization.h>
#include <smith/nn/modules/_functions.h>
#include <smith/nn/options/normalization.h>
#include <smith/nn/pimpl.h>
#include <smith/types.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace smith::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LayerNorm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Layer Normalization over a mini-batch of inputs as described in
/// the paper `Layer Normalization`_ .
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.LayerNorm to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::LayerNormOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LayerNorm model(LayerNormOptions({2,
/// 2}).elementwise_affine(false).eps(2e-5));
/// ```
class SMITH_API LayerNormImpl : public smith::nn::Cloneable<LayerNormImpl> {
 public:
  LayerNormImpl(std::vector<int64_t> normalized_shape)
      : LayerNormImpl(LayerNormOptions(std::move(normalized_shape))) {}
  explicit LayerNormImpl(LayerNormOptions options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `LayerNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies layer normalization over a mini-batch of inputs as described in
  /// the paper `Layer Normalization`_ .
  ///
  /// The mean and standard-deviation are calculated separately over the last
  /// certain number dimensions which have to be of the shape specified by
  /// input `normalized_shape`.
  ///
  /// `Layer Normalization`: https://arxiv.org/abs/1607.06450
  Tensor forward(const Tensor& input);

  /// The options with which this module was constructed.
  LayerNormOptions options;

  /// The learned weight.
  /// Initialized to ones if the `elementwise_affine` option is set to `true`
  /// upon construction.
  Tensor weight;

  /// The learned bias.
  /// Initialized to zeros `elementwise_affine` option is set to `true` upon
  /// construction.
  Tensor bias;
};

/// A `ModuleHolder` subclass for `LayerNormImpl`.
/// See the documentation for `LayerNormImpl` class to learn what methods it
/// provides, and examples of how to use `LayerNorm` with
/// `smith::nn::LayerNormOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(LayerNorm);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LocalResponseNorm
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies local response normalization over an input signal composed
/// of several input planes, where channels occupy the second dimension.
/// Applies normalization across channels.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.LocalResponseNorm to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::LocalResponseNormOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LocalResponseNorm
/// model(LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.));
/// ```
class SMITH_API LocalResponseNormImpl
    : public Cloneable<LocalResponseNormImpl> {
 public:
  LocalResponseNormImpl(int64_t size)
      : LocalResponseNormImpl(LocalResponseNormOptions(size)) {}
  explicit LocalResponseNormImpl(const LocalResponseNormOptions& options_);

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `LocalResponseNormImpl` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  LocalResponseNormOptions options;
};

/// A `ModuleHolder` subclass for `LocalResponseNormImpl`.
/// See the documentation for `LocalResponseNormImpl` class to learn what
/// methods it provides, and examples of how to use `LocalResponseNorm` with
/// `smith::nn::LocalResponseNormOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(LocalResponseNorm);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CrossMapLRN2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// See the documentation for `smith::nn::CrossMapLRN2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CrossMapLRN2d model(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10));
/// ```
class SMITH_API CrossMapLRN2dImpl
    : public smith::nn::Cloneable<CrossMapLRN2dImpl> {
 public:
  CrossMapLRN2dImpl(int64_t size)
      : CrossMapLRN2dImpl(CrossMapLRN2dOptions(size)) {}
  explicit CrossMapLRN2dImpl(const CrossMapLRN2dOptions& options_)
      : options(options_) {}

  void reset() override;

  /// Pretty prints the `CrossMapLRN2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  smith::Tensor forward(const smith::Tensor& input);

  CrossMapLRN2dOptions options;
};

/// A `ModuleHolder` subclass for `CrossMapLRN2dImpl`.
/// See the documentation for `CrossMapLRN2dImpl` class to learn what methods it
/// provides, and examples of how to use `CrossMapLRN2d` with
/// `smith::nn::CrossMapLRN2dOptions`. See the documentation for `ModuleHolder`
/// to learn about Blacksmith's module storage semantics.
SMITH_MODULE(CrossMapLRN2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GroupNorm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Group Normalization over a mini-batch of inputs as described in
/// the paper `Group Normalization`_ .
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.GroupNorm to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::GroupNormOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// GroupNorm model(GroupNormOptions(2, 2).eps(2e-5).affine(false));
/// ```
class SMITH_API GroupNormImpl : public smith::nn::Cloneable<GroupNormImpl> {
 public:
  GroupNormImpl(int64_t num_groups, int64_t num_channels)
      : GroupNormImpl(GroupNormOptions(num_groups, num_channels)) {}
  explicit GroupNormImpl(const GroupNormOptions& options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `GroupNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The options with which this module was constructed.
  GroupNormOptions options;

  /// The learned weight.
  Tensor weight;

  /// The learned bias.
  Tensor bias;
};

/// A `ModuleHolder` subclass for `GroupNormImpl`.
/// See the documentation for `GroupNormImpl` class to learn what methods it
/// provides, and examples of how to use `GroupNorm` with
/// `smith::nn::GroupNormOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(GroupNorm);

} // namespace smith::nn
