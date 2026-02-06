#pragma once

#include <smith/expanding_array.h>
#include <smith/nn/cloneable.h>
#include <smith/nn/functional/pooling.h>
#include <smith/nn/modules/common.h>
#include <smith/nn/options/pooling.h>

#include <smith/csrc/Export.h>

namespace smith::nn {

/// Base class for all (dimension-specialized) avgpool modules.
template <size_t D, typename Derived>
class SMITH_API AvgPoolImpl : public smith::nn::Cloneable<Derived> {
 public:
  AvgPoolImpl(ExpandingArray<D> kernel_size)
      : AvgPoolImpl(AvgPoolOptions<D>(kernel_size)) {}
  explicit AvgPoolImpl(const AvgPoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `AvgPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  AvgPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 1-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.AvgPool1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AvgPool1dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AvgPool1d model(AvgPool1dOptions(3).stride(2));
/// ```
class SMITH_API AvgPool1dImpl : public AvgPoolImpl<1, AvgPool1dImpl> {
 public:
  using AvgPoolImpl<1, AvgPool1dImpl>::AvgPoolImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AvgPool1dImpl`.
/// See the documentation for `AvgPool1dImpl` class to learn what methods it
/// provides, and examples of how to use `AvgPool1d` with
/// `smith::nn::AvgPool1dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(AvgPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 2-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.AvgPool2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AvgPool2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AvgPool2d model(AvgPool2dOptions({3, 2}).stride({2, 2}));
/// ```
class SMITH_API AvgPool2dImpl : public AvgPoolImpl<2, AvgPool2dImpl> {
 public:
  using AvgPoolImpl<2, AvgPool2dImpl>::AvgPoolImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AvgPool2dImpl`.
/// See the documentation for `AvgPool2dImpl` class to learn what methods it
/// provides, and examples of how to use `AvgPool2d` with
/// `smith::nn::AvgPool2dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(AvgPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 3-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.AvgPool3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AvgPool3dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AvgPool3d model(AvgPool3dOptions(5).stride(2));
/// ```
class SMITH_API AvgPool3dImpl : public AvgPoolImpl<3, AvgPool3dImpl> {
 public:
  using AvgPoolImpl<3, AvgPool3dImpl>::AvgPoolImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AvgPool3dImpl`.
/// See the documentation for `AvgPool3dImpl` class to learn what methods it
/// provides, and examples of how to use `AvgPool3d` with
/// `smith::nn::AvgPool3dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(AvgPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) maxpool modules.
template <size_t D, typename Derived>
class SMITH_API MaxPoolImpl : public smith::nn::Cloneable<Derived> {
 public:
  MaxPoolImpl(ExpandingArray<D> kernel_size)
      : MaxPoolImpl(MaxPoolOptions<D>(kernel_size)) {}
  explicit MaxPoolImpl(const MaxPoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `MaxPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  MaxPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 1-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.MaxPool1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::MaxPool1dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MaxPool1d model(MaxPool1dOptions(3).stride(2));
/// ```
class SMITH_API MaxPool1dImpl : public MaxPoolImpl<1, MaxPool1dImpl> {
 public:
  using MaxPoolImpl<1, MaxPool1dImpl>::MaxPoolImpl;
  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `smith::nn::MaxUnpool1d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `MaxPool1dImpl`.
/// See the documentation for `MaxPool1dImpl` class to learn what methods it
/// provides, and examples of how to use `MaxPool1d` with
/// `smith::nn::MaxPool1dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(MaxPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 2-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.MaxPool2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::MaxPool2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MaxPool2d model(MaxPool2dOptions({3, 2}).stride({2, 2}));
/// ```
class SMITH_API MaxPool2dImpl : public MaxPoolImpl<2, MaxPool2dImpl> {
 public:
  using MaxPoolImpl<2, MaxPool2dImpl>::MaxPoolImpl;
  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `smith::nn::MaxUnpool2d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `MaxPool2dImpl`.
/// See the documentation for `MaxPool2dImpl` class to learn what methods it
/// provides, and examples of how to use `MaxPool2d` with
/// `smith::nn::MaxPool2dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(MaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 3-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.MaxPool3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::MaxPool3dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MaxPool3d model(MaxPool3dOptions(3).stride(2));
/// ```
class SMITH_API MaxPool3dImpl : public MaxPoolImpl<3, MaxPool3dImpl> {
 public:
  using MaxPoolImpl<3, MaxPool3dImpl>::MaxPoolImpl;
  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `smith::nn::MaxUnpool3d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `MaxPool3dImpl`.
/// See the documentation for `MaxPool3dImpl` class to learn what methods it
/// provides, and examples of how to use `MaxPool3d` with
/// `smith::nn::MaxPool3dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(MaxPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) adaptive maxpool modules.
template <size_t D, typename output_size_t, typename Derived>
class SMITH_API AdaptiveMaxPoolImpl : public smith::nn::Cloneable<Derived> {
 public:
  AdaptiveMaxPoolImpl(output_size_t output_size)
      : AdaptiveMaxPoolImpl(
            AdaptiveMaxPoolOptions<output_size_t>(output_size)) {}
  explicit AdaptiveMaxPoolImpl(
      const AdaptiveMaxPoolOptions<output_size_t>& options_)
      : options(options_) {}

  void reset() override {}

  /// Pretty prints the `AdaptiveMaxPool{1,2,3}d` module into the given
  /// `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "smith::nn::AdaptiveMaxPool" << D << 'd'
           << "(output_size=" << options.output_size() << ')';
  }

  /// The options with which this `Module` was constructed.
  AdaptiveMaxPoolOptions<output_size_t> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive maxpool over a 1-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.AdaptiveMaxPool1d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AdaptiveMaxPool1dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveMaxPool1d model(AdaptiveMaxPool1dOptions(3));
/// ```
class SMITH_API AdaptiveMaxPool1dImpl
    : public AdaptiveMaxPoolImpl<1, ExpandingArray<1>, AdaptiveMaxPool1dImpl> {
 public:
  using AdaptiveMaxPoolImpl<1, ExpandingArray<1>, AdaptiveMaxPool1dImpl>::
      AdaptiveMaxPoolImpl;

  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool1d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool1dImpl`.
/// See the documentation for `AdaptiveMaxPool1dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveMaxPool1d` with
/// `smith::nn::AdaptiveMaxPool1dOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(AdaptiveMaxPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive maxpool over a 2-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.AdaptiveMaxPool2d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AdaptiveMaxPool2dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
/// ```
class SMITH_API AdaptiveMaxPool2dImpl : public AdaptiveMaxPoolImpl<
                                            2,
                                            ExpandingArrayWithOptionalElem<2>,
                                            AdaptiveMaxPool2dImpl> {
 public:
  using AdaptiveMaxPoolImpl<
      2,
      ExpandingArrayWithOptionalElem<2>,
      AdaptiveMaxPool2dImpl>::AdaptiveMaxPoolImpl;

  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool2d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool2dImpl`.
/// See the documentation for `AdaptiveMaxPool2dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveMaxPool2d` with
/// `smith::nn::AdaptiveMaxPool2dOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(AdaptiveMaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive maxpool over a 3-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.AdaptiveMaxPool3d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AdaptiveMaxPool3dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveMaxPool3d model(AdaptiveMaxPool3dOptions(3));
/// ```
class SMITH_API AdaptiveMaxPool3dImpl : public AdaptiveMaxPoolImpl<
                                            3,
                                            ExpandingArrayWithOptionalElem<3>,
                                            AdaptiveMaxPool3dImpl> {
 public:
  using AdaptiveMaxPoolImpl<
      3,
      ExpandingArrayWithOptionalElem<3>,
      AdaptiveMaxPool3dImpl>::AdaptiveMaxPoolImpl;

  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool3d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool3dImpl`.
/// See the documentation for `AdaptiveMaxPool3dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveMaxPool3d` with
/// `smith::nn::AdaptiveMaxPool3dOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(AdaptiveMaxPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) adaptive avgpool modules.
template <size_t D, typename output_size_t, typename Derived>
class SMITH_API AdaptiveAvgPoolImpl : public smith::nn::Cloneable<Derived> {
 public:
  AdaptiveAvgPoolImpl(output_size_t output_size)
      : AdaptiveAvgPoolImpl(
            AdaptiveAvgPoolOptions<output_size_t>(output_size)) {}
  explicit AdaptiveAvgPoolImpl(
      const AdaptiveAvgPoolOptions<output_size_t>& options_)
      : options(options_) {}

  void reset() override {}

  /// Pretty prints the `AdaptiveAvgPool{1,2,3}d` module into the given
  /// `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "smith::nn::AdaptiveAvgPool" << D << 'd'
           << "(output_size=" << options.output_size() << ')';
  }

  /// The options with which this `Module` was constructed.
  AdaptiveAvgPoolOptions<output_size_t> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveAvgPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive avgpool over a 1-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.AdaptiveAvgPool1d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AdaptiveAvgPool1dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveAvgPool1d model(AdaptiveAvgPool1dOptions(5));
/// ```
class SMITH_API AdaptiveAvgPool1dImpl
    : public AdaptiveAvgPoolImpl<1, ExpandingArray<1>, AdaptiveAvgPool1dImpl> {
 public:
  using AdaptiveAvgPoolImpl<1, ExpandingArray<1>, AdaptiveAvgPool1dImpl>::
      AdaptiveAvgPoolImpl;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveAvgPool1dImpl`.
/// See the documentation for `AdaptiveAvgPool1dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveAvgPool1d` with
/// `smith::nn::AdaptiveAvgPool1dOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(AdaptiveAvgPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveAvgPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive avgpool over a 2-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.AdaptiveAvgPool2d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AdaptiveAvgPool2dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveAvgPool2d model(AdaptiveAvgPool2dOptions({3, 2}));
/// ```
class SMITH_API AdaptiveAvgPool2dImpl : public AdaptiveAvgPoolImpl<
                                            2,
                                            ExpandingArrayWithOptionalElem<2>,
                                            AdaptiveAvgPool2dImpl> {
 public:
  using AdaptiveAvgPoolImpl<
      2,
      ExpandingArrayWithOptionalElem<2>,
      AdaptiveAvgPool2dImpl>::AdaptiveAvgPoolImpl;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveAvgPool2dImpl`.
/// See the documentation for `AdaptiveAvgPool2dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveAvgPool2d` with
/// `smith::nn::AdaptiveAvgPool2dOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(AdaptiveAvgPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveAvgPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive avgpool over a 3-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.AdaptiveAvgPool3d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::AdaptiveAvgPool3dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveAvgPool3d model(AdaptiveAvgPool3dOptions(3));
/// ```
class SMITH_API AdaptiveAvgPool3dImpl : public AdaptiveAvgPoolImpl<
                                            3,
                                            ExpandingArrayWithOptionalElem<3>,
                                            AdaptiveAvgPool3dImpl> {
 public:
  using AdaptiveAvgPoolImpl<
      3,
      ExpandingArrayWithOptionalElem<3>,
      AdaptiveAvgPool3dImpl>::AdaptiveAvgPoolImpl;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveAvgPool3dImpl`.
/// See the documentation for `AdaptiveAvgPool3dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveAvgPool3d` with
/// `smith::nn::AdaptiveAvgPool3dOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(AdaptiveAvgPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) maxunpool modules.
template <size_t D, typename Derived>
class SMITH_API MaxUnpoolImpl : public smith::nn::Cloneable<Derived> {
 public:
  MaxUnpoolImpl(ExpandingArray<D> kernel_size)
      : MaxUnpoolImpl(MaxUnpoolOptions<D>(kernel_size)) {}
  explicit MaxUnpoolImpl(const MaxUnpoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `MaxUnpool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  MaxUnpoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxUnpool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxunpool over a 1-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.MaxUnpool1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::MaxUnpool1dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MaxUnpool1d model(MaxUnpool1dOptions(3).stride(2).padding(1));
/// ```
class SMITH_API MaxUnpool1dImpl : public MaxUnpoolImpl<1, MaxUnpool1dImpl> {
 public:
  using MaxUnpoolImpl<1, MaxUnpool1dImpl>::MaxUnpoolImpl;
  Tensor forward(
      const Tensor& input,
      const Tensor& indices,
      const std::optional<std::vector<int64_t>>& output_size = std::nullopt);

 protected:
  FORWARD_HAS_DEFAULT_ARGS({2, AnyValue(std::optional<std::vector<int64_t>>())})
};

/// A `ModuleHolder` subclass for `MaxUnpool1dImpl`.
/// See the documentation for `MaxUnpool1dImpl` class to learn what methods it
/// provides, and examples of how to use `MaxUnpool1d` with
/// `smith::nn::MaxUnpool1dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(MaxUnpool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxUnpool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxunpool over a 2-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.MaxUnpool2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::MaxUnpool2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MaxUnpool2d model(MaxUnpool2dOptions(3).stride(2).padding(1));
/// ```
class SMITH_API MaxUnpool2dImpl : public MaxUnpoolImpl<2, MaxUnpool2dImpl> {
 public:
  using MaxUnpoolImpl<2, MaxUnpool2dImpl>::MaxUnpoolImpl;
  Tensor forward(
      const Tensor& input,
      const Tensor& indices,
      const std::optional<std::vector<int64_t>>& output_size = std::nullopt);

 protected:
  FORWARD_HAS_DEFAULT_ARGS({2, AnyValue(std::optional<std::vector<int64_t>>())})
};

/// A `ModuleHolder` subclass for `MaxUnpool2dImpl`.
/// See the documentation for `MaxUnpool2dImpl` class to learn what methods it
/// provides, and examples of how to use `MaxUnpool2d` with
/// `smith::nn::MaxUnpool2dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(MaxUnpool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxUnpool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxunpool over a 3-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.MaxUnpool3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::MaxUnpool3dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MaxUnpool3d model(MaxUnpool3dOptions(3).stride(2).padding(1));
/// ```
class SMITH_API MaxUnpool3dImpl : public MaxUnpoolImpl<3, MaxUnpool3dImpl> {
 public:
  using MaxUnpoolImpl<3, MaxUnpool3dImpl>::MaxUnpoolImpl;
  Tensor forward(
      const Tensor& input,
      const Tensor& indices,
      const std::optional<std::vector<int64_t>>& output_size = std::nullopt);

 protected:
  FORWARD_HAS_DEFAULT_ARGS({2, AnyValue(std::optional<std::vector<int64_t>>())})
};

/// A `ModuleHolder` subclass for `MaxUnpool3dImpl`.
/// See the documentation for `MaxUnpool3dImpl` class to learn what methods it
/// provides, and examples of how to use `MaxUnpool3d` with
/// `smith::nn::MaxUnpool3dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(MaxUnpool3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FractionalMaxPool2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies fractional maxpool over a 2-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.FractionalMaxPool2d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::FractionalMaxPool2dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// FractionalMaxPool2d model(FractionalMaxPool2dOptions(5).output_size(1));
/// ```
class SMITH_API FractionalMaxPool2dImpl
    : public smith::nn::Cloneable<FractionalMaxPool2dImpl> {
 public:
  FractionalMaxPool2dImpl(ExpandingArray<2> kernel_size)
      : FractionalMaxPool2dImpl(FractionalMaxPool2dOptions(kernel_size)) {}
  explicit FractionalMaxPool2dImpl(FractionalMaxPool2dOptions options_);

  void reset() override;

  /// Pretty prints the `FractionalMaxPool2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `smith::nn::MaxUnpool2d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);

  /// The options with which this `Module` was constructed.
  FractionalMaxPool2dOptions options;

  Tensor _random_samples;
};

/// A `ModuleHolder` subclass for `FractionalMaxPool2dImpl`.
/// See the documentation for `FractionalMaxPool2dImpl` class to learn what
/// methods it provides, and examples of how to use `FractionalMaxPool2d` with
/// `smith::nn::FractionalMaxPool2dOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(FractionalMaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FractionalMaxPool3d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies fractional maxpool over a 3-D input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.FractionalMaxPool3d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::FractionalMaxPool3dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// FractionalMaxPool3d model(FractionalMaxPool3dOptions(5).output_size(1));
/// ```
class SMITH_API FractionalMaxPool3dImpl
    : public smith::nn::Cloneable<FractionalMaxPool3dImpl> {
 public:
  FractionalMaxPool3dImpl(ExpandingArray<3> kernel_size)
      : FractionalMaxPool3dImpl(FractionalMaxPool3dOptions(kernel_size)) {}
  explicit FractionalMaxPool3dImpl(FractionalMaxPool3dOptions options_);

  void reset() override;

  /// Pretty prints the `FractionalMaxPool3d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `smith::nn::MaxUnpool3d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);

  /// The options with which this `Module` was constructed.
  FractionalMaxPool3dOptions options;

  Tensor _random_samples;
};

/// A `ModuleHolder` subclass for `FractionalMaxPool3dImpl`.
/// See the documentation for `FractionalMaxPool3dImpl` class to learn what
/// methods it provides, and examples of how to use `FractionalMaxPool3d` with
/// `smith::nn::FractionalMaxPool3dOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(FractionalMaxPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) lppool modules.
template <size_t D, typename Derived>
class SMITH_API LPPoolImpl : public smith::nn::Cloneable<Derived> {
 public:
  LPPoolImpl(double norm_type, ExpandingArray<D> kernel_size)
      : LPPoolImpl(LPPoolOptions<D>(norm_type, kernel_size)) {}
  explicit LPPoolImpl(const LPPoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `LPPool{1,2}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  LPPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LPPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LPPool1d function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.LPPool1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::LPPool1dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LPPool1d model(LPPool1dOptions(1, 2).stride(5).ceil_mode(true));
/// ```
class SMITH_API LPPool1dImpl : public LPPoolImpl<1, LPPool1dImpl> {
 public:
  using LPPoolImpl<1, LPPool1dImpl>::LPPoolImpl;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `LPPool1dImpl`.
/// See the documentation for `LPPool1dImpl` class to learn what methods it
/// provides, and examples of how to use `LPPool1d` with
/// `smith::nn::LPPool1dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(LPPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LPPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LPPool2d function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.LPPool2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::LPPool2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LPPool2d model(LPPool2dOptions(1, std::vector<int64_t>({3, 4})).stride({5,
/// 6}).ceil_mode(true));
/// ```
class SMITH_API LPPool2dImpl : public LPPoolImpl<2, LPPool2dImpl> {
 public:
  using LPPoolImpl<2, LPPool2dImpl>::LPPoolImpl;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `LPPool2dImpl`.
/// See the documentation for `LPPool2dImpl` class to learn what methods it
/// provides, and examples of how to use `LPPool2d` with
/// `smith::nn::LPPool2dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(LPPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LPPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LPPool3d function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.LPPool3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::LPPool3dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LPPool3d model(LPPool3dOptions(1, std::vector<int64_t>({3, 4, 5})).stride(
/// {5, 6, 7}).ceil_mode(true));
/// ```
class SMITH_API LPPool3dImpl : public LPPoolImpl<3, LPPool3dImpl> {
 public:
  using LPPoolImpl<3, LPPool3dImpl>::LPPoolImpl;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `LPPool3dImpl`.
/// See the documentation for `LPPool3dImpl` class to learn what methods it
/// provides, and examples of how to use `LPPool3d` with
/// `smith::nn::LPPool3dOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(LPPool3d);

} // namespace smith::nn
