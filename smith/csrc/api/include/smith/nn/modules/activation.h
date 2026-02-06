#pragma once

#include <smith/nn/cloneable.h>
#include <smith/nn/functional/activation.h>
#include <smith/nn/modules/common.h>
#include <smith/nn/modules/linear.h>
#include <smith/nn/options/activation.h>

#include <smith/csrc/Export.h>

namespace smith::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies elu over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.ELU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::ELUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ELU model(ELUOptions().alpha(42.42).inplace(true));
/// ```
class SMITH_API ELUImpl : public smith::nn::Cloneable<ELUImpl> {
 public:
  explicit ELUImpl(const ELUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `ELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ELUOptions options;
};

/// A `ModuleHolder` subclass for `ELUImpl`.
/// See the documentation for `ELUImpl` class to learn what methods it
/// provides, and examples of how to use `ELU` with `smith::nn::ELUOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(ELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the selu function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.SELU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::SELUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// SELU model(SELUOptions().inplace(true));
/// ```
class SMITH_API SELUImpl : public smith::nn::Cloneable<SELUImpl> {
 public:
  explicit SELUImpl(const SELUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `SELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  SELUOptions options;
};

/// A `ModuleHolder` subclass for `SELUImpl`.
/// See the documentation for `SELUImpl` class to learn what methods it
/// provides, and examples of how to use `SELU` with `smith::nn::SELUOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(SELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the hard shrinkage function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Hardshrink to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::HardshrinkOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Hardshrink model(HardshrinkOptions().lambda(42.42));
/// ```
class SMITH_API HardshrinkImpl : public smith::nn::Cloneable<HardshrinkImpl> {
 public:
  explicit HardshrinkImpl(const HardshrinkOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Hardshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  HardshrinkOptions options;
};

/// A `ModuleHolder` subclass for `HardshrinkImpl`.
/// See the documentation for `HardshrinkImpl` class to learn what methods it
/// provides, and examples of how to use `Hardshrink` with
/// `smith::nn::HardshrinkOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(Hardshrink);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardtanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the HardTanh function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Hardtanh to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::HardtanhOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Hardtanh
/// model(HardtanhOptions().min_val(-42.42).max_val(0.42).inplace(true));
/// ```
class SMITH_API HardtanhImpl : public smith::nn::Cloneable<HardtanhImpl> {
 public:
  explicit HardtanhImpl(const HardtanhOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `Hardtanh` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  HardtanhOptions options;
};

/// A `ModuleHolder` subclass for `HardtanhImpl`.
/// See the documentation for `HardtanhImpl` class to learn what methods it
/// provides, and examples of how to use `Hardtanh` with
/// `smith::nn::HardtanhOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(Hardtanh);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LeakyReLU function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.LeakyReLU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::LeakyReLUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LeakyReLU model(LeakyReLUOptions().negative_slope(0.42).inplace(true));
/// ```
class SMITH_API LeakyReLUImpl : public smith::nn::Cloneable<LeakyReLUImpl> {
 public:
  explicit LeakyReLUImpl(const LeakyReLUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `LeakyReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  LeakyReLUOptions options;
};

/// A `ModuleHolder` subclass for `LeakyReLUImpl`.
/// See the documentation for `LeakyReLUImpl` class to learn what methods it
/// provides, and examples of how to use `LeakyReLU` with
/// `smith::nn::LeakyReLUOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(LeakyReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LogSigmoid function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.LogSigmoid to learn
/// about the exact behavior of this module.
class SMITH_API LogSigmoidImpl : public smith::nn::Cloneable<LogSigmoidImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `LogSigmoid` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `LogSigmoidImpl`.
/// See the documentation for `LogSigmoidImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(LogSigmoid);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Softmax function.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Softmax to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::SoftmaxOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Softmax model(SoftmaxOptions(1));
/// ```
class SMITH_API SoftmaxImpl : public smith::nn::Cloneable<SoftmaxImpl> {
 public:
  explicit SoftmaxImpl(int64_t dim) : SoftmaxImpl(SoftmaxOptions(dim)) {}
  explicit SoftmaxImpl(const SoftmaxOptions& options_);

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softmax` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  SoftmaxOptions options;
};

/// A `ModuleHolder` subclass for `SoftmaxImpl`.
/// See the documentation for `SoftmaxImpl` class to learn what methods it
/// provides, and examples of how to use `Softmax` with
/// `smith::nn::SoftmaxOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(Softmax);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Softmin function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Softmin to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::SoftminOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Softmin model(SoftminOptions(1));
/// ```
class SMITH_API SoftminImpl : public smith::nn::Cloneable<SoftminImpl> {
 public:
  explicit SoftminImpl(int64_t dim) : SoftminImpl(SoftminOptions(dim)) {}
  explicit SoftminImpl(const SoftminOptions& options_);

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softmin` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  SoftminOptions options;
};

/// A `ModuleHolder` subclass for `SoftminImpl`.
/// See the documentation for `SoftminImpl` class to learn what methods it
/// provides, and examples of how to use `Softmin` with
/// `smith::nn::SoftminOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(Softmin);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSoftmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LogSoftmax function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.LogSoftmax to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::LogSoftmaxOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LogSoftmax model(LogSoftmaxOptions(1));
/// ```
class SMITH_API LogSoftmaxImpl : public smith::nn::Cloneable<LogSoftmaxImpl> {
 public:
  explicit LogSoftmaxImpl(int64_t dim)
      : LogSoftmaxImpl(LogSoftmaxOptions(dim)) {}
  explicit LogSoftmaxImpl(const LogSoftmaxOptions& options_);

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `LogSoftmax` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  LogSoftmaxOptions options;
};

/// A `ModuleHolder` subclass for `LogSoftmaxImpl`.
/// See the documentation for `LogSoftmaxImpl` class to learn what methods it
/// provides, and examples of how to use `LogSoftmax` with
/// `smith::nn::LogSoftmaxOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(LogSoftmax);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Softmax2d function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Softmax2d to learn
/// about the exact behavior of this module.
class SMITH_API Softmax2dImpl : public smith::nn::Cloneable<Softmax2dImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softmax2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `Softmax2dImpl`.
/// See the documentation for `Softmax2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(Softmax2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the PReLU function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.PReLU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::PReLUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PReLU model(PReLUOptions().num_parameters(42));
/// ```
class SMITH_API PReLUImpl : public smith::nn::Cloneable<PReLUImpl> {
 public:
  explicit PReLUImpl(const PReLUOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `PReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  PReLUOptions options;

  /// The learned weight.
  Tensor weight;
};

/// A `ModuleHolder` subclass for `PReLUImpl`.
/// See the documentation for `PReLUImpl` class to learn what methods it
/// provides, and examples of how to use `PReLU` with `smith::nn::PReLUOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(PReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ReLU function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.ReLU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::ReLUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ReLU model(ReLUOptions().inplace(true));
/// ```
class SMITH_API ReLUImpl : public smith::nn::Cloneable<ReLUImpl> {
 public:
  explicit ReLUImpl(const ReLUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `ReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReLUOptions options;
};

/// A `ModuleHolder` subclass for `ReLUImpl`.
/// See the documentation for `ReLUImpl` class to learn what methods it
/// provides, and examples of how to use `ReLU` with `smith::nn::ReLUOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(ReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU6 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ReLU6 function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.ReLU6 to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::ReLU6Options` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ReLU6 model(ReLU6Options().inplace(true));
/// ```
class SMITH_API ReLU6Impl : public smith::nn::Cloneable<ReLU6Impl> {
 public:
  explicit ReLU6Impl(const ReLU6Options& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `ReLU6` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReLU6Options options;
};

/// A `ModuleHolder` subclass for `ReLU6Impl`.
/// See the documentation for `ReLU6Impl` class to learn what methods it
/// provides, and examples of how to use `ReLU6` with `smith::nn::ReLU6Options`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(ReLU6);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the RReLU function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.RReLU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::RReLUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// RReLU model(RReLUOptions().lower(0.24).upper(0.42).inplace(true));
/// ```
class SMITH_API RReLUImpl : public smith::nn::Cloneable<RReLUImpl> {
 public:
  explicit RReLUImpl(const RReLUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `RReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  RReLUOptions options;
};

/// A `ModuleHolder` subclass for `RReLUImpl`.
/// See the documentation for `RReLUImpl` class to learn what methods it
/// provides, and examples of how to use `RReLU` with `smith::nn::RReLUOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(RReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies celu over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.CELU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::CELUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CELU model(CELUOptions().alpha(42.42).inplace(true));
/// ```
class SMITH_API CELUImpl : public smith::nn::Cloneable<CELUImpl> {
 public:
  explicit CELUImpl(const CELUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `CELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  CELUOptions options;
};

/// A `ModuleHolder` subclass for `CELUImpl`.
/// See the documentation for `CELUImpl` class to learn what methods it
/// provides, and examples of how to use `CELU` with `smith::nn::CELUOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(CELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies glu over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.GLU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::GLUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// GLU model(GLUOptions(1));
/// ```
class SMITH_API GLUImpl : public smith::nn::Cloneable<GLUImpl> {
 public:
  explicit GLUImpl(const GLUOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `GLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  GLUOptions options;
};

/// A `ModuleHolder` subclass for `GLUImpl`.
/// See the documentation for `GLUImpl` class to learn what methods it
/// provides, and examples of how to use `GLU` with `smith::nn::GLUOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(GLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies gelu over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.GELU to learn
/// about the exact behavior of this module.
class SMITH_API GELUImpl : public smith::nn::Cloneable<GELUImpl> {
 public:
  explicit GELUImpl(GELUOptions options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `GELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  GELUOptions options;
};

/// A `ModuleHolder` subclass for `GELUImpl`.
/// See the documentation for `GELUImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(GELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SiLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies silu over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.SiLU to learn
/// about the exact behavior of this module.
class SMITH_API SiLUImpl : public smith::nn::Cloneable<SiLUImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `SiLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `SiLUImpl`.
/// See the documentation for `SiLUImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(SiLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mish ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies mish over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Mish to learn
/// about the exact behavior of this module.
class SMITH_API MishImpl : public smith::nn::Cloneable<MishImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Mish` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `MishImpl`.
/// See the documentation for `MishImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(Mish);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies sigmoid over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Sigmoid to learn
/// about the exact behavior of this module.
class SMITH_API SigmoidImpl : public smith::nn::Cloneable<SigmoidImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Sigmoid` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `SigmoidImpl`.
/// See the documentation for `SigmoidImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(Sigmoid);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softplus ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies softplus over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Softplus to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::SoftplusOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Softplus model(SoftplusOptions().beta(0.24).threshold(42.42));
/// ```
class SMITH_API SoftplusImpl : public smith::nn::Cloneable<SoftplusImpl> {
 public:
  explicit SoftplusImpl(const SoftplusOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softplus` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  SoftplusOptions options;
};

/// A `ModuleHolder` subclass for `SoftplusImpl`.
/// See the documentation for `SoftplusImpl` class to learn what methods it
/// provides, and examples of how to use `Softplus` with
/// `smith::nn::SoftplusOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(Softplus);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the soft shrinkage function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Softshrink to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::SoftshrinkOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Softshrink model(SoftshrinkOptions(42.42));
/// ```
class SMITH_API SoftshrinkImpl : public smith::nn::Cloneable<SoftshrinkImpl> {
 public:
  explicit SoftshrinkImpl(const SoftshrinkOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  SoftshrinkOptions options;
};

/// A `ModuleHolder` subclass for `SoftshrinkImpl`.
/// See the documentation for `SoftshrinkImpl` class to learn what methods it
/// provides, and examples of how to use `Softshrink` with
/// `smith::nn::SoftshrinkOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(Softshrink);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softsign ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Softsign over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Softsign to learn
/// about the exact behavior of this module.
class SMITH_API SoftsignImpl : public smith::nn::Cloneable<SoftsignImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softsign` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `SoftsignImpl`.
/// See the documentation for `SoftsignImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(Softsign);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Tanh over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Tanh to learn
/// about the exact behavior of this module.
class SMITH_API TanhImpl : public smith::nn::Cloneable<TanhImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Tanh` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `TanhImpl`.
/// See the documentation for `TanhImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(Tanh);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tanhshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Tanhshrink over a given input.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Tanhshrink to learn
/// about the exact behavior of this module.
class SMITH_API TanhshrinkImpl : public smith::nn::Cloneable<TanhshrinkImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Tanhshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `TanhshrinkImpl`.
/// See the documentation for `TanhshrinkImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(Tanhshrink);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Threshold ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Threshold function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Threshold to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::ThresholdOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Threshold model(ThresholdOptions(42.42, 24.24).inplace(true));
/// ```
class SMITH_API ThresholdImpl : public smith::nn::Cloneable<ThresholdImpl> {
 public:
  ThresholdImpl(double threshold, double value)
      : ThresholdImpl(ThresholdOptions(threshold, value)) {}
  explicit ThresholdImpl(const ThresholdOptions& options_);

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `Threshold` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ThresholdOptions options;
};

/// A `ModuleHolder` subclass for `ThresholdImpl`.
/// See the documentation for `ThresholdImpl` class to learn what methods it
/// provides, and examples of how to use `Threshold` with
/// `smith::nn::ThresholdOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(Threshold);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiheadAttention ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the MultiheadAttention function element-wise.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.MultiheadAttention
/// to learn about the exact behavior of this module.
///
/// See the documentation for `smith::nn::MultiheadAttentionOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MultiheadAttention model(MultiheadAttentionOptions(20, 10).bias(false));
/// ```
class SMITH_API MultiheadAttentionImpl
    : public smith::nn::Cloneable<MultiheadAttentionImpl> {
 public:
  MultiheadAttentionImpl(int64_t embed_dim, int64_t num_heads)
      : MultiheadAttentionImpl(
            MultiheadAttentionOptions(embed_dim, num_heads)) {}
  explicit MultiheadAttentionImpl(const MultiheadAttentionOptions& options_);

  std::tuple<Tensor, Tensor> forward(
      const Tensor& query,
      const Tensor& key,
      const Tensor& value,
      const Tensor& key_padding_mask = {},
      bool need_weights = true,
      const Tensor& attn_mask = {},
      bool average_attn_weights = true);

 protected:
  FORWARD_HAS_DEFAULT_ARGS(
      {3, AnyValue(Tensor())},
      {4, AnyValue(true)},
      {5, AnyValue(Tensor())},
      {6, AnyValue(true)})

 public:
  void reset() override;

  void _reset_parameters();

  /// The options with which this `Module` was constructed.
  MultiheadAttentionOptions options;

  bool _qkv_same_embed_dim{};
  Tensor in_proj_weight;
  Tensor in_proj_bias;
  Tensor bias_k;
  Tensor bias_v;
  Linear out_proj = nullptr;
  Tensor q_proj_weight;
  Tensor k_proj_weight;
  Tensor v_proj_weight;
  int64_t head_dim{};
};

/// A `ModuleHolder` subclass for `MultiheadAttentionImpl`.
/// See the documentation for `MultiheadAttentionImpl` class to learn what
/// methods it provides, and examples of how to use `MultiheadAttention` with
/// `smith::nn::MultiheadAttentionOptions`. See the documentation for
/// `ModuleHolder` to learn about Blacksmith's module storage semantics.
SMITH_MODULE(MultiheadAttention);

} // namespace smith::nn
