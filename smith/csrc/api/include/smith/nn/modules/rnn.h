#pragma once

#include <smith/nn/cloneable.h>
#include <smith/nn/modules/common.h>
#include <smith/nn/modules/dropout.h>
#include <smith/nn/options/rnn.h>
#include <smith/nn/pimpl.h>
#include <smith/nn/utils/rnn.h>
#include <smith/types.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace smith::nn {

namespace detail {
/// Base class for all RNN implementations (intended for code sharing).
template <typename Derived>
class SMITH_API RNNImplBase : public smith::nn::Cloneable<Derived> {
 public:
  explicit RNNImplBase(const RNNOptionsBase& options_);

  /// Initializes the parameters of the RNN module.
  void reset() override;

  void reset_parameters();

  /// Overrides `nn::Module::to()` to call `flatten_parameters()` after the
  /// original operation.
  void to(smith::Device device, smith::Dtype dtype, bool non_blocking = false)
      override;
  void to(smith::Dtype dtype, bool non_blocking = false) override;
  void to(smith::Device device, bool non_blocking = false) override;

  /// Pretty prints the RNN module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Modifies the internal storage of weights for optimization purposes.
  ///
  /// On CPU, this method should be called if any of the weight or bias vectors
  /// are changed (i.e. weights are added or removed). On GPU, it should be
  /// called __any time the storage of any parameter is modified__, e.g. any
  /// time a parameter is assigned a new value. This allows using the fast path
  /// in cuDNN implementations of respective RNN `forward()` methods. It is
  /// called once upon construction, inside `reset()`.
  void flatten_parameters();

  std::vector<Tensor> all_weights() const;

  /// The RNN's options.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  RNNOptionsBase options_base;

 protected:
  // Resets flat_weights_
  // Note: be v. careful before removing this, as 3rd party device types
  // likely rely on this behavior to properly .to() modules like LSTM.
  void reset_flat_weights();

  void check_input(const Tensor& input, const Tensor& batch_sizes) const;

  std::tuple<int64_t, int64_t, int64_t> get_expected_hidden_size(
      const Tensor& input,
      const Tensor& batch_sizes) const;

  void check_hidden_size(
      const Tensor& hx,
      std::tuple<int64_t, int64_t, int64_t> expected_hidden_size,
      std::string msg = "Expected hidden size {1}, got {2}") const;

  void check_forward_args(Tensor input, Tensor hidden, Tensor batch_sizes)
      const;

  Tensor permute_hidden(Tensor hx, const Tensor& permutation) const;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::string> flat_weights_names_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::vector<std::string>> all_weights_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<Tensor> flat_weights_;
};
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A multi-layer Elman RNN module with Tanh or ReLU activation.
/// See https://blacksmith.org/docs/main/generated/smith.nn.RNN.html to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::RNNOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// RNN model(RNNOptions(128,
/// 64).num_layers(3).dropout(0.2).nonlinearity(smith::kTanh));
/// ```
class SMITH_API RNNImpl : public detail::RNNImplBase<RNNImpl> {
 public:
  RNNImpl(int64_t input_size, int64_t hidden_size)
      : RNNImpl(RNNOptions(input_size, hidden_size)) {}
  explicit RNNImpl(const RNNOptions& options_);

  std::tuple<Tensor, Tensor> forward(const Tensor& input, Tensor hx = {});

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})

 public:
  std::tuple<smith::nn::utils::rnn::PackedSequence, Tensor>
  forward_with_packed_input(
      const smith::nn::utils::rnn::PackedSequence& packed_input,
      Tensor hx = {});

  RNNOptions options;

 protected:
  std::tuple<Tensor, Tensor> forward_helper(
      const Tensor& input,
      const Tensor& batch_sizes,
      const Tensor& sorted_indices,
      int64_t max_batch_size,
      Tensor hx);
};

/// A `ModuleHolder` subclass for `RNNImpl`.
/// See the documentation for `RNNImpl` class to learn what methods it
/// provides, and examples of how to use `RNN` with `smith::nn::RNNOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(RNN);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A multi-layer long-short-term-memory (LSTM) module.
/// See https://blacksmith.org/docs/main/generated/smith.nn.LSTM.html to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::LSTMOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LSTM model(LSTMOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```
class SMITH_API LSTMImpl : public detail::RNNImplBase<LSTMImpl> {
 public:
  LSTMImpl(int64_t input_size, int64_t hidden_size)
      : LSTMImpl(LSTMOptions(input_size, hidden_size)) {}
  explicit LSTMImpl(const LSTMOptions& options_);

  std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward(
      const Tensor& input,
      std::optional<std::tuple<Tensor, Tensor>> hx_opt = {});

 protected:
  FORWARD_HAS_DEFAULT_ARGS(
      {1, AnyValue(std::optional<std::tuple<Tensor, Tensor>>())})

 public:
  std::tuple<smith::nn::utils::rnn::PackedSequence, std::tuple<Tensor, Tensor>>
  forward_with_packed_input(
      const smith::nn::utils::rnn::PackedSequence& packed_input,
      std::optional<std::tuple<Tensor, Tensor>> hx_opt = {});

  LSTMOptions options;

 protected:
  void check_forward_args(
      const Tensor& input,
      std::tuple<Tensor, Tensor> hidden,
      const Tensor& batch_sizes) const;

  std::tuple<int64_t, int64_t, int64_t> get_expected_cell_size(
      const Tensor& input,
      const Tensor& batch_sizes) const;

  std::tuple<Tensor, Tensor> permute_hidden(
      std::tuple<Tensor, Tensor> hx,
      const Tensor& permutation) const;

  std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_helper(
      const Tensor& input,
      const Tensor& batch_sizes,
      const Tensor& sorted_indices,
      int64_t max_batch_size,
      std::optional<std::tuple<Tensor, Tensor>> hx_opt);
};

/// A `ModuleHolder` subclass for `LSTMImpl`.
/// See the documentation for `LSTMImpl` class to learn what methods it
/// provides, and examples of how to use `LSTM` with `smith::nn::LSTMOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(LSTM);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A multi-layer gated recurrent unit (GRU) module.
/// See https://blacksmith.org/docs/main/generated/smith.nn.GRU.html to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::GRUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// GRU model(GRUOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```
class SMITH_API GRUImpl : public detail::RNNImplBase<GRUImpl> {
 public:
  GRUImpl(int64_t input_size, int64_t hidden_size)
      : GRUImpl(GRUOptions(input_size, hidden_size)) {}
  explicit GRUImpl(const GRUOptions& options_);

  std::tuple<Tensor, Tensor> forward(const Tensor& input, Tensor hx = {});

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(smith::Tensor())})

 public:
  std::tuple<smith::nn::utils::rnn::PackedSequence, Tensor>
  forward_with_packed_input(
      const smith::nn::utils::rnn::PackedSequence& packed_input,
      Tensor hx = {});

  GRUOptions options;

 protected:
  std::tuple<Tensor, Tensor> forward_helper(
      const Tensor& input,
      const Tensor& batch_sizes,
      const Tensor& sorted_indices,
      int64_t max_batch_size,
      Tensor hx);
};

/// A `ModuleHolder` subclass for `GRUImpl`.
/// See the documentation for `GRUImpl` class to learn what methods it
/// provides, and examples of how to use `GRU` with `smith::nn::GRUOptions`.
/// See the documentation for `ModuleHolder` to learn about Blacksmith's
/// module storage semantics.
SMITH_MODULE(GRU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCellImplBase
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
/// Base class for all RNNCell implementations (intended for code sharing).
template <typename Derived>
class SMITH_API RNNCellImplBase : public smith::nn::Cloneable<Derived> {
 public:
  explicit RNNCellImplBase(const RNNCellOptionsBase& options_);

  /// Initializes the parameters of the RNNCell module.
  void reset() override;

  void reset_parameters();

  /// Pretty prints the RNN module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  RNNCellOptionsBase options_base;

  Tensor weight_ih;
  Tensor weight_hh;
  Tensor bias_ih;
  Tensor bias_hh;

 protected:
  void check_forward_input(const Tensor& input, const std::string& name) const;
  virtual std::string get_nonlinearity_str() const;
};
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// An Elman RNN cell with tanh or ReLU non-linearity.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.RNNCell to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::RNNCellOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// RNNCell model(RNNCellOptions(20,
/// 10).bias(false).nonlinearity(smith::kReLU));
/// ```
class SMITH_API RNNCellImpl : public detail::RNNCellImplBase<RNNCellImpl> {
 public:
  RNNCellImpl(int64_t input_size, int64_t hidden_size)
      : RNNCellImpl(RNNCellOptions(input_size, hidden_size)) {}
  explicit RNNCellImpl(const RNNCellOptions& options_);

  Tensor forward(const Tensor& input, const Tensor& hx = {});

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})

 public:
  RNNCellOptions options;

 protected:
  std::string get_nonlinearity_str() const override;
};

/// A `ModuleHolder` subclass for `RNNCellImpl`.
/// See the documentation for `RNNCellImpl` class to learn what methods it
/// provides, and examples of how to use `RNNCell` with
/// `smith::nn::RNNCellOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(RNNCell);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTMCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A long short-term memory (LSTM) cell.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.LSTMCell to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::LSTMCellOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LSTMCell model(LSTMCellOptions(20, 10).bias(false));
/// ```
class SMITH_API LSTMCellImpl : public detail::RNNCellImplBase<LSTMCellImpl> {
 public:
  LSTMCellImpl(int64_t input_size, int64_t hidden_size)
      : LSTMCellImpl(LSTMCellOptions(input_size, hidden_size)) {}
  explicit LSTMCellImpl(const LSTMCellOptions& options_);

  std::tuple<Tensor, Tensor> forward(
      const Tensor& input,
      std::optional<std::tuple<Tensor, Tensor>> hx_opt = {});

 protected:
  FORWARD_HAS_DEFAULT_ARGS(
      {1, AnyValue(std::optional<std::tuple<Tensor, Tensor>>())})

 public:
  LSTMCellOptions options;
};

/// A `ModuleHolder` subclass for `LSTMCellImpl`.
/// See the documentation for `LSTMCellImpl` class to learn what methods it
/// provides, and examples of how to use `LSTMCell` with
/// `smith::nn::LSTMCellOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(LSTMCell);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRUCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A gated recurrent unit (GRU) cell.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.GRUCell to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::GRUCellOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// GRUCell model(GRUCellOptions(20, 10).bias(false));
/// ```
class SMITH_API GRUCellImpl : public detail::RNNCellImplBase<GRUCellImpl> {
 public:
  GRUCellImpl(int64_t input_size, int64_t hidden_size)
      : GRUCellImpl(GRUCellOptions(input_size, hidden_size)) {}
  explicit GRUCellImpl(const GRUCellOptions& options_);

  Tensor forward(const Tensor& input, const Tensor& hx = {});

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})

 public:
  GRUCellOptions options;
};

/// A `ModuleHolder` subclass for `GRUCellImpl`.
/// See the documentation for `GRUCellImpl` class to learn what methods it
/// provides, and examples of how to use `GRUCell` with
/// `smith::nn::GRUCellOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
SMITH_MODULE(GRUCell);

} // namespace smith::nn
