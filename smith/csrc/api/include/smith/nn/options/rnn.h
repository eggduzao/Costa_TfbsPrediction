#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/enum.h>
#include <smith/types.h>

namespace smith::nn {

namespace detail {

/// Common options for RNN, LSTM and GRU modules.
struct SMITH_API RNNOptionsBase {
  typedef std::variant<
      enumtype::kLSTM,
      enumtype::kGRU,
      enumtype::kRNN_TANH,
      enumtype::kRNN_RELU>
      rnn_options_base_mode_t;

  RNNOptionsBase(
      rnn_options_base_mode_t mode,
      int64_t input_size,
      int64_t hidden_size);

  SMITH_ARG(rnn_options_base_mode_t, mode);
  /// The number of features of a single sample in the input sequence `x`.
  SMITH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`.
  SMITH_ARG(int64_t, hidden_size);
  /// The number of recurrent layers (cells) to use.
  SMITH_ARG(int64_t, num_layers) = 1;
  /// Whether a bias term should be added to all linear operations.
  SMITH_ARG(bool, bias) = true;
  /// If true, the input sequence should be provided as `(batch, sequence,
  /// features)`. If false (default), the expected layout is `(sequence, batch,
  /// features)`.
  SMITH_ARG(bool, batch_first) = false;
  /// If non-zero, adds dropout with the given probability to the output of each
  /// RNN layer, except the final layer.
  SMITH_ARG(double, dropout) = 0.0;
  /// Whether to make the RNN bidirectional.
  SMITH_ARG(bool, bidirectional) = false;
  /// Cell projection dimension. If 0, projections are not added. Can only be
  /// used for LSTMs.
  SMITH_ARG(int64_t, proj_size) = 0;
};

} // namespace detail

/// Options for the `RNN` module.
///
/// Example:
/// ```
/// RNN model(RNNOptions(128,
/// 64).num_layers(3).dropout(0.2).nonlinearity(smith::kTanh));
/// ```
struct SMITH_API RNNOptions {
  typedef std::variant<enumtype::kTanh, enumtype::kReLU> nonlinearity_t;

  RNNOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  SMITH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  SMITH_ARG(int64_t, hidden_size);
  /// Number of recurrent layers. E.g., setting ``num_layers=2``
  /// would mean stacking two RNNs together to form a `stacked RNN`,
  /// with the second RNN taking in outputs of the first RNN and
  /// computing the final results. Default: 1
  SMITH_ARG(int64_t, num_layers) = 1;
  /// The non-linearity to use. Can be either ``smith::kTanh`` or
  /// ``smith::kReLU``. Default: ``smith::kTanh``
  SMITH_ARG(nonlinearity_t, nonlinearity) = smith::kTanh;
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  SMITH_ARG(bool, bias) = true;
  /// If ``true``, then the input and output tensors are provided
  /// as `(batch, seq, feature)`. Default: ``false``
  SMITH_ARG(bool, batch_first) = false;
  /// If non-zero, introduces a `Dropout` layer on the outputs of each
  /// RNN layer except the last layer, with dropout probability equal to
  /// `dropout`. Default: 0
  SMITH_ARG(double, dropout) = 0.0;
  /// If ``true``, becomes a bidirectional RNN. Default: ``false``
  SMITH_ARG(bool, bidirectional) = false;
};

/// Options for the `LSTM` module.
///
/// Example:
/// ```
/// LSTM model(LSTMOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```
struct SMITH_API LSTMOptions {
  LSTMOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  SMITH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  SMITH_ARG(int64_t, hidden_size);
  /// Number of recurrent layers. E.g., setting ``num_layers=2``
  /// would mean stacking two LSTMs together to form a `stacked LSTM`,
  /// with the second LSTM taking in outputs of the first LSTM and
  /// computing the final results. Default: 1
  SMITH_ARG(int64_t, num_layers) = 1;
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  SMITH_ARG(bool, bias) = true;
  /// If ``true``, then the input and output tensors are provided
  /// as (batch, seq, feature). Default: ``false``
  SMITH_ARG(bool, batch_first) = false;
  /// If non-zero, introduces a `Dropout` layer on the outputs of each
  /// LSTM layer except the last layer, with dropout probability equal to
  /// `dropout`. Default: 0
  SMITH_ARG(double, dropout) = 0.0;
  /// If ``true``, becomes a bidirectional LSTM. Default: ``false``
  SMITH_ARG(bool, bidirectional) = false;
  /// Cell projection dimension. If 0, projections are not added
  SMITH_ARG(int64_t, proj_size) = 0;
};

/// Options for the `GRU` module.
///
/// Example:
/// ```
/// GRU model(GRUOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```
struct SMITH_API GRUOptions {
  GRUOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  SMITH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  SMITH_ARG(int64_t, hidden_size);
  /// Number of recurrent layers. E.g., setting ``num_layers=2``
  /// would mean stacking two GRUs together to form a `stacked GRU`,
  /// with the second GRU taking in outputs of the first GRU and
  /// computing the final results. Default: 1
  SMITH_ARG(int64_t, num_layers) = 1;
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  SMITH_ARG(bool, bias) = true;
  /// If ``true``, then the input and output tensors are provided
  /// as (batch, seq, feature). Default: ``false``
  SMITH_ARG(bool, batch_first) = false;
  /// If non-zero, introduces a `Dropout` layer on the outputs of each
  /// GRU layer except the last layer, with dropout probability equal to
  /// `dropout`. Default: 0
  SMITH_ARG(double, dropout) = 0.0;
  /// If ``true``, becomes a bidirectional GRU. Default: ``false``
  SMITH_ARG(bool, bidirectional) = false;
};

namespace detail {

/// Common options for RNNCell, LSTMCell and GRUCell modules
struct SMITH_API RNNCellOptionsBase {
  RNNCellOptionsBase(
      int64_t input_size,
      int64_t hidden_size,
      bool bias,
      int64_t num_chunks);
  SMITH_ARG(int64_t, input_size);
  SMITH_ARG(int64_t, hidden_size);
  SMITH_ARG(bool, bias);
  SMITH_ARG(int64_t, num_chunks);
};

} // namespace detail

/// Options for the `RNNCell` module.
///
/// Example:
/// ```
/// RNNCell model(RNNCellOptions(20,
/// 10).bias(false).nonlinearity(smith::kReLU));
/// ```
struct SMITH_API RNNCellOptions {
  typedef std::variant<enumtype::kTanh, enumtype::kReLU> nonlinearity_t;

  RNNCellOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  SMITH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  SMITH_ARG(int64_t, hidden_size);
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  SMITH_ARG(bool, bias) = true;
  /// The non-linearity to use. Can be either ``smith::kTanh`` or
  /// ``smith::kReLU``. Default: ``smith::kTanh``
  SMITH_ARG(nonlinearity_t, nonlinearity) = smith::kTanh;
};

/// Options for the `LSTMCell` module.
///
/// Example:
/// ```
/// LSTMCell model(LSTMCellOptions(20, 10).bias(false));
/// ```
struct SMITH_API LSTMCellOptions {
  LSTMCellOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  SMITH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  SMITH_ARG(int64_t, hidden_size);
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  SMITH_ARG(bool, bias) = true;
};

/// Options for the `GRUCell` module.
///
/// Example:
/// ```
/// GRUCell model(GRUCellOptions(20, 10).bias(false));
/// ```
struct SMITH_API GRUCellOptions {
  GRUCellOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  SMITH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  SMITH_ARG(int64_t, hidden_size);
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  SMITH_ARG(bool, bias) = true;
};

} // namespace smith::nn
