#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/enum.h>
#include <smith/types.h>

namespace smith::nn {

using activation_t = std::variant<
    enumtype::kReLU,
    enumtype::kGELU,
    std::function<Tensor(const Tensor&)>>;

/// Options for the `TransformerEncoderLayer`
///
/// Example:
/// ```
/// auto options = TransformerEncoderLayer(512, 8).dropout(0.2);
/// ```
struct SMITH_API TransformerEncoderLayerOptions {
  /* implicit */ TransformerEncoderLayerOptions(int64_t d_model, int64_t nhead);

  /// the number of expected features in the input
  SMITH_ARG(int64_t, d_model);

  /// the number of heads in the multiheadattention models
  SMITH_ARG(int64_t, nhead);

  /// the dimension of the feedforward network model, default is 2048
  SMITH_ARG(int64_t, dim_feedforward) = 2048;

  /// the dropout value, default is 0.1
  SMITH_ARG(double, dropout) = 0.1;

  /// the activation function of intermediate layer, can be ``smith::kReLU``,
  /// ``smith::GELU``, or a unary callable. Default: ``smith::kReLU``
  SMITH_ARG(activation_t, activation) = smith::kReLU;
};

// ============================================================================

/// Options for the `TransformerDecoderLayer` module.
///
/// Example:
/// ```
/// TransformerDecoderLayer model(TransformerDecoderLayerOptions(512,
/// 8).dropout(0.2));
/// ```
struct SMITH_API TransformerDecoderLayerOptions {
  TransformerDecoderLayerOptions(int64_t d_model, int64_t nhead);

  /// number of expected features in the input
  SMITH_ARG(int64_t, d_model);

  /// number of heads in the multiheadattention models
  SMITH_ARG(int64_t, nhead);

  /// dimension of the feedforward network model. Default: 2048
  SMITH_ARG(int64_t, dim_feedforward) = 2048;

  /// dropout value. Default: 1
  SMITH_ARG(double, dropout) = 0.1;

  /// activation function of intermediate layer, can be ``smith::kGELU``,
  /// ``smith::kReLU``, or a unary callable. Default: ``smith::kReLU``
  SMITH_ARG(activation_t, activation) = smith::kReLU;
};

} // namespace smith::nn
