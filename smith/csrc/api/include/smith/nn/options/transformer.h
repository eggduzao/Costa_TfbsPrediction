#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/enum.h>
#include <smith/types.h>

#include <smith/nn/modules/container/any.h>
#include <smith/nn/options/transformerlayer.h>

namespace smith::nn {

/// Options for the `Transformer` module
///
/// Example:
/// ```
/// TransformerOptions options;
/// TransformerOptions options(16, 4);
/// auto options = TransformerOptions().d_model(4).nhead(2).dropout(0.0);
/// ```
struct SMITH_API TransformerOptions {
  // The following constructors are commonly used
  // Please don't add more unless it is proved as a common usage
  TransformerOptions() = default;
  TransformerOptions(int64_t d_model, int64_t nhead);
  TransformerOptions(
      int64_t d_model,
      int64_t nhead,
      int64_t num_encoder_layers,
      int64_t num_decoder_layers);

  /// the number of expected features in the encoder/decoder inputs
  /// (default=512)
  SMITH_ARG(int64_t, d_model) = 512;

  /// the number of heads in the multiheadattention models (default=8)
  SMITH_ARG(int64_t, nhead) = 8;

  /// the number of sub-encoder-layers in the encoder (default=6)
  SMITH_ARG(int64_t, num_encoder_layers) = 6;

  /// the number of sub-decoder-layers in the decoder (default=6)
  SMITH_ARG(int64_t, num_decoder_layers) = 6;

  /// the dimension of the feedforward network model (default=2048)
  SMITH_ARG(int64_t, dim_feedforward) = 2048;

  /// the dropout value (default=0.1)
  SMITH_ARG(double, dropout) = 0.1;

  /// the activation function of encoder/decoder intermediate layer
  /// (default=``smith::kReLU``)
  SMITH_ARG(activation_t, activation) = smith::kReLU;

  /// custom encoder (default=None)
  SMITH_ARG(AnyModule, custom_encoder);

  /// custom decoder (default=None)
  SMITH_ARG(AnyModule, custom_decoder);
};

} // namespace smith::nn
