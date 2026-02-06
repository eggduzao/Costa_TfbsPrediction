#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/enum.h>
#include <smith/types.h>

namespace smith::nn {

/// Options for the `ELU` module.
///
/// Example:
/// ```
/// ELU model(ELUOptions().alpha(42.42).inplace(true));
/// ```
struct SMITH_API ELUOptions {
  /// The `alpha` value for the ELU formulation. Default: 1.0
  SMITH_ARG(double, alpha) = 1.0;

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for `smith::nn::functional::elu`.
///
/// See the documentation for `smith::nn::ELUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::elu(x, F::ELUFuncOptions().alpha(0.42).inplace(true));
/// ```
using ELUFuncOptions = ELUOptions;
} // namespace functional

// ============================================================================

/// Options for the `SELU` module.
///
/// Example:
/// ```
/// SELU model(SELUOptions().inplace(true));
/// ```
struct SMITH_API SELUOptions {
  /* implicit */ SELUOptions(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace);
};

namespace functional {
/// Options for `smith::nn::functional::selu`.
///
/// See the documentation for `smith::nn::SELUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::selu(input, F::SELUFuncOptions(false));
/// ```
using SELUFuncOptions = SELUOptions;
} // namespace functional

// ============================================================================

/// Options for the `GLU` module.
///
/// Example:
/// ```
/// GLU model(GLUOptions(1));
/// ```
struct SMITH_API GLUOptions {
  /* implicit */ GLUOptions(int64_t dim = -1);

  /// the dimension on which to split the input. Default: -1
  SMITH_ARG(int64_t, dim);
};

namespace functional {
/// Options for `smith::nn::functional::glu`.
///
/// See the documentation for `smith::nn::GLUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::glu(input, GLUFuncOptions(1));
/// ```
using GLUFuncOptions = GLUOptions;
} // namespace functional

// ============================================================================

/// Options for the `GELU` module.
///
/// Example:
/// ```
/// GELU model(GELUOptions().approximate("none"));
/// ```
struct SMITH_API GELUOptions {
  /// Specifies the approximation to apply to the output.
  SMITH_ARG(std::string, approximate) = "none";
};

namespace functional {
/// Options for `smith::nn::functional::gelu`.
///
/// See the documentation for `smith::nn::GELUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::gelu(input, F::GELUFuncOptions().approximate("none"));
/// ```
using GELUFuncOptions = GELUOptions;
} // namespace functional

// ============================================================================

/// Options for the `Hardshrink` module.
///
/// Example:
/// ```
/// Hardshrink model(HardshrinkOptions().lambda(42.42));
/// ```
struct SMITH_API HardshrinkOptions {
  /* implicit */ HardshrinkOptions(double lambda = 0.5);

  /// the `lambda` value for the Hardshrink formulation. Default: 0.5
  SMITH_ARG(double, lambda);
};

namespace functional {
/// Options for `smith::nn::functional::hardshrink`.
///
/// See the documentation for `smith::nn::HardshrinkOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::hardshrink(x, F::HardshrinkFuncOptions().lambda(0.42));
/// ```
using HardshrinkFuncOptions = HardshrinkOptions;
} // namespace functional

// ============================================================================

/// Options for the `Hardtanh` module.
///
/// Example:
/// ```
/// Hardtanh
/// model(HardtanhOptions().min_val(-42.42).max_val(0.42).inplace(true));
/// ```
struct SMITH_API HardtanhOptions {
  /// minimum value of the linear region range. Default: -1
  SMITH_ARG(double, min_val) = -1.0;

  /// maximum value of the linear region range. Default: 1
  SMITH_ARG(double, max_val) = 1.0;

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for `smith::nn::functional::hardtanh`.
///
/// See the documentation for `smith::nn::HardtanhOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::hardtanh(x,
/// F::HardtanhFuncOptions().min_val(-1.0).max_val(1.0).inplace(true));
/// ```
using HardtanhFuncOptions = HardtanhOptions;
} // namespace functional

// ============================================================================

/// Options for the `LeakyReLU` module.
///
/// Example:
/// ```
/// LeakyReLU model(LeakyReLUOptions().negative_slope(0.42).inplace(true));
/// ```
struct SMITH_API LeakyReLUOptions {
  /// Controls the angle of the negative slope. Default: 1e-2
  SMITH_ARG(double, negative_slope) = 1e-2;

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for `smith::nn::functional::leaky_relu`.
///
/// See the documentation for `smith::nn::LeakyReLUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::leaky_relu(x,
/// F::LeakyReLUFuncOptions().negative_slope(0.42).inplace(true));
/// ```
using LeakyReLUFuncOptions = LeakyReLUOptions;
} // namespace functional

// ============================================================================

/// Options for the `Softmax` module.
///
/// Example:
/// ```
/// Softmax model(SoftmaxOptions(1));
/// ```
struct SMITH_API SoftmaxOptions {
  SoftmaxOptions(int64_t dim);

  /// Dimension along which Softmax will be computed.
  SMITH_ARG(int64_t, dim);
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::softmax`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::softmax(input, F::SoftmaxFuncOptions(1));
/// ```
struct SMITH_API SoftmaxFuncOptions {
  SoftmaxFuncOptions(int64_t dim);

  /// Dimension along which Softmax will be computed.
  SMITH_ARG(int64_t, dim);

  /// the desired data type of returned tensor.
  /// If specified, the input tensor is casted to `dtype` before the operation
  /// is performed. This is useful for preventing data type overflows. Default:
  /// None.
  SMITH_ARG(std::optional<smith::Dtype>, dtype) = std::nullopt;
};

} // namespace functional

// ============================================================================

/// Options for the `Softmin` module.
///
/// Example:
/// ```
/// Softmin model(SoftminOptions(1));
/// ```
struct SMITH_API SoftminOptions {
  SoftminOptions(int64_t dim);

  /// Dimension along which Softmin will be computed.
  SMITH_ARG(int64_t, dim);
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::softmin`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::softmin(input, F::SoftminFuncOptions(1));
/// ```
struct SMITH_API SoftminFuncOptions {
  SoftminFuncOptions(int64_t dim);

  /// Dimension along which Softmin will be computed.
  SMITH_ARG(int64_t, dim);

  /// the desired data type of returned tensor.
  /// If specified, the input tensor is casted to `dtype` before the operation
  /// is performed. This is useful for preventing data type overflows. Default:
  /// None.
  SMITH_ARG(std::optional<smith::Dtype>, dtype) = std::nullopt;
};

} // namespace functional

// ============================================================================

/// Options for the `LogSoftmax` module.
///
/// Example:
/// ```
/// LogSoftmax model(LogSoftmaxOptions(1));
/// ```
struct SMITH_API LogSoftmaxOptions {
  LogSoftmaxOptions(int64_t dim);

  /// Dimension along which LogSoftmax will be computed.
  SMITH_ARG(int64_t, dim);
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::log_softmax`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::log_softmax(input, LogSoftmaxFuncOptions(1));
/// ```
struct SMITH_API LogSoftmaxFuncOptions {
  LogSoftmaxFuncOptions(int64_t dim);

  /// Dimension along which LogSoftmax will be computed.
  SMITH_ARG(int64_t, dim);

  /// the desired data type of returned tensor.
  /// If specified, the input tensor is casted to `dtype` before the operation
  /// is performed. This is useful for preventing data type overflows. Default:
  /// None.
  SMITH_ARG(std::optional<smith::Dtype>, dtype) = std::nullopt;
};

} // namespace functional

// ============================================================================

/// Options for the `PReLU` module.
///
/// Example:
/// ```
/// PReLU model(PReLUOptions().num_parameters(42));
/// ```
struct SMITH_API PReLUOptions {
  /// number of `a` to learn. Although it takes an int as input, there is only
  /// two values are legitimate: 1, or the number of channels at input. Default:
  /// 1
  SMITH_ARG(int64_t, num_parameters) = 1;

  /// the initial value of `a`. Default: 0.25
  SMITH_ARG(double, init) = 0.25;
};

// ============================================================================

/// Options for the `ReLU` module.
///
/// Example:
/// ```
/// ReLU model(ReLUOptions().inplace(true));
/// ```
struct SMITH_API ReLUOptions {
  /* implicit */ ReLUOptions(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace);
};

namespace functional {
/// Options for `smith::nn::functional::relu`.
///
/// See the documentation for `smith::nn::ReLUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::relu(x, F::ReLUFuncOptions().inplace(true));
/// ```
using ReLUFuncOptions = ReLUOptions;
} // namespace functional

// ============================================================================

/// Options for the `ReLU6` module.
///
/// Example:
/// ```
/// ReLU6 model(ReLU6Options().inplace(true));
/// ```
struct SMITH_API ReLU6Options {
  /* implicit */ ReLU6Options(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace);
};

namespace functional {
/// Options for `smith::nn::functional::relu6`.
///
/// See the documentation for `smith::nn::ReLU6Options` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::relu6(x, F::ReLU6FuncOptions().inplace(true));
/// ```
using ReLU6FuncOptions = ReLU6Options;
} // namespace functional

// ============================================================================

/// Options for the `RReLU` module.
///
/// Example:
/// ```
/// RReLU model(RReLUOptions().lower(0.24).upper(0.42).inplace(true));
/// ```
struct SMITH_API RReLUOptions {
  /// lower bound of the uniform distribution. Default: 1/8
  SMITH_ARG(double, lower) = 1.0 / 8.0;

  /// upper bound of the uniform distribution. Default: 1/3
  SMITH_ARG(double, upper) = 1.0 / 3.0;

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace) = false;
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::rrelu`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::rrelu(x, F::RReLUFuncOptions().lower(0.1).upper(0.4).inplace(true));
/// ```
struct SMITH_API RReLUFuncOptions {
  /// lower bound of the uniform distribution. Default: 1/8
  SMITH_ARG(double, lower) = 1.0 / 8.0;

  /// upper bound of the uniform distribution. Default: 1/3
  SMITH_ARG(double, upper) = 1.0 / 3.0;

  SMITH_ARG(bool, training) = false;

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace) = false;
};

} // namespace functional

// ============================================================================

/// Options for the `CELU` module.
///
/// Example:
/// ```
/// CELU model(CELUOptions().alpha(42.42).inplace(true));
/// ```
struct SMITH_API CELUOptions {
  /// The `alpha` value for the CELU formulation. Default: 1.0
  SMITH_ARG(double, alpha) = 1.0;

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for `smith::nn::functional::celu`.
///
/// See the documentation for `smith::nn::CELUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::celu(x, F::CELUFuncOptions().alpha(0.42).inplace(true));
/// ```
using CELUFuncOptions = CELUOptions;
} // namespace functional

// ============================================================================

/// Options for the `Softplus` module.
///
/// Example:
/// ```
/// Softplus model(SoftplusOptions().beta(0.24).threshold(42.42));
/// ```
struct SMITH_API SoftplusOptions {
  /// the `beta` value for the Softplus formulation. Default: 1
  SMITH_ARG(double, beta) = 1.0;

  /// values above this revert to a linear function. Default: 20
  SMITH_ARG(double, threshold) = 20.0;
};

namespace functional {
/// Options for `smith::nn::functional::softplus`.
///
/// See the documentation for `smith::nn::SoftplusOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::softplus(x, F::SoftplusFuncOptions().beta(0.5).threshold(3.0));
/// ```
using SoftplusFuncOptions = SoftplusOptions;
} // namespace functional

// ============================================================================

/// Options for the `Softshrink` module.
///
/// Example:
/// ```
/// Softshrink model(SoftshrinkOptions(42.42));
/// ```
struct SMITH_API SoftshrinkOptions {
  /* implicit */ SoftshrinkOptions(double lambda = 0.5);

  /// the `lambda` value for the Softshrink formulation. Default: 0.5
  SMITH_ARG(double, lambda);
};

namespace functional {
/// Options for `smith::nn::functional::softshrink`.
///
/// See the documentation for `smith::nn::SoftshrinkOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::softshrink(x, F::SoftshrinkFuncOptions(0.42));
/// ```
using SoftshrinkFuncOptions = SoftshrinkOptions;
} // namespace functional

// ============================================================================

/// Options for the `Threshold` module.
///
/// Example:
/// ```
/// Threshold model(ThresholdOptions(42.42, 24.24).inplace(true));
/// ```
struct SMITH_API ThresholdOptions {
  ThresholdOptions(double threshold, double value)
      : threshold_(threshold), value_(value) {}

  /// The value to threshold at
  SMITH_ARG(double, threshold);

  /// The value to replace with
  SMITH_ARG(double, value);

  /// can optionally do the operation in-place. Default: False
  SMITH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for `smith::nn::functional::threshold`.
///
/// See the documentation for `smith::nn::ThresholdOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::threshold(x, F::ThresholdFuncOptions(0.5, 0.5).inplace(true));
/// ```
using ThresholdFuncOptions = ThresholdOptions;
} // namespace functional

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::gumbel_softmax`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::gumbel_softmax(logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(-1));
/// ```
struct SMITH_API GumbelSoftmaxFuncOptions {
  /// non-negative scalar temperature
  SMITH_ARG(double, tau) = 1.0;

  /// returned samples will be discretized as one-hot vectors,
  /// but will be differentiated as if it is the soft sample in autograd.
  /// Default: False
  SMITH_ARG(bool, hard) = false;

  /// dimension along which softmax will be computed. Default: -1
  SMITH_ARG(int, dim) = -1;
};

} // namespace functional

// ============================================================================

/// Options for the `MultiheadAttention` module.
///
/// Example:
/// ```
/// MultiheadAttention model(MultiheadAttentionOptions(20, 10).bias(false));
/// ```
struct SMITH_API MultiheadAttentionOptions {
  MultiheadAttentionOptions(int64_t embed_dim, int64_t num_heads);

  /// total dimension of the model.
  SMITH_ARG(int64_t, embed_dim);

  /// parallel attention heads.
  SMITH_ARG(int64_t, num_heads);

  /// a Dropout layer on attn_output_weights. Default: 0.0.
  SMITH_ARG(double, dropout) = 0.0;

  /// add bias as module parameter. Default: true.
  SMITH_ARG(bool, bias) = true;

  /// add bias to the key and value sequences at dim=0.
  SMITH_ARG(bool, add_bias_kv) = false;

  /// add a new batch of zeros to the key and value sequences at dim=1.
  SMITH_ARG(bool, add_zero_attn) = false;

  /// total number of features in key. Default: std::nullopt.
  SMITH_ARG(int64_t, kdim);

  /// total number of features in key. Default: std::nullopt.
  SMITH_ARG(int64_t, vdim);
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::multi_head_attention_forward`
struct SMITH_API MultiheadAttentionForwardFuncOptions {
  MultiheadAttentionForwardFuncOptions(
      int64_t embed_dim_to_check,
      int64_t num_heads,
      Tensor in_proj_weight,
      Tensor in_proj_bias,
      Tensor bias_k,
      Tensor bias_v,
      bool add_zero_attn,
      double dropout_p,
      Tensor out_proj_weight,
      Tensor out_proj_bias);

  SMITH_ARG(int64_t, embed_dim_to_check);

  SMITH_ARG(int64_t, num_heads);

  SMITH_ARG(Tensor, in_proj_weight);

  SMITH_ARG(Tensor, in_proj_bias);

  SMITH_ARG(Tensor, bias_k);

  SMITH_ARG(Tensor, bias_v);

  SMITH_ARG(bool, add_zero_attn);

  SMITH_ARG(double, dropout_p);

  SMITH_ARG(Tensor, out_proj_weight);

  SMITH_ARG(Tensor, out_proj_bias);

  SMITH_ARG(bool, training) = true;

  SMITH_ARG(Tensor, key_padding_mask);

  SMITH_ARG(bool, need_weights) = true;

  SMITH_ARG(Tensor, attn_mask);

  SMITH_ARG(bool, use_separate_proj_weight) = false;

  SMITH_ARG(Tensor, q_proj_weight);

  SMITH_ARG(Tensor, k_proj_weight);

  SMITH_ARG(Tensor, v_proj_weight);

  SMITH_ARG(Tensor, static_k);

  SMITH_ARG(Tensor, static_v);

  SMITH_ARG(bool, average_attn_weights) = true;
};

} // namespace functional

} // namespace smith::nn
