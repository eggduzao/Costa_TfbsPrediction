#pragma once

#include <c10/util/irange.h>
#include <smith/nn/options/batchnorm.h>
#include <smith/types.h>

namespace smith::nn::functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor batch_norm(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    Tensor weight,
    Tensor bias,
    bool training,
    double momentum,
    double eps) {
  SMITH_CHECK(
      input.dim() >= 2,
      "Expected at least 2 input dimensions, but got ",
      input.dim());
  if (training) {
    auto size = input.sizes();
    int64_t size_prods = size[0];
    for (const auto i : c10::irange(size.size() - 2)) {
      size_prods *= size[i + 2];
    }
    SMITH_CHECK(
        size_prods != 1,
        "Expected more than 1 value per channel when training, got input size ",
        size);
  }

  return smith::batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      at::globalContext().userEnabledCuDNN());
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://blacksmith.org/docs/main/nn.functional.html#smith.nn.functional.batch_norm
/// about the exact behavior of this functional.
///
/// See the documentation for `smith::nn::functional::BatchNormFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::batch_norm(input, mean, variance,
/// F::BatchNormFuncOptions().weight(weight).bias(bias).momentum(0.1).eps(1e-05).training(false));
/// ```
inline Tensor batch_norm(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const BatchNormFuncOptions& options = {}) {
  return detail::batch_norm(
      input,
      running_mean,
      running_var,
      options.weight(),
      options.bias(),
      options.training(),
      options.momentum(),
      options.eps());
}

} // namespace smith::nn::functional
