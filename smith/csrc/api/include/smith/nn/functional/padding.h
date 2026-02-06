#pragma once

#include <ATen/PadNd.h>
#include <smith/nn/options/padding.h>

namespace smith::nn::functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor pad(
    const Tensor& input,
    IntArrayRef pad,
    PadFuncOptions::mode_t mode,
    double value) {
  const auto mode_enum = [&] {
    if (std::holds_alternative<enumtype::kConstant>(mode)) {
      return at::padding_mode::constant;
    } else if (std::holds_alternative<enumtype::kReflect>(mode)) {
      return at::padding_mode::reflect;
    } else if (std::holds_alternative<enumtype::kReplicate>(mode)) {
      return at::padding_mode::replicate;
    } else if (std::holds_alternative<enumtype::kCircular>(mode)) {
      return at::padding_mode::circular;
    }
    SMITH_CHECK(false, "Unrecognised padding mode");
  }();

  std::optional<double> fill_value;
  if (value != 0.0) {
    fill_value = value;
  }
  return at::_pad_enum(input, pad, static_cast<int64_t>(mode_enum), fill_value);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://blacksmith.org/docs/main/nn.functional.html#smith.nn.functional.pad
/// about the exact behavior of this functional.
///
/// See the documentation for `smith::nn::functional::PadFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::pad(input, F::PadFuncOptions({1, 2, 2, 1, 1,
/// 2}).mode(smith::kReplicate));
/// ```
inline Tensor pad(const Tensor& input, const PadFuncOptions& options) {
  return detail::pad(input, options.pad(), options.mode(), options.value());
}

} // namespace smith::nn::functional
