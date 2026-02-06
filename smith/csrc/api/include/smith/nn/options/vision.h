#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/enum.h>
#include <smith/types.h>

namespace smith::nn::functional {

/// Options for `smith::nn::functional::grid_sample`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::grid_sample(input, grid,
/// F::GridSampleFuncOptions().mode(smith::kBilinear).padding_mode(smith::kZeros).align_corners(true));
/// ```
struct SMITH_API GridSampleFuncOptions {
  typedef std::
      variant<enumtype::kBilinear, enumtype::kNearest, enumtype::kBicubic>
          mode_t;
  typedef std::
      variant<enumtype::kZeros, enumtype::kBorder, enumtype::kReflection>
          padding_mode_t;

  /// interpolation mode to calculate output values. Default: Bilinear
  SMITH_ARG(mode_t, mode) = smith::kBilinear;
  /// padding mode for outside grid values. Default: Zeros
  SMITH_ARG(padding_mode_t, padding_mode) = smith::kZeros;
  /// Specifies perspective to pixel as point. Default: false
  SMITH_ARG(std::optional<bool>, align_corners) = std::nullopt;
};

} // namespace smith::nn::functional
