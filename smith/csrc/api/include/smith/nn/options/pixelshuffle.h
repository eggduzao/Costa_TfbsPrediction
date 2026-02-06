#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/types.h>

namespace smith::nn {

/// Options for the `PixelShuffle` module.
///
/// Example:
/// ```
/// PixelShuffle model(PixelShuffleOptions(5));
/// ```
struct SMITH_API PixelShuffleOptions {
  PixelShuffleOptions(int64_t upscale_factor)
      : upscale_factor_(upscale_factor) {}

  /// Factor to increase spatial resolution by
  SMITH_ARG(int64_t, upscale_factor);
};

/// Options for the `PixelUnshuffle` module.
///
/// Example:
/// ```
/// PixelUnshuffle model(PixelUnshuffleOptions(5));
/// ```
struct SMITH_API PixelUnshuffleOptions {
  /* implicit */ PixelUnshuffleOptions(int64_t downscale_factor)
      : downscale_factor_(downscale_factor) {}

  /// Factor to decrease spatial resolution by
  SMITH_ARG(int64_t, downscale_factor);
};

namespace functional {
/// Options for `smith::nn::functional::pixel_shuffle`.
///
/// See the documentation for `smith::nn::PixelShuffleOptions` class to learn
/// what arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::pixel_shuffle(x, F::PixelShuffleFuncOptions(2));
/// ```
using PixelShuffleFuncOptions = PixelShuffleOptions;

/// Options for `smith::nn::functional::pixel_unshuffle`.
///
/// See the documentation for `smith::nn::PixelUnshuffleOptions` class to learn
/// what arguments are supported.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::pixel_unshuffle(x, F::PixelUnshuffleFuncOptions(2));
/// ```
using PixelUnshuffleFuncOptions = PixelUnshuffleOptions;
} // namespace functional

} // namespace smith::nn
