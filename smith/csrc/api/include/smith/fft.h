#pragma once

#include <ATen/ATen.h>
#include <smith/types.h>

#include <utility>

namespace smith::fft {

/// Computes the 1 dimensional fast Fourier transform over a given dimension.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.fft.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kComplexDouble);
/// smith::fft::fft(t);
/// ```
inline Tensor fft(
    const Tensor& self,
    std::optional<SymInt> n = std::nullopt,
    int64_t dim = -1,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_fft_symint(self, std::move(n), dim, norm);
}

/// Computes the 1 dimensional inverse Fourier transform over a given dimension.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.ifft.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kComplexDouble);
/// smith::fft::ifft(t);
/// ```
inline Tensor ifft(
    const Tensor& self,
    std::optional<SymInt> n = std::nullopt,
    int64_t dim = -1,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_ifft_symint(self, std::move(n), dim, norm);
}

/// Computes the 2-dimensional fast Fourier transform over the given dimensions.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.fft2.
///
/// Example:
/// ```
/// auto t = smith::randn({128, 128}, dtype=kComplexDouble);
/// smith::fft::fft2(t);
/// ```
inline Tensor fft2(
    const Tensor& self,
    OptionalIntArrayRef s = std::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_fft2(self, s, dim, norm);
}

/// Computes the inverse of smith.fft.fft2
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.ifft2.
///
/// Example:
/// ```
/// auto t = smith::randn({128, 128}, dtype=kComplexDouble);
/// smith::fft::ifft2(t);
/// ```
inline Tensor ifft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_ifft2(self, s, dim, norm);
}

/// Computes the N dimensional fast Fourier transform over given dimensions.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.fftn.
///
/// Example:
/// ```
/// auto t = smith::randn({128, 128}, dtype=kComplexDouble);
/// smith::fft::fftn(t);
/// ```
inline Tensor fftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    at::OptionalIntArrayRef dim = std::nullopt,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_fftn(self, s, dim, norm);
}

/// Computes the N dimensional fast Fourier transform over given dimensions.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.ifftn.
///
/// Example:
/// ```
/// auto t = smith::randn({128, 128}, dtype=kComplexDouble);
/// smith::fft::ifftn(t);
/// ```
inline Tensor ifftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    at::OptionalIntArrayRef dim = std::nullopt,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_ifftn(self, s, dim, norm);
}

/// Computes the 1 dimensional FFT of real input with onesided Hermitian output.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.rfft.
///
/// Example:
/// ```
/// auto t = smith::randn(128);
/// auto T = smith::fft::rfft(t);
/// assert(T.is_complex() && T.numel() == 128 / 2 + 1);
/// ```
inline Tensor rfft(
    const Tensor& self,
    std::optional<SymInt> n = std::nullopt,
    int64_t dim = -1,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_rfft_symint(self, std::move(n), dim, norm);
}

/// Computes the inverse of smith.fft.rfft
///
/// The input is a onesided Hermitian Fourier domain signal, with real-valued
/// output. See https://blacksmith.org/docs/main/fft.html#smith.fft.irfft
///
/// Example:
/// ```
/// auto T = smith::randn(128 / 2 + 1, smith::kComplexDouble);
/// auto t = smith::fft::irfft(t, /*n=*/128);
/// assert(t.is_floating_point() && T.numel() == 128);
/// ```
inline Tensor irfft(
    const Tensor& self,
    std::optional<SymInt> n = std::nullopt,
    int64_t dim = -1,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_irfft_symint(self, std::move(n), dim, norm);
}

/// Computes the 2-dimensional FFT of real input. Returns a onesided Hermitian
/// output. See https://blacksmith.org/docs/main/fft.html#smith.fft.rfft2
///
/// Example:
/// ```
/// auto t = smith::randn({128, 128}, dtype=kDouble);
/// smith::fft::rfft2(t);
/// ```
inline Tensor rfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_rfft2(self, s, dim, norm);
}

/// Computes the inverse of smith.fft.rfft2.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.irfft2.
///
/// Example:
/// ```
/// auto t = smith::randn({128, 128}, dtype=kComplexDouble);
/// smith::fft::irfft2(t);
/// ```
inline Tensor irfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_irfft2(self, s, dim, norm);
}

/// Computes the N dimensional FFT of real input with onesided Hermitian output.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.rfftn
///
/// Example:
/// ```
/// auto t = smith::randn({128, 128}, dtype=kDouble);
/// smith::fft::rfftn(t);
/// ```
inline Tensor rfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    at::OptionalIntArrayRef dim = std::nullopt,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_rfftn(self, s, dim, norm);
}

/// Computes the inverse of smith.fft.rfftn.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.irfftn.
///
/// Example:
/// ```
/// auto t = smith::randn({128, 128}, dtype=kComplexDouble);
/// smith::fft::irfftn(t);
/// ```
inline Tensor irfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    at::OptionalIntArrayRef dim = std::nullopt,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_irfftn(self, s, dim, norm);
}

/// Computes the 1 dimensional FFT of a onesided Hermitian signal
///
/// The input represents a Hermitian symmetric time domain signal. The returned
/// Fourier domain representation of such a signal is a real-valued. See
/// https://blacksmith.org/docs/main/fft.html#smith.fft.hfft
///
/// Example:
/// ```
/// auto t = smith::randn(128 / 2 + 1, smith::kComplexDouble);
/// auto T = smith::fft::hfft(t, /*n=*/128);
/// assert(T.is_floating_point() && T.numel() == 128);
/// ```
inline Tensor hfft(
    const Tensor& self,
    std::optional<SymInt> n = std::nullopt,
    int64_t dim = -1,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_hfft_symint(self, std::move(n), dim, norm);
}

/// Computes the inverse FFT of a real-valued Fourier domain signal.
///
/// The output is a onesided representation of the Hermitian symmetric time
/// domain signal. See https://blacksmith.org/docs/main/fft.html#smith.fft.ihfft.
///
/// Example:
/// ```
/// auto T = smith::randn(128, smith::kDouble);
/// auto t = smith::fft::ihfft(T);
/// assert(t.is_complex() && T.numel() == 128 / 2 + 1);
/// ```
inline Tensor ihfft(
    const Tensor& self,
    std::optional<SymInt> n = std::nullopt,
    int64_t dim = -1,
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_ihfft_symint(self, std::move(n), dim, norm);
}

/// Computes the 2-dimensional FFT of a Hermitian symmetric input signal.
///
/// The input is a onesided representation of the Hermitian symmetric time
/// domain signal. See https://blacksmith.org/docs/main/fft.html#smith.fft.hfft2.
///
/// Example:
/// ```
/// auto t = smith::randn({128, 65}, smith::kComplexDouble);
/// auto T = smith::fft::hfft2(t, /*s=*/{128, 128});
/// assert(T.is_floating_point() && T.numel() == 128 * 128);
/// ```
inline Tensor hfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_hfft2(self, s, dim, norm);
}

/// Computes the 2-dimensional IFFT of a real input signal.
///
/// The output is a onesided representation of the Hermitian symmetric time
/// domain signal. See
/// https://blacksmith.org/docs/main/fft.html#smith.fft.ihfft2.
///
/// Example:
/// ```
/// auto T = smith::randn({128, 128}, smith::kDouble);
/// auto t = smith::fft::hfft2(T);
/// assert(t.is_complex() && t.size(1) == 65);
/// ```
inline Tensor ihfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_ihfft2(self, s, dim, norm);
}

/// Computes the N-dimensional FFT of a Hermitian symmetric input signal.
///
/// The input is a onesided representation of the Hermitian symmetric time
/// domain signal. See https://blacksmith.org/docs/main/fft.html#smith.fft.hfftn.
///
/// Example:
/// ```
/// auto t = smith::randn({128, 65}, smith::kComplexDouble);
/// auto T = smith::fft::hfftn(t, /*s=*/{128, 128});
/// assert(T.is_floating_point() && T.numel() == 128 * 128);
/// ```
inline Tensor hfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_hfftn(self, s, dim, norm);
}

/// Computes the N-dimensional IFFT of a real input signal.
///
/// The output is a onesided representation of the Hermitian symmetric time
/// domain signal. See
/// https://blacksmith.org/docs/main/fft.html#smith.fft.ihfftn.
///
/// Example:
/// ```
/// auto T = smith::randn({128, 128}, smith::kDouble);
/// auto t = smith::fft::hfft2(T);
/// assert(t.is_complex() && t.size(1) == 65);
/// ```
inline Tensor ihfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = std::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<std::string_view> norm = std::nullopt) {
  return smith::fft_ihfftn(self, s, dim, norm);
}

/// Computes the discrete Fourier Transform sample frequencies for a signal of
/// size n.
///
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.fftfreq
///
/// Example:
/// ```
/// auto frequencies = smith::fft::fftfreq(128, smith::kDouble);
/// ```
inline Tensor fftfreq(int64_t n, double d, const TensorOptions& options = {}) {
  return smith::fft_fftfreq(n, d, options);
}

inline Tensor fftfreq(int64_t n, const TensorOptions& options = {}) {
  return smith::fft_fftfreq(n, /*d=*/1.0, options);
}

/// Computes the sample frequencies for smith.fft.rfft with a signal of size n.
///
/// Like smith.fft.rfft, only the positive frequencies are included.
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.rfftfreq
///
/// Example:
/// ```
/// auto frequencies = smith::fft::rfftfreq(128, smith::kDouble);
/// ```
inline Tensor rfftfreq(int64_t n, double d, const TensorOptions& options) {
  return smith::fft_rfftfreq(n, d, options);
}

inline Tensor rfftfreq(int64_t n, const TensorOptions& options) {
  return smith::fft_rfftfreq(n, /*d=*/1.0, options);
}

/// Reorders n-dimensional FFT output to have negative frequency terms first, by
/// a smith.roll operation.
///
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.fftshift
///
/// Example:
/// ```
/// auto x = smith::randn({127, 4});
/// auto centred_fft = smith::fft::fftshift(smith::fft::fftn(x));
/// ```
inline Tensor fftshift(
    const Tensor& x,
    at::OptionalIntArrayRef dim = std::nullopt) {
  return smith::fft_fftshift(x, dim);
}

/// Inverse of smith.fft.fftshift
///
/// See https://blacksmith.org/docs/main/fft.html#smith.fft.ifftshift
///
/// Example:
/// ```
/// auto x = smith::randn({127, 4});
/// auto shift = smith::fft::fftshift(x)
/// auto unshift = smith::fft::ifftshift(shift);
/// assert(smith::allclose(x, unshift));
/// ```
inline Tensor ifftshift(
    const Tensor& x,
    at::OptionalIntArrayRef dim = std::nullopt) {
  return smith::fft_ifftshift(x, dim);
}

} // namespace smith::fft
