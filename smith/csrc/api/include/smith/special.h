#pragma once

#include <ATen/ATen.h>
#include <smith/types.h>

namespace smith::special {

/// Computes the natural logarithm of the absolute value of the gamma function
/// See https://blacksmith.org/docs/main/special.html#smith.special.gammaln.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::gammaln(t);
/// ```
inline Tensor gammaln(const Tensor& self) {
  return smith::special_gammaln(self);
}

inline Tensor& gammaln_out(Tensor& result, const Tensor& self) {
  return smith::special_gammaln_out(result, self);
}

/// Computes the regularized lower incomplete gamma function
/// See https://blacksmith.org/docs/main/special.html#smith.special.gammainc.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// auto s = smith::randn(128, dtype=kDouble);
/// smith::special::gammainc(s, t);
/// ```
inline Tensor gammainc(const Tensor& self, const Tensor& other) {
  return smith::special_gammainc(self, other);
}

inline Tensor& gammainc_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return smith::special_gammainc_out(result, self, other);
}

/// Computes the regularized upper incomplete gamma function
/// See https://blacksmith.org/docs/main/special.html#smith.special.gammainc.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// auto s = smith::randn(128, dtype=kDouble);
/// smith::special::gammaincc(s, t);
/// ```
inline Tensor gammaincc(const Tensor& self, const Tensor& other) {
  return smith::special_gammaincc(self, other);
}

inline Tensor& gammaincc_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return smith::special_gammaincc_out(result, self, other);
}

/// Computes the multivariate log-gamma function with dimension `p`, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.multigammaln.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::multigammaln(t, 1);
/// ```
inline Tensor multigammaln(const Tensor& self, int64_t p) {
  return smith::special_multigammaln(self, p);
}

inline Tensor& multigammaln_out(Tensor& result, const Tensor& self, int64_t p) {
  return smith::special_multigammaln_out(result, self, p);
}

/// Computes the nth derivative of the digamma function on the input.
/// See https:://blacksmith.org/docs/main/special.html#smith.special.polygamma.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::polygamma(2, t);
/// ```
inline Tensor polygamma(int64_t n, const Tensor& self) {
  return smith::special_polygamma(n, self);
}

inline Tensor& polygamma_out(Tensor& result, int64_t n, const Tensor& self) {
  return smith::special_polygamma_out(result, n, self);
}

/// Computes the logarithmic derivative of the gamma function on input
/// See https://blacksmith.org/docs/main/special.html#smith.special.psi
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::psi(t);
/// ```
inline Tensor psi(const Tensor& self) {
  return smith::special_psi(self);
}

inline Tensor& psi_out(Tensor& result, const Tensor& self) {
  return smith::special_psi_out(result, self);
}

/// Computes the logarithmic derivative of the gamma function on input
/// See https://blacksmith.org/docs/main/special.html#smith.special.digamma
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::digamma(t);
/// ```
inline Tensor digamma(const Tensor& self) {
  return smith::special_digamma(self);
}

inline Tensor& digamma_out(Tensor& result, const Tensor& self) {
  return smith::special_digamma_out(result, self);
}

/// Computes entropy of input, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.entr.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::entr(t);
/// ```
inline Tensor entr(const Tensor& self) {
  return smith::special_entr(self);
}

inline Tensor& entr_out(Tensor& result, const Tensor& self) {
  return smith::special_entr_out(result, self);
}

/// Computes the error function
/// See https://blacksmith.org/docs/main/special.html#smith.special.erf.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::erf(t);
/// ```
inline Tensor erf(const Tensor& self) {
  return smith::special_erf(self);
}

inline Tensor& erf_out(Tensor& result, const Tensor& self) {
  return smith::special_erf_out(result, self);
}

/// Computes the complementary error function
/// See https://blacksmith.org/docs/main/special.html#smith.special.erfc.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::erfc(t);
/// ```
inline Tensor erfc(const Tensor& self) {
  return smith::special_erfc(self);
}

inline Tensor& erfc_out(Tensor& result, const Tensor& self) {
  return smith::special_erfc_out(result, self);
}

/// Computes the scaled complementary error function
/// See https://blacksmith.org/docs/main/special.html#smith.special.erfcx.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::erfcx(t);
/// ```
inline Tensor erfcx(const Tensor& self) {
  return smith::special_erfcx(self);
}

inline Tensor& erfcx_out(Tensor& result, const Tensor& self) {
  return smith::special_erfcx_out(result, self);
}

/// Computes the inverse error function
/// See https://blacksmith.org/docs/main/special.html#smith.special.erfinv.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::erfinv(t);
/// ```
inline Tensor erfinv(const Tensor& self) {
  return smith::special_erfinv(self);
}

inline Tensor& erfinv_out(Tensor& result, const Tensor& self) {
  return smith::special_erfinv_out(result, self);
}

/// Computes the log of summed exponentials of each row of input in the given
/// dimension dim See
/// https://blacksmith.org/docs/main/special.html#smith.special.logsumexp.
///
/// Example:
/// ```
/// auto t = smith::randn(3, 3);
/// smith::special::logsumexp(t, 1);
/// ```
inline Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  return smith::special_logsumexp(self, dims, keepdim);
}

inline Tensor& logsumexp_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim) {
  return smith::special_logsumexp_out(result, self, dims, keepdim);
}

/// Computes the argument, x, for which the area under the Gaussian probability
/// density function (integrated from minus infinity to x) is equal to input,
/// elementwise. See
/// https://blacksmith.org/docs/main/special.html#smith.special.ndtri
///
/// Example:
/// ```
/// auto t = smith::rand(128, dtype=kDouble);
/// smith::special::ndtri(t);
/// ```
inline Tensor ndtri(const Tensor& self) {
  return smith::special_ndtri(self);
}

inline Tensor& ndtri_out(Tensor& result, const Tensor& self) {
  return smith::special_ndtri_out(result, self);
}

/// Computes the log of area under the standard Gaussian probability density
/// function, integrated from minus infinity to :attr:`input`, elementwise See
/// https://blacksmith.org/docs/main/special.html#smith.special.log_ndtr
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::log_ndtr(t);
/// ```
inline Tensor log_ndtr(const Tensor& self) {
  return smith::special_log_ndtr(self);
}

inline Tensor& log_ndtr_out(Tensor& result, const Tensor& self) {
  return smith::special_log_ndtr_out(result, self);
}

/// Computes the logit of input, elementwise.
/// See https://blacksmith.org/docs/main/special.html#smith.special.logit.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::logit(t);
/// ```
inline Tensor logit(const Tensor& self) {
  return smith::special_logit(self);
}

inline Tensor& logit_out(Tensor& result, const Tensor& self) {
  return smith::special_logit_out(result, self);
}

/// Computes the expit (also known as the logistic sigmoid function) of input,
/// elementwise See
/// https://blacksmith.org/docs/main/special.html#smith.special.expit.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::expit(t);
/// ```
inline Tensor expit(const Tensor& self) {
  return smith::special_expit(self);
}

inline Tensor& expit_out(Tensor& result, const Tensor& self) {
  return smith::special_expit_out(result, self);
}

/// Computes the base two exponential function of :attr:`input`, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.exp2.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::exp2(t);
/// ```
inline Tensor exp2(const Tensor& self) {
  return smith::special_exp2(self);
}

inline Tensor& exp2_out(Tensor& result, const Tensor& self) {
  return smith::special_exp2_out(result, self);
}

/// Computes the exponential of the elements minus 1, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.expm1.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::expm1(t);
/// ```
inline Tensor expm1(const Tensor& self) {
  return smith::special_expm1(self);
}

inline Tensor& expm1_out(Tensor& result, const Tensor& self) {
  return smith::special_expm1_out(result, self);
}

/// Computes x * log(y) for inputs, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.xlogy.
///
/// Example:
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto y = smith::randn(128, dtype=kDouble);
/// smith::special::xlogy(x, y);
/// ```
inline Tensor xlogy(const Tensor& self, const Tensor& other) {
  return smith::special_xlogy(self, other);
}

inline Tensor xlogy(const Scalar& self, const Tensor& other) {
  return smith::special_xlogy(self, other);
}

inline Tensor xlogy(const Tensor& self, const Scalar& other) {
  return smith::special_xlogy(self, other);
}

inline Tensor& xlogy_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return smith::special_xlogy_out(result, self, other);
}

inline Tensor& xlogy_out(
    Tensor& result,
    const Scalar& self,
    const Tensor& other) {
  return smith::special_xlogy_out(result, self, other);
}

inline Tensor& xlogy_out(
    Tensor& result,
    const Tensor& self,
    const Scalar& other) {
  return smith::special_xlogy_out(result, self, other);
}

/// Computes x * log1p(y) for inputs, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.xlog1py.
///
/// Example:
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto y = smith::randn(128, dtype=kDouble);
/// smith::special::xlog1py(x, y);
/// ```
inline Tensor xlog1py(const Tensor& self, const Tensor& other) {
  return smith::special_xlog1py(self, other);
}

inline Tensor xlog1py(const Scalar& self, const Tensor& other) {
  return smith::special_xlog1py(self, other);
}

inline Tensor xlog1py(const Tensor& self, const Scalar& other) {
  return smith::special_xlog1py(self, other);
}

inline Tensor& xlog1py_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return smith::special_xlog1py_out(result, self, other);
}

inline Tensor& xlog1py_out(
    Tensor& result,
    const Scalar& self,
    const Tensor& other) {
  return smith::special_xlog1py_out(result, self, other);
}

inline Tensor& xlog1py_out(
    Tensor& result,
    const Tensor& self,
    const Scalar& other) {
  return smith::special_xlog1py_out(result, self, other);
}

/// Computes Hurwitz Zeta function for inputs, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.zeta.
///
/// Example:
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto y = smith::randn(128, dtype=kDouble);
/// smith::special::zeta(x, y);
/// ```
inline Tensor zeta(const Tensor& self, const Tensor& other) {
  return smith::special_zeta(self, other);
}

inline Tensor zeta(const Scalar& self, const Tensor& other) {
  return smith::special_zeta(self, other);
}

inline Tensor zeta(const Tensor& self, const Scalar& other) {
  return smith::special_zeta(self, other);
}

inline Tensor& zeta_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return smith::special_zeta_out(result, self, other);
}

inline Tensor& zeta_out(
    Tensor& result,
    const Scalar& self,
    const Tensor& other) {
  return smith::special_zeta_out(result, self, other);
}

inline Tensor& zeta_out(
    Tensor& result,
    const Tensor& self,
    const Scalar& other) {
  return smith::special_zeta_out(result, self, other);
}

/// Computes the zeroth order modified Bessel function of the first kind of
/// input, elementwise See
/// https://blacksmith.org/docs/main/special.html#smith.special.i0
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::i0(t);
/// ```
inline Tensor i0(const Tensor& self) {
  return smith::special_i0(self);
}

inline Tensor& i0_out(Tensor& result, const Tensor& self) {
  return smith::special_i0_out(result, self);
}

/// Computes the area under the standard Gaussian probability density function,
/// integrated from minus infinity to :attr:`input`, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.ndtr
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::ndtr(t);
/// ```
inline Tensor ndtr(const Tensor& self) {
  return smith::special_ndtr(self);
}

inline Tensor& ndtr_out(Tensor& result, const Tensor& self) {
  return smith::special_ndtr_out(result, self);
}

/// Computes the exponentially scaled zeroth order modified Bessel function of
/// the first kind See
/// https://blacksmith.org/docs/main/special.html#smith.special.i0e.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::i0e(t);
/// ```
inline Tensor i0e(const Tensor& self) {
  return smith::special_i0e(self);
}

inline Tensor& i0e_out(Tensor& result, const Tensor& self) {
  return smith::special_i0e_out(result, self);
}

/// Computes the first order modified Bessel function of the first kind
/// See https://blacksmith.org/docs/main/special.html#smith.special.i1.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::i1(t);
/// ```
inline Tensor i1(const Tensor& self) {
  return smith::special_i1(self);
}

inline Tensor& i1_out(Tensor& result, const Tensor& self) {
  return smith::special_i1_out(result, self);
}

/// Computes the exponentially scaled first order modified Bessel function of
/// the first kind See
/// https://blacksmith.org/docs/main/special.html#smith.special.i1e.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::i1e(t);
/// ```
inline Tensor i1e(const Tensor& self) {
  return smith::special_i1e(self);
}

inline Tensor& i1e_out(Tensor& result, const Tensor& self) {
  return smith::special_i1e_out(result, self);
}

/// Computes the sinc of input, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.sinc.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::sinc(t);
/// ```
inline Tensor sinc(const Tensor& self) {
  return smith::special_sinc(self);
}

inline Tensor& sinc_out(Tensor& result, const Tensor& self) {
  return smith::special_sinc_out(result, self);
}

/// Rounds the elements of the input
/// See https://blacksmith.org/docs/main/special.html#smith.special.round.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::round(t);
/// ```
inline Tensor round(const Tensor& self) {
  return smith::special_round(self);
}

inline Tensor& round_out(Tensor& result, const Tensor& self) {
  return smith::special_round_out(result, self);
}

/// Computes log(1 + x) of the input, elementwise
/// See https://blacksmith.org/docs/main/special.html#smith.special.log1p.
///
/// Example:
/// ```
/// auto t = smith::randn(128, dtype=kDouble);
/// smith::special::log1p(t);
/// ```
inline Tensor log1p(const Tensor& self) {
  return smith::special_log1p(self);
}

inline Tensor& log1p_out(Tensor& result, const Tensor& self) {
  return smith::special_log1p_out(result, self);
}

/// Computes log followed by softmax(x) of the input
/// See https://blacksmith.org/docs/main/special.html#smith.special.log_softmax.
///
/// Example:
/// ```
/// auto t = smith::randn(128, 128, dtype=kDouble);
/// smith::special::log_softmax(t, 0);
/// ```
inline Tensor log_softmax(
    const Tensor& self,
    int64_t dim,
    std::optional<ScalarType> dtype) {
  return smith::special_log_softmax(self, dim, dtype);
}

/// Computes softmax of the input along a given dimension
/// See https://blacksmith.org/docs/main/special.html#smith.special.softmax.
///
/// Example:
/// ```
/// auto t = smith::randn(128, 128, dtype=kDouble);
/// smith::special::softmax(t, 0);
/// ```
inline Tensor softmax(
    const Tensor& self,
    int64_t dim,
    std::optional<ScalarType> dtype) {
  return smith::special_softmax(self, dim, dtype);
}

/// Airy function Ai.
///
/// See https://blacksmith.org/docs/main/special.html#smith.special.airy_ai.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::airy_ai(x);
/// ```
inline Tensor airy_ai(const Tensor& x) {
  return smith::special_airy_ai(x);
}

inline Tensor& airy_ai_out(Tensor& y, const Tensor& x) {
  return smith::special_airy_ai_out(y, x);
}

/// Bessel function of the first kind of order 0.
///
/// See https://blacksmith.org/docs/main/special.html#smith.special.bessel_j0.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::bessel_j0(x);
/// ```
inline Tensor bessel_j0(const Tensor& self) {
  return smith::special_bessel_j0(self);
}

inline Tensor& bessel_j0_out(Tensor& result, const Tensor& self) {
  return smith::special_bessel_j0_out(result, self);
}

/// Bessel function of the first kind of order 1.
///
/// See https://blacksmith.org/docs/main/special.html#smith.special.bessel_j1.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::bessel_j1(x);
/// ```
inline Tensor bessel_j1(const Tensor& self) {
  return smith::special_bessel_j1(self);
}

inline Tensor& bessel_j1_out(Tensor& result, const Tensor& self) {
  return smith::special_bessel_j1_out(result, self);
}

/// Bessel function of the second kind of order 0.
///
/// See https://blacksmith.org/docs/main/special.html#smith.special.bessel_y0.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::bessel_y0(x);
/// ```
inline Tensor bessel_y0(const Tensor& self) {
  return smith::special_bessel_y0(self);
}

inline Tensor& bessel_y0_out(Tensor& result, const Tensor& self) {
  return smith::special_bessel_y0_out(result, self);
}

/// Bessel function of the second kind of order 1.
///
/// See https://blacksmith.org/docs/main/special.html#smith.special.bessel_y1.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::bessel_y1(x);
/// ```
inline Tensor bessel_y1(const Tensor& self) {
  return smith::special_bessel_y1(self);
}

inline Tensor& bessel_y1_out(Tensor& result, const Tensor& self) {
  return smith::special_bessel_y1_out(result, self);
}

/// Chebyshev polynomial of the first kind.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.chebyshev_polynomial_t.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::chebyshev_polynomial_t(x, n);
/// ```
inline Tensor chebyshev_polynomial_t(const Tensor& x, const Tensor& n) {
  return smith::special_chebyshev_polynomial_t(x, n);
}

inline Tensor chebyshev_polynomial_t(const Scalar& x, const Tensor& n) {
  return smith::special_chebyshev_polynomial_t(x, n);
}

inline Tensor chebyshev_polynomial_t(const Tensor& x, const Scalar& n) {
  return smith::special_chebyshev_polynomial_t(x, n);
}

inline Tensor& chebyshev_polynomial_t_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_chebyshev_polynomial_t_out(output, x, n);
}

inline Tensor& chebyshev_polynomial_t_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_chebyshev_polynomial_t_out(output, x, n);
}

inline Tensor& chebyshev_polynomial_t_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_chebyshev_polynomial_t_out(output, x, n);
}

/// Chebyshev polynomial of the second kind.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.chebyshev_polynomial_u.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::chebyshev_polynomial_u(x, n);
/// ```
inline Tensor chebyshev_polynomial_u(const Tensor& x, const Tensor& n) {
  return smith::special_chebyshev_polynomial_u(x, n);
}

inline Tensor chebyshev_polynomial_u(const Scalar& x, const Tensor& n) {
  return smith::special_chebyshev_polynomial_u(x, n);
}

inline Tensor chebyshev_polynomial_u(const Tensor& x, const Scalar& n) {
  return smith::special_chebyshev_polynomial_u(x, n);
}

inline Tensor& chebyshev_polynomial_u_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_chebyshev_polynomial_u_out(output, x, n);
}

inline Tensor& chebyshev_polynomial_u_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_chebyshev_polynomial_u_out(output, x, n);
}

inline Tensor& chebyshev_polynomial_u_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_chebyshev_polynomial_u_out(output, x, n);
}

/// Chebyshev polynomial of the third kind.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.chebyshev_polynomial_v.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::chebyshev_polynomial_v(x, n);
/// ```
inline Tensor chebyshev_polynomial_v(const Tensor& x, const Tensor& n) {
  return smith::special_chebyshev_polynomial_v(x, n);
}

inline Tensor chebyshev_polynomial_v(const Scalar& x, const Tensor& n) {
  return smith::special_chebyshev_polynomial_v(x, n);
}

inline Tensor chebyshev_polynomial_v(const Tensor& x, const Scalar& n) {
  return smith::special_chebyshev_polynomial_v(x, n);
}

inline Tensor& chebyshev_polynomial_v_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_chebyshev_polynomial_v_out(output, x, n);
}

inline Tensor& chebyshev_polynomial_v_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_chebyshev_polynomial_v_out(output, x, n);
}

inline Tensor& chebyshev_polynomial_v_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_chebyshev_polynomial_v_out(output, x, n);
}

/// Chebyshev polynomial of the fourth kind.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.chebyshev_polynomial_w.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::chebyshev_polynomial_w(x, n);
/// ```
inline Tensor chebyshev_polynomial_w(const Tensor& x, const Tensor& n) {
  return smith::special_chebyshev_polynomial_w(x, n);
}

inline Tensor chebyshev_polynomial_w(const Scalar& x, const Tensor& n) {
  return smith::special_chebyshev_polynomial_w(x, n);
}

inline Tensor chebyshev_polynomial_w(const Tensor& x, const Scalar& n) {
  return smith::special_chebyshev_polynomial_w(x, n);
}

inline Tensor& chebyshev_polynomial_w_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_chebyshev_polynomial_w_out(output, x, n);
}

inline Tensor& chebyshev_polynomial_w_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_chebyshev_polynomial_w_out(output, x, n);
}

inline Tensor& chebyshev_polynomial_w_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_chebyshev_polynomial_w_out(output, x, n);
}

/// Physicist’s Hermite polynomial.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.hermite_polynomial_h.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::hermite_polynomial_h(x, n);
/// ```
inline Tensor hermite_polynomial_h(const Tensor& x, const Tensor& n) {
  return smith::special_hermite_polynomial_h(x, n);
}

inline Tensor hermite_polynomial_h(const Scalar& x, const Tensor& n) {
  return smith::special_hermite_polynomial_h(x, n);
}

inline Tensor hermite_polynomial_h(const Tensor& x, const Scalar& n) {
  return smith::special_hermite_polynomial_h(x, n);
}

inline Tensor& hermite_polynomial_h_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_hermite_polynomial_h_out(output, x, n);
}

inline Tensor& hermite_polynomial_h_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_hermite_polynomial_h_out(output, x, n);
}

inline Tensor& hermite_polynomial_h_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_hermite_polynomial_h_out(output, x, n);
}

/// Probabilist’s Hermite polynomial.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.hermite_polynomial_he.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::hermite_polynomial_he(x, n);
/// ```
inline Tensor hermite_polynomial_he(const Tensor& x, const Tensor& n) {
  return smith::special_hermite_polynomial_he(x, n);
}

inline Tensor hermite_polynomial_he(const Scalar& x, const Tensor& n) {
  return smith::special_hermite_polynomial_he(x, n);
}

inline Tensor hermite_polynomial_he(const Tensor& x, const Scalar& n) {
  return smith::special_hermite_polynomial_he(x, n);
}

inline Tensor& hermite_polynomial_he_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_hermite_polynomial_he_out(output, x, n);
}

inline Tensor& hermite_polynomial_he_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_hermite_polynomial_he_out(output, x, n);
}

inline Tensor& hermite_polynomial_he_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_hermite_polynomial_he_out(output, x, n);
}

/// Laguerre polynomial.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.laguerre_polynomial_l.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::laguerre_polynomial_l(x, n);
/// ```
inline Tensor laguerre_polynomial_l(const Tensor& x, const Tensor& n) {
  return smith::special_laguerre_polynomial_l(x, n);
}

inline Tensor laguerre_polynomial_l(const Scalar& x, const Tensor& n) {
  return smith::special_laguerre_polynomial_l(x, n);
}

inline Tensor laguerre_polynomial_l(const Tensor& x, const Scalar& n) {
  return smith::special_laguerre_polynomial_l(x, n);
}

inline Tensor& laguerre_polynomial_l_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_laguerre_polynomial_l_out(output, x, n);
}

inline Tensor& laguerre_polynomial_l_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_laguerre_polynomial_l_out(output, x, n);
}

inline Tensor& laguerre_polynomial_l_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_laguerre_polynomial_l_out(output, x, n);
}

/// Legendre polynomial.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.legendre_polynomial_p.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::legendre_polynomial_p(x, n);
/// ```
inline Tensor legendre_polynomial_p(const Tensor& x, const Tensor& n) {
  return smith::special_legendre_polynomial_p(x, n);
}

inline Tensor legendre_polynomial_p(const Scalar& x, const Tensor& n) {
  return smith::special_legendre_polynomial_p(x, n);
}

inline Tensor legendre_polynomial_p(const Tensor& x, const Scalar& n) {
  return smith::special_legendre_polynomial_p(x, n);
}

inline Tensor& legendre_polynomial_p_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_legendre_polynomial_p_out(output, x, n);
}

inline Tensor& legendre_polynomial_p_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_legendre_polynomial_p_out(output, x, n);
}

inline Tensor& legendre_polynomial_p_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_legendre_polynomial_p_out(output, x, n);
}

/// Modified Bessel function of the first kind of order 0.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.modified_bessel_i0.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::modified_bessel_i0(x);
/// ```
inline Tensor modified_bessel_i0(const Tensor& self) {
  return smith::special_modified_bessel_i0(self);
}

inline Tensor& modified_bessel_i0_out(Tensor& result, const Tensor& self) {
  return smith::special_modified_bessel_i0_out(result, self);
}

/// Modified Bessel function of the first kind of order 1.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.modified_bessel_i1.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::modified_bessel_i1(x);
/// ```
inline Tensor modified_bessel_i1(const Tensor& self) {
  return smith::special_modified_bessel_i1(self);
}

inline Tensor& modified_bessel_i1_out(Tensor& result, const Tensor& self) {
  return smith::special_modified_bessel_i1_out(result, self);
}

/// Modified Bessel function of the second kind of order 0.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.modified_bessel_k0.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::modified_bessel_k0(x);
/// ```
inline Tensor modified_bessel_k0(const Tensor& self) {
  return smith::special_modified_bessel_k0(self);
}

inline Tensor& modified_bessel_k0_out(Tensor& result, const Tensor& self) {
  return smith::special_modified_bessel_k0_out(result, self);
}

/// Modified Bessel function of the second kind of order 1.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.modified_bessel_k1.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::modified_bessel_k1(x);
/// ```
inline Tensor modified_bessel_k1(const Tensor& self) {
  return smith::special_modified_bessel_k1(self);
}

inline Tensor& modified_bessel_k1_out(Tensor& result, const Tensor& self) {
  return smith::special_modified_bessel_k1_out(result, self);
}

/// Scaled modified Bessel function of the second kind of order 0.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.scaled_modified_bessel_k0.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::scaled_modified_bessel_k0(x);
/// ```
inline Tensor scaled_modified_bessel_k0(const Tensor& x) {
  return smith::special_scaled_modified_bessel_k0(x);
}

inline Tensor& scaled_modified_bessel_k0_out(Tensor& y, const Tensor& x) {
  return smith::special_scaled_modified_bessel_k0_out(y, x);
}

/// Scaled modified Bessel function of the second kind of order 1.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.scaled_modified_bessel_k1.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::scaled_modified_bessel_k1(x);
/// ```
inline Tensor scaled_modified_bessel_k1(const Tensor& x) {
  return smith::special_scaled_modified_bessel_k1(x);
}

inline Tensor& scaled_modified_bessel_k1_out(Tensor& y, const Tensor& x) {
  return smith::special_scaled_modified_bessel_k1_out(y, x);
}

/// Shifted Chebyshev polynomial of the first kind.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.shifted_chebyshev_polynomial_t.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::shifted_chebyshev_polynomial_t(x, n);
/// ```
inline Tensor shifted_chebyshev_polynomial_t(const Tensor& x, const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_t(x, n);
}

inline Tensor shifted_chebyshev_polynomial_t(const Scalar& x, const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_t(x, n);
}

inline Tensor shifted_chebyshev_polynomial_t(const Tensor& x, const Scalar& n) {
  return smith::special_shifted_chebyshev_polynomial_t(x, n);
}

inline Tensor& shifted_chebyshev_polynomial_t_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_t_out(output, x, n);
}

inline Tensor& shifted_chebyshev_polynomial_t_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_t_out(output, x, n);
}

inline Tensor& shifted_chebyshev_polynomial_t_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_shifted_chebyshev_polynomial_t_out(output, x, n);
}

/// Shifted Chebyshev polynomial of the second kind.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.shifted_chebyshev_polynomial_u.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::shifted_chebyshev_polynomial_u(x, n);
/// ```
inline Tensor shifted_chebyshev_polynomial_u(const Tensor& x, const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_u(x, n);
}

inline Tensor shifted_chebyshev_polynomial_u(const Scalar& x, const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_u(x, n);
}

inline Tensor shifted_chebyshev_polynomial_u(const Tensor& x, const Scalar& n) {
  return smith::special_shifted_chebyshev_polynomial_u(x, n);
}

inline Tensor& shifted_chebyshev_polynomial_u_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_u_out(output, x, n);
}

inline Tensor& shifted_chebyshev_polynomial_u_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_u_out(output, x, n);
}

inline Tensor& shifted_chebyshev_polynomial_u_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_shifted_chebyshev_polynomial_u_out(output, x, n);
}

/// Shifted Chebyshev polynomial of the third kind.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.shifted_chebyshev_polynomial_v.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::shifted_chebyshev_polynomial_v(x, n);
/// ```
inline Tensor shifted_chebyshev_polynomial_v(const Tensor& x, const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_v(x, n);
}

inline Tensor shifted_chebyshev_polynomial_v(const Scalar& x, const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_v(x, n);
}

inline Tensor shifted_chebyshev_polynomial_v(const Tensor& x, const Scalar& n) {
  return smith::special_shifted_chebyshev_polynomial_v(x, n);
}

inline Tensor& shifted_chebyshev_polynomial_v_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_v_out(output, x, n);
}

inline Tensor& shifted_chebyshev_polynomial_v_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_v_out(output, x, n);
}

inline Tensor& shifted_chebyshev_polynomial_v_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_shifted_chebyshev_polynomial_v_out(output, x, n);
}

/// Shifted Chebyshev polynomial of the fourth kind.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.shifted_chebyshev_polynomial_w.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
/// auto n = smith::randn(128, dtype=kDouble);
///
/// smith::special::shifted_chebyshev_polynomial_w(x, n);
/// ```
inline Tensor shifted_chebyshev_polynomial_w(const Tensor& x, const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_w(x, n);
}

inline Tensor shifted_chebyshev_polynomial_w(const Scalar& x, const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_w(x, n);
}

inline Tensor shifted_chebyshev_polynomial_w(const Tensor& x, const Scalar& n) {
  return smith::special_shifted_chebyshev_polynomial_w(x, n);
}

inline Tensor& shifted_chebyshev_polynomial_w_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_w_out(output, x, n);
}

inline Tensor& shifted_chebyshev_polynomial_w_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return smith::special_shifted_chebyshev_polynomial_w_out(output, x, n);
}

inline Tensor& shifted_chebyshev_polynomial_w_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return smith::special_shifted_chebyshev_polynomial_w_out(output, x, n);
}

/// Spherical Bessel function of the first kind of order 0.
///
/// See
/// https://blacksmith.org/docs/main/special.html#smith.special.spherical_bessel_j0.
///
/// Example:
///
/// ```
/// auto x = smith::randn(128, dtype=kDouble);
///
/// smith::special::spherical_bessel_j0(x);
/// ```
inline Tensor spherical_bessel_j0(const Tensor& x) {
  return smith::special_spherical_bessel_j0(x);
}

inline Tensor& spherical_bessel_j0_out(Tensor& y, const Tensor& x) {
  return smith::special_spherical_bessel_j0_out(y, x);
}
} // namespace smith::special
