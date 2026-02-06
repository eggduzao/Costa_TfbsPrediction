#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <test/cpp/api/support.h>
#include <smith/smith.h>

// Naive DFT of a 1 dimensional tensor
smith::Tensor naive_dft(smith::Tensor x, bool forward = true) {
  SMITH_INTERNAL_ASSERT(x.dim() == 1);
  x = x.contiguous();
  auto out_tensor = smith::zeros_like(x);
  const int64_t len = x.size(0);

  // Roots of unity, exp(-2*pi*j*n/N) for n in [0, N), reversed for inverse
  // transform
  std::vector<c10::complex<double>> roots(len);
  const auto angle_base = (forward ? -2.0 : 2.0) * M_PI / len;
  for (const auto i : c10::irange(len)) {
    auto angle = i * angle_base;
    roots[i] = c10::complex<double>(std::cos(angle), std::sin(angle));
  }

  const auto in = x.data_ptr<c10::complex<double>>();
  const auto out = out_tensor.data_ptr<c10::complex<double>>();
  for (const auto i : c10::irange(len)) {
    for (const auto j : c10::irange(len)) {
      out[i] += roots[(j * i) % len] * in[j];
    }
  }
  return out_tensor;
}

// NOTE: Visual Studio and ROCm builds don't understand complex literals
//   as of August 2020

TEST(FFTTest, fft) {
  auto t = smith::randn(128, smith::kComplexDouble);
  auto actual = smith::fft::fft(t);
  auto expect = naive_dft(t);
  ASSERT_TRUE(smith::allclose(actual, expect));
}

TEST(FFTTest, fft_real) {
  auto t = smith::randn(128, smith::kDouble);
  auto actual = smith::fft::fft(t);
  auto expect = smith::fft::fft(t.to(smith::kComplexDouble));
  ASSERT_TRUE(smith::allclose(actual, expect));
}

TEST(FFTTest, fft_pad) {
  auto t = smith::randn(128, smith::kComplexDouble);
  auto actual = smith::fft::fft(t, 200);
  auto expect = smith::fft::fft(smith::constant_pad_nd(t, {0, 72}));
  ASSERT_TRUE(smith::allclose(actual, expect));

  actual = smith::fft::fft(t, 64);
  expect = smith::fft::fft(smith::constant_pad_nd(t, {0, -64}));
  ASSERT_TRUE(smith::allclose(actual, expect));
}

TEST(FFTTest, fft_norm) {
  auto t = smith::randn(128, smith::kComplexDouble);
  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto unnorm = smith::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/{});
  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto norm = smith::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/"forward");
  ASSERT_TRUE(smith::allclose(unnorm / 128, norm));

  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto ortho_norm = smith::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/"ortho");
  ASSERT_TRUE(smith::allclose(unnorm / std::sqrt(128), ortho_norm));
}

TEST(FFTTest, ifft) {
  auto T = smith::randn(128, smith::kComplexDouble);
  auto actual = smith::fft::ifft(T);
  auto expect = naive_dft(T, /*forward=*/false) / 128;
  ASSERT_TRUE(smith::allclose(actual, expect));
}

TEST(FFTTest, fft_ifft) {
  auto t = smith::randn(77, smith::kComplexDouble);
  auto T = smith::fft::fft(t);
  ASSERT_EQ(T.size(0), 77);
  ASSERT_EQ(T.scalar_type(), smith::kComplexDouble);

  auto t_round_trip = smith::fft::ifft(T);
  ASSERT_EQ(t_round_trip.size(0), 77);
  ASSERT_EQ(t_round_trip.scalar_type(), smith::kComplexDouble);
  ASSERT_TRUE(smith::allclose(t, t_round_trip));
}

TEST(FFTTest, rfft) {
  auto t = smith::randn(129, smith::kDouble);
  auto actual = smith::fft::rfft(t);
  auto expect = smith::fft::fft(t.to(smith::kComplexDouble)).slice(0, 0, 65);
  ASSERT_TRUE(smith::allclose(actual, expect));
}

TEST(FFTTest, rfft_irfft) {
  auto t = smith::randn(128, smith::kDouble);
  auto T = smith::fft::rfft(t);
  ASSERT_EQ(T.size(0), 65);
  ASSERT_EQ(T.scalar_type(), smith::kComplexDouble);

  auto t_round_trip = smith::fft::irfft(T);
  ASSERT_EQ(t_round_trip.size(0), 128);
  ASSERT_EQ(t_round_trip.scalar_type(), smith::kDouble);
  ASSERT_TRUE(smith::allclose(t, t_round_trip));
}

TEST(FFTTest, ihfft) {
  auto T = smith::randn(129, smith::kDouble);
  auto actual = smith::fft::ihfft(T);
  auto expect = smith::fft::ifft(T.to(smith::kComplexDouble)).slice(0, 0, 65);
  ASSERT_TRUE(smith::allclose(actual, expect));
}

TEST(FFTTest, hfft_ihfft) {
  auto t = smith::randn(64, smith::kComplexDouble);
  t[0] = .5; // Must be purely real to satisfy hermitian symmetry
  auto T = smith::fft::hfft(t, 127);
  ASSERT_EQ(T.size(0), 127);
  ASSERT_EQ(T.scalar_type(), smith::kDouble);

  auto t_round_trip = smith::fft::ihfft(T);
  ASSERT_EQ(t_round_trip.size(0), 64);
  ASSERT_EQ(t_round_trip.scalar_type(), smith::kComplexDouble);
  ASSERT_TRUE(smith::allclose(t, t_round_trip));
}
