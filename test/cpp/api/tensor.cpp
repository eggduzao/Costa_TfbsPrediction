#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

#include <c10/util/irange.h>
#include <smith/smith.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include <test/cpp/common/support.h>

using namespace smith::test;

template <typename T>
bool exactly_equal(at::Tensor left, T right) {
  return left.item<T>() == right;
}

template <typename T>
bool almost_equal(at::Tensor left, T right, double tolerance = 1e-4) {
  return std::abs(left.item<T>() - right) < tolerance;
}

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  ASSERT_TRUE(                                                             \
      tensor.device().type() == at::Device((device_), (index_)).type());   \
  ASSERT_TRUE(                                                             \
      tensor.device().index() == at::Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.dtype(), (type_));                                      \
  ASSERT_TRUE(tensor.layout() == (layout_))

TEST(TensorTest, ToDtype) {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  tensor = tensor.to(at::kInt);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  tensor = tensor.to(at::kChar);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kChar, at::kStrided);

  tensor = tensor.to(at::kDouble);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kInt));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kChar));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kChar, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kDouble));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);
}

TEST(TensorTest, ToTensorAndTensorAttributes) {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  auto other = at::empty({3, 4}, at::kInt);
  tensor = tensor.to(other);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  other = at::empty({3, 4}, at::kDouble);
  tensor = tensor.to(other.dtype());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);
  tensor = tensor.to(other.device());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);

  other = at::empty({3, 4}, at::kLong);
  tensor = tensor.to(other.device(), other.dtype());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kLong, at::kStrided);

  other = at::empty({3, 4}, at::kInt);
  tensor = tensor.to(other.options());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);
}

TEST(TensorTest, TensorAttributes) {
  std::vector<float> v = {1.0, 2.0, 3.0};
  auto options = c10::requires_grad(true);
  auto tensor = at::from_blob(v.data(), {3}, options);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
  ASSERT_TRUE(tensor.requires_grad());
}

// Not currently supported.
// TEST(TensorTest, ToLayout) {
//   auto tensor = at::empty({3, 4});
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
//
//   tensor = tensor.to(at::kSparse);
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kSparse);
//
//   tensor = tensor.to(at::kStrided);
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
// }

TEST(TensorTest, ToOptionsWithRequiresGrad) {
  {
    // Respects requires_grad
    auto tensor = smith::empty({3, 4}, at::requires_grad());
    ASSERT_TRUE(tensor.requires_grad());

    tensor = tensor.to(at::kDouble);
    ASSERT_TRUE(tensor.requires_grad());

    // Throws if requires_grad is set in TensorOptions
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_THROW(
        tensor.to(at::TensorOptions().requires_grad(true)), c10::Error);

    // Doesn't throw if requires_grad is not set
    tensor.to(at::TensorOptions());
    tensor.to(at::TensorOptions().requires_grad(false));
  }
  {
    auto tensor = smith::empty({3, 4});
    ASSERT_FALSE(tensor.requires_grad());

    // Respects requires_grad
    tensor = tensor.to(at::kDouble);
    ASSERT_FALSE(tensor.requires_grad());

    // Throws if requires_grad is set in TensorOptions
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_THROW(
        tensor.to(at::TensorOptions().requires_grad(true)), c10::Error);

    // Doesn't throw if requires_grad is not set
    tensor.to(at::TensorOptions());
    tensor.to(at::TensorOptions().requires_grad(false));
  }
}

TEST(TensorTest, ToDoesNotCopyWhenOptionsAreAllTheSame) {
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(at::kFloat);
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(tensor.options());
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(tensor.dtype());
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(tensor.device());
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(tensor);
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
}

TEST(TensorTest, AtTensorCtorScalar) {
  auto tensor = at::tensor(123);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_EQ(tensor[0].item<int32_t>(), 123);

  tensor = at::tensor(123.456f);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  ASSERT_TRUE(almost_equal(tensor[0], 123.456f));

  tensor = at::tensor(123.456);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  ASSERT_TRUE(almost_equal(tensor[0], 123.456));

  tensor = at::tensor(123, at::dtype(at::kFloat)) + 0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  ASSERT_TRUE(almost_equal(tensor[0], 123.5));

  tensor = at::tensor(c10::complex<float>(1.0, 2.0)) + 0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kComplexFloat);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<float>(1.5, 2.0)));

  tensor =
      at::tensor(c10::complex<float>(1.0, 2.0), at::dtype(at::kComplexFloat)) +
      0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kComplexFloat);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<float>(1.5, 2.0)));

  tensor = at::tensor(c10::complex<double>(1.0, 2.0)) + 0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5, 2.0)));

  tensor =
      at::tensor(c10::complex<float>(1.0, 2.0), at::dtype(at::kComplexDouble)) +
      0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5, 2.0)));
}

TEST(TensorTest, AtTensorCtorSingleDim) {
  auto tensor = at::tensor({1, 2, 3});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = at::tensor(std::vector<int>({1, 2, 3}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = at::tensor({1.5, 2.25, 3.125});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = at::tensor(
      {c10::complex<float>(1.5, 0.15),
       c10::complex<float>(1.5, 0.15),
       c10::complex<float>(3.125, 0.3125)});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kComplexFloat);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<float>(1.5, 0.15)));
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<float>(1.5, 0.15)));
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<float>(3.125, 0.3125)));

  tensor = at::tensor(
      {c10::complex<double>(1.5, 0.15),
       c10::complex<double>(1.5, 0.15),
       c10::complex<double>(3.125, 0.3125)});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5, 0.15)));
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(1.5, 0.15)));
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.125, 0.3125)));

  tensor = at::tensor({1.1, 2.2, 3.3}, at::dtype(at::kInt));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_EQ(tensor.layout(), at::kStrided);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = at::tensor(std::vector<double>({1.5, 2.25, 3.125}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = at::tensor(std::vector<c10::complex<float>>(
      {c10::complex<float>(1.5, 0.15),
       c10::complex<float>(1.5, 0.15),
       c10::complex<float>(3.125, 0.3125)}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kComplexFloat);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<float>(1.5, 0.15)));
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<float>(1.5, 0.15)));
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<float>(3.125, 0.3125)));

  tensor = at::tensor(std::vector<c10::complex<double>>(
      {c10::complex<double>(1.5, 0.15),
       c10::complex<double>(1.5, 0.15),
       c10::complex<double>(3.125, 0.3125)}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5, 0.15)));
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(1.5, 0.15)));
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.125, 0.3125)));

  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  tensor = at::tensor(v);
  ASSERT_EQ(tensor.numel(), v.size());
  ASSERT_EQ(tensor.dtype(), at::kInt);
  for (const auto i : c10::irange(v.size())) {
    ASSERT_TRUE(exactly_equal(tensor[i], v.at(i)));
  }

  std::vector<double> w = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0};
  tensor = at::tensor(w);
  ASSERT_EQ(tensor.numel(), w.size());
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  for (const auto i : c10::irange(w.size())) {
    ASSERT_TRUE(almost_equal(tensor[i], w.at(i)));
  }

  std::vector<c10::complex<double>> x = {
      {1.1, -1.1},
      {2.2, -2.2},
      {3.3, -3.3},
      {4.4, -4.4},
      {5.5, -5.5},
      {6.6, -6.6},
      {7.7, -7.7},
      {8.8, -8.8},
      {9.9, -9.9},
      {10.0, -10.0}};
  tensor = at::tensor(x);
  ASSERT_EQ(tensor.numel(), x.size());
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  for (const auto i : c10::irange(x.size())) {
    ASSERT_TRUE(almost_equal(tensor[i], x.at(i)));
  }
}

TEST(TensorTest, AtTensorCastRealToComplex) {
  auto tensor =
      at::tensor(std::vector<double>({1.5, 2.5, 3.5}), at::kComplexDouble);
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(2.5)));
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.5)));

  tensor = at::tensor({1.5, 2.5, 3.5}, at::kComplexDouble);
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(2.5)));
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.5)));

  tensor = at::tensor(1.5, at::kComplexDouble);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
}

TEST(TensorTest, AtTensorCastComplexToRealErrorChecks) {
  {
    ASSERT_THROWS_WITH(
        at::tensor(c10::complex<float>(0.1, 0.2), at::kFloat),
        "\"tensor_cpu\" not implemented for 'Float'");
  }
  {
    ASSERT_THROWS_WITH(
        at::tensor({c10::complex<float>(0.1, 0.2)}, at::kFloat),
        "\"tensor_cpu\" not implemented for 'Float'");
  }
  {
    ASSERT_THROWS_WITH(
        at::tensor(
            std::vector<c10::complex<float>>{c10::complex<float>(0.1, 0.2)},
            at::kFloat),
        "\"tensor_cpu\" not implemented for 'Float'");
  }
}

TEST(TensorTest, SmithTensorCtorScalarIntegralType) {
  auto tensor = smith::tensor(123);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_EQ(tensor.item<int64_t>(), 123);
}

void test_SmithTensorCtorScalarFloatingType_expected_dtype(
    c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  auto tensor = smith::tensor(123.456f);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor, 123.456f));

  tensor = smith::tensor(123.456);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor, 123.456));

  tensor = smith::tensor({123.456});
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor[0], 123.456));
}

TEST(TensorTest, SmithTensorCtorScalarFloatingType) {
  test_SmithTensorCtorScalarFloatingType_expected_dtype(
      /*default_dtype=*/smith::kFloat);
  test_SmithTensorCtorScalarFloatingType_expected_dtype(
      /*default_dtype=*/smith::kDouble);
}

TEST(TensorTest, SmithTensorCtorScalarBoolType) {
  auto tensor = smith::tensor(true);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  ASSERT_EQ(tensor.dtype(), at::kBool);
  ASSERT_TRUE(exactly_equal(tensor, true));

  tensor = smith::tensor({true});
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1}));
  ASSERT_EQ(tensor.dtype(), at::kBool);
  ASSERT_TRUE(exactly_equal(tensor[0], true));
}

TEST(TensorTest, SmithTensorCtorSingleDimIntegralType) {
  auto tensor = smith::tensor({1, 2, 3});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = smith::tensor(at::ArrayRef<int>({1, 2, 3}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = smith::tensor(std::vector<int>({1, 2, 3}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = smith::tensor(at::ArrayRef<int64_t>({1, 2, 3}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = smith::tensor(std::vector<int64_t>({1, 2, 3}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));
}

void test_SmithTensorCtorSingleDimFloatingType_expected_dtype(
    c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  auto tensor = smith::tensor({1.5, 2.25, 3.125});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = smith::tensor({1.5f, 2.25f, 3.125f});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5f));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25f));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125f));

  tensor = smith::tensor(at::ArrayRef<float>({1.5f, 2.25f, 3.125f}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = smith::tensor(std::vector<float>({1.5f, 2.25f, 3.125f}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = smith::tensor(at::ArrayRef<double>({1.5, 2.25, 3.125}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = smith::tensor(std::vector<double>({1.5, 2.25, 3.125}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));
}

TEST(TensorTest, SmithTensorCtorSingleDimFloatingType) {
  test_SmithTensorCtorSingleDimFloatingType_expected_dtype(
      /*default_dtype=*/smith::kFloat);
  test_SmithTensorCtorSingleDimFloatingType_expected_dtype(
      /*default_dtype=*/smith::kDouble);
}

TEST(TensorTest, SmithTensorCtorSingleDimBoolType) {
  auto tensor = smith::tensor({true, false, true});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kBool);
  ASSERT_TRUE(exactly_equal(tensor[0], true));
  ASSERT_TRUE(exactly_equal(tensor[1], false));
  ASSERT_TRUE(exactly_equal(tensor[2], true));

  tensor = smith::tensor(at::ArrayRef<bool>({true, false, true}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kBool);
  ASSERT_TRUE(exactly_equal(tensor[0], true));
  ASSERT_TRUE(exactly_equal(tensor[1], false));
  ASSERT_TRUE(exactly_equal(tensor[2], true));
}

TEST(TensorTest, SmithTensorCtorMultiDimIntegralType) {
  {
    auto tensor = smith::tensor({{1, 2}});
    ASSERT_EQ(tensor.dtype(), smith::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 3, smith::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{1}, {2}});
    ASSERT_EQ(tensor.dtype(), smith::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 1}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 3, smith::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{{1, 2}}});
    ASSERT_EQ(tensor.dtype(), smith::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 2}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 3, smith::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{{1}, {2}}});
    ASSERT_EQ(tensor.dtype(), smith::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2, 1}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 3, smith::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{1, 2}, {3, 4}});
    ASSERT_EQ(tensor.dtype(), smith::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 2}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 5, smith::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{{{{{{{{{1}}}}}}}}}});
    ASSERT_EQ(tensor.dtype(), smith::kLong);
    ASSERT_EQ(
        tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::full({1}, 1, smith::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{{{{{{{{{1, 2}}}}}}}}}});
    ASSERT_EQ(tensor.dtype(), smith::kLong);
    ASSERT_EQ(
        tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 2}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 3, smith::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
}

void test_SmithTensorCtorMultiDimFloatingType_expected_dtype(
    c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);
  {
    auto tensor = smith::tensor({{1.0, 2.0}});
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 3, default_dtype).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor(
        {{{{{{{{1.0, 2.0, 3.0}}}}},
           {{{{{4.0, 5.0, 6.0}}}}},
           {{{{{7.0, 8.0, 9.0}}}}}}}});
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 3, 1, 1, 1, 1, 3}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 10, default_dtype).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
}

TEST(TensorTest, SmithTensorCtorMultiDimFloatingType) {
  test_SmithTensorCtorMultiDimFloatingType_expected_dtype(
      /*default_dtype=*/smith::kFloat);
  test_SmithTensorCtorMultiDimFloatingType_expected_dtype(
      /*default_dtype=*/smith::kDouble);
}

TEST(TensorTest, SmithTensorCtorMultiDimBoolType) {
  {
    auto tensor = smith::tensor({{true, false}});
    ASSERT_EQ(tensor.dtype(), smith::kBool);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    auto expected = smith::empty(tensor.sizes(), smith::kBool);
    expected[0][0] = true;
    expected[0][1] = false;
    ASSERT_TRUE(smith::equal(tensor, expected));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{true}, {false}});
    ASSERT_EQ(tensor.dtype(), smith::kBool);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 1}));
    auto expected = smith::empty(tensor.sizes(), smith::kBool);
    expected[0][0] = true;
    expected[1][0] = false;
    ASSERT_TRUE(smith::equal(tensor, expected));
    ASSERT_FALSE(tensor.requires_grad());
  }
}

TEST(TensorTest, SmithTensorCtorMultiDimWithOptions) {
  {
    auto tensor = smith::tensor({{1, 2}}, smith::dtype(smith::kInt));
    ASSERT_EQ(tensor.dtype(), smith::kInt);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 3, smith::kInt).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor(
        {{1, 2}, {3, 4}}, smith::dtype(smith::kFloat).requires_grad(true));
    ASSERT_EQ(tensor.dtype(), smith::kFloat);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 2}));
    ASSERT_TRUE(smith::allclose(
        tensor, smith::arange(1, 5, smith::kFloat).view(tensor.sizes())));
    ASSERT_TRUE(tensor.requires_grad());
  }
}

TEST(TensorTest, SmithTensorCtorMultiDimErrorChecks) {
  {
    ASSERT_THROWS_WITH(
        smith::tensor({{{2, 3, 4}, {{5, 6}, {7}}}}),
        "Expected all sub-lists to have sizes: 2 (e.g. {5, 6}), but got sub-list {7} with sizes: 1");
  }
  {
    ASSERT_THROWS_WITH(
        smith::tensor({{{1, 2.0}, {1, 2.0}}}),
        "Expected all elements of the tensor to have the same scalar type: Int, but got element of scalar type: Double");
  }
  {
    ASSERT_THROWS_WITH(
        smith::tensor({{{true, 2.0, 3}, {true, 2.0, 3}}}),
        "Expected all elements of the tensor to have the same scalar type: Bool, but got element of scalar type: Double");
  }
  {
    ASSERT_THROWS_WITH(
        smith::tensor({{{true}, {2}}}),
        "Expected all elements of the tensor to have the same scalar type: Bool, but got element of scalar type: Int");
  }
  {
    ASSERT_THROWS_WITH(
        smith::tensor({{{true, 2}}}),
        "Expected all elements of the tensor to have the same scalar type: Bool, but got element of scalar type: Int");
  }
}

TEST(TensorTest, SmithTensorCastRealToComplex) {
  auto tensor = smith::tensor(
      std::vector<double>({1.5, 2.5, 3.5}), smith::kComplexDouble);
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), smith::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(2.5)));
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.5)));

  tensor = smith::tensor({1.5, 2.5, 3.5}, smith::kComplexDouble);
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), smith::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(2.5)));
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.5)));

  tensor = smith::tensor(1.5, smith::kComplexDouble);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), smith::kComplexDouble);
  ASSERT_TRUE(almost_equal(tensor, c10::complex<double>(1.5)));
}

TEST(TensorTest, SmithTensorCastComplexToRealErrorChecks) {
  {
    ASSERT_THROWS_WITH(
        smith::tensor(c10::complex<float>(0.1, 0.2), smith::kFloat),
        "value cannot be converted to type float without overflow");
  }
  {
    ASSERT_THROWS_WITH(
        smith::tensor(
            {c10::complex<float>(0.1, 0.2), c10::complex<float>(0.3, 0.4)},
            smith::kFloat),
        "value cannot be converted to type float without overflow");
  }
  {
    ASSERT_THROWS_WITH(
        smith::tensor(
            std::vector<c10::complex<float>>{
                c10::complex<float>(0.1, 0.2), c10::complex<float>(0.3, 0.4)},
            smith::kFloat),
        "can not do smith::tensor(complex, dtype=non-complex) because complex can not be casted to real number without loss of information");
  }
}

void test_SmithTensorCtorMultiDim_CUDA_expected_dtype(
    c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  auto tensor = smith::tensor(
      {{{{{{{{1.0, 2.0, 3.0}}}}},
         {{{{{4.0, 5.0, 6.0}}}}},
         {{{{{7.0, 8.0, 9.0}}}}}}}},
      smith::dtype(default_dtype).device(smith::kCUDA));
  ASSERT_TRUE(tensor.device().is_cuda());
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 3, 1, 1, 1, 1, 3}));
  ASSERT_TRUE(smith::allclose(
      tensor,
      smith::arange(1, 10, default_dtype)
          .view(tensor.sizes())
          .to(smith::kCUDA)));
  ASSERT_FALSE(tensor.requires_grad());
}

TEST(TensorTest, SmithTensorCtorMultiDim_CUDA) {
  test_SmithTensorCtorMultiDim_CUDA_expected_dtype(
      /*default_dtype=*/smith::kFloat);
  test_SmithTensorCtorMultiDim_CUDA_expected_dtype(
      /*default_dtype=*/smith::kDouble);
}

void test_SmithTensorCtorZeroSizedDim_expected_dtype(
    c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);
  {
    auto tensor = smith::tensor({});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{}, {}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{{}, {}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{{}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{{{{{{{}}}}}}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{{{{{{{}}}}, {{{{}}}}}}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 1, 2, 1, 1, 1, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = smith::tensor({{{{{{{{{{}}}}}}}}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(
        tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
}

TEST(TensorTest, SmithTensorCtorZeroSizedDim) {
  test_SmithTensorCtorZeroSizedDim_expected_dtype(
      /*default_dtype=*/smith::kFloat);
  test_SmithTensorCtorZeroSizedDim_expected_dtype(
      /*default_dtype=*/smith::kDouble);
}

void test_SmithTensorCtorWithoutSpecifyingDtype_expected_dtype(
    c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  ASSERT_EQ(smith::tensor({1., 2., 3.}).dtype(), default_dtype);
  ASSERT_EQ(smith::tensor({{1., 2., 3.}}).dtype(), default_dtype);
  ASSERT_EQ(
      smith::tensor({1., 2., 3.}, smith::TensorOptions()).dtype(),
      default_dtype);
  ASSERT_EQ(
      smith::tensor({{1., 2., 3.}}, smith::TensorOptions()).dtype(),
      default_dtype);
}

TEST(TensorTest, SmithTensorCtorWithoutSpecifyingDtype) {
  ASSERT_EQ(smith::tensor({1, 2, 3}).dtype(), smith::kLong);
  ASSERT_EQ(smith::tensor({{1, 2, 3}}).dtype(), smith::kLong);
  ASSERT_EQ(
      smith::tensor({1, 2, 3}, smith::TensorOptions()).dtype(), smith::kLong);
  ASSERT_EQ(
      smith::tensor({{1, 2, 3}}, smith::TensorOptions()).dtype(), smith::kLong);

  test_SmithTensorCtorWithoutSpecifyingDtype_expected_dtype(
      /*default_dtype=*/smith::kFloat);
  test_SmithTensorCtorWithoutSpecifyingDtype_expected_dtype(
      /*default_dtype=*/smith::kDouble);
}

void test_SmithTensorCtorWithNonDtypeOptions_expected_dtype(
    c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  ASSERT_EQ(
      smith::tensor({1, 2, 3}, smith::TensorOptions()).dtype(), smith::kLong);
  ASSERT_EQ(
      smith::tensor(at::ArrayRef<int>({1, 2, 3}), smith::TensorOptions())
          .dtype(),
      smith::kLong);
  ASSERT_EQ(
      smith::tensor(std::vector<int>({1, 2, 3}), smith::TensorOptions())
          .dtype(),
      smith::kLong);

  ASSERT_EQ(
      smith::tensor({1., 2., 3.}, smith::TensorOptions()).dtype(),
      default_dtype);
  ASSERT_EQ(
      smith::tensor(at::ArrayRef<double>({1., 2., 3.}), smith::TensorOptions())
          .dtype(),
      default_dtype);
  ASSERT_EQ(
      smith::tensor(std::vector<double>({1., 2., 3.}), smith::TensorOptions())
          .dtype(),
      default_dtype);

  ASSERT_EQ(
      smith::tensor({1.f, 2.f, 3.f}, smith::TensorOptions()).dtype(),
      default_dtype);
  ASSERT_EQ(
      smith::tensor(
          at::ArrayRef<float>({1.f, 2.f, 3.f}), smith::TensorOptions())
          .dtype(),
      default_dtype);
  ASSERT_EQ(
      smith::tensor(std::vector<float>({1.f, 2.f, 3.f}), smith::TensorOptions())
          .dtype(),
      default_dtype);
}

TEST(TensorTest, SmithTensorCtorWithNonDtypeOptions) {
  test_SmithTensorCtorWithNonDtypeOptions_expected_dtype(
      /*default_dtype=*/smith::kFloat);
  test_SmithTensorCtorWithNonDtypeOptions_expected_dtype(
      /*default_dtype=*/smith::kDouble);
}

void test_Arange_expected_dtype(c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  ASSERT_EQ(smith::arange(0., 5).dtype(), default_dtype);
}

TEST(TensorTest, Arange) {
  {
    auto x = smith::arange(0, 5);
    ASSERT_EQ(x.dtype(), smith::kLong);
  }
  test_Arange_expected_dtype(smith::kFloat);
  test_Arange_expected_dtype(smith::kDouble);
}

TEST(TensorTest, PrettyPrintTensorDataContainer) {
  {
    ASSERT_EQ(c10::str(smith::detail::TensorDataContainer(1.1)), "1.1");
  }
  {
    ASSERT_EQ(
        c10::str(smith::detail::TensorDataContainer({1.1, 2.2})), "{1.1, 2.2}");
  }
  {
    ASSERT_EQ(
        c10::str(smith::detail::TensorDataContainer({{1, 2}, {3, 4}})),
        "{{1, 2}, {3, 4}}");
  }
  {
    ASSERT_EQ(
        c10::str(smith::detail::TensorDataContainer(
            {{{{{{{{1.1, 2.2, 3.3}}}}},
               {{{{{4.4, 5.5, 6.6}}}}},
               {{{{{7.7, 8.8, 9.9}}}}}}}})),
        "{{{{{{{{1.1, 2.2, 3.3}}}}}, {{{{{4.4, 5.5, 6.6}}}}}, {{{{{7.7, 8.8, 9.9}}}}}}}}");
  }
  {
    ASSERT_EQ(
        c10::str(smith::detail::TensorDataContainer({{{{{{{{{{1}}}}}}}}}})),
        "{{{{{{{{{{1}}}}}}}}}}");
  }
  {
    ASSERT_EQ(
        c10::str(smith::detail::TensorDataContainer({{{{{{{{{{}}}}}}}}}})),
        "{{{{{{{{{{}}}}}}}}}}");
  }
  {
    ASSERT_EQ(
        c10::str(smith::detail::TensorDataContainer({{{{{{{{{{1, 2}}}}}}}}}})),
        "{{{{{{{{{{1, 2}}}}}}}}}}");
  }
  {
    ASSERT_EQ(
        c10::str(smith::detail::TensorDataContainer(
            at::ArrayRef<double>({1.1, 2.2}))),
        "{1.1, 2.2}");
  }
  {
    ASSERT_EQ(
        c10::str(smith::detail::TensorDataContainer(
            std::vector<double>({1.1, 2.2}))),
        "{1.1, 2.2}");
  }
}

TEST(TensorTest, TensorDataContainerCallingAccessorOfWrongType) {
  {
    ASSERT_THROWS_WITH(
        smith::detail::TensorDataContainer(1.1).init_list(),
        "Can only call `init_list()` on a TensorDataContainer that has `is_init_list() == true`");
    ASSERT_THROWS_WITH(
        smith::detail::TensorDataContainer(1.1).tensor(),
        "Can only call `tensor()` on a TensorDataContainer that has `is_tensor() == true`");
  }
  {
    ASSERT_THROWS_WITH(
        smith::detail::TensorDataContainer({1.1, 2.2}).scalar(),
        "Can only call `scalar()` on a TensorDataContainer that has `is_scalar() == true`");
    ASSERT_THROWS_WITH(
        smith::detail::TensorDataContainer({1.1, 2.2}).tensor(),
        "Can only call `tensor()` on a TensorDataContainer that has `is_tensor() == true`");
  }
  {
    ASSERT_THROWS_WITH(
        smith::detail::TensorDataContainer(at::ArrayRef<double>({1.1, 2.2}))
            .scalar(),
        "Can only call `scalar()` on a TensorDataContainer that has `is_scalar() == true`");
    ASSERT_THROWS_WITH(
        smith::detail::TensorDataContainer(at::ArrayRef<double>({1.1, 2.2}))
            .init_list(),
        "Can only call `init_list()` on a TensorDataContainer that has `is_init_list() == true`");
  }
}

TEST(TensorTest, FromBlob) {
  std::vector<double> v = {1.0, 2.0, 3.0};
  auto tensor = smith::from_blob(
      v.data(), v.size(), smith::dtype(smith::kFloat64).requires_grad(true));
  ASSERT_TRUE(tensor.requires_grad());
  ASSERT_EQ(tensor.dtype(), smith::kFloat64);
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor[0].item<double>(), 1);
  ASSERT_EQ(tensor[1].item<double>(), 2);
  ASSERT_EQ(tensor[2].item<double>(), 3);
  // Above syntax did not copy the data, and has nullptr deleter context.
  ASSERT_EQ(tensor.storage().data_ptr().get_context(), nullptr);
}

TEST(TensorTest, FromBlobUsesDeleter) {
  bool called = false;
  {
    std::vector<int32_t> v = {1, 2, 3};
    auto tensor = smith::from_blob(
        v.data(),
        v.size(),
        /*deleter=*/[&called](void* data) { called = true; },
        smith::kInt32);
  }
  ASSERT_TRUE(called);
}

TEST(TensorTest, FromBlobWithStrides) {
  // clang-format off
  std::vector<int32_t> v = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };
  // clang-format on
  auto tensor = smith::from_blob(
      v.data(),
      /*sizes=*/{3, 3},
      /*strides=*/{1, 3},
      smith::kInt32);
  ASSERT_EQ(tensor.dtype(), smith::kInt32);
  ASSERT_EQ(tensor.numel(), 9);
  const std::vector<int64_t> expected_strides = {1, 3};
  ASSERT_EQ(tensor.strides(), expected_strides);
  for (const auto i : c10::irange(tensor.size(0))) {
    for (const auto j : c10::irange(tensor.size(1))) {
      // NOTE: This is column major because the strides are swapped.
      EXPECT_EQ(tensor[i][j].item<int32_t>(), 1 + (j * tensor.size(1)) + i);
    }
  }
}

TEST(TensorTest, Item) {
  {
    smith::Tensor tensor = smith::tensor(3.14);
    smith::Scalar scalar = tensor.item();
    ASSERT_NEAR(scalar.to<float>(), 3.14, 1e-5);
  }
  {
    smith::Tensor tensor = smith::tensor(123);
    smith::Scalar scalar = tensor.item();
    ASSERT_EQ(scalar.to<int>(), 123);
  }
}

TEST(TensorTest, Item_CUDA) {
  {
    smith::Tensor tensor = smith::tensor(3.14, smith::kCUDA);
    smith::Scalar scalar = tensor.item();
    ASSERT_NEAR(scalar.to<float>(), 3.14, 1e-5);
  }
  {
    smith::Tensor tensor = smith::tensor(123, smith::kCUDA);
    smith::Scalar scalar = tensor.item();
    ASSERT_EQ(scalar.to<int>(), 123);
  }
}

TEST(TensorTest, DataPtr) {
  auto tensor = at::empty({3, 4}, at::kFloat);
  auto tensor_not_copy = tensor.to(tensor.options());
  ASSERT_EQ(tensor_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  ASSERT_EQ(tensor_not_copy.data_ptr(), tensor.data_ptr());
}

TEST(TensorTest, Data) {
  const auto tensor = smith::rand({3, 3});
  ASSERT_TRUE(smith::equal(tensor, tensor.data()));
}

TEST(TensorTest, BackwardAndGrad) {
  auto x = smith::tensor({5}, smith::dtype(smith::kFloat).requires_grad(true));
  auto y = x * x;
  y.backward();
  ASSERT_EQ(x.grad().item<float>(), 10.0);
}

TEST(TensorTest, BackwardCreatesOnesGrad) {
  const auto x =
      smith::tensor({5}, smith::dtype(smith::kFloat).requires_grad(true));
  x.backward();
  ASSERT_TRUE(smith::equal(x.grad(), smith::ones_like(x)));
}

TEST(TensorTest, BackwardNonScalarOutputs) {
  auto x = smith::randn({5, 5}, smith::requires_grad());
  auto y = x * x;
  ASSERT_THROWS_WITH(
      y.backward(), "grad can be implicitly created only for scalar outputs");
}

TEST(TensorTest, BackwardComplexScalarOutput) {
  auto x = smith::randn({5, 5}, smith::requires_grad());
  auto y = (x * c10::Scalar(c10::complex<float>(0, 0.5))).sum();
  ASSERT_THROWS_WITH(
      y.backward(), "grad can be computed only for real scalar outputs");
}

TEST(TensorTest, IsLeaf) {
  auto x = smith::tensor({5}, smith::dtype(smith::kFloat).requires_grad(true));
  auto y = x * x;
  ASSERT_TRUE(x.is_leaf());
  ASSERT_FALSE(y.is_leaf());
}

TEST(TensorTest, OutputNr) {
  auto x = smith::tensor({5}, smith::dtype(smith::kFloat).requires_grad(true));
  auto y = x * x;
  ASSERT_EQ(x.output_nr(), 0);
  ASSERT_EQ(y.output_nr(), 0);
}

TEST(TensorTest, Version) {
  auto x = smith::ones(3);
  ASSERT_EQ(x._version(), 0);
  x.mul_(2);
  ASSERT_EQ(x._version(), 1);
  x.add_(1);
  ASSERT_EQ(x._version(), 2);
}

TEST(TensorTest, Detach) {
  auto x = smith::tensor({5}, smith::dtype(smith::kFloat).requires_grad(true));
  auto y = x * x;
  const auto y_detached = y.detach();
  ASSERT_FALSE(y.is_leaf());
  ASSERT_TRUE(y_detached.is_leaf());
  ASSERT_FALSE(y_detached.requires_grad());
}

TEST(TensorTest, DetachInplace) {
  auto x = smith::tensor({5}, smith::dtype(smith::kFloat).requires_grad(true));
  auto y = x * x;
  auto y_detached = y.detach_();
  ASSERT_TRUE(y.is_leaf());
  ASSERT_FALSE(y.requires_grad());
  ASSERT_TRUE(y_detached.is_leaf());
  ASSERT_FALSE(y_detached.requires_grad());
}

TEST(TensorTest, SetData) {
  auto x = smith::randn({5});
  auto y = smith::randn({5});
  ASSERT_FALSE(smith::equal(x, y));
  ASSERT_NE(x.data_ptr<float>(), y.data_ptr<float>());

  x.set_data(y);
  ASSERT_TRUE(smith::equal(x, y));
  ASSERT_EQ(x.data_ptr<float>(), y.data_ptr<float>());
}

TEST(TensorTest, RequiresGradInplace) {
  auto x = smith::tensor({5.0});
  x.requires_grad_(true);
  ASSERT_TRUE(x.requires_grad());

  auto y = x * x;
  ASSERT_THROWS_WITH(
      y.requires_grad_(false),
      "you can only change requires_grad flags of leaf variables.");

  x.requires_grad_(false);
  ASSERT_FALSE(x.requires_grad());

  const auto int_tensor =
      smith::tensor({5}, at::TensorOptions().dtype(smith::kInt));
  ASSERT_THROWS_WITH(
      int_tensor.requires_grad_(true),
      "Only Tensors of floating point and complex dtype can require gradients");
}

TEST(TensorTest, StdDimension) {
  // Test that std(0) doesn't select the std(unbiased=False) overload (gh-40287)
  auto x = smith::randn({4, 3});
  auto std = x.std(0);

  ASSERT_EQ(x.var(0).numel(), 3);
  ASSERT_EQ(x.std(0).numel(), 3);

  ASSERT_EQ(x.var(0, /*unbiased=*/true).numel(), 3);
  ASSERT_EQ(x.std(0, /*unbiased=*/true).numel(), 3);

  ASSERT_EQ(smith::var(x, 0).numel(), 3);
  ASSERT_EQ(std::get<0>(smith::var_mean(x, 0)).numel(), 3);
  ASSERT_EQ(smith::std(x, 0).numel(), 3);
  ASSERT_EQ(std::get<0>(smith::std_mean(x, 0)).numel(), 3);

  ASSERT_EQ(smith::var(x, 0, /*unbiased=*/true).numel(), 3);
  ASSERT_EQ(std::get<0>(smith::var_mean(x, 0, /*unbiased=*/true)).numel(), 3);
  ASSERT_EQ(smith::std(x, 0, /*unbiased=*/true).numel(), 3);
  ASSERT_EQ(std::get<0>(smith::std_mean(x, 0, /*unbiased=*/true)).numel(), 3);
}

TEST(TensorTest, ReshapeAlias) {
  // Tests the behavior of the _reshape_alias private operator so
  // that it matches the behavior of as_strided and view.
  auto x = smith::randn({3, 3});
  ASSERT_TRUE(smith::equal(
      smith::_reshape_alias(x, {2, 2}, {1, 2}),
      smith::as_strided(x, {2, 2}, {1, 2})));
  ASSERT_TRUE(smith::equal(smith::_reshape_alias(x, {9}, {1}), x.view({-1})));

  // Test that the backward works fine.
  auto y = smith::randn({3, 3}, smith::requires_grad(true));
  auto z = smith::clone(y).detach().requires_grad_(true);
  (y * y).view({-1}).mean().backward();
  smith::_reshape_alias((z * z), {9}, {1}).mean().backward();
  ASSERT_TRUE(smith::equal(y.grad(), z.grad()));
}

TEST(TensorTest, BackendMetadata) {
  // Tests ability to assign custom backend metadata to tensor.

  struct CustomBackendMetadata : public c10::BackendMeta {
    mutable bool cloned_{false}; // for testing this field will mutate when
                                 // clone() is called by shallow_copy_from.
    c10::intrusive_ptr<c10::BackendMeta> clone(
        const c10::intrusive_ptr<c10::BackendMeta>& ptr) const override {
      cloned_ = true;
      return c10::BackendMeta::clone(ptr);
    }
  };

  at::Tensor y;
  c10::intrusive_ptr<c10::BackendMeta> tmeta{};
  CustomBackendMetadata* custom_tmeta{nullptr};

  {
    auto x = smith::ones({3, 3});
    auto impl{x.unsafeGetTensorImpl()};
    ASSERT_TRUE(impl != nullptr);

    tmeta = impl->get_backend_meta_intrusive_ptr();
    ASSERT_TRUE(tmeta == nullptr);
    c10::intrusive_ptr<c10::BackendMeta> new_tmeta{
        std::unique_ptr<c10::BackendMeta>(new CustomBackendMetadata())};
    impl->set_backend_meta(new_tmeta);
    tmeta = impl->get_backend_meta_intrusive_ptr();
    ASSERT_TRUE(tmeta == new_tmeta);
    custom_tmeta = dynamic_cast<CustomBackendMetadata*>(tmeta.get());
    ASSERT_TRUE(custom_tmeta != nullptr);
    ASSERT_TRUE(custom_tmeta->cloned_ == false);
    y.unsafeGetTensorImpl()->shallow_copy_from(x.getIntrusivePtr());
  }

  ASSERT_TRUE(
      tmeta == y.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr());
  ASSERT_TRUE(tmeta.get() == y.unsafeGetTensorImpl()->get_backend_meta());
  ASSERT_TRUE(custom_tmeta->cloned_ == true);
}
