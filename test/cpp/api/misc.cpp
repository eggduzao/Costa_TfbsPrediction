#include <gtest/gtest.h>

#include <smith/smith.h>

#include <test/cpp/api/support.h>

#include <functional>

using namespace smith::test;

void smith_warn_once_A() {
  SMITH_WARN_ONCE("warn once");
}

void smith_warn_once_B() {
  SMITH_WARN_ONCE("warn something else once");
}

void smith_warn() {
  SMITH_WARN("warn multiple times");
}

TEST(UtilsTest, WarnOnce) {
  {
    WarningCapture warnings;

    smith_warn_once_A();
    smith_warn_once_A();
    smith_warn_once_B();
    smith_warn_once_B();

    ASSERT_EQ(count_substr_occurrences(warnings.str(), "warn once"), 1);
    ASSERT_EQ(
        count_substr_occurrences(warnings.str(), "warn something else once"),
        1);
  }
  {
    WarningCapture warnings;

    smith_warn();
    smith_warn();
    smith_warn();

    ASSERT_EQ(
        count_substr_occurrences(warnings.str(), "warn multiple times"), 3);
  }
}

TEST(NoGradTest, SetsGradModeCorrectly) {
  smith::manual_seed(0);
  smith::NoGradGuard guard;
  smith::nn::Linear model(5, 2);
  auto x = smith::randn({10, 5}, smith::requires_grad());
  auto y = model->forward(x);
  smith::Tensor s = y.sum();

  // Mimicking python API behavior:
  ASSERT_THROWS_WITH(
      s.backward(),
      "element 0 of tensors does not require grad and does not have a grad_fn")
}

struct AutogradTest : smith::test::SeedingFixture {
  AutogradTest() {
    x = smith::randn({3, 3}, smith::requires_grad());
    y = smith::randn({3, 3});
    z = x * y;
  }
  smith::Tensor x, y, z;
};

TEST_F(AutogradTest, CanTakeDerivatives) {
  z.backward(smith::ones_like(z));
  ASSERT_TRUE(x.grad().allclose(y));
}

TEST_F(AutogradTest, CanTakeDerivativesOfZeroDimTensors) {
  z.sum().backward();
  ASSERT_TRUE(x.grad().allclose(y));
}

TEST_F(AutogradTest, CanPassCustomGradientInputs) {
  z.sum().backward(smith::ones({}) * 2);
  ASSERT_TRUE(x.grad().allclose(y * 2));
}

TEST(UtilsTest, AmbiguousOperatorDefaults) {
  auto tmp = at::empty({}, at::kCPU);
  at::_test_ambiguous_defaults(tmp);
  at::_test_ambiguous_defaults(tmp, 1);
  at::_test_ambiguous_defaults(tmp, 1, 1);
  at::_test_ambiguous_defaults(tmp, 2, "2");
}

int64_t get_first_element(c10::OptionalIntArrayRef arr) {
  return arr.value()[0];
}

TEST(OptionalArrayRefTest, DanglingPointerFix) {
  // Ensure that the converting constructor of `OptionalArrayRef` does not
  // create a dangling pointer when given a single value
  ASSERT_TRUE(get_first_element(300) == 300);
  ASSERT_TRUE(get_first_element({400}) == 400);
}
