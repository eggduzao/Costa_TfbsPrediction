#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <gtest/gtest.h>
#include <test/cpp/lazy/test_lazy_ops_util.h>
#include <smith/csrc/lazy/core/debug_util.h>
#include <smith/csrc/lazy/core/helpers.h>
#include <smith/csrc/lazy/core/ir_builder.h>
#include <smith/csrc/lazy/core/lazy_graph_executor.h>
#include <smith/csrc/lazy/core/metrics.h>
#include <smith/csrc/lazy/core/permutation_util.h>
#include <smith/csrc/lazy/ts_backend/dynamic_ir.h>
#include <smith/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <smith/smith.h>

namespace smith {
namespace lazy {

// Lazy Tensor is disabled in FBCODE until addressing non-virtual methods (e.g.
// sizes) in TensorImpl
#ifndef FBCODE_CAFFE2

namespace {
// This registers the smithscript backend, without which lazy device won't work.
// FIXME: This registers the backend for the whole test binary. We should
// probably do it and undo it in the test fixture below.
static bool inline init_backend() {
  smith::lazy::InitSmithScriptBackend();
  return true;
}
static const bool backend_initialized = init_backend();

} // namespace

class LazyTsTest : public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  static void CommonSetup() {}

  void ExpectCounterNotChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {}

  void ExpectCounterChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {}

  void ResetCounters() {}

 private:
  void MakeEndSnapshot() {}
};

class LazyOpsTestBase : public LazyTsTest {
 protected:
  static void SetUpTestCase() {}
};

void LazyTsTest::SetUp() {
  (void)backend_initialized; // avoid unused parameter warning
  at::manual_seed(42);
  smith::lazy::LazyGraphExecutor::Get()->SetRngSeed(
      smith::lazy::BackendDevice(), 42);
}

void LazyTsTest::TearDown() {}

namespace {
using smith::lazy::DebugUtil;

class LazyOpsTest : public LazyOpsTestBase {};

static inline bool IsCuda() {
  return smith::lazy::getBackend()->EagerFallbackDeviceType() == at::kCUDA;
}

static inline at::DeviceType DefaultDevice() {
  return smith::lazy::getBackend()->EagerFallbackDeviceType();
}

} // namespace

TEST_F(LazyOpsTest, TestScalarTensor) {
  smith::Tensor scalar_tensor = smith::scalar_tensor(
      1., smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_scalar_tensor = smith::scalar_tensor(
        1., smith::TensorOptions(smith::kFloat).device(smith::kLazy));
    AllClose(scalar_tensor, lazy_scalar_tensor);
  });
}

TEST_F(LazyOpsTest, TestClone) {
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = lazy_a.clone();
    AllClose(a, lazy_b);
    lazy_a.add_(1.0);
    AllClose(a, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestTo) {
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestIsFloatingPoint) {
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    bool is_float = smith::is_floating_point(a);
    bool lazy_is_float = smith::is_floating_point(lazy_a);
    EXPECT_EQ(is_float, lazy_is_float);
  });
}

TEST_F(LazyOpsTest, TestIsSigned) {
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    bool is_signed = smith::is_signed(a);
    bool lazy_is_signed = smith::is_signed(lazy_a);
    EXPECT_EQ(is_signed, lazy_is_signed);
  });
}

TEST_F(LazyOpsTest, TestCastByte) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::_cast_Byte(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::_cast_Byte(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastChar) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::_cast_Char(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::_cast_Char(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastShort) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::_cast_Short(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::_cast_Short(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastInt) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::_cast_Int(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::_cast_Int(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastLong) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::_cast_Long(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::_cast_Long(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastFloat) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::_cast_Float(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::_cast_Float(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestRetainType) {
  smith::Tensor lazy_a = smith::zeros(
      {2, 2}, smith::TensorOptions(smith::kByte).device(smith::kLazy));
  smith::Tensor lazy_b = smith::ones(
      {2, 2}, smith::TensorOptions(smith::kByte).device(smith::kLazy));
  smith::Tensor lazy_c = lazy_a + lazy_b;
  EXPECT_EQ(lazy_c.scalar_type(), smith::ScalarType::Byte);
}

TEST_F(LazyOpsTest, TestLogicalTypeWithInterop) {
  smith::Tensor query = smith::rand(
      {2, 12, 20, 64},
      smith::TensorOptions(smith::kFloat).device(smith::kLazy));
  smith::Tensor key = smith::rand(
      {2, 12, 64, 20},
      smith::TensorOptions(smith::kFloat).device(smith::kLazy));
  smith::Tensor scores =
      smith::matmul(query, key) /
      smith::scalar_tensor(
          8, smith::TensorOptions(smith::kDouble).device(smith::kLazy));
  smith::Tensor p_attn = smith::softmax(scores, /*dim=*/-1);
  EXPECT_EQ(p_attn.scalar_type(), smith::ScalarType::Float);
}

TEST_F(LazyOpsTest, TestAdd) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::add(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddHalf) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kHalf).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kHalf).device(DefaultDevice()));
  smith::Tensor c = smith::add(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddMixedPrecision) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kHalf).device(DefaultDevice()));
  smith::Tensor c = smith::add(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddInPlace) {
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor c = a.add_(b);
    smith::Tensor lazy_c = lazy_a.add_(lazy_b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddScalar) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar b(1);
  smith::Tensor c = smith::add(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_c = smith::add(lazy_a, b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddScalarInPlace) {
  smith::Scalar b(1);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor c = a.add_(b);
    smith::Tensor lazy_c = lazy_a.add_(b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddZeroSizeDim) {
  smith::Tensor a = smith::rand(
      {0, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {1, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::add(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSub) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::sub(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::sub(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubInPlace) {
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor c = a.sub_(b);
    smith::Tensor lazy_c = lazy_a.sub_(lazy_b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubScalar) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar b(1);
  smith::Tensor c = smith::sub(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_c = smith::sub(lazy_a, b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubScalarInPlace) {
  smith::Scalar b(1);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor c = a.sub_(b);
    smith::Tensor lazy_c = lazy_a.sub_(b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMul) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::mul(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::mul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulInPlace) {
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor c = a.mul_(b);
    smith::Tensor lazy_c = lazy_a.mul_(lazy_b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulScalar) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar b(3);
  smith::Tensor c = smith::mul(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_c = smith::mul(lazy_a, b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulScalarInPlace) {
  smith::Scalar b(3);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a = smith::rand(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor c = a.mul_(b);
    smith::Tensor lazy_c = lazy_a.mul_(b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestDiv) {
  for (smith::ScalarType scalar_type1 :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor a = isFloatingType(scalar_type1)
        ? smith::rand({3, 4}, smith::TensorOptions(scalar_type1))
        : smith::randint(0, 100, {3, 4}, smith::TensorOptions(scalar_type1));
    for (smith::ScalarType scalar_type2 :
         {smith::kFloat,
          smith::kByte,
          smith::kChar,
          smith::kShort,
          smith::kInt,
          smith::kLong}) {
      smith::Tensor b = isFloatingType(scalar_type2)
          ? smith::rand({3, 4}, smith::TensorOptions(scalar_type2))
          : smith::randint(1, 100, {3, 4}, smith::TensorOptions(scalar_type2));
      smith::Tensor c = smith::div(a, b);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a, device);
        smith::Tensor lazy_b = CopyToDevice(b, device);
        smith::Tensor lazy_c = smith::div(lazy_a, lazy_b);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivWithRoundingMode) {
  std::optional<std::string_view> rounding_modes[] = {
      "trunc", "floor", std::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (smith::ScalarType scalar_type1 :
         {smith::kFloat,
          smith::kByte,
          smith::kChar,
          smith::kShort,
          smith::kInt,
          smith::kLong}) {
      int lower_bound = (scalar_type1 == smith::kByte) ? 0 : -100;
      smith::Tensor a = isFloatingType(scalar_type1)
          ? smith::rand({3, 4}, smith::TensorOptions(scalar_type1))
          : smith::randint(
                lower_bound, 50, {3, 4}, smith::TensorOptions(scalar_type1));
      for (smith::ScalarType scalar_type2 :
           {smith::kFloat,
            smith::kByte,
            smith::kChar,
            smith::kShort,
            smith::kInt,
            smith::kLong}) {
        smith::Tensor b = isFloatingType(scalar_type2)
            ? smith::rand({3, 4}, smith::TensorOptions(scalar_type2))
            : smith::randint(
                  51, 100, {3, 4}, smith::TensorOptions(scalar_type2));
        smith::Tensor c = smith::div(a, b, rounding_mode);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          smith::Tensor lazy_b = CopyToDevice(b, device);
          smith::Tensor lazy_c = smith::div(lazy_a, lazy_b, rounding_mode);
          AllClose(c, lazy_c);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestDivInPlace) {
  for (smith::ScalarType scalar_type1 : {smith::kFloat}) {
    smith::Tensor a = isFloatingType(scalar_type1)
        ? smith::rand({3, 4}, smith::TensorOptions(scalar_type1))
        : smith::randint(0, 100, {3, 4}, smith::TensorOptions(scalar_type1));
    for (smith::ScalarType scalar_type2 : {smith::kFloat}) {
      smith::Tensor b = isFloatingType(scalar_type2)
          ? smith::rand({3, 4}, smith::TensorOptions(scalar_type2))
          : smith::randint(1, 100, {3, 4}, smith::TensorOptions(scalar_type2));
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a, device);
        smith::Tensor c = a.div_(b);
        smith::Tensor lazy_b = CopyToDevice(b, device);
        smith::Tensor lazy_c = lazy_a.div_(lazy_b);
        ;
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivInPlaceWithRoundingMode) {
  std::optional<std::string_view> rounding_modes[] = {
      "trunc", "floor", std::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (smith::ScalarType scalar_type1 : {smith::kFloat}) {
      smith::Tensor a = isFloatingType(scalar_type1)
          ? smith::rand({3, 4}, smith::TensorOptions(scalar_type1))
          : smith::randint(
                -100, 100, {3, 4}, smith::TensorOptions(scalar_type1));
      for (smith::ScalarType scalar_type2 : {smith::kFloat}) {
        smith::Tensor b = isFloatingType(scalar_type2)
            ? smith::rand({3, 4}, smith::TensorOptions(scalar_type2))
            : smith::randint(
                  1, 100, {3, 4}, smith::TensorOptions(scalar_type2));
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          smith::Tensor c = a.div_(b, rounding_mode);
          smith::Tensor lazy_b = CopyToDevice(b, device);
          smith::Tensor lazy_c = lazy_a.div_(lazy_b, rounding_mode);
          AllClose(c, lazy_c);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestDivScalar) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor a = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 4}, smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              1,
              100,
              {3, 4},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool is_float : {true, false}) {
      smith::Scalar b = is_float ? smith::Scalar(3.0) : smith::Scalar(3);
      smith::Tensor c = smith::div(a, b);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a, device);
        smith::Tensor lazy_c = smith::div(lazy_a, b);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivScalarInPlace) {
  for (smith::ScalarType scalar_type : {smith::kFloat}) {
    smith::Tensor a = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 4}, smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              1,
              100,
              {3, 4},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool is_float : {true, false}) {
      smith::Scalar b = is_float ? smith::Scalar(3.0) : smith::Scalar(3);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a, device);
        smith::Tensor c = a.div_(b);
        smith::Tensor lazy_c = lazy_a.div_(b);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivOut) {
  for (smith::ScalarType scalar_type : {smith::kFloat, smith::kDouble}) {
    smith::Tensor a = smith::rand(
        {3, 4}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor b = smith::rand(
        {3, 4}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor c = smith::empty(
        {3, 4}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::div_out(c, a, b);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = smith::empty({3, 4}, lazy_b.options());
      smith::div_out(lazy_c, lazy_a, lazy_b);
      AllClose(c, lazy_c);
    });
  }
}

TEST_F(LazyOpsTest, TestRsubScalar) {
  smith::Tensor input = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar other(1.5);
  smith::Scalar alpha(2.5);
  smith::Tensor result = smith::rsub(input, other, alpha);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::rsub(lazy_input, other, alpha);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestNe) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::ne(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::ne(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestNeInplace) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor a_copy = a.clone();
  smith::Tensor b = a.clone();
  b[0] += 1;
  a.ne_(b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a_copy, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.ne_(lazy_b);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestEq) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.clone();
  smith::Tensor c = smith::eq(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::eq(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEqInplace) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.clone();
  b[0] += 1;
  smith::Tensor a_copy = a.clone();
  a.eq_(b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a_copy, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.eq_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestGe) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.clone();
  smith::Tensor c = smith::ge(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::ge(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGeInplace) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  smith::Tensor a_copy = a.clone();
  a.ge_(b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a_copy, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.ge_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestLe) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.clone();
  smith::Tensor c = smith::le(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::le(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestLeInplace) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  smith::Tensor a_copy = a.clone();
  a.le_(b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a_copy, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.le_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestGt) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::add(a.clone(), smith::ones_like(a));
  smith::Tensor c = smith::gt(b, a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::gt(lazy_b, lazy_a);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGtInplace) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  smith::Tensor a_copy = a.clone();
  a.gt_(b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a_copy, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.gt_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestLt) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::add(a.clone(), smith::ones_like(a));
  smith::Tensor c = smith::lt(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::lt(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestLtInplace) {
  smith::Tensor a = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  smith::Tensor a_copy = a.clone();
  a.lt_(b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a_copy, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.lt_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestNeScalar) {
  smith::Tensor input = smith::ones({2, 3});
  smith::Scalar other(float(0));
  smith::Tensor result = smith::ne(input, other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::ne(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEqScalar) {
  smith::Tensor input = smith::ones({2, 3});
  smith::Scalar other(float(1));
  smith::Tensor result = smith::eq(input, other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::eq(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGeScalar) {
  smith::Tensor input = smith::ones({2, 3});
  smith::Scalar other(float(1));
  smith::Tensor result = smith::ge(input, other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::ge(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGeScalarInplace) {
  smith::Tensor input = smith::arange(
      -1.,
      1.5,
      0.5,
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar other(float(0));
  smith::Tensor input_copy = input.clone();
  input.ge_(other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.ge_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestLeScalar) {
  smith::Tensor input = smith::ones({2, 3});
  smith::Scalar other(float(1));
  smith::Tensor result = smith::le(input, other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::le(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLeScalarInplace) {
  smith::Tensor input = smith::arange(
      -1.,
      1.5,
      0.5,
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar other(float(0));
  smith::Tensor input_copy = input.clone();
  input.le_(other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.le_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestGtScalar) {
  smith::Tensor input = smith::ones({2, 3});
  smith::Scalar other(float(0.5));
  smith::Tensor result = smith::gt(input, other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::gt(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGtScalarInplace) {
  smith::Tensor input = smith::arange(
      -1.,
      1.5,
      0.5,
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar other(float(0));
  smith::Tensor input_copy = input.clone();
  input.gt_(other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.gt_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestLtScalar) {
  smith::Tensor input = smith::ones({2, 3});
  smith::Scalar other(float(1.5));
  smith::Tensor result = smith::lt(input, other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::lt(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLtScalarInplace) {
  smith::Tensor input = smith::arange(
      -1.,
      1.5,
      0.5,
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar other(float(0));
  smith::Tensor input_copy = input.clone();
  input.lt_(other);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.lt_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestIntegerAdd) {
  std::vector<smith::ScalarType> types(
      {smith::kByte, smith::kChar, smith::kShort, smith::kInt, smith::kLong});

  ForEachDevice([&](const smith::Device& device) {
    for (auto type : types) {
      smith::Tensor a =
          smith::randint(0, 63, {2, 2}, smith::TensorOptions(type));
      smith::Tensor b =
          smith::randint(0, 63, {2, 2}, smith::TensorOptions(type));
      smith::Scalar one =
          isIntegralType(type, false) ? smith::Scalar(1) : smith::Scalar(1.0);
      smith::Tensor c = smith::add(b, one);

      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = smith::add(lazy_b, one);

      AllEqual(c, lazy_c);
    }
  });
}

TEST_F(LazyOpsTest, TestSVD) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      smith::Tensor a = smith::rand(
          {m, n}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      auto b = smith::svd(a, /*some=*/true, /*compute_uv=*/true);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a, device);
        auto lazy_b = smith::svd(lazy_a, /*some=*/true, /*compute_uv=*/true);
        // The U and V matrices might have different sign for column vectors, so
        // cannot be compared if not by absolute value.
        AllClose(
            std::get<0>(b).abs(),
            std::get<0>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        smith::Tensor diag = std::get<1>(b);
        smith::Tensor lazy_diag = std::get<1>(lazy_b);
        ASSERT_EQ(diag.sizes(), lazy_diag.sizes());
        AllClose(
            diag,
            lazy_diag,
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        AllClose(
            std::get<2>(b).abs(),
            std::get<2>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestQR) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      smith::Tensor a = smith::rand(
          {m, n}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      auto b = smith::qr(a);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a, device);
        auto lazy_b = smith::qr(lazy_a);
        AllClose(
            std::get<0>(b).abs(),
            std::get<0>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        AllClose(
            std::get<1>(b).abs(),
            std::get<1>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestCholesky) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (bool upper : {true, false}) {
      smith::Tensor a = smith::rand(
          {3, m, m},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor pd_a =
          smith::matmul(a, smith::transpose(a, 1, 2)) +
          smith::eye(
              m, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      auto b = smith::cholesky(pd_a, upper);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(pd_a, device);
        auto lazy_b = smith::cholesky(lazy_a, upper);
        AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestLogDet) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    smith::Tensor a = smith::rand(
        {3, m, m}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor pd_a = smith::matmul(a, smith::transpose(a, 1, 2)) +
        smith::eye(m,
                   smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor b = smith::logdet(pd_a);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(pd_a, device);
      smith::Tensor lazy_b = smith::logdet(lazy_a);
      AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestTriangularSolve) {
  static const int dims[] = {4, 7};
  for (bool batched_a : {true, false}) {
    for (bool batched_b : {true, false}) {
      for (auto m : dims) {
        for (auto n : dims) {
          for (bool upper : {true, false}) {
            for (bool transpose : {true, false}) {
              for (bool unitriangular : {true, false}) {
                smith::Tensor a = smith::randn(
                    {m, m},
                    smith::TensorOptions(smith::kFloat)
                        .device(DefaultDevice()));
                smith::Tensor b = smith::randn(
                    {m, n},
                    smith::TensorOptions(smith::kFloat)
                        .device(DefaultDevice()));
                a = batched_a ? a.expand({3, m, m}).clone() : a;
                b = batched_b ? b.expand({3, m, n}).clone() : b;
                auto result = smith::triangular_solve(
                    b,
                    a,
                    /*upper=*/upper,
                    /*transpose=*/transpose,
                    /*unitriangular=*/unitriangular);
                ForEachDevice([&](const smith::Device& device) {
                  smith::Tensor lazy_a = CopyToDevice(a, device);
                  smith::Tensor lazy_b = CopyToDevice(b, device);
                  auto lazy_result = smith::triangular_solve(
                      lazy_b,
                      lazy_a,
                      /*upper=*/upper,
                      /*transpose=*/transpose,
                      /*unitriangular=*/unitriangular);
                  AllClose(
                      std::get<0>(result),
                      std::get<0>(lazy_result),
                      /*rtol=*/1e-3,
                      /*atol=*/1e-4);
                  AllClose(
                      std::get<1>(result),
                      std::get<1>(lazy_result),
                      /*rtol=*/1e-3,
                      /*atol=*/1e-4);
                });
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestKthValue) {
  smith::Tensor a = smith::rand(
      {4, 5, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool keepdim : {false, true}) {
        auto b = smith::kthvalue(a, k, dim, keepdim);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = smith::kthvalue(lazy_a, k, dim, keepdim);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTopK) {
  smith::Tensor a = smith::rand(
      {4, 5, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool largest : {false, true}) {
        auto b = smith::topk(a, k, dim, largest, /*sorted=*/true);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = smith::topk(lazy_a, k, dim, largest, /*sorted=*/true);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSort) {
  smith::Tensor a = smith::rand(
      {4, 5, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        auto b = smith::sort(a, dim, descending);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = smith::sort(lazy_a, dim, descending);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSortDescWithMinValue) {
  std::vector<int8_t> values{-128, 100};
  smith::Tensor input =
      smith::tensor(values, smith::TensorOptions(smith::kChar));
  auto output = smith::sort(input, /*dim=*/0, /*descending=*/true);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    auto lazy_output = smith::sort(lazy_input, /*dim=*/0, /*descending=*/true);
    AllEqual(std::get<0>(output), std::get<0>(lazy_output));
    AllEqual(std::get<1>(output), std::get<1>(lazy_output));
  });
}

TEST_F(LazyOpsTest, TestArgSort) {
  smith::Tensor a = smith::rand(
      {4, 5, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        smith::Tensor b = smith::argsort(a, dim, descending);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          smith::Tensor lazy_b = smith::argsort(lazy_a, dim, descending);
          AllEqual(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMin) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::min(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::min(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMax) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::max(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::max(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestUnaryMin) {
  smith::Tensor input = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::min(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::min(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestUnaryMax) {
  smith::Tensor input = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::max(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::max(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAll) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor a = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 4}, smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {3, 4},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor b = smith::all(a);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::all(lazy_a);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAllDim) {
  smith::Tensor a = smith::randint(
      0,
      5,
      {2, 3, 4},
      smith::TensorOptions(smith::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::all(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::all(lazy_a, dim, /*keepdim=*/false);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAllDimKeep) {
  smith::Tensor a = smith::randint(
      0,
      5,
      {2, 3, 4},
      smith::TensorOptions(smith::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::all(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::all(lazy_a, dim, /*keepdim=*/true);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAmax) {
  smith::Tensor input = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor values = smith::amax(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_values =
            smith::amax(lazy_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, lazy_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        smith::Tensor values =
            smith::amax(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_input = CopyToDevice(input, device);
          smith::Tensor lazy_values =
              smith::amax(lazy_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, lazy_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("xla::amax", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestAmin) {
  smith::Tensor input = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor values = smith::amin(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_values =
            smith::amin(lazy_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, lazy_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        smith::Tensor values =
            smith::amin(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_input = CopyToDevice(input, device);
          smith::Tensor lazy_values =
              smith::amin(lazy_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, lazy_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("xla::amin", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestAny) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor a = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 4}, smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {3, 4},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor b = smith::any(a);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::any(lazy_a);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAnyDim) {
  smith::Tensor a = smith::randint(
      0,
      5,
      {2, 3, 4},
      smith::TensorOptions(smith::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::any(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::any(lazy_a, dim, /*keepdim=*/false);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAnyDimKeep) {
  smith::Tensor a = smith::randint(
      0,
      5,
      {2, 3, 4},
      smith::TensorOptions(smith::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::any(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::any(lazy_a, dim, /*keepdim=*/true);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMean) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::mean(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::mean(lazy_a);
    ASSERT_EQ(b.sizes(), lazy_b.sizes());
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestMeanCast) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::mean(a, smith::kDouble);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::mean(lazy_a, smith::kDouble);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestMeanInDim) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::mean(a, {dim});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::mean(lazy_a, {dim});
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDims) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    smith::Tensor b = smith::mean(a, dims);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::mean(lazy_a, dims);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDimsKeepCast) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    smith::Tensor b = smith::mean(a, dims, true, smith::kDouble);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::mean(lazy_a, dims, true, smith::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDimOut) {
  smith::Tensor a = smith::rand(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::empty(
        {4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::mean_out(b, a, {dim});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::empty({4, 4}, lazy_a.options());
      smith::mean_out(lazy_b, lazy_a, {dim});
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestStd) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto unbiased : {true, false}) {
    smith::Tensor b = smith::std(a, unbiased);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::std(lazy_a, unbiased);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestStdInDim) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (auto unbiased : {true, false}) {
    for (auto keepdim : {true, false}) {
      for (int dim = -rank; dim < rank; ++dim) {
        smith::Tensor b = smith::std(a, {dim}, unbiased, keepdim);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          smith::Tensor lazy_b = smith::std(lazy_a, {dim}, unbiased, keepdim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestStdWithCorrection) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // int rank = a.dim();
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        smith::Tensor b = smith::std(a, dim, correction, keepdim);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          smith::Tensor lazy_b = smith::std(lazy_a, dim, correction, keepdim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestStdMeanWithCorrection) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // int rank = a.dim();
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        auto b = smith::std_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = smith::std_mean(lazy_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllClose(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSum) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::sum(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::sum(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumCast) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::sum(a, smith::kDouble);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::sum(lazy_a, smith::kDouble);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumU8) {
  smith::Tensor a = smith::ones(
      {256}, smith::TensorOptions(smith::kByte).device(DefaultDevice()));
  smith::Tensor b = smith::sum(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::sum(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumInDim) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::sum(a, {dim});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::sum(lazy_a, {dim});
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDims) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    smith::Tensor b = smith::sum(a, dims);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::sum(lazy_a, dims);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDimsKeep) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    smith::Tensor b = smith::sum(a, dims, /*keepdim=*/true);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::sum(lazy_a, dims, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDimsKeepCast) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    smith::Tensor b = smith::sum(a, dims, /*keepdim=*/true, smith::kDouble);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b =
          smith::sum(lazy_a, dims, /*keepdim=*/true, smith::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestVar) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (bool unbiased : {true, false}) {
    smith::Tensor b = smith::var(a, unbiased);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::var(lazy_a, unbiased);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestVarWithDim) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (bool unbiased : {true, false}) {
        smith::Tensor b = smith::var(a, dims, unbiased, keepDim);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          smith::Tensor lazy_b = smith::var(lazy_a, dims, unbiased, keepDim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestVarWithCorrection) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (const auto& correction : corrections) {
        smith::Tensor b = smith::var(a, dim, correction, keepDim);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          smith::Tensor lazy_b = smith::var(lazy_a, dim, correction, keepDim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::var", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestVarMeanWithCorrection) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (const auto& correction : corrections) {
      for (auto keepdim : {true, false}) {
        auto b = smith::var_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = smith::var_mean(lazy_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllClose(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxInDim) {
  smith::Tensor input = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = smith::max(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        auto lazy_values_indices =
            smith::max(lazy_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(lazy_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(lazy_values_indices));
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMinInDim) {
  smith::Tensor input = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = smith::min(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        auto lazy_values_indices =
            smith::min(lazy_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(lazy_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(lazy_values_indices));
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNorm) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::norm(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::norm(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestNormInDim) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    smith::Tensor b = smith::norm(a, 2, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::norm(lazy_a, 2, {dim}, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestNormInDims) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    smith::Tensor b = smith::norm(a, 2, dims, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::norm(lazy_a, 2, dims, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestNormInDimsKeep) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    smith::Tensor b = smith::norm(a, 2, dims, /*keepdim=*/true);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::norm(lazy_a, 2, dims, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestNormalTwoTensor) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const smith::Device& device) {
    at::Tensor lazy_mean = CopyToDevice(mean, device);
    at::Tensor lazy_std = CopyToDevice(std, device);
    at::Tensor lazy_normal = at::normal(lazy_mean, lazy_std);
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestNormalDoubleMean) {
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const smith::Device& device) {
    at::Tensor lazy_std = CopyToDevice(std, device);
    at::Tensor lazy_normal = at::normal(0, lazy_std);
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestNormalDoubleStd) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const smith::Device& device) {
    at::Tensor lazy_mean = CopyToDevice(mean, device);
    at::Tensor lazy_normal = at::normal(lazy_mean, 1);
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestNormalInPlace) {
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const smith::Device& device) {
    at::Tensor lazy_a = CopyToDevice(a, device);
    lazy_a.normal_(/*mean=*/0, /*std=*/1);
    double res_mean = lazy_a.mean().item().toDouble();
    double res_std = lazy_a.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestUniformInPlace) {
  const double eps = 1e-3;
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const smith::Device& device) {
    at::Tensor lazy_a = CopyToDevice(a, device);
    lazy_a.uniform_(/*from=*/0, /*to=*/1);
    at::Tensor cpu_a = ToCpuTensor(lazy_a);
    double res_min = cpu_a.min().item().toDouble();
    double res_max = cpu_a.max().item().toDouble();
    EXPECT_GT(res_min, 0.0 - eps);
    EXPECT_LT(res_max, 1.0 + eps);
  });
}

TEST_F(LazyOpsTest, TestRandomInPlace) {
  for (auto dtype :
       {smith::kFloat,
        smith::kDouble,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    const double eps = 0.2;
    smith::Tensor a = smith::zeros({10, 10, 10}, smith::TensorOptions(dtype));
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      lazy_a.random_(/*from=*/0, /*to=*/10);
      double res_mean = lazy_a.sum().item().toDouble() / a.numel();
      double res_min = lazy_a.min().item().toDouble();
      double res_max = lazy_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(LazyOpsTest, TestRandomInPlaceDefaultFrom) {
  for (auto dtype :
       {smith::kFloat,
        smith::kDouble,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    const double eps = 0.2;
    smith::Tensor a = smith::zeros({10, 10, 10}, smith::TensorOptions(dtype));
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      lazy_a.random_(/*to=*/10);
      double res_mean = lazy_a.sum().item().toDouble() / a.numel();
      double res_min = lazy_a.min().item().toDouble();
      double res_max = lazy_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(LazyOpsTest, TestRandomInPlaceDefault) {
  for (auto dtype :
       {smith::kFloat,
        smith::kDouble,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    auto input = smith::zeros({10}, smith::TensorOptions(dtype));
    ForEachDevice([&](const smith::Device& device) {
      auto lazyInput = CopyToDevice(input, device);
      lazyInput.random_();
      auto output = ToCpuTensor(lazyInput);
      EXPECT_TRUE(smith::all(output.ne(input)).item<bool>());
    });
  }
}

TEST_F(LazyOpsTest, TestNormGeneral) {
  smith::Tensor a = smith::randn(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::norm(a, 3.5);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::norm(lazy_a, 3.5);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestNormNuclear) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::norm(a, 1);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::norm(lazy_a, 1);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFrobeniusNormInDim) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    smith::Tensor b = smith::frobenius_norm(a, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b =
          smith::frobenius_norm(lazy_a, {dim}, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestFrobeniusNormInDims) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    smith::Tensor b = smith::frobenius_norm(a, dims, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b =
          smith::frobenius_norm(lazy_a, dims, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestGroupNorm) {
  int num_channels = 6;
  smith::Tensor input = smith::rand(
      {20, num_channels, 10, 10},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {num_channels},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor bias = smith::rand(
      {num_channels},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  double eps = 1e-05;
  for (int num_groups : {3, 6, 1}) {
    smith::Tensor output = smith::group_norm(
        input,
        num_groups,
        weight,
        bias,
        eps,
        /*cudnn_enabled=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_weight = CopyToDevice(weight, device);
      smith::Tensor lazy_bias = CopyToDevice(bias, device);
      smith::Tensor lazy_output = smith::group_norm(
          lazy_input,
          num_groups,
          lazy_weight,
          lazy_bias,
          eps,
          /*cudnn_enabled=*/false);
      AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
    });
  }
}

TEST_F(LazyOpsTest, TestGroupNormBackward) {
  int num_channels = 6;
  smith::Tensor input = smith::rand(
      {2, num_channels, 5, 5},
      smith::TensorOptions(smith::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  smith::Tensor weight = smith::rand(
      {num_channels},
      smith::TensorOptions(smith::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  smith::Tensor bias = smith::rand(
      {num_channels},
      smith::TensorOptions(smith::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int num_groups : {3, 6, 1}) {
      auto testfn =
          [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
        return smith::group_norm(
            /*input=*/inputs[0],
            num_groups,
            inputs[1],
            inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      smith::Tensor undef;
      ForEachDevice([&](const smith::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device,
            testfn,
            /*rtol=*/1e-3,
            /*atol=*/1e-3,
            /*derivative_level=*/2);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestInstanceNorm) {
  int batch = 5;
  int num_channels = 20;
  smith::Tensor input = smith::rand(
      {batch, num_channels, 10, 10},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {num_channels},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor bias = smith::rand(
      {num_channels},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor running_mean = smith::zeros(
      {num_channels},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor running_var = smith::ones(
      {num_channels},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  double momentum = 0.1;
  double eps = 1e-05;
  smith::Tensor output = smith::instance_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      /*use_input_stats=*/true,
      momentum,
      eps,
      /*cudnn_enabled=*/false);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_weight = CopyToDevice(weight, device);
    smith::Tensor lazy_bias = CopyToDevice(bias, device);
    smith::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
    smith::Tensor lazy_running_var = CopyToDevice(running_var, device);
    smith::Tensor lazy_output = smith::instance_norm(
        lazy_input,
        lazy_weight,
        lazy_bias,
        lazy_running_mean,
        lazy_running_var,
        /*use_input_stats=*/true,
        momentum,
        eps,
        /*cudnn_enabled=*/false);
    AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLayerNorm) {
  smith::Tensor input = smith::rand(
      {20, 10, 10, 10},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  double eps = 1e-05;
  smith::Tensor undef;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      smith::Tensor weight = smith::rand(
          normalized_shape,
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor bias = smith::rand(
          normalized_shape,
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor output = smith::layer_norm(
          input,
          normalized_shape,
          undef_weight ? undef : weight,
          undef_weight ? undef : bias,
          eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        smith::Tensor lazy_bias =
            undef_weight ? undef : CopyToDevice(bias, device);
        smith::Tensor lazy_output = smith::layer_norm(
            lazy_input,
            normalized_shape,
            lazy_weight,
            lazy_bias,
            eps,
            /*cudnn_enabled=*/false);
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestLayerNormBackward) {
  smith::Tensor input = smith::rand(
      {2, 3, 3, 3},
      smith::TensorOptions(smith::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 3);
      auto testfn =
          [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
        return smith::layer_norm(
            /*input=*/inputs[0],
            normalized_shape,
            inputs[1],
            inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      smith::Tensor weight = smith::rand(
          normalized_shape,
          smith::TensorOptions(smith::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      smith::Tensor bias = smith::rand(
          normalized_shape,
          smith::TensorOptions(smith::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      smith::Tensor undef;
      ForEachDevice([&](const smith::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device,
            testfn,
            /*rtol=*/1e-3,
            /*atol=*/1e-4,
            /*derivative_level=*/2);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNuclearNorm) {
  smith::Tensor a = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::nuclear_norm(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::nuclear_norm(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestPairwiseDistance) {
  smith::Tensor x1 = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor x2 = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  double eps = 1e-6;
  for (bool keepdim : {false, true}) {
    for (double p : {1, 2, 3, 4}) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor output =
            smith::pairwise_distance(x1, x2, p, eps, keepdim);
        smith::Tensor lazy_x1 = CopyToDevice(x1, device);
        smith::Tensor lazy_x2 = CopyToDevice(x2, device);
        smith::Tensor lazy_output =
            smith::pairwise_distance(lazy_x1, lazy_x2, p, eps, keepdim);
        AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestCosineSimilarity) {
  smith::Tensor x1 = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor x2 = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  double eps = 1e-8;
  int rank = x1.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor output = smith::cosine_similarity(x1, x2, dim, eps);
      smith::Tensor lazy_x1 = CopyToDevice(x1, device);
      smith::Tensor lazy_x2 = CopyToDevice(x2, device);
      smith::Tensor lazy_output =
          smith::cosine_similarity(lazy_x1, lazy_x2, dim, eps);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestCosineEmbeddingLoss) {
  smith::Tensor input1 = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor input2 = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor target = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::Mean, smith::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor output = smith::cosine_embedding_loss(
            input1, input2, target, margin, reduction);
        smith::Tensor lazy_input1 = CopyToDevice(input1, device);
        smith::Tensor lazy_input2 = CopyToDevice(input2, device);
        smith::Tensor lazy_target = CopyToDevice(target, device);
        smith::Tensor lazy_output = smith::cosine_embedding_loss(
            lazy_input1, lazy_input2, lazy_target, margin, reduction);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestHingeEmbeddingLoss) {
  smith::Tensor input = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor target = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::Mean, smith::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor output =
            smith::hinge_embedding_loss(input, target, margin, reduction);
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_target = CopyToDevice(target, device);
        smith::Tensor lazy_output = smith::hinge_embedding_loss(
            lazy_input, lazy_target, margin, reduction);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestTripletMarginLoss) {
  smith::Tensor anchor = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor positive = smith::abs(smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())));
  smith::Tensor negative = smith::neg(smith::abs(smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()))));
  double eps = 1e-6;
  for (double margin : {0., 0.2}) {
    for (double p : {1, 2, 3, 4}) {
      for (bool swap : {false, true}) {
        for (smith::Reduction::Reduction reduction :
             {smith::Reduction::Mean, smith::Reduction::Sum}) {
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor output = smith::triplet_margin_loss(
                anchor, positive, negative, margin, p, eps, swap, reduction);
            smith::Tensor lazy_anchor = CopyToDevice(anchor, device);
            smith::Tensor lazy_positive = CopyToDevice(positive, device);
            smith::Tensor lazy_negative = CopyToDevice(negative, device);
            smith::Tensor lazy_output = smith::triplet_margin_loss(
                lazy_anchor,
                lazy_positive,
                lazy_negative,
                margin,
                p,
                eps,
                swap,
                reduction);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestBinaryCrossEntropy) {
  int batch = 10;
  int classes = 5;
  smith::Tensor input = smith::rand(
      {batch, classes},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor target = smith::rand(
      {batch, classes},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {batch, classes},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor undef;
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::Mean,
        smith::Reduction::Sum,
        smith::Reduction::None}) {
    for (bool undef_weight : {false, true}) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor output = smith::binary_cross_entropy(
            input, target, undef_weight ? undef : weight, reduction);
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_target = CopyToDevice(target, device);
        smith::Tensor lazy_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        smith::Tensor lazy_output = smith::binary_cross_entropy(
            lazy_input, lazy_target, lazy_weight, reduction);
        AllClose(output, lazy_output, /*rtol=*/1e-4, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMarginRankingLoss) {
  smith::Tensor input1 = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor input2 = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor target = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::Mean, smith::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor output = smith::margin_ranking_loss(
            input1, input2, target, margin, reduction);
        smith::Tensor lazy_input1 = CopyToDevice(input1, device);
        smith::Tensor lazy_input2 = CopyToDevice(input2, device);
        smith::Tensor lazy_target = CopyToDevice(target, device);
        smith::Tensor lazy_output = smith::margin_ranking_loss(
            lazy_input1, lazy_input2, lazy_target, margin, reduction);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestBCEWithLogits) {
  int batch = 10;
  int classes = 5;
  smith::Tensor input = smith::rand(
      {batch, classes},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor target = smith::rand(
      {batch, classes},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {classes}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor pos_weight = smith::rand(
      {classes}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor undef;
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::Mean, smith::Reduction::Sum}) {
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor output = smith::binary_cross_entropy_with_logits(
              input,
              target,
              undef_weight ? undef : weight,
              undef_pos_weight ? undef : pos_weight,
              reduction);
          smith::Tensor lazy_input = CopyToDevice(input, device);
          smith::Tensor lazy_target = CopyToDevice(target, device);
          smith::Tensor lazy_weight =
              undef_weight ? undef : CopyToDevice(weight, device);
          smith::Tensor lazy_pos_weight =
              undef_pos_weight ? undef : CopyToDevice(pos_weight, device);
          smith::Tensor lazy_output = smith::binary_cross_entropy_with_logits(
              lazy_input, lazy_target, lazy_weight, lazy_pos_weight, reduction);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestKlDiv) {
  smith::Tensor input = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor target = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (bool log_target : {true, false}) {
    for (smith::Reduction::Reduction reduction :
         {smith::Reduction::Mean, smith::Reduction::Sum}) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor output =
            smith::kl_div(input, target, reduction, log_target);
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_target = CopyToDevice(target, device);
        smith::Tensor lazy_output =
            smith::kl_div(lazy_input, lazy_target, reduction, log_target);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestProd) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::prod(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::prod(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestProdCast) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::prod(a, smith::kDouble);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::prod(lazy_a, smith::kDouble);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestProdInDim) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::prod(a, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::prod(lazy_a, dim);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestProdInDimKeepCast) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::prod(a, dim, /*keepdim=*/true, smith::kDouble);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b =
          smith::prod(lazy_a, dim, /*keepdim=*/true, smith::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestProdInDimKeep) {
  smith::Tensor a = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor b = smith::prod(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::prod(lazy_a, dim, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSum) {
  smith::Tensor input = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor result = smith::cumsum(input, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_result = smith::cumsum(lazy_input, dim);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSumCast) {
  smith::Tensor input = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor result = smith::cumsum(input, dim, smith::kDouble);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_result =
          smith::cumsum(lazy_input, dim, smith::kDouble);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSumLong) {
  smith::Tensor input = smith::randint(
      1000,
      {4, 3, 4},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor result = smith::cumsum(input, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_result = smith::cumsum(lazy_input, dim);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSumCastLong) {
  smith::Tensor input = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor result = smith::cumsum(input, dim, smith::kLong);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_result = smith::cumsum(lazy_input, dim, smith::kLong);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProd) {
  smith::Tensor input = smith::rand(
      {4, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor result = smith::cumprod(input, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_result = smith::cumprod(lazy_input, dim);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdCast) {
  smith::Tensor input = smith::mul(
      smith::rand(
          {4, 3, 4},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice())),
      10);
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor result = smith::cumprod(input, dim, smith::kDouble);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_result =
          smith::cumprod(lazy_input, dim, smith::kDouble);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdLong) {
  smith::Tensor input = smith::randint(
      7, {2, 3}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor result = smith::cumsum(input, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_result = smith::cumsum(lazy_input, dim);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdCastLong) {
  smith::Tensor input =
      smith::rand(
          {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      7;
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor result = smith::cumsum(input, dim, smith::kLong);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_result = smith::cumsum(lazy_input, dim, smith::kLong);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMin) {
  smith::Tensor a = smith::rand(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::argmin(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b =
        smith::argmin(lazy_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMinDim) {
  smith::Tensor a = smith::rand(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    smith::Tensor b = smith::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::argmin(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMinDimKeep) {
  smith::Tensor a = smith::rand(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    smith::Tensor b = smith::argmin(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::argmin(lazy_a, dim, /*keepdim=*/true);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMinSameValue) {
  smith::Tensor a = smith::ones(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::argmin(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::argmin(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMinWrapper) {
  smith::Tensor a = smith::rand(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    smith::Tensor b = smith::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::argmin(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMax) {
  smith::Tensor a = smith::rand(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::argmax(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b =
        smith::argmax(lazy_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMaxDim) {
  smith::Tensor a = smith::rand(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    smith::Tensor b = smith::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::argmax(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMaxDimKeep) {
  smith::Tensor a = smith::rand(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    smith::Tensor b = smith::argmax(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::argmax(lazy_a, dim, /*keepdim=*/true);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMaxSameValue) {
  smith::Tensor a = smith::ones(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::argmax(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b =
        smith::argmax(lazy_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMaxWrapper) {
  smith::Tensor a = smith::rand(
      {4, 4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    smith::Tensor b = smith::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::argmax(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAsin) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::asin(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::asin(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAsinh) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::asinh(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::asinh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAsinhInPlace) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = smith::asinh_(a);
    smith::Tensor lazy_b = smith::asinh_(lazy_a);
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSin) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::sin(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::sin(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSinh) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::sinh(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::sinh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAcos) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::acos(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::acos(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAcosh) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100;
  smith::Tensor b = smith::acosh(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::acosh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAcoshInPlace) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = smith::acosh_(a);
    smith::Tensor lazy_b = smith::acosh_(lazy_a);
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestCos) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::cos(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::cos(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestCosh) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::cosh(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::cosh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtan) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::atan(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::atan(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtanh) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::atanh(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::atanh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtanhInPlace) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = smith::atanh_(a);
    smith::Tensor lazy_b = smith::atanh_(lazy_a);
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtan2) {
  smith::Tensor a = smith::randn(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::randn(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::atan2(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::atan2(lazy_a, lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestTan) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::tan(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::tan(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestTanh) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::tanh(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::tanh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestClampMinMax) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar min_val(0.311);
  smith::Scalar max_val(0.409);
  smith::Tensor b = smith::clamp(a, min_val, max_val);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::clamp(lazy_a, min_val, max_val);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMin) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar min_val(0.311);
  smith::Tensor b = smith::clamp(a, min_val, std::nullopt);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::clamp(lazy_a, min_val, std::nullopt);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMax) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar max_val(0.409);
  smith::Tensor b = smith::clamp(a, std::nullopt, max_val);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::clamp(lazy_a, std::nullopt, max_val);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMinExplicit) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar min_val(0.311);
  smith::Tensor b = smith::clamp_min(a, min_val);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::clamp_min(lazy_a, min_val);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMaxExplicit) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar max_val(0.409);
  smith::Tensor b = smith::clamp_max(a, max_val);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::clamp_max(lazy_a, max_val);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMinExplicitInPlace) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar min_val(0.311);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = smith::clamp_min_(a, min_val);
    smith::Tensor lazy_b = smith::clamp_min_(lazy_a, min_val);
    AllClose(a, lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMaxExplicitInPlace) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar max_val(0.409);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = smith::clamp_max_(a, max_val);
    smith::Tensor lazy_b = smith::clamp_max_(lazy_a, max_val);
    AllClose(a, lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCeil) {
  smith::Tensor a =
      smith::randn(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::ceil(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::ceil(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFloor) {
  smith::Tensor a =
      smith::randn(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::floor(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::floor(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestRound) {
  smith::Tensor a = smith::cat(
      {smith::randn(
           {8}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
           100.0,
       // Special case: 0.5, -0.5. lazy::Round impl rounds to -1/1 whereas
       // lazy::RoundToEven properly implements bankers rounding.
       smith::tensor(
           {-0.5, 0.5},
           smith::TensorOptions(smith::kFloat).device(DefaultDevice()))},
      0);
  smith::Tensor b = smith::round(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::round(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestTrunc) {
  smith::Tensor a =
      smith::randn(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::trunc(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::trunc(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFrac) {
  smith::Tensor a =
      smith::randn(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::frac(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::frac(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestNeg) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::neg(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::neg(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestBitwiseNot) {
  std::vector<smith::ScalarType> types(
      {smith::kByte, smith::kChar, smith::kShort, smith::kInt, smith::kLong});

  ForEachDevice([&](const smith::Device& device) {
    for (auto type : types) {
      smith::Tensor a =
          smith::randint(0, 63, {2, 2}, smith::TensorOptions(type));
      smith::Tensor b = smith::bitwise_not(a);
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = smith::bitwise_not(lazy_a);
      AllEqual(b, lazy_b);
    }
  });
}

TEST_F(LazyOpsTest, TestBitwiseNotInPlace) {
  std::vector<smith::ScalarType> types(
      {smith::kByte, smith::kChar, smith::kShort, smith::kInt, smith::kLong});

  ForEachDevice([&](const smith::Device& device) {
    for (auto type : types) {
      smith::Tensor a =
          smith::randint(0, 63, {2, 2}, smith::TensorOptions(type));
      smith::Tensor lazy_a = CopyToDevice(a, device);
      a.bitwise_not_();
      lazy_a.bitwise_not_();
      AllEqual(a, lazy_a);
    }
  });
}

TEST_F(LazyOpsTest, TestSign) {
  smith::Tensor a =
      smith::randn(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b = smith::sign(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::sign(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSignByte) {
  smith::Tensor a = smith::randint(
      256, {2, 2}, smith::TensorOptions(smith::kByte).device(DefaultDevice()));
  smith::Tensor b = smith::sign(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::sign(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestAbs) {
  smith::Tensor a = smith::randn(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::abs(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::abs(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestAbsByte) {
  smith::Tensor a = smith::randint(
      256, {2, 2}, smith::TensorOptions(smith::kByte).device(DefaultDevice()));
  smith::Tensor b = smith::abs(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::abs(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestEmptyLike) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::empty_like(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::empty_like(lazy_a);
    EXPECT_EQ(b.sizes(), lazy_b.sizes());
  });
}

TEST_F(LazyOpsTest, TestEmptyLikeOptions) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::empty_like(
      a, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::empty_like(
        lazy_a, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    EXPECT_EQ(b.sizes(), lazy_b.sizes());
  });
}

TEST_F(LazyOpsTest, TestEmpty) {
  smith::Tensor a = smith::zeros(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = smith::empty(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(device));
    EXPECT_EQ(a.sizes(), lazy_a.sizes());
  });
}

TEST_F(LazyOpsTest, TestZeroInPlace) {
  smith::Tensor input = smith::ones(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));

  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazyInput = CopyToDevice(input, device);
    auto& output = smith::zero_(input);
    auto& lazyOutput = smith::zero_(lazyInput);
    AllClose(output, lazyOutput);
  });
}

TEST_F(LazyOpsTest, TestZerosLike) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::zeros_like(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::zeros_like(lazy_a);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestZerosLikeOptions) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::zeros_like(
      a, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::zeros_like(
        lazy_a, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestZeros) {
  smith::Tensor a = smith::zeros(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = smith::zeros(
        {2, 2}, smith::TensorOptions(smith::kFloat).device(device));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestOnes) {
  smith::Tensor a = smith::ones(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a =
        smith::ones({2, 2}, smith::TensorOptions(smith::kFloat).device(device));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestOnesLike) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::ones_like(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::ones_like(lazy_a);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestOnesLikeOptions) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::ones_like(
      a, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::ones_like(
        lazy_a, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFull) {
  smith::Tensor a = smith::full(
      {2, 2},
      3.1165,
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = smith::full(
        {2, 2}, 3.1165, smith::TensorOptions(smith::kFloat).device(device));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFullLike) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::full_like(a, 3.1165);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::full_like(lazy_a, 3.1165);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFullLikeOptions) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::full_like(
      a, 3.1165, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::full_like(
        lazy_a,
        3.1165,
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestARange) {
  for (auto& ranges : std::vector<std::vector<float>>{
           {0.0, 100.0, 0.5}, {0.0, -100.0, -0.5}}) {
    smith::Tensor a = smith::arange(
        ranges[0],
        ranges[1],
        ranges[2],
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = smith::arange(
          ranges[0],
          ranges[1],
          ranges[2],
          smith::TensorOptions(smith::kFloat).device(device));
      AllClose(a, lazy_a);
    });
  }
}

TEST_F(LazyOpsTest, TestARangeOut) {
  smith::Tensor a = smith::randn(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto& ranges : std::vector<std::vector<float>>{
           {0.0, 100.0, 0.5}, {0.0, -100.0, -0.5}}) {
    smith::Tensor b = smith::arange_out(a, ranges[0], ranges[1], ranges[2]);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b =
          smith::arange_out(lazy_a, ranges[0], ranges[1], ranges[2]);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestDimARange) {
  smith::Tensor like = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor a = smith::_dim_arange(like, 1);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_like = CopyToDevice(like, device);
    smith::Tensor lazy_a = smith::_dim_arange(lazy_like, 1);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestBartlettWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor output = smith::bartlett_window(
          window_length,
          periodic,
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));

      smith::Tensor lazy_output = smith::bartlett_window(
          window_length,
          periodic,
          smith::TensorOptions(smith::kFloat).device(device));
      AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });
  }
}

TEST_F(LazyOpsTest, TestBlackmanWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor output = smith::blackman_window(
          window_length,
          periodic,
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor lazy_output = smith::blackman_window(
          window_length,
          periodic,
          smith::TensorOptions(smith::kFloat).device(device));
      AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });
  }
}

TEST_F(LazyOpsTest, TestHammingWindow) {
  double alpha = 0.54;
  double beta = 0.46;
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor output = smith::hamming_window(
          window_length,
          periodic,
          alpha,
          beta,
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor lazy_output = smith::hamming_window(
          window_length,
          periodic,
          alpha,
          beta,
          smith::TensorOptions(smith::kFloat).device(device));
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestHannWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor output = smith::hann_window(
          window_length,
          periodic,
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor lazy_output = smith::hann_window(
          window_length,
          periodic,
          smith::TensorOptions(smith::kFloat).device(device));
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestLogSigmoid) {
  smith::Tensor a = smith::empty(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  a.uniform_(-1.0, 1.0);
  smith::Tensor b = smith::log_sigmoid(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::log_sigmoid(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLogSigmoidForward) {
  smith::Tensor a = smith::empty(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  a.uniform_(-1.0, 1.0);
  auto tuple = smith::log_sigmoid_forward(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    auto lazy_tuple = smith::log_sigmoid_forward(lazy_a);
    AllClose(
        std::get<0>(tuple),
        std::get<0>(lazy_tuple),
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
    AllClose(
        std::get<1>(tuple),
        std::get<1>(lazy_tuple),
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLogsumexp) {
  smith::Tensor a = smith::rand(
      {3, 4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepdim : {false, true}) {
      smith::Tensor b = smith::logsumexp(a, dims, keepdim);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a, device);
        smith::Tensor lazy_b = smith::logsumexp(lazy_a, dims, keepdim);
        AllClose(b, lazy_b);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestSiLU) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::silu(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::silu(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterChanged("lazy::silu_out", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestSigmoid) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::sigmoid(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::sigmoid(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestMatmul_1x1) {
  smith::Tensor a = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::matmul(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMatmul_2x1) {
  smith::Tensor a = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::matmul(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMatmul_1x2) {
  smith::Tensor a = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::matmul(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMatmul_2x2) {
  smith::Tensor a = smith::rand(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::matmul(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestMatmulBcast) {
  smith::Tensor a = smith::rand(
      {4, 2, 3, 2, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 1, 4, 3},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::matmul(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestDot) {
  smith::Tensor a = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::dot(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::dot(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestTensorDot) {
  smith::Tensor a = smith::rand(
      {6, 4, 8}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {4, 7, 8}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> dims_a = {1, 2};
  std::vector<int64_t> dims_b = {0, 2};
  smith::Tensor c = smith::tensordot(a, b, dims_a, dims_b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::tensordot(lazy_a, lazy_b, dims_a, dims_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGer) {
  smith::Tensor a = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::ger(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::ger(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMv) {
  smith::Tensor a = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::mv(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::mv(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMvOut) {
  smith::Tensor a = smith::rand(
      {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::mv_out(c, a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::empty({4}, lazy_b.options());
    smith::mv_out(lazy_c, lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestBatchAddBatchMatMul) {
  smith::Tensor a = smith::rand(
      {3, 6, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 6, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::rand(
      {3, 4, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar alpha = 0.5;
  smith::Scalar beta = 1.5;
  smith::Tensor d = smith::baddbmm(a, b, c, beta, alpha);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = smith::baddbmm(lazy_a, lazy_b, lazy_c, beta, alpha);
    AllClose(d, lazy_d, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestBatchAddBatchMatMulInPlace) {
  smith::Tensor a = smith::rand(
      {3, 6, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 6, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::rand(
      {3, 4, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar alpha = 0.5;
  smith::Scalar beta = 1.5;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor d = a.baddbmm_(b, c, beta, alpha);
    smith::Tensor lazy_d = lazy_a.baddbmm_(lazy_b, lazy_c, beta, alpha);
    AllClose(d, lazy_d, /*rtol=*/1e-3, /*atol=*/1e-4);
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestBatchMatMul) {
  smith::Tensor a = smith::rand(
      {3, 6, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 4, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::bmm(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::bmm(lazy_a, lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestChainMatMul) {
  smith::Tensor a = smith::rand(
      {5, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {4, 6}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::rand(
      {6, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor d = smith::rand(
      {2, 7}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor result = smith::chain_matmul({a, b, c, d});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = CopyToDevice(d, device);
    smith::Tensor lazy_result =
        smith::chain_matmul({lazy_a, lazy_b, lazy_c, lazy_d});
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestLinear) {
  smith::Tensor input = smith::rand(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor bias = smith::rand(
      {3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor result = smith::linear(input, weight);
  smith::Tensor result_with_bias = smith::linear(input, weight, bias);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_weight = CopyToDevice(weight, device);
    smith::Tensor lazy_bias = CopyToDevice(bias, device);
    smith::Tensor lazy_result = smith::linear(lazy_input, lazy_weight);
    smith::Tensor lazy_result_with_bias =
        smith::linear(lazy_input, lazy_weight, lazy_bias);
    AllClose(result, lazy_result, /*rtol=*/1e-2, /*atol=*/1e-4);
    AllClose(
        result_with_bias,
        lazy_result_with_bias,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestPinverse) {
  smith::Tensor input = smith::rand(
      {4, 6}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor result = smith::pinverse(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::pinverse(lazy_input);
    AllClose(result, lazy_result, /*rtol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestEinsumOuter) {
  smith::Tensor a = smith::rand(
      {5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::string equation = "i,j->ij";
  smith::Tensor c = smith::einsum(equation, {a, b});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::einsum(equation, {lazy_a, lazy_b});
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEinsumOuterBackward) {
  smith::Tensor a = smith::rand(
      {5},
      smith::TensorOptions(smith::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  smith::Tensor b = smith::rand(
      {5},
      smith::TensorOptions(smith::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  std::string equation = "i,j->ij";
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::einsum(equation, inputs);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward({a, b}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestEinsumBatchMatMul) {
  smith::Tensor a = smith::rand(
      {3, 2, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 5, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::string equation = "bij,bjk->bik";
  smith::Tensor c = smith::einsum(equation, {a, b});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::einsum(equation, {lazy_a, lazy_b});
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEinsumBlacksmithLowerBilinear) {
  smith::Tensor a = smith::rand(
      {3, 5, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor l = smith::rand(
      {2, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor r = smith::rand(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::string equation = "bn,anm,bm->ba";
  smith::Tensor c = smith::einsum(equation, {l, a, r});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_l = CopyToDevice(l, device);
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_r = CopyToDevice(r, device);
    smith::Tensor lazy_c = smith::einsum(equation, {lazy_l, lazy_a, lazy_r});
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEinsumBlacksmithLowerDiagonal) {
  smith::Tensor input = smith::rand(
      {3, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::string equation = "ii->i";
  smith::Tensor result = smith::einsum(equation, {input});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::einsum(equation, {lazy_input});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEinsumBlacksmithLowerBatchDiagonal) {
  smith::Tensor input = smith::rand(
      {4, 3, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::string equation = "...ii->...i";
  smith::Tensor result = smith::einsum(equation, {input});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::einsum(equation, {lazy_input});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEinsumBlacksmithLowerBatchPermute) {
  smith::Tensor input = smith::rand(
      {2, 3, 4, 5},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::string equation = "...ij->...ji";
  smith::Tensor result = smith::einsum(equation, {input});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::einsum(equation, {lazy_input});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEinsumBlacksmithLowerRepeatedAxis) {
  smith::Tensor x = smith::rand(
      {2, 3, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor y = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::string equation = "ijj,k->ik";
  smith::Tensor result = smith::einsum(equation, {x, y});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_x = CopyToDevice(x, device);
    smith::Tensor lazy_y = CopyToDevice(y, device);
    smith::Tensor lazy_result = smith::einsum(equation, {lazy_x, lazy_y});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBilinear) {
  int batch_size = 16;
  int in1_features = 4;
  int in2_features = 6;
  int out_features = 8;
  smith::Tensor input1 = smith::rand(
      {batch_size, in1_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor input2 = smith::rand(
      {batch_size, in2_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {out_features, in1_features, in2_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor bias = smith::rand(
      {out_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input1 = CopyToDevice(input1, device);
    smith::Tensor lazy_input2 = CopyToDevice(input2, device);
    smith::Tensor lazy_weight = CopyToDevice(weight, device);
    smith::Tensor lazy_bias = CopyToDevice(bias, device);
    smith::Tensor result = smith::bilinear(input1, input2, weight, bias);
    smith::Tensor lazy_result =
        smith::bilinear(lazy_input1, lazy_input2, lazy_weight, lazy_bias);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2D) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  smith::Tensor input = smith::rand(
      {batch_size, chans, h, w},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor result = smith::upsample_nearest2d(input, {uh, uw});
    smith::Tensor lazy_result = smith::upsample_nearest2d(lazy_input, {uh, uw});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2DBackward) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::upsample_nearest2d(inputs[0], {uh, uw});
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {batch_size, chans, h, w},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2DWithScale) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int chans = 2;
  double scale_h = 2.5;
  double scale_w = 3.4;
  smith::Tensor input = smith::rand(
      {batch_size, chans, h, w},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor result = smith::upsample_nearest2d(
        input, std::nullopt, at::ArrayRef<double>{scale_h, scale_w});
    smith::Tensor lazy_result = smith::upsample_nearest2d(
        lazy_input, std::nullopt, at::ArrayRef<double>{scale_h, scale_w});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2DBackwardWithScale) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int chans = 2;
  double scale_h = 2.5;
  double scale_w = 3.4;
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::upsample_nearest2d(
        inputs[0], std::nullopt, at::ArrayRef<double>{scale_h, scale_w});
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {batch_size, chans, h, w},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestUpsampleBilinear2D) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  for (bool align_corners : {true, false}) {
    smith::Tensor input = smith::rand(
        {batch_size, chans, h, w},
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor result =
          smith::upsample_bilinear2d(input, {uh, uw}, align_corners);
      smith::Tensor lazy_result =
          smith::upsample_bilinear2d(lazy_input, {uh, uw}, align_corners);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestUpsampleBilinear2DBackward) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  for (bool align_corners : {true, false}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::upsample_bilinear2d(inputs[0], {uh, uw}, align_corners);
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
              {batch_size, chans, h, w},
              smith::TensorOptions(smith::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAddCMul) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor d = smith::addcmul(a, b, c, 3.1165);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = smith::addcmul(lazy_a, lazy_b, lazy_c, 3.1165);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestAddCDiv) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c =
      smith::abs(smith::rand(
          {2, 2},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()))) +
      1.0;
  smith::Tensor d = smith::addcdiv(a, b, c, 3.1165);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = smith::addcdiv(lazy_a, lazy_b, lazy_c, 3.1165);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestAddCDivWithBroadcast) {
  smith::Tensor a = smith::rand(
      {1, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 1}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c =
      smith::abs(smith::rand(
          {1, 3},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()))) +
      1.0;
  smith::Tensor d = smith::addcdiv(a, b, c, 3.1165);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = smith::addcdiv(lazy_a, lazy_b, lazy_c, 3.1165);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestSize) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      EXPECT_EQ(smith::size(input, dim), smith::size(lazy_input, dim));
    }
  });
}

TEST_F(LazyOpsTest, TestSelect) {
  std::vector<int64_t> input_sizes = {14, 24, 8};
  int rank = input_sizes.size();
  for (int dim = -rank; dim < rank; ++dim) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::select(inputs[0], dim, 0);
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
              input_sizes,
              smith::TensorOptions(smith::kFloat).requires_grad(true))},
          device,
          testfn);
    });
  };
}

TEST_F(LazyOpsTest, TestBernoulliScalarProb) {
  smith::Tensor input = smith::zeros(
      1000, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::bernoulli(lazy_input, 0.1);
    double frac = lazy_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestBernoulliTensorProb) {
  std::vector<float> prob_values(1000, 0.1);
  smith::Tensor input = smith::tensor(
      prob_values, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::bernoulli(lazy_input);
    double frac = lazy_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestBernoulliScalarProbInPlace) {
  smith::Tensor input = smith::zeros(
      1000, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    lazy_input.bernoulli_(0.1);
    double frac = lazy_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestBernoulliTensorProbInPlace) {
  smith::Tensor input = smith::zeros(
      1000, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor prob = smith::scalar_tensor(
      0.1, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_prob = CopyToDevice(prob, device);
    lazy_input.bernoulli_(lazy_prob);
    double frac = lazy_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestDropout) {
  smith::Tensor a = smith::rand(
      {17, 21}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::dropout(lazy_a, 0.1, /*train=*/true);
    double prob =
        static_cast<double>(lazy_b.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    EXPECT_GT(prob, 0.86);
    EXPECT_LT(prob, 0.94);
  });
}

TEST_F(LazyOpsTest, TestDropoutInPlace) {
  smith::Tensor a = smith::rand(
      {17, 21}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::dropout_(lazy_a, 0.1, /*train=*/true);
    double prob =
        static_cast<double>(lazy_a.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    EXPECT_GT(prob, 0.85);
    EXPECT_LT(prob, 0.94);
  });
}

TEST_F(LazyOpsTest, TestRandperm) {
  unsigned n = 5;
  smith::Tensor shuffle = smith::randperm(
      n, smith::TensorOptions(smith::kLong).device(smith::kLazy));
  smith::Tensor shuffle_cpu = CopyToDevice(shuffle, smith::kCPU);
  std::vector<int64_t> shuffle_data(
      shuffle_cpu.data_ptr<int64_t>(), shuffle_cpu.data_ptr<int64_t>() + n);
  EXPECT_TRUE(
      shuffle_data.size() == n && smith::lazy::IsPermutation(shuffle_data));
}

TEST_F(LazyOpsTest, TestSlice) {
  smith::Tensor a = smith::rand(
      {32, 24, 16},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::slice(a, 1, 0, 16, 1);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::slice(lazy_a, 1, 0, 16, 1);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestTake) {
  smith::Tensor a = smith::rand(
      {4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::randint(
      16, {5}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor c = smith::take(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::take(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestTakeBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::take(inputs[0], inputs[1]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
             {4, 4},
             smith::TensorOptions(smith::kFloat)
                 .device(DefaultDevice())
                 .requires_grad(true)),
         smith::randint(
             16,
             {5},
             smith::TensorOptions(smith::kLong).device(DefaultDevice()))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestStack) {
  smith::Tensor a = smith::rand(
      {2, 4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::rand(
      {2, 4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = a.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor d = smith::stack({a, b, c}, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = CopyToDevice(c, device);
      smith::Tensor lazy_d = smith::stack({lazy_a, lazy_b, lazy_c}, dim);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestCat) {
  smith::Tensor a = smith::rand(
      {2, 1, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::rand(
      {2, 3, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    smith::Tensor d = smith::cat({a, b, c}, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = CopyToDevice(c, device);
      smith::Tensor lazy_d = smith::cat({lazy_a, lazy_b, lazy_c}, dim);
      EXPECT_TRUE(d.sizes() == lazy_d.sizes() && d.dtype() == lazy_d.dtype());
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestUnbind) {
  smith::Tensor input = smith::rand(
      {4, 3, 7}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<smith::Tensor> output = smith::unbind(input, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      std::vector<smith::Tensor> lazy_output = smith::unbind(lazy_input, dim);
      ASSERT_EQ(output.size(), lazy_output.size());
      for (size_t i = 0; i < output.size(); ++i) {
        AllClose(output[i], lazy_output[i]);
      }
    });
  }
}

TEST_F(LazyOpsTest, TestRepeat) {
  std::vector<std::vector<int64_t>> repeats_list = {{4, 2}, {4, 2, 3}};
  std::vector<std::vector<int64_t>> input_size_list = {{3}, {2, 4}};
  for (const auto& repeats : repeats_list) {
    for (const auto& input_size : input_size_list) {
      smith::Tensor input = smith::rand(
          input_size,
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor output = input.repeat(repeats);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_output = lazy_input.repeat(repeats);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestGather) {
  smith::Tensor a = smith::rand(
      {3, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::empty(
      {3, 3}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      b[i][j] = (i + j) % 3;
    }
  }
  for (bool sparse_grad : {false, true}) {
    smith::Tensor c = smith::gather(a, 1, b, sparse_grad);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = smith::gather(lazy_a, 1, lazy_b, sparse_grad);
      AllClose(c, lazy_c);
    });
  }
}

TEST_F(LazyOpsTest, TestScatter) {
  smith::Tensor a = smith::rand(
      {3, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {3, 5}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    smith::Tensor d = smith::scatter(a, dim, c, b);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = CopyToDevice(c, device);
      smith::Tensor lazy_d = smith::scatter(lazy_a, dim, lazy_c, lazy_b);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterR1) {
  smith::Tensor a = smith::rand(
      {5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {2}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  c[0] = 1;
  c[1] = 3;
  smith::Tensor d = smith::scatter(a, 0, c, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = smith::scatter(lazy_a, 0, lazy_c, lazy_b);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestScatterR3) {
  smith::Tensor a = smith::rand(
      {3, 5, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 4, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {3, 4, 2}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 2; k++) {
        c[i][j][k] = (i + j + k) % 4;
      }
    }
  }
  smith::Tensor d = smith::scatter(a, 1, c, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = smith::scatter(lazy_a, 1, lazy_c, lazy_b);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestScatterBiggerSource) {
  smith::Tensor a = smith::rand(
      {4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {8, 8}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {4, 4}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    smith::Tensor d = smith::scatter(a, dim, c, b);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = CopyToDevice(c, device);
      smith::Tensor lazy_d = smith::scatter(lazy_a, dim, lazy_c, lazy_b);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterScalar) {
  smith::Tensor a = smith::rand(
      {4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar b = 1.0f;
  smith::Tensor c = smith::empty(
      {4, 4}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    smith::Tensor d = smith::scatter(a, dim, c, b);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_c = CopyToDevice(c, device);
      smith::Tensor lazy_d = smith::scatter(lazy_a, dim, lazy_c, b);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterReduceAdd) {
  smith::Tensor a = smith::rand(
      {3, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {3, 5}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    smith::Tensor d = smith::scatter(a, dim, c, b, "add");
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = CopyToDevice(c, device);
      smith::Tensor lazy_d = smith::scatter(lazy_a, dim, lazy_c, lazy_b, "add");
      AllClose(d, lazy_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::scatter_out", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestScatterAdd) {
  smith::Tensor a = smith::rand(
      {3, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {3, 5}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    smith::Tensor d = smith::scatter_add(a, dim, c, b);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = CopyToDevice(c, device);
      smith::Tensor lazy_d = smith::scatter_add(lazy_a, dim, lazy_c, lazy_b);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterAddInPlace) {
  smith::Tensor b = smith::rand(
      {4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {4, 4}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor a = smith::rand(
          {4, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor d = a.scatter_add_(dim, c, b);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c = CopyToDevice(c, device);
      smith::Tensor lazy_d = lazy_a.scatter_add_(dim, lazy_c, lazy_b);
      AllClose(d, lazy_d);
      AllClose(a, lazy_a);
    });
  }
}

TEST_F(LazyOpsTest, TestIndexSelect) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor a = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 4}, smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {3, 4},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (smith::ScalarType index_scalar_type : {smith::kInt, smith::kLong}) {
      smith::Tensor b = smith::empty(
          {2}, smith::TensorOptions(index_scalar_type).device(DefaultDevice()));
      b[0] = 0;
      b[1] = 2;
      for (auto offset : {-2, 0}) {
        smith::Tensor c0 = smith::index_select(a, 0 + offset, b);
        smith::Tensor c1 = smith::index_select(a, 1 + offset, b);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a, device);
          smith::Tensor lazy_b = CopyToDevice(b, device);
          smith::Tensor lazy_c0 =
              smith::index_select(lazy_a, 0 + offset, lazy_b);
          smith::Tensor lazy_c1 =
              smith::index_select(lazy_a, 1 + offset, lazy_b);
          AllEqual(c0, lazy_c0);
          AllEqual(c1, lazy_c1);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestIndexSelectRank0) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor a = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 4}, smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {3, 4},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor b = smith::scalar_tensor(
        2, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor c0 = smith::index_select(a, 0, b);
    smith::Tensor c1 = smith::index_select(a, 1, b);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_a = CopyToDevice(a, device);
      smith::Tensor lazy_b = CopyToDevice(b, device);
      smith::Tensor lazy_c0 = smith::index_select(lazy_a, 0, lazy_b);
      smith::Tensor lazy_c1 = smith::index_select(lazy_a, 1, lazy_b);
      AllEqual(c0, lazy_c0);
      AllEqual(c1, lazy_c1);
    });
  }
}

TEST_F(LazyOpsTest, TestInverse) {
  if (IsCuda()) {
    // TODO(whc) debug failure on cuda, lazy_b comes back transposed
    GTEST_SKIP();
  }
  smith::Tensor a = smith::randn(
      {5, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::inverse(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::inverse(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestIsnan) {
  smith::Tensor a = smith::tensor(
      {1.0, 2.0, std::nan("1"), 4.0},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::isnan(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::isnan(lazy_a);
    AllEqual(b, lazy_b);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::isnan", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestExpand) {
  smith::Tensor a = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.expand({2, 3, 4}, /*implicit=*/false);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = lazy_a.expand({2, 3, 4}, /*implicit=*/false);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestExpandBack) {
  smith::Tensor a = smith::rand(
      {3, 1}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = a.expand({3, 4}, /*implicit=*/false);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = lazy_a.expand({3, 4}, /*implicit=*/false);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestExpandAs) {
  smith::Tensor a = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::native::expand_as(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::native::expand_as(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEye) {
  int n = 5;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor out = smith::eye(
        n, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_out =
        smith::eye(n, smith::TensorOptions(smith::kFloat).device(device));
    AllClose(out, lazy_out);
  });
}

TEST_F(LazyOpsTest, TestEyeWide) {
  int lines = 3;
  int cols = 5;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor out = smith::eye(
        lines,
        cols,
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_out = smith::eye(
        lines, cols, smith::TensorOptions(smith::kFloat).device(device));
    AllClose(out, lazy_out);
  });
}

TEST_F(LazyOpsTest, TestEyeNarrow) {
  int lines = 5;
  int cols = 3;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor out = smith::eye(
        lines,
        cols,
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_out = smith::eye(
        lines, cols, smith::TensorOptions(smith::kFloat).device(device));
    AllClose(out, lazy_out);
  });
}

TEST_F(LazyOpsTest, TestBroadcastTensors) {
  smith::Tensor a = smith::rand(
      {2, 1, 1}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2, 1}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<smith::Tensor> c = smith::broadcast_tensors({a, b});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    std::vector<smith::Tensor> lazy_c =
        smith::broadcast_tensors({lazy_a, lazy_b});
    ASSERT_EQ(c.size(), lazy_c.size());
    for (size_t i = 0; i < c.size(); ++i) {
      AllClose(c[i], lazy_c[i]);
    }
  });
}

TEST_F(LazyOpsTest, TestOneIndex) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor indices = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor result = smith::index(params, {indices});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_params = CopyToDevice(params, device);
      smith::Tensor lazy_indices = CopyToDevice(indices, device);
      smith::Tensor lazy_result = smith::index(lazy_params, {lazy_indices});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestOneIndexTransfer) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor indices = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor result = smith::index(params, {indices});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_params = CopyToDevice(params, device);
      smith::Tensor lazy_result = smith::index(lazy_params, {indices.cpu()});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestNonzero) {
  smith::Tensor a = smith::zeros(
      {4, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  a[0][1] = 1.0;
  a[1][0] = 2.0;
  a[3][1] = 3.0;
  smith::Tensor b = smith::nonzero(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::nonzero(lazy_a);
    AllClose(b, lazy_b);

    if (DebugUtil::ExperimentEnabled("nonzero")) {
      // If the nonzero support is enabled, we must not see any aten:: calls.
      ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    }
    ResetCounters();
  });
}

TEST_F(LazyOpsTest, TestMaskedSelect) {
  smith::Tensor a = smith::rand(
      {3, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::randint(
      0, 2, {5}, smith::TensorOptions(smith::kBool).device(DefaultDevice()));
  smith::Tensor c = smith::masked_select(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::masked_select(lazy_a, lazy_b);
    AllClose(c, lazy_c);

    if (DebugUtil::ExperimentEnabled("masked_select")) {
      // If the masked_select support is enabled, we must not see any aten::
      // calls.
      ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    }
    ResetCounters();
  });
}

TEST_F(LazyOpsTest, TestMaskedScatter) {
  smith::Tensor a = smith::rand(
      {3, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::randint(
      0, 2, {3, 5}, smith::TensorOptions(smith::kBool).device(DefaultDevice()));
  smith::Tensor c = smith::rand(
      {15}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor d = smith::masked_scatter(a, b, c);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = smith::masked_scatter(lazy_a, lazy_b, lazy_c);
    AllClose(d, lazy_d);

    if (DebugUtil::ExperimentEnabled("masked_scatter")) {
      // If the masked_select support is enabled, we must not see any aten::
      // calls.
      ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    }
    ResetCounters();
  });
}

TEST_F(LazyOpsTest, TestMultiIndexHeadNull) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor indices_null;
    smith::Tensor indices_0 = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor indices_1 = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor result =
        smith::index(params, {indices_null, indices_0, indices_1});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_params = CopyToDevice(params, device);
      smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      smith::Tensor lazy_result = smith::index(
          lazy_params, {indices_null, lazy_indices_0, lazy_indices_1});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexMiddleNull) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor indices_0 = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor indices_null;
    smith::Tensor indices_1 = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor result =
        smith::index(params, {indices_0, indices_null, indices_1});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_params = CopyToDevice(params, device);
      smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      smith::Tensor lazy_result = smith::index(
          lazy_params, {lazy_indices_0, indices_null, lazy_indices_1});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexTailNull) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor indices_0 = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor indices_null;
    smith::Tensor indices_1 = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor result =
        smith::index(params, {indices_0, indices_1, indices_null});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_params = CopyToDevice(params, device);
      smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      smith::Tensor lazy_result = smith::index(
          lazy_params, {lazy_indices_0, lazy_indices_1, indices_null});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexMiddleBroadcast) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor indices_0 = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor indices_1 = smith::randint(
        -3,
        3,
        {2, 1, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor result = smith::index(params, {indices_0, indices_1});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_params = CopyToDevice(params, device);
      smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      smith::Tensor lazy_result =
          smith::index(lazy_params, {lazy_indices_0, lazy_indices_1});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexTailBroadcast) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor indices_0 = smith::randint(
        -3,
        3,
        {2, 1, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor indices_1 = smith::randint(
        -3,
        3,
        {2, 1},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor result = smith::index(params, {indices_0, indices_1});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_params = CopyToDevice(params, device);
      smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      smith::Tensor lazy_result =
          smith::index(lazy_params, {lazy_indices_0, lazy_indices_1});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMaskIndex) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {2, 2}, smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {2, 2},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor indices = smith::randint(
        0,
        2,
        {2, 2},
        smith::TensorOptions(smith::kBool).device(DefaultDevice()));
    smith::Tensor result = smith::index(params, {indices});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_params = CopyToDevice(params, device);
      smith::Tensor lazy_indices = CopyToDevice(indices, device);
      smith::Tensor lazy_result = smith::index(lazy_params, {lazy_indices});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestOneIndexPut) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor indices = smith::randint(
        -3,
        3,
        {2, 4, 3},
        smith::TensorOptions(smith::kLong).device(DefaultDevice()));
    smith::Tensor values = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      smith::Tensor result =
          smith::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_params = CopyToDevice(params, device);
        smith::Tensor lazy_indices = CopyToDevice(indices, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::index_put(
            lazy_params, {lazy_indices}, lazy_values, accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestOneIndexPutInPlace) {
  smith::Tensor indices = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor values = smith::ones(
        {3, 5, 6, 7},
        smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor params = isFloatingType(scalar_type)
            ? smith::rand(
                  {4, 3, 5, 6, 7},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()))
            : smith::randint(
                  100,
                  {4, 3, 5, 6, 7},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()));
        smith::Tensor lazy_params = CopyToDevice(params.clone(), device);
        smith::Tensor result =
            smith::index_put_(params, {indices}, values, accumulate);
        smith::Tensor lazy_indices = CopyToDevice(indices, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::index_put_(
            lazy_params, {lazy_indices}, lazy_values, accumulate);
        AllEqual(result, lazy_result);
        AllEqual(params, lazy_params);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestOneIndexPutTransfer) {
  smith::Tensor indices = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor values = smith::ones(
        {3, 5, 6, 7},
        smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      smith::Tensor result =
          smith::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_params = CopyToDevice(params, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result =
            smith::index_put(lazy_params, {indices}, lazy_values, accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPut) {
  smith::Tensor indices_0 = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor indices_1 = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor values = smith::ones(
        {5, 6, 7}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      smith::Tensor result =
          smith::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_params = CopyToDevice(params, device);
        smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutHeadNull) {
  smith::Tensor indices_0 = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor indices_null;
  smith::Tensor indices_1 = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 3, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 3, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor values = smith::ones(
        {3, 6, 7}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      smith::Tensor result = smith::index_put(
          params, {indices_null, indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_params = CopyToDevice(params, device);
        smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::index_put(
            lazy_params,
            {indices_null, lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutMiddleNull) {
  smith::Tensor indices_0 = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor indices_null;
  smith::Tensor indices_1 = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 3, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 3, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor values = smith::ones(
        {3, 6, 7}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      smith::Tensor result = smith::index_put(
          params, {indices_0, indices_null, indices_1}, values, accumulate);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_params = CopyToDevice(params, device);
        smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::index_put(
            lazy_params,
            {lazy_indices_0, indices_null, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutTailNull) {
  smith::Tensor indices_0 = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor indices_1 = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor indices_null;
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 3, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 3, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor values = smith::ones(
        {3, 6, 7}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      smith::Tensor result = smith::index_put(
          params, {indices_0, indices_1, indices_null}, values, accumulate);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_params = CopyToDevice(params, device);
        smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1, indices_null},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutMiddleBroadcast) {
  smith::Tensor indices_0 = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor indices_1 = smith::randint(
      -3,
      3,
      {2, 1, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor values = smith::ones(
        {5, 6, 7}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      smith::Tensor result =
          smith::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_params = CopyToDevice(params, device);
        smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutTailBroadcast) {
  smith::Tensor indices_0 = smith::randint(
      -3,
      3,
      {2, 1, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor indices_1 = smith::randint(
      -3,
      3,
      {2, 1},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {4, 3, 5, 6, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor values = smith::ones(
        {5, 6, 7}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      smith::Tensor result =
          smith::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_params = CopyToDevice(params, device);
        smith::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        smith::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMaskIndexPut) {
  smith::Tensor indices =
      smith::tensor(
          {0, 1}, smith::TensorOptions(smith::kByte).device(DefaultDevice()))
          .to(smith::kBool);
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor params = isFloatingType(scalar_type)
        ? smith::rand(
              {2, 2}, smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {2, 2},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor values = smith::ones(
        {2}, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      smith::Tensor result =
          smith::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_params = CopyToDevice(params, device);
        smith::Tensor lazy_indices = CopyToDevice(indices, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::index_put(
            lazy_params, {lazy_indices}, lazy_values, accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexPutImpl) {
  smith::Tensor indices = smith::randint(
      -3,
      3,
      {2, 4, 3},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor values = smith::ones(
        {3, 5, 6, 7},
        smith::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor params = isFloatingType(scalar_type)
            ? smith::rand(
                  {4, 3, 5, 6, 7},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()))
            : smith::randint(
                  100,
                  {4, 3, 5, 6, 7},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()));
        smith::Tensor lazy_params = CopyToDevice(params.clone(), device);
        smith::Tensor result = smith::_index_put_impl_(
            params, {indices}, values, accumulate, /*unsafe=*/true);
        smith::Tensor lazy_indices = CopyToDevice(indices, device);
        smith::Tensor lazy_values = CopyToDevice(values, device);
        smith::Tensor lazy_result = smith::_index_put_impl_(
            lazy_params,
            {lazy_indices},
            lazy_values,
            accumulate,
            /*unsafe=*/true);
        AllEqual(result, lazy_result);
        AllEqual(params, lazy_params);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithScalar) {
  smith::Tensor index = smith::tensor(
      {0, 2}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Scalar value = 42;
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor base = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 4, 5},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {3, 4, 5},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor result = smith::index_fill(base, dim, index, value);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_base = CopyToDevice(base, device);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_result =
            smith::index_fill(lazy_base, dim, lazy_index, value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithScalarInPlace) {
  smith::Tensor index = smith::tensor(
      {0, 2}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Scalar value = 42;
  int rank = 3;
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor base = isFloatingType(scalar_type)
            ? smith::rand(
                  {3, 4, 5},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()))
            : smith::randint(
                  100,
                  {3, 4, 5},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()));
        smith::Tensor lazy_base = CopyToDevice(base.clone(), device);
        smith::Tensor result = base.index_fill_(dim, index, value);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_result =
            lazy_base.index_fill_(dim, lazy_index, value);
        AllEqual(result, lazy_result);
        AllEqual(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithTensor) {
  smith::Tensor index = smith::tensor(
      {0, 2}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor base = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 4, 5},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {3, 4, 5},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor value = smith::scalar_tensor(
        42, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor result = smith::index_fill(base, dim, index, value);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_base = CopyToDevice(base, device);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_value = CopyToDevice(value, device);
        smith::Tensor lazy_result =
            smith::index_fill(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithTensorInPlace) {
  smith::Tensor index = smith::tensor(
      {0, 2}, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor value = smith::scalar_tensor(
        42, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = 3;
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor base = isFloatingType(scalar_type)
            ? smith::rand(
                  {3, 4, 5},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()))
            : smith::randint(
                  100,
                  {3, 4, 5},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()));
        smith::Tensor lazy_base = CopyToDevice(base.clone(), device);
        smith::Tensor result = base.index_fill_(dim, index, value);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_value = CopyToDevice(value, device);
        smith::Tensor lazy_result =
            lazy_base.index_fill_(dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
        AllEqual(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillRank0) {
  smith::Tensor index = smith::scalar_tensor(
      2, smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor base = isFloatingType(scalar_type)
        ? smith::rand(
              {3, 4, 5},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {3, 4, 5},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    smith::Tensor value = smith::scalar_tensor(
        42, smith::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor result = smith::index_fill(base, dim, index, value);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_base = CopyToDevice(base, device);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_value = CopyToDevice(value, device);
        smith::Tensor lazy_result =
            smith::index_fill(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexAdd) {
  int index_size = 10;
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor base = isFloatingType(scalar_type)
        ? smith::rand(
              {5, 3, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {5, 3, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (smith::ScalarType index_scalar_type : {smith::kInt, smith::kLong}) {
        smith::Tensor index = smith::randint(
            0,
            base.size(dim),
            {index_size},
            smith::TensorOptions(index_scalar_type).device(DefaultDevice()));
        std::vector<int64_t> value_sizes(
            base.sizes().begin(), base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        smith::Tensor value = isFloatingType(scalar_type)
            ? smith::rand(
                  value_sizes,
                  smith::TensorOptions(scalar_type).device(DefaultDevice()))
            : smith::randint(
                  100,
                  value_sizes,
                  smith::TensorOptions(scalar_type).device(DefaultDevice()));
        smith::Tensor result = smith::index_add(base, dim, index, value);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_base = CopyToDevice(base, device);
          smith::Tensor lazy_index = CopyToDevice(index, device);
          smith::Tensor lazy_value = CopyToDevice(value, device);
          smith::Tensor lazy_result =
              smith::index_add(lazy_base, dim, lazy_index, lazy_value);
          AllClose(result, lazy_result);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestIndexAddInPlace) {
  int index_size = 10;
  int rank = 3;
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor base = isFloatingType(scalar_type)
            ? smith::rand(
                  {5, 3, 7},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()))
            : smith::randint(
                  100,
                  {5, 3, 7},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()));
        smith::Tensor index = smith::randint(
            0,
            base.size(dim),
            {index_size},
            smith::TensorOptions(smith::kLong).device(DefaultDevice()));
        std::vector<int64_t> value_sizes(
            base.sizes().begin(), base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        smith::Tensor value = isFloatingType(scalar_type)
            ? smith::rand(
                  value_sizes,
                  smith::TensorOptions(scalar_type).device(DefaultDevice()))
            : smith::randint(
                  100,
                  value_sizes,
                  smith::TensorOptions(scalar_type).device(DefaultDevice()));
        smith::Tensor lazy_base = CopyToDevice(base.clone(), device);
        smith::Tensor result = base.index_add_(dim, index, value);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_value = CopyToDevice(value, device);
        smith::Tensor lazy_result =
            lazy_base.index_add_(dim, lazy_index, lazy_value);
        AllClose(result, lazy_result);
        AllClose(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexAddRank0) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor base = isFloatingType(scalar_type)
        ? smith::rand(
              {5, 3, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {5, 3, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor index = smith::randint(
          0,
          base.size(dim),
          at::IntArrayRef{},
          smith::TensorOptions(smith::kLong).device(DefaultDevice()));
      std::vector<int64_t> value_sizes(
          base.sizes().begin(), base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = 1;
      smith::Tensor value = isFloatingType(scalar_type)
          ? smith::rand(
                value_sizes,
                smith::TensorOptions(scalar_type).device(DefaultDevice()))
          : smith::randint(
                100,
                value_sizes,
                smith::TensorOptions(scalar_type).device(DefaultDevice()));
      smith::Tensor result = smith::index_add(base, dim, index, value);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_base = CopyToDevice(base, device);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_value = CopyToDevice(value, device);
        smith::Tensor lazy_result =
            smith::index_add(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexCopy) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor base = isFloatingType(scalar_type)
        ? smith::rand(
              {5, 3, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {5, 3, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor index = smith::randperm(
          base.size(dim),
          smith::TensorOptions(smith::kLong).device(DefaultDevice()));
      smith::Tensor value = isFloatingType(scalar_type)
          ? smith::rand(
                base.sizes(),
                smith::TensorOptions(scalar_type).device(DefaultDevice()))
          : smith::randint(
                100,
                base.sizes(),
                smith::TensorOptions(scalar_type).device(DefaultDevice()));
      smith::Tensor result = smith::index_copy(base, dim, index, value);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_base = CopyToDevice(base, device);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_value = CopyToDevice(value, device);
        smith::Tensor lazy_result =
            smith::index_copy(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexCopyInPlace) {
  if (IsCuda()) {
    GTEST_SKIP();
  }
  int index_size = 10;
  int rank = 3;
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor base = isFloatingType(scalar_type)
            ? smith::rand(
                  {5, 3, 7},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()))
            : smith::randint(
                  100,
                  {5, 3, 7},
                  smith::TensorOptions(scalar_type).device(DefaultDevice()));
        smith::Tensor index = smith::randint(
            0,
            base.size(dim),
            {index_size},
            smith::TensorOptions(smith::kLong).device(DefaultDevice()));
        std::vector<int64_t> value_sizes(
            base.sizes().begin(), base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        smith::Tensor value = isFloatingType(scalar_type)
            ? smith::rand(
                  value_sizes,
                  smith::TensorOptions(scalar_type).device(DefaultDevice()))
            : smith::randint(
                  100,
                  value_sizes,
                  smith::TensorOptions(scalar_type).device(DefaultDevice()));
        smith::Tensor lazy_base = CopyToDevice(base.clone(), device);
        smith::Tensor result = base.index_copy_(dim, index, value);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_value = CopyToDevice(value, device);
        smith::Tensor lazy_result =
            lazy_base.index_copy_(dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
        AllEqual(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexCopyRank0) {
  for (smith::ScalarType scalar_type :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor base = isFloatingType(scalar_type)
        ? smith::rand(
              {5, 3, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()))
        : smith::randint(
              100,
              {5, 3, 7},
              smith::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor index = smith::randint(
          0,
          base.size(dim),
          at::IntArrayRef{},
          smith::TensorOptions(smith::kLong).device(DefaultDevice()));
      std::vector<int64_t> value_sizes(
          base.sizes().begin(), base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = 1;
      smith::Tensor value = isFloatingType(scalar_type)
          ? smith::rand(
                value_sizes,
                smith::TensorOptions(scalar_type).device(DefaultDevice()))
          : smith::randint(
                100,
                value_sizes,
                smith::TensorOptions(scalar_type).device(DefaultDevice()));
      smith::Tensor result = smith::index_copy(base, dim, index, value);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_base = CopyToDevice(base, device);
        smith::Tensor lazy_index = CopyToDevice(index, device);
        smith::Tensor lazy_value = CopyToDevice(value, device);
        smith::Tensor lazy_result =
            smith::index_copy(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestRelu) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::relu(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::relu(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReluInPlace) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = smith::relu_(input);
    smith::Tensor lazy_output = smith::relu_(lazy_input);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestHardshrink) {
  smith::Tensor input = smith::randn(
      {10}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::hardshrink(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::hardshrink(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardSigmoid) {
  smith::Tensor input = smith::randn(
      {10}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::hardsigmoid(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::hardsigmoid(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardSigmoidInPlace) {
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor input = smith::randn(
        {10}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = smith::hardsigmoid_(input);
    smith::Tensor lazy_output = smith::hardsigmoid_(lazy_input);
    AllClose(input, lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardSigmoidBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::hardsigmoid(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::randn(
            {10},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestSoftshrink) {
  smith::Tensor input = smith::randn(
      {10}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::softshrink(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::softshrink(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardtanh) {
  smith::Tensor input = smith::randn(
      {10}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::hardtanh(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::hardtanh(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardtanhInPlace) {
  smith::Tensor input = smith::randn(
      {10}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = smith::hardtanh_(input);
    smith::Tensor lazy_output = smith::hardtanh_(lazy_input);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestLeakyRelu) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  double negative_slope = 0.01;
  smith::Tensor output = smith::leaky_relu(input, negative_slope);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::leaky_relu(lazy_input, negative_slope);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestLeakyReluInPlace) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  double negative_slope = 0.01;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = smith::leaky_relu_(input, negative_slope);
    smith::Tensor lazy_output = smith::leaky_relu_(lazy_input, negative_slope);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestExp) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::exp(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::exp(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestExpm1) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::expm1(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::expm1(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::log(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::log(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog2) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::log2(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::log2(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog10) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::log10(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::log10(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog1p) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::log1p(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::log1p(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestErf) {
  smith::Tensor a = smith::randn(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::erf(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::erf(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestErfc) {
  smith::Tensor a = smith::randn(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::erfc(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::erfc(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestErfinv) {
  smith::Tensor a = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::erfinv(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::erfinv(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSqrt) {
  smith::Tensor a = smith::abs(smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())));
  smith::Tensor b = smith::sqrt(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::sqrt(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestRsqrt) {
  smith::Tensor a = smith::abs(smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())));
  smith::Tensor b = smith::rsqrt(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::rsqrt(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestReciprocal) {
  smith::Tensor a = smith::randn(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::reciprocal(a);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::reciprocal(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorScalar) {
  smith::Tensor base = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar exponent = 4.09;
  smith::Tensor result = smith::pow(base, exponent);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_base = CopyToDevice(base, device);
    smith::Tensor lazy_result = smith::pow(lazy_base, exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorScalarInPlace) {
  smith::Tensor base = smith::rand(
      {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar exponent = 4.09;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_base = CopyToDevice(base.clone(), device);
    smith::Tensor result = base.pow_(exponent);
    smith::Tensor lazy_result = lazy_base.pow_(exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, lazy_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorTensor) {
  smith::Tensor base = smith::abs(smith::rand(
      {4, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())));
  smith::Tensor exponent = smith::rand(
      {4, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor result = smith::pow(base, exponent);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_base = CopyToDevice(base, device);
    smith::Tensor lazy_exponent = CopyToDevice(exponent, device);
    smith::Tensor lazy_result = smith::pow(lazy_base, lazy_exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorTensorInPlace) {
  smith::Tensor base = smith::abs(smith::rand(
      {4, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())));
  smith::Tensor exponent = smith::rand(
      {4, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_base = CopyToDevice(base.clone(), device);
    smith::Tensor result = base.pow_(exponent);
    smith::Tensor lazy_exponent = CopyToDevice(exponent, device);
    smith::Tensor lazy_result = lazy_base.pow_(lazy_exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, lazy_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorTensorBroadcast) {
  smith::Tensor base = smith::abs(smith::rand(
      {4, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())));
  smith::Tensor exponent = smith::rand(
      {4, 1}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor result = smith::pow(base, exponent);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_base = CopyToDevice(base, device);
    smith::Tensor lazy_exponent = CopyToDevice(exponent, device);
    smith::Tensor lazy_result = smith::pow(lazy_base, lazy_exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowScalarTensor) {
  smith::Scalar base = 3.5;
  smith::Tensor exponent = smith::rand({4, 2});
  smith::Tensor result = smith::pow(base, exponent);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_exponent = CopyToDevice(exponent, device);
    smith::Tensor lazy_result = smith::pow(base, lazy_exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowIntExponent) {
  smith::Tensor base = smith::abs(smith::rand(
      {4, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())));
  smith::Scalar exponent = 3;
  smith::Tensor result = smith::pow(base, exponent);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_base = CopyToDevice(base, device);
    smith::Tensor lazy_result = smith::pow(lazy_base, exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestFmodScalar) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Scalar divisor = 2.0;
  smith::Tensor b = smith::fmod(a, divisor);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::fmod(lazy_a, divisor);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFmodScalarInPlace) {
  smith::Scalar divisor = 2.0;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a =
        smith::rand(
            {2, 2},
            smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
        100.0;
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = a.fmod_(divisor);
    smith::Tensor lazy_b = lazy_a.fmod_(divisor);
    AllClose(b, lazy_b);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFmodTensor) {
  smith::Tensor a =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      10.0;
  smith::Tensor c = smith::fmod(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::fmod(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestFmodTensorInPlace) {
  smith::Tensor b =
      smith::rand(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      10.0;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a =
        smith::rand(
            {2, 2},
            smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
        100.0;
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor c = a.fmod_(b);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = lazy_a.fmod_(lazy_b);
    AllClose(c, lazy_c);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestRemainderScalar) {
  smith::Tensor a =
      smith::randn(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Scalar divisor = -2.0;
  smith::Tensor b = smith::remainder(a, divisor);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = smith::remainder(lazy_a, divisor);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestRemainderScalarInPlace) {
  smith::Scalar divisor = -2.0;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a =
        smith::randn(
            {2, 2},
            smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
        100.0;
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor b = a.remainder_(divisor);
    smith::Tensor lazy_b = lazy_a.remainder_(divisor);
    AllClose(b, lazy_b);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestRemainderTensor) {
  smith::Tensor a =
      smith::randn(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      100.0;
  smith::Tensor b =
      smith::randn(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      10.0;
  smith::Tensor c = smith::remainder(a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = smith::remainder(lazy_a, lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(LazyOpsTest, TestRemainderTensorInPlace) {
  smith::Tensor b =
      smith::randn(
          {2, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
      10.0;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor a =
        smith::randn(
            {2, 2},
            smith::TensorOptions(smith::kFloat).device(DefaultDevice())) *
        100.0;
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor c = a.remainder_(b);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = lazy_a.remainder_(lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-4, /*atol=*/1e-6);
    AllClose(a, lazy_a, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(LazyOpsTest, TestWhere) {
  smith::Tensor a = smith::rand(
      {3, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {3, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {3, 3}, smith::TensorOptions(smith::kByte).device(DefaultDevice()));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  smith::Tensor d = smith::where(c, a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = smith::where(lazy_c, lazy_a, lazy_b);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestWhereBroadcast) {
  smith::Tensor a = smith::rand(
      {3, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::zeros(
      {}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::empty(
      {3, 3}, smith::TensorOptions(smith::kByte).device(DefaultDevice()));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  smith::Tensor d = smith::where(c, a, b);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    smith::Tensor lazy_d = smith::where(lazy_c, lazy_a, lazy_b);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestThreshold) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  float threshold = 0.4;
  float value = 20;
  smith::Tensor output = smith::threshold(input, threshold, value);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::threshold(lazy_input, threshold, value);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestThresholdBackward) {
  float threshold = 0.4;
  float value = 20;

  auto testFunction =
      [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::threshold(inputs[0], threshold, value);
  };

  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 1, 4, 6},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testFunction);
  });
}

TEST_F(LazyOpsTest, TestThresholdInPlace) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = input.clone();
  float threshold = 0.4;
  float value = 20;
  smith::threshold_(output, threshold, value);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_output = CopyToDevice(input, device);
    smith::threshold_(lazy_output, threshold, value);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestElu) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar alpha = 0.5;
  smith::Scalar scale = 2.5;
  smith::Scalar input_scale = 1.5;
  smith::Tensor output = smith::elu(input, alpha, scale, input_scale);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output =
        smith::elu(lazy_input, alpha, scale, input_scale);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestEluInPlace) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar alpha = 0.5;
  smith::Scalar scale = 2.5;
  smith::Scalar input_scale = 1.5;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = smith::elu_(input, alpha, scale, input_scale);
    smith::Tensor lazy_output =
        smith::elu_(lazy_input, alpha, scale, input_scale);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestSelu) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::selu(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::selu(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSeluInPlace) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = smith::selu_(input);
    smith::Tensor lazy_output = smith::selu_(lazy_input);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestCelu) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar alpha = 2.5;
  smith::Tensor output = smith::celu(input, alpha);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::celu(lazy_input, alpha);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestCeluInPlace) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar alpha = 2.5;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = smith::celu_(input, alpha);
    smith::Tensor lazy_output = smith::celu_(lazy_input, alpha);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestGelu) {
  smith::Tensor input = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::gelu(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::gelu(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAddMatMul) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  smith::Tensor input = smith::rand(
      {in_channels, out_channels},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {out_channels, labels},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor bias = smith::rand(
      {labels}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    smith::Tensor output = smith::addmm(bias, input, weight, /*beta=*/beta);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_weight = CopyToDevice(weight, device);
      smith::Tensor lazy_bias = CopyToDevice(bias, device);
      smith::Tensor lazy_output =
          smith::addmm(lazy_bias, lazy_input, lazy_weight, /*beta=*/beta);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestEmbedding) {
  smith::Tensor a = smith::rand(
      {32, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor i = smith::randint(
      0,
      31,
      {3, 4},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor b = smith::embedding(
      a,
      i,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_i = CopyToDevice(i, device);
    smith::Tensor lazy_b = smith::embedding(
        lazy_a,
        lazy_i,
        /*padding_idx=*/0,
        /*scale_grad_by_freq=*/false,
        /*sparse=*/false);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestOneHot) {
  int num_classes = 5;
  smith::Tensor input = smith::randint(
      0,
      num_classes,
      {10},
      smith::TensorOptions(smith::kLong).device(DefaultDevice()));
  smith::Tensor output = smith::one_hot(input, num_classes);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::one_hot(lazy_input, num_classes);
    AllEqual(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTranspose) {
  smith::Tensor input = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::t(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::t(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTransposeInPlace) {
  smith::Tensor input = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = input.t_();
    smith::Tensor lazy_output = lazy_input.t_();
    EXPECT_EQ(lazy_output.sizes(), output.sizes());
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestReshape) {
  smith::Tensor input = smith::rand(
      {32, 20, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::reshape(input, {-1, 320});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::reshape(lazy_input, {-1, 320});
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestResize) {
  // Testing a resize_() with target size bigger than original size is not
  // possible, as we fill with zeros, while blacksmith fills with random garbage.
  smith::Tensor input = smith::rand(
      {2, 2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor saved_input = input.clone();
  input.resize_({3, 3});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(saved_input, device);
    lazy_input.resize_({3, 3});
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestViewResize) {
  smith::Tensor input = smith::zeros(
      {8, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor saved_input = input.clone();
  smith::Tensor output = input.view({4, 4});
  output.resize_({3, 3});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(saved_input, device);
    smith::Tensor lazy_output = lazy_input.view({4, 4});
    lazy_output.resize_({3, 3});
    AllClose(input, lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestView) {
  smith::Tensor input = smith::rand(
      {32, 20, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = input.view({-1, 320});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = lazy_input.view({-1, 320});
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestViewMod) {
  smith::Tensor input = smith::zeros(
      {32, 20, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor one = smith::tensor(
      1.0, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = input.view({-1, 320});
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor xinput = smith::zeros(
        {32, 20, 4, 4},
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_input = CopyToDevice(xinput, device);
    smith::Tensor lazy_one = CopyToDevice(one, device);
    smith::Tensor lazy_output = lazy_input.view({-1, 320});
    lazy_output.add_(lazy_one, 1.0);
    lazy_input.add_(lazy_one, 1.0);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestViewModComplex) {
  smith::Tensor input = smith::zeros(
      {32, 20, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor one = smith::tensor(
      1.0, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  smith::Tensor output2 = input.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor xinput = smith::zeros(
        {32, 20, 4, 4},
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_input = CopyToDevice(xinput, device);
    smith::Tensor lazy_one = CopyToDevice(one, device);
    smith::Tensor lazy_output1 = lazy_input.view({-1, 320});
    lazy_output1.add_(lazy_one, 1.0);
    smith::Tensor lazy_output2 = lazy_input.view({-1, 160});
    lazy_output2.add_(lazy_one, 1.0);
    AllClose(output1, lazy_output1);
    AllClose(output2, lazy_output2);
  });
}

TEST_F(LazyOpsTest, TestViewOfViewMod) {
  smith::Tensor input = smith::zeros(
      {32, 20, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor one = smith::tensor(
      1.0, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  smith::Tensor output2 = output1.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor xinput = smith::zeros(
        {32, 20, 4, 4},
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_input = CopyToDevice(xinput, device);
    smith::Tensor lazy_one = CopyToDevice(one, device);
    smith::Tensor lazy_output1 = lazy_input.view({-1, 320});
    lazy_output1.add_(lazy_one, 1.0);
    smith::Tensor lazy_output2 = lazy_output1.view({-1, 160});
    lazy_output2.add_(lazy_one, 1.0);
    AllClose(output1, lazy_output1);
    AllClose(output2, lazy_output2);
  });
}

TEST_F(LazyOpsTest, TestViewSqueezeAddInPlace) {
  smith::Tensor input = smith::zeros(
      {2, 3, 1}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> view_size = {2, 3, 1, 1};
  int squeeze_dim = 2;
  smith::Tensor one = smith::tensor(
      1.0, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = input.view(view_size);
    output.squeeze_(squeeze_dim);
    output.add_(one, 1.0);
    smith::Tensor lazy_one = CopyToDevice(one, device);
    smith::Tensor lazy_output = lazy_input.view(view_size);
    lazy_output.squeeze_(squeeze_dim);
    lazy_output.add_(lazy_one, 1.0);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestUnsafeView) {
  smith::Tensor input = smith::rand(
      {32, 20, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::_unsafe_view(input, {-1, 320});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::_unsafe_view(lazy_input, {-1, 320});
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestNarrow) {
  smith::Tensor a = smith::rand(
      {8, 10, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int64_t dim : {1, -3}) {
    for (int64_t start : {2, -8}) {
      smith::Tensor b = a.narrow(dim, start, 6);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a, device);
        smith::Tensor lazy_b = lazy_a.narrow(dim, start, 6);
        AllClose(b, lazy_b);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdate) {
  for (int64_t dim : {1, -2}) {
    for (int64_t start : {2, -6}) {
      smith::Tensor a = smith::rand(
          {3, 8, 3},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor a_copy = a.clone();
      smith::Tensor b = smith::rand(
          {3, 4, 3},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a_copy, device);
        smith::Tensor lazy_b = CopyToDevice(b, device);
        smith::Tensor lazy_c = lazy_a.narrow(dim, start, 4);
        lazy_c.add_(lazy_b, 1.0);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdateBaseCheck) {
  for (int64_t dim : {0, -2}) {
    for (int64_t start : {2, -6}) {
      smith::Tensor a = smith::zeros(
          {8, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor a_copy = a.clone();
      smith::Tensor b = smith::ones(
          {4, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a_copy, device);
        smith::Tensor lazy_b = CopyToDevice(b, device);
        smith::Tensor lazy_c = lazy_a.narrow(dim, start, 4);
        lazy_c.add_(lazy_b, 1.0);
        AllClose(a, lazy_a);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdateTwoSlices) {
  for (int64_t dim : {0, -2}) {
    for (int64_t start0 : {2, -6}) {
      for (int64_t start1 : {6, -2}) {
        smith::Tensor a = smith::zeros(
            {8, 3},
            smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
        smith::Tensor a_copy = a.clone();
        smith::Tensor b = smith::ones(
            {2, 3},
            smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
        smith::Tensor c = b + 1;
        smith::Tensor d = a.narrow(dim, start0, 2);
        smith::Tensor e = a.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        e.add_(c, 1.0);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a_copy, device);
          smith::Tensor lazy_b = CopyToDevice(b, device);
          smith::Tensor lazy_c = CopyToDevice(c, device);
          smith::Tensor lazy_d = lazy_a.narrow(dim, start0, 2);
          smith::Tensor lazy_e = lazy_a.narrow(dim, start1, 2);
          lazy_d.add_(lazy_b, 1.0);
          lazy_e.add_(lazy_c, 1.0);
          AllClose(d, lazy_d);
          AllClose(e, lazy_e);
          AllClose(a, lazy_a);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdateView) {
  for (int64_t dim : {0, -3}) {
    for (int64_t start : {2, -6}) {
      smith::Tensor a = smith::rand(
          {8, 2, 3},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor a_copy = a.clone();
      smith::Tensor b = smith::rand(
          {4, 6}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor c = a.narrow(dim, start, 4);
      smith::Tensor d = c.view({4, 6});
      d.add_(b, 1.0);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_a = CopyToDevice(a_copy, device);
        smith::Tensor lazy_b = CopyToDevice(b, device);
        smith::Tensor lazy_c = lazy_a.narrow(dim, start, 4);
        smith::Tensor lazy_d = lazy_c.view({4, 6});
        lazy_d.add_(lazy_b, 1.0);
        AllClose(d, lazy_d);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowInNarrowUpdate) {
  for (int64_t dim : {1, -2}) {
    for (int64_t start0 : {1, -7}) {
      for (int64_t start1 : {1, -5}) {
        smith::Tensor a = smith::rand(
            {3, 8, 3},
            smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
        smith::Tensor a_copy = a.clone();
        smith::Tensor b = smith::rand(
            {3, 2, 3},
            smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
        smith::Tensor c = a.narrow(dim, start0, 6);
        smith::Tensor d = c.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        ForEachDevice([&](const smith::Device& device) {
          smith::Tensor lazy_a = CopyToDevice(a_copy, device);
          smith::Tensor lazy_b = CopyToDevice(b, device);
          smith::Tensor lazy_c = lazy_a.narrow(dim, start0, 6);
          smith::Tensor lazy_d = lazy_c.narrow(dim, start1, 2);
          lazy_d.add_(lazy_b, 1.0);
          AllClose(a, lazy_a);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowCopy) {
  for (int64_t dim : {1, -3}) {
    for (int64_t start : {2, -8}) {
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor input = smith::rand(
            {8, 10, 4, 4},
            smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor result = input.narrow_copy(dim, start, 6);
        input.add_(1);
        smith::Tensor lazy_result = lazy_input.narrow_copy(dim, start, 6);
        lazy_input.add_(1);
        AllClose(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestViewAs) {
  smith::Tensor input = smith::rand(
      {32, 20, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor empty = smith::empty({32, 320});
  smith::Tensor output = input.view_as(empty);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_empty = CopyToDevice(empty, device);
    smith::Tensor lazy_output = lazy_input.view_as(lazy_empty);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestLogSoftmax) {
  smith::Tensor input = smith::rand(
      {5, 3, 4, 2},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor output = smith::log_softmax(input, dim);
      smith::Tensor lazy_output = smith::log_softmax(lazy_input, dim);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestLogSoftmaxCast) {
  smith::Tensor input = smith::rand(
      {5, 3, 4, 2},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor output = smith::log_softmax(input, dim, smith::kDouble);
      smith::Tensor lazy_output =
          smith::log_softmax(lazy_input, dim, smith::kDouble);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestLogSoftmaxWrapper) {
  smith::Tensor input = smith::rand(
      {10, 2, 6, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor output =
          smith::_log_softmax(input, dim, /*half_to_float=*/false);
      smith::Tensor lazy_output =
          smith::_log_softmax(lazy_input, dim, /*half_to_float=*/false);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftmax) {
  smith::Tensor input = smith::rand(
      {10, 2, 6, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor output = smith::softmax(input, dim);
      smith::Tensor lazy_output = smith::softmax(lazy_input, dim);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftmaxCast) {
  smith::Tensor input = smith::rand(
      {10, 2, 6, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor output = smith::softmax(input, dim, smith::kDouble);
      smith::Tensor lazy_output =
          smith::softmax(lazy_input, dim, smith::kDouble);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftmaxWrapper) {
  smith::Tensor input = smith::rand(
      {10, 2, 6, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      smith::Tensor output =
          smith::_softmax(input, dim, /*half_to_float=*/false);
      smith::Tensor lazy_output =
          smith::_softmax(lazy_input, dim, /*half_to_float=*/false);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftplus) {
  smith::Tensor input = smith::rand(
      {2, 1, 4, 6},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::softplus(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::softplus(lazy_input);
    AllClose(output, lazy_output, /*rtol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestMaxPool1D) {
  smith::Tensor input = smith::rand(
      {1, 16, 56}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output = smith::max_pool1d(
              input,
              /*kernel_size=*/{kernel_size},
              /*stride=*/{stride},
              /*padding=*/{padding},
              /*dilation=*/{dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::max_pool1d(
                lazy_input,
                /*kernel_size=*/{kernel_size},
                /*stride=*/{stride},
                /*padding=*/{padding},
                /*dilation=*/{dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2D) {
  smith::Tensor input = smith::rand(
      {1, 4, 14, 14},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output = smith::max_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::max_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DWithIndices) {
  smith::Tensor input = smith::rand(
      {1, 4, 14, 14},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto outputs = smith::max_pool2d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            auto lazy_outputs = smith::max_pool2d_with_indices(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(std::get<0>(outputs), std::get<0>(lazy_outputs));
            AllClose(std::get<1>(outputs), std::get<1>(lazy_outputs));
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DNonSquare) {
  smith::Tensor input = smith::rand(
      {1, 4, 14, 14},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output = smith::max_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::max_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1},
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3D) {
  smith::Tensor input = smith::rand(
      {1, 1, 8, 8, 8},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output = smith::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DWithIndices) {
  smith::Tensor input = smith::rand(
      {1, 1, 8, 8, 8},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto outputs = smith::max_pool3d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            auto lazy_outputs = smith::max_pool3d_with_indices(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);

            AllClose(std::get<0>(outputs), std::get<0>(lazy_outputs));
            AllClose(std::get<1>(outputs), std::get<1>(lazy_outputs));
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DIncompleteAttributes) {
  smith::Tensor input = smith::rand(
      {1, 1, 8, 8, 8},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output = smith::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
                /*padding=*/{padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DNonSquare) {
  smith::Tensor input = smith::rand(
      {1, 1, 8, 8, 8},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output = smith::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DNoBatch) {
  smith::Tensor input = smith::rand(
      {4, 14, 14}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output = smith::max_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::max_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DNoBatch) {
  smith::Tensor input = smith::rand(
      {1, 8, 8, 8},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output = smith::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool1D) {
  smith::Tensor input = smith::rand(
      {4, 1, 28}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          smith::Tensor output = smith::avg_pool1d(
              input,
              /*kernel_size=*/{kernel_size},
              /*stride=*/{stride},
              /*padding=*/{padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::avg_pool1d(
                lazy_input,
                /*kernel_size=*/{kernel_size},
                /*stride=*/{stride},
                /*padding=*/{padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2D) {
  smith::Tensor input = smith::rand(
      {2, 1, 14, 14},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          smith::Tensor output = smith::avg_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const smith::Device& device) {
            // smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::avg_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output.to(smith::kCPU));
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2DNonSquare) {
  smith::Tensor input = smith::rand(
      {2, 1, 14, 14},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          smith::Tensor output = smith::avg_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::avg_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1},
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3D) {
  smith::Tensor input = smith::rand(
      {1, 1, 7, 7, 7},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          smith::Tensor output = smith::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DIncompleteAttributes) {
  smith::Tensor input = smith::rand(
      {1, 1, 7, 7, 7},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          smith::Tensor output = smith::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding, padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DNonSquare) {
  smith::Tensor input = smith::rand(
      {1, 1, 7, 7, 7},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          smith::Tensor output = smith::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2DNoBatch) {
  smith::Tensor input = smith::rand(
      {1, 7, 7}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          smith::Tensor output = smith::avg_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::avg_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DNoBatch) {
  smith::Tensor input = smith::rand(
      {1, 7, 7, 7},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          smith::Tensor output = smith::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output = smith::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2D) {
  smith::Tensor input = smith::rand(
      {4, 1, 28, 28},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int64_t output_size : {7, 4}) {
    smith::Tensor output =
        smith::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output =
          smith::adaptive_avg_pool2d(lazy_input, {output_size, output_size});
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3D) {
  smith::Tensor input = smith::rand(
      {9, 4, 56, 28, 28},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int64_t output_size : {7, 4}) {
    smith::Tensor output = smith::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::adaptive_avg_pool3d(
          lazy_input, {output_size, output_size, output_size});
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3DNoBatch) {
  smith::Tensor input = smith::rand(
      {3, 56, 28, 28},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int64_t output_size : {7, 4}) {
    smith::Tensor output = smith::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::adaptive_avg_pool3d(
          lazy_input, {output_size, output_size, output_size});
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2DNoBatch) {
  smith::Tensor input = smith::rand(
      {1, 56, 56}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int64_t output_size : {7, 8}) {
    smith::Tensor output =
        smith::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output =
          smith::adaptive_avg_pool2d(lazy_input, {output_size, output_size});
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool2D) {
  int kernel_size = 2;
  smith::Tensor input = smith::rand(
      {2, 2, 8, 8},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output;
          smith::Tensor indices;
          std::tie(output, indices) = smith::max_pool2d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size({input.size(2), input.size(3)});
          at::Tensor utensor =
              smith::max_unpool2d(output, indices, output_size);

          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_output = CopyToDevice(output, device);
            smith::Tensor lazy_indices = CopyToDevice(indices, device);
            at::Tensor lazy_utensor =
                smith::max_unpool2d(lazy_output, lazy_indices, output_size);
            AllClose(utensor, lazy_utensor);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool3D) {
  int kernel_size = 2;
  smith::Tensor input = smith::rand(
      {1, 1, 4, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output;
          smith::Tensor indices;
          std::tie(output, indices) = smith::max_pool3d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size(
              {input.size(2), input.size(3), input.size(4)});
          at::Tensor utensor = smith::max_unpool3d(
              output,
              indices,
              output_size,
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding});

          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_output = CopyToDevice(output, device);
            smith::Tensor lazy_indices = CopyToDevice(indices, device);
            at::Tensor lazy_utensor = smith::max_unpool3d(
                lazy_output,
                lazy_indices,
                output_size,
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding});
            AllClose(utensor, lazy_utensor);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLoss) {
  int batch = 6;
  int classes = 2;
  // TODO(asuhan): Fix the smith::kDouble case.
  for (auto dtype : {smith::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        smith::Tensor input = smith::rand(
            {batch, classes},
            smith::TensorOptions(dtype).device(DefaultDevice()));
        smith::Tensor target = smith::randint(
            std::min(ignore_index, 0),
            classes,
            {batch},
            smith::TensorOptions(smith::kLong).device(DefaultDevice()));
        smith::Tensor weight;
        if (def_weight) {
          weight = smith::rand(
              {classes}, smith::TensorOptions(dtype).device(DefaultDevice()));
        }
        for (smith::Reduction::Reduction reduction :
             {smith::Reduction::Mean,
              smith::Reduction::Sum,
              smith::Reduction::None}) {
          smith::Tensor output = smith::nll_loss(
              /*self=*/input,
              /*target=*/target,
              /*weight=*/weight,
              /*reduction=*/reduction,
              /*ignore_index=*/ignore_index);

          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_target = CopyToDevice(target, device);
            smith::Tensor lazy_weight =
                def_weight ? CopyToDevice(weight, device) : smith::Tensor();
            smith::Tensor lazy_output = smith::nll_loss(
                /*self=*/lazy_input,
                /*target=*/lazy_target,
                /*weight=*/lazy_weight,
                /*reduction=*/reduction,
                /*ignore_index=*/ignore_index);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLoss2d) {
  int batch = 6;
  int classes = 2;
  int height = 3;
  int width = 3;
  // TODO(asuhan): Fix the smith::kDouble case.
  for (auto dtype : {smith::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        smith::Tensor input = smith::rand(
            {batch, classes, height, width},
            smith::TensorOptions(dtype).device(DefaultDevice()));
        smith::Tensor target = smith::randint(
            std::min(ignore_index, 0),
            classes,
            {batch, height, width},
            smith::TensorOptions(smith::kLong).device(DefaultDevice()));
        smith::Tensor weight;
        if (def_weight) {
          weight = smith::rand(
              {classes}, smith::TensorOptions(dtype).device(DefaultDevice()));
        }
        for (smith::Reduction::Reduction reduction :
             {smith::Reduction::Mean,
              smith::Reduction::Sum,
              smith::Reduction::None}) {
          smith::Tensor output = smith::nll_loss2d(
              /*self=*/input,
              /*target=*/target,
              /*weight=*/weight,
              /*reduction=*/reduction,
              /*ignore_index=*/ignore_index);

          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_target = CopyToDevice(target, device);
            smith::Tensor lazy_weight =
                def_weight ? CopyToDevice(weight, device) : smith::Tensor();
            smith::Tensor lazy_output = smith::nll_loss2d(
                /*self=*/lazy_input,
                /*target=*/lazy_target,
                /*weight=*/lazy_weight,
                /*reduction=*/reduction,
                /*ignore_index=*/ignore_index);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSmoothL1Loss) {
  smith::Tensor input = smith::randn(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor target = smith::randn(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::None,
        smith::Reduction::Mean,
        smith::Reduction::Sum}) {
    for (double beta : {0.25, 1.}) {
      smith::Tensor output =
          smith::smooth_l1_loss(input, target, reduction, beta);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_target = CopyToDevice(target, device);
        smith::Tensor lazy_output =
            smith::smooth_l1_loss(lazy_input, lazy_target, reduction, beta);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestL1Loss) {
  smith::Tensor input = smith::randn(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor target = smith::randn(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::None,
        smith::Reduction::Mean,
        smith::Reduction::Sum}) {
    smith::Tensor output = smith::l1_loss(input, target, reduction);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_target = CopyToDevice(target, device);
      smith::Tensor lazy_output =
          smith::l1_loss(lazy_input, lazy_target, reduction);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestL1LossBackward) {
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::None,
        smith::Reduction::Mean,
        smith::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::l1_loss(inputs[0], inputs[1], reduction);
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
               {2, 4},
               smith::TensorOptions(smith::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           smith::rand(
               {2, 4},
               smith::TensorOptions(smith::kFloat).device(DefaultDevice()))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestMseLoss) {
  smith::Tensor input = smith::randn(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor target = smith::randn(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::None,
        smith::Reduction::Mean,
        smith::Reduction::Sum}) {
    smith::Tensor output = smith::mse_loss(input, target, reduction);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_target = CopyToDevice(target, device);
      smith::Tensor lazy_output =
          smith::mse_loss(lazy_input, lazy_target, reduction);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestMseLossBackward) {
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::None,
        smith::Reduction::Mean,
        smith::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::mse_loss(inputs[0], inputs[1], reduction);
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
               {2, 4},
               smith::TensorOptions(smith::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           smith::rand(
               {2, 4},
               smith::TensorOptions(smith::kFloat).device(DefaultDevice()))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestBatchNorm1D) {
  int num_features = 3;
  smith::Tensor input = smith::rand(
      {2, num_features, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {num_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor bias = smith::rand(
      {num_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor running_mean = smith::zeros(
      {num_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor running_var = smith::ones(
      {num_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  double momentum = 0.1;
  double eps = 0.5;
  smith::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      smith::Tensor output = smith::batch_norm(
          /*input=*/input,
          /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean,
          /*running_var=*/running_var,
          /*training=*/training,
          /*momentum=*/momentum,
          /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        smith::Tensor lazy_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        smith::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
        smith::Tensor lazy_running_var = CopyToDevice(running_var, device);
        smith::Tensor lazy_output = smith::batch_norm(
            /*input=*/lazy_input,
            /*weight=*/lazy_weight,
            /*bias=*/lazy_bias,
            /*running_mean=*/lazy_running_mean,
            /*running_var=*/lazy_running_var,
            /*training=*/training,
            /*momentum=*/momentum,
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestBatchNorm2D) {
  int num_features = 3;
  smith::Tensor input = smith::rand(
      {2, num_features, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {num_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor bias = smith::rand(
      {num_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor running_mean = smith::zeros(
      {num_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor running_var = smith::ones(
      {num_features},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  double momentum = 0.1;
  double eps = 0.5;
  smith::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      smith::Tensor output = smith::batch_norm(
          /*input=*/input,
          /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean,
          /*running_var=*/running_var,
          /*training=*/training,
          /*momentum=*/momentum,
          /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        smith::Tensor lazy_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        smith::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
        smith::Tensor lazy_running_var = CopyToDevice(running_var, device);
        smith::Tensor lazy_output = smith::batch_norm(
            /*input=*/lazy_input,
            /*weight=*/lazy_weight,
            /*bias=*/lazy_bias,
            /*running_mean=*/lazy_running_mean,
            /*running_var=*/lazy_running_var,
            /*training=*/training,
            /*momentum=*/momentum,
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDim) {
  smith::Tensor input = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    EXPECT_EQ(input.dim(), lazy_input.dim());
  });
}

TEST_F(LazyOpsTest, TestContiguous) {
  smith::Tensor input = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::native::contiguous(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::native::contiguous(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSqueezeAll) {
  smith::Tensor input = smith::rand(
      {2, 1, 3, 1},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::squeeze(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::squeeze(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSqueezeAllInPlace) {
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor input = smith::rand(
        {2, 1, 3, 1},
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = input.squeeze_();
    smith::Tensor lazy_output = lazy_input.squeeze_();
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
    ASSERT_EQ(input.dim(), lazy_input.dim());
    for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
      ASSERT_EQ(input.size(dim_idx), lazy_input.size(dim_idx));
    }
  });
}

TEST_F(LazyOpsTest, TestSqueezeOne) {
  smith::Tensor input = smith::rand(
      {2, 1, 3, 1},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor output = smith::squeeze(input, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::squeeze(lazy_input, dim);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestSqueezeOneInPlace) {
  int rank = 4;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor input = smith::rand(
          {2, 1, 3, 1},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor output = input.squeeze_(dim);
      smith::Tensor lazy_output = lazy_input.squeeze_(dim);
      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
      ASSERT_EQ(input.dim(), lazy_input.dim());
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), lazy_input.size(dim_idx));
      }
    });
  }
}

TEST_F(LazyOpsTest, TestUnsqueeze) {
  smith::Tensor input = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor output = smith::unsqueeze(input, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::unsqueeze(lazy_input, dim);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestUnsqueezeInPlace) {
  smith::Tensor input = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor output = input.unsqueeze_(dim);
      smith::Tensor lazy_output = lazy_input.unsqueeze_(dim);
      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
      ASSERT_EQ(input.dim(), lazy_input.dim());
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), lazy_input.size(dim_idx));
      }
    });
  }
}

TEST_F(LazyOpsTest, TestMaskedFill) {
  smith::Tensor input = smith::rand(
      {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor mask = smith::randint(
      0, 2, {2, 3}, smith::TensorOptions(smith::kBool).device(DefaultDevice()));
  smith::Scalar value(42);
  smith::Tensor result = smith::masked_fill(input, mask, value);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_mask = CopyToDevice(mask, device);
    smith::Tensor lazy_result =
        smith::masked_fill(lazy_input, lazy_mask, value);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestMaskedFillInPlace) {
  smith::Scalar value(42);
  smith::Tensor mask = smith::randint(
      0, 2, {2, 3}, smith::TensorOptions(smith::kBool).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor input = smith::rand(
        {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_mask = CopyToDevice(mask, device);
    smith::Tensor result = input.masked_fill_(mask, value);
    smith::Tensor lazy_result = lazy_input.masked_fill_(lazy_mask, value);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestMaskedFillBroadcast) {
  smith::Tensor input = smith::rand(
      {2, 5, 4, 3},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor mask = smith::randint(
      0, 2, {4, 1}, smith::TensorOptions(smith::kBool).device(DefaultDevice()));
  smith::Scalar value(42);
  smith::Tensor result = smith::masked_fill(input, mask, value);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_mask = CopyToDevice(mask, device);
    smith::Tensor lazy_result =
        smith::masked_fill(lazy_input, lazy_mask, value);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestFill) {
  smith::Scalar value(42);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor input = smith::empty(
        {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor result = smith::fill_(input, value);
    smith::Tensor lazy_result = smith::fill_(lazy_input, value);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestFillWithRank0) {
  smith::Tensor value = smith::scalar_tensor(42);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor input = smith::empty(
        {2, 3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor result = smith::fill_(input, value);
    smith::Tensor lazy_value = CopyToDevice(value, device);
    smith::Tensor lazy_result = smith::fill_(lazy_input, value);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestPermute) {
  smith::Tensor input = smith::rand(
      {2, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  int rank = input.dim();
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(
            dims_permutation.begin(),
            dims_permutation.end(),
            [rank](int64_t& dim) { dim -= rank; });
      }
      smith::Tensor output = input.permute(dims_permutation);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_output = lazy_input.permute(dims_permutation);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestPermuteMod) {
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  std::vector<int64_t> input_sizes = {2, 3, 4};
  int rank = input_sizes.size();
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(
            dims_permutation.begin(),
            dims_permutation.end(),
            [rank](int64_t& dim) { dim -= rank; });
      }
      smith::Tensor input = smith::zeros(
          input_sizes,
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor one = smith::tensor(
          1.0, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor output = input.permute(dims_permutation);
      output.add_(one, 1.0);
      input.add_(one, 1.0);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor xinput = smith::zeros(
            input_sizes,
            smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
        smith::Tensor lazy_input = CopyToDevice(xinput, device);
        smith::Tensor lazy_one = CopyToDevice(one, device);
        smith::Tensor lazy_output = lazy_input.permute(dims_permutation);
        lazy_output.add_(lazy_one, 1.0);
        lazy_input.add_(lazy_one, 1.0);
        AllClose(output, lazy_output);
        AllClose(input, lazy_input);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestFlip) {
  smith::Tensor input = smith::rand(
      {2, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<std::vector<int64_t>> dim_powerset = {
      {0}, {1}, {2}, {0, 1}, {1, 2}, {2, 0}, {0, 1, 2}};
  for (std::vector<int64_t> flip_dims : dim_powerset) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(
            flip_dims.begin(), flip_dims.end(), [](int64_t& dim) { dim -= 3; });
      }
      smith::Tensor output = smith::flip(input, flip_dims);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        smith::Tensor lazy_output = smith::flip(lazy_input, flip_dims);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestPixelShuffle) {
  smith::Tensor input = smith::rand(
      {5, 18, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int upscale_factor = 3;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = smith::pixel_shuffle(input, upscale_factor);
    smith::Tensor lazy_output =
        smith::pixel_shuffle(lazy_input, upscale_factor);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSumToSize) {
  smith::Tensor input = smith::rand(
      {4, 6, 3, 7},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> out_size = {4, 1, 1, 7};
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = input.sum_to_size(out_size);
    smith::Tensor lazy_output = lazy_input.sum_to_size(out_size);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTransposeDims) {
  smith::Tensor input = smith::rand(
      {2, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int dim0 = 0;
  int dim1 = 2;
  smith::Tensor output = smith::transpose(input, dim0, dim1);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::transpose(lazy_input, dim0, dim1);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTransposeDimsMod) {
  std::vector<int64_t> input_sizes = {2, 3, 4};
  int dim0 = 0;
  int dim1 = 2;
  smith::Tensor input = smith::zeros(
      input_sizes, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor one = smith::tensor(
      1.0, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::transpose(input, dim0, dim1);
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor xinput = smith::zeros(
        input_sizes,
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor lazy_input = CopyToDevice(xinput, device);
    smith::Tensor lazy_one = CopyToDevice(one, device);
    smith::Tensor lazy_output = smith::transpose(lazy_input, dim0, dim1);
    lazy_output.add_(lazy_one, 1.0);
    lazy_input.add_(lazy_one, 1.0);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestTransposeDimsInPlace) {
  smith::Tensor input = smith::rand(
      {2, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int dim0 = 0;
  int dim1 = 2;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output = input.transpose_(dim0, dim1);
    smith::Tensor lazy_output = lazy_input.transpose_(dim0, dim1);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestSplit) {
  smith::Tensor input = smith::rand(
      {7, 8, 9}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int split_size : {2, 3}) {
    for (int dim = -rank; dim < rank; ++dim) {
      std::vector<smith::Tensor> outputs = smith::split(input, split_size, dim);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_input = CopyToDevice(input, device);
        std::vector<smith::Tensor> lazy_outputs =
            smith::split(lazy_input, split_size, dim);
        ASSERT_EQ(outputs.size(), lazy_outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
          AllClose(outputs[i], lazy_outputs[i]);
        }
      });
    }
  }
}

TEST_F(LazyOpsTest, TestSplitEmpty) {
  smith::Tensor input = smith::rand(
      {0}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int split_size = 0;
  int dim = 0;
  std::vector<smith::Tensor> outputs = smith::split(input, split_size, dim);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    std::vector<smith::Tensor> lazy_outputs =
        smith::split(lazy_input, split_size, dim);
    ASSERT_EQ(outputs.size(), lazy_outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      AllClose(outputs[i], lazy_outputs[i]);
    }
  });
}

TEST_F(LazyOpsTest, TestSplitWithSizes) {
  smith::Tensor input = smith::rand(
      {15, 15, 15},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<smith::Tensor> outputs =
        smith::split_with_sizes(input, {4, 5, 6}, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      std::vector<smith::Tensor> lazy_outputs =
          smith::split_with_sizes(lazy_input, {4, 5, 6}, dim);
      ASSERT_EQ(outputs.size(), lazy_outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        AllClose(outputs[i], lazy_outputs[i]);
      }
    });
  }
}

TEST_F(LazyOpsTest, TestCrossImplicitDim) {
  std::vector<std::vector<int64_t>> dim_sizes = {
      {4, 5, 3}, {4, 3, 5}, {3, 4, 5}};
  for (auto dim_size : dim_sizes) {
    smith::Tensor input = smith::rand(
        dim_size, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor other = smith::rand(
        dim_size, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    smith::Tensor result = smith::cross(input, other);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_other = CopyToDevice(other, device);
      smith::Tensor lazy_result = smith::cross(lazy_input, lazy_other);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCrossExplicitDim) {
  std::vector<int64_t> dim_size = {3, 3};
  smith::Tensor input = smith::rand(
      dim_size, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor other = smith::rand(
      dim_size, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  int rank = dim_size.size();
  for (int dim = -rank; dim < rank; ++dim) {
    smith::Tensor result = smith::cross(input, other, dim);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_other = CopyToDevice(other, device);
      smith::Tensor lazy_result = smith::cross(lazy_input, lazy_other, dim);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCrossZeroDim) {
  smith::Tensor input = smith::rand(
      {0, 1, 3, 0},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor result = smith::cross(input, input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::cross(lazy_input, lazy_input);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestTriu) {
  int size = 5;
  smith::Tensor input = smith::rand(
      {size, size},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output = smith::triu(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::triu(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTriuNonSquare) {
  int size = 5;
  smith::Tensor input = smith::rand(
      {size, size + 1},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output = smith::triu(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::triu(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTriuBatch) {
  int size = 5;
  int batch_size = 3;
  smith::Tensor input = smith::rand(
      {batch_size, size, size},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output = smith::triu(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::triu(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTril) {
  int size = 5;
  smith::Tensor input = smith::rand(
      {size, size},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output = smith::tril(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::tril(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTrilNonSquare) {
  int size = 5;
  smith::Tensor input = smith::rand(
      {size, size + 1},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output = smith::tril(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::tril(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTrilBatch) {
  int size = 5;
  int batch_size = 3;
  smith::Tensor input = smith::rand(
      {batch_size, size, size},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output = smith::tril(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::tril(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTriuInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor input = smith::rand(
          {size, size},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor output = input.triu_(diagonal);
      smith::Tensor lazy_output = lazy_input.triu_(diagonal);
      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
    });
  }
}

TEST_F(LazyOpsTest, TestTrilInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor input = smith::rand(
          {size, size},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor output = input.tril_(diagonal);
      smith::Tensor lazy_output = lazy_input.tril_(diagonal);
      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
    });
  }
}

TEST_F(LazyOpsTest, TestTrace) {
  int n = 5;
  smith::Tensor input = smith::rand(
      {n, n}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::trace(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::trace(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTraceWide) {
  int lines = 3;
  int cols = 5;
  smith::Tensor input = smith::rand(
      {lines, cols},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::trace(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::trace(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTraceNarrow) {
  int lines = 5;
  int cols = 3;
  smith::Tensor input = smith::rand(
      {lines, cols},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor output = smith::trace(input);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::trace(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestDiagRank1) {
  int size = 7;
  smith::Tensor input = smith::rand(
      {size}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -2 * size; diagonal <= 2 * size; ++diagonal) {
    smith::Tensor output = smith::diag(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::diag(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagRank2) {
  int size = 7;
  smith::Tensor input = smith::rand(
      {size, size},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output = smith::diag(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::diag(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagFlat) {
  smith::Tensor input = smith::rand(
      {4, 3, 6, 7},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int diagonal = -10; diagonal < 10; ++diagonal) {
    smith::Tensor output = smith::diagflat(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::diagflat(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonal) {
  int size = 5;
  smith::Tensor input = smith::rand(
      {size, size},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output = smith::diagonal(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::diagonal(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonalUpdate) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    auto input = smith::rand(
        {size, size},
        smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
    auto input_clone = input.clone();
    auto output = smith::diagonal(input, diagonal);
    output.add_(1);

    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input_clone, device);
      smith::Tensor lazy_output = smith::diagonal(lazy_input, diagonal);
      lazy_output.add_(1);

      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonalNonSquare) {
  int size = 5;
  smith::Tensor input = smith::rand(
      {size, size + 1},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output = smith::diagonal(input, diagonal);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output = smith::diagonal(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonalBatch) {
  int size = 5;
  int batch_size = 3;
  int dim1 = 1;
  int dim2 = 2;
  smith::Tensor input = smith::rand(
      {batch_size, size, size},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    smith::Tensor output =
        smith::diagonal(input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor lazy_input = CopyToDevice(input, device);
      smith::Tensor lazy_output =
          smith::diagonal(lazy_input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestFlatten) {
  smith::Tensor input = smith::rand({4, 7, 5, 3});
  int rank = input.dim();
  for (int pos_start_dim = 0; pos_start_dim < rank; ++pos_start_dim) {
    for (int pos_end_dim = pos_start_dim; pos_end_dim < rank; ++pos_end_dim) {
      for (bool negative_start_dim : {false, true}) {
        for (bool negative_end_dim : {false, true}) {
          int start_dim =
              negative_start_dim ? pos_start_dim - rank : pos_start_dim;
          int end_dim = negative_end_dim ? pos_end_dim - rank : pos_end_dim;
          smith::Tensor output = smith::flatten(input, start_dim, end_dim);
          ForEachDevice([&](const smith::Device& device) {
            smith::Tensor lazy_input = CopyToDevice(input, device);
            smith::Tensor lazy_output =
                smith::flatten(lazy_input, start_dim, end_dim);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestLogicalAnd) {
  for (smith::ScalarType scalar_type1 :
       {smith::kFloat,
        smith::kByte,
        smith::kChar,
        smith::kShort,
        smith::kInt,
        smith::kLong}) {
    smith::Tensor lhs = isFloatingType(scalar_type1)
        ? smith::rand({3, 4}, smith::TensorOptions(scalar_type1))
        : smith::randint(0, 100, {3, 4}, smith::TensorOptions(scalar_type1));
    for (smith::ScalarType scalar_type2 :
         {smith::kFloat,
          smith::kByte,
          smith::kChar,
          smith::kShort,
          smith::kInt,
          smith::kLong}) {
      smith::Tensor rhs = isFloatingType(scalar_type2)
          ? smith::rand({3, 4}, smith::TensorOptions(scalar_type2))
          : smith::randint(1, 100, {3, 4}, smith::TensorOptions(scalar_type2));
      smith::Tensor result = smith::logical_and(lhs, rhs);
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
        smith::Tensor lazy_rhs = CopyToDevice(rhs, device);
        smith::Tensor lazy_result = smith::logical_and(lazy_lhs, lazy_rhs);
        AllEqual(result, lazy_result);
      });
    }
  }

  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("xla::logical_and_out", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestBitwiseAnd) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Tensor rhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor lazy_rhs = CopyToDevice(rhs, device);
    smith::Tensor lazy_result = lazy_lhs.__and__(lazy_rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndInPlace) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Tensor rhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor result = lhs.__iand__(rhs);
    smith::Tensor lazy_rhs = CopyToDevice(rhs, device);
    smith::Tensor lazy_result = lazy_lhs.__iand__(lazy_rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndScalar) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Scalar rhs(123456789);
  smith::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor lazy_result = lazy_lhs.__and__(rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndScalarInPlace) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Scalar rhs(123456789);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor result = lhs.__iand__(rhs);
    smith::Tensor lazy_result = lazy_lhs.__iand__(rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndPromotion) {
  smith::Tensor input = smith::rand(
      {4, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor view = input.reshape(-1);
  smith::Tensor result = smith::__and__(view.gt(0), view.ne(0));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_view = lazy_input.reshape(-1);
    smith::Tensor lazy_result =
        smith::__and__(lazy_view.gt(0), lazy_view.ne(0));
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOr) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Tensor rhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor lazy_rhs = CopyToDevice(rhs, device);
    smith::Tensor lazy_result = lazy_lhs.__or__(lazy_rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOrInPlace) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Tensor rhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor result = lhs.__ior__(rhs);
    smith::Tensor lazy_rhs = CopyToDevice(rhs, device);
    smith::Tensor lazy_result = lazy_lhs.__ior__(lazy_rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOrScalar) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Scalar rhs(123456789);
  smith::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor lazy_result = lazy_lhs.__or__(rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOrScalarInPlace) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Scalar rhs(123456789);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor result = lhs.__ior__(rhs);
    smith::Tensor lazy_result = lazy_lhs.__ior__(rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseXor) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Tensor rhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor lazy_rhs = CopyToDevice(rhs, device);
    smith::Tensor lazy_result = lazy_lhs.__xor__(lazy_rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseXorInPlace) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Tensor rhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor result = lhs.__ixor__(rhs);
    smith::Tensor lazy_rhs = CopyToDevice(rhs, device);
    smith::Tensor lazy_result = lazy_lhs.__ixor__(lazy_rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseXorScalar) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Scalar rhs(123456789);
  smith::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor lazy_result = lazy_lhs.__xor__(rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseXorScalarInPlace) {
  smith::Tensor lhs = smith::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      smith::TensorOptions(smith::kInt));
  smith::Scalar rhs(123456789);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_lhs = CopyToDevice(lhs, device);
    smith::Tensor result = lhs.__ixor__(rhs);
    smith::Tensor lazy_result = lazy_lhs.__ixor__(rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestLshift) {
  smith::Tensor input = smith::ones(
      {4, 2}, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Tensor shift_amount = smith::randint(
      16,
      input.sizes(),
      smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Tensor result = smith::__lshift__(input, shift_amount);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    smith::Tensor lazy_result =
        smith::__lshift__(lazy_input, lazy_shift_amount);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLshiftInPlace) {
  smith::Tensor input = smith::ones(
      {4, 2}, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor shift_amount = smith::randint(
        16,
        input.sizes(),
        smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
    smith::Tensor result = input.__ilshift__(shift_amount);
    smith::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    smith::Tensor lazy_result = lazy_input.__ilshift__(lazy_shift_amount);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestLshiftScalar) {
  smith::Tensor input = smith::ones(
      {4, 2}, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Scalar shift_amount = 3;
  smith::Tensor result = smith::__lshift__(input, shift_amount);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::__lshift__(lazy_input, shift_amount);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLshiftScalarInPlace) {
  smith::Tensor input = smith::ones(
      {4, 2}, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Scalar shift_amount = 3;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor result = input.__ilshift__(shift_amount);
    smith::Tensor lazy_result = lazy_input.__ilshift__(shift_amount);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestRshift) {
  smith::Tensor input = smith::ones(
      {4, 2}, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Tensor shift_amount = smith::randint(
      16,
      input.sizes(),
      smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Tensor result = smith::__rshift__(input, shift_amount);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    smith::Tensor lazy_result =
        smith::__rshift__(lazy_input, lazy_shift_amount);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestRshiftInPlace) {
  smith::Tensor input = smith::ones(
      {4, 2}, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor shift_amount = smith::randint(
        16,
        input.sizes(),
        smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
    smith::Tensor result = input.__irshift__(shift_amount);
    smith::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    smith::Tensor lazy_result = lazy_input.__irshift__(lazy_shift_amount);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestRshiftScalar) {
  smith::Tensor input = smith::ones(
      {4, 2}, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Scalar shift_amount = 3;
  smith::Tensor result = smith::__rshift__(input, shift_amount);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_result = smith::__rshift__(lazy_input, shift_amount);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestRshiftScalarInPlace) {
  smith::Tensor input = smith::ones(
      {4, 2}, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Scalar shift_amount = 3;
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor result = input.__irshift__(shift_amount);
    smith::Tensor lazy_result = lazy_input.__irshift__(shift_amount);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestMeshgrid) {
  smith::Tensor a = smith::rand(
      {3}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor b = smith::rand(
      {2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor c = smith::rand(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  auto d = smith::meshgrid({a, b, c});
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_a = CopyToDevice(a, device);
    smith::Tensor lazy_b = CopyToDevice(b, device);
    smith::Tensor lazy_c = CopyToDevice(c, device);
    auto lazy_d = smith::meshgrid({lazy_a, lazy_b, lazy_c});
    EXPECT_EQ(d.size(), lazy_d.size());
    for (size_t i = 0; i < d.size(); ++i) {
      AllClose(d[i], lazy_d[i]);
    }
  });
}

TEST_F(LazyOpsTest, TestConstantPad) {
  smith::Tensor input = smith::rand(
      {4, 2, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 2, 3, 4, 5, 6};
  float pad_value = 5;
  smith::Tensor output = smith::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output =
        smith::constant_pad_nd(lazy_input, pad, pad_value);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestConstantPadIncomplete) {
  smith::Tensor input = smith::rand(
      {4, 2, 5}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 2};
  float pad_value = 5;
  smith::Tensor output = smith::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output =
        smith::constant_pad_nd(lazy_input, pad, pad_value);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReflectionPad2dRank3) {
  smith::Tensor input = smith::rand(
      {2, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{2, 2, 2, 2};
  smith::Tensor output = smith::reflection_pad2d(input, pad);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::reflection_pad2d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReflectionPad2dRank4) {
  smith::Tensor input = smith::rand(
      {2, 2, 3, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{2, 2, 2, 2};
  smith::Tensor output = smith::reflection_pad2d(input, pad);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::reflection_pad2d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReflectionPad2dBackward) {
  std::vector<int64_t> pad{2, 3, 1, 2};
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::reflection_pad2d(inputs[0], pad);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {1, 2, 4, 4},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad1d) {
  smith::Tensor input = smith::rand(
      {1, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 2};
  smith::Tensor output = smith::replication_pad1d(input, pad);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::replication_pad1d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad1dZeroPad) {
  smith::Tensor input = smith::rand(
      {1, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 0};
  smith::Tensor output = smith::replication_pad1d(input, pad);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::replication_pad1d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad1dBackward) {
  std::vector<int64_t> pad{2, 3};
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::replication_pad1d(inputs[0], pad);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 4},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad2d) {
  smith::Tensor input = smith::rand(
      {1, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 2, 2, 1};
  smith::Tensor output = smith::replication_pad2d(input, pad);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::replication_pad2d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad2dZeroPad) {
  smith::Tensor input = smith::rand(
      {1, 3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 0, 0, 1};
  smith::Tensor output = smith::replication_pad2d(input, pad);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::replication_pad2d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad2dBackward) {
  std::vector<int64_t> pad{2, 3, 1, 1};
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::replication_pad2d(inputs[0], pad);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 3, 4},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestAsStrided) {
  smith::Tensor input = smith::rand(
      {128, 320}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  smith::Tensor output =
      smith::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output =
        smith::as_strided(lazy_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAsStridedInPlace) {
  smith::Tensor input = smith::rand(
      {128, 320}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor output =
        smith::as_strided_(input, /*size=*/size, /*stride=*/stride);
    smith::Tensor lazy_output =
        smith::as_strided_(lazy_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestAsStridedWithOffset) {
  smith::Tensor input = smith::rand(
      {4, 8, 2}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  int64_t storage_offset = 4;
  smith::Tensor output = smith::as_strided(
      input,
      /*size=*/size,
      /*stride=*/stride,
      /*storage_offset=*/storage_offset);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input, device);
    smith::Tensor lazy_output = smith::as_strided(
        lazy_input,
        /*size=*/size,
        /*stride=*/stride,
        /*storage_offset=*/storage_offset);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAsStridedWithInplaceCopy) {
  smith::Tensor grad = smith::ones(
      {4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  std::vector<int64_t> size = {4};
  std::vector<int64_t> stride = {1};
  smith::Tensor output = smith::zeros({4}, grad.options());
  output.as_strided(size, stride).copy_(grad);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_grad = CopyToDevice(grad, device);
    smith::Tensor lazy_output = smith::zeros({4}, lazy_grad.options());
    lazy_output.as_strided(size, stride).copy_(lazy_grad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestEmptyStrided) {
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  smith::Tensor output = smith::empty_strided(/*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_output =
        smith::empty_strided(/*size=*/size, /*stride=*/stride);
    EXPECT_EQ(output.sizes(), lazy_output.sizes());
    EXPECT_EQ(output.strides(), lazy_output.strides());
  });
}

TEST_F(LazyOpsTest, TestAvgPool2DBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
            return smith::avg_pool2d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const smith::Device& device) {
            TestBackward(
                {smith::rand(
                    {1, 1, 7, 7},
                    smith::TensorOptions(smith::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
            return smith::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const smith::Device& device) {
            TestBackward(
                {smith::rand(
                    {1, 1, 7, 7, 7},
                    smith::TensorOptions(smith::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2DNoBatchBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
            return smith::avg_pool2d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const smith::Device& device) {
            TestBackward(
                {smith::rand(
                    {1, 7, 7},
                    smith::TensorOptions(smith::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DNoBatchBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
            return smith::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const smith::Device& device) {
            TestBackward(
                {smith::rand(
                    {1, 7, 7, 7},
                    smith::TensorOptions(smith::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3DNoBatchBackward) {
  if (IsCuda()) {
    GTEST_SKIP();
  }
  for (int64_t output_size : {7, 4}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::adaptive_avg_pool3d(
          inputs[0], {output_size, output_size, output_size});
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
              {1, 56, 28, 28},
              smith::TensorOptions(smith::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3DBackward) {
  if (IsCuda()) {
    GTEST_SKIP();
  }
  for (int64_t output_size : {7, 4}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::adaptive_avg_pool3d(
          inputs[0], {output_size, output_size, output_size});
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
              {4, 1, 56, 28, 28},
              smith::TensorOptions(smith::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2DBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
              {4, 1, 56, 56},
              smith::TensorOptions(smith::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2DNoBatchBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
              {1, 56, 56},
              smith::TensorOptions(smith::kFloat).requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestConv2D) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 3; ++dilation) {
          for (int groups :
               {1, 2, 4}) { // covers normal, grouped, depthwise conv.
            ForEachDevice([&](const smith::Device& device) {
              smith::Tensor input = smith::rand(
                  {1, in_channels, 7, 7},
                  smith::TensorOptions(smith::kDouble).device(DefaultDevice()));
              smith::Tensor weight = smith::rand(
                  {out_channels,
                   in_channels / groups,
                   kernel_size,
                   kernel_size},
                  smith::TensorOptions(smith::kDouble).device(DefaultDevice()));
              smith::Tensor bias = with_bias
                  ? smith::rand(
                        {out_channels},
                        smith::TensorOptions(smith::kDouble)
                            .device(DefaultDevice()))
                  : smith::Tensor();

              smith::Tensor lazy_input = CopyToDevice(input, device);
              smith::Tensor lazy_weight = CopyToDevice(weight, device);
              smith::Tensor lazy_bias =
                  with_bias ? CopyToDevice(bias, device) : smith::Tensor();

              smith::Tensor output = smith::conv2d(
                  input,
                  weight,
                  bias,
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*dilation=*/{dilation, dilation},
                  groups);
              smith::Tensor lazy_output = smith::conv2d(
                  lazy_input,
                  lazy_weight,
                  lazy_bias,
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*dilation=*/{dilation, dilation},
                  groups);
              AllClose(output, lazy_output);
            });
          }
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestConv2DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 3; ++dilation) {
          for (int groups :
               {1, 2, 4}) { // covers normal, grouped, depthwise conv.
            auto testfn =
                [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
              return smith::conv2d(
                  inputs[0],
                  inputs[1],
                  inputs[2],
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*dilation=*/{dilation, dilation},
                  groups);
            };

            ForEachDevice([&](const smith::Device& device) {
              smith::Tensor bias = with_bias
                  ? smith::rand(
                        {out_channels},
                        smith::TensorOptions(smith::kDouble)
                            .device(DefaultDevice()))
                  : smith::Tensor();
              TestBackward(
                  {smith::rand(
                       {1, in_channels, 7, 7},
                       smith::TensorOptions(smith::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   smith::rand(
                       {out_channels,
                        in_channels / groups,
                        kernel_size,
                        kernel_size},
                       smith::TensorOptions(smith::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   bias},
                  device,
                  testfn);
            });
          }
        };
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTransposedConv2DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (int dilation = 1; dilation <= 2; ++dilation) {
        for (int output_padding = 0;
             output_padding < std::max(stride, dilation);
             ++output_padding) {
          for (bool with_bias : {true, false}) {
            for (int groups :
                 {1, 2, 4}) { // covers normal, grouped, depthwise conv.
              auto testfn = [&](const std::vector<smith::Tensor>& inputs)
                  -> smith::Tensor {
                return smith::conv_transpose2d(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    /*stride=*/{stride, stride + 1},
                    /*padding=*/{padding, padding + 1},
                    /*output_padding=*/output_padding,
                    /*groups=*/groups,
                    /*dilation=*/{dilation, dilation + 1});
              };
              ForEachDevice([&](const smith::Device& device) {
                smith::Tensor input = smith::rand(
                    {4, out_channels, 7, 7},
                    smith::TensorOptions(smith::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true));
                smith::Tensor weight = smith::rand(
                    {out_channels,
                     in_channels / groups,
                     kernel_size,
                     kernel_size},
                    smith::TensorOptions(smith::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true));
                smith::Tensor bias = with_bias
                    ? smith::rand(
                          {in_channels},
                          smith::TensorOptions(smith::kFloat)
                              .device(DefaultDevice())
                              .requires_grad(true))
                    : smith::Tensor();
                TestBackward(
                    {input, weight, bias},
                    device,
                    testfn,
                    /*rtol=*/1e-5,
                    /*atol=*/1e-5);
              });
            }
          };
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestConv3DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 1; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          for (int groups :
               {1, 2, 4}) { // covers normal, grouped, depthwise conv.
            auto testfn =
                [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
              return smith::conv3d(
                  inputs[0],
                  inputs[1],
                  inputs[2],
                  /*stride=*/{stride, stride, stride},
                  /*padding=*/{padding, padding, padding},
                  /*dilation=*/{dilation, dilation, dilation},
                  groups);
            };

            ForEachDevice([&](const smith::Device& device) {
              smith::Tensor bias = with_bias
                  ? smith::rand(
                        {out_channels},
                        smith::TensorOptions(smith::kDouble)
                            .device(DefaultDevice()))
                  : smith::Tensor();
              TestBackward(
                  {smith::rand(
                       {4, in_channels, 7, 7, 7},
                       smith::TensorOptions(smith::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   smith::rand(
                       {out_channels,
                        in_channels / groups,
                        kernel_size,
                        kernel_size,
                        kernel_size},
                       smith::TensorOptions(smith::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   bias},
                  device,
                  testfn);
            });
          }
        };
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTransposedConv3DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (int dilation = 1; dilation <= 2; ++dilation) {
        for (int output_padding = 0;
             output_padding < std::max(stride, dilation);
             ++output_padding) {
          for (bool with_bias : {true, false}) {
            for (int groups :
                 {1, 2, 4}) { // covers normal, grouped, depthwise conv.
              auto testfn = [&](const std::vector<smith::Tensor>& inputs)
                  -> smith::Tensor {
                return smith::conv_transpose3d(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    /*stride=*/{stride, stride + 1, stride},
                    /*padding=*/{padding, padding + 1, stride},
                    /*output_padding=*/output_padding,
                    /*groups=*/groups,
                    /*dilation=*/{dilation, dilation + 1, dilation});
              };
              ForEachDevice([&](const smith::Device& device) {
                smith::Tensor input = smith::rand(
                    {4, out_channels, 7, 7, 7},
                    smith::TensorOptions(smith::kDouble)
                        .device(DefaultDevice())
                        .requires_grad(true));
                smith::Tensor weight = smith::rand(
                    {out_channels,
                     in_channels / groups,
                     kernel_size,
                     kernel_size,
                     kernel_size},
                    smith::TensorOptions(smith::kDouble)
                        .device(DefaultDevice())
                        .requires_grad(true));
                smith::Tensor bias = with_bias
                    ? smith::rand(
                          {in_channels},
                          smith::TensorOptions(smith::kDouble)
                              .device(DefaultDevice())
                              .requires_grad(true))
                    : smith::Tensor();
                TestBackward({input, weight, bias}, device, testfn);
              });
            }
          };
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
          return smith::max_pool2d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const smith::Device& device) {
          TestBackward(
              {smith::rand(
                  {1, 2, 8, 8},
                  smith::TensorOptions(smith::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
          return smith::max_pool3d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const smith::Device& device) {
          TestBackward(
              {smith::rand(
                  {1, 2, 4, 4, 4},
                  smith::TensorOptions(smith::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DNoBatchBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
          return smith::max_pool2d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const smith::Device& device) {
          TestBackward(
              {smith::rand(
                  {2, 8, 8},
                  smith::TensorOptions(smith::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DNoBatchBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
          return smith::max_pool3d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const smith::Device& device) {
          TestBackward(
              {smith::rand(
                  {2, 4, 4, 4},
                  smith::TensorOptions(smith::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool2DBackward) {
  int kernel_size = 2;
  smith::Tensor input = smith::rand(
      {2, 2, 8, 8},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output;
          smith::Tensor indices;
          std::tie(output, indices) = smith::max_pool2d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size({input.size(2), input.size(3)});
          auto testfn =
              [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
            return smith::max_unpool2d(inputs[0], inputs[1], output_size);
          };

          ForEachDevice([&](const smith::Device& device) {
            TestBackward(
                {output.requires_grad_(true), indices}, device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool3DBackward) {
  int kernel_size = 2;
  smith::Tensor input = smith::rand(
      {1, 1, 4, 4, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          smith::Tensor output;
          smith::Tensor indices;
          std::tie(output, indices) = smith::max_pool3d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size(
              {input.size(2), input.size(3), input.size(4)});
          auto testfn =
              [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
            return smith::max_unpool3d(
                inputs[0],
                inputs[1],
                output_size,
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding});
          };

          ForEachDevice([&](const smith::Device& device) {
            TestBackward(
                {output.requires_grad_(true), indices}, device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTanhBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::tanh(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 2},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn,
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSigmoidBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::sigmoid(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 2},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestLogSigmoidBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::log_sigmoid(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 2},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn,
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLogSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::log_softmax(inputs[0], dim);
    };

    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
              {5, 3, 4, 2},
              smith::TensorOptions(smith::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn,
          /*rtol=*/1e-3,
          /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::softmax(inputs[0], dim);
    };

    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
              {5, 3, 4, 2},
              smith::TensorOptions(smith::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn,
          /*rtol=*/1e-3,
          /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestSoftplusBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::softplus(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 1, 4, 6},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn,
        /*rtol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestReluBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::relu(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 1, 4, 6},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestRreluBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::rrelu(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 1, 4, 6},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestHardshrinkBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::hardshrink(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::randn(
            {100},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestSoftshrinkBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::softshrink(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::randn(
            {100},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestHardtanhBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::hardtanh(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::randn(
            {100},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestEluBackward) {
  smith::Scalar alpha = 0.5;
  smith::Scalar scale = 2.5;
  smith::Scalar input_scale = 1.5;
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::elu(inputs[0], alpha, scale, input_scale);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 1, 4, 6},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestGeluBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::gelu(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 3},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
  ExpectCounterChanged("lazy::gelu_backward", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLeakyReluBackward) {
  double negative_slope = 0.01;
  auto testfn = [=](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::leaky_relu(inputs[0], negative_slope);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 1, 4, 6},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestTransposeBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::t(inputs[0]);
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {2, 3},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestAddMatMulBackward) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::addmm(inputs[0], inputs[1], inputs[2], /*beta=*/beta);
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {smith::rand(
               {labels},
               smith::TensorOptions(smith::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           smith::rand(
               {in_channels, out_channels},
               smith::TensorOptions(smith::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           smith::rand(
               {out_channels, labels},
               smith::TensorOptions(smith::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestBinaryCrossEntropyBackward) {
  int batch = 6;
  int classes = 2;
  // TODO(asuhan): Fix the smith::kDouble case.
  for (auto dtype : {smith::kFloat}) {
    for (bool def_weight : {false, true}) {
      smith::Tensor input = smith::rand(
          {batch, classes}, smith::TensorOptions(dtype).requires_grad(true));
      smith::Tensor target =
          smith::rand({batch, classes}, smith::TensorOptions(dtype));
      smith::Tensor weight;
      if (def_weight) {
        weight = smith::rand({batch, classes}, smith::TensorOptions(dtype));
      }
      for (smith::Reduction::Reduction reduction :
           {smith::Reduction::Mean,
            smith::Reduction::Sum,
            smith::Reduction::None}) {
        auto testfn =
            [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
          return smith::binary_cross_entropy(
              /*self=*/inputs[0],
              /*target=*/inputs[1],
              /*weight=*/inputs[2],
              /*reduction=*/reduction);
        };
        ForEachDevice([&](const smith::Device& device) {
          TestBackward(
              {input, target, weight},
              device,
              testfn,
              /*rtol=*/1e-4,
              /*atol=*/1e-7);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLossBackward) {
  int batch = 6;
  int classes = 2;
  // TODO(asuhan): Fix the smith::kDouble case.
  for (auto dtype : {smith::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        smith::Tensor input = smith::rand(
            {batch, classes},
            smith::TensorOptions(dtype)
                .device(DefaultDevice())
                .requires_grad(true));
        smith::Tensor target = smith::randint(
            std::min(ignore_index, 0),
            classes,
            {batch},
            smith::TensorOptions(smith::kLong).device(DefaultDevice()));
        smith::Tensor weight;
        if (def_weight) {
          weight = smith::rand(
              {classes}, smith::TensorOptions(dtype).device(DefaultDevice()));
        }
        for (smith::Reduction::Reduction reduction :
             {smith::Reduction::Mean,
              smith::Reduction::Sum,
              smith::Reduction::None}) {
          auto testfn =
              [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
            return smith::nll_loss(
                /*self=*/inputs[0],
                /*target=*/inputs[1],
                /*weight=*/inputs[2],
                /*reduction=*/reduction,
                /*ignore_index=*/ignore_index);
          };
          ForEachDevice([&](const smith::Device& device) {
            TestBackward(
                {input, target, weight},
                device,
                testfn,
                /*rtol=*/1e-5,
                /*atol=*/1e-8);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLoss2dBackward) {
  int batch = 6;
  int classes = 2;
  int height = 3;
  int width = 3;
  // TODO(asuhan): Fix the smith::kDouble case.
  for (auto dtype : {smith::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        smith::Tensor input = smith::rand(
            {batch, classes, height, width},
            smith::TensorOptions(dtype)
                .device(DefaultDevice())
                .requires_grad(true));
        smith::Tensor target = smith::randint(
            std::min(ignore_index, 0),
            classes,
            {batch, height, width},
            smith::TensorOptions(smith::kLong).device(DefaultDevice()));
        smith::Tensor weight;
        if (def_weight) {
          weight = smith::rand(
              {classes}, smith::TensorOptions(dtype).device(DefaultDevice()));
        }
        for (smith::Reduction::Reduction reduction :
             {smith::Reduction::Mean,
              smith::Reduction::Sum,
              smith::Reduction::None}) {
          auto testfn =
              [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
            return smith::nll_loss2d(
                /*self=*/inputs[0],
                /*target=*/inputs[1],
                /*weight=*/inputs[2],
                /*reduction=*/reduction,
                /*ignore_index=*/ignore_index);
          };
          ForEachDevice([&](const smith::Device& device) {
            TestBackward(
                {input, target, weight},
                device,
                testfn,
                /*rtol=*/1e-5,
                /*atol=*/1e-8);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSmoothL1LossBackward) {
  smith::Tensor input = smith::randn(
      {2, 4},
      smith::TensorOptions(smith::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  smith::Tensor target = smith::randn(
      {2, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::None,
        smith::Reduction::Mean,
        smith::Reduction::Sum}) {
    for (double beta : {0.25, 1.}) {
      auto testfn =
          [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
        return smith::smooth_l1_loss(
            /*input=*/inputs[0],
            /*target=*/inputs[1],
            /*reduction=*/reduction,
            /*beta=*/beta);
      };
      ForEachDevice([&](const smith::Device& device) {
        TestBackward(
            {input, target},
            device,
            testfn,
            /*rtol=*/1e-5,
            /*atol=*/1e-8);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestViewBackward) {
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return inputs[0].view({-1, 320});
  };
  ForEachDevice([&](const smith::Device& device) {
    TestBackward(
        {smith::rand(
            {32, 20, 4, 4},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestBatchNorm2DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::batch_norm(
        /*input=*/inputs[0],
        /*weight=*/inputs[1],
        /*bias=*/inputs[2],
        /*running_mean=*/inputs[3],
        /*running_var=*/inputs[4],
        /*training=*/true,
        /*momentum=*/momentum,
        /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  smith::Tensor undef;
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor input = smith::rand(
          {2, num_features, 4, 4},
          smith::TensorOptions(smith::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      smith::Tensor weight = undef_weight_bias
          ? undef
          : smith::rand(
                {num_features},
                smith::TensorOptions(smith::kFloat)
                    .device(DefaultDevice())
                    .requires_grad(true));
      smith::Tensor bias = undef_weight_bias
          ? undef
          : smith::rand(
                {num_features},
                smith::TensorOptions(smith::kFloat)
                    .device(DefaultDevice())
                    .requires_grad(true));
      smith::Tensor running_mean = smith::zeros(
          {num_features},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor running_var = smith::ones(
          {num_features},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      TestBackward(
          {input, weight, bias, running_mean, running_var},
          device,
          testfn,
          /*rtol=*/1e-3,
          /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestBatchNorm3DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
    return smith::batch_norm(
        /*input=*/inputs[0],
        /*weight=*/inputs[1],
        /*bias=*/inputs[2],
        /*running_mean=*/inputs[3],
        /*running_var=*/inputs[4],
        /*training=*/true,
        /*momentum=*/momentum,
        /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  smith::Tensor undef;
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const smith::Device& device) {
      smith::Tensor input = smith::rand(
          {2, num_features, 4, 4, 2},
          smith::TensorOptions(smith::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      smith::Tensor weight = undef_weight_bias
          ? undef
          : smith::rand(
                {num_features},
                smith::TensorOptions(smith::kFloat)
                    .device(DefaultDevice())
                    .requires_grad(true));
      smith::Tensor bias = undef_weight_bias
          ? undef
          : smith::rand(
                {num_features},
                smith::TensorOptions(smith::kFloat)
                    .device(DefaultDevice())
                    .requires_grad(true));
      smith::Tensor running_mean = smith::zeros(
          {num_features},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      smith::Tensor running_var = smith::ones(
          {num_features},
          smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
      TestBackward(
          {input, weight, bias, running_mean, running_var},
          device,
          testfn,
          /*rtol=*/1e-3,
          /*atol=*/1e-3);
    });
  }
}

TEST_F(LazyOpsTest, TestBCEWithLogitsBackward) {
  int batch = 10;
  int classes = 5;
  smith::Tensor undef;
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::None,
        smith::Reduction::Mean,
        smith::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::binary_cross_entropy_with_logits(
          /*input=*/inputs[0],
          /*target=*/inputs[1],
          /*weight=*/inputs[2],
          /*pos_weight=*/inputs[3],
          /*reduction=*/reduction);
    };
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        smith::Tensor input = smith::rand(
            {batch, classes},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true));
        smith::Tensor target = smith::rand(
            {batch, classes},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true));
        smith::Tensor weight = undef_weight
            ? undef
            : smith::rand(
                  {classes},
                  smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
        smith::Tensor pos_weight = undef_pos_weight
            ? undef
            : smith::rand(
                  {classes},
                  smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
        ForEachDevice([&](const smith::Device& device) {
          TestBackward(
              {input, target, weight, pos_weight},
              device,
              testfn,
              /*rtol=*/1e-3,
              /*atol=*/1e-5);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestKlDivBackward) {
  smith::Tensor input = smith::rand(
      {4, 3},
      smith::TensorOptions(smith::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  smith::Tensor target = smith::rand(
      {4, 3},
      smith::TensorOptions(smith::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  for (smith::Reduction::Reduction reduction :
       {smith::Reduction::Mean,
        smith::Reduction::Sum,
        smith::Reduction::None}) {
    auto testfn =
        [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
      return smith::kl_div(/*self=*/inputs[0], /*target=*/inputs[1], reduction);
    };
    ForEachDevice([&](const smith::Device& device) {
      TestBackward(
          {input, target},
          device,
          testfn,
          /*rtol=*/1e-4,
          /*atol=*/1e-5);
    });
  }
}

TEST_F(LazyOpsTest, TestEmbeddingBackward) {
  int num_weights = 32;
  for (int padding_idx = -1; padding_idx < num_weights; ++padding_idx) {
    for (bool scale_grad_by_freq : {false, true}) {
      auto testfn =
          [&](const std::vector<smith::Tensor>& inputs) -> smith::Tensor {
        return smith::embedding(
            inputs[0],
            inputs[1],
            /*padding_idx=*/padding_idx,
            /*scale_grad_by_freq=*/scale_grad_by_freq,
            /*sparse=*/false);
      };
      ForEachDevice([&](const smith::Device& device) {
        smith::Tensor weight = smith::rand(
            {num_weights, 7},
            smith::TensorOptions(smith::kFloat)
                .device(DefaultDevice())
                .requires_grad(true));
        smith::Tensor indices = smith::randint(
            num_weights,
            {3, 9, 4},
            smith::TensorOptions(smith::kLong).device(DefaultDevice()));
        TestBackward(
            {weight, indices},
            device,
            testfn,
            /*rtol=*/1e-5,
            /*atol=*/1e-8);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestAmpForeachNonFiniteCheckAndUnscale) {
  if (IsCuda()) {
    // TODO(whc) debug failure on cuda
    GTEST_SKIP();
  }

  smith::Tensor grads0 = smith::tensor(
      {1, 2, 3, 4},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor grads1 = smith::tensor(
      {1.0, 2.0, std::nan("1"), 4.0},
      smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor inv_scale = smith::scalar_tensor(
      0.2, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor found_inf = smith::scalar_tensor(
      0, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor grads_output0 = grads0 * inv_scale;
  smith::Tensor found_inf_output0 = smith::scalar_tensor(
      0, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor found_inf_output1 = smith::scalar_tensor(
      1, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const smith::Device& device) {
    if (grads0.device() == at::kCPU) {
      GTEST_SKIP();
    }
    smith::Tensor lazy_grads0 = CopyToDevice(grads0, device);
    smith::Tensor lazy_inv_scale = CopyToDevice(inv_scale, device);
    smith::Tensor lazy_found_inf = CopyToDevice(found_inf, device);
    smith::_amp_foreach_non_finite_check_and_unscale_(
        lazy_grads0, lazy_found_inf, lazy_inv_scale);
    AllClose(grads_output0, lazy_grads0, /*rtol=*/1e-2, /*atol=*/1e-4);
    AllEqual(found_inf_output0, lazy_found_inf);

    smith::Tensor lazy_grads1 = CopyToDevice(grads1, device);
    smith::_amp_foreach_non_finite_check_and_unscale_(
        lazy_grads1, lazy_found_inf, lazy_inv_scale);
    AllEqual(found_inf_output1, lazy_found_inf);
  });
}

TEST_F(LazyOpsTest, TestAmpUpdateScale) {
  smith::Tensor growth_tracker = smith::scalar_tensor(
      0, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Tensor current_scale = smith::scalar_tensor(
      4, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor found_inf = smith::scalar_tensor(
      1, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor not_found_inf = smith::scalar_tensor(
      0, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  float scale_growth_factor = 2.0;
  float scale_backoff_factor = 0.5;
  int growth_interval = 3;

  smith::Tensor growth_tracker_result0 = smith::scalar_tensor(
      1, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Tensor current_scale_result0 = smith::scalar_tensor(
      4, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor growth_tracker_result1 = smith::scalar_tensor(
      2, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Tensor current_scale_result1 = smith::scalar_tensor(
      4, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor growth_tracker_result2 = smith::scalar_tensor(
      0, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Tensor current_scale_result2 = smith::scalar_tensor(
      8, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor growth_tracker_result3 = smith::scalar_tensor(
      0, smith::TensorOptions(smith::kInt32).device(DefaultDevice()));
  smith::Tensor current_scale_result3 = smith::scalar_tensor(
      4, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));

  ForEachDevice([&](const smith::Device& device) {
    if (growth_tracker.device() == at::kCPU) {
      GTEST_SKIP();
    }
    smith::Tensor lazy_growth_tracker = CopyToDevice(growth_tracker, device);
    smith::Tensor lazy_current_scale = CopyToDevice(current_scale, device);
    smith::Tensor lazy_found_inf = CopyToDevice(found_inf, device);
    smith::Tensor lazy_not_found_inf = CopyToDevice(not_found_inf, device);

    smith::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_not_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    AllClose(
        current_scale_result0,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    AllEqual(growth_tracker_result0, lazy_growth_tracker);

    smith::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_not_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    AllClose(
        current_scale_result1,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    AllEqual(growth_tracker_result1, lazy_growth_tracker);

    // smith::_amp_update_scale_ returns the reference of current_scale
    lazy_current_scale = smith::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_not_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    AllClose(
        current_scale_result2,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    AllEqual(growth_tracker_result2, lazy_growth_tracker);

    lazy_current_scale = smith::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    AllClose(
        current_scale_result3,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    AllEqual(growth_tracker_result3, lazy_growth_tracker);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::_amp_update_scale_", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestEarlySyncLiveTensors) {
  smith::Tensor scalar_tensor = smith::scalar_tensor(
      1., smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar scalar1 = scalar_tensor.item();
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_scalar_tensor = CopyToDevice(scalar_tensor, device);
    smith::Scalar scalar2 = lazy_scalar_tensor.item();
    ASSERT_EQ(scalar1.to<float>(), scalar2.to<float>());
  });
  if (DebugUtil::ExperimentEnabled("early_sync")) {
    ExpectCounterChanged("EarlySyncLiveTensorsCount", GetIgnoredCounters());
  } else {
    ExpectCounterNotChanged("EarlySyncLiveTensorsCount", GetIgnoredCounters());
  }
  ExpectCounterChanged("aten::_local_scalar_dense", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerp) {
  smith::Tensor start = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor end = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor res = smith::lerp(start, end, weight);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_start = CopyToDevice(start, device);
    smith::Tensor lazy_end = CopyToDevice(end, device);
    smith::Tensor lazy_weight = CopyToDevice(weight, device);
    smith::Tensor lazy_res = smith::lerp(lazy_start, lazy_end, lazy_weight);
    AllClose(res, lazy_res);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpScalar) {
  smith::Tensor start = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor end = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar weight = smith::Scalar(3.0);
  smith::Tensor res = smith::lerp(start, end, weight);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_start = CopyToDevice(start, device);
    smith::Tensor lazy_end = CopyToDevice(end, device);
    smith::Tensor lazy_res = smith::lerp(lazy_start, lazy_end, weight);
    AllClose(res, lazy_res);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpInplace) {
  smith::Tensor input = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor end = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor input_copy = input.clone();
  input.lerp_(end, weight);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input_copy, device);
    smith::Tensor lazy_end = CopyToDevice(end, device);
    smith::Tensor lazy_weight = CopyToDevice(weight, device);
    lazy_input.lerp_(lazy_end, lazy_weight);
    AllClose(lazy_input, input);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpScalarInplace) {
  smith::Tensor input = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor end = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar weight = smith::Scalar(3.0);
  smith::Tensor input_copy = input.clone();
  input.lerp_(end, weight);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_input = CopyToDevice(input_copy, device);
    smith::Tensor lazy_end = CopyToDevice(end, device);
    lazy_input.lerp_(lazy_end, weight);
    AllClose(lazy_input, input);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpOut) {
  smith::Tensor start = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor end = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor weight = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor res = smith::empty(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  ;
  smith::lerp_out(res, start, end, weight);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_start = CopyToDevice(start, device);
    smith::Tensor lazy_end = CopyToDevice(end, device);
    smith::Tensor lazy_weight = CopyToDevice(weight, device);
    smith::Tensor lazy_res = smith::empty({3, 4}, lazy_start.options());
    smith::lerp_out(lazy_res, lazy_start, lazy_end, lazy_weight);
    AllClose(res, lazy_res);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpScalarOut) {
  smith::Tensor start = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Tensor end = smith::rand(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::Scalar weight = smith::Scalar(3.0);
  smith::Tensor res = smith::empty(
      {3, 4}, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  smith::lerp_out(res, start, end, weight);
  ForEachDevice([&](const smith::Device& device) {
    smith::Tensor lazy_start = CopyToDevice(start, device);
    smith::Tensor lazy_end = CopyToDevice(end, device);
    smith::Tensor lazy_res = smith::empty({3, 4}, lazy_start.options());
    smith::lerp_out(lazy_res, lazy_start, lazy_end, weight);
    AllClose(res, lazy_res);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, IsAliasOf) {
  auto a = smith::empty(
      4, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));
  auto b = smith::empty(
      4, smith::TensorOptions(smith::kFloat).device(DefaultDevice()));

  ForEachDevice([&](const smith::Device& device) {
    auto lazy_a = CopyToDevice(a, device);
    auto lazy_b = CopyToDevice(b, device);
    EXPECT_EQ(!a.is_alias_of(b), !lazy_a.is_alias_of(lazy_b));

    auto c = a.view({2, 2});
    auto lazy_c = lazy_a.view({2, 2});
    EXPECT_EQ(a.is_alias_of(c), lazy_a.is_alias_of(lazy_c));

    auto d = c.view({1, 4});
    auto lazy_d = lazy_c.view({1, 4});
    EXPECT_EQ(d.is_alias_of(c), lazy_d.is_alias_of(lazy_c));
    EXPECT_EQ(d.is_alias_of(a), lazy_d.is_alias_of(lazy_a));
  });
}

#endif // FBCODE_CAFFE2

} // namespace lazy
} // namespace smith
