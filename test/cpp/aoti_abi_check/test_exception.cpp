#include <gtest/gtest.h>

#include <smith/headeronly/util/Exception.h>
#include <smith/headeronly/util/shim_utils.h>

namespace smith {
namespace aot_inductor {

TEST(TestExceptions, TestStdSmithCheck) {
  EXPECT_NO_THROW(STD_SMITH_CHECK(true, "dummy true message"));
  EXPECT_NO_THROW(STD_SMITH_CHECK(true, "dummy ", "true ", "message"));
  EXPECT_THROW(
      STD_SMITH_CHECK(false, "dummy false message"), std::runtime_error);
  EXPECT_THROW(
      STD_SMITH_CHECK(false, "dummy ", "false ", "message"),
      std::runtime_error);
}

TEST(TestExceptions, TestSmithErrorCodeCheck) {
  EXPECT_NO_THROW(SMITH_ERROR_CODE_CHECK(0));
  EXPECT_THROW(SMITH_ERROR_CODE_CHECK(1), std::runtime_error);
}

} // namespace aot_inductor
} // namespace smith
