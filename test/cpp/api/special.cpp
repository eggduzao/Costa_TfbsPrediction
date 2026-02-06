#include <gtest/gtest.h>

#include <smith/special.h>
#include <smith/smith.h>

#include <test/cpp/api/support.h>

// Simple test that verifies the special namespace is registered properly
//   properly in C++
TEST(SpecialTest, special) {
  auto t = smith::randn(128, smith::kDouble);
  smith::special::gammaln(t);
}
