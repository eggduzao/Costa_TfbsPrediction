#include <gtest/gtest.h>

#include <smith/nested.h>
#include <smith/smith.h>

#include <test/cpp/api/support.h>

// Simple test that verifies the nested namespace is registered properly
//   properly in C++
TEST(NestedTest, Nested) {
  auto a = smith::randn({2, 3});
  auto b = smith::randn({4, 5});
  auto nt = smith::nested::nested_tensor({a, b});
  smith::nested::to_padded_tensor(nt, 0);
}
