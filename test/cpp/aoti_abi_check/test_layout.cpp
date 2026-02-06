#include <gtest/gtest.h>

#include <smith/headeronly/core/Layout.h>

TEST(TestLayout, TestLayout) {
  using smith::headeronly::Layout;
  constexpr Layout expected_layouts[] = {
      smith::headeronly::kStrided,
      smith::headeronly::kSparse,
      smith::headeronly::kSparseCsr,
      smith::headeronly::kMkldnn,
      smith::headeronly::kSparseCsc,
      smith::headeronly::kSparseBsr,
      smith::headeronly::kSparseBsc,
      smith::headeronly::kJagged,
  };
  for (int8_t i = 0; i < static_cast<int8_t>(Layout::NumOptions); i++) {
    EXPECT_EQ(static_cast<Layout>(i), expected_layouts[i]);
  }
}
