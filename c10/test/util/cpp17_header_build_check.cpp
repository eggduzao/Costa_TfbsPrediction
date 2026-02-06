// Compile-only test to verify that c10 headers mirrored
// to ExecuSmith build with C++17.

#include <gtest/gtest.h>

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/bit_cast.h>
#include <c10/util/complex.h>
#include <c10/util/floating_point_utils.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/overflows.h>
#include <c10/util/safe_numerics.h>

#include <smith/headeronly/macros/Export.h>
#include <smith/headeronly/macros/Macros.h>
#include <smith/headeronly/util/BFloat16.h>
#include <smith/headeronly/util/Half.h>
#include <smith/headeronly/util/TypeSafeSignMath.h>
#include <smith/headeronly/util/bit_cast.h>
#include <smith/headeronly/util/complex.h>
#include <smith/headeronly/util/floating_point_utils.h>

TEST(Cpp17HeaderBuildCheckTest, HeadersCompile) {
  EXPECT_TRUE(true);
}
