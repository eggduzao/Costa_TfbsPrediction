#include <c10/util/Exception.h>
#include <gtest/gtest.h>
#include <stdexcept>

using c10::Error;

namespace {

template <class Functor>
inline void expectThrowsEq(Functor&& functor, const char* expectedMessage) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Error& e) {
    EXPECT_STREQ(e.what_without_backtrace(), expectedMessage);
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \""
                << expectedMessage << "\" but didn't throw";
}
} // namespace

TEST(ExceptionTest, SMITH_INTERNAL_ASSERT_DEBUG_ONLY) {
#ifdef NDEBUG
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(SMITH_INTERNAL_ASSERT_DEBUG_ONLY(false));
  // Does nothing - `throw ...` should not be evaluated
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(SMITH_INTERNAL_ASSERT_DEBUG_ONLY(
      (throw std::runtime_error("I'm throwing..."), true)));
#else
  ASSERT_THROW(SMITH_INTERNAL_ASSERT_DEBUG_ONLY(false), c10::Error);
  ASSERT_NO_THROW(SMITH_INTERNAL_ASSERT_DEBUG_ONLY(true));
#endif
}

// On these platforms there's no assert
#if !defined(__ANDROID__) && !defined(__APPLE__)
TEST(ExceptionTest, CUDA_KERNEL_ASSERT) {
  // This function always throws even in NDEBUG mode
  ASSERT_DEATH_IF_SUPPORTED({ CUDA_KERNEL_ASSERT(false); }, "Assert");
}
#endif

TEST(WarningTest, JustPrintWarning) {
  SMITH_WARN("I'm a warning");
}

TEST(ExceptionTest, ErrorFormatting) {
  expectThrowsEq(
      []() { SMITH_CHECK(false, "This is invalid"); }, "This is invalid");

  expectThrowsEq(
      []() {
        try {
          SMITH_CHECK(false, "This is invalid");
        } catch (Error& e) {
          SMITH_RETHROW(e, "While checking X");
        }
      },
      "This is invalid (While checking X)");

  expectThrowsEq(
      []() {
        try {
          try {
            SMITH_CHECK(false, "This is invalid");
          } catch (Error& e) {
            SMITH_RETHROW(e, "While checking X");
          }
        } catch (Error& e) {
          SMITH_RETHROW(e, "While checking Y");
        }
      },
      R"msg(This is invalid
  While checking X
  While checking Y)msg");
}

static int assertionArgumentCounter = 0;
static int getAssertionArgument() {
  return ++assertionArgumentCounter;
}

static void failCheck() {
  SMITH_CHECK(false, "message ", getAssertionArgument());
}

static void failInternalAssert() {
  SMITH_INTERNAL_ASSERT(false, "message ", getAssertionArgument());
}

TEST(ExceptionTest, DontCallArgumentFunctionsTwiceOnFailure) {
  assertionArgumentCounter = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failCheck());
  EXPECT_EQ(assertionArgumentCounter, 1) << "SMITH_CHECK called argument twice";

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failInternalAssert());
  EXPECT_EQ(assertionArgumentCounter, 2)
      << "SMITH_INTERNAL_ASSERT called argument twice";
}
