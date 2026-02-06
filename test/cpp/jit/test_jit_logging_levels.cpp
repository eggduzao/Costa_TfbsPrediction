#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>

#include <smith/csrc/jit/jit_log.h>
#include <sstream>

namespace smith {
namespace jit {

TEST(JitLoggingTest, CheckSetLoggingLevel) {
  ::smith::jit::set_jit_logging_levels("file_to_test");
  ASSERT_TRUE(::smith::jit::is_enabled(
      "file_to_test.cpp", JitLoggingLevels::GRAPH_DUMP));
}

TEST(JitLoggingTest, CheckSetMultipleLogLevels) {
  ::smith::jit::set_jit_logging_levels("f1:>f2:>>f3");
  ASSERT_TRUE(::smith::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));
  ASSERT_TRUE(
      ::smith::jit::is_enabled("f2.cpp", JitLoggingLevels::GRAPH_UPDATE));
  ASSERT_TRUE(
      ::smith::jit::is_enabled("f3.cpp", JitLoggingLevels::GRAPH_DEBUG));
}

TEST(JitLoggingTest, CheckLoggingLevelAfterUnset) {
  ::smith::jit::set_jit_logging_levels("f1");
  ASSERT_EQ("f1", ::smith::jit::get_jit_logging_levels());
  ::smith::jit::set_jit_logging_levels("invalid");
  ASSERT_FALSE(
      ::smith::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));
}

TEST(JitLoggingTest, CheckAfterChangingLevel) {
  ::smith::jit::set_jit_logging_levels("f1");
  ::smith::jit::set_jit_logging_levels(">f1");
  ASSERT_TRUE(
      ::smith::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_UPDATE));
}

TEST(JitLoggingTest, CheckOutputStreamSetting) {
  ::smith::jit::set_jit_logging_levels("test_jit_logging_levels");
  std::ostringstream test_stream;
  ::smith::jit::set_jit_logging_output_stream(test_stream);
  /* Using JIT_LOG checks if this file has logging enabled with
    is_enabled(__FILE__, level) making the test fail. since we are only testing
    the OutputStreamSetting we can forcefully output to it directly.
  */
  ::smith::jit::get_jit_logging_output_stream() << ::smith::jit::jit_log_prefix(
      ::smith::jit::JitLoggingLevels::GRAPH_DUMP,
      __FILE__,
      __LINE__,
      ::c10::str("Message"));
  ASSERT_TRUE(test_stream.str().size() > 0);
}

} // namespace jit
} // namespace smith
