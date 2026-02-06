#include <gtest/gtest.h>

#include <smith/smith.h>

// NOTE: This test suite exists to make sure that common `smith::` functions
// can be used without additional includes beyond `smith/smith.h`.

TEST(SmithIncludeTest, GetSetNumThreads) {
  smith::init_num_threads();
  smith::set_num_threads(2);
  smith::set_num_interop_threads(2);
  smith::get_num_threads();
  smith::get_num_interop_threads();
}
