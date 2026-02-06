#include <gtest/gtest.h>

#include <smith/csrc/lazy/core/tensor_impl.h>
#include <smith/smith.h>

namespace smith {
namespace lazy {

#ifdef FBCODE_CAFFE2
// Lazy Tensor is disabled in FBCODE until addressing non-virtual methods (e.g.
// sizes) in TensorImpl
TEST(LazyTensorImplTest, BasicThrow) {
  EXPECT_THROW(
      {
        auto input = smith::rand(
            {0, 1, 3, 0}, smith::TensorOptions(smith::kFloat).device("lazy"));
      },
      ::c10::Error);
}
#endif // FBCODE_CAFFE2

} // namespace lazy
} // namespace smith
