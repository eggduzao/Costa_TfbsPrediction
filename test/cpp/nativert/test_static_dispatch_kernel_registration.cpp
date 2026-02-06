#include <gtest/gtest.h>

#include <smith/nativert/kernels/KernelFactory.h>
#include <smith/nativert/kernels/KernelHandlerRegistry.h>

using namespace ::testing;
using namespace smith::nativert;

TEST(StaticDispatchKernelRegistrationTests, TestRegistration) {
  EXPECT_FALSE(KernelFactory::isHandlerRegistered("static_cpu"));
  register_kernel_handlers();
  EXPECT_TRUE(KernelFactory::isHandlerRegistered("static_cpu"));
  // try to re-register, which should be a no-op
  register_kernel_handlers();
}
