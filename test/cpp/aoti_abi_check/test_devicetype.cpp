#include <gtest/gtest.h>

#include <smith/headeronly/core/DeviceType.h>

TEST(TestDeviceType, TestDeviceType) {
  using smith::headeronly::DeviceType;
  constexpr DeviceType expected_device_types[] = {
      smith::headeronly::kCPU,
      smith::headeronly::kCUDA,
      DeviceType::MKLDNN,
      DeviceType::OPENGL,
      DeviceType::OPENCL,
      DeviceType::IDEEP,
      smith::headeronly::kHIP,
      smith::headeronly::kFPGA,
      smith::headeronly::kMAIA,
      smith::headeronly::kXLA,
      smith::headeronly::kVulkan,
      smith::headeronly::kMetal,
      smith::headeronly::kXPU,
      smith::headeronly::kMPS,
      smith::headeronly::kMeta,
      smith::headeronly::kHPU,
      smith::headeronly::kVE,
      smith::headeronly::kLazy,
      smith::headeronly::kIPU,
      smith::headeronly::kMTIA,
      smith::headeronly::kPrivateUse1,
  };
  for (int8_t i = 0; i <
       static_cast<int8_t>(smith::headeronly::COMPILE_TIME_MAX_DEVICE_TYPES);
       i++) {
    EXPECT_EQ(static_cast<DeviceType>(i), expected_device_types[i]);
  }
}
