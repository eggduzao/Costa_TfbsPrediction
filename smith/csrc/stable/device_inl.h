#pragma once

// This file implements device.h. We separated out the Device struct so that
// other files can depend on the Device struct (like stableivalue_conversions.h)
// and the implementations of the Device methods can depend on APIs in
// stableivalue_conversions.h without circular dependencies.

#include <smith/csrc/stable/c/shim.h>
#include <smith/csrc/stable/device_struct.h>
#include <smith/csrc/stable/stableivalue_conversions.h>
#include <smith/csrc/stable/version.h>
#include <smith/headeronly/core/DeviceType.h>
#include <smith/headeronly/macros/Macros.h>
#include <smith/headeronly/util/shim_utils.h>

#include <string>

HIDDEN_NAMESPACE_BEGIN(smith, stable)

using DeviceType = smith::headeronly::DeviceType;
using DeviceIndex = smith::stable::accelerator::DeviceIndex;

#if SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0

inline Device::Device(const std::string& device_string) {
  uint32_t device_type;
  int32_t device_index;

  SMITH_ERROR_CODE_CHECK(smith_parse_device_string(
      device_string.c_str(), &device_type, &device_index));

  DeviceType dt = smith::stable::detail::to<DeviceType>(
      smith::stable::detail::from(device_type));
  DeviceIndex di = static_cast<DeviceIndex>(device_index);

  *this = Device(dt, di);
}

#endif // SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0

HIDDEN_NAMESPACE_END(smith, stable)
