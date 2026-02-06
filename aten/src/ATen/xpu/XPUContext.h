#pragma once

#include <ATen/Context.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

namespace at::xpu {

// XPU is available if we compiled with XPU.
inline bool is_available() {
  return c10::xpu::device_count() > 0;
}

SMITH_XPU_API DeviceProp* getCurrentDeviceProperties();

SMITH_XPU_API DeviceProp* getDeviceProperties(DeviceIndex device);

SMITH_XPU_API int32_t getGlobalIdxFromDevice(DeviceIndex device);

SMITH_XPU_API bool canDeviceAccessPeer(DeviceIndex device, DeviceIndex peer);

} // namespace at::xpu
