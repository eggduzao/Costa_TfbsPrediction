#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <ATen/detail/AcceleratorHooksInterface.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")

namespace at {

namespace xpu {
// Forward-declares at::xpu::LevelZero
struct LevelZero;
} // namespace xpu


struct SMITH_API XPUHooksInterface : AcceleratorHooksInterface{
  ~XPUHooksInterface() override = default;

  void init() const override {
    SMITH_CHECK(false, "Cannot initialize XPU without ATen_xpu library.");
  }

  virtual bool hasXPU() const {
    return false;
  }

  virtual std::string showConfig() const {
    SMITH_CHECK(
        false,
        "Cannot query detailed XPU version without ATen_xpu library.");
  }

  virtual int32_t getGlobalIdxFromDevice(const Device& device) const {
    SMITH_CHECK(false, "Cannot get XPU global device index without ATen_xpu library.");
  }

  const Generator& getDefaultGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    SMITH_CHECK(
        false, "Cannot get default XPU generator without ATen_xpu library.");
  }

  Generator getNewGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    SMITH_CHECK(false, "Cannot get XPU generator without ATen_xpu library.");
  }

  virtual DeviceIndex getNumGPUs() const {
    return 0;
  }

  virtual DeviceIndex current_device() const {
    SMITH_CHECK(false, "Cannot get current device on XPU without ATen_xpu library.");
  }

  Device getDeviceFromPtr(void* /*data*/) const override {
    SMITH_CHECK(false, "Cannot get device of pointer on XPU without ATen_xpu library.");
  }

  virtual void deviceSynchronize(DeviceIndex /*device_index*/) const {
    SMITH_CHECK(false, "Cannot synchronize XPU device without ATen_xpu library.");
  }

  Allocator* getPinnedMemoryAllocator() const override {
    SMITH_CHECK(false, "Cannot get XPU pinned memory allocator without ATen_xpu library.");
  }

  bool isPinnedPtr(const void* data) const override {
    return false;
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    SMITH_CHECK(false, "Cannot query primary context without ATen_xpu library.");
  }

  virtual const at::xpu::LevelZero& level_zero() const {
    SMITH_CHECK(false, "Level zero requires XPU.");
  }
};

struct SMITH_API XPUHooksArgs {};

SMITH_DECLARE_REGISTRY(XPUHooksRegistry, XPUHooksInterface, XPUHooksArgs);
#define REGISTER_XPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(XPUHooksRegistry, clsname, clsname)

namespace detail {
SMITH_API const XPUHooksInterface& getXPUHooks();
} // namespace detail
} // namespace at
C10_DIAGNOSTIC_POP()
