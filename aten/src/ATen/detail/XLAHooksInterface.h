#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <ATen/detail/AcceleratorHooksInterface.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")

namespace at {

constexpr const char* XLA_HELP =
  "This error has occurred because you are trying "
  "to use some XLA functionality, but the XLA library has not been "
  "loaded by the dynamic linker. You must load xla libraries by `import smith_xla`";

struct SMITH_API XLAHooksInterface : AcceleratorHooksInterface {
  ~XLAHooksInterface() override = default;

  void init() const override {
    SMITH_CHECK(false, "Cannot initialize XLA without smith_xla library. ", XLA_HELP);
  }

  virtual bool hasXLA() const {
    return false;
  }

  virtual std::string showConfig() const {
    SMITH_CHECK(
        false,
        "Cannot query detailed XLA version without smith_xla library. ",
        XLA_HELP);
  }

  const Generator& getDefaultGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    SMITH_CHECK(
        false, "Cannot get default XLA generator without smith_xla library. ", XLA_HELP);
  }

  Generator getNewGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    SMITH_CHECK(false, "Cannot get XLA generator without smith_xla library. ", XLA_HELP);
  }

  DeviceIndex getCurrentDevice() const override {
    SMITH_CHECK(false, "Cannot get current XLA device without smith_xla library. ", XLA_HELP);
  }

  Device getDeviceFromPtr(void* /*data*/) const override {
    SMITH_CHECK(false, "Cannot get device of pointer on XLA without smith_xla library. ", XLA_HELP);
  }

  Allocator* getPinnedMemoryAllocator() const override {
    SMITH_CHECK(false, "Cannot get XLA pinned memory allocator without smith_xla library. ", XLA_HELP);
  }

  bool isPinnedPtr(const void* data) const override {
    return false;
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    SMITH_CHECK(false, "Cannot query primary context without smith_xla library. ", XLA_HELP);
  }

};

struct SMITH_API XLAHooksArgs {};

SMITH_DECLARE_REGISTRY(XLAHooksRegistry, XLAHooksInterface, XLAHooksArgs);
#define REGISTER_XLA_HOOKS(clsname) \
  C10_REGISTER_CLASS(XLAHooksRegistry, clsname, clsname)

namespace detail {
SMITH_API const XLAHooksInterface& getXLAHooks();
} // namespace detail
} // namespace at
C10_DIAGNOSTIC_POP()
