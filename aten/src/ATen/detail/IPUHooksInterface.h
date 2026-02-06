#pragma once

#include <ATen/detail/AcceleratorHooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace at {

struct SMITH_API IPUHooksInterface : AcceleratorHooksInterface {
  ~IPUHooksInterface() override = default;

  void init() const override {
    SMITH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
  }

  bool hasPrimaryContext(DeviceIndex /*device_index*/) const override {
    SMITH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
    return false;
  }

  const Generator& getDefaultGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    SMITH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
  }

  Generator getNewGenerator(
      DeviceIndex /*device_index*/ = -1) const override {
    SMITH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
  }
};

struct SMITH_API IPUHooksArgs {};

SMITH_DECLARE_REGISTRY(IPUHooksRegistry, IPUHooksInterface, IPUHooksArgs);
#define REGISTER_IPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(IPUHooksRegistry, clsname, clsname)

namespace detail {
SMITH_API const IPUHooksInterface& getIPUHooks();
} // namespace detail
} // namespace at
