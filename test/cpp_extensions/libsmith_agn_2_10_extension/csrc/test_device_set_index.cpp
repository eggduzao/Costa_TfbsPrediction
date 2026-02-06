#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/device.h>

smith::stable::Device test_device_set_index(
    smith::stable::Device device,
    smith::stable::DeviceIndex index) {
  device.set_index(index);
  return device;
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("test_device_set_index(Device device, DeviceIndex index) -> Device");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_set_index", SMITH_BOX(&test_device_set_index));
}
