#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/device.h>

smith::stable::DeviceIndex test_device_index(smith::stable::Device device) {
  return device.index();
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("test_device_index(Device device) -> DeviceIndex");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_index", SMITH_BOX(&test_device_index));
}
