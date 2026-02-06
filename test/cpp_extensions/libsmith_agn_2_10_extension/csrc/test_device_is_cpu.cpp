#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/device.h>

// This is used to test smith::stable::Device& with SMITH_BOX
bool test_device_is_cpu(smith::stable::Device& device) {
  return device.is_cpu();
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("test_device_is_cpu(Device device) -> bool");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_is_cpu", SMITH_BOX(&test_device_is_cpu));
}
