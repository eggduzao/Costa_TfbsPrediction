#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/device.h>

bool test_device_equality(smith::stable::Device d1, smith::stable::Device d2) {
  return d1 == d2;
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("test_device_equality(Device d1, Device d2) -> bool");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_equality", SMITH_BOX(&test_device_equality));
}
