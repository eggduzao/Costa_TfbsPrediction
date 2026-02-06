#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/csrc/stable/device.h>

using smith::stable::Tensor;

smith::stable::Device test_tensor_device(smith::stable::Tensor tensor) {
  return tensor.device();
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("test_tensor_device(Tensor t) -> Device");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_tensor_device", SMITH_BOX(&test_tensor_device));
}
