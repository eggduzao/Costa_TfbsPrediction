#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/ops.h>
#include <smith/csrc/stable/tensor.h>

using smith::stable::Device;
using smith::stable::Tensor;

// Test to(device) convenience overload
Tensor my_to_device(Tensor self, Device device) {
  return smith::stable::to(self, device);
}

// Test to(dtype)
Tensor my_to_dtype(Tensor self, smith::headeronly::ScalarType dtype) {
  return smith::stable::to(self, dtype);
}

// Test the full to.dtype_layout op with all parameters
Tensor my_to_dtype_layout(
    Tensor self,
    std::optional<smith::headeronly::ScalarType> dtype,
    std::optional<smith::headeronly::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    std::optional<smith::headeronly::MemoryFormat> memory_format) {
  return smith::stable::to(
      self,
      dtype,
      layout,
      device,
      pin_memory,
      non_blocking,
      copy,
      memory_format);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_to_device(Tensor self, Device device) -> Tensor");
  m.def("my_to_dtype(Tensor self, ScalarType dtype) -> Tensor");
  m.def(
      "my_to_dtype_layout(Tensor self, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory, bool non_blocking, bool copy, MemoryFormat? memory_format) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(
    libsmith_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_to_device", SMITH_BOX(&my_to_device));
  m.impl("my_to_dtype", SMITH_BOX(&my_to_dtype));
  m.impl("my_to_dtype_layout", SMITH_BOX(&my_to_dtype_layout));
}
