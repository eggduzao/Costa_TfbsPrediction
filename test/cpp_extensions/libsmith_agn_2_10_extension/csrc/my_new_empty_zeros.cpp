#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/ops.h>
#include <smith/csrc/stable/tensor.h>

using smith::stable::Device;
using smith::stable::Tensor;

// Test new_empty with all kwargs
Tensor my_new_empty(
    Tensor self,
    smith::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<smith::headeronly::ScalarType> dtype,
    std::optional<smith::headeronly::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return smith::stable::new_empty(self, size, dtype, layout, device, pin_memory);
}

// Test new_zeros with all kwargs
Tensor my_new_zeros(
    Tensor self,
    smith::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<smith::headeronly::ScalarType> dtype,
    std::optional<smith::headeronly::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return smith::stable::new_zeros(self, size, dtype, layout, device, pin_memory);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def(
      "my_new_empty(Tensor self, int[] size, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def(
      "my_new_zeros(Tensor self, int[] size, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_new_empty", SMITH_BOX(&my_new_empty));
  m.impl("my_new_zeros", SMITH_BOX(&my_new_zeros));
}
