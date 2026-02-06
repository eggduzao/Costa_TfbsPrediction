#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/device.h>
#include <smith/csrc/stable/ops.h>
#include <smith/csrc/stable/tensor.h>

#include <optional>

using smith::stable::Tensor;

Tensor my_full(
    std::vector<int64_t> size,
    double fill_value,
    std::optional<smith::headeronly::ScalarType> dtype,
    std::optional<smith::headeronly::Layout> layout,
    std::optional<smith::stable::Device> device,
    std::optional<bool> pin_memory) {
  return smith::stable::full(size, fill_value, dtype, layout, device, pin_memory);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def(
      "my_full(int[] size, float fill_value, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_full", SMITH_BOX(&my_full));
}
