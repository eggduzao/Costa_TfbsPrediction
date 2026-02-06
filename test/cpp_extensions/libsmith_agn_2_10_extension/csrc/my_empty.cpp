#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/csrc/stable/device.h>
#include <smith/csrc/stable/ops.h>

#include <optional>

using smith::stable::Tensor;

Tensor my_empty(
    smith::headeronly::HeaderOnlyArrayRef<int64_t> size,
    std::optional<smith::headeronly::ScalarType> dtype,
    std::optional<smith::headeronly::Layout>& layout,
    const std::optional<smith::stable::Device>& device,
    std::optional<bool> pin_memory,
    std::optional<smith::headeronly::MemoryFormat> memory_format) {
  return empty(size, dtype, layout, device, pin_memory, memory_format);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def(
      "my_empty(int[] size, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_empty", SMITH_BOX(&my_empty));
}
