#include <smith/csrc/stable/device.h>
#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/ops.h>
#include <smith/csrc/stable/tensor.h>

using smith::stable::Tensor;

// Wrapper for smith::stable::from_blob with all parameters
// Note: We pass data_ptr as int64_t since we can't pass void* through the
// dispatcher
Tensor my_from_blob(
    int64_t data_ptr,
    smith::headeronly::HeaderOnlyArrayRef<int64_t> sizes,
    smith::headeronly::HeaderOnlyArrayRef<int64_t> strides,
    smith::stable::Device device,
    smith::headeronly::ScalarType dtype) {
  void* data = reinterpret_cast<void*>(data_ptr);
  return smith::stable::from_blob(
      data, sizes, strides, device, dtype);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def(
      "my_from_blob(int data_ptr, int[] sizes, int[] strides, Device device, ScalarType dtype) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(
    libsmith_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_from_blob", SMITH_BOX(&my_from_blob));
}
