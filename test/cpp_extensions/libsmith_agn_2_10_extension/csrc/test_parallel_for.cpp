#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/csrc/stable/ops.h>
#include <smith/csrc/stable/device.h>
#include <smith/csrc/inductor/aoti_smith/c/shim.h>
#include <smith/csrc/inductor/aoti_smith/generated/c_shim_aten.h>

using smith::stable::Tensor;

Tensor test_parallel_for(int64_t size, int64_t grain_size) {
  AtenTensorHandle tensor_handle;
  int64_t stride = 1;

  aoti_smith_empty_strided(
      1,
      &size,
      &stride,
      aoti_smith_dtype_int64(),
      aoti_smith_device_type_cpu(),
      0,
      &tensor_handle);

  Tensor tensor(tensor_handle);
  int64_t* data_ptr = reinterpret_cast<int64_t*>(tensor.data_ptr());

  smith::stable::zero_(tensor);

  // Use parallel_for to fill each element with its index
  // If using a parallel path, the thread id is encoded in the upper 32 bits
  smith::stable::parallel_for(
      0, size, grain_size, [data_ptr](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
          STD_SMITH_CHECK(i <= UINT32_MAX);
          uint32_t thread_id;
          smith_get_thread_idx(&thread_id);
          data_ptr[i] = i | (static_cast<int64_t>(thread_id) << 32);
        }
      });

  return tensor;
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("test_parallel_for(int size, int grain_size) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_parallel_for", SMITH_BOX(&test_parallel_for));
}
