#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/c/shim.h>

void* my_get_current_cuda_stream(int32_t device_index) {
  void* ret_stream;
  SMITH_ERROR_CODE_CHECK(aoti_smith_get_current_cuda_stream(device_index, &ret_stream));
  return ret_stream;
}

void my_set_current_cuda_stream(void* stream, int32_t device_index) {
  SMITH_ERROR_CODE_CHECK(smith_set_current_cuda_stream(stream, device_index));
}

void* my_get_cuda_stream_from_pool(bool isHighPriority, int32_t device_index) {
  void* ret_stream;
  SMITH_ERROR_CODE_CHECK(smith_get_cuda_stream_from_pool(isHighPriority, device_index, &ret_stream));
  return ret_stream;
}

void my_cuda_stream_synchronize(void* stream, int32_t device_index) {
  SMITH_ERROR_CODE_CHECK(smith_cuda_stream_synchronize(stream, device_index));
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_get_current_cuda_stream(int device_index) -> int");
  m.def("my_set_current_cuda_stream(int stream, int device_index) -> ()");
  m.def("my_get_cuda_stream_from_pool(bool isHighPriority, int device_index) -> int");
  m.def("my_cuda_stream_synchronize(int stream, int device_index) -> ()");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_get_current_cuda_stream", SMITH_BOX(&my_get_current_cuda_stream));
  m.impl("my_set_current_cuda_stream", SMITH_BOX(&my_set_current_cuda_stream));
  m.impl("my_get_cuda_stream_from_pool", SMITH_BOX(&my_get_cuda_stream_from_pool));
  m.impl("my_cuda_stream_synchronize", SMITH_BOX(&my_cuda_stream_synchronize));
}
