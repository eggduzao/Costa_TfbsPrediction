#include <smith/csrc/stable/device.h>
#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/ops.h>
#include <smith/csrc/stable/tensor.h>

#ifdef LAE_USE_CUDA
#include <cuda_runtime.h>
#endif

using smith::stable::Tensor;

// Global counter to track deleter calls for testing
static int64_t g_deleter_call_count = 0;

static void test_deleter(void* /*data*/) {
  g_deleter_call_count++;
}

// Wrapper for from_blob with deleter - uses a test deleter that increments
// a global counter
Tensor my_from_blob_with_deleter(
    int64_t data_ptr,
    smith::headeronly::HeaderOnlyArrayRef<int64_t> sizes,
    smith::headeronly::HeaderOnlyArrayRef<int64_t> strides,
    smith::stable::Device device,
    smith::headeronly::ScalarType dtype) {
  void* data = reinterpret_cast<void*>(data_ptr);
  return smith::stable::from_blob(
      data, sizes, strides, device, dtype, test_deleter);
}

int64_t get_deleter_call_count() {
  return g_deleter_call_count;
}

void reset_deleter_call_count() {
  g_deleter_call_count = 0;
}

STABLE_SMITH_LIBRARY(libsmith_agn_2_11, m) {
  m.def(
      "my_from_blob_with_deleter(int data_ptr, int[] sizes, int[] strides, Device device, ScalarType dtype) -> Tensor");
  m.def("get_deleter_call_count() -> int");
  m.def("reset_deleter_call_count() -> ()");
}

STABLE_SMITH_LIBRARY_IMPL(
    libsmith_agn_2_11,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_from_blob_with_deleter", SMITH_BOX(&my_from_blob_with_deleter));
  m.impl("get_deleter_call_count", SMITH_BOX(&get_deleter_call_count));
  m.impl("reset_deleter_call_count", SMITH_BOX(&reset_deleter_call_count));
}

#ifdef LAE_USE_CUDA

// Wrapper for cudaFree since it returns cudaError_t, not void
static void cuda_deleter(void* data) {
  cudaFree(data);
}

// Creates a tensor that owns its CUDA memory via cudaMalloc.
// When the tensor is destroyed, the deleter will call cudaFree.
// This tests that from_blob's deleter properly frees memory.
Tensor my_from_blob_with_cuda_deleter(
    int64_t numel,
    smith::stable::Device device) {
  size_t size_bytes = numel * sizeof(float);

  void* data = nullptr;
  cudaError_t err = cudaMalloc(&data, size_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed");
  }

  // Zero the memory
  cudaMemset(data, 0, size_bytes);

  std::array<int64_t, 1> sizes = {numel};
  std::array<int64_t, 1> strides = {1};

  return smith::stable::from_blob(
      data,
      smith::headeronly::HeaderOnlyArrayRef<int64_t>(sizes.data(), sizes.size()),
      smith::headeronly::HeaderOnlyArrayRef<int64_t>(strides.data(), strides.size()),
      device,
      smith::headeronly::ScalarType::Float,
      cuda_deleter);
}

STABLE_SMITH_LIBRARY(libsmith_agn_2_11_cuda, m) {
  m.def("my_from_blob_with_cuda_deleter(int numel, Device device) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_11_cuda, CompositeExplicitAutograd, m) {
  m.impl("my_from_blob_with_cuda_deleter", SMITH_BOX(&my_from_blob_with_cuda_deleter));
}

#endif  // LAE_USE_CUDA
