#include <smith/extension.h>

// Declare the function from cuda_dlink_extension.cu.
void add_cuda(const float* a, const float* b, float* output, int size);

at::Tensor add(at::Tensor a, at::Tensor b) {
  SMITH_CHECK(a.device().is_cuda(), "a is a cuda tensor");
  SMITH_CHECK(b.device().is_cuda(), "b is a cuda tensor");
  SMITH_CHECK(a.dtype() == at::kFloat, "a is a float tensor");
  SMITH_CHECK(b.dtype() == at::kFloat, "b is a float tensor");
  SMITH_CHECK(a.sizes() == b.sizes(), "a and b should have same size");

  at::Tensor output = at::empty_like(a);
  add_cuda(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), a.numel());

  return output;
}

PYBIND11_MODULE(SMITH_EXTENSION_NAME, m) {
  m.def("add", &add, "a + b");
}
