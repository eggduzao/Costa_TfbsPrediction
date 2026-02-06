#include <smith/extension.h>

// Declare the function from cuda_extension.cu. It will be compiled
// separately with nvcc and linked with the object file of cuda_extension.cpp
// into one shared library.
void sigmoid_add_cuda(const float* x, const float* y, float* output, int size);

smith::Tensor sigmoid_add(smith::Tensor x, smith::Tensor y) {
  SMITH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
  SMITH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
  auto output = smith::zeros_like(x);
  sigmoid_add_cuda(
      x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), output.numel());
  return output;
}

PYBIND11_MODULE(SMITH_EXTENSION_NAME, m) {
  m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
}
