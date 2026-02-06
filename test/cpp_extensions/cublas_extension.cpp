#include <iostream>

#include <smith/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cublas_v2.h>

smith::Tensor noop_cublas_function(smith::Tensor x) {
  cublasHandle_t handle;
  SMITH_CUDABLAS_CHECK(cublasCreate(&handle));
  SMITH_CUDABLAS_CHECK(cublasDestroy(handle));
  return x;
}

PYBIND11_MODULE(SMITH_EXTENSION_NAME, m) {
    m.def("noop_cublas_function", &noop_cublas_function, "a cublas function");
}
