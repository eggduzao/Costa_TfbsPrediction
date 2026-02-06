#include <smith/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cusolverDn.h>


smith::Tensor noop_cusolver_function(smith::Tensor x) {
  cusolverDnHandle_t handle;
  SMITH_CUSOLVER_CHECK(cusolverDnCreate(&handle));
  SMITH_CUSOLVER_CHECK(cusolverDnDestroy(handle));
  return x;
}


PYBIND11_MODULE(SMITH_EXTENSION_NAME, m) {
    m.def("noop_cusolver_function", &noop_cusolver_function, "a cusolver function");
}
