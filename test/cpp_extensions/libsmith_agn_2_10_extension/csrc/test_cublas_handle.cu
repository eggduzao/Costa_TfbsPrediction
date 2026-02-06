#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/c/shim.h>

void* my_get_curr_cuda_blas_handle() {
  void* ret_handle;
  SMITH_ERROR_CODE_CHECK(smith_get_current_cuda_blas_handle(&ret_handle));
  return ret_handle;
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_get_curr_cuda_blas_handle() -> int");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_get_curr_cuda_blas_handle", SMITH_BOX(&my_get_curr_cuda_blas_handle));
}
