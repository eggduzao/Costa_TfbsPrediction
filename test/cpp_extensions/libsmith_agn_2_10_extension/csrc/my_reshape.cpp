#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/csrc/stable/ops.h>

using smith::stable::Tensor;

Tensor my_reshape(Tensor t, smith::headeronly::HeaderOnlyArrayRef<int64_t> shape) {
  return reshape(t, shape);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_reshape(Tensor t, int[] shape) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_reshape", SMITH_BOX(&my_reshape));
}
