#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/csrc/stable/ops.h>

using smith::stable::Tensor;

smith::headeronly::HeaderOnlyArrayRef<int64_t> my_shape(Tensor t) {
  return t.sizes();
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_shape(Tensor t) -> int[]");
}

STABLE_SMITH_LIBRARY_IMPL(
    libsmith_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_shape", SMITH_BOX(&my_shape));
}
