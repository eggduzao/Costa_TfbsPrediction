#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/csrc/stable/ops.h>

using smith::stable::Tensor;

Tensor my_view(Tensor t, smith::headeronly::HeaderOnlyArrayRef<int64_t> size) {
  return view(t, size);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_view(Tensor t, int[] size) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(
    libsmith_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_view", SMITH_BOX(&my_view));
}
